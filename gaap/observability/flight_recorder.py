"""
GAAP Flight Recorder Module - Black Box Recording

Provides crash-safe black box recording for all operations:
- Circular buffer for recent events
- Crash-safe persistence
- Export to JSON/Parquet
- Memory-efficient event storage

Usage:
    from gaap.observability import FlightRecorder

    recorder = FlightRecorder(max_events=10000)
    recorder.record("llm_call", {"model": "gpt-4", "tokens": 100})
    recorder.record("error", {"error": "timeout", "layer": "execution"})

    # Export for analysis
    recorder.export_json("flight_data.json")
    recorder.export_parquet("flight_data.parquet")
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Iterator
import hashlib

logger = logging.getLogger("gaap.observability.flight_recorder")


class FlightEventType(Enum):
    """Types of flight recorder events."""

    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    WARNING = "warning"
    STATE_CHANGE = "state_change"
    OODA_PHASE = "ooda_phase"
    HEALING = "healing"
    AXIOM_CHECK = "axiom_check"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    USER_INPUT = "user_input"
    SYSTEM = "system"
    CHECKPOINT = "checkpoint"


@dataclass
class FlightEvent:
    """A single event in the flight recorder."""

    event_id: str
    event_type: FlightEventType
    timestamp: str
    data: dict[str, Any]
    sequence_number: int
    process_id: int
    thread_id: int
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        content = f"{self.event_id}{self.event_type.value}{json.dumps(self.data, sort_keys=True, default=str)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify event integrity."""
        return self.checksum == self._compute_checksum()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "sequence_number": self.sequence_number,
            "process_id": self.process_id,
            "thread_id": self.thread_id,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlightEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=FlightEventType(data["event_type"]),
            timestamp=data["timestamp"],
            data=data["data"],
            sequence_number=data["sequence_number"],
            process_id=data["process_id"],
            thread_id=data["thread_id"],
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )


class CircularBuffer:
    """
    Thread-safe circular buffer for events.

    Provides O(1) append and memory-bounded storage.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque[FlightEvent] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._total_events = 0
        self._dropped_events = 0

    def append(self, event: FlightEvent) -> None:
        """Add an event to the buffer."""
        with self._lock:
            if len(self._buffer) >= self.max_size:
                self._dropped_events += 1
            self._buffer.append(event)
            self._total_events += 1

    def get_all(self) -> list[FlightEvent]:
        """Get all events in the buffer."""
        with self._lock:
            return list(self._buffer)

    def get_range(self, start: int, end: int) -> list[FlightEvent]:
        """Get events in a range."""
        with self._lock:
            events = list(self._buffer)
            return events[start:end]

    def get_since(self, sequence_number: int) -> list[FlightEvent]:
        """Get events since a sequence number."""
        with self._lock:
            return [e for e in self._buffer if e.sequence_number > sequence_number]

    def get_latest(self, count: int = 100) -> list[FlightEvent]:
        """Get the latest N events."""
        with self._lock:
            return list(self._buffer)[-count:]

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._dropped_events = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> dict[str, int]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._buffer),
                "max_size": self.max_size,
                "total_events": self._total_events,
                "dropped_events": self._dropped_events,
            }


class FlightRecorder:
    """
    Black box recorder for all GAAP operations.

    Features:
    - Circular buffer for memory-bounded storage
    - Crash-safe persistence with periodic saves
    - Signal handlers for graceful shutdown
    - Export to JSON and Parquet formats
    - Event type filtering and querying

    Usage:
        recorder = FlightRecorder(max_events=50000, persist_path="./flight_data")

        # Record events
        recorder.record("llm_call", {"model": "gpt-4", "tokens": 100})
        recorder.record_error("TimeoutError", {"layer": "execution"})

        # Query events
        recent = recorder.get_recent_events(100)
        errors = recorder.get_events_by_type(FlightEventType.ERROR)

        # Export
        recorder.export_json("flight_data.json")
    """

    DEFAULT_PERSIST_PATH = ".gaap/flight_recorder"

    _instance: Optional["FlightRecorder"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "FlightRecorder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        max_events: int = 50000,
        persist_path: str | None = None,
        auto_persist: bool = True,
        persist_interval: int = 60,
        enable_crash_handler: bool = True,
    ) -> None:
        if self._initialized:
            return

        self.max_events = max_events
        self.persist_path = Path(persist_path or self.DEFAULT_PERSIST_PATH)
        self.auto_persist = auto_persist
        self.persist_interval = persist_interval

        self._buffer = CircularBuffer(max_events)
        self._sequence_counter = 0
        self._lock = threading.RLock()
        self._start_time = datetime.now()
        self._last_persist_time: datetime | None = None
        self._persist_thread: threading.Thread | None = None
        self._shutdown_requested = False
        self._enabled = True

        self.persist_path.mkdir(parents=True, exist_ok=True)

        if enable_crash_handler:
            self._setup_crash_handlers()

        if auto_persist:
            self._start_persist_thread()

        atexit.register(self._on_exit)

        self._initialized = True
        logger.info(f"FlightRecorder initialized (max_events={max_events})")

    def _setup_crash_handlers(self) -> None:
        """Set up signal handlers for crash-safe persistence."""

        def signal_handler(signum: int, frame: Any) -> None:
            logger.warning(f"Received signal {signum}, flushing flight recorder...")
            self._emergency_persist()
            sys.exit(signum)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _start_persist_thread(self) -> None:
        """Start the background persistence thread."""

        def persist_loop() -> None:
            while not self._shutdown_requested:
                time.sleep(self.persist_interval)
                if not self._shutdown_requested:
                    try:
                        self._periodic_persist()
                    except Exception as e:
                        logger.error(f"Periodic persist failed: {e}")

        self._persist_thread = threading.Thread(target=persist_loop, daemon=True)
        self._persist_thread.start()

    def _on_exit(self) -> None:
        """Handle process exit."""
        self._shutdown_requested = True
        self._emergency_persist()

    def _emergency_persist(self) -> None:
        """Emergency persistence on shutdown or crash."""
        try:
            events = self._buffer.get_all()
            if not events:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flight_emergency_{timestamp}.json"
            filepath = self.persist_path / filename

            data = {
                "emergency": True,
                "timestamp": datetime.now().isoformat(),
                "total_events": len(events),
                "events": [e.to_dict() for e in events],
            }

            temp_path = filepath.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
            temp_path.rename(filepath)

            logger.info(f"Emergency persist saved: {filepath}")
        except Exception as e:
            logger.error(f"Emergency persist failed: {e}")

    def _periodic_persist(self) -> None:
        """Periodic background persistence."""
        if not self.auto_persist:
            return

        events = self._buffer.get_latest(1000)
        if not events:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"flight_periodic_{timestamp}.jsonl"
        filepath = self.persist_path / filename

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict(), default=str) + "\n")

            self._last_persist_time = datetime.now()
        except Exception as e:
            logger.error(f"Periodic persist failed: {e}")

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        return f"{int(time.time() * 1000000)}_{self._sequence_counter}"

    def _next_sequence_number(self) -> int:
        """Get the next sequence number."""
        with self._lock:
            self._sequence_counter += 1
            return self._sequence_counter

    def enable(self) -> None:
        """Enable recording."""
        self._enabled = True

    def disable(self) -> None:
        """Disable recording."""
        self._enabled = False

    def record(
        self,
        event_type: str | FlightEventType,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record an event.

        Args:
            event_type: Type of event (string or FlightEventType)
            data: Event data
            metadata: Optional metadata

        Returns:
            Event ID
        """
        if not self._enabled:
            return ""

        if isinstance(event_type, str):
            try:
                event_type = FlightEventType(event_type)
            except ValueError:
                event_type = FlightEventType.SYSTEM

        event = FlightEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data,
            sequence_number=self._next_sequence_number(),
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
            metadata=metadata or {},
        )

        self._buffer.append(event)
        return event.event_id

    def record_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> str:
        """Record an LLM API call."""
        return self.record(
            FlightEventType.LLM_CALL,
            {
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "success": success,
                "error": error,
            },
        )

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        success: bool,
        latency_ms: float | None = None,
    ) -> str:
        """Record a tool call."""
        return self.record(
            FlightEventType.TOOL_CALL,
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": str(result)[:500] if result else None,
                "success": success,
                "latency_ms": latency_ms,
            },
        )

    def record_error(
        self,
        error_type: str,
        context: dict[str, Any],
        severity: str = "error",
    ) -> str:
        """Record an error."""
        return self.record(
            FlightEventType.ERROR,
            {
                "error_type": error_type,
                "context": context,
                "severity": severity,
            },
        )

    def record_ooda_phase(
        self,
        phase: str,
        iteration: int,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Record an OODA phase transition."""
        return self.record(
            FlightEventType.OODA_PHASE,
            {
                "phase": phase,
                "iteration": iteration,
                "details": details or {},
            },
        )

    def record_healing(
        self,
        level: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Record a healing event."""
        return self.record(
            FlightEventType.HEALING,
            {
                "level": level,
                "success": success,
                "details": details or {},
            },
        )

    def record_checkpoint(
        self,
        name: str,
        state: dict[str, Any] | None = None,
    ) -> str:
        """Record a checkpoint."""
        return self.record(
            FlightEventType.CHECKPOINT,
            {
                "name": name,
                "state_summary": {k: type(v).__name__ for k, v in (state or {}).items()},
            },
        )

    def get_recent_events(self, count: int = 100) -> list[FlightEvent]:
        """Get the most recent events."""
        return self._buffer.get_latest(count)

    def get_events_by_type(self, event_type: FlightEventType) -> list[FlightEvent]:
        """Get all events of a specific type."""
        return [e for e in self._buffer.get_all() if e.event_type == event_type]

    def get_events_since(self, sequence_number: int) -> list[FlightEvent]:
        """Get events since a sequence number."""
        return self._buffer.get_since(sequence_number)

    def get_errors(self) -> list[FlightEvent]:
        """Get all error events."""
        return self.get_events_by_type(FlightEventType.ERROR)

    def search_events(
        self,
        event_type: FlightEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        data_contains: dict[str, Any] | None = None,
    ) -> list[FlightEvent]:
        """
        Search for events matching criteria.

        Args:
            event_type: Filter by event type
            start_time: Filter events after this time
            end_time: Filter events before this time
            data_contains: Filter events containing these key-value pairs in data

        Returns:
            List of matching events
        """
        events = self._buffer.get_all()

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if start_time:
            events = [e for e in events if datetime.fromisoformat(e.timestamp) >= start_time]

        if end_time:
            events = [e for e in events if datetime.fromisoformat(e.timestamp) <= end_time]

        if data_contains:

            def matches(e: FlightEvent) -> bool:
                for key, value in data_contains.items():
                    if key not in e.data or e.data[key] != value:
                        return False
                return True

            events = [e for e in events if matches(e)]

        return events

    def get_stats(self) -> dict[str, Any]:
        """Get recorder statistics."""
        buffer_stats = self._buffer.stats
        events = self._buffer.get_all()

        event_counts: dict[str, int] = {}
        for event in events:
            key = event.event_type.value
            event_counts[key] = event_counts.get(key, 0) + 1

        return {
            "start_time": self._start_time.isoformat(),
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            "last_persist": self._last_persist_time.isoformat()
            if self._last_persist_time
            else None,
            "buffer": buffer_stats,
            "event_counts": event_counts,
            "enabled": self._enabled,
        }

    def export_json(self, filepath: str, indent: int = 2) -> bool:
        """
        Export all events to a JSON file.

        Args:
            filepath: Output file path
            indent: JSON indentation

        Returns:
            True if export succeeded
        """
        try:
            events = self._buffer.get_all()
            data = {
                "export_time": datetime.now().isoformat(),
                "total_events": len(events),
                "events": [e.to_dict() for e in events],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)

            logger.info(f"Exported {len(events)} events to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Export to JSON failed: {e}")
            return False

    def export_parquet(self, filepath: str) -> bool:
        """
        Export all events to a Parquet file.

        Requires pyarrow or fastparquet.

        Args:
            filepath: Output file path

        Returns:
            True if export succeeded
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("pyarrow not installed, cannot export to Parquet")
            return False

        try:
            events = self._buffer.get_all()

            records = []
            for event in events:
                records.append(
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp,
                        "sequence_number": event.sequence_number,
                        "process_id": event.process_id,
                        "thread_id": event.thread_id,
                        "data": json.dumps(event.data, default=str),
                        "checksum": event.checksum,
                    }
                )

            if not records:
                logger.warning("No events to export")
                return False

            table = pa.Table.from_pylist(records)
            pq.write_table(table, filepath)

            logger.info(f"Exported {len(records)} events to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Export to Parquet failed: {e}")
            return False

    def export_jsonl(self, filepath: str) -> bool:
        """
        Export all events to JSONL (JSON Lines) format.

        Args:
            filepath: Output file path

        Returns:
            True if export succeeded
        """
        try:
            events = self._buffer.get_all()

            with open(filepath, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict(), default=str) + "\n")

            logger.info(f"Exported {len(events)} events to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Export to JSONL failed: {e}")
            return False

    def clear(self) -> None:
        """Clear all recorded events."""
        self._buffer.clear()
        self._sequence_counter = 0
        logger.info("Flight recorder cleared")

    def verify_integrity(self) -> dict[str, Any]:
        """
        Verify integrity of all events.

        Returns:
            Dictionary with verification results
        """
        events = self._buffer.get_all()
        valid = 0
        invalid = []

        for event in events:
            if event.verify_integrity():
                valid += 1
            else:
                invalid.append(event.event_id)

        return {
            "total_events": len(events),
            "valid_events": valid,
            "invalid_events": len(invalid),
            "invalid_ids": invalid[:10],
        }
