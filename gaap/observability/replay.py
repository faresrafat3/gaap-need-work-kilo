"""
GAAP Session Replay Module - Record and Replay Sessions

Provides session recording and replay capabilities:
- SessionRecorder: Record all LLM I/O and tool results
- SessionReplay: Load and replay sessions step-by-step
- Time-travel to specific steps
- State inspection at any point
- Ability to modify state and resume

Usage:
    from gaap.observability import SessionRecorder, SessionReplay

    # Recording
    recorder = SessionRecorder()
    recorder.start_session("my_session")
    recorder.record_llm_call(messages, response)
    recorder.record_tool_call("read_file", args, result)
    recorder.save_session()

    # Replay
    replay = SessionReplay()
    replay.load_session("my_session")
    replay.step_forward()
    state = replay.get_current_state()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
import hashlib

logger = logging.getLogger("gaap.observability.replay")


class StepType(Enum):
    """Types of recorded steps."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    USER_INPUT = "user_input"
    SYSTEM_EVENT = "system_event"
    OODA_PHASE = "ooda_phase"
    HEALING_EVENT = "healing_event"
    AXIOM_CHECK = "axiom_check"


@dataclass
class RecordedStep:
    """A single recorded step in a session."""

    step_id: str
    step_type: StepType
    timestamp: str
    data: dict[str, Any]
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_step_id: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "state_snapshot": self.state_snapshot,
            "metadata": self.metadata,
            "parent_step_id": self.parent_step_id,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordedStep":
        """Create from dictionary."""
        return cls(
            step_id=data["step_id"],
            step_type=StepType(data["step_type"]),
            timestamp=data["timestamp"],
            data=data["data"],
            state_snapshot=data.get("state_snapshot", {}),
            metadata=data.get("metadata", {}),
            parent_step_id=data.get("parent_step_id"),
            duration_ms=data.get("duration_ms"),
        )


@dataclass
class SessionState:
    """
    Captured state at a point in time.

    Contains all relevant state information for inspection and replay.
    """

    step_index: int
    total_steps: int
    messages: list[dict[str, Any]]
    tool_results: dict[str, Any]
    variables: dict[str, Any]
    memory: dict[str, Any]
    ooda_state: dict[str, Any] = field(default_factory=dict)
    task_context: dict[str, Any] = field(default_factory=dict)
    healing_history: list[dict[str, Any]] = field(default_factory=list)
    axiom_violations: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "messages": self.messages,
            "tool_results": self.tool_results,
            "variables": self.variables,
            "memory": self.memory,
            "ooda_state": self.ooda_state,
            "task_context": self.task_context,
            "healing_history": self.healing_history,
            "axiom_violations": self.axiom_violations,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        return cls(
            step_index=data["step_index"],
            total_steps=data["total_steps"],
            messages=data.get("messages", []),
            tool_results=data.get("tool_results", {}),
            variables=data.get("variables", {}),
            memory=data.get("memory", {}),
            ooda_state=data.get("ooda_state", {}),
            task_context=data.get("task_context", {}),
            healing_history=data.get("healing_history", []),
            axiom_violations=data.get("axiom_violations", []),
            metrics=data.get("metrics", {}),
        )


@dataclass
class RecordedSession:
    """A complete recorded session."""

    session_id: str
    name: str
    created_at: str
    ended_at: str | None
    steps: list[RecordedStep]
    initial_state: dict[str, Any]
    final_state: dict[str, Any] | None
    metadata: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "steps": [s.to_dict() for s in self.steps],
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "metadata": self.metadata,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordedSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            name=data["name"],
            created_at=data["created_at"],
            ended_at=data.get("ended_at"),
            steps=[RecordedStep.from_dict(s) for s in data.get("steps", [])],
            initial_state=data.get("initial_state", {}),
            final_state=data.get("final_state"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


class SessionRecorder:
    """
    Records all operations in a session for later replay.

    Features:
    - Records LLM calls, tool calls, state changes
    - Captures state snapshots at each step
    - Stores sessions in .gaap/sessions/
    - Supports tagging and notes

    Usage:
        recorder = SessionRecorder()
        session_id = recorder.start_session("debug_session")
        recorder.record_llm_request(messages, model, params)
        recorder.record_llm_response(response, latency_ms)
        recorder.save_session()
    """

    DEFAULT_STORAGE_PATH = ".gaap/sessions"

    def __init__(
        self,
        storage_path: str | None = None,
        auto_save: bool = True,
        max_steps: int = 10000,
    ):
        self.storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        self.max_steps = max_steps

        self._current_session: RecordedSession | None = None
        self._current_state: dict[str, Any] = {}
        self._step_counter: int = 0
        self._last_step_id: str | None = None

    def start_session(
        self,
        name: str,
        initial_state: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Start a new recording session.

        Args:
            name: Session name
            initial_state: Initial state to record
            tags: Optional tags for categorization
            metadata: Optional metadata

        Returns:
            Session ID
        """
        session_id = self._generate_session_id(name)

        self._current_session = RecordedSession(
            session_id=session_id,
            name=name,
            created_at=datetime.now().isoformat(),
            ended_at=None,
            steps=[],
            initial_state=initial_state or {},
            final_state=None,
            metadata=metadata or {},
            tags=tags or [],
        )

        self._current_state = dict(initial_state or {})
        self._step_counter = 0
        self._last_step_id = None

        start_step = RecordedStep(
            step_id=f"{session_id}_step_0",
            step_type=StepType.SESSION_START,
            timestamp=datetime.now().isoformat(),
            data={"name": name},
            state_snapshot=dict(self._current_state),
        )
        self._current_session.steps.append(start_step)
        self._step_counter = 1

        logger.info(f"Started session recording: {session_id}")
        return session_id

    def _generate_session_id(self, name: str) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        return f"{timestamp}_{hash_part}"

    def end_session(self, notes: str = "") -> None:
        """
        End the current session.

        Args:
            notes: Optional notes to add to the session
        """
        if self._current_session is None:
            logger.warning("No active session to end")
            return

        end_step = RecordedStep(
            step_id=f"{self._current_session.session_id}_step_{self._step_counter}",
            step_type=StepType.SESSION_END,
            timestamp=datetime.now().isoformat(),
            data={"notes": notes},
            state_snapshot=dict(self._current_state),
        )
        self._current_session.steps.append(end_step)

        self._current_session.ended_at = datetime.now().isoformat()
        self._current_session.final_state = dict(self._current_state)
        self._current_session.notes = notes

        if self.auto_save:
            self.save_session()

        logger.info(f"Ended session: {self._current_session.session_id}")
        self._current_session = None

    def record_llm_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        params: dict[str, Any] | None = None,
        provider: str = "unknown",
    ) -> str:
        """
        Record an LLM request.

        Args:
            messages: List of messages sent to the LLM
            model: Model name
            params: Optional request parameters
            provider: Provider name

        Returns:
            Step ID for correlating with response
        """
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        step_id = f"{self._current_session.session_id}_step_{self._step_counter}"
        step = RecordedStep(
            step_id=step_id,
            step_type=StepType.LLM_REQUEST,
            timestamp=datetime.now().isoformat(),
            data={
                "messages": messages,
                "model": model,
                "params": params or {},
                "provider": provider,
            },
            state_snapshot=dict(self._current_state),
            parent_step_id=self._last_step_id,
        )

        self._current_session.steps.append(step)
        self._step_counter += 1
        self._last_step_id = step_id

        if "llm_requests" not in self._current_state:
            self._current_state["llm_requests"] = []
        self._current_state["llm_requests"].append(step_id)

        return step_id

    def record_llm_response(
        self,
        response: dict[str, Any],
        latency_ms: float | None = None,
        request_step_id: str | None = None,
    ) -> str:
        """
        Record an LLM response.

        Args:
            response: LLM response data
            latency_ms: Latency in milliseconds
            request_step_id: ID of the corresponding request step

        Returns:
            Step ID
        """
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        step_id = f"{self._current_session.session_id}_step_{self._step_counter}"
        step = RecordedStep(
            step_id=step_id,
            step_type=StepType.LLM_RESPONSE,
            timestamp=datetime.now().isoformat(),
            data={
                "response": response,
                "latency_ms": latency_ms,
            },
            state_snapshot=dict(self._current_state),
            parent_step_id=request_step_id or self._last_step_id,
            duration_ms=latency_ms,
        )

        self._current_session.steps.append(step)
        self._step_counter += 1
        self._last_step_id = step_id

        self._current_state["last_llm_response"] = step_id

        return step_id

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        step_id: str | None = None,
    ) -> str:
        """
        Record a tool call initiation.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            step_id: Optional custom step ID

        Returns:
            Step ID for correlating with result
        """
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        sid = step_id or f"{self._current_session.session_id}_step_{self._step_counter}"
        step = RecordedStep(
            step_id=sid,
            step_type=StepType.TOOL_CALL,
            timestamp=datetime.now().isoformat(),
            data={
                "tool_name": tool_name,
                "arguments": arguments,
            },
            state_snapshot=dict(self._current_state),
            parent_step_id=self._last_step_id,
        )

        self._current_session.steps.append(step)
        self._step_counter += 1
        self._last_step_id = sid

        if "tool_calls" not in self._current_state:
            self._current_state["tool_calls"] = []
        self._current_state["tool_calls"].append(sid)

        return sid

    def record_tool_result(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        latency_ms: float | None = None,
        call_step_id: str | None = None,
    ) -> str:
        """
        Record a tool call result.

        Args:
            tool_name: Name of the tool
            result: Tool result
            success: Whether the call succeeded
            latency_ms: Latency in milliseconds
            call_step_id: ID of the corresponding tool call step

        Returns:
            Step ID
        """
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        step_id = f"{self._current_session.session_id}_step_{self._step_counter}"
        step = RecordedStep(
            step_id=step_id,
            step_type=StepType.TOOL_RESULT,
            timestamp=datetime.now().isoformat(),
            data={
                "tool_name": tool_name,
                "result": result,
                "success": success,
            },
            state_snapshot=dict(self._current_state),
            parent_step_id=call_step_id or self._last_step_id,
            duration_ms=latency_ms,
        )

        self._current_session.steps.append(step)
        self._step_counter += 1
        self._last_step_id = step_id

        if "tool_results" not in self._current_state:
            self._current_state["tool_results"] = {}
        self._current_state["tool_results"][tool_name] = {
            "result": result,
            "success": success,
            "step_id": step_id,
        }

        return step_id

    def record_state_change(
        self,
        key: str,
        old_value: Any,
        new_value: Any,
        reason: str = "",
    ) -> str:
        """
        Record a state change.

        Args:
            key: State key that changed
            old_value: Previous value
            new_value: New value
            reason: Reason for the change

        Returns:
            Step ID
        """
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        self._current_state[key] = new_value

        step_id = f"{self._current_session.session_id}_step_{self._step_counter}"
        step = RecordedStep(
            step_id=step_id,
            step_type=StepType.STATE_CHANGE,
            timestamp=datetime.now().isoformat(),
            data={
                "key": key,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
            },
            state_snapshot=dict(self._current_state),
            parent_step_id=self._last_step_id,
        )

        self._current_session.steps.append(step)
        self._step_counter += 1
        self._last_step_id = step_id

        return step_id

    def record_error(
        self,
        error: Exception | str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Record an error.

        Args:
            error: The error that occurred
            context: Optional context information

        Returns:
            Step ID
        """
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        error_data = {
            "error_type": type(error).__name__ if isinstance(error, Exception) else "string",
            "error_message": str(error),
            "context": context or {},
        }

        step_id = f"{self._current_session.session_id}_step_{self._step_counter}"
        step = RecordedStep(
            step_id=step_id,
            step_type=StepType.ERROR,
            timestamp=datetime.now().isoformat(),
            data=error_data,
            state_snapshot=dict(self._current_state),
            parent_step_id=self._last_step_id,
        )

        self._current_session.steps.append(step)
        self._step_counter += 1
        self._last_step_id = step_id

        if "errors" not in self._current_state:
            self._current_state["errors"] = []
        self._current_state["errors"].append(step_id)

        return step_id

    def record_ooda_phase(
        self,
        phase: str,
        iteration: int,
        details: dict[str, Any] | None = None,
    ) -> str:
        """
        Record an OODA phase transition.

        Args:
            phase: OODA phase name
            iteration: Current iteration number
            details: Optional details

        Returns:
            Step ID
        """
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        step_id = f"{self._current_session.session_id}_step_{self._step_counter}"
        step = RecordedStep(
            step_id=step_id,
            step_type=StepType.OODA_PHASE,
            timestamp=datetime.now().isoformat(),
            data={
                "phase": phase,
                "iteration": iteration,
                "details": details or {},
            },
            state_snapshot=dict(self._current_state),
            parent_step_id=self._last_step_id,
        )

        self._current_session.steps.append(step)
        self._step_counter += 1
        self._last_step_id = step_id

        return step_id

    def create_checkpoint(self, name: str = "") -> str:
        """
        Create a checkpoint in the session.

        Args:
            name: Optional checkpoint name

        Returns:
            Checkpoint step ID
        """
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        step_id = f"{self._current_session.session_id}_checkpoint_{self._step_counter}"
        step = RecordedStep(
            step_id=step_id,
            step_type=StepType.CHECKPOINT,
            timestamp=datetime.now().isoformat(),
            data={"name": name},
            state_snapshot=dict(self._current_state),
            parent_step_id=self._last_step_id,
        )

        self._current_session.steps.append(step)
        self._step_counter += 1

        return step_id

    def save_session(self, session_id: str | None = None) -> str:
        """
        Save the current or specified session.

        Args:
            session_id: Optional session ID (uses current if not specified)

        Returns:
            Path to saved session file
        """
        session = self._current_session
        if session is None:
            raise RuntimeError("No active session to save")

        filename = f"{session.session_id}.json"
        filepath = self.storage_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Session saved: {filepath}")
        return str(filepath)

    def get_current_session_id(self) -> str | None:
        """Get the current session ID."""
        return self._current_session.session_id if self._current_session else None

    def get_current_state(self) -> dict[str, Any]:
        """Get the current state."""
        return dict(self._current_state)


class SessionReplay:
    """
    Load and replay recorded sessions.

    Features:
    - Load sessions from storage
    - Step forward/backward through session
    - Jump to specific steps
    - Inspect state at any point
    - Modify state and resume execution

    Usage:
        replay = SessionReplay()
        replay.load_session("session_id")
        replay.step_forward()
        state = replay.get_current_state()
    """

    def __init__(self, storage_path: str | None = None):
        self.storage_path = Path(storage_path or SessionRecorder.DEFAULT_STORAGE_PATH)
        self._session: RecordedSession | None = None
        self._current_step_index: int = 0
        self._replay_state: SessionState | None = None
        self._modified_state: dict[str, Any] | None = None

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all available sessions.

        Returns:
            List of session metadata
        """
        sessions = []
        if not self.storage_path.exists():
            return sessions

        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "session_id": data["session_id"],
                        "name": data["name"],
                        "created_at": data["created_at"],
                        "step_count": len(data.get("steps", [])),
                        "tags": data.get("tags", []),
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load session {filepath}: {e}")

        return sorted(sessions, key=lambda s: s["created_at"], reverse=True)

    def load_session(self, session_id: str) -> bool:
        """
        Load a session for replay.

        Args:
            session_id: Session ID to load

        Returns:
            True if loaded successfully
        """
        filepath = self.storage_path / f"{session_id}.json"
        if not filepath.exists():
            for f in self.storage_path.glob("*.json"):
                if session_id in f.name:
                    filepath = f
                    break

        if not filepath.exists():
            logger.error(f"Session not found: {session_id}")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._session = RecordedSession.from_dict(data)
            self._current_step_index = 0
            self._replay_state = None
            self._modified_state = None
            self._build_initial_state()
            logger.info(f"Loaded session: {session_id} ({len(self._session.steps)} steps)")
            return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load session: {e}")
            return False

    def _build_initial_state(self) -> None:
        """Build the initial state from session."""
        if self._session is None:
            return

        self._replay_state = SessionState(
            step_index=0,
            total_steps=len(self._session.steps),
            messages=[],
            tool_results={},
            variables=dict(self._session.initial_state),
            memory={},
        )

    def step_forward(self) -> RecordedStep | None:
        """
        Advance to the next step.

        Returns:
            The next step, or None if at end
        """
        if self._session is None or self._replay_state is None:
            return None

        if self._current_step_index >= len(self._session.steps) - 1:
            return None

        self._current_step_index += 1
        step = self._session.steps[self._current_step_index]
        self._apply_step_to_state(step)
        self._replay_state.step_index = self._current_step_index

        return step

    def step_backward(self) -> RecordedStep | None:
        """
        Go back to the previous step.

        Returns:
            The previous step, or None if at beginning
        """
        if self._session is None or self._replay_state is None:
            return None

        if self._current_step_index <= 0:
            return None

        self._current_step_index -= 1
        self._rebuild_state_to_step(self._current_step_index)

        return self._session.steps[self._current_step_index]

    def jump_to_step(self, step_index: int) -> RecordedStep | None:
        """
        Jump to a specific step.

        Args:
            step_index: Step index to jump to

        Returns:
            The step at that index, or None if invalid
        """
        if self._session is None or self._replay_state is None:
            return None

        if step_index < 0 or step_index >= len(self._session.steps):
            return None

        self._current_step_index = step_index
        self._rebuild_state_to_step(step_index)

        return self._session.steps[step_index]

    def jump_to_checkpoint(self, checkpoint_name: str | None = None) -> RecordedStep | None:
        """
        Jump to a checkpoint.

        Args:
            checkpoint_name: Optional checkpoint name (jumps to last if not specified)

        Returns:
            The checkpoint step, or None if not found
        """
        if self._session is None:
            return None

        checkpoint_steps = [
            (i, s) for i, s in enumerate(self._session.steps) if s.step_type == StepType.CHECKPOINT
        ]

        if not checkpoint_steps:
            return None

        if checkpoint_name:
            for i, step in checkpoint_steps:
                if step.data.get("name") == checkpoint_name:
                    return self.jump_to_step(i)
            return None

        return self.jump_to_step(checkpoint_steps[-1][0])

    def _apply_step_to_state(self, step: RecordedStep) -> None:
        """Apply a step's changes to the replay state."""
        if self._replay_state is None:
            return

        if step.step_type == StepType.LLM_REQUEST:
            self._replay_state.messages.extend(step.data.get("messages", []))

        elif step.step_type == StepType.LLM_RESPONSE:
            response = step.data.get("response", {})
            if "content" in response:
                self._replay_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["content"],
                    }
                )

        elif step.step_type == StepType.TOOL_RESULT:
            tool_name = step.data.get("tool_name", "unknown")
            self._replay_state.tool_results[tool_name] = step.data.get("result")

        elif step.step_type == StepType.STATE_CHANGE:
            key = step.data.get("key")
            new_value = step.data.get("new_value")
            if key:
                self._replay_state.variables[key] = new_value

        elif step.step_type == StepType.OODA_PHASE:
            self._replay_state.ooda_state = step.data.get("details", {})

        elif step.step_type == StepType.ERROR:
            self._replay_state.healing_history.append(step.data)

    def _rebuild_state_to_step(self, target_index: int) -> None:
        """Rebuild state by replaying from beginning to target step."""
        if self._session is None or self._replay_state is None:
            return

        self._build_initial_state()

        for i in range(1, target_index + 1):
            if i < len(self._session.steps):
                self._apply_step_to_state(self._session.steps[i])

        self._replay_state.step_index = target_index

    def get_current_step(self) -> RecordedStep | None:
        """Get the current step."""
        if self._session is None:
            return None
        return self._session.steps[self._current_step_index]

    def get_current_state(self) -> SessionState | None:
        """Get the current replay state."""
        if self._replay_state:
            return SessionState(
                step_index=self._replay_state.step_index,
                total_steps=self._replay_state.total_steps,
                messages=list(self._replay_state.messages),
                tool_results=dict(self._replay_state.tool_results),
                variables=dict(self._replay_state.variables),
                memory=dict(self._replay_state.memory),
                ooda_state=dict(self._replay_state.ooda_state),
                task_context=dict(self._replay_state.task_context),
                healing_history=list(self._replay_state.healing_history),
                axiom_violations=list(self._replay_state.axiom_violations),
                metrics=dict(self._replay_state.metrics),
            )
        return None

    def modify_state(self, key: str, value: Any) -> None:
        """
        Modify the current state.

        Args:
            key: State key to modify
            value: New value
        """
        if self._replay_state is None:
            return

        if self._modified_state is None:
            self._modified_state = {}

        self._modified_state[key] = value
        self._replay_state.variables[key] = value

    def get_all_steps(self) -> list[RecordedStep]:
        """Get all steps in the session."""
        if self._session is None:
            return []
        return list(self._session.steps)

    def get_step_count(self) -> int:
        """Get the total number of steps."""
        if self._session is None:
            return 0
        return len(self._session.steps)

    def get_current_index(self) -> int:
        """Get the current step index."""
        return self._current_step_index

    def search_steps(
        self,
        step_type: StepType | None = None,
        tool_name: str | None = None,
        text_contains: str | None = None,
    ) -> list[tuple[int, RecordedStep]]:
        """
        Search for steps matching criteria.

        Args:
            step_type: Filter by step type
            tool_name: Filter by tool name (for tool calls/results)
            text_contains: Filter by text content

        Returns:
            List of (index, step) tuples
        """
        if self._session is None:
            return []

        results = []
        for i, step in enumerate(self._session.steps):
            if step_type and step.step_type != step_type:
                continue
            if tool_name and step.data.get("tool_name") != tool_name:
                continue
            if text_contains:
                step_str = json.dumps(step.data, default=str).lower()
                if text_contains.lower() not in step_str:
                    continue
            results.append((i, step))

        return results

    def export_session(self, output_path: str, format: str = "json") -> bool:
        """
        Export session to a file.

        Args:
            output_path: Output file path
            format: Export format (json, markdown)

        Returns:
            True if exported successfully
        """
        if self._session is None:
            return False

        try:
            if format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(self._session.to_dict(), f, indent=2, default=str)
            elif format == "markdown":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(self._session_to_markdown())
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to export session: {e}")
            return False

    def _session_to_markdown(self) -> str:
        """Convert session to markdown format."""
        if self._session is None:
            return ""

        lines = [
            f"# Session: {self._session.name}",
            f"",
            f"**Session ID:** {self._session.session_id}",
            f"**Created:** {self._session.created_at}",
            f"**Steps:** {len(self._session.steps)}",
            f"**Tags:** {', '.join(self._session.tags) or 'none'}",
            f"",
            f"## Steps",
            f"",
        ]

        for i, step in enumerate(self._session.steps):
            lines.append(f"### Step {i}: {step.step_type.value}")
            lines.append(f"**Timestamp:** {step.timestamp}")
            if step.duration_ms:
                lines.append(f"**Duration:** {step.duration_ms:.2f}ms")
            lines.append(f"")
            lines.append("```json")
            lines.append(json.dumps(step.data, indent=2, default=str))
            lines.append("```")
            lines.append("")

        return "\n".join(lines)
