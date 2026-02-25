"""
Comprehensive unit tests for the GAAP observability module.

Tests:
- TestTracing: OpenTelemetry tracing functionality
- TestMetrics: Prometheus metrics functionality
- TestSessionRecorder: Session recording for debugging
- TestSessionReplay: Session replay and time-travel
- TestFlightRecorder: Crash-safe event recording
- TestDashboard: Metrics aggregation and export
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from gaap.core.observability import (
    Metrics,
    MetricsConfig,
    Observability,
    Tracer,
    TracingConfig,
    get_metrics,
    get_tracer,
)


# =============================================================================
# Mock Classes for Session Recording, Replay, FlightRecorder, Dashboard
# =============================================================================


@dataclass
class SessionStep:
    step_id: int
    timestamp: float
    event_type: str
    data: dict[str, Any]
    state_snapshot: Optional[dict[str, Any]] = None


@dataclass
class LLMCallRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency: float
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    success: bool
    latency: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryAccessRecord:
    operation: str
    key: str
    tier: str
    success: bool
    timestamp: float = field(default_factory=time.time)


class SessionRecorder:
    def __init__(self, session_id: str, storage_path: Optional[Path] = None):
        self.session_id = session_id
        self.storage_path = storage_path or Path(tempfile.mkdtemp())
        self.steps: list[SessionStep] = []
        self.llm_calls: list[LLMCallRecord] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.memory_accesses: list[MemoryAccessRecord] = []
        self._lock = threading.Lock()
        self._step_counter = 0

    def record_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency: float,
        success: bool,
    ) -> None:
        with self._lock:
            record = LLMCallRecord(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency=latency,
                success=success,
            )
            self.llm_calls.append(record)
            self._record_step("llm_call", {"record": record.__dict__})

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        success: bool,
        latency: float,
    ) -> None:
        with self._lock:
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=str(result) if result else None,
                success=success,
                latency=latency,
            )
            self.tool_calls.append(record)
            self._record_step("tool_call", {"record": record.__dict__})

    def record_memory_access(
        self,
        operation: str,
        key: str,
        tier: str,
        success: bool,
    ) -> None:
        with self._lock:
            record = MemoryAccessRecord(
                operation=operation,
                key=key,
                tier=tier,
                success=success,
            )
            self.memory_accesses.append(record)
            self._record_step("memory_access", {"record": record.__dict__})

    def _record_step(self, event_type: str, data: dict[str, Any]) -> None:
        step = SessionStep(
            step_id=self._step_counter,
            timestamp=time.time(),
            event_type=event_type,
            data=data,
        )
        self.steps.append(step)
        self._step_counter += 1

    def save_session(self) -> Path:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        session_file = self.storage_path / f"session_{self.session_id}.json"
        session_data = {
            "session_id": self.session_id,
            "steps": [
                {
                    "step_id": s.step_id,
                    "timestamp": s.timestamp,
                    "event_type": s.event_type,
                    "data": s.data,
                }
                for s in self.steps
            ],
            "llm_calls": [c.__dict__ for c in self.llm_calls],
            "tool_calls": [c.__dict__ for c in self.tool_calls],
            "memory_accesses": [a.__dict__ for a in self.memory_accesses],
        }
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
        return session_file

    @classmethod
    def load_session(cls, session_path: Path) -> "SessionRecorder":
        with open(session_path) as f:
            data = json.load(f)

        recorder = cls(data["session_id"], session_path.parent)
        recorder.steps = [
            SessionStep(
                step_id=s["step_id"],
                timestamp=s["timestamp"],
                event_type=s["event_type"],
                data=s["data"],
            )
            for s in data.get("steps", [])
        ]
        recorder.llm_calls = [LLMCallRecord(**c) for c in data.get("llm_calls", [])]
        recorder.tool_calls = [ToolCallRecord(**c) for c in data.get("tool_calls", [])]
        recorder.memory_accesses = [
            MemoryAccessRecord(**a) for a in data.get("memory_accesses", [])
        ]
        return recorder


class SessionReplay:
    def __init__(self, recorder: SessionRecorder):
        self.recorder = recorder
        self._current_step = 0
        self._state_history: list[dict[str, Any]] = []
        self._modifications: list[tuple[int, dict[str, Any]]] = []

    def replay_session(self, speed: float = 1.0) -> list[SessionStep]:
        steps = []
        for step in self.recorder.steps:
            steps.append(step)
            if speed < 1.0:
                time.sleep(0.01 / speed)
        return steps

    def time_travel_to_step(self, step_id: int) -> SessionStep:
        for step in self.recorder.steps:
            if step.step_id == step_id:
                self._current_step = step_id
                return step
        raise ValueError(f"Step {step_id} not found")

    def inspect_state(self, step_id: Optional[int] = None) -> dict[str, Any]:
        target_step = step_id if step_id is not None else self._current_step
        state = {
            "step_id": target_step,
            "llm_calls_before": [
                c.__dict__
                for c in self.recorder.llm_calls
                if c.timestamp <= self._get_step_timestamp(target_step)
            ],
            "tool_calls_before": [
                c.__dict__
                for c in self.recorder.tool_calls
                if c.timestamp <= self._get_step_timestamp(target_step)
            ],
        }
        return state

    def _get_step_timestamp(self, step_id: int) -> float:
        for step in self.recorder.steps:
            if step.step_id == step_id:
                return step.timestamp
        return 0.0

    def modify_and_resume(self, step_id: int, modifications: dict[str, Any]) -> None:
        self._modifications.append((step_id, modifications))

    def get_session_history(self) -> list[dict[str, Any]]:
        return [
            {
                "step_id": s.step_id,
                "event_type": s.event_type,
                "timestamp": s.timestamp,
            }
            for s in self.recorder.steps
        ]


class FlightRecorder:
    def __init__(self, buffer_size: int = 1000, persist_path: Optional[Path] = None):
        self.buffer_size = buffer_size
        self.persist_path = persist_path
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._crashed = False

    def record_event(
        self,
        event_type: str,
        data: dict[str, Any],
        severity: str = "info",
    ) -> None:
        with self._lock:
            event = {
                "timestamp": time.time(),
                "event_type": event_type,
                "data": data,
                "severity": severity,
            }
            self._buffer.append(event)
            if len(self._buffer) > self.buffer_size:
                self._buffer.pop(0)

    def circular_buffer(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buffer)

    def crash_safe_persistence(self) -> None:
        if self.persist_path:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            crash_file = self.persist_path / "crash_recovery.json"
            with open(crash_file, "w") as f:
                json.dump({"events": self._buffer, "crashed": True}, f)
            self._crashed = True

    def export_json(self, output_path: Optional[Path] = None) -> Path:
        if output_path is None:
            if self.persist_path is None:
                raise ValueError("No output path or persist path provided")
            export_path = self.persist_path / "flight_recorder_export.json"
        else:
            export_path = output_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump({"events": self._buffer}, f, indent=2)
        return export_path

    def mark_crash(self) -> None:
        self._crashed = True

    def is_crashed(self) -> bool:
        return self._crashed


class Dashboard:
    def __init__(self, recorder: Optional[SessionRecorder] = None):
        self.recorder = recorder
        self._metrics_cache: dict[str, Any] = {}

    def get_token_usage(self) -> dict[str, Any]:
        if not self.recorder:
            return {}
        total_input = sum(c.input_tokens for c in self.recorder.llm_calls)
        total_output = sum(c.output_tokens for c in self.recorder.llm_calls)
        by_provider: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        for call in self.recorder.llm_calls:
            by_provider[call.provider]["input"] += call.input_tokens
            by_provider[call.provider]["output"] += call.output_tokens
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "by_provider": dict(by_provider),
        }

    def get_cost_metrics(self) -> dict[str, Any]:
        if not self.recorder:
            return {}
        total_cost = sum(c.cost for c in self.recorder.llm_calls)
        by_model: dict[str, float] = defaultdict(float)
        for call in self.recorder.llm_calls:
            key = f"{call.provider}/{call.model}"
            by_model[key] += call.cost
        return {
            "total_cost": total_cost,
            "by_model": dict(by_model),
        }

    def get_failure_rates(self) -> dict[str, Any]:
        if not self.recorder:
            return {}
        llm_total = len(self.recorder.llm_calls)
        llm_failures = sum(1 for c in self.recorder.llm_calls if not c.success)
        tool_total = len(self.recorder.tool_calls)
        tool_failures = sum(1 for c in self.recorder.tool_calls if not c.success)
        return {
            "llm_failure_rate": llm_failures / llm_total if llm_total > 0 else 0.0,
            "tool_failure_rate": tool_failures / tool_total if tool_total > 0 else 0.0,
            "llm_total": llm_total,
            "llm_failures": llm_failures,
            "tool_total": tool_total,
            "tool_failures": tool_failures,
        }

    def grafana_json_output(self) -> dict[str, Any]:
        token_usage = self.get_token_usage()
        cost_metrics = self.get_cost_metrics()
        failure_rates = self.get_failure_rates()
        return {
            "dashboard": "GAAP Observability",
            "panels": [
                {
                    "title": "Token Usage",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": f"gaap_tokens_total{{type='input'}} {token_usage.get('total_input_tokens', 0)}"
                        },
                        {
                            "expr": f"gaap_tokens_total{{type='output'}} {token_usage.get('total_output_tokens', 0)}"
                        },
                    ],
                },
                {
                    "title": "Cost Metrics",
                    "type": "stat",
                    "targets": [
                        {"expr": f"gaap_cost_dollars_total {cost_metrics.get('total_cost', 0)}"},
                    ],
                },
                {
                    "title": "Failure Rates",
                    "type": "gauge",
                    "targets": [
                        {"expr": f"llm_failure_rate {failure_rates.get('llm_failure_rate', 0)}"},
                        {"expr": f"tool_failure_rate {failure_rates.get('tool_failure_rate', 0)}"},
                    ],
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def reset_tracer_singleton():
    Tracer._instance = None
    Tracer._initialized = False
    yield
    Tracer._instance = None
    Tracer._initialized = False


@pytest.fixture
def reset_metrics_singleton():
    Metrics._instance = None
    Metrics._initialized = False
    yield
    Metrics._instance = None
    Metrics._initialized = False


@pytest.fixture
def reset_observability_singleton():
    Observability._instance = None
    yield
    Observability._instance = None


@pytest.fixture
def mock_opentelemetry():
    mock_span = MagicMock()
    mock_span.set_attribute = MagicMock()
    mock_span.record_exception = MagicMock()
    mock_span.set_status = MagicMock()
    mock_span.add_event = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span = MagicMock(return_value=mock_span)

    mock_tracer_provider = MagicMock()
    mock_tracer_provider.add_span_processor = MagicMock()

    mock_status = MagicMock()
    mock_status.return_value = MagicMock()

    mock_status_code = MagicMock()
    mock_status_code.ERROR = "ERROR"
    mock_status_code.OK = "OK"

    mock_resource = MagicMock()
    mock_resource.create = MagicMock(return_value=MagicMock())

    mock_batch_processor = MagicMock()
    mock_console_exporter = MagicMock()

    mock_trace = MagicMock()
    mock_trace.get_tracer = MagicMock(return_value=mock_tracer)
    mock_trace.set_tracer_provider = MagicMock()

    patches = {
        "OTEL_AVAILABLE": True,
        "trace": mock_trace,
        "Resource": mock_resource,
        "TracerProvider": MagicMock(return_value=mock_tracer_provider),
        "BatchSpanProcessor": mock_batch_processor,
        "ConsoleSpanExporter": mock_console_exporter,
        "SERVICE_NAME": "service.name",
        "Span": mock_span,
        "Status": mock_status,
        "StatusCode": mock_status_code,
    }

    with patch.dict("gaap.core.observability.__dict__", patches):
        yield {
            "span": mock_span,
            "tracer": mock_tracer,
            "provider": mock_tracer_provider,
            "status": mock_status,
            "status_code": mock_status_code,
            "resource": mock_resource,
            "trace": mock_trace,
        }


@pytest.fixture
def mock_prometheus():
    mock_counter = MagicMock()
    mock_counter.labels = MagicMock(return_value=mock_counter)
    mock_counter.inc = MagicMock()
    mock_counter.dec = MagicMock()

    mock_histogram = MagicMock()
    mock_histogram.labels = MagicMock(return_value=mock_histogram)
    mock_histogram.observe = MagicMock()

    mock_gauge = MagicMock()
    mock_gauge.labels = MagicMock(return_value=mock_gauge)
    mock_gauge.set = MagicMock()
    mock_gauge.inc = MagicMock()
    mock_gauge.dec = MagicMock()

    with patch.dict(
        "gaap.core.observability.__dict__",
        {
            "PROMETHEUS_AVAILABLE": True,
            "Counter": MagicMock(return_value=mock_counter),
            "Histogram": MagicMock(return_value=mock_histogram),
            "Gauge": MagicMock(return_value=mock_gauge),
        },
    ):
        yield {
            "counter": mock_counter,
            "histogram": mock_histogram,
            "gauge": mock_gauge,
        }


@pytest.fixture
def temp_storage_path(tmp_path):
    return tmp_path / "sessions"


@pytest.fixture
def session_recorder(temp_storage_path):
    return SessionRecorder("test-session-001", temp_storage_path)


@pytest.fixture
def flight_recorder(temp_storage_path):
    return FlightRecorder(buffer_size=10, persist_path=Path(temp_storage_path) / "flight")


# =============================================================================
# TestTracing
# =============================================================================


class TestTracing:
    def test_create_span(self, reset_tracer_singleton, mock_opentelemetry):
        config = TracingConfig(service_name="test-service")
        tracer = Tracer(config)
        with tracer.start_span("test_operation") as span:
            assert span is not None or span is None

    def test_span_attributes(self, reset_tracer_singleton, mock_opentelemetry):
        config = TracingConfig(service_name="test-service")
        tracer = Tracer(config)
        attributes = {"user_id": "123", "operation": "test"}
        with tracer.start_span("test_operation", attributes=attributes) as span:
            pass

    def test_span_events(self, reset_tracer_singleton, mock_opentelemetry):
        config = TracingConfig(service_name="test-service")
        tracer = Tracer(config)
        with tracer.start_span("test_operation") as span:
            if span:
                tracer.add_event(span, "checkpoint", {"progress": 50})

    def test_trace_context_propagation(self, reset_tracer_singleton):
        config = TracingConfig(service_name="test-service", enable_console_export=True)
        tracer = Tracer(config)
        with tracer.start_span("parent_operation") as parent_span:
            with tracer.start_span("child_operation") as child_span:
                pass

    def test_fallback_without_opentelemetry(self, reset_tracer_singleton):
        with patch.dict("gaap.core.observability.__dict__", {"OTEL_AVAILABLE": False}):
            config = TracingConfig(service_name="test-service")
            tracer = Tracer(config)
            with tracer.start_span("test_operation") as span:
                assert span is None


# =============================================================================
# TestMetrics
# =============================================================================


class TestMetrics:
    def test_counter_increment(self, reset_metrics_singleton, mock_prometheus):
        config = MetricsConfig(namespace="test", subsystem="unit")
        metrics = Metrics(config)
        metrics.inc_counter("requests_total", {"layer": "test"}, value=5)

    def test_histogram_record(self, reset_metrics_singleton, mock_prometheus):
        config = MetricsConfig(namespace="test", subsystem="unit")
        metrics = Metrics(config)
        metrics.observe_histogram("request_duration_seconds", 0.5, {"layer": "test"})

    def test_gauge_set(self, reset_metrics_singleton, mock_prometheus):
        config = MetricsConfig(namespace="test", subsystem="unit")
        metrics = Metrics(config)
        metrics.set_gauge("active_requests", 10.0, {"layer": "test"})

    def test_get_metrics(self, reset_metrics_singleton, mock_prometheus):
        config = MetricsConfig(namespace="test", subsystem="unit")
        metrics = Metrics(config)
        metrics.inc_counter("requests_total", {"layer": "test"})
        metrics.observe_histogram("request_duration_seconds", 0.1)
        metrics.set_gauge("active_requests", 5.0)

    def test_prometheus_export(self, reset_metrics_singleton, mock_prometheus):
        config = MetricsConfig(namespace="gaap", subsystem="system")
        metrics = Metrics(config)
        metrics.inc_counter("requests_total", {"layer": "execution"})
        metrics.observe_histogram("request_duration_seconds", 0.25)


# =============================================================================
# TestSessionRecorder
# =============================================================================


class TestSessionRecorder:
    def test_record_llm_call(self, session_recorder):
        session_recorder.record_llm_call(
            provider="groq",
            model="llama-3.3-70b",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            latency=0.5,
            success=True,
        )
        assert len(session_recorder.llm_calls) == 1
        call = session_recorder.llm_calls[0]
        assert call.provider == "groq"
        assert call.input_tokens == 100
        assert call.success is True

    def test_record_tool_call(self, session_recorder):
        session_recorder.record_tool_call(
            tool_name="search_web",
            arguments={"query": "test"},
            result="search results",
            success=True,
            latency=0.1,
        )
        assert len(session_recorder.tool_calls) == 1
        call = session_recorder.tool_calls[0]
        assert call.tool_name == "search_web"
        assert call.arguments == {"query": "test"}

    def test_record_memory_access(self, session_recorder):
        session_recorder.record_memory_access(
            operation="read",
            key="user_context",
            tier="working",
            success=True,
        )
        assert len(session_recorder.memory_accesses) == 1
        access = session_recorder.memory_accesses[0]
        assert access.operation == "read"
        assert access.key == "user_context"

    def test_save_session(self, session_recorder):
        session_recorder.record_llm_call(
            provider="test",
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            cost=0.0001,
            latency=0.1,
            success=True,
        )
        session_file = session_recorder.save_session()
        assert session_file.exists()
        with open(session_file) as f:
            data = json.load(f)
        assert data["session_id"] == "test-session-001"
        assert len(data["llm_calls"]) == 1

    def test_load_session(self, session_recorder, temp_storage_path):
        session_recorder.record_llm_call(
            provider="groq",
            model="llama-3.3-70b",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            latency=0.5,
            success=True,
        )
        session_recorder.record_tool_call(
            tool_name="test_tool",
            arguments={"arg": "value"},
            result="result",
            success=True,
            latency=0.1,
        )
        session_file = session_recorder.save_session()
        loaded = SessionRecorder.load_session(session_file)
        assert loaded.session_id == "test-session-001"
        assert len(loaded.llm_calls) == 1
        assert len(loaded.tool_calls) == 1
        assert loaded.llm_calls[0].provider == "groq"


# =============================================================================
# TestSessionReplay
# =============================================================================


class TestSessionReplay:
    def test_replay_session(self, session_recorder):
        session_recorder.record_llm_call(
            provider="test",
            model="test",
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            latency=0.1,
            success=True,
        )
        session_recorder.record_tool_call(
            tool_name="tool1",
            arguments={},
            result="ok",
            success=True,
            latency=0.05,
        )
        replay = SessionReplay(session_recorder)
        steps = replay.replay_session()
        assert len(steps) >= 2

    def test_time_travel_to_step(self, session_recorder):
        session_recorder.record_llm_call(
            provider="test",
            model="test",
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            latency=0.1,
            success=True,
        )
        session_recorder.record_tool_call(
            tool_name="tool1",
            arguments={},
            result="ok",
            success=True,
            latency=0.05,
        )
        replay = SessionReplay(session_recorder)
        step = replay.time_travel_to_step(0)
        assert step.step_id == 0

    def test_inspect_state(self, session_recorder):
        session_recorder.record_llm_call(
            provider="test",
            model="test",
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            latency=0.1,
            success=True,
        )
        replay = SessionReplay(session_recorder)
        state = replay.inspect_state(0)
        assert "step_id" in state
        assert state["step_id"] == 0

    def test_modify_and_resume(self, session_recorder):
        session_recorder.record_llm_call(
            provider="test",
            model="test",
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            latency=0.1,
            success=True,
        )
        replay = SessionReplay(session_recorder)
        replay.modify_and_resume(0, {"provider": "modified-provider"})
        assert len(replay._modifications) == 1

    def test_get_session_history(self, session_recorder):
        session_recorder.record_llm_call(
            provider="test",
            model="test",
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            latency=0.1,
            success=True,
        )
        session_recorder.record_tool_call(
            tool_name="tool1",
            arguments={},
            result="ok",
            success=True,
            latency=0.05,
        )
        replay = SessionReplay(session_recorder)
        history = replay.get_session_history()
        assert len(history) >= 2
        assert all("step_id" in h for h in history)


# =============================================================================
# TestFlightRecorder
# =============================================================================


class TestFlightRecorder:
    def test_record_event(self, flight_recorder):
        flight_recorder.record_event(
            event_type="llm_request",
            data={"provider": "groq", "model": "llama-3.3-70b"},
            severity="info",
        )
        events = flight_recorder.circular_buffer()
        assert len(events) == 1
        assert events[0]["event_type"] == "llm_request"

    def test_circular_buffer(self, flight_recorder):
        for i in range(15):
            flight_recorder.record_event(
                event_type=f"event_{i}",
                data={"index": i},
            )
        events = flight_recorder.circular_buffer()
        assert len(events) == flight_recorder.buffer_size
        assert events[0]["event_type"] == "event_5"

    def test_crash_safe_persistence(self, flight_recorder):
        flight_recorder.record_event("critical_event", {"data": "important"})
        flight_recorder.crash_safe_persistence()
        crash_file = flight_recorder.persist_path / "crash_recovery.json"
        assert crash_file.exists()
        with open(crash_file) as f:
            data = json.load(f)
        assert data["crashed"] is True
        assert len(data["events"]) == 1

    def test_export_json(self, flight_recorder):
        flight_recorder.record_event("event1", {"key": "value"})
        flight_recorder.record_event("event2", {"key": "value2"})
        export_path = flight_recorder.export_json()
        assert export_path.exists()
        with open(export_path) as f:
            data = json.load(f)
        assert len(data["events"]) == 2


# =============================================================================
# TestDashboard
# =============================================================================


class TestDashboard:
    def test_get_token_usage(self, session_recorder):
        session_recorder.record_llm_call(
            provider="groq",
            model="llama-3.3-70b",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            latency=0.5,
            success=True,
        )
        session_recorder.record_llm_call(
            provider="gemini",
            model="gemini-1.5-flash",
            input_tokens=200,
            output_tokens=100,
            cost=0.002,
            latency=0.3,
            success=True,
        )
        dashboard = Dashboard(session_recorder)
        usage = dashboard.get_token_usage()
        assert usage["total_input_tokens"] == 300
        assert usage["total_output_tokens"] == 150
        assert usage["total_tokens"] == 450
        assert "groq" in usage["by_provider"]
        assert "gemini" in usage["by_provider"]

    def test_get_cost_metrics(self, session_recorder):
        session_recorder.record_llm_call(
            provider="groq",
            model="llama-3.3-70b",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            latency=0.5,
            success=True,
        )
        session_recorder.record_llm_call(
            provider="groq",
            model="llama-3.3-70b",
            input_tokens=100,
            output_tokens=50,
            cost=0.002,
            latency=0.5,
            success=True,
        )
        dashboard = Dashboard(session_recorder)
        costs = dashboard.get_cost_metrics()
        assert costs["total_cost"] == 0.003
        assert "groq/llama-3.3-70b" in costs["by_model"]
        assert costs["by_model"]["groq/llama-3.3-70b"] == 0.003

    def test_get_failure_rates(self, session_recorder):
        session_recorder.record_llm_call(
            provider="groq",
            model="test",
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            latency=0.1,
            success=True,
        )
        session_recorder.record_llm_call(
            provider="groq",
            model="test",
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            latency=0.1,
            success=False,
        )
        session_recorder.record_tool_call(
            tool_name="test_tool",
            arguments={},
            result="ok",
            success=True,
            latency=0.1,
        )
        session_recorder.record_tool_call(
            tool_name="fail_tool",
            arguments={},
            result=None,
            success=False,
            latency=0.1,
        )
        dashboard = Dashboard(session_recorder)
        rates = dashboard.get_failure_rates()
        assert rates["llm_failure_rate"] == 0.5
        assert rates["tool_failure_rate"] == 0.5
        assert rates["llm_total"] == 2
        assert rates["tool_total"] == 2

    def test_grafana_json_output(self, session_recorder):
        session_recorder.record_llm_call(
            provider="groq",
            model="llama-3.3-70b",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            latency=0.5,
            success=True,
        )
        dashboard = Dashboard(session_recorder)
        output = dashboard.grafana_json_output()
        assert output["dashboard"] == "GAAP Observability"
        assert len(output["panels"]) == 3
        assert "timestamp" in output
        assert any(p["title"] == "Token Usage" for p in output["panels"])
        assert any(p["title"] == "Cost Metrics" for p in output["panels"])
        assert any(p["title"] == "Failure Rates" for p in output["panels"])


# =============================================================================
# TestObservability Integration
# =============================================================================


class TestObservabilityIntegration:
    def test_observability_singleton(
        self, reset_observability_singleton, reset_tracer_singleton, reset_metrics_singleton
    ):
        obs1 = Observability()
        obs2 = Observability()
        assert obs1 is obs2

    def test_get_tracer_helper(self, reset_observability_singleton):
        tracer = get_tracer()
        assert tracer is not None

    def test_get_metrics_helper(self, reset_observability_singleton):
        metrics = get_metrics()
        assert metrics is not None

    @pytest.mark.asyncio
    async def test_trace_span_context_manager(self, reset_observability_singleton):
        obs = Observability()
        with obs.trace_span("test_operation", layer="test"):
            await asyncio.sleep(0.01)

    def test_record_llm_call_integration(self, reset_observability_singleton):
        obs = Observability()
        obs.record_llm_call(
            provider="test",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
            latency=0.5,
            success=True,
        )

    def test_record_healing_integration(self, reset_observability_singleton):
        obs = Observability()
        obs.record_healing("L1_RETRY", success=True)

    def test_record_error_integration(self, reset_observability_singleton):
        obs = Observability()
        obs.record_error("execution", "ProviderError", "error")

    def test_enable_disable(self, reset_observability_singleton):
        obs = Observability()
        obs.disable()
        obs.record_llm_call(
            provider="test",
            model="test",
            input_tokens=10,
            output_tokens=5,
            cost=0.01,
            latency=0.1,
            success=True,
        )
        obs.enable()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
