"""
Comprehensive unit tests for the GAAP observability replay module.

Tests:
- TestSessionRecorder: Recording sessions, events, state management
- TestSessionReplay: Loading, replaying, state navigation
- TestRecordedStep: Step serialization and deserialization
- TestSessionState: State management and snapshots
- TestRecordedSession: Session serialization
- TestExportFormats: JSON and Markdown export functionality
- TestEdgeCases: Empty sessions, corrupted data, large recordings
"""

import json
import os
import tempfile
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest

from gaap.observability.replay import (
    StepType,
    RecordedStep,
    SessionState,
    RecordedSession,
    SessionRecorder,
    SessionReplay,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_storage_path(tmp_path: Path) -> Path:
    """Create a temporary storage path for sessions."""
    return tmp_path / "sessions"


@pytest.fixture
def recorder(temp_storage_path: Path) -> SessionRecorder:
    """Create a SessionRecorder with temporary storage."""
    return SessionRecorder(storage_path=str(temp_storage_path), auto_save=False)


@pytest.fixture
def sample_session_id(temp_storage_path: Path) -> str:
    """Create a sample recording session with some steps and return session ID."""
    recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
    session_id = recorder.start_session(
        name="test_session",
        initial_state={"var1": "value1", "count": 0},
        tags=["test", "sample"],
        metadata={"author": "test", "version": "1.0"},
    )
    # Add some steps
    recorder.record_llm_request(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
        params={"temperature": 0.7},
        provider="openai",
    )
    recorder.record_llm_response(
        response={"content": "Hi there!", "model": "gpt-4"},
        latency_ms=500.0,
    )
    recorder.record_tool_call(
        tool_name="search",
        arguments={"query": "test"},
    )
    recorder.record_tool_result(
        tool_name="search",
        result={"items": ["result1", "result2"]},
        success=True,
        latency_ms=100.0,
    )
    recorder.record_state_change(
        key="count",
        old_value=0,
        new_value=1,
        reason="increment",
    )
    recorder.end_session(notes="Test session completed")
    return session_id


@pytest.fixture
def loaded_replay(temp_storage_path: Path, sample_session_id: str) -> SessionReplay:
    """Create and load a session for replay testing."""
    replay = SessionReplay(storage_path=str(temp_storage_path))
    result = replay.load_session(sample_session_id)
    assert result, f"Failed to load session {sample_session_id}"
    return replay


@pytest.fixture
def mock_datetime():
    """Mock datetime for deterministic timestamps."""
    fixed_time = datetime(2024, 1, 15, 10, 30, 0)
    with patch("gaap.observability.replay.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_time
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock_dt


@pytest.fixture
def mock_time():
    """Mock time for deterministic timing."""
    with patch("gaap.observability.replay.time") as mock_t:
        mock_t.time.return_value = 1705312200.0
        yield mock_t


# =============================================================================
# Test RecordedStep
# =============================================================================


class TestRecordedStep:
    """Tests for RecordedStep dataclass."""

    def test_recorded_step_creation(self):
        """Test creating a RecordedStep."""
        step = RecordedStep(
            step_id="step_001",
            step_type=StepType.LLM_REQUEST,
            timestamp="2024-01-15T10:30:00",
            data={"messages": [{"role": "user", "content": "Hello"}]},
            state_snapshot={"var": "value"},
            metadata={"source": "test"},
            parent_step_id="step_000",
            duration_ms=100.0,
        )
        assert step.step_id == "step_001"
        assert step.step_type == StepType.LLM_REQUEST
        assert step.timestamp == "2024-01-15T10:30:00"
        assert step.data["messages"][0]["content"] == "Hello"
        assert step.state_snapshot == {"var": "value"}
        assert step.metadata == {"source": "test"}
        assert step.parent_step_id == "step_000"
        assert step.duration_ms == 100.0

    def test_recorded_step_to_dict(self):
        """Test converting RecordedStep to dictionary."""
        step = RecordedStep(
            step_id="step_001",
            step_type=StepType.TOOL_CALL,
            timestamp="2024-01-15T10:30:00",
            data={"tool_name": "search", "args": {}},
        )
        d = step.to_dict()
        assert d["step_id"] == "step_001"
        assert d["step_type"] == "tool_call"
        assert d["timestamp"] == "2024-01-15T10:30:00"
        assert d["data"]["tool_name"] == "search"
        assert d["state_snapshot"] == {}
        assert d["metadata"] == {}

    def test_recorded_step_from_dict(self):
        """Test creating RecordedStep from dictionary."""
        data = {
            "step_id": "step_002",
            "step_type": "llm_response",
            "timestamp": "2024-01-15T10:31:00",
            "data": {"response": "Hello!"},
            "state_snapshot": {"last_message": "Hello"},
            "metadata": {"model": "gpt-4"},
            "parent_step_id": "step_001",
            "duration_ms": 250.0,
        }
        step = RecordedStep.from_dict(data)
        assert step.step_id == "step_002"
        assert step.step_type == StepType.LLM_RESPONSE
        assert step.timestamp == "2024-01-15T10:31:00"
        assert step.data["response"] == "Hello!"
        assert step.state_snapshot["last_message"] == "Hello"
        assert step.metadata["model"] == "gpt-4"
        assert step.parent_step_id == "step_001"
        assert step.duration_ms == 250.0

    def test_recorded_step_all_step_types(self):
        """Test RecordedStep with all step types."""
        for step_type in StepType:
            step = RecordedStep(
                step_id=f"step_{step_type.value}",
                step_type=step_type,
                timestamp="2024-01-15T10:30:00",
                data={"test": True},
            )
            d = step.to_dict()
            restored = RecordedStep.from_dict(d)
            assert restored.step_type == step_type


# =============================================================================
# Test SessionState
# =============================================================================


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_session_state_creation(self):
        """Test creating a SessionState."""
        state = SessionState(
            step_index=5,
            total_steps=10,
            messages=[{"role": "user", "content": "Hello"}],
            tool_results={"search": {"items": []}},
            variables={"count": 5},
            memory={"context": "test"},
            ooda_state={"phase": "observe"},
            task_context={"task_id": "123"},
            healing_history=[{"error": "fixed"}],
            axiom_violations=[{"axiom": "test"}],
            metrics={"latency": 100},
        )
        assert state.step_index == 5
        assert state.total_steps == 10
        assert len(state.messages) == 1
        assert state.tool_results["search"]["items"] == []
        assert state.variables["count"] == 5

    def test_session_state_defaults(self):
        """Test SessionState with default values."""
        state = SessionState(
            step_index=0,
            total_steps=1,
            messages=[],
            tool_results={},
            variables={},
            memory={},
        )
        assert state.ooda_state == {}
        assert state.task_context == {}
        assert state.healing_history == []
        assert state.axiom_violations == []
        assert state.metrics == {}

    def test_session_state_to_dict(self):
        """Test converting SessionState to dictionary."""
        state = SessionState(
            step_index=1,
            total_steps=5,
            messages=[{"role": "assistant", "content": "Hi"}],
            tool_results={"calc": 42},
            variables={"x": 1},
            memory={},
        )
        d = state.to_dict()
        assert d["step_index"] == 1
        assert d["total_steps"] == 5
        assert d["messages"][0]["content"] == "Hi"
        assert d["tool_results"]["calc"] == 42

    def test_session_state_from_dict(self):
        """Test creating SessionState from dictionary."""
        data = {
            "step_index": 3,
            "total_steps": 8,
            "messages": [{"role": "user", "content": "Test"}],
            "tool_results": {"tool1": "result"},
            "variables": {"key": "value"},
            "memory": {},
            "ooda_state": {"observe": True},
            "task_context": {"id": "task1"},
            "healing_history": [],
            "axiom_violations": [],
            "metrics": {"calls": 10},
        }
        state = SessionState.from_dict(data)
        assert state.step_index == 3
        assert state.total_steps == 8
        assert state.messages[0]["content"] == "Test"
        assert state.variables["key"] == "value"
        assert state.ooda_state["observe"] is True


# =============================================================================
# Test RecordedSession
# =============================================================================


class TestRecordedSession:
    """Tests for RecordedSession dataclass."""

    def test_recorded_session_creation(self):
        """Test creating a RecordedSession."""
        step = RecordedStep(
            step_id="step_0",
            step_type=StepType.SESSION_START,
            timestamp="2024-01-15T10:00:00",
            data={},
        )
        session = RecordedSession(
            session_id="sess_001",
            name="Test Session",
            created_at="2024-01-15T10:00:00",
            ended_at="2024-01-15T10:30:00",
            steps=[step],
            initial_state={"var": "init"},
            final_state={"var": "final"},
            metadata={"author": "test"},
            tags=["test", "sample"],
            notes="Test notes",
        )
        assert session.session_id == "sess_001"
        assert session.name == "Test Session"
        assert len(session.steps) == 1
        assert session.initial_state == {"var": "init"}
        assert session.final_state == {"var": "final"}
        assert session.tags == ["test", "sample"]
        assert session.notes == "Test notes"

    def test_recorded_session_to_dict(self):
        """Test converting RecordedSession to dictionary."""
        step = RecordedStep(
            step_id="step_0",
            step_type=StepType.SESSION_START,
            timestamp="2024-01-15T10:00:00",
            data={"name": "test"},
        )
        session = RecordedSession(
            session_id="sess_001",
            name="Test",
            created_at="2024-01-15T10:00:00",
            ended_at=None,
            steps=[step],
            initial_state={},
            final_state=None,
            metadata={},
        )
        d = session.to_dict()
        assert d["session_id"] == "sess_001"
        assert d["name"] == "Test"
        assert d["ended_at"] is None
        assert len(d["steps"]) == 1
        assert d["steps"][0]["step_type"] == "session_start"

    def test_recorded_session_from_dict(self):
        """Test creating RecordedSession from dictionary."""
        data = {
            "session_id": "sess_002",
            "name": "Loaded Session",
            "created_at": "2024-01-15T11:00:00",
            "ended_at": "2024-01-15T11:30:00",
            "steps": [
                {
                    "step_id": "step_0",
                    "step_type": "session_start",
                    "timestamp": "2024-01-15T11:00:00",
                    "data": {},
                    "state_snapshot": {},
                    "metadata": {},
                }
            ],
            "initial_state": {"initialized": True},
            "final_state": {"completed": True},
            "metadata": {"version": "2.0"},
            "tags": ["production"],
            "notes": "Production run",
        }
        session = RecordedSession.from_dict(data)
        assert session.session_id == "sess_002"
        assert session.name == "Loaded Session"
        assert session.ended_at == "2024-01-15T11:30:00"
        assert len(session.steps) == 1
        assert session.steps[0].step_type == StepType.SESSION_START
        assert session.initial_state["initialized"] is True
        assert session.tags == ["production"]


# =============================================================================
# Test SessionRecorder
# =============================================================================


class TestSessionRecorder:
    """Tests for SessionRecorder class."""

    def test_init_default_storage(self):
        """Test SessionRecorder initialization with default storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(SessionRecorder, "DEFAULT_STORAGE_PATH", ".gaap/sessions"):
                recorder = SessionRecorder()
                assert recorder.auto_save is True
                assert recorder.max_steps == 10000
                assert recorder._current_session is None

    def test_init_custom_storage(self, temp_storage_path: Path):
        """Test SessionRecorder initialization with custom storage."""
        recorder = SessionRecorder(
            storage_path=str(temp_storage_path),
            auto_save=False,
            max_steps=5000,
        )
        assert recorder.storage_path == temp_storage_path
        assert recorder.auto_save is False
        assert recorder.max_steps == 5000
        assert temp_storage_path.exists()

    def test_start_session(self, recorder: SessionRecorder):
        """Test starting a recording session."""
        session_id = recorder.start_session(
            name="my_test_session",
            initial_state={"counter": 0, "name": "test"},
            tags=["debug", "feature_x"],
            metadata={"version": "1.0", "env": "test"},
        )
        assert session_id is not None
        assert recorder._current_session is not None
        assert recorder._current_session.name == "my_test_session"
        assert recorder._current_session.initial_state == {"counter": 0, "name": "test"}
        assert recorder._current_session.tags == ["debug", "feature_x"]
        assert recorder._current_session.metadata == {"version": "1.0", "env": "test"}
        assert len(recorder._current_session.steps) == 1
        assert recorder._current_session.steps[0].step_type == StepType.SESSION_START

    def test_start_session_generates_unique_ids(self, recorder: SessionRecorder):
        """Test that start_session generates unique session IDs."""
        id1 = recorder.start_session("session1")
        recorder.end_session()
        id2 = recorder.start_session("session2")
        assert id1 != id2

    def test_end_session(self, temp_storage_path: Path):
        """Test ending a recording session."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        recorder.start_session("test_session")
        recorder.end_session(notes="Session completed successfully")

        # Session should be cleared
        assert recorder._current_session is None

        # Check saved file
        session_files = list(temp_storage_path.glob("*.json"))
        assert len(session_files) == 1

        with open(session_files[0]) as f:
            data = json.load(f)
        assert data["name"] == "test_session"
        assert data["notes"] == "Session completed successfully"
        assert data["ended_at"] is not None
        assert len(data["steps"]) >= 2  # SESSION_START + SESSION_END

    def test_end_session_no_active_session(self, recorder: SessionRecorder, caplog):
        """Test ending session when no session is active."""
        with caplog.at_level("WARNING"):
            recorder.end_session()
        assert "No active session to end" in caplog.text

    def test_record_llm_request(self, recorder: SessionRecorder):
        """Test recording an LLM request."""
        session_id = recorder.start_session("test")
        step_id = recorder.record_llm_request(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            model="gpt-4",
            params={"temperature": 0.7, "max_tokens": 100},
            provider="openai",
        )
        assert step_id is not None
        assert step_id.startswith(session_id)
        assert len(recorder._current_session.steps) == 2  # start + request
        step = recorder._current_session.steps[-1]
        assert step.step_type == StepType.LLM_REQUEST
        assert step.data["model"] == "gpt-4"
        assert step.data["provider"] == "openai"
        assert len(step.data["messages"]) == 2
        assert recorder._current_state["llm_requests"] == [step_id]

    def test_record_llm_request_no_session(self, recorder: SessionRecorder):
        """Test recording LLM request without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.record_llm_request([{"role": "user", "content": "Hello"}], "gpt-4")

    def test_record_llm_response(self, recorder: SessionRecorder):
        """Test recording an LLM response."""
        recorder.start_session("test")
        request_id = recorder.record_llm_request(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
        )
        step_id = recorder.record_llm_response(
            response={
                "content": "Hi there! How can I help?",
                "finish_reason": "stop",
            },
            latency_ms=523.5,
            request_step_id=request_id,
        )
        assert step_id is not None
        step = recorder._current_session.steps[-1]
        assert step.step_type == StepType.LLM_RESPONSE
        assert step.data["response"]["content"] == "Hi there! How can I help?"
        assert step.duration_ms == 523.5
        assert step.parent_step_id == request_id
        assert recorder._current_state["last_llm_response"] == step_id

    def test_record_llm_response_no_session(self, recorder: SessionRecorder):
        """Test recording LLM response without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.record_llm_response({"content": "Hi"})

    def test_record_tool_call(self, recorder: SessionRecorder):
        """Test recording a tool call."""
        session_id = recorder.start_session("test")
        step_id = recorder.record_tool_call(
            tool_name="file_search",
            arguments={"pattern": "*.py", "path": "/project"},
            step_id="custom_step_id",
        )
        assert step_id == "custom_step_id"
        step = recorder._current_session.steps[-1]
        assert step.step_type == StepType.TOOL_CALL
        assert step.data["tool_name"] == "file_search"
        assert step.data["arguments"]["pattern"] == "*.py"
        assert recorder._current_state["tool_calls"] == ["custom_step_id"]

    def test_record_tool_call_auto_id(self, recorder: SessionRecorder):
        """Test recording a tool call with auto-generated ID."""
        session_id = recorder.start_session("test")
        step_id = recorder.record_tool_call(
            tool_name="calculator",
            arguments={"expr": "2+2"},
        )
        assert step_id is not None
        assert step_id.startswith(session_id)

    def test_record_tool_call_no_session(self, recorder: SessionRecorder):
        """Test recording tool call without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.record_tool_call("search", {})

    def test_record_tool_result(self, recorder: SessionRecorder):
        """Test recording a tool result."""
        recorder.start_session("test")
        call_id = recorder.record_tool_call("search", {"query": "python"})
        step_id = recorder.record_tool_result(
            tool_name="search",
            result={"hits": ["result1", "result2"]},
            success=True,
            latency_ms=150.0,
            call_step_id=call_id,
        )
        assert step_id is not None
        step = recorder._current_session.steps[-1]
        assert step.step_type == StepType.TOOL_RESULT
        assert step.data["tool_name"] == "search"
        assert step.data["success"] is True
        assert step.parent_step_id == call_id
        assert step.duration_ms == 150.0
        assert recorder._current_state["tool_results"]["search"]["result"]["hits"] == [
            "result1",
            "result2",
        ]

    def test_record_tool_result_failure(self, recorder: SessionRecorder):
        """Test recording a failed tool result."""
        recorder.start_session("test")
        recorder.record_tool_call("risky_op", {})
        step_id = recorder.record_tool_result(
            tool_name="risky_op",
            result="Error: Connection refused",
            success=False,
            latency_ms=50.0,
        )
        step = recorder._current_session.steps[-1]
        assert step.data["success"] is False
        assert recorder._current_state["tool_results"]["risky_op"]["success"] is False

    def test_record_tool_result_no_session(self, recorder: SessionRecorder):
        """Test recording tool result without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.record_tool_result("tool", "result")

    def test_record_state_change(self, recorder: SessionRecorder):
        """Test recording a state change."""
        recorder.start_session("test")
        step_id = recorder.record_state_change(
            key="user_name",
            old_value=None,
            new_value="John Doe",
            reason="User login",
        )
        assert step_id is not None
        step = recorder._current_session.steps[-1]
        assert step.step_type == StepType.STATE_CHANGE
        assert step.data["key"] == "user_name"
        assert step.data["old_value"] is None
        assert step.data["new_value"] == "John Doe"
        assert step.data["reason"] == "User login"
        assert recorder._current_state["user_name"] == "John Doe"

    def test_record_state_change_no_session(self, recorder: SessionRecorder):
        """Test recording state change without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.record_state_change("key", "old", "new")

    def test_record_error_with_exception(self, recorder: SessionRecorder):
        """Test recording an error with exception."""
        recorder.start_session("test")
        try:
            raise ValueError("Something went wrong")
        except ValueError as e:
            step_id = recorder.record_error(
                error=e,
                context={"operation": "data_processing", "input": "invalid"},
            )
        step = recorder._current_session.steps[-1]
        assert step.step_type == StepType.ERROR
        assert step.data["error_type"] == "ValueError"
        assert step.data["error_message"] == "Something went wrong"
        assert step.data["context"]["operation"] == "data_processing"
        assert recorder._current_state["errors"] == [step_id]

    def test_record_error_with_string(self, recorder: SessionRecorder):
        """Test recording an error with string."""
        recorder.start_session("test")
        step_id = recorder.record_error("Simple error message")
        step = recorder._current_session.steps[-1]
        assert step.data["error_type"] == "string"
        assert step.data["error_message"] == "Simple error message"

    def test_record_error_no_session(self, recorder: SessionRecorder):
        """Test recording error without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.record_error("error")

    def test_record_ooda_phase(self, recorder: SessionRecorder):
        """Test recording an OODA phase."""
        recorder.start_session("test")
        step_id = recorder.record_ooda_phase(
            phase="observe",
            iteration=1,
            details={"sensors": ["visual", "auditory"]},
        )
        step = recorder._current_session.steps[-1]
        assert step.step_type == StepType.OODA_PHASE
        assert step.data["phase"] == "observe"
        assert step.data["iteration"] == 1
        assert step.data["details"]["sensors"] == ["visual", "auditory"]

    def test_record_ooda_phase_no_session(self, recorder: SessionRecorder):
        """Test recording OODA phase without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.record_ooda_phase("orient", 1)

    def test_create_checkpoint(self, recorder: SessionRecorder):
        """Test creating a checkpoint."""
        recorder.start_session("test")
        checkpoint_id = recorder.create_checkpoint(name="before_critical_op")
        assert checkpoint_id is not None
        assert "checkpoint" in checkpoint_id
        step = recorder._current_session.steps[-1]
        assert step.step_type == StepType.CHECKPOINT
        assert step.data["name"] == "before_critical_op"

    def test_create_checkpoint_no_session(self, recorder: SessionRecorder):
        """Test creating checkpoint without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.create_checkpoint()

    def test_save_session_explicit(self, recorder: SessionRecorder, temp_storage_path: Path):
        """Test explicitly saving a session."""
        recorder.start_session("test_save")
        recorder.record_llm_request([{"role": "user", "content": "Hi"}], "gpt-4")
        filepath = recorder.save_session()

        assert Path(filepath).exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["name"] == "test_save"
        assert len(data["steps"]) == 2

    def test_save_session_no_active_session(self, recorder: SessionRecorder):
        """Test saving without active session."""
        with pytest.raises(RuntimeError, match="No active session"):
            recorder.save_session()

    def test_get_current_session_id(self, recorder: SessionRecorder):
        """Test getting current session ID."""
        assert recorder.get_current_session_id() is None
        session_id = recorder.start_session("test")
        assert recorder.get_current_session_id() == session_id
        recorder.end_session()
        assert recorder.get_current_session_id() is None

    def test_get_current_state(self, recorder: SessionRecorder):
        """Test getting current state."""
        assert recorder.get_current_state() == {}
        recorder.start_session("test", initial_state={"count": 0})
        assert recorder.get_current_state() == {"count": 0}
        recorder.record_state_change("count", 0, 1)
        assert recorder.get_current_state()["count"] == 1


# =============================================================================
# Test SessionReplay
# =============================================================================


class TestSessionReplay:
    """Tests for SessionReplay class."""

    def test_init_default_storage(self):
        """Test SessionReplay initialization with default storage."""
        replay = SessionReplay()
        assert replay._session is None
        assert replay._current_step_index == 0

    def test_init_custom_storage(self, temp_storage_path: Path):
        """Test SessionReplay initialization with custom storage."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.storage_path == temp_storage_path

    def test_list_sessions_empty(self, temp_storage_path: Path):
        """Test listing sessions with empty storage."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        sessions = replay.list_sessions()
        assert sessions == []

    def test_list_sessions_with_data(self, temp_storage_path: Path):
        """Test listing available sessions."""
        # Create a session
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        recorder.start_session("session1", tags=["test"])
        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        sessions = replay.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["name"] == "session1"
        assert sessions[0]["tags"] == ["test"]
        assert "step_count" in sessions[0]

    def test_list_sessions_invalid_json(self, temp_storage_path: Path, caplog):
        """Test listing sessions with invalid JSON file."""
        temp_storage_path.mkdir(parents=True, exist_ok=True)
        # Create invalid JSON file
        invalid_file = temp_storage_path / "invalid.json"
        invalid_file.write_text("not valid json")

        replay = SessionReplay(storage_path=str(temp_storage_path))
        with caplog.at_level("WARNING"):
            sessions = replay.list_sessions()
        assert sessions == []
        assert "Failed to load session" in caplog.text

    def test_load_session_success(self, temp_storage_path: Path):
        """Test loading a session successfully."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("test_load")
        recorder.record_llm_request([{"role": "user", "content": "Hello"}], "gpt-4")
        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        result = replay.load_session(session_id)
        assert result is True
        assert replay._session is not None
        assert replay._session.session_id == session_id
        assert len(replay._session.steps) >= 2

    def test_load_session_not_found(self, temp_storage_path: Path, caplog):
        """Test loading a non-existent session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        with caplog.at_level("ERROR"):
            result = replay.load_session("nonexistent_session")
        assert result is False
        assert "Session not found" in caplog.text

    def test_load_session_partial_id(self, temp_storage_path: Path):
        """Test loading a session with partial ID match."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("test_partial")
        recorder.end_session()

        # Use only first part of session ID
        partial_id = session_id.split("_")[0]

        replay = SessionReplay(storage_path=str(temp_storage_path))
        result = replay.load_session(partial_id)
        assert result is True

    def test_load_session_corrupted(self, temp_storage_path: Path, caplog):
        """Test loading a corrupted session file."""
        temp_storage_path.mkdir(parents=True, exist_ok=True)
        corrupted_file = temp_storage_path / "corrupted_session.json"
        corrupted_file.write_text('{"session_id": "test", "invalid":}')

        replay = SessionReplay(storage_path=str(temp_storage_path))
        with caplog.at_level("ERROR"):
            result = replay.load_session("corrupted_session")
        assert result is False
        assert "Failed to load session" in caplog.text

    def test_step_forward(self, loaded_replay: SessionReplay):
        """Test stepping forward through session."""
        # Start at step 0
        assert loaded_replay._current_step_index == 0

        # Step forward
        step1 = loaded_replay.step_forward()
        assert step1 is not None
        assert loaded_replay._current_step_index == 1

        # Continue stepping
        while loaded_replay.step_forward():
            pass

        # Should be at last step
        assert loaded_replay._current_step_index == len(loaded_replay._session.steps) - 1

        # Step past end returns None
        result = loaded_replay.step_forward()
        assert result is None

    def test_step_forward_no_session(self, temp_storage_path: Path):
        """Test step_forward without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.step_forward() is None

    def test_step_backward(self, loaded_replay: SessionReplay):
        """Test stepping backward through session."""
        # Move to end
        while loaded_replay.step_forward():
            pass

        last_index = loaded_replay._current_step_index
        assert last_index > 0

        # Step backward
        step = loaded_replay.step_backward()
        assert step is not None
        assert loaded_replay._current_step_index == last_index - 1

        # Go back to beginning
        while loaded_replay.step_backward():
            pass

        assert loaded_replay._current_step_index == 0

        # Step before start returns None
        result = loaded_replay.step_backward()
        assert result is None

    def test_step_backward_no_session(self, temp_storage_path: Path):
        """Test step_backward without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.step_backward() is None

    def test_jump_to_step(self, loaded_replay: SessionReplay):
        """Test jumping to a specific step."""
        step = loaded_replay.jump_to_step(3)
        assert step is not None
        assert loaded_replay._current_step_index == 3

    def test_jump_to_step_invalid(self, loaded_replay: SessionReplay):
        """Test jumping to invalid step indices."""
        assert loaded_replay.jump_to_step(-1) is None
        assert loaded_replay.jump_to_step(1000) is None

    def test_jump_to_step_no_session(self, temp_storage_path: Path):
        """Test jump_to_step without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.jump_to_step(0) is None

    def test_jump_to_checkpoint_by_name(self, temp_storage_path: Path):
        """Test jumping to a checkpoint by name."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("test_checkpoints")
        recorder.create_checkpoint(name="checkpoint_a")
        recorder.record_llm_request([{"role": "user", "content": "Hello"}], "gpt-4")
        recorder.create_checkpoint(name="checkpoint_b")
        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        result = replay.jump_to_checkpoint("checkpoint_b")
        assert result is not None
        assert result.data["name"] == "checkpoint_b"

    def test_jump_to_checkpoint_last(self, temp_storage_path: Path):
        """Test jumping to last checkpoint."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("test")
        recorder.create_checkpoint(name="first")
        recorder.create_checkpoint(name="second")
        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        result = replay.jump_to_checkpoint()  # No name = last checkpoint
        assert result is not None
        assert result.data["name"] == "second"

    def test_jump_to_checkpoint_not_found(self, loaded_replay: SessionReplay):
        """Test jumping to non-existent checkpoint."""
        assert loaded_replay.jump_to_checkpoint("nonexistent") is None

    def test_jump_to_checkpoint_no_checkpoints(self, temp_storage_path: Path):
        """Test jumping to checkpoint when none exist."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("test_no_checkpoints")
        recorder.record_llm_request([{"role": "user", "content": "Hello"}], "gpt-4")
        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        assert replay.jump_to_checkpoint() is None

    def test_get_current_step(self, loaded_replay: SessionReplay):
        """Test getting current step."""
        step = loaded_replay.get_current_step()
        assert step is not None
        assert step.step_type == StepType.SESSION_START

        loaded_replay.step_forward()
        step = loaded_replay.get_current_step()
        assert step.step_type == StepType.LLM_REQUEST

    def test_get_current_step_no_session(self, temp_storage_path: Path):
        """Test get_current_step without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.get_current_step() is None

    def test_get_current_state(self, loaded_replay: SessionReplay):
        """Test getting current replay state."""
        state = loaded_replay.get_current_state()
        assert state is not None
        assert state.step_index == 0
        assert state.total_steps == len(loaded_replay._session.steps)

        # Step forward and check state updates
        loaded_replay.step_forward()  # LLM_REQUEST
        state = loaded_replay.get_current_state()
        assert len(state.messages) == 1

        loaded_replay.step_forward()  # LLM_RESPONSE
        state = loaded_replay.get_current_state()
        assert len(state.messages) == 2

        loaded_replay.step_forward()  # TOOL_CALL
        loaded_replay.step_forward()  # TOOL_RESULT
        state = loaded_replay.get_current_state()
        assert "search" in state.tool_results

    def test_get_current_state_no_session(self, temp_storage_path: Path):
        """Test get_current_state without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.get_current_state() is None

    def test_modify_state(self, loaded_replay: SessionReplay):
        """Test modifying replay state."""
        loaded_replay.modify_state("custom_key", "custom_value")
        state = loaded_replay.get_current_state()
        assert state.variables["custom_key"] == "custom_value"

        loaded_replay.modify_state("count", 999)
        state = loaded_replay.get_current_state()
        assert state.variables["count"] == 999

    def test_modify_state_no_session(self, temp_storage_path: Path):
        """Test modify_state without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.modify_state("key", "value")  # Should not raise
        assert replay._modified_state is None

    def test_get_all_steps(self, loaded_replay: SessionReplay):
        """Test getting all steps."""
        steps = loaded_replay.get_all_steps()
        assert len(steps) == len(loaded_replay._session.steps)
        assert steps[0].step_type == StepType.SESSION_START

    def test_get_all_steps_no_session(self, temp_storage_path: Path):
        """Test get_all_steps without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.get_all_steps() == []

    def test_get_step_count(self, loaded_replay: SessionReplay):
        """Test getting step count."""
        assert loaded_replay.get_step_count() == len(loaded_replay._session.steps)

    def test_get_step_count_no_session(self, temp_storage_path: Path):
        """Test get_step_count without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.get_step_count() == 0

    def test_get_current_index(self, loaded_replay: SessionReplay):
        """Test getting current index."""
        assert loaded_replay.get_current_index() == 0
        loaded_replay.step_forward()
        assert loaded_replay.get_current_index() == 1

    def test_search_steps_by_type(self, loaded_replay: SessionReplay):
        """Test searching steps by type."""
        results = loaded_replay.search_steps(step_type=StepType.LLM_REQUEST)
        assert len(results) == 1
        assert results[0][1].step_type == StepType.LLM_REQUEST

    def test_search_steps_by_tool_name(self, loaded_replay: SessionReplay):
        """Test searching steps by tool name."""
        results = loaded_replay.search_steps(tool_name="search")
        assert len(results) == 2  # TOOL_CALL and TOOL_RESULT

    def test_search_steps_by_text(self, loaded_replay: SessionReplay):
        """Test searching steps by text content."""
        results = loaded_replay.search_steps(text_contains="gpt-4")
        assert len(results) >= 1

    def test_search_steps_combined(self, loaded_replay: SessionReplay):
        """Test searching with combined criteria."""
        results = loaded_replay.search_steps(
            step_type=StepType.TOOL_CALL,
            tool_name="search",
        )
        assert len(results) == 1

    def test_search_steps_no_session(self, temp_storage_path: Path):
        """Test search_steps without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        assert replay.search_steps() == []


# =============================================================================
# Test Export Formats
# =============================================================================


class TestExportFormats:
    """Tests for export functionality."""

    def test_export_json(
        self, temp_storage_path: Path, loaded_replay: SessionReplay, tmp_path: Path
    ):
        """Test exporting session to JSON."""
        output_path = tmp_path / "exported.json"
        result = loaded_replay.export_session(str(output_path), format="json")
        assert result is True
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
        assert data["session_id"] == loaded_replay._session.session_id
        assert "steps" in data

    def test_export_markdown(self, loaded_replay: SessionReplay, tmp_path: Path):
        """Test exporting session to Markdown."""
        output_path = tmp_path / "exported.md"
        result = loaded_replay.export_session(str(output_path), format="markdown")
        assert result is True
        assert output_path.exists()

        content = output_path.read_text()
        assert "# Session:" in content
        assert "Session ID:" in content
        assert "## Steps" in content
        assert "```json" in content

    def test_export_invalid_format(self, loaded_replay: SessionReplay, tmp_path: Path):
        """Test exporting with invalid format."""
        output_path = tmp_path / "exported.txt"
        result = loaded_replay.export_session(str(output_path), format="txt")
        assert result is False

    def test_export_no_session(self, temp_storage_path: Path, tmp_path: Path):
        """Test exporting without loaded session."""
        replay = SessionReplay(storage_path=str(temp_storage_path))
        output_path = tmp_path / "exported.json"
        result = replay.export_session(str(output_path), format="json")
        assert result is False

    def test_export_write_error(self, loaded_replay: SessionReplay, tmp_path: Path):
        """Test handling export write error."""
        # Try to write to a read-only directory
        output_path = "/nonexistent_dir/exported.json"
        result = loaded_replay.export_session(output_path, format="json")
        assert result is False

    def test_session_to_markdown_content(self, temp_storage_path: Path):
        """Test markdown content generation."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session(
            name="markdown_test",
            tags=["test", "export"],
        )
        recorder.record_llm_request(
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4",
        )
        recorder.record_llm_response(
            response={"content": "Response"},
            latency_ms=100.0,
        )
        recorder.end_session(notes="Test notes")

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        md = replay._session_to_markdown()
        assert "# Session: markdown_test" in md
        assert "**Tags:** test, export" in md
        assert "### Step" in md
        assert "**Timestamp:**" in md
        assert "**Duration:**" in md


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_session(self, temp_storage_path: Path):
        """Test handling of empty session (only start and end)."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("empty_session")
        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        assert replay.get_step_count() == 2  # SESSION_START + SESSION_END
        assert replay.step_forward() is not None  # Can step to SESSION_END
        assert replay.step_forward() is None  # Cannot step past end

    def test_large_recording(self, temp_storage_path: Path):
        """Test handling of large recordings."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("large_session")

        # Add many steps
        for i in range(100):
            recorder.record_llm_request(
                messages=[{"role": "user", "content": f"Message {i}"}],
                model="gpt-4",
            )
            recorder.record_state_change("counter", i, i + 1)

        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        assert replay.get_step_count() > 200  # start + 100*2 + end

        # Test jumping around in large session
        replay.jump_to_step(50)
        assert replay.get_current_index() == 50

        replay.jump_to_step(150)
        assert replay.get_current_index() == 150

    def test_corrupted_step_data(self, temp_storage_path: Path):
        """Test handling of corrupted step data in session."""
        temp_storage_path.mkdir(parents=True, exist_ok=True)

        # Create a session file with valid structure but corrupted step type
        corrupted_session = {
            "session_id": "corrupted_test",
            "name": "Corrupted Session",
            "created_at": "2024-01-15T10:00:00",
            "ended_at": "2024-01-15T10:30:00",
            "steps": [
                {
                    "step_id": "step_0",
                    "step_type": "invalid_step_type",  # Invalid step type
                    "timestamp": "2024-01-15T10:00:00",
                    "data": {},
                }
            ],
            "initial_state": {},
            "final_state": None,
            "metadata": {},
        }

        session_file = temp_storage_path / "corrupted_test.json"
        with open(session_file, "w") as f:
            json.dump(corrupted_session, f)

        replay = SessionReplay(storage_path=str(temp_storage_path))
        # Invalid step type raises ValueError during from_dict
        with pytest.raises(ValueError, match="'invalid_step_type' is not a valid StepType"):
            replay.load_session("corrupted_test")

    def test_missing_step_fields(self):
        """Test RecordedStep with missing optional fields."""
        # Minimal data
        data = {
            "step_id": "step_1",
            "step_type": "llm_request",
            "timestamp": "2024-01-15T10:00:00",
            "data": {},
        }
        step = RecordedStep.from_dict(data)
        assert step.state_snapshot == {}
        assert step.metadata == {}
        assert step.parent_step_id is None
        assert step.duration_ms is None

    def test_concurrent_recording_not_supported(self, recorder: SessionRecorder):
        """Test that only one session can be recorded at a time per recorder."""
        session_id1 = recorder.start_session("session1")

        # Starting a new session should replace the old one
        session_id2 = recorder.start_session("session2")

        assert session_id1 != session_id2
        assert recorder._current_session.name == "session2"

    def test_rebuild_state_consistency(self, temp_storage_path: Path):
        """Test that state rebuilding produces consistent results."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("state_test", initial_state={"counter": 0})

        # Add interleaved LLM and tool operations
        for i in range(5):
            recorder.record_llm_request(
                messages=[{"role": "user", "content": f"Q{i}"}],
                model="gpt-4",
            )
            recorder.record_llm_response(
                response={"content": f"A{i}"},
            )
            recorder.record_tool_call("tool", {"i": i})
            recorder.record_tool_result("tool", {"result": i})
            recorder.record_state_change("counter", i, i + 1)

        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        # Step through and record state at each step
        states = []
        while True:
            state = replay.get_current_state()
            states.append((replay.get_current_index(), state.variables.copy()))
            if replay.step_forward() is None:
                break

        # Go back and verify states are the same
        for idx, expected_vars in reversed(states):
            replay.jump_to_step(idx)
            state = replay.get_current_state()
            assert state.variables == expected_vars

    def test_state_snapshot_isolation(self, recorder: SessionRecorder):
        """Test that state snapshots are isolated between steps."""
        recorder.start_session("isolation_test", initial_state={"list": [1, 2, 3]})

        # Modify state
        recorder.record_state_change("list", [1, 2, 3], [1, 2, 3, 4])

        # Verify snapshot is not affected by later changes
        step1 = recorder._current_session.steps[1]
        assert step1.state_snapshot["list"] == [1, 2, 3, 4]

        # Original step 0 snapshot should be unchanged
        step0 = recorder._current_session.steps[0]
        assert step0.state_snapshot["list"] == [1, 2, 3]

    def test_unicode_in_session(self, temp_storage_path: Path):
        """Test handling of Unicode characters in session data."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("unicode_test")
        recorder.record_llm_request(
            messages=[{"role": "user", "content": "Hello 世界 🌍 ñoño"}],
            model="gpt-4",
        )
        recorder.record_state_change("name", "", "José García")
        recorder.end_session(notes="Notes: 测试 тест")

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        step = replay._session.steps[1]
        assert "世界" in step.data["messages"][0]["content"]

    def test_nested_data_structures(self, recorder: SessionRecorder):
        """Test handling of nested data structures."""
        recorder.start_session("nested_test")

        nested_data = {
            "level1": {"level2": {"level3": ["a", "b", {"key": "value"}]}},
            "numbers": [1, 2, [3, 4, [5, 6]]],
        }

        recorder.record_state_change("data", {}, nested_data)

        step = recorder._current_session.steps[-1]
        assert step.data["new_value"]["level1"]["level2"]["level3"][2]["key"] == "value"

    def test_special_characters_in_filename(self, temp_storage_path: Path):
        """Test that session IDs with special characters are handled."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("session/with/slashes")
        recorder.end_session()

        # File should exist with the session_id in its name
        files = list(temp_storage_path.glob("*.json"))
        assert len(files) == 1

    def test_replay_after_modification(self, temp_storage_path: Path):
        """Test replay behavior after state modification."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session("mod_test", initial_state={"value": 0})
        recorder.record_state_change("value", 0, 1)
        recorder.record_state_change("value", 1, 2)
        recorder.end_session()

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        # Step forward and modify
        replay.step_forward()
        replay.modify_state("value", 100)

        # Verify modification persists
        state = replay.get_current_state()
        assert state.variables["value"] == 100

        # After stepping backward, the state is rebuilt and modifications are lost
        # This is expected behavior - modifications are transient
        replay.step_backward()
        state = replay.get_current_state()
        assert state.variables["value"] == 0  # Reset to initial state


# =============================================================================
# Test Async Operations (if any in future)
# =============================================================================


class TestAsyncOperations:
    """Tests for async operations (placeholders for future async support)."""

    @pytest.mark.asyncio
    async def test_async_session_operations(self, temp_storage_path: Path):
        """Test that sync operations work in async context."""
        recorder = SessionRecorder(storage_path=str(temp_storage_path))
        session_id = await asyncio.to_thread(recorder.start_session, "async_test")
        assert session_id is not None

        await asyncio.to_thread(
            recorder.record_llm_request, [{"role": "user", "content": "Hello"}], "gpt-4"
        )

        await asyncio.to_thread(recorder.end_session)


# =============================================================================
# Test StepType Enum
# =============================================================================


class TestStepType:
    """Tests for StepType enum."""

    def test_all_step_types_defined(self):
        """Test that all expected step types are defined."""
        expected_types = [
            "session_start",
            "session_end",
            "llm_request",
            "llm_response",
            "tool_call",
            "tool_result",
            "state_change",
            "error",
            "checkpoint",
            "user_input",
            "system_event",
            "ooda_phase",
            "healing_event",
            "axiom_check",
        ]
        for type_name in expected_types:
            assert hasattr(StepType, type_name.upper())

    def test_step_type_values(self):
        """Test that step type values match their names."""
        assert StepType.SESSION_START.value == "session_start"
        assert StepType.LLM_REQUEST.value == "llm_request"
        assert StepType.ERROR.value == "error"


# =============================================================================
# Test Session Metadata
# =============================================================================


class TestSessionMetadata:
    """Tests for session metadata handling."""

    def test_session_metadata_preserved(self, temp_storage_path: Path):
        """Test that session metadata is preserved through save/load."""
        metadata = {
            "version": "1.0.0",
            "environment": "production",
            "git_commit": "abc123",
        }
        tags = ["release", "v1.0"]

        recorder = SessionRecorder(storage_path=str(temp_storage_path), auto_save=True)
        session_id = recorder.start_session(
            name="metadata_test",
            metadata=metadata,
            tags=tags,
        )
        recorder.end_session(notes="Test session with metadata")

        replay = SessionReplay(storage_path=str(temp_storage_path))
        replay.load_session(session_id)

        session = replay._session
        assert session.metadata == metadata
        assert session.tags == tags
        assert session.notes == "Test session with metadata"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
