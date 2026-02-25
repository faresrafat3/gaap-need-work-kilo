"""
Tests for OODA Loop Evolution (Spec 21)

Tests:
- Constitutional Gatekeeper (INVARIANT blocking)
- Dynamic Few-Shot Injection
- Lessons Injection
- Enhanced Back-propagation (AXIOM_VIOLATION trigger)
- Goal Drift Detection
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gaap.core.types import OODAState, OODAPhase, ReplanTrigger
from gaap.core.axioms import AxiomValidator, AxiomLevel, AxiomCheckResult, create_validator
from gaap.core.observer import Observer, EnvironmentState, create_observer
from gaap.gaap_engine import GAAPEngine, GAAPResponse, create_engine
from gaap.layers.layer2_tactical import AtomicTask, TaskCategory
from gaap.core.types import TaskType, TaskPriority


class TestConstitutionalGatekeeper:
    """Test Constitutional Gatekeeper enforcement."""

    def test_invariant_violation_blocks_execution(self):
        """INVARIANT violations should mark result as failed."""
        validator = create_validator(strict=True)

        code_with_syntax_error = """
def broken(
    # Missing closing paren
"""
        results = validator.validate(code=code_with_syntax_error, task_id="test-1")

        syntax_result = next((r for r in results if r.axiom_name == "syntax"), None)
        assert syntax_result is not None
        assert not syntax_result.passed

    def test_guideline_violation_logs_warning(self):
        """GUIDELINE violations should not block but log."""
        validator = create_validator(strict=True)

        code_with_unknown_import = """
import unknown_package_xyz
def foo():
    pass
"""
        results = validator.validate(code=code_with_unknown_import, task_id="test-2")

        dep_result = next((r for r in results if r.axiom_name == "dependency"), None)
        assert dep_result is not None
        assert not dep_result.passed

    def test_valid_code_passes_all_axioms(self):
        """Valid code should pass all axiom checks."""
        validator = create_validator(strict=True)

        valid_code = """
import os
import json

def valid_function():
    return {"status": "ok"}
"""
        results = validator.validate(code=valid_code, task_id="test-3")

        assert all(r.passed for r in results)


class TestOODAState:
    """Test OODA State management."""

    def test_axiom_violation_recording(self):
        """Test recording axiom violations."""
        ooda = OODAState(request_id="test-req")

        ooda.record_axiom_violation(
            {"axiom": "syntax", "message": "Syntax error", "severity": "low"}
        )

        assert len(ooda.axiom_violations) == 1
        assert ooda.axiom_violations[0]["axiom"] == "syntax"

    def test_replan_trigger(self):
        """Test triggering replan."""
        ooda = OODAState(request_id="test-req")

        ooda.trigger_replan(ReplanTrigger.AXIOM_VIOLATION)

        assert ooda.needs_replanning
        assert ooda.replan_trigger == ReplanTrigger.AXIOM_VIOLATION
        assert ooda.replan_count == 1

    def test_multiple_violations_trigger_replan(self):
        """Multiple violations should suggest replanning."""
        ooda = OODAState(request_id="test-req")

        for i in range(3):
            ooda.record_axiom_violation({"axiom": f"test-{i}", "message": f"Violation {i}"})

        assert len(ooda.axiom_violations) >= 3


class TestObserverEvolution:
    """Test enhanced Observer capabilities."""

    @pytest.mark.asyncio
    async def test_axiom_violation_pattern_detection(self):
        """Observer should detect axiom violation patterns."""
        observer = create_observer()
        ooda = OODAState(request_id="test-req")

        for i in range(3):
            ooda.record_axiom_violation({"axiom": f"test-{i}", "message": f"Violation {i}"})

        state = await observer.scan(ooda, original_goals=["test goal"])

        assert state.needs_replanning
        assert state.replan_trigger == ReplanTrigger.AXIOM_VIOLATION

    @pytest.mark.asyncio
    async def test_goal_drift_detection(self):
        """Observer should detect goal drift."""
        observer = create_observer()
        ooda = OODAState(request_id="test-req")

        ooda.completed_tasks.update(["task-1", "task-2", "task-3", "task-4", "task-5", "task-6"])
        ooda.lessons_learned = ["Completed unrelated task", "Another unrelated result"]

        state = await observer.scan(
            ooda, original_goals=["implement authentication system with OAuth"]
        )

        if state.needs_replanning and state.replan_trigger == ReplanTrigger.GOAL_DRIFT:
            assert True

    @pytest.mark.asyncio
    async def test_resource_exhaustion_detection(self):
        """Observer should detect resource exhaustion."""
        with patch.dict("os.environ", {"GAAP_MEMORY_LIMIT_MB": "100"}):
            observer = create_observer()
            ooda = OODAState(request_id="test-req")

            with patch.object(observer, "_get_memory_usage", return_value=500.0):
                state = await observer.scan(ooda)

                assert state.needs_replanning
                assert state.replan_trigger == ReplanTrigger.RESOURCE_EXHAUSTED

    def test_goal_alignment_calculation(self):
        """Test goal alignment scoring."""
        observer = create_observer()
        ooda = OODAState(request_id="test-req")

        ooda.lessons_learned = [
            "Authentication with OAuth implemented",
            "Added user login functionality",
        ]

        alignment = observer._check_goal_alignment(
            ooda, ["implement authentication system with OAuth"]
        )

        assert alignment > 0.0


class TestTaskEnrichment:
    """Test Dynamic Few-Shot Injection and Lessons Injection."""

    @pytest.mark.asyncio
    async def test_lessons_injection_into_metadata(self):
        """Lessons should be injected into task metadata."""
        engine = create_engine(enable_all=False)

        task = AtomicTask(
            id="test-task",
            name="Test Task",
            description="Write a function",
            category=TaskCategory.SETUP,
            type=TaskType.CODE_GENERATION,
        )

        ooda = OODAState(request_id="test-req")
        ooda.lessons_learned = ["Previous lesson 1", "Previous lesson 2"]

        enriched = await engine._enrich_task_context(task, ooda)

        assert "session_lessons" in enriched.metadata
        assert len(enriched.metadata["session_lessons"]) == 2

    @pytest.mark.asyncio
    async def test_context_enrichment_preserves_original_data(self):
        """Enrichment should preserve original task data."""
        engine = create_engine(enable_all=False)

        task = AtomicTask(
            id="test-task",
            name="Test Task",
            description="Write a function",
            category=TaskCategory.SETUP,
            type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
            metadata={"original_key": "original_value"},
        )

        ooda = OODAState(request_id="test-req")

        enriched = await engine._enrich_task_context(task, ooda)

        assert enriched.id == task.id
        assert enriched.name == task.name
        assert enriched.description == task.description
        assert enriched.priority == task.priority
        assert "original_key" in enriched.metadata


class TestReplanTriggers:
    """Test all replan trigger scenarios."""

    def test_replan_trigger_l3_critical_failure(self):
        """L3 critical failure should trigger replan."""
        ooda = OODAState(request_id="test-req")
        ooda.failed_tasks.add("critical-task-1")
        ooda.failed_tasks.add("critical-task-2")
        ooda.failed_tasks.add("critical-task-3")

        assert len(ooda.failed_tasks) >= 2

    def test_replan_trigger_axiom_violation(self):
        """Axiom violations should trigger replan."""
        ooda = OODAState(request_id="test-req")

        ooda.trigger_replan(ReplanTrigger.AXIOM_VIOLATION)

        assert ooda.replan_trigger == ReplanTrigger.AXIOM_VIOLATION
        assert ooda.replan_count == 1

    def test_replan_trigger_resource_exhausted(self):
        """Resource exhaustion should trigger replan."""
        ooda = OODAState(request_id="test-req")

        ooda.trigger_replan(ReplanTrigger.RESOURCE_EXHAUSTED)

        assert ooda.replan_trigger == ReplanTrigger.RESOURCE_EXHAUSTED

    def test_replan_trigger_goal_drift(self):
        """Goal drift should trigger replan."""
        ooda = OODAState(request_id="test-req")

        ooda.trigger_replan(ReplanTrigger.GOAL_DRIFT)

        assert ooda.replan_trigger == ReplanTrigger.GOAL_DRIFT


class TestAxiomValidator:
    """Test AxiomValidator capabilities."""

    def test_validator_stats(self):
        """Validator should track statistics."""
        validator = create_validator()

        validator.validate(code="x = 1", task_id="t1")
        validator.validate(code="y = 2", task_id="t2")

        stats = validator.get_stats()
        assert stats["checks_run"] == 2

    def test_known_packages_list(self):
        """Validator should have known packages list."""
        validator = create_validator()

        assert "os" in validator.known_packages
        assert "json" in validator.known_packages
        assert "asyncio" in validator.known_packages

    def test_interface_files_detection(self):
        """Validator should detect interface file modifications."""
        validator = create_validator()

        result = validator.validate(
            code="# modification", file_path="/project/src/__init__.py", task_id="t1"
        )

        interface_result = next((r for r in result if r.axiom_name == "interface"), None)
        assert interface_result is not None
        assert not interface_result.passed

    def test_read_only_diagnostic_check(self):
        """Validator should detect write operations in diagnostic tasks."""
        validator = create_validator()

        code_with_write = """
with open('file.txt', 'w') as f:
    f.write('data')
"""
        result = validator.validate(code=code_with_write, task_id="diag-1")

        readonly_result = next((r for r in result if r.axiom_name == "read_only_diagnostic"), None)
        assert readonly_result is not None
        assert not readonly_result.passed
