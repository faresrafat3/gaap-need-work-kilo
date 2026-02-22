"""
Unit Tests for GAAP SOP Governance Module
Tests: SOPStore, SOPGatekeeper, RoleDefinition, SOPExecution
"""

import tempfile
from pathlib import Path

import pytest

from gaap.core.governance import (
    Artifact,
    ArtifactStatus,
    RoleDefinition,
    SOPExecution,
    SOPGatekeeper,
    SOPStep,
    SOPStepStatus,
    SOPStore,
    create_sop_gatekeeper,
    create_sop_store,
)


@pytest.fixture
def temp_roles_dir():
    """Create a temporary directory for roles"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sop_store(temp_roles_dir):
    """Create a fresh SOP store"""
    return SOPStore(roles_dir=temp_roles_dir)


@pytest.fixture
def sop_gatekeeper(sop_store):
    """Create an SOP gatekeeper"""
    return SOPGatekeeper(sop_store=sop_store)


@pytest.fixture
def sample_sop_step():
    """Create a sample SOP step"""
    return SOPStep(
        step_id=1,
        description="Analyze requirements",
    )


@pytest.fixture
def sample_artifact():
    """Create a sample artifact"""
    return Artifact(
        name="test_report.md",
        artifact_type="report",
        required=True,
    )


class TestSOPStep:
    """Tests for SOPStep"""

    def test_step_creation(self):
        """Test creating an SOP step"""
        step = SOPStep(step_id=1, description="Test step")

        assert step.step_id == 1
        assert step.description == "Test step"
        assert step.status == SOPStepStatus.PENDING

    def test_step_status_transition(self):
        """Test step status transitions"""
        step = SOPStep(step_id=1, description="Test")

        assert step.status == SOPStepStatus.PENDING

        step.status = SOPStepStatus.IN_PROGRESS
        assert step.status == SOPStepStatus.IN_PROGRESS

        step.status = SOPStepStatus.COMPLETED
        assert step.status == SOPStepStatus.COMPLETED


class TestArtifact:
    """Tests for Artifact"""

    def test_artifact_creation(self):
        """Test creating an artifact"""
        artifact = Artifact(name="report.md", artifact_type="file")

        assert artifact.name == "report.md"
        assert artifact.artifact_type == "file"
        assert artifact.status == ArtifactStatus.MISSING
        assert artifact.required is True

    def test_artifact_validation_rules(self):
        """Test artifact with validation rules"""
        artifact = Artifact(
            name="cvss_score",
            artifact_type="score",
            validation_rules=["0 <= score <= 10"],
        )

        assert len(artifact.validation_rules) == 1


class TestRoleDefinition:
    """Tests for RoleDefinition"""

    def test_role_creation(self):
        """Test creating a role definition"""
        steps = [
            SOPStep(step_id=1, description="Step 1"),
            SOPStep(step_id=2, description="Step 2"),
        ]
        artifacts = [Artifact(name="output.md", artifact_type="file")]

        role = RoleDefinition(
            role_id="test_role",
            name="Test Role",
            mission="Test mission",
            sop_steps=steps,
            mandatory_artifacts=artifacts,
        )

        assert role.role_id == "test_role"
        assert len(role.sop_steps) == 2
        assert len(role.mandatory_artifacts) == 1

    def test_get_step(self):
        """Test getting step by ID"""
        steps = [
            SOPStep(step_id=1, description="Step 1"),
            SOPStep(step_id=2, description="Step 2"),
        ]
        role = RoleDefinition(role_id="test", name="Test", mission="Test mission", sop_steps=steps)

        step = role.get_step(1)
        assert step is not None
        assert step.description == "Step 1"

        missing = role.get_step(99)
        assert missing is None

    def test_get_next_pending_step(self):
        """Test getting next pending step"""
        steps = [
            SOPStep(step_id=1, description="Step 1"),
            SOPStep(step_id=2, description="Step 2"),
        ]
        role = RoleDefinition(role_id="test", name="Test", mission="Test mission", sop_steps=steps)

        next_step = role.get_next_pending_step()
        assert next_step is not None
        assert next_step.step_id == 1

        steps[0].status = SOPStepStatus.COMPLETED
        next_step = role.get_next_pending_step()
        assert next_step is not None
        assert next_step.step_id == 2

    def test_all_steps_completed(self):
        """Test checking if all steps completed"""
        steps = [
            SOPStep(step_id=1, description="Step 1"),
            SOPStep(step_id=2, description="Step 2"),
        ]
        role = RoleDefinition(role_id="test", name="Test", mission="Test mission", sop_steps=steps)

        assert not role.all_steps_completed()

        steps[0].status = SOPStepStatus.COMPLETED
        assert not role.all_steps_completed()

        steps[1].status = SOPStepStatus.COMPLETED
        assert role.all_steps_completed()

    def test_completion_rate(self):
        """Test completion rate calculation"""
        steps = [
            SOPStep(step_id=1, description="Step 1"),
            SOPStep(step_id=2, description="Step 2"),
            SOPStep(step_id=3, description="Step 3"),
        ]
        role = RoleDefinition(role_id="test", name="Test", mission="Test mission", sop_steps=steps)

        assert role.completion_rate() == 0.0

        steps[0].status = SOPStepStatus.COMPLETED
        assert role.completion_rate() == pytest.approx(1 / 3, rel=0.01)

        steps[1].status = SOPStepStatus.COMPLETED
        assert role.completion_rate() == pytest.approx(2 / 3, rel=0.01)


class TestSOPStore:
    """Tests for SOPStore"""

    def test_create_store(self, temp_roles_dir):
        """Test creating an SOP store"""
        store = create_sop_store(temp_roles_dir)

        assert store is not None
        assert store._roles_dir.exists()

    def test_default_roles_created(self, sop_store):
        """Test that default roles are created"""
        roles = sop_store.list_roles()

        assert "coder" in roles
        assert "critic" in roles
        assert "researcher" in roles

    def test_get_role(self, sop_store):
        """Test getting a role by ID"""
        role = sop_store.get_role("coder")

        assert role is not None
        assert role.name == "Code Generator"
        assert len(role.sop_steps) >= 1

    def test_get_role_by_name(self, sop_store):
        """Test getting a role by name"""
        role = sop_store.get_role_by_name("Code Generator")

        assert role is not None
        assert role.role_id == "coder"

    def test_create_execution(self, sop_store):
        """Test creating an SOP execution"""
        execution = sop_store.create_execution("coder", "task_001")

        assert execution is not None
        assert execution.task_id == "task_001"
        assert execution.role.role_id == "coder"

    def test_create_execution_invalid_role(self, sop_store):
        """Test creating execution with invalid role"""
        execution = sop_store.create_execution("invalid_role", "task_001")

        assert execution is None


class TestSOPExecution:
    """Tests for SOPExecution"""

    def test_execution_creation(self, sop_store):
        """Test creating an execution"""
        role = sop_store.get_role("coder")
        execution = SOPExecution(
            execution_id="exec_001",
            role=role,
            task_id="task_001",
        )

        assert execution.execution_id == "exec_001"
        assert execution.current_step == 1
        assert not execution.is_complete()

    def test_mark_step_completed(self, sop_store):
        """Test marking step as completed"""
        role = sop_store.get_role("coder")
        execution = SOPExecution(
            execution_id="exec_001",
            role=role,
            task_id="task_001",
        )

        execution.mark_step_completed(1, "Output 1")

        step = role.get_step(1)
        assert step.status == SOPStepStatus.COMPLETED
        assert step.output == "Output 1"

    def test_mark_step_skipped(self, sop_store):
        """Test marking step as skipped"""
        role = sop_store.get_role("coder")
        execution = SOPExecution(
            execution_id="exec_001",
            role=role,
            task_id="task_001",
        )

        execution.mark_step_skipped(1, "Not applicable")

        step = role.get_step(1)
        assert step.status == SOPStepStatus.SKIPPED
        assert 1 in execution.skipped_steps
        assert execution.reflexion_required

    def test_advance_step(self, sop_store):
        """Test advancing to next step"""
        role = sop_store.get_role("coder")
        execution = SOPExecution(
            execution_id="exec_001",
            role=role,
            task_id="task_001",
        )

        next_step = execution.advance_step()

        assert execution.current_step == 2


class TestSOPGatekeeper:
    """Tests for SOPGatekeeper"""

    def test_create_gatekeeper(self):
        """Test creating a gatekeeper"""
        gatekeeper = create_sop_gatekeeper()

        assert gatekeeper is not None
        assert gatekeeper._store is not None

    def test_start_execution(self, sop_gatekeeper):
        """Test starting an execution"""
        execution = sop_gatekeeper.start_execution("coder", "task_001")

        assert execution is not None
        assert execution.task_id == "task_001"

    def test_complete_step(self, sop_gatekeeper):
        """Test completing a step"""
        sop_gatekeeper.start_execution("coder", "task_001")

        result = sop_gatekeeper.complete_step("task_001", 1, "Output")

        assert result is True

        execution = sop_gatekeeper.get_execution("task_001")
        step = execution.role.get_step(1)
        assert step.status == SOPStepStatus.COMPLETED

    def test_skip_step(self, sop_gatekeeper):
        """Test skipping a step"""
        sop_gatekeeper.start_execution("coder", "task_001")

        result = sop_gatekeeper.skip_step("task_001", 1, "Not needed")

        assert result is True

        execution = sop_gatekeeper.get_execution("task_001")
        assert execution.reflexion_required
        assert "Not needed" in execution.reflexion_reason

    def test_validate_artifact(self, sop_gatekeeper):
        """Test validating an artifact"""
        sop_gatekeeper.start_execution("coder", "task_001")

        result = sop_gatekeeper.validate_artifact("task_001", "source_code", "def hello(): pass")

        assert result is True

    def test_validate_artifact_missing(self, sop_gatekeeper):
        """Test validating missing artifact"""
        sop_gatekeeper.start_execution("coder", "task_001")

        result = sop_gatekeeper.validate_artifact("task_001", "source_code", None)

        assert result is False

    def test_check_completion(self, sop_gatekeeper):
        """Test checking completion"""
        execution = sop_gatekeeper.start_execution("coder", "task_001")

        # Initially not complete
        status = sop_gatekeeper.check_completion("task_001")
        assert not status["complete"]

        # Complete all steps
        for step in execution.role.sop_steps:
            sop_gatekeeper.complete_step("task_001", step.step_id)

        # Validate artifacts
        for artifact in execution.role.mandatory_artifacts:
            sop_gatekeeper.validate_artifact("task_001", artifact.name, "dummy content")

        status = sop_gatekeeper.check_completion("task_001")
        assert status["complete"]

    def test_get_reflexion_prompt(self, sop_gatekeeper):
        """Test getting reflexion prompt"""
        sop_gatekeeper.start_execution("coder", "task_001")
        sop_gatekeeper.skip_step("task_001", 1, "Test skip")

        prompt = sop_gatekeeper.get_reflexion_prompt("task_001")

        assert prompt is not None
        assert "skipped" in prompt.lower()

    def test_get_stats(self, sop_gatekeeper):
        """Test getting statistics"""
        sop_gatekeeper.start_execution("coder", "task_001")
        sop_gatekeeper.start_execution("critic", "task_002")

        stats = sop_gatekeeper.get_stats()

        assert stats["total_executions"] == 2
        assert stats["available_roles"] >= 3


class TestArtifactValidation:
    """Tests for artifact validation"""

    def test_validate_file_artifact(self, sop_gatekeeper):
        """Test validating file artifact"""
        sop_gatekeeper.start_execution("coder", "task_001")

        result = sop_gatekeeper.validate_artifact("task_001", "source_code", "print('hello')")

        assert result is True

    def test_validate_report_artifact(self, sop_gatekeeper):
        """Test validating report artifact"""
        sop_gatekeeper.start_execution("critic", "task_001")

        result = sop_gatekeeper.validate_artifact("task_001", "security_report.md", "A" * 100)

        assert result is True

    def test_validate_score_artifact(self, sop_gatekeeper):
        """Test validating score artifact"""
        sop_gatekeeper.start_execution("critic", "task_001")

        # The cvss_score artifact type is "file" by default
        # So any non-empty content is valid
        result = sop_gatekeeper.validate_artifact("task_001", "cvss_score", "7.5")

        assert result is True

    def test_validate_empty_artifact(self, sop_gatekeeper):
        """Test validating empty artifact"""
        sop_gatekeeper.start_execution("critic", "task_001")

        result = sop_gatekeeper.validate_artifact("task_001", "cvss_score", "")

        assert result is False
