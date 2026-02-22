"""
SOP Governance & Role Enforcement
Implements: docs/evolution_plan_2026/47_SOP_GOVERNANCE.md

Process-Driven Intelligence for reliable agent behavior.

Features:
- Role definitions with SOP steps
- Mandatory artifact validation
- SOP Gatekeeper for task completion
- Reflexion on skipped steps
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml


class SOPStepStatus(Enum):
    """حالة خطوة SOP"""

    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    SKIPPED = auto()
    FAILED = auto()


class ArtifactStatus(Enum):
    """حالة Artifact"""

    MISSING = auto()
    INVALID = auto()
    VALID = auto()


@dataclass
class SOPStep:
    """خطوة في SOP"""

    step_id: int
    description: str
    status: SOPStepStatus = SOPStepStatus.PENDING
    completed_at: datetime | None = None
    output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """مخرجات مطلوبة"""

    name: str
    artifact_type: str  # file, report, score, code
    required: bool = True
    validation_rules: list[str] = field(default_factory=list)
    status: ArtifactStatus = ArtifactStatus.MISSING
    content: Any = None
    validated_at: datetime | None = None


@dataclass
class RoleDefinition:
    """تعريف الدور"""

    role_id: str
    name: str
    mission: str
    sop_steps: list[SOPStep] = field(default_factory=list)
    mandatory_artifacts: list[Artifact] = field(default_factory=list)
    priority: int = 1
    timeout_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: int) -> SOPStep | None:
        """Get step by ID"""
        for step in self.sop_steps:
            if step.step_id == step_id:
                return step
        return None

    def get_next_pending_step(self) -> SOPStep | None:
        """Get next pending step"""
        for step in self.sop_steps:
            if step.status == SOPStepStatus.PENDING:
                return step
        return None

    def all_steps_completed(self) -> bool:
        """Check if all steps completed"""
        return all(
            s.status in [SOPStepStatus.COMPLETED, SOPStepStatus.SKIPPED] for s in self.sop_steps
        )

    def completion_rate(self) -> float:
        """Calculate completion rate"""
        if not self.sop_steps:
            return 1.0
        completed = sum(1 for s in self.sop_steps if s.status == SOPStepStatus.COMPLETED)
        return completed / len(self.sop_steps)


@dataclass
class SOPExecution:
    """تنفيذ SOP"""

    execution_id: str
    role: RoleDefinition
    task_id: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    current_step: int = 1
    skipped_steps: list[int] = field(default_factory=list)
    artifacts_produced: dict[str, Any] = field(default_factory=dict)
    reflexion_required: bool = False
    reflexion_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def advance_step(self) -> SOPStep | None:
        """Advance to next step"""
        self.current_step += 1
        return self.role.get_step(self.current_step)

    def mark_step_completed(self, step_id: int, output: str | None = None) -> None:
        """Mark step as completed"""
        step = self.role.get_step(step_id)
        if step:
            step.status = SOPStepStatus.COMPLETED
            step.completed_at = datetime.now()
            step.output = output

    def mark_step_skipped(self, step_id: int, reason: str) -> None:
        """Mark step as skipped and trigger reflexion"""
        step = self.role.get_step(step_id)
        if step:
            step.status = SOPStepStatus.SKIPPED
            self.skipped_steps.append(step_id)
            self.reflexion_required = True
            self.reflexion_reason = f"Step {step_id} skipped: {reason}"

    def is_complete(self) -> bool:
        """Check if execution is complete"""
        return self.role.all_steps_completed()


class SOPStore:
    """
    Store for SOP Role Definitions.

    Loads roles from YAML files and provides lookup.
    """

    DEFAULT_ROLES_DIR = ".gaap/roles"

    def __init__(self, roles_dir: str | None = None) -> None:
        self._roles_dir = Path(roles_dir or self.DEFAULT_ROLES_DIR)
        self._roles: dict[str, RoleDefinition] = {}
        self._logger = logging.getLogger("gaap.governance.sop_store")

        self._ensure_roles_dir()
        self._load_roles()

    def _ensure_roles_dir(self) -> None:
        """Ensure roles directory exists"""
        self._roles_dir.mkdir(parents=True, exist_ok=True)
        self._create_default_roles()

    def _create_default_roles(self) -> None:
        """Create default role files if missing"""
        default_roles = {
            "coder.yaml": {
                "role": "Code Generator",
                "mission": "Generate high-quality, tested code that meets specifications.",
                "sop_steps": [
                    "Analyze requirements and constraints",
                    "Design solution architecture",
                    "Implement core functionality",
                    "Add error handling",
                    "Write unit tests",
                    "Document the code",
                ],
                "mandatory_artifacts": ["source_code", "tests", "documentation"],
            },
            "critic.yaml": {
                "role": "Security Analyst",
                "mission": "Identify vulnerabilities and ensure code quality.",
                "sop_steps": [
                    "Scan for hardcoded credentials",
                    "Check for unsafe library imports",
                    "Analyze data flow for injection risks",
                    "Review error handling",
                    "Generate Security Risk Table",
                ],
                "mandatory_artifacts": ["security_report.md", "cvss_score"],
            },
            "researcher.yaml": {
                "role": "Research Analyst",
                "mission": "Investigate and document best practices and solutions.",
                "sop_steps": [
                    "Identify research questions",
                    "Search for relevant documentation",
                    "Analyze findings",
                    "Synthesize recommendations",
                    "Document results",
                ],
                "mandatory_artifacts": ["research_report", "recommendations"],
            },
            "architect.yaml": {
                "role": "System Architect",
                "mission": "Design scalable and maintainable system architecture.",
                "sop_steps": [
                    "Analyze requirements and constraints",
                    "Evaluate technology options",
                    "Design component structure",
                    "Define interfaces and contracts",
                    "Document architecture decisions",
                ],
                "mandatory_artifacts": ["architecture_spec", "component_diagram", "decision_log"],
            },
        }

        for filename, role_data in default_roles.items():
            filepath = self._roles_dir / filename
            if not filepath.exists():
                with open(filepath, "w") as f:
                    yaml.dump(role_data, f, default_flow_style=False, sort_keys=False)
                self._logger.info(f"Created default role: {filename}")

    def _load_roles(self) -> None:
        """Load all role definitions from YAML files"""
        if not self._roles_dir.exists():
            return

        for filepath in self._roles_dir.glob("*.yaml"):
            try:
                with open(filepath) as f:
                    data = yaml.safe_load(f)

                if data:
                    role = self._parse_role(filepath.stem, data)
                    self._roles[role.role_id] = role
                    self._logger.debug(f"Loaded role: {role.name}")

            except Exception as e:
                self._logger.warning(f"Failed to load role {filepath}: {e}")

    def _parse_role(self, role_id: str, data: dict[str, Any]) -> RoleDefinition:
        """Parse role definition from YAML data"""
        steps = []
        for i, step_desc in enumerate(data.get("sop_steps", []), 1):
            if isinstance(step_desc, str):
                steps.append(SOPStep(step_id=i, description=step_desc))
            elif isinstance(step_desc, dict):
                steps.append(
                    SOPStep(
                        step_id=i,
                        description=step_desc.get("description", ""),
                        metadata=step_desc.get("metadata", {}),
                    )
                )

        artifacts = []
        for artifact_name in data.get("mandatory_artifacts", []):
            artifacts.append(Artifact(name=artifact_name, artifact_type="file"))

        return RoleDefinition(
            role_id=role_id,
            name=data.get("role", role_id),
            mission=data.get("mission", ""),
            sop_steps=steps,
            mandatory_artifacts=artifacts,
            priority=data.get("priority", 1),
            metadata=data.get("metadata", {}),
        )

    def get_role(self, role_id: str) -> RoleDefinition | None:
        """Get role by ID"""
        return self._roles.get(role_id)

    def get_role_by_name(self, name: str) -> RoleDefinition | None:
        """Get role by name"""
        for role in self._roles.values():
            if role.name.lower() == name.lower():
                return role
        return None

    def list_roles(self) -> list[str]:
        """List all available role IDs"""
        return list(self._roles.keys())

    def create_execution(self, role_id: str, task_id: str) -> SOPExecution | None:
        """Create a new SOP execution"""
        import uuid

        role = self.get_role(role_id)
        if not role:
            return None

        return SOPExecution(
            execution_id=f"sop_{uuid.uuid4().hex[:8]}",
            role=role,
            task_id=task_id,
        )


class SOPGatekeeper:
    """
    Validates that tasks meet SOP requirements.

    A task is not considered "done" until:
    - All SOP steps are completed
    - All mandatory artifacts are present and valid
    """

    def __init__(self, sop_store: SOPStore | None = None) -> None:
        self._store = sop_store or SOPStore()
        self._logger = logging.getLogger("gaap.governance.gatekeeper")
        self._executions: dict[str, SOPExecution] = {}

    def start_execution(self, role_id: str, task_id: str) -> SOPExecution | None:
        """Start a new SOP execution"""
        execution = self._store.create_execution(role_id, task_id)
        if execution:
            self._executions[task_id] = execution
            self._logger.info(f"Started SOP execution: {execution.execution_id} for role {role_id}")
        return execution

    def get_execution(self, task_id: str) -> SOPExecution | None:
        """Get execution by task ID"""
        return self._executions.get(task_id)

    def complete_step(
        self,
        task_id: str,
        step_id: int,
        output: str | None = None,
    ) -> bool:
        """Mark a step as completed"""
        execution = self._executions.get(task_id)
        if not execution:
            return False

        execution.mark_step_completed(step_id, output)
        self._logger.info(f"Step {step_id} completed for task {task_id}")
        return True

    def skip_step(
        self,
        task_id: str,
        step_id: int,
        reason: str,
    ) -> bool:
        """Skip a step (triggers reflexion)"""
        execution = self._executions.get(task_id)
        if not execution:
            return False

        execution.mark_step_skipped(step_id, reason)
        self._logger.warning(f"Step {step_id} skipped for task {task_id}: {reason}")
        return True

    def validate_artifact(
        self,
        task_id: str,
        artifact_name: str,
        content: Any,
    ) -> bool:
        """Validate an artifact"""
        execution = self._executions.get(task_id)
        if not execution:
            return False

        for artifact in execution.role.mandatory_artifacts:
            if artifact.name == artifact_name:
                is_valid = self._validate_artifact_content(artifact, content)
                artifact.status = ArtifactStatus.VALID if is_valid else ArtifactStatus.INVALID
                artifact.content = content
                artifact.validated_at = datetime.now()

                execution.artifacts_produced[artifact_name] = content

                self._logger.info(f"Artifact {artifact_name} validated: {artifact.status.name}")
                return is_valid

        return False

    def _validate_artifact_content(self, artifact: Artifact, content: Any) -> bool:
        """Validate artifact content based on type"""
        if content is None:
            return False

        if artifact.artifact_type == "file":
            return bool(content)

        if artifact.artifact_type == "report":
            return isinstance(content, str) and len(content) > 50

        if artifact.artifact_type == "score":
            try:
                score = float(content)
                return 0.0 <= score <= 10.0
            except (TypeError, ValueError):
                return False

        if artifact.artifact_type == "code":
            return isinstance(content, str) and len(content) > 10

        return bool(content)

    def check_completion(self, task_id: str) -> dict[str, Any]:
        """
        Check if task meets SOP completion requirements.

        Returns completion status with details.
        """
        execution = self._executions.get(task_id)
        if not execution:
            return {
                "complete": False,
                "reason": "No execution found",
            }

        steps_complete = execution.role.all_steps_completed()
        completion_rate = execution.role.completion_rate()

        missing_artifacts = []
        invalid_artifacts = []

        for artifact in execution.role.mandatory_artifacts:
            if artifact.status == ArtifactStatus.MISSING:
                missing_artifacts.append(artifact.name)
            elif artifact.status == ArtifactStatus.INVALID:
                invalid_artifacts.append(artifact.name)

        all_artifacts_valid = not missing_artifacts and not invalid_artifacts

        is_complete = steps_complete and all_artifacts_valid

        result = {
            "complete": is_complete,
            "steps_completion_rate": completion_rate,
            "steps_complete": steps_complete,
            "artifacts_valid": all_artifacts_valid,
            "missing_artifacts": missing_artifacts,
            "invalid_artifacts": invalid_artifacts,
            "skipped_steps": execution.skipped_steps,
            "reflexion_required": execution.reflexion_required,
            "reflexion_reason": execution.reflexion_reason,
        }

        if is_complete:
            execution.completed_at = datetime.now()

        return result

    def get_reflexion_prompt(self, task_id: str) -> str | None:
        """Get reflexion prompt for skipped steps"""
        execution = self._executions.get(task_id)
        if not execution or not execution.reflexion_required:
            return None

        skipped = [execution.role.get_step(s) for s in execution.skipped_steps]
        skipped_desc = [s.description for s in skipped if s]

        return (
            f"System Message: You skipped the following SOP steps: {', '.join(skipped_desc)}. "
            f"Reason provided: {execution.reflexion_reason}. "
            f"Please explain why you deviated from the protocol and whether this affects the task outcome."
        )

    def get_stats(self) -> dict[str, Any]:
        """Get gatekeeper statistics"""
        total = len(self._executions)
        completed = sum(1 for e in self._executions.values() if e.is_complete() and e.completed_at)
        reflexions = sum(1 for e in self._executions.values() if e.reflexion_required)

        return {
            "total_executions": total,
            "completed": completed,
            "completion_rate": completed / max(total, 1),
            "reflexions_triggered": reflexions,
            "available_roles": len(self._store.list_roles()),
        }


def create_sop_store(roles_dir: str | None = None) -> SOPStore:
    """Create an SOPStore instance"""
    return SOPStore(roles_dir=roles_dir)


def create_sop_gatekeeper(sop_store: SOPStore | None = None) -> SOPGatekeeper:
    """Create an SOPGatekeeper instance"""
    return SOPGatekeeper(sop_store=sop_store)
