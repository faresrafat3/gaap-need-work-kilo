"""
SOP Manager - MetaGPT-Inspired Role-Based Standard Operating Procedures

Manages Standard Operating Procedures for agents:
- Role-based SOP definitions
- Artifact validation against SOPs
- Step progression tracking
- Quality gate enforcement

Inspired by MetaGPT: https://github.com/geekan/MetaGPT
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("gaap.layers.sop_manager")


class StepType(Enum):
    """Types of SOP steps"""

    ACTION = auto()
    DECISION = auto()
    VALIDATION = auto()
    ARTIFACT_CREATION = auto()
    REVIEW = auto()
    APPROVAL = auto()


class QualityGateStatus(Enum):
    """Status of a quality gate check"""

    PENDING = auto()
    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()
    WAIVED = auto()


@dataclass
class QualityGate:
    """
    A quality gate in an SOP.

    Defines a checkpoint that must be passed before proceeding.
    """

    name: str
    description: str = ""
    check_function: str = ""
    required_artifacts: list[str] = field(default_factory=list)
    failure_action: str = "halt"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "check_function": self.check_function,
            "required_artifacts": self.required_artifacts,
            "failure_action": self.failure_action,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityGate:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            check_function=data.get("check_function", ""),
            required_artifacts=data.get("required_artifacts", []),
            failure_action=data.get("failure_action", "halt"),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SOPStep:
    """
    A single step in an SOP.

    Represents one action or decision point in the procedure.
    """

    step_id: int
    name: str
    description: str = ""
    step_type: StepType = StepType.ACTION
    expected_artifacts: list[str] = field(default_factory=list)
    quality_gates: list[QualityGate] = field(default_factory=list)
    dependencies: list[int] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "step_type": self.step_type.name,
            "expected_artifacts": self.expected_artifacts,
            "quality_gates": [qg.to_dict() for qg in self.quality_gates],
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SOPStep:
        return cls(
            step_id=data.get("step_id", 0),
            name=data.get("name", ""),
            description=data.get("description", ""),
            step_type=StepType[data.get("step_type", "ACTION")],
            expected_artifacts=data.get("expected_artifacts", []),
            quality_gates=[QualityGate.from_dict(qg) for qg in data.get("quality_gates", [])],
            dependencies=data.get("dependencies", []),
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SOP:
    """
    Standard Operating Procedure for a role.

    Defines the complete workflow for a role including:
    - Ordered steps to execute
    - Artifacts to produce
    - Quality gates to pass
    - Dependencies between steps

    Attributes:
        id: Unique identifier
        role: Role this SOP applies to
        name: Human-readable name
        description: What this SOP accomplishes
        steps: Ordered list of steps
        artifacts_produced: List of artifact types produced
        quality_gates: Global quality gates
        version: SOP version
        created_at: Creation timestamp
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""
    name: str = ""
    description: str = ""
    steps: list[SOPStep] = field(default_factory=list)
    artifacts_produced: list[str] = field(default_factory=list)
    quality_gates: list[QualityGate] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: int) -> SOPStep | None:
        """Get a step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_next_step(self, current_step_id: int) -> SOPStep | None:
        """Get the next step after current_step_id"""
        for i, step in enumerate(self.steps):
            if step.step_id == current_step_id and i + 1 < len(self.steps):
                return self.steps[i + 1]
        return None

    def get_step_dependencies(self, step_id: int) -> list[int]:
        """Get dependencies for a step"""
        step = self.get_step(step_id)
        return step.dependencies if step else []

    def validate_step_order(self) -> tuple[bool, list[str]]:
        """Validate that step dependencies are satisfiable"""
        errors = []
        step_ids = {s.step_id for s in self.steps}

        for step in self.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    errors.append(f"Step {step.step_id} depends on non-existent step {dep_id}")

        for i, step in enumerate(self.steps):
            if step.step_id != i + 1:
                errors.append(f"Step IDs should be sequential, got {step.step_id} at index {i}")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "artifacts_produced": self.artifacts_produced,
            "quality_gates": [qg.to_dict() for qg in self.quality_gates],
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SOP:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data.get("role", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=[SOPStep.from_dict(s) for s in data.get("steps", [])],
            artifacts_produced=data.get("artifacts_produced", []),
            quality_gates=[QualityGate.from_dict(qg) for qg in data.get("quality_gates", [])],
            version=data.get("version", "1.0.0"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ArtifactValidationResult:
    """Result of validating an artifact against SOP"""

    artifact_name: str
    is_valid: bool
    quality_gate_results: list[tuple[str, QualityGateStatus]]
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "is_valid": self.is_valid,
            "quality_gate_results": [
                {"gate": g, "status": s.name} for g, s in self.quality_gate_results
            ],
            "issues": self.issues,
            "suggestions": self.suggestions,
            "score": self.score,
        }


class SOPManager:
    """
    Manages SOPs for different roles.

    Provides:
    - SOP registration and lookup
    - Artifact validation against SOPs
    - Step progression tracking
    - Quality gate enforcement

    Usage:
        manager = SOPManager(sop_dir=".gaap/sops")

        # Get SOP for a role
        sop = manager.get_sop_for_role("coder")

        # Validate an artifact
        result = manager.validate_artifact_against_sop(
            artifact=my_artifact,
            sop=sop,
        )

        # Get next step
        next_step = manager.get_next_step(sop, current_step=2)
    """

    DEFAULT_SOPS_DIR = ".gaap/sops"

    def __init__(self, sop_dir: str | None = None) -> None:
        self._sop_dir = Path(sop_dir or self.DEFAULT_SOPS_DIR)
        self._sops: dict[str, SOP] = {}
        self._role_index: dict[str, str] = {}
        self._logger = logging.getLogger("gaap.layers.sop_manager")

        self._ensure_sop_dir()
        self._load_sops()

    def _ensure_sop_dir(self) -> None:
        """Ensure SOP directory exists"""
        self._sop_dir.mkdir(parents=True, exist_ok=True)
        self._create_default_sops()

    def _create_default_sops(self) -> None:
        """Create default SOP files if missing"""
        default_sops = {
            "coder.yaml": {
                "role": "coder",
                "name": "Code Generation SOP",
                "description": "Standard procedure for generating high-quality code",
                "steps": [
                    {
                        "step_id": 1,
                        "name": "Analyze Requirements",
                        "description": "Understand and document requirements",
                        "step_type": "ACTION",
                        "expected_artifacts": ["requirements_doc"],
                    },
                    {
                        "step_id": 2,
                        "name": "Design Solution",
                        "description": "Design the solution architecture",
                        "step_type": "ACTION",
                        "dependencies": [1],
                        "expected_artifacts": ["design_doc"],
                    },
                    {
                        "step_id": 3,
                        "name": "Implement Code",
                        "description": "Write the actual code",
                        "step_type": "ARTIFACT_CREATION",
                        "dependencies": [2],
                        "expected_artifacts": ["source_code"],
                        "quality_gates": [
                            {
                                "name": "syntax_check",
                                "description": "Verify code syntax is valid",
                            }
                        ],
                    },
                    {
                        "step_id": 4,
                        "name": "Write Tests",
                        "description": "Create unit tests for the code",
                        "step_type": "ARTIFACT_CREATION",
                        "dependencies": [3],
                        "expected_artifacts": ["test_code"],
                    },
                    {
                        "step_id": 5,
                        "name": "Run Tests",
                        "description": "Execute tests and verify passing",
                        "step_type": "VALIDATION",
                        "dependencies": [4],
                        "expected_artifacts": ["test_results"],
                    },
                    {
                        "step_id": 6,
                        "name": "Document Code",
                        "description": "Add documentation and comments",
                        "step_type": "ARTIFACT_CREATION",
                        "dependencies": [3],
                        "expected_artifacts": ["documentation"],
                    },
                ],
                "artifacts_produced": [
                    "source_code",
                    "test_code",
                    "documentation",
                ],
                "quality_gates": [
                    {
                        "name": "all_tests_pass",
                        "description": "All tests must pass",
                        "required_artifacts": ["test_results"],
                    },
                    {
                        "name": "coverage_threshold",
                        "description": "Code coverage must be >= 80%",
                        "required_artifacts": ["coverage_report"],
                    },
                ],
            },
            "reviewer.yaml": {
                "role": "reviewer",
                "name": "Code Review SOP",
                "description": "Standard procedure for reviewing code",
                "steps": [
                    {
                        "step_id": 1,
                        "name": "Read Code",
                        "description": "Read and understand the code changes",
                        "step_type": "ACTION",
                        "expected_artifacts": ["code_understanding"],
                    },
                    {
                        "step_id": 2,
                        "name": "Check Security",
                        "description": "Analyze for security vulnerabilities",
                        "step_type": "VALIDATION",
                        "dependencies": [1],
                        "expected_artifacts": ["security_check"],
                        "quality_gates": [
                            {
                                "name": "no_critical_vulnerabilities",
                                "description": "No critical security issues",
                            }
                        ],
                    },
                    {
                        "step_id": 3,
                        "name": "Check Performance",
                        "description": "Analyze performance implications",
                        "step_type": "VALIDATION",
                        "dependencies": [1],
                        "expected_artifacts": ["performance_check"],
                    },
                    {
                        "step_id": 4,
                        "name": "Check Style",
                        "description": "Verify code style compliance",
                        "step_type": "VALIDATION",
                        "dependencies": [1],
                        "expected_artifacts": ["style_check"],
                    },
                    {
                        "step_id": 5,
                        "name": "Write Review",
                        "description": "Compile review findings",
                        "step_type": "ARTIFACT_CREATION",
                        "dependencies": [2, 3, 4],
                        "expected_artifacts": ["review_report"],
                    },
                ],
                "artifacts_produced": ["review_report"],
            },
            "architect.yaml": {
                "role": "architect",
                "name": "Architecture Design SOP",
                "description": "Standard procedure for designing system architecture",
                "steps": [
                    {
                        "step_id": 1,
                        "name": "Gather Requirements",
                        "description": "Collect and analyze requirements",
                        "step_type": "ACTION",
                        "expected_artifacts": ["requirements"],
                    },
                    {
                        "step_id": 2,
                        "name": "Analyze Constraints",
                        "description": "Identify technical and business constraints",
                        "step_type": "ACTION",
                        "dependencies": [1],
                        "expected_artifacts": ["constraints"],
                    },
                    {
                        "step_id": 3,
                        "name": "Design Components",
                        "description": "Design system components and interfaces",
                        "step_type": "ARTIFACT_CREATION",
                        "dependencies": [2],
                        "expected_artifacts": ["component_design"],
                    },
                    {
                        "step_id": 4,
                        "name": "Define Data Model",
                        "description": "Design data structures and flows",
                        "step_type": "ARTIFACT_CREATION",
                        "dependencies": [3],
                        "expected_artifacts": ["data_model"],
                    },
                    {
                        "step_id": 5,
                        "name": "Review Architecture",
                        "description": "Review and validate architecture decisions",
                        "step_type": "REVIEW",
                        "dependencies": [3, 4],
                        "expected_artifacts": ["architecture_review"],
                    },
                    {
                        "step_id": 6,
                        "name": "Document Architecture",
                        "description": "Create architecture documentation",
                        "step_type": "ARTIFACT_CREATION",
                        "dependencies": [5],
                        "expected_artifacts": ["architecture_doc"],
                    },
                ],
                "artifacts_produced": [
                    "component_design",
                    "data_model",
                    "architecture_doc",
                ],
            },
        }

        for filename, sop_data in default_sops.items():
            filepath = self._sop_dir / filename
            if not filepath.exists():
                with open(filepath, "w") as f:
                    yaml.dump(sop_data, f, default_flow_style=False, sort_keys=False)
                self._logger.info(f"Created default SOP: {filename}")

    def _load_sops(self) -> None:
        """Load all SOPs from the SOP directory"""
        if not self._sop_dir.exists():
            return

        for filepath in self._sop_dir.glob("*.yaml"):
            try:
                with open(filepath) as f:
                    data = yaml.safe_load(f)

                if data:
                    sop = self._parse_sop(data)
                    self._sops[sop.id] = sop
                    self._role_index[sop.role] = sop.id
                    self._logger.debug(f"Loaded SOP: {sop.name} for role {sop.role}")

            except Exception as e:
                self._logger.warning(f"Failed to load SOP {filepath}: {e}")

    def _parse_sop(self, data: dict[str, Any]) -> SOP:
        """Parse SOP from dictionary"""
        steps = []
        for step_data in data.get("steps", []):
            quality_gates = [QualityGate.from_dict(qg) for qg in step_data.get("quality_gates", [])]

            step = SOPStep(
                step_id=step_data.get("step_id", 0),
                name=step_data.get("name", ""),
                description=step_data.get("description", ""),
                step_type=StepType[step_data.get("step_type", "ACTION")],
                expected_artifacts=step_data.get("expected_artifacts", []),
                quality_gates=quality_gates,
                dependencies=step_data.get("dependencies", []),
                timeout_seconds=step_data.get("timeout_seconds", 300),
                retry_count=step_data.get("retry_count", 0),
                metadata=step_data.get("metadata", {}),
            )
            steps.append(step)

        quality_gates = [QualityGate.from_dict(qg) for qg in data.get("quality_gates", [])]

        return SOP(
            role=data.get("role", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=steps,
            artifacts_produced=data.get("artifacts_produced", []),
            quality_gates=quality_gates,
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
        )

    def get_sop_for_role(self, role: str) -> SOP | None:
        """Get SOP for a specific role"""
        sop_id = self._role_index.get(role)
        if sop_id:
            return self._sops.get(sop_id)
        return None

    def get_sop(self, sop_id: str) -> SOP | None:
        """Get SOP by ID"""
        return self._sops.get(sop_id)

    def register_sop(self, sop: SOP) -> None:
        """Register a new SOP"""
        is_valid, errors = sop.validate_step_order()
        if not is_valid:
            raise ValueError(f"Invalid SOP: {errors}")

        self._sops[sop.id] = sop
        self._role_index[sop.role] = sop.id

        self._save_sop(sop)
        self._logger.info(f"Registered SOP: {sop.name} for role {sop.role}")

    def _save_sop(self, sop: SOP) -> None:
        """Save SOP to file"""
        filepath = self._sop_dir / f"{sop.role}.yaml"
        with open(filepath, "w") as f:
            yaml.dump(sop.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate_artifact_against_sop(
        self,
        artifact: Any,
        sop: SOP,
        step_id: int | None = None,
    ) -> ArtifactValidationResult:
        """
        Validate an artifact against SOP requirements.

        Args:
            artifact: The artifact to validate
            sop: The SOP to validate against
            step_id: Optional specific step to validate for

        Returns:
            ArtifactValidationResult with validation details
        """
        artifact_name = getattr(artifact, "name", "unknown")
        issues: list[str] = []
        suggestions: list[str] = []
        gate_results: list[tuple[str, QualityGateStatus]] = []

        if artifact is None:
            issues.append("Artifact is None")
            return ArtifactValidationResult(
                artifact_name=artifact_name,
                is_valid=False,
                quality_gate_results=gate_results,
                issues=issues,
                score=0.0,
            )

        gates_to_check = sop.quality_gates
        if step_id is not None:
            step = sop.get_step(step_id)
            if step:
                gates_to_check = step.quality_gates

        for gate in gates_to_check:
            status = self._check_quality_gate(artifact, gate)
            gate_results.append((gate.name, status))

            if status == QualityGateStatus.FAILED:
                issues.append(f"Quality gate '{gate.name}' failed")
                if gate.description:
                    suggestions.append(f"Address: {gate.description}")

        if step_id is not None:
            step = sop.get_step(step_id)
            if step and step.expected_artifacts:
                if artifact_name not in step.expected_artifacts:
                    issues.append(
                        f"Artifact '{artifact_name}' not expected for step {step_id}. "
                        f"Expected: {step.expected_artifacts}"
                    )

        is_valid = all(
            status
            in (QualityGateStatus.PASSED, QualityGateStatus.SKIPPED, QualityGateStatus.WAIVED)
            for _, status in gate_results
        )

        passed_gates = sum(1 for _, status in gate_results if status == QualityGateStatus.PASSED)
        total_gates = len(gate_results) if gate_results else 1
        score = passed_gates / total_gates

        return ArtifactValidationResult(
            artifact_name=artifact_name,
            is_valid=is_valid and len(issues) == 0,
            quality_gate_results=gate_results,
            issues=issues,
            suggestions=suggestions,
            score=score,
        )

    def _check_quality_gate(
        self,
        artifact: Any,
        gate: QualityGate,
    ) -> QualityGateStatus:
        """Check if an artifact passes a quality gate"""
        if gate.required_artifacts:
            artifact_name = getattr(artifact, "name", "")
            if artifact_name not in gate.required_artifacts:
                return QualityGateStatus.SKIPPED

        artifact_type = getattr(artifact, "type", None)
        artifact_content = getattr(artifact, "content", None)

        if artifact_content is None:
            return QualityGateStatus.FAILED

        if "syntax" in gate.name.lower():
            if isinstance(artifact_content, str):
                if len(artifact_content) > 0:
                    return QualityGateStatus.PASSED
                return QualityGateStatus.FAILED

        if "test" in gate.name.lower():
            if isinstance(artifact_content, dict):
                passed = artifact_content.get("passed", 0)
                total = artifact_content.get("total", 1)
                if passed >= total:
                    return QualityGateStatus.PASSED
            return QualityGateStatus.FAILED

        if "security" in gate.name.lower():
            if isinstance(artifact_content, dict):
                vulnerabilities = artifact_content.get("vulnerabilities", [])
                critical = [v for v in vulnerabilities if v.get("severity") == "critical"]
                if not critical:
                    return QualityGateStatus.PASSED
            return QualityGateStatus.FAILED

        if artifact_content:
            return QualityGateStatus.PASSED

        return QualityGateStatus.PENDING

    def get_next_step(self, sop: SOP, current_step: int) -> int:
        """
        Get the next step ID in the SOP.

        Args:
            sop: The SOP
            current_step: Current step ID

        Returns:
            Next step ID, or -1 if at end
        """
        next_step = sop.get_next_step(current_step)
        return next_step.step_id if next_step else -1

    def get_executable_steps(
        self,
        sop: SOP,
        completed_steps: set[int],
    ) -> list[int]:
        """
        Get steps that can be executed given completed steps.

        A step is executable if all its dependencies are completed.
        """
        executable = []

        for step in sop.steps:
            if step.step_id in completed_steps:
                continue

            deps_satisfied = all(dep in completed_steps for dep in step.dependencies)

            if deps_satisfied:
                executable.append(step.step_id)

        return executable

    def list_sops(self) -> list[str]:
        """List all SOP IDs"""
        return list(self._sops.keys())

    def list_roles(self) -> list[str]:
        """List all roles with SOPs"""
        return list(self._role_index.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get SOP manager statistics"""
        total_steps = sum(len(sop.steps) for sop in self._sops.values())
        total_gates = sum(
            len(sop.quality_gates) + sum(len(step.quality_gates) for step in sop.steps)
            for sop in self._sops.values()
        )

        return {
            "total_sops": len(self._sops),
            "total_roles": len(self._role_index),
            "total_steps": total_steps,
            "total_quality_gates": total_gates,
            "sop_directory": str(self._sop_dir),
        }


def create_sop_manager(sop_dir: str | None = None) -> SOPManager:
    """Factory function to create an SOPManager"""
    return SOPManager(sop_dir=sop_dir)
