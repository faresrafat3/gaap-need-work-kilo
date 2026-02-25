"""
Intelligent Task Schema - DSPy-Style Structural Definitions
============================================================

Evolution 2026: Tasks with semantic understanding, dynamic schemas,
and intelligent tool recommendations.

Key Features:
- Semantic intent extraction (LLM-generated)
- Dynamic input/output schemas
- Tool recommendations with reasoning
- Risk assessment per task
- Success indicators (not generic acceptance criteria)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Literal
import hashlib
import json


class TaskPhase(Enum):
    """Phase status for rolling wave planning"""

    PLACEHOLDER = auto()  # Not yet expanded
    EXPANDING = auto()  # Currently being expanded
    EXPANDED = auto()  # Fully expanded into atomic tasks
    EXECUTING = auto()  # Tasks are being executed
    COMPLETED = auto()  # All tasks completed
    FAILED = auto()  # Phase failed


class RiskLevel(Enum):
    """Risk level for tasks"""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class RiskType(Enum):
    """Types of risks"""

    BREAKING_CHANGE = auto()
    DATA_LOSS = auto()
    SECURITY = auto()
    PERFORMANCE = auto()
    DEPENDENCY = auto()
    INTEGRATION = auto()
    RESOURCE = auto()
    UNKNOWN = auto()


@dataclass
class SchemaDefinition:
    """
    Detailed schema definition for input/output.

    DSPy-style structured specification.
    """

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    constraints: dict[str, Any] = field(default_factory=dict)
    examples: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "constraints": self.constraints,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaDefinition":
        return cls(
            name=data.get("name", ""),
            type=data.get("type", "any"),
            description=data.get("description", ""),
            required=data.get("required", True),
            default=data.get("default"),
            constraints=data.get("constraints", {}),
            examples=data.get("examples", []),
        )


@dataclass
class ToolRecommendation:
    """
    Tool recommendation with reasoning.

    Not just "use pytest" but WHY.
    """

    tool: str
    reason: str
    priority: int = 1  # 1 = highest
    alternatives: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "reason": self.reason,
            "priority": self.priority,
            "alternatives": self.alternatives,
            "parameters": self.parameters,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolRecommendation":
        return cls(
            tool=data.get("tool", ""),
            reason=data.get("reason", ""),
            priority=data.get("priority", 1),
            alternatives=data.get("alternatives", []),
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 0.8),
        )


@dataclass
class RiskFactor:
    """
    Identified risk factor for a task.
    """

    type: RiskType
    level: RiskLevel
    description: str
    affected_files: list[str] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)
    mitigation: str = ""
    probability: float = 0.5
    impact: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.name,
            "level": self.level.name,
            "description": self.description,
            "affected_files": self.affected_files,
            "affected_components": self.affected_components,
            "mitigation": self.mitigation,
            "probability": self.probability,
            "impact": self.impact,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskFactor":
        return cls(
            type=RiskType[data.get("type", "UNKNOWN")],
            level=RiskLevel[data.get("level", "MEDIUM")],
            description=data.get("description", ""),
            affected_files=data.get("affected_files", []),
            affected_components=data.get("affected_components", []),
            mitigation=data.get("mitigation", ""),
            probability=data.get("probability", 0.5),
            impact=data.get("impact", 0.5),
        )


@dataclass
class SuccessIndicator:
    """
    Specific, measurable success criteria for a task.

    Not generic "works correctly" but specific metrics.
    """

    indicator_type: str  # test_coverage, performance, correctness, etc.
    description: str
    target: Any  # Target value
    scope: str = ""  # What scope this applies to
    measurement_method: str = ""  # How to measure
    weight: float = 1.0  # Importance weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator_type": self.indicator_type,
            "description": self.description,
            "target": self.target,
            "scope": self.scope,
            "measurement_method": self.measurement_method,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuccessIndicator":
        return cls(
            indicator_type=data.get("indicator_type", "correctness"),
            description=data.get("description", ""),
            target=data.get("target"),
            scope=data.get("scope", ""),
            measurement_method=data.get("measurement_method", ""),
            weight=data.get("weight", 1.0),
        )


@dataclass
class IntelligentTask:
    """
    Enhanced task with semantic understanding and structured schemas.

    This replaces simple task descriptions with rich, LLM-generated
    specifications that enable:
    - Precise dependency detection
    - Accurate tool selection
    - Risk-aware execution
    - Measurable success criteria
    """

    # Core identification
    id: str
    name: str
    description: str
    category: str

    # Phase membership (for rolling wave planning)
    phase_id: str | None = None
    expansion_status: Literal["placeholder", "expanded"] = "expanded"

    # Semantic understanding (LLM-extracted)
    semantic_intent: str = ""  # What the task REALLY wants to accomplish
    semantic_scope: list[str] = field(default_factory=list)  # Files/modules affected
    semantic_keywords: list[str] = field(default_factory=list)  # Key concepts

    # Dynamic schemas (DSPy-style)
    input_schema: dict[str, SchemaDefinition] = field(default_factory=dict)
    output_schema: dict[str, SchemaDefinition] = field(default_factory=dict)

    # Tool recommendations with reasoning
    recommended_tools: list[ToolRecommendation] = field(default_factory=list)

    # Risk assessment
    risk_factors: list[RiskFactor] = field(default_factory=list)
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM

    # Success criteria (specific, not generic)
    success_indicators: list[SuccessIndicator] = field(default_factory=list)

    # Dependencies
    dependencies: list[str] = field(default_factory=list)
    dependency_reasoning: dict[str, str] = field(default_factory=dict)  # Why each dep

    # Execution metadata
    estimated_complexity: float = 0.5  # 0.0-1.0
    estimated_duration_minutes: int = 5
    priority: int = 2  # 1=critical, 2=high, 3=normal, 4=low

    # Status
    status: Literal["pending", "ready", "executing", "completed", "failed", "skipped"] = "pending"
    result: dict[str, Any] = field(default_factory=dict)

    # Learning
    lessons_learned: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_id_hash(self) -> str:
        """Get unique hash for this task"""
        content = f"{self.name}:{self.description}:{self.semantic_intent}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "phase_id": self.phase_id,
            "expansion_status": self.expansion_status,
            "semantic_intent": self.semantic_intent,
            "semantic_scope": self.semantic_scope,
            "semantic_keywords": self.semantic_keywords,
            "input_schema": {k: v.to_dict() for k, v in self.input_schema.items()},
            "output_schema": {k: v.to_dict() for k, v in self.output_schema.items()},
            "recommended_tools": [t.to_dict() for t in self.recommended_tools],
            "risk_factors": [r.to_dict() for r in self.risk_factors],
            "overall_risk_level": self.overall_risk_level.name,
            "success_indicators": [s.to_dict() for s in self.success_indicators],
            "dependencies": self.dependencies,
            "dependency_reasoning": self.dependency_reasoning,
            "estimated_complexity": self.estimated_complexity,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "priority": self.priority,
            "status": self.status,
            "result": self.result,
            "lessons_learned": self.lessons_learned,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntelligentTask":
        """Create from dictionary"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            phase_id=data.get("phase_id"),
            expansion_status=data.get("expansion_status", "expanded"),
            semantic_intent=data.get("semantic_intent", ""),
            semantic_scope=data.get("semantic_scope", []),
            semantic_keywords=data.get("semantic_keywords", []),
            input_schema={
                k: SchemaDefinition.from_dict(v) for k, v in data.get("input_schema", {}).items()
            },
            output_schema={
                k: SchemaDefinition.from_dict(v) for k, v in data.get("output_schema", {}).items()
            },
            recommended_tools=[
                ToolRecommendation.from_dict(t) for t in data.get("recommended_tools", [])
            ],
            risk_factors=[RiskFactor.from_dict(r) for r in data.get("risk_factors", [])],
            overall_risk_level=RiskLevel[data.get("overall_risk_level", "MEDIUM")],
            success_indicators=[
                SuccessIndicator.from_dict(s) for s in data.get("success_indicators", [])
            ],
            dependencies=data.get("dependencies", []),
            dependency_reasoning=data.get("dependency_reasoning", {}),
            estimated_complexity=data.get("estimated_complexity", 0.5),
            estimated_duration_minutes=data.get("estimated_duration_minutes", 5),
            priority=data.get("priority", 2),
            status=data.get("status", "pending"),
            result=data.get("result", {}),
            lessons_learned=data.get("lessons_learned", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            metadata=data.get("metadata", {}),
        )

    def is_high_risk(self) -> bool:
        """Check if task is high risk"""
        return self.overall_risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def get_sorted_tools(self) -> list[ToolRecommendation]:
        """Get tools sorted by priority"""
        return sorted(self.recommended_tools, key=lambda t: t.priority)

    def get_risk_score(self) -> float:
        """Calculate overall risk score (0.0-1.0)"""
        if not self.risk_factors:
            return 0.0

        total = 0.0
        for risk in self.risk_factors:
            level_score = {
                RiskLevel.LOW: 0.25,
                RiskLevel.MEDIUM: 0.5,
                RiskLevel.HIGH: 0.75,
                RiskLevel.CRITICAL: 1.0,
            }.get(risk.level, 0.5)
            total += level_score * risk.probability * risk.impact

        return min(total / len(self.risk_factors), 1.0)


@dataclass
class Phase:
    """
    High-level phase (epic) for rolling wave planning.

    Phases start as placeholders and get expanded into tasks
    when they become active.
    """

    id: str
    name: str
    description: str
    order: int  # Phase order (1, 2, 3, ...)

    # Status
    status: TaskPhase = TaskPhase.PLACEHOLDER

    # Tasks (empty until expanded)
    tasks: list[IntelligentTask] = field(default_factory=list)

    # Phase-level metadata
    semantic_goal: str = ""  # What this phase accomplishes
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # Dependencies on other phases
    depends_on_phases: list[str] = field(default_factory=list)

    # Estimated metrics
    estimated_tasks: int = 0
    estimated_duration_minutes: int = 0
    estimated_complexity: float = 0.5

    # Actual metrics (after completion)
    actual_tasks: int = 0
    actual_duration_minutes: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    expanded_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Reassessment tracking
    reassessment_count: int = 0
    last_reassessment_reason: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "order": self.order,
            "status": self.status.name,
            "tasks": [t.to_dict() for t in self.tasks],
            "semantic_goal": self.semantic_goal,
            "risk_level": self.risk_level.name,
            "depends_on_phases": self.depends_on_phases,
            "estimated_tasks": self.estimated_tasks,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "estimated_complexity": self.estimated_complexity,
            "actual_tasks": self.actual_tasks,
            "actual_duration_minutes": self.actual_duration_minutes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expanded_at": self.expanded_at.isoformat() if self.expanded_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "reassessment_count": self.reassessment_count,
            "last_reassessment_reason": self.last_reassessment_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Phase":
        """Create from dictionary"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            order=data.get("order", 0),
            status=TaskPhase[data.get("status", "PLACEHOLDER")],
            tasks=[IntelligentTask.from_dict(t) for t in data.get("tasks", [])],
            semantic_goal=data.get("semantic_goal", ""),
            risk_level=RiskLevel[data.get("risk_level", "MEDIUM")],
            depends_on_phases=data.get("depends_on_phases", []),
            estimated_tasks=data.get("estimated_tasks", 0),
            estimated_duration_minutes=data.get("estimated_duration_minutes", 0),
            estimated_complexity=data.get("estimated_complexity", 0.5),
            actual_tasks=data.get("actual_tasks", 0),
            actual_duration_minutes=data.get("actual_duration_minutes", 0),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            expanded_at=datetime.fromisoformat(data["expanded_at"])
            if data.get("expanded_at")
            else None,
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            reassessment_count=data.get("reassessment_count", 0),
            last_reassessment_reason=data.get("last_reassessment_reason", ""),
            metadata=data.get("metadata", {}),
        )

    def is_ready_to_expand(self) -> bool:
        """Check if phase is ready for expansion"""
        if self.status != TaskPhase.PLACEHOLDER:
            return False

        # Check dependencies
        # This would be checked against completed phases
        return True

    def is_ready_to_execute(self) -> bool:
        """Check if phase is ready for execution"""
        return self.status == TaskPhase.EXPANDED and len(self.tasks) > 0

    def get_progress(self) -> float:
        """Get completion progress (0.0-1.0)"""
        if not self.tasks:
            return 0.0

        completed = sum(1 for t in self.tasks if t.status == "completed")
        return completed / len(self.tasks)


@dataclass
class ReassessmentResult:
    """
    Result of phase reassessment after completion.

    LLM-generated analysis of what needs to change.
    """

    # Decision
    replan_needed: bool
    reasoning: str
    confidence: float

    # Changes
    new_tasks_to_inject: list[IntelligentTask] = field(default_factory=list)
    tasks_to_remove: list[str] = field(default_factory=list)
    tasks_to_modify: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Phase changes
    phases_to_add: list[Phase] = field(default_factory=list)
    phases_to_remove: list[str] = field(default_factory=list)
    phases_to_reorder: dict[str, int] = field(default_factory=dict)

    # Risk alerts
    new_risks_identified: list[RiskFactor] = field(default_factory=list)

    # Dependencies
    new_dependencies: list[tuple[str, str, str]] = field(default_factory=list)  # (from, to, reason)
    dependencies_to_remove: list[tuple[str, str]] = field(default_factory=list)  # (from, to)

    # Impact analysis
    affected_files: list[str] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)

    # Metadata
    analysis_mode: str = "risk_based"  # or "full"
    analysis_duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "replan_needed": self.replan_needed,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "new_tasks_to_inject": [t.to_dict() for t in self.new_tasks_to_inject],
            "tasks_to_remove": self.tasks_to_remove,
            "tasks_to_modify": self.tasks_to_modify,
            "phases_to_add": [p.to_dict() for p in self.phases_to_add],
            "phases_to_remove": self.phases_to_remove,
            "phases_to_reorder": self.phases_to_reorder,
            "new_risks_identified": [r.to_dict() for r in self.new_risks_identified],
            "new_dependencies": self.new_dependencies,
            "dependencies_to_remove": self.dependencies_to_remove,
            "affected_files": self.affected_files,
            "affected_components": self.affected_components,
            "analysis_mode": self.analysis_mode,
            "analysis_duration_ms": self.analysis_duration_ms,
        }
