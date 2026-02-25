"""
Phase Planner - Intelligent Phase Discovery and Reassessment
==============================================================

Evolution 2026: LLM-driven phase planning with configurable modes.

Key Features:
- Dynamic phase discovery (not templates)
- Risk-based or full reassessment
- Adaptive to code changes
- Learning from phase outcomes

Modes:
- auto: LLM decides based on context
- risk_based: Focus on high-risk areas
- full: Complete reanalysis
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.layers.layer1_strategic import ArchitectureSpec, ArchitectureParadigm
from gaap.layers.layer2_config import Layer2Config
from gaap.layers.task_schema import (
    IntelligentTask,
    Phase,
    ReassessmentResult,
    RiskLevel,
    RiskFactor,
    RiskType,
    TaskPhase,
)

logger = get_logger("gaap.layer2.phase_planner")


@dataclass
class PhaseDiscoveryContext:
    """Context for phase discovery"""

    architecture_spec: ArchitectureSpec
    original_request: str
    complexity_score: float = 0.5
    has_security_requirements: bool = False
    has_performance_requirements: bool = False
    has_integration_requirements: bool = False
    codebase_context: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseExpansionContext:
    """Context for phase expansion"""

    phase: Phase
    codebase_state: dict[str, Any]
    completed_phases: list[Phase]
    execution_signals: dict[str, Any] = field(default_factory=dict)
    previous_failures: list[Any] = field(default_factory=list)


class PhaseDiscoveryEngine:
    """
    LLM-driven phase discovery.

    Discovers optimal phases based on architecture paradigm,
    not fixed templates.
    """

    def __init__(
        self,
        provider: Any = None,
        config: Layer2Config | None = None,
    ):
        self._provider = provider
        self._config = config or Layer2Config()
        self._logger = logger

        self._discovery_count = 0
        self._llm_discoveries = 0
        self._fallback_discoveries = 0

    async def discover_phases(self, context: PhaseDiscoveryContext) -> list[Phase]:
        """
        Discover phases based on architecture spec.

        Uses LLM for intelligent discovery, falls back to paradigm-based
        phases if LLM unavailable.
        """
        self._discovery_count += 1

        mode = self._config.phase_discovery_mode

        if mode == "auto":
            mode = await self._infer_discovery_mode(context)

        if self._provider and mode == "deep":
            phases = await self._llm_discover_phases(context)
            if phases:
                self._llm_discoveries += 1
                return phases

        phases = self._fallback_discover_phases(context)
        self._fallback_discoveries += 1
        return phases

    async def _infer_discovery_mode(
        self, context: PhaseDiscoveryContext
    ) -> Literal["standard", "deep"]:
        """Let LLM decide which mode to use"""
        if not self._provider:
            return "standard"

        complexity = context.complexity_score
        has_special_reqs = (
            context.has_security_requirements
            or context.has_performance_requirements
            or context.has_integration_requirements
        )

        if complexity > 0.7 or has_special_reqs:
            return "deep"

        return "standard"

    async def _llm_discover_phases(self, context: PhaseDiscoveryContext) -> list[Phase]:
        """Use LLM to discover phases dynamically"""

        spec = context.architecture_spec

        prompt = f"""You are an expert software project planner. Discover the optimal phases for implementing this architecture.

ARCHITECTURE:
- Paradigm: {spec.paradigm.value}
- Data Strategy: {spec.data_strategy.value if spec.data_strategy else "Not specified"}
- Communication: {spec.communication.value if spec.communication else "Not specified"}
- Components: {json.dumps([c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in spec.components[:10]], indent=2) if spec.components else "Not specified"}

ORIGINAL REQUEST:
{context.original_request}

REQUIREMENTS:
- Complexity: {context.complexity_score:.2f}
- Security Requirements: {context.has_security_requirements}
- Performance Requirements: {context.has_performance_requirements}
- Integration Requirements: {context.has_integration_requirements}

TASK:
Discover 3-6 optimal phases for implementing this architecture. Each phase should:
1. Have a clear semantic goal (what it accomplishes)
2. Be ordered logically (dependencies considered)
3. Have an estimated risk level

Output ONLY valid JSON array:
[
  {{
    "name": "Phase Name",
    "description": "What this phase accomplishes",
    "order": 1,
    "semantic_goal": "The core purpose of this phase",
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "estimated_tasks": 5,
    "depends_on_phases": []
  }}
]

IMPORTANT: Generate phases specific to this architecture, not generic templates.
"""

        try:
            response = await self._provider.chat_completion(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=getattr(self._provider, "default_model", "llama-3.3-70b-versatile"),
                temperature=self._config.llm_temperature,
                max_tokens=2048,
            )

            if not response.choices:
                return []

            content = response.choices[0].message.content
            return self._parse_phase_response(content)

        except Exception as e:
            self._logger.warning(f"LLM phase discovery failed: {e}")
            return []

    def _parse_phase_response(self, content: str) -> list[Phase]:
        """Parse LLM response into Phase objects"""
        import re

        phases: list[Phase] = []

        json_match = re.search(r"\[[\s\S]*\]", content)
        if not json_match:
            return phases

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return phases

        for i, item in enumerate(data[:6]):
            try:
                risk_level = RiskLevel.MEDIUM
                risk_str = item.get("risk_level", "MEDIUM").upper()
                for level in RiskLevel:
                    if level.name == risk_str:
                        risk_level = level
                        break

                phase = Phase(
                    id=f"phase_{i + 1}",
                    name=item.get("name", f"Phase {i + 1}"),
                    description=item.get("description", ""),
                    order=item.get("order", i + 1),
                    status=TaskPhase.PLACEHOLDER,
                    semantic_goal=item.get("semantic_goal", ""),
                    risk_level=risk_level,
                    depends_on_phases=item.get("depends_on_phases", []),
                    estimated_tasks=item.get("estimated_tasks", 5),
                    estimated_duration_minutes=item.get("estimated_tasks", 5) * 10,
                )
                phases.append(phase)
            except Exception as e:
                self._logger.debug(f"Failed to parse phase {i}: {e}")
                continue

        phases.sort(key=lambda p: p.order)
        for i, phase in enumerate(phases):
            phase.order = i + 1

        return phases

    def _fallback_discover_phases(self, context: PhaseDiscoveryContext) -> list[Phase]:
        """Fallback phase discovery based on architecture paradigm"""

        spec = context.architecture_spec
        paradigm = spec.paradigm

        phases = []

        if paradigm in (ArchitectureParadigm.MICROSERVICES, ArchitectureParadigm.EVENT_DRIVEN):
            phases = self._distributed_system_phases(context)
        elif paradigm == ArchitectureParadigm.SERVERLESS:
            phases = self._serverless_phases(context)
        elif paradigm == ArchitectureParadigm.HEXAGONAL:
            phases = self._hexagonal_phases(context)
        else:
            phases = self._standard_phases(context)

        if context.has_security_requirements:
            phases = self._add_security_phase(phases, context)

        if context.has_integration_requirements:
            phases = self._add_integration_phase(phases, context)

        for i, phase in enumerate(phases):
            phase.order = i + 1
            phase.id = f"phase_{i + 1}"

        return phases

    def _standard_phases(self, context: PhaseDiscoveryContext) -> list[Phase]:
        """Standard phased approach for monoliths"""
        return [
            Phase(
                id="phase_1",
                name="Core Domain",
                description="Implement core domain models and business logic",
                order=1,
                semantic_goal="Establish the foundation of the application",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=8,
            ),
            Phase(
                id="phase_2",
                name="API Layer",
                description="Build API endpoints and controllers",
                order=2,
                semantic_goal="Expose business logic through APIs",
                risk_level=RiskLevel.LOW,
                estimated_tasks=6,
                depends_on_phases=["phase_1"],
            ),
            Phase(
                id="phase_3",
                name="Data Layer",
                description="Implement data persistence and access",
                order=3,
                semantic_goal="Enable data storage and retrieval",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=5,
                depends_on_phases=["phase_1"],
            ),
            Phase(
                id="phase_4",
                name="Testing & Quality",
                description="Comprehensive testing and quality assurance",
                order=4,
                semantic_goal="Ensure reliability and correctness",
                risk_level=RiskLevel.LOW,
                estimated_tasks=7,
                depends_on_phases=["phase_2", "phase_3"],
            ),
        ]

    def _distributed_system_phases(self, context: PhaseDiscoveryContext) -> list[Phase]:
        """Phases for microservices/event-driven architectures"""
        return [
            Phase(
                id="phase_1",
                name="Service Boundaries",
                description="Define and implement core service boundaries",
                order=1,
                semantic_goal="Establish clear service domains and contracts",
                risk_level=RiskLevel.HIGH,
                estimated_tasks=10,
            ),
            Phase(
                id="phase_2",
                name="Core Services",
                description="Implement primary business services",
                order=2,
                semantic_goal="Build the main service implementations",
                risk_level=RiskLevel.HIGH,
                estimated_tasks=12,
                depends_on_phases=["phase_1"],
            ),
            Phase(
                id="phase_3",
                name="Communication Layer",
                description="Set up service communication and messaging",
                order=3,
                semantic_goal="Enable inter-service communication",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=6,
                depends_on_phases=["phase_2"],
            ),
            Phase(
                id="phase_4",
                name="API Gateway",
                description="Implement API gateway and routing",
                order=4,
                semantic_goal="Provide unified entry point for clients",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=5,
                depends_on_phases=["phase_3"],
            ),
            Phase(
                id="phase_5",
                name="Integration Testing",
                description="End-to-end service integration tests",
                order=5,
                semantic_goal="Verify service interactions",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=8,
                depends_on_phases=["phase_4"],
            ),
        ]

    def _serverless_phases(self, context: PhaseDiscoveryContext) -> list[Phase]:
        """Phases for serverless architectures"""
        return [
            Phase(
                id="phase_1",
                name="Function Design",
                description="Design and implement core functions",
                order=1,
                semantic_goal="Create the serverless function handlers",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=8,
            ),
            Phase(
                id="phase_2",
                name="Event Triggers",
                description="Configure event sources and triggers",
                order=2,
                semantic_goal="Set up function invocation paths",
                risk_level=RiskLevel.LOW,
                estimated_tasks=4,
                depends_on_phases=["phase_1"],
            ),
            Phase(
                id="phase_3",
                name="Data Integration",
                description="Connect functions to data stores",
                order=3,
                semantic_goal="Enable persistent data access",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=5,
                depends_on_phases=["phase_2"],
            ),
            Phase(
                id="phase_4",
                name="Testing & Deployment",
                description="Test and deploy functions",
                order=4,
                semantic_goal="Verify and release to production",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=6,
                depends_on_phases=["phase_3"],
            ),
        ]

    def _hexagonal_phases(self, context: PhaseDiscoveryContext) -> list[Phase]:
        """Phases for hexagonal architectures"""
        return [
            Phase(
                id="phase_1",
                name="Domain Core",
                description="Implement pure domain logic",
                order=1,
                semantic_goal="Establish business rules and domain models",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=10,
            ),
            Phase(
                id="phase_2",
                name="Ports",
                description="Define ports (interfaces)",
                order=2,
                semantic_goal="Specify input/output boundaries",
                risk_level=RiskLevel.LOW,
                estimated_tasks=4,
                depends_on_phases=["phase_1"],
            ),
            Phase(
                id="phase_3",
                name="Adapters",
                description="Implement adapters for external systems",
                order=3,
                semantic_goal="Connect domain to external world",
                risk_level=RiskLevel.MEDIUM,
                estimated_tasks=8,
                depends_on_phases=["phase_2"],
            ),
            Phase(
                id="phase_4",
                name="Integration",
                description="Wire everything together",
                order=4,
                semantic_goal="Complete the hexagon",
                risk_level=RiskLevel.LOW,
                estimated_tasks=5,
                depends_on_phases=["phase_3"],
            ),
        ]

    def _add_security_phase(
        self, phases: list[Phase], context: PhaseDiscoveryContext
    ) -> list[Phase]:
        """Add security phase if needed"""
        security_phase = Phase(
            id="phase_security",
            name="Security Hardening",
            description="Implement security controls and audits",
            order=len(phases) + 1,
            semantic_goal="Protect the application from threats",
            risk_level=RiskLevel.HIGH,
            estimated_tasks=6,
            depends_on_phases=[p.id for p in phases[-2:]] if len(phases) >= 2 else [],
        )
        phases.append(security_phase)
        return phases

    def _add_integration_phase(
        self, phases: list[Phase], context: PhaseDiscoveryContext
    ) -> list[Phase]:
        """Add integration phase if needed"""
        integration_phase = Phase(
            id="phase_integration",
            name="External Integrations",
            description="Connect to external systems and APIs",
            order=len(phases),
            semantic_goal="Enable interoperability with external services",
            risk_level=RiskLevel.MEDIUM,
            estimated_tasks=5,
            depends_on_phases=[p.id for p in phases[-2:]] if len(phases) >= 2 else [],
        )
        phases.insert(-1, integration_phase)
        return phases

    def get_stats(self) -> dict[str, int]:
        """Get discovery statistics"""
        return {
            "total_discoveries": self._discovery_count,
            "llm_discoveries": self._llm_discoveries,
            "fallback_discoveries": self._fallback_discoveries,
        }


class PhaseReassessor:
    """
    Intelligent phase reassessment after completion.

    Supports:
    - auto: LLM decides risk_based or full
    - risk_based: Analyze high-risk areas only
    - full: Complete reanalysis
    """

    def __init__(
        self,
        provider: Any = None,
        config: Layer2Config | None = None,
    ):
        self._provider = provider
        self._config = config or Layer2Config()
        self._logger = logger

        self._reassessments = 0
        self._replans_triggered = 0

    async def reassess(
        self,
        completed_phase: Phase,
        code_changes: dict[str, str],
        remaining_phases: list[Phase],
        execution_signals: dict[str, Any],
    ) -> ReassessmentResult:
        """
        Reassess remaining phases after a phase completes.

        Mode depends on config:
        - auto: LLM decides based on complexity
        - risk_based: Focus on risk areas
        - full: Complete analysis
        """
        start_time = time.time()
        self._reassessments += 1

        mode = self._config.phase_reassessment_mode

        if mode == "auto":
            mode = await self._infer_reassessment_mode(
                completed_phase, remaining_phases, execution_signals
            )

        if self._provider:
            result = await self._llm_reassess(
                completed_phase, code_changes, remaining_phases, execution_signals, mode
            )
        else:
            result = self._fallback_reassess(completed_phase, code_changes, remaining_phases)

        result.analysis_mode = mode
        result.analysis_duration_ms = int((time.time() - start_time) * 1000)

        if result.replan_needed:
            self._replans_triggered += 1
            completed_phase.reassessment_count += 1

        return result

    async def _infer_reassessment_mode(
        self,
        completed_phase: Phase,
        remaining_phases: list[Phase],
        execution_signals: dict[str, Any],
    ) -> Literal["risk_based", "full"]:
        """Let LLM decide which mode to use"""

        had_failures = execution_signals.get("failed_tasks", 0) > 0
        high_risk_remaining = any(
            p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) for p in remaining_phases
        )
        complexity = completed_phase.estimated_complexity

        if had_failures or high_risk_remaining or complexity > 0.7:
            return "full"

        return "risk_based"

    async def _llm_reassess(
        self,
        completed_phase: Phase,
        code_changes: dict[str, str],
        remaining_phases: list[Phase],
        execution_signals: dict[str, Any],
        mode: str,
    ) -> ReassessmentResult:
        """Use LLM for intelligent reassessment"""

        changes_summary = self._summarize_changes(code_changes)

        remaining_summary = "\n".join(
            [f"- {p.name}: {p.description} (Risk: {p.risk_level.name})" for p in remaining_phases]
        )

        prompt = f"""You are an expert project planner. Analyze the impact of completed work on remaining phases.

COMPLETED PHASE:
- Name: {completed_phase.name}
- Goal: {completed_phase.semantic_goal}
- Status: {completed_phase.status.name}
- Tasks Completed: {completed_phase.actual_tasks}

CODE CHANGES SUMMARY:
{changes_summary[:2000]}

EXECUTION SIGNALS:
- Failed Tasks: {execution_signals.get("failed_tasks", 0)}
- Warnings: {execution_signals.get("warnings", [])}
- Performance: {execution_signals.get("performance_notes", "Normal")}

REMAINING PHASES:
{remaining_summary}

ANALYSIS MODE: {mode.upper()}

TASK:
Analyze whether remaining phases need to be adjusted based on the completed work.
{"Focus on HIGH and CRITICAL risk areas." if mode == "risk_based" else "Perform complete analysis of all phases."}

Output ONLY valid JSON:
{{
  "replan_needed": true|false,
  "reasoning": "Why replan is or isn't needed",
  "confidence": 0.0-1.0,
  "new_tasks_to_inject": [
    {{
      "name": "Task name",
      "description": "What it does",
      "category": "setup|api|testing|...",
      "phase_id": "phase_X",
      "priority": 1-4,
      "semantic_intent": "The core intent"
    }}
  ],
  "tasks_to_remove": ["task_id_1"],
  "new_risks": [
    {{
      "type": "BREAKING_CHANGE|SECURITY|PERFORMANCE|...",
      "level": "LOW|MEDIUM|HIGH|CRITICAL",
      "description": "Risk description",
      "mitigation": "How to mitigate"
    }}
  ],
  "affected_files": ["file1.py"],
  "affected_components": ["component1"]
}}
"""

        try:
            response = await self._provider.chat_completion(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=getattr(self._provider, "default_model", "llama-3.3-70b-versatile"),
                temperature=self._config.llm_temperature,
                max_tokens=2048,
            )

            if not response.choices:
                return self._fallback_reassess(completed_phase, code_changes, remaining_phases)

            content = response.choices[0].message.content
            return self._parse_reassessment_response(content, remaining_phases)

        except Exception as e:
            self._logger.warning(f"LLM reassessment failed: {e}")
            return self._fallback_reassess(completed_phase, code_changes, remaining_phases)

    def _summarize_changes(self, code_changes: dict[str, str]) -> str:
        """Summarize code changes for LLM context"""
        if not code_changes:
            return "No code changes recorded"

        summary = []
        for file_path, content in list(code_changes.items())[:10]:
            lines = len(content.split("\n"))
            summary.append(f"- {file_path}: {lines} lines")

        return "\n".join(summary)

    def _parse_reassessment_response(
        self,
        content: str,
        remaining_phases: list[Phase],
    ) -> ReassessmentResult:
        """Parse LLM reassessment response"""
        import re

        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return ReassessmentResult(replan_needed=False, reasoning="Parse error", confidence=0.5)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ReassessmentResult(replan_needed=False, reasoning="JSON error", confidence=0.5)

        new_tasks = []
        for task_data in data.get("new_tasks_to_inject", []):
            task = IntelligentTask(
                id=f"injected_{int(time.time() * 1000)}",
                name=task_data.get("name", "Injected Task"),
                description=task_data.get("description", ""),
                category=task_data.get("category", "setup"),
                phase_id=task_data.get("phase_id"),
                priority=task_data.get("priority", 2),
                semantic_intent=task_data.get("semantic_intent", ""),
            )
            new_tasks.append(task)

        new_risks = []
        for risk_data in data.get("new_risks", []):
            try:
                risk = RiskFactor(
                    type=RiskType[risk_data.get("type", "UNKNOWN")],
                    level=RiskLevel[risk_data.get("level", "MEDIUM")],
                    description=risk_data.get("description", ""),
                    mitigation=risk_data.get("mitigation", ""),
                )
                new_risks.append(risk)
            except (KeyError, ValueError):
                continue

        return ReassessmentResult(
            replan_needed=data.get("replan_needed", False),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5),
            new_tasks_to_inject=new_tasks,
            tasks_to_remove=data.get("tasks_to_remove", []),
            new_risks_identified=new_risks,
            affected_files=data.get("affected_files", []),
            affected_components=data.get("affected_components", []),
        )

    def _fallback_reassess(
        self,
        completed_phase: Phase,
        code_changes: dict[str, str],
        remaining_phases: list[Phase],
    ) -> ReassessmentResult:
        """Simple fallback reassessment"""

        has_changes = len(code_changes) > 0
        high_risk_remaining = any(
            p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) for p in remaining_phases
        )

        replan_needed = has_changes and high_risk_remaining

        return ReassessmentResult(
            replan_needed=replan_needed,
            reasoning="Fallback: Replan recommended if code changes affect high-risk phases",
            confidence=0.6,
            affected_files=list(code_changes.keys())[:5] if code_changes else [],
        )

    def get_stats(self) -> dict[str, int]:
        """Get reassessment statistics"""
        return {
            "total_reassessments": self._reassessments,
            "replans_triggered": self._replans_triggered,
        }


class PhaseExpander:
    """
    Expands phases into atomic tasks.

    Uses LLM for intelligent task generation.
    """

    def __init__(
        self,
        provider: Any = None,
        config: Layer2Config | None = None,
    ):
        self._provider = provider
        self._config = config or Layer2Config()
        self._logger = logger

    async def expand_phase(self, context: PhaseExpansionContext) -> list[IntelligentTask]:
        """
        Expand a phase into atomic tasks.
        """
        phase = context.phase

        if self._provider:
            tasks = await self._llm_expand_phase(context)
            if tasks:
                phase.tasks = tasks
                phase.status = TaskPhase.EXPANDED
                phase.expanded_at = datetime.now()
                return tasks

        tasks = self._fallback_expand_phase(phase)
        phase.tasks = tasks
        phase.status = TaskPhase.EXPANDED
        phase.expanded_at = datetime.now()
        return tasks

    async def _llm_expand_phase(self, context: PhaseExpansionContext) -> list[IntelligentTask]:
        """Use LLM to generate tasks for a phase"""

        phase = context.phase

        prompt = f"""You are an expert software engineer. Generate atomic tasks for this phase.

PHASE:
- Name: {phase.name}
- Description: {phase.description}
- Goal: {phase.semantic_goal}
- Risk Level: {phase.risk_level.name}

COMPLETED PHASES:
{self._format_completed_phases(context.completed_phases)}

Generate 3-8 atomic tasks for this phase. Each task should be:
1. Specific and actionable
2. Have clear inputs and outputs
3. Include tool recommendations
4. Identify potential risks

Output ONLY valid JSON:
[
  {{
    "name": "Task name",
    "description": "Detailed description",
    "category": "setup|api|database|testing|...",
    "semantic_intent": "Core intent",
    "semantic_scope": ["file1.py", "module1"],
    "inputs": [
      {{"name": "input1", "type": "file", "description": "Input file"}}
    ],
    "outputs": [
      {{"name": "output1", "type": "file", "description": "Output file"}}
    ],
    "tools": [
      {{"name": "pytest", "reason": "For testing", "priority": 1}}
    ],
    "risks": [
      {{"type": "BREAKING_CHANGE", "level": "MEDIUM", "description": "Risk desc"}}
    ],
    "priority": 1,
    "estimated_minutes": 15
  }}
]
"""

        try:
            response = await self._provider.chat_completion(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=getattr(self._provider, "default_model", "llama-3.3-70b-versatile"),
                temperature=self._config.llm_temperature,
                max_tokens=4096,
            )

            if not response.choices:
                return []

            content = response.choices[0].message.content
            return self._parse_tasks_response(content, phase.id)

        except Exception as e:
            self._logger.warning(f"LLM phase expansion failed: {e}")
            return []

    def _format_completed_phases(self, phases: list[Phase]) -> str:
        if not phases:
            return "None"
        return "\n".join([f"- {p.name}: {p.semantic_goal}" for p in phases])

    def _parse_tasks_response(
        self,
        content: str,
        phase_id: str,
    ) -> list[IntelligentTask]:
        """Parse LLM tasks response"""
        import re

        tasks: list[IntelligentTask] = []

        json_match = re.search(r"\[[\s\S]*\]", content)
        if not json_match:
            return tasks

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return tasks

        for i, task_data in enumerate(data[:10]):
            try:
                task = IntelligentTask(
                    id=f"{phase_id}_task_{i + 1}",
                    name=task_data.get("name", f"Task {i + 1}"),
                    description=task_data.get("description", ""),
                    category=task_data.get("category", "setup"),
                    phase_id=phase_id,
                    semantic_intent=task_data.get("semantic_intent", ""),
                    semantic_scope=task_data.get("semantic_scope", []),
                    priority=task_data.get("priority", 2),
                    estimated_duration_minutes=task_data.get("estimated_minutes", 10),
                )

                for inp in task_data.get("inputs", []):
                    from gaap.layers.task_schema import SchemaDefinition

                    task.input_schema[inp["name"]] = SchemaDefinition(
                        name=inp["name"],
                        type=inp.get("type", "any"),
                        description=inp.get("description", ""),
                    )

                for outp in task_data.get("outputs", []):
                    from gaap.layers.task_schema import SchemaDefinition

                    task.output_schema[outp["name"]] = SchemaDefinition(
                        name=outp["name"],
                        type=outp.get("type", "any"),
                        description=outp.get("description", ""),
                    )

                for tool in task_data.get("tools", []):
                    from gaap.layers.task_schema import ToolRecommendation

                    task.recommended_tools.append(
                        ToolRecommendation(
                            tool=tool["name"],
                            reason=tool.get("reason", ""),
                            priority=tool.get("priority", 1),
                        )
                    )

                for risk in task_data.get("risks", []):
                    from gaap.layers.task_schema import RiskFactor, RiskType, RiskLevel

                    try:
                        task.risk_factors.append(
                            RiskFactor(
                                type=RiskType[risk.get("type", "UNKNOWN")],
                                level=RiskLevel[risk.get("level", "MEDIUM")],
                                description=risk.get("description", ""),
                            )
                        )
                    except (KeyError, ValueError):
                        pass

                tasks.append(task)
            except Exception as e:
                self._logger.debug(f"Failed to parse task {i}: {e}")
                continue

        return tasks

    def _fallback_expand_phase(self, phase: Phase) -> list[IntelligentTask]:
        """Fallback task generation for a phase"""

        tasks = []
        num_tasks = phase.estimated_tasks or 4

        for i in range(num_tasks):
            task = IntelligentTask(
                id=f"{phase.id}_task_{i + 1}",
                name=f"Implement {phase.name} - Part {i + 1}",
                description=f"Implementation task {i + 1} for {phase.name}",
                category="setup",
                phase_id=phase.id,
                semantic_intent=f"Contribute to {phase.semantic_goal}",
                priority=2,
                estimated_duration_minutes=10,
            )
            tasks.append(task)

        if len(tasks) > 1:
            for i in range(1, len(tasks)):
                tasks[i].dependencies.append(tasks[i - 1].id)

        return tasks


def create_phase_planner(
    provider: Any = None,
    config: Layer2Config | None = None,
) -> tuple[PhaseDiscoveryEngine, PhaseReassessor, PhaseExpander]:
    """Factory function to create phase planning components"""
    config = config or Layer2Config()

    discovery = PhaseDiscoveryEngine(provider=provider, config=config)
    reassessor = PhaseReassessor(provider=provider, config=config)
    expander = PhaseExpander(provider=provider, config=config)

    return discovery, reassessor, expander
