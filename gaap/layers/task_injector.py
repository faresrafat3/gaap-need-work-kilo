"""
Task Injector - Dynamic Task Injection
=======================================

Evolution 2026: Intelligent task injection with autonomy levels.

Key Features:
- Analyzes execution context to detect missing tasks
- Supports multiple autonomy levels
- Risk-aware injection decisions
- Learns from past injection patterns

Autonomy Levels:
- auto: LLM decides based on risk and context
- autonomous: Always inject without asking
- semi: Inject low-risk, ask for high-risk
- conservative: Always ask before injecting
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.layers.layer2_config import Layer2Config
from gaap.layers.task_schema import (
    IntelligentTask,
    Phase,
    RiskLevel,
    RiskFactor,
    RiskType,
    TaskPhase,
)

logger = get_logger("gaap.layer2.injector")


@dataclass
class InjectionContext:
    """Context for task injection decision"""

    current_phase: Phase
    completed_phases: list[Phase]
    pending_tasks: list[IntelligentTask]
    execution_signals: dict[str, Any] = field(default_factory=dict)
    code_changes: dict[str, str] = field(default_factory=dict)
    failed_tasks: list[Any] = field(default_factory=list)
    architecture_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class InjectionDecision:
    """Decision about task injection"""

    should_inject: bool
    tasks_to_inject: list[IntelligentTask] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5
    requires_user_approval: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    injection_type: str = "adaptive"


@dataclass
class InjectionPattern:
    """Learned pattern for task injection"""

    trigger_conditions: list[str]
    task_template: dict[str, Any]
    success_rate: float = 0.5
    times_applied: int = 0
    last_applied: float = 0.0


class DynamicTaskInjector:
    """
    Intelligent task injector with configurable autonomy.

    Analyzes execution context and injects tasks when:
    - Missing dependencies detected
    - Failed tasks need recovery
    - Security/testing gaps identified
    - Integration points missed
    """

    INJECTION_TYPES = {
        "dependency_gap": "Missing dependency detected",
        "failure_recovery": "Task to recover from failure",
        "security_audit": "Security check needed",
        "integration_test": "Integration test missing",
        "data_migration": "Data migration step",
        "validation": "Validation step needed",
        "documentation": "Documentation task",
        "cleanup": "Cleanup task needed",
    }

    def __init__(
        self,
        provider: Any = None,
        config: Layer2Config | None = None,
    ):
        self._provider = provider
        self._config = config or Layer2Config()
        self._logger = logger

        self._injections = 0
        self._approved_injections = 0
        self._rejected_injections = 0
        self._autonomous_injections = 0

        self._patterns: list[InjectionPattern] = []
        self._injection_history: list[dict[str, Any]] = []

    async def analyze_and_inject(
        self,
        context: InjectionContext,
    ) -> InjectionDecision:
        """
        Analyze context and decide if tasks need injection.

        Uses configured autonomy level:
        - auto: LLM decides based on risk
        - autonomous: Always inject
        - semi: Inject low-risk, ask for high
        - conservative: Always ask
        """
        start_time = time.time()
        self._injections += 1

        autonomy = self._config.injection_autonomy
        if autonomy == "auto":
            autonomy = await self._infer_autonomy(context)

        decision = await self._analyze_for_injection(context)

        if not decision.should_inject or not decision.tasks_to_inject:
            return decision

        decision.risk_level = self._assess_injection_risk(decision.tasks_to_inject)

        if autonomy == "autonomous":
            decision.requires_user_approval = False
            self._autonomous_injections += 1
        elif autonomy == "conservative":
            decision.requires_user_approval = True
        elif autonomy == "semi":
            risk_score = self._get_risk_score(decision.risk_level)
            threshold = self._config.injection_risk_threshold
            decision.requires_user_approval = risk_score >= threshold

        elapsed = time.time() - start_time
        self._logger.info(
            f"Injection analysis: {len(decision.tasks_to_inject)} tasks, "
            f"autonomy={autonomy}, approval={decision.requires_user_approval}, "
            f"{elapsed:.2f}s"
        )

        self._injection_history.append(
            {
                "timestamp": time.time(),
                "injection_type": decision.injection_type,
                "tasks_count": len(decision.tasks_to_inject),
                "autonomy": autonomy,
                "requires_approval": decision.requires_user_approval,
                "risk_level": decision.risk_level.name,
            }
        )

        return decision

    async def _infer_autonomy(
        self,
        context: InjectionContext,
    ) -> Literal["autonomous", "semi", "conservative"]:
        """Let LLM decide autonomy level"""

        has_failures = len(context.failed_tasks) > 0
        high_risk_phase = context.current_phase.risk_level in (
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        )

        if high_risk_phase and has_failures:
            return "conservative"
        elif has_failures:
            return "semi"
        else:
            return "autonomous"

    async def _analyze_for_injection(
        self,
        context: InjectionContext,
    ) -> InjectionDecision:
        """Analyze if tasks need to be injected"""

        if self._provider:
            decision = await self._llm_analyze_injection(context)
            if decision:
                return decision

        return await self._rule_based_analysis(context)

    async def _llm_analyze_injection(
        self,
        context: InjectionContext,
    ) -> InjectionDecision | None:
        """Use LLM for intelligent injection analysis"""

        phase = context.current_phase
        completed_summary = self._summarize_completed_phases(context.completed_phases)
        pending_summary = self._summarize_pending_tasks(context.pending_tasks)
        signals_summary = self._format_signals(context.execution_signals)

        prompt = f"""You are an expert project manager. Analyze the current execution state to identify missing tasks.

CURRENT PHASE:
- Name: {phase.name}
- Goal: {phase.semantic_goal}
- Status: {phase.status.name}
- Risk Level: {phase.risk_level.name}

COMPLETED PHASES:
{completed_summary}

PENDING TASKS:
{pending_summary}

EXECUTION SIGNALS:
{signals_summary}

FAILED TASKS:
{len(context.failed_tasks)} failures detected

CODE CHANGES:
{len(context.code_changes)} files modified

Analyze and identify if any tasks are missing that should be injected. Consider:
1. Missing dependency tasks
2. Recovery tasks for failures
3. Security or validation tasks
4. Integration tasks
5. Documentation tasks

Output ONLY valid JSON:
{{
  "should_inject": true|false,
  "reasoning": "Why injection is or isn't needed",
  "confidence": 0.0-1.0,
  "injection_type": "dependency_gap|failure_recovery|security_audit|...",
  "tasks": [
    {{
      "name": "Task name",
      "description": "What it does",
      "category": "setup|api|testing|security|...",
      "priority": 1-4,
      "semantic_intent": "Core intent",
      "risk_level": "LOW|MEDIUM|HIGH"
    }}
  ]
}}

IMPORTANT: Only inject tasks that are truly missing and necessary.
"""

        try:
            response = await self._provider.chat_completion(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=getattr(self._provider, "default_model", "llama-3.3-70b-versatile"),
                temperature=self._config.llm_temperature,
                max_tokens=2048,
            )

            if not response.choices:
                return None

            content = response.choices[0].message.content
            return self._parse_injection_response(content, phase.id)

        except Exception as e:
            self._logger.warning(f"LLM injection analysis failed: {e}")
            return None

    def _parse_injection_response(
        self,
        content: str,
        phase_id: str,
    ) -> InjectionDecision | None:
        """Parse LLM injection response"""
        import re

        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

        tasks = []
        for i, task_data in enumerate(data.get("tasks", [])):
            risk_str = task_data.get("risk_level", "MEDIUM").upper()
            risk_level = RiskLevel.MEDIUM
            for level in RiskLevel:
                if level.name == risk_str:
                    risk_level = level
                    break

            task = IntelligentTask(
                id=f"injected_{phase_id}_{int(time.time() * 1000)}_{i}",
                name=task_data.get("name", f"Injected Task {i + 1}"),
                description=task_data.get("description", ""),
                category=task_data.get("category", "setup"),
                phase_id=phase_id,
                priority=task_data.get("priority", 2),
                semantic_intent=task_data.get("semantic_intent", ""),
                overall_risk_level=risk_level,
            )
            tasks.append(task)

        return InjectionDecision(
            should_inject=data.get("should_inject", False),
            tasks_to_inject=tasks,
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5),
            injection_type=data.get("injection_type", "adaptive"),
        )

    async def _rule_based_analysis(
        self,
        context: InjectionContext,
    ) -> InjectionDecision:
        """Rule-based injection analysis"""

        tasks_to_inject = []
        reasoning_parts = []

        if context.failed_tasks:
            recovery_tasks = self._create_recovery_tasks(
                context.failed_tasks,
                context.current_phase.id,
            )
            tasks_to_inject.extend(recovery_tasks)
            reasoning_parts.append(f"{len(recovery_tasks)} recovery tasks for failures")

        signals = context.execution_signals
        if signals.get("security_concerns"):
            security_task = self._create_security_task(
                context.current_phase.id,
                signals["security_concerns"],
            )
            tasks_to_inject.append(security_task)
            reasoning_parts.append("security audit task")

        if signals.get("missing_tests"):
            test_task = self._create_test_task(
                context.current_phase.id,
                signals["missing_tests"],
            )
            tasks_to_inject.append(test_task)
            reasoning_parts.append("test task")

        if signals.get("integration_gaps"):
            integration_tasks = self._create_integration_tasks(
                context.current_phase.id,
                signals["integration_gaps"],
            )
            tasks_to_inject.extend(integration_tasks)
            reasoning_parts.append(f"{len(integration_tasks)} integration tasks")

        should_inject = len(tasks_to_inject) > 0

        return InjectionDecision(
            should_inject=should_inject,
            tasks_to_inject=tasks_to_inject,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No injection needed",
            confidence=0.7 if should_inject else 0.9,
            injection_type="rule_based",
        )

    def _create_recovery_tasks(
        self,
        failed_tasks: list[Any],
        phase_id: str,
    ) -> list[IntelligentTask]:
        """Create recovery tasks for failures"""

        tasks = []
        for i, failed in enumerate(failed_tasks[:3]):
            task_name = getattr(failed, "name", str(failed))[:50]
            task = IntelligentTask(
                id=f"recovery_{phase_id}_{int(time.time() * 1000)}_{i}",
                name=f"Recover: {task_name}",
                description=f"Recovery task for failed execution: {task_name}",
                category="setup",
                phase_id=phase_id,
                priority=1,
                semantic_intent="Recover from failed task execution",
                overall_risk_level=RiskLevel.MEDIUM,
            )
            tasks.append(task)

        return tasks

    def _create_security_task(
        self,
        phase_id: str,
        concerns: Any,
    ) -> IntelligentTask:
        """Create security audit task"""

        return IntelligentTask(
            id=f"security_{phase_id}_{int(time.time() * 1000)}",
            name="Security Audit",
            description=f"Perform security audit. Concerns: {str(concerns)[:200]}",
            category="security",
            phase_id=phase_id,
            priority=1,
            semantic_intent="Identify and mitigate security vulnerabilities",
            overall_risk_level=RiskLevel.HIGH,
        )

    def _create_test_task(
        self,
        phase_id: str,
        missing_tests: Any,
    ) -> IntelligentTask:
        """Create test task"""

        return IntelligentTask(
            id=f"test_{phase_id}_{int(time.time() * 1000)}",
            name="Add Missing Tests",
            description=f"Write tests for: {str(missing_tests)[:200]}",
            category="testing",
            phase_id=phase_id,
            priority=2,
            semantic_intent="Ensure code coverage for untested components",
            overall_risk_level=RiskLevel.LOW,
        )

    def _create_integration_tasks(
        self,
        phase_id: str,
        gaps: Any,
    ) -> list[IntelligentTask]:
        """Create integration tasks"""

        return [
            IntelligentTask(
                id=f"integration_{phase_id}_{int(time.time() * 1000)}",
                name="Integration Test",
                description=f"Add integration test for: {str(gaps)[:200]}",
                category="testing",
                phase_id=phase_id,
                priority=2,
                semantic_intent="Verify component integration",
                overall_risk_level=RiskLevel.MEDIUM,
            )
        ]

    def _assess_injection_risk(
        self,
        tasks: list[IntelligentTask],
    ) -> RiskLevel:
        """Assess overall risk of injection"""

        if not tasks:
            return RiskLevel.LOW

        max_risk = RiskLevel.LOW
        for task in tasks:
            if task.overall_risk_level.value > max_risk.value:
                max_risk = task.overall_risk_level

        return max_risk

    def _get_risk_score(self, level: RiskLevel) -> float:
        """Convert risk level to score"""

        return {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0,
        }.get(level, 0.5)

    def _summarize_completed_phases(self, phases: list[Phase]) -> str:
        if not phases:
            return "None"

        return "\n".join(
            [f"- {p.name}: {p.semantic_goal} ({p.actual_tasks} tasks)" for p in phases[-5:]]
        )

    def _summarize_pending_tasks(self, tasks: list[IntelligentTask]) -> str:
        if not tasks:
            return "None"

        return "\n".join([f"- {t.name}: {t.semantic_intent[:50]}" for t in tasks[:10]])

    def _format_signals(self, signals: dict[str, Any]) -> str:
        if not signals:
            return "No signals"

        parts = []
        for key, value in list(signals.items())[:5]:
            parts.append(f"- {key}: {str(value)[:100]}")

        return "\n".join(parts)

    def record_approval(self, decision: InjectionDecision, approved: bool) -> None:
        """Record user approval/rejection for learning"""

        if approved:
            self._approved_injections += 1
        else:
            self._rejected_injections += 1

        if self._config.learning_enabled and decision.tasks_to_inject:
            pattern = InjectionPattern(
                trigger_conditions=[decision.injection_type],
                task_template={
                    "category": decision.tasks_to_inject[0].category,
                    "injection_type": decision.injection_type,
                },
                success_rate=1.0 if approved else 0.0,
                times_applied=1,
                last_applied=time.time(),
            )
            self._patterns.append(pattern)

    def get_stats(self) -> dict[str, Any]:
        """Get injection statistics"""

        return {
            "total_injections": self._injections,
            "approved_injections": self._approved_injections,
            "rejected_injections": self._rejected_injections,
            "autonomous_injections": self._autonomous_injections,
            "learned_patterns": len(self._patterns),
        }


def create_task_injector(
    provider: Any = None,
    config: Layer2Config | None = None,
) -> DynamicTaskInjector:
    """Factory function to create DynamicTaskInjector"""

    return DynamicTaskInjector(provider=provider, config=config)
