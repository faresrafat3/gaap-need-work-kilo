"""
Real-Time Reflector - Immediate Learning from Execution
Implements: docs/evolution_plan_2026/21_ENGINE_AUDIT_SPEC.md

Analyzes execution results immediately and extracts lessons for MemoRAG.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger("gaap.reflection")


class ReflectionType(Enum):
    """أنواع الانعكاس"""

    SUCCESS_PATTERN = auto()
    FAILURE_ANALYSIS = auto()
    OPTIMIZATION = auto()
    SECURITY_INSIGHT = auto()
    PERFORMANCE_TIP = auto()


@dataclass
class Reflection:
    """انعكاس مستخلص"""

    type: ReflectionType
    lesson: str
    context: str
    confidence: float
    applicable_to: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionSummary:
    """ملخص تنفيذ"""

    task_id: str
    success: bool
    duration_ms: float
    tokens_used: int
    cost_usd: float
    model: str
    provider: str
    error: str | None = None
    output_preview: str = ""
    quality_score: float = 0.0


class RealTimeReflector:
    """
    Real-Time Reflection Engine

    Analyzes execution results immediately after completion and:
    - Extracts lessons from successes and failures
    - Identifies patterns across executions
    - Stores insights in MemoRAG
    - Updates semantic rules
    """

    SUCCESS_PATTERNS = [
        ("fast_execution", lambda s: s.duration_ms < 1000, "Fast execution achieved with {model}"),
        ("low_cost", lambda s: s.cost_usd < 0.01, "Cost-effective execution at ${cost:.4f}"),
        ("high_quality", lambda s: s.quality_score > 0.9, "High quality output achieved"),
    ]

    FAILURE_PATTERNS = [
        (
            "timeout",
            lambda s: "timeout" in (s.error or "").lower(),
            "Consider increasing timeout for similar tasks",
        ),
        (
            "rate_limit",
            lambda s: "rate limit" in (s.error or "").lower(),
            "Implement backoff for rate-limited providers",
        ),
        (
            "auth_error",
            lambda s: "auth" in (s.error or "").lower(),
            "Check API credentials before execution",
        ),
        (
            "syntax_error",
            lambda s: "syntax" in (s.error or "").lower(),
            "Validate code syntax before execution",
        ),
    ]

    def __init__(self, memorag: Any = None) -> None:
        self._memorag = memorag
        self._reflections: list[Reflection] = []
        self._execution_history: list[ExecutionSummary] = []
        self._patterns_learned: dict[str, int] = {}
        self._logger = logging.getLogger("gaap.reflection")

        if self._memorag is None:
            try:
                from gaap.memory.memorag import get_memorag

                self._memorag = get_memorag()
            except Exception as e:
                self._logger.warning(f"MemoRAG unavailable: {e}")

    def reflect(
        self,
        task_id: str,
        success: bool,
        duration_ms: float,
        tokens_used: int,
        cost_usd: float,
        model: str,
        provider: str,
        error: str | None = None,
        output: str | None = None,
        quality_score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> list[Reflection]:
        """
        Analyze execution and extract lessons

        Args:
            task_id: Task identifier
            success: Whether execution succeeded
            duration_ms: Execution duration
            tokens_used: Tokens consumed
            cost_usd: Cost in USD
            model: Model used
            provider: Provider used
            error: Error message if failed
            output: Output preview
            quality_score: Quality score (0-1)
            metadata: Additional metadata

        Returns:
            List of extracted reflections
        """
        summary = ExecutionSummary(
            task_id=task_id,
            success=success,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            model=model,
            provider=provider,
            error=error,
            output_preview=output[:200] if output else "",
            quality_score=quality_score,
        )

        self._execution_history.append(summary)

        reflections = []

        if success:
            reflections = self._analyze_success(summary)
        else:
            reflections = self._analyze_failure(summary)

        for r in reflections:
            self._reflections.append(r)
            self._store_reflection(r, summary)

        if len(self._execution_history) % 10 == 0:
            self._analyze_patterns()

        return reflections

    def _analyze_success(self, summary: ExecutionSummary) -> list[Reflection]:
        """Analyze successful execution"""
        reflections = []

        for pattern_name, check_fn, template in self.SUCCESS_PATTERNS:
            if check_fn(summary):
                lesson = template.format(
                    model=summary.model,
                    cost=summary.cost_usd,
                    duration=summary.duration_ms,
                )
                reflection = Reflection(
                    type=ReflectionType.SUCCESS_PATTERN,
                    lesson=lesson,
                    context=f"Task {summary.task_id} with {summary.model}",
                    confidence=0.8,
                    applicable_to=[summary.model, summary.provider],
                    metadata={
                        "pattern": pattern_name,
                        "duration_ms": summary.duration_ms,
                        "cost_usd": summary.cost_usd,
                    },
                )
                reflections.append(reflection)
                self._patterns_learned[pattern_name] = (
                    self._patterns_learned.get(pattern_name, 0) + 1
                )

        if summary.quality_score > 0.8 and summary.tokens_used > 0:
            efficiency = summary.quality_score / (summary.tokens_used / 1000)
            if efficiency > 1.0:
                reflection = Reflection(
                    type=ReflectionType.OPTIMIZATION,
                    lesson=f"High efficiency achieved: {efficiency:.2f} quality per 1k tokens",
                    context=f"Model {summary.model} with {summary.tokens_used} tokens",
                    confidence=0.7,
                    applicable_to=[summary.model],
                )
                reflections.append(reflection)

        return reflections

    def _analyze_failure(self, summary: ExecutionSummary) -> list[Reflection]:
        """Analyze failed execution"""
        reflections = []

        for pattern_name, check_fn, lesson in self.FAILURE_PATTERNS:
            if check_fn(summary):
                reflection = Reflection(
                    type=ReflectionType.FAILURE_ANALYSIS,
                    lesson=lesson,
                    context=f"Task {summary.task_id} failed: {summary.error}",
                    confidence=0.9,
                    applicable_to=[summary.provider],
                    metadata={
                        "pattern": pattern_name,
                        "error": summary.error,
                    },
                )
                reflections.append(reflection)
                self._patterns_learned[pattern_name] = (
                    self._patterns_learned.get(pattern_name, 0) + 1
                )

        if summary.error:
            reflection = Reflection(
                type=ReflectionType.FAILURE_ANALYSIS,
                lesson=f"Failure in {summary.provider}/{summary.model}: {summary.error[:100]}",
                context=f"Task {summary.task_id}",
                confidence=0.7,
                applicable_to=[summary.provider, summary.model],
                metadata={"error_type": "general"},
            )
            reflections.append(reflection)

        return reflections

    def _store_reflection(self, reflection: Reflection, summary: ExecutionSummary) -> None:
        """Store reflection in MemoRAG"""
        if not self._memorag:
            return

        try:
            category = reflection.type.name.lower()
            self._memorag.store_lesson(
                lesson=reflection.lesson,
                context=reflection.context,
                category=category,
                success=reflection.type == ReflectionType.SUCCESS_PATTERN,
                task_type=summary.model,
            )
        except Exception as e:
            self._logger.debug(f"Failed to store reflection: {e}")

    def _analyze_patterns(self) -> None:
        """Analyze patterns across execution history"""
        if len(self._execution_history) < 10:
            return

        recent = self._execution_history[-20:]
        success_rate = sum(1 for e in recent if e.success) / len(recent)

        if success_rate < 0.5:
            self._logger.warning(f"Low success rate detected: {success_rate:.0%}")

        model_stats: dict[str, dict[str, Any]] = {}
        for e in recent:
            if e.model not in model_stats:
                model_stats[e.model] = {"count": 0, "success": 0, "duration": 0.0}
            model_stats[e.model]["count"] += 1
            if e.success:
                model_stats[e.model]["success"] += 1
            model_stats[e.model]["duration"] += e.duration_ms

        best_model = None
        best_score = 0
        for model, stats in model_stats.items():
            if stats["count"] >= 2:
                score = stats["success"] / stats["count"]
                if score > best_score:
                    best_score = score
                    best_model = model

        if best_model and best_score > 0.8:
            reflection = Reflection(
                type=ReflectionType.OPTIMIZATION,
                lesson=f"Model {best_model} shows {best_score:.0%} success rate",
                context="Pattern analysis across recent executions",
                confidence=0.85,
                applicable_to=[best_model],
                metadata={
                    "success_rate": best_score,
                    "sample_size": model_stats[best_model]["count"],
                },
            )
            self._reflections.append(reflection)
            self._store_reflection(reflection, self._execution_history[-1])

    def get_recent_lessons(self, limit: int = 10) -> list[str]:
        """Get recent lessons learned"""
        return [r.lesson for r in self._reflections[-limit:]]

    def get_patterns(self) -> dict[str, int]:
        """Get learned patterns"""
        return self._patterns_learned.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get reflection statistics"""
        recent = self._execution_history[-20:] if self._execution_history else []
        return {
            "total_reflections": len(self._reflections),
            "total_executions": len(self._execution_history),
            "patterns_learned": len(self._patterns_learned),
            "recent_success_rate": sum(1 for e in recent if e.success) / max(len(recent), 1),
            "reflection_types": {
                t.name: sum(1 for r in self._reflections if r.type == t) for t in ReflectionType
            },
        }


_reflector_instance: RealTimeReflector | None = None


def get_reflector() -> RealTimeReflector:
    """Get singleton RealTimeReflector instance"""
    global _reflector_instance
    if _reflector_instance is None:
        _reflector_instance = RealTimeReflector()
    return _reflector_instance
