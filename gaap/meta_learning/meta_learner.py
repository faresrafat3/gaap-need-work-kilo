"""
Meta-Learner - Recursive Self-Improvement Coordinator
=====================================================

Coordinates all meta-learning components during dream cycles:
- Wisdom Distillation: Extract principles from successes
- Failure Analysis: Learn from mistakes
- Axiom Proposals: Suggest new constitutional rules
- Confidence Tracking: Monitor decision quality

Usage:
    learner = MetaLearner(
        episodic_store=episodic,
        axiom_validator=validator
    )

    # Run dream cycle
    result = await learner.run_dream_cycle()

    # Get heuristics for a task
    wisdom = learner.get_wisdom_for_task("code generation")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from gaap.core.axioms import AxiomValidator
from gaap.memory.hierarchical import (
    EpisodicMemory,
    EpisodicMemoryStore,
    HierarchicalMemory,
)
from gaap.meta_learning.axiom_bridge import AxiomBridge, AxiomProposal, ProposalStatus
from gaap.meta_learning.confidence import (
    ActionRecommendation,
    ConfidenceCalculator,
    ConfidenceResult,
)
from gaap.meta_learning.failure_store import (
    CorrectiveAction,
    FailedTrace,
    FailureStore,
    FailureType,
)
from gaap.meta_learning.wisdom_distiller import (
    DistillationResult,
    ProjectHeuristic,
    WisdomDistiller,
)

logger = logging.getLogger("gaap.meta_learning.learner")


@dataclass
class DreamCycleResult:
    """نتيجة دورة التعلم الليلية"""

    started_at: datetime
    completed_at: datetime | None = None

    episodes_analyzed: int = 0
    heuristics_distilled: int = 0
    failures_analyzed: int = 0
    axioms_proposed: int = 0
    axioms_committed: int = 0

    new_heuristics: list[ProjectHeuristic] = field(default_factory=list)
    new_failures: list[FailedTrace] = field(default_factory=list)
    new_proposals: list[AxiomProposal] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "episodes_analyzed": self.episodes_analyzed,
            "heuristics_distilled": self.heuristics_distilled,
            "failures_analyzed": self.failures_analyzed,
            "axioms_proposed": self.axioms_proposed,
            "axioms_committed": self.axioms_committed,
            "new_heuristics": [h.to_dict() for h in self.new_heuristics],
            "new_failures": [f.to_dict() for f in self.new_failures],
            "new_proposals": [p.to_dict() for p in self.new_proposals],
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
        }

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.completed_at is not None


@dataclass
class WisdomContext:
    """سياق الحكمة لمهمة معينة"""

    task_description: str
    relevant_heuristics: list[ProjectHeuristic]
    pitfall_warnings: list[str]
    confidence: ConfidenceResult
    similar_failures: list[tuple[FailedTrace, list[CorrectiveAction]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_description": self.task_description,
            "relevant_heuristics": [h.to_dict() for h in self.relevant_heuristics],
            "pitfall_warnings": self.pitfall_warnings,
            "confidence": self.confidence.to_dict(),
            "similar_failures": [
                {"trace": t.to_dict(), "corrections": [c.to_dict() for c in cs]}
                for t, cs in self.similar_failures
            ],
        }


class MetaLearner:
    """
    Coordinates all meta-learning activities.

    Features:
    - Dream cycle orchestration
    - Cross-component integration
    - Wisdom retrieval for tasks
    - Failure-aware decision making
    - Automatic axiom promotion
    """

    DEFAULT_STORAGE_PATH = ".gaap/memory/meta_learning"

    def __init__(
        self,
        storage_path: str | None = None,
        episodic_store: EpisodicMemoryStore | None = None,
        hierarchical_memory: HierarchicalMemory | None = None,
        axiom_validator: AxiomValidator | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._episodic = episodic_store
        self._hierarchical = hierarchical_memory
        self._validator = axiom_validator
        self._llm = llm_client

        self.distiller = WisdomDistiller(
            storage_path=str(self.storage_path / "wisdom"),
            episodic_store=episodic_store,
            llm_client=llm_client,
        )

        self.failures = FailureStore(
            storage_path=str(self.storage_path / "failures"),
            episodic_store=episodic_store,
        )

        self.axiom_bridge = AxiomBridge(
            storage_path=str(self.storage_path / "proposals"),
            axiom_validator=axiom_validator,
        )

        self.confidence_calc = ConfidenceCalculator()

        self._logger = logger

        self._last_dream_cycle: datetime | None = None
        self._dream_cycle_history: list[DreamCycleResult] = []

    async def run_dream_cycle(
        self,
        days: int = 1,
        categories: list[str] | None = None,
    ) -> DreamCycleResult:
        """
        Run a complete dream cycle.

        Args:
            days: Number of days to analyze
            categories: Optional categories to focus on

        Returns:
            DreamCycleResult with all activities
        """
        import time

        start_time = time.time()
        result = DreamCycleResult(started_at=datetime.now())

        try:
            episodes = self._get_recent_episodes(days)
            result.episodes_analyzed = len(episodes)

            await self._run_wisdom_distillation(episodes, categories, result)

            await self._run_failure_analysis(episodes, result)

            await self._run_axiom_promotion(result)

            await self._run_heuristic_validation(episodes, result)

        except Exception as e:
            result.errors.append(f"Dream cycle error: {str(e)}")
            self._logger.error(f"Dream cycle failed: {e}", exc_info=True)

        result.completed_at = datetime.now()
        result.processing_time_ms = (time.time() - start_time) * 1000

        self._last_dream_cycle = result.completed_at
        self._dream_cycle_history.append(result)

        self._logger.info(
            f"Dream cycle complete: {result.heuristics_distilled} heuristics, "
            f"{result.axioms_proposed} proposals, {result.processing_time_ms:.0f}ms"
        )

        return result

    def get_wisdom_for_task(
        self,
        task_description: str,
        task_type: str | None = None,
    ) -> WisdomContext:
        """
        Get accumulated wisdom for a task.

        Args:
            task_description: Description of the task
            task_type: Optional task type filter

        Returns:
            WisdomContext with heuristics, warnings, and confidence
        """
        heuristics = self.distiller.get_heuristics_for_context(
            task_description,
            min_confidence=0.5,
            limit=5,
        )

        warnings = self.failures.get_pitfall_warnings(task_description)

        similar_failures = self.failures.find_similar(
            task_description,
            task_type=task_type,
            limit=3,
        )

        successful_episodes = (
            [e for e in self._episodic._episodes if e.success] if self._episodic else []
        )
        similar_successes = [
            e for e in successful_episodes if self._is_similar_task(task_description, e.action)
        ]

        confidence = self.confidence_calc.calculate_for_task(
            task_description=task_description,
            similar_successes=similar_successes,
            similar_failures=[t for t, _ in similar_failures],
        )

        return WisdomContext(
            task_description=task_description,
            relevant_heuristics=heuristics,
            pitfall_warnings=warnings,
            confidence=confidence,
            similar_failures=similar_failures,
        )

    def record_failure(
        self,
        task_type: str,
        hypothesis: str,
        error: str,
        context: dict[str, Any] | None = None,
        corrective_action: str | None = None,
        agent_thoughts: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """
        Record a failure for learning.

        Args:
            task_type: Type of task that failed
            hypothesis: What the agent thought would work
            error: The error that occurred
            context: Additional context
            corrective_action: What fixed it (if known)
            agent_thoughts: Agent's reasoning before failure
            task_id: Associated task ID

        Returns:
            Failure ID
        """
        error_type = self.failures.classify_error(error)

        trace = FailedTrace(
            task_type=task_type,
            hypothesis=hypothesis,
            error=error,
            error_type=error_type,
            context=context or {},
            agent_thoughts=agent_thoughts,
            task_id=task_id,
        )

        action = corrective_action or None
        failure_id = self.failures.record(trace, action)

        self._logger.info(f"Recorded failure: {failure_id} ({error_type.name}) - {error[:50]}")

        return failure_id

    def record_success(
        self,
        task_type: str,
        action: str,
        result: str,
        task_id: str | None = None,
        lessons: list[str] | None = None,
    ) -> None:
        """
        Record a successful episode.

        Args:
            task_type: Type of task
            action: What the agent did
            result: The result
            task_id: Associated task ID
            lessons: Lessons learned
        """
        if not self._episodic:
            return

        episode = EpisodicMemory(
            task_id=task_id or "",
            action=action,
            result=result,
            success=True,
            category=task_type,
            lessons=lessons or [],
        )

        self._episodic.record(episode)

    def should_research(self, task_description: str) -> bool:
        """
        Check if a task requires research before execution.

        Args:
            task_description: Description of the task

        Returns:
            True if research is needed
        """
        wisdom = self.get_wisdom_for_task(task_description)
        return wisdom.confidence.recommendation == ActionRecommendation.RESEARCH_REQUIRED

    def get_action_recommendation(
        self,
        task_description: str,
    ) -> ActionRecommendation:
        """
        Get recommended action for a task.

        Args:
            task_description: Description of the task

        Returns:
            ActionRecommendation
        """
        wisdom = self.get_wisdom_for_task(task_description)
        return wisdom.confidence.recommendation

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive meta-learner statistics."""
        return {
            "distiller": self.distiller.get_stats(),
            "failures": self.failures.get_stats(),
            "axiom_bridge": self.axiom_bridge.get_stats(),
            "confidence": self.confidence_calc.get_stats(),
            "last_dream_cycle": self._last_dream_cycle.isoformat()
            if self._last_dream_cycle
            else None,
            "dream_cycles_run": len(self._dream_cycle_history),
        }

    def get_ready_axioms(self) -> list[ProjectHeuristic]:
        """Get heuristics ready to become axioms."""
        return self.distiller.get_heuristics_ready_for_axiom()

    async def _run_wisdom_distillation(
        self,
        episodes: list[EpisodicMemory],
        categories: list[str] | None,
        result: DreamCycleResult,
    ) -> None:
        """Run wisdom distillation on episodes."""
        successful = [e for e in episodes if e.success]
        if not successful:
            return

        if categories:
            for category in categories:
                cat_episodes = [e for e in successful if e.category == category]
                if len(cat_episodes) >= 3:
                    distill_result = await self.distiller.distill(cat_episodes)
                    if distill_result.heuristic:
                        result.new_heuristics.append(distill_result.heuristic)
                        result.heuristics_distilled += 1
        else:
            distill_result = await self.distiller.distill(successful)
            if distill_result.heuristic:
                result.new_heuristics.append(distill_result.heuristic)
                result.heuristics_distilled += 1

    async def _run_failure_analysis(
        self,
        episodes: list[EpisodicMemory],
        result: DreamCycleResult,
    ) -> None:
        """Analyze failures from episodes."""
        failed = [e for e in episodes if not e.success]
        result.failures_analyzed = len(failed)

        for episode in failed:
            if episode.result and "error" in episode.result.lower():
                trace = FailedTrace(
                    task_type=episode.category,
                    hypothesis=episode.action[:200],
                    error=episode.result,
                    error_type=self.failures.classify_error(episode.result),
                    task_id=episode.task_id,
                )

                failure_id = self.failures.record(trace)

                if self._episodic:
                    stored = self.failures._failures.get(failure_id)
                    if stored:
                        result.new_failures.append(stored)

    async def _run_axiom_promotion(self, result: DreamCycleResult) -> None:
        """Promote ready heuristics to axioms."""
        ready = self.distiller.get_heuristics_ready_for_axiom()

        for heuristic in ready:
            proposal = self.axiom_bridge.propose_from_heuristic(heuristic)

            if proposal.status == ProposalStatus.APPROVED:
                committed = self.axiom_bridge.commit(proposal.get_id())
                if committed:
                    result.axioms_committed += 1

            result.new_proposals.append(proposal)
            result.axioms_proposed += 1

    async def _run_heuristic_validation(
        self,
        episodes: list[EpisodicMemory],
        result: DreamCycleResult,
    ) -> None:
        """Validate existing heuristics against recent episodes."""
        cutoff = datetime.now() - timedelta(days=30)
        recent = [e for e in episodes if e.timestamp >= cutoff]

        for hid, heuristic in list(self.distiller._heuristics.items()):
            if heuristic.status.name == "DEPRECATED":
                continue

            is_valid = self.distiller.validate_heuristic(hid, recent)

            if not is_valid:
                result.warnings.append(f"Heuristic deprecated: {heuristic.principle[:50]}")

    def _get_recent_episodes(self, days: int) -> list[EpisodicMemory]:
        """Get episodes from recent days."""
        if not self._episodic:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in self._episodic._episodes if e.timestamp >= cutoff]

    def _is_similar_task(self, desc1: str, desc2: str) -> bool:
        """Check if two task descriptions are similar."""
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        return overlap >= min(len(words1), len(words2)) // 2


async def run_meta_learning_cycle(
    episodic_store: EpisodicMemoryStore | None = None,
    axiom_validator: AxiomValidator | None = None,
) -> DreamCycleResult:
    """Run a meta-learning dream cycle."""
    learner = MetaLearner(
        episodic_store=episodic_store,
        axiom_validator=axiom_validator,
    )
    return await learner.run_dream_cycle()


def create_meta_learner(
    storage_path: str | None = None,
    episodic_store: EpisodicMemoryStore | None = None,
    axiom_validator: AxiomValidator | None = None,
    llm_client: Any | None = None,
) -> MetaLearner:
    """Create a MetaLearner instance."""
    return MetaLearner(
        storage_path=storage_path,
        episodic_store=episodic_store,
        axiom_validator=axiom_validator,
        llm_client=llm_client,
    )
