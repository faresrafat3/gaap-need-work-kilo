"""
Confidence Scorer for Layer0 Interface
======================================

Pre-execution confidence assessment that integrates:
- KnowledgeMap (novelty detection)
- EpisodicMemory (similar past tasks)
- ConfidenceCalculator (scoring)

Determines if the agent should:
- Execute directly (high confidence)
- Proceed with caution (medium confidence)
- Research first (low confidence)

Usage:
    scorer = ConfidenceScorer(episodic_store, knowledge_map)

    result = await scorer.assess(intent)
    if result.needs_research():
        trigger_research()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from gaap.core.knowledge_map import KnowledgeMap, KnowledgeLevel
from gaap.meta_learning.confidence import (
    ConfidenceCalculator,
    ConfidenceResult,
    ConfidenceLevel,
    ActionRecommendation,
)
from gaap.memory.hierarchical import EpisodicMemoryStore

logger = logging.getLogger("gaap.core.confidence_scorer")


@dataclass
class AssessmentResult:
    """نتيجة تقييم الثقة"""

    confidence: ConfidenceResult
    knowledge_gaps: list[str] = field(default_factory=list)
    novelty_score: float = 0.0
    similar_tasks_found: int = 0
    research_topics: list[str] = field(default_factory=list)
    caution_mode: bool = False
    epistemic_humility: float = 0.5

    def needs_research(self) -> bool:
        return self.confidence.recommendation == ActionRecommendation.RESEARCH_REQUIRED

    def needs_caution(self) -> bool:
        return self.confidence.recommendation == ActionRecommendation.PROCEED_WITH_CAUTION

    def is_reliable(self) -> bool:
        return self.confidence.is_reliable()

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence.to_dict(),
            "knowledge_gaps": self.knowledge_gaps,
            "novelty_score": self.novelty_score,
            "similar_tasks_found": self.similar_tasks_found,
            "research_topics": self.research_topics,
            "caution_mode": self.caution_mode,
            "epistemic_humility": self.epistemic_humility,
        }


class ConfidenceScorer:
    """
    Pre-execution confidence assessment.

    Integrates multiple signals to determine if the agent
    should proceed, research, or use caution.

    Signals:
    - Similarity to past successful tasks (EpisodicMemory)
    - Novelty of concepts (KnowledgeMap)
    - Knowledge gaps (KnowledgeMap)
    - Historical success rate (EpisodicMemory)

    Formula:
        FinalConfidence = (S * 0.5) + ((1-N) * 0.3) + (E * 0.2)

        Where:
        - S = Similarity score (0-1)
        - N = Novelty score (0-1)
        - E = Evidence count factor (0-1)
    """

    RESEARCH_THRESHOLD = 0.4
    CAUTION_THRESHOLD = 0.7

    def __init__(
        self,
        episodic_store: EpisodicMemoryStore | None = None,
        knowledge_map: KnowledgeMap | None = None,
        calculator: ConfidenceCalculator | None = None,
    ) -> None:
        self.episodic = episodic_store
        self.knowledge_map = knowledge_map or KnowledgeMap()
        self.calculator = calculator or ConfidenceCalculator()

        self._logger = logger

        self._assessments: list[AssessmentResult] = []

    async def assess(
        self,
        task_description: str,
        task_type: str | None = None,
        experts_opinions: list[float] | None = None,
    ) -> AssessmentResult:
        """
        Assess confidence for a task.

        Args:
            task_description: Description of the task
            task_type: Optional task type filter
            experts_opinions: Optional confidence scores from experts

        Returns:
            AssessmentResult with confidence and recommendations
        """
        novelty = self.knowledge_map.assess_novelty(task_description)

        knowledge_gaps = self.knowledge_map.get_knowledge_gaps(task_description)

        similar_successes, similar_failures = self._find_similar_tasks(task_description, task_type)

        similarity = self._calculate_similarity_score(similar_successes, similar_failures)

        evidence_count = len(similar_successes) + len(similar_failures)

        historical_success = self._calculate_historical_success(similar_successes, similar_failures)

        confidence = self.calculator.calculate(
            similarity=similarity,
            novelty=novelty,
            consensus_variance=self._calculate_variance(experts_opinions),
            evidence_count=evidence_count,
            historical_success=historical_success,
            context={"task": task_description[:200]},
        )

        research_topics = [g.entity_name for g in knowledge_gaps[:5]]

        caution_mode = confidence.recommendation == ActionRecommendation.PROCEED_WITH_CAUTION

        epistemic_humility = self.calculator.get_epistemic_humility_score(confidence.score)

        result = AssessmentResult(
            confidence=confidence,
            knowledge_gaps=[g.entity_name for g in knowledge_gaps],
            novelty_score=novelty,
            similar_tasks_found=evidence_count,
            research_topics=research_topics,
            caution_mode=caution_mode,
            epistemic_humility=epistemic_humility,
        )

        self._assessments.append(result)

        self._logger.info(
            f"Confidence assessment: {confidence.score:.0%} "
            f"(novelty={novelty:.0%}, similarity={similarity:.0%}, "
            f"gaps={len(knowledge_gaps)}, recommendation={confidence.recommendation.name})"
        )

        return result

    def assess_sync(
        self,
        task_description: str,
        task_type: str | None = None,
    ) -> AssessmentResult:
        """Synchronous wrapper for assess."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.assess(task_description, task_type),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.assess(task_description, task_type))

    def get_epistemic_humility(self, confidence: float) -> float:
        """Calculate epistemic humility score."""
        return self.calculator.get_epistemic_humility_score(confidence)

    def get_stats(self) -> dict[str, Any]:
        """Get scorer statistics."""
        if not self._assessments:
            return {
                "total_assessments": 0,
                "research_triggered": 0,
                "caution_triggered": 0,
                "avg_confidence": 0.0,
            }

        research_count = sum(1 for a in self._assessments if a.needs_research())
        caution_count = sum(1 for a in self._assessments if a.needs_caution())

        return {
            "total_assessments": len(self._assessments),
            "research_triggered": research_count,
            "caution_triggered": caution_count,
            "avg_confidence": sum(a.confidence.score for a in self._assessments)
            / len(self._assessments),
            "knowledge_map": self.knowledge_map.get_stats(),
        }

    def _find_similar_tasks(
        self,
        task_description: str,
        task_type: str | None = None,
    ) -> tuple[list[Any], list[Any]]:
        successes: list[Any] = []
        failures: list[Any] = []

        if not self.episodic:
            return successes, failures

        keywords = set(self._extract_keywords(task_description))

        for episode in self.episodic._episodes:
            if task_type and episode.category != task_type:
                continue

            episode_keywords = set(self._extract_keywords(episode.action))
            overlap = len(keywords & episode_keywords)

            if overlap >= 2:
                if episode.success:
                    successes.append(episode)
                else:
                    failures.append(episode)

        return successes[:20], failures[:10]

    def _calculate_similarity_score(
        self,
        successes: list[Any],
        failures: list[Any],
    ) -> float:
        """Calculate similarity score from past tasks."""
        total = len(successes) + len(failures)
        if total == 0:
            return 0.5

        success_rate = len(successes) / total

        count_factor = min(total / 10, 1.0)

        return success_rate * 0.6 + count_factor * 0.4

    def _calculate_historical_success(
        self,
        successes: list[Any],
        failures: list[Any],
    ) -> float:
        """Calculate historical success rate."""
        total = len(successes) + len(failures)
        if total == 0:
            return 0.5
        return len(successes) / total

    def _calculate_variance(self, values: list[float] | None) -> float:
        """Calculate variance of expert opinions."""
        if not values or len(values) < 2:
            return 0.5

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)

        return min(variance * 4, 1.0)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        import re

        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        stopwords = {
            "the",
            "and",
            "for",
            "was",
            "were",
            "been",
            "being",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "with",
            "from",
            "when",
            "what",
            "where",
        }
        return [w for w in words if w not in stopwords]


def create_confidence_scorer(
    episodic_store: EpisodicMemoryStore | None = None,
    knowledge_map: KnowledgeMap | None = None,
) -> ConfidenceScorer:
    """Create a ConfidenceScorer instance."""
    return ConfidenceScorer(
        episodic_store=episodic_store,
        knowledge_map=knowledge_map,
    )
