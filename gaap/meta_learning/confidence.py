"""
Confidence Calculator - Epistemic Humility
=========================================

Calculates confidence scores for decisions, predictions, and heuristics.
Implements "knowing what it doesn't know" through multiple factors.

Usage:
    calc = ConfidenceCalculator()

    score = calc.calculate(
        similarity=0.8,
        novelty=0.2,
        consensus_variance=0.1
    )
    print(f"Confidence: {score:.2%}")

    if score < 0.4:
        print("Need more research before proceeding")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger("gaap.meta_learning.confidence")


class ConfidenceLevel(Enum):
    """مستويات الثقة"""

    VERY_LOW = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    VERY_HIGH = auto()


class ActionRecommendation(Enum):
    """توصيات بناءً على مستوى الثقة"""

    RESEARCH_REQUIRED = auto()
    PROCEED_WITH_CAUTION = auto()
    NORMAL_EXECUTION = auto()
    DIRECT_EXECUTION = auto()


@dataclass
class ConfidenceFactors:
    """
    Factors that influence confidence calculation.

    All values should be in range [0.0, 1.0].
    """

    similarity: float = 0.5
    novelty: float = 0.5
    consensus_variance: float = 0.5
    evidence_count: int = 0
    recency: float = 0.5
    source_reliability: float = 0.5
    cross_validation: float = 0.0
    historical_success: float = 0.5

    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity": self.similarity,
            "novelty": self.novelty,
            "consensus_variance": self.consensus_variance,
            "evidence_count": self.evidence_count,
            "recency": self.recency,
            "source_reliability": self.source_reliability,
            "cross_validation": self.cross_validation,
            "historical_success": self.historical_success,
            "context": self.context,
        }


@dataclass
class ConfidenceResult:
    """نتيجة حساب الثقة"""

    score: float
    level: ConfidenceLevel
    recommendation: ActionRecommendation
    factors: ConfidenceFactors
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level.name,
            "recommendation": self.recommendation.name,
            "factors": self.factors.to_dict(),
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
        }

    def needs_research(self) -> bool:
        return self.score < 0.4

    def needs_caution(self) -> bool:
        return 0.4 <= self.score < 0.7

    def is_reliable(self) -> bool:
        return self.score >= 0.7


class ConfidenceCalculator:
    """
    Calculates confidence scores for decisions.

    Uses a weighted formula with multiple factors:
    - Similarity: How similar is this to past successes?
    - Novelty: How new/unusual is this situation?
    - Consensus Variance: How much do experts disagree?
    - Evidence Count: How much data supports this?
    - Recency: How recent is the supporting evidence?
    - Source Reliability: How reliable are the sources?
    - Cross Validation: Has this been validated across domains?
    - Historical Success: Past success rate with similar decisions

    Formula:
        FinalConfidence = (S * w1) + ((1-N) * w2) + ((1-V) * w3) +
                         (E_factor * w4) + (R * w5) + (SR * w6) +
                         (CV * w7) + (HS * w8)

    Where w1-w8 are configurable weights.
    """

    DEFAULT_WEIGHTS = {
        "similarity": 0.25,
        "novelty": 0.15,
        "consensus_variance": 0.10,
        "evidence_count": 0.15,
        "recency": 0.10,
        "source_reliability": 0.10,
        "cross_validation": 0.05,
        "historical_success": 0.10,
    }

    RESEARCH_THRESHOLD = 0.4
    CAUTION_THRESHOLD = 0.7
    HIGH_CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._normalize_weights()
        self._logger = logger

        self._calculations: list[ConfidenceResult] = []

    def calculate(
        self,
        similarity: float = 0.5,
        novelty: float = 0.5,
        consensus_variance: float = 0.5,
        evidence_count: int = 0,
        recency: float = 0.5,
        source_reliability: float = 0.5,
        cross_validation: float = 0.0,
        historical_success: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> ConfidenceResult:
        """
        Calculate confidence score from factors.

        Args:
            similarity: Similarity to past successes (0-1)
            novelty: Task novelty/unusualness (0-1)
            consensus_variance: Disagreement level (0-1, lower = more agreement)
            evidence_count: Number of supporting cases
            recency: How recent is evidence (0-1)
            source_reliability: Reliability of sources (0-1)
            cross_validation: Cross-domain validation (0-1)
            historical_success: Past success rate (0-1)
            context: Additional context

        Returns:
            ConfidenceResult with score, level, and recommendation
        """
        factors = ConfidenceFactors(
            similarity=similarity,
            novelty=novelty,
            consensus_variance=consensus_variance,
            evidence_count=evidence_count,
            recency=recency,
            source_reliability=source_reliability,
            cross_validation=cross_validation,
            historical_success=historical_success,
            context=context or {},
        )

        evidence_factor = self._normalize_evidence_count(evidence_count)

        score = (
            similarity * self.weights["similarity"]
            + (1 - novelty) * self.weights["novelty"]
            + (1 - consensus_variance) * self.weights["consensus_variance"]
            + evidence_factor * self.weights["evidence_count"]
            + recency * self.weights["recency"]
            + source_reliability * self.weights["source_reliability"]
            + cross_validation * self.weights["cross_validation"]
            + historical_success * self.weights["historical_success"]
        )

        score = max(0.0, min(1.0, score))

        level = self._determine_level(score)
        recommendation = self._determine_recommendation(score)
        explanation = self._generate_explanation(factors, score)

        result = ConfidenceResult(
            score=score,
            level=level,
            recommendation=recommendation,
            factors=factors,
            explanation=explanation,
        )

        self._calculations.append(result)
        return result

    def calculate_from_factors(self, factors: ConfidenceFactors) -> ConfidenceResult:
        """Calculate confidence from ConfidenceFactors object."""
        return self.calculate(
            similarity=factors.similarity,
            novelty=factors.novelty,
            consensus_variance=factors.consensus_variance,
            evidence_count=factors.evidence_count,
            recency=factors.recency,
            source_reliability=factors.source_reliability,
            cross_validation=factors.cross_validation,
            historical_success=factors.historical_success,
            context=factors.context,
        )

    def calculate_for_task(
        self,
        task_description: str,
        similar_successes: list[Any],
        similar_failures: list[Any],
        expert_opinions: list[float] | None = None,
    ) -> ConfidenceResult:
        """
        Calculate confidence for a specific task.

        Args:
            task_description: Description of the task
            similar_successes: List of similar successful episodes
            similar_failures: List of similar failed episodes
            expert_opinions: List of confidence scores from experts

        Returns:
            ConfidenceResult for the task
        """
        similarity = len(similar_successes) / max(
            len(similar_successes) + len(similar_failures) + 1, 1
        )

        total_similar = len(similar_successes) + len(similar_failures)
        novelty = 1.0 - min(total_similar / 10, 1.0)

        if expert_opinions and len(expert_opinions) > 1:
            mean = sum(expert_opinions) / len(expert_opinions)
            variance = sum((x - mean) ** 2 for x in expert_opinions) / len(expert_opinions)
            consensus_variance = min(variance * 4, 1.0)
        else:
            consensus_variance = 0.5

        evidence_count = total_similar

        historical_success = (
            len(similar_successes) / max(total_similar, 1) if total_similar > 0 else 0.5
        )

        return self.calculate(
            similarity=similarity,
            novelty=novelty,
            consensus_variance=consensus_variance,
            evidence_count=evidence_count,
            historical_success=historical_success,
            context={"task": task_description[:200]},
        )

    def get_epistemic_humility_score(self, confidence: float) -> float:
        """
        Calculate epistemic humility score.

        Higher humility means the agent acknowledges uncertainty appropriately.

        Args:
            confidence: Current confidence score

        Returns:
            Humility score (0-1, higher = more humble/appropriate)
        """
        if confidence < 0.3:
            return 1.0
        elif confidence > 0.95:
            return 0.5
        else:
            return 1.0 - abs(confidence - 0.7) * 1.5

    def get_stats(self) -> dict[str, Any]:
        """Get calculator statistics."""
        if not self._calculations:
            return {
                "total_calculations": 0,
                "avg_confidence": 0.0,
                "by_level": {},
            }

        by_level: dict[str, int] = {}
        for calc in self._calculations:
            by_level[calc.level.name] = by_level.get(calc.level.name, 0) + 1

        return {
            "total_calculations": len(self._calculations),
            "avg_confidence": sum(c.score for c in self._calculations) / len(self._calculations),
            "by_level": by_level,
            "research_needed": sum(1 for c in self._calculations if c.needs_research()),
            "caution_needed": sum(1 for c in self._calculations if c.needs_caution()),
        }

    def _normalize_weights(self) -> None:
        """Ensure weights sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def _normalize_evidence_count(self, count: int) -> float:
        """Normalize evidence count to [0, 1] range."""
        return min(count / 20, 1.0)

    def _determine_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        if score < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.4:
            return ConfidenceLevel.LOW
        elif score < 0.6:
            return ConfidenceLevel.MODERATE
        elif score < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def _determine_recommendation(self, score: float) -> ActionRecommendation:
        """Determine action recommendation from score."""
        if score < self.RESEARCH_THRESHOLD:
            return ActionRecommendation.RESEARCH_REQUIRED
        elif score < self.CAUTION_THRESHOLD:
            return ActionRecommendation.PROCEED_WITH_CAUTION
        elif score < self.HIGH_CONFIDENCE_THRESHOLD:
            return ActionRecommendation.NORMAL_EXECUTION
        else:
            return ActionRecommendation.DIRECT_EXECUTION

    def _generate_explanation(
        self,
        factors: ConfidenceFactors,
        score: float,
    ) -> str:
        """Generate human-readable explanation."""
        parts = []

        if factors.similarity > 0.7:
            parts.append(f"high similarity ({factors.similarity:.0%})")
        elif factors.similarity < 0.3:
            parts.append(f"low similarity ({factors.similarity:.0%})")

        if factors.novelty > 0.7:
            parts.append("novel situation")
        elif factors.novelty < 0.3:
            parts.append("familiar territory")

        if factors.consensus_variance > 0.5:
            parts.append("expert disagreement")
        elif factors.consensus_variance < 0.2:
            parts.append("strong consensus")

        if factors.evidence_count >= 10:
            parts.append(f"strong evidence ({factors.evidence_count} cases)")
        elif factors.evidence_count < 3:
            parts.append("limited evidence")

        if not parts:
            return f"Confidence score: {score:.0%}"

        return f"Confidence {score:.0%} based on: {', '.join(parts)}"


def create_confidence_calculator(
    weights: dict[str, float] | None = None,
) -> ConfidenceCalculator:
    """Create a ConfidenceCalculator instance."""
    return ConfidenceCalculator(weights=weights)
