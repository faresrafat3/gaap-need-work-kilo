"""
Self-Evolving Profiles - MorphAgent-Inspired Agent Evolution

Implements agents that can update their own identity based on:
- Performance analytics
- Task outcome patterns
- Capability drift detection
- Confidence-based evolution

Inspired by MorphAgent: https://arxiv.org/abs/2408.11242
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("gaap.swarm.profile_evolver")


class EvolutionTrigger(Enum):
    """Triggers for profile evolution"""

    PERFORMANCE_IMPROVEMENT = auto()
    PERFORMANCE_DECLINE = auto()
    CAPABILITY_EXPANSION = auto()
    CAPABILITY_NARROWING = auto()
    TASK_SUCCESS_PATTERN = auto()
    TASK_FAILURE_PATTERN = auto()
    DOMAIN_SHIFT = auto()
    MANUAL_REQUEST = auto()


class EvolutionStatus(Enum):
    """Status of an evolution"""

    PROPOSED = auto()
    VALIDATING = auto()
    APPLIED = auto()
    REJECTED = auto()
    ROLLED_BACK = auto()


@dataclass
class ProfileEvolution:
    """
    Record of a profile change.

    Tracks the complete history of how a fractal's profile
    has evolved over time.

    Attributes:
        id: Unique identifier
        fractal_id: ID of the fractal being evolved
        old_specialty: Previous specialty
        new_specialty: New specialty
        old_capabilities: Previous capabilities
        new_capabilities: New capabilities
        reason: Human-readable reason for change
        trigger: What triggered this evolution
        confidence: Confidence level in the evolution
        status: Current status
        created_at: When the evolution was proposed
        applied_at: When the evolution was applied
        performance_before: Performance metrics before
        performance_after: Performance metrics after
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fractal_id: str = ""
    old_specialty: str = ""
    new_specialty: str = ""
    old_capabilities: dict[str, float] = field(default_factory=dict)
    new_capabilities: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    trigger: EvolutionTrigger = EvolutionTrigger.MANUAL_REQUEST
    confidence: float = 0.5
    status: EvolutionStatus = EvolutionStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: datetime | None = None
    performance_before: dict[str, float] = field(default_factory=dict)
    performance_after: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fractal_id": self.fractal_id,
            "old_specialty": self.old_specialty,
            "new_specialty": self.new_specialty,
            "old_capabilities": self.old_capabilities,
            "new_capabilities": self.new_capabilities,
            "reason": self.reason,
            "trigger": self.trigger.name,
            "confidence": self.confidence,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "performance_before": self.performance_before,
            "performance_after": self.performance_after,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileEvolution:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            fractal_id=data.get("fractal_id", ""),
            old_specialty=data.get("old_specialty", ""),
            new_specialty=data.get("new_specialty", ""),
            old_capabilities=data.get("old_capabilities", {}),
            new_capabilities=data.get("new_capabilities", {}),
            reason=data.get("reason", ""),
            trigger=EvolutionTrigger[data.get("trigger", "MANUAL_REQUEST")],
            confidence=data.get("confidence", 0.5),
            status=EvolutionStatus[data.get("status", "PROPOSED")],
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
            applied_at=(
                datetime.fromisoformat(data["applied_at"]) if data.get("applied_at") else None
            ),
            performance_before=data.get("performance_before", {}),
            performance_after=data.get("performance_after", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PerformanceSnapshot:
    """
    Snapshot of fractal performance at a point in time.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_quality_score: float = 0.0
    avg_latency_ms: float = 0.0
    domain_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    capability_scores: dict[str, float] = field(default_factory=dict)
    predicted_failures: int = 0
    actual_failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    @property
    def failure_prediction_accuracy(self) -> float:
        total_predictions = self.predicted_failures + self.actual_failures
        if total_predictions == 0:
            return 1.0
        return self.predicted_failures / total_predictions

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.success_rate,
            "avg_quality_score": self.avg_quality_score,
            "avg_latency_ms": self.avg_latency_ms,
            "domain_breakdown": self.domain_breakdown,
            "capability_scores": self.capability_scores,
            "predicted_failures": self.predicted_failures,
            "actual_failures": self.actual_failures,
            "failure_prediction_accuracy": self.failure_prediction_accuracy,
        }


@dataclass
class EvolutionRule:
    """
    Rule for triggering profile evolution.

    Defines conditions under which evolution should be proposed.
    """

    name: str
    trigger: EvolutionTrigger
    condition: Callable[[PerformanceSnapshot, PerformanceSnapshot], bool]
    min_confidence: float = 0.6
    cooldown_hours: int = 24
    priority: int = 1


class ProfileEvolver:
    """
    Updates agent profiles based on performance.

    Inspired by MorphAgent's self-evolving agents:
    1. Analyze performance patterns
    2. Detect capability drift
    3. Propose profile evolutions
    4. Validate and apply changes
    5. Track evolution history

    The evolver enables agents to adapt their identities
    based on what they're actually good at, rather than
    static assignments.

    Usage:
        evolver = ProfileEvolver(reputation_store=reputation_store)

        # Analyze a fractal's performance
        analysis = evolver.analyze_performance("coder_01")

        # Get evolution suggestion
        evolution = evolver.suggest_evolution("coder_01")

        # Apply evolution
        evolver.apply_evolution(evolution)

        # View history
        history = evolver.get_evolution_history("coder_01")
    """

    def __init__(
        self,
        reputation_store: Any = None,
        storage_path: str | None = None,
        min_tasks_for_evolution: int = 10,
        evolution_cooldown_hours: int = 24,
    ) -> None:
        self._reputation_store = reputation_store
        self._storage_path = Path(storage_path) if storage_path else None
        self._min_tasks_for_evolution = min_tasks_for_evolution
        self._evolution_cooldown = timedelta(hours=evolution_cooldown_hours)

        self._evolutions: dict[str, list[ProfileEvolution]] = {}
        self._performance_history: dict[str, list[PerformanceSnapshot]] = {}
        self._current_profiles: dict[str, dict[str, Any]] = {}
        self._last_evolution: dict[str, datetime] = {}

        self._logger = logging.getLogger("gaap.swarm.profile_evolver")

        self._evolution_rules = self._create_default_rules()

        if self._storage_path:
            self._load_from_storage()

    def _create_default_rules(self) -> list[EvolutionRule]:
        """Create default evolution rules"""
        return [
            EvolutionRule(
                name="consistent_domain_success",
                trigger=EvolutionTrigger.TASK_SUCCESS_PATTERN,
                condition=lambda current, previous: (
                    current.success_rate > 0.8
                    and current.total_tasks >= self._min_tasks_for_evolution
                ),
                min_confidence=0.7,
            ),
            EvolutionRule(
                name="consistent_domain_failure",
                trigger=EvolutionTrigger.TASK_FAILURE_PATTERN,
                condition=lambda current, previous: (
                    current.success_rate < 0.3
                    and current.total_tasks >= self._min_tasks_for_evolution
                ),
                min_confidence=0.6,
            ),
            EvolutionRule(
                name="capability_expansion",
                trigger=EvolutionTrigger.CAPABILITY_EXPANSION,
                condition=lambda current, previous: (
                    len(current.capability_scores) > len(previous.capability_scores)
                ),
                min_confidence=0.8,
            ),
            EvolutionRule(
                name="performance_improvement",
                trigger=EvolutionTrigger.PERFORMANCE_IMPROVEMENT,
                condition=lambda current, previous: (
                    current.success_rate > previous.success_rate + 0.2
                    and current.avg_quality_score > previous.avg_quality_score + 0.1
                ),
                min_confidence=0.75,
            ),
            EvolutionRule(
                name="performance_decline",
                trigger=EvolutionTrigger.PERFORMANCE_DECLINE,
                condition=lambda current, previous: (
                    current.success_rate < previous.success_rate - 0.3
                ),
                min_confidence=0.7,
            ),
        ]

    def register_profile(
        self,
        fractal_id: str,
        specialty: str,
        capabilities: dict[str, float],
    ) -> None:
        """Register a fractal's initial profile"""
        self._current_profiles[fractal_id] = {
            "specialty": specialty,
            "capabilities": capabilities,
            "registered_at": datetime.now().isoformat(),
        }

        if fractal_id not in self._performance_history:
            self._performance_history[fractal_id] = []

        self._logger.debug(f"Registered profile for {fractal_id}: {specialty}")

    def record_performance(
        self,
        fractal_id: str,
        snapshot: PerformanceSnapshot,
    ) -> None:
        """Record a performance snapshot"""
        if fractal_id not in self._performance_history:
            self._performance_history[fractal_id] = []

        self._performance_history[fractal_id].append(snapshot)

        self._performance_history[fractal_id] = self._performance_history[fractal_id][-100:]

        if self._storage_path:
            self._save_to_storage()

    def analyze_performance(self, fractal_id: str) -> dict[str, Any]:
        """
        Analyze a fractal's performance patterns.

        Returns comprehensive analysis including:
        - Success rate trends
        - Domain strengths/weaknesses
        - Capability evolution
        - Recommended actions
        """
        history = self._performance_history.get(fractal_id, [])

        if not history:
            return {
                "fractal_id": fractal_id,
                "status": "no_data",
                "message": "No performance data available",
            }

        current = history[-1]
        previous = history[-2] if len(history) > 1 else current

        trend = self._compute_trend(history)

        strengths = self._identify_strengths(current)
        weaknesses = self._identify_weaknesses(current)

        recommendations = self._generate_recommendations(current, previous, strengths, weaknesses)

        return {
            "fractal_id": fractal_id,
            "status": "analyzed",
            "current_performance": current.to_dict(),
            "trend": trend,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "total_snapshots": len(history),
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def _compute_trend(self, history: list[PerformanceSnapshot]) -> str:
        """Compute performance trend"""
        if len(history) < 3:
            return "insufficient_data"

        recent = history[-3:]
        rates = [s.success_rate for s in recent]

        if all(rates[i] < rates[i + 1] for i in range(len(rates) - 1)):
            return "improving"
        elif all(rates[i] > rates[i + 1] for i in range(len(rates) - 1)):
            return "declining"
        else:
            return "stable"

    def _identify_strengths(self, snapshot: PerformanceSnapshot) -> list[str]:
        """Identify domain strengths"""
        strengths = []

        for domain, metrics in snapshot.domain_breakdown.items():
            success_rate = metrics.get("success_rate", 0)
            if success_rate > 0.7:
                strengths.append(domain)

        return strengths

    def _identify_weaknesses(self, snapshot: PerformanceSnapshot) -> list[str]:
        """Identify domain weaknesses"""
        weaknesses = []

        for domain, metrics in snapshot.domain_breakdown.items():
            success_rate = metrics.get("success_rate", 0)
            if success_rate < 0.4:
                weaknesses.append(domain)

        return weaknesses

    def _generate_recommendations(
        self,
        current: PerformanceSnapshot,
        previous: PerformanceSnapshot,
        strengths: list[str],
        weaknesses: list[str],
    ) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if strengths:
            recommendations.append(f"Consider specializing in: {', '.join(strengths)}")

        if weaknesses:
            recommendations.append(f"Consider avoiding tasks in: {', '.join(weaknesses)}")

        if current.failure_prediction_accuracy > 0.8:
            recommendations.append("Good failure prediction - continue using epistemic humility")
        elif current.failure_prediction_accuracy < 0.5:
            recommendations.append("Improve failure prediction - review task estimation")

        return recommendations

    def suggest_evolution(self, fractal_id: str) -> ProfileEvolution | None:
        """
        Suggest a profile evolution based on performance.

        Analyzes recent performance and proposes an evolution
        if warranted by the evolution rules.
        """
        if fractal_id not in self._performance_history:
            return None

        history = self._performance_history[fractal_id]
        if len(history) < 2:
            return None

        if fractal_id in self._last_evolution:
            time_since = datetime.now() - self._last_evolution[fractal_id]
            if time_since < self._evolution_cooldown:
                self._logger.debug(f"Evolution on cooldown for {fractal_id}: {time_since}")
                return None

        current = history[-1]
        previous = history[-2]

        triggered_rule = None
        for rule in sorted(self._evolution_rules, key=lambda r: -r.priority):
            try:
                if rule.condition(current, previous):
                    triggered_rule = rule
                    break
            except Exception as e:
                self._logger.warning(f"Rule {rule.name} evaluation failed: {e}")

        if not triggered_rule:
            return None

        profile = self._current_profiles.get(fractal_id, {})
        old_specialty = profile.get("specialty", "unknown")
        old_capabilities = profile.get("capabilities", {})

        new_specialty, new_capabilities = self._compute_evolution(
            fractal_id, current, previous, triggered_rule
        )

        if new_specialty == old_specialty and new_capabilities == old_capabilities:
            return None

        confidence = self._compute_evolution_confidence(
            current, triggered_rule, old_capabilities, new_capabilities
        )

        evolution = ProfileEvolution(
            fractal_id=fractal_id,
            old_specialty=old_specialty,
            new_specialty=new_specialty,
            old_capabilities=dict(old_capabilities),
            new_capabilities=new_capabilities,
            reason=f"Triggered by rule: {triggered_rule.name}",
            trigger=triggered_rule.trigger,
            confidence=confidence,
            status=EvolutionStatus.PROPOSED,
            performance_before=previous.to_dict(),
        )

        self._logger.info(
            f"Proposed evolution for {fractal_id}: {old_specialty} -> {new_specialty} "
            f"(confidence: {confidence:.2f})"
        )

        return evolution

    def _compute_evolution(
        self,
        fractal_id: str,
        current: PerformanceSnapshot,
        previous: PerformanceSnapshot,
        rule: EvolutionRule,
    ) -> tuple[str, dict[str, float]]:
        """Compute the proposed evolution"""
        profile = self._current_profiles.get(fractal_id, {})
        old_capabilities = profile.get("capabilities", {})

        strengths = self._identify_strengths(current)
        weaknesses = self._identify_weaknesses(current)

        new_capabilities = dict(old_capabilities)

        if rule.trigger == EvolutionTrigger.TASK_SUCCESS_PATTERN:
            for strength in strengths:
                if strength not in new_capabilities:
                    new_capabilities[strength] = 0.5
                new_capabilities[strength] = min(1.0, new_capabilities[strength] + 0.1)

        elif rule.trigger == EvolutionTrigger.TASK_FAILURE_PATTERN:
            for weakness in weaknesses:
                if weakness in new_capabilities:
                    new_capabilities[weakness] = max(0.0, new_capabilities[weakness] - 0.1)

        elif rule.trigger == EvolutionTrigger.CAPABILITY_EXPANSION:
            for cap, score in current.capability_scores.items():
                if cap not in new_capabilities:
                    new_capabilities[cap] = score

        elif rule.trigger == EvolutionTrigger.PERFORMANCE_IMPROVEMENT:
            for cap, score in current.capability_scores.items():
                if cap in new_capabilities:
                    new_capabilities[cap] = (new_capabilities[cap] + score) / 2

        elif rule.trigger == EvolutionTrigger.PERFORMANCE_DECLINE:
            for cap in list(new_capabilities.keys()):
                if cap not in current.capability_scores:
                    new_capabilities[cap] *= 0.9

        top_capability = (
            max(new_capabilities.items(), key=lambda x: x[1])[0] if new_capabilities else "general"
        )
        new_specialty = top_capability

        return new_specialty, new_capabilities

    def _compute_evolution_confidence(
        self,
        current: PerformanceSnapshot,
        rule: EvolutionRule,
        old_capabilities: dict[str, float],
        new_capabilities: dict[str, float],
    ) -> float:
        """Compute confidence in the evolution"""
        base_confidence = rule.min_confidence

        sample_confidence = min(1.0, current.total_tasks / 50)

        capability_change = len(set(new_capabilities.keys()) ^ set(old_capabilities.keys()))
        change_confidence = max(0.5, 1.0 - capability_change * 0.1)

        final_confidence = base_confidence * 0.4 + sample_confidence * 0.3 + change_confidence * 0.3

        return min(1.0, final_confidence)

    def apply_evolution(self, evolution: ProfileEvolution) -> bool:
        """
        Apply a proposed evolution.

        Validates and applies the evolution, updating the fractal's profile.
        """
        if evolution.status != EvolutionStatus.PROPOSED:
            self._logger.warning(f"Cannot apply evolution in status: {evolution.status}")
            return False

        if evolution.confidence < 0.5:
            evolution.status = EvolutionStatus.REJECTED
            self._logger.info(f"Rejected evolution due to low confidence: {evolution.confidence}")
            return False

        evolution.status = EvolutionStatus.APPLIED
        evolution.applied_at = datetime.now()

        fractal_id = evolution.fractal_id
        self._current_profiles[fractal_id] = {
            "specialty": evolution.new_specialty,
            "capabilities": evolution.new_capabilities,
            "last_evolution": evolution.id,
            "evolved_at": evolution.applied_at.isoformat(),
        }

        self._last_evolution[fractal_id] = evolution.applied_at

        if fractal_id not in self._evolutions:
            self._evolutions[fractal_id] = []
        self._evolutions[fractal_id].append(evolution)

        if self._reputation_store:
            try:
                self._reputation_store.update_capabilities(
                    fractal_id,
                    evolution.new_capabilities,
                )
            except Exception as e:
                self._logger.warning(f"Failed to update reputation store: {e}")

        self._logger.info(
            f"Applied evolution for {fractal_id}: {evolution.old_specialty} -> "
            f"{evolution.new_specialty}"
        )

        if self._storage_path:
            self._save_to_storage()

        return True

    def get_evolution_history(
        self,
        fractal_id: str,
        limit: int = 10,
    ) -> list[ProfileEvolution]:
        """Get evolution history for a fractal"""
        evolutions = self._evolutions.get(fractal_id, [])
        return evolutions[-limit:]

    def get_current_profile(self, fractal_id: str) -> dict[str, Any] | None:
        """Get the current profile for a fractal"""
        return self._current_profiles.get(fractal_id)

    def rollback_evolution(self, evolution_id: str) -> bool:
        """Rollback a specific evolution"""
        for fractal_id, evolutions in self._evolutions.items():
            for i, evo in enumerate(evolutions):
                if evo.id == evolution_id:
                    if evo.status != EvolutionStatus.APPLIED:
                        return False

                    evo.status = EvolutionStatus.ROLLED_BACK

                    if i > 0:
                        prev_evo = evolutions[i - 1]
                        self._current_profiles[fractal_id] = {
                            "specialty": prev_evo.new_specialty,
                            "capabilities": prev_evo.new_capabilities,
                            "rolled_back_from": evolution_id,
                        }
                    else:
                        self._current_profiles[fractal_id] = {
                            "specialty": evo.old_specialty,
                            "capabilities": evo.old_capabilities,
                            "rolled_back_from": evolution_id,
                        }

                    self._logger.info(f"Rolled back evolution {evolution_id}")

                    if self._storage_path:
                        self._save_to_storage()

                    return True

        return False

    def get_stats(self) -> dict[str, Any]:
        """Get evolver statistics"""
        total_evolutions = sum(len(e) for e in self._evolutions.values())
        applied = sum(
            1
            for evos in self._evolutions.values()
            for e in evos
            if e.status == EvolutionStatus.APPLIED
        )
        rejected = sum(
            1
            for evos in self._evolutions.values()
            for e in evos
            if e.status == EvolutionStatus.REJECTED
        )
        rolled_back = sum(
            1
            for evos in self._evolutions.values()
            for e in evos
            if e.status == EvolutionStatus.ROLLED_BACK
        )

        return {
            "total_profiles": len(self._current_profiles),
            "total_evolutions": total_evolutions,
            "applied": applied,
            "rejected": rejected,
            "rolled_back": rolled_back,
            "fractals_with_history": len(self._evolutions),
        }

    def _save_to_storage(self) -> None:
        """Save evolver state to storage"""
        if not self._storage_path:
            return

        self._storage_path.mkdir(parents=True, exist_ok=True)

        profiles_file = self._storage_path / "profiles.json"
        with open(profiles_file, "w") as f:
            json.dump(self._current_profiles, f, indent=2, default=str)

        evolutions_file = self._storage_path / "evolutions.json"
        all_evolutions = [e.to_dict() for evos in self._evolutions.values() for e in evos]
        with open(evolutions_file, "w") as f:
            json.dump(all_evolutions, f, indent=2, default=str)

    def _load_from_storage(self) -> None:
        """Load evolver state from storage"""
        if not self._storage_path:
            return

        profiles_file = self._storage_path / "profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file) as f:
                    self._current_profiles = json.load(f)
            except Exception as e:
                self._logger.error(f"Failed to load profiles: {e}")

        evolutions_file = self._storage_path / "evolutions.json"
        if evolutions_file.exists():
            try:
                with open(evolutions_file) as f:
                    data = json.load(f)
                for item in data:
                    evo = ProfileEvolution.from_dict(item)
                    if evo.fractal_id not in self._evolutions:
                        self._evolutions[evo.fractal_id] = []
                    self._evolutions[evo.fractal_id].append(evo)
            except Exception as e:
                self._logger.error(f"Failed to load evolutions: {e}")


def create_profile_evolver(
    reputation_store: Any = None,
    storage_path: str | None = None,
) -> ProfileEvolver:
    """Factory function to create a ProfileEvolver"""
    return ProfileEvolver(
        reputation_store=reputation_store,
        storage_path=storage_path,
    )
