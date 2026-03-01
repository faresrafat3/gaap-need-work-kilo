"""
Layer 2 Learner - Tactical Learning Component
==============================================

Evolution 2026: Tactical-level learning for improved task planning.

Key Features:
- Learns from phase execution outcomes
- Tracks dependency patterns
- Improves task estimation over time
- Stores execution episodes for future reference

Learning Areas:
- Task duration estimation
- Dependency prediction
- Risk assessment refinement
- Phase planning optimization
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from gaap.core.logging import get_standard_logger as get_logger
from gaap.layers.layer2_config import Layer2Config
from gaap.layers.task_schema import (
    IntelligentTask,
    Phase,
    RiskLevel,
)

logger = get_logger("gaap.layer2.learner")


@dataclass
class ExecutionEpisode:
    """Record of a single task execution"""

    task_id: str
    task_name: str
    category: str
    phase_id: str | None

    estimated_duration_minutes: int
    actual_duration_minutes: int
    estimation_error: float = 0.0

    status: str = "completed"
    retry_count: int = 0

    risk_level: str = "MEDIUM"
    actual_risk_level: str = "MEDIUM"

    dependencies: list[str] = field(default_factory=list)
    missing_dependencies: list[str] = field(default_factory=list)

    tools_used: list[str] = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)

    def get_id_hash(self) -> str:
        content = f"{self.task_name}:{self.category}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class PhaseEpisode:
    """Record of a phase execution"""

    phase_id: str
    phase_name: str
    semantic_goal: str

    estimated_tasks: int
    actual_tasks: int

    estimated_duration_minutes: int
    actual_duration_minutes: int

    status: str = "completed"
    reassessment_count: int = 0

    task_episodes: list[ExecutionEpisode] = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)


@dataclass
class LearningPattern:
    """Learned pattern for tactical decisions"""

    pattern_type: str  # duration, dependency, risk, phase
    pattern_key: str  # Key for matching

    observed_values: list[float] = field(default_factory=list)
    average_value: float = 0.0
    confidence: float = 0.0

    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)

    def update(self, new_value: float) -> None:
        self.observed_values.append(new_value)
        self.sample_count = len(self.observed_values)

        if self.sample_count > 0:
            self.average_value = sum(self.observed_values) / self.sample_count

        if self.sample_count >= 3:
            variance = (
                sum((v - self.average_value) ** 2 for v in self.observed_values) / self.sample_count
            )
            std_dev = variance**0.5
            self.confidence = max(0.0, min(1.0, 1.0 - (std_dev / max(self.average_value, 1.0))))

        self.last_updated = time.time()


@dataclass
class TacticalInsights:
    """Aggregated insights from learning"""

    duration_adjustments: dict[str, float] = field(default_factory=dict)
    dependency_corrections: dict[str, list[str]] = field(default_factory=dict)
    risk_adjustments: dict[str, str] = field(default_factory=dict)
    phase_recommendations: dict[str, Any] = field(default_factory=dict)

    confidence_score: float = 0.0
    last_updated: float = field(default_factory=time.time)


class Layer2Learner:
    """
    Tactical learning component for Layer 2.

    Learns from execution history to improve:
    - Task duration estimates
    - Dependency predictions
    - Risk assessments
    - Phase planning
    """

    def __init__(
        self,
        config: Layer2Config | None = None,
        storage: Any = None,
    ):
        self._config = config or Layer2Config()
        self._storage = storage
        self._logger = logger

        self._task_episodes: list[ExecutionEpisode] = []
        self._phase_episodes: list[PhaseEpisode] = []

        self._duration_patterns: dict[str, LearningPattern] = {}
        self._dependency_patterns: dict[str, LearningPattern] = {}
        self._risk_patterns: dict[str, LearningPattern] = {}

        self._insights = TacticalInsights()

        self._learning_enabled = self._config.learning_enabled
        self._store_episodes = self._config.store_episodes

    def record_task_execution(
        self,
        task: IntelligentTask,
        actual_duration_minutes: int,
        status: str,
        retry_count: int = 0,
        tools_used: list[str] | None = None,
    ) -> None:
        """Record a task execution for learning"""

        if not self._learning_enabled:
            return

        estimation_error = abs(actual_duration_minutes - task.estimated_duration_minutes) / max(
            task.estimated_duration_minutes, 1
        )

        episode = ExecutionEpisode(
            task_id=task.id,
            task_name=task.name,
            category=task.category,
            phase_id=task.phase_id,
            estimated_duration_minutes=task.estimated_duration_minutes,
            actual_duration_minutes=actual_duration_minutes,
            estimation_error=estimation_error,
            status=status,
            retry_count=retry_count,
            risk_level=task.overall_risk_level.name,
            dependencies=task.dependencies.copy(),
            tools_used=tools_used or [],
        )

        self._task_episodes.append(episode)

        self._update_duration_pattern(task, actual_duration_minutes)
        self._update_risk_pattern(task, status, retry_count)

        if status == "failed" and task.dependencies:
            self._update_dependency_pattern(task, episode)

    def record_phase_execution(
        self,
        phase: Phase,
        actual_duration_minutes: int,
        status: str,
    ) -> None:
        """Record a phase execution for learning"""

        if not self._learning_enabled:
            return

        task_episodes = [ep for ep in self._task_episodes if ep.phase_id == phase.id]

        episode = PhaseEpisode(
            phase_id=phase.id,
            phase_name=phase.name,
            semantic_goal=phase.semantic_goal,
            estimated_tasks=phase.estimated_tasks,
            actual_tasks=len(phase.tasks),
            estimated_duration_minutes=phase.estimated_duration_minutes,
            actual_duration_minutes=actual_duration_minutes,
            status=status,
            reassessment_count=phase.reassessment_count,
            task_episodes=task_episodes,
        )

        self._phase_episodes.append(episode)

    def _update_duration_pattern(
        self,
        task: IntelligentTask,
        actual_duration: int,
    ) -> None:
        """Update duration estimation pattern"""

        key = f"{task.category}:{self._extract_task_type(task.name)}"

        if key not in self._duration_patterns:
            self._duration_patterns[key] = LearningPattern(
                pattern_type="duration",
                pattern_key=key,
            )

        self._duration_patterns[key].update(float(actual_duration))

        if self._duration_patterns[key].confidence >= 0.5:
            avg = self._duration_patterns[key].average_value
            adjustment = avg / max(task.estimated_duration_minutes, 1)
            self._insights.duration_adjustments[key] = adjustment

    def _update_risk_pattern(
        self,
        task: IntelligentTask,
        status: str,
        retry_count: int,
    ) -> None:
        """Update risk assessment pattern"""

        key = f"{task.category}:{task.overall_risk_level.name}"

        if key not in self._risk_patterns:
            self._risk_patterns[key] = LearningPattern(
                pattern_type="risk",
                pattern_key=key,
            )

        success_value = 1.0 if status == "completed" else 0.0
        self._risk_patterns[key].update(success_value)

        if retry_count > 2 and task.overall_risk_level != RiskLevel.HIGH:
            self._insights.risk_adjustments[key] = "HIGH"

    def _update_dependency_pattern(
        self,
        task: IntelligentTask,
        episode: ExecutionEpisode,
    ) -> None:
        """Update dependency prediction pattern"""

        if task.category not in self._insights.dependency_corrections:
            self._insights.dependency_corrections[task.category] = []

        for dep in task.dependencies:
            if dep not in self._insights.dependency_corrections[task.category]:
                self._insights.dependency_corrections[task.category].append(dep)

    def _extract_task_type(self, task_name: str) -> str:
        """Extract task type from name"""

        name_lower = task_name.lower()

        if "test" in name_lower:
            return "testing"
        elif "api" in name_lower or "endpoint" in name_lower:
            return "api"
        elif "database" in name_lower or "db" in name_lower or "model" in name_lower:
            return "database"
        elif "security" in name_lower:
            return "security"
        elif "document" in name_lower:
            return "documentation"
        else:
            return "general"

    def get_duration_estimate(
        self,
        category: str,
        task_type: str,
        default: int = 10,
    ) -> tuple[int, float]:
        """
        Get learned duration estimate.

        Returns:
            (estimated_minutes, confidence)
        """

        key = f"{category}:{task_type}"

        if key in self._duration_patterns:
            pattern = self._duration_patterns[key]
            if pattern.confidence >= 0.3:
                return (int(pattern.average_value), pattern.confidence)

        return (default, 0.0)

    def get_risk_recommendation(
        self,
        category: str,
        current_risk: RiskLevel,
    ) -> tuple[RiskLevel, float]:
        """
        Get learned risk recommendation.

        Returns:
            (recommended_risk, confidence)
        """

        key = f"{category}:{current_risk.name}"

        if key in self._risk_patterns:
            pattern = self._risk_patterns[key]

            if pattern.confidence >= 0.5:
                success_rate = pattern.average_value

                if success_rate < 0.5 and current_risk != RiskLevel.CRITICAL:
                    levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
                    current_idx = levels.index(current_risk)
                    if current_idx < len(levels) - 1:
                        return (levels[current_idx + 1], pattern.confidence)

        return (current_risk, 0.0)

    def get_dependency_suggestions(
        self,
        category: str,
    ) -> list[str]:
        """Get learned dependency suggestions for a category"""

        return self._insights.dependency_corrections.get(category, [])

    def get_insights(self) -> TacticalInsights:
        """Get current tactical insights"""

        self._insights.last_updated = time.time()

        total_confidence = 0.0
        pattern_count = 0

        for pattern in self._duration_patterns.values():
            total_confidence += pattern.confidence
            pattern_count += 1

        for pattern in self._risk_patterns.values():
            total_confidence += pattern.confidence
            pattern_count += 1

        if pattern_count > 0:
            self._insights.confidence_score = total_confidence / pattern_count

        return self._insights

    def apply_learning_to_task(
        self,
        task: IntelligentTask,
    ) -> IntelligentTask:
        """Apply learned insights to improve a task"""

        if not self._learning_enabled:
            return task

        task_type = self._extract_task_type(task.name)

        estimated_minutes, confidence = self.get_duration_estimate(
            task.category, task_type, task.estimated_duration_minutes
        )

        if confidence >= 0.5:
            task.estimated_duration_minutes = estimated_minutes
            task.metadata["learned_estimate"] = True
            task.metadata["estimate_confidence"] = confidence

        recommended_risk, risk_confidence = self.get_risk_recommendation(
            task.category, task.overall_risk_level
        )

        if risk_confidence >= 0.5:
            task.overall_risk_level = recommended_risk
            task.metadata["learned_risk"] = True
            task.metadata["risk_confidence"] = risk_confidence

        suggested_deps = self.get_dependency_suggestions(task.category)
        for dep in suggested_deps:
            if dep not in task.dependencies:
                task.dependencies.append(dep)
                task.metadata["learned_dependency"] = True

        return task

    def get_stats(self) -> dict[str, Any]:
        """Get learner statistics"""

        return {
            "task_episodes": len(self._task_episodes),
            "phase_episodes": len(self._phase_episodes),
            "duration_patterns": len(self._duration_patterns),
            "risk_patterns": len(self._risk_patterns),
            "insights_confidence": self._insights.confidence_score,
            "learning_enabled": self._learning_enabled,
        }

    def save_episodes(self) -> dict[str, Any]:
        """Export episodes for persistence"""

        return {
            "task_episodes": [
                {
                    "task_id": ep.task_id,
                    "task_name": ep.task_name,
                    "category": ep.category,
                    "phase_id": ep.phase_id,
                    "estimated_duration": ep.estimated_duration_minutes,
                    "actual_duration": ep.actual_duration_minutes,
                    "status": ep.status,
                    "retry_count": ep.retry_count,
                    "timestamp": ep.timestamp,
                }
                for ep in self._task_episodes[-100:]
            ],
            "phase_episodes": [
                {
                    "phase_id": ep.phase_id,
                    "phase_name": ep.phase_name,
                    "estimated_tasks": ep.estimated_tasks,
                    "actual_tasks": ep.actual_tasks,
                    "status": ep.status,
                    "timestamp": ep.timestamp,
                }
                for ep in self._phase_episodes[-50:]
            ],
            "patterns": {
                "duration": {
                    k: {
                        "average": v.average_value,
                        "confidence": v.confidence,
                        "samples": v.sample_count,
                    }
                    for k, v in self._duration_patterns.items()
                },
                "risk": {
                    k: {
                        "average": v.average_value,
                        "confidence": v.confidence,
                        "samples": v.sample_count,
                    }
                    for k, v in self._risk_patterns.items()
                },
            },
        }

    def load_episodes(self, data: dict[str, Any]) -> None:
        """Import episodes from persistence"""

        for ep_data in data.get("task_episodes", []):
            episode = ExecutionEpisode(
                task_id=ep_data.get("task_id", ""),
                task_name=ep_data.get("task_name", ""),
                category=ep_data.get("category", ""),
                phase_id=ep_data.get("phase_id"),
                estimated_duration_minutes=ep_data.get("estimated_duration", 10),
                actual_duration_minutes=ep_data.get("actual_duration", 10),
                status=ep_data.get("status", "completed"),
                retry_count=ep_data.get("retry_count", 0),
                timestamp=ep_data.get("timestamp", time.time()),
            )
            self._task_episodes.append(episode)

        for ep_data in data.get("phase_episodes", []):
            phase_episode = PhaseEpisode(
                phase_id=ep_data.get("phase_id", ""),
                phase_name=ep_data.get("phase_name", ""),
                semantic_goal=ep_data.get("semantic_goal", ""),
                estimated_tasks=ep_data.get("estimated_tasks", 0),
                actual_tasks=ep_data.get("actual_tasks", 0),
                estimated_duration_minutes=ep_data.get("estimated_duration", 0),
                actual_duration_minutes=ep_data.get("actual_duration", 0),
                status=ep_data.get("status", "completed"),
                timestamp=ep_data.get("timestamp", time.time()),
            )
            self._phase_episodes.append(phase_episode)

        for key, pattern_data in data.get("patterns", {}).get("duration", {}).items():
            self._duration_patterns[key] = LearningPattern(
                pattern_type="duration",
                pattern_key=key,
                average_value=pattern_data.get("average", 0),
                confidence=pattern_data.get("confidence", 0),
                sample_count=pattern_data.get("samples", 0),
            )

        for key, pattern_data in data.get("patterns", {}).get("risk", {}).items():
            self._risk_patterns[key] = LearningPattern(
                pattern_type="risk",
                pattern_key=key,
                average_value=pattern_data.get("average", 0),
                confidence=pattern_data.get("confidence", 0),
                sample_count=pattern_data.get("samples", 0),
            )


def create_layer2_learner(
    config: Layer2Config | None = None,
) -> Layer2Learner:
    """Factory function to create Layer2Learner"""

    return Layer2Learner(config=config)
