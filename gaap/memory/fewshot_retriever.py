"""
Dynamic Few-Shot Retriever - Medprompt-Inspired Example Selection

Implements intelligent few-shot selection based on:
- Task similarity matching
- Success trajectory analysis
- Contextual example ranking
- Performance-based filtering

Inspired by Medprompt: https://arxiv.org/abs/2311.16452
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.memory.fewshot_retriever")


class TaskCategory(Enum):
    """Categories of tasks"""

    CODE_GENERATION = auto()
    CODE_REVIEW = auto()
    DEBUGGING = auto()
    REFACTORING = auto()
    DOCUMENTATION = auto()
    TESTING = auto()
    ANALYSIS = auto()
    PLANNING = auto()
    RESEARCH = auto()
    UNKNOWN = auto()


class SuccessLevel(Enum):
    """Level of success for a trajectory"""

    FAILED = 0
    PARTIAL = 1
    SUCCESS = 2
    EXEMPLARY = 3


@dataclass
class TrajectoryStep:
    """
    A single step in a task execution trajectory.

    Represents one action taken during task execution.
    """

    step_id: int
    action: str
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "reasoning": self.reasoning,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryStep:
        return cls(
            step_id=data.get("step_id", 0),
            action=data.get("action", ""),
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            reasoning=data.get("reasoning", ""),
            duration_ms=data.get("duration_ms", 0.0),
            success=data.get("success", True),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SuccessMetrics:
    """
    Metrics measuring the success of a trajectory.

    Combines multiple factors to determine overall quality.
    """

    completion_rate: float = 0.0
    accuracy_score: float = 0.0
    efficiency_score: float = 0.0
    quality_score: float = 0.0
    user_satisfaction: float | None = None

    @property
    def overall_score(self) -> float:
        """Compute overall success score"""
        base = (
            self.completion_rate * 0.3
            + self.accuracy_score * 0.3
            + self.efficiency_score * 0.2
            + self.quality_score * 0.2
        )

        if self.user_satisfaction is not None:
            base = base * 0.8 + self.user_satisfaction * 0.2

        return min(1.0, max(0.0, base))

    def to_dict(self) -> dict[str, Any]:
        return {
            "completion_rate": self.completion_rate,
            "accuracy_score": self.accuracy_score,
            "efficiency_score": self.efficiency_score,
            "quality_score": self.quality_score,
            "user_satisfaction": self.user_satisfaction,
            "overall_score": self.overall_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuccessMetrics:
        return cls(
            completion_rate=data.get("completion_rate", 0.0),
            accuracy_score=data.get("accuracy_score", 0.0),
            efficiency_score=data.get("efficiency_score", 0.0),
            quality_score=data.get("quality_score", 0.0),
            user_satisfaction=data.get("user_satisfaction"),
        )


@dataclass
class Trajectory:
    """
    A complete task execution trajectory.

    Records the full path of a successful (or failed) task execution,
    enabling similar task retrieval and learning.

    Attributes:
        id: Unique identifier
        task_type: Category of the task
        task_description: Human-readable task description
        steps: List of execution steps
        result: Final result of the trajectory
        success_metrics: Metrics measuring success
        success_level: Categorized success level
        signature_name: Associated signature (if any)
        created_at: When this trajectory was recorded
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskCategory = TaskCategory.UNKNOWN
    task_description: str = ""
    steps: list[TrajectoryStep] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    success_metrics: SuccessMetrics = field(default_factory=SuccessMetrics)
    success_level: SuccessLevel = SuccessLevel.PARTIAL
    signature_name: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float:
        """Total duration of all steps"""
        return sum(s.duration_ms for s in self.steps)

    @property
    def step_count(self) -> int:
        """Number of steps in trajectory"""
        return len(self.steps)

    def compute_embedding_key(self) -> str:
        """Compute a key for embedding/lookup"""
        content = f"{self.task_type.name}:{self.task_description}"
        return hashlib.md5(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type.name,
            "task_description": self.task_description,
            "steps": [s.to_dict() for s in self.steps],
            "result": self.result,
            "success_metrics": self.success_metrics.to_dict(),
            "success_level": self.success_level.name,
            "signature_name": self.signature_name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            task_type=TaskCategory[data.get("task_type", "UNKNOWN")],
            task_description=data.get("task_description", ""),
            steps=[TrajectoryStep.from_dict(s) for s in data.get("steps", [])],
            result=data.get("result", {}),
            success_metrics=SuccessMetrics.from_dict(data.get("success_metrics", {})),
            success_level=SuccessLevel[data.get("success_level", "PARTIAL")],
            signature_name=data.get("signature_name"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """Result of few-shot retrieval"""

    query: str
    trajectories: list[Trajectory]
    similarity_scores: list[float]
    total_candidates: int
    retrieval_time_ms: float

    def get_best(self) -> Trajectory | None:
        """Get the best matching trajectory"""
        if not self.trajectories:
            return None
        return self.trajectories[0]


class FewShotRetriever:
    """
    Retrieves similar successful trajectories for few-shot prompting.

    Inspired by Medprompt's approach to dynamic few-shot selection:
    1. Index successful trajectories with embeddings
    2. Retrieve similar tasks based on semantic similarity
    3. Rank by success metrics and relevance
    4. Build few-shot prompts from best examples

    Usage:
        retriever = FewShotRetriever(vector_store=my_vector_store)

        # Index a successful trajectory
        retriever.index_trajectory(trajectory)

        # Retrieve similar examples
        results = retriever.retrieve_similar("Write a function to sort a list", k=3)

        # Build a few-shot prompt
        prompt = retriever.build_few_shot_prompt(task, examples)
    """

    def __init__(
        self,
        vector_store: Any = None,
        storage_path: str | None = None,
        min_success_level: SuccessLevel = SuccessLevel.SUCCESS,
        max_trajectory_age_days: int = 90,
    ) -> None:
        self._vector_store = vector_store
        self._storage_path = Path(storage_path) if storage_path else None
        self._min_success_level = min_success_level
        self._max_trajectory_age_days = max_trajectory_age_days

        self._trajectories: dict[str, Trajectory] = {}
        self._signature_index: dict[str, list[str]] = {}
        self._type_index: dict[TaskCategory, list[str]] = {}

        self._logger = logging.getLogger("gaap.memory.fewshot_retriever")

        if self._storage_path:
            self._load_trajectories()

    def index_trajectory(self, trajectory: Trajectory) -> str:
        """
        Index a trajectory for retrieval.

        Only indexes trajectories that meet minimum success level.

        Args:
            trajectory: The trajectory to index

        Returns:
            The trajectory ID
        """
        if trajectory.success_level.value < self._min_success_level.value:
            self._logger.debug(f"Skipping trajectory {trajectory.id}: success level too low")
            return trajectory.id

        self._trajectories[trajectory.id] = trajectory

        if trajectory.task_type not in self._type_index:
            self._type_index[trajectory.task_type] = []
        if trajectory.id not in self._type_index[trajectory.task_type]:
            self._type_index[trajectory.task_type].append(trajectory.id)

        if trajectory.signature_name:
            if trajectory.signature_name not in self._signature_index:
                self._signature_index[trajectory.signature_name] = []
            if trajectory.id not in self._signature_index[trajectory.signature_name]:
                self._signature_index[trajectory.signature_name].append(trajectory.id)

        if self._vector_store:
            embedding_text = self._create_embedding_text(trajectory)
            self._vector_store.add(
                content=embedding_text,
                metadata={
                    "trajectory_id": trajectory.id,
                    "task_type": trajectory.task_type.name,
                    "success_level": trajectory.success_level.name,
                    "success_score": trajectory.success_metrics.overall_score,
                },
            )

        self._logger.debug(f"Indexed trajectory: {trajectory.id}")

        if self._storage_path:
            self._save_trajectories()

        return trajectory.id

    def _create_embedding_text(self, trajectory: Trajectory) -> str:
        """Create text for embedding from trajectory"""
        parts = [
            f"Task: {trajectory.task_description}",
            f"Type: {trajectory.task_type.name}",
        ]

        for step in trajectory.steps[:3]:
            parts.append(f"Step: {step.action}")
            if step.reasoning:
                parts.append(f"Reasoning: {step.reasoning[:200]}")

        return " ".join(parts)

    def retrieve_similar(
        self,
        task_description: str,
        k: int = 3,
        task_type: TaskCategory | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """
        Retrieve similar trajectories.

        Args:
            task_description: Description of the task to match
            k: Maximum number of results
            task_type: Optional filter by task type
            min_score: Minimum success score threshold

        Returns:
            RetrievalResult with matching trajectories
        """
        start_time = datetime.now()

        candidates: list[tuple[Trajectory, float]] = []

        if self._vector_store:
            search_results = self._vector_store.search(
                query=task_description,
                n_results=k * 2,
            )

            for result in search_results:
                traj_id = (
                    result.metadata.get("trajectory_id") if hasattr(result, "metadata") else None
                )
                if traj_id and traj_id in self._trajectories:
                    traj = self._trajectories[traj_id]
                    score = result.score if hasattr(result, "score") else 0.5

                    if task_type and traj.task_type != task_type:
                        continue
                    if traj.success_metrics.overall_score < min_score:
                        continue

                    candidates.append((traj, score))

        if not candidates:
            search_set = list(self._trajectories.values())

            if task_type:
                type_ids = self._type_index.get(task_type, [])
                search_set = [self._trajectories[i] for i in type_ids if i in self._trajectories]

            for traj in search_set:
                score = self._compute_similarity(task_description, traj)
                if traj.success_metrics.overall_score >= min_score:
                    candidates.append((traj, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        top_candidates = candidates[:k]

        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000

        return RetrievalResult(
            query=task_description,
            trajectories=[c[0] for c in top_candidates],
            similarity_scores=[c[1] for c in top_candidates],
            total_candidates=len(candidates),
            retrieval_time_ms=retrieval_time,
        )

    def _compute_similarity(self, query: str, trajectory: Trajectory) -> float:
        """Compute simple similarity score between query and trajectory"""
        query_words = set(query.lower().split())
        desc_words = set(trajectory.task_description.lower().split())

        word_overlap = len(query_words & desc_words) / max(len(query_words), 1)

        success_boost = trajectory.success_metrics.overall_score * 0.3

        recency_score = 0.0
        age_days = (datetime.now() - trajectory.created_at).days
        if age_days < self._max_trajectory_age_days:
            recency_score = (
                (self._max_trajectory_age_days - age_days) / self._max_trajectory_age_days * 0.1
            )

        return min(1.0, word_overlap * 0.6 + success_boost + recency_score)

    def get_examples_for_signature(
        self,
        signature_name: str,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Get examples for a specific signature.

        Args:
            signature_name: Name of the signature
            k: Maximum number of examples

        Returns:
            List of example dictionaries
        """
        traj_ids = self._signature_index.get(signature_name, [])

        trajectories = [self._trajectories[tid] for tid in traj_ids if tid in self._trajectories]

        trajectories.sort(
            key=lambda t: t.success_metrics.overall_score,
            reverse=True,
        )

        examples = []
        for traj in trajectories[:k]:
            if traj.steps:
                inputs = traj.steps[0].input_data if traj.steps else {}
                outputs = traj.result

                examples.append(
                    {
                        "inputs": inputs,
                        "outputs": outputs,
                        "quality_score": traj.success_metrics.overall_score,
                        "source": f"trajectory:{traj.id}",
                    }
                )

        return examples

    def build_few_shot_prompt(
        self,
        task_description: str,
        examples: list[Trajectory] | None = None,
        k: int = 3,
        include_reasoning: bool = True,
    ) -> str:
        """
        Build a few-shot prompt from examples.

        Args:
            task_description: Description of the current task
            examples: Optional pre-retrieved examples
            k: Number of examples if retrieving
            include_reasoning: Whether to include reasoning traces

        Returns:
            Formatted few-shot prompt
        """
        if examples is None:
            result = self.retrieve_similar(task_description, k=k)
            examples = result.trajectories

        if not examples:
            return f"Task: {task_description}\n\nNo similar examples available."

        lines = ["Here are some examples of similar tasks:\n"]

        for i, traj in enumerate(examples, 1):
            lines.append(f"--- Example {i} ---")
            lines.append(f"Task: {traj.task_description}")

            if include_reasoning and traj.steps:
                lines.append("\nApproach:")
                for j, step in enumerate(traj.steps[:3], 1):
                    lines.append(f"  {j}. {step.action}")
                    if step.reasoning:
                        lines.append(f"     Reasoning: {step.reasoning[:100]}")

            if traj.result:
                lines.append("\nResult:")
                result_preview = json.dumps(traj.result, indent=2)[:500]
                lines.append(result_preview)

            lines.append("")

        lines.append(f"--- Current Task ---")
        lines.append(f"Task: {task_description}")

        return "\n".join(lines)

    def get_trajectory(self, trajectory_id: str) -> Trajectory | None:
        """Get a specific trajectory by ID"""
        return self._trajectories.get(trajectory_id)

    def get_stats(self) -> dict[str, Any]:
        """Get retriever statistics"""
        type_counts = {t.name: len(ids) for t, ids in self._type_index.items() if ids}

        success_dist = {level.name: 0 for level in SuccessLevel}
        for traj in self._trajectories.values():
            success_dist[traj.success_level.name] += 1

        avg_score = 0.0
        if self._trajectories:
            avg_score = sum(
                t.success_metrics.overall_score for t in self._trajectories.values()
            ) / len(self._trajectories)

        return {
            "total_trajectories": len(self._trajectories),
            "by_task_type": type_counts,
            "by_success_level": success_dist,
            "avg_success_score": round(avg_score, 3),
            "signatures_indexed": len(self._signature_index),
        }

    def _save_trajectories(self) -> None:
        """Save trajectories to storage"""
        if not self._storage_path:
            return

        self._storage_path.mkdir(parents=True, exist_ok=True)
        file_path = self._storage_path / "trajectories.json"

        data = [t.to_dict() for t in self._trajectories.values()]

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_trajectories(self) -> None:
        """Load trajectories from storage"""
        if not self._storage_path:
            return

        file_path = self._storage_path / "trajectories.json"

        if not file_path.exists():
            return

        try:
            with open(file_path) as f:
                data = json.load(f)

            for item in data:
                traj = Trajectory.from_dict(item)
                self._trajectories[traj.id] = traj

                if traj.task_type not in self._type_index:
                    self._type_index[traj.task_type] = []
                self._type_index[traj.task_type].append(traj.id)

                if traj.signature_name:
                    if traj.signature_name not in self._signature_index:
                        self._signature_index[traj.signature_name] = []
                    self._signature_index[traj.signature_name].append(traj.id)

            self._logger.info(f"Loaded {len(self._trajectories)} trajectories")

        except Exception as e:
            self._logger.error(f"Failed to load trajectories: {e}")

    def clear_old_trajectories(self, max_age_days: int | None = None) -> int:
        """Remove trajectories older than max_age_days"""
        max_age = max_age_days or self._max_trajectory_age_days
        cutoff = datetime.now()
        cutoff = cutoff.replace(day=cutoff.day - max_age)

        to_remove = [tid for tid, traj in self._trajectories.items() if traj.created_at < cutoff]

        for tid in to_remove:
            traj = self._trajectories[tid]
            self._type_index[traj.task_type].remove(tid)
            if traj.signature_name and traj.signature_name in self._signature_index:
                self._signature_index[traj.signature_name].remove(tid)
            del self._trajectories[tid]

        if self._storage_path and to_remove:
            self._save_trajectories()

        return len(to_remove)


def create_fewshot_retriever(
    vector_store: Any = None,
    storage_path: str | None = None,
) -> FewShotRetriever:
    """Factory function to create a FewShotRetriever"""
    return FewShotRetriever(
        vector_store=vector_store,
        storage_path=storage_path,
    )
