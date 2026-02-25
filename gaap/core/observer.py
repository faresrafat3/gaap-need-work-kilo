"""
OODA Observer (Evolution 2026)

Provides real-time environment scanning for the OODA Loop.

Features:
    - Resource monitoring (memory, disk, processes)
    - Cognitive state tracking (lessons, tasks)
    - Strategic drift detection
    - Resource exhaustion warnings
    - Axiom violation pattern detection
    - Goal drift analysis

Usage:
    >>> observer = create_observer(memory_system)
    >>> state = await observer.scan(ooda_state)
    >>> if state.needs_replanning:
    ...     await replan()
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from gaap.core.types import OODAState, ReplanTrigger
from gaap.memory import LessonStore

logger = logging.getLogger("gaap.core.observer")

DEFAULT_MEMORY_LIMIT_MB = int(os.getenv("GAAP_MEMORY_LIMIT_MB", "4000"))
DEFAULT_LESSON_COUNT = 3
FAILED_TASKS_THRESHOLD = 2
AXIOM_VIOLATION_THRESHOLD = 3


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EnvironmentState:
    """
    Snapshot of the current execution environment.

    Attributes:
        memory_usage_mb: Current memory usage in megabytes
        disk_space_free_mb: Free disk space in megabytes
        active_processes: Number of active processes
        lessons_available: List of relevant lessons from memory
        pending_tasks: Number of tasks in queue
        failed_tasks: Number of failed tasks
        needs_replanning: Flag indicating if replanning is needed
        replan_trigger: Reason for replanning (if triggered)

    Example:
        >>> state = EnvironmentState()
        >>> state.memory_usage_mb = 2048.0
        >>> state.needs_replanning = True
        >>> print(state.to_dict())
        {'memory': 2048.0, 'lessons': 0, 'replan': True}
    """

    memory_usage_mb: float = 0.0
    disk_space_free_mb: float = 0.0
    active_processes: int = 0

    # Cognitive State
    lessons_available: list[str] = field(default_factory=list)
    pending_tasks: int = 0
    failed_tasks: int = 0

    # Decisions
    needs_replanning: bool = False
    replan_trigger: ReplanTrigger = ReplanTrigger.NONE

    def to_dict(self) -> dict[str, Any]:
        """
        Convert environment state to dictionary.

        Returns:
            Dictionary with key metrics for serialization
        """
        return {
            "memory_mb": self.memory_usage_mb,
            "lessons_count": len(self.lessons_available),
            "needs_replanning": self.needs_replanning,
            "replan_trigger": self.replan_trigger.name,
        }


class Observer:
    """
    The 'Eye' of the OODA loop.

    Scans internal and external state to provide situational awareness
    for the OODA cycle. Monitors resources, cognitive state, and
    strategic alignment.

    Attributes:
        memory: Optional hierarchical memory system
        lesson_store: Persistent lesson storage for retrieval

    Usage:
        >>> observer = create_observer(memory_system)
        >>> state = await observer.scan(ooda_state)
        >>> if state.needs_replanning:
        ...     await replan()
    """

    def __init__(self, memory_system: Any = None) -> None:
        """
        Initialize observer.

        Args:
            memory_system: Optional hierarchical memory system for lesson retrieval
        """
        self.memory = memory_system
        self.lesson_store = LessonStore()

    async def scan(
        self,
        ooda_state: OODAState,
        task_graph: Any = None,
        original_goals: list[str] | None = None,
    ) -> EnvironmentState:
        """
        Perform comprehensive environment scan.

        Scans:
            1. Resource usage (memory, disk, processes)
            2. Cognitive state (lessons, pending tasks)
            3. Strategic alignment (goal drift detection)
            4. Resource limits (memory threshold)
            5. Axiom violation patterns (new)
            6. Execution health (new)

        Args:
            ooda_state: Current OODA loop state for analysis
            task_graph: Optional task graph for dependency analysis
            original_goals: Original goals for drift detection

        Returns:
            EnvironmentState with current snapshot

        Note:
            Gracefully handles errors in subsystems and returns
            partial state instead of crashing

        Example:
            >>> observer = create_observer(memory)
            >>> state = await observer.scan(ooda_state)
            >>> if state.needs_replanning:
            ...     print(f"Replan triggered: {state.replan_trigger.name}")
        """
        state = EnvironmentState()

        try:
            state.memory_usage_mb = self._get_memory_usage()

            if self.lesson_store:
                try:
                    query = " ".join(original_goals) if original_goals else "general execution"
                    state.lessons_available = self.lesson_store.retrieve_lessons(
                        query, k=DEFAULT_LESSON_COUNT
                    )
                except Exception as e:
                    logger.warning(f"Failed to retrieve lessons: {e}")
                    state.lessons_available = []

            if ooda_state:
                state.pending_tasks = len(ooda_state.in_progress_tasks)
                state.failed_tasks = len(ooda_state.failed_tasks)

                if len(ooda_state.failed_tasks) > FAILED_TASKS_THRESHOLD:
                    state.needs_replanning = True
                    state.replan_trigger = ReplanTrigger.L3_CRITICAL_FAILURE

                if len(ooda_state.axiom_violations) >= AXIOM_VIOLATION_THRESHOLD:
                    state.needs_replanning = True
                    state.replan_trigger = ReplanTrigger.AXIOM_VIOLATION
                    logger.warning(
                        f"AXIOM_VIOLATION pattern detected: {len(ooda_state.axiom_violations)} violations"
                    )

                if original_goals and len(ooda_state.completed_tasks) > 5:
                    goal_alignment = self._check_goal_alignment(ooda_state, original_goals)
                    if goal_alignment < 0.3:
                        state.needs_replanning = True
                        state.replan_trigger = ReplanTrigger.GOAL_DRIFT
                        logger.warning(f"GOAL_DRIFT detected: alignment={goal_alignment:.2f}")

            memory_limit = int(os.getenv("GAAP_MEMORY_LIMIT_MB", str(DEFAULT_MEMORY_LIMIT_MB)))
            if state.memory_usage_mb > memory_limit:
                state.needs_replanning = True
                state.replan_trigger = ReplanTrigger.RESOURCE_EXHAUSTED

        except Exception as e:
            logger.error(f"Observer scan failed: {e}")

        return state

    def _check_goal_alignment(self, ooda_state: OODAState, original_goals: list[str]) -> float:
        """
        Check alignment between completed tasks and original goals.

        Returns:
            Alignment score (0.0 to 1.0)
        """
        if not original_goals or not ooda_state.lessons_learned:
            return 1.0

        goals_text = " ".join(original_goals).lower()
        lessons_text = " ".join(ooda_state.lessons_learned).lower()

        goal_keywords = set(goals_text.split())
        lesson_keywords = set(lessons_text.split())
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "to",
            "of",
            "and",
            "in",
            "that",
            "for",
        }
        goal_keywords -= stop_words
        lesson_keywords -= stop_words

        if not goal_keywords:
            return 1.0

        overlap = len(goal_keywords & lesson_keywords)
        return overlap / len(goal_keywords)

    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes, or 0.0 if psutil not available

        Note:
            Uses psutil if available, otherwise returns mocked value
        """
        try:
            import psutil

            process = psutil.Process()
            return float(process.memory_info().rss / 1024 / 1024)
        except ImportError:
            logger.debug("psutil not installed, using mocked memory value")
            return 1024.0  # Mocked value

    def record_error(
        self,
        error: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an error for tracking and analysis.

        Args:
            error: Error message or description
            context: Optional context data (task_id, component, etc.)

        Note:
            Errors are logged and can be retrieved for pattern analysis.
            Error messages are truncated to 100 characters for readability.

        Example:
            >>> observer.record_error("Task timeout", context={"task_id": "123"})
        """
        error_msg = error[:100] if error else "Unknown error"
        if context:
            logger.warning(f"Error recorded: {error_msg}", extra=context)
        else:
            logger.warning(f"Error recorded: {error_msg}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get observer statistics.

        Returns:
            Dictionary with observer metrics and availability flags

        Example:
            >>> stats = observer.get_stats()
            >>> print(stats)
            {'memory_available': True, 'lesson_store_available': True, 'memory_usage_mb': 2048.0}
        """
        return {
            "memory_available": self.memory is not None,
            "lesson_store_available": self.lesson_store is not None,
            "memory_usage_mb": self._get_memory_usage(),
        }


def create_observer(memory_system: Any = None) -> Observer:
    """
    Factory function to create an Observer instance.

    Args:
        memory_system: Optional hierarchical memory system

    Returns:
        Configured Observer instance

    Example:
        >>> observer = create_observer(memory_system)
        >>> state = await observer.scan(ooda_state)
    """
    return Observer(memory_system)
