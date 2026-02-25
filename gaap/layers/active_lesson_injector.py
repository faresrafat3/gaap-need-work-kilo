"""
Active Lesson Injector - Pre-Execution Lesson Retrieval
========================================================

Evolution 2026: Active lesson injection before execution.

Key Features:
- Retrieves relevant lessons before execution
- Filters by relevance and age
- Injects into prompt context
- Learns from injection outcomes

Instead of just writing lessons passively, actively retrieves
and injects them to prevent repeating past mistakes.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.layers.layer3_config import Layer3Config, LessonInjectionConfig
from gaap.layers.layer2_tactical import AtomicTask, TaskCategory

logger = get_logger("gaap.layer3.lesson_injector")


@dataclass
class Lesson:
    """A learned lesson from past execution"""

    lesson_id: str
    content: str
    category: str
    context: str = ""

    task_type: str = "general"
    success: bool = False

    relevance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    source_task_id: str = ""
    source_error: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "content": self.content,
            "category": self.category,
            "context": self.context,
            "task_type": self.task_type,
            "success": self.success,
            "relevance_score": self.relevance_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source_task_id": self.source_task_id,
            "source_error": self.source_error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lesson":
        return cls(
            lesson_id=data.get("lesson_id", ""),
            content=data.get("content", ""),
            category=data.get("category", "general"),
            context=data.get("context", ""),
            task_type=data.get("task_type", "general"),
            success=data.get("success", False),
            relevance_score=data.get("relevance_score", 0.0),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            source_task_id=data.get("source_task_id", ""),
            source_error=data.get("source_error", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InjectionResult:
    """Result of lesson injection"""

    injected: bool
    lessons_count: int
    injected_content: str = ""
    lessons: list[Lesson] = field(default_factory=list)

    injection_position: str = "system"
    relevance_threshold_met: bool = True


class ActiveLessonInjector:
    """
    Active lesson injection before execution.

    Retrieves relevant lessons from memory and injects them
    into the prompt to prevent repeating past mistakes.
    """

    CATEGORY_MAPPING = {
        TaskCategory.API: "code",
        TaskCategory.DATABASE: "code",
        TaskCategory.FRONTEND: "code",
        TaskCategory.TESTING: "code",
        TaskCategory.SECURITY: "security",
        TaskCategory.INFRASTRUCTURE: "code",
        TaskCategory.INTEGRATION: "code",
        TaskCategory.INFORMATION_GATHERING: "research",
        TaskCategory.SOURCE_VERIFICATION: "research",
        TaskCategory.DATA_SYNTHESIS: "research",
        TaskCategory.LITERATURE_REVIEW: "research",
        TaskCategory.REPRODUCTION: "diagnostic",
        TaskCategory.LOG_ANALYSIS: "diagnostic",
        TaskCategory.ROOT_CAUSE_ANALYSIS: "diagnostic",
        TaskCategory.DIAGNOSTIC_ACTION: "diagnostic",
        TaskCategory.ANALYSIS: "analysis",
    }

    def __init__(
        self,
        config: Layer3Config | None = None,
        memory: Any = None,
    ):
        self._config = config or Layer3Config()
        self._memory = memory
        self._logger = logger

        self._injections = 0
        self._lessons_retrieved = 0
        self._cache_hits = 0

        self._lesson_cache: dict[str, list[Lesson]] = {}

    async def get_relevant_lessons(
        self,
        query: str,
        task: AtomicTask | None = None,
        k: int = 5,
    ) -> list[Lesson]:
        """
        Retrieve relevant lessons from memory.

        Args:
            query: Search query (usually task description)
            task: Task for category filtering
            k: Maximum lessons to retrieve

        Returns:
            List of relevant lessons
        """

        cache_key = f"{query[:100]}:{k}"
        if cache_key in self._lesson_cache:
            self._cache_hits += 1
            return self._lesson_cache[cache_key][:k]

        lessons = []

        if self._memory:
            try:
                lessons = await self._retrieve_from_memory(query, task, k)
            except Exception as e:
                self._logger.warning(f"Failed to retrieve from memory: {e}")

        if not lessons:
            lessons = self._get_default_lessons(task)

        self._lessons_retrieved += len(lessons)
        self._lesson_cache[cache_key] = lessons

        return lessons[:k]

    async def _retrieve_from_memory(
        self,
        query: str,
        task: AtomicTask | None,
        k: int,
    ) -> list[Lesson]:
        """Retrieve lessons from vector memory"""

        lessons = []

        config = self._config.lesson_injection

        categories = []
        if task and config.category_filter:
            categories = config.category_filter
        elif task:
            category = self.CATEGORY_MAPPING.get(task.category, "general")
            categories = [category]

        if hasattr(self._memory, "search"):
            results = await self._memory.search(
                query=query,
                k=k,
                filters={"category": categories} if categories else None,
            )

            for result in results:
                lesson = Lesson(
                    lesson_id=result.get("id", ""),
                    content=result.get("content", result.get("lesson", "")),
                    category=result.get("category", "general"),
                    context=result.get("context", ""),
                    relevance_score=result.get("score", 0.0),
                    created_at=datetime.fromisoformat(result["created_at"])
                    if result.get("created_at")
                    else datetime.now(),
                )

                if config.include_failures_only and lesson.success:
                    continue

                if lesson.relevance_score >= config.relevance_threshold:
                    lessons.append(lesson)

        elif hasattr(self._memory, "retrieve"):
            raw_lessons = self._memory.retrieve(query=query, k=k)

            for i, content in enumerate(raw_lessons):
                lesson = Lesson(
                    lesson_id=f"retrieved_{i}",
                    content=content,
                    category="general",
                    context="retrieved from memory",
                    relevance_score=0.7,
                )
                lessons.append(lesson)

        max_age_days = config.max_lesson_age_days
        if max_age_days > 0:
            cutoff = datetime.now() - timedelta(days=max_age_days)
            lessons = [l for l in lessons if l.created_at >= cutoff]

        return lessons

    def _get_default_lessons(self, task: AtomicTask | None) -> list[Lesson]:
        """Get default lessons based on task category"""

        defaults = {
            "code": [
                Lesson(
                    lesson_id="default_code_1",
                    content="Always check for edge cases and handle exceptions properly.",
                    category="code",
                ),
                Lesson(
                    lesson_id="default_code_2",
                    content="Use meaningful variable names and add docstrings.",
                    category="code",
                ),
            ],
            "security": [
                Lesson(
                    lesson_id="default_security_1",
                    content="Never trust user input - always validate and sanitize.",
                    category="security",
                ),
                Lesson(
                    lesson_id="default_security_2",
                    content="Avoid using eval(), exec(), or other dangerous functions.",
                    category="security",
                ),
            ],
            "diagnostic": [
                Lesson(
                    lesson_id="default_diag_1",
                    content="Start with the simplest hypothesis first.",
                    category="diagnostic",
                ),
            ],
            "research": [
                Lesson(
                    lesson_id="default_research_1",
                    content="Always verify sources from multiple perspectives.",
                    category="research",
                ),
            ],
        }

        if task:
            category = self.CATEGORY_MAPPING.get(task.category, "code")
            return defaults.get(category, defaults["code"])

        return defaults["code"]

    async def inject_lessons(
        self,
        messages: list[Message],
        task: AtomicTask,
        k: int | None = None,
    ) -> InjectionResult:
        """
        Inject relevant lessons into the message list.

        Args:
            messages: Current message list
            task: Task to inject lessons for
            k: Maximum lessons (uses config if not provided)

        Returns:
            InjectionResult with injection details
        """

        config = self._config.lesson_injection

        if not config.enabled:
            return InjectionResult(
                injected=False,
                lessons_count=0,
            )

        self._injections += 1

        k = k or config.max_lessons

        query = f"{task.name}: {task.description[:200]}"

        lessons = await self.get_relevant_lessons(query, task, k)

        if not lessons:
            return InjectionResult(
                injected=False,
                lessons_count=0,
            )

        injected_content = self._format_lessons(lessons)

        injection_position = config.injection_position

        if injection_position == "system":
            self._inject_as_system(messages, injected_content)
        elif injection_position == "user":
            self._inject_as_user(messages, injected_content)
        else:
            self._inject_prepend(messages, injected_content)

        return InjectionResult(
            injected=True,
            lessons_count=len(lessons),
            injected_content=injected_content,
            lessons=lessons,
            injection_position=injection_position,
        )

    def _format_lessons(self, lessons: list[Lesson]) -> str:
        """Format lessons for injection"""

        lines = ["## Lessons from Past Executions", ""]

        for i, lesson in enumerate(lessons, 1):
            lines.append(f"{i}. {lesson.content}")

            if lesson.source_error:
                lines.append(f"   - Previous error: {lesson.source_error[:100]}")

            if not lesson.success:
                lines.append("   - This was learned from a past failure")

        lines.append("")
        lines.append("Apply these lessons to avoid repeating past mistakes.")

        return "\n".join(lines)

    def _inject_as_system(self, messages: list[Message], content: str) -> None:
        """Inject as a system message at the beginning"""

        for i, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM:
                enhanced_content = f"{msg.content}\n\n{content}"
                messages[i] = Message(
                    role=MessageRole.SYSTEM,
                    content=enhanced_content,
                )
                return

        messages.insert(0, Message(role=MessageRole.SYSTEM, content=content))

    def _inject_as_user(self, messages: list[Message], content: str) -> None:
        """Inject as a user message"""

        messages.append(
            Message(
                role=MessageRole.USER,
                content=content,
            )
        )

    def _inject_prepend(self, messages: list[Message], content: str) -> None:
        """Prepend to the first user message"""

        for i, msg in enumerate(messages):
            if msg.role == MessageRole.USER:
                enhanced_content = f"{content}\n\n{msg.content}"
                messages[i] = Message(
                    role=MessageRole.USER,
                    content=enhanced_content,
                )
                return

        messages.insert(0, Message(role=MessageRole.USER, content=content))

    def inject_lessons_into_prompt(
        self,
        prompt: str,
        task: AtomicTask,
        k: int | None = None,
    ) -> str:
        """
        Inject lessons directly into a prompt string.

        Args:
            prompt: Original prompt
            task: Task to inject lessons for
            k: Maximum lessons

        Returns:
            Enhanced prompt with lessons
        """

        config = self._config.lesson_injection

        if not config.enabled:
            return prompt

        k = k or config.max_lessons

        query = f"{task.name}: {task.description[:200]}"

        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return prompt

            lessons = loop.run_until_complete(self.get_relevant_lessons(query, task, k))
        except Exception:
            lessons = self._get_default_lessons(task)

        if not lessons:
            return prompt

        lessons_content = self._format_lessons(lessons)

        return f"{lessons_content}\n\n---\n\n{prompt}"

    def record_injection_outcome(
        self,
        task_id: str,
        lessons_used: list[Lesson],
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record whether injected lessons helped"""

        if success:
            self._logger.debug(f"Injection successful for task {task_id}")
        else:
            self._logger.info(f"Injection didn't prevent failure for task {task_id}: {error}")

    def get_stats(self) -> dict[str, Any]:
        """Get injector statistics"""

        return {
            "total_injections": self._injections,
            "lessons_retrieved": self._lessons_retrieved,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._lesson_cache),
        }

    def clear_cache(self) -> None:
        """Clear the lesson cache"""

        self._lesson_cache.clear()
        self._logger.info("Lesson cache cleared")


def create_lesson_injector(
    config: Layer3Config | None = None,
    memory: Any = None,
) -> ActiveLessonInjector:
    """Factory function to create ActiveLessonInjector"""

    return ActiveLessonInjector(config=config, memory=memory)
