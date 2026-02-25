"""
Reflexion Engine - Self-Reflection for Failure Recovery
========================================================

Implements the Reflexion pattern for self-improving error recovery.
Instead of generic error templates, the agent generates its own
analysis of why it failed and what to do differently.

Workflow:
1. Error occurs during task execution
2. Agent generates self-reflection: "I failed because X. Plan: Y."
3. New prompt = Original prompt + Reflection context

This creates personalized, context-aware recovery strategies.

Reference: docs/evolution_plan_2026/26_HEALING_AUDIT_SPEC.md

Usage:
    from gaap.healing.reflexion import ReflexionEngine, Reflection

    engine = ReflexionEngine(llm_provider)

    # Generate reflection on failure
    reflection = await engine.reflect(
        error=ValueError("Invalid input"),
        task=task,
        previous_attempt=previous_output
    )

    # Refine prompt with reflection
    refined = engine.refine_prompt(original_prompt, reflection)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from gaap.healing.healing_config import ReflexionConfig

logger = logging.getLogger("gaap.healing.reflexion")


class ReflectionDepth(Enum):
    """عمق التفكير الذاتي"""

    SURFACE = auto()  # تحليل سطحي للخطأ
    MODERATE = auto()  # تحليل متوسط مع خطة
    DEEP = auto()  # تحليل عميق مع root cause


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider"""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


@dataclass
class Reflection:
    """
    نتائج التفكير الذاتي في الفشل

    Attributes:
        failure_analysis: لماذا فشلت المحاولة
        root_cause: السبب الجذري (إن وجد)
        proposed_fix: الحل المقترح
        confidence: ثقة الوكيل في الحل (0-1)
        alternative_approaches: بدائل مقترحة
        lessons_learned: دروس مستفادة
    """

    failure_analysis: str
    root_cause: str | None = None
    proposed_fix: str | None = None
    confidence: float = 0.5
    alternative_approaches: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    depth: ReflectionDepth = ReflectionDepth.MODERATE
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_analysis": self.failure_analysis,
            "root_cause": self.root_cause,
            "proposed_fix": self.proposed_fix,
            "confidence": self.confidence,
            "alternative_approaches": self.alternative_approaches,
            "lessons_learned": self.lessons_learned,
            "depth": self.depth.name,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reflection":
        return cls(
            failure_analysis=data.get("failure_analysis", ""),
            root_cause=data.get("root_cause"),
            proposed_fix=data.get("proposed_fix"),
            confidence=data.get("confidence", 0.5),
            alternative_approaches=data.get("alternative_approaches", []),
            lessons_learned=data.get("lessons_learned", []),
            depth=ReflectionDepth[data.get("depth", "MODERATE")],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(),
        )

    def to_prompt_context(self) -> str:
        """تحويل التفكير لسياق يمكن إضافته للـ prompt"""
        parts = [f"## PREVIOUS ATTEMPT FAILED\n\n**Failure Analysis:** {self.failure_analysis}"]

        if self.root_cause:
            parts.append(f"\n**Root Cause:** {self.root_cause}")

        if self.proposed_fix:
            parts.append(f"\n**Proposed Fix:** {self.proposed_fix}")

        if self.alternative_approaches:
            parts.append("\n**Alternative Approaches:**")
            for i, alt in enumerate(self.alternative_approaches, 1):
                parts.append(f"  {i}. {alt}")

        if self.lessons_learned:
            parts.append("\n**Lessons Learned:**")
            for lesson in self.lessons_learned:
                parts.append(f"  - {lesson}")

        parts.append(
            "\n**Instructions:** Apply these insights to avoid repeating the same mistakes."
        )

        return "\n".join(parts)


class ReflexionEngine:
    """
    محرك التفكير الذاتي في الفشل

    Uses LLM to generate intelligent self-reflections on failures,
    enabling context-aware recovery strategies.

    Features:
    - LLM-powered failure analysis
    - Root cause identification
    - Proposed fix generation
    - Alternative approach suggestions
    - Confidence scoring
    """

    REFLECTION_PROMPT = """You are an AI assistant analyzing a failed task execution. Generate a structured self-reflection.

## Original Task
{task_description}

## Previous Attempt Output
{previous_output}

## Error Encountered
Error Type: {error_type}
Error Message: {error_message}

## Context
{context}

---

Generate a self-reflection in the following format:

**Failure Analysis:** [Brief analysis of why the attempt failed]

**Root Cause:** [The underlying reason for the failure]

**Proposed Fix:** [Specific fix to apply]

**Alternative Approaches:**
1. [First alternative]
2. [Second alternative]

**Lessons Learned:**
- [Lesson 1]
- [Lesson 2]

**Confidence:** [0.0-1.0 score of confidence in the proposed fix]

Be specific and actionable. Focus on what went wrong and how to fix it."""

    DEEP_REFLECTION_PROMPT = """You are an AI assistant performing a deep analysis of a repeated failure.

## Original Task
{task_description}

## Previous Attempts
{attempt_history}

## Current Error
Error Type: {error_type}
Error Message: {error_message}

## Context
{context}

---

This task has failed multiple times. Perform a DEEP analysis:

1. **Pattern Recognition:** What patterns do you see in the failures?
2. **Fundamental Issue:** Is there a fundamental misunderstanding of the task?
3. **Strategy Pivot:** Should we approach this task differently?

Generate your analysis:

**Failure Analysis:** [Comprehensive analysis]

**Root Cause:** [Deep root cause]

**Proposed Fix:** [Radical fix or strategy change]

**Alternative Approaches:**
1. [Fundamentally different approach]
2. [Simplified approach]
3. [Different paradigm]

**Lessons Learned:**
- [Key insight 1]
- [Key insight 2]

**Confidence:** [0.0-1.0]"""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        model: str = "gpt-4o-mini",
        max_reflection_tokens: int = 1000,
        enable_deep_reflection: bool = True,
        config: "ReflexionConfig | None" = None,
    ):
        from gaap.healing.healing_config import ReflexionConfig

        self._config = config or ReflexionConfig()
        self._llm_provider = llm_provider
        self._model = self._config.model if config else model
        self._max_tokens = self._config.max_tokens if config else max_reflection_tokens
        self._enable_deep = self._config.enable_deep_reflexion if config else enable_deep_reflection
        self._deep_threshold = self._config.deep_reflexion_threshold
        self._cache_enabled = self._config.cache_reflections
        self._logger = logger

        self._reflection_cache: dict[str, Reflection] = {}

    @property
    def config(self) -> "ReflexionConfig":
        return self._config

    async def reflect(
        self,
        error: Exception,
        task_description: str,
        previous_output: str = "",
        context: dict[str, Any] | None = None,
        attempt_count: int = 1,
    ) -> Reflection:
        """
        Generate self-reflection on a failure.

        Args:
            error: The exception that occurred
            task_description: Original task description
            previous_output: Output from the failed attempt
            context: Additional context
            attempt_count: Number of attempts so far

        Returns:
            Reflection with analysis and proposed fix
        """
        context = context or {}
        cache_key = f"{task_description}:{str(error)}:{attempt_count}"

        if self._cache_enabled and cache_key in self._reflection_cache:
            self._logger.debug("Using cached reflection")
            return self._reflection_cache[cache_key]

        if self._llm_provider is None:
            return self._fallback_reflection(error, task_description, previous_output)

        try:
            depth = (
                ReflectionDepth.DEEP
                if attempt_count >= self._deep_threshold and self._enable_deep
                else ReflectionDepth.MODERATE
            )

            prompt = self._build_reflection_prompt(
                error=error,
                task_description=task_description,
                previous_output=previous_output,
                context=context,
                attempt_count=attempt_count,
                depth=depth,
            )

            messages = [{"role": "user", "content": prompt}]
            response = await self._llm_provider.complete(
                messages,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._config.temperature,
            )

            reflection = self._parse_reflection(response, depth)
            if self._cache_enabled:
                self._reflection_cache[cache_key] = reflection

            self._logger.info(f"Generated reflection with confidence {reflection.confidence:.2f}")
            return reflection

        except Exception as e:
            self._logger.warning(f"LLM reflection failed, using fallback: {e}")
            return self._fallback_reflection(error, task_description, previous_output)

    def _build_reflection_prompt(
        self,
        error: Exception,
        task_description: str,
        previous_output: str,
        context: dict[str, Any],
        attempt_count: int,
        depth: ReflectionDepth,
    ) -> str:
        if depth == ReflectionDepth.DEEP:
            template = self.DEEP_REFLECTION_PROMPT
            attempt_history = context.get(
                "attempt_history", f"Attempt {attempt_count}: {previous_output[:500]}"
            )
            return template.format(
                task_description=task_description[:1000],
                attempt_history=attempt_history,
                error_type=type(error).__name__,
                error_message=str(error)[:500],
                context=self._format_context(context),
            )
        else:
            template = self.REFLECTION_PROMPT
            return template.format(
                task_description=task_description[:1000],
                previous_output=previous_output[:1000]
                if previous_output
                else "No output generated",
                error_type=type(error).__name__,
                error_message=str(error)[:500],
                context=self._format_context(context),
            )

    def _format_context(self, context: dict[str, Any]) -> str:
        if not context:
            return "No additional context available"

        parts = []
        for key, value in context.items():
            if key in ("provider", "model", "task_id"):
                continue
            str_value = str(value)[:200]
            parts.append(f"- {key}: {str_value}")

        return "\n".join(parts) if parts else "No additional context available"

    def _parse_reflection(self, response: str, depth: ReflectionDepth) -> Reflection:
        def extract_section(text: str, marker: str) -> str:
            start = text.find(marker)
            if start == -1:
                return ""

            start += len(marker)
            end_markers = ["**", "\n\n", "\n**"]
            end = len(text)

            for em in end_markers:
                pos = text.find(em, start)
                if pos != -1 and pos < end:
                    end = pos

            return text[start:end].strip()

        failure_analysis = extract_section(response, "**Failure Analysis:**")
        root_cause = extract_section(response, "**Root Cause:**")
        proposed_fix = extract_section(response, "**Proposed Fix:**")
        confidence_str = extract_section(response, "**Confidence:**")

        alternative_approaches = []
        alt_section = response.find("**Alternative Approaches:**")
        if alt_section != -1:
            alt_text = response[alt_section : alt_section + 500]
            for line in alt_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    cleaned = line.lstrip("0123456789.- ").strip()
                    if cleaned and not cleaned.startswith("**"):
                        alternative_approaches.append(cleaned)

        lessons_learned = []
        lesson_section = response.find("**Lessons Learned:**")
        if lesson_section != -1:
            lesson_text = response[lesson_section : lesson_section + 500]
            for line in lesson_text.split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    cleaned = line.lstrip("- ").strip()
                    if cleaned and not cleaned.startswith("**"):
                        lessons_learned.append(cleaned)

        try:
            confidence = float(confidence_str.replace("/", ".").split()[0])
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, IndexError):
            confidence = 0.5

        return Reflection(
            failure_analysis=failure_analysis or "Unable to analyze failure",
            root_cause=root_cause or None,
            proposed_fix=proposed_fix or None,
            confidence=confidence,
            alternative_approaches=alternative_approaches[:3],
            lessons_learned=lessons_learned[:3],
            depth=depth,
        )

    def _fallback_reflection(
        self,
        error: Exception,
        task_description: str,
        previous_output: str,
    ) -> Reflection:
        error_type = type(error).__name__
        error_msg = str(error)

        if "syntax" in error_msg.lower() or "parse" in error_msg.lower():
            return Reflection(
                failure_analysis=f"Syntax or parsing error: {error_msg[:200]}",
                root_cause="The code or response had invalid syntax",
                proposed_fix="Review and fix syntax errors. Ensure all brackets, quotes are matched.",
                confidence=0.7,
                lessons_learned=["Validate syntax before processing"],
                depth=ReflectionDepth.SURFACE,
            )

        elif "timeout" in error_msg.lower():
            return Reflection(
                failure_analysis="Operation timed out",
                root_cause="The task took too long to complete",
                proposed_fix="Simplify the approach or break into smaller steps",
                confidence=0.6,
                alternative_approaches=["Use a simpler algorithm", "Reduce output size"],
                depth=ReflectionDepth.SURFACE,
            )

        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
            return Reflection(
                failure_analysis="Network or connection error",
                root_cause="Unable to reach external service",
                proposed_fix="Retry with exponential backoff",
                confidence=0.8,
                lessons_learned=["Network issues are often transient"],
                depth=ReflectionDepth.SURFACE,
            )

        else:
            return Reflection(
                failure_analysis=f"Error of type {error_type}: {error_msg[:200]}",
                root_cause="Unknown - requires further investigation",
                proposed_fix="Try alternative approach or simplify task",
                confidence=0.4,
                alternative_approaches=["Break task into smaller parts", "Use different method"],
                depth=ReflectionDepth.SURFACE,
            )

    def refine_prompt(
        self,
        original_prompt: str,
        reflection: Reflection,
        max_length: int = 8000,
    ) -> str:
        """
        Refine the original prompt with reflection context.

        Args:
            original_prompt: The original task prompt
            reflection: The generated reflection
            max_length: Maximum length for the refined prompt

        Returns:
            Refined prompt with reflection context
        """
        reflection_context = reflection.to_prompt_context()

        refined = f"{original_prompt}\n\n{reflection_context}"

        if len(refined) > max_length:
            truncated_original = original_prompt[: max_length - len(reflection_context) - 100]
            refined = f"{truncated_original}\n\n[...truncated...]\n\n{reflection_context}"

        return refined

    def clear_cache(self) -> None:
        """Clear the reflection cache"""
        self._reflection_cache.clear()
        self._logger.debug("Reflection cache cleared")
