# Self Healer
import asyncio
import logging
import random
import re
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

from gaap.core.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from gaap.core.types import (
    HealingLevel,
    Task,
    TaskComplexity,
)

if TYPE_CHECKING:
    from gaap.healing.reflexion import ReflexionEngine
    from gaap.healing.healing_config import HealingConfig
    from gaap.meta_learning.failure_store import FailureStore

# =============================================================================
# Logger Setup
# =============================================================================


from gaap.core.logging import get_standard_logger as get_logger


# =============================================================================
# Enums
# =============================================================================


class ErrorCategory(Enum):
    """تصنيفات الأخطاء"""

    TRANSIENT = auto()  # خطأ عابر (شبكة، timeout)
    SYNTAX = auto()  # خطأ صيغة
    LOGIC = auto()  # خطأ منطقي
    MODEL_LIMIT = auto()  # حدود النموذج
    RESOURCE = auto()  # موارد (ميزانية، rate limit)
    CRITICAL = auto()  # خطأ حرج
    UNKNOWN = auto()  # غير معروف


class RecoveryAction(Enum):
    """إجراءات الاسترداد"""

    RETRY = auto()
    REFINE_PROMPT = auto()
    CHANGE_MODEL = auto()
    SIMPLIFY_TASK = auto()
    DECOMPOSE_TASK = auto()
    ESCALATE = auto()
    ABORT = auto()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ErrorContext:
    """سياق الخطأ"""

    error: Exception
    category: ErrorCategory
    message: str
    task_id: str
    provider: str = ""
    model: str = ""
    attempt: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": type(self.error).__name__,
            "category": self.category.name,
            "message": self.message,
            "task_id": self.task_id,
            "provider": self.provider,
            "model": self.model,
            "attempt": self.attempt,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RecoveryResult:
    """نتيجة الاسترداد"""

    success: bool
    action: RecoveryAction
    level: HealingLevel
    result: Any | None = None
    error: str | None = None
    attempts: int = 0
    time_spent_ms: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingRecord:
    """سجل تعافي"""

    task_id: str
    level: HealingLevel
    action: RecoveryAction
    success: bool
    duration_ms: float
    error_category: ErrorCategory
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Error Classifier
# =============================================================================


class ErrorClassifier:
    """مصنف الأخطاء"""

    # أنماط الأخطاء
    PATTERNS = {
        ErrorCategory.TRANSIENT: [
            r"connection\s+(reset|refused)",
            r"network\s+error",
            r"service\s+unavailable",
            r"temporary\s+failure",
            r"rate\s+limit",
            r"too\s+many\s+requests",
            r"exhausted",
            r"concurrency",
        ],
        ErrorCategory.MODEL_LIMIT: [
            r"timeout",  # Timeouts = model too slow, not transient
            r"timed\s+out",
            r"maximum\s+context",
            r"token\s+limit",
            r"content\s+policy",
            r"safety\s+filter",
            r"model\s+overloaded",
        ],
        ErrorCategory.RESOURCE: [
            r"budget\s+exceeded",
            r"quota\s+exceeded",
            r"out\s+of\s+memory",
            r"disk\s+full",
        ],
        ErrorCategory.CRITICAL: [
            r"security\s+violation",
            r"unauthorized",
            r"forbidden",
            r"fatal\s+error",
        ],
    }

    @classmethod
    def classify(cls, error: Exception) -> ErrorCategory:
        """تصنيف الخطأ"""
        error_message = str(error).lower()

        # تصنيف حسب نوع الاستثناء أولاً (أكثر دقة من regex)
        if isinstance(error, ProviderRateLimitError):
            return ErrorCategory.TRANSIENT

        if isinstance(error, ProviderTimeoutError):
            return ErrorCategory.MODEL_LIMIT  # Timeout = model slow, skip L1_RETRY

        if isinstance(error, ProviderError):
            return ErrorCategory.MODEL_LIMIT

        if isinstance(error, TimeoutError):
            return ErrorCategory.TRANSIENT  # Generic timeout = retryable

        if "auth" in type(error).__name__.lower():
            return ErrorCategory.CRITICAL

        # فحص الأنماط (fallback for untyped errors)
        for category, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return category

        return ErrorCategory.UNKNOWN


class SemanticErrorClassifier:
    """
    مصنف الأخطاء الدلالي

    Uses LLM for semantic classification when regex fails.
    Provides more intelligent error categorization.
    """

    CLASSIFICATION_PROMPT = """Analyze this error and classify it into ONE category.

Error Type: {error_type}
Error Message: {error_message}
Task Context: {task_context}

Categories:
- TRANSIENT: Network issues, rate limits, temporary failures (retry immediately)
- SYNTAX: Code syntax errors, parsing failures (fix and retry)
- LOGIC: Wrong output, validation failures, incorrect results (redesign approach)
- MODEL_LIMIT: Token limits, timeouts, content policy (change model or simplify)
- RESOURCE: Budget exceeded, quota issues, memory issues (escalate or wait)
- CRITICAL: Security violations, auth failures (immediate escalation)

Respond with ONLY the category name, nothing else."""

    def __init__(self, llm_provider: Any = None, model: str = "gpt-4o-mini"):
        self._llm_provider = llm_provider
        self._model = model
        self._logger = get_logger("gaap.healing.semantic_classifier")

    async def classify(
        self,
        error: Exception,
        task_context: str = "",
    ) -> ErrorCategory:
        """
        Classify error using LLM for semantic understanding.

        Args:
            error: The exception to classify
            task_context: Context about the task being executed

        Returns:
            ErrorCategory classification
        """
        regex_result = ErrorClassifier.classify(error)
        if regex_result != ErrorCategory.UNKNOWN:
            return regex_result

        if self._llm_provider is None:
            return ErrorCategory.UNKNOWN

        try:
            prompt = self.CLASSIFICATION_PROMPT.format(
                error_type=type(error).__name__,
                error_message=str(error)[:500],
                task_context=task_context[:300],
            )

            messages = [{"role": "user", "content": prompt}]
            response = await self._llm_provider.complete(
                messages,
                model=self._model,
                max_tokens=20,
                temperature=0.1,
            )

            category_str = response.strip().upper()
            for cat in ErrorCategory:
                if cat.name == category_str:
                    self._logger.debug(f"Semantic classification: {cat.name}")
                    return cat

            return ErrorCategory.UNKNOWN

        except Exception as e:
            self._logger.warning(f"Semantic classification failed: {e}")
            return ErrorCategory.UNKNOWN


# =============================================================================
# Prompt Refiner
# =============================================================================


class PromptRefiner:
    """محسن الـ Prompts"""

    # قوالب التحسين السيادي (Reflexion-based)
    REFINEMENT_TEMPLATES = {
        "syntax_error": """
CRITICAL: The previous attempt resulted in a SYNTAX ERROR. 

Error Message: {error_message}
Original Request: {original_prompt}

[INSTRUCTIONS]
1. Analyze WHY the syntax error occurred (e.g., missing imports, unclosed brackets).
2. Think Step-by-Step about the fix.
3. Provide the corrected implementation ensuring full syntax validity.
4. Do NOT repeat the same error.
""",
        "timeout": """
CRITICAL: The previous attempt TIMED OUT (Transient/Performance Issue).

Original Request: {original_prompt}

[INSTRUCTIONS]
1. Identify potential performance bottlenecks in your previous logic.
2. Provide a more EFFICIENT and STREAMLINED solution.
3. Remove unnecessary complexity that might be causing latency.
""",
        "validation_failed": """
CRITICAL: The previous output FAILED logic/quality validation.

Validation Issues: {error_message}
Original Request: {original_prompt}

[INSTRUCTIONS]
1. Explain why your previous solution failed to meet the criteria.
2. Redesign the approach to specifically address all validation errors listed above.
3. Ensure high-fidelity logic and alignment with requirements.
""",
        "general": """
CRITICAL: The previous attempt failed for unknown or multiple reasons.

Error Context: {error_message}
Original Request: {original_prompt}

[INSTRUCTIONS]
1. Perform a Root Cause Analysis (RCA) on the failure.
2. Pivot your strategy if necessary.
3. Provide a robust, improved implementation that handles edge cases and errors.
""",
    }

    @classmethod
    def refine(cls, original_prompt: str, error: Exception, error_category: ErrorCategory) -> str:
        """تحسين الـ Prompt"""
        template_key = "general"

        if error_category == ErrorCategory.SYNTAX:
            template_key = "syntax_error"
        elif error_category == ErrorCategory.TRANSIENT:
            template_key = "timeout"
        elif error_category == ErrorCategory.LOGIC:
            template_key = "validation_failed"

        template = cls.REFINEMENT_TEMPLATES.get(template_key, cls.REFINEMENT_TEMPLATES["general"])

        return template.format(original_prompt=original_prompt, error_message=str(error)[:500])


# =============================================================================
# Self-Healing System
# =============================================================================


class SelfHealingSystem:
    """
    نظام التعافي الذاتي

    يستخدم 5 مستويات للتعافي:
    - L1: إعادة المحاولة البسيطة
    - L2: تحسين الـ Prompt (مع Reflexion)
    - L3: تغيير النموذج
    - L4: تغيير الاستراتيجية (تفكيك المهمة)
    - L5: تصعيد بشري

    Features:
    - ReflexionEngine للتفكير الذاتي في الفشل
    - SemanticErrorClassifier للتصنيف الذكي
    - Post-Mortem Memory لتخزين التجارب الفاشلة
    - Pattern Detection للكشف عن أنماط الفشل
    - Configurable via HealingConfig

    """

    def __init__(
        self,
        max_level: HealingLevel = HealingLevel.L4_STRATEGY_SHIFT,
        on_escalate: Callable[[ErrorContext], Awaitable[None]] | None = None,
        memory: Any = None,
        reflexion_engine: "ReflexionEngine | None" = None,
        failure_store: "FailureStore | None" = None,
        semantic_classifier: SemanticErrorClassifier | None = None,
        llm_provider: Any = None,
        config: "HealingConfig | None" = None,
    ):
        from gaap.healing.healing_config import HealingConfig

        self._config = config or HealingConfig()
        self.max_level = HealingLevel(self._config.max_healing_level)
        self.on_escalate = on_escalate
        self._logger = get_logger("gaap.healing")
        self._memory = memory
        self._llm_provider = llm_provider
        self._reflexion_engine = reflexion_engine
        self._failure_store = failure_store
        self._semantic_classifier = semantic_classifier or SemanticErrorClassifier(
            llm_provider=llm_provider,
            model=self._config.semantic_classifier.model,
        )
        self._records: list[HealingRecord] = []
        self._error_history: dict[str, list[ErrorContext]] = {}
        self._reflection_history: dict[str, list[Any]] = {}
        self._pattern_history: dict[str, list[dict[str, Any]]] = {}
        self._cooldown_patterns: dict[str, float] = {}
        self._total_healing_attempts = 0
        self._successful_recoveries = 0
        self._escalations = 0
        self._patterns_detected = 0

    @property
    def config(self) -> "HealingConfig":
        return self._config

    # =========================================================================
    # Main Healing Method
    # =========================================================================

    async def heal(
        self,
        error: Exception,
        task: Task,
        execute_func: Callable[[Task], Awaitable[Any]],
        context: dict[str, Any] | None = None,
    ) -> RecoveryResult:
        """
        محاولة التعافي من خطأ

        Args:
            error: الخطأ الذي حدث
            task: المهمة المراد تنفيذها
            execute_func: دالة التنفيذ
            context: سياق إضافي

        Returns:
            نتيجة الاسترداد
        """
        start_time = time.time()
        self._total_healing_attempts += 1

        if context is None:
            context = {}

        if self._memory and task.description:
            self._inject_few_shot_examples(task, context)

        error_category = ErrorClassifier.classify(error)
        if error_category == ErrorCategory.UNKNOWN and self._semantic_classifier:
            try:
                error_category = await self._semantic_classifier.classify(error, task.description)
            except Exception as e:
                self._logger.debug(f"Semantic classification failed: {e}")

        error_ctx = ErrorContext(
            error=error,
            category=error_category,
            message=str(error),
            task_id=task.id,
            provider=context.get("provider", "") if context else "",
            model=context.get("model", "") if context else "",
            attempt=0,
            stack_trace=traceback.format_exc(),
        )

        self._record_error(error_ctx)

        pattern_id = self._detect_failure_pattern(error_ctx)
        if pattern_id and self._should_auto_escalate(pattern_id):
            escalation_level = self._get_pattern_escalation_level(pattern_id)
            self._logger.warning(
                f"Auto-escalating due to pattern {pattern_id} to {escalation_level.name}"
            )
            current_level = escalation_level
        else:
            current_level = self._determine_start_level(error_category)

        self._logger.info(
            f"Starting healing for task {task.id} "
            f"at level {current_level.name}, "
            f"error category: {error_category.name}"
        )

        while current_level.value <= self.max_level.value:
            result = await self._attempt_level(
                level=current_level,
                error_ctx=error_ctx,
                task=task,
                execute_func=execute_func,
                context=context,
            )

            if result.success:
                self._successful_recoveries += 1
                result.time_spent_ms = (time.time() - start_time) * 1000
                self._record_healing(
                    task_id=task.id,
                    level=current_level,
                    action=result.action,
                    success=True,
                    error_category=error_category,
                )
                if self._failure_store and error_ctx.attempt > 1:
                    await self._record_successful_recovery(error_ctx, task, current_level)
                try:
                    from gaap.core.observability import observability as _obs

                    _obs.record_healing(level=current_level.name, success=True)
                except Exception:
                    pass
                return result

            next_level = self._get_next_level(current_level)
            if next_level is None or next_level.value > self.max_level.value:
                break

            current_level = next_level

        self._escalations += 1
        result = RecoveryResult(
            success=False,
            action=RecoveryAction.ESCALATE,
            level=current_level,
            error="All healing levels exhausted",
            attempts=error_ctx.attempt,
            time_spent_ms=(time.time() - start_time) * 1000,
        )

        if self._failure_store:
            await self._record_failure(error_ctx, task)

        if self.on_escalate:
            await self.on_escalate(error_ctx)

        self._record_healing(
            task_id=task.id,
            level=current_level,
            action=RecoveryAction.ESCALATE,
            success=False,
            error_category=error_category,
            details=str(error),
        )
        try:
            from gaap.core.observability import observability as _obs

            _obs.record_healing(level=current_level.name, success=False)
        except Exception:
            pass

        return result

    def _determine_start_level(self, error_category: ErrorCategory) -> HealingLevel:
        """تحديد مستوى البداية"""
        if error_category == ErrorCategory.TRANSIENT:
            return HealingLevel.L1_RETRY
        elif error_category == ErrorCategory.SYNTAX:
            return HealingLevel.L2_REFINE
        elif error_category == ErrorCategory.MODEL_LIMIT:
            return HealingLevel.L3_PIVOT
        elif error_category == ErrorCategory.LOGIC:
            return HealingLevel.L2_REFINE
        elif error_category == ErrorCategory.CRITICAL:
            return HealingLevel.L5_HUMAN_ESCALATION
        else:
            return HealingLevel.L1_RETRY

    def _calculate_delay(self, level: HealingLevel, attempt: int) -> float:
        """
        Calculate delay with exponential backoff and jitter.

        Args:
            level: Current healing level
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        if attempt == 0:
            return 0.0

        base_delay = self._config.base_delay_seconds
        delay: float

        if self._config.exponential_backoff:
            level_multipliers: dict[HealingLevel, float] = {
                HealingLevel.L1_RETRY: 1.0,
                HealingLevel.L2_REFINE: 2.0,
                HealingLevel.L3_PIVOT: 3.0,
                HealingLevel.L4_STRATEGY_SHIFT: 5.0,
                HealingLevel.L5_HUMAN_ESCALATION: 0.0,
            }
            multiplier: float = level_multipliers.get(level, 1.0)
            delay = base_delay * multiplier * (2 ** (attempt - 1))
        else:
            delay = base_delay * (attempt + 1)

        delay = min(delay, self._config.max_delay_seconds)

        if self._config.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)

    async def _attempt_level(
        self,
        level: HealingLevel,
        error_ctx: ErrorContext,
        task: Task,
        execute_func: Callable[[Task], Awaitable[Any]],
        context: dict[str, Any] | None = None,
    ) -> RecoveryResult:
        """محاولة مستوى معين"""
        max_retries = self._config.max_retries_per_level

        for attempt in range(max_retries):
            error_ctx.attempt = attempt + 1

            self._logger.info(f"Attempting {level.name} (attempt {attempt + 1}/{max_retries})")

            delay = self._calculate_delay(level, attempt)
            if delay > 0:
                await asyncio.sleep(delay)

            try:
                # تنفيذ الإجراء المناسب
                action, modified_task = await self._apply_level_action(
                    level, error_ctx, task, context
                )

                # محاولة التنفيذ
                result = await execute_func(modified_task)

                # Check if the result object itself indicates failure
                # (execute_func may return ExecutionResult with success=False
                #  without raising an exception)
                result_success = getattr(result, "success", True)
                if not result_success:
                    error_msg = getattr(result, "error", "Execution returned success=False")
                    self._logger.warning(
                        f"{level.name} attempt {attempt + 1}: execution returned failure: {str(error_msg)[:100]}"
                    )
                    error_ctx.error = Exception(str(error_msg))
                    continue

                return RecoveryResult(
                    success=True,
                    action=action,
                    level=level,
                    result=result,
                    attempts=attempt + 1,
                    metadata={"action": action.name},
                )

            except Exception as e:
                self._logger.warning(f"{level.name} attempt {attempt + 1} failed: {e}")
                error_ctx.error = e
                continue

        return RecoveryResult(
            success=False,
            action=RecoveryAction.RETRY,
            level=level,
            error=str(error_ctx.error),
            attempts=max_retries,
        )

    async def _apply_level_action(
        self,
        level: HealingLevel,
        error_ctx: ErrorContext,
        task: Task,
        context: dict[str, Any] | None = None,
    ) -> tuple[RecoveryAction, Task]:
        """تطبيق إجراء المستوى"""
        context = context or {}

        if level == HealingLevel.L1_RETRY:
            return RecoveryAction.RETRY, task

        elif level == HealingLevel.L2_REFINE:
            refined_description = await self._refine_with_reflexion(task, error_ctx, context)

            modified_task = Task(
                id=task.id,
                description=refined_description,
                type=task.type,
                priority=task.priority,
                complexity=task.complexity,
                context=task.context,
                constraints=task.constraints,
                metadata={
                    **task.metadata,
                    "refined": True,
                    "reflexion_used": self._reflexion_engine is not None,
                },
            )

            return RecoveryAction.REFINE_PROMPT, modified_task

        elif level == HealingLevel.L3_PIVOT:
            modified_task = Task(
                id=task.id,
                description=task.description,
                type=task.type,
                priority=task.priority,
                complexity=task.complexity,
                context=task.context,
                constraints={**task.constraints, "force_model_tier": "higher"},
                metadata={**task.metadata, "model_pivoted": True},
            )

            return RecoveryAction.CHANGE_MODEL, modified_task

        elif level == HealingLevel.L4_STRATEGY_SHIFT:
            simplified_description = self._simplify_task(task)

            modified_task = Task(
                id=task.id,
                description=simplified_description,
                type=task.type,
                priority=task.priority,
                complexity=TaskComplexity.SIMPLE,
                context=task.context,
                constraints={**task.constraints, "simplified": True},
                metadata={**task.metadata, "simplified": True},
            )

            return RecoveryAction.SIMPLIFY_TASK, modified_task

        else:
            return RecoveryAction.ESCALATE, task

    def _simplify_task(self, task: Task) -> str:
        """تبسيط وصف المهمة"""
        # استخراج المتطلبات الأساسية
        description = task.description

        # إضافة تعليمات التبسيط
        simplified = f"""
IMPORTANT: Simplified version of the original task.

Original task: {description}

Please provide a MINIMAL implementation that:
1. Addresses only the core requirement
2. Uses the simplest possible approach
3. Avoids complex optimizations
4. Focuses on correctness over elegance

This is a simplified version due to previous failures with the full implementation.
"""
        return simplified

    def _get_next_level(self, current: HealingLevel) -> HealingLevel | None:
        """الحصول على المستوى التالي"""
        levels = list(HealingLevel)
        current_idx = levels.index(current)

        if current_idx < len(levels) - 1:
            return levels[current_idx + 1]
        return None

    # =========================================================================
    # Reflexion Integration
    # =========================================================================

    async def _refine_with_reflexion(
        self,
        task: Task,
        error_ctx: ErrorContext,
        context: dict[str, Any],
    ) -> str:
        """
        Refine prompt using ReflexionEngine if available.
        Falls back to PromptRefiner if no engine configured.
        """
        previous_output = context.get("previous_output", "")
        attempt_count = len(self._error_history.get(task.id, []))

        if self._reflexion_engine:
            try:
                reflection = await self._reflexion_engine.reflect(
                    error=error_ctx.error,
                    task_description=task.description,
                    previous_output=previous_output,
                    context=context,
                    attempt_count=attempt_count,
                )

                if task.id not in self._reflection_history:
                    self._reflection_history[task.id] = []
                self._reflection_history[task.id].append(reflection)

                refined = self._reflexion_engine.refine_prompt(task.description, reflection)
                self._logger.info(
                    f"Generated reflexion with confidence {reflection.confidence:.2f}"
                )
                return refined

            except Exception as e:
                self._logger.warning(f"Reflexion failed, using fallback: {e}")

        return PromptRefiner.refine(task.description, error_ctx.error, error_ctx.category)

    async def _record_failure(
        self,
        error_ctx: ErrorContext,
        task: Task,
    ) -> None:
        """Record failure to FailureStore for future avoidance."""
        if not self._failure_store:
            return

        try:
            from gaap.meta_learning.failure_store import FailedTrace, FailureType

            failure_type_map = {
                ErrorCategory.SYNTAX: FailureType.SYNTAX,
                ErrorCategory.LOGIC: FailureType.LOGIC,
                ErrorCategory.TRANSIENT: FailureType.TIMEOUT,
                ErrorCategory.MODEL_LIMIT: FailureType.RESOURCE,
                ErrorCategory.RESOURCE: FailureType.RESOURCE,
                ErrorCategory.CRITICAL: FailureType.SECURITY,
                ErrorCategory.UNKNOWN: FailureType.UNKNOWN,
            }

            reflection = None
            if task.id in self._reflection_history and self._reflection_history[task.id]:
                reflection = self._reflection_history[task.id][-1]

            trace = FailedTrace(
                task_type=task.type.name if hasattr(task.type, "name") else str(task.type),
                hypothesis=task.description[:500],
                error=str(error_ctx.error)[:500],
                error_type=failure_type_map.get(error_ctx.category, FailureType.UNKNOWN),
                context={
                    "provider": error_ctx.provider,
                    "model": error_ctx.model,
                    "attempts": error_ctx.attempt,
                },
                agent_thoughts=reflection.failure_analysis if reflection else None,
                task_id=task.id,
            )

            self._failure_store.record(trace)
            self._logger.info(f"Recorded failure trace for task {task.id}")

        except Exception as e:
            self._logger.warning(f"Failed to record to FailureStore: {e}")

    async def _record_successful_recovery(
        self,
        error_ctx: ErrorContext,
        task: Task,
        level: HealingLevel,
    ) -> None:
        """Record successful recovery as corrective action."""
        if not self._failure_store:
            return

        try:
            from gaap.meta_learning.failure_store import CorrectiveAction

            reflection = None
            if task.id in self._reflection_history and self._reflection_history[task.id]:
                reflection = self._reflection_history[task.id][-1]

            trace_id = None
            if hasattr(self._failure_store, "_failures"):
                for fid, f in self._failure_store._failures.items():
                    if f.task_id == task.id:
                        trace_id = fid
                        break

            if trace_id and reflection and reflection.proposed_fix:
                action = CorrectiveAction(
                    failure_id=trace_id,
                    solution=reflection.proposed_fix,
                    explanation=reflection.failure_analysis,
                    source="reflexion",
                )
                if trace_id not in self._failure_store._corrections:
                    self._failure_store._corrections[trace_id] = []
                self._failure_store._corrections[trace_id].append(action)
                self._failure_store._save()
                self._logger.info(f"Recorded corrective action for failure {trace_id}")

        except Exception as e:
            self._logger.warning(f"Failed to record successful recovery: {e}")

    # =========================================================================
    # Recording Methods
    # =========================================================================

    def _record_error(self, error_ctx: ErrorContext) -> None:
        """تسجيل الخطأ"""
        task_id = error_ctx.task_id

        if task_id not in self._error_history:
            self._error_history[task_id] = []

        self._error_history[task_id].append(error_ctx)

    def _record_healing(
        self,
        task_id: str,
        level: HealingLevel,
        action: RecoveryAction,
        success: bool,
        error_category: ErrorCategory,
        details: str = "",
    ) -> None:
        """تسجيل محاولة التعافي"""
        record = HealingRecord(
            task_id=task_id,
            level=level,
            action=action,
            success=success,
            duration_ms=0,
            error_category=error_category,
            details=details,
        )

        self._records.append(record)

    # =========================================================================
    # Pattern Detection
    # =========================================================================

    def _detect_failure_pattern(self, error_ctx: ErrorContext) -> str | None:
        """
        Detect if this error matches a repeated pattern.

        Returns:
            Pattern ID if pattern detected, None otherwise
        """
        if not self._config.pattern_detection.enabled:
            return None

        error_signature = self._compute_error_signature(error_ctx)

        if error_signature not in self._pattern_history:
            self._pattern_history[error_signature] = []

        now = time.time()
        window_seconds = self._config.pattern_detection.time_window_hours * 3600

        recent_occurrences = [
            occ
            for occ in self._pattern_history[error_signature]
            if now - occ["timestamp"] < window_seconds
        ]

        recent_occurrences.append(
            {
                "timestamp": now,
                "task_id": error_ctx.task_id,
                "error": str(error_ctx.error)[:200],
                "category": error_ctx.category.name,
            }
        )

        self._pattern_history[error_signature] = recent_occurrences

        if len(recent_occurrences) >= self._config.pattern_detection.detection_threshold:
            self._patterns_detected += 1
            self._logger.warning(
                f"Failure pattern detected: {error_signature} "
                f"({len(recent_occurrences)} occurrences)"
            )
            return error_signature

        return None

    def _compute_error_signature(self, error_ctx: ErrorContext) -> str:
        """
        Compute a unique signature for an error for pattern matching.

        Uses error type, category, and message patterns.
        """
        error_type = type(error_ctx.error).__name__
        category = error_ctx.category.name

        import hashlib

        message_normalized = re.sub(r"\d+", "N", str(error_ctx.error).lower()[:100])
        message_hash = hashlib.md5(message_normalized.encode()).hexdigest()[:8]

        return f"{error_type}:{category}:{message_hash}"

    def _should_auto_escalate(self, pattern_id: str) -> bool:
        """
        Check if pattern should trigger auto-escalation.

        Args:
            pattern_id: The detected pattern signature

        Returns:
            True if should auto-escalate
        """
        if not self._config.pattern_detection.auto_escalate_patterns:
            return False

        now = time.time()
        cooldown_seconds = self._config.pattern_detection.pattern_cooldown_minutes * 60

        if pattern_id in self._cooldown_patterns:
            if now - self._cooldown_patterns[pattern_id] < cooldown_seconds:
                return False

        self._cooldown_patterns[pattern_id] = now
        return True

    def _get_pattern_escalation_level(self, pattern_id: str) -> HealingLevel:
        """
        Determine the escalation level for a detected pattern.

        Patterns are escalated to a higher level than normal.
        """
        occurrences = len(self._pattern_history.get(pattern_id, []))

        if occurrences >= 5:
            return HealingLevel.L5_HUMAN_ESCALATION
        elif occurrences >= 4:
            return HealingLevel.L4_STRATEGY_SHIFT
        elif occurrences >= 3:
            return HealingLevel.L3_PIVOT
        else:
            return HealingLevel.L2_REFINE

    # =========================================================================
    # Dynamic Few-Shot Injection
    # =========================================================================

    def _inject_few_shot_examples(self, task: Task, context: dict[str, Any]) -> None:
        """Inject relevant examples from memory into context"""
        try:
            if hasattr(self._memory, "retrieve"):
                results = self._memory.retrieve(query=task.description, k=3)
                if results:
                    examples = []
                    for r in results[:3]:
                        if hasattr(r, "content"):
                            examples.append(r.content[:200])
                        elif isinstance(r, dict) and "content" in r:
                            examples.append(r["content"][:200])
                    if examples:
                        context["few_shot_examples"] = examples
                        self._logger.debug(f"Injected {len(examples)} few-shot examples")
            elif hasattr(self._memory, "search"):
                results = self._memory.search(task.description, n_results=3)
                if results:
                    context["few_shot_examples"] = [r[:200] for r in results[:3]]
                    self._logger.debug(f"Injected {len(results)} few-shot examples from search")
        except Exception as e:
            self._logger.debug(f"Failed to inject few-shot examples: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات التعافي"""
        return {
            "total_attempts": self._total_healing_attempts,
            "successful_recoveries": self._successful_recoveries,
            "escalations": self._escalations,
            "recovery_rate": (
                self._successful_recoveries / self._total_healing_attempts
                if self._total_healing_attempts > 0
                else 0
            ),
            "errors_by_category": self._get_errors_by_category(),
            "healing_by_level": self._get_healing_by_level(),
        }

    def _get_errors_by_category(self) -> dict[str, int]:
        """الأخطاء حسب التصنيف"""
        counts: dict[str, int] = {}
        for errors in self._error_history.values():
            for error in errors:
                cat = error.category.name
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _get_healing_by_level(self) -> dict[str, dict[str, int]]:
        """التعافي حسب المستوى"""
        stats: dict[str, dict[str, int]] = {}

        for record in self._records:
            level = record.level.name
            if level not in stats:
                stats[level] = {"success": 0, "failed": 0}

            if record.success:
                stats[level]["success"] += 1
            else:
                stats[level]["failed"] += 1

        return stats

    def get_error_history(self, task_id: str) -> list[ErrorContext]:
        """تاريخ أخطاء مهمة"""
        return self._error_history.get(task_id, [])
