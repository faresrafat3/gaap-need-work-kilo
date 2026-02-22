# Self Healer
import asyncio
import logging
import re
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

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
    - L2: تحسين الـ Prompt
    - L3: تغيير النموذج
    - L4: تغيير الاستراتيجية (تفكيك المهمة)
    - L5: تصعيد بشري
    """

    # حدود المستويات
    MAX_RETRIES_PER_LEVEL = {
        HealingLevel.L1_RETRY: 1,  # Was 3 — reduced: retrying timeouts 3× wastes 540s
        HealingLevel.L2_REFINE: 1,  # Was 2 — reduced: with only 1 provider, repeated retries just timeout
        HealingLevel.L3_PIVOT: 1,  # Was 2 — reduced
        HealingLevel.L4_STRATEGY_SHIFT: 1,
        HealingLevel.L5_HUMAN_ESCALATION: 0,
    }

    # تأخيرات المستويات
    LEVEL_DELAYS = {
        HealingLevel.L1_RETRY: 1.0,
        HealingLevel.L2_REFINE: 2.0,
        HealingLevel.L3_PIVOT: 3.0,
        HealingLevel.L4_STRATEGY_SHIFT: 5.0,
        HealingLevel.L5_HUMAN_ESCALATION: 0.0,
    }

    def __init__(
        self,
        max_level: HealingLevel = HealingLevel.L4_STRATEGY_SHIFT,
        on_escalate: Callable[[ErrorContext], Awaitable[None]] | None = None,
        memory: Any = None,
    ):
        self.max_level = max_level
        self.on_escalate = on_escalate
        self._logger = get_logger("gaap.healing")
        self._memory = memory

        self._records: list[HealingRecord] = []
        self._error_history: dict[str, list[ErrorContext]] = {}

        # إحصائيات
        self._total_healing_attempts = 0
        self._successful_recoveries = 0
        self._escalations = 0

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

        # إنشاء سياق الخطأ
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

        # تسجيل الخطأ
        self._record_error(error_ctx)

        # تحديد نقطة البداية
        current_level = self._determine_start_level(error_category)

        self._logger.info(
            f"Starting healing for task {task.id} "
            f"at level {current_level.name}, "
            f"error category: {error_category.name}"
        )

        # محاولة التعافي
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
                # v2: Observability
                try:
                    from gaap.core.observability import observability as _obs

                    _obs.record_healing(level=current_level.name, success=True)
                except Exception as e:
                    pass  # observability is never allowed to crash healing
                return result

            # الانتقال للمستوى التالي
            next_level = self._get_next_level(current_level)
            if next_level is None or next_level.value > self.max_level.value:
                break

            current_level = next_level

        # فشل التعافي
        self._escalations += 1
        result = RecoveryResult(
            success=False,
            action=RecoveryAction.ESCALATE,
            level=current_level,
            error="All healing levels exhausted",
            attempts=error_ctx.attempt,
            time_spent_ms=(time.time() - start_time) * 1000,
        )

        # التصعيد
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
        # v2: Observability
        try:
            from gaap.core.observability import observability as _obs

            _obs.record_healing(level=current_level.name, success=False)
        except Exception as e:
            pass  # observability is never allowed to crash healing

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

    async def _attempt_level(
        self,
        level: HealingLevel,
        error_ctx: ErrorContext,
        task: Task,
        execute_func: Callable[[Task], Awaitable[Any]],
        context: dict[str, Any] | None = None,
    ) -> RecoveryResult:
        """محاولة مستوى معين"""
        max_retries = self.MAX_RETRIES_PER_LEVEL.get(level, 1)
        delay = self.LEVEL_DELAYS.get(level, 1.0)

        for attempt in range(max_retries):
            error_ctx.attempt = attempt + 1

            self._logger.info(f"Attempting {level.name} (attempt {attempt + 1}/{max_retries})")

            # انتظار قبل المحاولة
            if attempt > 0 and delay > 0:
                await asyncio.sleep(delay * attempt)

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
            # إعادة المحاولة كما هي
            return RecoveryAction.RETRY, task

        elif level == HealingLevel.L2_REFINE:
            # تحسين الـ Prompt
            refined_description = PromptRefiner.refine(
                task.description, error_ctx.error, error_ctx.category
            )

            modified_task = Task(
                id=task.id,
                description=refined_description,
                type=task.type,
                priority=task.priority,
                complexity=task.complexity,
                context=task.context,
                constraints=task.constraints,
                metadata={**task.metadata, "refined": True},
            )

            return RecoveryAction.REFINE_PROMPT, modified_task

        elif level == HealingLevel.L3_PIVOT:
            # تغيير النموذج
            # (يتم التعامل معه خارجياً عبر context)
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
            # تبسيط المهمة
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
