# Fallback Manager
import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from gaap.core.exceptions import (
    MaxRetriesExceededError,
)
from gaap.core.types import (
    ChatCompletionResponse,
    Message,
    ProviderType,
    Task,
)
from gaap.providers.base_provider import BaseProvider
from gaap.routing.router import SmartRouter

# =============================================================================
# Logger Setup
# =============================================================================


def get_logger(name: str) -> logging.Logger:
    """إنشاء مسجل"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# Fallback Configuration
# =============================================================================


@dataclass
class FallbackConfig:
    """تكوين البدائل"""

    max_fallbacks: int = 3  # الحد الأقصى للبدائل
    retry_delay_base: float = 1.0  # تأخير أساسي
    retry_delay_max: float = 30.0  # تأخير أقصى
    exponential_backoff: bool = True  # تأخير أسي
    jitter: bool = True  # إضافة عشوائية
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5  # عدد الفشل لفتح الدائرة
    circuit_breaker_timeout: int = 60  # ثواني قبل المحاولة مجدداً


@dataclass
class ProviderHealth:
    """صحة المزود"""

    name: str
    is_healthy: bool = True
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure: datetime | None = None
    last_success: datetime | None = None
    last_check: datetime | None = None
    circuit_open: bool = False
    circuit_open_since: datetime | None = None

    def record_success(self) -> None:
        """تسجيل نجاح"""
        self.is_healthy = True
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.last_success = datetime.now()
        self.circuit_open = False
        self.circuit_open_since = None

    def record_failure(self) -> None:
        """تسجيل فشل"""
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_failure = datetime.now()
        self.is_healthy = False

    def open_circuit(self) -> None:
        """فتح الدائرة (إيقاف الطلبات)"""
        self.circuit_open = True
        self.circuit_open_since = datetime.now()

    def should_try_circuit(self, timeout_seconds: int) -> bool:
        """هل يجب محاولة اختبار الدائرة؟"""
        if not self.circuit_open or self.circuit_open_since is None:
            return True

        elapsed = (datetime.now() - self.circuit_open_since).total_seconds()
        return elapsed >= timeout_seconds


# =============================================================================
# Fallback Chain
# =============================================================================


@dataclass
class FallbackChain:
    """سلسلة البدائل"""

    primary: str  # المزود الرئيسي
    fallbacks: list[str] = field(default_factory=list)  # البدائل
    current_index: int = 0

    def get_next(self) -> str | None:
        """الحصول على المزود التالي"""
        if self.current_index == 0:
            self.current_index += 1
            return self.primary

        if self.current_index <= len(self.fallbacks):
            provider = self.fallbacks[self.current_index - 1]
            self.current_index += 1
            return provider

        return None

    def reset(self) -> None:
        """إعادة تعيين"""
        self.current_index = 0

    def is_exhausted(self) -> bool:
        """هل استنفدت الخيارات؟"""
        return self.current_index > len(self.fallbacks)


# =============================================================================
# Fallback Manager
# =============================================================================


class FallbackManager:
    """
    مدير البدائل - يدير التبديل بين المزودين عند الفشل

    الميزات:
    - سلاسل بدائل قابلة للتخصيص
    - Circuit Breaker لحماية المزودين
    - تأخير أسي مع عشوائية
    - تتبع صحة المزودين
    """

    def __init__(self, router: SmartRouter, config: FallbackConfig | None = None):
        self._router = router
        self._config = config or FallbackConfig()
        self._logger = get_logger("gaap.fallback")

        # حالة المزودين
        self._health: dict[str, ProviderHealth] = {}
        self._chains: dict[str, FallbackChain] = {}

        # إحصائيات
        self._fallback_events: list[dict[str, Any]] = []
        self._total_fallbacks = 0
        self._successful_recoveries = 0

    # =========================================================================
    # Fallback Chain Management
    # =========================================================================

    def define_fallback_chain(self, primary: str, fallbacks: list[str]) -> None:
        """تحديد سلسلة بدائل"""
        self._chains[primary] = FallbackChain(primary=primary, fallbacks=fallbacks)
        self._logger.info(f"Defined fallback chain: {primary} -> {' -> '.join(fallbacks)}")

    def get_default_fallbacks(self, provider_type: ProviderType) -> list[str]:
        """الحصول على البدائل الافتراضية"""
        default_chains = {
            ProviderType.CHAT_BASED: ["groq", "gemini"],
            ProviderType.FREE_TIER: ["g4f", "groq"],
            ProviderType.PAID: ["groq", "gemini", "g4f"],
            ProviderType.LOCAL: ["groq", "g4f"],
        }
        return default_chains.get(provider_type, ["groq", "gemini"])

    # =========================================================================
    # Execution with Fallback
    # =========================================================================

    async def execute_with_fallback(
        self,
        messages: list[Message],
        primary_provider: str,
        primary_model: str,
        task: Task | None = None,
        **kwargs,
    ) -> ChatCompletionResponse:
        """
        تنفيذ مع بدائل تلقائية

        Args:
            messages: الرسائل
            primary_provider: المزود الرئيسي
            primary_model: النموذج الرئيسي
            task: المهمة (اختياري)

        Returns:
            استجابة الإكمال
        """
        # بناء سلسلة البدائل
        chain = self._chains.get(primary_provider)
        if chain is None:
            fallbacks = self.get_default_fallbacks(self._get_provider_type(primary_provider))
            chain = FallbackChain(
                primary=primary_provider, fallbacks=fallbacks[: self._config.max_fallbacks]
            )
        else:
            chain.reset()

        errors: list[Exception] = []
        attempts: list[dict[str, Any]] = []

        while not chain.is_exhausted():
            provider_name = chain.get_next()
            if provider_name is None:
                break

            # التحقق من صحة المزود
            if not self._is_provider_available(provider_name):
                self._logger.warning(f"Provider {provider_name} is not available, skipping")
                continue

            # الحصول على المزود
            provider = self._router.get_provider(provider_name)
            if provider is None:
                continue

            # تحديد النموذج
            model = self._select_model(provider, primary_model, chain.current_index)

            # حساب التأخير
            delay = self._calculate_delay(chain.current_index)
            if delay > 0:
                await asyncio.sleep(delay)

            # محاولة التنفيذ
            attempt_start = time.time()

            try:
                self._logger.info(
                    f"Attempting {provider_name}/{model} " f"(attempt {chain.current_index})"
                )

                response = await provider.chat_completion(messages=messages, model=model, **kwargs)

                # نجاح!
                self._record_success(provider_name)

                if chain.current_index > 1:
                    self._successful_recoveries += 1
                    self._log_fallback(
                        primary=primary_provider, fallback=provider_name, success=True
                    )

                return response

            except Exception as e:
                errors.append(e)
                latency = (time.time() - attempt_start) * 1000

                self._record_failure(provider_name)

                attempts.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "error": str(e),
                        "latency_ms": latency,
                    }
                )

                self._logger.warning(f"Provider {provider_name} failed: {e}")

                # التحقق من Circuit Breaker
                self._check_circuit_breaker(provider_name)

        # فشل جميع المحاولات
        self._total_fallbacks += 1
        self._log_fallback(
            primary=primary_provider, fallback="none", success=False, attempts=attempts
        )

        raise MaxRetriesExceededError(
            task_id=task.id if task else "unknown",
            max_retries=len(attempts),
            last_error=str(errors[-1]) if errors else "Unknown error",
        )

    def _select_model(self, provider: BaseProvider, primary_model: str, attempt: int) -> str:
        """اختيار نموذج مناسب"""
        # محاولة استخدام نفس النموذج
        if provider.is_model_available(primary_model):
            return primary_model

        # اختيار نموذج مشابه
        available = provider.get_available_models()
        if available:
            # محاولة العثور على نموذج مشابه
            for model in available:
                if any(x in model.lower() for x in ["llama", "mixtral", "gpt", "claude"]):
                    return model

            return available[0]

        return provider.default_model

    def _calculate_delay(self, attempt: int) -> float:
        """حساب التأخير"""
        if attempt <= 1:
            return 0

        if self._config.exponential_backoff:
            delay = self._config.retry_delay_base * (2 ** (attempt - 2))
        else:
            delay = self._config.retry_delay_base

        delay = min(delay, self._config.retry_delay_max)

        if self._config.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def _is_provider_available(self, provider_name: str) -> bool:
        """التحقق من توفر المزود"""
        health = self._health.get(provider_name)

        if health is None:
            return True

        # Don't circuit-break the only registered provider —
        # there's nothing to fall back to, so keep trying
        if len(self._router.get_all_providers()) <= 1:
            return True

        if health.circuit_open:
            return health.should_try_circuit(self._config.circuit_breaker_timeout)

        return (
            health.is_healthy
            or health.consecutive_failures < self._config.circuit_breaker_threshold
        )

    def _get_provider_type(self, provider_name: str) -> ProviderType:
        """الحصول على نوع المزود"""
        provider = self._router.get_provider(provider_name)
        return provider.provider_type if provider else ProviderType.FREE_TIER

    # =========================================================================
    # Health Tracking
    # =========================================================================

    def _record_success(self, provider_name: str) -> None:
        """تسجيل نجاح"""
        if provider_name not in self._health:
            self._health[provider_name] = ProviderHealth(name=provider_name)

        self._health[provider_name].record_success()

    def _record_failure(self, provider_name: str) -> None:
        """تسجيل فشل"""
        if provider_name not in self._health:
            self._health[provider_name] = ProviderHealth(name=provider_name)

        self._health[provider_name].record_failure()

    def _check_circuit_breaker(self, provider_name: str) -> None:
        """التحقق من Circuit Breaker"""
        health = self._health.get(provider_name)
        if health is None:
            return

        if health.consecutive_failures >= self._config.circuit_breaker_threshold:
            health.open_circuit()
            self._logger.warning(
                f"Circuit breaker opened for {provider_name} "
                f"after {health.consecutive_failures} failures"
            )

    # =========================================================================
    # Logging & Statistics
    # =========================================================================

    def _log_fallback(
        self, primary: str, fallback: str, success: bool, attempts: list[dict] | None = None
    ) -> None:
        """تسجيل حدث البديل"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "primary_provider": primary,
            "fallback_provider": fallback,
            "success": success,
            "attempts": attempts or [],
        }

        self._fallback_events.append(event)

        # تقليم السجلات
        if len(self._fallback_events) > 1000:
            self._fallback_events = self._fallback_events[-1000:]

    def get_health_status(self) -> dict[str, Any]:
        """الحصول على حالة الصحة"""
        return {
            name: {
                "is_healthy": health.is_healthy,
                "consecutive_failures": health.consecutive_failures,
                "circuit_open": health.circuit_open,
                "last_failure": health.last_failure.isoformat() if health.last_failure else None,
                "last_success": health.last_success.isoformat() if health.last_success else None,
            }
            for name, health in self._health.items()
        }

    def get_fallback_stats(self) -> dict[str, Any]:
        """إحصائيات البدائل"""
        return {
            "total_fallbacks": self._total_fallbacks,
            "successful_recoveries": self._successful_recoveries,
            "recovery_rate": (
                self._successful_recoveries / self._total_fallbacks
                if self._total_fallbacks > 0
                else 0
            ),
            "recent_events": self._fallback_events[-10:],
        }

    def reset_health(self, provider_name: str | None = None) -> None:
        """إعادة تعيين حالة الصحة"""
        if provider_name:
            if provider_name in self._health:
                self._health[provider_name] = ProviderHealth(name=provider_name)
        else:
            self._health.clear()


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """
    Circuit Breaker - يحمي من الطلبات المتكررة لمزود فاشل

    الحالات:
    - CLOSED: يعمل بشكل طبيعي
    - OPEN: يرفض الطلبات
    - HALF_OPEN: يسمح بطلب اختباري
    """

    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, success_threshold: int = 3, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout

        self._state = self.State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None

    def can_execute(self) -> bool:
        """هل يمكن تنفيذ طلب؟"""
        if self._state == self.State.CLOSED:
            return True

        if self._state == self.State.OPEN:
            if self._last_failure_time is None:
                return False

            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self.timeout:
                self._state = self.State.HALF_OPEN
                self._success_count = 0
                return True
            return False

        # HALF_OPEN
        return True

    def record_success(self) -> None:
        """تسجيل نجاح"""
        if self._state == self.State.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = self.State.CLOSED
                self._failure_count = 0
        elif self._state == self.State.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """تسجيل فشل"""
        self._last_failure_time = datetime.now()

        if self._state == self.State.HALF_OPEN:
            self._state = self.State.OPEN
        elif self._state == self.State.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = self.State.OPEN

    @property
    def state(self) -> State:
        """الحالة الحالية"""
        return self._state


# =============================================================================
# Convenience Functions
# =============================================================================


def create_fallback_manager(
    router: SmartRouter, max_fallbacks: int = 3, circuit_breaker_threshold: int = 5
) -> FallbackManager:
    """إنشاء مدير بدائل بسهولة"""
    config = FallbackConfig(
        max_fallbacks=max_fallbacks, circuit_breaker_threshold=circuit_breaker_threshold
    )
    return FallbackManager(router=router, config=config)
