import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    TypeVar,
)

from gaap.core.exceptions import (
    ModelNotFoundError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from gaap.core.types import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    Message,
    MessageRole,
    ModelTier,
    ProviderType,
    Usage,
)

T = TypeVar("T")


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
# Rate Limiter
# =============================================================================


@dataclass
class RateLimitState:
    """حالة حد الطلبات"""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    current_requests: int = 0
    current_tokens: int = 0
    window_start: datetime = field(default_factory=datetime.now)

    def is_allowed(self, tokens: int = 0) -> bool:
        """التحقق من السماح بالطلب"""
        now = datetime.now()
        # إعادة تعيين النافذة إذا مرت دقيقة
        if (now - self.window_start).total_seconds() >= 60:
            self.current_requests = 0
            self.current_tokens = 0
            self.window_start = now

        return (
            self.current_requests < self.requests_per_minute
            and self.current_tokens + tokens < self.tokens_per_minute
        )

    def record_request(self, tokens: int) -> None:
        """تسجيل طلب"""
        self.current_requests += 1
        self.current_tokens += tokens


class RateLimiter:
    """محدد معدل الطلبات"""

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 100000):
        self.state = RateLimitState(
            requests_per_minute=requests_per_minute, tokens_per_minute=tokens_per_minute
        )
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 0) -> bool:
        """محاولة الحصول على إذن"""
        async with self._lock:
            if self.state.is_allowed(tokens):
                self.state.record_request(tokens)
                return True
            return False

    async def wait_for_slot(self, tokens: int = 0, timeout: float = 30) -> bool:
        """الانتظار حتى يتوفر مكان"""
        start = time.time()
        while time.time() - start < timeout:
            if await self.acquire(tokens):
                return True
            await asyncio.sleep(0.5)
        return False


# =============================================================================
# Retry Manager
# =============================================================================


@dataclass
class RetryConfig:
    """تكوين إعادة المحاولة"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retry_on_status_codes: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


class RetryManager:
    """مدير إعادة المحاولة"""

    def __init__(self, config: RetryConfig):
        self.config = config
        self._attempt_counts: dict[str, int] = {}

    def get_delay(self, attempt: int) -> float:
        """حساب التأخير للمحاولة"""
        delay = self.config.base_delay * (self.config.exponential_base**attempt)
        return min(delay, self.config.max_delay)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """هل يجب إعادة المحاولة؟"""
        if attempt >= self.config.max_retries:
            return False

        # إعادة المحاولة لأخطاء الشبكة
        if isinstance(error, (asyncio.TimeoutError, ConnectionError)):
            return True

        # إعادة المحاولة لأخطاء HTTP محددة
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            return status_code in self.config.retry_on_status_codes

        return isinstance(error, ProviderRateLimitError)

    async def execute_with_retry(self, func: Callable[[], Awaitable[T]], key: str = "default") -> T:
        """تنفيذ مع إعادة المحاولة"""
        attempt = 0
        last_error: Exception | None = None

        while attempt < self.config.max_retries:
            try:
                result = await func()
                # إعادة تعيين العداد عند النجاح
                self._attempt_counts[key] = 0
                return result
            except Exception as e:
                last_error = e
                attempt += 1
                self._attempt_counts[key] = attempt

                if not self.should_retry(attempt, e):
                    raise

                delay = self.get_delay(attempt)
                await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error


# =============================================================================
# Usage Tracker
# =============================================================================


@dataclass
class UsageRecord:
    """سجل استخدام"""

    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    error: str | None = None


class UsageTracker:
    """متتبع الاستخدام"""

    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self._records: list[UsageRecord] = []
        self._totals: dict[str, dict[str, float]] = {}

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> UsageRecord:
        """تسجيل استخدام"""
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

        self._records.append(record)

        # تقليم السجلات القديمة
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

        # تحديث الإجماليات
        if provider not in self._totals:
            self._totals[provider] = {
                "total_tokens": 0,
                "total_cost": 0,
                "total_requests": 0,
                "failed_requests": 0,
            }

        self._totals[provider]["total_tokens"] += record.total_tokens
        self._totals[provider]["total_cost"] += cost_usd
        self._totals[provider]["total_requests"] += 1
        if not success:
            self._totals[provider]["failed_requests"] += 1

        return record

    def get_totals(self, provider: str | None = None) -> dict[str, Any]:
        """الحصول على الإجماليات"""
        if provider:
            return self._totals.get(provider, {})
        return self._totals.copy()

    def get_recent_records(self, limit: int = 100) -> list[UsageRecord]:
        """الحصول على السجلات الأخيرة"""
        return self._records[-limit:]


# =============================================================================
# Base Provider
# =============================================================================


class BaseProvider(ABC):
    """
    الفئة الأساسية لجميع مزودي النماذج

    توفر:
    - واجهة موحدة للاتصال بالنماذج
    - إدارة المهلات وإعادة المحاولة
    - تتبع الاستخدام والتكلفة
    - تحديد معدل الطلبات
    """

    def __init__(
        self,
        name: str,
        provider_type: ProviderType,
        models: list[str],
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limit_rpm: int = 60,
        rate_limit_tpm: int = 100000,
        timeout: float = 120.0,
        max_retries: int = 3,
        default_model: str | None = None,
    ):
        self.name = name
        self.provider_type = provider_type
        self.models = models
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_model = default_model or (models[0] if models else "")

        # المكونات الداخلية
        self._logger = get_logger(f"gaap.provider.{name}")
        self._rate_limiter = RateLimiter(rate_limit_rpm, rate_limit_tpm)
        self._retry_manager = RetryManager(RetryConfig(max_retries=max_retries))
        self._usage_tracker = UsageTracker()
        self._is_initialized = False

        # إحصائيات
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def _make_request(self, messages: list[Message], model: str, **kwargs) -> dict[str, Any]:
        """
        تنفيذ الطلب الفعلي للمزود

        يجب تنفيذها في الفئات الفرعية
        """
        pass

    @abstractmethod
    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        تنفيذ الطلب المتدفق

        يجب تنفيذها في الفئات الفرعية
        """
        yield ""  # pragma: no cover

    @abstractmethod
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """حساب تكلفة الطلب"""
        pass

    # =========================================================================
    # Public Methods
    # =========================================================================

    async def chat_completion(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> ChatCompletionResponse:
        """
        إكمال محادثة

        Args:
            messages: قائمة الرسائل
            model: اسم النموذج (اختياري، يستخدم الافتراضي)
            temperature: درجة الحرارة
            max_tokens: الحد الأقصى للرموز
            top_p: معامل top_p
            stop: قائمة كلمات التوقف

        Returns:
            استجابة الإكمال
        """
        model = model or self.default_model

        # التحقق من توفر النموذج
        if not self.is_model_available(model):
            raise ModelNotFoundError(
                model_name=model, provider_name=self.name, available_models=self.models
            )

        start_time = time.time()

        try:
            # الانتظار لمحدد الطلبات
            estimated_tokens = sum(len(m.content.split()) * 2 for m in messages) + max_tokens
            if not await self._rate_limiter.wait_for_slot(estimated_tokens):
                raise ProviderRateLimitError(provider_name=self.name, retry_after=60)

            # تنفيذ الطلب مع إعادة المحاولة
            async with self._timeout_context():
                response_data = await self._retry_manager.execute_with_retry(
                    lambda: self._make_request(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop,
                        **kwargs,
                    )
                )

            # تحليل الاستجابة
            result = self._parse_response(response_data, model)

            # حساب التكلفة
            if result.usage:
                cost = self._calculate_cost(
                    model=model,
                    input_tokens=result.usage.prompt_tokens,
                    output_tokens=result.usage.completion_tokens,
                )
            else:
                cost = 0.0

            # تسجيل الاستخدام
            latency_ms = (time.time() - start_time) * 1000
            self._record_usage(
                model=model,
                input_tokens=result.usage.prompt_tokens if result.usage else 0,
                output_tokens=result.usage.completion_tokens if result.usage else 0,
                cost=cost,
                latency_ms=latency_ms,
                success=True,
            )

            result.provider = self.name
            result.latency_ms = latency_ms
            result.metadata["cost_usd"] = cost

            return result

        except asyncio.TimeoutError:
            raise ProviderTimeoutError(provider_name=self.name, timeout_seconds=self.timeout)
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_usage(
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )
            raise

    async def stream_chat_completion(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        إكمال محادثة بتدفق

        يُرجع المحتوى تدريجياً
        """
        model = model or self.default_model

        if not self.is_model_available(model):
            raise ModelNotFoundError(
                model_name=model, provider_name=self.name, available_models=self.models
            )

        try:
            stream = self._stream_request(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            async for chunk in stream:  # type: ignore[misc]
                yield chunk
        except Exception as e:
            self._logger.error(f"Stream error: {e}")
            raise

    def is_model_available(self, model: str) -> bool:
        """التحقق من توفر النموذج"""
        return model in self.models

    def get_available_models(self) -> list[str]:
        """الحصول على قائمة النماذج المتاحة"""
        return self.models.copy()

    def get_model_tier(self, model: str) -> ModelTier:
        """الحصول على مستوى النموذج"""
        # يمكن تجاوز هذه الدالة في الفئات الفرعية
        model_lower = model.lower()

        if any(x in model_lower for x in ["opus", "o1", "gpt-4-turbo"]):
            return ModelTier.TIER_1_STRATEGIC
        elif any(x in model_lower for x in ["sonnet", "gpt-4o", "claude-3-5"]):
            return ModelTier.TIER_2_TACTICAL
        elif any(x in model_lower for x in ["haiku", "mini", "flash"]):
            return ModelTier.TIER_3_EFFICIENT
        elif any(x in model_lower for x in ["llama", "mistral", "local"]):
            return ModelTier.TIER_4_PRIVATE

        return ModelTier.TIER_2_TACTICAL  # افتراضي

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @asynccontextmanager
    async def _timeout_context(self):
        """سياق مع مهلة زمنية"""
        try:
            async with asyncio.timeout(self.timeout):
                yield
        except asyncio.TimeoutError:
            raise ProviderTimeoutError(provider_name=self.name, timeout_seconds=self.timeout)

    def _parse_response(self, response_data: dict[str, Any], model: str) -> ChatCompletionResponse:
        """تحليل استجابة المزود"""
        # التنفيذ الافتراضي، يمكن تجاوزه
        choices = []
        for i, choice_data in enumerate(response_data.get("choices", [])):
            message_data = choice_data.get("message", {})
            message = Message(
                role=MessageRole(message_data.get("role", "assistant")),
                content=message_data.get("content", ""),
            )
            choices.append(
                ChatCompletionChoice(
                    index=i, message=message, finish_reason=choice_data.get("finish_reason", "stop")
                )
            )

        usage_data = response_data.get("usage", {})
        usage = (
            Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            if usage_data
            else None
        )

        return ChatCompletionResponse(
            id=response_data.get("id", ""), model=model, choices=choices, usage=usage
        )

    def _record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """تسجيل الاستخدام"""
        self._total_requests += 1
        if success:
            self._successful_requests += 1
            self._total_tokens += input_tokens + output_tokens
            self._total_cost += cost
        else:
            self._failed_requests += 1

        self._usage_tracker.record(
            provider=self.name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

    def get_stats(self) -> dict[str, Any]:
        """الحصول على إحصائيات المزود"""
        return {
            "name": self.name,
            "type": self.provider_type.name,
            "models": self.models,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / self._total_requests if self._total_requests > 0 else 0
            ),
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "usage_by_model": self._usage_tracker.get_totals(),
        }

    def initialize(self) -> None:
        """تهيئة المزود"""
        self._is_initialized = True
        self._logger.info(f"Provider {self.name} initialized")

    def shutdown(self) -> None:
        """إيقاف المزود"""
        self._is_initialized = False
        self._logger.info(f"Provider {self.name} shut down")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, models={len(self.models)})>"


# =============================================================================
# Provider Factory
# =============================================================================


class ProviderFactory:
    """مصنع المزودين"""

    _providers: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, provider_class: type) -> None:
        """تسجيل مزود جديد"""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(
        cls, name: str, api_key: str | None = None, base_url: str | None = None, **kwargs
    ) -> BaseProvider:
        """إنشاء مزود"""
        provider_class = cls._providers.get(name.lower())
        if not provider_class:
            raise ProviderNotFoundError(
                provider_name=name, available_providers=list(cls._providers.keys())
            )

        return provider_class(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """قائمة المزودين المسجلين"""
        return list(cls._providers.keys())


# =============================================================================
# Decorator for Provider Registration
# =============================================================================


def register_provider(name: str):
    """مُزخرف لتسجيل مزود"""

    def decorator(cls):
        ProviderFactory.register(name, cls)
        return cls

    return decorator
