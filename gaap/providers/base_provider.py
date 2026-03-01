"""
Base Provider Module for GAAP System

Provides foundational base classes for all LLM providers:

Classes:
    - RateLimitState: Rate limit tracking
    - RateLimiter: Async rate limiting
    - RetryConfig: Retry configuration
    - RetryManager: Exponential backoff retry logic
    - BaseProvider: Abstract base class for all providers

Usage:
    from gaap.providers import BaseProvider, RateLimiter, RetryManager

    class MyProvider(BaseProvider):
        async def _make_request(self, messages, model, **kwargs):
            # Implementation here
            pass

Features:
    - Rate limiting with token bucket
    - Exponential backoff retry
    - Streaming support
    - Cost calculation
    - Model tier tracking
"""

import asyncio
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

# =============================================================================
# Constants
# =============================================================================

DEFAULT_REQUESTS_PER_MINUTE = 60
DEFAULT_TOKENS_PER_MINUTE = 100000
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_TIMEOUT = 30.0

T = TypeVar("T")


# =============================================================================
# Logger Setup
# =============================================================================


from gaap.core.logging import get_standard_logger as get_logger


@dataclass
class RateLimitState:
    """
    Rate limit state tracking.

    Tracks request counts and token usage within a time window.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute
        tokens_per_minute: Maximum tokens allowed per minute
        current_requests: Current request count in window
        current_tokens: Current token count in window
        window_start: Start time of current window

    Usage:
        >>> state = RateLimitState(requests_per_minute=60)
        >>> if state.is_allowed(tokens=100):
        ...     state.record_request(tokens=100)
    """

    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE
    tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE
    current_requests: int = 0
    current_tokens: int = 0
    window_start: datetime = field(default_factory=datetime.now)

    def is_allowed(self, tokens: int = 0) -> bool:
        """
        Check if request is allowed under rate limits.

        Args:
            tokens: Number of tokens for this request

        Returns:
            True if request is allowed, False otherwise

        Note:
            Automatically resets counters if window has expired
        """
        now = datetime.now()
        # Reset window if minute has passed
        if (now - self.window_start).total_seconds() >= 60:
            self.current_requests = 0
            self.current_tokens = 0
            self.window_start = now

        return (
            self.current_requests < self.requests_per_minute
            and self.current_tokens + tokens < self.tokens_per_minute
        )

    def record_request(self, tokens: int) -> None:
        """
        Record a request against rate limit.

        Args:
            tokens: Number of tokens used in request

        Note:
            Call this after making a successful request
        """
        self.current_requests += 1
        self.current_tokens += tokens


class RateLimiter:
    """
    Async rate limiter with token bucket algorithm.

    Provides rate limiting for API calls with:
    - Configurable requests per minute
    - Configurable tokens per minute
    - Async-safe with locking
    - Wait for slot capability

    Attributes:
        state: Current rate limit state
        _lock: Async lock for thread safety

    Usage:
        >>> limiter = RateLimiter(requests_per_minute=60)
        >>> if await limiter.acquire():
        ...     await make_api_call()
    """

    def __init__(
        self,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute (default: 60)
            tokens_per_minute: Max tokens per minute (default: 100000)
        """
        self.state = RateLimitState(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        )
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 0) -> bool:
        """
        Try to acquire a rate limit slot.

        Args:
            tokens: Number of tokens for this request

        Returns:
            True if slot acquired, False if rate limited

        Example:
            >>> if await limiter.acquire(tokens=100):
            ...     # Make API call
            ...     pass
        """
        async with self._lock:
            if self.state.is_allowed(tokens):
                self.state.record_request(tokens)
                return True
            return False

    async def wait_for_slot(self, tokens: int = 0, timeout: float = DEFAULT_TIMEOUT) -> bool:
        """
        Wait until a rate limit slot is available.

        Args:
            tokens: Number of tokens for this request
            timeout: Maximum time to wait in seconds (default: 30)

        Returns:
            True if slot acquired within timeout, False otherwise

        Example:
            >>> if await limiter.wait_for_slot(tokens=100, timeout=60):
            ...     # Make API call
            ...     pass
        """
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
    """
    Retry configuration for exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        retry_on_status_codes: HTTP status codes to retry on

    Usage:
        >>> config = RetryConfig(max_retries=5, base_delay=0.5)
        >>> manager = RetryManager(config)
    """

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retry_on_status_codes: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


class RetryManager:
    """
    Exponential backoff retry manager.

    Provides intelligent retry logic with:
    - Exponential backoff delays
    - Configurable max retries
    - Status code-based retry decisions
    - Network error detection

    Attributes:
        config: Retry configuration
        _attempt_counts: Per-key attempt counters

    Usage:
        >>> config = RetryConfig(max_retries=3)
        >>> manager = RetryManager(config)
        >>> result = await manager.execute_with_retry(my_async_func)
    """

    def __init__(self, config: RetryConfig) -> None:
        """
        Initialize retry manager.

        Args:
            config: Retry configuration settings
        """
        self.config = config
        self._attempt_counts: dict[str, int] = {}

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt using exponential backoff.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds, capped at max_delay

        Example:
            >>> manager.get_delay(0)  # First retry
            1.0
            >>> manager.get_delay(2)  # Third retry
            4.0
        """
        delay = self.config.base_delay * (self.config.exponential_base**attempt)
        return min(delay, self.config.max_delay)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """
        Determine if retry should be attempted based on error type.

        Args:
            attempt: Current attempt number
            error: Exception that occurred

        Returns:
            True if retry should be attempted, False otherwise

        Retry Conditions:
            - attempt < max_retries
            - Network errors (TimeoutError, ConnectionError)
            - Rate limit errors (ProviderRateLimitError)
            - Server errors (HTTP 500, 502, 503, 504)
        """
        if attempt >= self.config.max_retries:
            return False

        # Retry for network errors
        if isinstance(error, (asyncio.TimeoutError, ConnectionError)):
            return True

        # Retry for specific HTTP status codes
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            return status_code in self.config.retry_on_status_codes

        # Retry for rate limits
        return isinstance(error, ProviderRateLimitError)

    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        key: str = "default",
    ) -> T:
        """
        Execute async function with automatic retry on failure.

        Args:
            func: Async function to execute
            key: Unique key for tracking attempts

        Returns:
            Result from successful function execution

        Raises:
            Exception: Last exception if all retries exhausted

        Example:
            >>> result = await manager.execute_with_retry(
            ...     lambda: api_call(),
            ...     key="kimi_request"
            ... )
        """
        attempt = 0
        last_error: Exception | None = None

        while attempt < self.config.max_retries:
            try:
                result = await func()
                # Reset counter on success
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

        if last_error is None:
            raise RuntimeError("Retry logic failed: no error recorded")
        raise last_error


# =============================================================================
# Usage Tracker
# =============================================================================


@dataclass
class UsageRecord:
    """
    Usage record for tracking API calls.

    Attributes:
        timestamp: Time of the request
        provider: Provider name (e.g., "kimi", "gemini")
        model: Model used (e.g., "llama-3.3-70b")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens (input + output)
        cost_usd: Cost in USD
        latency_ms: Request latency in milliseconds
        success: Whether request succeeded
        error: Error message if failed

    Usage:
        >>> record = UsageRecord(
        ...     timestamp=datetime.now(),
        ...     provider="kimi",
        ...     model="llama-3.3-70b",
        ...     input_tokens=100,
        ...     output_tokens=50,
        ...     cost_usd=0.001,
        ...     latency_ms=250.0,
        ...     success=True
        ... )
    """

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
    """
    Tracks API usage across all providers.

    Features:
    - Per-provider usage tracking
    - Token and cost aggregation
    - Recent record history
    - Automatic trimming of old records

    Attributes:
        max_records: Maximum records to keep (default: 10000)
        _records: List of usage records
        _totals: Per-provider totals

    Usage:
        >>> tracker = UsageTracker(max_records=5000)
        >>> tracker.record(
        ...     provider="kimi",
        ...     model="llama-3.3-70b",
        ...     input_tokens=100,
        ...     output_tokens=50,
        ...     cost_usd=0.001,
        ...     latency_ms=250.0,
        ...     success=True
        ... )
    """

    def __init__(self, max_records: int = 10000) -> None:
        """
        Initialize usage tracker.

        Args:
            max_records: Maximum number of records to retain
        """
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
        """
        Record an API usage event.

        Args:
            provider: Provider name
            model: Model used
            input_tokens: Input token count
            output_tokens: Output token count
            cost_usd: Cost in USD
            latency_ms: Latency in milliseconds
            success: Success status
            error: Error message if failed

        Returns:
            Created UsageRecord instance

        Note:
            Automatically trims old records if exceeding max_records
        """
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

        # Trim old records
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

        # Update totals
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
        """
        Get usage totals.

        Args:
            provider: Optional provider name (returns all if None)

        Returns:
            Dictionary with totals for specified provider or all

        Example:
            >>> totals = tracker.get_totals("kimi")
            >>> print(totals["total_cost"])
            0.05
        """
        if provider:
            return self._totals.get(provider, {})
        return self._totals.copy()

    def get_recent_records(self, limit: int = 100) -> list[UsageRecord]:
        """
        Get most recent usage records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent UsageRecord objects

        Example:
            >>> records = tracker.get_recent_records(limit=10)
            >>> for record in records:
            ...     print(f"{record.provider}: {record.cost_usd}")
        """
        return self._records[-limit:]


# =============================================================================
# Base Provider
# =============================================================================


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Provides unified interface for LLM interactions with:
    - Rate limiting
    - Automatic retry with exponential backoff
    - Usage tracking and cost calculation
    - Timeout management
    - Streaming support

    Attributes:
        name: Provider name
        provider_type: Type of provider (FREE_TIER, PAID, etc.)
        models: List of available models
        api_key: API key for authentication
        base_url: Base URL for API
        timeout: Request timeout in seconds
        default_model: Default model to use

    Usage:
        >>> class MyProvider(BaseProvider):
        ...     async def _make_request(self, messages, model, **kwargs):
        ...         # Implementation
        ...         pass
        ...
        >>> provider = MyProvider(name="my_provider", ...)
        >>> response = await provider.chat_completion(messages)
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
    ) -> None:
        """
        Initialize base provider.

        Args:
            name: Provider name
            provider_type: Type of provider
            models: List of available model names
            api_key: Optional API key
            base_url: Optional custom base URL
            rate_limit_rpm: Requests per minute limit
            rate_limit_tpm: Tokens per minute limit
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            default_model: Default model (uses first if not specified)
        """
        self.name = name
        self.provider_type = provider_type
        self.models = models
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_model = default_model or (models[0] if models else "")

        # Internal components
        self._logger = get_logger(f"gaap.provider.{name}")
        self._rate_limiter = RateLimiter(rate_limit_rpm, rate_limit_tpm)
        self._retry_manager = RetryManager(RetryConfig(max_retries=max_retries))
        self._usage_tracker = UsageTracker()
        self._is_initialized = False

        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def _make_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Make actual request to provider.

        Must be implemented by subclasses.

        Args:
            messages: List of conversation messages
            model: Model name to use
            **kwargs: Additional provider-specific parameters

        Returns:
            Raw response data from provider

        Example:
            >>> response_data = await self._make_request(messages, model)
        """

    @abstractmethod
    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Make streaming request to provider.

        Must be implemented by subclasses.

        Args:
            messages: List of conversation messages
            model: Model name to use
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of response content as they arrive
        """
        yield ""  # pragma: no cover

    @abstractmethod
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for a request.

        Must be implemented by subclasses.

        Args:
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """

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
        **kwargs: Any,
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

            # v2: Observability — record per-call metrics
            try:
                from gaap.core.observability import observability as _obs

                _obs.record_llm_call(
                    provider=self.name,
                    model=model,
                    input_tokens=result.usage.prompt_tokens if result.usage else 0,
                    output_tokens=result.usage.completion_tokens if result.usage else 0,
                    cost=cost,
                    latency=latency_ms / 1000,
                    success=True,
                )
            except Exception as e:
                pass  # observability is never allowed to crash the main flow is never allowed to crash the main flow

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
            # v2: Observability — record failure
            try:
                from gaap.core.observability import observability as _obs

                _obs.record_llm_call(
                    provider=self.name,
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    latency=latency_ms / 1000,
                    success=False,
                )
            except Exception:
                pass  # observability is never allowed to crash the main flow
            raise

    async def stream_chat_completion(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
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
            async for chunk in stream:
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
    async def _timeout_context(self) -> AsyncGenerator[None, None]:
        """سياق مع مهلة زمنية"""
        try:
            async with asyncio.timeout(self.timeout):  # type: ignore[attr-defined]
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
        cls, name: str, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> BaseProvider:
        """إنشاء مزود"""
        provider_class = cls._providers.get(name.lower())
        if not provider_class:
            raise ProviderNotFoundError(
                provider_name=name, available_providers=list(cls._providers.keys())
            )

        provider: BaseProvider = provider_class(api_key=api_key, base_url=base_url, **kwargs)
        return provider

    @classmethod
    def list_providers(cls) -> list[str]:
        """قائمة المزودين المسجلين"""
        return list(cls._providers.keys())


# =============================================================================
# Decorator for Provider Registration
# =============================================================================


def register_provider(name: str) -> Callable[[type], type]:
    """مُزخرف لتسجيل مزود"""

    def decorator(cls: type) -> type:
        ProviderFactory.register(name, cls)
        return cls

    return decorator
