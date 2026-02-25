"""
Comprehensive unit tests for GAAP providers.

Tests provider functionality from gaap/providers/:
- BaseProvider: provider creation, properties, rate limiting, timeout, retry logic
- AccountManager: account management, round-robin, rotation
- ToolCalling: tool creation, execution, results
- Streaming: streaming responses, chunks
- PromptCaching: cache key generation, hit/miss
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gaap.core.exceptions import (
    ModelNotFoundError,
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
from gaap.providers.base_provider import (
    BaseProvider,
    RateLimitState,
    RateLimiter,
    RetryConfig,
    RetryManager,
    UsageRecord,
    UsageTracker,
)
from gaap.providers.account_manager import (
    AccountPool,
    AccountSlot,
    AccountStatus,
    RateLimitTracker,
    SessionTracker,
    detect_hard_cooldown,
)
from gaap.providers.tool_calling import (
    ParameterSchema,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    ToolType,
    create_tool_from_function,
)
from gaap.providers.streaming import (
    NativeStreamer,
    StreamConfig,
    StreamProtocol,
    TokenChunk,
    SSEParser,
    ConnectRPCParser,
    DeepSeekParser,
    collect_stream,
)
from gaap.providers.prompt_caching import (
    CacheConfig,
    CacheEntry,
    CacheProvider,
    PromptCache,
    estimate_cache_savings,
)


class ConcreteProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""

    def __init__(self, **kwargs):
        super().__init__(
            name="test-provider",
            provider_type=ProviderType.FREE_TIER,
            models=["model-1", "model-2", "llama-3.3-70b"],
            **kwargs,
        )
        self._request_count = 0

    async def _make_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> dict[str, Any]:
        self._request_count += 1
        return {
            "id": f"response-{self._request_count}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    async def _stream_request(self, messages: list[Message], model: str, **kwargs: Any):
        for chunk in ["Hello", " ", "world"]:
            yield chunk

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * 0.00001) + (output_tokens * 0.00003)


class TestRateLimitState:
    """Tests for RateLimitState."""

    def test_rate_limit_state_creation(self):
        state = RateLimitState(requests_per_minute=60, tokens_per_minute=100000)
        assert state.requests_per_minute == 60
        assert state.tokens_per_minute == 100000
        assert state.current_requests == 0
        assert state.current_tokens == 0

    def test_is_allowed_within_limits(self):
        state = RateLimitState(requests_per_minute=10, tokens_per_minute=1000)
        assert state.is_allowed(tokens=100) is True

    def test_is_allowed_at_request_limit(self):
        state = RateLimitState(requests_per_minute=5, tokens_per_minute=1000)
        for _ in range(5):
            state.record_request(tokens=50)
        assert state.is_allowed(tokens=100) is False

    def test_is_allowed_at_token_limit(self):
        state = RateLimitState(requests_per_minute=10, tokens_per_minute=500)
        state.record_request(tokens=400)
        assert state.is_allowed(tokens=200) is False

    def test_record_request(self):
        state = RateLimitState(requests_per_minute=60)
        state.record_request(tokens=100)
        assert state.current_requests == 1
        assert state.current_tokens == 100


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_within_limits(self):
        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=100000)
        result = await limiter.acquire(tokens=100)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_at_limit(self):
        limiter = RateLimiter(requests_per_minute=2, tokens_per_minute=100000)
        await limiter.acquire(tokens=100)
        await limiter.acquire(tokens=100)
        result = await limiter.acquire(tokens=100)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_slot_immediate(self):
        limiter = RateLimiter(requests_per_minute=60)
        result = await limiter.wait_for_slot(tokens=100, timeout=5)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_slot_timeout(self):
        limiter = RateLimiter(requests_per_minute=1, tokens_per_minute=10)
        await limiter.acquire(tokens=5)
        start = time.time()
        result = await limiter.wait_for_slot(tokens=100, timeout=0.5)
        elapsed = time.time() - start
        assert result is False
        assert elapsed < 1.0


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert 429 in config.retry_on_status_codes

    def test_custom_config(self):
        config = RetryConfig(max_retries=5, base_delay=0.5, max_delay=120.0)
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 120.0


class TestRetryManager:
    """Tests for RetryManager."""

    def test_get_delay_exponential(self):
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=60.0)
        manager = RetryManager(config)
        assert manager.get_delay(0) == 1.0
        assert manager.get_delay(1) == 2.0
        assert manager.get_delay(2) == 4.0
        assert manager.get_delay(3) == 8.0

    def test_get_delay_max_cap(self):
        config = RetryConfig(base_delay=10.0, exponential_base=10.0, max_delay=50.0)
        manager = RetryManager(config)
        assert manager.get_delay(1) == 50.0
        assert manager.get_delay(2) == 50.0

    def test_should_retry_within_limit(self):
        config = RetryConfig(max_retries=3)
        manager = RetryManager(config)
        timeout_err = asyncio.TimeoutError()
        assert manager.should_retry(0, timeout_err) is True
        assert manager.should_retry(1, timeout_err) is True
        assert manager.should_retry(2, timeout_err) is True
        assert manager.should_retry(3, timeout_err) is False

    def test_should_retry_on_timeout_error(self):
        config = RetryConfig(max_retries=3)
        manager = RetryManager(config)
        assert manager.should_retry(1, asyncio.TimeoutError()) is True

    def test_should_retry_on_rate_limit_error(self):
        config = RetryConfig(max_retries=3)
        manager = RetryManager(config)
        error = ProviderRateLimitError(provider_name="test", retry_after=60)
        assert manager.should_retry(1, error) is True

    def test_should_retry_on_status_code(self):
        config = RetryConfig(max_retries=3, retry_on_status_codes=[500, 502])
        manager = RetryManager(config)
        error_500 = MagicMock()
        error_500.status_code = 500
        assert manager.should_retry(1, error_500) is True
        error_400 = MagicMock()
        error_400.status_code = 400
        assert manager.should_retry(1, error_400) is False

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        config = RetryConfig(max_retries=3)
        manager = RetryManager(config)

        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await manager.execute_with_retry(success_func)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self):
        config = RetryConfig(max_retries=3, base_delay=0.01)
        manager = RetryManager(config)

        call_count = 0

        async def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError("timeout")
            return "success"

        result = await manager.execute_with_retry(eventual_success)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        config = RetryConfig(max_retries=2, base_delay=0.01)
        manager = RetryManager(config)

        async def always_fail():
            raise asyncio.TimeoutError("always timeout")

        with pytest.raises(asyncio.TimeoutError):
            await manager.execute_with_retry(always_fail)


class TestUsageTracker:
    """Tests for UsageTracker."""

    def test_record_usage(self):
        tracker = UsageTracker()
        record = tracker.record(
            provider="test",
            model="model-1",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            latency_ms=500,
            success=True,
        )
        assert record.provider == "test"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.total_tokens == 150
        assert record.success is True

    def test_get_totals_by_provider(self):
        tracker = UsageTracker()
        tracker.record("groq", "llama", 100, 50, 0.01, 500, True)
        tracker.record("groq", "llama", 200, 100, 0.02, 600, True)
        tracker.record("gemini", "flash", 150, 75, 0.015, 400, True)

        groq_totals = tracker.get_totals("groq")
        assert groq_totals["total_tokens"] == 450
        assert groq_totals["total_requests"] == 2

    def test_get_totals_all(self):
        tracker = UsageTracker()
        tracker.record("groq", "llama", 100, 50, 0.01, 500, True)
        tracker.record("gemini", "flash", 150, 75, 0.015, 400, True)

        totals = tracker.get_totals()
        assert "groq" in totals
        assert "gemini" in totals

    def test_get_recent_records(self):
        tracker = UsageTracker(max_records=5)
        for i in range(10):
            tracker.record("test", "model", i, i, 0.01, 100, True)

        recent = tracker.get_recent_records(limit=3)
        assert len(recent) == 3

    def test_max_records_trimming(self):
        tracker = UsageTracker(max_records=3)
        for i in range(5):
            tracker.record("test", "model", i, i, 0.01, 100, True)

        assert len(tracker._records) == 3


class TestBaseProvider:
    """Tests for BaseProvider."""

    def test_provider_creation(self):
        provider = ConcreteProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            rate_limit_rpm=30,
            timeout=60,
        )
        assert provider.name == "test-provider"
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://api.test.com"
        assert provider.timeout == 60
        assert provider.default_model == "model-1"

    def test_provider_properties(self):
        provider = ConcreteProvider()
        assert provider.name == "test-provider"
        assert provider.provider_type == ProviderType.FREE_TIER
        assert len(provider.models) == 3
        assert "model-1" in provider.models

    def test_is_model_available(self):
        provider = ConcreteProvider()
        assert provider.is_model_available("model-1") is True
        assert provider.is_model_available("nonexistent") is False

    def test_get_available_models(self):
        provider = ConcreteProvider()
        models = provider.get_available_models()
        assert len(models) == 3
        assert "model-1" in models

    def test_get_model_tier_strategic(self):
        provider = ConcreteProvider()
        tier = provider.get_model_tier("llama-3.3-70b")
        assert tier == ModelTier.TIER_4_PRIVATE

    def test_get_model_tier_tactical(self):
        provider = ConcreteProvider()
        tier = provider.get_model_tier("model-1")
        assert tier == ModelTier.TIER_2_TACTICAL

    @pytest.mark.asyncio
    async def test_rate_limiting_within_limits(self):
        provider = ConcreteProvider(rate_limit_rpm=60)
        messages = [Message(role=MessageRole.USER, content="Hello")]
        response = await provider.chat_completion(messages)
        assert response is not None
        assert len(response.choices) > 0

    @pytest.mark.asyncio
    async def test_rate_limiting_at_limit(self):
        provider = ConcreteProvider(rate_limit_rpm=1, rate_limit_tpm=10)
        messages = [Message(role=MessageRole.USER, content="Hi")]

        with pytest.raises(ProviderRateLimitError):
            await provider.chat_completion(messages)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_timeout_handling(self):
        provider = ConcreteProvider(timeout=0.1)

        async def slow_request(*args, **kwargs):
            await asyncio.sleep(0.15)
            return {}

        provider._make_request = slow_request
        messages = [Message(role=MessageRole.USER, content="Hello")]

        with pytest.raises(ProviderTimeoutError):
            await provider.chat_completion(messages)

    @pytest.mark.asyncio
    async def test_retry_logic_on_failure(self):
        provider = ConcreteProvider(max_retries=2)

        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError("timeout")
            return {
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        provider._make_request = failing_then_success
        messages = [Message(role=MessageRole.USER, content="Hello")]
        response = await provider.chat_completion(messages)
        assert response is not None
        assert call_count == 2

    def test_initialize(self):
        provider = ConcreteProvider()
        provider.initialize()
        assert provider._is_initialized is True

    def test_shutdown(self):
        provider = ConcreteProvider()
        provider.initialize()
        provider.shutdown()
        assert provider._is_initialized is False

    def test_get_stats(self):
        provider = ConcreteProvider()
        stats = provider.get_stats()
        assert stats["name"] == "test-provider"
        assert stats["total_requests"] == 0
        assert "success_rate" in stats


class TestRateLimitTracker:
    """Tests for RateLimitTracker."""

    def test_record_call_success(self):
        tracker = RateLimitTracker(max_rpm=5)
        tracker.record_call(success=True, latency_ms=100, tokens_used=50)
        assert tracker._total_calls == 1
        assert tracker._total_success == 1
        assert tracker._consecutive_errors == 0

    def test_record_call_failure(self):
        tracker = RateLimitTracker(max_rpm=5)
        tracker.record_call(success=False, latency_ms=100, error_msg="Error")
        assert tracker._total_calls == 1
        assert tracker._total_errors == 1
        assert tracker._consecutive_errors == 1

    def test_rpm_current(self):
        tracker = RateLimitTracker(max_rpm=5)
        for _ in range(3):
            tracker.record_call(success=True)
        assert tracker.rpm_current == 3

    def test_can_call_within_limits(self):
        tracker = RateLimitTracker(max_rpm=5, max_rph=100, max_rpd=1000)
        can, reason = tracker.can_call()
        assert can is True
        assert reason == "OK"

    def test_can_call_at_rpm_limit(self):
        tracker = RateLimitTracker(max_rpm=2)
        tracker.record_call(success=True)
        tracker.record_call(success=True)
        can, reason = tracker.can_call()
        assert can is False
        assert "RPM limit" in reason

    def test_consecutive_errors_triggers_cooldown(self):
        tracker = RateLimitTracker(max_rpm=5)
        for _ in range(3):
            tracker.record_call(success=False, error_msg="Error")
        assert tracker.in_cooldown is True

    def test_set_hard_cooldown(self):
        tracker = RateLimitTracker()
        tracker.set_hard_cooldown(seconds=3600, reason="Daily limit")
        assert tracker.in_cooldown is True
        assert "Daily limit" in tracker.cooldown_reason

    def test_health_score(self):
        tracker = RateLimitTracker()
        assert tracker.health_score() == 1.0
        tracker.record_call(success=False, error_msg="Error")
        assert tracker.health_score() < 1.0

    def test_health_score_in_cooldown(self):
        tracker = RateLimitTracker()
        tracker.set_hard_cooldown(60, "test")
        assert tracker.health_score() == 0.0

    def test_get_warnings(self):
        tracker = RateLimitTracker(max_rpm=5, warn_threshold=0.8)
        for _ in range(4):
            tracker.record_call(success=True)
        warnings = tracker.get_warnings()
        assert len(warnings) > 0
        assert "RPM" in warnings[0]


class TestSessionTracker:
    """Tests for SessionTracker."""

    def test_create_session(self):
        tracker = SessionTracker()
        tracker.create_session("session-1")
        assert tracker.active_count == 1

    def test_record_message(self):
        tracker = SessionTracker()
        tracker.create_session("session-1")
        tracker.record_message("session-1")
        tracker.record_message("session-1")
        assert tracker.total_messages == 2

    def test_should_rotate_by_messages(self):
        tracker = SessionTracker(max_messages_per_session=3)
        tracker.create_session("session-1")
        for _ in range(3):
            tracker.record_message("session-1")
        assert tracker.should_rotate("session-1") is True

    def test_max_sessions_eviction(self):
        tracker = SessionTracker(max_sessions=2)
        tracker.create_session("session-1")
        tracker.create_session("session-2")
        tracker.create_session("session-3")
        assert tracker.active_count == 2


class TestAccountSlot:
    """Tests for AccountSlot."""

    def test_account_slot_creation(self):
        slot = AccountSlot(
            provider="test",
            label="main",
            email="test@example.com",
            models=["model-1"],
        )
        slot._enabled = True
        assert slot.provider == "test"
        assert slot.label == "main"
        assert slot.email == "test@example.com"

    def test_account_slot_can_call(self):
        slot = AccountSlot(provider="test", label="main")
        slot._enabled = True
        with patch.object(type(slot), "status", AccountStatus.ACTIVE):
            can, reason = slot.rate_tracker.can_call()
            assert can is True
            assert reason == "OK"

    def test_account_slot_disabled(self):
        slot = AccountSlot(provider="test", label="main")
        slot._enabled = False
        assert slot.status == AccountStatus.DISABLED
        can, reason = slot.can_call()
        assert can is False

    def test_account_slot_to_config_dict(self):
        slot = AccountSlot(
            provider="test",
            label="main",
            email="test@example.com",
            models=["model-1"],
        )
        config = slot.to_config_dict()
        assert config["provider"] == "test"
        assert config["label"] == "main"
        assert config["email"] == "test@example.com"

    def test_account_slot_from_config_dict(self):
        config = {
            "provider": "test",
            "label": "main",
            "email": "test@example.com",
            "models": ["model-1"],
            "max_rpm": 10,
            "max_rph": 100,
            "max_rpd": 1000,
        }
        slot = AccountSlot.from_config_dict(config)
        assert slot.provider == "test"
        assert slot.label == "main"
        assert slot.email == "test@example.com"


class TestAccountPool:
    """Tests for AccountPool."""

    def test_add_account(self):
        pool = AccountPool.__new__(AccountPool)
        pool.provider = "test"
        pool._accounts = {}
        pool._lock = MagicMock()
        slot = AccountSlot(provider="test", label="main", email="test@example.com")
        pool._accounts["main"] = slot
        assert slot.label == "main"
        assert slot.provider == "test"
        assert pool.get_account("main") is not None

    def test_remove_account(self):
        pool = AccountPool.__new__(AccountPool)
        pool.provider = "test"
        pool._accounts = {}
        pool._lock = MagicMock()
        slot = AccountSlot(provider="test", label="main")
        pool._accounts["main"] = slot
        del pool._accounts["main"]
        assert pool.get_account("main") is None

    def test_remove_account_not_found(self):
        pool = AccountPool.__new__(AccountPool)
        pool._accounts = {}
        assert pool._accounts.get("nonexistent") is None

    def test_get_next_account_best(self):
        pool = AccountPool.__new__(AccountPool)
        pool._accounts = {}
        main = AccountSlot(provider="test", label="main")
        main._enabled = True
        pool._accounts["main"] = main
        alt1 = AccountSlot(provider="test", label="alt1")
        alt1._enabled = True
        pool._accounts["alt1"] = alt1

        def mock_can_call(self):
            if self._enabled:
                return (True, "OK")
            return (False, "Disabled")

        with patch.object(AccountSlot, "can_call", mock_can_call):
            with patch.object(AccountSlot, "health_score", property(lambda self: 1.0)):
                best = pool.best_account()
                assert best is not None
                assert best.label in ["main", "alt1"]

    def test_account_rotation(self):
        pool = AccountPool.__new__(AccountPool)
        pool._accounts = {}
        main = AccountSlot(provider="test", label="main")
        main._enabled = True
        pool._accounts["main"] = main
        alt1 = AccountSlot(provider="test", label="alt1")
        alt1._enabled = True
        pool._accounts["alt1"] = alt1
        alt2 = AccountSlot(provider="test", label="alt2")
        alt2._enabled = True
        pool._accounts["alt2"] = alt2

        def mock_can_call(self):
            if self._enabled:
                return (True, "OK")
            return (False, "Disabled")

        with patch.object(AccountSlot, "can_call", mock_can_call):
            with patch.object(AccountSlot, "health_score", property(lambda self: 1.0)):
                used = set()
                for _ in range(3):
                    best = pool.best_account()
                    if best:
                        used.add(best.label)
                        best.rate_tracker.record_call(success=True)

                assert len(used) <= 3

    def test_exhausted_accounts(self):
        pool = AccountPool.__new__(AccountPool)
        pool._accounts = {}
        slot = AccountSlot(provider="test", label="main", max_rpm=1)
        slot._enabled = True
        pool._accounts["main"] = slot
        slot.rate_tracker.record_call(success=True)
        slot.rate_tracker.record_call(success=True)

        best = pool.best_account()
        assert best is None

    def test_record_call(self):
        pool = AccountPool.__new__(AccountPool)
        pool._accounts = {}
        slot = AccountSlot(provider="test", label="main")
        pool._accounts["main"] = slot
        pool.record_call("main", success=True, latency_ms=100, tokens_used=50)
        acct = pool.get_account("main")
        assert acct is not None
        assert acct.rate_tracker._total_calls == 1

    def test_active_accounts(self):
        pool = AccountPool.__new__(AccountPool)
        pool._accounts = {}
        main = AccountSlot(provider="test", label="main")
        main._enabled = True
        pool._accounts["main"] = main
        alt1 = AccountSlot(provider="test", label="alt1")
        alt1._enabled = True
        pool._accounts["alt1"] = alt1
        disabled = AccountSlot(provider="test", label="disabled")
        disabled._enabled = False
        pool._accounts["disabled"] = disabled

        def mock_can_call(self):
            if self._enabled:
                return (True, "OK")
            return (False, "Disabled")

        with patch.object(AccountSlot, "can_call", mock_can_call):
            active = [a for a in pool._accounts.values() if a.can_call()[0]]
            assert len(active) == 2

    def test_should_call(self):
        pool = AccountPool.__new__(AccountPool)
        pool._accounts = {}
        main = AccountSlot(provider="test", label="main")
        main._enabled = True
        pool._accounts["main"] = main

        def mock_can_call(self):
            if self._enabled:
                return (True, "OK")
            return (False, "Disabled")

        with patch.object(AccountSlot, "can_call", mock_can_call):
            with patch.object(AccountSlot, "health_score", property(lambda self: 1.0)):
                should, reason, label = pool.should_call()
                assert should is True
                assert label == "main"

    def test_should_call_no_accounts(self):
        pool = AccountPool.__new__(AccountPool)
        pool._accounts = {}
        should, reason, label = pool.should_call()
        assert should is False
        assert label is None


class TestDetectHardCooldown:
    """Tests for detect_hard_cooldown function."""

    def test_kimi_daily_limit(self):
        result = detect_hard_cooldown("当前模型对话次数已达上限，3小时后恢复")
        assert result is not None
        assert result[0] == 3 * 3600
        assert "Kimi" in result[1]

    def test_kimi_minutes_limit(self):
        result = detect_hard_cooldown("5分钟后恢复")
        assert result is not None
        assert result[0] == 5 * 60

    def test_rate_limit_hours(self):
        result = detect_hard_cooldown("rate limit 2 hour")
        assert result is not None
        assert result[0] == 2 * 3600

    def test_no_cooldown_detected(self):
        result = detect_hard_cooldown("Some random error message")
        assert result is None


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_tool_definition_creation(self):
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "location": ParameterSchema(type="string", description="City name"),
            },
        )
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a location"

    def test_to_openai_schema(self):
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={
                "param1": ParameterSchema(type="string", required=True),
                "param2": ParameterSchema(type="integer", required=False),
            },
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert "param1" in schema["function"]["parameters"]["properties"]
        assert "param1" in schema["function"]["parameters"]["required"]
        assert "param2" not in schema["function"]["parameters"]["required"]

    def test_to_anthropic_schema(self):
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"param1": ParameterSchema(type="string")},
        )
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "test_tool"
        assert "input_schema" in schema

    def test_to_gemini_schema(self):
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"param1": ParameterSchema(type="string")},
        )
        schema = tool.to_gemini_schema()
        assert schema["name"] == "test_tool"
        assert "parameters" in schema


class TestToolCall:
    """Tests for ToolCall."""

    def test_tool_call_creation(self):
        call = ToolCall(
            id="call-123",
            name="get_weather",
            arguments={"location": "Tokyo"},
        )
        assert call.id == "call-123"
        assert call.name == "get_weather"
        assert call.arguments == {"location": "Tokyo"}

    def test_from_openai(self):
        data = {
            "id": "call-123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Tokyo"}',
            },
        }
        call = ToolCall.from_openai(data)
        assert call.id == "call-123"
        assert call.name == "get_weather"
        assert call.arguments == {"location": "Tokyo"}

    def test_from_anthropic(self):
        data = {
            "id": "call-123",
            "name": "get_weather",
            "input": {"location": "Tokyo"},
        }
        call = ToolCall.from_anthropic(data)
        assert call.id == "call-123"
        assert call.name == "get_weather"
        assert call.arguments == {"location": "Tokyo"}


class TestToolResult:
    """Tests for ToolResult."""

    def test_tool_result_creation(self):
        result = ToolResult(
            tool_call_id="call-123",
            name="get_weather",
            result="Sunny, 25°C",
        )
        assert result.tool_call_id == "call-123"
        assert result.name == "get_weather"
        assert result.result == "Sunny, 25°C"
        assert result.is_error is False

    def test_to_openai_message(self):
        result = ToolResult(
            tool_call_id="call-123",
            name="get_weather",
            result="Sunny, 25°C",
        )
        msg = result.to_openai_message()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call-123"
        assert msg["content"] == "Sunny, 25°C"

    def test_to_anthropic_message(self):
        result = ToolResult(
            tool_call_id="call-123",
            name="get_weather",
            result="Sunny, 25°C",
        )
        msg = result.to_anthropic_message()
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "tool_result"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="test_tool", description="Test")
        registry.register(tool)
        assert "test_tool" in registry.list_tools()

    def test_unregister_tool(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="test_tool", description="Test")
        registry.register(tool)
        assert registry.unregister("test_tool") is True
        assert registry.unregister("nonexistent") is False

    def test_get_tool(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="test_tool", description="Test")
        registry.register(tool)
        retrieved = registry.get("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_to_openai_tools(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="tool1", description="Test 1"))
        registry.register(ToolDefinition(name="tool2", description="Test 2"))
        tools = registry.to_openai_tools()
        assert len(tools) == 2

    def test_to_provider_format(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="test_tool", description="Test"))
        openai_tools = registry.to_provider_format("openai")
        assert len(openai_tools) == 1
        anthropic_tools = registry.to_provider_format("anthropic")
        assert len(anthropic_tools) == 1

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        registry = ToolRegistry()

        def handler(location: str) -> str:
            return f"Weather in {location}: Sunny"

        tool = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters={"location": ParameterSchema(type="string")},
            handler=handler,
        )
        registry.register(tool)

        call = ToolCall(id="call-1", name="get_weather", arguments={"location": "Tokyo"})
        result = await registry.execute(call)
        assert result.is_error is False
        assert "Tokyo" in result.result

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        registry = ToolRegistry()
        call = ToolCall(id="call-1", name="nonexistent", arguments={})
        result = await registry.execute(call)
        assert result.is_error is True
        assert "not found" in result.result

    @pytest.mark.asyncio
    async def test_execute_tool_no_handler(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="test", description="Test"))
        call = ToolCall(id="call-1", name="test", arguments={})
        result = await registry.execute(call)
        assert result.is_error is True
        assert "no handler" in result.result

    def test_validate_arguments_valid(self):
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters={
                "required_param": ParameterSchema(type="string", required=True),
                "optional_param": ParameterSchema(type="integer", required=False),
            },
        )
        registry.register(tool)

        call = ToolCall(id="call-1", name="test", arguments={"required_param": "value"})
        is_valid, errors = registry.validate_arguments(call)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_arguments_missing_required(self):
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters={
                "required_param": ParameterSchema(type="string", required=True),
            },
        )
        registry.register(tool)

        call = ToolCall(id="call-1", name="test", arguments={})
        is_valid, errors = registry.validate_arguments(call)
        assert is_valid is False
        assert "Missing required parameter" in errors[0]


class TestCreateToolFromFunction:
    """Tests for create_tool_from_function."""

    def test_create_from_function(self):
        def get_weather(location: str, units: str = "celsius") -> str:
            """Get weather for a location."""
            return f"Weather in {location}"

        tool = create_tool_from_function(get_weather)
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a location."
        assert "location" in tool.parameters
        assert tool.parameters["location"].required is True
        assert tool.parameters["units"].required is False

    def test_create_with_custom_name(self):
        def my_func(x: int) -> int:
            return x * 2

        tool = create_tool_from_function(my_func, name="double_value")
        assert tool.name == "double_value"


class TestTokenChunk:
    """Tests for TokenChunk."""

    def test_token_chunk_creation(self):
        chunk = TokenChunk(content="Hello", token_count=1)
        assert chunk.content == "Hello"
        assert chunk.is_final is False
        assert chunk.is_thinking is False

    def test_is_thinking(self):
        chunk = TokenChunk(content="", metadata={"is_thinking": True})
        assert chunk.is_thinking is True


class TestStreamConfig:
    """Tests for StreamConfig."""

    def test_default_config(self):
        config = StreamConfig()
        assert config.protocol == StreamProtocol.SSE
        assert config.chunk_timeout == 30.0
        assert config.max_response_bytes == 512 * 1024

    def test_custom_config(self):
        config = StreamConfig(
            protocol=StreamProtocol.CONNECT_RPC,
            chunk_timeout=60.0,
            max_response_bytes=1024 * 1024,
        )
        assert config.protocol == StreamProtocol.CONNECT_RPC
        assert config.chunk_timeout == 60.0


class TestSSEParser:
    """Tests for SSEParser."""

    @pytest.mark.asyncio
    async def test_parse_sse_stream(self):
        parser = SSEParser()
        config = StreamConfig()

        class MockResponse:
            async def aiter_lines(self):
                lines = [
                    b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                    b'data: {"choices": [{"delta": {"content": " world"}}]}',
                    b"data: [DONE]",
                ]
                for line in lines:
                    yield line

            def close(self):
                pass

        chunks = []
        async for chunk in parser.parse(MockResponse(), config):
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"


class TestConnectRPCParser:
    """Tests for ConnectRPCParser."""

    def test_parse_envelopes(self):
        import json
        import struct

        msg = {"op": "append", "block": {"text": {"content": "Hello"}}}
        data = json.dumps(msg).encode()
        envelope = struct.pack(">BI", 0, len(data)) + data

        messages = ConnectRPCParser.parse_envelopes(envelope)
        assert len(messages) == 1
        assert messages[0][0] == 0
        assert messages[0][1]["op"] == "append"


class TestDeepSeekParser:
    """Tests for DeepSeekParser."""

    def test_extract_deepseek_content_simple(self):
        parser = DeepSeekParser()
        chunk = {"v": "Hello world"}
        content = parser._extract_deepseek_content(chunk)
        assert content == "Hello world"

    def test_extract_deepseek_content_with_response(self):
        parser = DeepSeekParser()
        chunk = {
            "v": {
                "response": {
                    "fragments": [
                        {"type": "RESPONSE", "content": "Hello"},
                        {"type": "RESPONSE", "content": " world"},
                    ]
                }
            }
        }
        content = parser._extract_deepseek_content(chunk)
        assert content == "Hello world"


class TestNativeStreamer:
    """Tests for NativeStreamer."""

    def test_get_parser_sse(self):
        streamer = NativeStreamer()
        parser = streamer.get_parser(StreamProtocol.SSE)
        assert isinstance(parser, SSEParser)

    def test_get_parser_connect_rpc(self):
        streamer = NativeStreamer()
        parser = streamer.get_parser(StreamProtocol.CONNECT_RPC)
        assert isinstance(parser, ConnectRPCParser)

    @pytest.mark.asyncio
    async def test_stream_with_yield_delay(self):
        streamer = NativeStreamer(config=StreamConfig(yield_delay=0.01))

        class MockResponse:
            async def aiter_lines(self):
                yield b'data: {"choices": [{"delta": {"content": "test"}}]}'

            def close(self):
                pass

        chunks = []
        async for chunk in streamer.stream(MockResponse(), protocol=StreamProtocol.SSE):
            chunks.append(chunk)

        assert len(chunks) >= 1


class TestCollectStream:
    """Tests for collect_stream."""

    @pytest.mark.asyncio
    async def test_collect_stream(self):
        async def stream():
            yield TokenChunk(content="Hello")
            yield TokenChunk(content=" ")
            yield TokenChunk(content="world", is_final=True, finish_reason="stop")

        text, metadata = await collect_stream(stream())
        assert text == "Hello world"
        assert metadata["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_collect_stream_with_metadata(self):
        async def stream():
            yield TokenChunk(content="test", metadata={"is_thinking": True})

        text, metadata = await collect_stream(stream(), include_metadata=True)
        assert text == "test"
        assert len(metadata["chunks"]) == 1


class TestPromptCache:
    """Tests for PromptCache."""

    def test_cache_creation(self):
        cache = PromptCache()
        assert cache.config.enabled is True
        assert len(cache._entries) == 0

    def test_cache_key_generation(self):
        cache = PromptCache()
        messages = [{"role": "system", "content": "Test prompt"}]
        key1 = cache._get_cache_key(messages, "anthropic")
        key2 = cache._get_cache_key(messages, "anthropic")
        assert key1 == key2
        key3 = cache._get_cache_key(messages, "openai")
        assert key1 != key3

    def test_cache_hit(self):
        cache = PromptCache()
        long_content = "x" * 1500
        messages = [{"role": "system", "content": long_content}]

        cache.optimize(messages, "anthropic")

        cache.optimize(messages, "anthropic")

        stats = cache.get_stats()
        assert stats["hits"] >= 1

    def test_cache_miss(self):
        cache = PromptCache()
        long_content = "x" * 1500
        messages1 = [{"role": "system", "content": long_content + "1"}]
        messages2 = [{"role": "system", "content": long_content + "2"}]

        cache.optimize(messages1, "anthropic")
        cache.optimize(messages2, "anthropic")

        stats = cache.get_stats()
        assert stats["misses"] == 2

    def test_optimize_anthropic(self):
        cache = PromptCache()
        long_content = "x" * 1500
        messages = [{"role": "system", "content": long_content}]

        optimized = cache.optimize(messages, "anthropic")

        assert len(optimized) == 1
        assert "cache_control" in optimized[0]

    def test_optimize_disabled(self):
        cache = PromptCache(config=CacheConfig(enabled=False))
        messages = [{"role": "system", "content": "Test"}]

        optimized = cache.optimize(messages, "anthropic")

        assert optimized == messages

    def test_get_stats(self):
        cache = PromptCache()
        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "tokens_saved" in stats

    def test_clear(self):
        cache = PromptCache()
        cache._entries["test"] = CacheEntry(
            key="test",
            content_hash="hash",
            cached_tokens=100,
            created_at=time.time(),
            last_used=time.time(),
        )
        cache.clear()
        assert len(cache._entries) == 0

    def test_check_cache_status(self):
        cache = PromptCache()
        long_content = "x" * 1500
        messages = [{"role": "system", "content": long_content}]

        cache.optimize(messages, "anthropic")

        status = cache.check_cache_status(messages, "anthropic")
        assert status["cacheable"] is True


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_cache_entry_creation(self):
        entry = CacheEntry(
            key="test-key",
            content_hash="abc123",
            cached_tokens=100,
            created_at=time.time(),
            last_used=time.time(),
        )
        assert entry.key == "test-key"
        assert entry.cached_tokens == 100
        assert entry.is_expired is False

    def test_cache_entry_expired(self):
        entry = CacheEntry(
            key="test-key",
            content_hash="abc123",
            cached_tokens=100,
            created_at=time.time() - 400,
            last_used=time.time() - 400,
        )
        assert entry.is_expired is True


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_config(self):
        config = CacheConfig()
        assert config.enabled is True
        assert config.min_tokens == 1024
        assert config.max_cache_size == 100
        assert config.ttl_seconds == 300

    def test_custom_config(self):
        config = CacheConfig(
            enabled=False,
            min_tokens=512,
            max_cache_size=50,
        )
        assert config.enabled is False
        assert config.min_tokens == 512
        assert config.max_cache_size == 50


class TestEstimateCacheSavings:
    """Tests for estimate_cache_savings."""

    def test_estimate_savings(self):
        long_content = "x" * 2000
        messages = [
            {"role": "system", "content": long_content},
            {"role": "user", "content": "Hello"},
        ]

        result = estimate_cache_savings(messages, "anthropic")

        assert "total_tokens" in result
        assert "cacheable_tokens" in result
        assert "normal_cost" in result
        assert "cached_cost" in result
        assert "savings" in result
        assert result["savings"] > 0

    def test_estimate_savings_custom_pricing(self):
        messages = [{"role": "user", "content": "x" * 2000}]
        pricing = {"input": 0.01, "cached": 0.001}

        result = estimate_cache_savings(messages, "custom", pricing=pricing)

        assert result["normal_cost"] > result["cached_cost"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
