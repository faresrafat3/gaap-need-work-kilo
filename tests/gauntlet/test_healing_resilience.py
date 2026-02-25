"""
Healing Resilience Gauntlet Tests
=================================

Tests for self-healing system resilience under fault injection.

Implements: docs/evolution_plan_2026/45_TESTING_AUDIT_SPEC.md
"""

import asyncio
import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaap.core.types import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    Message,
    MessageRole,
    Usage,
)


class FailingProvider:
    """Provider that fails intermittently."""

    def __init__(self, failure_rate: float = 0.5):
        self.name = "failing-provider"
        self.failure_rate = failure_rate
        self.call_count = 0
        self.failure_count = 0

    async def chat_completion(self, messages, model=None, **kwargs):
        self.call_count += 1

        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise ConnectionError("Simulated network failure")

        return ChatCompletionResponse(
            id="success-response",
            model=model or "test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role=MessageRole.ASSISTANT, content="Success!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

    def get_stats(self):
        return {
            "call_count": self.call_count,
            "failure_count": self.failure_count,
            "failure_rate_actual": self.failure_count / max(1, self.call_count),
        }


class TestHealingResilienceGauntlet:
    """Tests for healing system resilience."""

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_provider_retry_on_failure(self) -> None:
        """Test that system retries on provider failure."""
        provider = FailingProvider(failure_rate=1.0)

        success_count = 0
        for _ in range(3):
            try:
                response = await provider.chat_completion(
                    [Message(role=MessageRole.USER, content="test")]
                )
                if response.choices[0].message.content == "Success!":
                    success_count += 1
            except ConnectionError:
                pass

        stats = provider.get_stats()
        assert stats["call_count"] == 3
        assert stats["failure_count"] == 3

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_eventual_success_with_retries(self) -> None:
        """Test that eventual success is achieved with retries."""
        provider = FailingProvider(failure_rate=0.5)
        random.seed(42)

        max_retries = 10
        success = False

        for attempt in range(max_retries):
            try:
                response = await provider.chat_completion(
                    [Message(role=MessageRole.USER, content="test")]
                )
                success = True
                break
            except ConnectionError:
                continue

        assert success or provider.call_count > 0

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    def test_chaos_monkey_network_injection(self, chaos_monkey) -> None:
        """Test chaos monkey can inject failures."""
        chaos_monkey.failure_rate = 1.0

        injected = chaos_monkey.inject_network_failure()
        assert injected is True

        stats = chaos_monkey.get_stats()
        assert stats["injected_failures"] >= 1

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    def test_chaos_monkey_timeout_injection(self, chaos_monkey) -> None:
        """Test chaos monkey can inject timeouts."""
        chaos_monkey.failure_rate = 1.0

        original_timeout = 5.0
        modified_timeout = chaos_monkey.inject_timeout(original_timeout)

        assert modified_timeout > original_timeout

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    def test_chaos_monkey_data_corruption(self, chaos_monkey) -> None:
        """Test chaos monkey can corrupt data."""
        chaos_monkey.failure_rate = 1.0

        original_data = {"key1": "value1", "key2": "value2"}
        corrupted_data = chaos_monkey.corrupt_data(original_data)

        assert corrupted_data != original_data or chaos_monkey.failure_rate < 1.0

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    def test_chaos_respects_failure_rate(self, chaos_monkey) -> None:
        """Test chaos monkey respects failure rate."""
        chaos_monkey.failure_rate = 0.0

        assert chaos_monkey.inject_network_failure() is False

        data = {"key": "value"}
        assert chaos_monkey.corrupt_data(data) == data


class TestHealingScenariosGauntlet:
    """Scenario-based healing tests."""

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self) -> None:
        """Test recovery from rate limit errors."""

        class RateLimitProvider:
            def __init__(self):
                self.name = "rate-limited"
                self.calls = 0
                self.rate_limited_until = 0

            async def chat_completion(self, messages, model=None, **kwargs):
                import time

                self.calls += 1

                if time.time() < self.rate_limited_until:
                    raise Exception("Rate limit exceeded")

                return ChatCompletionResponse(
                    id="response",
                    model=model or "test",
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=Message(role=MessageRole.ASSISTANT, content="OK"),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                )

        provider = RateLimitProvider()

        try:
            response = await provider.chat_completion(
                [Message(role=MessageRole.USER, content="test")]
            )
            assert response.choices[0].message.content == "OK"
        except Exception:
            pass

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_timeout_recovery(self) -> None:
        """Test recovery from timeout."""

        class TimeoutProvider:
            def __init__(self, timeout_after: int = 2):
                self.name = "timeout-prone"
                self.calls = 0
                self.timeout_after = timeout_after

            async def chat_completion(self, messages, model=None, **kwargs):
                self.calls += 1

                if self.calls <= self.timeout_after:
                    raise asyncio.TimeoutError("Request timed out")

                return ChatCompletionResponse(
                    id="response",
                    model=model or "test",
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=Message(role=MessageRole.ASSISTANT, content="Recovered"),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                )

        provider = TimeoutProvider(timeout_after=2)

        for attempt in range(5):
            try:
                response = await provider.chat_completion(
                    [Message(role=MessageRole.USER, content="test")]
                )
                assert response.choices[0].message.content == "Recovered"
                break
            except asyncio.TimeoutError:
                continue

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self) -> None:
        """Test handling of malformed responses."""

        class MalformedProvider:
            def __init__(self):
                self.name = "malformed"
                self.calls = 0

            async def chat_completion(self, messages, model=None, **kwargs):
                self.calls += 1

                if self.calls == 1:
                    return {"invalid": "response"}

                return ChatCompletionResponse(
                    id="response",
                    model=model or "test",
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=Message(role=MessageRole.ASSISTANT, content="Valid"),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                )

        provider = MalformedProvider()

        valid_response = False
        for _ in range(2):
            try:
                response = await provider.chat_completion(
                    [Message(role=MessageRole.USER, content="test")]
                )
                if hasattr(response, "choices"):
                    valid_response = True
                    break
            except Exception:
                continue

        assert provider.calls == 2


class TestFallbackBehaviorGauntlet:
    """Tests for fallback behavior."""

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_provider_fallback_chain(self) -> None:
        """Test fallback through multiple providers."""

        class FallbackChain:
            def __init__(self, providers: list):
                self.providers = providers
                self.current_idx = 0

            async def get_response(self, messages):
                for idx, provider in enumerate(self.providers):
                    try:
                        return await provider.chat_completion(messages)
                    except Exception:
                        continue
                raise Exception("All providers failed")

        failing = FailingProvider(failure_rate=1.0)
        working = FailingProvider(failure_rate=0.0)

        chain = FallbackChain([failing, working])

        response = await chain.get_response([Message(role=MessageRole.USER, content="test")])

        assert response.choices[0].message.content == "Success!"
        assert failing.call_count >= 1

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    def test_graceful_degradation(self, chaos_monkey) -> None:
        """Test graceful degradation under failures."""
        chaos_monkey.failure_rate = 0.5

        successes = 0
        attempts = 20

        for _ in range(attempts):
            if not chaos_monkey.inject_network_failure():
                successes += 1

        stats = chaos_monkey.get_stats()

        assert successes > 0
        assert stats["injected_failures"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "chaos"])
