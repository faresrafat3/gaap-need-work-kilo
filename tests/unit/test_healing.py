"""
Unit tests for GAAP healing system
"""

import asyncio

import pytest

from gaap.core.exceptions import ProviderRateLimitError, ProviderTimeoutError
from gaap.core.types import HealingLevel


class TestHealingLevels:
    """Test healing level enumeration"""

    def test_healing_levels_order(self):
        """Test healing levels are ordered correctly"""
        levels = list(HealingLevel)
        assert levels[0] == HealingLevel.L1_RETRY
        assert levels[-1] == HealingLevel.L5_HUMAN_ESCALATION

    def test_healing_level_names(self):
        """Test healing level names"""
        assert HealingLevel.L1_RETRY.name == "L1_RETRY"
        assert HealingLevel.L2_REFINE.name == "L2_REFINE"
        assert HealingLevel.L3_PIVOT.name == "L3_PIVOT"
        assert HealingLevel.L4_STRATEGY_SHIFT.name == "L4_STRATEGY_SHIFT"
        assert HealingLevel.L5_HUMAN_ESCALATION.name == "L5_HUMAN_ESCALATION"


class TestErrorRecoverability:
    """Test error recoverability logic"""

    def test_rate_limit_recoverable(self):
        """Test rate limit errors are recoverable"""
        exc = ProviderRateLimitError(provider_name="test", retry_after=60)
        assert exc.recoverable

    def test_timeout_recoverable(self):
        """Test timeout errors are recoverable"""
        exc = ProviderTimeoutError(provider_name="test", timeout_seconds=30)
        assert exc.recoverable

    def test_exception_details(self):
        """Test exception details"""
        exc = ProviderRateLimitError(
            provider_name="groq",
            retry_after=120,
        )
        assert exc.details["provider"] == "groq"
        assert exc.details["retry_after_seconds"] == 120


class TestHealingWorkflow:
    """Test healing workflow scenarios"""

    @pytest.mark.asyncio
    async def test_retry_scenario(self):
        """Test L1 retry scenario"""
        attempts = 0
        max_attempts = 3

        async def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ProviderRateLimitError(provider_name="test")
            return "success"

        result = None
        for _ in range(max_attempts):
            try:
                result = await flaky_operation()
                break
            except ProviderRateLimitError:
                await asyncio.sleep(0.01)

        assert result == "success"
        assert attempts == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
