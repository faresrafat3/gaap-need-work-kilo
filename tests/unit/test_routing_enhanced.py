"""
Enhanced tests for Router module
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gaap.routing.router import (
    KeyState,
    RoutingStrategy,
    ProviderScore,
    SmartRouter,
)


class TestKeyState:
    """Tests for KeyState"""

    def test_create_key_state(self):
        """Test creating a key state"""
        state = KeyState(key_index=0)
        assert state.key_index == 0
        assert state.requests_this_minute == 0
        assert state.requests_today == 0

    def test_key_state_is_available(self):
        """Test key state availability"""
        state = KeyState(key_index=0)
        state.requests_this_minute = 5

        available = state.is_available(rpm_limit=10)
        assert available is True

    def test_key_state_not_available_rpm(self):
        """Test key state not available due to RPM"""
        state = KeyState(key_index=0)
        state.requests_this_minute = 15

        available = state.is_available(rpm_limit=10)
        assert available is False

    def test_key_state_not_available_exhausted(self):
        """Test key state not available when exhausted"""
        state = KeyState(key_index=0)
        state.is_exhausted = True

        available = state.is_available()
        assert available is False

    def test_key_state_not_available_errors(self):
        """Test key state not available due to errors"""
        state = KeyState(key_index=0)
        state.consecutive_errors = 10

        available = state.is_available()
        assert available is False


class TestRoutingStrategy:
    """Tests for RoutingStrategy enum"""

    def test_routing_strategy_values(self):
        """Test routing strategy values"""
        assert RoutingStrategy.QUALITY_FIRST is not None
        assert RoutingStrategy.COST_OPTIMIZED is not None
        assert RoutingStrategy.SPEED_FIRST is not None
        assert RoutingStrategy.BALANCED is not None
        assert RoutingStrategy.SMART is not None

    def test_routing_strategy_string_values(self):
        """Test routing strategy string values"""
        assert RoutingStrategy.QUALITY_FIRST.value == "quality_first"
        assert RoutingStrategy.COST_OPTIMIZED.value == "cost_optimized"
        assert RoutingStrategy.SPEED_FIRST.value == "speed_first"
        assert RoutingStrategy.BALANCED.value == "balanced"
        assert RoutingStrategy.SMART.value == "smart"


class TestProviderScore:
    """Tests for ProviderScore"""

    def test_create_provider_score(self):
        """Test creating a provider score"""
        score = ProviderScore(
            provider_name="test-provider",
            model="test-model",
            quality_score=0.9,
            cost_score=0.8,
            speed_score=0.7,
            availability_score=1.0,
        )
        assert score.provider_name == "test-provider"
        assert score.quality_score == 0.9

    def test_provider_score_calculation(self):
        """Test provider score calculation"""
        score = ProviderScore(
            provider_name="test",
            model="model",
            quality_score=0.9,
            cost_score=0.8,
            speed_score=0.7,
            availability_score=1.0,
        )

        weights = {"quality": 0.4, "cost": 0.3, "speed": 0.2, "availability": 0.1}
        score.calculate_final_score(weights)

        expected = 0.9 * 0.4 + 0.8 * 0.3 + 0.7 * 0.2 + 1.0 * 0.1
        assert abs(score.final_score - expected) < 0.001

    def test_provider_score_defaults(self):
        """Test provider score defaults"""
        score = ProviderScore(provider_name="test", model="model")
        assert score.quality_score == 0.0
        assert score.cost_score == 0.0
        assert score.speed_score == 0.0
        assert score.final_score == 0.0


class TestSmartRouter:
    """Tests for SmartRouter"""

    def test_router_initialization(self):
        """Test router initialization"""
        router = SmartRouter()
        assert router is not None

    def test_router_initialization_with_strategy(self):
        """Test router initialization with strategy"""
        router = SmartRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        assert router is not None

    def test_router_initialization_with_providers(self):
        """Test router initialization with providers"""
        mock_provider = MagicMock()
        mock_provider.name = "test-provider"

        router = SmartRouter(providers=[mock_provider])
        assert router is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
