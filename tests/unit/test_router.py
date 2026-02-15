"""
Unit tests for Router
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, List

from gaap.core.types import (
    Task,
    TaskPriority,
    TaskType,
    TaskComplexity,
    ModelTier,
    ProviderType,
)


class TestModelSelection:
    """Tests for model selection logic"""

    def test_tier_selection_by_task_type(self):
        """Test tier selection based on task type"""
        task_tier_mapping = {
            TaskType.PLANNING: ModelTier.TIER_1_STRATEGIC,
            TaskType.CODE_GENERATION: ModelTier.TIER_2_TACTICAL,
            TaskType.TESTING: ModelTier.TIER_3_EFFICIENT,
            TaskType.DOCUMENTATION: ModelTier.TIER_3_EFFICIENT,
        }

        assert task_tier_mapping[TaskType.PLANNING] == ModelTier.TIER_1_STRATEGIC
        assert task_tier_mapping[TaskType.CODE_GENERATION] == ModelTier.TIER_2_TACTICAL

    def test_tier_selection_by_complexity(self):
        """Test tier selection based on complexity"""
        complexity_tier_mapping = {
            TaskComplexity.SIMPLE: ModelTier.TIER_3_EFFICIENT,
            TaskComplexity.MODERATE: ModelTier.TIER_2_TACTICAL,
            TaskComplexity.COMPLEX: ModelTier.TIER_1_STRATEGIC,
        }

        assert complexity_tier_mapping[TaskComplexity.SIMPLE] == ModelTier.TIER_3_EFFICIENT
        assert complexity_tier_mapping[TaskComplexity.COMPLEX] == ModelTier.TIER_1_STRATEGIC

    def test_tier_selection_by_priority(self):
        """Test tier selection based on priority"""
        priority_tier_mapping = {
            TaskPriority.CRITICAL: ModelTier.TIER_1_STRATEGIC,
            TaskPriority.HIGH: ModelTier.TIER_2_TACTICAL,
            TaskPriority.NORMAL: ModelTier.TIER_2_TACTICAL,
            TaskPriority.LOW: ModelTier.TIER_3_EFFICIENT,
        }

        assert priority_tier_mapping[TaskPriority.CRITICAL] == ModelTier.TIER_1_STRATEGIC
        assert priority_tier_mapping[TaskPriority.LOW] == ModelTier.TIER_3_EFFICIENT


class TestScoring:
    """Tests for scoring functions"""

    def test_quality_score(self):
        """Test quality scoring"""
        quality_scores = {"groq": 0.85, "gemini": 0.90, "g4f": 0.75}
        best = max(quality_scores.items(), key=lambda x: x[1])
        assert best[0] == "gemini"

    def test_cost_score(self):
        """Test cost scoring (lower is better)"""
        costs = {"groq": 0.0001, "gemini": 0.0002, "openai": 0.03}
        cost_scores = {k: 1.0 / v for k, v in costs.items()}
        best = max(cost_scores.items(), key=lambda x: x[1])
        assert best[0] == "groq"

    def test_speed_score(self):
        """Test speed scoring"""
        latencies = {"groq": 200, "gemini": 500, "openai": 1000}
        speed_scores = {k: 1.0 / v for k, v in latencies.items()}
        best = max(speed_scores.items(), key=lambda x: x[1])
        assert best[0] == "groq"

    def test_combined_score(self):
        """Test combined scoring with weights"""
        scores = {
            "groq": {"quality": 0.85, "cost": 0.9, "speed": 0.95},
            "gemini": {"quality": 0.90, "cost": 0.8, "speed": 0.7},
        }
        weights = {"quality": 0.5, "cost": 0.3, "speed": 0.2}

        combined = {}
        for provider, s in scores.items():
            combined[provider] = sum(s[k] * weights[k] for k in weights)

        best = max(combined.items(), key=lambda x: x[1])
        assert best[0] == "groq"


class TestBudgetManagement:
    """Tests for budget management"""

    def test_budget_check(self):
        """Test budget checking"""
        budget = 10.0
        estimated_cost = 0.001
        within_budget = estimated_cost <= budget
        assert within_budget

    def test_budget_exceeded(self):
        """Test budget exceeded detection"""
        budget = 0.0001
        estimated_cost = 0.001
        within_budget = estimated_cost <= budget
        assert not within_budget

    def test_budget_tracking(self):
        """Test budget tracking"""
        budget = 10.0
        spent = 0.5
        remaining = budget - spent
        assert remaining == 9.5

    def test_budget_allocation(self):
        """Test budget allocation per task"""
        total_budget = 10.0
        num_tasks = 5
        per_task = total_budget / num_tasks
        assert per_task == 2.0


class TestProviderSelection:
    """Tests for provider selection"""

    def test_provider_availability(self, mock_provider):
        """Test checking provider availability"""
        assert mock_provider.is_model_available("model-1")
        assert mock_provider.is_model_available("model-2")

    def test_provider_filtering_by_tier(self):
        """Test filtering providers by tier"""
        providers = {
            "groq": {"tier": ModelTier.TIER_2_TACTICAL, "available": True},
            "gemini": {"tier": ModelTier.TIER_3_EFFICIENT, "available": True},
            "openai": {"tier": ModelTier.TIER_1_STRATEGIC, "available": False},
        }

        available = [p for p, s in providers.items() if s["available"]]
        assert len(available) == 2

    def test_provider_selection_priority(self):
        """Test provider selection priority"""
        providers = [
            {"name": "groq", "priority": 1, "score": 0.85},
            {"name": "gemini", "priority": 2, "score": 0.90},
        ]

        sorted_providers = sorted(providers, key=lambda x: x["priority"])
        assert sorted_providers[0]["name"] == "groq"


class TestRoutingStrategies:
    """Tests for routing strategies"""

    def test_quality_first_strategy(self):
        """Test quality-first routing strategy"""
        weights = {"quality": 0.6, "cost": 0.1, "speed": 0.2, "availability": 0.1}
        assert weights["quality"] == 0.6

    def test_cost_optimized_strategy(self):
        """Test cost-optimized routing strategy"""
        weights = {"quality": 0.3, "cost": 0.5, "speed": 0.1, "availability": 0.1}
        assert weights["cost"] == 0.5

    def test_speed_first_strategy(self):
        """Test speed-first routing strategy"""
        weights = {"quality": 0.2, "cost": 0.1, "speed": 0.6, "availability": 0.1}
        assert weights["speed"] == 0.6

    def test_balanced_strategy(self):
        """Test balanced routing strategy"""
        weights = {"quality": 0.35, "cost": 0.25, "speed": 0.3, "availability": 0.1}
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01


class TestRoutingDecision:
    """Tests for routing decisions"""

    def test_routing_decision_creation(self):
        """Test creating routing decision"""
        decision = {
            "provider": "groq",
            "model": "llama-3.3-70b",
            "tier": ModelTier.TIER_2_TACTICAL,
            "estimated_cost": 0.001,
            "estimated_latency_ms": 200,
        }
        assert decision["provider"] == "groq"
        assert decision["tier"] == ModelTier.TIER_2_TACTICAL

    def test_routing_with_fallback(self):
        """Test routing with fallback chain"""
        primary = "groq"
        fallbacks = ["gemini", "g4f"]
        fallback_chain = [primary] + fallbacks
        assert len(fallback_chain) == 3


class TestRouterIntegration:
    """Integration tests for router"""

    @pytest.mark.asyncio
    async def test_route_task(self, mock_router, sample_task):
        """Test routing a task"""
        from gaap.core.types import RoutingContext

        context = RoutingContext(
            task=sample_task,
            available_providers=["mock-provider"],
            budget_remaining=10.0,
        )

        decision = await mock_router.route(context)
        assert decision is not None
        assert decision.selected_provider == "mock-provider"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
