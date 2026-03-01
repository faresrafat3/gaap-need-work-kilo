"""
GAAP Routing Module
===================

Intelligent provider selection and failover:

Smart Router:
    - RoutingStrategy: COST, SPEED, QUALITY, BALANCED, SMART
    - SmartRouter: AI-powered provider selection
    - ProviderScore: Provider quality metrics

Fallback System:
    - FallbackManager: Automatic failover on errors
    - FallbackChain: Ordered fallback providers
    - ProviderHealth: Health tracking per provider
    - CircuitBreaker: Prevent cascading failures

Pricing Table:
    - MODEL_PRICING: Live pricing per model
    - ModelPricing: Pricing data class
    - estimate_cost: Cost calculation utilities

Features:
    - Task complexity analysis
    - Budget-aware routing
    - Multi-key rotation
    - Rate limit handling
    - Quality-based selection

Usage:
    from gaap.routing import SmartRouter, RoutingStrategy

    router = SmartRouter(providers, strategy=RoutingStrategy.SMART)
    decision = await router.route(messages, task)
"""

from .cascade_router import (
    CascadeResult,
    CascadeRouter,
    QualityGate,
    QualityGateResult,
    create_cascade_router,
)
from .fallback import (
    CircuitBreaker,
    FallbackChain,
    FallbackConfig,
    FallbackManager,
    ProviderHealth,
    create_fallback_manager,
)
from .pricing_table import (
    MODEL_PRICING,
    ModelPricing,
    count_tokens,
    estimate_cost,
    get_all_pricing,
    get_best_value_model,
    get_cheapest_model,
    get_pricing,
    get_pricing_by_provider,
)
from .router import (
    COMPLEXITY_MODEL_TIER,
    PRIORITY_MULTIPLIERS,
    TASK_MODEL_RECOMMENDATIONS,
    ProviderScore,
    RoutingStrategy,
    SmartRouter,
    create_router,
)

__all__ = [
    # Router
    "SmartRouter",
    "RoutingStrategy",
    "ProviderScore",
    "TASK_MODEL_RECOMMENDATIONS",
    "PRIORITY_MULTIPLIERS",
    "COMPLEXITY_MODEL_TIER",
    "create_router",
    # Fallback
    "FallbackManager",
    "FallbackConfig",
    "FallbackChain",
    "ProviderHealth",
    "CircuitBreaker",
    "create_fallback_manager",
    # Pricing
    "MODEL_PRICING",
    "ModelPricing",
    "get_pricing",
    "get_all_pricing",
    "get_pricing_by_provider",
    "estimate_cost",
    "get_cheapest_model",
    "get_best_value_model",
    "count_tokens",
]
