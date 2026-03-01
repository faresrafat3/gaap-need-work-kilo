"""
Smart Router Module for GAAP System

Provides intelligent provider routing with:

Features:
    - 5 routing strategies (QUALITY_FIRST, COST_OPTIMIZED, SPEED_FIRST, BALANCED, SMART)
    - Provider scoring system
    - Task-based model recommendations
    - Priority and complexity-based routing
    - Budget management

Classes:
    - RoutingStrategy: Routing strategy enumeration
    - ProviderScore: Provider scoring dataclass
    - SmartRouter: Main router implementation

Usage:
    from gaap.routing import SmartRouter, RoutingStrategy

    router = SmartRouter(
        providers=[provider1, provider2],
        strategy=RoutingStrategy.BALANCED,
        budget_limit=50.0
    )

    decision = router.route(task, messages)
"""

# Router
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from gaap.core.exceptions import BudgetExceededError, NoAvailableProviderError
from gaap.core.types import (
    Message,
    ModelTier,
    ProviderType,
    RoutingDecision,
    Task,
    TaskComplexity,
    TaskPriority,
    TaskType,
)
from gaap.providers.base_provider import BaseProvider

# =============================================================================
# Helper Classes for Key Management
# =============================================================================


@dataclass
class KeyState:
    """Track state of a single API key for rate-limit awareness"""

    key_index: int
    last_used: float = 0.0
    requests_this_minute: int = 0
    requests_today: int = 0
    minute_window_start: float = 0.0
    day_start: float = 0.0
    consecutive_errors: int = 0
    is_exhausted: bool = False

    def __post_init__(self) -> None:
        now = time.time()
        self.minute_window_start = now
        self.day_start = now

    def reset_minute_if_needed(self) -> None:
        now = time.time()
        if now - self.minute_window_start >= 60:
            self.requests_this_minute = 0
            self.minute_window_start = now

    def is_available(self, rpm_limit: int = 10) -> bool:
        self.reset_minute_if_needed()
        if self.is_exhausted or self.consecutive_errors >= 5:
            return False
        return self.requests_this_minute < rpm_limit


# =============================================================================
# Constants
# =============================================================================

# Default scoring weights
DEFAULT_WEIGHTS = {
    "quality": 0.4,
    "cost": 0.3,
    "speed": 0.2,
    "availability": 0.1,
}

# =============================================================================
# Logger Setup
# =============================================================================


from gaap.core.logging import get_standard_logger as get_logger


class RoutingStrategy(Enum):
    """
    Routing strategy enumeration.

    Defines how providers are selected for tasks:

    Members:
        QUALITY_FIRST: Best quality regardless of cost
        COST_OPTIMIZED: Lowest cost option
        SPEED_FIRST: Fastest response time
        BALANCED: Balance between all factors
        SMART: Context-aware intelligent decision

    Usage:
        >>> strategy = RoutingStrategy.BALANCED
        >>> print(strategy.value)
        'balanced'
    """

    QUALITY_FIRST = "quality_first"  # أفضل جودة مهما كلف الأمر
    COST_OPTIMIZED = "cost_optimized"  # أقل تكلفة
    SPEED_FIRST = "speed_first"  # أسرع استجابة
    BALANCED = "balanced"  # توازن بين الجميع
    SMART = "smart"  # قرار ذكي بناءً على السياق


# =============================================================================
# Provider Scoring
# =============================================================================


@dataclass
class ProviderScore:
    """
    Provider scoring dataclass.

    Represents a scored provider with multiple criteria.

    Attributes:
        provider_name: Provider name
        model: Model name
        quality_score: Quality score (0.0-1.0)
        cost_score: Cost score (0.0-1.0)
        speed_score: Speed score (0.0-1.0)
        availability_score: Availability score (0.0-1.0)
        final_score: Weighted final score
        estimated_cost: Estimated cost in USD
        estimated_latency_ms: Estimated latency in milliseconds

    Usage:
        >>> score = ProviderScore(
        ...     provider_name="kimi",
        ...     model="llama-3.3-70b",
        ...     quality_score=0.9,
        ...     cost_score=1.0,
        ...     speed_score=0.95
        ... )
        >>> score.calculate_final_score({"quality": 0.4, "cost": 0.3, "speed": 0.2})
    """

    provider_name: str
    model: str
    quality_score: float = 0.0
    cost_score: float = 0.0
    speed_score: float = 0.0
    availability_score: float = 0.0
    final_score: float = 0.0
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0

    def calculate_final_score(self, weights: dict[str, float]) -> None:
        """
        Calculate weighted final score.

        Args:
            weights: Dictionary with scoring weights
            Default: quality=0.4, cost=0.3, speed=0.2, availability=0.1

        Example:
            >>> score.calculate_final_score({"quality": 0.4, "cost": 0.3, "speed": 0.2})
            >>> print(f"Final score: {score.final_score:.2f}")
        """
        self.final_score = (
            self.quality_score * weights.get("quality", 0.4)
            + self.cost_score * weights.get("cost", 0.3)
            + self.speed_score * weights.get("speed", 0.2)
            + self.availability_score * weights.get("availability", 0.1)
        )


# =============================================================================
# Model Recommendations
# =============================================================================

TASK_MODEL_RECOMMENDATIONS: dict[TaskType, dict[str, Any]] = {
    # مهام التخطيط الاستراتيجي
    TaskType.PLANNING: {
        "recommended_tier": ModelTier.TIER_1_STRATEGIC,
        "models": ["claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro"],
        "min_quality": 0.9,
    },
    # مهام كتابة الكود
    TaskType.CODE_GENERATION: {
        "recommended_tier": ModelTier.TIER_2_TACTICAL,
        "models": ["gpt-4o", "claude-3-5-sonnet", "llama-3.3-70b-versatile"],
        "min_quality": 0.85,
    },
    # مراجعة الكود
    TaskType.CODE_REVIEW: {
        "recommended_tier": ModelTier.TIER_2_TACTICAL,
        "models": ["gpt-4o-mini", "llama-3.1-70b-versatile", "claude-3-5-sonnet"],
        "min_quality": 0.8,
    },
    # التصحيح
    TaskType.DEBUGGING: {
        "recommended_tier": ModelTier.TIER_2_TACTICAL,
        "models": ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-flash"],
        "min_quality": 0.85,
    },
    # البحث
    TaskType.RESEARCH: {
        "recommended_tier": ModelTier.TIER_3_EFFICIENT,
        "models": ["gpt-4o-mini", "gemini-1.5-flash", "llama-3.1-8b-instant"],
        "min_quality": 0.7,
    },
    # التحليل
    TaskType.ANALYSIS: {
        "recommended_tier": ModelTier.TIER_2_TACTICAL,
        "models": ["claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro"],
        "min_quality": 0.85,
    },
    # الاختبار
    TaskType.TESTING: {
        "recommended_tier": ModelTier.TIER_3_EFFICIENT,
        "models": ["gpt-4o-mini", "llama-3.1-8b-instant", "gemini-1.5-flash"],
        "min_quality": 0.75,
    },
    # التوثيق
    TaskType.DOCUMENTATION: {
        "recommended_tier": ModelTier.TIER_3_EFFICIENT,
        "models": ["gpt-4o-mini", "llama-3.1-8b-instant", "gemini-1.5-flash-8b"],
        "min_quality": 0.7,
    },
}
"""
Task-based model recommendations.

Maps task types to recommended model tiers and specific models.
Used by SmartRouter for intelligent provider selection.

Example:
    >>> recommendations = TASK_MODEL_RECOMMENDATIONS[TaskType.CODE_GENERATION]
    >>> print(f"Recommended tier: {recommendations['recommended_tier']}")
    >>> print(f"Models: {recommendations['models']}")
"""

PRIORITY_MULTIPLIERS: dict[TaskPriority, float] = {
    TaskPriority.CRITICAL: 1.5,
    TaskPriority.HIGH: 1.25,
    TaskPriority.NORMAL: 1.0,
    TaskPriority.LOW: 0.8,
    TaskPriority.BACKGROUND: 0.6,
}
"""
Priority multipliers for resource allocation.

Higher priority tasks get more resources and better models.

Example:
    >>> multiplier = PRIORITY_MULTIPLIERS[TaskPriority.CRITICAL]
    >>> print(f"Critical task multiplier: {multiplier}")
    1.5
"""

COMPLEXITY_MODEL_TIER: dict[TaskComplexity, ModelTier] = {
    TaskComplexity.TRIVIAL: ModelTier.TIER_3_EFFICIENT,
    TaskComplexity.SIMPLE: ModelTier.TIER_3_EFFICIENT,
    TaskComplexity.MODERATE: ModelTier.TIER_2_TACTICAL,
    TaskComplexity.COMPLEX: ModelTier.TIER_2_TACTICAL,
    TaskComplexity.ARCHITECTURAL: ModelTier.TIER_1_STRATEGIC,
}
"""
Complexity-based model tier mapping.

Maps task complexity to appropriate model tier.
Higher complexity tasks require more capable models.

Example:
    >>> tier = COMPLEXITY_MODEL_TIER[TaskComplexity.ARCHITECTURAL]
    >>> print(f"Architectural tasks use: {tier}")
    ModelTier.TIER_1_STRATEGIC
"""


# =============================================================================
# Smart Router
# =============================================================================


class SmartRouter:
    """
    Smart Router for intelligent provider selection.

    Decides the best provider and model for each task based on:
    - Task complexity and priority
    - Available budget
    - Speed and quality requirements
    - User preferences
    - Provider health and availability

    Attributes:
        _providers: Dictionary of registered providers
        _strategy: Current routing strategy
        _budget_limit: Maximum budget limit
        _budget_spent: Amount of budget spent
        _quality_threshold: Minimum quality threshold
        _logger: Logger instance
        _routing_history: History of routing decisions
        _provider_stats: Statistics for each provider

    Usage:
        from gaap.routing import SmartRouter, RoutingStrategy

        router = SmartRouter(
            providers=[provider1, provider2],
            strategy=RoutingStrategy.BALANCED,
            budget_limit=100.0
        )

        decision = await router.route(messages, task)
    """

    def __init__(
        self,
        providers: list[BaseProvider] | None = None,
        strategy: RoutingStrategy = RoutingStrategy.SMART,
        budget_limit: float = 100.0,
        quality_threshold: float = 0.8,
    ) -> None:
        """
        Initialize SmartRouter.

        Args:
            providers: List of initial providers (optional)
            strategy: Routing strategy (default: SMART)
            budget_limit: Maximum budget limit (default: 100.0)
            quality_threshold: Minimum quality threshold (default: 0.8)
        """
        self._providers: dict[str, BaseProvider] = {}
        self._strategy = strategy
        self._budget_limit = budget_limit
        self._budget_spent = 0.0
        self._quality_threshold = quality_threshold
        self._logger = get_logger("gaap.router")

        # Statistics
        self._routing_history: list[RoutingDecision] = []
        self._provider_stats: dict[str, dict[str, Any]] = {}

        # تتبع حالة المفاتيح (v2.1 Sovereign Feature)
        self._provider_states: dict[str, list[KeyState]] = {}

        # v2: Adaptive Rate Limiter — self-adjusts based on provider success/failure
        self._adaptive_rate_limiter: Any = None
        try:
            from gaap.core.rate_limiter import AdaptiveRateLimiter, RateLimitConfig

            self._adaptive_rate_limiter = AdaptiveRateLimiter(
                RateLimitConfig(
                    requests_per_second=5.0,
                    burst_capacity=20,
                    min_rate=0.5,
                    max_rate=20.0,
                )
            )
        except Exception as e:
            self._logger.debug(f"Adaptive rate limiter init failed: {e}")

        # Register providers
        if providers:
            for provider in providers:
                self.register_provider(provider)

    def __repr__(self) -> str:
        return f"SmartRouter(providers={len(self._providers)}, strategy={self._strategy.name})"

    # =========================================================================
    # Provider Management
    # =========================================================================

    def register_provider(self, provider: BaseProvider) -> None:
        """
        Register a new provider.

        Args:
            provider: Provider instance to register

        Note:
            Initializes provider statistics tracking and key state management
        """
        self._providers[provider.name] = provider
        self._provider_stats[provider.name] = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0,
        }
        # تهيئة مفتاح واحد افتراضي لكل مزود
        if provider.name not in self._provider_states:
            self._provider_states[provider.name] = [KeyState(key_index=0)]

        self._logger.info(f"Registered provider: {provider.name}")

    def unregister_provider(self, provider_name: str) -> None:
        """
        Unregister a provider.

        Args:
            provider_name: Name of provider to remove

        Example:
            >>> router.unregister_provider("kimi")
        """
        if provider_name in self._providers:
            del self._providers[provider_name]
            del self._provider_stats[provider_name]
            self._logger.info(f"Unregistered provider: {provider_name}")

    def get_provider(self, provider_name: str) -> BaseProvider | None:
        """
        Get a provider by name.

        Args:
            provider_name: Provider name

        Returns:
            Provider instance or None if not found

        Example:
            >>> provider = router.get_provider("kimi")
        """
        return self._providers.get(provider_name)

    def get_all_providers(self) -> list[BaseProvider]:
        """
        Get all registered providers.

        Returns:
            List of all provider instances

        Example:
            >>> providers = router.get_all_providers()
        """
        return list(self._providers.values())

    # =========================================================================
    # Routing Logic
    # =========================================================================

    async def route(
        self,
        messages: list[Message],
        task: Task | None = None,
        preferred_model: str | None = None,
        excluded_providers: list[str] | None = None,
        **kwargs: Any,
    ) -> RoutingDecision:
        """
        Route request to best provider.

        Args:
            messages: List of messages to process
            task: Associated task (optional)
            preferred_model: Preferred model name (optional)
            excluded_providers: Providers to exclude (optional)
            **kwargs: Additional routing parameters

        Returns:
            RoutingDecision with selected provider and model

        Raises:
            BudgetExceededError: If estimated cost exceeds budget
            NoAvailableProviderError: If no suitable provider found

        Process:
            1. Analyze task/messages
            2. Check budget
            3. Get candidate providers
            4. Score candidates
            5. Select best provider
            6. Record decision

        Example:
            >>> decision = await router.route(messages, task)
            >>> print(f"Selected: {decision.selected_provider}/{decision.selected_model}")
        """
        start_time = time.time()
        excluded_providers = excluded_providers or []

        # Analyze task
        task_info = self._analyze_task(task) if task else self._analyze_messages(messages)

        # Check budget
        estimated_cost = self._estimate_cost(messages, task_info)
        if self._budget_spent + estimated_cost > self._budget_limit:
            raise BudgetExceededError(
                budget=self._budget_limit - self._budget_spent, required=estimated_cost
            )

        # Get candidates
        candidates = await self._get_candidates(
            task_info=task_info,
            preferred_model=preferred_model,
            excluded_providers=excluded_providers,
        )

        if not candidates:
            raise NoAvailableProviderError(
                requirements={
                    "task_type": task_info.get("type", "unknown"),
                    "min_tier": task_info.get("min_tier", ModelTier.TIER_2_TACTICAL).name,
                }
            )

        # Score candidates
        scored_candidates = await self._score_candidates(candidates, task_info)

        # Select best
        best = scored_candidates[0]

        # Build routing decision
        decision = RoutingDecision(
            selected_provider=best.provider_name,
            selected_model=best.model,
            reasoning=self._build_reasoning(best, task_info),
            alternatives=[c.provider_name for c in scored_candidates[1:4]],
            estimated_cost=best.estimated_cost,
            estimated_latency_ms=best.estimated_latency_ms,
            confidence=best.final_score / 100.0,
            metadata={
                "task_type": task_info.get("type"),
                "strategy": self._strategy.value,
                "candidates_count": len(scored_candidates),
            },
        )

        # Record decision
        self._routing_history.append(decision)

        self._logger.info(
            f"Routed to {best.provider_name}/{best.model} "
            f"(score: {best.final_score:.2f}, cost: ${best.estimated_cost:.4f})"
        )

        return decision

    def _analyze_task(self, task: Task) -> dict[str, Any]:
        """
        Analyze task to determine requirements.

        Args:
            task: Task to analyze

        Returns:
            Dictionary with task analysis results including:
            - type: Task type name
            - priority: Task priority
            - complexity: Task complexity name
            - min_tier: Minimum required model tier
            - min_quality: Minimum quality threshold
            - priority_multiplier: Priority multiplier
            - recommended_models: List of recommended models

        Example:
            >>> task_info = router._analyze_task(task)
            >>> print(f"Tier: {task_info['min_tier']}")
        """
        task_type = task.type
        priority = task.priority
        complexity = task.complexity

        # Get recommendations
        recommendations: dict[str, Any] = TASK_MODEL_RECOMMENDATIONS.get(
            task_type, {"recommended_tier": ModelTier.TIER_2_TACTICAL, "min_quality": 0.8}
        )

        # Adjust tier based on complexity
        rec_tier = recommendations.get("recommended_tier", ModelTier.TIER_2_TACTICAL)
        base_tier: ModelTier = (
            rec_tier if isinstance(rec_tier, ModelTier) else ModelTier.TIER_2_TACTICAL
        )
        complexity_tier: ModelTier = COMPLEXITY_MODEL_TIER.get(complexity, base_tier) or base_tier

        # Choose higher tier
        tier_order: list[ModelTier] = [
            ModelTier.TIER_4_PRIVATE,
            ModelTier.TIER_3_EFFICIENT,
            ModelTier.TIER_2_TACTICAL,
            ModelTier.TIER_1_STRATEGIC,
        ]
        final_tier: ModelTier = (
            base_tier
            if tier_order.index(base_tier) >= tier_order.index(complexity_tier)
            else complexity_tier
        )

        # Honor force_model_tier from healing system L3_PIVOT
        if hasattr(task, "constraints") and task.constraints:
            force_tier = task.constraints.get("force_model_tier")
            if force_tier == "higher":
                current_idx = tier_order.index(final_tier)
                if current_idx < len(tier_order) - 1:
                    final_tier = tier_order[current_idx + 1]
                    self._logger.info(f"L3_PIVOT: Bumped model tier to {final_tier.name}")

        return {
            "type": task_type.name,
            "priority": priority,
            "complexity": complexity.name,
            "min_tier": final_tier,
            "min_quality": recommendations.get("min_quality", 0.8),
            "priority_multiplier": PRIORITY_MULTIPLIERS.get(priority, 1.0),
            "recommended_models": recommendations.get("models", []),
        }

    def _analyze_messages(self, messages: list[Message]) -> dict[str, Any]:
        """
        Analyze messages when no task is provided.

        Args:
            messages: List of messages to analyze

        Returns:
            Dictionary with message analysis results

        Note:
            Estimates complexity from message length
        """
        # Estimate complexity from length
        total_length = sum(len(m.content) for m in messages)

        if total_length < 500:
            complexity = TaskComplexity.SIMPLE
            tier = ModelTier.TIER_3_EFFICIENT
        elif total_length < 2000:
            complexity = TaskComplexity.MODERATE
            tier = ModelTier.TIER_2_TACTICAL
        else:
            complexity = TaskComplexity.COMPLEX
            tier = ModelTier.TIER_2_TACTICAL

        return {
            "type": "UNKNOWN",
            "priority": TaskPriority.NORMAL,
            "complexity": complexity.name,
            "min_tier": tier,
            "min_quality": 0.8,
            "priority_multiplier": 1.0,
            "recommended_models": [],
        }

    def _estimate_cost(self, messages: list[Message], task_info: dict[str, Any]) -> float:
        """
        Estimate request cost.

        Args:
            messages: List of messages
            task_info: Task analysis results

        Returns:
            Estimated cost in USD

        Note:
            Uses average cost per 1K tokens
        """
        # Estimate tokens
        input_tokens = sum(len(m.content.split()) * 1.5 for m in messages)
        output_tokens = 500  # Average estimate

        # Estimate cost (using average)
        avg_cost_per_1k = 0.01  # Approximate average

        return (input_tokens + output_tokens) * avg_cost_per_1k / 1000

    async def _get_candidates(
        self,
        task_info: dict[str, Any],
        preferred_model: str | None,
        excluded_providers: list[str],
    ) -> list[tuple[BaseProvider, str]]:
        """الحصول على المرشحين مع فحص التوفر والقيود (Sovereign Filtering)"""
        candidates = []
        min_tier = task_info.get("min_tier", ModelTier.TIER_2_TACTICAL)
        recommended = task_info.get("recommended_models", [])

        for provider_name, provider in self._providers.items():
            if provider_name in excluded_providers:
                continue

            # 1. فحص توفر المفاتيح (Rate Limit Awareness)
            states = self._provider_states.get(provider_name, [])
            if states and not any(s.is_available() for s in states):
                self._logger.debug(
                    f"Router: Skipping provider {provider_name} (All keys exhausted)"
                )
                continue

            # 2. فحص النماذج المتاحة
            for model in provider.get_available_models():
                # Skip if preferred model specified
                if preferred_model and model != preferred_model:
                    continue

                # Check model tier
                model_tier = provider.get_model_tier(model)
                if self._tier_sufficient(model_tier, min_tier):
                    candidates.append((provider, model))

        # Sort by recommendations
        if recommended:
            candidates.sort(key=lambda x: (recommended.index(x[1]) if x[1] in recommended else 999))

        return candidates

    def _tier_sufficient(self, model_tier: ModelTier, required_tier: ModelTier) -> bool:
        """
        Check if model tier meets requirements.

        Args:
            model_tier: Model's actual tier
            required_tier: Minimum required tier

        Returns:
            True if model tier is sufficient

        Example:
            >>> router._tier_sufficient(ModelTier.TIER_2_TACTICAL, ModelTier.TIER_3_EFFICIENT)
            True
        """
        tier_order = [
            ModelTier.TIER_4_PRIVATE,
            ModelTier.TIER_3_EFFICIENT,
            ModelTier.TIER_2_TACTICAL,
            ModelTier.TIER_1_STRATEGIC,
        ]
        return tier_order.index(model_tier) >= tier_order.index(required_tier)

    async def _score_candidates(
        self, candidates: list[tuple[BaseProvider, str]], task_info: dict[str, Any]
    ) -> list[ProviderScore]:
        """
        Score all candidates.

        Args:
            candidates: List of (provider, model) tuples
            task_info: Task analysis results

        Returns:
            List of ProviderScore objects, sorted by final_score (descending)

        Example:
            >>> scores = await router._score_candidates(candidates, task_info)
        """
        scores = []

        # Get weights based on strategy
        weights = self._get_strategy_weights()

        for provider, model in candidates:
            score = ProviderScore(provider_name=provider.name, model=model)

            # Quality score
            score.quality_score = self._score_quality(provider, model, task_info)

            # Cost score
            score.cost_score, score.estimated_cost = self._score_cost(provider, model, task_info)

            # Speed score
            score.speed_score, score.estimated_latency_ms = self._score_speed(provider, model)

            # Availability score
            score.availability_score = self._score_availability(provider)

            # Calculate final score
            score.calculate_final_score(weights)

            scores.append(score)

        # Sort descending by final score
        scores.sort(key=lambda x: x.final_score, reverse=True)

        return scores

    def _get_strategy_weights(self) -> dict[str, float]:
        """
        Get scoring weights based on current strategy.

        Returns:
            Dictionary with quality, cost, speed, availability weights

        Example:
            >>> weights = router._get_strategy_weights()
            >>> print(weights['quality'])
            0.4
        """
        weights = {
            RoutingStrategy.QUALITY_FIRST: {
                "quality": 0.6,
                "cost": 0.1,
                "speed": 0.2,
                "availability": 0.1,
            },
            RoutingStrategy.COST_OPTIMIZED: {
                "quality": 0.3,
                "cost": 0.5,
                "speed": 0.1,
                "availability": 0.1,
            },
            RoutingStrategy.SPEED_FIRST: {
                "quality": 0.2,
                "cost": 0.1,
                "speed": 0.6,
                "availability": 0.1,
            },
            RoutingStrategy.BALANCED: {
                "quality": 0.35,
                "cost": 0.25,
                "speed": 0.3,
                "availability": 0.1,
            },
            RoutingStrategy.SMART: {"quality": 0.4, "cost": 0.3, "speed": 0.2, "availability": 0.1},
        }
        return weights.get(self._strategy, weights[RoutingStrategy.SMART])

    def _score_quality(
        self, provider: BaseProvider, model: str, task_info: dict[str, Any]
    ) -> float:
        """
        Score provider/model quality.

        Args:
            provider: Provider instance
            model: Model name
            task_info: Task analysis results

        Returns:
            Quality score (0-100)
        """
        tier = provider.get_model_tier(model)
        min_quality = task_info.get("min_quality", 0.8)

        # Base score from tier
        tier_scores = {
            ModelTier.TIER_1_STRATEGIC: 100,
            ModelTier.TIER_2_TACTICAL: 85,
            ModelTier.TIER_3_EFFICIENT: 70,
            ModelTier.TIER_4_PRIVATE: 75,
        }

        base_score = tier_scores.get(tier, 70)

        # Penalize if below min quality threshold
        if base_score / 100.0 < min_quality:
            base_score -= 10

        # Adjust for recommendations
        if model in task_info.get("recommended_models", []):
            base_score += 10

        # Factor in historical performance if available
        provider_name = getattr(provider, "name", str(provider))
        stats = self._provider_stats.get(provider_name, {})
        total = stats.get("success", 0) + stats.get("failure", 0)
        if total >= 5:  # Only use stats if we have enough data
            success_rate = stats.get("success", 0) / total
            base_score = base_score * 0.7 + (success_rate * 100) * 0.3

        return min(base_score, 100)

    def _score_cost(
        self, provider: BaseProvider, model: str, task_info: dict[str, Any]
    ) -> tuple[float, float]:
        """
        Score provider/model cost using tier-based differentiation.

        Args:
            provider: Provider instance
            model: Model name
            task_info: Task analysis results

        Returns:
            Tuple of (cost_score, estimated_cost)
        """
        tier = provider.get_model_tier(model)

        # Differentiate cost by tier (approximate $/1K tokens)
        tier_cost_map = {
            ModelTier.TIER_1_STRATEGIC: 0.03,  # Premium models
            ModelTier.TIER_2_TACTICAL: 0.005,  # Mid-tier models
            ModelTier.TIER_3_EFFICIENT: 0.001,  # Efficient models
            ModelTier.TIER_4_PRIVATE: 0.0,  # Local/private models
        }

        cost_per_1k = tier_cost_map.get(tier, 0.005)
        estimated_tokens = 2000  # Average estimate
        estimated_cost = estimated_tokens * cost_per_1k / 1000

        # Inverse score (lower cost = higher score)
        if estimated_cost < 0.001:
            cost_score = 100
        elif estimated_cost < 0.01:
            cost_score = 85
        elif estimated_cost < 0.05:
            cost_score = 65
        else:
            cost_score = 40

        return cost_score, estimated_cost

    def _score_speed(self, provider: BaseProvider, model: str) -> tuple[float, float]:
        """
        Score provider speed.

        Args:
            provider: Provider instance
            model: Model name

        Returns:
            Tuple of (speed_score, estimated_latency_ms)
        """
        # Estimates by provider type
        latency_estimates = {
            ProviderType.CHAT_BASED: (5000, 60),  # Slower
            ProviderType.FREE_TIER: (1500, 80),  # Medium-fast
            ProviderType.PAID: (2000, 75),  # Medium
            ProviderType.LOCAL: (3000, 70),  # Depends
        }

        latency, score = latency_estimates.get(provider.provider_type, (3000, 70))

        # Kimi is much faster
        if provider.name.lower() == "kimi":
            latency = 200
            score = 100

        return score, latency

    def _score_availability(self, provider: BaseProvider) -> float:
        """
        Score provider availability.

        Args:
            provider: Provider instance

        Returns:
            Availability score (0-100) based on success rate
        """
        stats = self._provider_stats.get(provider.name, {})

        if stats.get("requests", 0) == 0:
            return 100  # No data, assume available

        success_rate: float = float(stats["successes"]) / float(stats["requests"])
        return success_rate * 100

    def _build_reasoning(self, best: ProviderScore, task_info: dict[str, Any]) -> str:
        """
        Build reasoning for routing decision.

        Args:
            best: Best provider score
            task_info: Task analysis results

        Returns:
            Human-readable reasoning string
        """
        reasons: list[str] = []

        reasons.append(
            f"Selected {best.provider_name}/{best.model} with score {best.final_score:.1f}/100"
        )

        if best.quality_score >= 90:
            reasons.append("High quality match for task requirements")

        if best.cost_score >= 80:
            reasons.append("Cost-effective option")

        if best.speed_score >= 90:
            reasons.append("Fast response expected")

        return ". ".join(reasons)

    # =========================================================================
    # Budget Management
    # =========================================================================

    def set_budget(self, limit: float) -> None:
        """
        Set budget limit.

        Args:
            limit: New budget limit in USD

        Example:
            >>> router.set_budget(50.0)
        """
        self._budget_limit = limit

    def record_spending(self, amount: float) -> None:
        """
        Record budget spending.

        Args:
            amount: Amount spent in USD

        Example:
            >>> router.record_spending(0.05)
        """
        self._budget_spent += amount

    def get_budget_status(self) -> dict[str, float]:
        """
        Get current budget status.

        Returns:
            Dictionary with limit, spent, remaining, utilization

        Example:
            >>> status = router.get_budget_status()
            >>> print(f"Remaining: ${status['remaining']:.2f}")
        """
        return {
            "limit": self._budget_limit,
            "spent": self._budget_spent,
            "remaining": self._budget_limit - self._budget_spent,
            "utilization": self._budget_spent / self._budget_limit if self._budget_limit > 0 else 0,
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def record_result(
        self, provider_name: str, success: bool, latency_ms: float, cost: float
    ) -> None:
        """
        Record request result for statistics.

        Args:
            provider_name: Provider name
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            cost: Request cost in USD

        Note:
            Updates provider statistics including success rate and average latency

        Example:
            >>> router.record_result("kimi", True, 250.0, 0.001)
        """
        if provider_name not in self._provider_stats:
            return

        stats = self._provider_stats[provider_name]
        stats["requests"] += 1

        if success:
            stats["successes"] += 1
            stats["total_cost"] += cost
            # Update average latency
            old_avg = stats["avg_latency"]
            n = stats["successes"]
            stats["avg_latency"] = old_avg + (latency_ms - old_avg) / n
        else:
            stats["failures"] += 1

    def get_routing_stats(self) -> dict[str, Any]:
        """
        Get comprehensive routing statistics.

        Returns:
            Dictionary with total_routes, providers, budget

        Example:
            >>> stats = router.get_routing_stats()
            >>> print(f"Total routes: {stats['total_routes']}")
        """
        return {
            "total_routes": len(self._routing_history),
            "providers": self._provider_stats.copy(),
            "budget": self.get_budget_status(),
        }

    # =========================================================================
    # Strategy Management
    # =========================================================================

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """
        Change routing strategy.

        Args:
            strategy: New routing strategy

        Example:
            >>> router.set_strategy(RoutingStrategy.COST_OPTIMIZED)
        """
        self._strategy = strategy
        self._logger.info(f"Routing strategy changed to: {strategy.value}")

    def get_strategy(self) -> RoutingStrategy:
        """
        Get current routing strategy.

        Returns:
            Current routing strategy

        Example:
            >>> strategy = router.get_strategy()
            >>> print(strategy.value)
        """
        return self._strategy


# =============================================================================
# Convenience Functions
# =============================================================================


def create_router(
    providers: list[BaseProvider] | None = None,
    strategy: str = "smart",
    budget: float = 100.0,
) -> SmartRouter:
    """
    Create a SmartRouter instance easily.

    Args:
        providers: List of providers (optional)
        strategy: Routing strategy name (default: "smart")
        budget: Budget limit in USD (default: 100.0)

    Returns:
        Configured SmartRouter instance

    Example:
        >>> router = create_router(providers=[kimi], strategy="balanced", budget=50.0)
    """
    strategy_enum = RoutingStrategy(strategy.lower())
    return SmartRouter(providers=providers, strategy=strategy_enum, budget_limit=budget)
