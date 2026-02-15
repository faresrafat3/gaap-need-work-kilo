# Router
import logging
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
# Logger Setup
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """إنشاء مسجل"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# Routing Strategy
# =============================================================================

class RoutingStrategy(Enum):
    """استراتيجيات التوجيه"""
    QUALITY_FIRST = "quality_first"      # أفضل جودة مهما كلف الأمر
    COST_OPTIMIZED = "cost_optimized"    # أقل تكلفة
    SPEED_FIRST = "speed_first"          # أسرع استجابة
    BALANCED = "balanced"                # توازن بين الجميع
    SMART = "smart"                      # قرار ذكي بناءً على السياق


# =============================================================================
# Provider Scoring
# =============================================================================

@dataclass
class ProviderScore:
    """درجة المزود"""
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
        """حساب الدرجة النهائية"""
        self.final_score = (
            self.quality_score * weights.get("quality", 0.4) +
            self.cost_score * weights.get("cost", 0.3) +
            self.speed_score * weights.get("speed", 0.2) +
            self.availability_score * weights.get("availability", 0.1)
        )


# =============================================================================
# Model Recommendations
# =============================================================================

TASK_MODEL_RECOMMENDATIONS = {
    # مهام التخطيط الاستراتيجي
    TaskType.PLANNING: {
        "recommended_tier": ModelTier.TIER_1_STRATEGIC,
        "models": ["claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro"],
        "min_quality": 0.9
    },

    # مهام كتابة الكود
    TaskType.CODE_GENERATION: {
        "recommended_tier": ModelTier.TIER_2_TACTICAL,
        "models": ["gpt-4o", "claude-3-5-sonnet", "llama-3.3-70b-versatile"],
        "min_quality": 0.85
    },

    # مراجعة الكود
    TaskType.CODE_REVIEW: {
        "recommended_tier": ModelTier.TIER_2_TACTICAL,
        "models": ["gpt-4o-mini", "llama-3.1-70b-versatile", "claude-3-5-sonnet"],
        "min_quality": 0.8
    },

    # التصحيح
    TaskType.DEBUGGING: {
        "recommended_tier": ModelTier.TIER_2_TACTICAL,
        "models": ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-flash"],
        "min_quality": 0.85
    },

    # البحث
    TaskType.RESEARCH: {
        "recommended_tier": ModelTier.TIER_3_EFFICIENT,
        "models": ["gpt-4o-mini", "gemini-1.5-flash", "llama-3.1-8b-instant"],
        "min_quality": 0.7
    },

    # التحليل
    TaskType.ANALYSIS: {
        "recommended_tier": ModelTier.TIER_2_TACTICAL,
        "models": ["claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro"],
        "min_quality": 0.85
    },

    # الاختبار
    TaskType.TESTING: {
        "recommended_tier": ModelTier.TIER_3_EFFICIENT,
        "models": ["gpt-4o-mini", "llama-3.1-8b-instant", "gemini-1.5-flash"],
        "min_quality": 0.75
    },

    # التوثيق
    TaskType.DOCUMENTATION: {
        "recommended_tier": ModelTier.TIER_3_EFFICIENT,
        "models": ["gpt-4o-mini", "llama-3.1-8b-instant", "gemini-1.5-flash-8b"],
        "min_quality": 0.7
    },
}

PRIORITY_MULTIPLIERS = {
    TaskPriority.CRITICAL: 1.5,
    TaskPriority.HIGH: 1.25,
    TaskPriority.NORMAL: 1.0,
    TaskPriority.LOW: 0.8,
    TaskPriority.BACKGROUND: 0.6,
}

COMPLEXITY_MODEL_TIER = {
    TaskComplexity.TRIVIAL: ModelTier.TIER_3_EFFICIENT,
    TaskComplexity.SIMPLE: ModelTier.TIER_3_EFFICIENT,
    TaskComplexity.MODERATE: ModelTier.TIER_2_TACTICAL,
    TaskComplexity.COMPLEX: ModelTier.TIER_2_TACTICAL,
    TaskComplexity.ARCHITECTURAL: ModelTier.TIER_1_STRATEGIC,
}


# =============================================================================
# Smart Router
# =============================================================================

class SmartRouter:
    """
    الموجه الذكي للطلبات
    
    يقرر أفضل مزود ونموذج لكل مهمة بناءً على:
    - تعقيد المهمة وأولويتها
    - الميزانية المتبقية
    - متطلبات السرعة والجودة
    - تفضيلات المستخدم
    - حالة المزودين
    """

    def __init__(
        self,
        providers: list[BaseProvider] | None = None,
        strategy: RoutingStrategy = RoutingStrategy.SMART,
        budget_limit: float = 100.0,
        quality_threshold: float = 0.8
    ):
        self._providers: dict[str, BaseProvider] = {}
        self._strategy = strategy
        self._budget_limit = budget_limit
        self._budget_spent = 0.0
        self._quality_threshold = quality_threshold
        self._logger = get_logger("gaap.router")

        # إحصائيات
        self._routing_history: list[RoutingDecision] = []
        self._provider_stats: dict[str, dict[str, Any]] = {}

        # تسجيل المزودين
        if providers:
            for provider in providers:
                self.register_provider(provider)

    # =========================================================================
    # Provider Management
    # =========================================================================

    def register_provider(self, provider: BaseProvider) -> None:
        """تسجيل مزود جديد"""
        self._providers[provider.name] = provider
        self._provider_stats[provider.name] = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0,
        }
        self._logger.info(f"Registered provider: {provider.name}")

    def unregister_provider(self, provider_name: str) -> None:
        """إلغاء تسجيل مزود"""
        if provider_name in self._providers:
            del self._providers[provider_name]
            del self._provider_stats[provider_name]
            self._logger.info(f"Unregistered provider: {provider_name}")

    def get_provider(self, provider_name: str) -> BaseProvider | None:
        """الحصول على مزود"""
        return self._providers.get(provider_name)

    def get_all_providers(self) -> list[BaseProvider]:
        """الحصول على جميع المزودين"""
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
        **kwargs
    ) -> RoutingDecision:
        """
        توجيه الطلب لأفضل مزود
        
        Args:
            messages: قائمة الرسائل
            task: المهمة المرتبطة (اختياري)
            preferred_model: نموذج مفضل (اختياري)
            excluded_providers: مزودين مستبعدين (اختياري)
        
        Returns:
            قرار التوجيه
        """
        start_time = time.time()
        excluded_providers = excluded_providers or []

        # تحليل المهمة
        task_info = self._analyze_task(task) if task else self._analyze_messages(messages)

        # التحقق من الميزانية
        estimated_cost = self._estimate_cost(messages, task_info)
        if self._budget_spent + estimated_cost > self._budget_limit:
            raise BudgetExceededError(
                budget=self._budget_limit - self._budget_spent,
                required=estimated_cost
            )

        # الحصول على المرشحين
        candidates = await self._get_candidates(
            task_info=task_info,
            preferred_model=preferred_model,
            excluded_providers=excluded_providers
        )

        if not candidates:
            raise NoAvailableProviderError(
                requirements={
                    "task_type": task_info.get("type", "unknown"),
                    "min_tier": task_info.get("min_tier", ModelTier.TIER_2_TACTICAL).name,
                }
            )

        # تقييم المرشحين
        scored_candidates = await self._score_candidates(candidates, task_info)

        # اختيار الأفضل
        best = scored_candidates[0]

        # بناء قرار التوجيه
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
            }
        )

        # تسجيل القرار
        self._routing_history.append(decision)

        self._logger.info(
            f"Routed to {best.provider_name}/{best.model} "
            f"(score: {best.final_score:.2f}, cost: ${best.estimated_cost:.4f})"
        )

        return decision

    def _analyze_task(self, task: Task) -> dict[str, Any]:
        """تحليل المهمة"""
        task_type = task.type
        priority = task.priority
        complexity = task.complexity

        # الحصول على التوصيات
        recommendations = TASK_MODEL_RECOMMENDATIONS.get(
            task_type,
            {"recommended_tier": ModelTier.TIER_2_TACTICAL, "min_quality": 0.8}
        )

        # تعديل المستوى بناءً على التعقيد
        base_tier = recommendations["recommended_tier"]
        complexity_tier = COMPLEXITY_MODEL_TIER.get(complexity, base_tier)

        # اختيار الأعلى
        tier_order = [
            ModelTier.TIER_3_EFFICIENT,
            ModelTier.TIER_4_PRIVATE,
            ModelTier.TIER_2_TACTICAL,
            ModelTier.TIER_1_STRATEGIC
        ]
        final_tier = base_tier if tier_order.index(base_tier) >= tier_order.index(complexity_tier) else complexity_tier

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
        """تحليل الرسائل (عند عدم وجود مهمة)"""
        # تقدير التعقيد من طول الرسائل
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

    def _estimate_cost(
        self,
        messages: list[Message],
        task_info: dict[str, Any]
    ) -> float:
        """تقدير تكلفة الطلب"""
        # تقدير عدد الرموز
        input_tokens = sum(len(m.content.split()) * 1.5 for m in messages)
        output_tokens = 500  # تقدير متوسط

        # تقدير التكلفة (باستخدام متوسط التكاليف)
        avg_cost_per_1k = 0.01  # متوسط تقريبي

        return (input_tokens + output_tokens) * avg_cost_per_1k / 1000

    async def _get_candidates(
        self,
        task_info: dict[str, Any],
        preferred_model: str | None,
        excluded_providers: list[str]
    ) -> list[tuple[BaseProvider, str]]:
        """الحصول على المرشحين المتاحين"""
        candidates = []
        min_tier = task_info.get("min_tier", ModelTier.TIER_2_TACTICAL)
        recommended = task_info.get("recommended_models", [])

        for provider_name, provider in self._providers.items():
            if provider_name in excluded_providers:
                continue

            # التحقق من النماذج المتاحة
            for model in provider.get_available_models():
                # إذا كان هناك نموذج مفضل
                if preferred_model and model != preferred_model:
                    continue

                # التحقق من مستوى النموذج
                model_tier = provider.get_model_tier(model)
                if self._tier_sufficient(model_tier, min_tier):
                    candidates.append((provider, model))

        # ترتيب حسب التوصيات
        if recommended:
            candidates.sort(
                key=lambda x: (
                    recommended.index(x[1]) if x[1] in recommended else 999
                )
            )

        return candidates

    def _tier_sufficient(
        self,
        model_tier: ModelTier,
        required_tier: ModelTier
    ) -> bool:
        """التحقق من كفاية مستوى النموذج"""
        tier_order = [
            ModelTier.TIER_3_EFFICIENT,
            ModelTier.TIER_4_PRIVATE,
            ModelTier.TIER_2_TACTICAL,
            ModelTier.TIER_1_STRATEGIC
        ]
        return tier_order.index(model_tier) >= tier_order.index(required_tier)

    async def _score_candidates(
        self,
        candidates: list[tuple[BaseProvider, str]],
        task_info: dict[str, Any]
    ) -> list[ProviderScore]:
        """تقييم المرشحين"""
        scores = []

        # أوزان التقييم حسب الاستراتيجية
        weights = self._get_strategy_weights()

        for provider, model in candidates:
            score = ProviderScore(
                provider_name=provider.name,
                model=model
            )

            # درجة الجودة
            score.quality_score = self._score_quality(provider, model, task_info)

            # درجة التكلفة
            score.cost_score, score.estimated_cost = self._score_cost(
                provider, model, task_info
            )

            # درجة السرعة
            score.speed_score, score.estimated_latency_ms = self._score_speed(
                provider, model
            )

            # درجة التوفر
            score.availability_score = self._score_availability(provider)

            # الدرجة النهائية
            score.calculate_final_score(weights)

            scores.append(score)

        # ترتيب تنازلي
        scores.sort(key=lambda x: x.final_score, reverse=True)

        return scores

    def _get_strategy_weights(self) -> dict[str, float]:
        """أوزان الاستراتيجية"""
        weights = {
            RoutingStrategy.QUALITY_FIRST: {"quality": 0.6, "cost": 0.1, "speed": 0.2, "availability": 0.1},
            RoutingStrategy.COST_OPTIMIZED: {"quality": 0.3, "cost": 0.5, "speed": 0.1, "availability": 0.1},
            RoutingStrategy.SPEED_FIRST: {"quality": 0.2, "cost": 0.1, "speed": 0.6, "availability": 0.1},
            RoutingStrategy.BALANCED: {"quality": 0.35, "cost": 0.25, "speed": 0.3, "availability": 0.1},
            RoutingStrategy.SMART: {"quality": 0.4, "cost": 0.3, "speed": 0.2, "availability": 0.1},
        }
        return weights.get(self._strategy, weights[RoutingStrategy.SMART])

    def _score_quality(
        self,
        provider: BaseProvider,
        model: str,
        task_info: dict[str, Any]
    ) -> float:
        """درجة الجودة"""
        tier = provider.get_model_tier(model)
        min_quality = task_info.get("min_quality", 0.8)

        # درجة أساسية من المستوى
        tier_scores = {
            ModelTier.TIER_1_STRATEGIC: 100,
            ModelTier.TIER_2_TACTICAL: 85,
            ModelTier.TIER_3_EFFICIENT: 70,
            ModelTier.TIER_4_PRIVATE: 75,
        }

        base_score = tier_scores.get(tier, 70)

        # تعديل بناءً على التوصيات
        if model in task_info.get("recommended_models", []):
            base_score += 10

        return min(base_score, 100)

    def _score_cost(
        self,
        provider: BaseProvider,
        model: str,
        task_info: dict[str, Any]
    ) -> tuple[float, float]:
        """درجة التكلفة"""
        # تقدير الرموز
        estimated_tokens = 2000  # تقدير
        estimated_cost = estimated_tokens * 0.001  # تقريبي

        # درجة عكسية (أقل تكلفة = درجة أعلى)
        if estimated_cost < 0.001:
            cost_score = 100
        elif estimated_cost < 0.01:
            cost_score = 90
        elif estimated_cost < 0.1:
            cost_score = 70
        else:
            cost_score = 50

        return cost_score, estimated_cost

    def _score_speed(
        self,
        provider: BaseProvider,
        model: str
    ) -> tuple[float, float]:
        """درجة السرعة"""
        # تقديرات بناءً على نوع المزود
        latency_estimates = {
            ProviderType.CHAT_BASED: (5000, 60),   # أبطأ
            ProviderType.FREE_TIER: (1500, 80),    # متوسط-سريع
            ProviderType.PAID: (2000, 75),         # متوسط
            ProviderType.LOCAL: (3000, 70),        # يعتمد
        }

        latency, score = latency_estimates.get(
            provider.provider_type,
            (3000, 70)
        )

        # Groq أسرع بكثير
        if provider.name.lower() == "groq":
            latency = 200
            score = 100

        return score, latency

    def _score_availability(self, provider: BaseProvider) -> float:
        """درجة التوفر"""
        stats = self._provider_stats.get(provider.name, {})

        if stats.get("requests", 0) == 0:
            return 100  # لا توجد بيانات، نفترض أنه متاح

        success_rate = stats["successes"] / stats["requests"]
        return success_rate * 100

    def _build_reasoning(
        self,
        best: ProviderScore,
        task_info: dict[str, Any]
    ) -> str:
        """بناء سبب القرار"""
        reasons = []

        reasons.append(
            f"Selected {best.provider_name}/{best.model} "
            f"with score {best.final_score:.1f}/100"
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
        """تحديد الميزانية"""
        self._budget_limit = limit

    def record_spending(self, amount: float) -> None:
        """تسجيل إنفاق"""
        self._budget_spent += amount

    def get_budget_status(self) -> dict[str, float]:
        """حالة الميزانية"""
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
        self,
        provider_name: str,
        success: bool,
        latency_ms: float,
        cost: float
    ) -> None:
        """تسجيل نتيجة طلب"""
        if provider_name not in self._provider_stats:
            return

        stats = self._provider_stats[provider_name]
        stats["requests"] += 1

        if success:
            stats["successes"] += 1
            stats["total_cost"] += cost
            # تحديث متوسط التأخير
            old_avg = stats["avg_latency"]
            n = stats["successes"]
            stats["avg_latency"] = old_avg + (latency_ms - old_avg) / n
        else:
            stats["failures"] += 1

    def get_routing_stats(self) -> dict[str, Any]:
        """إحصائيات التوجيه"""
        return {
            "total_routes": len(self._routing_history),
            "providers": self._provider_stats.copy(),
            "budget": self.get_budget_status(),
        }

    # =========================================================================
    # Strategy Management
    # =========================================================================

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """تحديد الاستراتيجية"""
        self._strategy = strategy
        self._logger.info(f"Routing strategy changed to: {strategy.value}")

    def get_strategy(self) -> RoutingStrategy:
        """الحصول على الاستراتيجية الحالية"""
        return self._strategy


# =============================================================================
# Convenience Functions
# =============================================================================

def create_router(
    providers: list[BaseProvider] | None = None,
    strategy: str = "smart",
    budget: float = 100.0
) -> SmartRouter:
    """إنشاء موجه بسهولة"""
    strategy_enum = RoutingStrategy(strategy.lower())
    return SmartRouter(
        providers=providers,
        strategy=strategy_enum,
        budget_limit=budget
    )
