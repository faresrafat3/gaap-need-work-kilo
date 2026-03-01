"""
Quality Gate and Model Cascading
Implements: docs/evolution_plan_2026/37_ROUTER_AUDIT_SPEC.md

Features:
- Quality validation for model output
- Model cascading through tiers
- Automatic escalation on quality-based routing
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, ModelTier, Task

if TYPE_CHECKING:
    from gaap.providers.base_provider import BaseProvider
    from gaap.routing.router import SmartRouter


@dataclass
class QualityGateResult:
    passed: bool
    score: float = 0.0
    issues: list[str] = field(default_factory=list)
    needs_escalation: bool = False


class QualityGate:
    """
    Validates output quality for Model Cascading.

    Checks:
    - Response completeness
    - Code syntax validity
    - Error indicators
    - Response length
    """

    MIN_RESPONSE_LENGTH = 10
    ERROR_INDICATORS = ["error", "failed", "unable to", "cannot", "sorry"]

    def __init__(self, min_quality: float = 0.7):
        self.min_quality = min_quality

    def evaluate(
        self,
        response: str,
        task_type: str = "unknown",
        expected_format: str | None = None,
    ) -> QualityGateResult:
        issues: list[str] = []
        score = 1.0

        if not response or len(response.strip()) < self.MIN_RESPONSE_LENGTH:
            issues.append("Response too short or empty")
            score -= 0.5

        response_lower = response.lower() if response else ""
        for indicator in self.ERROR_INDICATORS:
            if indicator in response_lower:
                score -= 0.1
                issues.append(f"Error indicator found: '{indicator}'")

        if expected_format == "code":
            try:
                ast.parse(response)
            except SyntaxError:
                issues.append("Code response has syntax errors")
                score -= 0.3

        if len(response) > 1000 and task_type in ("planning", "analysis"):
            score += 0.1

        score = max(0.0, min(1.0, score))
        passed = score >= self.min_quality

        return QualityGateResult(
            passed=passed,
            score=score,
            issues=issues,
            needs_escalation=not passed,
        )


@dataclass
class CascadeResult:
    final_response: str
    provider_used: str
    model_used: str
    attempts: int = 1
    escalated: bool = False
    total_cost: float = 0.0
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider_used,
            "model": self.model_used,
            "attempts": self.attempts,
            "escalated": self.escalated,
            "cost": self.total_cost,
            "quality": self.quality_score,
        }


class CascadeRouter:
    """
    Model Cascading Router - Routes through model tiers based on quality.

    Flow:
    1. Start with cheapest model meeting min_quality
    2. Validate output with QualityGate
    3. If quality < threshold, escalate to next tier
    4. Repeat until success or no more providers

    Usage:
        cascade = CascadeRouter(router)
        result = await cascade.route_with_cascade(messages, task)
    """

    TIER_ORDER = [
        ModelTier.TIER_3_EFFICIENT,
        ModelTier.TIER_2_TACTICAL,
        ModelTier.TIER_1_STRATEGIC,
    ]

    def __init__(
        self,
        router: "SmartRouter",
        quality_threshold: float = 0.7,
        max_attempts: int = 3,
    ) -> None:
        self.router = router
        self.quality_gate = QualityGate(min_quality=quality_threshold)
        self.max_attempts = max_attempts
        self._logger = get_logger("gaap.cascade_router")

    async def route_with_cascade(
        self,
        messages: list[Message],
        task: Task | None = None,
        provider_call: Any | None = None,
    ) -> CascadeResult:
        attempts = 0
        total_cost = 0.0
        last_response = ""
        last_provider = ""
        last_model = ""

        for tier in self.TIER_ORDER:
            if attempts >= self.max_attempts:
                break

            candidates = await self._get_tier_candidates(tier)
            if not candidates:
                continue

            for provider, model in candidates:
                attempts += 1

                try:
                    decision = await self.router.route(messages, task)

                    if decision.selected_provider != provider.name:
                        continue

                    last_provider = provider.name
                    last_model = decision.selected_model
                    total_cost += decision.estimated_cost

                    if provider_call and callable(provider_call):
                        result = provider_call(provider, model, messages)
                        response = await result if hasattr(result, "__await__") else result
                    else:
                        response = f"Simulated response from {provider.name}/{model}"

                    last_response = response

                    gate_result = self.quality_gate.evaluate(
                        response,
                        task_type=task.type.name if task else "unknown",
                    )

                    if gate_result.passed:
                        return CascadeResult(
                            final_response=response,
                            provider_used=last_provider,
                            model_used=last_model,
                            attempts=attempts,
                            escalated=attempts > 1,
                            total_cost=total_cost,
                            quality_score=gate_result.score,
                        )

                    self._logger.info(
                        f"Quality gate failed (score: {gate_result.score:.2f}), escalating..."
                    )

                except Exception as e:
                    self._logger.warning(f"Provider {provider.name} failed: {e}")
                    continue

        return CascadeResult(
            final_response=last_response,
            provider_used=last_provider,
            model_used=last_model,
            attempts=attempts,
            escalated=True,
            total_cost=total_cost,
            quality_score=0.0,
        )

    async def _get_tier_candidates(
        self,
        tier: ModelTier,
    ) -> list[tuple["BaseProvider", str]]:
        candidates: list[tuple["BaseProvider", str]] = []

        for provider in self.router.get_all_providers():
            for model in provider.get_available_models():
                if provider.get_model_tier(model) == tier:
                    candidates.append((provider, model))

        from gaap.routing.pricing_table import get_pricing

        def get_cost(item: tuple["BaseProvider", str]) -> float:
            pricing = get_pricing(item[1])
            return pricing.input_cost_per_1m if pricing else 0.0

        candidates.sort(key=get_cost)
        return candidates

    def get_stats(self) -> dict[str, Any]:
        return {
            "quality_threshold": self.quality_gate.min_quality,
            "max_attempts": self.max_attempts,
            "tier_order": [t.name for t in self.TIER_ORDER],
        }


def create_cascade_router(
    router: "SmartRouter",
    quality_threshold: float = 0.7,
) -> CascadeRouter:
    return CascadeRouter(router, quality_threshold=quality_threshold)
