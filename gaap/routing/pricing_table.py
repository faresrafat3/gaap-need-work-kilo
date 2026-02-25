"""
Model Pricing Table - Live Cost Tracking
Implements: docs/evolution_plan_2026/37_ROUTER_AUDIT_SPEC.md

Features:
- Up-to-date pricing per 1M tokens
- Input/output token costs
- Provider-specific pricing
- Cost estimation utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelPricing:
    model_id: str
    provider: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    context_window: int = 128000
    max_output_tokens: int = 4096
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_1m
        return input_cost + output_cost

    def estimate_cost(self, estimated_input: int = 1000, estimated_output: int = 500) -> float:
        return self.calculate_cost(estimated_input, estimated_output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "input_cost_per_1m": self.input_cost_per_1m,
            "output_cost_per_1m": self.output_cost_per_1m,
            "context_window": self.context_window,
            "supports_vision": self.supports_vision,
            "supports_tools": self.supports_tools,
        }


MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI Models
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        provider="openai",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_tools=True,
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        provider="openai",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_tools=True,
    ),
    "gpt-4-turbo": ModelPricing(
        model_id="gpt-4-turbo",
        provider="openai",
        input_cost_per_1m=10.00,
        output_cost_per_1m=30.00,
        context_window=128000,
        max_output_tokens=4096,
        supports_vision=True,
        supports_tools=True,
    ),
    "o1": ModelPricing(
        model_id="o1",
        provider="openai",
        input_cost_per_1m=15.00,
        output_cost_per_1m=60.00,
        context_window=200000,
        max_output_tokens=100000,
        supports_vision=False,
        supports_tools=False,
    ),
    "o1-mini": ModelPricing(
        model_id="o1-mini",
        provider="openai",
        input_cost_per_1m=1.50,
        output_cost_per_1m=6.00,
        context_window=128000,
        max_output_tokens=65536,
        supports_vision=False,
        supports_tools=False,
    ),
    # Anthropic Models
    "claude-3-5-sonnet-20241022": ModelPricing(
        model_id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        context_window=200000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_tools=True,
    ),
    "claude-3-opus-20240229": ModelPricing(
        model_id="claude-3-opus-20240229",
        provider="anthropic",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        context_window=200000,
        max_output_tokens=4096,
        supports_vision=True,
        supports_tools=True,
    ),
    "claude-3-haiku-20240307": ModelPricing(
        model_id="claude-3-haiku-20240307",
        provider="anthropic",
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
        context_window=200000,
        max_output_tokens=4096,
        supports_vision=True,
        supports_tools=True,
    ),
    # Google Models
    "gemini-1.5-pro": ModelPricing(
        model_id="gemini-1.5-pro",
        provider="google",
        input_cost_per_1m=1.25,
        output_cost_per_1m=5.00,
        context_window=1000000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_tools=True,
    ),
    "gemini-1.5-flash": ModelPricing(
        model_id="gemini-1.5-flash",
        provider="google",
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
        context_window=1000000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_tools=True,
    ),
    "gemini-2.0-flash": ModelPricing(
        model_id="gemini-2.0-flash",
        provider="google",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
        context_window=1000000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_tools=True,
    ),
    # Groq Models (Fast Inference)
    "llama-3.3-70b-versatile": ModelPricing(
        model_id="llama-3.3-70b-versatile",
        provider="groq",
        input_cost_per_1m=0.59,
        output_cost_per_1m=0.79,
        context_window=128000,
        max_output_tokens=8192,
        supports_vision=False,
        supports_tools=True,
    ),
    "llama-3.1-8b-instant": ModelPricing(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.08,
        context_window=128000,
        max_output_tokens=8192,
        supports_vision=False,
        supports_tools=True,
    ),
    "mixtral-8x7b-32768": ModelPricing(
        model_id="mixtral-8x7b-32768",
        provider="groq",
        input_cost_per_1m=0.24,
        output_cost_per_1m=0.24,
        context_window=32768,
        max_output_tokens=4096,
        supports_vision=False,
        supports_tools=True,
    ),
    # DeepSeek Models
    "deepseek-chat": ModelPricing(
        model_id="deepseek-chat",
        provider="deepseek",
        input_cost_per_1m=0.14,
        output_cost_per_1m=0.28,
        context_window=64000,
        max_output_tokens=4096,
        supports_vision=False,
        supports_tools=True,
    ),
    "deepseek-reasoner": ModelPricing(
        model_id="deepseek-reasoner",
        provider="deepseek",
        input_cost_per_1m=0.55,
        output_cost_per_1m=2.19,
        context_window=64000,
        max_output_tokens=4096,
        supports_vision=False,
        supports_tools=False,
    ),
    # Mistral Models
    "mistral-large-latest": ModelPricing(
        model_id="mistral-large-latest",
        provider="mistral",
        input_cost_per_1m=2.00,
        output_cost_per_1m=6.00,
        context_window=128000,
        max_output_tokens=4096,
        supports_vision=False,
        supports_tools=True,
    ),
    "mistral-small-latest": ModelPricing(
        model_id="mistral-small-latest",
        provider="mistral",
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.60,
        context_window=128000,
        max_output_tokens=4096,
        supports_vision=False,
        supports_tools=True,
    ),
    # Local/Free Models (Estimated)
    "local-llama-3": ModelPricing(
        model_id="local-llama-3",
        provider="local",
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        context_window=8192,
        max_output_tokens=2048,
        supports_vision=False,
        supports_tools=False,
    ),
}


def get_pricing(model_id: str) -> ModelPricing | None:
    return MODEL_PRICING.get(model_id)


def get_all_pricing() -> dict[str, ModelPricing]:
    return MODEL_PRICING.copy()


def get_pricing_by_provider(provider: str) -> dict[str, ModelPricing]:
    return {
        model_id: pricing
        for model_id, pricing in MODEL_PRICING.items()
        if pricing.provider == provider
    }


def estimate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    pricing = get_pricing(model_id)
    if pricing:
        return pricing.calculate_cost(input_tokens, output_tokens)
    return None


def get_cheapest_model(min_context: int = 4096) -> str:
    valid_models = [
        (model_id, pricing.input_cost_per_1m)
        for model_id, pricing in MODEL_PRICING.items()
        if pricing.context_window >= min_context
    ]
    if valid_models:
        return min(valid_models, key=lambda x: x[1])[0]
    return ""


def get_best_value_model(min_context: int = 4096) -> str:
    valid_models = [
        (model_id, pricing.input_cost_per_1m + pricing.output_cost_per_1m)
        for model_id, pricing in MODEL_PRICING.items()
        if pricing.context_window >= min_context and pricing.provider != "local"
    ]
    if valid_models:
        return min(valid_models, key=lambda x: x[1])[0]
    return ""


def count_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)
