"""
Kilo Gateway Provider - Universal AI Inference API

Access 500+ models from Anthropic, OpenAI, Google, xAI, Mistral, MiniMax, and more
through a single unified endpoint.

Features:
- One API key for 500+ models
- Free tier available (with credits)
- OpenAI-compatible API
- Built-in load balancing
- No markup on token usage

Usage:
    from gaap.providers.kilo_gateway import KiloGatewayProvider

    provider = KiloGatewayProvider(api_key="your-kilo-api-key")
    response = await provider.chat_completion(messages, model="anthropic/claude-sonnet-4.5")

NOTE: For free models with Kilo Code JWT token:
    - Use model name with ":free" suffix (e.g., "z-ai/glm-5:free")
    - Get your token from Kilo Code CLI/Extension authentication
"""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from gaap.core.exceptions import (
    ProviderAuthenticationError,
    ProviderNotAvailableError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from gaap.core.types import Message, ModelTier, ProviderType
from gaap.providers.base_provider import BaseProvider, get_logger, register_provider

KILO_OPENROUTER_URL = "https://api.kilo.ai/api/openrouter/chat/completions"
KILO_GATEWAY_URL = "https://api.kilo.ai/api/gateway/chat/completions"

POPULAR_MODELS = [
    "z-ai/glm-5",
    "minimax/minimax-m2.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.6",
    "openai/gpt-5.2",
    "openai/gpt-4o",
    "google/gemini-3-pro",
    "x-ai/grok-code-1",
    "mistral/mistral-large",
    "stepfun/step-3.5-flash",
]

FREE_MODELS = [
    "z-ai/glm-5",
    "minimax/minimax-m2.5",
    "stepfun/step-3.5-flash",
    "arcee-ai/trinity-large-preview",
    "corethink",
]

MODEL_COSTS = {
    "z-ai/glm-5": {"input": 0.0, "output": 0.0},
    "minimax/minimax-m2.5": {"input": 0.0, "output": 0.0},
    "stepfun/step-3.5-flash": {"input": 0.0, "output": 0.0},
    "arcee-ai/trinity-large-preview": {"input": 0.0, "output": 0.0},
    "corethink": {"input": 0.0, "output": 0.0},
    "anthropic/claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
    "anthropic/claude-opus-4.6": {"input": 15.0, "output": 75.0},
    "openai/gpt-5.2": {"input": 2.5, "output": 10.0},
    "openai/gpt-4o": {"input": 2.5, "output": 10.0},
    "google/gemini-3-pro": {"input": 1.25, "output": 5.0},
    "x-ai/grok-code-1": {"input": 3.0, "output": 15.0},
    "mistral/mistral-large": {"input": 2.0, "output": 6.0},
}


@register_provider("kilo")
class KiloGatewayProvider(BaseProvider):
    """
    Kilo Gateway Provider - Universal AI Inference API.

    Access 500+ models through a single endpoint with:
    - Free tier credits ($5 for new users)
    - BYOK (Bring Your Own Key) support
    - No markup on tokens
    - Built-in load balancing

    Models: anthropic/*, openai/*, google/*, x-ai/*, mistral/*, minimax/*, z-ai/*

    NOTE: For free models, use the ":free" suffix:
        model="z-ai/glm-5:free"
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "z-ai/glm-5",
        use_free_suffix: bool = True,
        **kwargs: Any,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("KILO_API_KEY") or os.environ.get("KILO_TOKEN")

        all_models = POPULAR_MODELS + FREE_MODELS

        super().__init__(
            name="kilo",
            provider_type=ProviderType.FREE_TIER,
            models=all_models,
            api_key=api_key,
            base_url=KILO_OPENROUTER_URL,
            rate_limit_rpm=100,
            rate_limit_tpm=500000,
            timeout=120.0,
            max_retries=3,
            default_model=default_model,
        )

        self._session: aiohttp.ClientSession | None = None
        self._logger = get_logger("gaap.provider.kilo")
        self._free_credits_used: float = 0.0
        self._use_free_suffix: bool = use_free_suffix

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "GAAP-Kilo-Provider",
                "X-KILOCODE-EDITORNAME": "GAAP",
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    def _get_model_name(self, model: str) -> str:
        base_model = model.replace(":free", "")
        if self._use_free_suffix and base_model in FREE_MODELS and ":free" not in model:
            return f"{base_model}:free"
        return model

    async def _make_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> dict[str, Any]:
        session = await self._get_session()

        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        actual_model = self._get_model_name(model)

        payload = {
            "model": actual_model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if kwargs.get("stream"):
            payload["stream"] = True

        try:
            async with session.post(
                KILO_OPENROUTER_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status == 401:
                    raise ProviderAuthenticationError(provider_name=self.name)

                if response.status == 429:
                    retry_after = int(response.headers.get("retry-after", 60))
                    raise ProviderRateLimitError(provider_name=self.name, retry_after=retry_after)

                if response.status != 200:
                    error_body = await response.text()
                    raise ProviderResponseError(
                        provider_name=self.name,
                        status_code=response.status,
                        response_body=error_body,
                    )

                data: dict[str, Any] = await response.json()
                return dict(data)

        except aiohttp.ClientError as e:
            raise ProviderNotAvailableError(provider_name=self.name, reason=str(e))

    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        session = await self._get_session()

        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        actual_model = self._get_model_name(model)

        payload = {
            "model": actual_model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": True,
        }

        try:
            async with session.post(
                KILO_OPENROUTER_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise ProviderResponseError(
                        provider_name=self.name,
                        status_code=response.status,
                        response_body=error_body,
                    )

                async for line in response.content:
                    decoded_line = line.decode("utf-8").strip()

                    if not decoded_line or decoded_line == "data: [DONE]":
                        continue

                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        try:
                            chunk = json.loads(json_str)
                            content = (
                                chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            )
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            raise ProviderNotAvailableError(provider_name=self.name, reason=str(e))

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        base_model = model.replace(":free", "")
        costs = MODEL_COSTS.get(base_model, {"input": 0.0, "output": 0.0})
        return (input_tokens * costs["input"] / 1_000_000) + (
            output_tokens * costs["output"] / 1_000_000
        )

    def get_model_tier(self, model: str) -> ModelTier:
        base_model = model.replace(":free", "")
        if "opus" in base_model.lower() or "gpt-5" in base_model.lower():
            return ModelTier.TIER_1_STRATEGIC
        if "glm-5" in base_model.lower() or "minimax-m2.5" in base_model.lower():
            return ModelTier.TIER_1_STRATEGIC
        if "sonnet" in base_model.lower() or "gemini-3" in base_model.lower():
            return ModelTier.TIER_2_TACTICAL
        if base_model in FREE_MODELS:
            return ModelTier.TIER_3_EFFICIENT
        return super().get_model_tier(model)

    def get_free_models(self) -> list[str]:
        return FREE_MODELS.copy()

    def is_free_model(self, model: str) -> bool:
        base_model = model.replace(":free", "")
        return base_model in FREE_MODELS

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def shutdown(self) -> None:
        asyncio.create_task(self.close())
        super().shutdown()


def create_kilo_provider(
    api_key: str | None = None,
    default_model: str = "z-ai/glm-5",
) -> KiloGatewayProvider:
    return KiloGatewayProvider(api_key=api_key, default_model=default_model)


def list_kilo_models() -> list[str]:
    return POPULAR_MODELS.copy()


def get_free_kilo_models() -> list[str]:
    return FREE_MODELS.copy()
