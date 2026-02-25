import asyncio
import time
from typing import Any, AsyncGenerator

from gaap.core.types import (
    Message,
    ModelTier,
    ProviderType,
)
from gaap.providers.base_provider import BaseProvider, register_provider
from gaap.providers.chat_based.g4f_provider import G4FProvider  # Sovereign Replacement


@register_provider("unified_gaap")
class UnifiedGAAPProvider(BaseProvider):
    """
    Sovereign Unified Provider

    Acts as a high-level facade for the best available free-tier models (G4F).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gemini-1.5-flash",
        profile: str = "quality",
        **kwargs: Any,
    ) -> None:
        models = ["gemini-1.5-flash", "claude-3-5-sonnet", "gpt-4o-mini"]

        super().__init__(
            name="unified_gaap",
            provider_type=ProviderType.FREE_TIER,
            models=models,
            api_key=api_key,
            base_url=base_url,
            rate_limit_rpm=60,
            rate_limit_tpm=1000000,
            timeout=180.0,
            max_retries=2,
            default_model=default_model,
        )

        self._backend = G4FProvider()
        self._logger.info(f"UnifiedGAAPProvider initialized (Backend: G4F)")

    async def chat_completion(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward to backend"""
        return await self._backend.chat_completion(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

    async def _make_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Legacy internal method (not used if chat_completion overridden)"""
        return {}

    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Streaming proxy"""
        async for chunk in self._backend.stream_chat_completion(messages, model, **kwargs):
            yield chunk

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Cost is 0 for free WebChat providers."""
        return 0.0

    def get_model_tier(self, model: str) -> ModelTier:
        return ModelTier.TIER_1_STRATEGIC
