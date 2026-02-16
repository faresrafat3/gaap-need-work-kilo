import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from gaap.core.types import (
    Message,
    MessageRole,
    ModelTier,
    ProviderType,
)
from gaap.providers.base_provider import BaseProvider, register_provider
from gaap.providers.unified_provider import UnifiedProvider


@register_provider("unified_gaap")
class UnifiedGAAPProvider(BaseProvider):
    """
    Bridge between GAAP BaseProvider and the smart UnifiedProvider.
    This enables Kimi-first routing with the full GAAP engine features.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "kimi",
        profile: str = "quality",
        **kwargs,
    ):
        # We don't call super().__init__ with all models yet because they are dynamic
        models = ["kimi", "deepseek", "glm", "gemini-2.5-flash"]

        super().__init__(
            name="unified_gaap",
            provider_type=ProviderType.FREE_TIER,
            models=models,
            api_key=api_key,
            base_url=base_url,
            rate_limit_rpm=60,
            rate_limit_tpm=1000000,
            timeout=180.0,  # WebChat needs longer timeouts
            max_retries=2,
            default_model=default_model,
        )

        self._unified = UnifiedProvider(profile=profile, verbose=True)
        self._logger.info(f"UnifiedGAAPProvider initialized with profile: {profile}")

    async def _make_request(self, messages: list[Message], model: str, **kwargs) -> dict[str, Any]:
        """Execute request via UnifiedProvider fallback chain (Kimi -> DeepSeek -> etc.)"""

        # Convert Message objects to simple dicts
        prompt_parts = []
        system_content = None

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                prompt_parts.append(msg.content)

        full_prompt = "\n\n".join(prompt_parts)

        # Call the unified provider (blocking call run in thread)
        def sync_call():
            return self._unified.call(
                prompt=full_prompt, system=system_content, timeout=int(self.timeout)
            )

        try:
            content, model_used, latency_ms = await asyncio.to_thread(sync_call)

            return {
                "id": f"unified-{int(time.time())}",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(full_prompt) // 4,
                    "completion_tokens": len(content) // 4,
                    "total_tokens": (len(full_prompt) + len(content)) // 4,
                },
                "model_used": model_used,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            self._logger.error(f"Unified fallback chain failed: {e}")
            raise

    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Streaming is not natively supported by the unified bridge yet,
        so we yield the full response as a single chunk."""
        result = await self._make_request(messages, model, **kwargs)
        content = result["choices"][0]["message"]["content"]
        yield content

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Cost is 0 for free WebChat providers."""
        return 0.0

    def get_model_tier(self, model: str) -> ModelTier:
        return ModelTier.TIER_1_STRATEGIC  # Kimi/DeepSeek/GLM are high-tier
