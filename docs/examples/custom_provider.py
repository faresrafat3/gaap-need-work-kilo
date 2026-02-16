#!/usr/bin/env python3
"""
GAAP Custom Provider Example

This example shows how to create and use a custom LLM provider.
"""

import asyncio
from typing import Any

import aiohttp

from gaap.core.types import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    Message,
    MessageRole,
    ModelInfo,
    ModelTier,
    ProviderType,
)
from gaap.providers.base_provider import BaseProvider


class MyCustomProvider(BaseProvider):
    """
    Example custom provider implementation.

    This provider connects to a hypothetical API endpoint.
    """

    def __init__(
        self, api_key: str, base_url: str = "https://api.example.com/v1", name: str = "my_custom"
    ):
        super().__init__(
            name=name,
            provider_type=ProviderType.PAID,
            api_key=api_key,
            base_url=base_url,
            models=["custom-model-7b", "custom-model-70b"],
            rate_limit=60,
            timeout=120,
            max_retries=3,
        )
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def chat_completion(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        session = await self._get_session()

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        data = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.base_url}/chat/completions"

        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()

            return ChatCompletionResponse(
                id=result.get("id", "unknown"),
                model=model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=result["choices"][0]["message"]["content"],
                        ),
                        finish_reason=result["choices"][0].get("finish_reason", "stop"),
                    )
                ],
            )

    def get_available_models(self) -> list[str]:
        return self.models

    def get_model_info(self, model: str) -> ModelInfo:
        return ModelInfo(
            name=model,
            provider=self.name,
            tier=ModelTier.TIER_2_TACTICAL,
            context_window=8192,
            max_output_tokens=4096,
        )

    def shutdown(self) -> None:
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())


async def main():
    print("=== GAAP Custom Provider Example ===\n")

    # Create custom provider
    custom_provider = MyCustomProvider(
        api_key="your-api-key", base_url="https://api.example.com/v1"
    )

    # Use with GAAP engine
    from gaap import GAAPEngine

    engine = GAAPEngine(providers=[custom_provider], budget=10.0)

    # The engine will now use your custom provider
    print("Custom provider registered!")
    print(f"Available models: {custom_provider.get_available_models()}")

    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
