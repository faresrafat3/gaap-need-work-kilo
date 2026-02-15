"""
GAAP Custom Provider Example

This example shows how to create and register a custom LLM provider.
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, AsyncGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaap.core.types import (
    Message,
    MessageRole,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Usage,
    ProviderType,
    ModelTier,
)
from gaap.providers.base_provider import BaseProvider, register_provider


@register_provider("custom")
class CustomProvider(BaseProvider):
    """Example custom provider"""

    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        super().__init__(
            name="custom",
            provider_type=ProviderType.FREE_TIER,
            models=["custom-model-1", "custom-model-2"],
            **kwargs,
        )
        self.api_key = api_key
        self.base_url = base_url or "https://api.custom.ai"

    async def _make_request(self, messages: List[Message], model: str, **kwargs) -> Dict[str, Any]:
        """Make a request to the custom provider"""

        # Simulate API call
        await asyncio.sleep(0.1)

        return {
            "id": "custom-response-id",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Custom response to: {messages[-1].content}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

    async def _stream_request(
        self, messages: List[Message], model: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from the custom provider"""
        response = f"Custom streaming response to: {messages[-1].content}"
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.05)

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for the request"""
        # Very cheap custom provider
        return (input_tokens * 0.00001) + (output_tokens * 0.00002)

    def get_model_tier(self, model: str) -> ModelTier:
        """Get the tier for a model"""
        return ModelTier.TIER_3_EFFICIENT


async def custom_provider_example():
    """Example using the custom provider"""
    print("\n=== Custom Provider Example ===")

    # Create provider directly
    provider = CustomProvider(api_key="test-key")

    # Chat completion
    messages = [Message(role=MessageRole.USER, content="Hello, custom provider!")]

    response = await provider.chat_completion(messages, model="custom-model-1")

    print(f"Model: {response.model}")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.total_tokens}")

    # Get stats
    stats = provider.get_stats()
    print(f"Provider stats: {stats}")


async def streaming_example():
    """Example with streaming response"""
    print("\n=== Streaming Example ===")

    provider = CustomProvider(api_key="test-key")
    messages = [Message(role=MessageRole.USER, content="Tell me a story")]

    print("Streaming response: ", end="", flush=True)
    async for chunk in provider.stream_completion(messages, model="custom-model-1"):
        print(chunk, end="", flush=True)
    print()


async def main():
    """Run all examples"""
    await custom_provider_example()
    await streaming_example()

    print("\nCustom provider example complete!")


if __name__ == "__main__":
    asyncio.run(main())
