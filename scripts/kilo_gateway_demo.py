#!/usr/bin/env python3
"""
Kilo Gateway Demo - Test the free AI gateway

Features:
- 500+ models from one API
- Free tier with credits
- Models: Claude, GPT, Gemini, Grok, etc.

Usage:
    export KILO_API_KEY=your-api-key
    python scripts/kilo_gateway_demo.py
"""

import asyncio
import os
import sys

from gaap.core.types import Message, MessageRole
from gaap.providers.kilo_gateway import KiloGatewayProvider, FREE_MODELS, POPULAR_MODELS


async def test_kilo_gateway() -> int:
    api_key = os.environ.get("KILO_API_KEY")

    if not api_key:
        print("=" * 60)
        print("  KILO API KEY REQUIRED")
        print("=" * 60)
        print("""
To use Kilo Gateway:

1. Get your API key from: https://kilo.ai/gateway
   - New users get $5 FREE credits
   - No credit card required

2. Set environment variable:
   export KILO_API_KEY=your-key-here

3. Run this script again:
   python scripts/kilo_gateway_demo.py

FREE MODELS AVAILABLE:
   - z-ai/glm-5 (FREE - Deep Thinking!)
   - z-ai/glm-4.7 (FREE)
   - minimax/m2.1 (FREE)

POPULAR MODELS:
   - anthropic/claude-sonnet-4.5
   - openai/gpt-5.2
   - google/gemini-3-pro
   - x-ai/grok-code-1
""")
        return 1

    print("=" * 60)
    print("  KILO GATEWAY DEMO")
    print("=" * 60)

    provider = KiloGatewayProvider(api_key=api_key)

    print(f"\nAPI Key: {api_key[:10]}...")
    print(f"Default Model: {provider.default_model}")
    print(f"Free Models: {provider.get_free_models()}")

    messages = [
        Message(role=MessageRole.USER, content="Say 'Hello from Kilo Gateway!' in one line.")
    ]

    for model in ["z-ai/glm-5", "anthropic/claude-sonnet-4.5"]:
        print(f"\n{'─' * 60}")
        print(f"Testing: {model}")
        print(f"{'─' * 60}")

        try:
            response = await provider.chat_completion(
                messages=messages,
                model=model,
                max_tokens=100,
            )

            if response.choices:
                content = response.choices[0].message.content
                print(f"Response: {content}")
                if response.usage:
                    print(f"Tokens: {response.usage.total_tokens}")
                print("Status: OK")
            else:
                print("No response")

        except Exception as e:
            print(f"Error: {e}")

    await provider.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(test_kilo_gateway()))
