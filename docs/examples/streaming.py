#!/usr/bin/env python3
"""
GAAP Streaming Example

This example demonstrates streaming responses (when supported by provider).
"""

import asyncio
import sys

from gaap import GAAPEngine, GAAPRequest
from gaap.core.types import Message, MessageRole


async def simulate_streaming():
    """
    Simulate streaming by processing chunks.

    Note: Full streaming support depends on provider implementation.
    This example shows the concept.
    """
    print("=== GAAP Streaming Example ===\n")

    engine = GAAPEngine(budget=10.0)

    # For providers that support streaming, you would typically:
    # 1. Set stream=True in the request
    # 2. Iterate over the response chunks

    # Current GAAP implementation returns complete responses
    # Here's how to process incrementally:

    prompt = "Write a short story about a robot learning to paint"

    print(f"Prompt: {prompt}\n")
    print("Response (streaming simulation):\n")

    request = GAAPRequest(text=prompt)
    response = await engine.process(request)

    if response.success and response.output:
        # Simulate streaming by printing character by character
        output = response.output
        chunk_size = 10  # characters per "chunk"

        for i in range(0, len(output), chunk_size):
            chunk = output[i : i + chunk_size]
            print(chunk, end="", flush=True)
            await asyncio.sleep(0.05)  # Simulate network delay

        print("\n")

    print(f"\n--- Stats ---")
    print(f"Total time: {response.total_time_ms:.0f}ms")
    print(f"Tokens: {response.total_tokens}")

    engine.shutdown()


async def main():
    await simulate_streaming()


if __name__ == "__main__":
    asyncio.run(main())
