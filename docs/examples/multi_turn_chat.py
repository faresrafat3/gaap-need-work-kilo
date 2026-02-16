#!/usr/bin/env python3
"""
GAAP Multi-Turn Chat Example

This example demonstrates a conversation with context preservation.
"""

import asyncio

from gaap import GAAPEngine, GAAPRequest


async def main():
    print("=== GAAP Multi-Turn Chat Example ===\n")

    # Create engine
    engine = GAAPEngine(budget=50.0)

    # Conversation history (managed by GAAP memory)
    conversation = []

    # First message
    print("User: My name is Ahmed")
    response1 = await engine.chat("My name is Ahmed")
    conversation.append(("user", "My name is Ahmed"))
    conversation.append(("assistant", response1))
    print(f"Assistant: {response1[:100]}...\n")

    # Second message - context should be preserved
    print("User: What is my name?")
    response2 = await engine.chat("What is my name?")
    print(f"Assistant: {response2}\n")

    # Third message - continue conversation
    print("User: Can you help me write a Python function?")
    response3 = await engine.chat("Can you help me write a Python function to calculate fibonacci?")
    print(f"Assistant: {response3[:200]}...\n")

    # Get memory stats
    stats = engine.get_stats()
    print("\n--- Memory Stats ---")
    print(f"Requests processed: {stats['requests_processed']}")
    print(f"Success rate: {stats['success_rate']:.1%}")

    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
