"""
GAAP Quick Start Example

This example demonstrates the basic usage of GAAP for simple chat interactions.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaap import GAAPEngine, GAAPRequest


async def simple_chat():
    """Simple chat example"""

    engine = GAAPEngine(
        budget=10.0,
        enable_healing=True,
        enable_memory=True,
    )

    request = GAAPRequest(
        text="Write a Python function to calculate the Fibonacci sequence",
        priority="NORMAL",
    )

    response = await engine.process(request)

    print(f"Success: {response.success}")
    print(f"Output: {response.output}")
    print(f"Time: {response.total_time_ms:.0f}ms")
    print(f"Cost: ${response.total_cost_usd:.4f}")

    engine.shutdown()


async def quick_chat():
    """Even simpler quick chat"""

    engine = GAAPEngine(budget=1.0)

    response = await engine.chat("What is the capital of France?")
    print(response)

    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(simple_chat())
