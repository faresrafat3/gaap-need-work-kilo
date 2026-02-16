#!/usr/bin/env python3
"""
GAAP Quick Start Example

This example demonstrates the simplest way to use GAAP.
"""

import asyncio

from gaap import GAAPEngine, GAAPRequest


async def main():
    print("=== GAAP Quick Start Example ===\n")

    # Create engine with default settings
    # Uses UnifiedGAAPProvider (Kimi-first with fallbacks)
    engine = GAAPEngine(budget=10.0)

    # Create request
    request = GAAPRequest(text="Write a Python function for binary search", priority="NORMAL")

    # Process request
    print("Processing request...")
    response = await engine.process(request)

    # Display results
    print("\n--- Response ---")
    print(f"Success: {response.success}")
    print(f"Request ID: {response.request_id}")

    if response.output:
        print(f"\nOutput:\n{response.output}")

    if response.error:
        print(f"\nError: {response.error}")

    print("\n--- Metrics ---")
    print(f"Total time: {response.total_time_ms:.0f}ms")
    print(f"Total cost: ${response.total_cost_usd:.4f}")
    print(f"Total tokens: {response.total_tokens}")
    print(f"Quality score: {response.quality_score:.1f}/100")

    # Cleanup
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
