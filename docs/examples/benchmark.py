#!/usr/bin/env python3
"""
GAAP Benchmark Example

This example shows how to run benchmarks to test provider performance.
"""

import asyncio
import time

from gaap import GAAPEngine, GAAPRequest
from gaap.providers import GroqProvider, GeminiProvider


async def benchmark_single_request(engine: GAAPEngine, prompt: str, name: str) -> dict:
    """Benchmark a single request."""
    start = time.time()

    request = GAAPRequest(text=prompt, priority="NORMAL")
    response = await engine.process(request)

    elapsed = time.time() - start

    return {
        "name": name,
        "success": response.success,
        "latency_ms": elapsed * 1000,
        "tokens": response.total_tokens,
        "cost_usd": response.total_cost_usd,
        "quality": response.quality_score,
    }


async def run_benchmarks():
    print("=== GAAP Benchmark Example ===\n")

    # Test prompts
    prompts = [
        "What is 2 + 2?",
        "Write a haiku about programming",
        "Explain quantum computing in one paragraph",
        "Write a Python function to reverse a string",
        "List 5 benefits of exercise",
    ]

    # Providers to benchmark
    providers_config = [
        ("Groq", lambda: GroqProvider(api_key="gsk_your_key")),
        # Add more providers as needed
    ]

    results = []

    for provider_name, provider_factory in providers_config:
        print(f"\nBenchmarking {provider_name}...")

        provider = provider_factory()
        engine = GAAPEngine(providers=[provider], budget=100.0)

        for i, prompt in enumerate(prompts):
            result = await benchmark_single_request(engine, prompt, f"{provider_name}_{i}")
            results.append(result)
            print(f"  Prompt {i + 1}: {result['latency_ms']:.0f}ms")

        engine.shutdown()

    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"{'Provider':<15} {'Avg Latency':>12} {'Success':>8} {'Avg Quality':>12}")
    print("-" * 50)

    for provider_name, _ in providers_config:
        provider_results = [r for r in results if r["name"].startswith(provider_name)]

        avg_latency = sum(r["latency_ms"] for r in provider_results) / len(provider_results)
        success_rate = sum(1 for r in provider_results if r["success"]) / len(provider_results)
        avg_quality = sum(r["quality"] for r in provider_results) / len(provider_results)

        print(
            f"{provider_name:<15} {avg_latency:>10.0f}ms {success_rate:>8.0%} {avg_quality:>12.1f}"
        )


async def main():
    await run_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
