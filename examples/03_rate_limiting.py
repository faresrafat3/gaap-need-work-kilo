"""
GAAP Rate Limiting Example

This example demonstrates the different rate limiting strategies available.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaap.core.rate_limiter import (
    RateLimitConfig,
    RateLimitStrategy,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    AdaptiveRateLimiter,
    create_rate_limiter,
)


async def token_bucket_example():
    """Token Bucket rate limiter example"""
    print("\n=== Token Bucket Example ===")

    config = RateLimitConfig(
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        requests_per_second=10,
        burst_capacity=20,
    )

    limiter = TokenBucketRateLimiter(config)

    for i in range(5):
        result = await limiter.acquire(1)
        print(f"Request {i + 1}: allowed={result.allowed}, remaining={result.tokens_remaining:.1f}")

    print(f"\nStats: {limiter.get_stats()}")


async def sliding_window_example():
    """Sliding Window rate limiter example"""
    print("\n=== Sliding Window Example ===")

    config = RateLimitConfig(
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        window_size_seconds=1.0,
        max_requests_per_window=5,
    )

    limiter = SlidingWindowRateLimiter(config)

    for i in range(7):
        result = await limiter.acquire(1)
        print(f"Request {i + 1}: allowed={result.allowed}")
        await asyncio.sleep(0.1)

    print(f"\nStats: {limiter.get_stats()}")


async def adaptive_example():
    """Adaptive rate limiter example"""
    print("\n=== Adaptive Rate Limiter Example ===")

    config = RateLimitConfig(
        strategy=RateLimitStrategy.ADAPTIVE,
        requests_per_second=10,
        burst_capacity=100,
        adaptive_increase_factor=1.5,
        adaptive_decrease_factor=0.7,
    )

    limiter = AdaptiveRateLimiter(config)

    for i in range(20):
        result = await limiter.acquire(1)
        if i % 5 == 0:
            stats = limiter.get_stats()
            print(
                f"Request {i + 1}: rate={stats['adaptive_rate']:.1f}, success_rate={stats['success_rate']:.1%}"
            )

    print(f"\nFinal Stats: {limiter.get_stats()}")


async def wait_for_token_example():
    """Example of waiting for available tokens"""
    print("\n=== Wait for Token Example ===")

    config = RateLimitConfig(
        requests_per_second=2,
        burst_capacity=2,
    )

    limiter = TokenBucketRateLimiter(config)

    for i in range(5):
        got_token = await limiter.wait_for_token(timeout=2.0)
        print(f"Request {i + 1}: got_token={got_token}")


async def main():
    """Run all examples"""
    await token_bucket_example()
    await sliding_window_example()
    await adaptive_example()
    await wait_for_token_example()

    print("\n=== Using Factory Function ===")
    limiter = create_rate_limiter(
        RateLimitConfig(
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_second=50,
        )
    )
    print(f"Created: {type(limiter).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
