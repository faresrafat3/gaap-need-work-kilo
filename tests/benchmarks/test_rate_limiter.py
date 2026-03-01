"""
Benchmark tests for Rate Limiter performance.

Measures the current O(n) bottleneck in rate limiting to demonstrate
need for optimization. Tests scaling behavior with increasing request counts.

Usage:
    pytest tests/benchmarks/test_rate_limiter.py -v --benchmark-only
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from gaap.observability.benchmarks import BenchmarkRunner, BenchmarkConfig, run_benchmark


# =============================================================================
# Mock Rate Limiter Implementations (to demonstrate O(n) bottleneck)
# =============================================================================


class SimpleRateLimiter:
    """
    Simple O(n) rate limiter - demonstrates the bottleneck.
    Stores all request timestamps in a list and scans on each check.
    """

    def __init__(self, max_requests: int = 100, window_sec: float = 60.0):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.requests: list[float] = []

    def is_allowed(self) -> bool:
        """Check if request is allowed - O(n) scan."""
        now = time.time()
        cutoff = now - self.window_sec

        # O(n) scan to remove old requests
        self.requests = [ts for ts in self.requests if ts > cutoff]

        # Check limit
        if len(self.requests) >= self.max_requests:
            return False

        self.requests.append(now)
        return True


class OptimizedRateLimiter:
    """
    Optimized O(1) rate limiter using circular buffer.
    Demonstrates the target optimization.
    """

    def __init__(self, max_requests: int = 100, window_sec: float = 60.0):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.requests: list[float | None] = [None] * max_requests
        self.index = 0
        self.count = 0

    def is_allowed(self) -> bool:
        """Check if request is allowed - O(1)."""
        now = time.time()
        cutoff = now - self.window_sec

        # Check oldest request (circular buffer)
        if self.count >= self.max_requests:
            oldest = self.requests[self.index]
            if oldest is not None and oldest > cutoff:
                return False

        # Add request - O(1)
        self.requests[self.index] = now
        self.index = (self.index + 1) % self.max_requests
        if self.count < self.max_requests:
            self.count += 1

        return True


class TokenBucketRateLimiter:
    """Token bucket rate limiter - O(1) per request."""

    def __init__(self, rate: float = 10.0, burst: int = 100):
        self.rate = rate  # tokens per second
        self.burst = burst  # max tokens
        self.tokens = float(burst)
        self.last_update = time.time()

    def is_allowed(self) -> bool:
        """Check if request is allowed - O(1)."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


# =============================================================================
# Benchmark Functions
# =============================================================================


@pytest.fixture
def benchmark_runner():
    """Create a benchmark runner with appropriate config."""
    config = BenchmarkConfig(
        default_iterations=1000,
        warmup_iterations=100,
        enable_memory_tracking=True,
    )
    return BenchmarkRunner(config)


class TestRateLimiterScaling:
    """Test rate limiter scaling behavior."""

    def test_simple_rate_limiter_scaling(self, benchmark_runner):
        """Benchmark O(n) rate limiter with increasing request counts."""

        def bench_small():
            limiter = SimpleRateLimiter(max_requests=10, window_sec=60.0)
            for _ in range(10):
                limiter.is_allowed()

        def bench_medium():
            limiter = SimpleRateLimiter(max_requests=100, window_sec=60.0)
            for _ in range(100):
                limiter.is_allowed()

        def bench_large():
            limiter = SimpleRateLimiter(max_requests=1000, window_sec=60.0)
            for _ in range(1000):
                limiter.is_allowed()

        benchmark_runner.register(bench_small, name="rate_limiter_simple_small", iterations=100)
        benchmark_runner.register(bench_medium, name="rate_limiter_simple_medium", iterations=100)
        benchmark_runner.register(bench_large, name="rate_limiter_simple_large", iterations=100)

        results = benchmark_runner.run_all()

        # Print scaling analysis
        print("\n" + "=" * 70)
        print("O(n) Rate Limiter Scaling Analysis")
        print("=" * 70)

        for result in results.results:
            if result.error:
                print(f"\n{result.name}: FAILED - {result.error}")
                continue

            stats = result.stats
            print(f"\n{result.name}:")
            print(f"  Requests processed: {stats.iterations}")
            print(f"  Total time: {stats.total_time_sec * 1000:.2f} ms")
            print(f"  Avg per iteration: {stats.avg_time_sec * 1000:.3f} ms")
            print(f"  Ops/sec: {stats.ops_per_sec:.2f}")

        # Verify O(n) behavior - large should be significantly slower per operation
        small_result = next(r for r in results.results if "small" in r.name)
        large_result = next(r for r in results.results if "large" in r.name)

        if small_result.stats.avg_time_sec > 0:
            slowdown_ratio = large_result.stats.avg_time_sec / small_result.stats.avg_time_sec
            print(f"\nScaling Factor (large/small): {slowdown_ratio:.2f}x")
            print("Expected: >5x slowdown for O(n) behavior with 100x requests")

    def test_optimized_rate_limiter_scaling(self, benchmark_runner):
        """Benchmark O(1) rate limiter with increasing request counts."""

        def bench_small():
            limiter = OptimizedRateLimiter(max_requests=10, window_sec=60.0)
            for _ in range(10):
                limiter.is_allowed()

        def bench_medium():
            limiter = OptimizedRateLimiter(max_requests=100, window_sec=60.0)
            for _ in range(100):
                limiter.is_allowed()

        def bench_large():
            limiter = OptimizedRateLimiter(max_requests=1000, window_sec=60.0)
            for _ in range(1000):
                limiter.is_allowed()

        benchmark_runner.register(bench_small, name="rate_limiter_optimized_small", iterations=100)
        benchmark_runner.register(
            bench_medium, name="rate_limiter_optimized_medium", iterations=100
        )
        benchmark_runner.register(bench_large, name="rate_limiter_optimized_large", iterations=100)

        results = benchmark_runner.run_all()

        print("\n" + "=" * 70)
        print("O(1) Rate Limiter Scaling Analysis")
        print("=" * 70)

        for result in results.results:
            if result.error:
                print(f"\n{result.name}: FAILED - {result.error}")
                continue

            stats = result.stats
            print(f"\n{result.name}:")
            print(f"  Requests processed: {stats.iterations}")
            print(f"  Total time: {stats.total_time_sec * 1000:.2f} ms")
            print(f"  Avg per iteration: {stats.avg_time_sec * 1000:.3f} ms")
            print(f"  Ops/sec: {stats.ops_per_sec:.2f}")

        # O(1) should show similar per-operation cost regardless of size
        small_result = next(r for r in results.results if "small" in r.name)
        large_result = next(r for r in results.results if "large" in r.name)

        if small_result.stats.avg_time_sec > 0:
            ratio = large_result.stats.avg_time_sec / small_result.stats.avg_time_sec
            print(f"\nScaling Factor (large/small): {ratio:.2f}x")
            print("Expected: ~1x (similar performance regardless of request count)")

    def test_comparison_simple_vs_optimized(self, benchmark_runner):
        """Compare O(n) vs O(1) rate limiters head-to-head."""

        def bench_simple():
            limiter = SimpleRateLimiter(max_requests=1000, window_sec=60.0)
            for _ in range(1000):
                limiter.is_allowed()

        def bench_optimized():
            limiter = OptimizedRateLimiter(max_requests=1000, window_sec=60.0)
            for _ in range(1000):
                limiter.is_allowed()

        def bench_token_bucket():
            limiter = TokenBucketRateLimiter(rate=100.0, burst=1000)
            for _ in range(1000):
                limiter.is_allowed()

        benchmark_runner.register(bench_simple, name="simple_O(n)", iterations=100)
        benchmark_runner.register(bench_optimized, name="optimized_O(1)", iterations=100)
        benchmark_runner.register(bench_token_bucket, name="token_bucket_O(1)", iterations=100)

        results = benchmark_runner.run_all()

        print("\n" + "=" * 70)
        print("Rate Limiter Comparison: O(n) vs O(1)")
        print("=" * 70)

        simple_time = None
        for result in results.results:
            if result.error:
                print(f"\n{result.name}: FAILED")
                continue

            stats = result.stats
            print(f"\n{result.name}:")
            print(f"  Avg time: {stats.avg_time_sec * 1000:.3f} ms")
            print(f"  Ops/sec: {stats.ops_per_sec:.2f}")
            print(f"  Memory: {stats.memory_avg_mb:.2f} MB avg")

            if "simple" in result.name.lower():
                simple_time = stats.avg_time_sec
            elif simple_time and simple_time > 0:
                speedup = simple_time / stats.avg_time_sec
                print(f"  Speedup vs O(n): {speedup:.1f}x")

    def test_concurrent_rate_limiting(self):
        """Benchmark rate limiter under concurrent load."""
        limiter = SimpleRateLimiter(max_requests=10000, window_sec=60.0)

        async def concurrent_requests(n: int) -> list[bool]:
            """Simulate n concurrent requests."""
            tasks = [asyncio.to_thread(limiter.is_allowed) for _ in range(n)]
            return await asyncio.gather(*tasks)

        async def run_benchmark():
            start = time.perf_counter()
            results = await concurrent_requests(1000)
            elapsed = time.perf_counter() - start

            allowed = sum(1 for r in results if r)
            print(f"\nConcurrent Benchmark (1000 requests):")
            print(f"  Total time: {elapsed * 1000:.2f} ms")
            print(f"  Avg per request: {elapsed * 1000 / 1000:.3f} ms")
            print(f"  Requests allowed: {allowed}/1000")
            print(f"  Throughput: {1000 / elapsed:.2f} req/sec")

            return elapsed

        asyncio.run(run_benchmark())


class TestRateLimiterMemory:
    """Test rate limiter memory usage patterns."""

    def test_memory_growth_simple_limiter(self):
        """Test memory growth in O(n) rate limiter over time."""
        import tracemalloc

        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        limiter = SimpleRateLimiter(max_requests=10000, window_sec=3600.0)

        # Simulate many requests
        memory_samples = []
        for i in range(1000):
            limiter.is_allowed()
            if i % 100 == 0:
                current, _ = tracemalloc.get_traced_memory()
                memory_samples.append((i, (current - start_mem) / (1024 * 1024)))

        tracemalloc.stop()

        print("\n" + "=" * 70)
        print("O(n) Rate Limiter Memory Growth")
        print("=" * 70)

        for req_count, mem_mb in memory_samples:
            print(f"  Requests: {req_count:4d}, Memory growth: {mem_mb:.3f} MB")

        # Check memory growth pattern
        if len(memory_samples) >= 2:
            initial_growth = memory_samples[1][1]
            final_growth = memory_samples[-1][1]
            growth_ratio = final_growth / max(initial_growth, 0.001)
            print(f"\nMemory growth ratio (final/initial): {growth_ratio:.2f}x")
            print("Expected: Linear growth with request count for O(n) implementation")

    def test_memory_usage_comparison(self):
        """Compare memory usage between implementations."""
        import tracemalloc

        def measure_memory(limiter_class, requests: int):
            tracemalloc.start()
            start_mem = tracemalloc.get_traced_memory()[0]

            limiter = limiter_class(max_requests=10000, window_sec=3600.0)
            for _ in range(requests):
                limiter.is_allowed()

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return {
                "final_mb": (current - start_mem) / (1024 * 1024),
                "peak_mb": (peak - start_mem) / (1024 * 1024),
            }

        print("\n" + "=" * 70)
        print("Memory Usage Comparison (1000 requests)")
        print("=" * 70)

        simple_mem = measure_memory(SimpleRateLimiter, 1000)
        optimized_mem = measure_memory(OptimizedRateLimiter, 1000)

        print(f"\nSimpleRateLimiter (O(n)):")
        print(f"  Final memory: {simple_mem['final_mb']:.3f} MB")
        print(f"  Peak memory: {simple_mem['peak_mb']:.3f} MB")

        print(f"\nOptimizedRateLimiter (O(1)):")
        print(f"  Final memory: {optimized_mem['final_mb']:.3f} MB")
        print(f"  Peak memory: {optimized_mem['peak_mb']:.3f} MB")

        if simple_mem["final_mb"] > 0:
            ratio = simple_mem["final_mb"] / max(optimized_mem["final_mb"], 0.001)
            print(f"\nMemory efficiency: {ratio:.1f}x less memory with optimized version")


class TestThroughputAnalysis:
    """Analyze throughput characteristics."""

    def test_sustained_throughput(self):
        """Test sustained throughput over time."""
        print("\n" + "=" * 70)
        print("Sustained Throughput Analysis")
        print("=" * 70)

        implementations = [
            ("Simple O(n)", SimpleRateLimiter),
            ("Optimized O(1)", OptimizedRateLimiter),
            ("Token Bucket", TokenBucketRateLimiter),
        ]

        for name, limiter_class in implementations:
            if limiter_class == TokenBucketRateLimiter:
                limiter = limiter_class(rate=10000.0, burst=100000)
            else:
                limiter = limiter_class(max_requests=100000, window_sec=3600.0)

            # Warmup
            for _ in range(100):
                limiter.is_allowed()

            # Measure sustained throughput
            start = time.perf_counter()
            request_count = 10000
            for _ in range(request_count):
                limiter.is_allowed()
            elapsed = time.perf_counter() - start

            throughput = request_count / elapsed
            latency_us = (elapsed / request_count) * 1_000_000

            print(f"\n{name}:")
            print(f"  Throughput: {throughput:,.0f} req/sec")
            print(f"  Latency: {latency_us:.1f} µs/op")

    def test_burst_handling(self):
        """Test rate limiter burst handling performance."""
        print("\n" + "=" * 70)
        print("Burst Handling Performance")
        print("=" * 70)

        burst_sizes = [10, 100, 1000]

        for burst_size in burst_sizes:
            print(f"\nBurst size: {burst_size}")

            limiter = SimpleRateLimiter(max_requests=burst_size * 2, window_sec=60.0)

            start = time.perf_counter()
            allowed = 0
            for _ in range(burst_size):
                if limiter.is_allowed():
                    allowed += 1
            elapsed = time.perf_counter() - start

            print(f"  Time to process burst: {elapsed * 1000:.2f} ms")
            print(f"  Requests allowed: {allowed}/{burst_size}")
            print(f"  Throughput: {burst_size / elapsed:,.0f} req/sec")


# =============================================================================
# Performance Regression Tests
# =============================================================================


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    def test_rate_limiter_latency_regression(self):
        """Test that rate limiter latency stays within acceptable bounds."""
        limiter = SimpleRateLimiter(max_requests=1000, window_sec=60.0)

        # Measure p99 latency
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            limiter.is_allowed()
            elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
            latencies.append(elapsed)

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        p95 = latencies[int(len(latencies) * 0.95)]
        avg = sum(latencies) / len(latencies)

        print("\n" + "=" * 70)
        print("Rate Limiter Latency Analysis")
        print("=" * 70)
        print(f"  Average: {avg:.1f} µs")
        print(f"  P95: {p95:.1f} µs")
        print(f"  P99: {p99:.1f} µs")
        print(f"  Max: {max(latencies):.1f} µs")

        # Assert acceptable performance
        # These thresholds should be adjusted based on target hardware
        assert avg < 1000, f"Average latency too high: {avg:.1f} µs"
        assert p99 < 5000, f"P99 latency too high: {p99:.1f} µs"
