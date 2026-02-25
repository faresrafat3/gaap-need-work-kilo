#!/usr/bin/env python3
"""
GAAP Benchmark Suite - Comprehensive Performance Testing

Tests real performance of:
1. MCTS Search (iterations, node creation, time)
2. Layer1 Strategic Planning
3. Swarm Orchestrator
4. Metacognition Engine
5. Memory Operations
6. Full Pipeline Integration

Run: python scripts/benchmark.py
"""

import asyncio
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    ops_per_second: float
    details: dict[str, Any]


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_result(result: BenchmarkResult) -> None:
    print(f"\n  📊 {result.name}")
    print(f"     Iterations: {result.iterations}")
    print(f"     Total Time: {result.total_time * 1000:.2f}ms")
    print(f"     Avg Time:   {result.avg_time * 1000:.2f}ms")
    print(f"     Min/Max:    {result.min_time * 1000:.2f}ms / {result.max_time * 1000:.2f}ms")
    print(f"     Ops/sec:    {result.ops_per_second:.2f}")
    if result.details:
        for k, v in result.details.items():
            print(f"     {k}: {v}")


def benchmark_function(
    name: str,
    func: callable,
    iterations: int = 100,
    warmup: int = 5,
    **kwargs,
) -> BenchmarkResult:
    """Benchmark a synchronous function."""

    # Warmup
    for _ in range(warmup):
        func(**kwargs)

    # Actual benchmark
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        func(**kwargs)
        end = time.perf_counter()
        times.append(end - start)

    total_time = sum(times)
    avg_time = statistics.mean(times)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total_time,
        avg_time=avg_time,
        min_time=min(times),
        max_time=max(times),
        ops_per_second=iterations / total_time if total_time > 0 else 0,
        details={},
    )


async def benchmark_async_function(
    name: str,
    func: callable,
    iterations: int = 100,
    warmup: int = 5,
    **kwargs,
) -> BenchmarkResult:
    """Benchmark an async function."""

    # Warmup
    for _ in range(warmup):
        await func(**kwargs)

    # Actual benchmark
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        await func(**kwargs)
        end = time.perf_counter()
        times.append(end - start)

    total_time = sum(times)
    avg_time = statistics.mean(times)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total_time,
        avg_time=avg_time,
        min_time=min(times),
        max_time=max(times),
        ops_per_second=iterations / total_time if total_time > 0 else 0,
        details={},
    )


# =============================================================================
# MCTS Benchmarks
# =============================================================================


def benchmark_mcts_node() -> BenchmarkResult:
    """Benchmark MCTS node operations."""
    from gaap.layers.mcts_logic import MCTSNode

    def node_ops():
        root = MCTSNode(content="Root")
        for i in range(10):
            child = root.add_child(f"Child_{i}")
            child.update(0.5 + i * 0.05)
            _ = child.uct_score()
        return root.visits

    return benchmark_function("MCTS Node Operations", node_ops, iterations=1000)


async def benchmark_mcts_search_small() -> BenchmarkResult:
    """Benchmark MCTS search with small config."""
    from gaap.layers.mcts_logic import MCTSConfig, MCTSStrategic
    from unittest.mock import MagicMock

    config = MCTSConfig(iterations=10, max_depth=3)
    mcts = MCTSStrategic(config=config)
    intent = MagicMock()

    async def run_search():
        return await mcts.search(intent, {}, "Test")

    return await benchmark_async_function("MCTS Search (10 iter)", run_search, iterations=50)


async def benchmark_mcts_search_medium() -> BenchmarkResult:
    """Benchmark MCTS search with medium config."""
    from gaap.layers.mcts_logic import MCTSConfig, MCTSStrategic
    from unittest.mock import MagicMock

    config = MCTSConfig(iterations=50, max_depth=4)
    mcts = MCTSStrategic(config=config)
    intent = MagicMock()

    async def run_search():
        return await mcts.search(intent, {}, "Test")

    return await benchmark_async_function("MCTS Search (50 iter)", run_search, iterations=20)


# =============================================================================
# Layer Benchmarks
# =============================================================================


async def benchmark_layer1_fallback() -> BenchmarkResult:
    """Benchmark Layer1 strategic planning (fallback mode)."""
    from gaap.layers import Layer1Strategic
    from gaap.layers.layer0_interface import StructuredIntent
    from datetime import datetime

    layer1 = Layer1Strategic(enable_mcts=False)

    async def process_intent():
        intent = StructuredIntent(
            request_id="bench",
            timestamp=datetime.now(),
            explicit_goals=["Build a REST API"],
        )
        return await layer1.process(intent)

    return await benchmark_async_function(
        "Layer1 Fallback (ToT+MAD)", process_intent, iterations=30
    )


# =============================================================================
# Swarm Benchmarks
# =============================================================================


def benchmark_swarm_orchestrator() -> BenchmarkResult:
    """Benchmark Swarm Orchestrator creation and operations."""
    from gaap.swarm import SwarmOrchestrator, TaskAuction
    from gaap.swarm.reputation import ReputationStore, TaskDomain

    def swarm_ops():
        orchestrator = SwarmOrchestrator()
        auction = TaskAuction()
        rep = ReputationStore()
        rep.register_fractal("agent_1", "Coder", TaskDomain.PYTHON)
        rep.register_fractal("agent_2", "Critic", TaskDomain.SECURITY)
        return len(rep._profiles)

    return benchmark_function("Swarm Orchestrator", swarm_ops, iterations=500)


# =============================================================================
# Metacognition Benchmarks
# =============================================================================


def benchmark_metacognition() -> BenchmarkResult:
    """Benchmark Metacognition Engine."""
    from gaap.meta_learning import create_metacognition_engine
    from unittest.mock import MagicMock

    engine = create_metacognition_engine()
    task = MagicMock()
    task.description = "Test task for benchmarking"

    def assess():
        return engine._confidence_scorer.calculate_confidence(task)

    return benchmark_function("Metacognition Assessment", assess, iterations=200)


# =============================================================================
# Memory Benchmarks
# =============================================================================


def benchmark_vector_store() -> BenchmarkResult:
    """Benchmark Vector Store operations."""
    from gaap.memory.vector_store import VectorStore
    import uuid

    store = VectorStore(collection_name="bench_test")
    store.reset()

    def store_ops():
        entry_id = str(uuid.uuid4())
        store.add(f"Test content {entry_id}", {"type": "bench"})
        results = store.search("Test", n_results=5)
        return len(results)

    result = benchmark_function("VectorStore Operations", store_ops, iterations=100)
    store.reset()
    return result


def benchmark_lesson_store() -> BenchmarkResult:
    """Benchmark Lesson Store operations."""
    from gaap.memory import LessonStore

    store = LessonStore()

    def store_lesson():
        return store.add_lesson(
            lesson="Test lesson for benchmarking",
            category="performance",
            task_type="benchmark",
            success=True,
        )

    return benchmark_function("LessonStore Operations", store_lesson, iterations=100)


# =============================================================================
# Governance Benchmarks
# =============================================================================


def benchmark_governance() -> BenchmarkResult:
    """Benchmark SOP Governance."""
    from gaap.core.governance import create_sop_gatekeeper

    gatekeeper = create_sop_gatekeeper()

    def governance_ops():
        execution = gatekeeper.start_execution("coder", "task_123")
        gatekeeper.complete_step("task_123", 1, "output")
        return gatekeeper.check_completion("task_123")

    return benchmark_function("SOP Governance", governance_ops, iterations=500)


# =============================================================================
# Integration Benchmarks
# =============================================================================


async def benchmark_full_pipeline() -> BenchmarkResult:
    """Benchmark full pipeline: Layer0 -> Layer1 -> MCTS."""
    from gaap.layers import Layer1Strategic
    from gaap.layers.layer0_interface import StructuredIntent
    from datetime import datetime

    layer1 = Layer1Strategic(enable_mcts=True)

    async def pipeline():
        intent = StructuredIntent(
            request_id="pipeline_bench",
            timestamp=datetime.now(),
            explicit_goals=["Design scalable API"],
        )
        intent.metadata = {"priority": None, "complexity": None}
        return await layer1.process(intent)

    return await benchmark_async_function("Full Pipeline (L0->L1+MCTS)", pipeline, iterations=20)


# =============================================================================
# Main Benchmark Runner
# =============================================================================


async def run_benchmarks() -> list[BenchmarkResult]:
    """Run all benchmarks."""
    results = []

    print_header("GAAP Benchmark Suite")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Python: {sys.version.split()[0]}")

    # Synchronous benchmarks
    print("\n--- Running Synchronous Benchmarks ---")
    results.append(benchmark_mcts_node())
    print_result(results[-1])

    results.append(benchmark_swarm_orchestrator())
    print_result(results[-1])

    results.append(benchmark_metacognition())
    print_result(results[-1])

    results.append(benchmark_vector_store())
    print_result(results[-1])

    results.append(benchmark_lesson_store())
    print_result(results[-1])

    results.append(benchmark_governance())
    print_result(results[-1])

    # Async benchmarks
    print("\n--- Running Async Benchmarks ---")

    results.append(await benchmark_mcts_search_small())
    print_result(results[-1])

    results.append(await benchmark_mcts_search_medium())
    print_result(results[-1])

    results.append(await benchmark_layer1_fallback())
    print_result(results[-1])

    results.append(await benchmark_full_pipeline())
    print_result(results[-1])

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print benchmark summary."""
    print_header("BENCHMARK SUMMARY")

    total_ops = sum(r.iterations for r in results)
    total_time = sum(r.total_time for r in results)

    print(f"\n  Total Operations: {total_ops}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Overall Ops/sec: {total_ops / total_time:.2f}" if total_time > 0 else "")

    print("\n  Results by Component:")
    print(f"  {'Component':<35} {'Avg Time':>12} {'Ops/sec':>12}")
    print(f"  {'-' * 35} {'-' * 12} {'-' * 12}")

    for r in results:
        print(f"  {r.name:<35} {r.avg_time * 1000:>10.2f}ms {r.ops_per_second:>12.2f}")

    # Performance ratings
    print("\n  Performance Ratings:")
    for r in results:
        if r.ops_per_second > 100:
            rating = "🚀 Excellent"
        elif r.ops_per_second > 50:
            rating = "✅ Good"
        elif r.ops_per_second > 10:
            rating = "⚠️  Acceptable"
        else:
            rating = "🐌 Slow"
        print(f"    {r.name}: {rating}")

    print(f"\n{'=' * 70}")


def main() -> int:
    """Main entry point."""
    results = asyncio.run(run_benchmarks())
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
