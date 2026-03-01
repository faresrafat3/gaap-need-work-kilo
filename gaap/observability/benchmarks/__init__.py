"""
GAAP Benchmark Framework

Benchmark runner framework for GAAP:
- Compare before/after performance
- Report generation
- Regression detection
- Baseline management

Usage:
    from gaap.observability.benchmarks import BenchmarkRunner, BenchmarkResult

    runner = BenchmarkRunner()

    # Define benchmark
    @runner.benchmark(name="sort_algorithm", iterations=100)
    def benchmark_sort():
        return sorted(data)

    # Run all benchmarks
    results = runner.run_all()

    # Compare with baseline
    comparison = runner.compare_with_baseline(results, "baseline.json")
"""

from __future__ import annotations

import json
import logging
import statistics
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generator, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger("gaap.observability.benchmarks")

T = TypeVar("T")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BenchmarkMeasurement:
    """A single benchmark measurement."""

    duration_sec: float
    memory_delta_mb: float = 0.0
    iterations: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkStats:
    """Statistics for a benchmark."""

    name: str = ""
    iterations: int = 0
    total_time_sec: float = 0.0
    avg_time_sec: float = 0.0
    min_time_sec: float = 0.0
    max_time_sec: float = 0.0
    median_time_sec: float = 0.0
    std_dev_sec: float = 0.0
    p95_time_sec: float = 0.0
    p99_time_sec: float = 0.0
    ops_per_sec: float = 0.0
    memory_avg_mb: float = 0.0
    memory_peak_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_sec": round(self.total_time_sec, 6),
            "avg_time_sec": round(self.avg_time_sec, 6),
            "min_time_sec": round(self.min_time_sec, 6),
            "max_time_sec": round(self.max_time_sec, 6),
            "median_time_sec": round(self.median_time_sec, 6),
            "std_dev_sec": round(self.std_dev_sec, 6),
            "p95_time_sec": round(self.p95_time_sec, 6),
            "p99_time_sec": round(self.p99_time_sec, 6),
            "ops_per_sec": round(self.ops_per_sec, 2),
            "memory_avg_mb": round(self.memory_avg_mb, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
        }


@dataclass
class BenchmarkResult:
    """Complete result from a benchmark run."""

    name: str
    stats: BenchmarkStats
    measurements: list[BenchmarkMeasurement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "stats": self.stats.to_dict(),
            "measurements_count": len(self.measurements),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


@dataclass
class ComparisonResult:
    """Comparison between two benchmark results."""

    benchmark_name: str
    baseline_stats: BenchmarkStats | None = None
    current_stats: BenchmarkStats | None = None
    time_change_pct: float = 0.0
    memory_change_pct: float = 0.0
    ops_change_pct: float = 0.0
    is_regression: bool = False
    is_improvement: bool = False
    regression_threshold_pct: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "baseline": self.baseline_stats.to_dict() if self.baseline_stats else None,
            "current": self.current_stats.to_dict() if self.current_stats else None,
            "time_change_pct": round(self.time_change_pct, 2),
            "memory_change_pct": round(self.memory_change_pct, 2),
            "ops_change_pct": round(self.ops_change_pct, 2),
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
        }


@dataclass
class BenchmarkSuiteResult:
    """Results from running a benchmark suite."""

    name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration_sec: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "results": [r.to_dict() for r in self.results],
            "total_duration_sec": round(self.total_duration_sec, 2),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    default_iterations: int = 100
    warmup_iterations: int = 10
    min_iterations: int = 10
    max_iterations: int = 10000
    max_duration_sec: float = 60.0
    enable_memory_tracking: bool = True
    enable_gc_between_runs: bool = True
    regression_threshold_pct: float = 10.0
    improvement_threshold_pct: float = 10.0
    output_dir: str = "./benchmark_results"


# =============================================================================
# Benchmark Runner
# =============================================================================


@runtime_checkable
class BenchmarkFunction(Protocol):
    """Protocol for benchmark functions."""

    def __call__(self) -> Any: ...


class BenchmarkRunner:
    """
    Benchmark runner framework for GAAP.

    Features:
    - Run benchmarks with multiple iterations
    - Track memory usage
    - Compare against baselines
    - Generate reports
    - Detect regressions

    Usage:
        runner = BenchmarkRunner()

        @runner.benchmark(name="my_func", iterations=1000)
        def bench_my_func():
            return my_function()

        results = runner.run_all()
        report = runner.generate_report(results)
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        self._config = config or BenchmarkConfig()
        self._benchmarks: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {}
        self._results: list[BenchmarkResult] = []

        # Ensure output directory exists
        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)

    def benchmark(
        self,
        name: str | None = None,
        iterations: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator to register a benchmark function.

        Args:
            name: Benchmark name (defaults to function name)
            iterations: Number of iterations (uses config default if None)
            metadata: Additional metadata for the benchmark

        Returns:
            Decorated function
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            bench_name = name or func.__name__
            bench_iterations = iterations or self._config.default_iterations

            self._benchmarks[bench_name] = (
                func,
                {
                    "iterations": bench_iterations,
                    "metadata": metadata or {},
                    "original_func": func,
                },
            )

            return func

        return decorator

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        iterations: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a benchmark function programmatically.

        Args:
            func: Function to benchmark
            name: Benchmark name
            iterations: Number of iterations
            metadata: Additional metadata
        """
        bench_name = name or func.__name__
        bench_iterations = iterations or self._config.default_iterations

        self._benchmarks[bench_name] = (
            func,
            {
                "iterations": bench_iterations,
                "metadata": metadata or {},
                "original_func": func,
            },
        )

    @contextmanager
    def _measure_context(self) -> Generator[dict[str, Any], None, None]:
        """Context manager for measuring a single iteration."""
        import gc

        if self._config.enable_gc_between_runs:
            gc.collect()

        start_mem = 0
        peak_mem = 0

        if self._config.enable_memory_tracking:
            tracemalloc.start()
            start_mem = tracemalloc.get_traced_memory()[0]

        start_time = time.perf_counter()

        result = {
            "duration": 0.0,
            "memory_delta": 0.0,
            "peak_memory": 0.0,
        }

        try:
            yield result
        finally:
            duration = time.perf_counter() - start_time

            if self._config.enable_memory_tracking:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                result["memory_delta"] = (current - start_mem) / (1024 * 1024)
                result["peak_memory"] = peak / (1024 * 1024)

            result["duration"] = duration

    def _run_single(
        self,
        func: Callable[..., Any],
        warmup: bool = False,
    ) -> BenchmarkMeasurement:
        """Run a single benchmark iteration."""
        with self._measure_context() as metrics:
            func()

        return BenchmarkMeasurement(
            duration_sec=metrics["duration"],
            memory_delta_mb=metrics["memory_delta"],
            iterations=1,
        )

    def run_benchmark(self, name: str) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            name: Name of the registered benchmark

        Returns:
            BenchmarkResult with statistics
        """
        if name not in self._benchmarks:
            raise ValueError(f"Benchmark '{name}' not found")

        func, config = self._benchmarks[name]
        iterations = config["iterations"]
        metadata = config["metadata"]

        logger.info(f"Running benchmark '{name}' with {iterations} iterations")

        # Warmup
        if self._config.warmup_iterations > 0:
            logger.debug(f"Warming up with {self._config.warmup_iterations} iterations")
            for _ in range(self._config.warmup_iterations):
                func()

        # Run iterations
        measurements: list[BenchmarkMeasurement] = []
        start_time = time.time()

        try:
            for i in range(iterations):
                # Check max duration
                if time.time() - start_time > self._config.max_duration_sec:
                    logger.warning(
                        f"Benchmark '{name}' exceeded max duration, stopping at {i} iterations"
                    )
                    break

                measurement = self._run_single(func)
                measurements.append(measurement)

            # Calculate statistics
            durations = [m.duration_sec for m in measurements]
            memories = [m.memory_delta_mb for m in measurements if m.memory_delta_mb > 0]

            stats = BenchmarkStats(
                name=name,
                iterations=len(measurements),
                total_time_sec=sum(durations),
                avg_time_sec=statistics.mean(durations),
                min_time_sec=min(durations),
                max_time_sec=max(durations),
                median_time_sec=statistics.median(durations),
                std_dev_sec=statistics.stdev(durations) if len(durations) > 1 else 0.0,
                p95_time_sec=self._percentile(durations, 95),
                p99_time_sec=self._percentile(durations, 99),
                ops_per_sec=1.0 / statistics.mean(durations) if durations else 0.0,
                memory_avg_mb=statistics.mean(memories) if memories else 0.0,
                memory_peak_mb=max(memories) if memories else 0.0,
            )

            result = BenchmarkResult(
                name=name,
                stats=stats,
                measurements=measurements,
                metadata=metadata,
            )

            self._results.append(result)
            logger.info(f"Benchmark '{name}' completed: {stats.avg_time_sec:.6f}s avg")

            return result

        except Exception as e:
            logger.error(f"Benchmark '{name}' failed: {e}")
            return BenchmarkResult(
                name=name,
                stats=BenchmarkStats(name=name),
                error=str(e),
                metadata=metadata,
            )

    def _percentile(self, values: list[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def run_all(self, pattern: str | None = None) -> BenchmarkSuiteResult:
        """
        Run all registered benchmarks.

        Args:
            pattern: Optional pattern to filter benchmark names

        Returns:
            BenchmarkSuiteResult with all results
        """
        start_time = time.time()
        results: list[BenchmarkResult] = []

        benchmarks = list(self._benchmarks.keys())
        if pattern:
            benchmarks = [b for b in benchmarks if pattern in b]

        logger.info(f"Running {len(benchmarks)} benchmarks")

        for name in benchmarks:
            result = self.run_benchmark(name)
            results.append(result)

        total_duration = time.time() - start_time

        return BenchmarkSuiteResult(
            name="benchmark_suite",
            results=results,
            total_duration_sec=total_duration,
            metadata={"benchmark_count": len(results)},
        )

    def compare_with_baseline(
        self,
        current: BenchmarkResult | BenchmarkSuiteResult,
        baseline_path: str,
    ) -> list[ComparisonResult]:
        """
        Compare current results with a baseline.

        Args:
            current: Current benchmark results
            baseline_path: Path to baseline JSON file

        Returns:
            List of ComparisonResult
        """
        # Load baseline
        baseline_path_obj = Path(baseline_path)
        if not baseline_path_obj.exists():
            logger.warning(f"Baseline file not found: {baseline_path}")
            return []

        with open(baseline_path) as f:
            baseline_data = json.load(f)

        comparisons: list[ComparisonResult] = []

        # Handle both single result and suite
        current_results = (
            current.results if isinstance(current, BenchmarkSuiteResult) else [current]
        )

        for current_result in current_results:
            name = current_result.name
            baseline_result = None

            # Find matching baseline
            if "results" in baseline_data:
                for r in baseline_data["results"]:
                    if r["name"] == name:
                        baseline_result = r
                        break
            elif baseline_data.get("name") == name:
                baseline_result = baseline_data

            if baseline_result and baseline_result.get("stats"):
                baseline_stats = BenchmarkStats(**baseline_result["stats"])
                comparison = self._compare_stats(name, baseline_stats, current_result.stats)
                comparisons.append(comparison)
            else:
                logger.warning(f"No baseline found for benchmark '{name}'")

        return comparisons

    def _compare_stats(
        self,
        name: str,
        baseline: BenchmarkStats,
        current: BenchmarkStats,
    ) -> ComparisonResult:
        """Compare two benchmark statistics."""
        # Calculate changes
        time_change = (
            ((current.avg_time_sec - baseline.avg_time_sec) / baseline.avg_time_sec * 100)
            if baseline.avg_time_sec > 0
            else 0
        )
        memory_change = (
            ((current.memory_avg_mb - baseline.memory_avg_mb) / baseline.memory_avg_mb * 100)
            if baseline.memory_avg_mb > 0
            else 0
        )
        ops_change = (
            ((current.ops_per_sec - baseline.ops_per_sec) / baseline.ops_per_sec * 100)
            if baseline.ops_per_sec > 0
            else 0
        )

        # Determine regression/improvement
        is_regression = time_change > self._config.regression_threshold_pct
        is_improvement = time_change < -self._config.improvement_threshold_pct

        return ComparisonResult(
            benchmark_name=name,
            baseline_stats=baseline,
            current_stats=current,
            time_change_pct=time_change,
            memory_change_pct=memory_change,
            ops_change_pct=ops_change,
            is_regression=is_regression,
            is_improvement=is_improvement,
        )

    def save_baseline(self, result: BenchmarkResult | BenchmarkSuiteResult, path: str) -> None:
        """
        Save benchmark result as baseline.

        Args:
            result: Result to save
            path: Path to save baseline
        """
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Baseline saved to {path}")

    def generate_report(
        self,
        suite_result: BenchmarkSuiteResult,
        comparisons: list[ComparisonResult] | None = None,
    ) -> str:
        """
        Generate a human-readable report.

        Args:
            suite_result: Benchmark suite results
            comparisons: Optional comparison results

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "GAAP Benchmark Report",
            "=" * 70,
            f"Timestamp: {suite_result.timestamp.isoformat()}",
            f"Total Duration: {suite_result.total_duration_sec:.2f}s",
            f"Benchmarks Run: {len(suite_result.results)}",
            "",
        ]

        # Individual results
        lines.append("Benchmark Results:")
        lines.append("-" * 70)

        for result in suite_result.results:
            if result.error:
                lines.append(f"\n{result.name}: FAILED - {result.error}")
                continue

            stats = result.stats
            lines.extend(
                [
                    f"\n{result.name}:",
                    f"  Iterations: {stats.iterations}",
                    f"  Avg Time: {stats.avg_time_sec * 1000:.3f} ms",
                    f"  Min/Max: {stats.min_time_sec * 1000:.3f} / {stats.max_time_sec * 1000:.3f} ms",
                    f"  P95/P99: {stats.p95_time_sec * 1000:.3f} / {stats.p99_time_sec * 1000:.3f} ms",
                    f"  Ops/sec: {stats.ops_per_sec:.2f}",
                ]
            )

            if stats.memory_avg_mb > 0:
                lines.append(f"  Memory Avg: {stats.memory_avg_mb:.2f} MB")

        # Comparisons
        if comparisons:
            lines.extend(
                [
                    "",
                    "Comparison with Baseline:",
                    "-" * 70,
                ]
            )

            for comp in comparisons:
                status = "✓" if not comp.is_regression else "✗ REGRESSION"
                if comp.is_improvement:
                    status = "↑ IMPROVEMENT"

                lines.extend(
                    [
                        f"\n{comp.benchmark_name}: {status}",
                        f"  Time Change: {comp.time_change_pct:+.1f}%",
                        f"  Ops Change: {comp.ops_change_pct:+.1f}%",
                    ]
                )

                if comp.memory_change_pct != 0:
                    lines.append(f"  Memory Change: {comp.memory_change_pct:+.1f}%")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def save_report(self, suite_result: BenchmarkSuiteResult, filename: str | None = None) -> str:
        """
        Save benchmark report to file.

        Args:
            suite_result: Benchmark suite results
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.json"

        path = Path(self._config.output_dir) / filename

        with open(path, "w") as f:
            json.dump(suite_result.to_dict(), f, indent=2)

        logger.info(f"Report saved to {path}")
        return str(path)


# =============================================================================
# Convenience Functions
# =============================================================================


def run_benchmark(
    func: Callable[..., Any],
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """
    Quick function to run a single benchmark.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Warmup iterations

    Returns:
        BenchmarkResult
    """
    runner = BenchmarkRunner(BenchmarkConfig(warmup_iterations=warmup))
    runner.register(func, name=func.__name__, iterations=iterations)
    return runner.run_benchmark(func.__name__)


def compare_performance(
    baseline_func: Callable[..., Any],
    current_func: Callable[..., Any],
    iterations: int = 100,
) -> ComparisonResult:
    """
    Compare two functions for performance.

    Args:
        baseline_func: Baseline implementation
        current_func: New implementation
        iterations: Number of iterations

    Returns:
        ComparisonResult
    """
    runner = BenchmarkRunner()

    runner.register(baseline_func, name="baseline", iterations=iterations)
    runner.register(current_func, name="current", iterations=iterations)

    baseline_result = runner.run_benchmark("baseline")
    current_result = runner.run_benchmark("current")

    return runner._compare_stats(
        baseline_func.__name__,
        baseline_result.stats,
        current_result.stats,
    )
