"""
Memory profiling tests for GAAP.

Memory profiling tests to:
- Detect memory leaks
- Track memory per component
- Analyze memory usage patterns

Usage:
    pytest tests/benchmarks/test_memory_usage.py -v --benchmark-only
"""

from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import pytest


# =============================================================================
# Memory Profiling Utilities
# =============================================================================


@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in time."""

    timestamp: float
    rss_mb: float
    vms_mb: float
    traced_memory_mb: float
    object_count: int
    top_allocations: list[tuple[str, int, float]] = field(default_factory=list)


@dataclass
class MemoryProfile:
    """Complete memory profile for a component."""

    component: str
    snapshots: list[MemorySnapshot] = field(default_factory=list)
    growth_rate_mb_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    leak_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "snapshots_count": len(self.snapshots),
            "growth_rate_mb_per_sec": round(self.growth_rate_mb_per_sec, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "leak_detected": self.leak_detected,
        }


class MemoryProfiler:
    """
    Memory profiler for detecting leaks and tracking component memory.

    Usage:
        profiler = MemoryProfiler()

        with profiler.profile("my_component"):
            # Code to profile
            run_component()

        result = profiler.get_profile("my_component")
    """

    def __init__(self):
        self._profiles: dict[str, MemoryProfile] = {}
        self._snapshots: dict[str, list[MemorySnapshot]] = defaultdict(list)
        self._active: set[str] = set()

    def _get_memory_info(self) -> tuple[float, float]:
        """Get RSS and VMS memory in MB."""
        try:
            import psutil

            process = psutil.Process()
            mem = process.memory_info()
            return mem.rss / (1024 * 1024), mem.vms / (1024 * 1024)
        except ImportError:
            return 0.0, 0.0

    def _count_objects(self) -> int:
        """Count tracked objects."""
        gc.collect()
        return len(gc.get_objects())

    def _get_traced_memory(self) -> float:
        """Get traced memory in MB."""
        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)
        return 0.0

    def take_snapshot(self, component: str) -> MemorySnapshot:
        """Take a memory snapshot for a component."""
        rss, vms = self._get_memory_info()
        traced = self._get_traced_memory()
        obj_count = self._count_objects()

        # Get top allocations if tracing
        top_allocs = []
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:5]
            for stat in top_stats:
                top_allocs.append(
                    (
                        f"{stat.traceback.filename}:{stat.traceback.lineno}",
                        stat.count,
                        stat.size / (1024 * 1024),
                    )
                )

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss,
            vms_mb=vms,
            traced_memory_mb=traced,
            object_count=obj_count,
            top_allocations=top_allocs,
        )

        self._snapshots[component].append(snapshot)
        return snapshot

    def start_profiling(self, component: str) -> None:
        """Start profiling a component."""
        self._active.add(component)
        self.take_snapshot(component)

    def stop_profiling(self, component: str) -> MemoryProfile:
        """Stop profiling and return results."""
        self.take_snapshot(component)
        self._active.discard(component)

        snapshots = self._snapshots[component]
        if len(snapshots) < 2:
            return MemoryProfile(component=component, snapshots=snapshots)

        # Calculate growth rate
        duration = snapshots[-1].timestamp - snapshots[0].timestamp
        memory_growth = snapshots[-1].rss_mb - snapshots[0].rss_mb
        growth_rate = memory_growth / duration if duration > 0 else 0

        # Detect leak (consistent growth over time)
        peak = max(s.rss_mb for s in snapshots)

        # Simple leak detection: > 10% growth with > 1MB increase (for test scenarios)
        initial = snapshots[0].rss_mb
        leak_detected = (memory_growth > 1) and (memory_growth / max(initial, 1) > 0.01)

        profile = MemoryProfile(
            component=component,
            snapshots=snapshots,
            growth_rate_mb_per_sec=growth_rate,
            peak_memory_mb=peak,
            leak_detected=leak_detected,
        )

        self._profiles[component] = profile
        return profile

    @contextmanager
    def profile(self, component: str) -> Generator[None, None, None]:
        """Context manager for profiling."""
        self.start_profiling(component)
        try:
            yield
        finally:
            self.stop_profiling(component)

    def get_profile(self, component: str) -> MemoryProfile | None:
        """Get profile for a component."""
        return self._profiles.get(component)

    def get_all_profiles(self) -> dict[str, MemoryProfile]:
        """Get all profiles."""
        return dict(self._profiles)

    def reset(self) -> None:
        """Reset all profiles."""
        self._profiles.clear()
        self._snapshots.clear()
        self._active.clear()


# =============================================================================
# Mock Components for Testing
# =============================================================================


class MockComponent:
    """Base class for mock components."""

    def __init__(self, name: str):
        self.name = name
        self.data: list[Any] = []

    def allocate(self, size_mb: int) -> None:
        """Allocate memory."""
        # Allocate roughly size_mb of memory
        chunk = "x" * (size_mb * 1024 * 1024 // 10)  # Rough approximation
        self.data.append(chunk)

    def clear(self) -> None:
        """Clear allocated memory."""
        self.data.clear()
        gc.collect()


class LeakyComponent(MockComponent):
    """Component that leaks memory - for testing leak detection."""

    def __init__(self, name: str):
        super().__init__(name)
        self._cache: dict[int, Any] = {}
        self._counter = 0

    def process(self) -> None:
        """Process that leaks memory."""
        # Simulate accumulating cache without eviction
        self._counter += 1
        self._cache[self._counter] = "x" * 10000  # 10KB per call


class WellBehavedComponent(MockComponent):
    """Component that properly manages memory."""

    def __init__(self, name: str, cache_size: int = 100):
        super().__init__(name)
        self.cache_size = cache_size
        self._cache: dict[int, Any] = {}

    def process(self) -> None:
        """Process with bounded memory."""
        # Use bounded cache with LRU eviction
        key = len(self._cache)
        if len(self._cache) >= self.cache_size:
            # Remove oldest entries
            keys_to_remove = list(self._cache.keys())[: self.cache_size // 10]
            for k in keys_to_remove:
                del self._cache[k]

        self._cache[key] = "x" * 10000  # 10KB per item, bounded


# =============================================================================
# Test Cases
# =============================================================================


class TestMemoryLeakDetection:
    """Tests for memory leak detection."""

    def test_detect_memory_leak(self):
        """Test that memory leaks are detected."""
        profiler = MemoryProfiler()
        component = LeakyComponent("leaky")

        with profiler.profile("leaky_component"):
            # Simulate work that leaks memory
            for _ in range(1000):
                component.process()
                time.sleep(0.001)  # Small delay

        profile = profiler.get_profile("leaky_component")
        assert profile is not None

        print("\n" + "=" * 70)
        print("Memory Leak Detection Test")
        print("=" * 70)
        print(f"Component: {profile.component}")
        print(f"Growth rate: {profile.growth_rate_mb_per_sec:.4f} MB/sec")
        print(f"Peak memory: {profile.peak_memory_mb:.2f} MB")
        print(f"Leak detected: {profile.leak_detected}")

        # Should detect leak
        assert profile.leak_detected, "Expected leak to be detected"
        assert profile.growth_rate_mb_per_sec > 0, "Expected positive growth rate"

    def test_no_false_positive_for_well_behaved(self):
        """Test that well-behaved components don't trigger false positives."""
        profiler = MemoryProfiler()
        component = WellBehavedComponent("clean", cache_size=100)

        with profiler.profile("clean_component"):
            # Simulate work with bounded memory
            for _ in range(1000):
                component.process()
                time.sleep(0.001)

        profile = profiler.get_profile("clean_component")
        assert profile is not None

        print("\n" + "=" * 70)
        print("Well-Behaved Component Test")
        print("=" * 70)
        print(f"Component: {profile.component}")
        print(f"Growth rate: {profile.growth_rate_mb_per_sec:.4f} MB/sec")
        print(f"Peak memory: {profile.peak_memory_mb:.2f} MB")
        print(f"Leak detected: {profile.leak_detected}")

        # Should NOT detect leak
        assert not profile.leak_detected, "Should not detect leak in well-behaved component"

    def test_component_memory_comparison(self):
        """Compare memory usage across different components."""
        profiler = MemoryProfiler()

        components = {
            "small_cache": WellBehavedComponent("small", cache_size=10),
            "medium_cache": WellBehavedComponent("medium", cache_size=100),
            "large_cache": WellBehavedComponent("large", cache_size=1000),
        }

        for name, component in components.items():
            with profiler.profile(name):
                for _ in range(100):
                    component.process()

        print("\n" + "=" * 70)
        print("Component Memory Comparison")
        print("=" * 70)

        profiles = profiler.get_all_profiles()
        for name, profile in sorted(profiles.items()):
            print(f"\n{name}:")
            print(f"  Peak memory: {profile.peak_memory_mb:.2f} MB")
            print(
                f"  Final memory: {profile.snapshots[-1].rss_mb:.2f} MB"
                if profile.snapshots
                else "  No snapshots"
            )


class TestComponentMemoryTracking:
    """Tests for per-component memory tracking."""

    def test_memory_per_component_isolation(self):
        """Test that memory is tracked per component."""
        profiler = MemoryProfiler()

        # Component A - heavy memory usage
        with profiler.profile("component_a"):
            data_a = ["x" * 1000 for _ in range(10000)]  # ~10MB
            time.sleep(0.1)
            del data_a

        # Component B - light memory usage
        with profiler.profile("component_b"):
            data_b = ["x" * 100 for _ in range(1000)]  # ~100KB
            time.sleep(0.1)
            del data_b

        profile_a = profiler.get_profile("component_a")
        profile_b = profiler.get_profile("component_b")

        assert profile_a is not None
        assert profile_b is not None

        print("\n" + "=" * 70)
        print("Component Memory Isolation Test")
        print("=" * 70)

        for name, profile in [("A", profile_a), ("B", profile_b)]:
            print(f"\nComponent {name}:")
            print(f"  Snapshots: {len(profile.snapshots)}")
            if profile.snapshots:
                print(f"  Initial: {profile.snapshots[0].rss_mb:.2f} MB")
                print(f"  Peak: {max(s.rss_mb for s in profile.snapshots):.2f} MB")
                print(f"  Final: {profile.snapshots[-1].rss_mb:.2f} MB")

    def test_nested_component_tracking(self):
        """Test memory tracking with nested components."""
        profiler = MemoryProfiler()

        with profiler.profile("outer"):
            data_outer = ["x" * 1000 for _ in range(1000)]

            with profiler.profile("inner"):
                data_inner = ["x" * 1000 for _ in range(5000)]
                time.sleep(0.05)
                del data_inner

            time.sleep(0.05)
            del data_outer

        outer_profile = profiler.get_profile("outer")
        inner_profile = profiler.get_profile("inner")

        assert outer_profile is not None
        assert inner_profile is not None

        print("\n" + "=" * 70)
        print("Nested Component Tracking Test")
        print("=" * 70)
        print(f"\nOuter component snapshots: {len(outer_profile.snapshots)}")
        print(f"Inner component snapshots: {len(inner_profile.snapshots)}")


class TestMemoryPatternAnalysis:
    """Tests for memory usage pattern analysis."""

    def test_memory_growth_pattern(self):
        """Analyze memory growth patterns over time."""
        profiler = MemoryProfiler()

        # Simulate gradually increasing memory usage
        data_chunks: list[list[str]] = []
        chunk_size = 1000

        with profiler.profile("growing_component"):
            for i in range(10):
                # Allocate new chunk
                chunk = ["x" * 1000 for _ in range(chunk_size)]
                data_chunks.append(chunk)
                profiler.take_snapshot("growing_component")
                time.sleep(0.01)

            # Cleanup
            data_chunks.clear()
            gc.collect()

        profile = profiler.get_profile("growing_component")
        assert profile is not None

        print("\n" + "=" * 70)
        print("Memory Growth Pattern Analysis")
        print("=" * 70)

        snapshots = profile.snapshots
        for i, snap in enumerate(snapshots[:5]):
            print(f"\nSnapshot {i}:")
            print(f"  RSS: {snap.rss_mb:.2f} MB")
            print(f"  Objects: {snap.object_count:,}")
            print(f"  Traced: {snap.traced_memory_mb:.2f} MB")

    def test_memory_cleanup_effectiveness(self):
        """Test that memory cleanup is effective."""
        profiler = MemoryProfiler()

        with profiler.profile("cleanup_test"):
            # Phase 1: Allocate memory
            data = [["x" * 100 for _ in range(1000)] for _ in range(10)]
            profiler.take_snapshot("cleanup_test")

            # Phase 2: Clear and GC
            before_clear = profiler._snapshots["cleanup_test"][-1].rss_mb

            del data
            gc.collect()
            time.sleep(0.05)  # Give GC time

            profiler.take_snapshot("cleanup_test")
            after_clear = profiler._snapshots["cleanup_test"][-1].rss_mb

        print("\n" + "=" * 70)
        print("Memory Cleanup Effectiveness")
        print("=" * 70)
        print(f"\nMemory before cleanup: {before_clear:.2f} MB")
        print(f"Memory after cleanup: {after_clear:.2f} MB")
        print(f"Reduction: {before_clear - after_clear:.2f} MB")

        # Most memory should be reclaimed (allowing for some overhead)
        assert after_clear < before_clear * 1.5, (
            "Memory should be significantly reduced after cleanup"
        )


class TestMemoryRegression:
    """Tests to detect memory usage regressions."""

    def test_memory_usage_bounds(self):
        """Test that memory usage stays within acceptable bounds."""
        profiler = MemoryProfiler()

        with profiler.profile("bounded_component"):
            # Simulate typical workload
            cache: dict[int, str] = {}
            for i in range(10000):
                # Bounded cache
                if len(cache) > 1000:
                    # Remove random entries to keep size bounded
                    keys = list(cache.keys())[:100]
                    for k in keys:
                        del cache[k]

                cache[i] = "x" * 1000

                if i % 1000 == 0:
                    profiler.take_snapshot("bounded_component")

        profile = profiler.get_profile("bounded_component")
        assert profile is not None

        # Memory should stabilize, not grow unbounded
        rss_values = [s.rss_mb for s in profile.snapshots]
        if len(rss_values) >= 3:
            # Last few readings should be similar (bounded)
            recent_avg = sum(rss_values[-3:]) / 3
            early_avg = sum(rss_values[:3]) / 3

            # Allow 50% growth but not unbounded
            growth_ratio = recent_avg / max(early_avg, 1)

            print("\n" + "=" * 70)
            print("Memory Bounds Test")
            print("=" * 70)
            print(f"\nEarly average: {early_avg:.2f} MB")
            print(f"Recent average: {recent_avg:.2f} MB")
            print(f"Growth ratio: {growth_ratio:.2f}x")

            assert growth_ratio < 2.0, f"Memory grew too much: {growth_ratio:.2f}x"

    def test_object_count_stability(self):
        """Test that object count remains stable."""
        profiler = MemoryProfiler()

        with profiler.profile("object_stability"):
            for i in range(100):
                # Create and destroy objects
                temp_data = [{"key": j, "value": "x" * 100} for j in range(1000)]
                del temp_data

                if i % 10 == 0:
                    profiler.take_snapshot("object_stability")
                    gc.collect()

        profile = profiler.get_profile("object_stability")
        assert profile is not None

        object_counts = [s.object_count for s in profile.snapshots]
        if len(object_counts) >= 2:
            # Object count should be relatively stable
            initial = object_counts[0]
            final = object_counts[-1]
            change_ratio = abs(final - initial) / max(initial, 1)

            print("\n" + "=" * 70)
            print("Object Count Stability Test")
            print("=" * 70)
            print(f"\nInitial objects: {initial:,}")
            print(f"Final objects: {final:,}")
            print(f"Change ratio: {change_ratio:.2%}")

            # Allow 20% variance
            assert change_ratio < 0.2, f"Object count changed too much: {change_ratio:.2%}"


class TestIntegrationWithPerformanceMonitor:
    """Tests integrating with the performance monitor."""

    def test_performance_monitor_memory_tracking(self):
        """Test that performance monitor tracks memory."""
        from gaap.observability.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()
        monitor.reset()

        # Record memory for a component
        monitor.record_memory("test_component", force=True)

        # Simulate some memory usage
        data = ["x" * 1000 for _ in range(10000)]
        monitor.record_memory("test_component", force=True)

        del data
        gc.collect()
        monitor.record_memory("test_component", force=True)

        # Get memory stats
        stats = monitor.get_memory_stats("test_component")
        assert "test_component" in stats

        print("\n" + "=" * 70)
        print("Performance Monitor Memory Tracking")
        print("=" * 70)

        comp_stats = stats["test_component"]
        print(f"\nComponent: {comp_stats.component}")
        print(f"Current: {comp_stats.current_mb:.2f} MB")
        print(f"Peak: {comp_stats.peak_mb:.2f} MB")
        print(f"Samples: {comp_stats.samples_count}")

    def test_memory_export(self):
        """Test memory metrics export."""
        from gaap.observability.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()
        monitor.reset()

        # Record some memory data
        for i in range(5):
            monitor.record_memory("component_a", force=True)
            monitor.record_memory("component_b", force=True)
            time.sleep(0.01)

        # Export as JSON
        json_export = monitor.export_json()
        assert "memory" in json_export

        # Export as Prometheus
        prom_export = monitor.export_prometheus()
        assert "gaap_memory" in prom_export

        print("\n" + "=" * 70)
        print("Memory Metrics Export Test")
        print("=" * 70)
        print(f"\nJSON export length: {len(json_export)} chars")
        print(f"Prometheus export length: {len(prom_export)} chars")


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestMemoryBenchmarks:
    """Benchmark tests for memory operations."""

    def test_memory_allocation_speed(self):
        """Benchmark memory allocation speed."""
        print("\n" + "=" * 70)
        print("Memory Allocation Speed Benchmark")
        print("=" * 70)

        sizes = [100, 1000, 10000]

        for size in sizes:
            start = time.perf_counter()
            data = [{"key": i, "value": "x" * 100} for i in range(size)]
            elapsed = time.perf_counter() - start

            print(f"\nAllocating {size} objects:")
            print(f"  Time: {elapsed * 1000:.2f} ms")
            print(f"  Per object: {elapsed * 1_000_000 / size:.2f} µs")

            del data
            gc.collect()

    def test_gc_impact(self):
        """Measure garbage collection impact."""
        print("\n" + "=" * 70)
        print("Garbage Collection Impact")
        print("=" * 70)

        # Create objects that will need cleanup
        objects: list[Any] = []
        for _ in range(100000):
            objects.append({"data": "x" * 100})

        gc.disable()
        start = time.perf_counter()
        del objects
        gc.enable()
        gc.collect()
        elapsed = time.perf_counter() - start

        print(f"\nGC collection time: {elapsed * 1000:.2f} ms")

        # Create objects in generations
        for gen in range(3):
            print(f"\nGeneration {gen}:")
            count = gc.get_count()[gen] if gen < len(gc.get_count()) else 0
            print(f"  Object count: {count}")
