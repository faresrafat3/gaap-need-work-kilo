"""
GAAP Performance Monitor Module

Comprehensive performance monitoring system for GAAP:
- Track latency metrics (p50, p95, p99)
- Track memory usage per component
- Track throughput (requests/sec)
- Track error rates
- Context manager for timing: `with monitor.timing("operation"):`
- Decorator for functions: `@monitor.timed`

Usage:
    from gaap.observability.performance_monitor import PerformanceMonitor, get_performance_monitor

    # Get singleton instance
    monitor = get_performance_monitor()

    # Context manager for timing
    with monitor.timing("database_query", component="storage"):
        result = db.query()

    # Decorator for functions
    @monitor.timed(component="router")
    def route_request(request):
        return process(request)

    # Get metrics
    metrics = monitor.get_metrics()
    print(f"p95 latency: {metrics.latency.p95_ms}ms")
    print(f"Throughput: {metrics.throughput.requests_per_sec}/sec")
"""

from __future__ import annotations

import functools
import gc
import logging
import random
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger("gaap.observability.performance")

T = TypeVar("T")


# =============================================================================
# Configuration Classes
# =============================================================================


class SamplingStrategy(Enum):
    """Sampling strategies for performance data collection."""

    ALL = "all"  # Collect all samples (100%)
    ADAPTIVE = "adaptive"  # Adaptive based on load
    FIXED = "fixed"  # Fixed percentage
    PROBABILISTIC = "probabilistic"  # Random sampling
    ON_DEMAND = "on_demand"  # Only when explicitly requested


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring.

    Attributes:
        enabled: Whether monitoring is enabled
        sampling_strategy: How to sample performance data
        sampling_rate: Fixed sampling rate (0.0-1.0) for FIXED/PROBABILISTIC
        max_samples_per_metric: Maximum samples to keep per metric
        max_age_minutes: Maximum age of samples before cleanup
        enable_memory_tracking: Whether to track memory usage
        enable_throughput: Whether to track throughput
        enable_latency_percentiles: Whether to calculate percentiles
        gc_during_cleanup: Whether to run GC during cleanup
        export_format: Default export format (json, prometheus)
    """

    enabled: bool = True
    sampling_strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE
    sampling_rate: float = 0.1  # 10% default for fixed/probabilistic
    max_samples_per_metric: int = 10000
    max_age_minutes: int = 60
    enable_memory_tracking: bool = True
    enable_throughput: bool = True
    enable_latency_percentiles: bool = True
    gc_during_cleanup: bool = True
    export_format: str = "json"

    # Component-specific settings
    component_sampling: dict[str, float] = field(default_factory=dict)
    component_memory_tracking: dict[str, bool] = field(default_factory=dict)


# =============================================================================
# Metric Data Classes
# =============================================================================


@dataclass
class LatencySample:
    """A single latency measurement."""

    operation: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics for an operation or component."""

    operation: str = ""
    component: str = ""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    avg_ms: float = 0.0
    std_dev_ms: float = 0.0
    samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes raw samples for size)."""
        return {
            "operation": self.operation,
            "component": self.component,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
        }


@dataclass
class MemorySample:
    """A single memory measurement."""

    component: str
    bytes_used: int
    timestamp: datetime = field(default_factory=datetime.now)
    rss_mb: float = 0.0
    vms_mb: float = 0.0


@dataclass
class MemoryStats:
    """Memory statistics for a component."""

    component: str = ""
    current_bytes: int = 0
    peak_bytes: int = 0
    avg_bytes: float = 0.0
    samples_count: int = 0
    rss_mb: float = 0.0
    vms_mb: float = 0.0

    @property
    def current_mb(self) -> float:
        """Current memory usage in MB."""
        return round(self.current_bytes / (1024 * 1024), 2)

    @property
    def peak_mb(self) -> float:
        """Peak memory usage in MB."""
        return round(self.peak_bytes / (1024 * 1024), 2)

    @property
    def avg_mb(self) -> float:
        """Average memory usage in MB."""
        return round(self.avg_bytes / (1024 * 1024), 2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "current_bytes": self.current_bytes,
            "current_mb": self.current_mb,
            "peak_bytes": self.peak_bytes,
            "peak_mb": self.peak_mb,
            "avg_bytes": round(self.avg_bytes, 0),
            "avg_mb": self.avg_mb,
            "samples_count": self.samples_count,
            "rss_mb": round(self.rss_mb, 2),
            "vms_mb": round(self.vms_mb, 2),
        }


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    operation: str = ""
    total_requests: int = 0
    requests_per_sec: float = 0.0
    requests_per_min: float = 0.0
    window_start: datetime = field(default_factory=datetime.now)
    window_duration_sec: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "total_requests": self.total_requests,
            "requests_per_sec": round(self.requests_per_sec, 2),
            "requests_per_min": round(self.requests_per_min, 2),
            "window_start": self.window_start.isoformat(),
            "window_duration_sec": self.window_duration_sec,
        }


@dataclass
class ErrorStats:
    """Error statistics."""

    component: str = ""
    total_errors: int = 0
    total_calls: int = 0
    error_rate: float = 0.0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    recent_errors: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "total_errors": self.total_errors,
            "total_calls": self.total_calls,
            "error_rate": round(self.error_rate, 4),
            "errors_by_type": dict(self.errors_by_type),
            "recent_errors_count": len(self.recent_errors),
        }


@dataclass
class PerformanceMetrics:
    """Complete performance metrics snapshot."""

    timestamp: datetime = field(default_factory=datetime.now)
    latency: dict[str, LatencyStats] = field(default_factory=dict)
    memory: dict[str, MemoryStats] = field(default_factory=dict)
    throughput: dict[str, ThroughputStats] = field(default_factory=dict)
    errors: dict[str, ErrorStats] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "latency": {k: v.to_dict() for k, v in self.latency.items()},
            "memory": {k: v.to_dict() for k, v in self.memory.items()},
            "throughput": {k: v.to_dict() for k, v in self.throughput.items()},
            "errors": {k: v.to_dict() for k, v in self.errors.items()},
            "system": self.system,
        }


# =============================================================================
# Core Performance Monitor
# =============================================================================


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for GAAP.

    Singleton class providing:
    - Latency tracking with percentiles (p50, p95, p99)
    - Memory usage tracking per component
    - Throughput measurement (requests/sec)
    - Error rate tracking
    - Context manager for timing
    - Function decorator for automatic timing

    Thread-safe and async-safe with minimal overhead (<1%).

    Usage:
        monitor = PerformanceMonitor()

        # Context manager
        with monitor.timing("db_query", component="storage"):
            result = database.query()

        # Decorator
        @monitor.timed(component="router")
        def process(request):
            return handle(request)

        # Get metrics
        metrics = monitor.get_metrics()
    """

    _instance: PerformanceMonitor | None = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> PerformanceMonitor:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: PerformanceConfig | None = None) -> None:
        if self._initialized:
            return

        self._config = config or PerformanceConfig()

        # Thread-safe storage
        self._latency_samples: dict[str, deque[LatencySample]] = defaultdict(
            lambda: deque(maxlen=self._config.max_samples_per_metric)
        )
        self._memory_samples: dict[str, deque[MemorySample]] = defaultdict(
            lambda: deque(maxlen=self._config.max_samples_per_metric)
        )
        self._throughput_counters: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
        self._error_stats: dict[str, ErrorStats] = defaultdict(lambda: ErrorStats(component=""))

        # Active timing contexts (for nested timing)
        self._active_timers: dict[int, list[tuple[str, float]]] = {}
        self._timer_lock = threading.Lock()

        # Memory tracking state
        self._memory_tracking_enabled = self._config.enable_memory_tracking
        self._tracemalloc_started = False

        # Cleanup state
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

        # Adaptive sampling state
        self._current_sampling_rate = self._config.sampling_rate
        self._adaptive_load_factor = 1.0

        self._initialized = True
        logger.info(
            f"PerformanceMonitor initialized (strategy={self._config.sampling_strategy.value}, "
            f"sampling_rate={self._current_sampling_rate:.2%})"
        )

    def _should_sample(self, component: str = "") -> bool:
        """Determine if current operation should be sampled."""
        if not self._config.enabled:
            return False

        # Check component-specific sampling
        if component and component in self._config.component_sampling:
            return random.random() < self._config.component_sampling[component]

        # Apply strategy
        if self._config.sampling_strategy == SamplingStrategy.ALL:
            return True
        elif self._config.sampling_strategy == SamplingStrategy.ON_DEMAND:
            return False
        elif self._config.sampling_strategy == SamplingStrategy.FIXED:
            return random.random() < self._config.sampling_rate
        elif self._config.sampling_strategy == SamplingStrategy.PROBABILISTIC:
            return random.random() < self._current_sampling_rate
        elif self._config.sampling_strategy == SamplingStrategy.ADAPTIVE:
            return random.random() < self._current_sampling_rate

        return True

    def _update_adaptive_rate(self, load_factor: float) -> None:
        """Update adaptive sampling rate based on load."""
        if self._config.sampling_strategy != SamplingStrategy.ADAPTIVE:
            return

        self._adaptive_load_factor = load_factor
        # Higher load = lower sampling rate (min 0.01)
        self._current_sampling_rate = max(
            0.01, min(1.0, self._config.sampling_rate / max(1.0, load_factor))
        )

    def _get_process_memory(self) -> tuple[float, float]:
        """Get current process memory (rss, vms) in MB."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / (1024 * 1024), mem_info.vms / (1024 * 1024)
        except ImportError:
            return 0.0, 0.0

    def _maybe_cleanup(self) -> None:
        """Perform periodic cleanup of old data."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=self._config.max_age_minutes)

            # Cleanup latency samples
            for key, samples in self._latency_samples.items():
                while samples and samples[0].timestamp < cutoff:
                    samples.popleft()

            # Cleanup memory samples
            for key, samples in self._memory_samples.items():
                while samples and samples[0].timestamp < cutoff:
                    samples.popleft()

            # Cleanup throughput counters
            cutoff_ts = datetime.now() - timedelta(minutes=self._config.max_age_minutes)
            for key, counters in self._throughput_counters.items():
                self._throughput_counters[key] = [
                    (ts, count) for ts, count in counters if ts > cutoff_ts
                ]

            if self._config.gc_during_cleanup:
                gc.collect()

            self._last_cleanup = now
            logger.debug("PerformanceMonitor cleanup completed")

    # =====================================================================
    # Context Manager for Timing
    # =====================================================================

    @contextmanager
    def timing(
        self,
        operation: str,
        component: str = "",
        tags: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager for timing operations.

        Usage:
            with monitor.timing("database_query", component="storage"):
                result = db.execute(query)

        Args:
            operation: Name of the operation being timed
            component: Component name for categorization
            tags: Additional tags for the measurement

        Yields:
            None
        """
        if not self._should_sample(component):
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            sample = LatencySample(
                operation=operation,
                duration_ms=duration_ms,
                component=component,
                tags=tags or {},
            )

            key = f"{component}:{operation}" if component else operation
            with self._lock:
                self._latency_samples[key].append(sample)

            # Track throughput
            if self._config.enable_throughput:
                self._track_throughput(key)

            self._maybe_cleanup()

    # =====================================================================
    # Decorator for Functions
    # =====================================================================

    def timed(
        self,
        operation: str | None = None,
        component: str = "",
        tags: dict[str, str] | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator for timing function calls.

        Usage:
            @monitor.timed(component="router")
            def route_request(request):
                return process(request)

            @monitor.timed(operation="custom_name", component="database")
            def query(data):
                return db.fetch(data)

        Args:
            operation: Operation name (defaults to function name)
            component: Component name for categorization
            tags: Additional tags for the measurement

        Returns:
            Decorated function
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            op_name = operation or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                with self.timing(op_name, component, tags):
                    return func(*args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                with self.timing(op_name, component, tags):
                    return await func(*args, **kwargs)

            # Return async wrapper if function is coroutine
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return wrapper

        return decorator

    # =====================================================================
    # Memory Tracking
    # =====================================================================

    def record_memory(
        self,
        component: str,
        bytes_used: int | None = None,
        force: bool = False,
    ) -> None:
        """
        Record memory usage for a component.

        Args:
            component: Component name
            bytes_used: Bytes used (auto-detected if None)
            force: Force recording even if sampling would skip
        """
        if not self._config.enable_memory_tracking:
            return

        if not force and not self._should_sample(component):
            return

        rss_mb, vms_mb = self._get_process_memory()

        if bytes_used is None:
            # Try to get component-specific memory using tracemalloc
            bytes_used = self._get_component_memory(component)

        sample = MemorySample(
            component=component,
            bytes_used=bytes_used,
            rss_mb=rss_mb,
            vms_mb=vms_mb,
        )

        with self._lock:
            self._memory_samples[component].append(sample)

    def _get_component_memory(self, component: str) -> int:
        """Get memory usage for a specific component."""
        # This is a placeholder - component-specific memory tracking
        # would require more sophisticated allocation tracking
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

    def start_memory_tracking(self) -> None:
        """Start detailed memory tracking with tracemalloc."""
        if not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
            logger.info("Started tracemalloc for detailed memory tracking")

    def stop_memory_tracking(self) -> None:
        """Stop detailed memory tracking."""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
            logger.info("Stopped tracemalloc")

    # =====================================================================
    # Throughput Tracking
    # =====================================================================

    def _track_throughput(self, operation: str) -> None:
        """Track a request for throughput calculation."""
        now = datetime.now()
        with self._lock:
            self._throughput_counters[operation].append((now, 1))

    def record_request(self, operation: str, count: int = 1) -> None:
        """
        Record request(s) for throughput tracking.

        Args:
            operation: Operation name
            count: Number of requests (default 1)
        """
        if not self._config.enable_throughput:
            return

        now = datetime.now()
        with self._lock:
            self._throughput_counters[operation].append((now, count))

    # =====================================================================
    # Error Tracking
    # =====================================================================

    def record_error(
        self,
        component: str,
        error_type: str,
        error_message: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an error occurrence.

        Args:
            component: Component where error occurred
            error_type: Type of error (exception class name)
            error_message: Error message
            context: Additional context
        """
        with self._lock:
            stats = self._error_stats[component]
            stats.component = component
            stats.total_errors += 1
            stats.errors_by_type[error_type] = stats.errors_by_type.get(error_type, 0) + 1

            stats.recent_errors.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": error_type,
                    "message": error_message[:200],  # Limit size
                    "context": context or {},
                }
            )

    def record_call(self, component: str, success: bool = True) -> None:
        """
        Record a function call for error rate calculation.

        Args:
            component: Component name
            success: Whether the call succeeded
        """
        with self._lock:
            stats = self._error_stats[component]
            stats.component = component
            stats.total_calls += 1
            if not success:
                stats.total_errors += 1
            stats.error_rate = stats.total_errors / max(stats.total_calls, 1)

    # =====================================================================
    # Metrics Calculation
    # =====================================================================

    def _calculate_percentiles(self, values: list[float]) -> tuple[float, float, float]:
        """Calculate p50, p95, p99 percentiles."""
        if not values:
            return 0.0, 0.0, 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        def percentile(p: float) -> float:
            k = (n - 1) * p / 100.0
            f = int(k)
            c = f + 1 if f + 1 < n else f
            if f == c:
                return sorted_values[f]
            return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

        return percentile(50), percentile(95), percentile(99)

    def _calculate_std_dev(self, values: list[float], mean: float) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def get_latency_stats(self, operation: str | None = None) -> dict[str, LatencyStats]:
        """
        Get latency statistics.

        Args:
            operation: Specific operation (None for all)

        Returns:
            Dictionary of operation -> LatencyStats
        """
        with self._lock:
            if operation:
                key = operation
                samples = list(self._latency_samples.get(key, []))
                if not samples:
                    return {}

                values = [s.duration_ms for s in samples]
                p50, p95, p99 = self._calculate_percentiles(values)
                avg = sum(values) / len(values)

                return {
                    key: LatencyStats(
                        operation=samples[0].operation,
                        component=samples[0].component,
                        count=len(values),
                        total_ms=sum(values),
                        min_ms=min(values),
                        max_ms=max(values),
                        p50_ms=p50,
                        p95_ms=p95,
                        p99_ms=p99,
                        avg_ms=avg,
                        std_dev_ms=self._calculate_std_dev(values, avg),
                        samples=values[-100:] if self._config.enable_latency_percentiles else [],
                    )
                }

            # Get all operations
            result = {}
            for key, samples in self._latency_samples.items():
                if not samples:
                    continue

                values = [s.duration_ms for s in samples]
                p50, p95, p99 = self._calculate_percentiles(values)
                avg = sum(values) / len(values)

                result[key] = LatencyStats(
                    operation=samples[0].operation,
                    component=samples[0].component,
                    count=len(values),
                    total_ms=sum(values),
                    min_ms=min(values),
                    max_ms=max(values),
                    p50_ms=p50,
                    p95_ms=p95,
                    p99_ms=p99,
                    avg_ms=avg,
                    std_dev_ms=self._calculate_std_dev(values, avg),
                )

            return result

    def get_memory_stats(self, component: str | None = None) -> dict[str, MemoryStats]:
        """
        Get memory statistics.

        Args:
            component: Specific component (None for all)

        Returns:
            Dictionary of component -> MemoryStats
        """
        with self._lock:
            if component:
                samples = list(self._memory_samples.get(component, []))
                if not samples:
                    return {}

                bytes_values = [s.bytes_used for s in samples]
                return {
                    component: MemoryStats(
                        component=component,
                        current_bytes=bytes_values[-1] if bytes_values else 0,
                        peak_bytes=max(bytes_values) if bytes_values else 0,
                        avg_bytes=sum(bytes_values) / len(bytes_values) if bytes_values else 0,
                        samples_count=len(samples),
                        rss_mb=samples[-1].rss_mb if samples else 0,
                        vms_mb=samples[-1].vms_mb if samples else 0,
                    )
                }

            result = {}
            for comp, samples in self._memory_samples.items():
                if not samples:
                    continue

                bytes_values = [s.bytes_used for s in samples]
                result[comp] = MemoryStats(
                    component=comp,
                    current_bytes=bytes_values[-1] if bytes_values else 0,
                    peak_bytes=max(bytes_values) if bytes_values else 0,
                    avg_bytes=sum(bytes_values) / len(bytes_values) if bytes_values else 0,
                    samples_count=len(samples),
                    rss_mb=samples[-1].rss_mb if samples else 0,
                    vms_mb=samples[-1].vms_mb if samples else 0,
                )

            return result

    def get_throughput_stats(self, operation: str | None = None) -> dict[str, ThroughputStats]:
        """
        Get throughput statistics.

        Args:
            operation: Specific operation (None for all)

        Returns:
            Dictionary of operation -> ThroughputStats
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=60)

        with self._lock:
            if operation:
                counters = self._throughput_counters.get(operation, [])
                recent = [(ts, count) for ts, count in counters if ts > window_start]
                total = sum(count for _, count in recent)

                return {
                    operation: ThroughputStats(
                        operation=operation,
                        total_requests=sum(count for _, count in counters),
                        requests_per_sec=total / 60.0,
                        requests_per_min=total,
                        window_start=window_start,
                    )
                }

            result = {}
            for op, counters in self._throughput_counters.items():
                recent = [(ts, count) for ts, count in counters if ts > window_start]
                total = sum(count for _, count in recent)

                result[op] = ThroughputStats(
                    operation=op,
                    total_requests=sum(count for _, count in counters),
                    requests_per_sec=total / 60.0,
                    requests_per_min=total,
                    window_start=window_start,
                )

            return result

    def get_error_stats(self, component: str | None = None) -> dict[str, ErrorStats]:
        """
        Get error statistics.

        Args:
            component: Specific component (None for all)

        Returns:
            Dictionary of component -> ErrorStats
        """
        with self._lock:
            if component:
                stats = self._error_stats.get(component)
                if stats:
                    return {component: stats}
                return {}

            return dict(self._error_stats)

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get complete performance metrics snapshot.

        Returns:
            PerformanceMetrics with all current statistics
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            system_info = {
                "process_memory_rss_mb": memory_info.rss / (1024 * 1024),
                "process_memory_vms_mb": memory_info.vms / (1024 * 1024),
                "cpu_percent": cpu_percent,
                "num_threads": process.num_threads(),
            }
        except ImportError:
            system_info = {}

        return PerformanceMetrics(
            timestamp=datetime.now(),
            latency=self.get_latency_stats(),
            memory=self.get_memory_stats(),
            throughput=self.get_throughput_stats(),
            errors=self.get_error_stats(),
            system=system_info,
        )

    # =====================================================================
    # Export Functions
    # =====================================================================

    def export_json(self) -> str:
        """Export metrics as JSON string."""
        import json

        metrics = self.get_metrics()
        return json.dumps(metrics.to_dict(), indent=2)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.get_metrics()

        # Latency metrics
        for key, stats in metrics.latency.items():
            prefix = f'gaap_latency{{operation="{stats.operation}",component="{stats.component}"}}'
            lines.append(f"{prefix}_p50 {stats.p50_ms}")
            lines.append(f"{prefix}_p95 {stats.p95_ms}")
            lines.append(f"{prefix}_p99 {stats.p99_ms}")
            lines.append(f"{prefix}_avg {stats.avg_ms}")
            lines.append(f"{prefix}_count {stats.count}")

        # Memory metrics
        for comp, stats in metrics.memory.items():
            prefix = f'gaap_memory{{component="{comp}"}}'
            lines.append(f"{prefix}_bytes {stats.current_bytes}")
            lines.append(f"{prefix}_peak_bytes {stats.peak_bytes}")

        # Throughput metrics
        for op, stats in metrics.throughput.items():
            prefix = f'gaap_throughput{{operation="{op}"}}'
            lines.append(f"{prefix}_rps {stats.requests_per_sec}")
            lines.append(f"{prefix}_rpm {stats.requests_per_min}")
            lines.append(f"{prefix}_total {stats.total_requests}")

        # Error metrics
        for comp, stats in metrics.errors.items():
            prefix = f'gaap_errors{{component="{comp}"}}'
            lines.append(f"{prefix}_total {stats.total_errors}")
            lines.append(f"{prefix}_rate {stats.error_rate}")

        return "\n".join(lines)

    # =====================================================================
    # Configuration Management
    # =====================================================================

    def configure(self, config: PerformanceConfig) -> None:
        """Update configuration."""
        with self._lock:
            self._config = config
            self._current_sampling_rate = config.sampling_rate

    def enable(self) -> None:
        """Enable monitoring."""
        self._config.enabled = True

    def disable(self) -> None:
        """Disable monitoring."""
        self._config.enabled = False

    def set_sampling_rate(self, rate: float) -> None:
        """Set sampling rate (0.0-1.0)."""
        self._current_sampling_rate = max(0.0, min(1.0, rate))

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._latency_samples.clear()
            self._memory_samples.clear()
            self._throughput_counters.clear()
            self._error_stats.clear()
            logger.info("PerformanceMonitor metrics reset")


# Import here to avoid circular dependency
import asyncio

# =============================================================================
# Global Instance
# =============================================================================


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global PerformanceMonitor singleton instance."""
    return PerformanceMonitor()


# Convenience functions for direct use
def timing(operation: str, component: str = "", tags: dict[str, str] | None = None):
    """Context manager shorthand for get_performance_monitor().timing()"""
    return get_performance_monitor().timing(operation, component, tags)


def timed(operation: str | None = None, component: str = "", tags: dict[str, str] | None = None):
    """Decorator shorthand for get_performance_monitor().timed()"""
    return get_performance_monitor().timed(operation, component, tags)


def record_memory(component: str, bytes_used: int | None = None) -> None:
    """Shorthand for get_performance_monitor().record_memory()"""
    return get_performance_monitor().record_memory(component, bytes_used)


def record_error(
    component: str,
    error_type: str,
    error_message: str = "",
    context: dict[str, Any] | None = None,
) -> None:
    """Shorthand for get_performance_monitor().record_error()"""
    return get_performance_monitor().record_error(component, error_type, error_message, context)


def get_metrics() -> PerformanceMetrics:
    """Shorthand for get_performance_monitor().get_metrics()"""
    return get_performance_monitor().get_metrics()
