"""
GAAP Metrics Collectors
=======================

Comprehensive metrics collectors for the GAAP system.

Provides:
- Request latency histograms
- Request counters
- Active connection gauges
- Error rate tracking
- Provider-specific metrics
- Token usage tracking
- Cost tracking

Usage:
    from gaap.metrics.collectors import (
        RequestMetrics,
        ProviderMetrics,
        SystemMetrics,
        BusinessMetrics,
    )

    # Request metrics
    request_metrics = RequestMetrics()
    request_metrics.record_request(
        method="POST",
        endpoint="/api/chat",
        status=200,
        duration=0.5,
    )

    # Provider metrics
    provider_metrics = ProviderMetrics()
    provider_metrics.record_call(
        provider="kimi",
        model="k2.5",
        latency=1.2,
        success=True,
    )
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar

from gaap.metrics import (
    PROMETHEUS_AVAILABLE,
    get_registry,
)

logger = logging.getLogger("gaap.metrics.collectors")

if PROMETHEUS_AVAILABLE:
    from prometheus_client import Counter, Gauge, Histogram, Summary

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Base Collector
# =============================================================================


class BaseCollector:
    """Base class for all metrics collectors."""

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "api",
        registry: Optional[Any] = None,
    ):
        self.namespace = namespace
        self.subsystem = subsystem
        self.registry = registry or get_registry()
        self._metrics: dict[str, Any] = {}
        self._initialized = False

    def _metric_name(self, name: str) -> str:
        """Get fully qualified metric name."""
        return f"{self.namespace}_{self.subsystem}_{name}"

    def _create_counter(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ) -> Optional[Any]:
        """Create a counter metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        try:
            counter = Counter(
                name=self._metric_name(name),
                documentation=description,
                labelnames=labels or [],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry,
            )
            self._metrics[name] = counter
            return counter
        except Exception as e:
            logger.warning(f"Failed to create counter {name}: {e}")
            return None

    def _create_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ) -> Optional[Any]:
        """Create a gauge metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        try:
            gauge = Gauge(
                name=self._metric_name(name),
                documentation=description,
                labelnames=labels or [],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry,
            )
            self._metrics[name] = gauge
            return gauge
        except Exception as e:
            logger.warning(f"Failed to create gauge {name}: {e}")
            return None

    def _create_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
        buckets: Optional[list[float]] = None,
    ) -> Optional[Any]:
        """Create a histogram metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        try:
            histogram = Histogram(
                name=self._metric_name(name),
                documentation=description,
                labelnames=labels or [],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry,
                buckets=buckets or Histogram.DEFAULT_BUCKETS,
            )
            self._metrics[name] = histogram
            return histogram
        except Exception as e:
            logger.warning(f"Failed to create histogram {name}: {e}")
            return None

    def _create_summary(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ) -> Optional[Any]:
        """Create a summary metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        try:
            summary = Summary(
                name=self._metric_name(name),
                documentation=description,
                labelnames=labels or [],
                namespace=self.namespace,
                subsystem=self.subsystem,
                registry=self.registry,
            )
            self._metrics[name] = summary
            return summary
        except Exception as e:
            logger.warning(f"Failed to create summary {name}: {e}")
            return None


# =============================================================================
# Request Metrics
# =============================================================================


class RequestMetrics(BaseCollector):
    """
    HTTP request metrics collector.

    Tracks:
    - Request count by method, endpoint, status
    - Request latency distribution
    - Request/response sizes
    - Active requests
    """

    # Standard latency buckets for API requests (in seconds)
    LATENCY_BUCKETS = [
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
    ]

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "api",
        registry: Optional[Any] = None,
    ):
        super().__init__(namespace, subsystem, registry)
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize request metrics."""
        # Request counter
        self._counter = self._create_counter(
            "requests_total",
            "Total number of HTTP requests",
            ["method", "endpoint", "status_code"],
        )

        # Request latency histogram
        self._latency = self._create_histogram(
            "request_duration_seconds",
            "HTTP request latency in seconds",
            ["method", "endpoint"],
            buckets=self.LATENCY_BUCKETS,
        )

        # Request size histogram
        self._request_size = self._create_histogram(
            "request_size_bytes",
            "HTTP request size in bytes",
            ["method", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
        )

        # Response size histogram
        self._response_size = self._create_histogram(
            "response_size_bytes",
            "HTTP response size in bytes",
            ["method", "endpoint", "status_code"],
            buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
        )

        # Active requests gauge
        self._active = self._create_gauge(
            "active_requests",
            "Number of requests currently being processed",
            ["method", "endpoint"],
        )

        self._initialized = True

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
    ) -> None:
        """
        Record an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: Request endpoint path
            status_code: HTTP status code
            duration: Request duration in seconds
            request_size: Request body size in bytes
            response_size: Response body size in bytes
        """
        if not PROMETHEUS_AVAILABLE:
            return

        method = method.upper()
        endpoint = self._normalize_endpoint(endpoint)
        status = str(status_code)

        if self._counter:
            self._counter.labels(method=method, endpoint=endpoint, status_code=status).inc()

        if self._latency:
            self._latency.labels(method=method, endpoint=endpoint).observe(duration)

        if self._request_size and request_size is not None:
            self._request_size.labels(method=method, endpoint=endpoint).observe(request_size)

        if self._response_size and response_size is not None:
            self._response_size.labels(
                method=method, endpoint=endpoint, status_code=status
            ).observe(response_size)

    def inc_active(self, method: str, endpoint: str) -> None:
        """Increment active request count."""
        if self._active and PROMETHEUS_AVAILABLE:
            self._active.labels(
                method=method.upper(), endpoint=self._normalize_endpoint(endpoint)
            ).inc()

    def dec_active(self, method: str, endpoint: str) -> None:
        """Decrement active request count."""
        if self._active and PROMETHEUS_AVAILABLE:
            self._active.labels(
                method=method.upper(), endpoint=self._normalize_endpoint(endpoint)
            ).dec()

    @contextmanager
    def track_request(
        self,
        method: str,
        endpoint: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Context manager to track a request.

        Usage:
            with request_metrics.track_request("POST", "/api/chat") as tracker:
                response = await handle_request()
                tracker["status_code"] = response.status
                tracker["response_size"] = len(response.body)
        """
        start_time = time.time()
        tracker = {"status_code": 200, "request_size": None, "response_size": None}

        self.inc_active(method, endpoint)

        try:
            yield tracker
        finally:
            duration = time.time() - start_time
            self.dec_active(method, endpoint)
            self.record_request(
                method=method,
                endpoint=endpoint,
                status_code=tracker["status_code"],
                duration=duration,
                request_size=tracker.get("request_size"),
                response_size=tracker.get("response_size"),
            )

    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint for consistent labeling."""
        # Remove query parameters
        endpoint = endpoint.split("?")[0]
        # Remove trailing slash
        endpoint = endpoint.rstrip("/")
        # Normalize IDs in path
        parts = endpoint.split("/")
        normalized = []
        for part in parts:
            # Replace UUIDs and numeric IDs with placeholders
            if part and (len(part) == 32 or part.replace("-", "").isalnum() and len(part) > 20):
                normalized.append(":id")
            elif part.isdigit():
                normalized.append(":id")
            else:
                normalized.append(part)
        return "/".join(normalized) or "/"


# =============================================================================
# Connection Metrics
# =============================================================================


class ConnectionMetrics(BaseCollector):
    """
    Connection metrics collector.

    Tracks:
    - Active connections
    - Connection duration
    - Connection errors
    - WebSocket connections
    """

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "api",
        registry: Optional[Any] = None,
    ):
        super().__init__(namespace, subsystem, registry)
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize connection metrics."""
        # Active connections gauge
        self._active = self._create_gauge(
            "active_connections",
            "Number of active connections",
            ["type", "channel"],
        )

        # Connection duration histogram
        self._duration = self._create_histogram(
            "connection_duration_seconds",
            "Connection duration in seconds",
            ["type", "channel"],
            buckets=[1.0, 5.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
        )

        # Connection errors counter
        self._errors = self._create_counter(
            "connection_errors_total",
            "Total number of connection errors",
            ["type", "error_type"],
        )

        # Total connections counter
        self._total = self._create_counter(
            "connections_total",
            "Total number of connections",
            ["type", "channel"],
        )

        self._initialized = True

    def inc_active(self, conn_type: str, channel: str = "default") -> None:
        """Increment active connections."""
        if self._active and PROMETHEUS_AVAILABLE:
            self._active.labels(type=conn_type, channel=channel).inc()

    def dec_active(self, conn_type: str, channel: str = "default") -> None:
        """Decrement active connections."""
        if self._active and PROMETHEUS_AVAILABLE:
            self._active.labels(type=conn_type, channel=channel).dec()

    def record_connection(
        self,
        conn_type: str,
        channel: str = "default",
        duration: Optional[float] = None,
    ) -> None:
        """Record a connection."""
        if self._total and PROMETHEUS_AVAILABLE:
            self._total.labels(type=conn_type, channel=channel).inc()

        if self._duration and duration is not None and PROMETHEUS_AVAILABLE:
            self._duration.labels(type=conn_type, channel=channel).observe(duration)

    def record_error(self, conn_type: str, error_type: str) -> None:
        """Record a connection error."""
        if self._errors and PROMETHEUS_AVAILABLE:
            self._errors.labels(type=conn_type, error_type=error_type).inc()

    @contextmanager
    def track_connection(
        self,
        conn_type: str,
        channel: str = "default",
    ) -> Generator[None, None, None]:
        """Context manager to track a connection."""
        start_time = time.time()
        self.inc_active(conn_type, channel)

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.dec_active(conn_type, channel)
            self.record_connection(conn_type, channel, duration)


# =============================================================================
# Error Metrics
# =============================================================================


class ErrorMetrics(BaseCollector):
    """
    Error metrics collector.

    Tracks:
    - Error counts by type, layer, severity
    - Error rates
    - Exception types
    """

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "api",
        registry: Optional[Any] = None,
    ):
        super().__init__(namespace, subsystem, registry)
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize error metrics."""
        # Error counter
        self._counter = self._create_counter(
            "errors_total",
            "Total number of errors",
            ["layer", "error_type", "severity"],
        )

        # Error rate gauge (calculated)
        self._rate = self._create_gauge(
            "error_rate",
            "Error rate (errors per second)",
            ["layer"],
        )

        # Exception counter
        self._exceptions = self._create_counter(
            "exceptions_total",
            "Total number of exceptions",
            ["exception_type", "module"],
        )

        # HTTP error counter
        self._http_errors = self._create_counter(
            "http_errors_total",
            "Total number of HTTP errors",
            ["status_code", "endpoint"],
        )

        self._initialized = True
        self._error_counts: dict[str, list[float]] = defaultdict(list)

    def record_error(
        self,
        error_type: str,
        layer: str = "unknown",
        severity: str = "error",
    ) -> None:
        """
        Record an error.

        Args:
            error_type: Type of error
            layer: System layer where error occurred
            severity: Error severity (error, warning, critical)
        """
        if self._counter and PROMETHEUS_AVAILABLE:
            self._counter.labels(layer=layer, error_type=error_type, severity=severity).inc()

        # Track for rate calculation
        self._error_counts[layer].append(time.time())

    def record_exception(
        self,
        exception: Exception,
        module: str = "unknown",
    ) -> None:
        """Record an exception."""
        if self._exceptions and PROMETHEUS_AVAILABLE:
            exc_type = type(exception).__name__
            self._exceptions.labels(exception_type=exc_type, module=module).inc()

    def record_http_error(self, status_code: int, endpoint: str) -> None:
        """Record an HTTP error."""
        if self._http_errors and PROMETHEUS_AVAILABLE:
            self._http_errors.labels(status_code=str(status_code), endpoint=endpoint).inc()

    def update_error_rate(self, layer: str, window_seconds: float = 60.0) -> None:
        """Update error rate for a layer."""
        if not self._rate or not PROMETHEUS_AVAILABLE:
            return

        now = time.time()
        cutoff = now - window_seconds

        # Remove old errors
        self._error_counts[layer] = [t for t in self._error_counts[layer] if t > cutoff]

        # Calculate rate
        count = len(self._error_counts[layer])
        rate = count / window_seconds if window_seconds > 0 else 0

        self._rate.labels(layer=layer).set(rate)


# =============================================================================
# Provider Metrics
# =============================================================================


class ProviderMetrics(BaseCollector):
    """
    LLM provider metrics collector.

    Tracks:
    - Provider latency
    - Provider request counts
    - Provider error rates
    - Model usage distribution
    """

    # Latency buckets for LLM calls (typically slower than HTTP)
    LATENCY_BUCKETS = [
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        15.0,
        30.0,
        45.0,
        60.0,
        120.0,
    ]

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "providers",
        registry: Optional[Any] = None,
    ):
        super().__init__(namespace, subsystem, registry)
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize provider metrics."""
        # Provider latency histogram
        self._latency = self._create_histogram(
            "latency_seconds",
            "Provider API call latency in seconds",
            ["provider", "model", "operation"],
            buckets=self.LATENCY_BUCKETS,
        )

        # Provider request counter
        self._requests = self._create_counter(
            "requests_total",
            "Total number of provider requests",
            ["provider", "model", "status"],
        )

        # Provider error counter
        self._errors = self._create_counter(
            "errors_total",
            "Total number of provider errors",
            ["provider", "error_type"],
        )

        # Active requests gauge
        self._active = self._create_gauge(
            "active_requests",
            "Number of active provider requests",
            ["provider"],
        )

        # Time to first token histogram
        self._ttft = self._create_histogram(
            "time_to_first_token_seconds",
            "Time to first token in streaming responses",
            ["provider", "model"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        )

        # Tokens per second histogram
        self._tps = self._create_histogram(
            "tokens_per_second",
            "Token generation rate",
            ["provider", "model"],
            buckets=[1, 5, 10, 20, 50, 100, 200],
        )

        self._initialized = True

    def record_call(
        self,
        provider: str,
        model: str,
        latency: float,
        success: bool,
        operation: str = "completion",
    ) -> None:
        """
        Record a provider API call.

        Args:
            provider: Provider name
            model: Model name
            latency: Call latency in seconds
            success: Whether the call succeeded
            operation: Operation type
        """
        if not PROMETHEUS_AVAILABLE:
            return

        status = "success" if success else "failure"

        if self._latency:
            self._latency.labels(provider=provider, model=model, operation=operation).observe(
                latency
            )

        if self._requests:
            self._requests.labels(provider=provider, model=model, status=status).inc()

    def record_error(
        self,
        provider: str,
        error_type: str,
    ) -> None:
        """Record a provider error."""
        if self._errors and PROMETHEUS_AVAILABLE:
            self._errors.labels(provider=provider, error_type=error_type).inc()

    def inc_active(self, provider: str) -> None:
        """Increment active requests for a provider."""
        if self._active and PROMETHEUS_AVAILABLE:
            self._active.labels(provider=provider).inc()

    def dec_active(self, provider: str) -> None:
        """Decrement active requests for a provider."""
        if self._active and PROMETHEUS_AVAILABLE:
            self._active.labels(provider=provider).dec()

    def record_ttft(
        self,
        provider: str,
        model: str,
        ttft: float,
    ) -> None:
        """Record time to first token."""
        if self._ttft and PROMETHEUS_AVAILABLE:
            self._ttft.labels(provider=provider, model=model).observe(ttft)

    def record_tps(
        self,
        provider: str,
        model: str,
        tps: float,
    ) -> None:
        """Record tokens per second."""
        if self._tps and PROMETHEUS_AVAILABLE:
            self._tps.labels(provider=provider, model=model).observe(tps)

    @contextmanager
    def track_call(
        self,
        provider: str,
        model: str,
        operation: str = "completion",
    ) -> Generator[dict[str, Any], None, None]:
        """
        Context manager to track a provider call.

        Usage:
            with provider_metrics.track_call("kimi", "k2.5") as tracker:
                result = await provider.complete(...)
                tracker["success"] = result.success
        """
        start_time = time.time()
        tracker = {"success": True}

        self.inc_active(provider)

        try:
            yield tracker
        finally:
            duration = time.time() - start_time
            self.dec_active(provider)
            self.record_call(
                provider=provider,
                model=model,
                latency=duration,
                success=tracker.get("success", True),
                operation=operation,
            )


# =============================================================================
# Token & Cost Metrics
# =============================================================================


class CostMetrics(BaseCollector):
    """
    Token usage and cost metrics collector.

    Tracks:
    - Token usage by type and model
    - Cost accumulation
    - Cost per user/session
    """

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "cost",
        registry: Optional[Any] = None,
    ):
        super().__init__(namespace, subsystem, registry)
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize cost metrics."""
        # Token counter
        self._tokens = self._create_counter(
            "tokens_total",
            "Total number of tokens processed",
            ["provider", "model", "type"],  # type: input, output, total
        )

        # Cost counter (in USD)
        self._cost = self._create_counter(
            "dollars_total",
            "Total cost in USD",
            ["provider", "model", "category"],
        )

        # Cost gauge (current period)
        self._current_cost = self._create_gauge(
            "current_period_dollars",
            "Cost in current billing period",
            ["category"],
        )

        # Token usage histogram
        self._token_histogram = self._create_histogram(
            "tokens_per_request",
            "Tokens per request",
            ["provider", "model", "type"],
            buckets=[10, 50, 100, 500, 1000, 2000, 4000, 8000, 16000, 32000],
        )

        # Cost per request histogram
        self._cost_histogram = self._create_histogram(
            "dollars_per_request",
            "Cost per request in USD",
            ["provider", "model"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        )

        self._initialized = True

    def record_tokens(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """
        Record token usage.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        if not PROMETHEUS_AVAILABLE:
            return

        if self._tokens:
            self._tokens.labels(provider=provider, model=model, type="input").inc(input_tokens)
            self._tokens.labels(provider=provider, model=model, type="output").inc(output_tokens)

        if self._token_histogram:
            self._token_histogram.labels(provider=provider, model=model, type="input").observe(
                input_tokens
            )
            self._token_histogram.labels(provider=provider, model=model, type="output").observe(
                output_tokens
            )

    def record_cost(
        self,
        provider: str,
        model: str,
        cost_usd: float,
        category: str = "llm",
    ) -> None:
        """
        Record cost.

        Args:
            provider: Provider name
            model: Model name
            cost_usd: Cost in USD
            category: Cost category
        """
        if not PROMETHEUS_AVAILABLE:
            return

        if self._cost:
            self._cost.labels(provider=provider, model=model, category=category).inc(cost_usd)

        if self._cost_histogram:
            self._cost_histogram.labels(provider=provider, model=model).observe(cost_usd)

    def set_current_cost(self, cost_usd: float, category: str = "llm") -> None:
        """Set current period cost."""
        if self._current_cost and PROMETHEUS_AVAILABLE:
            self._current_cost.labels(category=category).set(cost_usd)

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        category: str = "llm",
    ) -> None:
        """Record complete usage in one call."""
        self.record_tokens(provider, model, input_tokens, output_tokens)
        self.record_cost(provider, model, cost_usd, category)


# =============================================================================
# System Metrics
# =============================================================================


class SystemMetrics(BaseCollector):
    """
    System resource metrics collector.

    Tracks:
    - Memory usage
    - CPU usage
    - Disk usage
    - Network I/O
    """

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "system",
        registry: Optional[Any] = None,
    ):
        super().__init__(namespace, subsystem, registry)
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize system metrics."""
        # Memory usage gauge
        self._memory = self._create_gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],  # type: used, free, cached, available
        )

        # Memory percent gauge
        self._memory_percent = self._create_gauge(
            "memory_usage_percent",
            "Memory usage percentage",
            [],
        )

        # CPU usage gauge
        self._cpu = self._create_gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            ["mode"],  # mode: user, system, idle
        )

        # Disk usage gauge
        self._disk = self._create_gauge(
            "disk_usage_bytes",
            "Disk usage in bytes",
            ["mount", "type"],  # type: used, free, total
        )

        # Disk percent gauge
        self._disk_percent = self._create_gauge(
            "disk_usage_percent",
            "Disk usage percentage",
            ["mount"],
        )

        # Network I/O counter
        self._network = self._create_counter(
            "network_io_bytes_total",
            "Network I/O in bytes",
            ["direction", "interface"],  # direction: receive, transmit
        )

        # Open file descriptors gauge
        self._fd = self._create_gauge(
            "open_file_descriptors",
            "Number of open file descriptors",
            [],
        )

        # Process metrics
        self._process_cpu = self._create_gauge(
            "process_cpu_seconds_total",
            "Process CPU time in seconds",
            ["mode"],
        )

        self._process_memory = self._create_gauge(
            "process_memory_bytes",
            "Process memory usage in bytes",
            ["type"],
        )

        self._initialized = True

    def record_memory(
        self,
        used_bytes: int,
        free_bytes: int,
        available_bytes: int,
        cached_bytes: int = 0,
    ) -> None:
        """Record memory usage."""
        if not PROMETHEUS_AVAILABLE:
            return

        if self._memory:
            self._memory.labels(type="used").set(used_bytes)
            self._memory.labels(type="free").set(free_bytes)
            self._memory.labels(type="available").set(available_bytes)
            self._memory.labels(type="cached").set(cached_bytes)

        if self._memory_percent:
            total = used_bytes + available_bytes
            percent = (used_bytes / total * 100) if total > 0 else 0
            self._memory_percent.set(percent)

    def record_cpu(self, user_percent: float, system_percent: float, idle_percent: float) -> None:
        """Record CPU usage."""
        if self._cpu and PROMETHEUS_AVAILABLE:
            self._cpu.labels(mode="user").set(user_percent)
            self._cpu.labels(mode="system").set(system_percent)
            self._cpu.labels(mode="idle").set(idle_percent)

    def record_disk(
        self,
        mount: str,
        used_bytes: int,
        free_bytes: int,
        total_bytes: int,
    ) -> None:
        """Record disk usage."""
        if not PROMETHEUS_AVAILABLE:
            return

        if self._disk:
            self._disk.labels(mount=mount, type="used").set(used_bytes)
            self._disk.labels(mount=mount, type="free").set(free_bytes)
            self._disk.labels(mount=mount, type="total").set(total_bytes)

        if self._disk_percent:
            percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
            self._disk_percent.labels(mount=mount).set(percent)

    def record_network(
        self,
        direction: str,
        interface: str,
        bytes_count: int,
    ) -> None:
        """Record network I/O."""
        if self._network and PROMETHEUS_AVAILABLE:
            # Use inc for counters - but need to track delta
            # For simplicity, we just set a gauge here in practice
            pass

    def record_open_fd(self, count: int) -> None:
        """Record open file descriptor count."""
        if self._fd and PROMETHEUS_AVAILABLE:
            self._fd.set(count)

    def record_process_memory(self, rss_bytes: int, vms_bytes: int) -> None:
        """Record process memory."""
        if self._process_memory and PROMETHEUS_AVAILABLE:
            self._process_memory.labels(type="rss").set(rss_bytes)
            self._process_memory.labels(type="vms").set(vms_bytes)


# =============================================================================
# Business Metrics
# =============================================================================


class BusinessMetrics(BaseCollector):
    """
    Business-level metrics collector.

    Tracks:
    - Chat sessions
    - Messages per session
    - User activity
    - Feature usage
    """

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "business",
        registry: Optional[Any] = None,
    ):
        super().__init__(namespace, subsystem, registry)
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize business metrics."""
        # Chat sessions counter
        self._sessions = self._create_counter(
            "chat_sessions_total",
            "Total number of chat sessions",
            ["source", "status"],
        )

        # Active sessions gauge
        self._active_sessions = self._create_gauge(
            "chat_sessions_active",
            "Number of active chat sessions",
            [],
        )

        # Messages counter
        self._messages = self._create_counter(
            "messages_total",
            "Total number of messages",
            ["type", "source"],  # type: user, assistant, system
        )

        # Messages per session histogram
        self._mps = self._create_histogram(
            "messages_per_session",
            "Messages per session",
            [],
            buckets=[1, 2, 5, 10, 20, 50, 100, 200],
        )

        # Session duration histogram
        self._session_duration = self._create_histogram(
            "session_duration_seconds",
            "Chat session duration",
            [],
            buckets=[30, 60, 300, 600, 1800, 3600, 7200],
        )

        # Active users gauge
        self._active_users = self._create_gauge(
            "users_active",
            "Number of active users",
            ["period"],  # period: 1m, 5m, 1h, 1d
        )

        # Feature usage counter
        self._features = self._create_counter(
            "feature_usage_total",
            "Feature usage count",
            ["feature"],
        )

        self._initialized = True

    def record_session_start(self, source: str = "api") -> None:
        """Record a new session start."""
        if self._sessions and PROMETHEUS_AVAILABLE:
            self._sessions.labels(source=source, status="started").inc()

    def record_session_end(
        self,
        source: str = "api",
        duration_seconds: Optional[float] = None,
        message_count: Optional[int] = None,
    ) -> None:
        """Record a session end."""
        if not PROMETHEUS_AVAILABLE:
            return

        if self._sessions:
            self._sessions.labels(source=source, status="completed").inc()

        if self._session_duration and duration_seconds is not None:
            self._session_duration.observe(duration_seconds)

        if self._mps and message_count is not None:
            self._mps.observe(message_count)

    def set_active_sessions(self, count: int) -> None:
        """Set active session count."""
        if self._active_sessions and PROMETHEUS_AVAILABLE:
            self._active_sessions.set(count)

    def record_message(self, msg_type: str, source: str = "api") -> None:
        """Record a message."""
        if self._messages and PROMETHEUS_AVAILABLE:
            self._messages.labels(type=msg_type, source=source).inc()

    def set_active_users(self, count: int, period: str = "5m") -> None:
        """Set active user count."""
        if self._active_users and PROMETHEUS_AVAILABLE:
            self._active_users.labels(period=period).set(count)

    def record_feature_usage(self, feature: str) -> None:
        """Record feature usage."""
        if self._features and PROMETHEUS_AVAILABLE:
            self._features.labels(feature=feature).inc()


# =============================================================================
# Collector Instances
# =============================================================================

# Global collector instances
_request_metrics: Optional[RequestMetrics] = None
_connection_metrics: Optional[ConnectionMetrics] = None
_error_metrics: Optional[ErrorMetrics] = None
_provider_metrics: Optional[ProviderMetrics] = None
_cost_metrics: Optional[CostMetrics] = None
_system_metrics: Optional[SystemMetrics] = None
_business_metrics: Optional[BusinessMetrics] = None


def get_request_metrics() -> RequestMetrics:
    """Get the global request metrics collector."""
    global _request_metrics
    if _request_metrics is None:
        _request_metrics = RequestMetrics()
    return _request_metrics


def get_connection_metrics() -> ConnectionMetrics:
    """Get the global connection metrics collector."""
    global _connection_metrics
    if _connection_metrics is None:
        _connection_metrics = ConnectionMetrics()
    return _connection_metrics


def get_error_metrics() -> ErrorMetrics:
    """Get the global error metrics collector."""
    global _error_metrics
    if _error_metrics is None:
        _error_metrics = ErrorMetrics()
    return _error_metrics


def get_provider_metrics() -> ProviderMetrics:
    """Get the global provider metrics collector."""
    global _provider_metrics
    if _provider_metrics is None:
        _provider_metrics = ProviderMetrics()
    return _provider_metrics


def get_cost_metrics() -> CostMetrics:
    """Get the global cost metrics collector."""
    global _cost_metrics
    if _cost_metrics is None:
        _cost_metrics = CostMetrics()
    return _cost_metrics


def get_system_metrics() -> SystemMetrics:
    """Get the global system metrics collector."""
    global _system_metrics
    if _system_metrics is None:
        _system_metrics = SystemMetrics()
    return _system_metrics


def get_business_metrics() -> BusinessMetrics:
    """Get the global business metrics collector."""
    global _business_metrics
    if _business_metrics is None:
        _business_metrics = BusinessMetrics()
    return _business_metrics


# Initialize all collectors
def initialize_all_collectors() -> dict[str, BaseCollector]:
    """Initialize and return all collectors."""
    return {
        "request": get_request_metrics(),
        "connection": get_connection_metrics(),
        "error": get_error_metrics(),
        "provider": get_provider_metrics(),
        "cost": get_cost_metrics(),
        "system": get_system_metrics(),
        "business": get_business_metrics(),
    }


__all__ = [
    "BaseCollector",
    "RequestMetrics",
    "ConnectionMetrics",
    "ErrorMetrics",
    "ProviderMetrics",
    "CostMetrics",
    "SystemMetrics",
    "BusinessMetrics",
    "get_request_metrics",
    "get_connection_metrics",
    "get_error_metrics",
    "get_provider_metrics",
    "get_cost_metrics",
    "get_system_metrics",
    "get_business_metrics",
    "initialize_all_collectors",
]
