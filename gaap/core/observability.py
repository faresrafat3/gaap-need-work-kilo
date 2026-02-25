"""
Observability Module - OpenTelemetry Tracing & Prometheus Metrics

Provides comprehensive observability for the GAAP system:
- Distributed tracing with OpenTelemetry
- Metrics collection with Prometheus
- Automatic instrumentation helpers
"""

import asyncio
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, ContextManager, Optional, Protocol, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# Type Stubs for Optional Dependencies
# =============================================================================

if TYPE_CHECKING:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Span, Status, StatusCode
    from prometheus_client import Counter, Gauge, Histogram, Info
else:

    class Span(Protocol):
        def set_attribute(self, key: str, value: Any) -> None: ...
        def record_exception(self, exception: Exception) -> None: ...
        def set_status(self, status: Any) -> None: ...
        def add_event(self, name: str, attributes: dict | None = None) -> None: ...

    class Status(Protocol):
        def __init__(self, status_code: Any, description: str = ""): ...

    class StatusCode(Protocol):
        ERROR: Any
        OK: Any

    class Resource(Protocol):
        @staticmethod
        def create(attrs: dict) -> "Resource": ...

    class TracerProvider(Protocol):
        def add_span_processor(self, processor: Any) -> None: ...

    class BatchSpanProcessor(Protocol):
        def __init__(self, exporter: Any): ...

    class ConsoleSpanExporter(Protocol):
        pass

    SERVICE_NAME: str = ""

    class Counter(Protocol):
        def labels(self, **kwargs: str) -> "Counter": ...
        def inc(self, value: float = 1) -> None: ...

    class Histogram(Protocol):
        def labels(self, **kwargs: str) -> "Histogram": ...
        def observe(self, value: float) -> None: ...

    class Gauge(Protocol):
        def labels(self, **kwargs: str) -> "Gauge": ...
        def set(self, value: float) -> None: ...
        def inc(self, value: float = 1) -> None: ...
        def dec(self, value: float = 1) -> None: ...

    class Info(Protocol):
        def labels(self, **kwargs: str) -> "Info": ...
        def set(self, value: dict) -> None: ...

# =============================================================================
# Optional Imports
# =============================================================================

try:
    from opentelemetry import trace as _trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Span, Status, StatusCode

    OTEL_AVAILABLE = True
    trace = _trace
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

try:
    from prometheus_client import Counter, Gauge, Histogram, Info

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class TracingConfig:
    """Configuration for OpenTelemetry tracing"""

    def __init__(
        self,
        service_name: str = "gaap-sovereign",
        service_version: str = "2.1.0-SOVEREIGN",
        environment: str = "production",
        enable_console_export: bool = False,
        enable_otlp_export: bool = False,
        otlp_endpoint: str | None = None,
        sample_rate: float = 1.0,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.enable_console_export = enable_console_export
        self.enable_otlp_export = enable_otlp_export
        self.otlp_endpoint = otlp_endpoint
        self.sample_rate = sample_rate


class MetricsConfig:
    """Configuration for Prometheus metrics"""

    def __init__(
        self,
        enable_default_metrics: bool = True,
        metrics_port: int = 9090,
        metrics_path: str = "/metrics",
        namespace: str = "gaap",
        subsystem: str = "system",
    ):
        self.enable_default_metrics = enable_default_metrics
        self.metrics_port = metrics_port
        self.metrics_path = metrics_path
        self.namespace = namespace
        self.subsystem = subsystem


class Tracer:
    """
    OpenTelemetry Tracer Wrapper

    Provides easy-to-use tracing capabilities:
    - Automatic span creation
    - Exception tracking
    - Context propagation
    """

    _instance: Optional["Tracer"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "Tracer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: TracingConfig | None = None) -> None:
        if self._initialized:
            return

        self.config = config or TracingConfig()
        self._tracer: Any = None
        self._provider: Any = None

        if OTEL_AVAILABLE:
            self._setup_tracer()

        self._initialized = True

    def _setup_tracer(self) -> None:
        """Initialize OpenTelemetry tracer"""
        if not OTEL_AVAILABLE:
            return
        resource = Resource.create(
            {
                SERVICE_NAME: self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
            }
        )

        self._provider = TracerProvider(resource=resource)

        if self.config.enable_console_export:
            console_exporter = ConsoleSpanExporter()
            self._provider.add_span_processor(BatchSpanProcessor(console_exporter))

        if self.config.enable_otlp_export and self.config.otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
                self._provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except ImportError:
                pass

        if trace is not None:
            trace.set_tracer_provider(self._provider)
            self._tracer = trace.get_tracer(self.config.service_name, self.config.service_version)

    @property
    def tracer(self) -> Any:
        """Get the underlying tracer"""
        return self._tracer

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: Any = None,
    ) -> ContextManager[Any]:
        """Start a new span"""
        if not OTEL_AVAILABLE or self._tracer is None:

            @contextmanager
            def noop_span() -> Any:
                yield None

            return noop_span()

        return self._tracer.start_as_current_span(name, attributes=attributes, kind=kind)  # type: ignore[no-any-return]

    def record_exception(self, span: Any, exception: Exception) -> None:
        """Record an exception in a span"""
        if span is not None and OTEL_AVAILABLE:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))

    def add_event(self, span: Any, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to a span"""
        if span is not None and OTEL_AVAILABLE:
            span.add_event(name, attributes or {})


class Metrics:
    """
    Prometheus Metrics Wrapper

    Provides pre-defined metrics for the GAAP system:
    - Request counters
    - Latency histograms
    - Active gauges
    - Error tracking
    """

    _instance: Optional["Metrics"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "Metrics":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: MetricsConfig | None = None) -> None:
        if self._initialized:
            return

        self.config = config or MetricsConfig()
        self._metrics: dict[str, Any] = {}
        self._registry: Any = None

        if PROMETHEUS_AVAILABLE:
            self._setup_metrics()

        self._initialized = True

    def _setup_metrics(self) -> None:
        """Initialize Prometheus metrics"""
        ns = self.config.namespace
        sub = self.config.subsystem

        self._metrics["requests_total"] = Counter(
            f"{ns}_requests_total",
            "Total number of requests processed",
            ["layer", "provider", "model", "status"],
            subsystem=sub,
        )

        self._metrics["request_duration_seconds"] = Histogram(
            f"{ns}_request_duration_seconds",
            "Request duration in seconds",
            ["layer", "provider", "operation"],
            subsystem=sub,
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        self._metrics["active_requests"] = Gauge(
            f"{ns}_active_requests",
            "Number of active requests being processed",
            ["layer"],
            subsystem=sub,
        )

        self._metrics["tokens_total"] = Counter(
            f"{ns}_tokens_total",
            "Total number of tokens processed",
            ["provider", "model", "type"],
            subsystem=sub,
        )

        self._metrics["cost_dollars"] = Counter(
            f"{ns}_cost_dollars",
            "Total cost in dollars",
            ["provider", "model"],
            subsystem=sub,
        )

        self._metrics["errors_total"] = Counter(
            f"{ns}_errors_total",
            "Total number of errors",
            ["layer", "error_type", "severity"],
            subsystem=sub,
        )

        self._metrics["healing_attempts_total"] = Counter(
            f"{ns}_healing_attempts_total",
            "Total number of healing attempts",
            ["level", "success"],
            subsystem=sub,
        )

        self._metrics["llm_calls_total"] = Counter(
            f"{ns}_llm_calls_total",
            "Total number of LLM API calls",
            ["provider", "model", "status"],
            subsystem=sub,
        )

        self._metrics["queue_size"] = Gauge(
            f"{ns}_queue_size",
            "Current queue size",
            ["queue_name"],
            subsystem=sub,
        )

        self._metrics["memory_usage_bytes"] = Gauge(
            f"{ns}_memory_usage_bytes",
            "Memory usage in bytes",
            ["tier"],
            subsystem=sub,
        )

    def inc_counter(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1
    ) -> None:
        """Increment a counter"""
        if PROMETHEUS_AVAILABLE and name in self._metrics:
            metric = self._metrics[name]
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def dec_counter(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1
    ) -> None:
        """Decrement a counter"""
        if PROMETHEUS_AVAILABLE and name in self._metrics:
            metric = self._metrics[name]
            if labels:
                metric.labels(**labels).dec(value)
            else:
                metric.dec(value)

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Observe a value in a histogram"""
        if PROMETHEUS_AVAILABLE and name in self._metrics:
            metric = self._metrics[name]
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value"""
        if PROMETHEUS_AVAILABLE and name in self._metrics:
            metric = self._metrics[name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

    def inc_gauge(self, name: str, labels: dict[str, str] | None = None, value: float = 1) -> None:
        """Increment a gauge"""
        if PROMETHEUS_AVAILABLE and name in self._metrics:
            metric = self._metrics[name]
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def dec_gauge(self, name: str, labels: dict[str, str] | None = None, value: float = 1) -> None:
        """Decrement a gauge"""
        if PROMETHEUS_AVAILABLE and name in self._metrics:
            metric = self._metrics[name]
            if labels:
                metric.labels(**labels).dec(value)
            else:
                metric.dec(value)

    def time_histogram(
        self, name: str, labels: dict[str, str] | None = None
    ) -> ContextManager[Any]:
        """Time a block and record in histogram"""

        @contextmanager
        def timer() -> Any:
            start = time.time()
            yield
            elapsed = time.time() - start
            self.observe_histogram(name, elapsed, labels)

        return timer()


class Observability:
    """
    Unified Observability Manager

    Combines tracing and metrics into a single interface:
    - Single initialization
    - Convenient decorators
    - Context managers
    """

    _instance: Optional["Observability"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Observability":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        tracing_config: TracingConfig | None = None,
        metrics_config: MetricsConfig | None = None,
    ) -> None:
        self.tracer = Tracer(tracing_config)
        self.metrics = Metrics(metrics_config)
        self._enabled = True

    def enable(self) -> None:
        """Enable observability"""
        self._enabled = True

    def disable(self) -> None:
        """Disable observability"""
        self._enabled = False

    @contextmanager
    def trace_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        layer: str = "unknown",
    ) -> Any:
        """Context manager for tracing and timing"""
        if not self._enabled:
            yield None
            return

        labels = {"layer": layer}
        timer = self.metrics.time_histogram("request_duration_seconds", labels)
        self.metrics.inc_gauge("active_requests", {"layer": layer})

        with timer, self.tracer.start_span(name, attributes) as span:
            try:
                yield span
            except Exception as e:
                if span:
                    self.tracer.record_exception(span, e)
                self.metrics.inc_counter(
                    "errors_total",
                    {"layer": layer, "error_type": type(e).__name__, "severity": "error"},
                )
                raise
            finally:
                self.metrics.dec_gauge("active_requests", {"layer": layer})

    def traced(self, name: str | None = None, layer: str = "unknown") -> Callable[[F], F]:
        """
        Decorator for automatic tracing and metrics

        Usage:
            @observability.traced(layer="strategic")
            async def my_function():
                ...
        """

        def decorator(func: F) -> F:
            span_name = name or func.__name__

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.trace_span(span_name, layer=layer):
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.trace_span(span_name, layer=layer):
                    return func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            return cast(F, sync_wrapper)

        return decorator

    def record_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency: float,
        success: bool,
    ) -> None:
        """Record an LLM API call with all metrics"""
        if not self._enabled:
            return

        status = "success" if success else "failure"

        self.metrics.inc_counter(
            "llm_calls_total", {"provider": provider, "model": model, "status": status}
        )

        self.metrics.inc_counter(
            "tokens_total", {"provider": provider, "model": model, "type": "input"}, input_tokens
        )
        self.metrics.inc_counter(
            "tokens_total", {"provider": provider, "model": model, "type": "output"}, output_tokens
        )

        self.metrics.inc_counter("cost_dollars", {"provider": provider, "model": model}, cost)

        self.metrics.observe_histogram(
            "request_duration_seconds",
            latency,
            {"layer": "execution", "provider": provider, "operation": "llm_call"},
        )

    def record_healing(self, level: str, success: bool) -> None:
        """Record a healing attempt"""
        if not self._enabled:
            return

        self.metrics.inc_counter(
            "healing_attempts_total", {"level": level, "success": "true" if success else "false"}
        )

    def record_error(self, layer: str, error_type: str, severity: str = "error") -> None:
        """Record an error"""
        if not self._enabled:
            return

        self.metrics.inc_counter(
            "errors_total", {"layer": layer, "error_type": error_type, "severity": severity}
        )


observability = Observability()


def get_tracer() -> Tracer:
    """Get the global tracer instance"""
    return observability.tracer


def get_metrics() -> Metrics:
    """Get the global metrics instance"""
    return observability.metrics


def traced(name: str | None = None, layer: str = "unknown") -> Callable[[F], F]:
    """Convenience decorator using global observability"""
    return observability.traced(name, layer)
