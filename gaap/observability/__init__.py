"""
GAAP Observability Module

Comprehensive observability stack for the GAAP system:
- Distributed tracing with OpenTelemetry
- Metrics collection with Prometheus export
- Session recording and replay
- Flight recorder for crash analysis
- Dashboard data providers
- Performance monitoring with percentiles

Usage:
    from gaap.observability import (
        GAAPTracer,
        GAAPMetrics,
        SessionRecorder,
        SessionReplay,
        FlightRecorder,
        DashboardProvider,
        PerformanceMonitor,
    )

    tracer = GAAPTracer()
    with tracer.span("operation"):
        ...

    metrics = GAAPMetrics()
    metrics.record_request("layer1", "provider", "model", success=True)

    # Performance monitoring
    monitor = PerformanceMonitor()
    with monitor.timing("database_query", component="storage"):
        result = database.query()
"""

from .dashboard import DashboardData, DashboardProvider
from .flight_recorder import FlightEvent, FlightRecorder
from .metrics import GAAPMetrics, MetricsCollector, get_metrics
from .performance_monitor import (
    ErrorStats,
    LatencyStats,
    MemoryStats,
    PerformanceConfig,
    PerformanceMetrics,
    PerformanceMonitor,
    SamplingStrategy,
    ThroughputStats,
)
from .performance_monitor import get_metrics as get_performance_metrics
from .performance_monitor import (
    get_performance_monitor,
    record_error,
    record_memory,
    timed,
    timing,
)
from .replay import (
    RecordedSession,
    RecordedStep,
    SessionRecorder,
    SessionReplay,
    SessionState,
)
from .tracing import (
    GAAPTracer,
    SpanAttributes,
    SpanEvents,
    TraceContext,
    get_tracer,
    instrument_engine,
)

__all__ = [
    # Tracing
    "GAAPTracer",
    "TraceContext",
    "SpanAttributes",
    "SpanEvents",
    "instrument_engine",
    "get_tracer",
    # Metrics
    "GAAPMetrics",
    "MetricsCollector",
    "get_metrics",
    # Performance Monitor
    "PerformanceMonitor",
    "PerformanceConfig",
    "SamplingStrategy",
    "LatencyStats",
    "MemoryStats",
    "ThroughputStats",
    "ErrorStats",
    "PerformanceMetrics",
    "get_performance_monitor",
    "timing",
    "timed",
    "record_memory",
    "record_error",
    "get_performance_metrics",
    # Session
    "SessionRecorder",
    "SessionReplay",
    "RecordedSession",
    "RecordedStep",
    "SessionState",
    # Flight Recorder
    "FlightRecorder",
    "FlightEvent",
    # Dashboard
    "DashboardProvider",
    "DashboardData",
]
