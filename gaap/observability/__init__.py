"""
GAAP Observability Module

Comprehensive observability stack for the GAAP system:
- Distributed tracing with OpenTelemetry
- Metrics collection with Prometheus export
- Session recording and replay
- Flight recorder for crash analysis
- Dashboard data providers

Usage:
    from gaap.observability import (
        GAAPTracer,
        GAAPMetrics,
        SessionRecorder,
        SessionReplay,
        FlightRecorder,
        DashboardProvider,
    )

    tracer = GAAPTracer()
    with tracer.span("operation"):
        ...

    metrics = GAAPMetrics()
    metrics.record_request("layer1", "provider", "model", success=True)
"""

from .dashboard import DashboardProvider, DashboardData
from .flight_recorder import FlightRecorder, FlightEvent
from .metrics import GAAPMetrics, MetricsCollector, get_metrics
from .replay import (
    SessionRecorder,
    SessionReplay,
    RecordedSession,
    RecordedStep,
    SessionState,
)
from .tracing import (
    GAAPTracer,
    TraceContext,
    SpanAttributes,
    SpanEvents,
    instrument_engine,
    get_tracer,
)

__all__ = [
    "GAAPTracer",
    "TraceContext",
    "SpanAttributes",
    "SpanEvents",
    "instrument_engine",
    "get_tracer",
    "GAAPMetrics",
    "MetricsCollector",
    "get_metrics",
    "SessionRecorder",
    "SessionReplay",
    "RecordedSession",
    "RecordedStep",
    "SessionState",
    "FlightRecorder",
    "FlightEvent",
    "DashboardProvider",
    "DashboardData",
]
