"""
GAAP Tracing Module
===================

OpenTelemetry integration for distributed tracing.

Provides:
- Request tracing
- Database query tracing
- Provider call tracing
- Distributed tracing across services
- Trace correlation with logs

Usage:
    from gaap.tracing import get_tracer, trace_span

    # Simple span
    with trace_span("operation_name"):
        do_work()

    # Span with attributes
    with trace_span("db_query", {"table": "users", "operation": "select"}):
        db.execute()
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("gaap.tracing")

# OpenTelemetry availability
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import (
        DEPLOYMENT_ENVIRONMENT,
        SERVICE_NAME,
        SERVICE_VERSION,
        Resource,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    otel_trace = None
    SpanKind = None
    Status = None
    StatusCode = None

F = TypeVar("F", bound=Callable[..., Any])

# Context variables for trace propagation
_current_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)


class GAAPTracer:
    """OpenTelemetry tracer wrapper for GAAP."""

    _instance: Optional["GAAPTracer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        service_name: str = "gaap",
        service_version: str = "1.0.0",
        environment: str = "development",
        otlp_endpoint: Optional[str] = None,
        sample_rate: float = 1.0,
    ):
        if self._initialized:
            return

        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.sample_rate = sample_rate
        self._otlp_endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        self._tracer = None
        self._provider = None
        self._propagator = None

        if OTEL_AVAILABLE:
            self._setup_tracer()

        self._initialized = True
        logger.info(f"GAAPTracer initialized (OTel available: {OTEL_AVAILABLE})")

    def _setup_tracer(self) -> None:
        """Initialize OpenTelemetry tracer."""
        if not OTEL_AVAILABLE:
            return

        try:
            resource = Resource.create(
                {
                    SERVICE_NAME: self.service_name,
                    SERVICE_VERSION: self.service_version,
                    DEPLOYMENT_ENVIRONMENT: self.environment,
                }
            )

            self._provider = TracerProvider(resource=resource)

            # Console exporter for debugging
            if os.environ.get("OTEL_DEBUG", "false").lower() == "true":
                console_exporter = ConsoleSpanExporter()
                self._provider.add_span_processor(SimpleSpanProcessor(console_exporter))

            # OTLP exporter for production
            if self._otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    otlp_exporter = OTLPSpanExporter(endpoint=self._otlp_endpoint)
                    self._provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    logger.info(f"OTLP exporter configured: {self._otlp_endpoint}")
                except ImportError:
                    logger.warning("OTLP exporter not available")

            otel_trace.set_tracer_provider(self._provider)
            self._tracer = otel_trace.get_tracer(self.service_name, self.service_version)
            self._propagator = TraceContextTextMapPropagator()

        except Exception as e:
            logger.error(f"Failed to setup tracer: {e}")

    @property
    def is_available(self) -> bool:
        """Check if tracing is available."""
        return OTEL_AVAILABLE and self._tracer is not None

    def start_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        kind: Any = None,
    ) -> Any:
        """Start a new span."""
        if not self.is_available:
            return None

        span_kind = kind or SpanKind.INTERNAL
        return self._tracer.start_span(name, attributes=attributes, kind=span_kind)

    def get_current_span(self) -> Any:
        """Get the current span."""
        if not self.is_available:
            return None
        return otel_trace.get_current_span()

    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute on current span."""
        span = self.get_current_span()
        if span:
            span.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Add event to current span."""
        span = self.get_current_span()
        if span:
            span.add_event(name, attributes or {})

    def record_exception(self, exception: Exception) -> None:
        """Record exception on current span."""
        span = self.get_current_span()
        if span:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))

    def inject_context(self, carrier: dict[str, str]) -> None:
        """Inject trace context into carrier."""
        if self._propagator and self.is_available:
            self._propagator.inject(carrier)

    def extract_context(self, carrier: dict[str, str]) -> Any:
        """Extract trace context from carrier."""
        if self._propagator and self.is_available:
            return self._propagator.extract(carrier)
        return None


def get_tracer() -> GAAPTracer:
    """Get the global tracer instance."""
    return GAAPTracer()


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
    kind: Optional[str] = None,
):
    """
    Context manager for creating a span.

    Usage:
        with trace_span("db_query", {"table": "users"}):
            result = db.execute()
    """
    tracer = get_tracer()

    if not tracer.is_available:
        yield None
        return

    span_kind = None
    if kind:
        kind_map = {
            "internal": SpanKind.INTERNAL,
            "server": SpanKind.SERVER,
            "client": SpanKind.CLIENT,
            "producer": SpanKind.PRODUCER,
            "consumer": SpanKind.CONSUMER,
        }
        span_kind = kind_map.get(kind.lower())

    try:
        with tracer._tracer.start_as_current_span(
            name, attributes=attributes, kind=span_kind
        ) as span:
            yield span
    except Exception as e:
        if tracer.is_available:
            tracer.record_exception(e)
        raise


def traced(
    name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
):
    """
    Decorator for tracing functions.

    Usage:
        @traced("process_request")
        async def process_request(data):
            return await handle(data)
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_span(span_name, attributes):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name, attributes):
                return func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class TracingMiddleware:
    """FastAPI middleware for request tracing."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Process request with tracing."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        tracer = get_tracer()

        if not tracer.is_available:
            await self.app(scope, receive, send)
            return

        # Extract context from headers
        headers = dict(scope.get("headers", []))
        headers_str = {k.decode(): v.decode() for k, v in headers.items()}
        context = tracer.extract_context(headers_str)

        request_method = scope.get("method", "GET")
        request_path = scope.get("path", "/")

        attributes = {
            "http.method": request_method,
            "http.path": request_path,
            "http.host": headers_str.get("host", "unknown"),
            "http.user_agent": headers_str.get("user-agent", "unknown"),
        }

        with trace_span(f"{request_method} {request_path}", attributes, kind="server") as span:
            if span:
                # Store trace info in scope for access in endpoints
                scope["gaap.trace_id"] = (
                    format(span.context.trace_id, "032x") if hasattr(span, "context") else None
                )
                scope["gaap.span_id"] = (
                    format(span.context.span_id, "016x") if hasattr(span, "context") else None
                )

            async def wrapped_send(message: Any) -> None:
                if message["type"] == "http.response.start":
                    status_code = message.get("status", 200)
                    tracer.set_attribute("http.status_code", status_code)

                    if status_code >= 400:
                        tracer.get_current_span().set_status(
                            Status(StatusCode.ERROR, f"HTTP {status_code}")
                        )

                await send(message)

            try:
                await self.app(scope, receive, wrapped_send)
            except Exception as e:
                tracer.record_exception(e)
                raise


# Database tracing helpers
@contextmanager
def trace_db_query(
    query: str,
    table: Optional[str] = None,
    operation: Optional[str] = None,
):
    """
    Trace a database query.

    Usage:
        with trace_db_query("SELECT * FROM users", table="users", operation="select"):
            cursor.execute(query)
    """
    attributes = {
        "db.system": "sqlite",  # or postgresql, mysql, etc.
        "db.statement": query[:1000],  # Truncate long queries
    }
    if table:
        attributes["db.table"] = table
    if operation:
        attributes["db.operation"] = operation

    with trace_span("db.query", attributes, kind="client") as span:
        start_time = logging.time() if hasattr(logging, "time") else __import__("time").time()
        try:
            yield span
        finally:
            duration = (__import__("time").time() - start_time) * 1000
            if span:
                tracer = get_tracer()
                tracer.set_attribute("db.duration_ms", duration)


# Provider tracing helpers
@contextmanager
def trace_provider_call(
    provider: str,
    model: str,
    operation: str = "completion",
):
    """
    Trace an LLM provider call.

    Usage:
        with trace_provider_call("kimi", "k2.5") as span:
            result = await provider.complete(prompt)
            span.set_attribute("tokens.input", result.input_tokens)
    """
    attributes = {
        "llm.provider": provider,
        "llm.model": model,
        "llm.operation": operation,
    }

    with trace_span("llm.request", attributes, kind="client") as span:
        yield span


def add_llm_attributes(
    input_tokens: int,
    output_tokens: int,
    cost_usd: Optional[float] = None,
) -> None:
    """Add LLM-specific attributes to current span."""
    tracer = get_tracer()
    tracer.set_attribute("llm.tokens.input", input_tokens)
    tracer.set_attribute("llm.tokens.output", output_tokens)
    tracer.set_attribute("llm.tokens.total", input_tokens + output_tokens)
    if cost_usd is not None:
        tracer.set_attribute("llm.cost.usd", cost_usd)


def get_trace_context() -> dict[str, Optional[str]]:
    """Get current trace context for propagation."""
    tracer = get_tracer()

    if not tracer.is_available:
        return {"trace_id": None, "span_id": None}

    span = tracer.get_current_span()
    if span and hasattr(span, "get_span_context"):
        ctx = span.get_span_context()
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }

    return {"trace_id": None, "span_id": None}


__all__ = [
    "GAAPTracer",
    "get_tracer",
    "trace_span",
    "traced",
    "TracingMiddleware",
    "trace_db_query",
    "trace_provider_call",
    "add_llm_attributes",
    "get_trace_context",
    "OTEL_AVAILABLE",
]
