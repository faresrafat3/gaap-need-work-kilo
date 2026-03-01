"""
GAAP Tracing Module - OpenTelemetry Integration

Provides distributed tracing for the GAAP system:
- GAAPEngine instrumentation
- Span creation for each layer
- Custom attributes and events
- TraceContext propagation
- Graceful fallback if opentelemetry not installed

Usage:
    from gaap.observability import GAAPTracer, get_tracer

    tracer = get_tracer()
    with tracer.span("my_operation", attributes={"key": "value"}):
        # Do work
        tracer.add_event("important_event", {"detail": "something"})

    # Instrument the engine
    from gaap.observability import instrument_engine
    instrument_engine(engine)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import RESOURCE, Resource
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

if TYPE_CHECKING:
    from gaap.gaap_engine import GAAPEngine

logger = logging.getLogger("gaap.observability.tracing")


class SpanAttributes:
    """
    Standard span attribute names for GAAP operations.

    These follow OpenTelemetry semantic conventions where applicable
    and extend them with GAAP-specific attributes.
    """

    INTENT_TYPE = "gaap.intent.type"
    COMPLEXITY_SCORE = "gaap.complexity.score"
    MODEL_NAME = "gaap.model.name"
    MODEL_PROVIDER = "gaap.model.provider"
    LAYER_NAME = "gaap.layer.name"
    TASK_ID = "gaap.task.id"
    TASK_TYPE = "gaap.task.type"
    TASK_PRIORITY = "gaap.task.priority"
    SESSION_ID = "gaap.session.id"
    REQUEST_ID = "gaap.request.id"
    OODA_PHASE = "gaap.ooda.phase"
    ITERATION = "gaap.iteration"
    HEALING_LEVEL = "gaap.healing.level"
    AXIOM_VIOLATIONS = "gaap.axiom.violations"
    QUALITY_SCORE = "gaap.quality.score"
    TOKENS_INPUT = "gaap.tokens.input"
    TOKENS_OUTPUT = "gaap.tokens.output"
    COST_USD = "gaap.cost.usd"
    LATENCY_MS = "gaap.latency.ms"


class SpanEvents:
    """
    Standard event names for GAAP spans.

    Events are timestamped log entries within a span that capture
    significant moments during the operation.
    """

    TOT_BRANCH_GENERATED = "ToT_Branch_Generated"
    CRITIC_VOTE = "Critic_Vote"
    TOOL_CALL = "Tool_Call"
    TOOL_RESULT = "Tool_Result"
    HEALING_STARTED = "Healing_Started"
    HEALING_COMPLETED = "Healing_Completed"
    AXIOM_CHECK = "Axiom_Check"
    OODA_PHASE_CHANGE = "OODA_Phase_Change"
    TASK_STARTED = "Task_Started"
    TASK_COMPLETED = "Task_Completed"
    TASK_FAILED = "Task_Failed"
    LLM_REQUEST = "LLM_Request"
    LLM_RESPONSE = "LLM_Response"
    MEMORY_ACCESS = "Memory_Access"
    LESSON_LEARNED = "Lesson_Learned"


@dataclass
class TraceContext:
    """
    Trace context for propagation across async boundaries.

    Contains all information needed to continue a trace in a different
    execution context (e.g., across async tasks, between layers).
    """

    trace_id: str
    span_id: str
    trace_flags: int = 0
    trace_state: dict[str, str] = field(default_factory=dict)
    session_id: str | None = None
    request_id: str | None = None
    parent_span_id: str | None = None

    def to_w3c_header(self) -> str:
        """Convert to W3C TraceContext header format."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_w3c_header(cls, header: str) -> Optional["TraceContext"]:
        """Parse W3C TraceContext header."""
        try:
            parts = header.split("-")
            if len(parts) != 4:
                return None
            version, trace_id, span_id, flags = parts
            if version != "00":
                return None
            return cls(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=int(flags, 16),
            )
        except (ValueError, IndexError):
            return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "parent_span_id": self.parent_span_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceContext":
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            trace_flags=data.get("trace_flags", 0),
            trace_state=data.get("trace_state", {}),
            session_id=data.get("session_id"),
            request_id=data.get("request_id"),
            parent_span_id=data.get("parent_span_id"),
        )


_current_trace_context: ContextVar[Optional[TraceContext]] = ContextVar(
    "gaap_trace_context", default=None
)


def get_current_trace_context() -> Optional[TraceContext]:
    """Get the current trace context from context vars."""
    return _current_trace_context.get()


def set_current_trace_context(ctx: Optional[TraceContext]) -> None:
    """Set the current trace context in context vars."""
    _current_trace_context.set(ctx)


class GAAPTracer:
    """
    OpenTelemetry tracer wrapper for GAAP.

    Provides:
    - Easy span creation with attributes
    - Event recording
    - Context propagation
    - Graceful fallback when OpenTelemetry is not installed

    Usage:
        tracer = GAAPTracer(service_name="my-service")

        with tracer.span("operation", attributes={"key": "value"}) as span:
            tracer.add_event(span, "event_name", {"detail": "value"})

        # Propagate context
        ctx = tracer.get_context()
        # Later, in another context:
        tracer.set_context(ctx)
    """

    _instance: Optional["GAAPTracer"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "GAAPTracer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        service_name: str = "gaap-sovereign",
        service_version: str = "2.1.0-SOVEREIGN",
        environment: str = "production",
        enable_console_export: bool = False,
        otlp_endpoint: str | None = None,
        sample_rate: float = 1.0,
    ) -> None:
        if self._initialized:
            return

        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.sample_rate = sample_rate
        self._tracer: Any = None
        self._provider: Any = None
        self._propagator: Any = None
        self._enabled = True

        if OTEL_AVAILABLE:
            self._setup_tracer(
                enable_console_export=enable_console_export,
                otlp_endpoint=otlp_endpoint,
            )

        self._initialized = True
        logger.info(
            f"GAAPTracer initialized (OpenTelemetry: {'available' if OTEL_AVAILABLE else 'not available'})"
        )

    def _setup_tracer(
        self,
        enable_console_export: bool = False,
        otlp_endpoint: str | None = None,
    ) -> None:
        """Initialize OpenTelemetry tracer provider."""
        if not OTEL_AVAILABLE:
            return

        resource = Resource.create(
            {
                RESOURCE: self.service_name,
                "service.version": self.service_version,
                "deployment.environment": self.environment,
            }
        )

        self._provider = TracerProvider(resource=resource)

        if enable_console_export:
            console_exporter = ConsoleSpanExporter()
            self._provider.add_span_processor(SimpleSpanProcessor(console_exporter))

        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                self._provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except ImportError:
                logger.warning("OTLP exporter not available, skipping OTLP export")

        otel_trace.set_tracer_provider(self._provider)
        self._tracer = otel_trace.get_tracer(self.service_name, self.service_version)
        self._propagator = TraceContextTextMapPropagator()

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled and OTEL_AVAILABLE

    def enable(self) -> None:
        """Enable tracing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable tracing."""
        self._enabled = False

    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: Any = None,
        record_exception: bool = True,
    ) -> ContextManager[Any]:
        """
        Create a new span.

        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Span kind (INTERNAL, CLIENT, SERVER, etc.)
            record_exception: Whether to automatically record exceptions

        Returns:
            Context manager yielding the span (or None if disabled)
        """
        if not self.enabled or self._tracer is None:
            return self._noop_span()

        span_kind = kind if kind else SpanKind.INTERNAL

        @contextmanager
        def _span_context() -> Any:
            start_time = time.time()
            with self._tracer.start_as_current_span(
                name, attributes=attributes, kind=span_kind
            ) as span:
                trace_context = TraceContext(
                    trace_id=format(span.context.trace_id, "032x"),
                    span_id=format(span.context.span_id, "016x"),
                    trace_flags=span.context.trace_flags,
                    session_id=attributes.get(SpanAttributes.SESSION_ID) if attributes else None,
                    request_id=attributes.get(SpanAttributes.REQUEST_ID) if attributes else None,
                )
                set_current_trace_context(trace_context)

                try:
                    yield span
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                finally:
                    elapsed_ms = (time.time() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, elapsed_ms)
                    set_current_trace_context(None)

        return _span_context()

    @contextmanager
    def _noop_span(self) -> Any:
        """No-op span context manager when tracing is disabled."""
        yield None

    def add_event(
        self,
        span: Any,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an event to the current span.

        Args:
            span: The span to add the event to (can be None)
            name: Event name (use SpanEvents constants)
            attributes: Optional event attributes
        """
        if span is not None and self.enabled:
            span.add_event(name, attributes or {})

    def set_attribute(self, span: Any, key: str, value: Any) -> None:
        """
        Set an attribute on the span.

        Args:
            span: The span to set the attribute on
            key: Attribute key (use SpanAttributes constants)
            value: Attribute value
        """
        if span is not None and self.enabled:
            span.set_attribute(key, value)

    def record_exception(self, span: Any, exception: Exception) -> None:
        """
        Record an exception on the span.

        Args:
            span: The span to record the exception on
            exception: The exception to record
        """
        if span is not None and self.enabled:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))

    def get_context(self) -> Optional[TraceContext]:
        """
        Get the current trace context for propagation.

        Returns:
            Current TraceContext or None
        """
        return get_current_trace_context()

    def set_context(self, context: Optional[TraceContext]) -> None:
        """
        Set the trace context (for propagation from external source).

        Args:
            context: TraceContext to set
        """
        set_current_trace_context(context)

    def inject_context(self, carrier: dict[str, str]) -> None:
        """
        Inject trace context into a carrier for propagation.

        Args:
            carrier: Dictionary to inject context into
        """
        if self.enabled and self._propagator and OTEL_AVAILABLE:
            self._propagator.inject(carrier)

    def extract_context(self, carrier: dict[str, str]) -> Optional[TraceContext]:
        """
        Extract trace context from a carrier.

        Args:
            carrier: Dictionary containing trace context

        Returns:
            Extracted TraceContext or None
        """
        if self.enabled and self._propagator and OTEL_AVAILABLE:
            ctx = self._propagator.extract(carrier)
            span_context = otel_trace.get_current_span(ctx).get_span_context()
            if span_context.is_valid:
                return TraceContext(
                    trace_id=format(span_context.trace_id, "032x"),
                    span_id=format(span_context.span_id, "016x"),
                    trace_flags=span_context.trace_flags,
                )
        return None

    def traced(
        self,
        name: str | None = None,
        attributes: dict[str, Any] | None = None,
        layer: str = "unknown",
    ) -> Callable[[F], F]:
        """
        Decorator for automatic tracing.

        Args:
            name: Span name (defaults to function name)
            attributes: Static attributes to add to span
            layer: Layer name for categorization

        Returns:
            Decorated function

        Usage:
            @tracer.traced(layer="strategic")
            async def my_function():
                ...
        """

        def decorator(func: F) -> F:
            span_name = name or func.__name__
            func_attrs = dict(attributes) if attributes else {}
            func_attrs[SpanAttributes.LAYER_NAME] = layer

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(span_name, attributes=func_attrs):
                    return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(span_name, attributes=func_attrs):
                    return func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator


def get_tracer() -> GAAPTracer:
    """Get the global GAAPTracer instance."""
    return GAAPTracer()


def instrument_engine(engine: "GAAPEngine") -> "GAAPEngine":
    """
    Instrument a GAAPEngine instance with tracing.

    This wraps key methods of the engine and its layers to automatically
    create spans for each operation.

    Args:
        engine: The GAAPEngine instance to instrument

    Returns:
        The instrumented engine (same instance, modified in place)
    """
    tracer = get_tracer()

    if not tracer.enabled:
        logger.info("Tracing disabled, skipping engine instrumentation")
        return engine

    original_process = engine.process

    async def traced_process(request: Any) -> Any:
        attrs = {
            SpanAttributes.REQUEST_ID: getattr(request, "metadata", {}).get(
                "request_id", "unknown"
            ),
            SpanAttributes.SESSION_ID: getattr(request, "metadata", {}).get(
                "session_id", "unknown"
            ),
        }
        with tracer.span("gaap.engine.process", attributes=attrs) as span:
            tracer.add_event(span, SpanEvents.TASK_STARTED, {"request": str(request)[:100]})
            result = await original_process(request)
            tracer.add_event(span, SpanEvents.TASK_COMPLETED, {"success": result.success})

            tracer.set_attribute(span, SpanAttributes.QUALITY_SCORE, result.quality_score)
            tracer.set_attribute(span, SpanAttributes.COST_USD, result.total_cost_usd)
            tracer.set_attribute(span, SpanAttributes.TOKENS_OUTPUT, result.total_tokens)

            return result

    engine.process = traced_process

    if hasattr(engine, "layer0") and engine.layer0:
        original_layer0 = engine.layer0.process

        async def traced_layer0(request: Any) -> Any:
            with tracer.span(
                "gaap.layer0.process",
                attributes={SpanAttributes.LAYER_NAME: "interface"},
            ) as span:
                result = await original_layer0(request)
                if result:
                    tracer.set_attribute(span, SpanAttributes.INTENT_TYPE, result.intent_type.name)
                    tracer.set_attribute(
                        span, SpanAttributes.COMPLEXITY_SCORE, result.complexity_score
                    )
                return result

        engine.layer0.process = traced_layer0

    if hasattr(engine, "layer1") and engine.layer1:
        original_layer1 = engine.layer1.process

        async def traced_layer1(intent: Any) -> Any:
            with tracer.span(
                "gaap.layer1.process",
                attributes={SpanAttributes.LAYER_NAME: "strategic"},
            ) as span:
                result = await original_layer1(intent)
                tracer.add_event(span, SpanEvents.TOT_BRANCH_GENERATED)
                return result

        engine.layer1.process = traced_layer1

    if hasattr(engine, "layer2") and engine.layer2:
        original_layer2 = engine.layer2.process

        async def traced_layer2(spec: Any) -> Any:
            with tracer.span(
                "gaap.layer2.process",
                attributes={SpanAttributes.LAYER_NAME: "tactical"},
            ) as span:
                result = await original_layer2(spec)
                if result:
                    tracer.set_attribute(span, "gaap.task.count", result.total_tasks)
                return result

        engine.layer2.process = traced_layer2

    if hasattr(engine, "layer3") and engine.layer3:
        original_layer3 = engine.layer3.process

        async def traced_layer3(task: Any) -> Any:
            attrs = {
                SpanAttributes.LAYER_NAME: "execution",
                SpanAttributes.TASK_ID: getattr(task, "id", "unknown"),
                SpanAttributes.TASK_TYPE: getattr(task, "type", "unknown"),
            }
            with tracer.span("gaap.layer3.process", attributes=attrs) as span:
                result = await original_layer3(task)
                tracer.set_attribute(span, SpanAttributes.QUALITY_SCORE, result.quality_score)
                tracer.add_event(span, SpanEvents.TASK_COMPLETED)
                return result

        engine.layer3.process = traced_layer3

    logger.info("GAAPEngine instrumented with tracing")
    return engine
