"""
GAAP Metrics Module - Prometheus Metrics Collection

Provides comprehensive metrics collection for the GAAP system:
- Token usage tracking
- Cost tracking
- Latency histograms
- Counters for requests, errors, tool calls
- Gauges for active sessions, memory usage
- Prometheus export format support

Usage:
    from gaap.observability import GAAPMetrics, get_metrics

    metrics = get_metrics()

    # Record a request
    metrics.record_request("layer1", "groq", "llama-3", success=True)

    # Record LLM usage
    metrics.record_llm_usage("groq", "llama-3", input_tokens=100, output_tokens=200, cost=0.01)

    # Get metrics summary
    summary = metrics.get_metrics()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("gaap.observability.metrics")

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        CollectorRegistry,
        generate_latest,
        REGISTRY,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None
    Histogram = None
    Info = None
    CollectorRegistry = None
    REGISTRY = None


@dataclass
class MetricValue:
    """A single metric measurement with timestamp."""

    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Summary of collected metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0
    total_errors: int = 0
    total_tool_calls: int = 0
    total_healing_attempts: int = 0
    active_sessions: int = 0
    avg_latency_ms: float = 0.0
    requests_by_layer: dict[str, int] = field(default_factory=dict)
    requests_by_provider: dict[str, int] = field(default_factory=dict)
    errors_by_type: dict[str, int] = field(default_factory=dict)
    tokens_by_model: dict[str, int] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    latency_samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_errors": self.total_errors,
            "total_tool_calls": self.total_tool_calls,
            "total_healing_attempts": self.total_healing_attempts,
            "active_sessions": self.active_sessions,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "requests_by_layer": dict(self.requests_by_layer),
            "requests_by_provider": dict(self.requests_by_provider),
            "errors_by_type": dict(self.errors_by_type),
            "tokens_by_model": dict(self.tokens_by_model),
            "cost_by_model": {k: round(v, 6) for k, v in self.cost_by_model.items()},
        }


class MetricsCollector:
    """
    Thread-safe metrics collector without external dependencies.

    Collects and aggregates metrics in memory with minimal overhead.
    Used as fallback when Prometheus is not available.
    """

    def __init__(self, namespace: str = "gaap", subsystem: str = "system"):
        self.namespace = namespace
        self.subsystem = subsystem
        self._lock = threading.RLock()
        self._counters: dict[str, list[MetricValue]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._start_time = datetime.now()

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name].append(MetricValue(value, labels=labels or {}))

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value."""
        with self._lock:
            key = f"{name}:{sorted(labels.items())}" if labels else name
            self._gauges[key] = value

    def inc_gauge(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a gauge."""
        with self._lock:
            key = f"{name}:{sorted(labels.items())}" if labels else name
            self._gauges[key] = self._gauges.get(key, 0.0) + value

    def dec_gauge(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Decrement a gauge."""
        with self._lock:
            key = f"{name}:{sorted(labels.items())}" if labels else name
            self._gauges[key] = self._gauges.get(key, 0.0) - value

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Observe a value in a histogram."""
        with self._lock:
            self._histograms[name].append(value)

    def get_counter_value(self, name: str) -> float:
        """Get the total value of a counter."""
        with self._lock:
            return sum(v.value for v in self._counters.get(name, []))

    def get_gauge_value(self, name: str) -> float:
        """Get the current value of a gauge."""
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get statistics for a histogram."""
        with self._lock:
            values = self._histograms.get(name, [])
            if not values:
                return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
            return {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = datetime.now()


class GAAPMetrics:
    """
    Unified metrics collection for GAAP.

    Provides:
    - Prometheus-compatible metrics when available
    - Fallback to in-memory collection
    - Easy-to-use methods for common metric operations
    - Summary generation for dashboards

    Usage:
        metrics = GAAPMetrics()

        # Record operations
        metrics.record_request("strategic", "groq", "llama-3", success=True)
        metrics.record_llm_usage("groq", "llama-3", 100, 200, 0.01)
        metrics.record_tool_call("read_file", success=True)
        metrics.record_error("layer1", "TimeoutError")

        # Get summary
        summary = metrics.get_metrics()
    """

    _instance: Optional["GAAPMetrics"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "GAAPMetrics":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        namespace: str = "gaap",
        subsystem: str = "system",
        enable_prometheus: bool = True,
    ) -> None:
        if self._initialized:
            return

        self.namespace = namespace
        self.subsystem = subsystem
        self._collector = MetricsCollector(namespace, subsystem)
        self._prometheus_metrics: dict[str, Any] = {}
        self._enabled = True
        self._use_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        if self._use_prometheus:
            self._setup_prometheus_metrics()

        self._initialized = True
        logger.info(
            f"GAAPMetrics initialized (Prometheus: {'enabled' if self._use_prometheus else 'disabled'})"
        )

    def _setup_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        ns = self.namespace
        sub = self.subsystem

        self._prometheus_metrics["requests_total"] = Counter(
            f"{ns}_requests_total",
            "Total number of requests processed",
            ["layer", "provider", "model", "status"],
            subsystem=sub,
        )

        self._prometheus_metrics["request_duration_seconds"] = Histogram(
            f"{ns}_request_duration_seconds",
            "Request duration in seconds",
            ["layer", "provider", "operation"],
            subsystem=sub,
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        self._prometheus_metrics["active_sessions"] = Gauge(
            f"{ns}_active_sessions",
            "Number of active sessions",
            subsystem=sub,
        )

        self._prometheus_metrics["active_requests"] = Gauge(
            f"{ns}_active_requests",
            "Number of active requests being processed",
            ["layer"],
            subsystem=sub,
        )

        self._prometheus_metrics["tokens_total"] = Counter(
            f"{ns}_tokens_total",
            "Total number of tokens processed",
            ["provider", "model", "type"],
            subsystem=sub,
        )

        self._prometheus_metrics["cost_dollars"] = Counter(
            f"{ns}_cost_dollars",
            "Total cost in dollars",
            ["provider", "model"],
            subsystem=sub,
        )

        self._prometheus_metrics["errors_total"] = Counter(
            f"{ns}_errors_total",
            "Total number of errors",
            ["layer", "error_type", "severity"],
            subsystem=sub,
        )

        self._prometheus_metrics["tool_calls_total"] = Counter(
            f"{ns}_tool_calls_total",
            "Total number of tool calls",
            ["tool_name", "status"],
            subsystem=sub,
        )

        self._prometheus_metrics["healing_attempts_total"] = Counter(
            f"{ns}_healing_attempts_total",
            "Total number of healing attempts",
            ["level", "success"],
            subsystem=sub,
        )

        self._prometheus_metrics["thought_depth"] = Histogram(
            f"{ns}_thought_depth",
            "Depth of thought in ToT/GoT reasoning",
            ["layer"],
            subsystem=sub,
            buckets=[1, 2, 3, 5, 7, 10, 15, 20],
        )

        self._prometheus_metrics["llm_latency_seconds"] = Histogram(
            f"{ns}_llm_latency_seconds",
            "LLM API call latency in seconds",
            ["provider", "model"],
            subsystem=sub,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        self._prometheus_metrics["memory_usage_bytes"] = Gauge(
            f"{ns}_memory_usage_bytes",
            "Memory usage in bytes",
            ["tier"],
            subsystem=sub,
        )

        self._prometheus_metrics["quality_score"] = Histogram(
            f"{ns}_quality_score",
            "Quality score of generated outputs",
            ["layer", "task_type"],
            subsystem=sub,
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

    def enable(self) -> None:
        """Enable metrics collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable metrics collection."""
        self._enabled = False

    def _inc_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment counter in both collectors."""
        if not self._enabled:
            return
        self._collector.inc_counter(name, value, labels)
        if self._use_prometheus and name in self._prometheus_metrics:
            metric = self._prometheus_metrics[name]
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def _set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge in both collectors."""
        if not self._enabled:
            return
        self._collector.set_gauge(name, value, labels)
        if self._use_prometheus and name in self._prometheus_metrics:
            metric = self._prometheus_metrics[name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

    def _inc_gauge(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment gauge in both collectors."""
        if not self._enabled:
            return
        self._collector.inc_gauge(name, value, labels)
        if self._use_prometheus and name in self._prometheus_metrics:
            metric = self._prometheus_metrics[name]
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def _dec_gauge(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Decrement gauge in both collectors."""
        if not self._enabled:
            return
        self._collector.dec_gauge(name, value, labels)
        if self._use_prometheus and name in self._prometheus_metrics:
            metric = self._prometheus_metrics[name]
            if labels:
                metric.labels(**labels).dec(value)
            else:
                metric.dec(value)

    def _observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Observe value in histogram."""
        if not self._enabled:
            return
        self._collector.observe_histogram(name, value, labels)
        if self._use_prometheus and name in self._prometheus_metrics:
            metric = self._prometheus_metrics[name]
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

    def record_request(
        self,
        layer: str,
        provider: str,
        model: str,
        success: bool,
        latency_seconds: float | None = None,
    ) -> None:
        """
        Record a request.

        Args:
            layer: Layer name (interface, strategic, tactical, execution)
            provider: Provider name (groq, openai, etc.)
            model: Model name
            success: Whether the request succeeded
            latency_seconds: Optional latency in seconds
        """
        status = "success" if success else "failure"

        self._inc_counter(
            "requests_total",
            labels={"layer": layer, "provider": provider, "model": model, "status": status},
        )

        if latency_seconds is not None:
            self._observe_histogram(
                "request_duration_seconds",
                latency_seconds,
                labels={"layer": layer, "provider": provider, "operation": "request"},
            )

    def record_llm_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_seconds: float | None = None,
    ) -> None:
        """
        Record LLM usage.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            latency_seconds: Optional latency in seconds
        """
        self._inc_counter(
            "tokens_total",
            input_tokens,
            labels={"provider": provider, "model": model, "type": "input"},
        )
        self._inc_counter(
            "tokens_total",
            output_tokens,
            labels={"provider": provider, "model": model, "type": "output"},
        )
        self._inc_counter("cost_dollars", cost, labels={"provider": provider, "model": model})

        if latency_seconds is not None:
            self._observe_histogram(
                "llm_latency_seconds",
                latency_seconds,
                labels={"provider": provider, "model": model},
            )

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        latency_seconds: float | None = None,
    ) -> None:
        """
        Record a tool call.

        Args:
            tool_name: Name of the tool
            success: Whether the call succeeded
            latency_seconds: Optional latency in seconds
        """
        status = "success" if success else "failure"
        self._inc_counter("tool_calls_total", labels={"tool_name": tool_name, "status": status})

    def record_error(
        self,
        layer: str,
        error_type: str,
        severity: str = "error",
    ) -> None:
        """
        Record an error.

        Args:
            layer: Layer where the error occurred
            error_type: Type of error (exception class name)
            severity: Error severity (error, warning, critical)
        """
        self._inc_counter(
            "errors_total",
            labels={"layer": layer, "error_type": error_type, "severity": severity},
        )

    def record_healing_attempt(
        self,
        level: str,
        success: bool,
    ) -> None:
        """
        Record a healing attempt.

        Args:
            level: Healing level (L1, L2, L3, L4, L5)
            success: Whether the healing succeeded
        """
        self._inc_counter(
            "healing_attempts_total",
            labels={"level": level, "success": "true" if success else "false"},
        )

    def record_thought_depth(
        self,
        layer: str,
        depth: int,
    ) -> None:
        """
        Record thought depth for ToT/GoT reasoning.

        Args:
            layer: Layer name
            depth: Depth of the thought tree
        """
        self._observe_histogram("thought_depth", float(depth), labels={"layer": layer})

    def record_quality_score(
        self,
        layer: str,
        task_type: str,
        score: float,
    ) -> None:
        """
        Record quality score for generated output.

        Args:
            layer: Layer name
            task_type: Type of task
            score: Quality score (0.0 to 1.0)
        """
        self._observe_histogram(
            "quality_score",
            score,
            labels={"layer": layer, "task_type": task_type},
        )

    def set_active_sessions(self, count: int) -> None:
        """Set the number of active sessions."""
        self._set_gauge("active_sessions", float(count))

    def set_active_requests(self, layer: str, count: int) -> None:
        """Set the number of active requests for a layer."""
        self._set_gauge("active_requests", float(count), labels={"layer": layer})

    def inc_active_requests(self, layer: str) -> None:
        """Increment active requests for a layer."""
        self._inc_gauge("active_requests", labels={"layer": layer})

    def dec_active_requests(self, layer: str) -> None:
        """Decrement active requests for a layer."""
        self._dec_gauge("active_requests", labels={"layer": layer})

    def set_memory_usage(self, tier: str, bytes_used: int) -> None:
        """Set memory usage for a tier."""
        self._set_gauge("memory_usage_bytes", float(bytes_used), labels={"tier": tier})

    def get_metrics(self) -> dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Dictionary containing metrics summary
        """
        summary = MetricsSummary()

        requests = self._collector._counters.get("requests_total", [])
        for mv in requests:
            summary.total_requests += int(mv.value)
            layer = mv.labels.get("layer", "unknown")
            provider = mv.labels.get("provider", "unknown")
            status = mv.labels.get("status", "unknown")
            summary.requests_by_layer[layer] = summary.requests_by_layer.get(layer, 0) + int(
                mv.value
            )
            summary.requests_by_provider[provider] = summary.requests_by_provider.get(
                provider, 0
            ) + int(mv.value)
            if status == "success":
                summary.successful_requests += int(mv.value)
            else:
                summary.failed_requests += int(mv.value)

        tokens_input = self._collector._counters.get("tokens_total", [])
        for mv in tokens_input:
            if mv.labels.get("type") == "input":
                summary.total_tokens_input += int(mv.value)
                model = mv.labels.get("model", "unknown")
                summary.tokens_by_model[model] = summary.tokens_by_model.get(model, 0) + int(
                    mv.value
                )
            else:
                summary.total_tokens_output += int(mv.value)

        costs = self._collector._counters.get("cost_dollars", [])
        for mv in costs:
            summary.total_cost_usd += mv.value
            model = mv.labels.get("model", "unknown")
            summary.cost_by_model[model] = summary.cost_by_model.get(model, 0.0) + mv.value

        errors = self._collector._counters.get("errors_total", [])
        for mv in errors:
            summary.total_errors += int(mv.value)
            error_type = mv.labels.get("error_type", "unknown")
            summary.errors_by_type[error_type] = summary.errors_by_type.get(error_type, 0) + int(
                mv.value
            )

        tool_calls = self._collector._counters.get("tool_calls_total", [])
        for mv in tool_calls:
            summary.total_tool_calls += int(mv.value)

        healing = self._collector._counters.get("healing_attempts_total", [])
        for mv in healing:
            summary.total_healing_attempts += int(mv.value)

        latency_stats = self._collector.get_histogram_stats("request_duration_seconds")
        summary.avg_latency_ms = latency_stats.get("avg", 0.0) * 1000
        summary.latency_samples = self._collector._histograms.get("request_duration_seconds", [])[
            -100:
        ]

        summary.active_sessions = int(self._collector.get_gauge_value("active_sessions"))

        return summary.to_dict()

    def get_prometheus_export(self) -> bytes:
        """
        Get metrics in Prometheus text export format.

        Returns:
            Prometheus-formatted metrics as bytes
        """
        if self._use_prometheus and PROMETHEUS_AVAILABLE:
            return generate_latest(REGISTRY)
        return b""

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._collector.reset()
        logger.info("Metrics reset")


def get_metrics() -> GAAPMetrics:
    """Get the global GAAPMetrics instance."""
    return GAAPMetrics()
