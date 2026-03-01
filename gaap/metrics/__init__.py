"""
GAAP Metrics Module
===================

Prometheus client setup and core metrics infrastructure for GAAP.

This module provides a centralized metrics collection system with:
- Prometheus client initialization
- Registry management
- Metric naming conventions
- Export utilities

Usage:
    from gaap.metrics import get_registry, generate_metrics_report

    # Get the global registry
    registry = get_registry()

    # Generate metrics report
    metrics_data = generate_metrics_report()
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("gaap.metrics")

# Prometheus client availability
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )
    from prometheus_client.exposition import make_wsgi_app

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = Info = None
    CollectorRegistry = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    make_wsgi_app = None
    REGISTRY = None

# Global registry instance
_global_registry: Optional[CollectorRegistry] = None
_metrics_initialized = False


def get_registry() -> Optional[CollectorRegistry]:
    """Get the global Prometheus registry."""
    global _global_registry
    if not PROMETHEUS_AVAILABLE:
        return None
    if _global_registry is None:
        # Use default registry or create isolated one based on config
        use_isolated = os.environ.get("GAAP_METRICS_ISOLATED", "false").lower() == "true"
        _global_registry = CollectorRegistry() if use_isolated else REGISTRY
    return _global_registry


def create_isolated_registry() -> Optional[CollectorRegistry]:
    """Create a new isolated registry for testing."""
    if not PROMETHEUS_AVAILABLE:
        return None
    return CollectorRegistry()


def get_metrics_export() -> tuple[bytes, str]:
    """
    Generate Prometheus metrics export.

    Returns:
        Tuple of (metrics_data, content_type)
    """
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not available\n", CONTENT_TYPE_LATEST

    registry = get_registry()
    if registry is None:
        return b"# Registry not available\n", CONTENT_TYPE_LATEST

    return generate_latest(registry), CONTENT_TYPE_LATEST


def generate_metrics_report() -> dict[str, Any]:
    """
    Generate a comprehensive metrics report.

    Returns:
        Dictionary containing metrics summary
    """
    from gaap.observability import get_metrics

    metrics = get_metrics()
    return metrics.get_metrics()


def initialize_metrics(
    namespace: str = "gaap",
    subsystem: str = "api",
    enable_prometheus: bool = True,
) -> bool:
    """
    Initialize the metrics system.

    Args:
        namespace: Metrics namespace prefix
        subsystem: Metrics subsystem
        enable_prometheus: Whether to enable Prometheus export

    Returns:
        True if initialization successful
    """
    global _metrics_initialized

    if _metrics_initialized:
        return True

    if not enable_prometheus:
        logger.info("Metrics initialization skipped (disabled)")
        return True

    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available, using fallback metrics")
        # Still initialize the fallback metrics collector
        from gaap.observability import get_metrics

        get_metrics()
        _metrics_initialized = True
        return True

    # Initialize GAAPMetrics which sets up Prometheus metrics
    from gaap.observability import get_metrics

    metrics = get_metrics()
    logger.info(f"Metrics initialized with Prometheus: {metrics._use_prometheus}")

    _metrics_initialized = True
    return True


def get_metrics_status() -> dict[str, Any]:
    """Get current metrics system status."""
    return {
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "initialized": _metrics_initialized,
        "registry_type": "isolated" if os.environ.get("GAAP_METRICS_ISOLATED") else "default",
        "timestamp": time.time(),
    }


# Metric naming conventions
class MetricNames:
    """Standard metric names for GAAP."""

    # Request metrics
    REQUESTS_TOTAL = "requests_total"
    REQUEST_DURATION_SECONDS = "request_duration_seconds"
    REQUEST_SIZE_BYTES = "request_size_bytes"
    RESPONSE_SIZE_BYTES = "response_size_bytes"

    # Connection metrics
    ACTIVE_CONNECTIONS = "active_connections"
    ACTIVE_REQUESTS = "active_requests"
    CONNECTION_DURATION = "connection_duration_seconds"

    # Error metrics
    ERRORS_TOTAL = "errors_total"
    ERROR_RATE = "error_rate"

    # Provider metrics
    PROVIDER_LATENCY = "provider_latency_seconds"
    PROVIDER_REQUESTS = "provider_requests_total"
    PROVIDER_ERRORS = "provider_errors_total"

    # Token and cost metrics
    TOKENS_TOTAL = "tokens_total"
    COST_DOLLARS = "cost_dollars"
    TOKEN_USAGE = "token_usage"

    # System metrics
    MEMORY_USAGE_BYTES = "memory_usage_bytes"
    CPU_USAGE_PERCENT = "cpu_usage_percent"
    DISK_USAGE_BYTES = "disk_usage_bytes"

    # Business metrics
    CHAT_SESSIONS = "chat_sessions_total"
    MESSAGES_PER_SESSION = "messages_per_session"
    USERS_ACTIVE = "users_active"


# Default metric labels
DEFAULT_LABELS = {
    "service": "gaap",
    "version": "1.0.0",
}


def get_metric_fully_qualified_name(name: str, namespace: str = "gaap") -> str:
    """Get fully qualified metric name."""
    return f"{namespace}_{name}"


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "get_registry",
    "create_isolated_registry",
    "get_metrics_export",
    "generate_metrics_report",
    "initialize_metrics",
    "get_metrics_status",
    "MetricNames",
    "DEFAULT_LABELS",
    "get_metric_fully_qualified_name",
    "CONTENT_TYPE_LATEST",
]
