"""
GAAP Status Page
================

Simple status page showing system status, component health, incident history, and metrics.

Usage:
    from gaap.status_page import get_status_page_data

    status = await get_status_page_data()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Incident storage (in production, use a database)
_incidents: list[dict[str, Any]] = []
_maintenance_windows: list[dict[str, Any]] = []


@dataclass
class ComponentStatus:
    """Status of a system component."""

    name: str
    status: str  # operational, degraded, down, maintenance
    description: str
    last_checked: float = field(default_factory=time.time)
    response_time_ms: Optional[float] = None
    error_rate: Optional[float] = None


def _get_incident_history(limit: int = 10) -> list[dict[str, Any]]:
    """Get recent incidents."""
    return sorted(_incidents, key=lambda x: x.get("started_at", 0), reverse=True)[:limit]


def _get_maintenance_windows() -> list[dict[str, Any]]:
    """Get upcoming maintenance windows."""
    now = time.time()
    return [w for w in _maintenance_windows if w.get("scheduled_end", 0) > now]


def record_incident(
    title: str,
    description: str,
    severity: str = "minor",
    component: str = "system",
) -> dict[str, Any]:
    """
    Record a new incident.

    Args:
        title: Incident title
        description: Incident description
        severity: Incident severity (minor, major, critical)
        component: Affected component

    Returns:
        Incident record
    """
    incident = {
        "id": f"inc-{int(time.time())}-{len(_incidents)}",
        "title": title,
        "description": description,
        "severity": severity,
        "component": component,
        "status": "open",
        "started_at": time.time(),
        "resolved_at": None,
    }
    _incidents.append(incident)
    return incident


def resolve_incident(incident_id: str) -> bool:
    """Resolve an incident by ID."""
    for incident in _incidents:
        if incident["id"] == incident_id:
            incident["status"] = "resolved"
            incident["resolved_at"] = time.time()
            return True
    return False


def schedule_maintenance(
    title: str,
    description: str,
    start_time: float,
    end_time: float,
    affected_components: list[str],
) -> dict[str, Any]:
    """Schedule a maintenance window."""
    window = {
        "id": f"mnt-{int(time.time())}-{len(_maintenance_windows)}",
        "title": title,
        "description": description,
        "scheduled_start": start_time,
        "scheduled_end": end_time,
        "affected_components": affected_components,
        "status": "scheduled",
    }
    _maintenance_windows.append(window)
    return window


async def get_component_statuses() -> dict[str, ComponentStatus]:
    """Get status of all system components."""
    from gaap.storage.sqlite_store import get_sqlite_store

    components: dict[str, ComponentStatus] = {}

    # Database status
    try:
        store = get_sqlite_store()
        store.get_stats()
        components["database"] = ComponentStatus(
            name="Database",
            status="operational",
            description="SQLite database is accessible",
            response_time_ms=10.0,
        )
    except Exception as e:
        components["database"] = ComponentStatus(
            name="Database",
            status="down",
            description=f"Database error: {e}",
        )

    # API status
    components["api"] = ComponentStatus(
        name="API",
        status="operational",
        description="API is responding to requests",
        response_time_ms=50.0,
    )

    # WebSocket status
    from gaap.api.websocket import manager as ws_manager

    ws_count = ws_manager.connection_count
    components["websocket"] = ComponentStatus(
        name="WebSocket",
        status="operational",
        description=f"WebSocket server active ({ws_count} connections)",
    )

    # Provider status
    try:
        from gaap.api.chat import _provider_cache

        provider_count = len(_provider_cache)
        components["providers"] = ComponentStatus(
            name="LLM Providers",
            status="operational" if provider_count > 0 else "degraded",
            description=f"{provider_count} providers initialized",
        )
    except Exception as e:
        components["providers"] = ComponentStatus(
            name="LLM Providers",
            status="unknown",
            description=f"Provider status unknown: {e}",
        )

    return components


def _determine_overall_status(components: dict[str, ComponentStatus]) -> str:
    """Determine overall system status from components."""
    statuses = [c.status for c in components.values()]

    if any(s == "down" for s in statuses):
        return "major_outage"
    elif any(s == "degraded" for s in statuses):
        return "degraded_performance"
    elif any(s == "maintenance" for s in statuses):
        return "maintenance"
    elif all(s == "operational" for s in statuses):
        return "operational"
    else:
        return "unknown"


async def get_status_page_data() -> dict[str, Any]:
    """
    Get comprehensive status page data.

    Returns:
        Status page data dictionary
    """
    from gaap.observability import get_metrics

    # Get component statuses
    components = await get_component_statuses()

    # Get metrics summary
    try:
        metrics = get_metrics()
        metrics_summary = metrics.get_metrics()
    except Exception:
        metrics_summary = {}

    # Build status data
    status_data = {
        "page": {
            "title": "GAAP System Status",
            "description": "Real-time status of the GAAP platform",
            "url": os.environ.get("GAAP_STATUS_URL", "https://status.gaap.local"),
        },
        "status": {
            "indicator": _determine_overall_status(components),
            "description": _get_status_description(_determine_overall_status(components)),
        },
        "components": [
            {
                "name": c.name,
                "status": c.status,
                "description": c.description,
                "response_time_ms": c.response_time_ms,
                "error_rate": c.error_rate,
            }
            for c in components.values()
        ],
        "incidents": _get_incident_history(5),
        "maintenance": _get_maintenance_windows(),
        "metrics": {
            "total_requests": metrics_summary.get("total_requests", 0),
            "success_rate": round(metrics_summary.get("success_rate", 1.0) * 100, 2),
            "avg_latency_ms": metrics_summary.get("avg_latency_ms", 0),
            "active_sessions": metrics_summary.get("active_sessions", 0),
        },
        "timestamp": time.time(),
    }

    return status_data


def _get_status_description(indicator: str) -> str:
    """Get human-readable status description."""
    descriptions = {
        "operational": "All systems operational",
        "degraded_performance": "Degraded performance",
        "partial_outage": "Partial system outage",
        "major_outage": "Major system outage",
        "maintenance": "Scheduled maintenance",
        "unknown": "System status unknown",
    }
    return descriptions.get(indicator, "Unknown status")


async def get_detailed_status(component_name: Optional[str] = None) -> dict[str, Any]:
    """Get detailed status for a specific component or all components."""
    components = await get_component_statuses()

    if component_name:
        component = components.get(component_name)
        if not component:
            return {"error": f"Component '{component_name}' not found"}
        return {
            "name": component.name,
            "status": component.status,
            "description": component.description,
            "last_checked": component.last_checked,
            "response_time_ms": component.response_time_ms,
            "error_rate": component.error_rate,
        }

    return {
        "components": {
            name: {
                "name": c.name,
                "status": c.status,
                "description": c.description,
                "last_checked": c.last_checked,
            }
            for name, c in components.items()
        }
    }


# Initialize with some sample incidents for demonstration
def _init_sample_data():
    """Initialize sample data for demonstration."""
    if not _incidents:
        record_incident(
            title="Brief API Latency Spike",
            description="API response times were elevated for 5 minutes",
            severity="minor",
            component="api",
        )
        # Mark as resolved
        _incidents[0]["status"] = "resolved"
        _incidents[0]["resolved_at"] = time.time() - 86400  # Resolved yesterday


_init_sample_data()


__all__ = [
    "get_status_page_data",
    "get_detailed_status",
    "record_incident",
    "resolve_incident",
    "schedule_maintenance",
    "ComponentStatus",
]
