"""
System API - System Health and Management Endpoints
===================================================

Provides REST API for system monitoring and management.

Endpoints:
- GET /api/system/health - Get system health status
- GET /api/system/metrics - Get system metrics
- GET /api/system/logs - Get system logs
- POST /api/system/restart - Restart system components
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gaap.core.config import get_config
from gaap.core.events import EventEmitter, EventType
from gaap.core.observability import observability

logger = logging.getLogger("gaap.api.system")

router = APIRouter(prefix="/api/system", tags=["system"])

_start_time = time.time()


class HealthStatus(str):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health of a single component."""

    name: str
    status: str
    message: str = ""
    latency_ms: float | None = None
    details: dict[str, Any] = {}


class SystemHealthResponse(BaseModel):
    """System health response."""

    status: str
    version: str
    uptime_seconds: float
    timestamp: str
    components: list[ComponentHealth] = []


class MetricValue(BaseModel):
    """Single metric value."""

    name: str
    value: float
    unit: str
    timestamp: str


class MetricsResponse(BaseModel):
    """System metrics response."""

    system: dict[str, Any] = {}
    memory: dict[str, Any] = {}
    providers: dict[str, Any] = {}
    budget: dict[str, Any] = {}
    healing: dict[str, Any] = {}
    custom: list[MetricValue] = []


class LogEntry(BaseModel):
    """Single log entry."""

    timestamp: str
    level: str
    logger: str
    message: str
    extra: dict[str, Any] = {}


class LogsResponse(BaseModel):
    """Logs response."""

    logs: list[LogEntry]
    total: int


class RestartRequest(BaseModel):
    """Request to restart components."""

    components: list[str] = []
    graceful: bool = True
    timeout_seconds: int = 30


class RestartResponse(BaseModel):
    """Restart response."""

    success: bool
    restarted: list[str] = []
    failed: list[str] = []
    message: str = ""


class SystemInfoResponse(BaseModel):
    """System information response."""

    name: str
    version: str
    environment: str
    python_version: str
    platform: str
    architecture: str
    uptime_seconds: float
    config_path: str | None = None


def _check_memory_health() -> ComponentHealth:
    """Check memory system health."""
    try:
        from gaap.memory.hierarchical import HierarchicalMemory

        memory = HierarchicalMemory()
        stats = memory.get_stats()

        working_usage = (
            stats["working"]["size"] / stats["working"]["max_size"]
            if stats["working"]["max_size"] > 0
            else 0
        )

        if working_usage > 0.9:
            status = HealthStatus.DEGRADED
            message = "Working memory near capacity"
        else:
            status = HealthStatus.HEALTHY
            message = "Memory system operational"

        return ComponentHealth(
            name="memory",
            status=status,
            message=message,
            details=stats,
        )
    except Exception as e:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


def _check_config_health() -> ComponentHealth:
    """Check configuration health."""
    try:
        config = get_config()

        return ComponentHealth(
            name="config",
            status=HealthStatus.HEALTHY,
            message="Configuration loaded",
            details={
                "environment": config.system.environment,
                "log_level": config.system.log_level,
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="config",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


def _check_storage_health() -> ComponentHealth:
    """Check storage health."""
    try:
        from gaap.storage.sqlite_store import SQLiteConfig, SQLiteStore

        store = SQLiteStore(config=SQLiteConfig())
        stats = store.get_stats()

        return ComponentHealth(
            name="storage",
            status=HealthStatus.HEALTHY,
            message="Storage operational",
            details=stats,
        )
    except Exception as e:
        return ComponentHealth(
            name="storage",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


def _check_events_health() -> ComponentHealth:
    """Check event system health."""
    try:
        emitter = EventEmitter.get_instance()
        subscriber_count = emitter.subscriber_count()

        return ComponentHealth(
            name="events",
            status=HealthStatus.HEALTHY,
            message="Event system operational",
            details={
                "subscribers": subscriber_count,
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="events",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health() -> SystemHealthResponse:
    """Get system health status."""
    try:
        components = [
            _check_config_health(),
            _check_memory_health(),
            _check_storage_health(),
            _check_events_health(),
        ]

        statuses = [c.status for c in components]

        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        config = get_config()

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SYSTEM_HEALTH,
            {"status": overall_status, "components": len(components)},
            source="system_api",
        )

        return SystemHealthResponse(
            status=overall_status,
            version=config.system.version,
            uptime_seconds=time.time() - _start_time,
            timestamp=datetime.now().isoformat(),
            components=components,
        )
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics() -> MetricsResponse:
    """Get system metrics."""
    try:
        config = get_config()

        system_metrics = {
            "uptime_seconds": time.time() - _start_time,
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
        }

        memory_metrics = {}
        try:
            from gaap.memory.hierarchical import HierarchicalMemory

            memory = HierarchicalMemory()
            memory_metrics = memory.get_stats()
        except Exception:
            pass

        providers_metrics = {}
        try:
            providers_metrics = observability.get_provider_metrics()
        except Exception:
            pass

        budget_metrics = {}
        try:
            budget_metrics = {
                "monthly_limit": config.budget.monthly_limit,
                "daily_limit": config.budget.daily_limit,
            }
        except Exception:
            pass

        healing_metrics = {}
        try:
            healing_metrics = observability.get_healing_metrics()
        except Exception:
            pass

        custom_metrics = []
        try:
            for name, value in observability.metrics._counters.items():
                if isinstance(value, dict):
                    for labels, count in value.items():
                        custom_metrics.append(
                            MetricValue(
                                name=name,
                                value=count,
                                unit="count",
                                timestamp=datetime.now().isoformat(),
                            )
                        )
        except Exception:
            pass

        return MetricsResponse(
            system=system_metrics,
            memory=memory_metrics,
            providers=providers_metrics,
            budget=budget_metrics,
            healing=healing_metrics,
            custom=custom_metrics[:50],
        )
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs", response_model=LogsResponse)
async def get_system_logs(
    level: str | None = None,
    logger_name: str | None = None,
    limit: int = 100,
) -> LogsResponse:
    """Get system logs."""
    try:
        emitter = EventEmitter.get_instance()
        history = emitter.get_history(limit=limit)

        logs = []
        for event in history:
            log_entry = LogEntry(
                timestamp=event.timestamp.isoformat(),
                level="INFO",
                logger=event.source or "system",
                message=f"{event.type.name}: {event.data}",
                extra=event.data,
            )

            if level and log_entry.level != level.upper():
                continue
            if logger_name and logger_name not in log_entry.logger:
                continue

            logs.append(log_entry)

        return LogsResponse(logs=logs, total=len(logs))
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart", response_model=RestartResponse)
async def restart_system(request: RestartRequest) -> RestartResponse:
    """Restart system components."""
    try:
        restarted = []
        failed = []

        components_to_restart = request.components or ["memory", "events"]

        for component in components_to_restart:
            try:
                if component == "memory":
                    from gaap.memory.hierarchical import HierarchicalMemory

                    memory = HierarchicalMemory()
                    memory.working.clear()
                    restarted.append("memory")

                elif component == "events":
                    emitter = EventEmitter.get_instance()
                    emitter.clear_history()
                    restarted.append("events")

                elif component == "config":
                    from gaap.core.config import get_config_manager

                    manager = get_config_manager()
                    manager.reload()
                    restarted.append("config")

                else:
                    failed.append(component)

            except Exception as e:
                logger.error(f"Failed to restart {component}: {e}")
                failed.append(component)

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SYSTEM_HEALTH,
            {"action": "restart", "components": restarted},
            source="system_api",
        )

        message = f"Restarted: {', '.join(restarted)}" if restarted else ""
        if failed:
            message += f" Failed: {', '.join(failed)}"

        logger.info(f"System restart: {message}")

        return RestartResponse(
            success=len(failed) == 0,
            restarted=restarted,
            failed=failed,
            message=message,
        )
    except Exception as e:
        logger.error(f"Failed to restart system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info() -> SystemInfoResponse:
    """Get system information."""
    try:
        config = get_config()
        manager = None
        try:
            from gaap.core.config import get_config_manager

            manager = get_config_manager()
        except Exception:
            pass

        return SystemInfoResponse(
            name=config.system.name,
            version=config.system.version,
            environment=config.system.environment,
            python_version=sys.version.split()[0],
            platform=platform.system(),
            architecture=platform.machine(),
            uptime_seconds=time.time() - _start_time,
            config_path=manager._config_path if manager else None,
        )
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_recent_events(limit: int = 50) -> dict[str, Any]:
    """Get recent system events."""
    try:
        emitter = EventEmitter.get_instance()
        history = emitter.get_history(limit=limit)

        events = [e.to_dict() for e in history]

        return {
            "events": events,
            "total": len(events),
        }
    except Exception as e:
        logger.error(f"Failed to get recent events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_system_cache() -> dict[str, Any]:
    """Clear system caches."""
    try:
        cleared = []

        try:
            from gaap.memory.hierarchical import HierarchicalMemory

            memory = HierarchicalMemory()
            memory.working.clear()
            cleared.append("working_memory")
        except Exception:
            pass

        try:
            emitter = EventEmitter.get_instance()
            emitter.clear_history()
            cleared.append("event_history")
        except Exception:
            pass

        logger.info(f"Cleared caches: {cleared}")

        return {
            "success": True,
            "cleared": cleared,
            "message": f"Cleared: {', '.join(cleared)}",
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_routes(app: Any) -> None:
    """Register system routes with FastAPI app."""
    app.include_router(router)
