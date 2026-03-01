"""
GAAP FastAPI Application
========================

Main FastAPI application that combines all API routers.

Usage:
    # Run server
    uvicorn gaap.api.main:app --reload --port 8000

    # Or programmatically
    from gaap.api.main import create_app
    app = create_app()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from gaap.api.websocket import manager as ws_manager
from gaap.core.events import EventEmitter, EventType
from gaap.db import close_db, init_db
from gaap.logging_config import generate_correlation_id, set_correlation_id

# Monitoring and tracing imports
from gaap.metrics import get_metrics_export, initialize_metrics
from gaap.metrics.collectors import (
    get_connection_metrics,
    get_error_metrics,
    get_request_metrics,
    get_system_metrics,
    initialize_all_collectors,
)
from gaap.tracing import TracingMiddleware

try:
    pass

    _HAS_PROVIDERS = True
except ImportError:
    _HAS_PROVIDERS = False

logger = logging.getLogger("gaap.api")

# Global state for shutdown handling
_shutdown_event = asyncio.Event()
_active_requests: set[asyncio.Task] = set()
_start_time = time.time()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler with graceful shutdown."""
    logger.info("GAAP API starting up...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")

    # Initialize metrics
    initialize_metrics()
    initialize_all_collectors()
    logger.info("Metrics initialized")

    # Initialize event emitter
    emitter = EventEmitter.get_instance()

    # Subscribe WebSocket manager to all events
    async def broadcast_event(event):
        await ws_manager.broadcast_event(event)

    for event_type in EventType:
        emitter.subscribe_async(event_type, broadcast_event)

    # Log startup info
    logger.info(f"GAAP API version {app.version} started")

    yield

    # Graceful shutdown
    logger.info("GAAP API shutting down... Initiating graceful shutdown")

    # Signal shutdown
    _shutdown_event.set()

    # Wait for active requests to complete (with timeout)
    if _active_requests:
        logger.info(f"Waiting for {len(_active_requests)} active requests to complete...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*_active_requests, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Some requests did not complete in time, forcing shutdown")

    # Close WebSocket connections
    logger.info("Closing WebSocket connections...")
    await ws_manager.close_all()

    # Close database connections
    try:
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.warning(f"Error closing database: {e}")

    # Cleanup other resources
    logger.info("Cleanup complete")


# =============================================================================
# Health Check
# =============================================================================


async def get_health_status(detailed: bool = False) -> dict[str, Any]:
    """
    Comprehensive health check of all system components.

    Args:
        detailed: Include detailed component information

    Returns:
        Health status dictionary
    """
    try:
        import psutil

        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False

    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime_seconds": round(time.time() - _start_time, 2),
        "version": "1.0.0",
    }

    # System metrics
    if HAS_PSUTIL:
        try:
            import psutil

            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage("/")

            status["system"] = {
                "memory": {
                    "total_mb": round(memory.total / (1024 * 1024), 2),
                    "available_mb": round(memory.available / (1024 * 1024), 2),
                    "percent_used": memory.percent,
                },
                "cpu_percent": cpu_percent,
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": round((disk.used / disk.total) * 100, 2),
                },
            }

            # Record system metrics
            system_metrics = get_system_metrics()
            system_metrics.record_memory(
                used_bytes=memory.used,
                free_bytes=memory.free,
                available_bytes=memory.available,
            )
            system_metrics.record_disk(
                mount="/",
                used_bytes=disk.used,
                free_bytes=disk.free,
                total_bytes=disk.total,
            )

        except Exception as e:
            status["system"] = {"error": str(e)}
    else:
        status["system"] = {"available": False, "reason": "psutil not installed"}

    # WebSocket connections
    status["websocket"] = {
        "total_connections": ws_manager.connection_count,
        "channels": {
            channel: ws_manager.get_channel_count(channel)
            for channel in ["events", "ooda", "steering"]
        },
    }

    # Database health
    db_health_start = time.time()
    try:
        from sqlalchemy import text

        from gaap.db import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_response_time = (time.time() - db_health_start) * 1000
        status["database"] = {
            "status": "connected",
            "response_time_ms": round(db_response_time, 2),
        }
    except Exception as e:
        status["database"] = {"status": "error", "error": str(e)}

    # Redis health (if configured)
    redis_health_start = time.time()
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            import redis as redis_lib

            r = redis_lib.from_url(redis_url, socket_connect_timeout=2)
            r.ping()
            redis_response_time = (time.time() - redis_health_start) * 1000
            status["redis"] = {
                "status": "connected",
                "response_time_ms": round(redis_response_time, 2),
            }
        except ImportError:
            status["redis"] = {"status": "not_available", "reason": "redis not installed"}
        except Exception as e:
            status["redis"] = {"status": "error", "error": str(e)}
    else:
        status["redis"] = {"status": "not_configured"}

    # Provider health
    provider_status = {}
    provider_health_start = time.time()
    if _HAS_PROVIDERS:
        from gaap.api.chat import _provider_cache

        for name in ["kimi", "deepseek", "glm"]:
            cached = _provider_cache.get(name)
            provider_status[name] = {
                "status": "available" if cached else "not_initialized",
                "response_time_ms": round((time.time() - provider_health_start) * 1000, 2),
            }
    status["providers"] = provider_status

    # Overall status determination
    component_statuses = [
        status.get("database", {}).get("status") == "connected",
    ]

    if not all(component_statuses):
        status["status"] = "degraded"

    # Check for critical issues
    memory_info = status.get("system", {})
    if isinstance(memory_info, dict):
        memory_percent = memory_info.get("memory", {}).get("percent_used", 0)
        if memory_percent > 95:
            status["status"] = "critical"
            status["warnings"] = status.get("warnings", []) + ["High memory usage detected"]

        disk_percent = memory_info.get("disk", {}).get("percent_used", 0)
        if disk_percent > 90:
            status["status"] = "critical"
            status["warnings"] = status.get("warnings", []) + ["Low disk space"]

    # Include metrics summary if detailed
    if detailed:
        try:
            from gaap.observability import get_metrics

            metrics = get_metrics()
            status["metrics_summary"] = metrics.get_metrics()
        except Exception as e:
            status["metrics_summary"] = {"error": str(e)}

    return status


# =============================================================================
# WebSocket with Timeouts
# =============================================================================


async def handle_websocket_with_timeout(
    websocket: WebSocket,
    channel: str,
    handler: Callable,
) -> None:
    """Handle WebSocket with proper timeouts and ping/pong."""
    await ws_manager.connect(websocket, channel)
    last_ping = time.time()
    ping_interval = 30.0  # Send ping every 30 seconds
    receive_timeout = 60.0  # Timeout for receive operations

    try:
        while not _shutdown_event.is_set():
            # Send periodic ping
            if time.time() - last_ping > ping_interval:
                try:
                    await asyncio.wait_for(
                        websocket.send_json({"type": "ping", "timestamp": time.time()}),
                        timeout=5.0,
                    )
                    last_ping = time.time()
                except asyncio.TimeoutError:
                    logger.warning(f"Ping timeout on {channel} channel")
                    break

            # Receive with timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=receive_timeout,
                )

                # Handle ping/pong
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                elif data.get("type") == "pong":
                    pass  # Client responded to our ping
                else:
                    # Pass to handler
                    await handler(data)

            except asyncio.TimeoutError:
                # No message received in timeout period, send ping
                continue

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from {channel} channel")
    except Exception as e:
        logger.error(f"WebSocket error on {channel} channel: {e}")
    finally:
        ws_manager.disconnect(websocket)


# =============================================================================
# App Factory
# =============================================================================


def create_app(
    title: str = "GAAP API",
    version: str = "1.0.0",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure FastAPI application."""

    # Default CORS origins - production should restrict these
    if cors_origins is None:
        cors_origins_str = os.environ.get("CORS_ORIGINS", "")
        if cors_origins_str:
            cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        else:
            cors_origins = [
                "http://localhost:3000",
                "http://localhost:8080",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080",
            ]

    app = FastAPI(
        title=title,
        version=version,
        description="GAAP - General AI Agent Platform API",
        lifespan=lifespan,
    )

    # Store CORS origins for reference
    app.state.cors_origins = cors_origins

    # Tracing middleware (must be first)
    app.add_middleware(TracingMiddleware)

    # Request metrics middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Collect request metrics and add correlation IDs."""
        # Generate correlation ID if not present
        correlation_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()
        set_correlation_id(correlation_id)

        # Add correlation ID to response
        request.state.correlation_id = correlation_id

        # Track request metrics
        request_metrics = get_request_metrics()
        connection_metrics = get_connection_metrics()

        method = request.method
        path = request.url.path

        request_metrics.inc_active(method, path)

        start_time = time.time()
        status_code = 200

        try:
            response = await call_next(request)
            status_code = response.status_code

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response
        except Exception as e:
            status_code = 500
            error_metrics = get_error_metrics()
            error_metrics.record_exception(e, module="gaap.api.main")
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            request_metrics.dec_active(method, path)
            request_metrics.record_request(
                method=method,
                endpoint=path,
                status_code=status_code,
                duration=duration,
            )

            # Log request
            logger.info(
                f"{method} {path} - {status_code} - {duration * 1000:.2f}ms",
                extra={
                    "correlation_id": correlation_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 2),
                },
            )

    # CORS middleware with strict settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
        max_age=600,  # Cache preflight for 10 minutes
    )

    # Security middleware
    from .security_middleware import (
        SecurityHeadersMiddleware,
        RateLimitMiddleware,
        InputValidationMiddleware,
    )
    
    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add rate limiting (60 requests per minute)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    
    # Add input validation
    app.add_middleware(InputValidationMiddleware)

    # Include routers
    from .budget import router as budget_router
    from .chat import router as chat_router
    from .config import router as config_router
    from .context import router as context_router
    from .healing import router as healing_router
    from .knowledge import router as knowledge_router
    from .maintenance import router as maintenance_router
    from .memory import router as memory_router
    from .providers import router as providers_router
    from .providers_status import router as providers_status_router
    from .research import router as research_router
    from .sessions import router as sessions_router
    from .swarm import router as swarm_router
    from .system import router as system_router
    from .validators import router as validators_router

    app.include_router(config_router)
    app.include_router(providers_status_router)  # Must be BEFORE providers_router
    app.include_router(providers_router)
    app.include_router(research_router)
    app.include_router(healing_router)
    app.include_router(memory_router)
    app.include_router(budget_router)
    app.include_router(sessions_router)
    app.include_router(system_router)
    app.include_router(chat_router)
    app.include_router(validators_router)
    app.include_router(context_router)
    app.include_router(knowledge_router)
    app.include_router(maintenance_router)
    app.include_router(swarm_router)

    # Health check endpoint
    @app.get("/api/health")
    async def health_check(request: Request, detailed: bool = False) -> dict[str, Any]:
        """Health check endpoint with optional detailed metrics."""
        return await get_health_status(detailed=detailed)

    # Prometheus metrics endpoint
    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        """Prometheus metrics export endpoint."""
        data, content_type = get_metrics_export()
        return PlainTextResponse(content=data.decode("utf-8"), media_type=content_type)

    # Status page endpoint
    @app.get("/status")
    async def status_page() -> dict[str, Any]:
        """Public status page with system overview."""
        from gaap.status_page import get_status_page_data

        return await get_status_page_data()

    # WebSocket endpoints with improved handling
    @app.websocket("/ws/events")
    async def events_websocket(websocket: WebSocket):
        """Real-time events stream with timeout handling."""

        async def handler(data: dict) -> None:
            # Handle client messages (e.g., subscribe to specific events)
            if data.get("type") == "subscribe":
                event_type = data.get("event_type")
                logger.debug(f"Client subscribed to {event_type}")

        await handle_websocket_with_timeout(websocket, "events", handler)

    @app.websocket("/ws/ooda")
    async def ooda_websocket(websocket: WebSocket):
        """OODA loop visualization stream."""

        async def handler(data: dict) -> None:
            pass

        await handle_websocket_with_timeout(websocket, "ooda", handler)

    @app.websocket("/ws/steering")
    async def steering_websocket(websocket: WebSocket):
        """Steering commands stream."""

        async def handler(data: dict) -> None:
            emitter = EventEmitter.get_instance()

            if data.get("type") == "pause":
                await emitter.emit_async(
                    EventType.STEERING_PAUSE,
                    {"session_id": data.get("session_id")},
                    source="steering_ws",
                )
                await websocket.send_json({"type": "paused"})

            elif data.get("type") == "resume":
                await emitter.emit_async(
                    EventType.STEERING_RESUME,
                    {
                        "session_id": data.get("session_id"),
                        "instruction": data.get("instruction"),
                    },
                    source="steering_ws",
                )
                await websocket.send_json({"type": "resumed"})

            elif data.get("type") == "veto":
                await emitter.emit_async(
                    EventType.STEERING_VETO,
                    {"session_id": data.get("session_id")},
                    source="steering_ws",
                )
                await websocket.send_json({"type": "vetoed"})

        await handle_websocket_with_timeout(websocket, "steering", handler)

    # Root endpoint
    @app.get("/")
    async def root() -> dict[str, Any]:
        """Root endpoint with API info."""
        return {
            "name": title,
            "version": version,
            "docs": "/docs",
            "health": "/api/health",
        }

    # Startup/shutdown endpoints
    @app.get("/ready")
    async def readiness_check() -> dict[str, str]:
        """Kubernetes-style readiness check."""
        return {"status": "ready"}

    @app.get("/live")
    async def liveness_check() -> dict[str, str]:
        """Kubernetes-style liveness check."""
        return {"status": "alive"}

    return app


# Default app instance
app = create_app()
