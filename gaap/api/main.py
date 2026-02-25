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

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from gaap.core.events import EventEmitter, EventType
from gaap.api.websocket import manager as ws_manager

logger = logging.getLogger("gaap.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("GAAP API starting up...")

    # Initialize event emitter
    emitter = EventEmitter.get_instance()

    # Subscribe WebSocket manager to all events
    async def broadcast_event(event):
        await ws_manager.broadcast_event(event)

    for event_type in EventType:
        emitter.subscribe_async(event_type, broadcast_event)

    yield

    # Cleanup
    logger.info("GAAP API shutting down...")
    await ws_manager.close_all()


def create_app(
    title: str = "GAAP API",
    version: str = "1.0.0",
    cors_origins: list[str] = None,
) -> FastAPI:
    """Create and configure FastAPI application."""

    if cors_origins is None:
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

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from .config import router as config_router
    from .providers import router as providers_router
    from .research import router as research_router
    from .healing import router as healing_router
    from .memory import router as memory_router
    from .budget import router as budget_router
    from .sessions import router as sessions_router
    from .system import router as system_router

    app.include_router(config_router)
    app.include_router(providers_router)
    app.include_router(research_router)
    app.include_router(healing_router)
    app.include_router(memory_router)
    app.include_router(budget_router)
    app.include_router(sessions_router)
    app.include_router(system_router)

    # WebSocket endpoints
    @app.websocket("/ws/events")
    async def events_websocket(websocket: WebSocket):
        """Real-time events stream."""
        await ws_manager.connect(websocket, "events")
        try:
            while True:
                data = await websocket.receive_json()
                # Handle client messages (e.g., subscribe to specific events)
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            ws_manager.disconnect(websocket)

    @app.websocket("/ws/ooda")
    async def ooda_websocket(websocket: WebSocket):
        """OODA loop visualization stream."""
        await ws_manager.connect(websocket, "ooda")
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            ws_manager.disconnect(websocket)

    @app.websocket("/ws/steering")
    async def steering_websocket(websocket: WebSocket):
        """Steering commands stream."""
        await ws_manager.connect(websocket, "steering")
        try:
            while True:
                data = await websocket.receive_json()

                # Handle steering commands
                if data.get("type") == "pause":
                    emitter = EventEmitter.get_instance()
                    await emitter.emit_async(
                        EventType.STEERING_PAUSE,
                        {"session_id": data.get("session_id")},
                        source="steering_ws",
                    )
                    await websocket.send_json({"type": "paused"})

                elif data.get("type") == "resume":
                    emitter = EventEmitter.get_instance()
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
                    emitter = EventEmitter.get_instance()
                    await emitter.emit_async(
                        EventType.STEERING_VETO,
                        {"session_id": data.get("session_id")},
                        source="steering_ws",
                    )
                    await websocket.send_json({"type": "vetoed"})

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            ws_manager.disconnect(websocket)

    # Health check
    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "connections": ws_manager.connection_count,
        }

    # Root endpoint
    @app.get("/")
    async def root() -> dict[str, Any]:
        """Root endpoint with API info."""
        return {
            "name": title,
            "version": version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Default app instance
app = create_app()
