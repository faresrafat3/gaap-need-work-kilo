"""
Sessions API - Session Management Endpoints
===========================================

Provides REST API for managing execution sessions.

Endpoints:
- GET /api/sessions - List all sessions
- POST /api/sessions - Create new session
- GET /api/sessions/{id} - Get session details
- PUT /api/sessions/{id} - Update session
- DELETE /api/sessions/{id} - Delete session
- POST /api/sessions/{id}/pause - Pause session
- POST /api/sessions/{id}/resume - Resume session
- POST /api/sessions/{id}/export - Export session data
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gaap.core.events import EventEmitter, EventType
from gaap.storage.sqlite_store import SQLiteStore, SQLiteConfig

logger = logging.getLogger("gaap.api.sessions")

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_store: SQLiteStore | None = None


class SessionStatus(str, Enum):
    """Session status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionPriority(str, Enum):
    """Session priority."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    priority: SessionPriority = SessionPriority.NORMAL
    tags: list[str] = []
    config: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class SessionUpdateRequest(BaseModel):
    """Request to update a session."""

    name: str | None = None
    description: str | None = None
    priority: SessionPriority | None = None
    tags: list[str] | None = None
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    """Session response."""

    id: str
    name: str
    description: str
    status: SessionStatus
    priority: SessionPriority
    tags: list[str]
    config: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    progress: float = 0.0
    tasks_total: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    cost_usd: float = 0.0
    tokens_used: int = 0


class SessionListResponse(BaseModel):
    """List of sessions."""

    sessions: list[SessionResponse]
    total: int


class SessionExportResponse(BaseModel):
    """Export session data."""

    session: SessionResponse
    tasks: list[dict[str, Any]]
    logs: list[dict[str, Any]]
    metrics: dict[str, Any]


def get_store() -> SQLiteStore:
    """Get or create the store instance."""
    global _store
    if _store is None:
        _store = SQLiteStore(config=SQLiteConfig(db_path=".gaap/gaap.db"))
    return _store


def _session_to_response(data: dict[str, Any]) -> SessionResponse:
    """Convert session data to response."""
    session_data = data.get("data", data)
    return SessionResponse(
        id=data.get("id", session_data.get("id", "")),
        name=session_data.get("name", ""),
        description=session_data.get("description", ""),
        status=SessionStatus(session_data.get("status", "pending")),
        priority=SessionPriority(session_data.get("priority", "normal")),
        tags=session_data.get("tags", []),
        config=session_data.get("config", {}),
        metadata=session_data.get("metadata", {}),
        created_at=data.get("created_at", session_data.get("created_at", "")),
        updated_at=data.get("updated_at"),
        started_at=session_data.get("started_at"),
        completed_at=session_data.get("completed_at"),
        progress=session_data.get("progress", 0.0),
        tasks_total=session_data.get("tasks_total", 0),
        tasks_completed=session_data.get("tasks_completed", 0),
        tasks_failed=session_data.get("tasks_failed", 0),
        cost_usd=session_data.get("cost_usd", 0.0),
        tokens_used=session_data.get("tokens_used", 0),
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    status: SessionStatus | None = None,
    priority: SessionPriority | None = None,
    limit: int = 50,
    offset: int = 0,
) -> SessionListResponse:
    """List all sessions with optional filtering."""
    try:
        store = get_store()

        where = {}
        if status:
            where["status"] = status.value
        if priority:
            where["priority"] = priority.value

        sessions_data = store.query("sessions", where=where, limit=limit, offset=offset)
        total = store.count("sessions", where=where) if where else store.count("sessions")

        sessions = [_session_to_response(s) for s in sessions_data]

        return SessionListResponse(sessions=sessions, total=total)
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(request: SessionCreateRequest) -> SessionResponse:
    """Create a new session."""
    try:
        store = get_store()

        session_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()

        session_data = {
            "name": request.name,
            "description": request.description,
            "status": SessionStatus.PENDING.value,
            "priority": request.priority.value,
            "tags": request.tags,
            "config": request.config,
            "metadata": request.metadata,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "tasks_total": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "cost_usd": 0.0,
            "tokens_used": 0,
        }

        store.insert("sessions", session_data, item_id=session_id)

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_CREATED,
            {"session_id": session_id, "name": request.name},
            source="sessions_api",
        )

        logger.info(f"Created session {session_id}: {request.name}")

        return SessionResponse(
            id=session_id,
            name=request.name,
            description=request.description,
            status=SessionStatus.PENDING,
            priority=request.priority,
            tags=request.tags,
            config=request.config,
            metadata=request.metadata,
            created_at=now,
            progress=0.0,
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """Get session details."""
    try:
        store = get_store()
        data = store.get("sessions", session_id)

        if not data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return _session_to_response(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(session_id: str, request: SessionUpdateRequest) -> SessionResponse:
    """Update session."""
    try:
        store = get_store()
        data = store.get("sessions", session_id)

        if not data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session_data = data.get("data", {})

        if request.name is not None:
            session_data["name"] = request.name
        if request.description is not None:
            session_data["description"] = request.description
        if request.priority is not None:
            session_data["priority"] = request.priority.value
        if request.tags is not None:
            session_data["tags"] = request.tags
        if request.config is not None:
            session_data["config"] = request.config
        if request.metadata is not None:
            session_data["metadata"] = request.metadata

        store.update("sessions", session_id, session_data)

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_UPDATE,
            {"session_id": session_id, "changes": request.model_dump(exclude_none=True)},
            source="sessions_api",
        )

        logger.info(f"Updated session {session_id}")

        data = store.get("sessions", session_id)
        return _session_to_response(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session."""
    try:
        store = get_store()

        if not store.get("sessions", session_id):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        store.delete("sessions", session_id)

        logger.info(f"Deleted session {session_id}")

        return {"success": True, "message": f"Session {session_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/pause", response_model=SessionResponse)
async def pause_session(session_id: str) -> SessionResponse:
    """Pause a running session."""
    try:
        store = get_store()
        data = store.get("sessions", session_id)

        if not data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session_data = data.get("data", {})
        current_status = SessionStatus(session_data.get("status", "pending"))

        if current_status != SessionStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Can only pause running sessions")

        session_data["status"] = SessionStatus.PAUSED.value
        store.update("sessions", session_id, session_data)

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_PAUSED,
            {"session_id": session_id},
            source="sessions_api",
        )

        logger.info(f"Paused session {session_id}")

        data = store.get("sessions", session_id)
        return _session_to_response(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/resume", response_model=SessionResponse)
async def resume_session(session_id: str) -> SessionResponse:
    """Resume a paused session."""
    try:
        store = get_store()
        data = store.get("sessions", session_id)

        if not data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session_data = data.get("data", {})
        current_status = SessionStatus(session_data.get("status", "pending"))

        if current_status != SessionStatus.PAUSED:
            raise HTTPException(status_code=400, detail="Can only resume paused sessions")

        session_data["status"] = SessionStatus.RUNNING.value
        if not session_data.get("started_at"):
            session_data["started_at"] = datetime.now().isoformat()

        store.update("sessions", session_id, session_data)

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_RESUMED,
            {"session_id": session_id},
            source="sessions_api",
        )

        logger.info(f"Resumed session {session_id}")

        data = store.get("sessions", session_id)
        return _session_to_response(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/export", response_model=SessionExportResponse)
async def export_session(session_id: str) -> SessionExportResponse:
    """Export session data including tasks and logs."""
    try:
        store = get_store()
        data = store.get("sessions", session_id)

        if not data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session = _session_to_response(data)

        tasks = store.query("tasks", where={"session_id": session_id}, limit=1000)

        logs = store.query("logs", where={"session_id": session_id}, limit=1000)

        metrics = {
            "total_cost": session.cost_usd,
            "total_tokens": session.tokens_used,
            "tasks_total": session.tasks_total,
            "tasks_completed": session.tasks_completed,
            "tasks_failed": session.tasks_failed,
            "success_rate": (
                session.tasks_completed / session.tasks_total if session.tasks_total > 0 else 0.0
            ),
        }

        logger.info(f"Exported session {session_id}")

        return SessionExportResponse(
            session=session,
            tasks=tasks,
            logs=logs,
            metrics=metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/cancel", response_model=SessionResponse)
async def cancel_session(session_id: str) -> SessionResponse:
    """Cancel a session."""
    try:
        store = get_store()
        data = store.get("sessions", session_id)

        if not data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session_data = data.get("data", {})
        current_status = SessionStatus(session_data.get("status", "pending"))

        if current_status in [SessionStatus.COMPLETED, SessionStatus.CANCELLED]:
            raise HTTPException(
                status_code=400, detail="Cannot cancel completed or already cancelled sessions"
            )

        session_data["status"] = SessionStatus.CANCELLED.value
        session_data["completed_at"] = datetime.now().isoformat()
        store.update("sessions", session_id, session_data)

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_COMPLETED,
            {"session_id": session_id, "status": "cancelled"},
            source="sessions_api",
        )

        logger.info(f"Cancelled session {session_id}")

        data = store.get("sessions", session_id)
        return _session_to_response(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_routes(app: Any) -> None:
    """Register sessions routes with FastAPI app."""
    app.include_router(router)
