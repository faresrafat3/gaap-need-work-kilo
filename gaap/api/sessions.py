"""
Sessions API - Session Management Endpoints
============================================

Provides REST API for managing chat sessions using PostgreSQL database.

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
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gaap.core.events import EventEmitter, EventType
from gaap.db import get_session as get_db_session
from gaap.db.models import Session as SessionModel
from gaap.db.models import SessionPriority, SessionStatus
from gaap.db.models.message import Message
from gaap.db.repositories import MessageRepository, PaginationParams, SessionRepository

logger = logging.getLogger("gaap.api.sessions")

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionPriorityEnum(str, Enum):
    """Session priority (matches database enum)."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    priority: SessionPriorityEnum = SessionPriorityEnum.NORMAL
    tags: list[str] = []
    config: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class SessionUpdateRequest(BaseModel):
    """Request to update a session."""

    name: str | None = None
    description: str | None = None
    priority: SessionPriorityEnum | None = None
    tags: list[str] | None = None
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class MessageResponse(BaseModel):
    """Message in session export."""

    id: str
    role: str
    content: str
    created_at: str
    tokens: int = 0
    provider: str | None = None
    model: str | None = None


class SessionResponse(BaseModel):
    """Session response."""

    id: str
    name: str
    description: str | None
    status: str
    priority: str
    tags: list[str]
    config: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    last_message_at: str | None
    message_count: int
    total_tokens: int
    total_cost: float


class SessionListResponse(BaseModel):
    """List of sessions."""

    sessions: list[SessionResponse]
    total: int
    page: int = 1
    per_page: int = 50


class SessionExportResponse(BaseModel):
    """Export session data."""

    session: SessionResponse
    messages: list[MessageResponse]
    stats: dict[str, Any]


def _session_to_response(session: SessionModel | None) -> SessionResponse:
    """Convert database session to response model."""
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        id=session.id,
        name=session.title or "Untitled",
        description=session.description,
        status=session.status,
        priority=session.priority,
        tags=session.tags or [],
        config=session.config or {},
        metadata=session.metadata or {},
        created_at=session.created_at.isoformat() if session.created_at else "",
        updated_at=session.updated_at.isoformat() if session.updated_at else "",
        last_message_at=session.last_message_at.isoformat() if session.last_message_at else None,
        message_count=session.message_count or 0,
        total_tokens=session.total_tokens or 0,
        total_cost=session.total_cost or 0.0,
    )


def _message_to_response(msg: Message) -> MessageResponse:
    """Convert database message to response model."""
    return MessageResponse(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at.isoformat() if msg.created_at else "",
        tokens=msg.total_tokens or 0,
        provider=msg.provider,
        model=msg.model,
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    status: SessionStatus | None = None,
    priority: SessionPriorityEnum | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session),
) -> SessionListResponse:
    """List all sessions with optional filtering."""
    try:
        repo = SessionRepository(db)
        pagination = PaginationParams(page=(offset // limit) + 1, per_page=limit)

        # Use database status directly
        db_status = status

        result = await repo.find_many(
            pagination=pagination,
            **({"status": db_status.value} if db_status else {}),
        )

        sessions = [_session_to_response(s) for s in result.items]

        return SessionListResponse(
            sessions=sessions,
            total=result.total,
            page=result.page,
            per_page=result.per_page,
        )
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {e}")


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    request: SessionCreateRequest,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Create a new session."""
    try:
        repo = SessionRepository(db)

        session_id = uuid.uuid4().hex[:12]
        now = datetime.now()

        # Map priority
        priority_map = {
            SessionPriorityEnum.LOW: SessionPriority.LOW,
            SessionPriorityEnum.NORMAL: SessionPriority.NORMAL,
            SessionPriorityEnum.HIGH: SessionPriority.HIGH,
            SessionPriorityEnum.CRITICAL: SessionPriority.CRITICAL,
        }
        db_priority = priority_map.get(request.priority, SessionPriority.NORMAL)

        session = await repo.create(
            id=session_id,
            title=request.name,
            description=request.description or None,
            status=SessionStatus.ACTIVE.value,
            priority=db_priority.value,
            tags=request.tags,
            config=request.config,
            metadata=request.metadata,
            created_at=now,
            updated_at=now,
        )

        await db.commit()

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_CREATED,
            {"session_id": session_id, "name": request.name},
            source="sessions_api",
        )

        logger.info(f"Created session {session_id}: {request.name}")

        return _session_to_response(session)
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_detail(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Get session details."""
    try:
        repo = SessionRepository(db)
        session = await repo.get(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return _session_to_response(session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {e}")


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Update session."""
    try:
        repo = SessionRepository(db)
        session = await repo.get(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        update_data: dict[str, Any] = {"updated_at": datetime.now()}

        if request.name is not None:
            update_data["title"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.priority is not None:
            priority_map = {
                SessionPriorityEnum.LOW: SessionPriority.LOW.value,
                SessionPriorityEnum.NORMAL: SessionPriority.NORMAL.value,
                SessionPriorityEnum.HIGH: SessionPriority.HIGH.value,
                SessionPriorityEnum.CRITICAL: SessionPriority.CRITICAL.value,
            }
            update_data["priority"] = priority_map.get(
                request.priority, SessionPriority.NORMAL.value
            )
        if request.tags is not None:
            update_data["tags"] = request.tags
        if request.config is not None:
            update_data["config"] = request.config
        if request.metadata is not None:
            update_data["metadata"] = request.metadata

        updated = await repo.update(session_id, **update_data)
        await db.commit()

        if not updated:
            raise HTTPException(status_code=500, detail="Failed to update session")

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_UPDATE,
            {"session_id": session_id, "changes": request.model_dump(exclude_none=True)},
            source="sessions_api",
        )

        logger.info(f"Updated session {session_id}")

        return _session_to_response(updated)
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update session: {e}")


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Delete a session."""
    try:
        repo = SessionRepository(db)
        session = await repo.get(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Soft delete - mark as deleted status
        await repo.update(session_id, status=SessionStatus.ARCHIVED.value)
        await db.commit()

        logger.info(f"Deleted (archived) session {session_id}")

        return {"success": True, "message": f"Session {session_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {e}")


@router.post("/{session_id}/pause", response_model=SessionResponse)
async def pause_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Pause a running session."""
    try:
        repo = SessionRepository(db)
        session = await repo.get(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        if session.status != SessionStatus.ACTIVE.value:
            raise HTTPException(status_code=400, detail="Can only pause active sessions")

        updated = await repo.update(
            session_id,
            status=SessionStatus.PAUSED.value,
            updated_at=datetime.now(),
        )
        await db.commit()

        if not updated:
            raise HTTPException(status_code=500, detail="Failed to pause session")

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_PAUSED,
            {"session_id": session_id},
            source="sessions_api",
        )

        logger.info(f"Paused session {session_id}")

        return _session_to_response(updated)
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to pause session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause session: {e}")


@router.post("/{session_id}/resume", response_model=SessionResponse)
async def resume_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Resume a paused session."""
    try:
        repo = SessionRepository(db)
        session = await repo.get(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        if session.status != SessionStatus.PAUSED.value:
            raise HTTPException(status_code=400, detail="Can only resume paused sessions")

        updated = await repo.update(
            session_id,
            status=SessionStatus.ACTIVE.value,
            updated_at=datetime.now(),
        )
        await db.commit()

        if not updated:
            raise HTTPException(status_code=500, detail="Failed to resume session")

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_RESUMED,
            {"session_id": session_id},
            source="sessions_api",
        )

        logger.info(f"Resumed session {session_id}")

        return _session_to_response(updated)
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to resume session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume session: {e}")


@router.post("/{session_id}/export", response_model=SessionExportResponse)
async def export_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> SessionExportResponse:
    """Export session data including messages."""
    try:
        session_repo = SessionRepository(db)
        message_repo = MessageRepository(db)

        session = await session_repo.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Get messages
        messages_result = await message_repo.get_by_session(session_id)
        messages = [_message_to_response(m) for m in messages_result.items]

        # Calculate stats
        stats = await message_repo.get_token_stats(session_id)

        logger.info(f"Exported session {session_id}")

        return SessionExportResponse(
            session=_session_to_response(session),
            messages=messages,
            stats=stats,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export session: {e}")


@router.post("/{session_id}/cancel", response_model=SessionResponse)
async def cancel_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Cancel a session."""
    try:
        repo = SessionRepository(db)
        session = await repo.get(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        if session.status == SessionStatus.ARCHIVED.value:
            raise HTTPException(status_code=400, detail="Cannot cancel archived sessions")

        updated = await repo.update(
            session_id,
            status=SessionStatus.ARCHIVED.value,
            updated_at=datetime.now(),
        )
        await db.commit()

        if not updated:
            raise HTTPException(status_code=500, detail="Failed to cancel session")

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SESSION_COMPLETED,
            {"session_id": session_id, "status": "cancelled"},
            source="sessions_api",
        )

        logger.info(f"Cancelled session {session_id}")

        return _session_to_response(updated)
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to cancel session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel session: {e}")


def register_routes(app: Any) -> None:
    """Register sessions routes with FastAPI app."""
    app.include_router(router)
