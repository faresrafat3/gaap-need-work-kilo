"""
Chat API - Natural Language Chat Endpoint
=========================================
Uses WebChat Bridge Provider (Kimi, DeepSeek, GLM, Copilot)
Integrates with PostgreSQL database for session and message storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from gaap.core.types import Message as CoreMessage
from gaap.core.types import MessageRole as CoreMessageRole
from gaap.db import get_session as get_db_session
from gaap.db.models import Session as SessionModel
from gaap.db.models import SessionPriority, SessionStatus
from gaap.db.models.message import Message, MessageRole, MessageStatus
from gaap.db.repositories import MessageRepository, SessionRepository
from gaap.providers.webchat_bridge import (
    WebChatBridgeProvider,
    create_deepseek_provider,
    create_glm_provider,
    create_kimi_provider,
)

logger = logging.getLogger("gaap.api.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])

PROVIDERS = {
    "kimi": "kimi-k2.5-thinking",
    "deepseek": "deepseek-chat",
    "glm": "GLM-5",
}

MAX_PROVIDER_NAME_LENGTH = 50
MAX_MESSAGE_LENGTH = 50000
CACHE_TTL_SECONDS = 900


@dataclass
class CachedProvider:
    """Provider cache entry with TTL."""

    provider: WebChatBridgeProvider
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


_provider_cache: dict[str, CachedProvider] = {}
_cleanup_task: asyncio.Task | None = None
_cleanup_task_started: bool = False


async def _start_cache_cleanup_task() -> None:
    """Background task to periodically clean up expired cache entries."""
    global _cleanup_task
    logger.info("Starting provider cache cleanup background task")
    try:
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes (300 seconds)
            try:
                _cleanup_expired_cache()
                logger.debug("Cache cleanup completed")
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
    except asyncio.CancelledError:
        logger.info("Cache cleanup task cancelled")
        raise
    except Exception as e:
        logger.error(f"Cache cleanup task crashed: {e}")
        raise


def _start_cleanup_if_needed() -> None:
    """Start the cleanup task if not already running."""
    global _cleanup_task_started, _cleanup_task
    if not _cleanup_task_started:
        _cleanup_task_started = True
        _cleanup_task = asyncio.create_task(_start_cache_cleanup_task())
        logger.info("Provider cache cleanup task started")


def _cleanup_expired_cache() -> None:
    """Remove expired entries from provider cache."""
    now = time.time()
    expired = [
        name
        for name, cached in _provider_cache.items()
        if now - cached.last_accessed > CACHE_TTL_SECONDS
    ]
    for name in expired:
        del _provider_cache[name]
        logger.debug(f"Removed expired provider from cache: {name}")


def get_provider(provider_name: str = "kimi") -> WebChatBridgeProvider:
    """Get or create a provider instance by name with TTL caching."""
    if not provider_name or len(provider_name) > MAX_PROVIDER_NAME_LENGTH:
        raise ValueError(
            f"Invalid provider name length: {len(provider_name) if provider_name else 0}"
        )

    _start_cleanup_if_needed()

    now = time.time()

    if provider_name not in _provider_cache:
        logger.info(f"Creating new provider instance: {provider_name}")
        if provider_name == "kimi":
            provider = create_kimi_provider(
                model="kimi-k2.5-thinking", account="default", timeout=120
            )
        elif provider_name == "deepseek":
            provider = create_deepseek_provider(account="default", timeout=120)
        elif provider_name == "glm":
            provider = create_glm_provider(account="default", timeout=120)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        _provider_cache[provider_name] = CachedProvider(provider=provider)

    cached = _provider_cache[provider_name]
    cached.last_accessed = now
    cached.access_count += 1

    return cached.provider


def get_fallback_providers(primary: str) -> list[str]:
    """Get fallback providers in order, excluding the primary."""
    all_providers = list(PROVIDERS.keys())
    if primary in all_providers:
        all_providers.remove(primary)
    all_providers.insert(0, primary)
    return all_providers


class ChatRequest(BaseModel):
    """Chat request with strict validation."""

    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )
    context: dict | None = None
    provider: str = Field(default="kimi", max_length=MAX_PROVIDER_NAME_LENGTH)
    store_in_db: bool = Field(default=True, description="Whether to store message in database")

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Sanitize message input."""
        if not v or not isinstance(v, str):
            raise ValueError("Message must be a non-empty string")
        sanitized = "".join(char for char in v if ord(char) >= 32 or char in "\n\r\t")
        return sanitized.strip()

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name against allowed providers."""
        if v not in PROVIDERS:
            logger.warning(f"Unknown provider '{v}', defaulting to 'kimi'")
            return "kimi"
        return v

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: dict | None) -> dict | None:
        """Validate context data."""
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("Context must be a dictionary")
        try:
            context_size = len(json.dumps(v))
            if context_size > 100000:
                raise ValueError("Context too large (max 100KB)")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid context data: {e}")
        return v


class ChatResponse(BaseModel):
    """Chat response."""

    response: str
    session_id: str | None = None
    message_id: str | None = None
    provider_used: str | None = None
    model_used: str | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0


class AuditLogger:
    """Structured audit logging for chat requests."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("gaap.audit.chat")

    @staticmethod
    def _hash_sensitive(data: str) -> str:
        """Hash sensitive data for logging."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def log_request(
        self,
        request: ChatRequest,
        client_ip: str,
        success: bool,
        response_time_ms: float,
        provider_used: str | None = None,
        error: str | None = None,
    ) -> None:
        """Log chat request with hashed sensitive data."""
        message_hash = self._hash_sensitive(request.message)

        log_entry = {
            "event": "chat_request",
            "timestamp": time.time(),
            "client_ip_hash": self._hash_sensitive(client_ip),
            "provider_requested": request.provider,
            "provider_used": provider_used,
            "message_hash": message_hash,
            "message_length": len(request.message),
            "success": success,
            "response_time_ms": round(response_time_ms, 2),
            "error": error,
        }

        if success:
            self.logger.info("Chat request completed", extra=log_entry)
        else:
            self.logger.warning("Chat request failed", extra=log_entry)


_audit_logger = AuditLogger()


async def _store_user_message(
    db: AsyncSession,
    session_id: str,
    content: str,
) -> Message:
    """Store user message in database."""
    message_repo = MessageRepository(db)
    session_repo = SessionRepository(db)

    sequence = await message_repo.get_next_sequence(session_id)

    message = await message_repo.create(
        session_id=session_id,
        role=MessageRole.USER.value,
        content=content,
        sequence=sequence,
        status=MessageStatus.COMPLETED.value,
        prompt_tokens=len(content.split()),  # Approximate
        completion_tokens=0,
        total_tokens=len(content.split()),
    )

    # Update session stats
    await session_repo.update_message_stats(
        session_id,
        tokens=len(content.split()),
        cost=0.0,
    )

    await db.commit()
    return message


async def _store_assistant_message(
    db: AsyncSession,
    session_id: str,
    content: str,
    provider: str,
    model: str,
    latency_ms: float,
    tokens: int,
) -> Message:
    """Store assistant response in database."""
    message_repo = MessageRepository(db)
    session_repo = SessionRepository(db)

    sequence = await message_repo.get_next_sequence(session_id)

    message = await message_repo.create(
        session_id=session_id,
        role=MessageRole.ASSISTANT.value,
        content=content,
        sequence=sequence,
        status=MessageStatus.COMPLETED.value,
        provider=provider,
        model=model,
        prompt_tokens=0,
        completion_tokens=tokens,
        total_tokens=tokens,
        latency_ms=latency_ms,
    )

    # Update session stats
    await session_repo.update_message_stats(
        session_id,
        tokens=tokens,
        cost=0.0,  # Cost calculation would need pricing data
    )

    await db.commit()
    return message


async def _get_or_create_session(
    db: AsyncSession,
    session_id: str | None,
    user_message: str,
) -> tuple[SessionModel, bool]:
    """Get existing session or create new one."""
    session_repo = SessionRepository(db)

    if session_id:
        session = await session_repo.get(session_id)
        if session:
            return session, False

    # Create new session
    new_session_id = (
        session_id
        or f"chat_{int(time.time())}_{hashlib.md5(user_message.encode()).hexdigest()[:8]}"
    )
    session = await session_repo.create(
        id=new_session_id,
        title=user_message[:50] + "..." if len(user_message) > 50 else user_message,
        status=SessionStatus.ACTIVE.value,
        priority=SessionPriority.NORMAL.value,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    await db.commit()
    return session, True


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> ChatResponse:
    """Handle chat messages with provider selection, fallback, and database storage."""
    start_time = time.time()
    client_ip = http_request.client.host if http_request.client else "unknown"

    selected_provider = request.provider if request.provider in PROVIDERS else "kimi"
    fallback_order = get_fallback_providers(selected_provider)

    messages = [CoreMessage(role=CoreMessageRole.USER, content=request.message)]
    last_error = None
    provider_used = None
    session_id = None
    user_message_stored = False

    # Get or create session if storing in DB
    if request.store_in_db:
        try:
            session, is_new = await _get_or_create_session(db, request.session_id, request.message)
            session_id = session.id

            # Store user message
            user_msg = await _store_user_message(db, session_id, request.message)
            user_message_stored = True
            logger.debug(f"Stored user message {user_msg.id} in session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to store user message: {e}")
            await db.rollback()

    for prov_name in fallback_order:
        try:
            provider = get_provider(prov_name)
            model = PROVIDERS[prov_name]

            response = await asyncio.wait_for(
                provider._make_request(messages, model), timeout=120.0
            )

            if isinstance(response, dict) and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                # Try to extract token usage if available
                usage = response.get("usage", {})
                tokens = usage.get("total_tokens", len(content.split()))
            else:
                content = str(response) if response else "No response"
                tokens = len(content.split())

            provider_used = prov_name
            latency_ms = (time.time() - start_time) * 1000

            # Store assistant response if storing in DB
            assistant_message_id = None
            if request.store_in_db and session_id and user_message_stored:
                try:
                    assistant_msg = await _store_assistant_message(
                        db,
                        session_id,
                        content,
                        provider_used,
                        model,
                        latency_ms,
                        tokens,
                    )
                    assistant_message_id = assistant_msg.id
                    logger.debug(f"Stored assistant message {assistant_msg.id}")
                except Exception as e:
                    logger.warning(f"Failed to store assistant message: {e}")
                    await db.rollback()

            # Log successful request
            _audit_logger.log_request(
                request=request,
                client_ip=client_ip,
                success=True,
                response_time_ms=latency_ms,
                provider_used=provider_used,
            )

            return ChatResponse(
                response=content,
                session_id=session_id,
                message_id=assistant_message_id,
                provider_used=provider_used,
                model_used=PROVIDERS.get(provider_used),
                tokens_used=tokens,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Provider {prov_name} timed out, trying next...")
            last_error = "Timeout"
        except Exception as e:
            logger.warning(f"Provider {prov_name} failed: {e}, trying next...")
            last_error = str(e)

    response_time = (time.time() - start_time) * 1000

    _audit_logger.log_request(
        request=request,
        client_ip=client_ip,
        success=False,
        response_time_ms=response_time,
        error=last_error,
    )

    logger.error(f"All providers failed. Last error: {last_error}")
    raise HTTPException(status_code=503, detail=f"All providers failed: {last_error}")


@router.get("/providers", response_model=list[dict])
async def list_providers():
    """Get available providers and their status."""
    now = time.time()
    providers_info = []
    for name in PROVIDERS:
        cached = _provider_cache.get(name)
        status = "available" if cached is not None else "not_initialized"
        age_seconds = None
        if cached:
            age_seconds = round(now - cached.created_at, 2)

        providers_info.append(
            {
                "name": name,
                "model": PROVIDERS[name],
                "status": status,
                "is_default": name == "kimi",
                "cache_age_seconds": age_seconds,
            }
        )
    return providers_info


@router.get("/cache/stats")
async def get_cache_stats() -> dict[str, Any]:
    """Get provider cache statistics."""
    now = time.time()
    stats = {
        "total_cached": len(_provider_cache),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "providers": {},
    }

    for name, cached in _provider_cache.items():
        stats["providers"][name] = {
            "created_at": cached.created_at,
            "last_accessed": cached.last_accessed,
            "access_count": cached.access_count,
            "age_seconds": round(now - cached.created_at, 2),
            "idle_seconds": round(now - cached.last_accessed, 2),
        }

    return stats


@router.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Clear the provider cache (admin only)."""
    global _provider_cache
    count = len(_provider_cache)
    _provider_cache.clear()
    logger.info(f"Provider cache cleared: {count} entries removed")
    return {"status": "cleared", "entries_removed": str(count)}


@router.get("/{session_id}/history")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """Get chat history for a session."""
    try:
        message_repo = MessageRepository(db)
        messages = await message_repo.get_conversation_history(session_id, limit=limit)

        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "tokens": msg.total_tokens,
                "provider": msg.provider,
                "model": msg.model,
            }
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {e}")
