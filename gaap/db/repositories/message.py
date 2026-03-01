"""Message Repository

Message operations with conversation history and search.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gaap.db.models.message import Message, MessageRole
from gaap.db.repositories.base import BaseRepository, PaginatedResult, PaginationParams


class MessageRepository(BaseRepository[Message]):
    """Repository for message operations.

    Usage:
        repo = MessageRepository(session)
        messages = await repo.get_conversation_history(session_id)
        results = await repo.search(session_id, "error")
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Message)

    async def get_by_session(
        self,
        session_id: str,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[Message]:
        """Get messages for a session."""
        stmt = (
            select(Message).where(Message.session_id == session_id).order_by(asc(Message.sequence))
        )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await self.session.scalar(count_stmt) or 0

        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.per_page)

        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return PaginatedResult(
            items=items,
            total=total,
            page=pagination.page if pagination else 1,
            per_page=pagination.per_page if pagination else total,
        )

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[Message]:
        """Get conversation history (most recent messages)."""
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.sequence))
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        # Return in chronological order
        return list(reversed(result.scalars().all()))

    async def get_recent_messages(
        self,
        session_id: str,
        count: int = 10,
    ) -> list[Message]:
        """Get most recent messages."""
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.created_at))
            .limit(count)
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def search(
        self,
        session_id: str,
        query: str,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[Message]:
        """Search messages by content."""
        stmt = (
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.content.ilike(f"%{query}%"),
            )
            .order_by(desc(Message.created_at))
        )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await self.session.scalar(count_stmt) or 0

        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.per_page)

        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return PaginatedResult(
            items=items,
            total=total,
            page=pagination.page if pagination else 1,
            per_page=pagination.per_page if pagination else total,
        )

    async def get_by_role(
        self,
        session_id: str,
        role: MessageRole,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[Message]:
        """Get messages by role."""
        stmt = (
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.role == role.value,
            )
            .order_by(asc(Message.sequence))
        )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await self.session.scalar(count_stmt) or 0

        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.per_page)

        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return PaginatedResult(
            items=items,
            total=total,
            page=pagination.page if pagination else 1,
            per_page=pagination.per_page if pagination else total,
        )

    async def get_next_sequence(self, session_id: str) -> int:
        """Get next message sequence number for a session."""
        stmt = select(func.max(Message.sequence)).where(Message.session_id == session_id)
        result = await self.session.scalar(stmt)
        return (result or 0) + 1

    async def create_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        **kwargs: Any,
    ) -> Message:
        """Create a message with auto-incremented sequence."""
        sequence = await self.get_next_sequence(session_id)
        return await self.create(
            session_id=session_id,
            role=role.value,
            content=content,
            sequence=sequence,
            **kwargs,
        )

    async def get_token_stats(self, session_id: str) -> dict[str, Any]:
        """Get token usage statistics for a session."""
        stmt = select(
            func.sum(Message.prompt_tokens),
            func.sum(Message.completion_tokens),
            func.sum(Message.total_tokens),
            func.sum(Message.cost_usd),
            func.avg(Message.latency_ms),
        ).where(Message.session_id == session_id)

        result = await self.session.execute(stmt)
        stats = result.one()

        return {
            "prompt_tokens": stats[0] or 0,
            "completion_tokens": stats[1] or 0,
            "total_tokens": stats[2] or 0,
            "total_cost": stats[3] or 0.0,
            "avg_latency_ms": stats[4] or 0.0,
        }

    async def get_message_counts_by_role(self, session_id: str) -> dict[str, int]:
        """Get message counts by role for a session."""
        stmt = (
            select(Message.role, func.count())
            .where(Message.session_id == session_id)
            .group_by(Message.role)
        )

        result = await self.session.execute(stmt)
        return {row[0]: row[1] for row in result.all()}
