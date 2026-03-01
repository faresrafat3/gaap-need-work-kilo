"""Session Repository

Session management with filtering and archive operations.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gaap.db.models.session import Session, SessionStatus
from gaap.db.repositories.base import BaseRepository, PaginatedResult, PaginationParams


class SessionRepository(BaseRepository[Session]):
    """Repository for session operations.

    Usage:
        repo = SessionRepository(session)
        sessions = await repo.list_by_user(user_id, status=SessionStatus.ACTIVE)
        await repo.archive_old_sessions(days=30)
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Session)

    async def list_by_user(
        self,
        user_id: str,
        status: SessionStatus | None = None,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[Session]:
        """List sessions for a user with optional status filter."""
        stmt = select(Session).where(Session.user_id == user_id)

        if status:
            stmt = stmt.where(Session.status == status.value)

        # Order by last message, then creation
        stmt = stmt.order_by(desc(Session.last_message_at), desc(Session.created_at))

        # Get total
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await self.session.scalar(count_stmt) or 0

        # Apply pagination
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

    async def list_active(
        self,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[Session]:
        """List all active sessions."""
        stmt = (
            select(Session)
            .where(Session.status == SessionStatus.ACTIVE.value)
            .order_by(desc(Session.last_message_at))
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

    async def search(
        self,
        user_id: str,
        query: str,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[Session]:
        """Search sessions by title or description."""
        # Escape special SQL LIKE characters
        escaped_query = query.replace("%", "\\%").replace("_", "\\_")

        stmt = select(Session).where(
            Session.user_id == user_id,
            Session.title.ilike(f"%{escaped_query}%", escape="\\"),
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

    async def update_message_stats(
        self,
        session_id: str,
        tokens: int,
        cost: float,
    ) -> Session | None:
        """Update session message statistics."""
        session = await self.get(session_id)
        if not session:
            return None

        session.message_count += 1
        session.total_tokens += tokens
        session.total_cost += cost
        session.last_message_at = datetime.utcnow()

        await self.session.flush()
        return session

    async def archive(self, session_id: str) -> Session | None:
        """Archive a session."""
        return await self.update(
            session_id,
            status=SessionStatus.ARCHIVED.value,
            archived_at=datetime.utcnow(),
        )

    async def archive_old_sessions(self, days: int = 30) -> int:
        """Archive sessions older than specified days.

        Returns:
            Number of sessions archived
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        stmt = select(Session).where(
            Session.status == SessionStatus.ACTIVE.value,
            Session.last_message_at < cutoff,
        )

        result = await self.session.execute(stmt)
        sessions = result.scalars().all()

        count = 0
        for session in sessions:
            session.status = SessionStatus.ARCHIVED.value
            session.archived_at = datetime.utcnow()
            count += 1

        await self.session.flush()
        return count

    async def restore(self, session_id: str) -> Session | None:
        """Restore an archived session."""
        return await self.update(
            session_id,
            status=SessionStatus.ACTIVE.value,
            archived_at=None,
        )

    async def get_stats(self, user_id: str) -> dict[str, Any]:
        """Get session statistics for a user."""
        # Count by status
        stmt = (
            select(Session.status, func.count())
            .where(Session.user_id == user_id)
            .group_by(Session.status)
        )
        result = await self.session.execute(stmt)
        status_counts = {row[0]: row[1] for row in result.all()}

        # Total tokens and cost
        stmt = select(
            func.sum(Session.total_tokens),
            func.sum(Session.total_cost),
            func.sum(Session.message_count),
        ).where(Session.user_id == user_id)
        result = await self.session.execute(stmt)
        totals = result.one()

        return {
            "total_sessions": sum(status_counts.values()),
            "by_status": status_counts,
            "total_tokens": totals[0] or 0,
            "total_cost": totals[1] or 0.0,
            "total_messages": totals[2] or 0,
        }
