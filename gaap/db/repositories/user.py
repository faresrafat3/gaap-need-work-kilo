"""User Repository

User-specific repository with authentication methods.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gaap.db.models.user import APIKey, User, UserPreference
from gaap.db.repositories.base import BaseRepository, PaginatedResult, PaginationParams


class UserRepository(BaseRepository[User]):
    """Repository for user operations.

    Usage:
        repo = UserRepository(session)
        user = await repo.get_by_email("user@example.com")
        if user and await repo.verify_password(user, "password"):
            # Authenticated
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, User)

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        stmt = select(User).where(User.email == email.lower())
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username."""
        stmt = select(User).where(User.username == username.lower())
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_api_key_hash(self, key_hash: str) -> User | None:
        """Get user by API key hash."""
        stmt = (
            select(User)
            .join(APIKey, User.id == APIKey.user_id)
            .where(
                APIKey.key_hash == key_hash,
                APIKey.is_active.is_(True),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        username: str,
        hashed_password: str,
        **kwargs: Any,
    ) -> User:
        """Create a new user with preferences."""
        # Create user
        user = await self.create(
            email=email.lower(),
            username=username.lower(),
            hashed_password=hashed_password,
            **kwargs,
        )

        # Create default preferences
        preferences = UserPreference(user_id=user.id)
        self.session.add(preferences)
        await self.session.flush()

        return user

    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        await self.update(user_id, last_login=datetime.utcnow())

    async def activate_user(self, user_id: str) -> User | None:
        """Activate user account."""
        return await self.update(
            user_id,
            is_active=True,
            status="active",
            email_verified=True,
        )

    async def deactivate_user(self, user_id: str) -> User | None:
        """Deactivate user account."""
        return await self.update(user_id, is_active=False, status="inactive")

    async def list_active(
        self,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[User]:
        """List active users."""
        stmt = select(User).where(User.is_active.is_(True))

        # Get total
        from sqlalchemy import func

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await self.session.scalar(count_stmt) or 0

        # Get items
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
        query: str,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[User]:
        """Search users by email or username."""
        from sqlalchemy import or_

        stmt = select(User).where(
            or_(
                User.email.ilike(f"%{query}%"),
                User.username.ilike(f"%{query}%"),
            )
        )

        from sqlalchemy import func

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


class APIKeyRepository(BaseRepository[APIKey]):
    """Repository for API key operations."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, APIKey)

    async def get_by_hash(self, key_hash: str) -> APIKey | None:
        """Get API key by hash."""
        stmt = select(APIKey).where(APIKey.key_hash == key_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: str,
        active_only: bool = True,
    ) -> list[APIKey]:
        """List API keys for a user."""
        stmt = select(APIKey).where(APIKey.user_id == user_id)
        if active_only:
            stmt = stmt.where(APIKey.is_active.is_(True))

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def deactivate(self, key_id: str) -> APIKey | None:
        """Deactivate an API key."""
        return await self.update(key_id, is_active=False)

    async def update_last_used(self, key_id: str) -> None:
        """Update API key last used timestamp."""
        await self.update(key_id, last_used=datetime.utcnow())
