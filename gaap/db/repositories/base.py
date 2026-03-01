"""Base Repository

Generic repository with CRUD operations and pagination.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gaap.db import Base

ModelType = TypeVar("ModelType", bound=Base)


class PaginationParams:
    """Pagination parameters.

    Attributes:
        page: Page number (1-based)
        per_page: Items per page
    """

    def __init__(self, page: int = 1, per_page: int = 50) -> None:
        self.page = max(1, page)
        self.per_page = min(max(1, per_page), 100)  # Max 100 per page

    @property
    def offset(self) -> int:
        """Calculate offset for query."""
        return (self.page - 1) * self.per_page


class PaginatedResult(Generic[ModelType]):
    """Paginated query result.

    Attributes:
        items: List of items
        total: Total number of items
        page: Current page
        per_page: Items per page
        pages: Total number of pages
    """

    def __init__(
        self,
        items: list[ModelType],
        total: int,
        page: int,
        per_page: int,
    ) -> None:
        self.items = items
        self.total = total
        self.page = page
        self.per_page = per_page
        self.pages = (total + per_page - 1) // per_page if per_page > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "per_page": self.per_page,
            "pages": self.pages,
        }


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations.

    Usage:
        class UserRepository(BaseRepository[User]):
            def __init__(self, session: AsyncSession):
                super().__init__(session, User)

        repo = UserRepository(session)
        user = await repo.get(user_id)
    """

    def __init__(
        self,
        session: AsyncSession,
        model: type[ModelType],
    ) -> None:
        self.session = session
        self.model = model

    async def get(self, id: str) -> ModelType | None:
        """Get entity by ID."""
        return await self.session.get(self.model, id)

    async def get_by_ids(self, ids: list[str]) -> list[ModelType]:
        """Get multiple entities by IDs."""
        if not ids:
            return []
        stmt = select(self.model).where(self.model.id.in_(ids))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_all(
        self,
        pagination: PaginationParams | None = None,
    ) -> PaginatedResult[ModelType]:
        """Get all entities with optional pagination."""
        # Get total count
        count_stmt = select(func.count()).select_from(self.model)
        total = await self.session.scalar(count_stmt) or 0

        # Get items
        stmt = select(self.model)
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

    async def create(self, **kwargs: Any) -> ModelType:
        """Create new entity."""
        entity = self.model(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def create_many(self, data_list: list[dict[str, Any]]) -> list[ModelType]:
        """Create multiple entities."""
        entities = [self.model(**data) for data in data_list]
        self.session.add_all(entities)
        await self.session.flush()
        return entities

    async def update(self, id: str, **kwargs: Any) -> ModelType | None:
        """Update entity by ID."""
        entity = await self.get(id)
        if entity is None:
            return None

        for key, value in kwargs.items():
            if hasattr(entity, key):
                setattr(entity, key, value)

        await self.session.flush()
        return entity

    async def update_many(
        self,
        ids: list[str],
        **kwargs: Any,
    ) -> list[ModelType]:
        """Update multiple entities."""
        entities = []
        for entity_id in ids:
            entity = await self.update(entity_id, **kwargs)
            if entity:
                entities.append(entity)
        return entities

    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        entity = await self.get(id)
        if entity is None:
            return False

        await self.session.delete(entity)
        await self.session.flush()
        return True

    async def delete_many(self, ids: list[str]) -> int:
        """Delete multiple entities."""
        count = 0
        for entity_id in ids:
            if await self.delete(entity_id):
                count += 1
        return count

    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        stmt = select(func.count()).where(self.model.id == id)
        result = await self.session.scalar(stmt)
        return bool(result and result > 0)

    async def count(self) -> int:
        """Get total count of entities."""
        stmt = select(func.count()).select_from(self.model)
        return await self.session.scalar(stmt) or 0

    async def find_one(self, **filters: Any) -> ModelType | None:
        """Find one entity by filters."""
        stmt = select(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                stmt = stmt.where(getattr(self.model, key) == value)

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_many(
        self,
        pagination: PaginationParams | None = None,
        **filters: Any,
    ) -> PaginatedResult[ModelType]:
        """Find entities by filters with pagination."""
        stmt = select(self.model)

        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model, key):
                stmt = stmt.where(getattr(self.model, key) == value)

        # Get total count
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
