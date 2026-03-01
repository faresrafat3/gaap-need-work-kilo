"""GAAP Database Layer

Complete database layer with async SQLAlchemy support.

Usage:
    from gaap.db import AsyncSessionLocal, engine
    from gaap.db.models import User, Session

    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from gaap.db.config import get_db_settings
from gaap.db.models.base import Base

# Get database settings
settings = get_db_settings()

# Create async engine
engine = create_async_engine(
    settings.async_database_url,
    pool_size=settings.pool_size,
    max_overflow=settings.max_overflow,
    pool_pre_ping=settings.pool_pre_ping,
    pool_recycle=settings.pool_recycle,
    echo=settings.echo,
    future=True,
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session as async context manager.

    Usage:
        async with get_db_session() as session:
            user = await session.get(User, user_id)
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection.

    Usage with FastAPI:
        @router.get("/users/{user_id}")
        async def get_user(user_id: str, db: AsyncSession = Depends(get_session)):
            return await db.get(User, user_id)
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


__all__ = [
    "engine",
    "AsyncSessionLocal",
    "Base",
    "get_db_session",
    "get_session",
    "init_db",
    "drop_db",
    "close_db",
]
