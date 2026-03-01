"""Database Configuration

Environment-based database configuration with support for PostgreSQL and SQLite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class DatabaseSettings:
    """Database configuration settings.

    Attributes:
        database_url: Database connection URL
        async_database_url: Async database connection URL
        pool_size: Connection pool size
        max_overflow: Maximum pool overflow
        pool_pre_ping: Enable connection health checks
        pool_recycle: Connection recycle time in seconds
        echo: Enable SQL logging
        connect_args: Additional connection arguments
    """

    database_url: str
    async_database_url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    echo: bool = False
    connect_args: dict | None = None


def _get_database_url() -> tuple[str, str]:
    """Get database URL from environment or use defaults.

    Supports PostgreSQL and SQLite. For development, SQLite is used
    as fallback if no DATABASE_URL is set.

    Returns:
        Tuple of (sync_url, async_url)
    """
    # Check for explicit database URL
    database_url = os.getenv("DATABASE_URL", "").strip()

    if database_url:
        # Convert to async URL if needed
        if database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            async_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("sqlite://"):
            async_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        else:
            async_url = database_url
        return database_url, async_url

    # Check for PostgreSQL-specific environment variables
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_user = os.getenv("POSTGRES_USER", "gaap")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "gaap")
    pg_db = os.getenv("POSTGRES_DB", "gaap")

    # Use PostgreSQL if configured
    if os.getenv("USE_POSTGRES", "false").lower() == "true":
        sync_url = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        async_url = f"postgresql+asyncpg://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        return sync_url, async_url

    # Default to SQLite for development
    db_path = os.getenv("GAAP_DB_PATH", ".gaap/gaap.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    sync_url = f"sqlite:///{db_path}"
    async_url = f"sqlite+aiosqlite:///{db_path}"

    return sync_url, async_url


def get_db_settings() -> DatabaseSettings:
    """Get database settings from environment.

    Environment Variables:
        DATABASE_URL: Full database URL
        POSTGRES_HOST: PostgreSQL host
        POSTGRES_PORT: PostgreSQL port
        POSTGRES_USER: PostgreSQL user
        POSTGRES_PASSWORD: PostgreSQL password
        POSTGRES_DB: PostgreSQL database name
        USE_POSTGRES: Set to 'true' to use PostgreSQL
        GAAP_DB_PATH: SQLite database path
        DB_POOL_SIZE: Connection pool size (default: 10)
        DB_MAX_OVERFLOW: Max pool overflow (default: 20)
        DB_POOL_PRE_PING: Enable health checks (default: true)
        DB_POOL_RECYCLE: Connection recycle seconds (default: 3600)
        DB_ECHO: Enable SQL logging (default: false)

    Returns:
        DatabaseSettings instance
    """
    sync_url, async_url = _get_database_url()

    # Determine if using SQLite
    is_sqlite = "sqlite" in sync_url.lower()

    # SQLite-specific connect args
    connect_args = {}
    if is_sqlite:
        connect_args = {
            "check_same_thread": False,
        }

    return DatabaseSettings(
        database_url=sync_url,
        async_database_url=async_url,
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
        pool_pre_ping=os.getenv("DB_POOL_PRE_PING", "true").lower() == "true",
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
        echo=os.getenv("DB_ECHO", "false").lower() == "true",
        connect_args=connect_args if is_sqlite else None,
    )


# Default settings instance
default_settings: Final[DatabaseSettings] = get_db_settings()

__all__ = [
    "DatabaseSettings",
    "get_db_settings",
    "default_settings",
]
