"""Tests for database configuration."""

import os
from unittest.mock import patch

import pytest

from gaap.db.config import DatabaseSettings, get_db_settings


class TestDatabaseSettings:
    """Tests for DatabaseSettings."""

    def test_default_sqlite_settings(self):
        """Test default SQLite settings."""
        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            settings = get_db_settings()

        assert "sqlite" in settings.database_url
        assert "sqlite+aiosqlite" in settings.async_database_url
        assert settings.pool_size == 10
        assert settings.max_overflow == 20
        assert settings.pool_pre_ping is True

    def test_postgres_settings_from_env(self):
        """Test PostgreSQL settings from environment."""
        env = {
            "USE_POSTGRES": "true",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_DB": "testdb",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = get_db_settings()

        assert "postgresql://" in settings.database_url
        assert "postgresql+asyncpg://" in settings.async_database_url
        assert "testuser" in settings.database_url
        assert "testpass" in settings.database_url

    def test_database_url_override(self):
        """Test DATABASE_URL environment variable override."""
        env = {
            "DATABASE_URL": "postgresql://user:pass@host:5432/db",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = get_db_settings()

        assert settings.database_url == "postgresql://user:pass@host:5432/db"
        assert "postgresql+asyncpg://" in settings.async_database_url

    def test_custom_pool_settings(self):
        """Test custom pool settings."""
        env = {
            "DB_POOL_SIZE": "20",
            "DB_MAX_OVERFLOW": "30",
            "DB_POOL_PRE_PING": "false",
            "DB_ECHO": "true",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = get_db_settings()

        assert settings.pool_size == 20
        assert settings.max_overflow == 30
        assert settings.pool_pre_ping is False
        assert settings.echo is True


class TestDatabaseSettingsDataclass:
    """Tests for DatabaseSettings dataclass."""

    def test_immutability(self):
        """Test that settings are immutable."""
        settings = DatabaseSettings(
            database_url="sqlite:///test.db",
            async_database_url="sqlite+aiosqlite:///test.db",
        )

        with pytest.raises(AttributeError):
            settings.pool_size = 20
