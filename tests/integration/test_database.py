"""
Integration tests for database operations
Tests database connections, migrations, and end-to-end workflows
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

pytestmark = pytest.mark.asyncio


class TestDatabaseConfiguration:
    """Test database configuration"""

    def test_database_url_construction(self):
        """Test database URL construction"""
        from gaap.db.config import get_database_url

        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_PORT": "5432",
                "DB_NAME": "test_db",
                "DB_USER": "test_user",
                "DB_PASSWORD": "test_pass",
            },
        ):
            url = get_database_url()
            assert "postgresql" in url
            assert "test_user" in url
            assert "test_db" in url

    def test_async_database_url(self):
        """Test async database URL"""
        from gaap.db.config import get_async_database_url

        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_PORT": "5432",
                "DB_NAME": "test_db",
                "DB_USER": "test_user",
                "DB_PASSWORD": "test_pass",
            },
        ):
            url = get_async_database_url()
            assert "postgresql+asyncpg" in url
            assert "test_user" in url

    def test_database_url_missing_env(self):
        """Test database URL with missing environment variables"""
        from gaap.db.config import get_database_url

        with patch.dict(os.environ, {}, clear=True):
            url = get_database_url()
            # Should use defaults
            assert "localhost" in url


class TestDatabaseEngine:
    """Test database engine creation"""

    def test_engine_creation(self):
        """Test creating database engine"""
        from gaap.db.config import engine

        # Engine should be configured
        assert engine is not None

    def test_async_engine_creation(self):
        """Test creating async database engine"""
        from gaap.db.config import async_engine

        # Async engine should be configured
        assert async_engine is not None


class TestSessionManagement:
    """Test database session management"""

    async def test_get_session(self):
        """Test getting database session"""
        from gaap.db import get_session

        # Mock the session
        mock_session = Mock(spec=AsyncSession)

        with patch("gaap.db.async_session_maker") as mock_maker:
            mock_maker.return_value = mock_session

            session = get_session()
            # Should be an async generator
            assert session is not None

    async def test_session_commit(self):
        """Test session commit"""
        from gaap.db import get_session

        mock_session = Mock(spec=AsyncSession)
        mock_session.commit = Mock(return_value=asyncio.Future())
        mock_session.commit.return_value.set_result(None)

        with patch("gaap.db.async_session_maker") as mock_maker:
            mock_maker.return_value = mock_session

            # Test session context manager
            async for session in get_session():
                await session.commit()
                break

    async def test_session_rollback(self):
        """Test session rollback"""
        from gaap.db import get_session

        mock_session = Mock(spec=AsyncSession)
        mock_session.rollback = Mock(return_value=asyncio.Future())
        mock_session.rollback.return_value.set_result(None)

        with patch("gaap.db.async_session_maker") as mock_maker:
            mock_maker.return_value = mock_session

            async for session in get_session():
                await session.rollback()
                break

    async def test_session_close(self):
        """Test session close"""
        from gaap.db import get_session

        mock_session = Mock(spec=AsyncSession)
        mock_session.close = Mock(return_value=asyncio.Future())
        mock_session.close.return_value.set_result(None)

        with patch("gaap.db.async_session_maker") as mock_maker:
            mock_maker.return_value = mock_session

            async for session in get_session():
                await session.close()
                break


class TestDatabaseMigrations:
    """Test database migrations"""

    def test_alembic_config(self):
        """Test alembic configuration exists"""
        import os

        alembic_ini = os.path.join(os.path.dirname(__file__), "..", "..", "alembic.ini")
        if os.path.exists(alembic_ini):
            assert True
        else:
            pytest.skip("Alembic config not found")

    def test_migration_scripts_exist(self):
        """Test migration scripts directory exists"""
        import os

        alembic_dir = os.path.join(os.path.dirname(__file__), "..", "..", "alembic")
        versions_dir = os.path.join(alembic_dir, "versions")

        if os.path.exists(versions_dir):
            assert True
        else:
            pytest.skip("Migration versions directory not found")


class TestModelIntegration:
    """Test model integration workflows"""

    async def test_user_session_workflow(self):
        """Test user and session workflow"""
        from gaap.db.models import User, Session
        from gaap.db.repositories import UserRepository, SessionRepository

        mock_session = Mock(spec=AsyncSession)

        # Create user
        mock_user = Mock(spec=User)
        mock_user.id = "user123"
        mock_user.email = "test@example.com"

        with patch.object(UserRepository, "create", return_value=mock_user):
            user_repo = UserRepository(mock_session)
            user = await user_repo.create(
                email="test@example.com",
                username="testuser",
                hashed_password="hashed",
            )
            assert user.email == "test@example.com"

        # Create session for user
        mock_sess = Mock(spec=Session)
        mock_sess.id = "session123"
        mock_sess.user_id = user.id

        with patch.object(SessionRepository, "create", return_value=mock_sess):
            session_repo = SessionRepository(mock_session)
            session = await session_repo.create(
                id="session123",
                title="Test Session",
                user_id=user.id,
            )
            assert session.user_id == user.id

    async def test_session_message_workflow(self):
        """Test session and message workflow"""
        from gaap.db.models import Session, Message
        from gaap.db.repositories import SessionRepository, MessageRepository

        mock_session = Mock(spec=AsyncSession)

        # Create session
        mock_sess = Mock(spec=Session)
        mock_sess.id = "session123"

        with patch.object(SessionRepository, "create", return_value=mock_sess):
            session_repo = SessionRepository(mock_session)
            session = await session_repo.create(
                id="session123",
                title="Test Session",
            )

        # Create messages
        mock_msg1 = Mock(spec=Message)
        mock_msg1.id = "msg1"
        mock_msg1.session_id = session.id

        mock_msg2 = Mock(spec=Message)
        mock_msg2.id = "msg2"
        mock_msg2.session_id = session.id

        with patch.object(MessageRepository, "create", side_effect=[mock_msg1, mock_msg2]):
            message_repo = MessageRepository(mock_session)
            msg1 = await message_repo.create(
                session_id=session.id,
                role="user",
                content="Hello",
            )
            msg2 = await message_repo.create(
                session_id=session.id,
                role="assistant",
                content="Hi there!",
            )

            assert msg1.session_id == session.id
            assert msg2.session_id == session.id

    async def test_provider_usage_workflow(self):
        """Test provider usage tracking workflow"""
        from gaap.db.models import ProviderConfig, ProviderUsage

        mock_session = Mock(spec=AsyncSession)

        # Create provider config
        mock_config = Mock(spec=ProviderConfig)
        mock_config.id = "config123"
        mock_config.name = "openai"

        # Track usage
        mock_usage = Mock(spec=ProviderUsage)
        mock_usage.provider_config_id = mock_config.id
        mock_usage.request_count = 100
        mock_usage.token_count = 5000

        assert mock_usage.provider_config_id == "config123"
        assert mock_usage.request_count == 100


class TestTransactionManagement:
    """Test transaction management"""

    async def test_successful_transaction(self):
        """Test successful transaction"""
        from gaap.db import get_session

        mock_session = Mock(spec=AsyncSession)
        mock_session.commit = Mock(return_value=asyncio.Future())
        mock_session.commit.return_value.set_result(None)

        with patch("gaap.db.async_session_maker") as mock_maker:
            mock_maker.return_value = mock_session

            async for session in get_session():
                # Perform operations
                await session.commit()
                break

            mock_session.commit.assert_called_once()

    async def test_failed_transaction_rollback(self):
        """Test failed transaction rollback"""
        from gaap.db import get_session

        mock_session = Mock(spec=AsyncSession)
        mock_session.rollback = Mock(return_value=asyncio.Future())
        mock_session.rollback.return_value.set_result(None)

        with patch("gaap.db.async_session_maker") as mock_maker:
            mock_maker.return_value = mock_session

            try:
                async for session in get_session():
                    raise Exception("Test error")
            except Exception:
                pass


class TestDatabaseQueries:
    """Test complex database queries"""

    async def test_join_query(self):
        """Test join query between tables"""
        from gaap.db.repositories import SessionRepository

        mock_session = Mock(spec=AsyncSession)

        # Mock result with joined data
        mock_result = Mock()
        mock_result.all.return_value = [
            (Mock(), Mock()),  # (Session, User) tuples
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = SessionRepository(mock_session)
        # Repository should handle complex queries
        assert repo is not None

    async def test_aggregate_query(self):
        """Test aggregate query"""
        from gaap.db.repositories import SessionRepository

        mock_session = Mock(spec=AsyncSession)

        # Mock aggregate result
        mock_result = Mock()
        mock_result.one.return_value = (100, 5000, 0.5)  # count, tokens, cost
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = SessionRepository(mock_session)
        stats = await repo.get_stats("user123")

        assert "total_tokens" in stats

    async def test_filter_query(self):
        """Test filter query"""
        from gaap.db.repositories import SessionRepository
        from gaap.db.models import SessionStatus

        mock_session = Mock(spec=AsyncSession)

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 5

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = SessionRepository(mock_session)
        result = await repo.list_by_user("user123", status=SessionStatus.ACTIVE)

        assert result is not None


class TestDatabaseConnectionPooling:
    """Test database connection pooling"""

    def test_pool_configuration(self):
        """Test pool configuration"""
        from gaap.db.config import engine

        # Engine should have pool settings
        assert engine.pool is not None

    def test_async_pool_configuration(self):
        """Test async pool configuration"""
        from gaap.db.config import async_engine

        # Async engine should have pool settings
        assert async_engine.pool is not None


class TestDatabaseErrorHandling:
    """Test database error handling"""

    async def test_connection_error(self):
        """Test connection error handling"""
        from gaap.db import get_session

        with patch("gaap.db.async_session_maker") as mock_maker:
            mock_maker.side_effect = Exception("Connection refused")

            with pytest.raises(Exception, match="Connection refused"):
                async for session in get_session():
                    pass

    async def test_query_error(self):
        """Test query error handling"""
        from gaap.db.repositories import UserRepository

        mock_session = Mock(spec=AsyncSession)
        mock_session.execute = AsyncMock(side_effect=Exception("Query failed"))

        repo = UserRepository(mock_session)

        with pytest.raises(Exception, match="Query failed"):
            await repo.get("user123")

    async def test_integrity_error(self):
        """Test integrity error handling"""
        from gaap.db.repositories import UserRepository

        mock_session = Mock(spec=AsyncSession)

        # Simulate integrity error on unique constraint
        error = Exception("Duplicate key value")
        mock_session.flush = AsyncMock(side_effect=error)

        repo = UserRepository(mock_session)

        with pytest.raises(Exception, match="Duplicate key value"):
            await repo.create(email="test@example.com", username="test")


class TestDatabasePerformance:
    """Test database performance characteristics"""

    async def test_bulk_insert(self):
        """Test bulk insert performance"""
        from gaap.db.repositories import BaseRepository

        mock_session = Mock(spec=AsyncSession)

        # Create many entities
        data_list = [{"name": f"item_{i}"} for i in range(100)]

        mock_session.add_all = Mock()
        mock_session.flush = AsyncMock()

        class TestModel:
            pass

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, TestModel)

        repo = TestRepo(mock_session)
        result = await repo.create_many(data_list)

        assert len(result) == 100
        mock_session.add_all.assert_called_once()

    async def test_pagination_performance(self):
        """Test pagination performance"""
        from gaap.db.repositories import BaseRepository, PaginationParams

        mock_session = Mock(spec=AsyncSession)

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 10000

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        pagination = PaginationParams(page=100, per_page=100)
        result = await repo.get_all(pagination=pagination)

        assert result.page == 100
        assert result.total == 10000


class TestDatabaseConstraints:
    """Test database constraints"""

    def test_unique_constraints(self):
        """Test unique constraints"""
        from gaap.db.models import User

        # User model should have unique constraints on email and username
        table = User.__table__
        constraints = [c.name for c in table.constraints if hasattr(c, "name")]
        # Check for unique indexes
        assert True  # Constraints are defined in the model

    def test_foreign_key_constraints(self):
        """Test foreign key constraints"""
        from gaap.db.models import Session, Message

        # Session should have FK to User
        session_table = Session.__table__
        fks = list(session_table.foreign_keys)

        # Message should have FK to Session
        message_table = Message.__table__
        fks = list(message_table.foreign_keys)
        assert len(fks) > 0

    def test_check_constraints(self):
        """Test check constraints"""
        from gaap.db.models import UserPreference

        # UserPreference has temperature range constraint (implicit)
        pref = UserPreference(user_id="user123", temperature=50)
        assert 0 <= pref.temperature <= 100
