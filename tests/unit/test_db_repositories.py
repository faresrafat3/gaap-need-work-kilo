"""
Comprehensive tests for gaap/db/repositories/ module
Tests BaseRepository, SessionRepository, MessageRepository, and UserRepository
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.asyncio


class TestPaginationParams:
    """Test PaginationParams"""

    def test_default_values(self):
        """Test default pagination values"""
        from gaap.db.repositories.base import PaginationParams

        params = PaginationParams()
        assert params.page == 1
        assert params.per_page == 50

    def test_custom_values(self):
        """Test custom pagination values"""
        from gaap.db.repositories.base import PaginationParams

        params = PaginationParams(page=2, per_page=25)
        assert params.page == 2
        assert params.per_page == 25

    def test_page_minimum(self):
        """Test page minimum is 1"""
        from gaap.db.repositories.base import PaginationParams

        params = PaginationParams(page=0)
        assert params.page == 1

        params2 = PaginationParams(page=-5)
        assert params2.page == 1

    def test_per_page_minimum(self):
        """Test per_page minimum is 1"""
        from gaap.db.repositories.base import PaginationParams

        params = PaginationParams(per_page=0)
        assert params.per_page == 1

        params2 = PaginationParams(per_page=-10)
        assert params2.per_page == 1

    def test_per_page_maximum(self):
        """Test per_page maximum is 100"""
        from gaap.db.repositories.base import PaginationParams

        params = PaginationParams(per_page=200)
        assert params.per_page == 100

    def test_offset_calculation(self):
        """Test offset calculation"""
        from gaap.db.repositories.base import PaginationParams

        params = PaginationParams(page=3, per_page=20)
        assert params.offset == 40  # (3-1) * 20


class TestPaginatedResult:
    """Test PaginatedResult"""

    def test_creation(self):
        """Test creating paginated result"""
        from gaap.db.repositories.base import PaginatedResult

        items = [1, 2, 3]
        result = PaginatedResult(
            items=items,
            total=10,
            page=1,
            per_page=3,
        )

        assert result.items == items
        assert result.total == 10
        assert result.page == 1
        assert result.per_page == 3
        assert result.pages == 4  # ceil(10/3)

    def test_pages_calculation_exact(self):
        """Test pages calculation when exact"""
        from gaap.db.repositories.base import PaginatedResult

        result = PaginatedResult(
            items=[1, 2],
            total=10,
            page=1,
            per_page=5,
        )

        assert result.pages == 2

    def test_pages_calculation_zero_per_page(self):
        """Test pages calculation with zero per_page"""
        from gaap.db.repositories.base import PaginatedResult

        result = PaginatedResult(
            items=[],
            total=10,
            page=1,
            per_page=0,
        )

        assert result.pages == 0

    def test_to_dict(self):
        """Test converting to dictionary"""
        from gaap.db.repositories.base import PaginatedResult

        result = PaginatedResult(
            items=["a", "b"],
            total=10,
            page=1,
            per_page=2,
        )

        d = result.to_dict()
        assert d["items"] == ["a", "b"]
        assert d["total"] == 10
        assert d["page"] == 1
        assert d["per_page"] == 2
        assert d["pages"] == 5


class TestBaseRepository:
    """Test BaseRepository"""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session"""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def mock_model(self):
        """Create mock model class"""
        model = Mock()
        model.id = "test_id"
        return model

    async def test_get_by_id(self, mock_session):
        """Test getting entity by ID"""
        from gaap.db.repositories.base import BaseRepository

        mock_entity = Mock()
        mock_session.get = AsyncMock(return_value=mock_entity)

        # Create a concrete repository class for testing
        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.get("test_id")

        assert result is mock_entity
        mock_session.get.assert_called_once_with(Mock, "test_id")

    async def test_get_by_id_not_found(self, mock_session):
        """Test getting non-existent entity"""
        from gaap.db.repositories.base import BaseRepository

        mock_session.get = AsyncMock(return_value=None)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.get("nonexistent")

        assert result is None

    async def test_get_by_ids(self, mock_session):
        """Test getting multiple entities by IDs"""
        from gaap.db.repositories.base import BaseRepository

        mock_entities = [Mock(), Mock()]
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_entities
        mock_session.execute = AsyncMock(return_value=mock_result)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.get_by_ids(["id1", "id2"])

        assert len(result) == 2

    async def test_get_by_ids_empty_list(self, mock_session):
        """Test getting entities with empty ID list"""
        from gaap.db.repositories.base import BaseRepository

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.get_by_ids([])

        assert result == []

    async def test_get_all(self, mock_session):
        """Test getting all entities"""
        from gaap.db.repositories.base import BaseRepository, PaginationParams

        mock_entities = [Mock(), Mock(), Mock()]

        # Mock count query
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 3

        # Mock items query
        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = mock_entities

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.get_all()

        assert result.total == 3
        assert len(result.items) == 3

    async def test_get_all_with_pagination(self, mock_session):
        """Test getting all with pagination"""
        from gaap.db.repositories.base import BaseRepository, PaginationParams

        mock_entities = [Mock(), Mock()]

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 10

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = mock_entities

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        pagination = PaginationParams(page=2, per_page=2)
        result = await repo.get_all(pagination=pagination)

        assert result.page == 2
        assert result.per_page == 2

    async def test_create(self, mock_session):
        """Test creating entity"""
        from gaap.db.repositories.base import BaseRepository

        mock_entity = Mock()
        mock_model = Mock(return_value=mock_entity)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, mock_model)

        repo = TestRepo(mock_session)
        result = await repo.create(name="test", value=42)

        assert result is mock_entity
        mock_session.add.assert_called_once_with(mock_entity)
        mock_session.flush.assert_called_once()

    async def test_create_many(self, mock_session):
        """Test creating multiple entities"""
        from gaap.db.repositories.base import BaseRepository

        mock_model = Mock()

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, mock_model)

        repo = TestRepo(mock_session)
        data_list = [{"name": "a"}, {"name": "b"}]
        result = await repo.create_many(data_list)

        assert len(result) == 2
        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()

    async def test_update(self, mock_session):
        """Test updating entity"""
        from gaap.db.repositories.base import BaseRepository

        mock_entity = Mock()
        mock_entity.name = "original"
        mock_entity.value = 0

        mock_session.get = AsyncMock(return_value=mock_entity)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.update("test_id", name="updated", value=42)

        assert result is mock_entity
        assert mock_entity.name == "updated"
        assert mock_entity.value == 42
        mock_session.flush.assert_called_once()

    async def test_update_not_found(self, mock_session):
        """Test updating non-existent entity"""
        from gaap.db.repositories.base import BaseRepository

        mock_session.get = AsyncMock(return_value=None)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.update("nonexistent", name="updated")

        assert result is None

    async def test_update_many(self, mock_session):
        """Test updating multiple entities"""
        from gaap.db.repositories.base import BaseRepository

        mock_entity1 = Mock()
        mock_entity2 = Mock()

        mock_session.get = AsyncMock(side_effect=[mock_entity1, mock_entity2])

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.update_many(["id1", "id2"], status="updated")

        assert len(result) == 2

    async def test_delete(self, mock_session):
        """Test deleting entity"""
        from gaap.db.repositories.base import BaseRepository

        mock_entity = Mock()
        mock_session.get = AsyncMock(return_value=mock_entity)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.delete("test_id")

        assert result is True
        mock_session.delete.assert_called_once_with(mock_entity)
        mock_session.flush.assert_called_once()

    async def test_delete_not_found(self, mock_session):
        """Test deleting non-existent entity"""
        from gaap.db.repositories.base import BaseRepository

        mock_session.get = AsyncMock(return_value=None)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.delete("nonexistent")

        assert result is False

    async def test_delete_many(self, mock_session):
        """Test deleting multiple entities"""
        from gaap.db.repositories.base import BaseRepository

        mock_entity = Mock()
        mock_session.get = AsyncMock(side_effect=[mock_entity, None, mock_entity])

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.delete_many(["id1", "id2", "id3"])

        assert result == 2  # Two were found and deleted

    async def test_exists(self, mock_session):
        """Test checking entity existence"""
        from gaap.db.repositories.base import BaseRepository

        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute = AsyncMock(return_value=mock_result)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.exists("test_id")

        assert result is True

    async def test_exists_not_found(self, mock_session):
        """Test checking non-existent entity"""
        from gaap.db.repositories.base import BaseRepository

        mock_result = Mock()
        mock_result.scalar.return_value = 0
        mock_session.execute = AsyncMock(return_value=mock_result)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.exists("nonexistent")

        assert result is False

    async def test_count(self, mock_session):
        """Test counting entities"""
        from gaap.db.repositories.base import BaseRepository

        mock_result = Mock()
        mock_result.scalar.return_value = 42
        mock_session.execute = AsyncMock(return_value=mock_result)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.count()

        assert result == 42

    async def test_find_one(self, mock_session):
        """Test finding one entity by filters"""
        from gaap.db.repositories.base import BaseRepository

        mock_entity = Mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_entity
        mock_session.execute = AsyncMock(return_value=mock_result)

        class TestModel:
            name = "name"
            status = "status"

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, TestModel)

        repo = TestRepo(mock_session)
        result = await repo.find_one(name="test", status="active")

        assert result is mock_entity

    async def test_find_many(self, mock_session):
        """Test finding many entities by filters"""
        from gaap.db.repositories.base import BaseRepository, PaginationParams

        mock_entities = [Mock(), Mock()]

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 5

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = mock_entities

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        class TestModel:
            status = "status"

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, TestModel)

        repo = TestRepo(mock_session)
        result = await repo.find_many(status="active")

        assert result.total == 5
        assert len(result.items) == 2


class TestSessionRepository:
    """Test SessionRepository"""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session"""
        return AsyncMock(spec=AsyncSession)

    async def test_list_by_user(self, mock_session):
        """Test listing sessions by user"""
        from gaap.db.repositories.session import SessionRepository

        mock_sessions = [Mock(), Mock()]

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 5

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = mock_sessions

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = SessionRepository(mock_session)
        result = await repo.list_by_user("user123")

        assert result.total == 5
        assert len(result.items) == 2

    async def test_list_by_user_with_status(self, mock_session):
        """Test listing sessions by user with status filter"""
        from gaap.db.repositories.session import SessionRepository, SessionStatus

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 3

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = SessionRepository(mock_session)
        result = await repo.list_by_user("user123", status=SessionStatus.ACTIVE)

        assert result.total == 3

    async def test_list_active(self, mock_session):
        """Test listing active sessions"""
        from gaap.db.repositories.session import SessionRepository

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 10

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = SessionRepository(mock_session)
        result = await repo.list_active()

        assert result.total == 10

    async def test_search(self, mock_session):
        """Test searching sessions"""
        from gaap.db.repositories.session import SessionRepository

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 2

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = SessionRepository(mock_session)
        result = await repo.search("user123", "test query")

        assert result.total == 2

    async def test_update_message_stats(self, mock_session):
        """Test updating session message stats"""
        from gaap.db.repositories.session import SessionRepository

        mock_session_obj = Mock()
        mock_session_obj.message_count = 5
        mock_session_obj.total_tokens = 100
        mock_session_obj.total_cost = 0.01

        mock_session.get = AsyncMock(return_value=mock_session_obj)

        repo = SessionRepository(mock_session)
        result = await repo.update_message_stats("session123", tokens=50, cost=0.005)

        assert result is mock_session_obj
        assert result.message_count == 6
        assert result.total_tokens == 150
        assert result.total_cost == 0.015
        mock_session.flush.assert_called_once()

    async def test_update_message_stats_not_found(self, mock_session):
        """Test updating stats for non-existent session"""
        from gaap.db.repositories.session import SessionRepository

        mock_session.get = AsyncMock(return_value=None)

        repo = SessionRepository(mock_session)
        result = await repo.update_message_stats("nonexistent", tokens=50, cost=0.005)

        assert result is None

    async def test_archive(self, mock_session):
        """Test archiving session"""
        from gaap.db.repositories.session import SessionRepository

        mock_session_obj = Mock()
        mock_session.get = AsyncMock(return_value=mock_session_obj)

        repo = SessionRepository(mock_session)
        result = await repo.archive("session123")

        assert result.status == "archived"
        assert result.archived_at is not None

    async def test_restore(self, mock_session):
        """Test restoring archived session"""
        from gaap.db.repositories.session import SessionRepository

        mock_session_obj = Mock()
        mock_session.get = AsyncMock(return_value=mock_session_obj)

        repo = SessionRepository(mock_session)
        result = await repo.restore("session123")

        assert result.status == "active"
        assert result.archived_at is None

    async def test_get_stats(self, mock_session):
        """Test getting session statistics"""
        from gaap.db.repositories.session import SessionRepository

        # Mock status count query
        mock_status_result = Mock()
        mock_status_result.all.return_value = [("active", 5), ("archived", 3)]

        # Mock totals query
        mock_totals_result = Mock()
        mock_totals_result.one.return_value = (1000, 0.05, 50)

        mock_session.execute = AsyncMock(side_effect=[mock_status_result, mock_totals_result])

        repo = SessionRepository(mock_session)
        result = await repo.get_stats("user123")

        assert result["total_sessions"] == 8
        assert result["by_status"]["active"] == 5
        assert result["total_tokens"] == 1000
        assert result["total_cost"] == 0.05
        assert result["total_messages"] == 50


class TestMessageRepository:
    """Test MessageRepository"""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session"""
        return AsyncMock(spec=AsyncSession)

    async def test_get_by_session(self, mock_session):
        """Test getting messages by session"""
        from gaap.db.repositories.message import MessageRepository

        mock_messages = [Mock(), Mock()]

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 20

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = mock_messages

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = MessageRepository(mock_session)
        result = await repo.get_by_session("session123")

        assert result.total == 20
        assert len(result.items) == 2

    async def test_get_conversation_history(self, mock_session):
        """Test getting conversation history"""
        from gaap.db.repositories.message import MessageRepository

        mock_messages = [Mock(), Mock()]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_messages
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = MessageRepository(mock_session)
        result = await repo.get_conversation_history("session123")

        assert len(result) == 2

    async def test_get_next_sequence(self, mock_session):
        """Test getting next sequence number"""
        from gaap.db.repositories.message import MessageRepository

        mock_result = Mock()
        mock_result.scalar.return_value = 5
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = MessageRepository(mock_session)
        result = await repo.get_next_sequence("session123")

        assert result == 6  # 5 + 1

    async def test_get_next_sequence_empty(self, mock_session):
        """Test getting next sequence when no messages"""
        from gaap.db.repositories.message import MessageRepository

        mock_result = Mock()
        mock_result.scalar.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = MessageRepository(mock_session)
        result = await repo.get_next_sequence("session123")

        assert result == 1

    async def test_get_token_stats(self, mock_session):
        """Test getting token statistics"""
        from gaap.db.repositories.message import MessageRepository

        mock_result = Mock()
        mock_result.one.return_value = (1000, 500, 1500, 0.02)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = MessageRepository(mock_session)
        result = await repo.get_token_stats("session123")

        assert result["prompt_tokens"] == 1000
        assert result["completion_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["cost_usd"] == 0.02

    async def test_get_messages_by_role(self, mock_session):
        """Test getting messages by role"""
        from gaap.db.repositories.message import MessageRepository

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 5

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = MessageRepository(mock_session)
        result = await repo.get_messages_by_role("session123", "user")

        assert result.total == 5


class TestUserRepository:
    """Test UserRepository"""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session"""
        return AsyncMock(spec=AsyncSession)

    async def test_get_by_email(self, mock_session):
        """Test getting user by email"""
        from gaap.db.repositories.user import UserRepository

        mock_user = Mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = UserRepository(mock_session)
        result = await repo.get_by_email("test@example.com")

        assert result is mock_user

    async def test_get_by_email_not_found(self, mock_session):
        """Test getting non-existent user by email"""
        from gaap.db.repositories.user import UserRepository

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = UserRepository(mock_session)
        result = await repo.get_by_email("nonexistent@example.com")

        assert result is None

    async def test_get_by_username(self, mock_session):
        """Test getting user by username"""
        from gaap.db.repositories.user import UserRepository

        mock_user = Mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = UserRepository(mock_session)
        result = await repo.get_by_username("testuser")

        assert result is mock_user

    async def test_get_active_users(self, mock_session):
        """Test getting active users"""
        from gaap.db.repositories.user import UserRepository

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 10

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = UserRepository(mock_session)
        result = await repo.get_active_users()

        assert result.total == 10

    async def test_search_users(self, mock_session):
        """Test searching users"""
        from gaap.db.repositories.user import UserRepository

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 3

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        repo = UserRepository(mock_session)
        result = await repo.search("test")

        assert result.total == 3

    async def test_get_user_stats(self, mock_session):
        """Test getting user statistics"""
        from gaap.db.repositories.user import UserRepository

        mock_status_result = Mock()
        mock_status_result.all.return_value = [("active", 10), ("pending", 2)]

        mock_role_result = Mock()
        mock_role_result.all.return_value = [("user", 11), ("admin", 1)]

        mock_session.execute = AsyncMock(side_effect=[mock_status_result, mock_role_result])

        repo = UserRepository(mock_session)
        result = await repo.get_user_stats()

        assert result["total"] == 12
        assert result["by_status"]["active"] == 10
        assert result["by_role"]["user"] == 11

    async def test_update_last_login(self, mock_session):
        """Test updating last login"""
        from gaap.db.repositories.user import UserRepository
        from datetime import datetime

        mock_user = Mock()
        mock_session.get = AsyncMock(return_value=mock_user)

        repo = UserRepository(mock_session)
        result = await repo.update_last_login("user123")

        assert result is mock_user
        assert result.last_login is not None
        mock_session.flush.assert_called_once()

    async def test_deactivate_user(self, mock_session):
        """Test deactivating user"""
        from gaap.db.repositories.user import UserRepository

        mock_user = Mock()
        mock_user.is_active = True
        mock_session.get = AsyncMock(return_value=mock_user)

        repo = UserRepository(mock_session)
        result = await repo.deactivate("user123")

        assert result is mock_user
        assert result.is_active is False
        assert result.status == "inactive"


class TestRepositoryEdgeCases:
    """Test repository edge cases"""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session"""
        return AsyncMock(spec=AsyncSession)

    async def test_paginated_result_zero_total(self, mock_session):
        """Test paginated result with zero total"""
        from gaap.db.repositories.base import PaginatedResult

        result = PaginatedResult(
            items=[],
            total=0,
            page=1,
            per_page=10,
        )

        assert result.pages == 0
        assert result.items == []

    async def test_find_many_no_filters(self, mock_session):
        """Test find_many with no filters"""
        from gaap.db.repositories.base import BaseRepository

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 0

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, Mock)

        repo = TestRepo(mock_session)
        result = await repo.find_many()

        assert result.total == 0

    async def test_update_nonexistent_attribute(self, mock_session):
        """Test updating non-existent attribute"""
        from gaap.db.repositories.base import BaseRepository

        class TestModel:
            name = "name"
            existing_attr = "value"

        mock_entity = Mock()
        mock_entity.name = "test"
        mock_entity.existing_attr = "value"

        mock_session.get = AsyncMock(return_value=mock_entity)

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, TestModel)

        repo = TestRepo(mock_session)
        result = await repo.update("test_id", nonexistent_attr="value")

        # Should not set non-existent attribute
        assert not hasattr(mock_entity, "nonexistent_attr") or mock_entity.nonexistent_attr is None
