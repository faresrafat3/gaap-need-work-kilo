"""
Comprehensive tests for gaap/api/sessions.py module
Tests session CRUD, filtering, and pagination
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.asyncio


class TestSessionStatusEnum:
    """Test SessionStatusEnum"""

    def test_status_values(self):
        """Test all status values"""
        from gaap.api.sessions import SessionStatusEnum

        assert SessionStatusEnum.PENDING == "pending"
        assert SessionStatusEnum.RUNNING == "running"
        assert SessionStatusEnum.PAUSED == "paused"
        assert SessionStatusEnum.COMPLETED == "completed"
        assert SessionStatusEnum.FAILED == "failed"
        assert SessionStatusEnum.CANCELLED == "cancelled"


class TestSessionPriorityEnum:
    """Test SessionPriorityEnum"""

    def test_priority_values(self):
        """Test all priority values"""
        from gaap.api.sessions import SessionPriorityEnum

        assert SessionPriorityEnum.LOW == "low"
        assert SessionPriorityEnum.NORMAL == "normal"
        assert SessionPriorityEnum.HIGH == "high"
        assert SessionPriorityEnum.CRITICAL == "critical"


class TestSessionCreateRequest:
    """Test SessionCreateRequest model"""

    def test_valid_request(self):
        """Test valid create request"""
        from gaap.api.sessions import SessionCreateRequest

        request = SessionCreateRequest(name="Test Session")
        assert request.name == "Test Session"
        assert request.description == ""
        assert request.priority.value == "normal"
        assert request.tags == []
        assert request.config == {}
        assert request.metadata == {}

    def test_name_min_length(self):
        """Test name minimum length"""
        from gaap.api.sessions import SessionCreateRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SessionCreateRequest(name="")

    def test_name_max_length(self):
        """Test name maximum length"""
        from gaap.api.sessions import SessionCreateRequest
        from pydantic import ValidationError

        long_name = "x" * 201
        with pytest.raises(ValidationError):
            SessionCreateRequest(name=long_name)

    def test_custom_values(self):
        """Test create request with custom values"""
        from gaap.api.sessions import SessionCreateRequest, SessionPriorityEnum

        request = SessionCreateRequest(
            name="Test",
            description="A test session",
            priority=SessionPriorityEnum.HIGH,
            tags=["test", "example"],
            config={"key": "value"},
            metadata={"author": "test"},
        )

        assert request.description == "A test session"
        assert request.priority == SessionPriorityEnum.HIGH
        assert request.tags == ["test", "example"]
        assert request.config == {"key": "value"}
        assert request.metadata == {"author": "test"}


class TestSessionUpdateRequest:
    """Test SessionUpdateRequest model"""

    def test_all_fields_optional(self):
        """Test all fields are optional"""
        from gaap.api.sessions import SessionUpdateRequest

        request = SessionUpdateRequest()
        assert request.name is None
        assert request.description is None
        assert request.priority is None
        assert request.tags is None
        assert request.config is None
        assert request.metadata is None

    def test_partial_update(self):
        """Test partial update"""
        from gaap.api.sessions import SessionUpdateRequest, SessionPriorityEnum

        request = SessionUpdateRequest(name="Updated Name")
        assert request.name == "Updated Name"
        assert request.description is None


class TestMessageResponse:
    """Test MessageResponse model"""

    def test_message_response_creation(self):
        """Test creating message response"""
        from gaap.api.sessions import MessageResponse

        response = MessageResponse(
            id="msg123",
            role="user",
            content="Hello",
            created_at="2024-01-01T00:00:00",
        )

        assert response.id == "msg123"
        assert response.role == "user"
        assert response.content == "Hello"
        assert response.tokens == 0
        assert response.provider is None


class TestSessionResponse:
    """Test SessionResponse model"""

    def test_session_response_creation(self):
        """Test creating session response"""
        from gaap.api.sessions import SessionResponse

        response = SessionResponse(
            id="session123",
            name="Test Session",
            description="A test",
            status="active",
            priority="normal",
            tags=["test"],
            config={},
            metadata={},
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            last_message_at=None,
            message_count=0,
            total_tokens=0,
            total_cost=0.0,
        )

        assert response.id == "session123"
        assert response.name == "Test Session"
        assert response.message_count == 0


class TestSessionListResponse:
    """Test SessionListResponse model"""

    def test_list_response_defaults(self):
        """Test list response defaults"""
        from gaap.api.sessions import SessionListResponse

        response = SessionListResponse(sessions=[], total=0)
        assert response.sessions == []
        assert response.total == 0
        assert response.page == 1
        assert response.per_page == 50


class TestSessionExportResponse:
    """Test SessionExportResponse model"""

    def test_export_response(self):
        """Test export response"""
        from gaap.api.sessions import SessionExportResponse, SessionResponse

        session = SessionResponse(
            id="session123",
            name="Test",
            description=None,
            status="active",
            priority="normal",
            tags=[],
            config={},
            metadata={},
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            last_message_at=None,
            message_count=0,
            total_tokens=0,
            total_cost=0.0,
        )

        response = SessionExportResponse(
            session=session,
            messages=[],
            stats={},
        )

        assert response.session.id == "session123"


class TestSessionToResponse:
    """Test _session_to_response helper"""

    def test_session_to_response(self):
        """Test converting session to response"""
        from gaap.api.sessions import _session_to_response

        mock_session = Mock()
        mock_session.id = "session123"
        mock_session.title = "Test Session"
        mock_session.description = "A test"
        mock_session.status = "active"
        mock_session.priority = "normal"
        mock_session.tags = ["test"]
        mock_session.config = {"key": "value"}
        mock_session.metadata = {"author": "test"}
        mock_session.created_at = datetime(2024, 1, 1)
        mock_session.updated_at = datetime(2024, 1, 1)
        mock_session.last_message_at = None
        mock_session.message_count = 10
        mock_session.total_tokens = 100
        mock_session.total_cost = 0.01

        response = _session_to_response(mock_session)

        assert response.id == "session123"
        assert response.name == "Test Session"
        assert response.tags == ["test"]

    def test_session_to_response_none_raises(self):
        """Test converting None session raises HTTPException"""
        from gaap.api.sessions import _session_to_response

        with pytest.raises(HTTPException) as exc_info:
            _session_to_response(None)

        assert exc_info.value.status_code == 404

    def test_session_to_response_long_title(self):
        """Test session with long title"""
        from gaap.api.sessions import _session_to_response

        mock_session = Mock()
        mock_session.id = "session123"
        mock_session.title = "x" * 100
        mock_session.description = None
        mock_session.status = "active"
        mock_session.priority = "normal"
        mock_session.tags = None
        mock_session.config = None
        mock_session.metadata = None
        mock_session.created_at = datetime(2024, 1, 1)
        mock_session.updated_at = datetime(2024, 1, 1)
        mock_session.last_message_at = None
        mock_session.message_count = None
        mock_session.total_tokens = None
        mock_session.total_cost = None

        response = _session_to_response(mock_session)

        assert response.name == "x" * 100


class TestMessageToResponse:
    """Test _message_to_response helper"""

    def test_message_to_response(self):
        """Test converting message to response"""
        from gaap.api.sessions import _message_to_response

        mock_msg = Mock()
        mock_msg.id = "msg123"
        mock_msg.role = "user"
        mock_msg.content = "Hello"
        mock_msg.created_at = datetime(2024, 1, 1)
        mock_msg.total_tokens = 10
        mock_msg.provider = "kimi"
        mock_msg.model = "model1"

        response = _message_to_response(mock_msg)

        assert response.id == "msg123"
        assert response.role == "user"
        assert response.content == "Hello"
        assert response.tokens == 10


class TestListSessions:
    """Test list_sessions endpoint"""

    async def test_list_all_sessions(self):
        """Test listing all sessions"""
        from gaap.api.sessions import list_sessions

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_session = Mock()
            mock_session.id = "session123"
            mock_session.title = "Test"
            mock_session.status = "active"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_paginated = Mock()
            mock_paginated.items = [mock_session]
            mock_paginated.total = 1
            mock_paginated.page = 1
            mock_paginated.per_page = 50

            mock_repo.return_value.find_many = AsyncMock(return_value=mock_paginated)
            mock_repo_class.return_value = mock_repo

            result = await list_sessions(db=mock_db)

            assert result.total == 1
            assert len(result.sessions) == 1

    async def test_list_sessions_with_status_filter(self):
        """Test listing sessions with status filter"""
        from gaap.api.sessions import list_sessions, SessionStatusEnum

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_paginated = Mock()
            mock_paginated.items = []
            mock_paginated.total = 0
            mock_paginated.page = 1
            mock_paginated.per_page = 50

            mock_repo.return_value.find_many = AsyncMock(return_value=mock_paginated)
            mock_repo_class.return_value = mock_repo

            result = await list_sessions(status=SessionStatusEnum.RUNNING, db=mock_db)

            assert result.total == 0

    async def test_list_sessions_error(self):
        """Test listing sessions with error"""
        from gaap.api.sessions import list_sessions

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.return_value.find_many = AsyncMock(side_effect=Exception("DB error"))
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await list_sessions(db=mock_db)

            assert exc_info.value.status_code == 500


class TestCreateSession:
    """Test create_session endpoint"""

    async def test_create_session_success(self):
        """Test creating session successfully"""
        from gaap.api.sessions import create_session, SessionCreateRequest

        mock_db = AsyncMock()
        request = SessionCreateRequest(name="Test Session")

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_repo_class,
            patch("gaap.api.sessions.EventEmitter") as mock_emitter,
        ):
            mock_session = Mock()
            mock_session.id = "new_session_id"
            mock_session.title = "Test Session"
            mock_session.description = None
            mock_session.status = "active"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_repo = Mock()
            mock_repo.create = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            mock_emitter.get_instance.return_value = Mock(emit=Mock())

            result = await create_session(request, db=mock_db)

            assert result.name == "Test Session"
            mock_db.commit.assert_called_once()

    async def test_create_session_error(self):
        """Test creating session with error"""
        from gaap.api.sessions import create_session, SessionCreateRequest

        mock_db = AsyncMock()
        request = SessionCreateRequest(name="Test Session")

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.create = AsyncMock(side_effect=Exception("DB error"))
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await create_session(request, db=mock_db)

            assert exc_info.value.status_code == 500
            mock_db.rollback.assert_called_once()


class TestGetSessionDetail:
    """Test get_session_detail endpoint"""

    async def test_get_existing_session(self):
        """Test getting existing session"""
        from gaap.api.sessions import get_session_detail

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_session = Mock()
            mock_session.id = "session123"
            mock_session.title = "Test"
            mock_session.description = None
            mock_session.status = "active"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            result = await get_session_detail("session123", db=mock_db)

            assert result.id == "session123"

    async def test_get_nonexistent_session(self):
        """Test getting non-existent session"""
        from gaap.api.sessions import get_session_detail

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await get_session_detail("nonexistent", db=mock_db)

            assert exc_info.value.status_code == 404

    async def test_get_session_error(self):
        """Test getting session with error"""
        from gaap.api.sessions import get_session_detail

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.get = AsyncMock(side_effect=Exception("DB error"))
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await get_session_detail("session123", db=mock_db)

            assert exc_info.value.status_code == 500


class TestUpdateSession:
    """Test update_session endpoint"""

    async def test_update_session_success(self):
        """Test updating session successfully"""
        from gaap.api.sessions import update_session, SessionUpdateRequest

        mock_db = AsyncMock()
        request = SessionUpdateRequest(name="Updated Name")

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_repo_class,
            patch("gaap.api.sessions.EventEmitter") as mock_emitter,
        ):
            mock_session = Mock()
            mock_session.id = "session123"
            mock_session.title = "Updated Name"
            mock_session.description = None
            mock_session.status = "active"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo.update = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            mock_emitter.get_instance.return_value = Mock(emit=Mock())

            result = await update_session("session123", request, db=mock_db)

            assert result.name == "Updated Name"

    async def test_update_nonexistent_session(self):
        """Test updating non-existent session"""
        from gaap.api.sessions import update_session, SessionUpdateRequest

        mock_db = AsyncMock()
        request = SessionUpdateRequest(name="Updated")

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await update_session("nonexistent", request, db=mock_db)

            assert exc_info.value.status_code == 404

    async def test_update_session_priority(self):
        """Test updating session priority"""
        from gaap.api.sessions import update_session, SessionUpdateRequest, SessionPriorityEnum

        mock_db = AsyncMock()
        request = SessionUpdateRequest(priority=SessionPriorityEnum.HIGH)

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_repo_class,
            patch("gaap.api.sessions.EventEmitter") as mock_emitter,
        ):
            mock_session = Mock()
            mock_session.id = "session123"
            mock_session.title = "Test"
            mock_session.description = None
            mock_session.status = "active"
            mock_session.priority = "high"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo.update = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            mock_emitter.get_instance.return_value = Mock(emit=Mock())

            result = await update_session("session123", request, db=mock_db)

            assert result.priority == "high"


class TestDeleteSession:
    """Test delete_session endpoint"""

    async def test_delete_existing_session(self):
        """Test deleting existing session"""
        from gaap.api.sessions import delete_session

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_session = Mock()
            mock_session.id = "session123"

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo.update = AsyncMock()
            mock_repo_class.return_value = mock_repo

            result = await delete_session("session123", db=mock_db)

            assert result["success"] is True
            mock_db.commit.assert_called_once()

    async def test_delete_nonexistent_session(self):
        """Test deleting non-existent session"""
        from gaap.api.sessions import delete_session

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await delete_session("nonexistent", db=mock_db)

            assert exc_info.value.status_code == 404


class TestPauseSession:
    """Test pause_session endpoint"""

    async def test_pause_active_session(self):
        """Test pausing active session"""
        from gaap.api.sessions import pause_session

        mock_db = AsyncMock()

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_repo_class,
            patch("gaap.api.sessions.EventEmitter") as mock_emitter,
        ):
            mock_session = Mock()
            mock_session.id = "session123"
            mock_session.title = "Test"
            mock_session.description = None
            mock_session.status = "active"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo.update = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            mock_emitter.get_instance.return_value = Mock(emit=Mock())

            result = await pause_session("session123", db=mock_db)

            assert result.status == "paused"

    async def test_pause_non_active_session(self):
        """Test pausing non-active session"""
        from gaap.api.sessions import pause_session

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_session = Mock()
            mock_session.status = "paused"

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await pause_session("session123", db=mock_db)

            assert exc_info.value.status_code == 400


class TestResumeSession:
    """Test resume_session endpoint"""

    async def test_resume_paused_session(self):
        """Test resuming paused session"""
        from gaap.api.sessions import resume_session

        mock_db = AsyncMock()

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_repo_class,
            patch("gaap.api.sessions.EventEmitter") as mock_emitter,
        ):
            mock_session = Mock()
            mock_session.id = "session123"
            mock_session.title = "Test"
            mock_session.description = None
            mock_session.status = "paused"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo.update = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            mock_emitter.get_instance.return_value = Mock(emit=Mock())

            result = await resume_session("session123", db=mock_db)

            assert result.status == "active"


class TestExportSession:
    """Test export_session endpoint"""

    async def test_export_session_success(self):
        """Test exporting session successfully"""
        from gaap.api.sessions import export_session

        mock_db = AsyncMock()

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_sess_repo,
            patch("gaap.api.sessions.MessageRepository") as mock_msg_repo,
        ):
            mock_session = Mock()
            mock_session.id = "session123"
            mock_session.title = "Test"
            mock_session.description = None
            mock_session.status = "active"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 2
            mock_session.total_tokens = 100
            mock_session.total_cost = 0.01

            mock_sess = Mock()
            mock_sess.get = AsyncMock(return_value=mock_session)
            mock_sess_repo.return_value = mock_sess

            mock_msg = Mock()
            mock_msg.id = "msg1"
            mock_msg.role = "user"
            mock_msg.content = "Hello"
            mock_msg.created_at = datetime(2024, 1, 1)
            mock_msg.total_tokens = 10

            mock_paginated = Mock()
            mock_paginated.items = [mock_msg]

            mock_msg_repo_inst = Mock()
            mock_msg_repo_inst.get_by_session = AsyncMock(return_value=mock_paginated)
            mock_msg_repo_inst.get_token_stats = AsyncMock(return_value={"total": 100})
            mock_msg_repo.return_value = mock_msg_repo_inst

            result = await export_session("session123", db=mock_db)

            assert result.session.id == "session123"
            assert len(result.messages) == 1


class TestCancelSession:
    """Test cancel_session endpoint"""

    async def test_cancel_active_session(self):
        """Test cancelling active session"""
        from gaap.api.sessions import cancel_session

        mock_db = AsyncMock()

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_repo_class,
            patch("gaap.api.sessions.EventEmitter") as mock_emitter,
        ):
            mock_session = Mock()
            mock_session.id = "session123"
            mock_session.title = "Test"
            mock_session.description = None
            mock_session.status = "archived"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime(2024, 1, 1)
            mock_session.updated_at = datetime(2024, 1, 1)
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo.update = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            mock_emitter.get_instance.return_value = Mock(emit=Mock())

            result = await cancel_session("session123", db=mock_db)

            assert result.status == "archived"

    async def test_cancel_archived_session(self):
        """Test cancelling already archived session"""
        from gaap.api.sessions import cancel_session

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            mock_session = Mock()
            mock_session.status = "archived"

            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await cancel_session("session123", db=mock_db)

            assert exc_info.value.status_code == 400


class TestRegisterRoutes:
    """Test register_routes function"""

    def test_register_routes(self):
        """Test route registration"""
        from gaap.api.sessions import register_routes

        mock_app = Mock()
        register_routes(mock_app)

        mock_app.include_router.assert_called_once()
