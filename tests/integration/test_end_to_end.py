"""
End-to-end integration tests
Tests full chat flow, sessions, and complete user workflows
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


class TestFullChatFlow:
    """Test complete chat flow from start to finish"""

    async def test_complete_chat_session(self):
        """Test complete chat session workflow"""
        from gaap.api.chat import (
            chat,
            ChatRequest,
            _get_or_create_session,
            _store_user_message,
            _store_assistant_message,
        )
        from gaap.api.sessions import get_session_detail

        mock_db = AsyncMock()
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        # Step 1: Create a new session
        mock_session = Mock()
        mock_session.id = "test_session_123"
        mock_session.title = "Test Chat"
        mock_session.status = "active"

        with patch("gaap.api.chat._get_or_create_session") as mock_get_session:
            mock_get_session.return_value = (mock_session, True)

            with patch("gaap.api.chat.get_provider") as mock_get_provider:
                mock_provider = Mock()
                mock_provider._make_request = AsyncMock(
                    return_value={
                        "choices": [{"message": {"content": "Hello! How can I help?"}}],
                        "usage": {"total_tokens": 10},
                    }
                )
                mock_get_provider.return_value = mock_provider

                with (
                    patch("gaap.api.chat._store_user_message") as mock_store_user,
                    patch("gaap.api.chat._store_assistant_message") as mock_store_assist,
                ):
                    mock_store_user.return_value = Mock(id="user_msg_1")
                    mock_store_assist.return_value = Mock(id="assistant_msg_1")

                    # First user message
                    request = ChatRequest(
                        message="Hello, I have a question",
                        store_in_db=True,
                    )

                    response = await chat(request, mock_request, mock_db)

                    assert response.response == "Hello! How can I help?"
                    assert response.session_id == "test_session_123"

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation"""
        from gaap.api.chat import chat, ChatRequest, _store_user_message

        mock_db = AsyncMock()
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        session_id = "multi_turn_session"

        # Turn 1
        with (
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
            patch("gaap.api.chat._store_assistant_message") as mock_store_assist,
        ):
            mock_session = Mock()
            mock_session.id = session_id
            mock_get_session.return_value = (mock_session, False)

            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Response 1"}}],
                    "usage": {"total_tokens": 15},
                }
            )
            mock_get_provider.return_value = mock_provider

            mock_store_user.return_value = Mock(id="msg_1")
            mock_store_assist.return_value = Mock(id="msg_2")

            request = ChatRequest(
                message="Question 1",
                session_id=session_id,
            )

            response1 = await chat(request, mock_request, mock_db)
            assert response1.response == "Response 1"

            # Turn 2
            mock_provider._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Response 2"}}],
                    "usage": {"total_tokens": 20},
                }
            )

            mock_store_user.return_value = Mock(id="msg_3")
            mock_store_assist.return_value = Mock(id="msg_4")

            request2 = ChatRequest(
                message="Question 2",
                session_id=session_id,
            )

            response2 = await chat(request2, mock_request, mock_db)
            assert response2.response == "Response 2"

    async def test_conversation_with_context(self):
        """Test conversation with context"""
        from gaap.api.chat import chat, ChatRequest

        mock_db = AsyncMock()
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        context = {"previous_topics": ["Python", "Async"], "user_preferences": {"concise": True}}

        with (
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat.get_provider") as mock_get_provider,
        ):
            mock_session = Mock()
            mock_session.id = "context_session"
            mock_get_session.return_value = (mock_session, True)

            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Context-aware response"}}],
                    "usage": {"total_tokens": 25},
                }
            )
            mock_get_provider.return_value = mock_provider

            with (
                patch("gaap.api.chat._store_user_message"),
                patch("gaap.api.chat._store_assistant_message"),
            ):
                request = ChatRequest(
                    message="Tell me more",
                    context=context,
                )

                response = await chat(request, mock_request, mock_db)
                assert response.response == "Context-aware response"


class TestSessionWorkflow:
    """Test complete session workflow"""

    async def test_session_lifecycle(self):
        """Test complete session lifecycle"""
        from gaap.api.sessions import (
            create_session,
            get_session_detail,
            update_session,
            pause_session,
            resume_session,
            export_session,
            delete_session,
            SessionCreateRequest,
            SessionUpdateRequest,
        )

        mock_db = AsyncMock()

        # 1. Create session
        create_request = SessionCreateRequest(name="Test Session")

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_repo_class,
            patch("gaap.api.sessions.EventEmitter") as mock_emitter,
        ):
            mock_session = Mock()
            mock_session.id = "lifecycle_session"
            mock_session.title = "Test Session"
            mock_session.status = "active"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime.now()
            mock_session.updated_at = datetime.now()
            mock_session.last_message_at = None
            mock_session.message_count = 0
            mock_session.total_tokens = 0
            mock_session.total_cost = 0.0

            mock_repo = Mock()
            mock_repo.create = AsyncMock(return_value=mock_session)
            mock_repo.get = AsyncMock(return_value=mock_session)
            mock_repo.update = AsyncMock(return_value=mock_session)
            mock_repo_class.return_value = mock_repo

            mock_emitter.get_instance.return_value = Mock(emit=Mock())

            created = await create_session(create_request, db=mock_db)
            assert created.name == "Test Session"

            # 2. Get session
            fetched = await get_session_detail("lifecycle_session", db=mock_db)
            assert fetched.id == "lifecycle_session"

            # 3. Update session
            update_request = SessionUpdateRequest(name="Updated Session")
            updated = await update_session("lifecycle_session", update_request, db=mock_db)
            assert updated.name == "Updated Session"

            # 4. Pause session
            mock_session.status = "paused"
            paused = await pause_session("lifecycle_session", db=mock_db)
            assert paused.status == "paused"

            # 5. Resume session
            mock_session.status = "active"
            resumed = await resume_session("lifecycle_session", db=mock_db)
            assert resumed.status == "active"

    async def test_session_export_with_messages(self):
        """Test exporting session with messages"""
        from gaap.api.sessions import export_session

        mock_db = AsyncMock()

        with (
            patch("gaap.api.sessions.SessionRepository") as mock_sess_repo,
            patch("gaap.api.sessions.MessageRepository") as mock_msg_repo,
        ):
            mock_session = Mock()
            mock_session.id = "export_session"
            mock_session.title = "Export Test"
            mock_session.description = None
            mock_session.status = "active"
            mock_session.priority = "normal"
            mock_session.tags = []
            mock_session.config = {}
            mock_session.metadata = {}
            mock_session.created_at = datetime.now()
            mock_session.updated_at = datetime.now()
            mock_session.last_message_at = datetime.now()
            mock_session.message_count = 3
            mock_session.total_tokens = 150
            mock_session.total_cost = 0.01

            mock_sess = Mock()
            mock_sess.get = AsyncMock(return_value=mock_session)
            mock_sess_repo.return_value = mock_sess

            mock_messages = [
                Mock(
                    id="msg1",
                    role="user",
                    content="Hello",
                    created_at=datetime.now(),
                    total_tokens=5,
                    provider=None,
                    model=None,
                ),
                Mock(
                    id="msg2",
                    role="assistant",
                    content="Hi!",
                    created_at=datetime.now(),
                    total_tokens=10,
                    provider="kimi",
                    model="model1",
                ),
            ]

            mock_paginated = Mock()
            mock_paginated.items = mock_messages

            mock_msg = Mock()
            mock_msg.get_by_session = AsyncMock(return_value=mock_paginated)
            mock_msg.get_token_stats = AsyncMock(return_value={"total": 15})
            mock_msg_repo.return_value = mock_msg

            exported = await export_session("export_session", db=mock_db)

            assert exported.session.id == "export_session"
            assert len(exported.messages) == 2
            assert exported.stats == {"total": 15}


class TestProviderWorkflow:
    """Test complete provider workflow"""

    async def test_provider_management(self):
        """Test provider management workflow"""
        from gaap.api.providers import (
            add_provider,
            list_providers,
            get_provider,
            test_provider,
            disable_provider,
            enable_provider,
            remove_provider,
            ProviderConfig,
        )

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.ProviderFactory") as mock_factory,
            patch("gaap.api.providers.observability") as mock_observability,
        ):
            mock_router = Mock()

            # 1. Add provider
            mock_provider = Mock()
            mock_provider.name = "test_provider"
            mock_provider.provider_type.name = "chat"
            mock_provider.get_available_models.return_value = ["model1", "model2"]
            mock_provider.get_stats.return_value = {"requests": 0}

            mock_factory.create.return_value = mock_provider
            mock_router.get_provider.return_value = mock_provider
            mock_router.get_all_providers.return_value = [mock_provider]

            mock_get_router.return_value = mock_router

            config = ProviderConfig(
                name="test_provider",
                api_key="secret123",
                models=["model1", "model2"],
            )

            added = await add_provider(config)
            assert added.name == "test_provider"
            mock_router.register_provider.assert_called_once()

            # 2. List providers
            providers = await list_providers()
            assert len(providers) == 1

            # 3. Get provider details
            details = await get_provider("test_provider")
            assert details.name == "test_provider"

            # 4. Test provider
            test_result = await test_provider("test_provider")
            assert test_result.success is True

            # 5. Disable provider
            disabled = await disable_provider("test_provider")
            assert disabled["success"] is True

            # 6. Enable provider
            enabled = await enable_provider("test_provider")
            assert enabled["success"] is True

            # 7. Remove provider
            removed = await remove_provider("test_provider")
            assert removed["success"] is True


class TestErrorRecovery:
    """Test error recovery scenarios"""

    async def test_chat_with_provider_fallback(self):
        """Test chat with provider fallback on failure"""
        from gaap.api.chat import chat, ChatRequest

        mock_db = AsyncMock()
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        with patch("gaap.api.chat._get_or_create_session") as mock_get_session:
            mock_session = Mock()
            mock_session.id = "fallback_session"
            mock_get_session.return_value = (mock_session, True)

            with patch("gaap.api.chat.get_provider") as mock_get_provider:
                # Primary provider fails
                mock_provider1 = Mock()
                mock_provider1._make_request = AsyncMock(side_effect=Exception("Primary failed"))

                # Fallback succeeds
                mock_provider2 = Mock()
                mock_provider2._make_request = AsyncMock(
                    return_value={
                        "choices": [{"message": {"content": "Fallback success!"}}],
                        "usage": {"total_tokens": 10},
                    }
                )

                mock_get_provider.side_effect = [mock_provider1, mock_provider2]

                with (
                    patch("gaap.api.chat._store_user_message"),
                    patch("gaap.api.chat._store_assistant_message"),
                ):
                    request = ChatRequest(message="Test")
                    response = await chat(request, mock_request, mock_db)

                    assert response.response == "Fallback success!"
                    assert response.provider_used is not None

    async def test_chat_all_providers_fail(self):
        """Test when all providers fail"""
        from gaap.api.chat import chat, ChatRequest
        from fastapi import HTTPException

        mock_db = AsyncMock()
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        with patch("gaap.api.chat._get_or_create_session") as mock_get_session:
            mock_session = Mock()
            mock_session.id = "fail_session"
            mock_get_session.return_value = (mock_session, True)

            with patch("gaap.api.chat.get_provider") as mock_get_provider:
                # All providers fail
                mock_provider = Mock()
                mock_provider._make_request = AsyncMock(side_effect=Exception("All failed"))
                mock_get_provider.return_value = mock_provider

                with patch("gaap.api.chat._store_user_message"):
                    request = ChatRequest(message="Test")

                    with pytest.raises(HTTPException) as exc_info:
                        await chat(request, mock_request, mock_db)

                    assert exc_info.value.status_code == 503

    async def test_session_recovery_from_error(self):
        """Test session recovery from database error"""
        from gaap.api.sessions import create_session, SessionCreateRequest
        from fastapi import HTTPException

        mock_db = AsyncMock()

        with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
            # First call fails, second succeeds
            mock_repo = Mock()
            mock_repo.create = AsyncMock(side_effect=[Exception("DB error")])
            mock_repo_class.return_value = mock_repo

            request = SessionCreateRequest(name="Test")

            with pytest.raises(HTTPException) as exc_info:
                await create_session(request, db=mock_db)

            assert exc_info.value.status_code == 500
            mock_db.rollback.assert_called_once()


class TestConcurrentOperations:
    """Test concurrent operations"""

    async def test_concurrent_chat_requests(self):
        """Test handling concurrent chat requests"""
        from gaap.api.chat import chat, ChatRequest

        async def make_request(i):
            mock_db = AsyncMock()
            mock_request = Mock()
            mock_request.client.host = f"127.0.0.{i}"

            with (
                patch("gaap.api.chat._get_or_create_session") as mock_get_session,
                patch("gaap.api.chat.get_provider") as mock_get_provider,
                patch("gaap.api.chat._store_user_message"),
                patch("gaap.api.chat._store_assistant_message"),
            ):
                mock_session = Mock()
                mock_session.id = f"session_{i}"
                mock_get_session.return_value = (mock_session, True)

                mock_provider = Mock()
                mock_provider._make_request = AsyncMock(
                    return_value={
                        "choices": [{"message": {"content": f"Response {i}"}}],
                        "usage": {"total_tokens": 10},
                    }
                )
                mock_get_provider.return_value = mock_provider

                request = ChatRequest(message=f"Message {i}")
                response = await chat(request, mock_request, mock_db)
                return response.response

        # Run multiple concurrent requests
        responses = await asyncio.gather(*[make_request(i) for i in range(5)])

        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert f"Response {i}" in response

    async def test_concurrent_session_operations(self):
        """Test concurrent session operations"""
        from gaap.api.sessions import get_session_detail, update_session, SessionUpdateRequest

        async def read_session():
            mock_db = AsyncMock()

            with patch("gaap.api.sessions.SessionRepository") as mock_repo_class:
                mock_session = Mock()
                mock_session.id = "concurrent_session"
                mock_session.title = "Test"
                mock_session.status = "active"

                mock_repo = Mock()
                mock_repo.get = AsyncMock(return_value=mock_session)
                mock_repo_class.return_value = mock_repo

                return await get_session_detail("concurrent_session", db=mock_db)

        async def update_session_fn():
            mock_db = AsyncMock()

            with (
                patch("gaap.api.sessions.SessionRepository") as mock_repo_class,
                patch("gaap.api.sessions.EventEmitter") as mock_emitter,
            ):
                mock_session = Mock()
                mock_session.id = "concurrent_session"
                mock_session.title = "Updated"

                mock_repo = Mock()
                mock_repo.get = AsyncMock(return_value=mock_session)
                mock_repo.update = AsyncMock(return_value=mock_session)
                mock_repo_class.return_value = mock_repo

                mock_emitter.get_instance.return_value = Mock(emit=Mock())

                request = SessionUpdateRequest(name="Updated")
                return await update_session("concurrent_session", request, db=mock_db)

        # Run concurrent read and update
        results = await asyncio.gather(read_session(), update_session_fn())

        assert len(results) == 2


class TestDataIntegrity:
    """Test data integrity across operations"""

    async def test_message_sequence_integrity(self):
        """Test message sequence integrity"""
        from gaap.db.repositories.message import MessageRepository

        mock_db = AsyncMock()

        # Simulate sequence counter
        sequence = 0

        async def get_next_sequence(session_id):
            nonlocal sequence
            sequence += 1
            return sequence

        with patch.object(MessageRepository, "get_next_sequence", side_effect=get_next_sequence):
            repo = MessageRepository(mock_db)

            seq1 = await repo.get_next_sequence("session123")
            seq2 = await repo.get_next_sequence("session123")
            seq3 = await repo.get_next_sequence("session123")

            assert seq1 == 1
            assert seq2 == 2
            assert seq3 == 3

    async def test_session_stats_consistency(self):
        """Test session stats consistency"""
        from gaap.db.repositories.session import SessionRepository

        mock_db = AsyncMock()
        mock_session = Mock()
        mock_session.message_count = 0
        mock_session.total_tokens = 0
        mock_session.total_cost = 0.0

        with patch.object(SessionRepository, "get", return_value=mock_session):
            with patch.object(SessionRepository, "update_message_stats") as mock_update:

                async def update_stats(repo, session_id, tokens, cost):
                    mock_session.message_count += 1
                    mock_session.total_tokens += tokens
                    mock_session.total_cost += cost
                    return mock_session

                mock_update.side_effect = lambda sid, tokens, cost: update_stats(
                    None, sid, tokens, cost
                )

                repo = SessionRepository(mock_db)

                # Update stats multiple times
                await repo.update_message_stats("session123", tokens=10, cost=0.001)
                await repo.update_message_stats("session123", tokens=20, cost=0.002)
                await repo.update_message_stats("session123", tokens=15, cost=0.0015)

                assert mock_session.message_count == 3
                assert mock_session.total_tokens == 45
                assert mock_session.total_cost == 0.0045

    async def test_token_count_accuracy(self):
        """Test token count accuracy"""
        from gaap.api.chat import _store_user_message, _store_assistant_message

        mock_db = AsyncMock()

        with (
            patch("gaap.api.chat.MessageRepository") as mock_repo_class,
            patch("gaap.api.chat.SessionRepository") as mock_sess_repo,
        ):
            mock_msg = Mock()
            mock_msg.id = "msg123"
            mock_msg.total_tokens = 10

            mock_repo = Mock()
            mock_repo.create = AsyncMock(return_value=mock_msg)
            mock_repo.get_next_sequence = AsyncMock(return_value=1)
            mock_repo_class.return_value = mock_repo

            mock_sess = Mock()
            mock_sess.update_message_stats = AsyncMock()
            mock_sess_repo.return_value = mock_sess

            # Store user message
            user_msg = await _store_user_message(mock_db, "session123", "Hello world")
            assert user_msg.total_tokens == 10

            # Store assistant message
            mock_msg.total_tokens = 20
            assistant_msg = await _store_assistant_message(
                mock_db, "session123", "Response", "kimi", "model", 100.0, 20
            )
            assert assistant_msg.total_tokens == 20


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""

    async def test_user_onboarding_flow(self):
        """Test complete user onboarding flow"""
        from gaap.db.models import User
        from gaap.db.repositories import UserRepository

        mock_db = AsyncMock()

        # 1. Create user
        with patch.object(UserRepository, "create") as mock_create:
            mock_user = Mock(spec=User)
            mock_user.id = "user123"
            mock_user.email = "newuser@example.com"
            mock_user.username = "newuser"
            mock_user.status = "pending"
            mock_user.is_active = True

            mock_create.return_value = mock_user

            repo = UserRepository(mock_db)
            user = await repo.create(
                email="newuser@example.com",
                username="newuser",
                hashed_password="hashed_password",
            )

            assert user.email == "newuser@example.com"
            assert user.status == "pending"

        # 2. Activate user
        with patch.object(UserRepository, "update") as mock_update:
            mock_user.status = "active"
            mock_user.email_verified = True
            mock_update.return_value = mock_user

            updated = await repo.update("user123", status="active", email_verified=True)
            assert updated.status == "active"
            assert updated.email_verified is True

    async def test_chat_with_file_upload(self):
        """Test chat workflow with file upload simulation"""
        from gaap.api.chat import chat, ChatRequest

        mock_db = AsyncMock()
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        # Simulate file context
        file_context = {
            "files": [{"name": "document.pdf", "content_type": "application/pdf", "size": 1024}]
        }

        with (
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat.get_provider") as mock_get_provider,
        ):
            mock_session = Mock()
            mock_session.id = "file_session"
            mock_get_session.return_value = (mock_session, True)

            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "I can see your PDF file"}}],
                    "usage": {"total_tokens": 30},
                }
            )
            mock_get_provider.return_value = mock_provider

            with (
                patch("gaap.api.chat._store_user_message"),
                patch("gaap.api.chat._store_assistant_message"),
            ):
                request = ChatRequest(
                    message="Analyze this file",
                    context=file_context,
                )

                response = await chat(request, mock_request, mock_db)
                assert "PDF" in response.response

    async def test_bulk_operations(self):
        """Test bulk operations"""
        from gaap.db.repositories import BaseRepository, PaginationParams

        mock_db = AsyncMock()

        # Create many items
        mock_items = [Mock(id=f"item_{i}") for i in range(100)]

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 100

        mock_items_result = Mock()
        mock_items_result.scalars.return_value.all.return_value = mock_items[:50]

        mock_db.execute = AsyncMock(side_effect=[mock_count_result, mock_items_result])

        class TestModel:
            pass

        class TestRepo(BaseRepository):
            def __init__(self, session):
                super().__init__(session, TestModel)

        repo = TestRepo(mock_db)

        # Get first page
        pagination = PaginationParams(page=1, per_page=50)
        result = await repo.get_all(pagination=pagination)

        assert result.total == 100
        assert len(result.items) == 50
        assert result.pages == 2
