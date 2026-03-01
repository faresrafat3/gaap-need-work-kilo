"""
Comprehensive tests for gaap/api/chat.py module
Tests chat endpoint, streaming, fallback, and database integration
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Mock database dependencies before import
pytestmark = pytest.mark.asyncio


class TestChatRequest:
    """Test ChatRequest model validation"""

    def test_valid_request(self):
        """Test valid chat request"""
        from gaap.api.chat import ChatRequest

        request = ChatRequest(message="Hello, how are you?")
        assert request.message == "Hello, how are you?"
        assert request.provider == "kimi"
        assert request.store_in_db is True

    def test_message_validation_min_length(self):
        """Test message minimum length validation"""
        from gaap.api.chat import ChatRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_message_validation_max_length(self):
        """Test message maximum length validation"""
        from gaap.api.chat import ChatRequest
        from pydantic import ValidationError

        long_message = "x" * 50001
        with pytest.raises(ValidationError):
            ChatRequest(message=long_message)

    def test_provider_validation_default(self):
        """Test provider defaults to kimi"""
        from gaap.api.chat import ChatRequest

        request = ChatRequest(message="Hello")
        assert request.provider == "kimi"

    def test_provider_validation_unknown_provider(self):
        """Test unknown provider defaults to kimi"""
        from gaap.api.chat import ChatRequest

        request = ChatRequest(message="Hello", provider="unknown_provider")
        assert request.provider == "kimi"

    def test_sanitize_message_removes_control_chars(self):
        """Test message sanitization"""
        from gaap.api.chat import ChatRequest

        request = ChatRequest(message="Hello\x00world\x01")
        assert "\x00" not in request.message
        assert "\x01" not in request.message

    def test_context_validation_size_limit(self):
        """Test context size validation"""
        from gaap.api.chat import ChatRequest
        from pydantic import ValidationError

        large_context = {"data": "x" * 100001}
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello", context=large_context)


class TestChatResponse:
    """Test ChatResponse model"""

    def test_response_defaults(self):
        """Test response default values"""
        from gaap.api.chat import ChatResponse

        response = ChatResponse(response="Hello!")
        assert response.response == "Hello!"
        assert response.tokens_used == 0
        assert response.latency_ms == 0.0
        assert response.session_id is None
        assert response.message_id is None


class TestCachedProvider:
    """Test CachedProvider dataclass"""

    def test_cached_provider_creation(self):
        """Test creating cached provider"""
        from gaap.api.chat import CachedProvider

        mock_provider = Mock()
        cached = CachedProvider(provider=mock_provider)

        assert cached.provider is mock_provider
        assert cached.access_count == 0
        assert cached.created_at is not None
        assert cached.last_accessed is not None


class TestAuditLogger:
    """Test AuditLogger functionality"""

    def test_audit_logger_creation(self):
        """Test audit logger creation"""
        from gaap.api.chat import AuditLogger

        logger = AuditLogger()
        assert logger is not None

    def test_hash_sensitive(self):
        """Test sensitive data hashing"""
        from gaap.api.chat import AuditLogger

        logger = AuditLogger()
        hashed = logger._hash_sensitive("sensitive_data")

        assert len(hashed) == 16
        assert hashed == hashlib.sha256("sensitive_data".encode()).hexdigest()[:16]

    def test_log_request_success(self):
        """Test logging successful request"""
        from gaap.api.chat import AuditLogger, ChatRequest

        logger = AuditLogger()
        request = ChatRequest(message="Hello")

        # Should not raise
        logger.log_request(
            request=request,
            client_ip="127.0.0.1",
            success=True,
            response_time_ms=100.0,
            provider_used="kimi",
        )

    def test_log_request_failure(self):
        """Test logging failed request"""
        from gaap.api.chat import AuditLogger, ChatRequest

        logger = AuditLogger()
        request = ChatRequest(message="Hello")

        # Should not raise
        logger.log_request(
            request=request,
            client_ip="127.0.0.1",
            success=False,
            response_time_ms=100.0,
            error="Test error",
        )


class TestProviderCache:
    """Test provider cache functionality"""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset provider cache between tests"""
        from gaap.api.chat import _provider_cache

        _provider_cache.clear()
        yield
        _provider_cache.clear()

    def test_get_provider_invalid_name_length(self):
        """Test get_provider with invalid name length"""
        from gaap.api.chat import get_provider, MAX_PROVIDER_NAME_LENGTH

        long_name = "x" * (MAX_PROVIDER_NAME_LENGTH + 1)
        with pytest.raises(ValueError):
            get_provider(long_name)

    def test_get_provider_empty_name(self):
        """Test get_provider with empty name"""
        from gaap.api.chat import get_provider

        with pytest.raises(ValueError):
            get_provider("")

    @patch("gaap.api.chat.create_kimi_provider")
    def test_get_provider_creates_new(self, mock_create):
        """Test get_provider creates new provider"""
        from gaap.api.chat import get_provider, _provider_cache

        mock_provider = Mock()
        mock_create.return_value = mock_provider

        provider = get_provider("kimi")

        assert "kimi" in _provider_cache
        mock_create.assert_called_once()

    @patch("gaap.api.chat.create_kimi_provider")
    def test_get_provider_returns_cached(self, mock_create):
        """Test get_provider returns cached provider"""
        from gaap.api.chat import get_provider, _provider_cache

        mock_provider = Mock()
        mock_create.return_value = mock_provider

        # First call creates
        provider1 = get_provider("kimi")
        # Second call returns cached
        provider2 = get_provider("kimi")

        assert provider1 is provider2
        mock_create.assert_called_once()

    @patch("gaap.api.chat.create_kimi_provider")
    def test_get_provider_updates_access_stats(self, mock_create):
        """Test get_provider updates access stats"""
        from gaap.api.chat import get_provider, _provider_cache

        mock_provider = Mock()
        mock_create.return_value = mock_provider

        get_provider("kimi")
        initial_access_count = _provider_cache["kimi"].access_count

        get_provider("kimi")
        assert _provider_cache["kimi"].access_count == initial_access_count + 1

    @patch("gaap.api.chat.create_kimi_provider")
    def test_cleanup_expired_cache(self, mock_create):
        """Test cleanup of expired cache entries"""
        from gaap.api.chat import get_provider, _provider_cache, CACHE_TTL_SECONDS

        mock_provider = Mock()
        mock_create.return_value = mock_provider

        get_provider("kimi")

        # Manually expire the cache
        _provider_cache["kimi"].last_accessed = time.time() - CACHE_TTL_SECONDS - 1

        # Create another provider to trigger cleanup
        with patch("gaap.api.chat.create_deepseek_provider") as mock_deepseek:
            mock_deepseek.return_value = Mock()
            get_provider("deepseek")

        assert "kimi" not in _provider_cache

    def test_get_fallback_providers(self):
        """Test getting fallback providers"""
        from gaap.api.chat import get_fallback_providers

        fallbacks = get_fallback_providers("kimi")

        assert "kimi" in fallbacks
        assert "deepseek" in fallbacks
        assert "glm" in fallbacks

    def test_get_fallback_providers_excludes_primary(self):
        """Test fallback excludes primary from non-primary positions"""
        from gaap.api.chat import get_fallback_providers

        fallbacks = get_fallback_providers("kimi")

        # Primary should be first
        assert fallbacks[0] == "kimi"
        # Primary should only appear once
        assert fallbacks.count("kimi") == 1


class TestSessionHelpers:
    """Test session helper functions"""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        session = AsyncMock()
        return session

    async def test_get_or_create_session_new(self, mock_db):
        """Test creating new session"""
        from gaap.api.chat import _get_or_create_session

        with patch("gaap.api.chat.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=None)
            mock_repo.create = AsyncMock(return_value=Mock(id="test_session_id"))
            mock_repo_class.return_value = mock_repo

            session, is_new = await _get_or_create_session(mock_db, None, "Hello")

            assert is_new is True
            assert session.id == "test_session_id"

    async def test_get_or_create_session_existing(self, mock_db):
        """Test getting existing session"""
        from gaap.api.chat import _get_or_create_session

        existing_session = Mock(id="existing_id")

        with patch("gaap.api.chat.SessionRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.get = AsyncMock(return_value=existing_session)
            mock_repo_class.return_value = mock_repo

            session, is_new = await _get_or_create_session(mock_db, "existing_id", "Hello")

            assert is_new is False
            assert session.id == "existing_id"

    async def test_store_user_message(self, mock_db):
        """Test storing user message"""
        from gaap.api.chat import _store_user_message

        with (
            patch("gaap.api.chat.MessageRepository") as mock_msg_repo,
            patch("gaap.api.chat.SessionRepository") as mock_sess_repo,
        ):
            mock_msg = Mock()
            mock_msg_repo.return_value.create = AsyncMock(return_value=mock_msg)
            mock_msg_repo.return_value.get_next_sequence = AsyncMock(return_value=1)
            mock_sess_repo.return_value.update_message_stats = AsyncMock()

            result = await _store_user_message(mock_db, "session_id", "Hello")

            assert result is mock_msg
            mock_msg_repo.return_value.create.assert_called_once()

    async def test_store_assistant_message(self, mock_db):
        """Test storing assistant message"""
        from gaap.api.chat import _store_assistant_message

        with (
            patch("gaap.api.chat.MessageRepository") as mock_msg_repo,
            patch("gaap.api.chat.SessionRepository") as mock_sess_repo,
        ):
            mock_msg = Mock()
            mock_msg_repo.return_value.create = AsyncMock(return_value=mock_msg)
            mock_msg_repo.return_value.get_next_sequence = AsyncMock(return_value=2)
            mock_sess_repo.return_value.update_message_stats = AsyncMock()

            result = await _store_assistant_message(
                mock_db, "session_id", "Response", "kimi", "model", 100.0, 10
            )

            assert result is mock_msg
            mock_msg_repo.return_value.create.assert_called_once()


class TestChatEndpoint:
    """Test main chat endpoint"""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        return AsyncMock()

    @pytest.fixture
    def mock_request(self):
        """Create mock HTTP request"""
        request = Mock()
        request.client.host = "127.0.0.1"
        return request

    async def test_chat_success(self, mock_db, mock_request):
        """Test successful chat request"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello")

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
            patch("gaap.api.chat._store_assistant_message") as mock_store_assistant,
        ):
            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Hello!"}}],
                    "usage": {"total_tokens": 10},
                }
            )
            mock_get_provider.return_value = mock_provider

            mock_session = Mock(id="test_session_id")
            mock_get_session.return_value = (mock_session, True)

            mock_msg = Mock(id="msg_id")
            mock_store_user.return_value = mock_msg
            mock_store_assistant.return_value = mock_msg

            response = await chat(request, mock_request, mock_db)

            assert response.response == "Hello!"
            assert response.session_id == "test_session_id"
            assert response.tokens_used == 10

    async def test_chat_with_fallback(self, mock_db, mock_request):
        """Test chat with provider fallback"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello")

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
        ):
            # First provider fails, second succeeds
            mock_provider1 = Mock()
            mock_provider1._make_request = AsyncMock(side_effect=Exception("Failed"))

            mock_provider2 = Mock()
            mock_provider2._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Fallback response!"}}],
                    "usage": {"total_tokens": 15},
                }
            )

            mock_get_provider.side_effect = [mock_provider1, mock_provider2]

            mock_session = Mock(id="test_session_id")
            mock_get_session.return_value = (mock_session, True)
            mock_store_user.return_value = Mock(id="msg_id")

            response = await chat(request, mock_request, mock_db)

            assert response.response == "Fallback response!"
            assert response.tokens_used == 15

    async def test_chat_all_providers_fail(self, mock_db, mock_request):
        """Test chat when all providers fail"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello")

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
        ):
            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(side_effect=Exception("Failed"))
            mock_get_provider.return_value = mock_provider

            mock_session = Mock(id="test_session_id")
            mock_get_session.return_value = (mock_session, True)
            mock_store_user.return_value = Mock(id="msg_id")

            with pytest.raises(HTTPException) as exc_info:
                await chat(request, mock_request, mock_db)

            assert exc_info.value.status_code == 503

    async def test_chat_timeout(self, mock_db, mock_request):
        """Test chat timeout handling"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello")

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
            patch("gaap.api.chat.get_fallback_providers") as mock_fallbacks,
        ):
            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_get_provider.return_value = mock_provider

            mock_fallbacks.return_value = ["kimi"]

            mock_session = Mock(id="test_session_id")
            mock_get_session.return_value = (mock_session, True)
            mock_store_user.return_value = Mock(id="msg_id")

            with pytest.raises(HTTPException) as exc_info:
                await chat(request, mock_request, mock_db)

            assert exc_info.value.status_code == 503

    async def test_chat_no_db_storage(self, mock_db, mock_request):
        """Test chat without database storage"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello", store_in_db=False)

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
            patch("gaap.api.chat._store_assistant_message") as mock_store_assistant,
        ):
            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Hello!"}}],
                    "usage": {"total_tokens": 10},
                }
            )
            mock_get_provider.return_value = mock_provider

            response = await chat(request, mock_request, mock_db)

            assert response.response == "Hello!"
            # Should not call DB storage
            mock_get_session.assert_not_called()
            mock_store_user.assert_not_called()
            mock_store_assistant.assert_not_called()


class TestListProvidersEndpoint:
    """Test list providers endpoint"""

    async def test_list_providers(self):
        """Test listing providers"""
        from gaap.api.chat import list_providers, _provider_cache, CachedProvider

        # Clear and setup cache
        _provider_cache.clear()
        mock_provider = Mock()
        _provider_cache["kimi"] = CachedProvider(provider=mock_provider)

        result = await list_providers()

        assert isinstance(result, list)
        assert len(result) == 3  # kimi, deepseek, glm

        # Find kimi provider
        kimi = next(p for p in result if p["name"] == "kimi")
        assert kimi["status"] == "available"
        assert kimi["is_default"] is True


class TestCacheStatsEndpoint:
    """Test cache stats endpoint"""

    async def test_get_cache_stats(self):
        """Test getting cache stats"""
        from gaap.api.chat import get_cache_stats, _provider_cache, CachedProvider

        _provider_cache.clear()
        mock_provider = Mock()
        _provider_cache["kimi"] = CachedProvider(provider=mock_provider)
        _provider_cache["kimi"].access_count = 5

        result = await get_cache_stats()

        assert result["total_cached"] == 1
        assert result["cache_ttl_seconds"] == 900
        assert "providers" in result
        assert "kimi" in result["providers"]


class TestClearCacheEndpoint:
    """Test clear cache endpoint"""

    async def test_clear_cache(self):
        """Test clearing cache"""
        from gaap.api.chat import clear_cache, _provider_cache, CachedProvider

        _provider_cache.clear()
        mock_provider = Mock()
        _provider_cache["kimi"] = CachedProvider(provider=mock_provider)

        result = await clear_cache()

        assert result["status"] == "cleared"
        assert result["entries_removed"] == "1"
        assert len(_provider_cache) == 0


class TestGetChatHistoryEndpoint:
    """Test get chat history endpoint"""

    async def test_get_chat_history(self):
        """Test getting chat history"""
        from gaap.api.chat import get_chat_history

        mock_db = AsyncMock()

        with patch("gaap.api.chat.MessageRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_msg = Mock(
                id="msg1",
                role="user",
                content="Hello",
                created_at=datetime.now(),
                total_tokens=10,
                provider="kimi",
                model="model1",
            )
            mock_repo.return_value.get_conversation_history = AsyncMock(return_value=[mock_msg])
            mock_repo_class.return_value = mock_repo

            result = await get_chat_history("session_id", limit=50, db=mock_db)

            assert len(result) == 1
            assert result[0]["id"] == "msg1"
            assert result[0]["role"] == "user"

    async def test_get_chat_history_error(self):
        """Test getting chat history with error"""
        from gaap.api.chat import get_chat_history

        mock_db = AsyncMock()

        with patch("gaap.api.chat.MessageRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.return_value.get_conversation_history = AsyncMock(
                side_effect=Exception("DB error")
            )
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException) as exc_info:
                await get_chat_history("session_id", limit=50, db=mock_db)

            assert exc_info.value.status_code == 500


class TestChatEdgeCases:
    """Test edge cases"""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        return AsyncMock()

    @pytest.fixture
    def mock_request(self):
        """Create mock HTTP request"""
        request = Mock()
        request.client.host = "127.0.0.1"
        return request

    async def test_chat_missing_choices(self, mock_db, mock_request):
        """Test chat response without choices"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello")

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
        ):
            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(return_value={})
            mock_get_provider.return_value = mock_provider

            mock_session = Mock(id="test_session_id")
            mock_get_session.return_value = (mock_session, True)
            mock_store_user.return_value = Mock(id="msg_id")

            with pytest.raises(HTTPException) as exc_info:
                await chat(request, mock_request, mock_db)

            assert exc_info.value.status_code == 503

    async def test_chat_empty_choices(self, mock_db, mock_request):
        """Test chat response with empty choices"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello")

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
        ):
            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(return_value={"choices": []})
            mock_get_provider.return_value = mock_provider

            mock_session = Mock(id="test_session_id")
            mock_get_session.return_value = (mock_session, True)
            mock_store_user.return_value = Mock(id="msg_id")

            with pytest.raises(HTTPException) as exc_info:
                await chat(request, mock_request, mock_db)

            assert exc_info.value.status_code == 503

    async def test_chat_response_without_usage(self, mock_db, mock_request):
        """Test chat response without usage data"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello")

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
            patch("gaap.api.chat._store_user_message") as mock_store_user,
            patch("gaap.api.chat._store_assistant_message") as mock_store_assistant,
        ):
            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Hello!"}}]
                    # No usage field
                }
            )
            mock_get_provider.return_value = mock_provider

            mock_session = Mock(id="test_session_id")
            mock_get_session.return_value = (mock_session, True)
            mock_store_user.return_value = Mock(id="msg_id")
            mock_store_assistant.return_value = Mock(id="msg_id")

            response = await chat(request, mock_request, mock_db)

            assert response.response == "Hello!"
            # Should use word count as fallback
            assert response.tokens_used > 0

    async def test_chat_db_storage_failure(self, mock_db, mock_request):
        """Test chat when DB storage fails"""
        from gaap.api.chat import chat, ChatRequest

        request = ChatRequest(message="Hello", store_in_db=True)

        with (
            patch("gaap.api.chat.get_provider") as mock_get_provider,
            patch("gaap.api.chat._get_or_create_session") as mock_get_session,
        ):
            mock_provider = Mock()
            mock_provider._make_request = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "Hello!"}}],
                    "usage": {"total_tokens": 10},
                }
            )
            mock_get_provider.return_value = mock_provider

            # DB fails
            mock_get_session.side_effect = Exception("DB error")

            # Should still work without DB
            response = await chat(request, mock_request, mock_db)

            assert response.response == "Hello!"

    def test_provider_cache_with_glm(self):
        """Test creating glm provider"""
        from gaap.api.chat import get_provider

        with patch("gaap.api.chat.create_glm_provider") as mock_create:
            mock_create.return_value = Mock()
            provider = get_provider("glm")
            mock_create.assert_called_once()

    def test_provider_cache_with_deepseek(self):
        """Test creating deepseek provider"""
        from gaap.api.chat import get_provider

        with patch("gaap.api.chat.create_deepseek_provider") as mock_create:
            mock_create.return_value = Mock()
            provider = get_provider("deepseek")
            mock_create.assert_called_once()

    def test_request_with_context(self):
        """Test request with context"""
        from gaap.api.chat import ChatRequest

        context = {"key": "value", "nested": {"data": "test"}}
        request = ChatRequest(message="Hello", context=context)

        assert request.context == context

    def test_request_with_session_id(self):
        """Test request with session ID"""
        from gaap.api.chat import ChatRequest

        request = ChatRequest(message="Hello", session_id="session_123")

        assert request.session_id == "session_123"
