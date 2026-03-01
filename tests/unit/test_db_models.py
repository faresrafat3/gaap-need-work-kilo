"""
Comprehensive tests for gaap/db/models/ module
Tests all SQLAlchemy models including User, Session, Message, Memory, and Provider
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DBSession, sessionmaker

# We need to create a test database
from gaap.db import Base
from gaap.db.models import (
    APIKey,
    EpisodicMemoryEntry,
    MemoryEntry,
    MemoryTier,
    Message,
    MessageRole,
    ProceduralMemoryEntry,
    ProviderConfig,
    ProviderUsage,
    SemanticRuleEntry,
    Session,
    SessionPriority,
    SessionStatus,
    User,
    UserPreference,
)
from gaap.db.models.mixins import TimestampMixin, generate_uuid


class TestGenerateUuid:
    """Test generate_uuid function"""

    def test_generate_uuid_returns_string(self):
        """Test generate_uuid returns a string"""
        result = generate_uuid()
        assert isinstance(result, str)

    def test_generate_uuid_is_unique(self):
        """Test generate_uuid produces unique values"""
        uuids = [generate_uuid() for _ in range(100)]
        assert len(set(uuids)) == 100

    def test_generate_uuid_format(self):
        """Test generate_uuid format"""
        result = generate_uuid()
        # Should be a valid UUID string (36 chars with hyphens)
        assert len(result) == 36
        uuid.UUID(result)  # Should not raise


class TestTimestampMixin:
    """Test TimestampMixin"""

    def test_mixin_has_created_at(self):
        """Test TimestampMixin has created_at field"""

        # Create a test model with TimestampMixin
        class TestModel(Base):
            __tablename__ = "test_timestamp"
            id = Column(String(36), primary_key=True, default=generate_uuid)

        assert hasattr(TestModel, "created_at")
        assert hasattr(TestModel, "updated_at")


class TestUserModel:
    """Test User model"""

    def test_user_creation(self):
        """Test creating a user"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed_password",
        )

        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.hashed_password == "hashed_password"
        assert user.role == "user"
        assert user.status == "pending"
        assert user.is_active is True
        assert user.email_verified is False

    def test_user_repr(self):
        """Test User repr"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
        )
        repr_str = repr(user)

        assert "User" in repr_str
        assert "test@example.com" in repr_str

    def test_user_default_values(self):
        """Test User default values"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
        )

        assert user.metadata == {}
        assert user.last_login is None
        assert user.full_name is None
        assert user.avatar_url is None

    def test_user_roles(self):
        """Test user roles"""
        from gaap.db.models.user import UserRole

        assert UserRole.ADMIN == "admin"
        assert UserRole.USER == "user"
        assert UserRole.GUEST == "guest"

    def test_user_statuses(self):
        """Test user statuses"""
        from gaap.db.models.user import UserStatus

        assert UserStatus.ACTIVE == "active"
        assert UserStatus.INACTIVE == "inactive"
        assert UserStatus.SUSPENDED == "suspended"
        assert UserStatus.PENDING == "pending"


class TestAPIKeyModel:
    """Test APIKey model"""

    def test_api_key_creation(self):
        """Test creating an API key"""
        api_key = APIKey(
            user_id="user123",
            key_hash="hash123",
            encrypted_key="encrypted_secret",
        )

        assert api_key.user_id == "user123"
        assert api_key.key_hash == "hash123"
        assert api_key.encrypted_key == "encrypted_secret"
        assert api_key.name == "API Key"
        assert api_key.is_active is True
        assert api_key.rate_limit == 60

    def test_api_key_default_permissions(self):
        """Test API key default permissions"""
        api_key = APIKey(
            user_id="user123",
            key_hash="hash",
            encrypted_key="secret",
        )

        assert api_key.permissions == {}

    def test_api_key_repr(self):
        """Test APIKey repr"""
        api_key = APIKey(
            user_id="user123",
            key_hash="hash",
            encrypted_key="secret",
            name="Test Key",
        )
        repr_str = repr(api_key)

        assert "APIKey" in repr_str
        assert "Test Key" in repr_str


class TestUserPreferenceModel:
    """Test UserPreference model"""

    def test_preference_creation(self):
        """Test creating user preferences"""
        pref = UserPreference(
            user_id="user123",
        )

        assert pref.user_id == "user123"
        assert pref.theme == "system"
        assert pref.language == "en"
        assert pref.timezone == "UTC"
        assert pref.max_tokens == 4096
        assert pref.temperature == 70
        assert pref.notifications_enabled is True
        assert pref.auto_save is True

    def test_preference_defaults(self):
        """Test preference defaults"""
        pref = UserPreference(user_id="user123")

        assert pref.default_model is None
        assert pref.preferences == {}

    def test_preference_repr(self):
        """Test UserPreference repr"""
        pref = UserPreference(user_id="user123", theme="dark")
        repr_str = repr(pref)

        assert "UserPreference" in repr_str
        assert "dark" in repr_str


class TestSessionModel:
    """Test Session model"""

    def test_session_creation(self):
        """Test creating a session"""
        session = Session(
            title="Test Session",
            status="active",
            priority="normal",
        )

        assert session.title == "Test Session"
        assert session.status == "active"
        assert session.priority == "normal"
        assert session.message_count == 0
        assert session.total_tokens == 0
        assert session.total_cost == 0.0

    def test_session_defaults(self):
        """Test session default values"""
        session = Session()

        assert session.title == "New Chat"
        assert session.status == "active"
        assert session.priority == "normal"
        assert session.metadata == {}
        assert session.context == {}
        assert session.tags == []
        assert session.description is None
        assert session.last_message_at is None
        assert session.archived_at is None
        assert session.expires_at is None

    def test_session_statuses(self):
        """Test session statuses"""
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.PAUSED == "paused"
        assert SessionStatus.ARCHIVED == "archived"
        assert SessionStatus.DELETED == "deleted"

    def test_session_priorities(self):
        """Test session priorities"""
        assert SessionPriority.LOW == "low"
        assert SessionPriority.NORMAL == "normal"
        assert SessionPriority.HIGH == "high"
        assert SessionPriority.CRITICAL == "critical"

    def test_session_repr(self):
        """Test Session repr"""
        session = Session(title="Test Session")
        repr_str = repr(session)

        assert "Session" in repr_str
        assert "Test Session" in repr_str

    def test_session_repr_long_title(self):
        """Test Session repr with long title"""
        session = Session(title="x" * 100)
        repr_str = repr(session)

        # Should be truncated
        assert len(repr_str) < 150


class TestMessageModel:
    """Test Message model"""

    def test_message_creation(self):
        """Test creating a message"""
        message = Message(
            session_id="session123",
            role="user",
            content="Hello",
        )

        assert message.session_id == "session123"
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.status == "completed"
        assert message.sequence == 0
        assert message.total_tokens == 0

    def test_message_defaults(self):
        """Test message default values"""
        message = Message(
            session_id="session123",
            role="user",
            content="Hello",
        )

        assert message.name is None
        assert message.tool_calls is None
        assert message.tool_call_id is None
        assert message.prompt_tokens == 0
        assert message.completion_tokens == 0
        assert message.cost_usd == 0.0
        assert message.latency_ms == 0.0
        assert message.provider is None
        assert message.model is None
        assert message.metadata == {}

    def test_message_roles(self):
        """Test message roles"""
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.FUNCTION == "function"
        assert MessageRole.TOOL == "tool"

    def test_message_repr(self):
        """Test Message repr"""
        message = Message(
            session_id="session123",
            role="user",
            content="Hello world",
        )
        repr_str = repr(message)

        assert "Message" in repr_str
        assert "user" in repr_str

    def test_message_repr_long_content(self):
        """Test Message repr with long content"""
        message = Message(
            session_id="session123",
            role="user",
            content="x" * 1000,
        )
        repr_str = repr(message)

        # Should be truncated
        assert "..." in repr_str


class TestMemoryEntryModel:
    """Test MemoryEntry model"""

    def test_memory_entry_creation(self):
        """Test creating a memory entry"""
        memory = MemoryEntry(
            user_id="user123",
            content="Test memory",
            tier=MemoryTier.WORKING,
        )

        assert memory.user_id == "user123"
        assert memory.content == "Test memory"
        assert memory.tier == MemoryTier.WORKING
        assert memory.importance == 0.5
        assert memory.confidence == 0.5

    def test_memory_defaults(self):
        """Test memory default values"""
        memory = MemoryEntry(
            user_id="user123",
            content="Test",
        )

        assert memory.session_id is None
        assert memory.embedding_id is None
        assert memory.metadata == {}
        assert memory.source is None
        assert memory.expires_at is None
        assert memory.access_count == 0
        assert memory.last_accessed_at is None

    def test_memory_tiers(self):
        """Test memory tiers"""
        assert MemoryTier.WORKING == "working"
        assert MemoryTier.SHORT_TERM == "short_term"
        assert MemoryTier.LONG_TERM == "long_term"
        assert MemoryTier.EPISODIC == "episodic"
        assert MemoryTier.SEMANTIC == "semantic"


class TestEpisodicMemoryEntryModel:
    """Test EpisodicMemoryEntry model"""

    def test_episodic_memory_creation(self):
        """Test creating episodic memory"""
        memory = EpisodicMemoryEntry(
            user_id="user123",
            content="Test episodic memory",
            session_id="session123",
            event_timestamp=datetime.now(),
        )

        assert memory.user_id == "user123"
        assert memory.content == "Test episodic memory"
        assert memory.session_id == "session123"
        assert memory.tier == MemoryTier.EPISODIC

    def test_episodic_defaults(self):
        """Test episodic memory defaults"""
        memory = EpisodicMemoryEntry(
            user_id="user123",
            content="Test",
            event_timestamp=datetime.now(),
        )

        assert memory.context is None
        assert memory.emotional_valence is None
        assert memory.participants == []


class TestSemanticRuleEntryModel:
    """Test SemanticRuleEntry model"""

    def test_semantic_rule_creation(self):
        """Test creating semantic rule"""
        rule = SemanticRuleEntry(
            user_id="user123",
            content="Test rule",
            rule_type="preference",
        )

        assert rule.user_id == "user123"
        assert rule.content == "Test rule"
        assert rule.rule_type == "preference"
        assert rule.tier == MemoryTier.SEMANTIC

    def test_semantic_rule_defaults(self):
        """Test semantic rule defaults"""
        rule = SemanticRuleEntry(
            user_id="user123",
            content="Test",
            rule_type="fact",
        )

        assert rule.applies_to is None
        assert rule.priority == 1.0


class TestProceduralMemoryEntryModel:
    """Test ProceduralMemoryEntry model"""

    def test_procedural_memory_creation(self):
        """Test creating procedural memory"""
        memory = ProceduralMemoryEntry(
            user_id="user123",
            content="Test procedure",
            procedure_name="test_procedure",
        )

        assert memory.user_id == "user123"
        assert memory.content == "Test procedure"
        assert memory.procedure_name == "test_procedure"
        assert memory.tier == MemoryTier.LONG_TERM

    def test_procedural_memory_defaults(self):
        """Test procedural memory defaults"""
        memory = ProceduralMemoryEntry(
            user_id="user123",
            content="Test",
            procedure_name="test",
        )

        assert memory.steps == []
        assert memory.success_count == 0
        assert memory.failure_count == 0


class TestProviderConfigModel:
    """Test ProviderConfig model"""

    def test_provider_config_creation(self):
        """Test creating provider config"""
        config = ProviderConfig(
            name="openai",
            provider_type="openai",
            api_key_encrypted="encrypted_key",
        )

        assert config.name == "openai"
        assert config.provider_type == "openai"
        assert config.api_key_encrypted == "encrypted_key"
        assert config.is_active is True

    def test_provider_config_defaults(self):
        """Test provider config defaults"""
        config = ProviderConfig(
            name="openai",
            provider_type="openai",
        )

        assert config.api_key_encrypted is None
        assert config.base_url is None
        assert config.default_model is None
        assert config.models == []
        assert config.config == {}
        assert config.priority == 1
        assert config.rate_limit_rpm == 60
        assert config.timeout_seconds == 120


class TestProviderUsageModel:
    """Test ProviderUsage model"""

    def test_provider_usage_creation(self):
        """Test creating provider usage"""
        usage = ProviderUsage(
            provider_config_id="config123",
            date=datetime.now(),
        )

        assert usage.provider_config_id == "config123"
        assert usage.request_count == 0
        assert usage.token_count == 0
        assert usage.cost_usd == 0.0

    def test_provider_usage_defaults(self):
        """Test provider usage defaults"""
        usage = ProviderUsage(
            provider_config_id="config123",
            date=datetime.now(),
        )

        assert usage.error_count == 0
        assert usage.latency_avg_ms == 0.0
        assert usage.latency_p95_ms == 0.0
        assert usage.latency_p99_ms == 0.0


class TestModelRelationships:
    """Test model relationships"""

    def test_user_sessions_relationship(self):
        """Test User sessions relationship"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
        )

        # Check that relationship attribute exists
        assert hasattr(user, "sessions")

    def test_user_api_keys_relationship(self):
        """Test User api_keys relationship"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
        )

        assert hasattr(user, "api_keys")

    def test_user_preferences_relationship(self):
        """Test User preferences relationship"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
        )

        assert hasattr(user, "preferences")

    def test_session_messages_relationship(self):
        """Test Session messages relationship"""
        session = Session(title="Test")

        assert hasattr(session, "messages")

    def test_session_user_relationship(self):
        """Test Session user relationship"""
        session = Session(title="Test")

        assert hasattr(session, "user")

    def test_message_session_relationship(self):
        """Test Message session relationship"""
        message = Message(
            session_id="session123",
            role="user",
            content="Hello",
        )

        assert hasattr(message, "session")


class TestModelEdgeCases:
    """Test model edge cases"""

    def test_user_unicode_email(self):
        """Test user with unicode email"""
        user = User(
            email="用户@例子.com",
            username="user",
            hashed_password="hashed",
        )

        assert user.email == "用户@例子.com"

    def test_message_unicode_content(self):
        """Test message with unicode content"""
        message = Message(
            session_id="session123",
            role="user",
            content="Hello 世界 🌍",
        )

        assert message.content == "Hello 世界 🌍"

    def test_session_long_title(self):
        """Test session with long title"""
        long_title = "x" * 500
        session = Session(title=long_title)

        assert session.title == long_title

    def test_message_empty_content(self):
        """Test message with empty content"""
        message = Message(
            session_id="session123",
            role="user",
            content="",
        )

        assert message.content == ""

    def test_memory_entry_very_long_content(self):
        """Test memory entry with very long content"""
        long_content = "x" * 100000
        memory = MemoryEntry(
            user_id="user123",
            content=long_content,
        )

        assert memory.content == long_content

    def test_api_key_special_characters_in_name(self):
        """Test API key with special characters in name"""
        api_key = APIKey(
            user_id="user123",
            key_hash="hash",
            encrypted_key="secret",
            name="Test Key <script>alert('xss')</script>",
        )

        assert "<script>" in api_key.name

    def test_user_preference_temperature_range(self):
        """Test user preference temperature at boundaries"""
        pref = UserPreference(
            user_id="user123",
            temperature=0,  # Min
        )
        assert pref.temperature == 0

        pref2 = UserPreference(
            user_id="user123",
            temperature=100,  # Max
        )
        assert pref2.temperature == 100


class TestModelTableNames:
    """Test model table names"""

    def test_user_table_name(self):
        """Test User table name"""
        assert User.__tablename__ == "users"

    def test_api_key_table_name(self):
        """Test APIKey table name"""
        assert APIKey.__tablename__ == "api_keys"

    def test_user_preference_table_name(self):
        """Test UserPreference table name"""
        assert UserPreference.__tablename__ == "user_preferences"

    def test_session_table_name(self):
        """Test Session table name"""
        assert Session.__tablename__ == "sessions"

    def test_message_table_name(self):
        """Test Message table name"""
        assert Message.__tablename__ == "messages"

    def test_memory_entry_table_name(self):
        """Test MemoryEntry table name"""
        assert MemoryEntry.__tablename__ == "memory_entries"

    def test_provider_config_table_name(self):
        """Test ProviderConfig table name"""
        assert ProviderConfig.__tablename__ == "provider_configs"

    def test_provider_usage_table_name(self):
        """Test ProviderUsage table name"""
        assert ProviderUsage.__tablename__ == "provider_usage"
