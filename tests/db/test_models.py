"""Tests for database models."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gaap.db.models import (
    APIKey,
    Message,
    MessageRole,
    Session,
    SessionStatus,
    User,
)


class TestUserModel:
    """Tests for User model."""

    @pytest.mark.asyncio
    async def test_create_user(self, db_session: AsyncSession):
        """Test creating a user."""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed_password_123",
        )
        db_session.add(user)
        await db_session.commit()

        # Verify user was created
        result = await db_session.execute(select(User).where(User.email == "test@example.com"))
        saved_user = result.scalar_one()

        assert saved_user.email == "test@example.com"
        assert saved_user.username == "testuser"
        assert saved_user.is_active is True
        assert saved_user.id is not None

    @pytest.mark.asyncio
    async def test_user_unique_email(self, db_session: AsyncSession):
        """Test that email must be unique."""
        user1 = User(
            email="unique@example.com",
            username="user1",
            hashed_password="pass1",
        )
        db_session.add(user1)
        await db_session.commit()

        # Try to create another user with same email
        user2 = User(
            email="unique@example.com",
            username="user2",
            hashed_password="pass2",
        )
        db_session.add(user2)

        with pytest.raises(Exception):  # IntegrityError
            await db_session.commit()


class TestSessionModel:
    """Tests for Session model."""

    @pytest.mark.asyncio
    async def test_create_session(self, db_session: AsyncSession):
        """Test creating a session."""
        # First create a user
        user = User(
            email="session_test@example.com",
            username="sessiontest",
            hashed_password="pass",
        )
        db_session.add(user)
        await db_session.flush()

        # Create session
        session = Session(
            user_id=user.id,
            title="Test Session",
            status=SessionStatus.ACTIVE.value,
        )
        db_session.add(session)
        await db_session.commit()

        # Verify
        result = await db_session.execute(select(Session).where(Session.id == session.id))
        saved_session = result.scalar_one()

        assert saved_session.title == "Test Session"
        assert saved_session.user_id == user.id
        assert saved_session.message_count == 0


class TestMessageModel:
    """Tests for Message model."""

    @pytest.mark.asyncio
    async def test_create_message(self, db_session: AsyncSession):
        """Test creating a message."""
        # Create user and session
        user = User(
            email="msg_test@example.com",
            username="msgtest",
            hashed_password="pass",
        )
        db_session.add(user)
        await db_session.flush()

        session = Session(user_id=user.id, title="Test")
        db_session.add(session)
        await db_session.flush()

        # Create message
        message = Message(
            session_id=session.id,
            role=MessageRole.USER.value,
            content="Hello, world!",
            sequence=1,
        )
        db_session.add(message)
        await db_session.commit()

        # Verify
        result = await db_session.execute(select(Message).where(Message.id == message.id))
        saved_message = result.scalar_one()

        assert saved_message.content == "Hello, world!"
        assert saved_message.role == MessageRole.USER.value
        assert saved_message.total_tokens == 0


class TestAPIKeyModel:
    """Tests for APIKey model."""

    @pytest.mark.asyncio
    async def test_create_api_key(self, db_session: AsyncSession):
        """Test creating an API key."""
        # Create user
        user = User(
            email="apikey_test@example.com",
            username="apikeytest",
            hashed_password="pass",
        )
        db_session.add(user)
        await db_session.flush()

        # Create API key
        api_key = APIKey(
            user_id=user.id,
            key_hash="hash123",
            encrypted_key="encrypted_value",
            name="Test Key",
        )
        db_session.add(api_key)
        await db_session.commit()

        # Verify
        result = await db_session.execute(select(APIKey).where(APIKey.id == api_key.id))
        saved_key = result.scalar_one()

        assert saved_key.name == "Test Key"
        assert saved_key.is_active is True
        assert saved_key.rate_limit == 60
