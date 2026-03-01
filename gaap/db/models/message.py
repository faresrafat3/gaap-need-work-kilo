"""Message Model

Chat message storage with role, content, and token tracking.
"""

import enum

from sqlalchemy import (
    JSON,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from gaap.db.models.base import Base
from gaap.db.models.mixins import generate_uuid


class MessageRole(str, enum.Enum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class MessageStatus(str, enum.Enum):
    """Message status enumeration."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Message(Base):
    """Chat message model."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False, index=True)
    content = Column(Text, nullable=False)
    name = Column(String(100), nullable=True)
    tool_calls = Column(JSON, nullable=True)
    tool_call_id = Column(String(100), nullable=True)
    status = Column(String(20), default=MessageStatus.COMPLETED.value, nullable=False)
    sequence = Column(Integer, default=0, nullable=False, index=True)
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    cost_usd = Column(Float, default=0.0, nullable=False)
    latency_ms = Column(Float, default=0.0, nullable=False)
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    metadata_ = Column("metadata", JSON, default=dict, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_sequence", "session_id", "sequence"),
        Index("ix_messages_session_role", "session_id", "role"),
        Index("ix_messages_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        content_preview = self.content[:50] if self.content else ""
        return f"<Message(id={self.id}, role={self.role}, content={content_preview}...)>"
