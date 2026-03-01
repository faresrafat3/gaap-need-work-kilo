"""Session Model

Chat session management with metadata and status tracking.
"""

import enum

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
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


class SessionStatus(str, enum.Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DELETED = "deleted"


class SessionPriority(str, enum.Enum):
    """Session priority enumeration."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Session(Base):
    """Chat session model."""

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String(500), default="New Chat", nullable=False)
    status = Column(String(20), default=SessionStatus.ACTIVE.value, nullable=False, index=True)
    priority = Column(String(20), default=SessionPriority.NORMAL.value, nullable=False)
    description = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict, nullable=False)
    context = Column(JSON, default=dict, nullable=False)
    tags = Column(JSON, default=list, nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    total_cost = Column(Float, default=0.0, nullable=False)
    last_message_at = Column(DateTime, nullable=True, index=True)
    archived_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_sessions_user_status", "user_id", "status"),
        Index("ix_sessions_last_message", "user_id", "last_message_at"),
        Index("ix_sessions_archived", "archived_at"),
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, title={self.title[:30]}, status={self.status})>"
