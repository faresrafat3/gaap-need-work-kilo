"""Memory Model

Hierarchical memory storage for episodic, semantic, and procedural memories.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)

from gaap.db.models.base import Base
from gaap.db.models.mixins import generate_uuid


class MemoryTier(str, enum.Enum):
    """Memory tier enumeration."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryPriority(str, enum.Enum):
    """Memory priority enumeration."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class MemoryEntry(Base):
    """Base memory entry model."""

    __tablename__ = "memory_entries"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    tier = Column(String(20), nullable=False, index=True)
    content = Column(JSON, nullable=False)
    priority = Column(String(20), default=MemoryPriority.NORMAL.value, nullable=False)
    importance = Column(Float, default=1.0, nullable=False)
    decay_rate = Column(Float, default=0.1, nullable=False)
    access_count = Column(Integer, default=0, nullable=False)
    tags = Column(JSON, default=list, nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False)
    embedding = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_memory_user_tier", "user_id", "tier"),
        Index("ix_memory_accessed", "last_accessed"),
        Index("ix_memory_importance", "importance"),
    )

    def __repr__(self) -> str:
        return f"<MemoryEntry(id={self.id}, tier={self.tier}, importance={self.importance})>"


class EpisodicMemoryEntry(Base):
    """Episodic memory entry for event history."""

    __tablename__ = "episodic_memories"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    task_id = Column(String(100), nullable=False, index=True)
    action = Column(String(500), nullable=False)
    result = Column(Text, nullable=True)
    success = Column(Boolean, default=True, nullable=False, index=True)
    category = Column(String(50), default="general", nullable=False)
    duration_ms = Column(Float, default=0.0, nullable=False)
    tokens_used = Column(Integer, default=0, nullable=False)
    cost_usd = Column(Float, default=0.0, nullable=False)
    model = Column(String(100), default="unknown", nullable=False)
    provider = Column(String(50), default="unknown", nullable=False)
    lessons = Column(JSON, default=list, nullable=False)
    context = Column(JSON, default=dict, nullable=False)

    __table_args__ = (
        Index("ix_episodic_user_task", "user_id", "task_id"),
        Index("ix_episodic_category", "category"),
        Index("ix_episodic_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<EpisodicMemory(id={self.id}, task={self.task_id}, success={self.success})>"


class SemanticRuleEntry(Base):
    """Semantic rule entry for extracted patterns."""

    __tablename__ = "semantic_rules"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    condition = Column(String(1000), nullable=False)
    action = Column(String(1000), nullable=False)
    confidence = Column(Float, default=0.5, nullable=False)
    support_count = Column(Integer, default=1, nullable=False)
    source_episodes = Column(JSON, default=list, nullable=False)
    category = Column(String(50), default="general", nullable=False)
    tags = Column(JSON, default=list, nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)

    __table_args__ = (
        Index("ix_semantic_user_category", "user_id", "category"),
        Index("ix_semantic_confidence", "confidence"),
    )

    def __repr__(self) -> str:
        return f"<SemanticRule(id={self.id}, confidence={self.confidence})>"


class ProceduralMemoryEntry(Base):
    """Procedural memory entry for skills and templates."""

    __tablename__ = "procedural_memories"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    prompt_template = Column(Text, nullable=False)
    success_rate = Column(Float, default=0.0, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    examples = Column(JSON, default=list, nullable=False)
    parameters = Column(JSON, default=dict, nullable=False)
    tags = Column(JSON, default=list, nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)

    __table_args__ = (
        Index("ix_procedural_user_name", "user_id", "name"),
        Index("ix_procedural_success", "success_rate"),
    )

    def __repr__(self) -> str:
        return f"<ProceduralMemory(id={self.id}, name={self.name}, success={self.success_rate})>"
