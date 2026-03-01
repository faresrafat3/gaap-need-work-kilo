"""Provider Model

Provider configurations and usage statistics.
"""

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


class ProviderConfig(Base):
    """Provider configuration storage."""

    __tablename__ = "provider_configs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    name = Column(String(100), nullable=False)
    provider_type = Column(String(50), nullable=False, index=True)
    api_key_encrypted = Column(Text, nullable=True)
    base_url = Column(String(500), nullable=True)
    models = Column(JSON, default=list, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_default = Column(Boolean, default=False, nullable=False)
    rate_limit = Column(Integer, default=60, nullable=False)
    timeout = Column(Integer, default=120, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)
    retry_delay = Column(Float, default=1.0, nullable=False)
    cost_per_1k_input = Column(Float, default=0.0, nullable=False)
    cost_per_1k_output = Column(Float, default=0.0, nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)

    __table_args__ = (
        Index("ix_provider_user_type", "user_id", "provider_type"),
        Index("ix_provider_active", "is_active"),
        Index("ix_provider_default", "user_id", "is_default"),
    )

    def __repr__(self) -> str:
        return f"<ProviderConfig(id={self.id}, name={self.name}, type={self.provider_type})>"


class ProviderUsage(Base):
    """Provider usage statistics."""

    __tablename__ = "provider_usage"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    provider_config_id = Column(
        ForeignKey("provider_configs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    requests_count = Column(Integer, default=0, nullable=False)
    tokens_input = Column(Integer, default=0, nullable=False)
    tokens_output = Column(Integer, default=0, nullable=False)
    tokens_total = Column(Integer, default=0, nullable=False)
    cost_usd = Column(Float, default=0.0, nullable=False)
    latency_avg_ms = Column(Float, default=0.0, nullable=False)
    latency_p95_ms = Column(Float, default=0.0, nullable=False)
    errors_count = Column(Integer, default=0, nullable=False)
    success_rate = Column(Float, default=1.0, nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)

    __table_args__ = (Index("ix_usage_provider_date", "provider_config_id", "date"),)

    def __repr__(self) -> str:
        return f"<ProviderUsage(id={self.id}, date={self.date}, cost={self.cost_usd:.4f})>"
