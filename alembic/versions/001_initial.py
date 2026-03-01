"""Initial migration - Create all tables.

Revision ID: 001
Revises:
Create Date: 2026-02-28 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all initial tables."""
    # Users table
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("username", sa.String(100), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, default="user"),
        sa.Column("status", sa.String(20), nullable=False, default="pending"),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("email_verified", sa.Boolean, nullable=False, default=False),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column("full_name", sa.String(200), nullable=True),
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True, default=dict),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # API Keys table
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("encrypted_key", sa.Text, nullable=False),
        sa.Column("name", sa.String(100), nullable=False, default="API Key"),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("last_used", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("permissions", postgresql.JSONB, nullable=False, default=dict),
        sa.Column("rate_limit", sa.Integer, nullable=False, default=60),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # User Preferences table
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("theme", sa.String(20), nullable=False, default="system"),
        sa.Column("language", sa.String(10), nullable=False, default="en"),
        sa.Column("timezone", sa.String(50), nullable=False, default="UTC"),
        sa.Column("default_model", sa.String(100), nullable=True),
        sa.Column("max_tokens", sa.Integer, nullable=False, default=4096),
        sa.Column("temperature", sa.Integer, nullable=False, default=70),
        sa.Column("notifications_enabled", sa.Boolean, nullable=False, default=True),
        sa.Column("auto_save", sa.Boolean, nullable=False, default=True),
        sa.Column("preferences", postgresql.JSONB, nullable=False, default=dict),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Sessions table
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("title", sa.String(500), nullable=False, default="New Chat"),
        sa.Column("status", sa.String(20), nullable=False, default="active"),
        sa.Column("priority", sa.String(20), nullable=False, default="normal"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=False, default=dict),
        sa.Column("context", postgresql.JSONB, nullable=False, default=dict),
        sa.Column("tags", postgresql.JSONB, nullable=False, default=list),
        sa.Column("message_count", sa.Integer, nullable=False, default=0),
        sa.Column("total_tokens", sa.Integer, nullable=False, default=0),
        sa.Column("total_cost", sa.Float, nullable=False, default=0.0),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("name", sa.String(100), nullable=True),
        sa.Column("tool_calls", postgresql.JSONB, nullable=True),
        sa.Column("tool_call_id", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, default="completed"),
        sa.Column("sequence", sa.Integer, nullable=False, default=0),
        sa.Column("prompt_tokens", sa.Integer, nullable=False, default=0),
        sa.Column("completion_tokens", sa.Integer, nullable=False, default=0),
        sa.Column("total_tokens", sa.Integer, nullable=False, default=0),
        sa.Column("cost_usd", sa.Float, nullable=False, default=0.0),
        sa.Column("latency_ms", sa.Float, nullable=False, default=0.0),
        sa.Column("provider", sa.String(50), nullable=True),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=False, default=dict),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Memory Entries table
    op.create_table(
        "memory_entries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column("tier", sa.String(20), nullable=False),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column("priority", sa.String(20), nullable=False, default="normal"),
        sa.Column("importance", sa.Float, nullable=False, default=1.0),
        sa.Column("decay_rate", sa.Float, nullable=False, default=0.1),
        sa.Column("access_count", sa.Integer, nullable=False, default=0),
        sa.Column("tags", postgresql.JSONB, nullable=False, default=list),
        sa.Column("metadata", postgresql.JSONB, nullable=False, default=dict),
        sa.Column(
            "last_accessed",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Episodic Memories table
    op.create_table(
        "episodic_memories",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column("task_id", sa.String(100), nullable=False),
        sa.Column("action", sa.String(500), nullable=False),
        sa.Column("result", sa.Text, nullable=True),
        sa.Column("success", sa.Boolean, nullable=False, default=True),
        sa.Column("category", sa.String(50), nullable=False, default="general"),
        sa.Column("duration_ms", sa.Float, nullable=False, default=0.0),
        sa.Column("tokens_used", sa.Integer, nullable=False, default=0),
        sa.Column("cost_usd", sa.Float, nullable=False, default=0.0),
        sa.Column("model", sa.String(100), nullable=False, default="unknown"),
        sa.Column("provider", sa.String(50), nullable=False, default="unknown"),
        sa.Column("lessons", postgresql.JSONB, nullable=False, default=list),
        sa.Column("context", postgresql.JSONB, nullable=False, default=dict),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Semantic Rules table
    op.create_table(
        "semantic_rules",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column("condition", sa.String(1000), nullable=False),
        sa.Column("action", sa.String(1000), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, default=0.5),
        sa.Column("support_count", sa.Integer, nullable=False, default=1),
        sa.Column("source_episodes", postgresql.JSONB, nullable=False, default=list),
        sa.Column("category", sa.String(50), nullable=False, default="general"),
        sa.Column("tags", postgresql.JSONB, nullable=False, default=list),
        sa.Column("metadata", postgresql.JSONB, nullable=False, default=dict),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Procedural Memories table
    op.create_table(
        "procedural_memories",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("prompt_template", sa.Text, nullable=False),
        sa.Column("success_rate", sa.Float, nullable=False, default=0.0),
        sa.Column("usage_count", sa.Integer, nullable=False, default=0),
        sa.Column("examples", postgresql.JSONB, nullable=False, default=list),
        sa.Column("parameters", postgresql.JSONB, nullable=False, default=dict),
        sa.Column("tags", postgresql.JSONB, nullable=False, default=list),
        sa.Column("metadata", postgresql.JSONB, nullable=False, default=dict),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Provider Configs table
    op.create_table(
        "provider_configs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("provider_type", sa.String(50), nullable=False),
        sa.Column("api_key_encrypted", sa.Text, nullable=True),
        sa.Column("base_url", sa.String(500), nullable=True),
        sa.Column("models", postgresql.JSONB, nullable=False, default=list),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("is_default", sa.Boolean, nullable=False, default=False),
        sa.Column("rate_limit", sa.Integer, nullable=False, default=60),
        sa.Column("timeout", sa.Integer, nullable=False, default=120),
        sa.Column("max_retries", sa.Integer, nullable=False, default=3),
        sa.Column("retry_delay", sa.Float, nullable=False, default=1.0),
        sa.Column("cost_per_1k_input", sa.Float, nullable=False, default=0.0),
        sa.Column("cost_per_1k_output", sa.Float, nullable=False, default=0.0),
        sa.Column("metadata", postgresql.JSONB, nullable=False, default=dict),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # Provider Usage table
    op.create_table(
        "provider_usage",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "provider_config_id",
            sa.String(36),
            sa.ForeignKey("provider_configs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("requests_count", sa.Integer, nullable=False, default=0),
        sa.Column("tokens_input", sa.Integer, nullable=False, default=0),
        sa.Column("tokens_output", sa.Integer, nullable=False, default=0),
        sa.Column("tokens_total", sa.Integer, nullable=False, default=0),
        sa.Column("cost_usd", sa.Float, nullable=False, default=0.0),
        sa.Column("latency_avg_ms", sa.Float, nullable=False, default=0.0),
        sa.Column("latency_p95_ms", sa.Float, nullable=False, default=0.0),
        sa.Column("errors_count", sa.Integer, nullable=False, default=0),
        sa.Column("success_rate", sa.Float, nullable=False, default=1.0),
        sa.Column("metadata", postgresql.JSONB, nullable=False, default=dict),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("provider_usage")
    op.drop_table("provider_configs")
    op.drop_table("procedural_memories")
    op.drop_table("semantic_rules")
    op.drop_table("episodic_memories")
    op.drop_table("memory_entries")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("user_preferences")
    op.drop_table("api_keys")
    op.drop_table("users")
