"""Add indexes for performance.

Revision ID: 002
Revises: 001
Create Date: 2026-02-28 00:01:00.000000
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers
revision: str = "002"
down_revision: str = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add performance indexes."""
    # User indexes
    op.create_index("ix_users_email_status", "users", ["email", "status"])
    op.create_index("ix_users_role_active", "users", ["role", "is_active"])

    # API Key indexes
    op.create_index("ix_api_keys_user_active", "api_keys", ["user_id", "is_active"])
    op.create_index("ix_api_keys_expires", "api_keys", ["expires_at"])

    # Session indexes
    op.create_index("ix_sessions_user_status", "sessions", ["user_id", "status"])
    op.create_index("ix_sessions_last_message", "sessions", ["user_id", "last_message_at"])
    op.create_index("ix_sessions_archived", "sessions", ["archived_at"])

    # Message indexes
    op.create_index("ix_messages_session_sequence", "messages", ["session_id", "sequence"])
    op.create_index("ix_messages_session_role", "messages", ["session_id", "role"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])

    # Memory indexes
    op.create_index("ix_memory_user_tier", "memory_entries", ["user_id", "tier"])
    op.create_index("ix_memory_accessed", "memory_entries", ["last_accessed"])
    op.create_index("ix_memory_importance", "memory_entries", ["importance"])

    # Episodic memory indexes
    op.create_index("ix_episodic_user_task", "episodic_memories", ["user_id", "task_id"])
    op.create_index("ix_episodic_category", "episodic_memories", ["category"])
    op.create_index("ix_episodic_created", "episodic_memories", ["created_at"])

    # Semantic rules indexes
    op.create_index("ix_semantic_user_category", "semantic_rules", ["user_id", "category"])
    op.create_index("ix_semantic_confidence", "semantic_rules", ["confidence"])

    # Procedural memory indexes
    op.create_index("ix_procedural_user_name", "procedural_memories", ["user_id", "name"])
    op.create_index("ix_procedural_success", "procedural_memories", ["success_rate"])

    # Provider indexes
    op.create_index("ix_provider_user_type", "provider_configs", ["user_id", "provider_type"])
    op.create_index("ix_provider_active", "provider_configs", ["is_active"])
    op.create_index("ix_provider_default", "provider_configs", ["user_id", "is_default"])

    # Usage indexes
    op.create_index("ix_usage_provider_date", "provider_usage", ["provider_config_id", "date"])


def downgrade() -> None:
    """Remove indexes."""
    # Usage indexes
    op.drop_index("ix_usage_provider_date", table_name="provider_usage")

    # Provider indexes
    op.drop_index("ix_provider_default", table_name="provider_configs")
    op.drop_index("ix_provider_active", table_name="provider_configs")
    op.drop_index("ix_provider_user_type", table_name="provider_configs")

    # Procedural memory indexes
    op.drop_index("ix_procedural_success", table_name="procedural_memories")
    op.drop_index("ix_procedural_user_name", table_name="procedural_memories")

    # Semantic rules indexes
    op.drop_index("ix_semantic_confidence", table_name="semantic_rules")
    op.drop_index("ix_semantic_user_category", table_name="semantic_rules")

    # Episodic memory indexes
    op.drop_index("ix_episodic_created", table_name="episodic_memories")
    op.drop_index("ix_episodic_category", table_name="episodic_memories")
    op.drop_index("ix_episodic_user_task", table_name="episodic_memories")

    # Memory indexes
    op.drop_index("ix_memory_importance", table_name="memory_entries")
    op.drop_index("ix_memory_accessed", table_name="memory_entries")
    op.drop_index("ix_memory_user_tier", table_name="memory_entries")

    # Message indexes
    op.drop_index("ix_messages_created_at", table_name="messages")
    op.drop_index("ix_messages_session_role", table_name="messages")
    op.drop_index("ix_messages_session_sequence", table_name="messages")

    # Session indexes
    op.drop_index("ix_sessions_archived", table_name="sessions")
    op.drop_index("ix_sessions_last_message", table_name="sessions")
    op.drop_index("ix_sessions_user_status", table_name="sessions")

    # API Key indexes
    op.drop_index("ix_api_keys_expires", table_name="api_keys")
    op.drop_index("ix_api_keys_user_active", table_name="api_keys")

    # User indexes
    op.drop_index("ix_users_role_active", table_name="users")
    op.drop_index("ix_users_email_status", table_name="users")
