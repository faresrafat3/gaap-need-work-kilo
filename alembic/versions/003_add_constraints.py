"""Add constraints and foreign key enforcement.

Revision ID: 003
Revises: 002
Create Date: 2026-02-28 00:02:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = "003"
down_revision: str = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add constraints."""
    # Note: SQLite has limited support for ALTER TABLE
    # We use batch operations for SQLite compatibility
    
    # Add check constraints for enums using batch mode
    with op.batch_alter_table("users") as batch_op:
        batch_op.create_check_constraint(
            "ck_users_role",
            sa.text("role IN ('admin', 'user', 'guest')"),
        )
        batch_op.create_check_constraint(
            "ck_users_status",
            sa.text("status IN ('active', 'inactive', 'suspended', 'pending')"),
        )

    with op.batch_alter_table("sessions") as batch_op:
        batch_op.create_check_constraint(
            "ck_sessions_status",
            sa.text("status IN ('active', 'paused', 'archived', 'deleted')"),
        )
        batch_op.create_check_constraint(
            "ck_sessions_priority",
            sa.text("priority IN ('low', 'normal', 'high', 'critical')"),
        )

    with op.batch_alter_table("messages") as batch_op:
        batch_op.create_check_constraint(
            "ck_messages_role",
            sa.text("role IN ('system', 'user', 'assistant', 'function', 'tool')"),
        )
        batch_op.create_check_constraint(
            "ck_messages_status",
            sa.text("status IN ('pending', 'completed', 'failed', 'cancelled')"),
        )

    with op.batch_alter_table("memory_entries") as batch_op:
        batch_op.create_check_constraint(
            "ck_memory_tier",
            sa.text("tier IN ('working', 'episodic', 'semantic', 'procedural')"),
        )

    with op.batch_alter_table("user_preferences") as batch_op:
        batch_op.create_unique_constraint(
            "uq_user_preferences_user_id",
            ["user_id"],
        )


def downgrade() -> None:
    """Remove constraints using batch mode for SQLite compatibility."""
    with op.batch_alter_table("user_preferences") as batch_op:
        batch_op.drop_constraint("uq_user_preferences_user_id", type_="unique")

    with op.batch_alter_table("memory_entries") as batch_op:
        batch_op.drop_constraint("ck_memory_tier", type_="check")

    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_constraint("ck_messages_status", type_="check")
        batch_op.drop_constraint("ck_messages_role", type_="check")

    with op.batch_alter_table("sessions") as batch_op:
        batch_op.drop_constraint("ck_sessions_priority", type_="check")
        batch_op.drop_constraint("ck_sessions_status", type_="check")

    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("ck_users_status", type_="check")
        batch_op.drop_constraint("ck_users_role", type_="check")
