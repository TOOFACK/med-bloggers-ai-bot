"""Add user flags and free trial quota.

Revision ID: 0d23632935d3
Revises: 63a70a950071
Create Date: 2025-12-01 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0d23632935d3"
down_revision: Union[str, Sequence[str], None] = "63a70a950071"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


FREE_TRIAL_QUOTA = 5


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "is_test_end",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "is_blocked",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    op.alter_column(
        "subs_info",
        "photo_left",
        server_default=sa.text(str(FREE_TRIAL_QUOTA)),
    )
    op.alter_column(
        "subs_info",
        "text_left",
        server_default=sa.text(str(FREE_TRIAL_QUOTA)),
    )

    op.execute(
        sa.text(
            """
            UPDATE subs_info
            SET photo_left = :quota,
                text_left = :quota
            """
        ).bindparams(quota=FREE_TRIAL_QUOTA)
    )

    op.execute(
        sa.text(
            """
            INSERT INTO subs_info (tg_id, photo_left, text_left, created_at)
            SELECT u.tg_id, :quota, :quota, now()
            FROM users u
            WHERE NOT EXISTS (
                SELECT 1 FROM subs_info s WHERE s.tg_id = u.tg_id
            )
            """
        ).bindparams(quota=FREE_TRIAL_QUOTA)
    )


def downgrade() -> None:
    op.alter_column("subs_info", "text_left", server_default=sa.text("0"))
    op.alter_column("subs_info", "photo_left", server_default=sa.text("0"))

    op.drop_column("users", "is_blocked")
    op.drop_column("users", "is_test_end")
