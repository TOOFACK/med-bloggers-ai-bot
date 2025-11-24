"""create subs_info without fk

Revision ID: 604d70a0c9cf
Revises: 889b710e630c
Create Date: 2025-11-25 00:58:17.375105

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '604d70a0c9cf'
down_revision: Union[str, Sequence[str], None] = '3a31be92f987'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "subs_info",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tg_id", sa.String(), nullable=False),
        sa.Column("photo_left", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("text_left", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_unique_constraint("uq_subs_info_tg_id", "subs_info", ["tg_id"])
    op.create_index("idx_subs_info_tg_id", "subs_info", ["tg_id"])


def downgrade():
    op.drop_index("idx_subs_info_tg_id", table_name="subs_info")
    op.drop_constraint("uq_subs_info_tg_id", "subs_info", type_="unique")
    op.drop_table("subs_info")
