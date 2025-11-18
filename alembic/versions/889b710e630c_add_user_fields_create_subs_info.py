"""add user fields + create subs_info

Revision ID: 889b710e630c
Revises: 3a31be92f987
Create Date: 2025-11-18 22:30:40.263956
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = '889b710e630c'
down_revision: Union[str, Sequence[str], None] = '3a31be92f987'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1) Делаем tg_id уникальным (нужно для FK)
    op.create_unique_constraint("uq_users_tg_id", "users", ["tg_id"])

    # 2) Добавляем новые поля в users
    op.add_column("users", sa.Column("first_name", sa.String()))
    op.add_column("users", sa.Column("last_name", sa.String()))
    op.add_column("users", sa.Column("username", sa.String()))
    op.create_index("idx_users_username", "users", ["username"])

    # 3) Создаем таблицу subs_info со ссылкой на users.tg_id
    op.create_table(
        "subs_info",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tg_id", sa.String(), sa.ForeignKey("users.tg_id", ondelete="CASCADE"), nullable=False),
        sa.Column("photo_left", sa.Integer(), server_default="0", nullable=False),
        sa.Column("text_left", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime()),
    )

    op.create_index("idx_subs_info_tg_id", "subs_info", ["tg_id"])
    op.create_unique_constraint("uq_subs_info_tg_id", "subs_info", ["tg_id"])



def downgrade() -> None:
    op.drop_index("idx_subs_info_tg_id", table_name="subs_info")
    op.drop_table("subs_info")

    op.drop_index("idx_users_username", table_name="users")
    op.drop_column("users", "username")
    op.drop_column("users", "last_name")
    op.drop_column("users", "first_name")

    op.drop_constraint("uq_users_tg_id", "users", type_="unique")
    op.drop_constraint("uq_subs_info_tg_id", "subs_info", type_="unique")

