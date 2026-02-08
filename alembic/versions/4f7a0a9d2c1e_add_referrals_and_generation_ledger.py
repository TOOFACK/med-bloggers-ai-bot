"""add referrals and generation ledger

Revision ID: 4f7a0a9d2c1e
Revises: 0d23632935d3
Create Date: 2026-02-08 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "4f7a0a9d2c1e"
down_revision: Union[str, Sequence[str], None] = "0d23632935d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "refs",
        sa.Column("owner_tg_id", sa.String(), primary_key=True),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("reward_generations", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("idx_refs_code", "refs", ["code"], unique=True)

    generation_source = postgresql.ENUM(
        "ref",
        "paid",
        name="generation_source",
        create_type=False,
    )
    generation_source.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "generation_ledger",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("tg_id", sa.String(), nullable=False),
        sa.Column("source", generation_source, nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("referral_owner_tg_id", sa.String(), nullable=True),
        sa.Column("referral_code", sa.String(length=64), nullable=True),
        sa.Column("author", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_generation_ledger_tg_id", "generation_ledger", ["tg_id"]
    )
    op.create_index(
        "idx_generation_ledger_ref_code", "generation_ledger", ["referral_code"]
    )
    op.create_index(
        "uq_generation_ledger_ref_once",
        "generation_ledger",
        ["tg_id"],
        unique=True,
        postgresql_where=sa.text("source = 'ref'"),
    )


def downgrade() -> None:
    op.drop_index("uq_generation_ledger_ref_once", table_name="generation_ledger")
    op.drop_index("idx_generation_ledger_ref_code", table_name="generation_ledger")
    op.drop_index("idx_generation_ledger_tg_id", table_name="generation_ledger")
    op.drop_table("generation_ledger")

    generation_source = postgresql.ENUM(
        "ref",
        "paid",
        name="generation_source",
        create_type=False,
    )
    generation_source.drop(op.get_bind(), checkfirst=True)

    op.drop_index("idx_refs_code", table_name="refs")
    op.drop_table("refs")
