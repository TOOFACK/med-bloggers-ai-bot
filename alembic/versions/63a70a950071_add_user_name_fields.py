"""add user name fields

Revision ID: 63a70a950071
Revises: 604d70a0c9cf
Create Date: 2025-11-25 20:29:37.091454

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '63a70a950071'
down_revision: Union[str, Sequence[str], None] = '604d70a0c9cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("users", sa.Column("first_name", sa.String(), nullable=True))
    op.add_column("users", sa.Column("last_name", sa.String(), nullable=True))
    op.add_column("users", sa.Column("username", sa.String(), nullable=True))
    op.create_index("idx_users_username", "users", ["username"])


def downgrade():
    op.drop_index("idx_users_username", table_name="users")
    op.drop_column("users", "username")
    op.drop_column("users", "last_name")
    op.drop_column("users", "first_name")