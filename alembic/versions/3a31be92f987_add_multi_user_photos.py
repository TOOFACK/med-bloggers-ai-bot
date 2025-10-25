"""Add multi photo support for users.

Revision ID: 3a31be92f987
Revises: 925099f3c567
Create Date: 2024-11-02 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3a31be92f987"
down_revision: Union[str, Sequence[str], None] = "925099f3c567"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "users",
        sa.Column("photo_urls", postgresql.ARRAY(sa.String()), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("photo_object_keys", postgresql.ARRAY(sa.String()), nullable=True),
    )
    op.execute(
        """
        UPDATE users
        SET photo_urls = ARRAY[photo_url]
        WHERE photo_url IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE users
        SET photo_object_keys = ARRAY[photo_object_key]
        WHERE photo_object_key IS NOT NULL
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        """
        UPDATE users
        SET photo_url = photo_urls[array_upper(photo_urls, 1)]
        WHERE photo_urls IS NOT NULL
              AND array_length(photo_urls, 1) > 0
        """
    )
    op.execute(
        """
        UPDATE users
        SET photo_object_key = photo_object_keys[array_upper(photo_object_keys, 1)]
        WHERE photo_object_keys IS NOT NULL
              AND array_length(photo_object_keys, 1) > 0
        """
    )
    op.drop_column("users", "photo_object_keys")
    op.drop_column("users", "photo_urls")
