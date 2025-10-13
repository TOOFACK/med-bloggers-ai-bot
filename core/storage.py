from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    tg_id = Column(String, unique=True, index=True)
    photo_url = Column(String, nullable=True)
    photo_object_key = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


async def get_user_by_tg_id(session: AsyncSession, tg_id: int | str) -> Optional[User]:
    result = await session.execute(select(User).where(User.tg_id == str(tg_id)))
    return result.scalar_one_or_none()


async def ensure_user(session: AsyncSession, tg_id: int | str) -> User:
    user = await get_user_by_tg_id(session, tg_id)
    if user is None:
        user = User(tg_id=str(tg_id))
        session.add(user)
        await session.flush()
    return user


async def set_user_photo(
    session: AsyncSession,
    user: User,
    photo_url: str,
    photo_object_key: str,
) -> User:
    user.photo_url = photo_url
    user.photo_object_key = photo_object_key
    session.add(user)
    await session.flush()
    return user


async def clear_user_photo(session: AsyncSession, user: User) -> User:
    user.photo_url = None
    user.photo_object_key = None
    session.add(user)
    await session.flush()
    return user
