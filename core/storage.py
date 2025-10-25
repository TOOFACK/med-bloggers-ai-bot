from typing import List, Optional, Tuple

from sqlalchemy import Column, DateTime, Integer, String, func, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

Base = declarative_base()

MAX_REFERENCE_PHOTOS = 3


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    tg_id = Column(String, unique=True, index=True)
    photo_url = Column(String, nullable=True)
    photo_object_key = Column(String, nullable=True)
    photo_urls = Column(ARRAY(String), nullable=True)
    photo_object_keys = Column(ARRAY(String), nullable=True)
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
) -> Tuple[User, List[str]]:
    photo_urls = list(user.photo_urls or [])
    photo_object_keys = list(user.photo_object_keys or [])

    photo_urls.append(photo_url)
    photo_object_keys.append(photo_object_key)

    removed_keys: List[str] = []
    while len(photo_urls) > MAX_REFERENCE_PHOTOS:
        photo_urls.pop(0)
        removed_key = photo_object_keys.pop(0)
        if removed_key:
            removed_keys.append(removed_key)

    user.photo_urls = photo_urls
    user.photo_object_keys = photo_object_keys
    user.photo_url = photo_urls[-1] if photo_urls else None
    user.photo_object_key = photo_object_keys[-1] if photo_object_keys else None
    session.add(user)
    await session.flush()
    return user, removed_keys


async def clear_user_photo(session: AsyncSession, user: User) -> User:
    user.photo_url = None
    user.photo_object_key = None
    user.photo_urls = None
    user.photo_object_keys = None
    session.add(user)
    await session.flush()
    return user


def get_user_photo_urls(user: User) -> List[str]:
    if user.photo_urls:
        return [url for url in user.photo_urls if url]
    return [user.photo_url] if user.photo_url else []
