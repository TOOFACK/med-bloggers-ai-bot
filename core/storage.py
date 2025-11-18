from typing import List, Optional, Tuple

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    func,
    select,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base, relationship

import logging
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

MAX_REFERENCE_PHOTOS = 3


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tg_id", name="uq_users_tg_id"),
        Index("idx_users_username", "username"),
    )

    id = Column(Integer, primary_key=True)
    tg_id = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    username = Column(String, nullable=True)
    photo_url = Column(String, nullable=True)
    photo_object_key = Column(String, nullable=True)
    photo_urls = Column(ARRAY(String), nullable=True)
    photo_object_keys = Column(ARRAY(String), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    subscription = relationship("SubsInfo", back_populates="user", uselist=False)


class SubsInfo(Base):
    __tablename__ = "subs_info"
    __table_args__ = (
        UniqueConstraint("tg_id", name="uq_subs_info_tg_id"),
        Index("idx_subs_info_tg_id", "tg_id"),
    )

    id = Column(Integer, primary_key=True)
    tg_id = Column(String, ForeignKey("users.tg_id", ondelete="CASCADE"), nullable=False)
    photo_left = Column(Integer, nullable=False, default=0)
    text_left = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now())
    user = relationship("User", back_populates="subscription", uselist=False)


async def get_user_by_tg_id(session: AsyncSession, tg_id: int | str) -> Optional[User]:
    result = await session.execute(select(User).where(User.tg_id == str(tg_id)))
    return result.scalar_one_or_none()


async def ensure_user(
    session: AsyncSession,
    tg_id: int | str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    username: Optional[str] = None,
) -> User:
    user = await get_user_by_tg_id(session, tg_id)
    if user is None:
        user = User(
            tg_id=str(tg_id),
            first_name=first_name,
            last_name=last_name,
            username=username,
        )
        session.add(user)
        await session.flush()
        return user

    updated = False
    
    first_name_check = first_name is not None and user.first_name != first_name
    last_name_check = last_name is not None and user.last_name != last_name
    username_check = first_name is not None and user.username != username

    if first_name_check or last_name_check or username_check:
        user.first_name = first_name
        user.last_name = last_name
        user.username = username
        updated = True

    if updated:
        session.add(user)
        await session.flush()
    return user


async def ensure_subscription(session: AsyncSession, user: User) -> SubsInfo:
    result = await session.execute(
        select(SubsInfo).where(SubsInfo.tg_id == user.tg_id)
    )
    subscription = result.scalar_one_or_none()
    if subscription is None:
        subscription = SubsInfo(tg_id=user.tg_id)
        session.add(subscription)
        await session.flush()
    return subscription


async def ensure_user_with_subscription(
    session: AsyncSession,
    tg_id: int | str,
    *,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    username: Optional[str] = None,
) -> tuple[User, SubsInfo]:
    user = await ensure_user(
        session,
        tg_id,
        first_name=first_name,
        last_name=last_name,
        username=username,
    )
    subscription = await ensure_subscription(session, user)
    return user, subscription



async def decrement_text_quota(session: AsyncSession, user: User, amount: int = 1):
    subscription = await ensure_subscription(session, user)

    if subscription.text_left < amount:
        logging.warning("Недостаточно текстовой квоты: %s", user.tg_id)
        return

    subscription.text_left -= amount
    session.add(subscription)
    await session.flush()
    return



async def decrement_photo_quota(
    session: AsyncSession, user: User, amount: int = 1
):
    subscription = await ensure_subscription(session, user)

    if subscription.photo_left < amount:
        logging.warning("Недостаточно фото-квоты: %s", user.tg_id)
        return

    subscription.photo_left -= amount
    session.add(subscription)
    await session.flush()
    return




async def set_user_photo(
    session: AsyncSession,
    user: User,
    photo_url: str,
    photo_object_key: str,
) -> Tuple[User, List[str]]:
    
    logging.info(f'inside set_user_photo {user}')

    photo_urls = list(user.photo_urls or [])
    photo_object_keys = list(user.photo_object_keys or [])

    photo_urls.append(photo_url)
    photo_object_keys.append(photo_object_key)

    removed_keys: List[str] = []
    if len(photo_urls) > MAX_REFERENCE_PHOTOS:
        logging.info(f'user {user} has more than {MAX_REFERENCE_PHOTOS}, deleting old...')
    while len(photo_urls) > MAX_REFERENCE_PHOTOS:
        logging.info(f'deleting {photo_urls[0]} from user {user}')
        photo_urls.pop(0)
        removed_key = photo_object_keys.pop(0)
        if removed_key:
            removed_keys.append(removed_key)
    
    logging.info(f'Now user {user} has {photo_urls}')
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
