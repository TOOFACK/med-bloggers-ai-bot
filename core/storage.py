from typing import List, Optional, Tuple

import enum
import secrets

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base

import logging
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

MAX_REFERENCE_PHOTOS = 3
FREE_TRIAL_QUOTA = 5


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tg_id", name="uq_users_tg_id"),
        Index("idx_users_username", "username"),
    )

    id = Column(Integer, primary_key=True)
    tg_id = Column(String, nullable=False)

    first_name = Column(String)
    last_name = Column(String)
    username = Column(String)

    photo_url = Column(String)
    photo_object_key = Column(String)

    photo_urls = Column(ARRAY(String))
    photo_object_keys = Column(ARRAY(String))

    is_test_end = Column(Boolean, nullable=False, server_default="false")
    is_blocked = Column(Boolean, nullable=False, server_default="false")

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class SubsInfo(Base):
    __tablename__ = "subs_info"
    __table_args__ = (
        UniqueConstraint("tg_id", name="uq_subs_info_tg_id"),
        Index("idx_subs_info_tg_id", "tg_id"),
    )

    id = Column(Integer, primary_key=True)
    tg_id = Column(String, nullable=False)

    photo_left = Column(Integer, nullable=False, server_default=str(FREE_TRIAL_QUOTA))
    text_left = Column(Integer, nullable=False, server_default=str(FREE_TRIAL_QUOTA))

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now())


QUOTA_COLUMN_MAP = {
    "photo": SubsInfo.photo_left,
    "text": SubsInfo.text_left,
}


class GenerationSource(str, enum.Enum):
    REF = "ref"
    PAID = "paid"


class Referral(Base):
    __tablename__ = "refs"

    owner_tg_id = Column(String, primary_key=True)
    code = Column(String(64), nullable=False, unique=True, index=True)
    reward_generations = Column(Integer, nullable=False, server_default="1")
    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class GenerationLedger(Base):
    __tablename__ = "generation_ledger"
    __table_args__ = (
        Index("idx_generation_ledger_tg_id", "tg_id"),
        Index("idx_generation_ledger_ref_code", "referral_code"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tg_id = Column(String, nullable=False)
    source = Column(
        SAEnum(
            GenerationSource,
            name="generation_source",
            values_callable=lambda enum_cls: [item.value for item in enum_cls],
        ),
        nullable=False,
    )
    amount = Column(Integer, nullable=False)
    referral_owner_tg_id = Column(String)
    referral_code = Column(String(64))
    author = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _normalize_tg_id(tg_id: int | str) -> str:
    if tg_id is None:
        raise ValueError("Invalid tg_id: None")
    value = str(tg_id).strip()
    if not value:
        raise ValueError(f"Invalid tg_id: {tg_id!r}")
    return value


def generate_referral_code(length: int = 6) -> str:
    if length < 4:
        raise ValueError("Referral code length must be >= 4")
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789abcdefghijkmnpqrstuvwxyz"
    return "".join(secrets.choice(alphabet) for _ in range(length))


async def use_referral(
    session: AsyncSession,
    tg_id: int | str,
    code: str,
    *,
    author: str = "system",
) -> int:
    if not code:
        return 0
    normalized_tg_id = _normalize_tg_id(tg_id)
    result = await session.execute(
        select(Referral).where(Referral.code == code)
    )
    referral = result.scalar_one_or_none()
    if referral is None or not referral.is_active:
        return 0
    if referral.owner_tg_id == normalized_tg_id:
        return 0

    result = await session.execute(
        select(GenerationLedger.id).where(
            GenerationLedger.tg_id == normalized_tg_id,
            GenerationLedger.source == GenerationSource.REF,
        )
    )
    if result.scalar_one_or_none() is not None:
        return 0

    user, _ = await ensure_user_with_subscription(session, normalized_tg_id)

    reward = max(int(referral.reward_generations), 0)
    if reward <= 0:
        return 0

    history = GenerationLedger(
        tg_id=normalized_tg_id,
        source=GenerationSource.REF,
        amount=reward,
        referral_owner_tg_id=referral.owner_tg_id,
        referral_code=referral.code,
        author=author,
    )
    session.add(history)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        return 0

    await restore_quota(session, user, "photo", reward)
    await restore_quota(session, user, "text", reward)
    return reward


async def get_or_create_referral(
    session: AsyncSession,
    owner_tg_id: int | str,
    *,
    reward_generations: int = 1,
    is_active: bool = True,
    code_length: int = 6,
) -> Referral:
    normalized_tg_id = _normalize_tg_id(owner_tg_id)
    result = await session.execute(
        select(Referral).where(Referral.owner_tg_id == normalized_tg_id)
    )
    referral = result.scalar_one_or_none()
    if referral is not None:
        return referral

    attempts = 0
    while attempts < 10:
        code = generate_referral_code(code_length)
        referral = Referral(
            owner_tg_id=normalized_tg_id,
            code=code,
            reward_generations=reward_generations,
            is_active=is_active,
        )
        session.add(referral)
        try:
            await session.flush()
            return referral
        except IntegrityError:
            await session.rollback()
            attempts += 1

    raise RuntimeError("Failed to generate a unique referral code")


async def get_user_by_tg_id(session: AsyncSession, tg_id: int | str) -> Optional[User]:
    result = await session.execute(select(User).where(User.tg_id == _normalize_tg_id(tg_id)))
    return result.scalar_one_or_none()


def _quota_column(quota_type: str):
    column = QUOTA_COLUMN_MAP.get(quota_type)
    if column is None:
        raise ValueError(f"Unsupported quota type: {quota_type}")
    return column


async def _apply_quota_delta(
    session: AsyncSession,
    user: User,
    quota_type: str,
    delta: int,
    *,
    mark_test_end_on_exhaust: bool = False,
) -> bool:
    column = _quota_column(quota_type)
    stmt = update(SubsInfo).where(SubsInfo.tg_id == user.tg_id)
    if delta < 0:
        stmt = stmt.where(column >= -delta)
    stmt = stmt.values({column.key: column + delta}).returning(column)
    result = await session.execute(stmt)
    new_value = result.scalar_one_or_none()
    updated = new_value is not None
    if (
        updated
        and mark_test_end_on_exhaust
        and new_value is not None
        and new_value <= 0
        and not user.is_test_end
    ):
        user.is_test_end = True
        session.add(user)
    if updated:
        await session.flush()
    return updated


async def consume_quota(
    session: AsyncSession, user: User, quota_type: str, amount: int = 1
) -> bool:
    return await _apply_quota_delta(
        session,
        user,
        quota_type,
        -amount,
        mark_test_end_on_exhaust=True,
    )


async def restore_quota(
    session: AsyncSession, user: User, quota_type: str, amount: int = 1
) -> bool:
    return await _apply_quota_delta(session, user, quota_type, amount)


async def ensure_user(
    session: AsyncSession,
    tg_id: int | str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    username: Optional[str] = None,
) -> User:
    normalized_tg_id = _normalize_tg_id(tg_id)
    user = await get_user_by_tg_id(session, normalized_tg_id)
    if user is None:
        user = User(
            tg_id=normalized_tg_id,
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
    username_check = username is not None and user.username != username

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
        subscription = SubsInfo(
            tg_id=user.tg_id,
            photo_left=FREE_TRIAL_QUOTA,
            text_left=FREE_TRIAL_QUOTA,
        )
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



async def decrement_text_quota(
    session: AsyncSession, user: User, amount: int = 1
) -> bool:
    return await consume_quota(session, user, "text", amount)



async def decrement_photo_quota(
    session: AsyncSession, user: User, amount: int = 1
) -> bool:
    return await consume_quota(session, user, "photo", amount)




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


async def clear_user_photo(session: AsyncSession, user: User) -> List[str]:
    removed_keys = list(user.photo_object_keys or [])
    user.photo_url = None
    user.photo_object_key = None
    user.photo_urls = None
    user.photo_object_keys = None

    session.add(user)
    await session.flush()
    return removed_keys
