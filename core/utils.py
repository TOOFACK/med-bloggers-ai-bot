import base64
import binascii
from io import BytesIO
from typing import List, Optional, Tuple

from aiogram import Bot
from aiogram.types import BufferedInputFile

DEFAULT_SUGGESTION_PATTERNS = (
    "Highly detailed illustration of {subject}",
    "Photorealistic concept art featuring {subject}",
    "Creative cinematic scene inspired by {subject}",
)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def truncate_for_button(text: str, limit: int = 48) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_referral_link(bot_username: str, code: str) -> str:
    if not bot_username:
        raise ValueError("bot_username is required")
    if not code:
        raise ValueError("code is required")
    username = bot_username[1:] if bot_username.startswith("@") else bot_username
    return f"https://t.me/{username}?start={code}"


def generate_prompt_suggestions(
    text: str, patterns: tuple[str, ...] = DEFAULT_SUGGESTION_PATTERNS
) -> List[str]:
    subject = normalize_text(text)
    if not subject:
        return []
    return [pattern.format(subject=subject) for pattern in patterns]


async def build_file_url(bot: Bot, file_id: str) -> Optional[str]:
    if not file_id:
        return None
    file = await bot.get_file(file_id)
    return f"https://api.telegram.org/file/bot{bot.token}/{file.file_path}"


async def fetch_file_bytes(bot: Bot, file_id: str) -> Tuple[bytes, str]:
    file = await bot.get_file(file_id)
    destination = BytesIO()
    await bot.download_file(file.file_path, destination)
    filename = file.file_path.split("/")[-1]
    return destination.getvalue(), filename


def input_file_from_base64(
    data: str, filename: str = "generation.jpg"
) -> BufferedInputFile:
    header, _, body = data.partition(",")
    if _ and "base64" in header.lower():
        data = body
    try:
        binary = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 payload") from exc
    return BufferedInputFile(binary, filename=filename)
