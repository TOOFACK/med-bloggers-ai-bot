import asyncio
import html
import logging
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from aiogram.types import BufferedInputFile
from aiogram import Bot, F, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.filters.command import CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from sqlalchemy.exc import SQLAlchemyError

from config import (
    COMET_API_KEY,
    COMET_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PAYMENTS_ACTIVE,
    VERTEX_ASPECT_RATIO,
    VERTEX_CREDENTIALS_PATH,
    VERTEX_IMAGE_MODEL,
    VERTEX_LOCATION,
    VERTEX_PROJECT_ID,
)
from core.db import SessionLocal
from core.providers import init_image_providers, init_prompt_providers
from core.s3 import S3ConfigError, delete_object, upload_bytes
from core.services import ModelService, PromptService
from core.storage import (
    MAX_REFERENCE_PHOTOS,
    SubsInfo,
    consume_quota,
    ensure_user_with_subscription,
    get_user_photo_urls,
    restore_quota,
    set_user_photo,
    clear_user_photo
)
from core.utils import (
    build_file_url,
    fetch_file_bytes,
    generate_prompt_suggestions,
    input_file_from_base64,
    normalize_text,
)
import base64
from .keyboards import (
    PromptChoiceCallback,
    PromptRegenCallback,
    prompt_suggestions_keyboard,
    prompt_mode_keyboard,
    PromptModeCallback
)

from .constants import PROMPT_SUGGESTION_COUNT, SYSTEM_PROMPT_FOR_EDIT, SYSTEM_PROMPT_FOR_CREATING

from .notifications.salebot import SaleBotClient
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
router = Router()

image_providers = init_image_providers(
    comet_api_key=COMET_API_KEY,
    openrouter_api_key=OPENROUTER_API_KEY,
    comet_base_url=COMET_BASE_URL,
    openrouter_base_url=OPENROUTER_BASE_URL,
    vertex_credentials_path=VERTEX_CREDENTIALS_PATH,
    vertex_project=VERTEX_PROJECT_ID,
    vertex_location=VERTEX_LOCATION,
    vertex_model=VERTEX_IMAGE_MODEL,
    vertex_aspect_ratio=VERTEX_ASPECT_RATIO,
)
if not image_providers:
    logger.warning("На запуске не найдено ни одного провайдера генерации изображений.")

prompt_providers = init_prompt_providers(
    comet_api_key=COMET_API_KEY,
    openrouter_api_key=OPENROUTER_API_KEY,
    openrouter_base_url=OPENROUTER_BASE_URL,
)
if not prompt_providers:
    logger.warning("На запуске не найдено ни одного провайдера для генерации промптов.")

model_service = ModelService(image_providers)
prompt_service = PromptService(prompt_providers)
sale_client = SaleBotClient()

async def start_loading_animation(
    message: types.Message, text="⏳ Генерируем", delay=0.5
):
    """
    Показывает анимацию "⏳ Генерируем..." в виде точек.
    Возвращает tuple: (msg, stop_animation)
    """
    msg = await message.answer(f"{text}.")
    running = True

    async def animate():
        dots = 1
        while running:
            await asyncio.sleep(delay)
            dots = (dots % 3) + 1
            try:
                await msg.edit_text(f"{text}{'.' * dots}")
            except Exception:
                break

    task = asyncio.create_task(animate())

    def stop():
        nonlocal running
        running = False
        task.cancel()

    return msg, stop


def _format_prompt_message(prompts: List[str]) -> str:
    lines = ["<b>Варианты промптов:</b>"]
    for idx, prompt in enumerate(prompts, start=1):
        lines.append(f"{idx}. <code>{html.escape(prompt)}</code>")
    return "\n".join(lines)


def _prune_map(data: Dict[str, Any], keep: int = 20) -> Dict[str, Any]:
    if len(data) <= keep:
        return data
    keys = sorted(data.keys(), key=int)[-keep:]
    return {key: data[key] for key in keys}


def _user_profile_kwargs(user: Optional[types.User]) -> Dict[str, Optional[str]]:
    if not user:
        return {}
    return {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "username": user.username,
    }


def _format_subscription_status_message(tg_id: int, subscription: Optional[SubsInfo]) -> str:
    if not PAYMENTS_ACTIVE or subscription is None:
        return (
            "Сейчас генерации безлимитны.\n"
        )

    def _fmt(value: Optional[int]) -> str:
        if value is None:
            return "∞"
        return str(max(value, 0))

    return (
        "📊 <b>Личный кабинет</b>\n"
        f"Генераций изображений осталось: <b>{_fmt(subscription.photo_left)}</b>\n"
        f"Запросов промптов осталось: <b>{_fmt(subscription.text_left)}</b>\n"
        '<a href="https://t.me/m/OC5FwjxIMzEy">Пополни подписку в отделе заботы</a>\n'
    )


def _quota_warning_message(tg_id: int, quota_type: str) -> str:
    if quota_type == "photo":
        label = "генерации изображений"
    else:
        label = "запросы на промпты"

    return (
        f"Закончились {label}.\n"
        '<a href="https://t.me/m/OC5FwjxIMzEy">Пополни подписку в отделе заботы</a>\n'
        "Используй команду /cabinet, чтобы проверить баланс."
    )



QuotaType = Literal["photo", "text"]
ReserveStatus = Literal["ok", "blocked", "exhausted"]


async def _notify_missing_user(target: Message | CallbackQuery) -> None:
    if isinstance(target, CallbackQuery):
        await target.answer("Не распознали пользователя.", show_alert=True)
    else:
        await target.answer("Не распознали пользователя.")


async def _notify_blocked_user(target: Message | CallbackQuery) -> None:
    text = "Доступ к боту заблокирован. Обратись к администратору."
    if isinstance(target, CallbackQuery):
        await target.message.answer(text)
        try:
            await target.answer("Доступ заблокирован.", show_alert=True)
        except TelegramBadRequest:
            pass
    else:
        await target.answer(text)


async def _ensure_user_allowed(target: Message | CallbackQuery) -> bool:
    from_user = getattr(target, "from_user", None)
    if from_user is None:
        await _notify_missing_user(target)
        return False
    async with SessionLocal() as session:
        user, _ = await ensure_user_with_subscription(
            session,
            from_user.id,
            **_user_profile_kwargs(from_user),
        )
        await _commit_session(session)
    if user.is_blocked:
        await _notify_blocked_user(target)
        return False
    return True


async def _notify_quota_exhausted(
    target: Message | CallbackQuery, quota_type: QuotaType
) -> None:
    from_user = getattr(target, "from_user", None)
    tg_id = from_user.id if from_user else 0
    warning = _quota_warning_message(tg_id, quota_type)
    if isinstance(target, CallbackQuery):
        await target.message.answer(warning)
        try:
            await target.answer("Лимит закончился.", show_alert=True)
        except TelegramBadRequest:
            pass
    else:
        await target.answer(warning, parse_mode="HTML")


async def _reserve_quota(
    tg_id: int,
    quota_type: QuotaType,
    *,
    amount: int = 1,
    profile: Optional[Dict[str, Optional[str]]] = None,
) -> ReserveStatus:
    if not PAYMENTS_ACTIVE:
        return "ok"
    async with SessionLocal() as session:
        user, subscription = await ensure_user_with_subscription(
            session, tg_id, **(profile or {})
        )
        if user.is_blocked:
            await _commit_session(session)
            return "blocked"

        quota_left = (
            subscription.photo_left if quota_type == "photo" else subscription.text_left
        )

        if quota_left is not None and quota_left < amount:
            if not user.is_test_end:
                user.is_test_end = True
                session.add(user)
            await _commit_session(session)
            return "exhausted"

        updated = await consume_quota(session, user, quota_type, amount)
        if not updated:
            if not user.is_test_end:
                user.is_test_end = True
                session.add(user)
                await _commit_session(session)
            else:
                await session.rollback()
            return "exhausted"

        await _commit_session(session)
        return "ok"


async def _return_quota(
    tg_id: int, quota_type: QuotaType, *, amount: int = 1
) -> None:
    if not PAYMENTS_ACTIVE:
        return
    async with SessionLocal() as session:
        user, _ = await ensure_user_with_subscription(session, tg_id)
        await restore_quota(session, user, quota_type, amount)
        await _commit_session(session)


def quota_guard(quota_type: QuotaType, *, amount: int = 1):
    def decorator(func: Callable[..., Awaitable[bool]]):
        @wraps(func)
        async def wrapper(target, *args, **kwargs):
            from_user = getattr(target, "from_user", None)
            if from_user is None:
                await _notify_missing_user(target)
                return False
            profile = _user_profile_kwargs(from_user)
            if not PAYMENTS_ACTIVE:
                async with SessionLocal() as session:
                    user, _ = await ensure_user_with_subscription(
                        session, from_user.id, **profile
                    )
                    await _commit_session(session)
                if user.is_blocked:
                    await _notify_blocked_user(target)
                    return False
                return await func(target, *args, **kwargs)

            reserve_status = await _reserve_quota(
                from_user.id,
                quota_type,
                amount=amount,
                profile=profile,
            )
            if reserve_status == "blocked":
                await _notify_blocked_user(target)
                return False
            if reserve_status == "exhausted":
                await _notify_quota_exhausted(target, quota_type)
                return False
            try:
                result = await func(target, *args, **kwargs)
            except Exception as exc:
                sale_client.send_error_message(
                error_text=str(exc),
                error_place="quota_guard.decorator.result = await func(target, *args, **kwargs)",
            )
                await _return_quota(from_user.id, quota_type, amount=amount)
                raise
            if result is False:
                await _return_quota(from_user.id, quota_type, amount=amount)
            return result

        return wrapper

    return decorator


async def _commit_session(session):
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        logger.exception("DB error: %s", exc)
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="_commit_session")
        raise


async def _perform_generation(
    prompt: str, reference_urls: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    try:
        if reference_urls:
            return await model_service.generate_with_reference(prompt, reference_urls)
        return await model_service.generate_from_text(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation failed: %s", exc)

        sale_client.send_error_message(
            error_text=str(exc),
            error_place="model_service.generate_with_reference or model_service.generate_from_text")
        
        return None


async def _send_generation(message: Message, result: Dict[str, Any], caption: str):

    footer = '@med_ai_photo_redactor_bot'
    full_caption = f"Результат генрации\n{footer}"

    if len(full_caption) > 128:
        full_caption = full_caption[:128] + "…"
    output_type = result.get("type")
    data = result.get("data")

    if output_type == "url":
        url_candidate = None
        if isinstance(data, str):
            stripped = data.strip()
            if stripped.startswith(("http://", "https://")):
                url_candidate = stripped
            else:
                import re

                match = re.search(r"https?://\S+", stripped)
                if match:
                    url_candidate = match.group(0).rstrip("`\"' )]")
        if url_candidate:
            try:
                await message.answer_photo(url_candidate, caption=full_caption)
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to send image by URL: %s", exc)

                sale_client.send_error_message(
                    error_text=str(exc),
                    error_place="_send_generation.url_candidate.message.answer_photo(url_candidate, caption=full_caption)")
        
        # TODO: resend if so or any other logic
        await message.answer(
            "Провайдер не вернул ссылку на изображение. Текст ответа:\n"
            f"<code>{html.escape(str(data))}</code>"
        )
        sale_client.send_error_message(
                error_text=f"Провайдер не вернул ссылку на изображение. Текст ответа:\n<code>{html.escape(str(data))}</code>",
                error_place="_send_generation.url_candidate.message.answer_photo(url_candidate, caption=full_caption)")
        return

    if output_type == "base64":
        try:
            photo_file = input_file_from_base64(data)
        except ValueError as exc:
            logger.exception("Base64 decode failed: %s", exc)

            sale_client.send_error_message(
                error_text=str(exc),
                error_place="_send_generation.output_type.base64.input_file_from_base64(data)")
            
            await message.answer("Ошибка обработки изображения. Попробуйте снова.")
            
            return
        await message.answer_photo(photo_file, caption=full_caption)
        return

    if output_type == "text":
        # TODO: retry logic until image
        await message.answer(
            "Не удалось обработать результат генерации. Попробуй еще раз"
        )
        sale_client.send_error_message(
                error_text=f"Провайдер не вернул ссылку на изображение. Текст ответа:\n<code>{html.escape(str(data))}</code>",
                error_place="_send_generation.url_candidate.message.answer_photo(url_candidate, caption=full_caption)")
        return

    logger.warning("Unexpected generation payload type: %s", output_type)
    await message.answer("Не удалось обработать результат генерации. Попробуй еще раз")


@router.message(Command("start", "help", "info"))
async def start(message: Message):
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return

    if not await _ensure_user_allowed(message):
        return

    instructions = (
        "🎨 <b>Привет!</b>\n\n"
        "Я — AI-бот, который создаёт и редактирует фотографии по описанию.\n\n"

        "🪄 <b>Как работает генерация:</b>\n"
        "• Пришли до 3 своих фото, где хорошо видно лицо (лучше с разных ракурсов).\n"
        "• После загрузки фото <b>любое твоё текстовое сообщение</b> становится инструкцией "
        "для генерации изображения с твоим участием.\n"
        "• Опиши максимально <b>подробно</b> образ, позу, стиль, одежду, фон, эмоции, "
        "освещение и настроение — чем детальнее, тем лучше результат.\n\n"

        "🖼️ <b>Если нужна картинка без твоего лица</b>\n"
        "Используй команду <code>/free_gen</code> — она создаёт изображение полностью с нуля.\n\n"

        "📸 <b>Управление фотографиями:</b>\n"
        "• <code>/my_photos</code> — покажу загруженные тобой фото.\n"
        "• <code>/reset_photos</code> — удалю все твои фото.\n\n"

        "🧠 <b>Инструмент для авторов контента:</b>\n"
        "Отправь <code>/get_prompts</code> + текст для поста/видео — и я сгенерирую варианты промптов "
        "для обложек Reels, видео или постов.\n\n"

        "👤 <b>Личный кабинет</b>\n"
        "Проверить оставшиеся генерации: <code>/cabinet</code>\n\n"

        "💰 <b>Прейскурант</b>\n"
        "Каждому новому пользователю даётся:\n"
        "• 5 бесплатных генераций изображений\n"
        "• 5 бесплатных промптов\n\n"
        "После исчерпания лимита пополнить баланс можно в "
        "<a href='https://t.me/m/OC5FwjxIMzEy'>отделе заботы</a>\n\n"
        "💳 <b>Тарифы:</b>\n"
        "• 20 фото + 20 промптов — 590 ₽\n"
        "• 50 фото + 50 промптов — 990 ₽\n"
        "• 100 фото + 100 промптов — 1490 ₽\n\n"
        "👇 Нажми кнопку, чтобы начать!"
    )

    await message.answer(instructions, parse_mode="HTML")


@router.message(Command("cabinet", "balance"))
async def handle_status(message: Message):
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return

    if not PAYMENTS_ACTIVE:
        if not await _ensure_user_allowed(message):
            return
        await message.answer(
            _format_subscription_status_message(message.from_user.id, None), format_mode="HTML"
        )
        return

    async with SessionLocal() as session:
        user, subscription = await ensure_user_with_subscription(
            session,
            message.from_user.id,
            **_user_profile_kwargs(message.from_user),
        )
        await _commit_session(session)
    if user.is_blocked:
        await _notify_blocked_user(message)
        return

    await message.answer(_format_subscription_status_message(message.from_user.id, subscription), format_mode="HTML")


@router.callback_query(F.data == "start_work")
async def handle_start_work(callback: CallbackQuery):
    await callback.message.answer("Отправь фото или введи текст для генерации, чтобы начать 🚀")


@router.message(F.photo)
async def save_photos(message: Message, bot: Bot, album: list[Message] | None = None):
    """
    Универсальный обработчик:
    - album != None → это альбом (media_group)
    - album == None → одиночное фото
    """

    messages = album if album else [message]

    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return

    removed_keys_total = []

    async with SessionLocal() as session:
        # создаём/обновляем юзера + его подписку
        user, _ = await ensure_user_with_subscription(
            session,
            message.from_user.id,
            **_user_profile_kwargs(message.from_user),
        )
        if user.is_blocked:
            await _commit_session(session)
            await _notify_blocked_user(message)
            return

        # Обработать каждое фото в сообщениях
        for msg in messages:
            photo = msg.photo[-1]

            # 1) скачать файл
            try:
                file_bytes, filename = await fetch_file_bytes(bot, photo.file_id)
            except Exception as exc:
                logger.exception("Failed to download photo: %s", exc)
                await message.answer("Не получилось скачать фото.")
                continue

            # 2) загрузить в S3
            try:
                object_key, url = await upload_bytes(file_bytes, filename, message.from_user.id)
            except Exception as exc:
                logger.exception("S3 upload failed: %s", exc)
                await message.answer("Ошибка загрузки файла в S3.")
                continue

            # 3) сохранить в базе (FIFO)
            user, removed = await set_user_photo(session, user, url, object_key)
            removed_keys_total.extend(removed)

        await session.commit()

        # получаем обновлённый список фото
        reference_urls = get_user_photo_urls(user)

    # 4) удалить старые файлы из S3
    for stale_key in removed_keys_total:
        try:
            await delete_object(stale_key)
        except Exception:
            pass

    # 5) Ответ пользователю
    if album:
        await message.answer(
            f"Галерея сохранена! У тебя теперь {len(reference_urls)} "
            f"базовых фото (максимум {MAX_REFERENCE_PHOTOS})."
        )
    else:
        await message.answer(
            f"Фото сохранено! Теперь у тебя {len(reference_urls)} "
            f"базовых фото (максимум {MAX_REFERENCE_PHOTOS})."
        )


@quota_guard("photo")
async def _generate_from_text_payload(
    message: Message, *, prompt: str, reference_urls: List[str]
) -> bool:
    wait_msg, stop_animation = await start_loading_animation(
        message, "🎨 Генерируем изображение"
    )
    try:
        result = await _perform_generation(prompt, reference_urls=reference_urls)
        if not result:
            await message.answer(
                "❌ Не удалось сгенерировать изображение, попробуй позже."
            )
            return False

        await _send_generation(message, result, caption=f"Готово! 🎨\n\n{prompt}")
        return True
    except Exception as exc:
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="_generate_from_text_payload._perform_generation",
        )
        await message.answer(
            "❌ Не удалось сгенерировать изображение, попробуй позже."
        )
        return False
    finally:
        stop_animation()
        await wait_msg.delete()


@quota_guard("photo")
async def _prompt_choice_generation(
    callback: CallbackQuery,
    *,
    prompt: str,
    mode: str,
    reference_urls: Optional[List[str]],
) -> bool:
    wait_msg, stop_animation = await start_loading_animation(
        callback.message, "🎨 Генерируем изображение"
    )
    try:
        result = await _perform_generation(prompt, reference_urls=reference_urls)
        if not result:
            await callback.answer(
                "❌ Не удалось сгенерировать картинку, попробуй снова.",
                show_alert=False,
            )
            return False

        caption_header = (
            "🪄 Результат редактуры:" if mode == "edit" else "🌄 Новая генерация:"
        )
        await _send_generation(
            callback.message,
            result,
            caption=f"{caption_header}\n\n{prompt}",
        )
        return True
    except Exception as exc:
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="_prompt_choice_generation._perform_generation",
        )
        await wait_msg.edit_text("Что-то пошло не так, попробуйте позже...")
        return False
    finally:
        stop_animation()
        await wait_msg.delete()


@quota_guard("photo")
async def _iterative_edit_generation(
    message: Message,
    *,
    prompt_text: str,
    reference_url: str,
) -> bool:
    wait_msg, stop_animation = await start_loading_animation(
        message, "🪄 Применяем правки, подожди немного"
    )
    try:
        result = await _perform_generation(prompt_text, reference_urls=[reference_url])
        if not result:
            await message.answer(
                "❌ Не удалось сгенерировать изображение, попробуй позже."
            )
            return False

        await _send_generation(
            message, result, caption=f"✨ Новая версия по запросу:\n\n{prompt_text}"
        )
        return True
    except Exception as exc:
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="_iterative_edit_generation._perform_generation",
        )
        await message.answer("Что-то пошло не так, попробуйте позже...")
        return False
    finally:
        stop_animation()
        await wait_msg.delete()


@quota_guard("photo")
async def _generate_without_base_payload(
    message: Message, *, prompt: str
) -> bool:
    wait_msg, stop_animation = await start_loading_animation(
        message, "🎨 Генерируем изображение"
    )
    try:
        result = await _perform_generation(prompt)
        if not result:
            await message.answer(
                "❌ Не удалось сгенерировать изображение, попробуй позже."
            )
            return False

        await _send_generation(message, result, caption=f"Готово! 🖼\n\n{prompt}")
        return True
    except Exception as exc:
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="_generate_without_base_payload._perform_generation",
        )
        await message.answer("Что-то пошло не так, попробуйте позже...")
        return False
    finally:
        stop_animation()
        await wait_msg.delete()


@quota_guard("text")
async def _generate_prompt_mode_payload(
    callback: CallbackQuery,
    state: FSMContext,
    *,
    base_text: str,
    mode: str,
) -> bool:
    instruction = (
        SYSTEM_PROMPT_FOR_EDIT if mode == "edit" else SYSTEM_PROMPT_FOR_CREATING
    )
    wait_msg, stop_animation = await start_loading_animation(
        callback.message, "💭 Думаем над промптами"
    )
    try:
        prompts = await prompt_service.generate(
            text=base_text,
            count=PROMPT_SUGGESTION_COUNT,
            instruction=instruction,
        )
    except Exception as exc:
        logger.warning("Prompt generation failed: %s", exc)
        prompts = generate_prompt_suggestions(base_text)[:PROMPT_SUGGESTION_COUNT]
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="_generate_prompt_mode_payload.prompt_service",
        )
    finally:
        stop_animation()
        await wait_msg.delete()

    if not prompts:
        await callback.message.answer(
            "Не удалось составить варианты промптов, попробуй другой текст."
        )
        return False

    state_data = await state.get_data()
    prompt_sets: Dict[str, List[str]] = state_data.get("prompt_sets", {})
    prompt_sources: Dict[str, str] = state_data.get("prompt_sources", {})
    prompt_sets[str(callback.message.message_id)] = prompts
    prompt_sources[str(callback.message.message_id)] = base_text
    await state.update_data(
        prompt_sets=_prune_map(prompt_sets),
        prompt_sources=_prune_map(prompt_sources),
    )

    await callback.message.answer(
        _format_prompt_message(prompts),
        reply_markup=prompt_suggestions_keyboard(
            callback.message.message_id, prompts, mode
        ),
        disable_web_page_preview=True,
    )
    await callback.answer()
    return True


@quota_guard("text")
async def _prompt_regeneration_payload(
    callback: CallbackQuery,
    *,
    base_text: str,
    mode: str,
    message_id: int,
) -> bool:
    instruction = (
        SYSTEM_PROMPT_FOR_EDIT if mode == "edit" else SYSTEM_PROMPT_FOR_CREATING
    )
    wait_msg, stop_animation = await start_loading_animation(
        callback.message, "♻️ Генерируем новые промпты"
    )
    try:
        prompts = await prompt_service.generate(
            text=base_text,
            count=PROMPT_SUGGESTION_COUNT,
            instruction=instruction,
        )
    except Exception as exc:
        logger.warning("Prompt regenerate failed: %s", exc)
        prompts = generate_prompt_suggestions(base_text)[:PROMPT_SUGGESTION_COUNT]
    finally:
        stop_animation()
        await wait_msg.delete()

    if not prompts:
        await callback.message.answer(
            "Не удалось составить варианты промптов, попробуй другой текст."
        )
        return False

    await callback.message.answer(
        _format_prompt_message(prompts),
        reply_markup=prompt_suggestions_keyboard(message_id, prompts, mode),
        disable_web_page_preview=True,
    )
    await callback.answer("Готово!")
    return True


# @router.message(Command("gen"))
@router.message(F.text & ~F.text.startswith("/") & ~F.reply_to_message)
async def generate_from_text(message: Message):
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return
    prompt = message.text.strip()
    if not prompt:
        await message.answer("Опиши, какую фотографию ты хочешь создать.")
        return

    async with SessionLocal() as session:
        user, subscription = await ensure_user_with_subscription(
            session,
            message.from_user.id,
            **_user_profile_kwargs(message.from_user),
        )
        if user.is_blocked:
            await _commit_session(session)
            await _notify_blocked_user(message)
            return
        photo_urls = get_user_photo_urls(user)
        photo_left = subscription.photo_left if PAYMENTS_ACTIVE else None
        if (
            PAYMENTS_ACTIVE
            and photo_left is not None
            and photo_left <= 0
            and not user.is_test_end
        ):
            user.is_test_end = True
            session.add(user)
        await _commit_session(session)

    if PAYMENTS_ACTIVE and photo_left is not None and photo_left <= 0:
        await message.answer(_quota_warning_message(message.from_user.id, "photo"), parse_mode="HTML")
        return

    if not photo_urls:
        await message.answer(
            "Сначала отправь фото, чтобы мы могли использовать его в качестве референса для генераций."
        )
        return

    await _generate_from_text_payload(
        message,
        prompt=prompt,
        reference_urls=photo_urls[:MAX_REFERENCE_PHOTOS],
    )



@router.message(Command("get_prompts"))
async def handle_get_prompts(message: Message, state: FSMContext):
    logger.info("starting promts from message")
    if not message.from_user or not message.text:
        return

    if not message.from_user or not message.text:
        return

    # Убираем команду и оставляем только аргументы
    # message.text может быть "/get_prompts текст"
    parts = message.text.split(maxsplit=1)
    user_text = parts[1].strip() if len(parts) > 1 else ""

    # Проверка на пустой текст
    if not user_text:
        await message.answer(
            "После команды <code>/get_prompts</code> укажи текст, "
            "по которому нужно создать варианты промптов.\n\n"
            "Например:\n"
            "<code>/get_prompts Текст для моего видео про путешествия</code>"
        )
        return

    if not await _ensure_user_allowed(message):
        return
    
    normalized = normalize_text(message.text)
    await state.update_data(base_text=normalized)

    await message.answer(
        "Выбери, какие промпты сгенерировать:",
        reply_markup=prompt_mode_keyboard(),
    )

@router.message(F.text & ~F.via_bot & ~F.text.startswith("/") & ~F.reply_to_message)
async def handle_plain_text(message: Message, state: FSMContext):
    await message.answer("Используй одну из команд <code>/get_prompts</code>, <code>/free_gen</code>, <code>/gen</code>")

@router.callback_query(PromptChoiceCallback.filter())
async def handle_prompt_choice(
    callback: CallbackQuery,
    callback_data: PromptChoiceCallback,
    state: FSMContext,
):
    logger.info("Inside handle_prompt_choice")
    if not callback.from_user:
        await callback.answer("Не распознали пользователя", show_alert=True)
        return

    state_data = await state.get_data()
    prompt_sets: Dict[str, List[str]] = state_data.get("prompt_sets", {})
    prompts = prompt_sets.get(str(callback_data.message_id))
    if not prompts or callback_data.index >= len(prompts):
        await callback.answer(
            "Этот набор промптов устарел, пришли текст ещё раз.", show_alert=True
        )
        return

    prompt = prompts[callback_data.index]
    mode = state_data.get("prompt_mode", "edit")  # <- сохраняли при выборе стиля

    logger.info(f"Selected prompt {prompt} (mode={mode})")

    async with SessionLocal() as session:
        user, subscription = await ensure_user_with_subscription(
            session,
            callback.from_user.id,
            **_user_profile_kwargs(callback.from_user),
        )
        if user.is_blocked:
            await _commit_session(session)
            await _notify_blocked_user(callback)
            return
        photo_urls = get_user_photo_urls(user)
        photo_left = subscription.photo_left if PAYMENTS_ACTIVE else None
        if (
            PAYMENTS_ACTIVE
            and photo_left is not None
            and photo_left <= 0
            and not user.is_test_end
        ):
            user.is_test_end = True
            session.add(user)
        await _commit_session(session)

    if PAYMENTS_ACTIVE and photo_left is not None and photo_left <= 0:
        await callback.message.answer(
            _quota_warning_message(callback.from_user.id, "photo"), parse_mode="HTML"
        )
        await callback.answer("Закончились генерации", show_alert=True)
        return

    await callback.answer("Генерируем…", show_alert=False)
    reference_urls = (
        photo_urls[:MAX_REFERENCE_PHOTOS] if mode == "edit" and photo_urls else None
    )
    await _prompt_choice_generation(
        callback,
        prompt=prompt,
        mode=mode,
        reference_urls=reference_urls,
    )


@router.callback_query(PromptRegenCallback.filter())
async def handle_prompt_regenerate(callback: CallbackQuery, callback_data: PromptRegenCallback, state: FSMContext):
    if not await _ensure_user_allowed(callback):
        return
    mode = callback_data.mode
    state_data = await state.get_data()
    prompt_sources: Dict[str, str] = state_data.get("prompt_sources", {})
    base_text = prompt_sources.get(str(callback_data.message_id))
    if not base_text:
        await callback.answer("Не нашли исходный текст, пришли сообщение ещё раз.", show_alert=True)
        return

    await _prompt_regeneration_payload(
        callback,
        base_text=base_text,
        mode=mode,
        message_id=callback_data.message_id,
    )


@router.message(F.reply_to_message & F.text)
async def handle_iterative_edit(message: Message, bot: Bot):
    """
    Пользователь отвечает на фото (которое бот сгенерировал),
    и пишет текст для доработки — "Сделай ночь", "Добавь свет", и т.п.
    """
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return

    reply = message.reply_to_message

    # Проверяем, что ответ именно на фото
    if not reply.photo:
        return

    # Получаем URL файла из Telegram
    try:
        file_id = reply.photo[-1].file_id
        file_info = await bot.get_file(file_id)
        file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"
    except Exception as exc:
        logger.error(f"Не удалось получить фото из сообщения: {exc}")
        await message.answer("❌ Не удалось загрузить фото для правок.")
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="handle_iterative_edit")
        return


    await _iterative_edit_generation(
        message,
        prompt_text=message.text,
        reference_url=file_url,

    )

@router.message(Command("free_gen"))
async def generate_without_base(message: Message, command: CommandObject):
    """
    Генерация изображения по тексту без базового фото.
    """
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return

    prompt = (command.args or "").strip()
    if not prompt:
        await message.answer("Укажи текст после команды: `/free_gen твой промпт`.")
        return

    async with SessionLocal() as session:
        user, subscription = await ensure_user_with_subscription(
            session,
            message.from_user.id,
            **_user_profile_kwargs(message.from_user),
        )
        if user.is_blocked:
            await _commit_session(session)
            await _notify_blocked_user(message)
            return
        photo_left = subscription.photo_left if PAYMENTS_ACTIVE else None
        if (
            PAYMENTS_ACTIVE
            and photo_left is not None
            and photo_left <= 0
            and not user.is_test_end
        ):
            user.is_test_end = True
            session.add(user)
        await _commit_session(session)

    if PAYMENTS_ACTIVE and photo_left is not None and photo_left <= 0:
        await message.answer(_quota_warning_message(message.from_user.id, "photo"), parse_mode="HTML")
        return

    await _generate_without_base_payload(message, prompt=prompt)
        

@router.callback_query(PromptModeCallback.filter())
async def handle_prompt_mode(callback: CallbackQuery, callback_data: PromptModeCallback, state: FSMContext):
    mode = callback_data.mode  # "edit" or "new"
    state_data = await state.get_data()
    base_text = state_data.get("base_text")
    if not base_text:
        await callback.answer("Не нашли исходный текст, попробуй снова.", show_alert=True)
        return

    if not PAYMENTS_ACTIVE and not await _ensure_user_allowed(callback):
        return

    text_left: Optional[int] = None
    if PAYMENTS_ACTIVE:
        async with SessionLocal() as session:
            user, subscription = await ensure_user_with_subscription(
                session,
                callback.from_user.id,
                **_user_profile_kwargs(callback.from_user),
            )
            if user.is_blocked:
                await _commit_session(session)
                await _notify_blocked_user(callback)
                return
            text_left = subscription.text_left
            if text_left is not None and text_left <= 0 and not user.is_test_end:
                user.is_test_end = True
                session.add(user)
            await _commit_session(session)

        if text_left is not None and text_left <= 0:
            await callback.message.answer(
                _quota_warning_message(callback.from_user.id, "text"), parse_mode="HTML"
            )
            await callback.answer("Нет лимита на промпты", show_alert=True)
            return

    # Сохраняем текущий режим
    await state.update_data(prompt_mode=mode)

    await _generate_prompt_mode_payload(callback, state, base_text=base_text, mode=mode)

# @router.message(F.text == "/reset_photos")
@router.message(Command("reset_photos"))
async def reset_photos(message: Message):
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return



    async with SessionLocal() as session:
        user, _ = await ensure_user_with_subscription(session, message.from_user.id, **_user_profile_kwargs(message.from_user), )
        if user is None:
            await message.answer("У тебя нет сохранённых фотографий.")
            return
        if user.is_blocked:
            await _commit_session(session)
            await _notify_blocked_user(message)
            return

        # clear db
        removed_keys = await clear_user_photo(session, user)

        await session.commit()

    # Delete from S3
    for stale_key in removed_keys:
        try:
            await delete_object(stale_key)
        except Exception as exc:
            logger.warning(f"Failed to delete S3 object {stale_key}: {exc}")

    await message.answer(
        "Все базовые фотографии удалены!\n"
        "Можешь загрузить новые, отправив фото сюда."
    )



@router.message(Command("my_photos"))
async def my_photos(message: Message):
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return

    async with SessionLocal() as session:
        user, _ = await ensure_user_with_subscription(
            session,
            message.from_user.id,
            **_user_profile_kwargs(message.from_user),
        )
        photo_urls = get_user_photo_urls(user)
        await _commit_session(session)

    # Нет фото
    if not photo_urls:
        await message.answer(
            "У тебя пока нет загруженных фото.\n\n"
            "Отправь 1–3 своих фотографии, чтобы я мог использовать их при генерации."
        )
        return

    # Есть только одно фото — просто отправляем
    if len(photo_urls) == 1:
        try:
            await message.answer_photo(photo_urls[0])
        except Exception:
            await message.answer(f"⚠️ Не удалось загрузить фото:\n{photo_urls[0]}")
        return


    media_group = []
    for url in photo_urls:
        try:
            media_group.append(InputMediaPhoto(media=url))
        except Exception:
            await message.answer(f"⚠️ Не удалось добавить фото:\n{url}")

    # Отправляем альбом
    try:
        await message.answer_media_group(media_group)
    except Exception:
        # fallback — если Telegram ругнётся на какой-то URL
        await message.answer("⚠️ Не удалось отправить альбом. Покажу фото по одному.")
        for url in photo_urls:
            try:
                await message.answer_photo(url)
            except Exception:
                await message.answer(f"⚠️ Ошибка загрузки фото:\n{url}")
