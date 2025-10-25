import asyncio
import html
import logging
from typing import Any, Dict, List, Optional

from aiogram import Bot, F, Router, types
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
    PROMPT_MODEL
)
from core.db import SessionLocal
from core.providers import init_image_providers, init_prompt_providers
from core.s3 import S3ConfigError, delete_object, upload_bytes
from core.services import ModelService, PromptService
from core.storage import (
    ensure_user,
    get_user_photo_urls,
    set_user_photo,
    MAX_REFERENCE_PHOTOS,
)
from core.utils import (
    build_file_url,
    fetch_file_bytes,
    generate_prompt_suggestions,
    input_file_from_base64,
    normalize_text,
)

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
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

router = Router()

image_providers = init_image_providers(
    comet_api_key=COMET_API_KEY,
    openrouter_api_key=OPENROUTER_API_KEY,
    comet_base_url=COMET_BASE_URL,
    openrouter_base_url=OPENROUTER_BASE_URL,
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
    provider = result.get("provider")
    footer = f"\n\nИсточник: {provider}" if provider else ""
    full_caption = f"{caption}{footer}"

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


@router.message(Command("start"))
async def start(message: Message):
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return

    async with SessionLocal() as session:
        await ensure_user(session, message.from_user.id)
        await _commit_session(session)

    instructions = (
        "🎨 <b>Привет!</b>\n\n"
        "Я — AI-бот, который помогает <b>создавать</b> и <b>редактировать изображения</b> по описанию.\n\n"
        "🪄 Что я умею:\n"
        "• Пришли до 3 своих фото, где хорошо видно лицо (лучше с разных ракурсов) — я сохраню их как базу.\n"
        "• <code>/gen</code> + описание — отредактирую твоё фото по заданию.\n"
        "• <code>/free_gen</code> + описание — создам новое изображение с нуля.\n"
        "• Просто пришли текст (например, пост или идею для видео) — предложу варианты визуализаций. Ты выберешь: с нуля или на основе фото.\n"
        "• Можно редактировать итеративно — просто ответь на картинку и напиши, что изменить.\n\n"
        "👇 Нажми кнопку, чтобы начать!"
    )

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Начать работу", callback_data="start_work")]
        ]
    )

    await message.answer(instructions, reply_markup=keyboard)


@router.callback_query(F.data == "start_work")
async def handle_start_work(callback: CallbackQuery):
    await callback.message.answer("Отправь фото или введи /free_gen, чтобы начать 🚀")


@router.message(F.photo)
async def save_photo(message: Message, bot: Bot):
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return

    photo = message.photo[-1]
    try:
        file_bytes, filename = await fetch_file_bytes(bot, photo.file_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to download photo: %s", exc)
        await message.answer("Не получилось скачать фото, попробуй ещё раз.")
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="save_photo.fetch_file_bytes(bot, photo.file_id)")
        return

    removed_object_keys: List[str] = []
    reference_urls: List[str] = []

    async with SessionLocal() as session:
        user = await ensure_user(session, message.from_user.id)

        try:
            object_key, url = await upload_bytes(
                file_bytes, filename, message.from_user.id
            )
            logger.info(f" url {url}")
        except S3ConfigError as exc:
            logger.exception("S3 configuration error: %s", exc)
            await message.answer(
                "Хранилище изображений не настроено. Сообщи администратору."
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("S3 upload failed: %s", exc)
            await message.answer("Не удалось сохранить фото, попробуй позже.")
            sale_client.send_error_message(
                error_text=str(exc),
                error_place="save_photo.upload_bytes(file_bytes, filename, message.from_user.id)")
            return

        user, removed_object_keys = await set_user_photo(session, user, url, object_key)
        reference_urls = get_user_photo_urls(user)
        await _commit_session(session)

    for stale_key in removed_object_keys:
        try:
            await delete_object(stale_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to delete old S3 object %s: %s", stale_key, exc
            )
            sale_client.send_error_message(
                error_text=str(exc),
                error_place="save_photo.delete_object(stale_key)",
            )

    photo_count = len(reference_urls)
    base_text = (
        "Используй <code>/gen &lt;описание&gt;</code> для генераций с учётом снимков."
    )
    if photo_count <= 1:
        prefix = "Фото сохранено! Оно станет базовым."
    else:
        prefix = (
            f"Фото сохранено! Теперь у тебя {photo_count} базовых фото "
            f"(максимум {MAX_REFERENCE_PHOTOS})."
        )

    await message.answer(f"{prefix} {base_text}")


@router.message(Command("gen"))
async def generate_from_text(message: Message, command: CommandObject):
    if not message.from_user:
        await message.answer("Не распознали пользователя.")
        return
    prompt = (command.args or "").strip()
    if not prompt:
        await message.answer("Укажи текст после команды: `/gen твой промпт`.")
        return

    async with SessionLocal() as session:
        user = await ensure_user(session, message.from_user.id)
        photo_urls = get_user_photo_urls(user)
        await _commit_session(session)

    if not photo_urls:
        await message.answer(
            "Сначала отправь фото, чтобы мы могли использовать его в качестве референса для генераций."
        )
        return

    wait_msg, stop_animation = await start_loading_animation(
        message, "🎨 Генерируем изображение"
    )

    try:
        result = await _perform_generation(
            prompt, reference_urls=photo_urls[:MAX_REFERENCE_PHOTOS]
        )
        if not result:
            await message.answer(
                "❌ Не удалось сгенерировать изображение, попробуй позже."
            )
            stop_animation()
            await wait_msg.delete()
            return

        # Останавливаем анимацию
        stop_animation()
        await wait_msg.delete()

        await _send_generation(message, result, caption=f"Готово! 🎨\n\n{prompt}")

    except Exception as e:
        stop_animation()
        await wait_msg.edit_text(f"⚠️ Ошибка: {e}")



@router.message(F.text & ~F.via_bot & ~F.text.startswith("/") & ~F.reply_to_message)
async def handle_post(message: Message, state: FSMContext):
    logger.info("starting promts from message")
    if not message.from_user or not message.text:
        return

    normalized = normalize_text(message.text)
    await state.update_data(base_text=normalized)

    await message.answer(
        "Выбери, какие промпты сгенерировать:",
        reply_markup=prompt_mode_keyboard(),
    )


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
    await callback.answer("Генерируем…", show_alert=False)

    logger.info(f"Selected prompt {prompt} (mode={mode})")

    async with SessionLocal() as session:
        user = await ensure_user(session, callback.from_user.id)
        photo_urls = get_user_photo_urls(user)
        await _commit_session(session)

    # 🌀 Анимация
    wait_msg, stop_animation = await start_loading_animation(
        callback.message, "🎨 Генерируем изображение"
    )

    try:
        if mode == "edit" and photo_urls:
            logger.info(f"Using reference photos: {photo_urls}")
            result = await _perform_generation(
                prompt, reference_urls=photo_urls[:MAX_REFERENCE_PHOTOS]
            )
        else:
            logger.info("Generating from text only (no reference)")
            result = await _perform_generation(prompt)

        if not result:
            stop_animation()
            await callback.answer("❌ Не удалось сгенерировать картинку, попробуй снова.", show_alert=False)
            stop_animation()
            await wait_msg.delete()
            return

        stop_animation()
        await wait_msg.delete()

        caption_header = (
            "🪄 Результат редактуры:" if mode == "edit" else "🌄 Новая генерация:"
        )
        await _send_generation(
            callback.message,
            result,
            caption=f"{caption_header}\n\n{prompt}",
        )

    except Exception as e:
        stop_animation()
        sale_client.send_error_message(
            error_text=str(e),
            error_place="handle_prompt_choice._perform_generation"
        )
        await wait_msg.edit_text(f"⚠️ Ошибка: {e}")


@router.callback_query(PromptRegenCallback.filter())
async def handle_prompt_regenerate(callback: CallbackQuery, callback_data: PromptRegenCallback, state: FSMContext):
    mode = callback_data.mode
    state_data = await state.get_data()
    prompt_sources: Dict[str, str] = state_data.get("prompt_sources", {})
    base_text = prompt_sources.get(str(callback_data.message_id))
    if not base_text:
        await callback.answer("Не нашли исходный текст, пришли сообщение ещё раз.", show_alert=True)
        return

    if mode == "edit":
        instruction = SYSTEM_PROMPT_FOR_EDIT
    else:
        instruction = SYSTEM_PROMPT_FOR_CREATING

    wait_msg, stop_animation = await start_loading_animation(callback.message, "♻️ Генерируем новые промпты")

    try:
        prompts = await prompt_service.generate(
            text=base_text,
            count=PROMPT_SUGGESTION_COUNT,
            instruction=instruction
        )
    except Exception as exc:
        logger.warning("Prompt regenerate failed: %s", exc)
        prompts = generate_prompt_suggestions(base_text)[:PROMPT_SUGGESTION_COUNT]
    finally:
        stop_animation()
        await wait_msg.delete()

    await callback.message.answer(
        _format_prompt_message(prompts),
        reply_markup=prompt_suggestions_keyboard(callback_data.message_id, prompts, mode),
        disable_web_page_preview=True,
    )
    await callback.answer("Готово!")


@router.message(F.reply_to_message & F.text)
async def handle_iterative_edit(message: Message, bot: Bot):
    """
    Пользователь отвечает на фото (которое бот сгенерировал),
    и пишет текст для доработки — "Сделай ночь", "Добавь свет", и т.п.
    """
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

    # Запускаем анимацию "генерация"
    wait_msg, stop_animation = await start_loading_animation(
        message, "🪄 Применяем правки, подожди немного"
    )

    try:
        # Отправляем в пайплайн Nano-Banana (через Comet/Gemini)
        result = await _perform_generation(message.text, reference_urls=[file_url])
        if not result:
            stop_animation()
            await message.answer(
                "❌ Не удалось сгенерировать изображение, попробуй позже."
            )
            return

        stop_animation()
        await wait_msg.delete()

        await _send_generation(
            message, result, caption=f"✨ Новая версия по запросу:\n\n{message.text}"
        )

    except Exception as exc:
        stop_animation()
        await wait_msg.edit_text(f"⚠️ Ошибка при генерации: {exc}")
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="handle_iterative_edit._perform_generation")


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

    # Показываем анимацию
    wait_msg, stop_animation = await start_loading_animation(
        message, "🎨 Генерируем изображение"
    )

    try:
        # ⚡ Без reference_urls → чисто текстовая генерация
        result = await _perform_generation(prompt)
        if not result:
            stop_animation()
            await message.answer(
                "❌ Не удалось сгенерировать изображение, попробуй позже."
            )
            return

        stop_animation()
        await wait_msg.delete()

        await _send_generation(message, result, caption=f"Готово! 🖼\n\n{prompt}")

    except Exception as e:
        stop_animation()
        await wait_msg.edit_text(f"⚠️ Ошибка: {e}")
        sale_client.send_error_message(
            error_text=str(e),
            error_place="generate_without_base._perform_generation")
        

@router.callback_query(PromptModeCallback.filter())
async def handle_prompt_mode(callback: CallbackQuery, callback_data: PromptModeCallback, state: FSMContext):
    mode = callback_data.mode  # "edit" or "new"
    state_data = await state.get_data()
    base_text = state_data.get("base_text")
    if not base_text:
        await callback.answer("Не нашли исходный текст, попробуй снова.", show_alert=True)
        return

    # Сохраняем текущий режим
    await state.update_data(prompt_mode=mode)

    # Подбираем инструкцию
    if mode == "edit":
        instruction = SYSTEM_PROMPT_FOR_EDIT
    else:
        instruction = SYSTEM_PROMPT_FOR_CREATING

    wait_msg, stop_animation = await start_loading_animation(callback.message, "💭 Думаем над промптами")

    try:
        prompts = await prompt_service.generate(text=base_text, count=PROMPT_SUGGESTION_COUNT, instruction=instruction)
    except Exception as exc:
        logger.warning("Prompt generation failed: %s", exc)
        prompts = generate_prompt_suggestions(base_text)[:PROMPT_SUGGESTION_COUNT]
        sale_client.send_error_message(
            error_text=str(exc),
            error_place="handle_prompt_mode.prompt_service.generate")
        await callback.message.answer("❌ Не удалось сгенерировать промпты, попробуй позже.")
    finally:
        
        stop_animation()
        await wait_msg.delete()

    if not prompts:
        await callback.message.answer("Не удалось составить варианты промптов, попробуй другой текст.")
        return

    # Сохраняем
    state_data = await state.get_data()
    prompt_sets: Dict[str, List[str]] = state_data.get("prompt_sets", {})
    prompt_sources: Dict[str, str] = state_data.get("prompt_sources", {})
    prompt_sets[str(callback.message.message_id)] = prompts
    prompt_sources[str(callback.message.message_id)] = base_text
    await state.update_data(prompt_sets=_prune_map(prompt_sets), prompt_sources=_prune_map(prompt_sources))

    await callback.message.answer(
        _format_prompt_message(prompts),
        reply_markup=prompt_suggestions_keyboard(callback.message.message_id, prompts, mode),
        disable_web_page_preview=True,
    )
    await callback.answer()
