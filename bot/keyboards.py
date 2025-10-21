from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from core.utils import truncate_for_button


class PromptChoiceCallback(CallbackData, prefix="prompt"):
    message_id: int
    index: int


class PromptRegenCallback(CallbackData, prefix="regen"):
    message_id: int
    mode: str

class PromptModeCallback(CallbackData, prefix="mode"):
    mode: str  # "edit" | "new"

def prompt_mode_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="🧠 Для редактуры фото", callback_data=PromptModeCallback(mode="edit").pack())
    builder.button(text="🌄 Для генерации с нуля", callback_data=PromptModeCallback(mode="new").pack())
    builder.adjust(1)
    return builder.as_markup()

def prompt_suggestions_keyboard(message_id: int, prompts: list[str], mode: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for idx, prompt in enumerate(prompts):
        builder.button(
            text=f"Использовать промпт {idx+1}",
            callback_data=PromptChoiceCallback(message_id=message_id, index=idx).pack(),
        )
    builder.button(
        text="🔄 Ещё в том же стиле",
        callback_data=PromptRegenCallback(message_id=message_id, mode=mode).pack(),
    )
    # Дополнительно кнопки для переключения режима
    other_mode = "new" if mode == "edit" else "edit"
    builder.button(
        text="🧠 Регенерировать в другом стиле",
        callback_data=PromptRegenCallback(message_id=message_id, mode=other_mode).pack(),
    )
    builder.adjust(1)
    return builder.as_markup()