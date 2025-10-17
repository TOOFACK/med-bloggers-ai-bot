from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from core.utils import truncate_for_button


class PromptChoiceCallback(CallbackData, prefix="prompt"):
    message_id: int
    index: int


class PromptRegenCallback(CallbackData, prefix="regen"):
    message_id: int


def prompt_suggestions_keyboard(
    message_id: int, prompts: list[str]
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for idx, prompt in enumerate(prompts):
        builder.button(
            text=f"Использовать промпт {idx}",
            callback_data=PromptChoiceCallback(message_id=message_id, index=idx).pack(),
        )
    builder.button(
        text="🔄 Ещё варианты",
        callback_data=PromptRegenCallback(message_id=message_id).pack(),
    )
    builder.adjust(1)
    return builder.as_markup()
