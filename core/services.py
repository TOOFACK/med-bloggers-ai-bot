import logging
from typing import Any, Dict, List, Optional, Sequence

from .providers.base import BasePromptProvider, BaseProvider

logger = logging.getLogger(__name__)


class ModelService:
    """Фолбэк-сервис поверх списка провайдеров."""

    def __init__(self, providers: Sequence[BaseProvider]):
        self.providers = list(providers)

    async def generate_from_text(self, prompt: str) -> Dict[str, Any]:
        return await self._generate(prompt, reference_urls=None)

    async def generate_with_reference(
        self, prompt: str, reference_urls: Sequence[str]
    ) -> Dict[str, Any]:
        return await self._generate(prompt, reference_urls=reference_urls)

    async def _generate(
        self, prompt: str, reference_urls: Optional[Sequence[str]]
    ) -> Dict[str, Any]:
        if not self.providers:
            raise RuntimeError("Нет доступных провайдеров для генерации.")

        errors: list[str] = []
        for provider in self.providers:
            try:
                result = await provider.generate(prompt, reference_urls=reference_urls)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Provider %s failed: %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
                continue

            result.setdefault("provider", provider.name)
            if "type" not in result or "data" not in result:
                logger.warning(
                    "Provider %s returned unexpected payload: %s", provider.name, result
                )
                errors.append(f"{provider.name}: invalid payload")
                continue

            return result

        error_text = (
            "; ".join(errors) if errors else "не удалось сгенерировать изображение."
        )
        raise RuntimeError(f"Все провайдеры упали: {error_text}")


class PromptService:
    """Генерация списков промптов с фолбэком по провайдерам."""

    def __init__(self, providers: Sequence[BasePromptProvider]):
        self.providers = list(providers)

    async def generate(self, text: str, count: int, instruction:str) -> List[str]:
        if not self.providers:
            raise RuntimeError("Нет доступных провайдеров для генерации промптов.")

        errors: list[str] = []
        for provider in self.providers:
            logger.info(f"trying to use {provider} as prompt generator with instruction {instruction} and text {text}")
            try:
                prompts = await provider.generate_prompts(text, count, instruction)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Prompt provider %s failed: %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
                continue

            filtered = [p for p in prompts if p]
            if filtered:
                return filtered[:count]
            errors.append(f"{provider.name}: пустой результат")

        error_text = "; ".join(errors) if errors else "не удалось получить промпты."
        raise RuntimeError(f"Все провайдеры промптов упали: {error_text}")
