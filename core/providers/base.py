from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class BaseProvider(ABC):
    """
    Абстрактный класс для всех провайдеров моделей.
    Все реализации должны возвращать единый формат: dict с результатом.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        reference_urls: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Сгенерировать картинку по тексту (и опционально картинке).
        reference_urls — список ссылок на изображения-референсы.
        Возвращает dict с полями:
            {
              "type": "url" | "base64",
              "data": "...",   # ссылка или base64-строка
              "provider": "имя провайдера"
            }
        """
        pass


class BasePromptProvider(ABC):
    """Абстракция для сервисов, генерирующих текстовые промпты."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def generate_prompts(self, text: str, count: int) -> List[str]:
        pass
