from aiogram import BaseMiddleware
from aiogram.types import Message
import asyncio
from typing import Dict, List, Any, Callable, Awaitable


class AlbumMiddleware(BaseMiddleware):
    def __init__(self, latency: float = 0.2):
        """
        latency — время ожидания, пока Telegram пришлёт все элементы альбома.
        """
        super().__init__()
        self.latency = latency
        self.albums: Dict[str, List[Message]] = {}

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any]
    ) -> Any:

        # не альбом → обычное фото/сообщение
        if event.media_group_id is None:
            return await handler(event, data)

        # альбом
        media_group_id = event.media_group_id

        # собираем сообщения в группу
        if media_group_id not in self.albums:
            self.albums[media_group_id] = [event]
            await asyncio.sleep(self.latency)  # ждём остальные элементы
            data["album"] = self.albums[media_group_id]
            del self.albums[media_group_id]
            return await handler(event, data)

        else:
            # продолжаем наполнять
            self.albums[media_group_id].append(event)
            return None  # НЕ вызываем handler для промежуточных элементов
