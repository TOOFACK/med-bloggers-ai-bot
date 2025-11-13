import asyncio
import base64
from io import BytesIO
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
from openai import OpenAI
from PIL import Image
from .base import BasePromptProvider, BaseProvider

try:
    from google import genai
    from google.genai import types
    import google.auth
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    google = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class CometProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.cometapi.com/v1beta"):
        self.api_key = api_key
        self.base_url = base_url

    async def generate(
        self,
        prompt: str,
        reference_urls: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/models/gemini-2.5-flash-image-preview:generateContent"
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

        parts: List[Dict[str, Any]] = [{"text": prompt}]

        if reference_urls:
            logger.info(f'Start generating with CometProvider and reference_urls {len(reference_urls)}')
        else:
            logger.info(f'Start generating with CometProvider a new image')
        async with aiohttp.ClientSession() as session:
            for image_url in reference_urls or []:
                inline_data = await self._image_url_to_inline(session, image_url)
                if inline_data:
                    parts.append({"inline_data": inline_data})

            payload = {


                "contents": [{"role": "user", "parts": parts}],


                "generationConfig": {"responseModalities": ["IMAGE"]},
            }

            logger.info(f'Sending post to CometProvider')
            async with session.post(url, headers=headers, json=payload, timeout=40) as resp:
                if resp.status != 200:
                    details = await resp.text()
                    raise RuntimeError(f"CometProvider error {resp.status}: {details}")
                result = await resp.json()
                img_b64 = _extract_comet_base64(result)
                if not img_b64:
                    raise RuntimeError(
                        f"CometProvider invalid response: {json.dumps(result)[:400]}"
                    )
                return {"type": "base64", "data": img_b64, "provider": self.name}

    async def _image_url_to_inline(
        self, session: aiohttp.ClientSession, image_url: str
    ) -> Optional[Dict[str, Any]]:
        async with session.get(image_url) as response:
            if response.status != 200:
                return None
            mime = response.headers.get("Content-Type", "image/jpeg")
            if mime in ("application/octet-stream", None, ""):
                mime = "image/jpeg"  # 🔧 исправляем некорректный MIME из Telegram
            raw = await response.read()
            encoded = base64.b64encode(raw).decode("utf-8")
            return {"mime_type": mime, "data": encoded}


class OpenRouterProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.5-flash-image",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def generate(
        self,
        prompt: str,
        reference_urls: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if reference_urls:
            logger.info(f'Start generating with OpenRouterProvider and reference_urls {len(reference_urls)}')
        else:
            logger.info(f'Start generating with OpenRouterProvider a new image')
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        for image_url in reference_urls or []:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                modalities=["image", "text"],  # ✅ добавляем сюда
            ),
        )

        payload = _extract_openrouter_payload(completion)
        return payload | {"provider": self.name}


class VertexAIProvider(BaseProvider):
    def __init__(
        self,
        credentials_path: str,
        project: str,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash-image",
        scopes: Optional[Sequence[str]] = None,
        aspect_ratio: Optional[str] = None,
    ):
        if genai is None or types is None or google is None:
            raise RuntimeError(
                "Пакет google-genai не установлен. Добавьте его в зависимости проекта."
            )

        scope_list = list(scopes or ["https://www.googleapis.com/auth/cloud-platform"])
        credentials, _ = google.auth.load_credentials_from_file(
            credentials_path,
            scopes=scope_list,
        )

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )
        self.model = model
        self.aspect_ratio = aspect_ratio

    async def generate(
        self,
        prompt: str,
        reference_urls: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        logger.info("Start generating with VertexAIProvider")
        contents: List[Any] = [prompt]

        if reference_urls:
            async with aiohttp.ClientSession() as session:
                for image_url in reference_urls:
                    part = await self._image_url_to_part(session, image_url)
                    if part is not None:
                        contents.append(part)

        config_kwargs: Dict[str, Any] = {"response_modalities": ["IMAGE"]}
        if self.aspect_ratio:
            config_kwargs["image_config"] = types.ImageConfig(
                aspect_ratio=self.aspect_ratio
            )
        config = types.GenerateContentConfig(**config_kwargs)

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            ),
        )

        img_b64 = self._extract_image_base64(response)

        if not img_b64:
            raise RuntimeError("Vertex AI не вернул изображение.")
        return {"type": "base64", "data": img_b64, "provider": self.name}



    async def _image_url_to_part(self, session, image_url):
        # Google сам скачает и обработает URI, нам не нужно трогать байты
        try:
            return types.Part.from_uri(
                file_uri=image_url,
                mime_type="image/jpeg"
            )
        except Exception as exc:
            logger.warning("VertexAIProvider: can't create Part from %s: %s", image_url, exc)
            return None


    # def _extract_image_base64(self, response: Any) -> Optional[str]:
    #     parts = getattr(response, "parts", None)
    #     if not parts:
    #         logger.warning("VertexAIProvider: response.parts is empty or None")
    #         return None

    #     for part in parts:
    #         try:
    #             out_img = part.as_image()
    #         except Exception:
    #             continue

    #         if not out_img or not out_img.image_bytes:
    #             continue

    #         raw = out_img.image_bytes


    #         # --- 1. Попытка распознать base64 строку ---
    #         # PNG b64 начинается с iVBOR..., JPEG с /9j/
    #         if raw.startswith(b"iVBOR") or raw.startswith(b"/9j/"):
    #             try:
    #                 logger.info("Gemini returned base64-wrapped image, returning...")
    #                 raw = base64.b64decode(raw)
    #             except Exception as exc:
    #                 logger.warning("Failed to decode base64 image: %s", exc)
    #                 continue

    #         # --- 2. Проверяем сигнатуру ---
    #         if not (raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff")):
    #             # Это НЕ картинка
    #             try:
    #                 text_preview = raw.decode("utf-8", errors="replace")
    #             except Exception:
    #                 text_preview = str(raw)
    #             logger.error("===== GEMINI RETURNED NON-IMAGE DATA =====")
    #             logger.error("first 300 bytes:\n%s", text_preview[:300])
    #             continue

    #         # --- 3. Пытаемся открыть как PIL ---
    #         try:
    #             pil_img = Image.open(BytesIO(raw)).convert("RGB")
    #         except Exception as exc:
    #             logger.warning("PIL failed to decode image: %s", exc)
    #             continue

    #         # --- 4. Telegram-safe JPEG ---
    #         buf = BytesIO()
    #         pil_img.save(buf, "JPEG", quality=90)
    #         jpeg_bytes = buf.getvalue()

    #         # --- 5. Отдаём base64 JPEG ---
    #         return base64.b64encode(jpeg_bytes).decode("utf-8")

    #     return None

    def _extract_image_base64(self, response: Any) -> Optional[str]:
        parts = getattr(response, "parts", None)
        if not parts:
            return None

        for part in parts:
            try:
                out_img = part.as_image()
            except Exception:
                continue

            if not out_img or not out_img.image_bytes:
                continue

            raw = out_img.image_bytes

            # Если Gemini отдал PNG в виде base64 (iVBOR..., /9j/...), просто оформим это как base64
            try:
                # Проверяем, выглядит ли raw как base64-строка
                # (начинается с ASCII-символов и может быть декодирована)
                base64.b64decode(raw, validate=True)
                # Значит raw уже base64 → просто возвращаем строку
                return raw.decode("utf-8")
            except Exception:
                # Значит raw — это реальные байты изображения, нужно закодировать
                return base64.b64encode(raw).decode("utf-8")

        return None




def _extract_comet_base64(payload: Dict[str, Any]) -> Optional[str]:
    candidates = payload.get("candidates") or []
    if not candidates:
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    for part in parts:
        if not isinstance(part, dict):
            continue
        inline = part.get("inline_data") or part.get("inlineData")
        if inline and inline.get("data"):
            return inline["data"]
        file_data = part.get("file_data") or part.get("fileData")
        if file_data and file_data.get("data"):
            return file_data["data"]
    return None


def _extract_openrouter_payload(completion: Any) -> Dict[str, Any]:
    """
    Универсальный парсер ответа от OpenRouter / Gemini / Comet.
    Возвращает {"type": "base64"|"url"|"text", "data": ...}
    """

    choice = completion.choices[0]
    message = getattr(choice, "message", None)

    if message is None:
        return {"type": "text", "data": str(choice)}

    # ---- Универсальное извлечение контента ----
    raw_content = getattr(message, "content", None)
    if hasattr(message, "model_dump"):
        message_dict = message.model_dump()
        raw_content = message_dict.get("content", raw_content)

    images_base64: list[str] = []
    image_urls: list[str] = []
    texts: list[str] = []

    # ✅ 1. OpenRouter / Gemini формат с message["images"]
    if hasattr(message, "images") and message.images:
        for img in message.images:
            img_url = img.get("image_url", {}).get("url")
            if img_url:
                if img_url.startswith("data:image"):
                    # data:image/png;base64,...
                    b64 = img_url.split("base64,")[-1]
                    images_base64.append(b64)
                else:
                    image_urls.append(img_url)

    # ✅ 2. OpenAI-style content = list of parts
    def _collect_from_item(item: Any) -> None:
        if not isinstance(item, dict):
            if isinstance(item, str):
                texts.append(item)
            return
        item_type = item.get("type", "")
        if item_type in {"output_image", "image"}:
            if isinstance(item.get("image_base64"), str):
                images_base64.append(item["image_base64"])
            elif isinstance(item.get("data"), str):
                images_base64.append(item["data"])
            elif isinstance(item.get("b64_json"), str):
                images_base64.append(item["b64_json"])
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                image_urls.append(image_url["url"])
            elif isinstance(image_url, str):
                image_urls.append(image_url)
        elif item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url_value = image_url.get("url")
                if isinstance(url_value, str):
                    image_urls.append(url_value)
            elif isinstance(image_url, str):
                image_urls.append(image_url)
        elif item_type == "text":
            text_value = item.get("text")
            if isinstance(text_value, str):
                texts.append(text_value)
        else:
            for key in ("text", "content"):
                value = item.get(key)
                if isinstance(value, str):
                    texts.append(value)

    if isinstance(raw_content, list):
        for entry in raw_content:
            _collect_from_item(entry)
    elif isinstance(raw_content, str):
        texts.append(raw_content)
    elif raw_content is not None:
        _collect_from_item(raw_content)  # type: ignore[arg-type]

    # ✅ 3. Собираем итог
    if images_base64:
        return {"type": "base64", "data": images_base64[0]}
    if image_urls:
        return {"type": "url", "data": image_urls[0]}
    joined_text = "\n".join(texts).strip()
    return {"type": "text", "data": joined_text or ""}



class CometPromptProvider(BasePromptProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.cometapi.com/v1",
        model: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate_prompts(self, text: str, count: int, instruction: str) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        logger.info(f"use Comet with model {self.model} with prompt {instruction}")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            "temperature": 0.9,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=40) as resp:
                if resp.status != 200:
                    details = await resp.text()
                    raise RuntimeError(
                        f"CometPromptProvider error {resp.status}: {details}"
                    )

                result = await resp.json()
                try:
                    raw_text = result["choices"][0]["message"]["content"].strip()
                    logger.info(f"Comet returned (first 200 chars): {raw_text[:200]}")
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"CometPromptProvider invalid response: {result}"
                    ) from exc

        logger.info(f"Parse text from API {raw_text}")
        return _normalize_prompt_list(raw_text, count)


class OpenRouterPromptProvider(BasePromptProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.0-flash-001",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def generate_prompts(self, text: str, count: int ,instruction: str) -> List[str]:

        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text},
                ],
            ),
        )

        message_content = completion.choices[0].message.content
        if isinstance(message_content, str):
            raw_text = message_content
        elif isinstance(message_content, list) and message_content:
            item = message_content[0]
            if isinstance(item, dict):
                raw_text = item.get("text") or item.get("content") or str(item)
            else:
                raw_text = str(item)
        else:
            raw_text = str(message_content)
        return _normalize_prompt_list(raw_text, count)


def _normalize_prompt_list(raw_text: str, expected_count: int) -> List[str]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines:
            fence = lines[0]
            if fence.startswith("```"):
                lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    prompts: List[str] = []
    if raw_text:
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                prompts = [str(item).strip() for item in data if str(item).strip()]
        except json.JSONDecodeError:
            pass

    if not prompts:
        # fallback: split by newline or semicolon
        separators = ["\n", ";"]
        for sep in separators:
            if sep in raw_text:
                prompts = [
                    part.strip(" •\t") for part in raw_text.split(sep) if part.strip()
                ]
                break
        if not prompts and raw_text:
            prompts = [raw_text]

    unique_prompts = []
    seen = set()
    for prompt in prompts:
        if prompt and prompt not in seen:
            unique_prompts.append(prompt)
            seen.add(prompt)
            if len(unique_prompts) >= expected_count:
                break

    return unique_prompts[:expected_count]
