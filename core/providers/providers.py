import asyncio
import base64
import json
from typing import Optional, Dict, Any, List, Sequence

import aiohttp
from openai import OpenAI

from .base import BaseProvider, BasePromptProvider
import logging

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

        async with aiohttp.ClientSession() as session:
            for image_url in reference_urls or []:
                inline_data = await self._image_url_to_inline(session, image_url)
                if inline_data:
                    parts.append({"inline_data": inline_data})

            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseModalities": ["IMAGE"]},
            }

            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    details = await resp.text()
                    raise RuntimeError(f"CometProvider error {resp.status}: {details}")
                result = await resp.json()
                img_b64 = _extract_comet_base64(result)
                if not img_b64:
                    raise RuntimeError(f"CometProvider invalid response: {json.dumps(result)[:400]}")
                return {"type": "base64", "data": img_b64, "provider": self.name}

    async def _image_url_to_inline(self, session: aiohttp.ClientSession, image_url: str) -> Optional[Dict[str, Any]]:
        async with session.get(image_url) as response:
            if response.status != 200:
                return None
            mime = response.headers.get("Content-Type", "image/jpeg")
            raw = await response.read()
            encoded = base64.b64encode(raw).decode("utf-8")
            return {"mime_type": mime, "data": encoded}


class OpenRouterProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.5-flash-image-preview",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def generate(
        self,
        prompt: str,
        reference_urls: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        for image_url in reference_urls or []:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})

        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            ),
        )

        payload = _extract_openrouter_payload(completion)
        return payload | {"provider": self.name}


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
    choice = completion.choices[0]
    message = getattr(choice, "message", None)

    if message is None:
        return {"type": "text", "data": str(choice)}

    raw_content = getattr(message, "content", None)
    if hasattr(message, "model_dump"):
        message_dict = message.model_dump()
        raw_content = message_dict.get("content", raw_content)

    images_base64: List[str] = []
    image_urls: List[str] = []
    texts: List[str] = []

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
        base_url: str = "https://api.cometapi.com/v1beta",
        model: str = "models/gemini-2.0-flash:generateContent",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate_prompts(self, text: str, count: int) -> List[str]:
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/{self.model}"
        instruction = (
            "Сгенерируй {count} креативных промптов для генерации изображений на основе следующего текста. "
            "Ответ верни в формате JSON списка строк без дополнительных комментариев. Сделай промпты на русском языке"
        ).format(count=count)

        logger.info(f"use Comet with model {self.model} with prompt {instruction}")
        payload = {
            "contents": [
                {
                    "role": "system",
                    "parts": [{"text": instruction}],
                },
                {
                    "role": "user",
                    "parts": [{"text": text}],
                },
            ]
        }

        raw_text = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    details = await resp.text()
                    raise RuntimeError(f"CometPromptProvider error {resp.status}: {details}")
                result = await resp.json()
                try:
                    raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"Comet return {raw_text}")
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(f"CometPromptProvider invalid response: {result}") from exc
        return _normalize_prompt_list(raw_text, count)


class OpenRouterPromptProvider(BasePromptProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.0-flash-exp",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def generate_prompts(self, text: str, count: int) -> List[str]:
        prompt = (
            "Сформируй {count} коротких промптов для генерации изображений на русском языке, "
            "используя следующий контекст. Верни результат в виде JSON-массива строк."
        ).format(count=count)

        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
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
                prompts = [part.strip(" •\t") for part in raw_text.split(sep) if part.strip()]
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
