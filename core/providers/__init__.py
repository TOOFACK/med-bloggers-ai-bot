import logging
from pathlib import Path
from typing import List

from .base import BasePromptProvider, BaseProvider
from .providers import (
    CometPromptProvider,
    CometProvider,
    OpenRouterPromptProvider,
    OpenRouterProvider,
    VertexAIProvider,
)

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def init_image_providers(
    comet_api_key: str | None,
    openrouter_api_key: str | None,
    comet_base_url: str = "https://api.cometapi.com/v1beta",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_model: str = "google/gemini-2.5-flash-image-preview",
    vertex_credentials_path: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str = "us-central1",
    vertex_model: str = "gemini-2.5-flash-image",
    vertex_aspect_ratio: str | None = None,
) -> List[BaseProvider]:
    providers: List[BaseProvider] = []

    if vertex_credentials_path and vertex_project:
        resolved_path = _resolve_credentials_path(vertex_credentials_path)
        if not resolved_path:
            logger.warning(
                "VertexAIProvider пропущен: не найден файл с ключами %s",
                vertex_credentials_path,
            )
        else:
            providers.append(
                VertexAIProvider(
                    credentials_path=resolved_path,
                    project=vertex_project,
                    location=vertex_location,
                    model=vertex_model,
                    aspect_ratio=vertex_aspect_ratio,
                )
            )
    
    if comet_api_key:
        providers.append(CometProvider(api_key=comet_api_key, base_url=comet_base_url))
    if openrouter_api_key:
        providers.append(
            OpenRouterProvider(
                api_key=openrouter_api_key,
                base_url=openrouter_base_url,
                model=openrouter_model,
            )
        )


    return providers


def init_prompt_providers(
    comet_api_key: str | None,
    openrouter_api_key: str | None,
    comet_base_url: str = "https://api.cometapi.com/v1",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
) -> List[BasePromptProvider]:
    providers: List[BasePromptProvider] = []
    if comet_api_key:
        providers.append(
            CometPromptProvider(
                api_key=comet_api_key,
                base_url=comet_base_url,
            )
        )
    if openrouter_api_key:
        providers.append(
            OpenRouterPromptProvider(
                api_key=openrouter_api_key,
                base_url=openrouter_base_url,
            )
        )
    return providers


def _resolve_credentials_path(path_value: str) -> str | None:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (_PROJECT_ROOT / candidate).resolve()
    if candidate.exists():
        return str(candidate)
    return None
