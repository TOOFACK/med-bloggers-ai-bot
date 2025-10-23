from typing import List

from .base import BasePromptProvider, BaseProvider
from .providers import (
    CometPromptProvider,
    CometProvider,
    OpenRouterPromptProvider,
    OpenRouterProvider,
)


def init_image_providers(
    comet_api_key: str | None,
    openrouter_api_key: str | None,
    comet_base_url: str = "https://api.cometapi.com/v1beta",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_model: str = "google/gemini-2.5-flash-image-preview",
) -> List[BaseProvider]:
    providers: List[BaseProvider] = []
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
