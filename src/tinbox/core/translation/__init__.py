"""Translation engine for Tinbox."""

import os

from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import ModelType, TranslationConfig

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"


def create_translator(config: TranslationConfig) -> ModelInterface:
    """Create a translator instance based on configuration.

    Dispatches to the appropriate SDK-backed translator based on the
    model provider. Uses lazy imports so that missing optional extras
    (openai, anthropic) only fail when actually needed.

    Args:
        config: Translation configuration

    Returns:
        Configured translator instance

    Raises:
        TranslationError: If the provider is unsupported or SDK is missing
    """
    if config.model in {ModelType.OPENAI, ModelType.OLLAMA, ModelType.GEMINI}:
        from tinbox.core.translation.openai import OpenAITranslator

        return OpenAITranslator()

    if config.model is ModelType.ANTHROPIC:
        from tinbox.core.translation.anthropic import AnthropicTranslator

        return AnthropicTranslator()

    raise TranslationError(
        f"Unsupported model provider: {config.model}. "
        f"Supported providers: {', '.join(m.value for m in ModelType)}"
    )


def provider_endpoint(model: ModelType) -> str:
    """Return the base URL requests for ``model`` will go to.

    OpenAI and Anthropic honour their SDKs' ``*_BASE_URL`` environment variables,
    so a stale value can send requests somewhere unexpected; the plan and
    ``tinbox doctor`` show it.
    """
    if model is ModelType.OLLAMA:
        return OLLAMA_BASE_URL
    if model is ModelType.GEMINI:
        return GEMINI_BASE_URL
    if model is ModelType.ANTHROPIC:
        return _base_url("ANTHROPIC_BASE_URL", ANTHROPIC_DEFAULT_BASE_URL)
    return _base_url("OPENAI_BASE_URL", OPENAI_DEFAULT_BASE_URL)


def _base_url(variable: str, default: str) -> str:
    # Like the SDKs, fall back only when the variable is missing: an empty
    # value is used as-is, and every request then fails to connect
    value = os.environ.get(variable)
    if value is None:
        return default
    return value if value.strip() else f"none: {variable} is set but empty"


__all__ = [
    "ModelInterface",
    "TranslationError",
    "TranslationRequest",
    "TranslationResponse",
    "create_translator",
    "provider_endpoint",
]
