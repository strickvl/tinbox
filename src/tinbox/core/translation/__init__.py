"""Translation engine for Tinbox."""

from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import ModelType, TranslationConfig


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


__all__ = [
    "ModelInterface",
    "TranslationError",
    "TranslationRequest",
    "TranslationResponse",
    "create_translator",
]
