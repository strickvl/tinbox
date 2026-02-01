"""Translation engine for Tinbox."""

from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.translation.litellm import LiteLLMTranslator
from tinbox.core.types import TranslationConfig


def create_translator(config: TranslationConfig) -> ModelInterface:
    """Create a translator instance based on configuration.

    Note: Model selection (provider/model_name) is handled per-request via
    TranslationRequest.model and TranslationRequest.model_params, not at
    translator construction time.

    Args:
        config: Translation configuration

    Returns:
        Configured translator instance
    """
    return LiteLLMTranslator()


__all__ = [
    "LiteLLMTranslator",
    "ModelInterface",
    "TranslationError",
    "TranslationRequest",
    "TranslationResponse",
    "create_translator",
]
