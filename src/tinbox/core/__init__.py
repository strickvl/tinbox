"""Core functionality for Tinbox."""

from tinbox.core.processor import DocumentContent, load_document
from tinbox.core.translation import (
    ModelInterface,
    TranslationRequest,
    TranslationResponse,
    create_translator,
)
from tinbox.core.translation.algorithms import translate_document
from tinbox.core.types import (
    FileType,
    ModelType,
    TranslationConfig,
    TranslationResult,
)

__all__ = [
    "DocumentContent",
    "FileType",
    "ModelType",
    "TranslationConfig",
    "TranslationResult",
    "load_document",
    "translate_document",
]
