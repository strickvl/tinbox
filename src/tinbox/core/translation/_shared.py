"""Shared helpers for translation provider implementations.

Private module — not part of the public API. Contains prompt construction,
content preparation, response parsing, and validation logic shared across
OpenAI and Anthropic translators.
"""

from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tinbox.core.translation.interface import (
    TranslationError,
    TranslationRequest,
)
from tinbox.core.types import GlossaryEntry
from tinbox.utils.chunks import extract_whitespace_formatting
from tinbox.utils.language import LanguageError, validate_language_pair
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


def get_model_name(request: TranslationRequest) -> str:
    """Extract the model name from a translation request.

    Args:
        request: The translation request

    Returns:
        Model name string for the API call

    Raises:
        TranslationError: If model name is missing
    """
    model_name = request.model_params.get("model_name")
    if not model_name:
        raise TranslationError("No model name provided")
    return model_name


def build_retry_decorator(
    rate_limit_error_cls: type[Exception] | None,
    logger_instance: Any,
) -> Any:
    """Build a retry decorator for rate limit handling.

    Args:
        rate_limit_error_cls: The rate limit exception class to retry on,
            or None if the SDK is not installed.
        logger_instance: Logger to use for retry sleep logging.

    Returns:
        A retry decorator, or a no-op decorator if the exception class is None.
    """
    if rate_limit_error_cls is None:

        def noop(fn):  # type: ignore[no-untyped-def]
            return fn

        return noop
    return retry(
        retry=retry_if_exception_type(rate_limit_error_cls),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger_instance, log_level=20),
    )


def build_system_instruction(request: TranslationRequest) -> str:
    """Build the system-level instruction for translation.

    Args:
        request: The translation request

    Returns:
        System instruction string
    """
    if request.glossary:
        json_instruction = (
            "IMPORTANT: You MUST respond with a valid JSON object only, no markdown, no extra text. "
            "The JSON must have exactly this structure:\n"
            '{"translation": "<your translation here>", "glossary_extension": [{"term": "<source term>", "translation": "<target term>"}]}\n'
            "The glossary_extension array can be empty if no new terms need to be added."
        )
    else:
        json_instruction = (
            "IMPORTANT: You MUST respond with a valid JSON object only, no markdown, no extra text. "
            "The JSON must have exactly this structure:\n"
            '{"translation": "<your translation here>"}'
        )

    return (
        f"You are a professional translator. Translate the following content "
        f"from '{request.source_lang}' to '{request.target_lang}'. "
        f"Maintain the original formatting and structure (including whitespaces, line breaks, etc.). "
        f"Include ALL markup/formatting in the translation. But do not fix tags or formatting errors - "
        f"you only receive chunks of text to translate and later chunks might contain the 'missing' tags. "
        f"Translate only the content, do not add any explanations or notes. "
        f"IMPORTANT: Your translation should ALWAYS(!) be in '{request.target_lang}' language.\n\n"
        f"{json_instruction}"
    )


def build_glossary_instruction(request: TranslationRequest) -> str | None:
    """Build the glossary instruction text for a translation request.

    Args:
        request: The translation request

    Returns:
        Glossary instruction string, or None if no glossary is set
    """
    if not request.glossary:
        return None
    return (
        "Use this glossary for consistent translations:\n"
        f"{request.glossary.to_context_string()}\n\n"
        "When you encounter these terms, use the provided translations. "
        "If you encounter new important terms that benefit from consistent translation "
        "(technical terms, proper nouns, domain vocabulary, names, etc.), include them in the glossary_extension field in your response. "
        "Only include terms that are important for consistent translation."
    )


def build_user_messages(request: TranslationRequest, clean_content: str) -> list[dict]:
    """Build the user message sequence (context, glossary, translation instruction).

    Args:
        request: The translation request
        clean_content: Whitespace-stripped content to translate

    Returns:
        List of user message dicts with 'role' and 'content' keys
    """
    messages: list[dict] = []

    if request.context:
        messages.append(
            {
                "role": "user",
                "content": f"[TRANSLATION_CONTEXT]{request.context}[/TRANSLATION_CONTEXT]",
            }
        )

    glossary_text = build_glossary_instruction(request)
    if glossary_text:
        messages.append({"role": "user", "content": glossary_text})

    # Translation instruction
    if request.content_type.startswith("text/"):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"[TRANSLATE_THIS]{clean_content}[/TRANSLATE_THIS]\n\n"
                    f"Translate the text between the [TRANSLATE_THIS]-tags to '{request.target_lang}' (preserve ALL markup/formatting and line-breaks). "
                    f"Respond ONLY with the JSON object as instructed."
                ),
            }
        )

    return messages


def build_image_user_content(request: TranslationRequest, image_b64: str) -> list[dict]:
    """Build multimodal content blocks for image translation.

    Args:
        request: The translation request
        image_b64: Base64-encoded image data

    Returns:
        List of content blocks (text + image_url) for the final user message
    """
    return [
        {
            "type": "text",
            "text": (
                f"Translate the contents of the image from {request.source_lang} "
                f"to {request.target_lang}. Respond ONLY with the JSON object as instructed."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        },
    ]


def prepare_request(
    request: TranslationRequest,
) -> tuple[str, str, str] | None:
    """Validate and prepare a translation request.

    Validates language pair, checks for empty content, and extracts whitespace.

    Args:
        request: The translation request

    Returns:
        Tuple of (prefix, clean_content, suffix) for text requests,
        or None if the content is empty (caller should return early).

    Raises:
        TranslationError: If language validation fails
    """
    try:
        validate_language_pair(request.source_lang, request.target_lang)
    except LanguageError as e:
        raise TranslationError(f"Translation failed: {e}") from e

    if not request.content or (
        isinstance(request.content, str) and not request.content.strip()
    ):
        logger.info("Empty content, returning as-is")
        return None

    prefix, clean, suffix = extract_whitespace_formatting(request.content)
    logger.debug("Content prefix: ", prefix=prefix)
    logger.debug("Content suffix: ", suffix=suffix)
    logger.debug("Clean content: ", content=clean)
    return prefix, clean, suffix


def validate_image_content(content: bytes) -> None:
    """Validate that image bytes are a valid image.

    Args:
        content: Raw image bytes

    Raises:
        TranslationError: If the image data is invalid
    """
    try:
        Image.open(io.BytesIO(content))
    except Exception as e:
        raise TranslationError("Translation failed: Invalid image data") from e


def parse_translation_response(
    raw_text: str, glossary_enabled: bool
) -> tuple[str, list[GlossaryEntry]]:
    """Parse the JSON translation response from a model.

    Args:
        raw_text: Raw text content from the model response
        glossary_enabled: Whether glossary extension parsing is enabled

    Returns:
        Tuple of (translated_text, glossary_updates)

    Raises:
        TranslationError: If JSON is invalid or missing required fields
    """
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, AttributeError) as e:
        raise TranslationError(f"Invalid JSON response format: {e!s}")

    if not isinstance(parsed, dict) or "translation" not in parsed:
        raise TranslationError("Missing translation field in JSON response")

    text = parsed["translation"]

    if not text or not text.strip():
        raise TranslationError("No content returned from model")

    # Parse glossary updates if present and enabled
    glossary_updates: list[GlossaryEntry] = []
    if glossary_enabled and parsed.get("glossary_extension"):
        glossary_data = parsed["glossary_extension"]
        if isinstance(glossary_data, list):
            glossary_updates = [
                GlossaryEntry(term=entry["term"], translation=entry["translation"])
                for entry in glossary_data
                if isinstance(entry, dict)
                and "term" in entry
                and "translation" in entry
            ]

    logger.debug("Glossary updates: ", glossary_updates=glossary_updates)
    return text, glossary_updates
