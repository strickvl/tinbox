"""Shared helpers for translation provider implementations.

Private module — not part of the public API. Contains prompt construction,
content preparation, response parsing, and validation logic shared across
OpenAI and Anthropic translators.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable, Iterator
from typing import Any, NoReturn

from PIL import Image
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from tinbox.core.cost import calculate_usage_cost
from tinbox.core.translation.interface import (
    ProviderAccessError,
    TranslationError,
    TranslationRequest,
)
from tinbox.core.types import GlossaryEntry, ModelType
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


def usage_tokens(response: Any, input_field: str, output_field: str) -> tuple[int, int]:
    """Read (input, output) token counts from an SDK response; missing counts are 0."""
    usage = getattr(response, "usage", None)
    counts = [
        getattr(usage, name, 0) if usage else 0 for name in (input_field, output_field)
    ]
    input_tokens, output_tokens = (c if isinstance(c, int) else 0 for c in counts)
    return input_tokens, output_tokens


def request_cost(
    request: TranslationRequest,
    model: ModelType,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Price one request's usage with the request's model name and custom prices."""
    return calculate_usage_cost(
        model,
        input_tokens,
        output_tokens,
        model_name=request.model_params.get("model_name"),
        local=request.local_model,
        price_in=request.price_in,
        price_out=request.price_out,
    )


_MAX_RETRY_WAIT_SECONDS = 120.0
_exponential_wait = wait_exponential(multiplier=1, min=4, max=_MAX_RETRY_WAIT_SECONDS)


# HTTP statuses an account-limit error can arrive with; 402 always means no credit
_ACCOUNT_LIMIT_STATUSES = {400, 402, 403, 429}
_ACCOUNT_LIMIT_MESSAGE = re.compile(
    r"credit balance is too low"  # Anthropic, sent as a 400
    r"|\b(?:spending|spend)\s+cap\b"  # Gemini project spending cap
    r"|limit:\s*0(?!\d|\.\d)",  # Gemini: no quota at all for this model or tier
    re.IGNORECASE,
)


def _nested_dicts(value: object) -> Iterator[dict]:
    """Yield every dict inside a decoded JSON error body, outermost first."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _nested_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_dicts(item)


def is_quota_exhausted(error: BaseException) -> bool:
    """Return True when an API error means the account cannot make more requests.

    Waiting will not make these succeed, so they are neither retried nor
    treated as a one-page failure:

    - OpenAI: a 429 with code ``insufficient_quota``.
    - Anthropic: a 400 saying the credit balance is too low.
    - Gemini: a 429 for a daily quota (``quotaId`` containing ``PerDay`` or
      ``Daily``), a quota limit of 0, a spending cap, or a 402.

    Gemini also sends per-minute limits as 429 ``RESOURCE_EXHAUSTED``; those
    stay retryable.
    """
    status = getattr(error, "status_code", None)
    if status == 402:
        return True
    if status is not None and status not in _ACCOUNT_LIMIT_STATUSES:
        return False
    details = list(_nested_dicts(getattr(error, "body", None)))
    markers = {getattr(error, "code", None), getattr(error, "type", None)}
    markers.update(d.get(key) for d in details for key in ("code", "type"))
    if "insufficient_quota" in markers:
        return True
    quota_ids = [d.get("quotaId") for d in details]
    if any(
        isinstance(quota_id, str) and ("PerDay" in quota_id or "Daily" in quota_id)
        for quota_id in quota_ids
    ):
        return True
    return status is not None and bool(_ACCOUNT_LIMIT_MESSAGE.search(str(error)))


def _retry_after_seconds(error: BaseException | None) -> float | None:
    """Read a numeric ``retry-after`` header from an SDK error's HTTP response."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, _MAX_RETRY_WAIT_SECONDS)


def _wait_before_retry(retry_state: RetryCallState) -> float:
    error = retry_state.outcome.exception() if retry_state.outcome else None
    hinted = _retry_after_seconds(error)
    return hinted if hinted is not None else _exponential_wait(retry_state)


def build_retry_decorator(
    rate_limit_error_cls: type[Exception] | None,
    logger_instance: Any,
) -> Any:
    """Build a retry decorator for transient rate limit errors.

    Quota-exhausted errors are not retried. A numeric ``retry-after`` header
    sets the wait (capped at 120 s); otherwise the wait grows exponentially.
    After the last attempt the original exception is re-raised.

    Args:
        rate_limit_error_cls: The rate limit exception class to retry on,
            or None if the SDK is not installed.
        logger_instance: Logger to use for retry sleep logging.

    Returns:
        A retry decorator, or a no-op decorator if the exception class is None.
    """
    if rate_limit_error_cls is None:

        def noop[T: Callable](fn: T) -> T:
            return fn

        return noop

    def is_transient(error: BaseException) -> bool:
        return isinstance(error, rate_limit_error_cls) and not is_quota_exhausted(error)

    return retry(
        retry=retry_if_exception(is_transient),
        wait=_wait_before_retry,
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger_instance, log_level=20),
        reraise=True,
    )


def reraise_provider_error(
    error: Exception,
    rate_limit_error_cls: type[Exception] | None,
    access_error_classes: tuple[type[Exception], ...],
) -> NoReturn:
    """Re-raise an SDK error as the exception the retry and algorithm code expect.

    Authentication, permission, and account-limit errors (see
    ``is_quota_exhausted``) become ``ProviderAccessError``. Other rate-limit
    errors pass through unchanged so the retry decorator can retry them.
    Everything else becomes ``TranslationError``.
    """
    is_rate_limit = rate_limit_error_cls is not None and isinstance(
        error, rate_limit_error_cls
    )
    if isinstance(error, access_error_classes) or is_quota_exhausted(error):
        raise ProviderAccessError(f"Translation failed: {error!s}") from error
    if is_rate_limit:
        raise error
    raise TranslationError(f"Translation failed: {error!s}") from error


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
) -> tuple[str, str | bytes, str] | None:
    """Validate and prepare a translation request.

    Validates language pair, checks for empty content, and extracts whitespace.

    Args:
        request: The translation request

    Returns:
        Tuple of (prefix, clean_content, suffix), preserving image bytes,
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

    if isinstance(request.content, bytes):
        return "", request.content, ""

    prefix, clean, suffix = extract_whitespace_formatting(request.content)
    logger.debug("Content prefix: ", prefix=prefix)
    logger.debug("Content suffix: ", suffix=suffix)
    logger.debug("Clean content: ", content=clean)
    return prefix, clean, suffix


def validate_image_content(content: str | bytes) -> None:
    """Validate that image bytes are a valid image.

    Args:
        content: Raw image bytes

    Raises:
        TranslationError: If the image data is invalid
    """
    if not isinstance(content, bytes):
        raise TranslationError("Translation failed: Invalid image data")
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
