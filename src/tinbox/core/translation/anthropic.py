"""Anthropic SDK-based translation implementation.

Handles Anthropic (Claude) models through the anthropic Python SDK.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Literal

from tinbox.core.cost import calculate_usage_cost
from tinbox.core.translation._shared import (
    build_glossary_instruction,
    build_retry_decorator,
    build_system_instruction,
    get_model_name,
    parse_translation_response,
    prepare_request,
    validate_image_content,
)
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import ModelType
from tinbox.utils.logging import get_logger

try:
    from anthropic import AsyncAnthropic, RateLimitError
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment, misc]
    RateLimitError = None  # type: ignore[assignment, misc]

logger = get_logger(__name__)

# Thinking budget mapping for reasoning_effort levels
_THINKING_BUDGETS: dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}


def _check_anthropic_installed() -> None:
    """Raise a clear error if the anthropic package is not installed."""
    if AsyncAnthropic is None:
        raise TranslationError(
            "Anthropic support is not installed. "
            "Install with: pip install tinbox[anthropic] or pip install tinbox[all]"
        )


completion_with_retry = build_retry_decorator(RateLimitError, logger)


def _thinking_config(
    effort: Literal["minimal", "low", "medium", "high"],
) -> dict | None:
    """Map reasoning effort to Anthropic thinking configuration."""
    if effort == "minimal":
        return None
    budget = _THINKING_BUDGETS.get(effort)
    if budget is None:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _extract_text_from_response(content_blocks: list) -> str:
    """Extract text content from Anthropic response content blocks.

    Filters out thinking blocks and other non-text content, collecting
    only TextBlock text in order.
    """
    texts = []
    for block in content_blocks:
        if hasattr(block, "type") and block.type == "text":
            texts.append(block.text)

    if not texts:
        raise TranslationError("Invalid response format: no text content in response")

    return "".join(texts)


class AnthropicTranslator(ModelInterface):
    """Anthropic SDK-based translator for Claude models."""

    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 100000,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: AsyncAnthropic | None = None  # type: ignore[assignment]

    def _get_client(self) -> AsyncAnthropic:  # type: ignore[return]
        """Get or create an AsyncAnthropic client."""
        _check_anthropic_installed()
        if self._client is None:
            self._client = AsyncAnthropic()
        return self._client

    def _build_content_blocks(
        self, request: TranslationRequest, clean_content: str | bytes
    ) -> list[dict]:
        """Build the user content blocks for the Anthropic messages API.

        Anthropic uses a single user message with multiple content blocks
        rather than multiple user messages.
        """
        blocks: list[dict] = []

        if request.context:
            blocks.append(
                {
                    "type": "text",
                    "text": f"[TRANSLATION_CONTEXT]{request.context}[/TRANSLATION_CONTEXT]",
                }
            )

        glossary_text = build_glossary_instruction(request)
        if glossary_text:
            blocks.append({"type": "text", "text": glossary_text})

        if request.content_type.startswith("text/"):
            blocks.append(
                {
                    "type": "text",
                    "text": (
                        f"[TRANSLATE_THIS]{clean_content}[/TRANSLATE_THIS]\n\n"
                        f"Translate the text between the [TRANSLATE_THIS]-tags to '{request.target_lang}' "
                        f"(preserve ALL markup/formatting and line-breaks). "
                        f"Respond ONLY with the JSON object as instructed."
                    ),
                }
            )
        else:
            image_b64 = base64.b64encode(clean_content).decode()  # type: ignore[arg-type]
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                }
            )
            blocks.append(
                {
                    "type": "text",
                    "text": (
                        f"Translate the contents of the image from {request.source_lang} "
                        f"to {request.target_lang}. Respond ONLY with the JSON object as instructed."
                    ),
                }
            )

        return blocks

    @completion_with_retry
    async def _make_completion_request(
        self, request: TranslationRequest, clean_content: str | bytes
    ):  # type: ignore[no-untyped-def]
        """Make a messages API request with retry on rate limits."""
        client = self._get_client()

        kwargs: dict = {
            "model": get_model_name(request),
            "system": build_system_instruction(request),
            "messages": [
                {
                    "role": "user",
                    "content": self._build_content_blocks(request, clean_content),
                }
            ],
            "max_tokens": self.max_tokens,
        }

        thinking = _thinking_config(request.reasoning_effort)
        if thinking:
            kwargs["thinking"] = thinking
            # Anthropic requires temperature=1 when thinking is enabled, so we omit it
        else:
            kwargs["temperature"] = self.temperature

        try:
            return await client.messages.create(**kwargs)
        except Exception as e:
            if RateLimitError and isinstance(e, RateLimitError):
                raise
            raise TranslationError(f"Translation failed: {e!s}") from e

    async def translate(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Translate content using the Anthropic SDK."""
        start_time = datetime.now()

        try:
            prep = prepare_request(request)
            if prep is None:
                time_taken = (datetime.now() - start_time).total_seconds()
                return TranslationResponse(
                    text=request.content if isinstance(request.content, str) else "",
                    tokens_used=0,
                    cost=0.0,
                    time_taken=time_taken,
                )

            content_prefix, clean_content, content_suffix = prep

            if request.content_type.startswith("image/"):
                validate_image_content(clean_content)  # type: ignore[arg-type]

            response = await self._make_completion_request(request, clean_content)

            if not hasattr(response, "content") or not response.content:
                raise TranslationError("No response from model")

            if hasattr(response, "stop_reason") and response.stop_reason != "end_turn":
                raise TranslationError(
                    f"Invalid stop reason from model: {response.stop_reason}, expected 'end_turn'"
                )

            raw_text = _extract_text_from_response(response.content)

            text, glossary_updates = parse_translation_response(
                raw_text,
                glossary_enabled=bool(request.glossary),
            )

            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "input_tokens", 0)
                output_tokens = getattr(response.usage, "output_tokens", 0)

            tokens = input_tokens + output_tokens
            cost = calculate_usage_cost(
                ModelType.ANTHROPIC, input_tokens, output_tokens
            )
            time_taken = (datetime.now() - start_time).total_seconds()
            final_text = content_prefix + text + content_suffix
            logger.debug("Final text: ", final_text=final_text)

            return TranslationResponse(
                text=final_text,
                tokens_used=tokens,
                cost=cost,
                time_taken=time_taken,
                glossary_updates=glossary_updates,
            )

        except TranslationError:
            raise
        except Exception as e:
            logger.error(f"Translation failed: {e!s}")
            raise TranslationError(f"Translation failed: {e!s}") from e

    async def validate_model(self) -> bool:
        """Verify the model is available and properly configured."""
        try:
            client = self._get_client()
            response = await client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return hasattr(response, "content") and len(response.content) > 0
        except Exception as e:
            logger.error(f"Model validation failed: {e!s}")
            return False
