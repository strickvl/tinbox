"""OpenAI SDK-based translation implementation.

Handles OpenAI, Ollama (OpenAI-compatible), and Gemini (OpenAI-compatible)
providers through the openai Python SDK with different base URLs.
"""

from __future__ import annotations

import base64
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI as OpenAIClient

from tinbox.core.translation._shared import (
    build_glossary_instruction,
    build_image_user_content,
    build_retry_decorator,
    build_system_instruction,
    build_user_messages,
    get_model_name,
    parse_translation_response,
    prepare_request,
    request_cost,
    reraise_provider_error,
    usage_tokens,
    validate_image_content,
)
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
    TranslationWithGlossaryResponse,
    TranslationWithoutGlossaryResponse,
)
from tinbox.core.translation.translategemma import build_translategemma_prompt
from tinbox.core.types import GlossaryEntry, ModelType, TranslationProtocol
from tinbox.utils.logging import get_logger

try:
    from openai import (
        AsyncOpenAI,
        AuthenticationError,
        PermissionDeniedError,
        RateLimitError,
    )

    _ACCESS_ERRORS: tuple[type[Exception], ...] = (
        AuthenticationError,
        PermissionDeniedError,
    )
except ImportError:
    AsyncOpenAI = None
    RateLimitError = None
    _ACCESS_ERRORS = ()

try:
    # Private SDK helper; if a future SDK moves it, fall back to parse() below
    from openai.lib._parsing import type_to_response_format_param
except ImportError:
    type_to_response_format_param = None

logger = get_logger(__name__)

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_GPT6_MINIMAL_EFFORT = {
    "gpt-6.1-sol": "low",
    "gpt-6-astra": "low",
    "gpt-6-sol": "none",
    "gpt-6-luna": "none",
}


def _check_openai_installed() -> type[OpenAIClient]:
    """Raise a clear error if the openai package is not installed."""
    if AsyncOpenAI is None:
        raise TranslationError(
            "OpenAI-compatible support is not installed. "
            "Install with: pip install tinbox[openai] or pip install tinbox[all]"
        )
    return AsyncOpenAI


completion_with_retry = build_retry_decorator(RateLimitError, logger)


def _read_reply(
    request: TranslationRequest, response: Any
) -> tuple[str, list[GlossaryEntry]]:
    """Check a completion and return its text and glossary updates."""
    if not hasattr(response, "choices") or not response.choices:
        raise TranslationError("No response from model")

    if (
        not hasattr(response.choices[0], "finish_reason")
        or response.choices[0].finish_reason != "stop"
    ):
        raise TranslationError(
            f"Invalid finish reason from model: {response.choices[0].finish_reason}, expected 'stop'"
        )

    if request.translation_protocol == TranslationProtocol.TRANSLATEGEMMA:
        text = response.choices[0].text
        if not text or not text.strip():
            raise TranslationError("No content returned from model")
        return text, []

    if (
        not hasattr(response.choices[0], "message")
        or not response.choices[0].message.content
    ):
        raise TranslationError("Invalid response format: missing message content")

    return parse_translation_response(
        response.choices[0].message.content,
        glossary_enabled=bool(request.glossary),
    )


class OpenAITranslator(ModelInterface):
    """OpenAI SDK-based translator for OpenAI, Ollama, and Gemini providers."""

    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 16384,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._clients: dict[ModelType, OpenAIClient] = {}

    def _get_client(self, model: ModelType) -> OpenAIClient:
        """Get or create an AsyncOpenAI client for the given provider."""
        client_class = _check_openai_installed()

        if model not in self._clients:
            if model == ModelType.OLLAMA:
                self._clients[model] = client_class(
                    base_url=_OLLAMA_BASE_URL,
                    api_key="ollama",
                )
            elif model == ModelType.GEMINI:
                api_key = os.environ.get("GOOGLE_API_KEY", "")
                self._clients[model] = client_class(
                    base_url=_GEMINI_BASE_URL,
                    api_key=api_key,
                )
            else:
                self._clients[model] = client_class()

        return self._clients[model]

    def _build_messages(
        self, request: TranslationRequest, clean_content: str | bytes
    ) -> list[dict]:
        """Build the full message list for a chat completion request."""
        messages: list[dict] = [
            {"role": "system", "content": build_system_instruction(request)},
        ]

        if request.content_type.startswith("text/"):
            if not isinstance(clean_content, str):
                raise TranslationError("Text content must be a string")
            messages.extend(build_user_messages(request, clean_content))
        else:
            if not isinstance(clean_content, bytes):
                raise TranslationError("Translation failed: Invalid image data")
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

            image_b64 = base64.b64encode(clean_content).decode()
            messages.append(
                {
                    "role": "user",
                    "content": build_image_user_content(request, image_b64),
                }
            )

        logger.debug("Messages: ", messages=messages)
        return messages

    @completion_with_retry
    async def _make_completion_request(
        self, request: TranslationRequest, clean_content: str | bytes
    ):
        """Make a chat completion request with retry on rate limits."""
        client = self._get_client(request.model)

        if request.translation_protocol == TranslationProtocol.TRANSLATEGEMMA:
            prompt = build_translategemma_prompt(request, clean_content)
            try:
                return await client.completions.create(
                    model=get_model_name(request),
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=min(self.max_tokens, 2048),
                    stop=["<end_of_turn>", "<eos>"],
                )
            except Exception as e:
                reraise_provider_error(e, RateLimitError, _ACCESS_ERRORS)

        kwargs: dict = {
            "model": get_model_name(request),
            "messages": self._build_messages(request, clean_content),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Unlike parse() with a model class, create() with this schema dict returns
        # truncated or invalid replies instead of raising, so _read_reply rejects
        # them after their billed usage has been priced
        response_model = (
            TranslationWithGlossaryResponse
            if request.glossary
            else TranslationWithoutGlossaryResponse
        )
        kwargs["response_format"] = (
            type_to_response_format_param(response_model)
            if type_to_response_format_param is not None
            else response_model
        )

        if request.reasoning_effort != "minimal":
            kwargs["reasoning_effort"] = request.reasoning_effort

        for k, v in request.model_params.items():
            if k != "model_name" and k not in kwargs:
                kwargs[k] = v

        model_family = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", kwargs["model"])
        if request.model == ModelType.OPENAI and model_family in _GPT6_MINIMAL_EFFORT:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            effort = (
                _GPT6_MINIMAL_EFFORT[model_family]
                if request.reasoning_effort == "minimal"
                else request.reasoning_effort
            )
            kwargs["reasoning_effort"] = effort
            if effort != "none":
                for parameter in ("temperature", "top_p", "top_logprobs", "logprobs"):
                    kwargs.pop(parameter, None)

        try:
            if type_to_response_format_param is None:
                return await client.chat.completions.parse(**kwargs)
            return await client.chat.completions.create(**kwargs)
        except Exception as e:
            reraise_provider_error(e, RateLimitError, _ACCESS_ERRORS)

    async def translate(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Translate content using the OpenAI SDK."""
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
                validate_image_content(clean_content)

            response = await self._make_completion_request(request, clean_content)

            # The provider bills a reply even when it is rejected below
            input_tokens, output_tokens = usage_tokens(
                response, "prompt_tokens", "completion_tokens"
            )
            cost = request_cost(request, request.model, input_tokens, output_tokens)
            try:
                text, glossary_updates = _read_reply(request, response)
            except TranslationError as e:
                e.cost, e.input_tokens, e.output_tokens = (
                    cost,
                    input_tokens,
                    output_tokens,
                )
                raise

            tokens = input_tokens + output_tokens
            time_taken = (datetime.now() - start_time).total_seconds()
            final_text = content_prefix + text + content_suffix
            logger.debug("Final text: ", final_text=final_text)

            return TranslationResponse(
                text=final_text,
                tokens_used=tokens,
                cost=cost,
                time_taken=time_taken,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                glossary_updates=glossary_updates,
            )

        except TranslationError:
            raise
        except Exception as e:
            logger.error(f"Translation failed: {e!s}")
            raise TranslationError(f"Translation failed: {e!s}") from e
