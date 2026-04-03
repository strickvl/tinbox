"""OpenAI SDK-based translation implementation.

Handles OpenAI, Ollama (OpenAI-compatible), and Gemini (OpenAI-compatible)
providers through the openai Python SDK with different base URLs.
"""

from __future__ import annotations

import base64
import os
from datetime import datetime

from tinbox.core.cost import calculate_usage_cost
from tinbox.core.translation._shared import (
    build_glossary_instruction,
    build_image_user_content,
    build_retry_decorator,
    build_system_instruction,
    build_user_messages,
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
    TranslationWithGlossaryResponse,
    TranslationWithoutGlossaryResponse,
)
from tinbox.core.types import ModelType
from tinbox.utils.logging import get_logger

try:
    from openai import AsyncOpenAI, RateLimitError
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment, misc]
    RateLimitError = None  # type: ignore[assignment, misc]

logger = get_logger(__name__)

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _check_openai_installed() -> None:
    """Raise a clear error if the openai package is not installed."""
    if AsyncOpenAI is None:
        raise TranslationError(
            "OpenAI-compatible support is not installed. "
            "Install with: pip install tinbox[openai] or pip install tinbox[all]"
        )


completion_with_retry = build_retry_decorator(RateLimitError, logger)


class OpenAITranslator(ModelInterface):
    """OpenAI SDK-based translator for OpenAI, Ollama, and Gemini providers."""

    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 100000,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._clients: dict[ModelType, AsyncOpenAI] = {}  # type: ignore[type-arg]

    def _get_client(self, model: ModelType) -> AsyncOpenAI:  # type: ignore[return]
        """Get or create an AsyncOpenAI client for the given provider."""
        _check_openai_installed()

        if model not in self._clients:
            if model == ModelType.OLLAMA:
                self._clients[model] = AsyncOpenAI(
                    base_url=_OLLAMA_BASE_URL,
                    api_key="ollama",
                )
            elif model == ModelType.GEMINI:
                api_key = os.environ.get("GOOGLE_API_KEY", "")
                self._clients[model] = AsyncOpenAI(
                    base_url=_GEMINI_BASE_URL,
                    api_key=api_key,
                )
            else:
                self._clients[model] = AsyncOpenAI()

        return self._clients[model]

    def _build_messages(
        self, request: TranslationRequest, clean_content: str | bytes
    ) -> list[dict]:
        """Build the full message list for a chat completion request."""
        messages: list[dict] = [
            {"role": "system", "content": build_system_instruction(request)},
        ]

        if request.content_type.startswith("text/"):
            messages.extend(
                build_user_messages(request, clean_content)  # type: ignore[arg-type]
            )
        else:
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

            image_b64 = base64.b64encode(clean_content).decode()  # type: ignore[arg-type]
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
    ):  # type: ignore[no-untyped-def]
        """Make a chat completion request with retry on rate limits."""
        client = self._get_client(request.model)

        kwargs: dict = {
            "model": get_model_name(request),
            "messages": self._build_messages(request, clean_content),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if request.glossary:
            kwargs["response_format"] = TranslationWithGlossaryResponse
        else:
            kwargs["response_format"] = TranslationWithoutGlossaryResponse

        if request.reasoning_effort and request.reasoning_effort != "minimal":
            kwargs["reasoning_effort"] = request.reasoning_effort

        for k, v in request.model_params.items():
            if k != "model_name" and k not in kwargs:
                kwargs[k] = v

        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as e:
            if RateLimitError and isinstance(e, RateLimitError):
                raise
            raise TranslationError(f"Translation failed: {e!s}") from e

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
                validate_image_content(clean_content)  # type: ignore[arg-type]

            response = await self._make_completion_request(request, clean_content)

            if not hasattr(response, "choices") or not response.choices:
                raise TranslationError("No response from model")

            if (
                not hasattr(response.choices[0], "finish_reason")
                or response.choices[0].finish_reason != "stop"
            ):
                raise TranslationError(
                    f"Invalid finish reason from model: {response.choices[0].finish_reason}, expected 'stop'"
                )

            if (
                not hasattr(response.choices[0], "message")
                or not response.choices[0].message.content
            ):
                raise TranslationError(
                    "Invalid response format: missing message content"
                )

            text, glossary_updates = parse_translation_response(
                response.choices[0].message.content,
                glossary_enabled=bool(request.glossary),
            )

            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0)
                output_tokens = getattr(response.usage, "completion_tokens", 0)

            tokens = input_tokens + output_tokens
            cost = calculate_usage_cost(request.model, input_tokens, output_tokens)
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
            client = self._get_client(ModelType.OPENAI)
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return hasattr(response, "choices") and len(response.choices) > 0
        except Exception as e:
            logger.error(f"Model validation failed: {e!s}")
            return False
