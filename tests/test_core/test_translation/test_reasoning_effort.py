"""Tests for reasoning effort functionality across providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinbox.core.translation.anthropic import AnthropicTranslator
from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.translation.openai import OpenAITranslator
from tinbox.core.types import ModelType


def _mock_openai_response(
    content: str = '{"translation": "Hola, mundo!"}',
) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = "stop"
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.total_tokens = 100
    response.usage.prompt_tokens = 60
    response.usage.completion_tokens = 40
    return response


def _mock_anthropic_response(
    text: str = '{"translation": "Hola, mundo!"}',
) -> MagicMock:
    response = MagicMock()
    text_block = MagicMock(type="text", text=text)
    response.content = [text_block]
    response.stop_reason = "end_turn"
    response.usage = MagicMock(input_tokens=60, output_tokens=40)
    return response


@pytest.mark.asyncio
async def test_openai_reasoning_effort_passed():
    """Test that reasoning_effort is passed to OpenAI SDK."""
    translator = OpenAITranslator()

    with patch("tinbox.core.translation.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response()
        )
        mock_cls.return_value = mock_client

        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello!",
            context=None,
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
            reasoning_effort="high",
        )
        await translator.translate(request)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_openai_reasoning_effort_minimal_omitted():
    """Test that minimal reasoning_effort is not sent to OpenAI."""
    translator = OpenAITranslator()

    with patch("tinbox.core.translation.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response()
        )
        mock_cls.return_value = mock_client

        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello!",
            context=None,
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
            reasoning_effort="minimal",
        )
        await translator.translate(request)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
@pytest.mark.asyncio
async def test_openai_reasoning_effort_all_non_minimal(effort):
    """Test that non-minimal effort values are passed through."""
    translator = OpenAITranslator()

    with patch("tinbox.core.translation.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response()
        )
        mock_cls.return_value = mock_client

        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello!",
            context=None,
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
            reasoning_effort=effort,
        )
        await translator.translate(request)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == effort


@pytest.mark.asyncio
async def test_anthropic_reasoning_effort_maps_to_thinking():
    """Test that reasoning_effort maps to Anthropic thinking config."""
    translator = AnthropicTranslator()

    with patch("tinbox.core.translation.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
        mock_cls.return_value = mock_client

        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello!",
            context=None,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
            reasoning_effort="high",
        )
        await translator.translate(request)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8192}


@pytest.mark.asyncio
async def test_anthropic_reasoning_effort_minimal_no_thinking():
    """Test that minimal effort does not enable thinking for Anthropic."""
    translator = AnthropicTranslator()

    with patch("tinbox.core.translation.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
        mock_cls.return_value = mock_client

        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello!",
            context=None,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
            reasoning_effort="minimal",
        )
        await translator.translate(request)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs
