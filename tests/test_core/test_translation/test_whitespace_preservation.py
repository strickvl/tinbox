"""Tests for whitespace preservation functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinbox.core.translation.anthropic import AnthropicTranslator
from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.translation.openai import OpenAITranslator
from tinbox.core.types import ModelType
from tinbox.utils.chunks import extract_whitespace_formatting


class TestExtractWhitespaceFormatting:
    """Test whitespace extraction utility function."""

    def test_extract_whitespace_formatting_simple(self):
        content = "  Hello world  "
        prefix, core, suffix = extract_whitespace_formatting(content)
        assert prefix == "  "
        assert core == "Hello world"
        assert suffix == "  "

    def test_extract_whitespace_formatting_complex(self):
        content = "\n\n  Hello world\n  "
        prefix, core, suffix = extract_whitespace_formatting(content)
        assert prefix == "\n\n  "
        assert core == "Hello world"
        assert suffix == "\n  "

    def test_extract_whitespace_formatting_edge_cases(self):
        # Empty string
        prefix, core, suffix = extract_whitespace_formatting("")
        assert prefix == ""
        assert core == ""
        assert suffix == ""

        # Only whitespace
        prefix, core, suffix = extract_whitespace_formatting("   ")
        assert prefix == "   "
        assert core == ""
        assert suffix == ""

        # No whitespace
        prefix, core, suffix = extract_whitespace_formatting("Hello")
        assert prefix == ""
        assert core == "Hello"
        assert suffix == ""

        # Only prefix whitespace
        prefix, core, suffix = extract_whitespace_formatting("  Hello")
        assert prefix == "  "
        assert core == "Hello"
        assert suffix == ""

        # Only suffix whitespace
        prefix, core, suffix = extract_whitespace_formatting("Hello  ")
        assert prefix == ""
        assert core == "Hello"
        assert suffix == "  "

    def test_extract_whitespace_formatting_non_string(self):
        prefix, core, suffix = extract_whitespace_formatting(123)
        assert prefix == ""
        assert core == 123
        assert suffix == ""


def _mock_openai_response() -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = "stop"
    response.choices[0].message = MagicMock()
    response.choices[
        0
    ].message.content = '{"translation": "Translated text", "glossary_extension": []}'
    response.usage = MagicMock()
    response.usage.total_tokens = 10
    response.usage.prompt_tokens = 6
    response.usage.completion_tokens = 4
    return response


def _mock_anthropic_response() -> MagicMock:
    response = MagicMock()
    text_block = MagicMock(type="text")
    text_block.text = '{"translation": "Translated text", "glossary_extension": []}'
    response.content = [text_block]
    response.stop_reason = "end_turn"
    response.usage = MagicMock(input_tokens=6, output_tokens=4)
    return response


class TestWhitespacePreservationOpenAI:
    """Test end-to-end whitespace preservation via OpenAI translator."""

    @pytest.fixture
    def mock_openai(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response()
        )
        with patch(
            "tinbox.core.translation.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            yield mock_client

    @pytest.mark.asyncio
    async def test_simple_whitespace(self, mock_openai):
        translator = OpenAITranslator()
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="   Hello, world!   ",
            context=None,
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
        )
        response = await translator.translate(request)
        assert response.text == "   Translated text   "

    @pytest.mark.asyncio
    async def test_complex_whitespace(self, mock_openai):
        translator = OpenAITranslator()
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="\n\n  Complex whitespace  \n",
            context=None,
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
        )
        response = await translator.translate(request)
        assert response.text == "\n\n  Translated text  \n"

    @pytest.mark.asyncio
    async def test_whitespace_with_context(self, mock_openai):
        translator = OpenAITranslator()
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="\n  Current text  \n",
            context="[PREVIOUS_CHUNK]\nPrevious\n[/PREVIOUS_CHUNK]",
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
        )
        response = await translator.translate(request)
        assert response.text == "\n  Translated text  \n"


class TestWhitespacePreservationAnthropic:
    """Test end-to-end whitespace preservation via Anthropic translator."""

    @pytest.fixture
    def mock_anthropic(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
        with patch(
            "tinbox.core.translation.anthropic.AsyncAnthropic",
            return_value=mock_client,
        ):
            yield mock_client

    @pytest.mark.asyncio
    async def test_simple_whitespace(self, mock_anthropic):
        translator = AnthropicTranslator()
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="   Hello, world!   ",
            context=None,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )
        response = await translator.translate(request)
        assert response.text == "   Translated text   "

    @pytest.mark.asyncio
    async def test_complex_whitespace(self, mock_anthropic):
        translator = AnthropicTranslator()
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="\n\n  Complex whitespace  \n",
            context=None,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )
        response = await translator.translate(request)
        assert response.text == "\n\n  Translated text  \n"
