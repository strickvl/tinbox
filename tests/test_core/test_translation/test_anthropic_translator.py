"""Tests for the Anthropic SDK-based translator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from tinbox.core.translation.anthropic import (
    AnthropicTranslator,
    _extract_text_from_response,
    _thinking_config,
)
from tinbox.core.translation.interface import TranslationError, TranslationRequest
from tinbox.core.types import Glossary, ModelType


def _mock_anthropic_response(
    text: str = '{"translation": "Translated text", "glossary_extension": []}',
    stop_reason: str = "end_turn",
    input_tokens: int = 6,
    output_tokens: int = 4,
    include_thinking: bool = False,
) -> MagicMock:
    """Build a mock Anthropic Message response."""
    response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    content = []
    if include_thinking:
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me think about this..."
        content.append(thinking_block)
    content.append(text_block)
    response.content = content
    response.stop_reason = stop_reason
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response.usage = usage
    return response


@pytest.fixture
def translator() -> AnthropicTranslator:
    return AnthropicTranslator()


@pytest.fixture
def mock_anthropic_client():
    """Patch AsyncAnthropic so _get_client returns a mock."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response())
    with patch(
        "tinbox.core.translation.anthropic.AsyncAnthropic",
        return_value=mock_client,
    ):
        yield mock_client


# --- Unit tests for helpers ---


class TestThinkingConfig:
    def test_minimal_returns_none(self):
        assert _thinking_config("minimal") is None

    def test_low_returns_enabled(self):
        config = _thinking_config("low")
        assert config == {"type": "enabled", "budget_tokens": 1024}

    def test_medium_returns_enabled(self):
        config = _thinking_config("medium")
        assert config == {"type": "enabled", "budget_tokens": 4096}

    def test_high_returns_enabled(self):
        config = _thinking_config("high")
        assert config == {"type": "enabled", "budget_tokens": 8192}


class TestExtractTextFromResponse:
    def test_single_text_block(self):
        block = MagicMock()
        block.type = "text"
        block.text = "hello"
        assert _extract_text_from_response([block]) == "hello"

    def test_multiple_text_blocks(self):
        b1 = MagicMock(type="text", text="hello ")
        b2 = MagicMock(type="text", text="world")
        assert _extract_text_from_response([b1, b2]) == "hello world"

    def test_filters_thinking_blocks(self):
        thinking = MagicMock(type="thinking", thinking="...")
        text = MagicMock(type="text", text="result")
        assert _extract_text_from_response([thinking, text]) == "result"

    def test_no_text_blocks_raises(self):
        thinking = MagicMock(type="thinking", thinking="...")
        with pytest.raises(TranslationError, match="no text content"):
            _extract_text_from_response([thinking])

    def test_empty_content_raises(self):
        with pytest.raises(TranslationError, match="no text content"):
            _extract_text_from_response([])


# --- Integration tests ---


@pytest.mark.asyncio
async def test_text_translation(translator: AnthropicTranslator, mock_anthropic_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"
    assert response.tokens_used == 10
    assert response.time_taken > 0


@pytest.mark.asyncio
async def test_image_translation(
    translator: AnthropicTranslator, mock_anthropic_client, tmp_path: Path
):
    image = Image.new("RGB", (100, 100), color="white")
    image_path = tmp_path / "test.png"
    image.save(image_path)

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=image_path.read_bytes(),
        context=None,
        content_type="image/png",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_empty_content(translator: AnthropicTranslator, mock_anthropic_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    response = await translator.translate(request)
    assert response.text == ""
    assert response.tokens_used == 0
    assert response.cost == 0.0


@pytest.mark.asyncio
async def test_whitespace_preservation(
    translator: AnthropicTranslator, mock_anthropic_client
):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="   Hello!   ",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    response = await translator.translate(request)
    assert response.text == "   Translated text   "


@pytest.mark.asyncio
async def test_whitespace_preservation_complex(
    translator: AnthropicTranslator, mock_anthropic_client
):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="\n\n  Hello!\n  ",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    response = await translator.translate(request)
    assert response.text == "\n\n  Translated text\n  "


@pytest.mark.asyncio
async def test_glossary_extension(
    translator: AnthropicTranslator, mock_anthropic_client
):
    mock_anthropic_client.messages.create = AsyncMock(
        return_value=_mock_anthropic_response(
            text='{"translation": "Bonjour", "glossary_extension": [{"term": "AI", "translation": "IA"}]}'
        )
    )
    request = TranslationRequest(
        source_lang="en",
        target_lang="fr",
        content="Hello",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
        glossary=Glossary(entries={"CPU": "Processeur"}),
    )
    response = await translator.translate(request)
    assert response.text == "Bonjour"
    assert len(response.glossary_updates) == 1
    assert response.glossary_updates[0].term == "AI"


@pytest.mark.asyncio
async def test_invalid_json_response(
    translator: AnthropicTranslator, mock_anthropic_client
):
    mock_anthropic_client.messages.create = AsyncMock(
        return_value=_mock_anthropic_response(text="Plain text, not JSON")
    )
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    with pytest.raises(TranslationError, match="Invalid JSON response format"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_thinking_blocks_filtered(
    translator: AnthropicTranslator, mock_anthropic_client
):
    """Test that thinking blocks are filtered and only text blocks are used."""
    mock_anthropic_client.messages.create = AsyncMock(
        return_value=_mock_anthropic_response(
            text='{"translation": "Hola!"}',
            include_thinking=True,
        )
    )
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    response = await translator.translate(request)
    assert response.text == "Hola!"


@pytest.mark.asyncio
async def test_reasoning_effort_thinking_config(translator: AnthropicTranslator):
    """Test that reasoning effort maps to thinking config correctly."""
    with patch("tinbox.core.translation.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(text='{"translation": "Hola!"}')
        )
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
        # Temperature should not be set when thinking is enabled
        assert "temperature" not in call_kwargs


@pytest.mark.asyncio
async def test_reasoning_effort_minimal_no_thinking(translator: AnthropicTranslator):
    """Test that minimal reasoning effort does not enable thinking."""
    with patch("tinbox.core.translation.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(text='{"translation": "Hola!"}')
        )
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
        assert call_kwargs["temperature"] == 0.3


@pytest.mark.asyncio
async def test_error_handling(translator: AnthropicTranslator, mock_anthropic_client):
    mock_anthropic_client.messages.create = AsyncMock(
        side_effect=RuntimeError("API error")
    )
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    with pytest.raises(TranslationError, match="Translation failed: API error"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_invalid_language_codes(
    translator: AnthropicTranslator, mock_anthropic_client
):
    request = TranslationRequest(
        source_lang="invalid",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    with pytest.raises(
        TranslationError, match=r"Translation failed.*Unsupported language code"
    ):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_malformed_image(translator: AnthropicTranslator, mock_anthropic_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=b"not a valid image",
        context=None,
        content_type="image/png",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    with pytest.raises(
        TranslationError, match="Translation failed: Invalid image data"
    ):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_no_model_name(translator: AnthropicTranslator, mock_anthropic_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={},
    )
    with pytest.raises(TranslationError, match="No model name provided"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_context_handling(translator: AnthropicTranslator, mock_anthropic_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Current text",
        context="[PREVIOUS_CHUNK]\nPrevious\n[/PREVIOUS_CHUNK]",
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"
