"""Tests for the LiteLLM translator."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from tinbox.core.translation.interface import (
    TranslationError,
    TranslationRequest,
)
from tinbox.core.translation.litellm import LiteLLMTranslator
from tinbox.core.types import ModelType


@pytest.fixture
def translator() -> LiteLLMTranslator:
    """Create a LiteLLM translator instance."""
    return LiteLLMTranslator()


@pytest.fixture
def mock_completion():
    """Create a mock completion response."""
    with patch("tinbox.core.translation.litellm.completion") as mock:
        # Mock response content as JSON string (matching real LiteLLM response format)
        response_content = (
            '{"translation": "Translated text", "glossary_extension": []}'
        )

        mock.return_value = type(
            "CompletionResponse",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "finish_reason": "stop",
                            "message": type(
                                "Message",
                                (),
                                {"content": response_content},
                            )(),
                        },
                    )
                ],
                "usage": type(
                    "Usage",
                    (),
                    {
                        "total_tokens": 10,
                        "cost": 0.001,
                        "completion_time": 0.5,
                    },
                )(),
                "_hidden_params": {"response_cost": 0.001},
            },
        )
        yield mock


@pytest.mark.asyncio
async def test_text_translation(translator: LiteLLMTranslator, mock_completion):
    """Test translating text content."""
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
    assert response.cost == 0.001
    assert response.time_taken > 0  # Real time calculation


@pytest.mark.asyncio
async def test_image_translation(
    translator: LiteLLMTranslator, mock_completion, tmp_path: Path
):
    """Test translating image content."""
    # Create a valid PNG image
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
    assert response.tokens_used == 10
    assert response.cost == 0.001
    assert response.time_taken > 0  # Real time calculation


@pytest.mark.asyncio
async def test_translation_error_handling(translator: LiteLLMTranslator, monkeypatch):
    """Test error handling during translation."""

    def mock_error(*args: Any, **kwargs: dict[str, Any]) -> None:
        raise RuntimeError("API error")

    # Patch both the module and the imported function
    import litellm

    monkeypatch.setattr(litellm, "completion", mock_error)
    monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_error)

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    with pytest.raises(TranslationError, match="Translation failed: API error"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_empty_content(translator: LiteLLMTranslator, mock_completion):
    """Test handling of empty or whitespace-only content."""
    # Empty string
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
    assert response.time_taken > 0

    # Whitespace only
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="   \n   ",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    response = await translator.translate(request)
    assert response.text == "   \n   "
    assert response.tokens_used == 0
    assert response.cost == 0.0
    assert response.time_taken > 0


@pytest.mark.asyncio
async def test_long_content(translator: LiteLLMTranslator, mock_completion):
    """Test handling of very long content."""
    long_text = "Hello, world! " * 1000  # Create long repetitive text
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=long_text,
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    response = await translator.translate(request)
    # Should preserve the trailing space from the original long_text
    assert response.text == "Translated text "


@pytest.mark.asyncio
async def test_special_characters(translator: LiteLLMTranslator, mock_completion):
    """Test handling of special characters and emojis."""
    special_text = "Hello! 👋 This is a test with special chars: ¡¢£¤¥¦§¨©ª"
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=special_text,
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_invalid_model_params(translator: LiteLLMTranslator, mock_completion):
    """Test handling of invalid model parameters."""
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet", "invalid_param": "value"},
    )

    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_malformed_image(translator: LiteLLMTranslator, mock_completion):
    """Test handling of malformed image data."""
    invalid_image_data = b"This is not a valid image"
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=invalid_image_data,
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
async def test_timeout_handling(translator: LiteLLMTranslator, monkeypatch):
    """Test handling of network timeouts."""

    def mock_timeout(*args: Any, **kwargs: dict[str, Any]) -> Any:
        raise asyncio.TimeoutError("Request timed out")

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_timeout)
    monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_timeout)

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    with pytest.raises(TranslationError, match="Translation failed: Request timed out"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_rate_limit_handling(translator: LiteLLMTranslator, monkeypatch):
    """Test handling of rate limiting."""

    def mock_rate_limit(*args: Any, **kwargs: dict[str, Any]) -> Any:
        raise RuntimeError("Rate limit exceeded")

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_rate_limit)
    monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_rate_limit)

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    with pytest.raises(
        TranslationError, match="Translation failed: Rate limit exceeded"
    ):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_invalid_language_codes(translator: LiteLLMTranslator, mock_completion):
    """Test handling of invalid language codes."""
    # Invalid source language
    request = TranslationRequest(
        source_lang="invalid",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    with pytest.raises(
        TranslationError, match=r"Translation failed.*Unsupported language code"
    ):
        await translator.translate(request)

    # Invalid target language
    request = TranslationRequest(
        source_lang="en",
        target_lang="invalid",
        content="Hello, world!",
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
async def test_hyphenated_language_codes(
    translator: LiteLLMTranslator, mock_completion
):
    """Test that hyphenated language codes like zh-tw are supported."""
    # Traditional Chinese (zh-tw) should be valid
    request = TranslationRequest(
        source_lang="zh-tw",
        target_lang="en",
        content="你好世界",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    # Should not raise - hyphenated codes are valid
    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_auto_language_detection(translator: LiteLLMTranslator, mock_completion):
    """Test handling of 'auto' as source language."""
    request = TranslationRequest(
        source_lang="auto",
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
    assert response.cost == 0.001
    assert response.time_taken > 0  # Real time calculation


@pytest.mark.asyncio
async def test_mixed_content_handling(translator: LiteLLMTranslator, mock_completion):
    """Test handling of mixed content types."""
    # Text with embedded image-like content
    mixed_content = "Hello, world! data:image/png;base64,invalid_base64"
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=mixed_content,
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_response_validation(translator: LiteLLMTranslator, monkeypatch):
    """Test validation of malformed responses."""

    def mock_malformed_response(*args: Any, **kwargs: dict[str, Any]) -> Any:
        return type(
            "MalformedResponse",
            (),
            {
                "choices": [],  # Empty choices
                "usage": None,  # Missing usage info
            },
        )

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_malformed_response)
    monkeypatch.setattr(
        "tinbox.core.translation.litellm.completion", mock_malformed_response
    )

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    with pytest.raises(TranslationError, match="No response from model"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_whitespace_preservation_simple(
    translator: LiteLLMTranslator, mock_completion
):
    """Test simple whitespace preservation."""
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
    # Should preserve leading and trailing whitespace
    assert response.text == "   Translated text   "
    assert response.tokens_used == 10
    assert response.cost == 0.001
    assert response.time_taken > 0


@pytest.mark.asyncio
async def test_whitespace_preservation_complex(
    translator: LiteLLMTranslator, mock_completion
):
    """Test complex whitespace preservation with newlines."""
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="\n\n  Hello, world!\n  ",
        context=None,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    response = await translator.translate(request)
    # Should preserve complex whitespace patterns
    assert response.text == "\n\n  Translated text\n  "
    assert response.tokens_used == 10
    assert response.cost == 0.001
    assert response.time_taken > 0


@pytest.mark.asyncio
async def test_context_handling(translator: LiteLLMTranslator, mock_completion):
    """Test context handling in prompt construction."""
    context_info = "[PREVIOUS_CHUNK]\nPrevious text\n[/PREVIOUS_CHUNK]\n\n[PREVIOUS_CHUNK_TRANSLATION]\nTexto anterior\n[/PREVIOUS_CHUNK_TRANSLATION]\n\nUse this context to maintain consistency in terminology and style."

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Current text",
        context=context_info,
        content_type="text/plain",
        model=ModelType.ANTHROPIC,
        model_params={"model_name": "claude-3-sonnet"},
    )

    response = await translator.translate(request)
    assert response.text == "Translated text"
    assert response.tokens_used == 10
    assert response.cost == 0.001
    assert response.time_taken > 0


@pytest.mark.asyncio
async def test_context_without_context(translator: LiteLLMTranslator, mock_completion):
    """Test translation without context (context=None)."""
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
    assert response.cost == 0.001
    assert response.time_taken > 0
