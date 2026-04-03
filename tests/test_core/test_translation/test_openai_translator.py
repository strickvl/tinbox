"""Tests for the OpenAI SDK-based translator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from tinbox.core.translation.interface import TranslationError, TranslationRequest
from tinbox.core.translation.openai import OpenAITranslator
from tinbox.core.types import Glossary, ModelType


def _mock_openai_response(
    content: str = '{"translation": "Translated text", "glossary_extension": []}',
    finish_reason: str = "stop",
    prompt_tokens: int = 6,
    completion_tokens: int = 4,
) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = finish_reason
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


@pytest.fixture
def translator() -> OpenAITranslator:
    return OpenAITranslator()


@pytest.fixture
def mock_openai_client():
    """Patch AsyncOpenAI so _get_client returns a mock."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response()
    )
    with patch("tinbox.core.translation.openai.AsyncOpenAI", return_value=mock_client):
        yield mock_client


@pytest.mark.asyncio
async def test_text_translation(translator: OpenAITranslator, mock_openai_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"
    assert response.tokens_used == 10
    assert response.time_taken > 0


@pytest.mark.asyncio
async def test_image_translation(
    translator: OpenAITranslator, mock_openai_client, tmp_path: Path
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
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"
    assert response.tokens_used == 10


@pytest.mark.asyncio
async def test_empty_content(translator: OpenAITranslator, mock_openai_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == ""
    assert response.tokens_used == 0
    assert response.cost == 0.0


@pytest.mark.asyncio
async def test_whitespace_only_content(
    translator: OpenAITranslator, mock_openai_client
):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="   \n   ",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == "   \n   "
    assert response.tokens_used == 0


@pytest.mark.asyncio
async def test_whitespace_preservation(
    translator: OpenAITranslator, mock_openai_client
):
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
async def test_whitespace_preservation_complex(
    translator: OpenAITranslator, mock_openai_client
):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="\n\n  Hello, world!\n  ",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == "\n\n  Translated text\n  "


@pytest.mark.asyncio
async def test_error_handling(translator: OpenAITranslator, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("API error")
    )
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    with pytest.raises(TranslationError, match="Translation failed: API error"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_invalid_language_codes(translator: OpenAITranslator, mock_openai_client):
    request = TranslationRequest(
        source_lang="invalid",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    with pytest.raises(
        TranslationError, match=r"Translation failed.*Unsupported language code"
    ):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_hyphenated_language_codes(
    translator: OpenAITranslator, mock_openai_client
):
    request = TranslationRequest(
        source_lang="zh-tw",
        target_lang="en",
        content="你好世界",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_auto_language_detection(
    translator: OpenAITranslator, mock_openai_client
):
    request = TranslationRequest(
        source_lang="auto",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_malformed_image(translator: OpenAITranslator, mock_openai_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=b"not a valid image",
        context=None,
        content_type="image/png",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    with pytest.raises(
        TranslationError, match="Translation failed: Invalid image data"
    ):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_empty_choices(translator: OpenAITranslator, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response()
    )
    resp = mock_openai_client.chat.completions.create.return_value
    resp.choices = []

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    with pytest.raises(TranslationError, match="No response from model"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_invalid_json_response(translator: OpenAITranslator, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(content="Plain text, not JSON")
    )
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    with pytest.raises(TranslationError, match="Invalid JSON response format"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_missing_translation_field(
    translator: OpenAITranslator, mock_openai_client
):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(content='{"text": "wrong field"}')
    )
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    with pytest.raises(TranslationError, match="Missing translation field"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_no_model_name(translator: OpenAITranslator, mock_openai_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello!",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={},
    )
    with pytest.raises(TranslationError, match="No model name provided"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_glossary_extension(translator: OpenAITranslator, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(
            content='{"translation": "Bonjour", "glossary_extension": [{"term": "AI", "translation": "IA"}]}'
        )
    )
    request = TranslationRequest(
        source_lang="en",
        target_lang="fr",
        content="Hello",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
        glossary=Glossary(entries={"CPU": "Processeur"}),
    )
    response = await translator.translate(request)
    assert response.text == "Bonjour"
    assert len(response.glossary_updates) == 1
    assert response.glossary_updates[0].term == "AI"
    assert response.glossary_updates[0].translation == "IA"


@pytest.mark.asyncio
async def test_context_handling(translator: OpenAITranslator, mock_openai_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Current text",
        context="[PREVIOUS_CHUNK]\nPrevious\n[/PREVIOUS_CHUNK]",
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_ollama_uses_correct_base_url(translator: OpenAITranslator):
    """Test that Ollama requests use the OpenAI-compatible base URL."""
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
            model=ModelType.OLLAMA,
            model_params={"model_name": "llama3.1:8b"},
        )
        response = await translator.translate(request)
        assert response.text == "Translated text"

        # Verify AsyncOpenAI was called with Ollama base URL
        mock_cls.assert_called_once_with(
            base_url="http://localhost:11434/v1", api_key="ollama"
        )


@pytest.mark.asyncio
async def test_ollama_zero_cost(translator: OpenAITranslator):
    """Test that Ollama translations have zero cost."""
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
            model=ModelType.OLLAMA,
            model_params={"model_name": "llama3.1:8b"},
        )
        response = await translator.translate(request)
        assert response.cost == 0.0


@pytest.mark.asyncio
async def test_gemini_uses_correct_base_url(translator: OpenAITranslator):
    """Test that Gemini requests use the Google OpenAI-compatible base URL."""
    with patch("tinbox.core.translation.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response()
        )
        mock_cls.return_value = mock_client

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            request = TranslationRequest(
                source_lang="en",
                target_lang="es",
                content="Hello!",
                context=None,
                content_type="text/plain",
                model=ModelType.GEMINI,
                model_params={"model_name": "gemini-2.5-pro"},
            )
            response = await translator.translate(request)
            assert response.text == "Translated text"

        # Verify AsyncOpenAI was called with Gemini base URL
        mock_cls.assert_called_once_with(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="test-key",
        )


@pytest.mark.asyncio
async def test_reasoning_effort_passed(translator: OpenAITranslator):
    """Test that reasoning_effort is passed to the OpenAI call."""
    with patch("tinbox.core.translation.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(content='{"translation": "Hola!"}')
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

        # Verify reasoning_effort was passed
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_reasoning_effort_minimal_not_passed(translator: OpenAITranslator):
    """Test that minimal reasoning_effort is not passed (default behavior)."""
    with patch("tinbox.core.translation.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(content='{"translation": "Hola!"}')
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


@pytest.mark.asyncio
async def test_special_characters(translator: OpenAITranslator, mock_openai_client):
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello! 👋 Special: ¡¢£¤¥",
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"


@pytest.mark.asyncio
async def test_long_content(translator: OpenAITranslator, mock_openai_client):
    long_text = "Hello, world! " * 1000
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=long_text,
        context=None,
        content_type="text/plain",
        model=ModelType.OPENAI,
        model_params={"model_name": "gpt-4o"},
    )
    response = await translator.translate(request)
    # Should preserve trailing space from original content
    assert response.text == "Translated text "
