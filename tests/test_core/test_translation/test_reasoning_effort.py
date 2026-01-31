"""Tests for reasoning effort functionality."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.translation.litellm import LiteLLMTranslator
from tinbox.core.types import ModelType


@pytest.fixture
def translator():
    """Create a LiteLLM translator for testing."""
    return LiteLLMTranslator()


@pytest.mark.asyncio
async def test_reasoning_effort_passed_to_completion(translator: LiteLLMTranslator):
    """Test that reasoning_effort is correctly passed to the completion call."""
    
    def mock_completion(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        # Verify reasoning_effort is passed correctly
        assert "reasoning_effort" in kwargs
        assert kwargs["reasoning_effort"] == "high"
        
        # Return a mock response
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].finish_reason = "stop"
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = '{"translation": "Hola, mundo!"}'
        response.usage = MagicMock()
        response.usage.total_tokens = 100
        response._hidden_params = {"response_cost": 0.01}
        return response

    with patch("tinbox.core.translation.litellm.completion", side_effect=mock_completion):
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello, world!",
            context=None,
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
            reasoning_effort="high",
        )

        response = await translator.translate(request)
        assert response.text == "Hola, mundo!"


@pytest.mark.asyncio
async def test_reasoning_effort_default_minimal(translator: LiteLLMTranslator):
    """Test that reasoning_effort defaults to minimal."""
    
    def mock_completion(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        # Verify reasoning_effort defaults to minimal
        assert "reasoning_effort" in kwargs
        assert kwargs["reasoning_effort"] == "minimal"
        
        # Return a mock response
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].finish_reason = "stop"
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = '{"translation": "Hola, mundo!"}'
        response.usage = MagicMock()
        response.usage.total_tokens = 100
        response._hidden_params = {"response_cost": 0.01}
        return response

    with patch("tinbox.core.translation.litellm.completion", side_effect=mock_completion):
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello, world!",
            context=None,
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
            # reasoning_effort not specified, should default to minimal
        )

        response = await translator.translate(request)
        assert response.text == "Hola, mundo!"


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
@pytest.mark.asyncio
async def test_reasoning_effort_all_values(translator: LiteLLMTranslator, effort):
    """Test that all reasoning effort values are passed correctly."""
    
    def mock_completion(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        # Verify reasoning_effort is passed correctly
        assert "reasoning_effort" in kwargs
        assert kwargs["reasoning_effort"] == effort
        
        # Return a mock response
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].finish_reason = "stop"
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = '{"translation": "Hola, mundo!"}'
        response.usage = MagicMock()
        response.usage.total_tokens = 100
        response._hidden_params = {"response_cost": 0.01}
        return response

    with patch("tinbox.core.translation.litellm.completion", side_effect=mock_completion):
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello, world!",
            context=None,
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"},
            reasoning_effort=effort,
        )

        response = await translator.translate(request)
        assert response.text == "Hola, mundo!"
