"""Tests for whitespace preservation functionality."""

import pytest

from tinbox.utils.chunks import extract_whitespace_formatting
from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.translation.litellm import LiteLLMTranslator
from tinbox.core.types import ModelType


class TestExtractWhitespaceFormatting:
    """Test whitespace extraction utility function."""

    def test_extract_whitespace_formatting_simple(self):
        """Test extracting whitespace from simple text."""
        content = "  Hello world  "
        prefix, core, suffix = extract_whitespace_formatting(content)
        
        assert prefix == "  "
        assert core == "Hello world"
        assert suffix == "  "

    def test_extract_whitespace_formatting_complex(self):
        """Test extracting whitespace from complex text with newlines."""
        content = "\n\n  Hello world\n  "
        prefix, core, suffix = extract_whitespace_formatting(content)
        
        assert prefix == "\n\n  "
        assert core == "Hello world"
        assert suffix == "\n  "

    def test_extract_whitespace_formatting_edge_cases(self):
        """Test edge cases for whitespace extraction."""
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
        """Test handling of non-string input."""
        prefix, core, suffix = extract_whitespace_formatting(123)
        assert prefix == ""
        assert core == 123
        assert suffix == ""


class TestWhitespacePreservationEndToEnd:
    """Test end-to-end whitespace preservation."""

    @pytest.fixture
    def mock_completion_with_whitespace(self, monkeypatch):
        """Mock completion that simulates LLM stripping whitespace."""
        def mock_completion(*args, **kwargs):
            # Simulate LLM returning content without whitespace
            response_content = '{"translation": "Translated text", "glossary_extension": []}'

            return type(
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
                                    {"content": response_content},  # JSON response content
                                )()
                            },
                        )
                    ],
                    "usage": type(
                        "Usage",
                        (),
                        {
                            "total_tokens": 10,
                        },
                    )(),
                    "_hidden_params": {
                        "response_cost": 0.001
                    },
                },
            )

        import litellm
        monkeypatch.setattr(litellm, "completion", mock_completion)
        monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_completion)

    @pytest.mark.asyncio
    async def test_whitespace_preservation_end_to_end(self, mock_completion_with_whitespace):
        """Test complete whitespace preservation workflow."""
        translator = LiteLLMTranslator()
        
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
        
        # Should preserve original whitespace even though LLM stripped it
        assert response.text == "   Translated text   "
        assert response.tokens_used == 10
        assert response.cost == 0.001
        assert response.time_taken > 0

    @pytest.mark.asyncio
    async def test_whitespace_preservation_with_context(self, mock_completion_with_whitespace):
        """Test whitespace preservation with context."""
        translator = LiteLLMTranslator()
        
        context_info = "[PREVIOUS_CHUNK]\nPrevious\n[/PREVIOUS_CHUNK]\n\n[PREVIOUS_CHUNK_TRANSLATION]\nAnterior\n[/PREVIOUS_CHUNK_TRANSLATION]"
        
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="\n  Current text  \n",
            context=context_info,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )

        response = await translator.translate(request)
        
        # Should preserve whitespace even with context
        assert response.text == "\n  Translated text  \n"
        assert response.tokens_used == 10
        assert response.cost == 0.001
        assert response.time_taken > 0


    @pytest.mark.asyncio
    async def test_whitespace_preservation(self, mock_completion_with_whitespace):
        """Test whitespace preservation."""
        translator = LiteLLMTranslator()
        
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
        
        # Should preserve complex whitespace patterns
        assert response.text == "\n\n  Translated text  \n"
        assert response.tokens_used == 10
        assert response.cost == 0.001
        assert response.time_taken > 0
