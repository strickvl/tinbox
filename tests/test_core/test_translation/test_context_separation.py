"""Tests for context and content separation functionality."""

import pytest

from tinbox.core.translation.algorithms import build_translation_context_info
from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.types import ModelType


class TestTranslationRequestWithContext:
    """Test TranslationRequest with context field."""

    def test_translation_request_with_context(self):
        """Test creating TranslationRequest with context."""
        context_info = "[PREVIOUS_CHUNK]\nPrevious text\n[/PREVIOUS_CHUNK]\n\n[PREVIOUS_CHUNK_TRANSLATION]\nTexto anterior\n[/PREVIOUS_CHUNK_TRANSLATION]"

        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Current text",
            context=context_info,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )

        assert request.source_lang == "en"
        assert request.target_lang == "es"
        assert request.content == "Current text"
        assert request.context == context_info
        assert request.content_type == "text/plain"
        assert request.model == ModelType.ANTHROPIC

    def test_translation_request_without_context(self):
        """Test creating TranslationRequest without context (default None)."""
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello, world!",
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )

        assert request.source_lang == "en"
        assert request.target_lang == "es"
        assert request.content == "Hello, world!"
        assert request.context is None
        assert request.content_type == "text/plain"
        assert request.model == ModelType.ANTHROPIC

    def test_translation_request_explicit_none_context(self):
        """Test creating TranslationRequest with explicit None context."""
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello, world!",
            context=None,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )

        assert request.source_lang == "en"
        assert request.target_lang == "es"
        assert request.content == "Hello, world!"
        assert request.context is None
        assert request.content_type == "text/plain"
        assert request.model == ModelType.ANTHROPIC


class TestBuildTranslationContextInfo:
    """Test context information building function."""

    def test_context_info_building(self):
        """Test building context information from previous chunks."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="Previous text",
            previous_translation="Texto anterior",
        )

        assert context is not None
        assert "[PREVIOUS_CHUNK]" in context
        assert "Previous text" in context
        assert "[/PREVIOUS_CHUNK]" in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "Texto anterior" in context
        assert "[/PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "Use this context to maintain consistency" in context

    def test_context_info_with_previous_chunks(self):
        """Test context building with both previous chunk and translation."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk="Hello world",
            previous_translation="Bonjour le monde",
        )

        expected_parts = [
            "[PREVIOUS_CHUNK]\nHello world\n[/PREVIOUS_CHUNK]",
            "[PREVIOUS_CHUNK_TRANSLATION]\nBonjour le monde\n[/PREVIOUS_CHUNK_TRANSLATION]",
            "Use this context to maintain consistency in terminology and style.",
        ]

        for part in expected_parts:
            assert part in context

    def test_context_info_without_previous_chunks(self):
        """Test context building with missing previous information."""
        # No previous chunk
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk=None,
            previous_translation="Texto anterior",
        )
        assert context is None

        # No previous translation
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="Previous text",
            previous_translation=None,
        )
        assert context is None

        # Neither previous chunk nor translation
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk=None,
            previous_translation=None,
        )
        assert context is None

    def test_context_info_with_empty_strings(self):
        """Test context building with empty string inputs."""
        # Empty previous chunk
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="",
            previous_translation="Texto anterior",
        )
        assert context is None

        # Empty previous translation
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="Previous text",
            previous_translation="",
        )
        assert context is None

        # Both empty
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="",
            previous_translation="",
        )
        assert context is None

    def test_context_info_tag_based_notation(self):
        """Test that context uses proper tag-based notation."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="Sample text",
            previous_translation="Texto de muestra",
        )

        # Check tag structure
        assert context.count("[PREVIOUS_CHUNK]") == 1
        assert context.count("[/PREVIOUS_CHUNK]") == 1
        assert context.count("[PREVIOUS_CHUNK_TRANSLATION]") == 1
        assert context.count("[/PREVIOUS_CHUNK_TRANSLATION]") == 1

        # Check that content is properly enclosed in tags
        source_start = context.find("[PREVIOUS_CHUNK]") + len("[PREVIOUS_CHUNK]")
        source_end = context.find("[/PREVIOUS_CHUNK]")
        source_content = context[source_start:source_end].strip()
        assert source_content == "Sample text"

        trans_start = context.find("[PREVIOUS_CHUNK_TRANSLATION]") + len(
            "[PREVIOUS_CHUNK_TRANSLATION]"
        )
        trans_end = context.find("[/PREVIOUS_CHUNK_TRANSLATION]")
        trans_content = context[trans_start:trans_end].strip()
        assert trans_content == "Texto de muestra"

    def test_context_info_multiline_content(self):
        """Test context building with multiline previous content."""
        previous_chunk = "Line 1\nLine 2\nLine 3"
        previous_translation = "Línea 1\nLínea 2\nLínea 3"

        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk=previous_chunk,
            previous_translation=previous_translation,
        )

        assert context is not None
        assert previous_chunk in context
        assert previous_translation in context

        # Verify multiline content is preserved
        assert "Line 1\nLine 2\nLine 3" in context
        assert "Línea 1\nLínea 2\nLínea 3" in context

    def test_context_info_with_next_chunk_functionality(self):
        """Test the new next chunk functionality."""
        # Test with only next chunk (first chunk scenario)
        context = build_translation_context_info(
            source_lang="en", target_lang="es", next_chunk="Next chunk content"
        )

        assert context is not None
        assert "[NEXT_CHUNK]" in context
        assert "Next chunk content" in context
        assert "[/NEXT_CHUNK]" in context
        assert "[PREVIOUS_CHUNK]" not in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" not in context

    def test_context_info_with_all_context_types(self):
        """Test context building with previous, translation, and next chunks."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="Previous text",
            previous_translation="Texto anterior",
            next_chunk="Next text",
        )

        assert context is not None
        # Check all context types are present
        assert "[PREVIOUS_CHUNK]" in context
        assert "Previous text" in context
        assert "[/PREVIOUS_CHUNK]" in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "Texto anterior" in context
        assert "[/PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "[NEXT_CHUNK]" in context
        assert "Next text" in context
        assert "[/NEXT_CHUNK]" in context

    def test_context_info_edge_case_empty_next(self):
        """Test context building with empty next chunk."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="Previous text",
            previous_translation="Texto anterior",
            next_chunk="",
        )

        assert context is not None
        # Should only have previous context, not next
        assert "[PREVIOUS_CHUNK]" in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "[NEXT_CHUNK]" not in context

    def test_context_info_incomplete_previous_with_next(self):
        """Test that incomplete previous context is ignored when next is available."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="Previous text",
            previous_translation=None,  # Missing translation
            next_chunk="Next text",
        )

        assert context is not None
        # Should only have next context since previous is incomplete
        assert "[NEXT_CHUNK]" in context
        assert "Next text" in context
        assert "[PREVIOUS_CHUNK]" not in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" not in context


class TestBackwardCompatibility:
    """Test backward compatibility of the new structure."""

    def test_backward_compatibility(self):
        """Test that old code patterns still work."""
        # Old pattern: creating request without context
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello, world!",
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )

        # Should work with context defaulting to None
        assert request.context is None
        assert request.content == "Hello, world!"

    def test_request_immutability(self):
        """Test that TranslationRequest remains immutable."""
        request = TranslationRequest(
            source_lang="en",
            target_lang="es",
            content="Hello, world!",
            context="Some context",
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )

        # Should not be able to modify the request
        with pytest.raises(Exception):  # ValidationError or AttributeError
            request.content = "Modified content"

        with pytest.raises(Exception):  # ValidationError or AttributeError
            request.context = "Modified context"
