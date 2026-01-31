"""Tests for context-aware translation algorithm."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import (
    build_translation_context_info,
    smart_text_split,
    translate_context_aware,
)
from tinbox.core.translation.checkpoint import CheckpointManager, TranslationState
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationResponse,
)
from tinbox.core.types import ModelType, TranslationConfig


@pytest.fixture
def mock_translator():
    """Create a mock translator for context-aware testing."""
    translator = AsyncMock(spec=ModelInterface)

    async def mock_translate(request):
        # Handle the new structure where content is pure and context is separate
        content = request.content
        return TranslationResponse(
            text=f"Translated: {content}",
            tokens_used=len(content.split()) * 2 if isinstance(content, str) else 10,
            cost=0.001,
            time_taken=0.1,
        )

    translator.translate = AsyncMock(side_effect=mock_translate)
    return translator


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock checkpoint manager."""
    manager = AsyncMock(spec=CheckpointManager)
    manager.load = AsyncMock(return_value=None)  # No existing checkpoint by default
    manager.save = AsyncMock()
    manager.cleanup_old_checkpoints = AsyncMock()
    return manager


@pytest.fixture
def context_aware_config(tmp_path):
    """Create test configuration for context-aware algorithm."""
    return TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="context-aware",
        input_file=tmp_path / "test.txt",
        context_size=50,  # Small size for testing
        custom_split_token=None,
        checkpoint_dir=tmp_path,
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )


class TestSmartTextSplit:
    """Test smart text splitting functionality."""

    def test_split_empty_text(self):
        """Test splitting empty text."""
        result = smart_text_split("", 100)
        assert result == []

    def test_split_invalid_target_size(self):
        """Test splitting with invalid target size."""
        with pytest.raises(ValueError, match="target_size must be positive"):
            smart_text_split("test", 0)

        with pytest.raises(ValueError, match="target_size must be positive"):
            smart_text_split("test", -1)

    def test_split_text_smaller_than_target(self):
        """Test splitting text smaller than target size."""
        text = "Short text"
        result = smart_text_split(text, 100)
        assert result == [text]

    def test_split_by_custom_token(self):
        """Test splitting by custom split token."""
        text = "First part|||Second part|||Third part"
        result = smart_text_split(text, 100, custom_split_token="|||")
        expected = ["First part", "Second part", "Third part"]
        assert result == expected

    def test_split_by_custom_token_ignores_target_size(self):
        """Test that custom token splitting ignores target size."""
        text = "Very long first part that exceeds target|||Short second"
        result = smart_text_split(text, 10, custom_split_token="|||")
        expected = ["Very long first part that exceeds target", "Short second"]
        assert result == expected

    def test_split_by_paragraphs(self):
        """Test splitting by paragraph breaks."""
        text = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph."
        result = smart_text_split(text, 30)
        assert len(result) >= 2
        assert "First paragraph." in result[0]
        assert "Second paragraph" in result[1]

    def test_split_by_sentences(self):
        """Test splitting by sentence endings."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        result = smart_text_split(text, 25)
        assert len(result) >= 2
        assert any("First sentence." in chunk for chunk in result)
        assert any("Second sentence!" in chunk for chunk in result)

    def test_split_by_line_breaks(self):
        """Test splitting by line breaks."""
        text = "First line\nSecond line\nThird line\nFourth line"
        result = smart_text_split(text, 20)
        assert len(result) >= 2
        assert "First line" in result[0]
        assert "Second line" in result[0] or "Second line" in result[1]

    def test_split_by_clause_boundaries(self):
        """Test splitting by clause boundaries."""
        text = "First clause; second clause: third clause, fourth clause"
        result = smart_text_split(text, 25)
        assert len(result) >= 2
        assert "First clause;" in result[0]

    def test_split_by_word_boundaries(self):
        """Test splitting by word boundaries as fallback."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        result = smart_text_split(text, 20)
        assert len(result) >= 2
        assert all(chunk.strip() for chunk in result)  # No empty chunks

    def test_split_fallback_to_character_boundary(self):
        """Test fallback to character boundary when no word boundaries."""
        text = "verylongwordwithoutspacesorpunctuation"
        result = smart_text_split(text, 10)
        assert len(result) >= 2
        assert all(len(chunk) <= 10 for chunk in result)


class TestBuildTranslationContextInfo:
    """Test translation context information building."""

    def test_build_context_info_without_previous(self):
        """Test building context info for first chunk (no previous context)."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
        )

        assert context is None

    def test_build_context_info_with_previous(self):
        """Test building context info with previous chunk and translation."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk="First sentence",
            previous_translation="Première phrase",
        )

        assert context is not None
        assert "[PREVIOUS_CHUNK]" in context
        assert "First sentence" in context
        assert "[/PREVIOUS_CHUNK]" in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "Première phrase" in context
        assert "[/PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "Use this context to maintain consistency" in context

    def test_build_context_info_with_partial_previous(self):
        """Test building context info with only previous chunk (no translation)."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk="Previous text",
            previous_translation=None,
        )

        # Should return None if translation is missing
        assert context is None

        # Test with missing previous chunk
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk=None,
            previous_translation="Previous translation",
        )

        # Should return None if chunk is missing
        assert context is None

    def test_build_context_info_with_next_chunk_only(self):
        """Test building context info with only next chunk (first chunk scenario)."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            next_chunk="Next chunk content",
        )

        assert context is not None
        assert "[NEXT_CHUNK]" in context
        assert "Next chunk content" in context
        assert "[/NEXT_CHUNK]" in context
        assert "Use this context to maintain consistency" in context
        # Should not contain previous context tags
        assert "[PREVIOUS_CHUNK]" not in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" not in context

    def test_build_context_info_with_previous_and_next(self):
        """Test building context info with both previous and next chunks (middle chunk scenario)."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk="Previous content",
            previous_translation="Contenu précédent",
            next_chunk="Next content",
        )

        assert context is not None
        # Check previous context
        assert "[PREVIOUS_CHUNK]" in context
        assert "Previous content" in context
        assert "[/PREVIOUS_CHUNK]" in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "Contenu précédent" in context
        assert "[/PREVIOUS_CHUNK_TRANSLATION]" in context
        # Check next context
        assert "[NEXT_CHUNK]" in context
        assert "Next content" in context
        assert "[/NEXT_CHUNK]" in context
        assert "Use this context to maintain consistency" in context

    def test_build_context_info_with_previous_only(self):
        """Test building context info with only previous chunk (last chunk scenario)."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk="Previous content",
            previous_translation="Contenu précédent",
        )

        assert context is not None
        assert "[PREVIOUS_CHUNK]" in context
        assert "Previous content" in context
        assert "[/PREVIOUS_CHUNK]" in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "Contenu précédent" in context
        assert "[/PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "Use this context to maintain consistency" in context
        # Should not contain next context tags
        assert "[NEXT_CHUNK]" not in context
        assert "[/NEXT_CHUNK]" not in context

    def test_build_context_info_with_empty_next_chunk(self):
        """Test building context info with empty next chunk."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk="Previous content",
            previous_translation="Contenu précédent",
            next_chunk="",
        )

        assert context is not None
        # Should only contain previous context since next chunk is empty
        assert "[PREVIOUS_CHUNK]" in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "[NEXT_CHUNK]" not in context

    def test_build_context_info_with_none_next_chunk(self):
        """Test building context info with None next chunk."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk="Previous content",
            previous_translation="Contenu précédent",
            next_chunk=None,
        )

        assert context is not None
        # Should only contain previous context since next chunk is None
        assert "[PREVIOUS_CHUNK]" in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" in context
        assert "[NEXT_CHUNK]" not in context

    def test_build_context_info_previous_chunk_missing_translation_with_next(self):
        """Test that previous chunk without translation is ignored, but next chunk is used."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="fr",
            previous_chunk="Previous content",
            previous_translation=None,
            next_chunk="Next content",
        )

        assert context is not None
        # Should only contain next context since previous is incomplete
        assert "[NEXT_CHUNK]" in context
        assert "Next content" in context
        assert "[PREVIOUS_CHUNK]" not in context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" not in context

    def test_build_context_info_multiline_next_chunk(self):
        """Test building context info with multiline next chunk."""
        next_chunk = "Line 1\nLine 2\nLine 3"

        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            next_chunk=next_chunk,
        )

        assert context is not None
        assert next_chunk in context
        
        # Verify multiline content is preserved in next chunk
        assert "Line 1\nLine 2\nLine 3" in context
        
        # Check that content is properly enclosed in tags
        next_start = context.find("[NEXT_CHUNK]") + len("[NEXT_CHUNK]")
        next_end = context.find("[/NEXT_CHUNK]")
        next_content = context[next_start:next_end].strip()
        assert next_content == next_chunk

    def test_build_context_info_tag_structure_with_next(self):
        """Test proper tag structure when next chunk is included."""
        context = build_translation_context_info(
            source_lang="en",
            target_lang="es",
            previous_chunk="Previous text",
            previous_translation="Texto anterior",
            next_chunk="Next text",
        )

        # Check all tag counts
        assert context.count("[PREVIOUS_CHUNK]") == 1
        assert context.count("[/PREVIOUS_CHUNK]") == 1
        assert context.count("[PREVIOUS_CHUNK_TRANSLATION]") == 1
        assert context.count("[/PREVIOUS_CHUNK_TRANSLATION]") == 1
        assert context.count("[NEXT_CHUNK]") == 1
        assert context.count("[/NEXT_CHUNK]") == 1

        # Check that content is properly enclosed in tags
        prev_source_start = context.find("[PREVIOUS_CHUNK]") + len("[PREVIOUS_CHUNK]")
        prev_source_end = context.find("[/PREVIOUS_CHUNK]")
        prev_source_content = context[prev_source_start:prev_source_end].strip()
        assert prev_source_content == "Previous text"

        prev_trans_start = context.find("[PREVIOUS_CHUNK_TRANSLATION]") + len("[PREVIOUS_CHUNK_TRANSLATION]")
        prev_trans_end = context.find("[/PREVIOUS_CHUNK_TRANSLATION]")
        prev_trans_content = context[prev_trans_start:prev_trans_end].strip()
        assert prev_trans_content == "Texto anterior"

        next_source_start = context.find("[NEXT_CHUNK]") + len("[NEXT_CHUNK]")
        next_source_end = context.find("[/NEXT_CHUNK]")
        next_source_content = context[next_source_start:next_source_end].strip()
        assert next_source_content == "Next text"


class TestTranslateContextAware:
    """Test context-aware translation algorithm."""

    async def test_translate_single_chunk(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translating content that fits in a single chunk."""
        content = DocumentContent(
            pages=["Short text that fits in one chunk."],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should have one translation call
        assert mock_translator.translate.call_count == 1
        
        # Verify the call was made with proper structure
        call_args = mock_translator.translate.call_args[0][0]
        assert call_args.content == "Short text that fits in one chunk."
        
        # Should not have context for first chunk
        assert call_args.context is None

        # Verify result
        assert result.text
        assert result.tokens_used > 0
        assert result.cost > 0
        assert result.time_taken >= 0

    async def test_translate_multiple_chunks(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translating content that requires multiple chunks."""
        # Create content that will definitely be split into multiple chunks
        long_content = "This is the first sentence. " * 10 + "This is the second sentence. " * 10
        content = DocumentContent(
            pages=[long_content],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should have multiple translation calls
        assert mock_translator.translate.call_count >= 2

        # First call should have next chunk context (but no previous)
        first_call = mock_translator.translate.call_args_list[0][0][0]
        assert first_call.context is not None
        assert "[NEXT_CHUNK]" in first_call.context
        assert "[PREVIOUS_CHUNK]" not in first_call.context
        assert "[PREVIOUS_CHUNK_TRANSLATION]" not in first_call.context

        # Second call should have both previous and next context
        if mock_translator.translate.call_count > 1:
            second_call = mock_translator.translate.call_args_list[1][0][0]
            assert second_call.context is not None
            assert "[PREVIOUS_CHUNK]" in second_call.context
            assert "[PREVIOUS_CHUNK_TRANSLATION]" in second_call.context
            # Should also have next context unless it's the last chunk
            if mock_translator.translate.call_count > 2:
                assert "[NEXT_CHUNK]" in second_call.context

        # Verify result
        assert result.text
        assert result.tokens_used > 0
        assert result.cost > 0

    async def test_translate_with_custom_split_token(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translation with custom split token."""
        config = context_aware_config.model_copy(
            update={"custom_split_token": "|||"}
        )
        
        content = DocumentContent(
            pages=["First part|||Second part|||Third part"],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should have exactly 3 translation calls (one per part)
        assert mock_translator.translate.call_count == 3

        # Verify each call contains the expected content
        calls = mock_translator.translate.call_args_list
        assert calls[0][0][0].content == "First part"
        assert calls[1][0][0].content == "Second part"
        assert calls[2][0][0].content == "Third part"

        # Verify result
        assert result.text
        assert "Translated: First part" in result.text
        assert "Translated: Second part" in result.text
        assert "Translated: Third part" in result.text

    async def test_translate_with_checkpointing(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translation with checkpoint saving."""
        content = DocumentContent(
            pages=["Content that will be split into multiple chunks for testing checkpointing functionality."],
            content_type="text/plain",
            metadata={},
        )

        await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Verify checkpoint was loaded
        mock_checkpoint_manager.load.assert_called_once()

        # Verify checkpoint was saved (at least once)
        assert mock_checkpoint_manager.save.call_count >= 1

        # Note: Cleanup is now handled in CLI after output file is written

    async def test_translate_resume_from_checkpoint(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test resuming translation from checkpoint."""
        # Configure checkpoint manager to return existing checkpoint with partial completion
        existing_checkpoint = TranslationState(
            source_lang="en",
            target_lang="fr",
            algorithm="context-aware",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={1: "Previously translated chunk 1"},
            token_usage=50,
            cost=0.05,
            time_taken=5.0,
        )
        mock_checkpoint_manager.load = AsyncMock(return_value=existing_checkpoint)

        # Content that will create multiple chunks, so we can test resumption
        content = DocumentContent(
            pages=["First chunk content. Second chunk content that will be processed."],
            content_type="text/plain",
            metadata={},
        )

        # Mock translator to handle new structure
        def mock_context_translate(request):
            content = request.content
            return TranslationResponse(
                text=f"Translated: {content}",
                tokens_used=10,
                cost=0.01,
                time_taken=0.1,
            )
        
        mock_translator.translate = AsyncMock(side_effect=mock_context_translate)

        result = await translate_context_aware(
            content,
            context_aware_config,
            checkpoint_manager=mock_checkpoint_manager,
            translator=mock_translator,
        )

        # Should load checkpoint
        mock_checkpoint_manager.load.assert_called_once()

        # Should make translation calls for remaining chunks
        assert mock_translator.translate.called

        # Should include checkpoint data in final result
        assert result.tokens_used >= 50  # At least the checkpoint tokens
        assert result.cost >= 0.05  # At least the checkpoint cost
        assert "Previously translated chunk 1" in result.text

    async def test_translate_image_content_error(
        self, context_aware_config, mock_translator
    ):
        """Test that context-aware algorithm rejects image content."""
        content = DocumentContent(
            pages=[b"binary image data"],
            content_type="image/png",
            metadata={},
        )

        with pytest.raises(TranslationError, match="Context-aware algorithm not supported for image content"):
            await translate_context_aware(
                content,
                context_aware_config,
                mock_translator,
            )

    async def test_translate_with_translation_error(
        self, context_aware_config, mock_checkpoint_manager
    ):
        """Test handling of translation errors."""
        # Create a translator that always fails
        failing_translator = AsyncMock(spec=ModelInterface)
        failing_translator.translate.side_effect = TranslationError("Translation failed")

        content = DocumentContent(
            pages=["Test content"],
            content_type="text/plain",
            metadata={},
        )

        with pytest.raises(TranslationError, match="Translation failed"):
            await translate_context_aware(
                content,
                context_aware_config,
                failing_translator,
                checkpoint_manager=mock_checkpoint_manager,
            )

    async def test_translate_multiple_pages(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translation with multiple pages."""
        content = DocumentContent(
            pages=["First page content.", "Second page content.", "Third page content."],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should translate the combined content
        assert result.text
        assert "Translated:" in result.text

        # Verify the combined text was processed
        first_call = mock_translator.translate.call_args_list[0][0][0]
        # The content should contain the combined pages
        combined_content = first_call.content
        assert "First page content." in combined_content
        assert "Second page content." in combined_content

    async def test_translate_without_checkpoint_manager(
        self, context_aware_config, mock_translator
    ):
        """Test translation without checkpoint manager."""
        content = DocumentContent(
            pages=["Simple content"],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=None,
        )

        # Should still work without checkpoint manager
        assert result.text
        assert result.tokens_used > 0
        assert result.cost > 0
