"""Tests for translation algorithms."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import (
    create_windows,
    merge_chunks,
    translate_context_aware,
    translate_document,
    translate_page_by_page,
    translate_sliding_window,
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
    """Create a mock translator."""
    translator = AsyncMock(spec=ModelInterface)

    async def mock_translate(request):
        if hasattr(request, "content") and request.content == "Page 2":
            if getattr(mock_translate, "should_fail", False):
                raise TranslationError("Failed to translate page 2")
        return TranslationResponse(
            text="Translated text",
            tokens_used=10,
            cost=0.001,
            time_taken=0.5,
        )

    translator.translate = AsyncMock(side_effect=mock_translate)
    return translator


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock checkpoint manager."""
    manager = AsyncMock(spec=CheckpointManager)

    async def mock_load(*args, **kwargs):
        return TranslationState(
            source_lang="en",
            target_lang="fr",
            algorithm="page",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={1: "Translated page 1"},
            token_usage=10,
            cost=0.001,
            time_taken=1.0,
        )

    manager.load = AsyncMock(side_effect=mock_load)
    manager.save = AsyncMock()
    manager.cleanup_old_checkpoints = AsyncMock()
    return manager


@pytest.fixture
def test_content():
    """Create test document content."""
    pages = ["Page 1", "Page 2", "Page 3"]
    return DocumentContent(
        pages=pages,
        content_type="text/plain",
        metadata={"test": "metadata"},
    )


@pytest.fixture
def test_config(tmp_path):
    """Create test translation config with checkpointing."""
    return TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=tmp_path / "test.txt",
        checkpoint_dir=tmp_path,
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )


async def test_translate_page_by_page_with_checkpointing(
    test_content,
    test_config,
    mock_translator,
    mock_checkpoint_manager,
):
    """Test page-by-page translation with checkpointing."""
    # Translate document
    result = await translate_page_by_page(
        test_content,
        test_config,
        mock_translator,
        checkpoint_manager=mock_checkpoint_manager,
    )

    # Verify checkpoint loading
    mock_checkpoint_manager.load.assert_called_once()

    # Count translation calls for actual pages (excluding seam repair)
    page_translation_calls = [
        call
        for call in mock_translator.translate.call_args_list
        if "Page" in str(call[0][0].content)
    ]
    assert len(page_translation_calls) == 2  # Only pages 2 and 3

    # Verify checkpoint saving
    assert mock_checkpoint_manager.save.call_count >= 2
    saved_states = [call[0][0] for call in mock_checkpoint_manager.save.call_args_list]
    assert len(saved_states[-1].completed_pages) == 3  # All pages completed

    # Note: Cleanup is now handled in CLI after output file is written

    # Verify final result
    assert result.text  # Result after seam repair
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken > 0


async def test_translate_sliding_window_with_checkpointing(
    test_content,
    test_config,
    mock_translator,
    mock_checkpoint_manager,
):
    """Test sliding window translation with checkpointing."""
    # Create sliding window config with simple settings
    sliding_config = test_config.model_copy(
        update={
            "algorithm": "sliding-window",
            "window_size": 20,
            "overlap_size": 5,
            "checkpoint_dir": test_config.checkpoint_dir,
            "checkpoint_frequency": 1,
        }
    )

    # Create simple test content that will definitely be split into two windows
    test_content = DocumentContent(
        pages=["First window content. Second window content."],
        content_type="text/plain",
        metadata={"test": "metadata"},
    )

    print(f"Test content: {test_content.pages[0]}")  # Debug print
    print(f"Content length: {len(test_content.pages[0])}")  # Debug print

    # Track translation calls
    translation_calls = []

    async def mock_translate(request):
        print(f"Mock translate called with: {request.content}")  # Debug print
        translation_calls.append(request.content)
        return TranslationResponse(
            text=f"Translated: {request.content}",
            tokens_used=5,
            cost=0.001,
            time_taken=0.1,
        )

    mock_translator.translate = AsyncMock(side_effect=mock_translate)

    # Configure checkpoint manager to return no checkpoint
    mock_checkpoint_manager.load = AsyncMock(return_value=None)

    # Translate document
    result = await translate_sliding_window(
        test_content,
        sliding_config,
        mock_translator,
        checkpoint_manager=mock_checkpoint_manager,
    )

    # Debug prints
    print(f"Result text: {result.text}")
    print(f"Translation calls: {translation_calls}")

    # Verify we got at least two translation calls
    assert len(translation_calls) >= 2, (
        f"Expected at least 2 translation calls, got {len(translation_calls)}"
    )

    # Verify checkpoint was saved
    assert mock_checkpoint_manager.save.called, "Checkpoint manager save was not called"
    saved_states = [call[0][0] for call in mock_checkpoint_manager.save.call_args_list]
    print(f"Saved states: {saved_states}")  # Debug print

    # Basic result verification
    assert result.text
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken > 0


async def test_translation_without_checkpointing(
    test_content,
    mock_translator,
):
    """Test translation without checkpointing enabled."""
    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        checkpoint_dir=None,
    )

    # Set up mock translator with proper response
    async def mock_translate(request):
        return TranslationResponse(
            text="Translated text",
            tokens_used=10,
            cost=0.001,
            time_taken=0.5,
        )

    mock_translator.translate = AsyncMock(side_effect=mock_translate)

    # Translate document
    result = await translate_page_by_page(
        test_content,
        config,
        mock_translator,
    )

    # Count translation calls for actual pages (excluding seam repair)
    page_translation_calls = [
        call
        for call in mock_translator.translate.call_args_list
        if "Page" in str(call[0][0].content)
    ]
    assert len(page_translation_calls) == 3  # All three pages

    # Verify result
    assert result.text
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken > 0


async def test_translation_with_failed_pages(
    test_content,
    test_config,
    mock_translator,
    mock_checkpoint_manager,
):
    """Test translation with some failed pages."""
    # Create a new mock translator that fails on page 2
    translator = AsyncMock(spec=ModelInterface)

    async def mock_translate(request):
        if hasattr(request, "content") and request.content == "Page 2":
            raise TranslationError("Failed to translate page 2")
        return TranslationResponse(
            text="Translated text",
            tokens_used=10,
            cost=0.001,
            time_taken=0.5,
        )

    translator.translate = AsyncMock(side_effect=mock_translate)

    # Translate document - should succeed but mark page 2 as failed
    result = await translate_page_by_page(
        test_content,
        test_config,
        translator,
        checkpoint_manager=mock_checkpoint_manager,
    )

    # Verify checkpoint handling
    assert mock_checkpoint_manager.save.called
    saved_states = [call[0][0] for call in mock_checkpoint_manager.save.call_args_list]
    assert len(saved_states) > 0
    assert 2 in saved_states[-1].failed_pages  # Page 2 should be marked as failed

    # Verify result
    assert result.text  # Should contain pages 1 and 3
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken > 0


async def test_translation_all_pages_fail(
    test_content,
    test_config,
    mock_translator,
):
    """Test handling of all pages failing."""
    # Make all translations fail
    mock_translator.translate.side_effect = TranslationError(
        "No pages were successfully translated"
    )

    # Verify translation fails with appropriate error
    with pytest.raises(TranslationError) as exc_info:
        await translate_page_by_page(
            test_content,
            test_config,
            mock_translator,
        )

    # Check error message
    assert "No pages were successfully translated" in str(exc_info.value)
    assert "Failed pages: [1, 2, 3]" in str(exc_info.value)


# Helper function tests
def test_create_windows():
    """Test window creation for sliding window translation."""
    text = "This is a test text for window creation."
    windows = create_windows(text, window_size=10, overlap_size=3)

    assert len(windows) > 0
    assert all(isinstance(w, str) for w in windows)
    assert len(windows[0]) <= 10  # Should not exceed window size


def test_merge_chunks():
    """Test chunk merging for sliding window translation."""
    chunks = [
        "First chunk with overlap",
        "with overlap and more",
        "and more final content",
    ]

    # Test with smaller overlap to ensure proper merging
    result = merge_chunks(chunks, overlap_size=12)
    assert result
    assert isinstance(result, str)

    # Verify content preservation
    assert "First chunk" in result
    assert "and more" in result
    assert "final content" in result

    # Verify proper overlap handling
    assert result.count("with overlap") == 1  # Should only appear once after merging
    assert result.count("and more") == 1  # Should only appear once after merging

    # Test with no overlap
    no_overlap_result = merge_chunks(chunks, overlap_size=0)
    assert no_overlap_result == "".join(chunks)

    # Test with full overlap
    full_overlap_result = merge_chunks(chunks, overlap_size=len(chunks[0]))
    assert len(full_overlap_result) > 0
    assert "First chunk" in full_overlap_result
    assert "final content" in full_overlap_result


async def test_translate_document_context_aware(
    test_content,
    mock_translator,
    mock_checkpoint_manager,
):
    """Test translate_document function with context-aware algorithm."""
    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="context-aware",
        input_file=Path("test.txt"),
        context_size=50,
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
    )

    # Mock translator to handle context-aware format
    async def mock_context_translate(request, stream=False):
        content = request.content
        if "[TRANSLATE_THIS]" in content and "[/TRANSLATE_THIS]" in content:
            start = content.find("[TRANSLATE_THIS]") + len("[TRANSLATE_THIS]")
            end = content.find("[/TRANSLATE_THIS]")
            actual_content = content[start:end].strip()
            return TranslationResponse(
                text=f"Translated: {actual_content}",
                tokens_used=10,
                cost=0.001,
                time_taken=0.1,
            )
        return TranslationResponse(
            text="Translated text",
            tokens_used=10,
            cost=0.001,
            time_taken=0.1,
        )

    mock_translator.translate = AsyncMock(side_effect=mock_context_translate)

    # Configure checkpoint manager to return no checkpoint so translation actually runs
    mock_checkpoint_manager.load = AsyncMock(return_value=None)

    # Test the main translate_document function
    result = await translate_document(
        test_content,
        config,
        mock_translator,
        checkpoint_manager=mock_checkpoint_manager,
    )

    # Verify the result
    assert result.text
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken >= 0

    # Verify translator was called
    assert mock_translator.translate.called


async def test_translate_document_algorithm_routing():
    """Test that translate_document routes to correct algorithm."""
    mock_translator = AsyncMock(spec=ModelInterface)
    mock_translator.translate.return_value = TranslationResponse(
        text="Translated",
        tokens_used=10,
        cost=0.001,
        time_taken=0.1,
    )

    DocumentContent(
        pages=["Test content"],
        content_type="text/plain",
        metadata={},
    )

    # Test unknown algorithm - should fail at config validation time
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError,
        match="Input should be 'page', 'sliding-window' or 'context-aware'",
    ):
        TranslationConfig(
            source_lang="en",
            target_lang="fr",
            model=ModelType.ANTHROPIC,
            model_name="claude-3-sonnet",
            algorithm="unknown-algorithm",
            input_file=Path("test.txt"),
        )


# Comprehensive checkpoint recovery tests
async def test_page_by_page_checkpoint_recovery():
    """Test that page-by-page translation properly recovers from checkpoint."""
    # Create mock translator
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Newly translated",
        tokens_used=5,
        cost=0.0005,
        time_taken=0.1,
    )

    # Create checkpoint manager that returns partial completion
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = TranslationState(
        source_lang="en",
        target_lang="fr",
        algorithm="page",
        completed_pages=[1, 2],
        failed_pages=[],
        translated_chunks={1: "Translated page 1", 2: "Translated page 2"},
        token_usage=20,
        cost=0.002,
        time_taken=2.0,
    )
    checkpoint_manager.save = AsyncMock()
    checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    # Create content with 4 pages
    content = DocumentContent(
        pages=["Page 1", "Page 2", "Page 3", "Page 4"],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Translate
    result = await translate_page_by_page(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should load checkpoint
    checkpoint_manager.load.assert_called_once()

    # Should only translate remaining pages (3 and 4)
    assert translator.translate.call_count == 2

    # Should include checkpoint data in final result
    assert result.tokens_used >= 20  # At least the checkpoint tokens
    assert result.cost >= 0.002  # At least the checkpoint cost

    # Should save checkpoints for new translations
    assert checkpoint_manager.save.called

    # Note: Cleanup is now handled in CLI after output file is written


async def test_sliding_window_checkpoint_recovery():
    """Test that sliding window translation properly recovers from checkpoint."""
    # Create mock translator
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Newly translated window",
        tokens_used=5,
        cost=0.0005,
        time_taken=0.1,
    )

    # Create checkpoint manager that returns partial completion
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = TranslationState(
        source_lang="en",
        target_lang="fr",
        algorithm="sliding-window",
        completed_pages=[1],
        failed_pages=[],
        translated_chunks={1: "Translated window 1", 2: "Translated window 2"},
        token_usage=20,
        cost=0.002,
        time_taken=2.0,
    )
    checkpoint_manager.save = AsyncMock()
    checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    # Create content that will create multiple windows
    content = DocumentContent(
        pages=[
            "This is a long text that will be split into multiple windows for translation using sliding window algorithm"
        ],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="sliding-window",
        input_file=Path("test.txt"),
        window_size=20,
        overlap_size=5,
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Translate
    result = await translate_sliding_window(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should load checkpoint
    checkpoint_manager.load.assert_called_once()

    # Should translate remaining windows (not all windows)
    assert translator.translate.call_count >= 1  # At least some remaining windows

    # Should include checkpoint data in final result
    assert result.tokens_used >= 20  # At least the checkpoint tokens
    assert result.cost >= 0.002  # At least the checkpoint cost

    # Should save checkpoints for new translations
    assert checkpoint_manager.save.called

    # Note: Cleanup is now handled in CLI after output file is written


async def test_context_aware_checkpoint_recovery():
    """Test that context-aware translation properly recovers from checkpoint."""
    # Create mock translator that handles context format
    translator = AsyncMock(spec=ModelInterface)

    def mock_context_translate(request):
        content = request.content
        if "[TRANSLATE_THIS]" in content and "[/TRANSLATE_THIS]" in content:
            start = content.find("[TRANSLATE_THIS]") + len("[TRANSLATE_THIS]")
            end = content.find("[/TRANSLATE_THIS]")
            actual_content = content[start:end].strip()
            return TranslationResponse(
                text=f"Translated: {actual_content}",
                tokens_used=5,
                cost=0.0005,
                time_taken=0.1,
            )
        return TranslationResponse(
            text="Translated chunk",
            tokens_used=5,
            cost=0.0005,
            time_taken=0.1,
        )

    translator.translate = AsyncMock(side_effect=mock_context_translate)

    # Create checkpoint manager that returns partial completion
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = TranslationState(
        source_lang="en",
        target_lang="fr",
        algorithm="context-aware",
        completed_pages=[1],
        failed_pages=[],
        translated_chunks={1: "Translated chunk 1", 2: "Translated chunk 2"},
        token_usage=20,
        cost=0.002,
        time_taken=2.0,
    )
    checkpoint_manager.save = AsyncMock()
    checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    # Create content that will create multiple chunks
    content = DocumentContent(
        pages=[
            "This is chunk one content. This is chunk two content. This is chunk three content. This is chunk four content."
        ],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="context-aware",
        input_file=Path("test.txt"),
        context_size=30,  # Small size to create multiple chunks
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Translate
    result = await translate_context_aware(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should load checkpoint
    checkpoint_manager.load.assert_called_once()

    # Should translate remaining chunks (not all chunks)
    assert translator.translate.call_count >= 1  # At least some remaining chunks

    # Should include checkpoint data in final result
    assert result.tokens_used >= 20  # At least the checkpoint tokens
    assert result.cost >= 0.002  # At least the checkpoint cost

    # Should save checkpoints for new translations
    assert checkpoint_manager.save.called

    # Note: Cleanup is now handled in CLI after output file is written


async def test_no_checkpoint_available():
    """Test behavior when no checkpoint is available."""
    # Create mock translator
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Translated",
        tokens_used=5,
        cost=0.0005,
        time_taken=0.1,
    )

    # Create checkpoint manager that returns no checkpoint
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = None
    checkpoint_manager.save = AsyncMock()
    checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    content = DocumentContent(
        pages=["Page 1", "Page 2"],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Translate
    await translate_page_by_page(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should attempt to load checkpoint
    checkpoint_manager.load.assert_called_once()

    # Should translate all pages since no checkpoint
    assert translator.translate.call_count == 2

    # Should save checkpoints during translation
    assert checkpoint_manager.save.called

    # Note: Cleanup is now handled in CLI after output file is written


async def test_page_by_page_all_chunks_completed():
    """Test page-by-page translation when all chunks are already completed in checkpoint."""
    # Create mock translator - should NOT be called since all chunks are done
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="This should not be called",
        tokens_used=999,
        cost=999.0,
        time_taken=999.0,
    )

    # Create checkpoint manager that returns ALL chunks completed
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = TranslationState(
        source_lang="en",
        target_lang="fr",
        algorithm="page",
        completed_pages=[1, 2, 3],
        failed_pages=[],
        translated_chunks={
            1: "Translated page 1",
            2: "Translated page 2",
            3: "Translated page 3",
        },
        token_usage=150,
        cost=0.015,
        time_taken=30.0,
    )
    checkpoint_manager.save = AsyncMock()
    checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    # Create content with exactly 3 pages (same as completed chunks)
    content = DocumentContent(
        pages=["Page 1", "Page 2", "Page 3"],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Translate
    result = await translate_page_by_page(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should load checkpoint
    checkpoint_manager.load.assert_called_once()

    # Should NOT call translator since all chunks are completed
    translator.translate.assert_not_called()

    # Should return the checkpoint data directly
    assert result.tokens_used == 150
    assert result.cost == 0.015
    assert "Translated page 1" in result.text
    assert "Translated page 2" in result.text
    assert "Translated page 3" in result.text

    # Should NOT save new checkpoints since no new work was done
    checkpoint_manager.save.assert_not_called()


async def test_context_aware_all_chunks_completed():
    """Test context-aware translation when all chunks are already completed in checkpoint."""
    # Create mock translator - should NOT be called since all chunks are done
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="This should not be called",
        tokens_used=999,
        cost=999.0,
        time_taken=999.0,
    )

    # Create checkpoint manager that returns ALL chunks completed
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = TranslationState(
        source_lang="en",
        target_lang="fr",
        algorithm="context-aware",
        completed_pages=[1],
        failed_pages=[],
        translated_chunks={
            1: "Translated chunk 1",
            2: "Translated chunk 2",
            3: "Translated chunk 3",
        },
        token_usage=300,
        cost=0.03,
        time_taken=60.0,
    )
    checkpoint_manager.save = AsyncMock()
    checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    # Create content that will be split into exactly 3 chunks (same as completed chunks)
    content = DocumentContent(
        pages=[
            "Short chunk 1. Short chunk 2. Short chunk 3."
        ],  # Will create 3 chunks with small context_size
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="context-aware",
        input_file=Path("test.txt"),
        context_size=15,  # Small size to ensure exactly 3 chunks
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Translate
    result = await translate_context_aware(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should load checkpoint
    checkpoint_manager.load.assert_called_once()

    # Should NOT call translator since all chunks are completed
    translator.translate.assert_not_called()

    # Should return the checkpoint data directly
    assert result.tokens_used == 300
    assert result.cost == 0.03
    assert result.text == "Translated chunk 1Translated chunk 2Translated chunk 3"

    # Should NOT save new checkpoints since no new work was done
    checkpoint_manager.save.assert_not_called()


async def test_sliding_window_all_chunks_completed():
    """Test sliding window translation when all chunks are already completed in checkpoint."""
    # Create mock translator - should NOT be called since all chunks are done
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="This should not be called",
        tokens_used=999,
        cost=999.0,
        time_taken=999.0,
    )

    # Create checkpoint manager that returns ALL windows completed
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = TranslationState(
        source_lang="en",
        target_lang="fr",
        algorithm="sliding-window",
        completed_pages=[1],
        failed_pages=[],
        translated_chunks={1: "Translated window 1", 2: "Translated window 2"},
        token_usage=200,
        cost=0.02,
        time_taken=40.0,
    )
    checkpoint_manager.save = AsyncMock()
    checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    # Create content that will be split into exactly 2 windows (same as completed chunks)
    content = DocumentContent(
        pages=[
            "This is exactly thirty chars."
        ],  # Exactly 30 chars, will create exactly 2 windows with size 20 and overlap 5
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="sliding-window",
        input_file=Path("test.txt"),
        window_size=20,  # Size to ensure exactly 2 windows
        overlap_size=5,
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Translate
    result = await translate_sliding_window(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should load checkpoint
    checkpoint_manager.load.assert_called_once()

    # Should NOT call translator since all windows are completed
    translator.translate.assert_not_called()

    # Should return the checkpoint data directly
    assert result.tokens_used == 200
    assert result.cost == 0.02
    assert "Translated window 1" in result.text
    assert "Translated window 2" in result.text

    # Should NOT save new checkpoints since no new work was done
    checkpoint_manager.save.assert_not_called()


# Max cost tests
async def test_page_by_page_max_cost_exceeded():
    """Test that page-by-page translation stops when max_cost is exceeded."""
    # Create mock translator that returns expensive responses
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Expensive translation",
        tokens_used=1000,
        cost=5.0,  # High cost per page
        time_taken=1.0,
    )

    # Create checkpoint manager that returns no checkpoint
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = None
    checkpoint_manager.save = AsyncMock()

    content = DocumentContent(
        pages=["Page 1", "Page 2", "Page 3"],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        max_cost=8.0,  # Should allow 1-2 pages but not all 3
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Should raise TranslationError when max_cost is exceeded
    with pytest.raises(TranslationError) as exc_info:
        await translate_page_by_page(
            content, config, translator, checkpoint_manager=checkpoint_manager
        )

    # Check error message
    assert "Translation cost" in str(exc_info.value)
    assert "exceeded maximum cost" in str(exc_info.value)
    assert "8.00" in str(exc_info.value)

    # Should have translated at least one page before hitting the limit
    assert translator.translate.call_count >= 1


async def test_sliding_window_max_cost_exceeded():
    """Test that sliding window translation stops when max_cost is exceeded."""
    # Create mock translator that returns expensive responses
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Expensive translation",
        tokens_used=1000,
        cost=3.0,  # High cost per window
        time_taken=1.0,
    )

    # Create checkpoint manager that returns no checkpoint
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = None
    checkpoint_manager.save = AsyncMock()

    # Create content that will create multiple windows
    content = DocumentContent(
        pages=[
            "This is a long text that will be split into multiple windows for testing max cost functionality."
        ],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="sliding-window",
        input_file=Path("test.txt"),
        window_size=20,
        overlap_size=5,
        max_cost=5.0,  # Should allow 1-2 windows but not all
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Should raise TranslationError when max_cost is exceeded
    with pytest.raises(TranslationError) as exc_info:
        await translate_sliding_window(
            content, config, translator, checkpoint_manager=checkpoint_manager
        )

    # Check error message
    assert "Translation cost" in str(exc_info.value)
    assert "exceeded maximum cost" in str(exc_info.value)
    assert "5.00" in str(exc_info.value)

    # Should have translated at least one window before hitting the limit
    assert translator.translate.call_count >= 1


async def test_context_aware_max_cost_exceeded():
    """Test that context-aware translation stops when max_cost is exceeded."""
    # Create mock translator that returns expensive responses
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Expensive translation",
        tokens_used=1000,
        cost=4.0,  # High cost per chunk
        time_taken=1.0,
    )

    # Create checkpoint manager that returns no checkpoint
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = None
    checkpoint_manager.save = AsyncMock()

    # Create content that will create multiple chunks
    content = DocumentContent(
        pages=[
            "This is chunk one content. This is chunk two content. This is chunk three content."
        ],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="context-aware",
        input_file=Path("test.txt"),
        context_size=30,  # Small size to create multiple chunks
        max_cost=7.0,  # Should allow 1-2 chunks but not all
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Should raise TranslationError when max_cost is exceeded
    with pytest.raises(TranslationError) as exc_info:
        await translate_context_aware(
            content, config, translator, checkpoint_manager=checkpoint_manager
        )

    # Check error message
    assert "Translation cost" in str(exc_info.value)
    assert "exceeded maximum cost" in str(exc_info.value)
    assert "7.00" in str(exc_info.value)

    # Should have translated at least one chunk before hitting the limit
    assert translator.translate.call_count >= 1


async def test_page_by_page_max_cost_exactly_at_limit():
    """Test that page-by-page translation succeeds when cost is exactly at the limit."""
    # Create mock translator that returns responses with exact cost
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Translation",
        tokens_used=100,
        cost=2.5,  # Exact cost per page
        time_taken=0.5,
    )

    # Create checkpoint manager that returns no checkpoint
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = None
    checkpoint_manager.save = AsyncMock()

    content = DocumentContent(
        pages=["Page 1", "Page 2"],  # 2 pages * 2.5 = 5.0 exactly
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        max_cost=5.0,  # Exactly the total cost
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Should succeed without raising an error
    result = await translate_page_by_page(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should have translated all pages
    assert translator.translate.call_count == 2
    assert result.cost == 5.0
    assert result.text


async def test_page_by_page_no_max_cost():
    """Test that page-by-page translation works normally when no max_cost is set."""
    # Create mock translator that returns expensive responses
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Expensive translation",
        tokens_used=1000,
        cost=10.0,  # Very high cost per page
        time_taken=1.0,
    )

    # Create checkpoint manager that returns no checkpoint
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = None
    checkpoint_manager.save = AsyncMock()

    content = DocumentContent(
        pages=["Page 1", "Page 2", "Page 3"],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        max_cost=None,  # No cost limit
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Should succeed without raising an error
    result = await translate_page_by_page(
        content, config, translator, checkpoint_manager=checkpoint_manager
    )

    # Should have translated all pages despite high cost
    assert translator.translate.call_count == 3
    assert result.cost == 30.0  # 3 pages * 10.0
    assert result.text


async def test_max_cost_with_checkpoint_recovery():
    """Test max_cost check works correctly with checkpoint recovery."""
    # Create mock translator that returns expensive responses
    translator = AsyncMock(spec=ModelInterface)
    translator.translate.return_value = TranslationResponse(
        text="Expensive translation",
        tokens_used=1000,
        cost=4.0,  # High cost per page
        time_taken=1.0,
    )

    # Create checkpoint manager that returns partial completion with existing cost
    checkpoint_manager = AsyncMock(spec=CheckpointManager)
    checkpoint_manager.load.return_value = TranslationState(
        source_lang="en",
        target_lang="fr",
        algorithm="page",
        completed_pages=[1],
        failed_pages=[],
        translated_chunks={1: "Translated page 1"},
        token_usage=1000,
        cost=3.0,  # Already spent 3.0 from checkpoint
        time_taken=2.0,
    )
    checkpoint_manager.save = AsyncMock()

    content = DocumentContent(
        pages=["Page 1", "Page 2", "Page 3"],
        content_type="text/plain",
        metadata={},
    )

    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        max_cost=6.0,  # 3.0 (checkpoint) + 4.0 (one new page) = 7.0 > 6.0
        checkpoint_dir=Path("checkpoints"),
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )

    # Should raise TranslationError when max_cost is exceeded including checkpoint cost
    with pytest.raises(TranslationError) as exc_info:
        await translate_page_by_page(
            content, config, translator, checkpoint_manager=checkpoint_manager
        )

    # Check error message
    assert "Translation cost" in str(exc_info.value)
    assert "exceeded maximum cost" in str(exc_info.value)
    assert "6.00" in str(exc_info.value)

    # Should have translated exactly one new page before hitting the limit
    assert translator.translate.call_count == 1
