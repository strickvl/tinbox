"""Tests for translation algorithms."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

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


@pytest.mark.parametrize(
    "algorithm", [translate_sliding_window, translate_context_aware]
)
async def test_text_algorithms_reject_mixed_pages(
    algorithm, test_config, mock_translator
):
    content = DocumentContent(
        pages=["Text page", b"image page"], content_type="text/plain"
    )

    with pytest.raises(TranslationError, match="not supported for image content"):
        await algorithm(content, test_config, mock_translator)

    mock_translator.translate.assert_not_awaited()


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
@pytest.mark.parametrize(
    ("text", "window_size", "overlap_size", "expected"),
    [
        ("", 4, 0, []),
        ("abc", 4, 0, ["abc"]),
        ("abcd", 4, 0, ["abcd"]),
        ("abcdefgh", 4, 0, ["abcd", "efgh"]),
        ("abcdefghij", 4, 0, ["abcd", "efgh", "ij"]),
        ("abcdefghij", 4, 1, ["abcd", "defg", "ghij"]),
        ("abcdef", 4, 3, ["abcd", "bcde", "cdef"]),
    ],
)
def test_create_windows(text, window_size, overlap_size, expected):
    """Test window creation for sliding window translation."""
    assert create_windows(text, window_size, overlap_size) == expected


@pytest.mark.parametrize("overlap_size", [-1, 4, 5])
def test_create_windows_rejects_invalid_overlap(overlap_size):
    with pytest.raises(ValueError, match="Overlap size"):
        create_windows("abcdefghij", window_size=4, overlap_size=overlap_size)


async def test_translate_sliding_window_without_overlap(tmp_path, mock_translator):
    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="sliding-window",
        input_file=tmp_path / "test.txt",
        window_size=4,
        overlap_size=0,
    )
    content = DocumentContent(
        pages=["abcdefghij"],
        content_type="text/plain",
        metadata={},
    )

    async def uppercase(request):
        return TranslationResponse(
            text=request.content.upper(),
            tokens_used=1,
            cost=0.001,
            time_taken=0.01,
        )

    mock_translator.translate.side_effect = uppercase

    result = await translate_sliding_window(content, config, mock_translator)

    assert result.text == "ABCDEFGHIJ"


def test_translation_config_rejects_negative_overlap(test_config):
    with pytest.raises(ValidationError, match="overlap_size"):
        TranslationConfig.model_validate(
            test_config.model_dump() | {"overlap_size": -1}
        )


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
    with pytest.raises(
        ValidationError,
        match="Input should be 'page', 'sliding-window' or 'context-aware'",
    ):
        TranslationConfig.model_validate(
            {
                "source_lang": "en",
                "target_lang": "fr",
                "model": ModelType.ANTHROPIC,
                "model_name": "claude-3-sonnet",
                "algorithm": "unknown-algorithm",
                "input_file": Path("test.txt"),
            }
        )
