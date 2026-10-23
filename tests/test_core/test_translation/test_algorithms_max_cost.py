"""Tests for translation algorithms stopping at the max cost limit."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import (
    translate_context_aware,
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
