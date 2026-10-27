"""Tests for resuming translation algorithms from checkpoints."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import (
    translate_context_aware,
    translate_page_by_page,
    translate_sliding_window,
)
from tinbox.core.translation.checkpoint import CheckpointManager, TranslationState
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationResponse,
)
from tinbox.core.types import ModelType, TranslationConfig


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


async def test_sliding_window_ignores_checkpoint_from_old_window_split(tmp_path):
    """Checkpoints saved before windows split at natural breaks are not resumed."""
    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="sliding-window",
        input_file=tmp_path / "test.txt",
        checkpoint_dir=tmp_path,
        resume_from_checkpoint=True,
    )
    manager = CheckpointManager(config)
    await manager.save(
        TranslationState(
            source_lang="en",
            target_lang="fr",
            algorithm="sliding-window",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={1: "Translated window 1"},
            token_usage=10,
            cost=0.001,
            time_taken=1.0,
        )
    )
    assert await manager.load() is not None

    checkpoint_path = tmp_path / "test_checkpoint.json"
    data = json.loads(checkpoint_path.read_text())
    del data["config"]["sliding_window_split_version"]
    checkpoint_path.write_text(json.dumps(data))

    assert await manager.load() is None
