"""Comprehensive tests for checkpoint management."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tinbox.core.translation.checkpoint import (
    CheckpointManager,
    ResumeResult,
    TranslationState,
    load_checkpoint,
    resume_from_checkpoint,
    should_resume,
)
from tinbox.core.types import ModelType, TranslationConfig


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def sample_config(tmp_path, temp_checkpoint_dir):
    """Create a sample translation configuration."""
    input_file = tmp_path / "test_document.pdf"
    input_file.touch()  # Create the file

    return TranslationConfig(
        source_lang="en",
        target_lang="es",
        model=ModelType.OPENAI,
        model_name="gpt-4o",
        algorithm="context-aware",
        input_file=input_file,
        checkpoint_dir=temp_checkpoint_dir,
        resume_from_checkpoint=True,
    )


@pytest.fixture
def sample_translation_state():
    """Create a sample translation state."""
    return TranslationState(
        source_lang="en",
        target_lang="es",
        algorithm="context-aware",
        completed_pages=[1, 2, 3],
        failed_pages=[4],
        translated_chunks={
            1: "Translated chunk 1",
            2: "Translated chunk 2",
            3: "Translated chunk 3",
        },
        token_usage=1500,
        cost=0.15,
        time_taken=45.5,
    )


class TestTranslationState:
    """Test the TranslationState dataclass."""

    def test_translation_state_creation(self):
        """Test creating a TranslationState."""
        state = TranslationState(
            source_lang="en",
            target_lang="fr",
            algorithm="page",
            completed_pages=[1, 2],
            failed_pages=[3],
            translated_chunks={1: "Hello", 2: "World"},
            token_usage=100,
            cost=0.01,
            time_taken=10.5,
        )

        assert state.source_lang == "en"
        assert state.target_lang == "fr"
        assert state.algorithm == "page"
        assert state.completed_pages == [1, 2]
        assert state.failed_pages == [3]
        assert state.translated_chunks == {1: "Hello", 2: "World"}
        assert state.token_usage == 100
        assert state.cost == 0.01
        assert state.time_taken == 10.5


class TestCheckpointManager:
    """Test the CheckpointManager class."""

    def test_init(self, sample_config):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(sample_config)
        assert manager.config == sample_config
        assert manager._logger is not None

    def test_get_checkpoint_path(self, sample_config, temp_checkpoint_dir):
        """Test getting checkpoint path."""
        manager = CheckpointManager(sample_config)
        checkpoint_path = manager._get_checkpoint_path()

        expected_path = temp_checkpoint_dir / "test_document_checkpoint.json"
        assert checkpoint_path == expected_path
        assert temp_checkpoint_dir.exists()

    def test_get_checkpoint_path_no_dir_configured(self, sample_config):
        """Test getting checkpoint path when no directory is configured."""
        config = sample_config.model_copy(update={"checkpoint_dir": None})
        manager = CheckpointManager(config)

        with pytest.raises(ValueError, match="No checkpoint directory configured"):
            manager._get_checkpoint_path()

    async def test_save_checkpoint(
        self, sample_config, sample_translation_state, temp_checkpoint_dir
    ):
        """Test saving a checkpoint."""
        manager = CheckpointManager(sample_config)
        await manager.save(sample_translation_state)

        # Verify checkpoint file was created
        checkpoint_path = temp_checkpoint_dir / "test_document_checkpoint.json"
        assert checkpoint_path.exists()

        # Verify checkpoint content
        with open(checkpoint_path) as f:
            data = json.load(f)

        assert data["source_lang"] == "en"
        assert data["target_lang"] == "es"
        assert data["algorithm"] == "context-aware"
        assert data["completed_pages"] == [1, 2, 3]
        assert data["failed_pages"] == [4]
        assert data["translated_chunks"] == {
            "1": "Translated chunk 1",
            "2": "Translated chunk 2",
            "3": "Translated chunk 3",
        }
        assert data["token_usage"] == 1500
        assert data["cost"] == 0.15
        assert data["time_taken"] == 45.5

        # Verify config section
        config_data = data["config"]
        assert config_data["source_lang"] == "en"
        assert config_data["target_lang"] == "es"
        assert config_data["model"] == "openai"
        assert config_data["algorithm"] == "context-aware"

    async def test_save_checkpoint_atomic_write(
        self, sample_config, sample_translation_state, temp_checkpoint_dir
    ):
        """Test that checkpoint saving is atomic."""
        manager = CheckpointManager(sample_config)

        # Mock the json.dump to fail after creating temp file
        with patch("json.dump", side_effect=Exception("Write failed")):
            with pytest.raises(Exception, match="Write failed"):
                await manager.save(sample_translation_state)

        # Verify no checkpoint file exists (atomic write failed)
        checkpoint_path = temp_checkpoint_dir / "test_document_checkpoint.json"
        assert not checkpoint_path.exists()

        # Note: temp file might still exist if json.dump fails, which is expected behavior
        # The important thing is that the final checkpoint file doesn't exist

    async def test_load_checkpoint_success(
        self, sample_config, sample_translation_state, temp_checkpoint_dir
    ):
        """Test loading a checkpoint successfully."""
        manager = CheckpointManager(sample_config)

        # First save a checkpoint
        await manager.save(sample_translation_state)

        # Then load it
        loaded_state = await manager.load()

        assert loaded_state is not None
        assert loaded_state.source_lang == "en"
        assert loaded_state.target_lang == "es"
        assert loaded_state.algorithm == "context-aware"
        assert loaded_state.completed_pages == [1, 2, 3]
        assert loaded_state.failed_pages == [4]
        assert loaded_state.translated_chunks == {
            1: "Translated chunk 1",
            2: "Translated chunk 2",
            3: "Translated chunk 3",
        }
        assert loaded_state.token_usage == 1500
        assert loaded_state.cost == 0.15
        assert loaded_state.time_taken == 45.5

    async def test_load_checkpoint_no_file(self, sample_config):
        """Test loading checkpoint when no file exists."""
        manager = CheckpointManager(sample_config)
        loaded_state = await manager.load()
        assert loaded_state is None

    async def test_load_checkpoint_config_mismatch(
        self, sample_config, sample_translation_state
    ):
        """Test loading checkpoint with configuration mismatch."""
        manager = CheckpointManager(sample_config)

        # Save checkpoint with current config
        await manager.save(sample_translation_state)

        # Create new config with different settings
        different_config = sample_config.model_copy(update={"target_lang": "fr"})
        different_manager = CheckpointManager(different_config)

        # Try to load with different config
        loaded_state = await different_manager.load()
        assert loaded_state is None

    async def test_load_checkpoint_corrupted_file(
        self, sample_config, temp_checkpoint_dir
    ):
        """Test loading a corrupted checkpoint file."""
        manager = CheckpointManager(sample_config)

        # Create a corrupted checkpoint file
        checkpoint_path = temp_checkpoint_dir / "test_document_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            f.write("invalid json content")

        loaded_state = await manager.load()
        assert loaded_state is None

    async def test_load_checkpoint_missing_fields(
        self, sample_config, temp_checkpoint_dir
    ):
        """Test loading checkpoint with missing required fields."""
        manager = CheckpointManager(sample_config)

        # Create checkpoint with missing fields
        checkpoint_path = temp_checkpoint_dir / "test_document_checkpoint.json"
        incomplete_data = {
            "source_lang": "en",
            "target_lang": "es",
            # Missing other required fields
        }

        with open(checkpoint_path, "w") as f:
            json.dump(incomplete_data, f)

        loaded_state = await manager.load()
        assert loaded_state is None

    async def test_cleanup_old_checkpoints(
        self, sample_config, sample_translation_state, temp_checkpoint_dir
    ):
        """Test cleaning up old checkpoints."""
        manager = CheckpointManager(sample_config)

        # Create a checkpoint
        await manager.save(sample_translation_state)
        checkpoint_path = temp_checkpoint_dir / "test_document_checkpoint.json"
        assert checkpoint_path.exists()

        # Clean up checkpoints
        await manager.cleanup_old_checkpoints(sample_config.input_file)

        # Verify checkpoint was deleted
        assert not checkpoint_path.exists()

    async def test_cleanup_old_checkpoints_no_file(self, sample_config):
        """Test cleaning up checkpoints when no file exists."""
        manager = CheckpointManager(sample_config)

        # Should not raise an error
        await manager.cleanup_old_checkpoints(sample_config.input_file)

    async def test_cleanup_old_checkpoints_permission_error(
        self, sample_config, sample_translation_state, temp_checkpoint_dir
    ):
        """Test cleanup when file deletion fails."""
        manager = CheckpointManager(sample_config)

        # Create a checkpoint
        await manager.save(sample_translation_state)

        # Mock unlink to raise permission error
        with patch.object(
            Path, "unlink", side_effect=PermissionError("Permission denied")
        ):
            # Should not raise an error (graceful handling)
            await manager.cleanup_old_checkpoints(sample_config.input_file)


class TestUtilityFunctions:
    """Test utility functions for checkpoint management."""

    def test_should_resume_true(self, sample_config):
        """Test should_resume returns True when conditions are met."""
        assert should_resume(sample_config) is True

    def test_should_resume_no_checkpoint_dir(self, sample_config):
        """Test should_resume returns False when no checkpoint directory."""
        config = sample_config.model_copy(update={"checkpoint_dir": None})
        assert should_resume(config) is False

    def test_should_resume_resume_disabled(self, sample_config):
        """Test should_resume returns False when resume is disabled."""
        config = sample_config.model_copy(update={"resume_from_checkpoint": False})
        assert should_resume(config) is False

    def test_should_resume_dir_not_exists(self, sample_config, tmp_path):
        """Test should_resume returns False when checkpoint directory doesn't exist."""
        non_existent_dir = tmp_path / "non_existent"
        config = sample_config.model_copy(update={"checkpoint_dir": non_existent_dir})
        assert should_resume(config) is False

    async def test_load_checkpoint_should_not_resume(self, sample_config):
        """Test load_checkpoint returns None when should not resume."""
        config = sample_config.model_copy(update={"resume_from_checkpoint": False})
        result = await load_checkpoint(config)
        assert result is None

    async def test_load_checkpoint_calls_manager(self, sample_config):
        """Test load_checkpoint calls CheckpointManager.load."""
        with patch(
            "tinbox.core.translation.checkpoint.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = mock_manager_class.return_value
            mock_manager.load = AsyncMock(return_value=None)

            result = await load_checkpoint(sample_config)

            mock_manager_class.assert_called_once_with(sample_config)
            mock_manager.load.assert_called_once()
            assert result is None


class TestErrorHandling:
    """Test error handling in checkpoint management."""

    async def test_save_checkpoint_io_error(
        self, sample_config, sample_translation_state
    ):
        """Test save checkpoint handles IO errors."""
        manager = CheckpointManager(sample_config)

        with patch("builtins.open", side_effect=OSError("Disk full")):
            with pytest.raises(IOError, match="Disk full"):
                await manager.save(sample_translation_state)

    async def test_load_checkpoint_io_error(self, sample_config, temp_checkpoint_dir):
        """Test load checkpoint handles IO errors gracefully."""
        manager = CheckpointManager(sample_config)

        # Create a checkpoint file
        checkpoint_path = temp_checkpoint_dir / "test_document_checkpoint.json"
        checkpoint_path.touch()

        with patch("builtins.open", side_effect=OSError("Permission denied")):
            loaded_state = await manager.load()
            assert loaded_state is None


class TestResumeFromCheckpoint:
    """Test the resume_from_checkpoint utility function."""

    async def test_resume_no_checkpoint_manager(self, sample_config):
        """Test resume when no checkpoint manager is provided."""
        result = await resume_from_checkpoint(None, sample_config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is False
        assert result.translated_items == []
        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.metadata == {}

    async def test_resume_disabled(self, sample_config):
        """Test resume when resume_from_checkpoint is disabled."""
        config = sample_config.model_copy(update={"resume_from_checkpoint": False})
        manager = AsyncMock(spec=CheckpointManager)

        result = await resume_from_checkpoint(manager, config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is False
        assert result.translated_items == []
        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.metadata == {}

        # Manager.load should not be called
        manager.load.assert_not_called()

    async def test_resume_no_checkpoint_found(self, sample_config):
        """Test resume when no checkpoint file exists."""
        manager = AsyncMock(spec=CheckpointManager)
        manager.load.return_value = None

        result = await resume_from_checkpoint(manager, sample_config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is False
        assert result.translated_items == []
        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.metadata == {}

        manager.load.assert_called_once()

    async def test_resume_empty_checkpoint(self, sample_config):
        """Test resume when checkpoint has no translated chunks."""
        manager = AsyncMock(spec=CheckpointManager)
        empty_state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="page",
            completed_pages=[],
            failed_pages=[],
            translated_chunks={},  # Empty chunks
            token_usage=0,
            cost=0.0,
            time_taken=0.0,
        )
        manager.load.return_value = empty_state

        result = await resume_from_checkpoint(manager, sample_config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is False
        assert result.translated_items == []
        assert result.total_tokens == 0
        assert result.total_cost == 0.0
        assert result.metadata == {}

    async def test_resume_successful_page_algorithm(self, sample_config):
        """Test successful resume for page-by-page algorithm."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
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
        manager.load.return_value = state

        config = sample_config.model_copy(update={"algorithm": "page"})
        result = await resume_from_checkpoint(manager, config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        assert result.translated_items == [
            "Translated page 1",
            "Translated page 2",
            "Translated page 3",
        ]
        assert result.total_tokens == 150
        assert result.total_cost == 0.015
        # Metadata should include translated_chunks for page-number mapping
        assert "translated_chunks" in result.metadata
        assert result.metadata["translated_chunks"] == {
            1: "Translated page 1",
            2: "Translated page 2",
            3: "Translated page 3",
        }

        manager.load.assert_called_once()

    async def test_resume_successful_sliding_window_algorithm(self, sample_config):
        """Test successful resume for sliding window algorithm."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="sliding-window",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={
                1: "Translated window 1",
                2: "Translated window 2",
            },
            token_usage=100,
            cost=0.01,
            time_taken=20.0,
        )
        manager.load.return_value = state

        config = sample_config.model_copy(update={"algorithm": "sliding-window"})
        result = await resume_from_checkpoint(manager, config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        assert result.translated_items == ["Translated window 1", "Translated window 2"]
        assert result.total_tokens == 100
        assert result.total_cost == 0.01
        # Metadata should include translated_chunks
        assert "translated_chunks" in result.metadata
        assert result.metadata["translated_chunks"] == {
            1: "Translated window 1",
            2: "Translated window 2",
        }

    async def test_resume_successful_context_aware_algorithm(self, sample_config):
        """Test successful resume for context-aware algorithm."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="context-aware",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={
                1: "Translated chunk 1",
                2: "Translated chunk 2",
            },
            token_usage=200,
            cost=0.02,
            time_taken=40.0,
        )
        manager.load.return_value = state

        # Provide source chunks for context
        source_chunks = ["Source chunk 1", "Source chunk 2", "Source chunk 3"]

        result = await resume_from_checkpoint(manager, sample_config, source_chunks)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        assert result.translated_items == ["Translated chunk 1", "Translated chunk 2"]
        assert result.total_tokens == 200
        assert result.total_cost == 0.02

        # Should have context metadata for continuing translation
        assert "previous_chunk" in result.metadata
        assert "previous_translation" in result.metadata
        assert (
            result.metadata["previous_chunk"] == "Source chunk 2"
        )  # Last completed chunk (index 1)
        assert result.metadata["previous_translation"] == "Translated chunk 2"

    async def test_resume_context_aware_no_source_chunks(self, sample_config):
        """Test context-aware resume when no source chunks are provided."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="context-aware",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={
                1: "Translated chunk 1",
                2: "Translated chunk 2",
            },
            token_usage=200,
            cost=0.02,
            time_taken=40.0,
        )
        manager.load.return_value = state

        result = await resume_from_checkpoint(
            manager, sample_config
        )  # No chunks parameter

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        assert result.translated_items == ["Translated chunk 1", "Translated chunk 2"]
        assert result.total_tokens == 200
        assert result.total_cost == 0.02
        # Still has translated_chunks, but no context (previous_chunk/previous_translation) without source chunks
        assert "translated_chunks" in result.metadata
        assert "previous_chunk" not in result.metadata

    async def test_resume_context_aware_insufficient_chunks(self, sample_config):
        """Test context-aware resume when there are not enough source chunks."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
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
        manager.load.return_value = state

        # Only provide 2 source chunks, but we have 3 translated chunks
        source_chunks = ["Source chunk 1", "Source chunk 2"]

        result = await resume_from_checkpoint(manager, sample_config, source_chunks)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        assert result.translated_items == [
            "Translated chunk 1",
            "Translated chunk 2",
            "Translated chunk 3",
        ]
        assert result.total_tokens == 300
        assert result.total_cost == 0.03
        # Still has translated_chunks, but no context when chunk index is out of range
        assert "translated_chunks" in result.metadata
        assert "previous_chunk" not in result.metadata

    async def test_resume_with_non_sequential_chunk_keys(self, sample_config):
        """Test resume with non-sequential chunk keys in checkpoint."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="page",
            completed_pages=[1, 3, 5],
            failed_pages=[2, 4],
            translated_chunks={
                1: "Translated page 1",
                3: "Translated page 3",
                5: "Translated page 5",
                # Missing 2 and 4
            },
            token_usage=150,
            cost=0.015,
            time_taken=30.0,
        )
        manager.load.return_value = state

        config = sample_config.model_copy(update={"algorithm": "page"})
        result = await resume_from_checkpoint(manager, config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        # Should include chunks with keys that exist in the range 1 to len(chunks)
        assert result.translated_items == [
            "Translated page 1",
            "Translated page 3",
        ]  # "1" and "3" exist, "2" is missing
        assert result.total_tokens == 150
        assert result.total_cost == 0.015

    async def test_resume_with_all_integer_keys(self, sample_config):
        """Test resume works correctly with integer keys (as real checkpoints have)."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
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
        manager.load.return_value = state

        config = sample_config.model_copy(update={"algorithm": "page"})
        result = await resume_from_checkpoint(manager, config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        # Should find all integer keys
        assert result.translated_items == [
            "Translated page 1",
            "Translated page 2",
            "Translated page 3",
        ]
        assert result.total_tokens == 150
        assert result.total_cost == 0.015

    async def test_resume_with_integer_keys_like_real_checkpoint(self, sample_config):
        """Test resume with integer keys as they appear in real checkpoints."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="context-aware",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={
                1: "First chunk content",
                2: "Second chunk content",
                3: "Third chunk content",
                4: "Fourth chunk content",
                5: "Fifth chunk content",
                6: "Sixth chunk content",
                7: "Seventh chunk content",
                8: "Eighth chunk content",
            },
            token_usage=40805,
            cost=0.33133,
            time_taken=592.174002,
        )
        manager.load.return_value = state

        result = await resume_from_checkpoint(manager, sample_config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        # Should find all 8 chunks with integer keys
        assert len(result.translated_items) == 8
        assert result.translated_items[0] == "First chunk content"
        assert result.translated_items[7] == "Eighth chunk content"
        assert result.total_tokens == 40805
        assert result.total_cost == 0.33133

    async def test_checkpoint_save_load_roundtrip_key_consistency(
        self, sample_config, temp_checkpoint_dir
    ):
        """Test complete checkpoint save/load/resume round-trip maintains key consistency."""
        import json

        manager = CheckpointManager(sample_config)

        # Step 1: Create state with integer keys (as algorithms do)
        original_state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="context-aware",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={
                1: "First chunk content",
                2: "Second chunk content",
                3: "Third chunk content",
                4: "Fourth chunk content",
                5: "Fifth chunk content",
            },
            token_usage=1000,
            cost=0.1,
            time_taken=60.0,
        )

        # Step 2: Save the state (integer keys → JSON string keys)
        await manager.save(original_state)

        # Step 3: Verify JSON file has string keys
        checkpoint_path = temp_checkpoint_dir / "test_document_checkpoint.json"
        assert checkpoint_path.exists()

        with open(checkpoint_path) as f:
            json_data = json.load(f)

        # JSON should have string keys
        json_chunks = json_data["translated_chunks"]
        assert list(json_chunks.keys()) == ["1", "2", "3", "4", "5"]
        assert json_chunks["1"] == "First chunk content"
        assert json_chunks["5"] == "Fifth chunk content"

        # Step 4: Load the state back (JSON string keys → integer keys)
        loaded_state = await manager.load()

        assert loaded_state is not None
        # After loading, should have integer keys again
        assert isinstance(loaded_state.translated_chunks, dict)
        assert list(loaded_state.translated_chunks.keys()) == [1, 2, 3, 4, 5]
        assert loaded_state.translated_chunks[1] == "First chunk content"
        assert loaded_state.translated_chunks[5] == "Fifth chunk content"

        # Step 5: Test that resume_from_checkpoint works with the loaded state
        manager_mock = AsyncMock(spec=CheckpointManager)
        manager_mock.load.return_value = loaded_state

        result = await resume_from_checkpoint(manager_mock, sample_config)

        assert result.resumed is True
        assert len(result.translated_items) == 5
        assert result.translated_items == [
            "First chunk content",
            "Second chunk content",
            "Third chunk content",
            "Fourth chunk content",
            "Fifth chunk content",
        ]
        assert result.total_tokens == 1000
        assert result.total_cost == 0.1

        # Step 6: Verify the complete round-trip maintains data integrity
        assert original_state.translated_chunks == loaded_state.translated_chunks
        assert original_state.token_usage == loaded_state.token_usage
        assert original_state.cost == loaded_state.cost

    async def test_resume_all_chunks_completed(self, sample_config):
        """Test resume when all chunks have been translated previously."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
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
        manager.load.return_value = state

        config = sample_config.model_copy(update={"algorithm": "page"})
        result = await resume_from_checkpoint(manager, config)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        assert result.translated_items == [
            "Translated page 1",
            "Translated page 2",
            "Translated page 3",
        ]
        assert result.total_tokens == 150
        assert result.total_cost == 0.015
        assert "translated_chunks" in result.metadata

    async def test_resume_all_chunks_completed_context_aware(self, sample_config):
        """Test context-aware resume when all chunks have been translated previously."""
        manager = AsyncMock(spec=CheckpointManager)
        state = TranslationState(
            source_lang="en",
            target_lang="es",
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
        manager.load.return_value = state

        # Provide exactly 3 source chunks to match the 3 translated chunks
        source_chunks = ["Source chunk 1", "Source chunk 2", "Source chunk 3"]

        result = await resume_from_checkpoint(manager, sample_config, source_chunks)

        assert isinstance(result, ResumeResult)
        assert result.resumed is True
        assert result.translated_items == [
            "Translated chunk 1",
            "Translated chunk 2",
            "Translated chunk 3",
        ]
        assert result.total_tokens == 300
        assert result.total_cost == 0.03

        # Should still have context metadata for the last chunk
        assert "previous_chunk" in result.metadata
        assert "previous_translation" in result.metadata
        assert (
            result.metadata["previous_chunk"] == "Source chunk 3"
        )  # Last completed chunk (index 2)
        assert result.metadata["previous_translation"] == "Translated chunk 3"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_checkpoint_with_empty_chunks(self, sample_config):
        """Test checkpoint with empty translated chunks."""
        state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="context-aware",
            completed_pages=[],
            failed_pages=[],
            translated_chunks={},
            token_usage=0,
            cost=0.0,
            time_taken=0.0,
        )

        manager = CheckpointManager(sample_config)
        await manager.save(state)

        loaded_state = await manager.load()
        assert loaded_state is not None
        assert loaded_state.translated_chunks == {}
        assert loaded_state.token_usage == 0
        assert loaded_state.cost == 0.0

    async def test_checkpoint_with_large_data(self, sample_config):
        """Test checkpoint with large amounts of data."""
        # Create a large translated chunks dictionary
        large_chunks = {i: f"Large translated chunk {i}" * 1000 for i in range(100)}

        state = TranslationState(
            source_lang="en",
            target_lang="es",
            algorithm="context-aware",
            completed_pages=list(range(100)),
            failed_pages=[],
            translated_chunks=large_chunks,
            token_usage=100000,
            cost=10.0,
            time_taken=3600.0,
        )

        manager = CheckpointManager(sample_config)
        await manager.save(state)

        loaded_state = await manager.load()
        assert loaded_state is not None
        assert len(loaded_state.translated_chunks) == 100
        assert loaded_state.token_usage == 100000

    def test_checkpoint_path_with_special_characters(
        self, tmp_path, temp_checkpoint_dir
    ):
        """Test checkpoint path generation with special characters in filename."""
        input_file = tmp_path / "test file with spaces & symbols!.pdf"
        input_file.touch()

        config = TranslationConfig(
            source_lang="en",
            target_lang="es",
            model=ModelType.OPENAI,
            model_name="gpt-4o",
            algorithm="context-aware",
            input_file=input_file,
            checkpoint_dir=temp_checkpoint_dir,
        )

        manager = CheckpointManager(config)
        checkpoint_path = manager._get_checkpoint_path()

        expected_path = (
            temp_checkpoint_dir / "test file with spaces & symbols!_checkpoint.json"
        )
        assert checkpoint_path == expected_path

    async def test_concurrent_checkpoint_operations(
        self, sample_config, sample_translation_state
    ):
        """Test concurrent checkpoint save/load operations."""
        import asyncio

        manager = CheckpointManager(sample_config)

        # Run save and load concurrently
        async def save_task():
            await manager.save(sample_translation_state)

        async def load_task():
            return await manager.load()

        # This should not cause any race conditions or errors
        save_result, load_result = await asyncio.gather(
            save_task(), load_task(), return_exceptions=True
        )

        # Save should succeed
        assert not isinstance(save_result, Exception)

        # Load might return None or the state, both are valid
        assert load_result is None or isinstance(load_result, TranslationState)
