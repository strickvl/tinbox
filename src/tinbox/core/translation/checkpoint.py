"""Checkpoint management for translation tasks."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from tinbox.core.types import TranslationConfig
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranslationState:
    """State of a translation task for checkpointing."""

    source_lang: str
    target_lang: str
    algorithm: str
    completed_pages: list[int]
    failed_pages: list[int]
    translated_chunks: dict[int, str]
    token_usage: int
    cost: float
    time_taken: float
    # Glossary state: mapping term -> translation
    glossary_entries: dict[str, str] = field(default_factory=dict)


@dataclass
class ResumeResult:
    """Result of attempting to resume from checkpoint."""

    resumed: bool
    translated_items: List[str]
    total_tokens: int
    total_cost: float
    metadata: Dict[str, Any]
    # Glossary state carried on resume
    glossary_entries: Dict[str, str] = field(default_factory=dict)


class CheckpointManager:
    """Manages saving and loading translation checkpoints."""

    def __init__(self, config: TranslationConfig) -> None:
        """Initialize the checkpoint manager.

        Args:
            config: Translation configuration
        """
        self.config = config
        self._logger = logger

    def _get_checkpoint_path(self) -> Path:
        """Get the path for the checkpoint file.

        Returns:
            Path to the checkpoint file
        """
        if not self.config.checkpoint_dir:
            raise ValueError("No checkpoint directory configured")

        # Create checkpoint directory if it doesn't exist
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Use input filename as base for checkpoint
        checkpoint_name = f"{self.config.input_file.stem}_checkpoint.json"
        return self.config.checkpoint_dir / checkpoint_name

    async def save(self, state: TranslationState) -> None:
        """Save translation state to a checkpoint file.

        Args:
            state: Current translation state
        """
        try:
            checkpoint_path = self._get_checkpoint_path()
            checkpoint_data = {
                "source_lang": state.source_lang,
                "target_lang": state.target_lang,
                "algorithm": state.algorithm,
                "completed_pages": state.completed_pages,
                "failed_pages": state.failed_pages,
                "translated_chunks": state.translated_chunks,
                "token_usage": state.token_usage,
                "cost": state.cost,
                "time_taken": state.time_taken,
                "glossary_entries": state.glossary_entries,
                "config": {
                    "source_lang": self.config.source_lang,
                    "target_lang": self.config.target_lang,
                    "model": self.config.model.value,
                    "algorithm": self.config.algorithm,
                },
            }

            # Write checkpoint atomically
            temp_path = checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.rename(checkpoint_path)

            self._logger.debug(
                f"Saved checkpoint to {checkpoint_path}",
                pages=len(state.completed_pages) + len(state.failed_pages),
                tokens=state.token_usage,
                cost=state.cost,
            )

        except Exception as e:
            self._logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    async def load(self) -> Optional[TranslationState]:
        """Load translation state from a checkpoint file.

        Returns:
            Loaded translation state, or None if no valid checkpoint exists
        """
        try:
            checkpoint_path = self._get_checkpoint_path()
            if not checkpoint_path.exists():
                return None

            with open(checkpoint_path) as f:
                data = json.load(f)

            # Validate checkpoint matches current config
            config = data.get("config", {})
            if (
                config.get("source_lang") != self.config.source_lang
                or config.get("target_lang") != self.config.target_lang
                or config.get("model") != self.config.model.value
                or config.get("algorithm") != self.config.algorithm
            ):
                self._logger.warning(
                    "Checkpoint configuration mismatch",
                    checkpoint_config=config,
                    current_config={
                        "source_lang": self.config.source_lang,
                        "target_lang": self.config.target_lang,
                        "model": self.config.model.value,
                        "algorithm": self.config.algorithm,
                    },
                )
                return None

            # Convert string keys back to integers for translated_chunks
            translated_chunks = {}
            for key, value in data["translated_chunks"].items():
                try:
                    # Try to convert key to int, keep as string if it fails
                    int_key = int(key)
                    translated_chunks[int_key] = value
                except (ValueError, TypeError):
                    translated_chunks[key] = value

            glossary_entries = data.get("glossary_entries", {})
            state = TranslationState(
                source_lang=data["source_lang"],
                target_lang=data["target_lang"],
                algorithm=data["algorithm"],
                completed_pages=data["completed_pages"],
                failed_pages=data["failed_pages"],
                translated_chunks=translated_chunks,
                token_usage=data["token_usage"],
                cost=data["cost"],
                time_taken=data["time_taken"],
                glossary_entries=glossary_entries,
            )

            self._logger.debug(
                f"Loaded checkpoint from {checkpoint_path}",
                pages=len(state.completed_pages) + len(state.failed_pages),
                tokens=state.token_usage,
                cost=state.cost,
            )

            return state

        except Exception as e:
            self._logger.error(f"Failed to load checkpoint: {str(e)}")
            return None


    async def cleanup_old_checkpoints(self, input_file: Path) -> None:
        """Clean up old checkpoint files for the given input file.

        Args:
            input_file: The input file to clean up checkpoints for
        """
        try:
            checkpoint_path = self._get_checkpoint_path()
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                self._logger.debug(f"Cleaned up checkpoint file: {checkpoint_path}")
        except Exception as e:
            self._logger.error(f"Failed to clean up checkpoint: {str(e)}")
            # Don't raise - cleanup failure shouldn't stop the translation


def should_resume(config: TranslationConfig) -> bool:
    """Check if translation should resume from checkpoint.

    Args:
        config: Translation configuration

    Returns:
        True if should resume from checkpoint
    """
    return bool(
        config.checkpoint_dir
        and config.resume_from_checkpoint
        and config.checkpoint_dir.exists()
    )


async def load_checkpoint(config: TranslationConfig) -> Optional[TranslationState]:
    """Load translation state from checkpoint.

    Args:
        config: Translation configuration

    Returns:
        Loaded translation state, or None if no valid checkpoint exists
    """
    if not should_resume(config):
        return None

    manager = CheckpointManager(config)
    return await manager.load()


async def resume_from_checkpoint(
    checkpoint_manager: Optional[CheckpointManager],
    config: TranslationConfig,
    chunks: Optional[List[str]] = None,
) -> ResumeResult:
    """Attempt to resume translation from checkpoint.
    
    Args:
        checkpoint_manager: The checkpoint manager instance
        config: Translation configuration
        chunks: Optional list of source chunks for context-aware algorithm
        
    Returns:
        ResumeResult with resume status and loaded data
    """
    if not checkpoint_manager or not config.resume_from_checkpoint:
        return ResumeResult(
            resumed=False,
            translated_items=[],
            total_tokens=0,
            total_cost=0.0,
            metadata={},
            glossary_entries={},
        )
        
    logger.info("Checking for checkpoint")
    checkpoint = await checkpoint_manager.load()
    
    if not checkpoint or not checkpoint.translated_chunks:
        return ResumeResult(
            resumed=False,
            translated_items=[],
            total_tokens=0,
            total_cost=0.0,
            metadata={},
            glossary_entries={},
        )
        
    logger.debug("Found valid checkpoint, resuming from saved state", checkpoint=checkpoint)
    logger.info("Found valid checkpoint, resuming from saved state")
    
    # Load existing translated items in order
    # Checkpoint loading always converts string keys to integer keys
    translated_items = [
        checkpoint.translated_chunks[i]
        for i in range(1, len(checkpoint.translated_chunks) + 1)
        if i in checkpoint.translated_chunks
    ]
    
    # Prepare algorithm-specific metadata
    metadata = {
        # Include translated_chunks for algorithms that need page-number mapping
        "translated_chunks": checkpoint.translated_chunks,
    }

    # For context-aware algorithm, set up context from the last completed chunk
    if config.algorithm == "context-aware" and chunks and translated_items:
        chunk_index = len(translated_items) - 1
        if chunk_index < len(chunks):
            metadata["previous_chunk"] = chunks[chunk_index]
            metadata["previous_translation"] = translated_items[-1]

    result = ResumeResult(
        resumed=True,
        translated_items=translated_items,
        total_tokens=checkpoint.token_usage,
        total_cost=checkpoint.cost,
        metadata=metadata,
        glossary_entries=getattr(checkpoint, "glossary_entries", {}),
    )
    
    logger.info(f"Resumed with {len(translated_items)} completed items")
    return result


