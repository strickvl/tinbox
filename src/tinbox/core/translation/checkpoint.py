"""Checkpoint management for translation tasks."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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

            self._logger.info(
                f"Saved checkpoint to {checkpoint_path}",
                pages=len(state.completed_pages) + len(state.failed_pages),
                tokens=state.token_usage,
                cost=state.cost,
            )

        except Exception as e:
            self._logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    def load(self) -> Optional[TranslationState]:
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

            state = TranslationState(
                source_lang=data["source_lang"],
                target_lang=data["target_lang"],
                algorithm=data["algorithm"],
                completed_pages=data["completed_pages"],
                failed_pages=data["failed_pages"],
                translated_chunks=data["translated_chunks"],
                token_usage=data["token_usage"],
                cost=data["cost"],
                time_taken=data["time_taken"],
            )

            self._logger.info(
                f"Loaded checkpoint from {checkpoint_path}",
                pages=len(state.completed_pages) + len(state.failed_pages),
                tokens=state.token_usage,
                cost=state.cost,
            )

            return state

        except Exception as e:
            self._logger.error(f"Failed to load checkpoint: {str(e)}")
            return None


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


def load_checkpoint(config: TranslationConfig) -> Optional[TranslationState]:
    """Load translation state from checkpoint.

    Args:
        config: Translation configuration

    Returns:
        Loaded translation state, or None if no valid checkpoint exists
    """
    if not should_resume(config):
        return None

    manager = CheckpointManager(config)
    return manager.load()


async def save_checkpoint(
    config: TranslationConfig,
    pages: list[str],
    tokens_used: int,
    cost: float,
) -> None:
    """Save translation state to checkpoint.

    Args:
        config: Translation configuration
        pages: Translated pages
        tokens_used: Total tokens used
        cost: Total cost
    """
    if not config.checkpoint_dir:
        return

    state = TranslationState(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        algorithm=config.algorithm,
        completed_pages=[],
        failed_pages=[],
        translated_chunks={},
        token_usage=tokens_used,
        cost=cost,
        time_taken=0.0,
    )

    manager = CheckpointManager(config)
    await manager.save(state)
