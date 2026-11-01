"""Checkpoint management for translation tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinbox.core.types import TranslationConfig
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)

# Bump when sliding-window splits text differently, so old checkpoints are not resumed
SLIDING_WINDOW_SPLIT_VERSION = 2
# Bump when the fields a checkpoint must match change, so old checkpoints are not resumed
CHECKPOINT_FORMAT_VERSION = 2


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
    # Split of token_usage; 0 in checkpoints written before the split was saved
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ResumeResult:
    """Result of attempting to resume from checkpoint."""

    resumed: bool
    translated_items: list[str]
    total_tokens: int
    total_cost: float
    metadata: dict[str, Any]
    # Glossary state carried on resume
    glossary_entries: dict[str, str] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class CheckpointManager:
    """Manages saving and loading translation checkpoints."""

    def __init__(self, config: TranslationConfig) -> None:
        """Initialize the checkpoint manager.

        Args:
            config: Translation configuration
        """
        self.config = config
        self._logger = logger
        self._source_sha256: str | None = None

    def _source_digest(self) -> str:
        """Hash the source file once, so a checkpoint only resumes the same text."""
        if self._source_sha256 is None:
            with open(self.config.input_file, "rb") as f:
                self._source_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
        return self._source_sha256

    def _required_config(self) -> dict[str, Any]:
        """Settings a checkpoint must match before its chunks can be reused.

        Each setting here changes either the source chunks or their translations,
        so resuming across a change would skip or misplace chunks.
        """
        required: dict[str, Any] = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "source_sha256": self._source_digest(),
            "source_lang": self.config.source_lang,
            "target_lang": self.config.target_lang,
            "model": self.config.model.value,
            "model_name": self.config.model_name,
            "algorithm": self.config.algorithm,
            "translation_protocol": self.config.translation_protocol.value,
            "local_model": self.config.local_model,
        }
        if self.config.algorithm == "sliding-window":
            required["sliding_window_split_version"] = SLIDING_WINDOW_SPLIT_VERSION
            required["window_size"] = self.config.window_size
            required["overlap_size"] = self.config.overlap_size
        elif self.config.algorithm == "context-aware":
            required["context_size"] = self.config.context_size
            required["custom_split_token"] = self.config.custom_split_token
        return required

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
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "cost": state.cost,
                "time_taken": state.time_taken,
                "glossary_entries": state.glossary_entries,
                "config": {
                    "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                    "input_path": str(self.config.input_file.resolve()),
                    "source_sha256": self._source_digest(),
                    "source_lang": self.config.source_lang,
                    "target_lang": self.config.target_lang,
                    "model": self.config.model.value,
                    "model_name": self.config.model_name,
                    "algorithm": self.config.algorithm,
                    "translation_protocol": self.config.translation_protocol.value,
                    "local_model": self.config.local_model,
                    "sliding_window_split_version": SLIDING_WINDOW_SPLIT_VERSION,
                    "window_size": self.config.window_size,
                    "overlap_size": self.config.overlap_size,
                    "context_size": self.config.context_size,
                    "custom_split_token": self.config.custom_split_token,
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
            self._logger.error(f"Failed to save checkpoint: {e!s}")
            raise

    async def load(self) -> TranslationState | None:
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

            saved = data.get("config", {})
            if "source_sha256" not in saved:
                self._logger.warning(
                    "Ignoring checkpoint saved by an older tinbox version; "
                    "translating from the beginning",
                    checkpoint_path=str(checkpoint_path),
                )
                return None

            mismatched = {
                key: {"checkpoint": saved.get(key), "current": current}
                for key, current in self._required_config().items()
                if saved.get(key) != current
            }
            if mismatched:
                self._logger.warning(
                    "Ignoring checkpoint that does not match this run; "
                    "translating from the beginning",
                    checkpoint_path=str(checkpoint_path),
                    mismatched=mismatched,
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
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
            )

            self._logger.debug(
                f"Loaded checkpoint from {checkpoint_path}",
                pages=len(state.completed_pages) + len(state.failed_pages),
                tokens=state.token_usage,
                cost=state.cost,
            )

            return state

        except Exception as e:
            self._logger.error(f"Failed to load checkpoint: {e!s}")
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
            self._logger.error(f"Failed to clean up checkpoint: {e!s}")
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


async def load_checkpoint(config: TranslationConfig) -> TranslationState | None:
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
    checkpoint_manager: CheckpointManager | None,
    config: TranslationConfig,
    chunks: list[str] | None = None,
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

    logger.debug(
        "Found valid checkpoint, resuming from saved state", checkpoint=checkpoint
    )
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
        total_input_tokens=checkpoint.input_tokens,
        total_output_tokens=checkpoint.output_tokens,
    )

    logger.info(f"Resumed with {len(translated_items)} completed items")
    return result
