"""Glossary management for translation tasks."""

from __future__ import annotations

import json
from pathlib import Path

from tinbox.core.types import Glossary, GlossaryEntry
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


def _term_key(term: str) -> str:
    """Compare terms ignoring case and surrounding whitespace."""
    return term.strip().casefold()


class GlossaryManager:
    """Manages glossary state and persistence during translation."""

    def __init__(
        self, initial_glossary: Glossary | None = None, *, locked: bool = False
    ) -> None:
        """Start from ``initial_glossary``; with ``locked``, its entries never change.

        Locked entries come from a glossary file the user curated, so the model
        can add new terms but cannot replace those translations.
        """
        self.current_glossary: Glossary = (
            initial_glossary if initial_glossary is not None else Glossary()
        )
        self._locked_entries: dict[str, str] = (
            dict(self.current_glossary.entries) if locked else {}
        )

    def get_current_glossary(self) -> Glossary:
        """Get the current glossary state."""
        return self.current_glossary

    def update_glossary(self, new_entries: list[GlossaryEntry]) -> None:
        """Update the current glossary with new entries."""
        locked = {_term_key(term) for term in self._locked_entries}
        kept = [entry for entry in new_entries if _term_key(entry.term) not in locked]
        if len(kept) < len(new_entries):
            logger.debug(
                "Kept locked glossary entries",
                ignored_terms=[e.term for e in new_entries if e not in kept],
            )
        if not kept:
            return
        try:
            self.current_glossary = self.current_glossary.extend(kept)
            logger.debug(
                "Updated glossary",
                new_terms=len(kept),
                total_terms=len(self.current_glossary.entries),
            )
        except Exception as e:
            logger.error("Failed to update glossary", error=str(e))

    def save_to_file(self, file_path: Path) -> None:
        """Save current glossary to a JSON file."""
        try:
            payload = {"entries": self.current_glossary.entries}
            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            temp_path.rename(file_path)
            logger.debug("Saved glossary to file", path=str(file_path))
        except Exception as e:
            logger.error("Failed to save glossary", error=str(e))
            raise

    @classmethod
    def load_from_file(cls, file_path: Path) -> GlossaryManager:
        """Load a glossary file; its entries are locked against model changes.

        A missing file, an empty file, or ``{}`` gives an empty glossary, so the
        same path can be used with ``--save-glossary`` to start one.

        Raises:
            ValueError: If the file is not a valid glossary.
        """
        if not file_path.exists():
            logger.warning("Glossary file does not exist", path=str(file_path))
            return GlossaryManager()

        try:
            text = file_path.read_text(encoding="utf-8")
            data = json.loads(text) if text.strip() else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Could not read glossary file {file_path}: {e!s}") from e

        entries = data.get("entries") if isinstance(data, dict) else None
        if data == {}:
            # An empty file (e.g. after `touch g.json`) or `{}` is an empty glossary
            entries = {}
        if not isinstance(entries, dict) or not all(
            isinstance(term, str) and isinstance(translation, str)
            for term, translation in entries.items()
        ):
            raise ValueError(
                f"Invalid glossary file {file_path}: expected "
                '{"entries": {"term": "translation", ...}}'
            )

        logger.debug(
            "Loaded glossary from file", path=str(file_path), terms=len(entries)
        )
        return GlossaryManager(initial_glossary=Glossary(entries=entries), locked=True)

    def restore_from_checkpoint(self, glossary_entries: dict[str, str]) -> None:
        """Restore glossary state from checkpoint entries."""
        if glossary_entries:
            # The glossary file may have been edited since the checkpoint was saved
            entries = {**glossary_entries, **self._locked_entries}
            self.current_glossary = Glossary(entries=entries)
            logger.debug(
                "Restored glossary from checkpoint",
                terms=len(glossary_entries),
            )
