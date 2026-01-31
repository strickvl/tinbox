"""Glossary management for translation tasks."""

import json
from pathlib import Path
from typing import List, Optional

from tinbox.core.types import Glossary, GlossaryEntry
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class GlossaryManager:
    """Manages glossary state and persistence during translation."""

    def __init__(self, initial_glossary: Optional[Glossary] = None) -> None:
        self.current_glossary: Glossary = initial_glossary if initial_glossary is not None else Glossary()

    def get_current_glossary(self) -> Glossary:
        """Get the current glossary state."""
        return self.current_glossary

    def update_glossary(self, new_entries: List[GlossaryEntry]) -> None:
        """Update the current glossary with new entries."""
        if not new_entries:
            return
        try:
            self.current_glossary = self.current_glossary.extend(new_entries)
            logger.debug(
                "Updated glossary",
                new_terms=len(new_entries),
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
    def load_from_file(cls, file_path: Path) -> "GlossaryManager":
        """Load glossary from a JSON file."""
        try:
            if not file_path.exists():
                logger.warning("Glossary file does not exist", path=str(file_path))
                return GlossaryManager()

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entries_dict = data.get("entries", {})
            if not isinstance(entries_dict, dict):
                raise ValueError("Invalid glossary file format: 'entries' must be an object mapping terms to translations")

            glossary = Glossary(entries=entries_dict)
            manager = GlossaryManager(initial_glossary=glossary)
            logger.debug(
                "Loaded glossary from file",
                path=str(file_path),
                terms=len(entries_dict),
            )
            return manager
        except Exception as e:
            logger.error("Failed to load glossary", error=str(e))
            # Return empty manager for robustness; caller may proceed without glossary
            return GlossaryManager()

    def restore_from_checkpoint(self, glossary_entries: dict[str, str]) -> None:
        """Restore glossary state from checkpoint entries."""
        if glossary_entries:
            self.current_glossary = Glossary(entries=glossary_entries)
            logger.debug(
                "Restored glossary from checkpoint",
                terms=len(glossary_entries),
            )


