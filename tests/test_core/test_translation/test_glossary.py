"""Basic tests for glossary types and manager."""

import json
from pathlib import Path

import pytest

from tinbox.core.translation.glossary import GlossaryManager
from tinbox.core.types import Glossary, GlossaryEntry


def test_glossary_entry_and_extend():
    g = Glossary()
    g2 = g.extend([GlossaryEntry(term="CPU", translation="Processeur")])
    assert g.entries == {}
    assert g2.entries["CPU"] == "Processeur"
    ctx = g2.to_context_string()
    assert "[GLOSSARY]" in ctx and "CPU -> Processeur" in ctx


def test_glossary_manager_update():
    mgr = GlossaryManager()
    assert mgr.get_current_glossary().entries == {}
    mgr.update_glossary([GlossaryEntry(term="GPU", translation="Carte graphique")])
    assert mgr.get_current_glossary().entries["GPU"] == "Carte graphique"


def test_glossary_save_load_roundtrip(tmp_path: Path):
    mgr = GlossaryManager()
    mgr.update_glossary(
        [
            GlossaryEntry(term="RAM", translation="Mémoire vive"),
            GlossaryEntry(term="SSD", translation="Disque SSD"),
        ]
    )
    path = tmp_path / "terms.json"
    mgr.save_to_file(path)

    loaded = GlossaryManager.load_from_file(path)
    cur = loaded.get_current_glossary()
    assert cur.entries["RAM"] == "Mémoire vive"
    assert cur.entries["SSD"] == "Disque SSD"


def test_glossary_manager_restore_from_checkpoint():
    mgr = GlossaryManager()
    assert mgr.get_current_glossary().entries == {}

    # Simulate restoring from checkpoint
    checkpoint_entries = {
        "API": "Interface de programmation",
        "SDK": "Kit de développement",
    }
    mgr.restore_from_checkpoint(checkpoint_entries)

    current = mgr.get_current_glossary()
    assert current.entries["API"] == "Interface de programmation"
    assert current.entries["SDK"] == "Kit de développement"


def _write_glossary(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loaded_entries_are_locked_but_new_terms_are_added(tmp_path: Path):
    path = _write_glossary(tmp_path / "terms.json", {"entries": {"CPU": "Prozessor"}})
    mgr = GlossaryManager.load_from_file(path)

    mgr.update_glossary(
        [
            GlossaryEntry(term="CPU", translation="Hauptprozessor"),
            GlossaryEntry(term="GPU", translation="Grafikprozessor"),
        ]
    )

    assert mgr.get_current_glossary().entries == {
        "CPU": "Prozessor",
        "GPU": "Grafikprozessor",
    }


def test_model_terms_stay_last_write_wins():
    mgr = GlossaryManager()
    mgr.update_glossary([GlossaryEntry(term="GPU", translation="Grafikkarte")])
    mgr.update_glossary([GlossaryEntry(term="GPU", translation="Grafikprozessor")])

    assert mgr.get_current_glossary().entries == {"GPU": "Grafikprozessor"}


def test_restore_from_checkpoint_keeps_the_file_translations(tmp_path: Path):
    """An edit to the glossary file between runs wins over the checkpoint copy."""
    path = _write_glossary(tmp_path / "terms.json", {"entries": {"CPU": "Prozessor"}})
    mgr = GlossaryManager.load_from_file(path)

    mgr.restore_from_checkpoint(
        {"CPU": "Hauptprozessor", " cpu": "Rechenwerk", "GPU": "Grafikprozessor"}
    )

    assert mgr.get_current_glossary().entries == {
        "CPU": "Prozessor",
        "GPU": "Grafikprozessor",
    }


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"CPU": "Prozessor"}),
        json.dumps({"entries": ["CPU", "Prozessor"]}),
        json.dumps({"entries": {"CPU": 1}}),
        json.dumps(["entries"]),
    ],
)
def test_invalid_glossary_file_raises(tmp_path: Path, content: str):
    path = tmp_path / "terms.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="glossary file"):
        GlossaryManager.load_from_file(path)


@pytest.mark.parametrize("content", ["", "  \n", "{}"])
def test_empty_glossary_file_starts_empty(tmp_path: Path, content: str):
    path = tmp_path / "terms.json"
    path.write_text(content, encoding="utf-8")

    assert GlossaryManager.load_from_file(path).get_current_glossary().entries == {}


def test_locked_terms_match_ignoring_case_and_spaces(tmp_path: Path):
    path = _write_glossary(
        tmp_path / "terms.json", {"entries": {"API": "Schnittstelle"}}
    )
    mgr = GlossaryManager.load_from_file(path)

    mgr.update_glossary(
        [
            GlossaryEntry(term="api", translation="Api-y"),
            GlossaryEntry(term="API ", translation="z"),
        ]
    )

    assert mgr.get_current_glossary().entries == {"API": "Schnittstelle"}


def test_missing_glossary_file_starts_empty(tmp_path: Path):
    """Lets `--glossary-file g.json --save-glossary g.json` start a new glossary."""
    mgr = GlossaryManager.load_from_file(tmp_path / "new.json")

    assert mgr.get_current_glossary().entries == {}
