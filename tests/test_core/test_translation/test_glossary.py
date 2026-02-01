"""Basic tests for glossary types and manager."""

from pathlib import Path

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
