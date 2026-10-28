"""CLI tests for plain-text input formats and clean stdout output."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tinbox.cli import app
from tinbox.core.translation.interface import TranslationResponse

MARKDOWN = "# Notes\n\nThe library opens at **nine**.\n"


def run_translate(source: Path, *extra: str):
    with (
        patch("tinbox.cli.create_translator"),
        patch("tinbox.cli.translate_document") as translate,
    ):
        translate.return_value = TranslationResponse(
            text="# Notizen\n\nDie Bibliothek öffnet um **neun**.\n",
            tokens_used=20,
            cost=0.01,
            time_taken=1.0,
        )
        args = ["translate", str(source), "--to", "de", "--model", "openai:gpt-4o"]
        result = CliRunner().invoke(app, [*args, "--force", *extra])
    return result, translate


@pytest.mark.parametrize("name", ["notes.md", "notes.markdown", "NOTES.MD", "a.rst"])
def test_translate_reads_plain_text_extensions(tmp_path: Path, name: str):
    source = tmp_path / name
    source.write_text(MARKDOWN)
    output = tmp_path / "out.md"

    result, translate = run_translate(source, "--output", str(output))

    assert result.exit_code == 0, result.output
    assert translate.call_args.kwargs["content"].pages == [MARKDOWN]
    assert output.read_text() == "# Notizen\n\nDie Bibliothek öffnet um **neun**.\n"


def test_translate_rejects_unsupported_extension(tmp_path: Path):
    source = tmp_path / "table.csv"
    source.write_text("a,b\n")

    result, translate = run_translate(source)

    assert result.exit_code == 1
    assert "Unsupported file type '.csv'" in result.stderr
    translate.assert_not_called()


def test_translate_stdout_carries_only_translation(tmp_path: Path):
    source = tmp_path / "notes.md"
    source.write_text(MARKDOWN)

    result, _ = run_translate(source)

    assert result.exit_code == 0, result.output
    assert result.stdout == "# Notizen\n\nDie Bibliothek öffnet um **neun**.\n\n"
    assert "Cost Estimate" in result.stderr
    assert "Translation Statistics" in result.stderr
