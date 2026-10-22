"""Exercise pair export and offline reporting through the CLI."""

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import structlog
from typer.testing import CliRunner

from tinbox.cli import app
from tinbox.core.review import load_review_bundle
from tinbox.core.translation.interface import TranslationResponse
from tinbox.core.types import GlossaryEntry


@pytest.fixture
def translation_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "tinbox.core.translation.algorithms.logger", structlog.get_logger()
    )
    source = tmp_path / "story.txt"
    source.write_text("Only Elara knew.\n\n|Some birds returned.\n", encoding="utf-8")
    translations = ["Nur Elara wusste es.\n\n", "Einige Vögel kehrten zurück.\n"]
    translator = Mock()
    translator.translate = AsyncMock(
        side_effect=[
            TranslationResponse(text=text, tokens_used=10, cost=0.001, time_taken=0.1)
            for text in translations
        ]
    )
    factory = Mock(return_value=translator)
    monkeypatch.setattr("tinbox.cli.create_translator", factory)
    output = tmp_path / "translation.txt"
    bundle = tmp_path / "translation.review.json"
    args = [
        "translate",
        str(source),
        "--from",
        "en",
        "--to",
        "de",
        "--model",
        "openai:gpt-4o",
        "--output",
        str(output),
        "--split-token",
        "|",
        "--force",
    ]
    return source, output, bundle, args, translations, factory


def test_translate_then_review_offline(translation_run, monkeypatch):
    source, output, bundle_path, args, translations, factory = translation_run
    result = CliRunner().invoke(app, [*args, "--save-review-input", str(bundle_path)])
    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8") == "".join(translations)
    bundle = load_review_bundle(bundle_path)
    assert [pair.source for pair in bundle.pairs] == [
        call.args[0].content for call in factory.return_value.translate.await_args_list
    ]
    assert [pair.translation for pair in bundle.pairs] == translations
    assert bundle.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert bundle.source_name == "story.txt"
    assert bundle.source_lang == "en"
    assert bundle.target_lang == "de"
    assert bundle.provider == "openai"
    assert bundle.model_name == "gpt-4o"
    assert bundle.custom_split_token == "|"

    # Reporting needs neither the original files, provider SDKs nor credentials.
    source.unlink()
    output.unlink()
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    factory.side_effect = AssertionError("Report attempted to create a translator")
    report = bundle_path.with_suffix(".html")
    result = CliRunner().invoke(app, ["review", str(bundle_path), "-o", str(report)])
    assert result.exit_code == 0, result.output
    html = report.read_text(encoding="utf-8")
    assert "Only Elara knew." in html
    assert "Einige Vögel" in html
    assert "No automated quality assessment" in result.output


@pytest.mark.parametrize("with_review", [False, True])
def test_translation_json_format_is_unchanged(translation_run, with_review):
    _, output, bundle_path, args, translations, _ = translation_run
    args += ["--format", "json"]
    if with_review:
        args += ["--save-review-input", str(bundle_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["result"]["text"] == "".join(translations)
    assert set(data["result"]) == {
        "text",
        "tokens_used",
        "cost",
        "time_taken",
        "failed_pages",
        "page_errors",
        "warnings",
    }
    assert bundle_path.exists() is with_review


def test_review_export_dry_run_creates_nothing(translation_run):
    _, output, bundle_path, args, _, factory = translation_run
    result = CliRunner().invoke(
        app, [*args, "--save-review-input", str(bundle_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    factory.assert_not_called()
    assert not output.exists()
    assert not bundle_path.exists()


@pytest.mark.parametrize("algorithm", ["page", "sliding-window"])
def test_review_rejects_unsupported_algorithm_before_cost_or_translation(
    translation_run, monkeypatch, algorithm
):
    _, output, bundle_path, args, _, factory = translation_run
    estimate = Mock(side_effect=AssertionError("Cost estimation should not run"))
    monkeypatch.setattr("tinbox.cli.estimate_cost", estimate)
    result = CliRunner().invoke(
        app,
        [*args, "--save-review-input", str(bundle_path), "--algorithm", algorithm],
    )
    assert result.exit_code == 1
    assert "context-aware" in result.output
    factory.assert_not_called()
    estimate.assert_not_called()
    assert not output.exists()
    assert not bundle_path.exists()


@pytest.mark.parametrize(
    "destination", ["source", "translation", "existing", "missing_parent"]
)
def test_bundle_path_is_checked_before_translation(translation_run, destination):
    source, output, bundle_path, args, _, factory = translation_run
    if destination == "source":
        bundle_path = source
    elif destination == "translation":
        bundle_path = output
    elif destination == "existing":
        bundle_path.write_text("keep this", encoding="utf-8")
    else:
        bundle_path = bundle_path.parent / "missing" / "review.json"
    original_source = source.read_bytes()
    result = CliRunner().invoke(app, [*args, "--save-review-input", str(bundle_path)])
    assert result.exit_code == 1
    factory.assert_not_called()
    assert source.read_bytes() == original_source
    assert not output.exists()
    if destination == "existing":
        assert bundle_path.read_text(encoding="utf-8") == "keep this"


def test_bundle_write_failure_preserves_translation(translation_run, monkeypatch):
    _, output, bundle_path, args, translations, factory = translation_run
    glossary = output.with_name("glossary.json")
    monkeypatch.setattr(
        "tinbox.core.translation.glossary.logger", structlog.get_logger()
    )
    factory.return_value.translate.side_effect = [
        TranslationResponse(
            text=text,
            tokens_used=10,
            cost=0.001,
            time_taken=0.1,
            glossary_updates=[GlossaryEntry(term="birds", translation="Vögel")],
        )
        for text in translations
    ]
    monkeypatch.setattr(
        "tinbox.cli.save_review_bundle", Mock(side_effect=OSError("Disk full"))
    )
    result = CliRunner().invoke(
        app,
        [
            *args,
            "--save-review-input",
            str(bundle_path),
            "--glossary",
            "--save-glossary",
            str(glossary),
        ],
    )
    assert result.exit_code == 1
    assert "Review bundle could not be saved" in result.output
    assert "already been written" in result.output
    assert output.read_text(encoding="utf-8") == "".join(translations)
    assert json.loads(glossary.read_text(encoding="utf-8")) == {
        "entries": {"birds": "Vögel"}
    }
    assert not bundle_path.exists()


def test_bundle_cannot_use_checkpoint_path(translation_run):
    source, output, _, args, _, factory = translation_run
    checkpoint_dir = source.parent / "checkpoints"
    checkpoint_dir.mkdir()
    bundle_path = checkpoint_dir / f"{source.stem}_checkpoint.json"
    result = CliRunner().invoke(
        app,
        [
            *args,
            "--save-review-input",
            str(bundle_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-frequency",
            "99",
        ],
    )
    assert result.exit_code == 1
    assert "separate path" in result.output
    factory.assert_not_called()
    assert not output.exists()
    assert not bundle_path.exists()


@pytest.mark.parametrize("collision", ["source", "hardlink", "glossary"])
def test_review_export_preserves_source_and_translation_paths(
    translation_run, collision
):
    source, output, bundle_path, args, _, factory = translation_run
    original_source = source.read_bytes()
    if collision == "source":
        args[args.index("--output") + 1] = str(source)
    elif collision == "hardlink":
        output.hardlink_to(source)
    else:
        args += ["--glossary", "--save-glossary", str(output)]
    result = CliRunner().invoke(app, [*args, "--save-review-input", str(bundle_path)])
    assert result.exit_code == 1
    assert "requires separate" in result.output
    factory.assert_not_called()
    assert source.read_bytes() == original_source
    assert not bundle_path.exists()


@pytest.mark.parametrize("contents", ["{", '{"schema_version": 99}', "[]"])
def test_review_rejects_invalid_bundle_without_report(tmp_path, contents):
    bundle = tmp_path / "review.json"
    bundle.write_text(contents, encoding="utf-8")
    report = tmp_path / "report.html"
    result = CliRunner().invoke(app, ["review", str(bundle), "-o", str(report)])
    assert result.exit_code == 1
    assert "Could not create review report" in result.output
    assert not report.exists()


def test_review_cannot_overwrite_bundle(translation_run):
    _, _, bundle_path, args, _, _ = translation_run
    result = CliRunner().invoke(app, [*args, "--save-review-input", str(bundle_path)])
    assert result.exit_code == 0, result.output
    original = bundle_path.read_bytes()
    result = CliRunner().invoke(
        app, ["review", str(bundle_path), "-o", str(bundle_path)]
    )
    assert result.exit_code == 1
    assert bundle_path.read_bytes() == original
