"""Tests for `translate --pages`, the TINBOX_* defaults and the endpoint rows."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pypdf import PdfWriter
from typer.testing import CliRunner

from tinbox.cli import (
    app,
    format_page_selection,
    parse_model_spec,
    parse_page_selection,
)
from tinbox.core.cost import estimate_cost
from tinbox.core.doctor import check_base_url
from tinbox.core.processor import DocumentContent
from tinbox.core.translation import provider_endpoint
from tinbox.core.translation.interface import TranslationResponse
from tinbox.core.types import ModelType


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def five_page_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "five.pdf"
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


@pytest.mark.parametrize(
    ("spec", "pages"),
    [("3", [3]), ("1-3,7", [1, 2, 3, 7]), (" 5-6 , 2 ", [2, 5, 6]), ("2,2-3", [2, 3])],
)
def test_parse_page_selection(spec: str, pages: list[int]):
    assert parse_page_selection(spec, page_count=10) == pages


@pytest.mark.parametrize("spec", ["0", "3-1", "x", "1,,2", "-2", "1-", "2-x"])
def test_parse_page_selection_rejects_bad_values(spec: str):
    with pytest.raises(ValueError, match="--pages"):
        parse_page_selection(spec, page_count=10)


def test_parse_page_selection_checks_the_page_count_before_expanding():
    """A huge range fails at once instead of building a set of a billion pages."""
    with pytest.raises(ValueError, match="page 999999999, but the PDF has 7"):
        parse_page_selection("1-999999999", page_count=7)


def test_format_page_selection():
    assert format_page_selection([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"


def _run(cli_runner: CliRunner, args: list[str], env: dict[str, str] | None = None):
    return cli_runner.invoke(app, ["translate", *args], env=env)


def test_pages_loads_and_prices_only_the_selection(cli_runner, five_page_pdf, tmp_path):
    content = DocumentContent(
        pages=[b"p2", b"p4"], content_type="image/png", page_numbers=[2, 4]
    )
    response = TranslationResponse(text="ok", tokens_used=1, cost=0.0, time_taken=0)
    with (
        patch("tinbox.cli.load_document", new_callable=AsyncMock) as load,
        patch("tinbox.cli.translate_document", new_callable=AsyncMock) as translate,
        patch("tinbox.cli.create_translator"),
        patch("tinbox.cli.estimate_cost", wraps=estimate_cost) as estimate,
        patch("tinbox.cli.status_console"),
    ):
        load.return_value = content
        translate.return_value = response
        result = _run(
            cli_runner,
            [
                str(five_page_pdf),
                "--model",
                "ollama:gemma4",
                "--pages",
                "2,4",
                "--format",
                "json",
                "--output",
                str(tmp_path / "out.txt"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert load.await_args is not None
    assert load.await_args.kwargs["pages"] == [2, 4]
    assert translate.await_args is not None
    assert translate.await_args.kwargs["config"].pages == [2, 4]
    assert json.loads((tmp_path / "out.txt").read_text())["metadata"]["pages"] == [2, 4]
    assert estimate.call_args.kwargs["page_count"] == 2
    priced = estimate_cost(*estimate.call_args.args, **estimate.call_args.kwargs)
    assert priced.estimated_tokens == 2030  # 2 pages x 500, in and out


@pytest.mark.parametrize(
    ("spec", "message"),
    [("6", "has 5 page(s)"), ("2-1", "ranges must go up")],
)
def test_pages_errors_stop_before_the_estimate(
    cli_runner, five_page_pdf, spec, message
):
    with (
        patch("tinbox.cli.estimate_cost") as estimate,
        patch("tinbox.cli.status_console") as console,
    ):
        result = _run(
            cli_runner,
            [str(five_page_pdf), "--model", "openai:gpt-5.5", "--pages", spec],
        )

    assert result.exit_code == 1
    assert message in " ".join(str(c) for c in console.print.call_args_list)
    estimate.assert_not_called()


def test_pages_is_refused_for_text_files(cli_runner, tmp_path):
    text = tmp_path / "doc.txt"
    text.write_text("Hallo")
    with patch("tinbox.cli.status_console") as console:
        result = _run(
            cli_runner, [str(text), "--model", "openai:gpt-5.5", "--pages", "1"]
        )

    assert result.exit_code == 1
    assert "only with PDF files" in str(console.print.call_args_list)


def test_env_vars_supply_model_and_languages(cli_runner, tmp_path):
    text = tmp_path / "doc.txt"
    text.write_text("Hallo Welt")
    env = {"TINBOX_MODEL": "ollama:gemma4", "TINBOX_FROM": "de", "TINBOX_TO": "fr"}
    with patch("tinbox.cli.status_console"), patch("tinbox.cli.display_plan") as plan:
        result = _run(cli_runner, [str(text), "--dry-run"], env=env)

    assert result.exit_code == 0, result.output
    rows = dict(plan.call_args.args[0])
    assert rows["Model"] == "ollama:gemma4"
    assert rows["Languages"] == "de to fr"


def test_flags_override_env_vars(cli_runner, tmp_path):
    text = tmp_path / "doc.txt"
    text.write_text("Hallo Welt")
    env = {"TINBOX_MODEL": "ollama:gemma4", "TINBOX_TO": "fr"}
    with patch("tinbox.cli.status_console"), patch("tinbox.cli.display_plan") as plan:
        result = _run(
            cli_runner,
            [str(text), "--dry-run", "--model", "ollama:qwen3", "--to", "es"],
            env=env,
        )

    assert result.exit_code == 0, result.output
    rows = dict(plan.call_args.args[0])
    assert rows["Model"] == "ollama:qwen3"
    assert rows["Languages"] == "auto to es"


def test_provider_endpoint_reads_sdk_base_urls(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert provider_endpoint(ModelType.OPENAI) == "https://api.openai.com/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    assert provider_endpoint(ModelType.OPENAI) == "http://localhost:8000/v1"
    # Gemini and Ollama use fixed endpoints, whatever OPENAI_BASE_URL says
    assert provider_endpoint(ModelType.OLLAMA) == "http://localhost:11434/v1"
    assert "generativelanguage" in provider_endpoint(ModelType.GEMINI)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy.test")
    assert provider_endpoint(ModelType.ANTHROPIC) == "http://proxy.test"


def test_doctor_reports_a_base_url_override(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    check = check_base_url("OPENAI_BASE_URL", "OpenAI", "https://api.openai.com/v1")
    assert check.details == "Set: OpenAI requests go to http://localhost:8000/v1"

    monkeypatch.delenv("OPENAI_BASE_URL")
    check = check_base_url("OPENAI_BASE_URL", "OpenAI", "https://api.openai.com/v1")
    assert check.details == "Not set: OpenAI requests go to https://api.openai.com/v1"


def test_model_spec_drops_stray_whitespace_and_invisible_characters():
    assert parse_model_spec(" openai:gpt-5.5\u200b\r\n") == (
        ModelType.OPENAI,
        "gpt-5.5",
    )


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_base_url_is_shown_as_broken(monkeypatch, value: str):
    """The SDKs use an empty variable as-is, so every request fails to connect."""
    monkeypatch.setenv("OPENAI_BASE_URL", value)

    assert "set but empty" in provider_endpoint(ModelType.OPENAI)
    check = check_base_url("OPENAI_BASE_URL", "OpenAI", "https://api.openai.com/v1")
    assert not check.ok
    assert check.details == "Set but empty: OpenAI requests will fail to connect"
