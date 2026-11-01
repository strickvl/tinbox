"""Tests for the pre-flight cost gates and partial-failure exit of `translate`."""

import json
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

import tinbox.cli as cli
from tinbox import __version__
from tinbox.cli import app
from tinbox.core.cost import CostEstimate
from tinbox.core.processor import DocumentContent
from tinbox.core.translation.checkpoint import CheckpointManager
from tinbox.core.translation.interface import TranslationResponse


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def input_file(tmp_path: Path) -> Path:
    path = tmp_path / "input.txt"
    path.write_text("Test content")
    return path


def _estimate(
    cost: float, *, requires_confirmation: bool | None = None
) -> CostEstimate:
    return CostEstimate(
        estimated_tokens=1000,
        estimated_cost=cost,
        estimated_time=10.0,
        warnings=[],
        requires_confirmation=requires_confirmation,
    )


@contextmanager
def _run_patches(
    response: TranslationResponse, estimate: CostEstimate
) -> Generator[None]:
    """Fake everything past the CLI edge so the gates run against real code."""
    content = DocumentContent(pages=["Test content"], content_type="text/plain")
    with ExitStack() as stack:
        stack.enter_context(patch("tinbox.cli.estimate_cost", return_value=estimate))
        stack.enter_context(
            patch(
                "tinbox.cli.load_document", new_callable=AsyncMock, return_value=content
            )
        )
        stack.enter_context(
            patch("tinbox.cli.create_translator", return_value=MagicMock())
        )
        stack.enter_context(
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=response,
            )
        )
        yield


def _printed(mock_console: MagicMock) -> str:
    return " ".join(str(call) for call in mock_console.print.call_args_list)


def test_cheap_cloud_estimate_runs_without_prompt(cli_runner, input_file, tmp_path):
    """A sub-threshold estimate no longer stops at a confirmation prompt."""
    output = tmp_path / "out.txt"
    response = TranslationResponse(text="Hola", tokens_used=10, cost=0.01, time_taken=1)
    with (
        _run_patches(response, _estimate(0.03)),
        patch("tinbox.cli.status_console"),
        patch("typer.confirm") as confirm,
        patch("tinbox.cli._confirmation_available", return_value=False),
    ):
        result = cli_runner.invoke(
            app,
            [
                "translate",
                str(input_file),
                "--to",
                "es",
                "--model",
                "openai:gpt-4o",
                "--output",
                str(output),
            ],
        )

    assert result.exit_code == 0, result.output
    confirm.assert_not_called()
    assert output.read_text() == "Hola"


def test_confirmation_without_terminal_refuses_and_names_force(cli_runner, input_file):
    """Needing confirmation with no TTY fails fast instead of hanging on a prompt."""
    with (
        patch("tinbox.cli.estimate_cost", return_value=_estimate(3.0)),
        patch("tinbox.cli.status_console") as mock_console,
        patch("tinbox.cli._confirmation_available", return_value=False),
        patch("typer.confirm") as confirm,
        patch("tinbox.cli.load_document") as load,
    ):
        result = cli_runner.invoke(
            app,
            ["translate", str(input_file), "--to", "es", "--model", "openai:gpt-4o"],
        )

    assert result.exit_code == 1
    assert "no terminal is attached" in _printed(mock_console)
    assert "--force" in _printed(mock_console)
    confirm.assert_not_called()
    load.assert_not_called()


def test_force_overrides_max_cost_refusal(cli_runner, input_file, tmp_path):
    output = tmp_path / "out.txt"
    response = TranslationResponse(text="Hola", tokens_used=10, cost=0.01, time_taken=1)
    with (
        _run_patches(response, _estimate(3.0)),
        patch("tinbox.cli.status_console"),
        patch("typer.confirm") as confirm,
    ):
        result = cli_runner.invoke(
            app,
            [
                "translate",
                str(input_file),
                "--to",
                "es",
                "--model",
                "openai:gpt-4o",
                "--max-cost",
                "1.0",
                "--force",
                "--output",
                str(output),
            ],
        )

    assert result.exit_code == 0, result.output
    confirm.assert_not_called()
    assert output.read_text() == "Hola"


def test_local_model_estimate_ignores_max_cost(cli_runner, input_file, tmp_path):
    """Local models cost nothing, so a stale estimate must not block them."""
    output = tmp_path / "out.txt"
    response = TranslationResponse(text="Hola", tokens_used=10, cost=0.0, time_taken=1)
    # estimate_cost never asks for confirmation on local models; mirror that here
    with (
        _run_patches(response, _estimate(3.0, requires_confirmation=False)),
        patch("tinbox.cli.status_console"),
        patch("typer.confirm") as confirm,
    ):
        result = cli_runner.invoke(
            app,
            [
                "translate",
                str(input_file),
                "--to",
                "es",
                "--model",
                "ollama:gemma",
                "--max-cost",
                "1.0",
                "--output",
                str(output),
            ],
        )

    assert result.exit_code == 0, result.output
    confirm.assert_not_called()


def test_failed_pages_exit_nonzero_and_keep_checkpoint(
    cli_runner, input_file, tmp_path
):
    """Partial output still gets written, but the run reports failure and keeps state."""
    output = tmp_path / "out.txt"
    checkpoint_dir = tmp_path / "checkpoints"
    response = TranslationResponse(
        text="Page one\n\n[TRANSLATION_FAILED: Page 2]",
        tokens_used=10,
        cost=0.01,
        time_taken=1,
        failed_pages=[2],
        page_errors={2: "boom"},
        warnings=["Translation incomplete: 1 page(s) failed to translate (pages: 2)"],
    )
    manager = MagicMock(spec=CheckpointManager)
    manager.cleanup_old_checkpoints = AsyncMock()
    with (
        _run_patches(response, _estimate(0.03)),
        patch("tinbox.cli.CheckpointManager", return_value=manager),
        patch("tinbox.cli.status_console") as mock_console,
    ):
        result = cli_runner.invoke(
            app,
            [
                "translate",
                str(input_file),
                "--to",
                "es",
                "--model",
                "openai:gpt-4o",
                "--output",
                str(output),
                "--checkpoint-dir",
                str(checkpoint_dir),
            ],
        )

    assert result.exit_code == 1
    assert output.read_text().startswith("Page one")
    manager.cleanup_old_checkpoints.assert_not_called()
    printed = _printed(mock_console)
    assert "1 page(s) failed to translate: 2" in printed
    assert "Checkpoint kept in" in printed
    assert "retry only the failed pages" in printed


def test_failed_pages_without_checkpoint_still_exit_nonzero(
    cli_runner, input_file, tmp_path
):
    output = tmp_path / "out.txt"
    response = TranslationResponse(
        text="[TRANSLATION_FAILED: Page 1]",
        tokens_used=0,
        cost=0.0,
        time_taken=1,
        failed_pages=[1],
        page_errors={1: "boom"},
    )
    with (
        _run_patches(response, _estimate(0.03)),
        patch("tinbox.cli.status_console") as mock_console,
    ):
        result = cli_runner.invoke(
            app,
            [
                "translate",
                str(input_file),
                "--to",
                "es",
                "--model",
                "openai:gpt-4o",
                "--output",
                str(output),
            ],
        )

    assert result.exit_code == 1
    assert output.exists()
    assert "Checkpoint kept" not in _printed(mock_console)


def test_invalid_language_exits_cleanly(cli_runner, input_file):
    """`typer.Exit` must not be swallowed by the catch-all error handler."""
    with patch("tinbox.cli.status_console") as mock_console:
        result = cli_runner.invoke(
            app,
            ["translate", str(input_file), "--to", "xx", "--model", "openai:gpt-4o"],
        )

    assert result.exit_code == 1
    assert "Translation failed" not in _printed(mock_console)


def test_stopped_chunks_exit_nonzero_and_keep_checkpoint(
    cli_runner, input_file, tmp_path
):
    """A context-aware stop writes the paid-for chunks, keeps state and fails."""
    output = tmp_path / "out.txt"
    response = TranslationResponse(
        text="Hola\n\n[TRANSLATION_FAILED: Chunks 3-4 of 4]",
        tokens_used=10,
        cost=0.01,
        time_taken=1,
        failed_chunks=[3, 4],
        warnings=["Translation incomplete: chunks 3-4 of 4 not translated."],
    )
    manager = MagicMock(spec=CheckpointManager)
    manager.cleanup_old_checkpoints = AsyncMock()
    with (
        _run_patches(response, _estimate(0.03)),
        patch("tinbox.cli.CheckpointManager", return_value=manager),
        patch("tinbox.cli.status_console") as mock_console,
    ):
        result = cli_runner.invoke(
            app,
            [
                *("translate", str(input_file), "--to", "es"),
                *("--model", "openai:gpt-4o", "--output", str(output)),
                *("--checkpoint-dir", str(tmp_path / "checkpoints")),
            ],
        )

    assert result.exit_code == 1
    assert output.read_text().startswith("Hola")
    manager.cleanup_old_checkpoints.assert_not_called()
    printed = _printed(mock_console)
    assert "Translation stopped before chunk 3 of 4" in printed
    assert "continue from chunk 3" in printed


@pytest.mark.parametrize(
    "stop",
    [
        {"failed_chunks": [5]},
        {"failed_pages": [2], "page_errors": {2: "Skipped: Translation cost"}},
    ],
)
def test_cost_stop_says_to_raise_max_cost(cli_runner, input_file, tmp_path, stop):
    """Past the cap, the same command stops at once, so the hint names --max-cost."""
    output = tmp_path / "out.json"
    response = TranslationResponse(
        text="Hola", tokens_used=10, cost=0.005, time_taken=1, **stop
    )
    manager = MagicMock(spec=CheckpointManager)
    with (
        _run_patches(response, _estimate(0.001)),
        patch("tinbox.cli.CheckpointManager", return_value=manager),
        patch("tinbox.cli.status_console") as mock_console,
    ):
        result = cli_runner.invoke(
            app,
            [
                *("translate", str(input_file), "--to", "es", "--format", "json"),
                *("--model", "openai:gpt-4o", "--output", str(output)),
                *("--checkpoint-dir", str(tmp_path / "checkpoints")),
                *("--max-cost", "0.0015"),
            ],
        )

    assert result.exit_code == 1
    printed = _printed(mock_console)
    assert "rerun with --max-cost above $0.0050" in printed
    assert "rerun the same command" not in printed
    if "failed_chunks" in stop:
        errors = json.loads(output.read_text())["errors"]
        assert errors == [
            "Translation stopped before chunk 5 of 5; chunk 5 was not translated"
        ]


def test_price_in_requires_price_out(cli_runner, input_file):
    with patch("tinbox.cli.status_console") as mock_console:
        result = cli_runner.invoke(
            app,
            [
                *("translate", str(input_file), "--to", "es"),
                *("--model", "openai:my-proxy-model", "--price-in", "1.5"),
            ],
        )

    assert result.exit_code == 1
    assert "must be given together" in _printed(mock_console)


def test_custom_prices_and_metadata_reach_estimate_config_and_json(
    cli_runner, input_file, tmp_path
):
    output = tmp_path / "out.json"
    response = TranslationResponse(
        text="Hola",
        tokens_used=15,
        cost=0.01,
        time_taken=1,
        input_tokens=10,
        output_tokens=5,
    )
    with (
        _run_patches(response, _estimate(0.03)),
        patch("tinbox.cli.status_console"),
    ):
        result = cli_runner.invoke(
            app,
            [
                *("translate", str(input_file), "--to", "es"),
                *("--model", "openai:my-proxy-model", "--format", "json"),
                *("--price-in", "1.5", "--price-out", "6", "--output", str(output)),
            ],
        )
        estimate_kwargs = cast(MagicMock, cli.estimate_cost).call_args.kwargs
        config = cast(MagicMock, cli.translate_document).call_args.kwargs["config"]

    assert result.exit_code == 0, result.output
    assert estimate_kwargs["model_name"] == "my-proxy-model"
    assert (estimate_kwargs["price_in"], estimate_kwargs["price_out"]) == (1.5, 6.0)
    assert (config.price_in, config.price_out) == (1.5, 6.0)
    metadata = json.loads(output.read_text())["metadata"]
    assert metadata["model"] == "openai"
    assert metadata["model_name"] == "my-proxy-model"
    assert (metadata["input_tokens"], metadata["output_tokens"]) == (10, 5)
    assert metadata["tinbox_version"] == __version__
