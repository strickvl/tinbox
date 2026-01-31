"""Tests for the CLI interface."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tinbox.cli import app
from tinbox.core.cost import CostEstimate
from tinbox.core.processor import DocumentContent
from tinbox.core.types import FileType, ModelType, TranslationResult, TranslationConfig
from tinbox.core.translation.interface import ModelInterface, TranslationResponse
from tinbox.core.translation.checkpoint import CheckpointManager


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_console():
    """Create a mock console for testing."""
    return MagicMock(spec=Console)


@pytest.fixture
def mock_cost_estimate():
    """Create a mock cost estimate."""
    return CostEstimate(
        estimated_tokens=1000,
        estimated_cost=0.03,
        estimated_time=60.0,
        warnings=[],
    )


@pytest.fixture
def mock_translation_result():
    """Create a mock translation result."""
    return TranslationResult(
        text="Translated text",
        tokens_used=1000,
        cost=0.03,
        time_taken=30.0,
    )


@pytest.fixture
def mock_document_content():
    """Create a mock document content."""
    return DocumentContent(
        pages=["Test content"],
        content_type="text/plain",
        metadata={"test": "metadata"},
    )


@pytest.fixture
def mock_model_interface():
    """Create a mock model interface."""
    interface = AsyncMock(spec=ModelInterface)
    interface.translate = AsyncMock(
        return_value=TranslationResult(
            text="Translated text",
            tokens_used=1000,
            cost=0.03,
            time_taken=30.0,
        )
    )
    return interface


def test_version(cli_runner):
    """Test version command."""
    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Tinbox version:" in result.stdout


def test_translate_basic(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
):
    """Test basic translation command."""
    # Create a temporary test file
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=mock_translation_result,
            ),
            patch(
                "tinbox.cli.load_document",
                new_callable=AsyncMock,
                return_value=mock_document_content,
            ),
            patch("tinbox.cli.create_translator", return_value=mock_model_interface),
            patch("tinbox.cli.console"),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--model",
                    "anthropic:claude-3-sonnet",
                ],
            )
            assert result.exit_code == 0

    finally:
        # Clean up test file
        test_file.unlink()


def test_translate_dry_run(cli_runner, mock_cost_estimate):
    """Test dry run mode."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch("tinbox.cli.console"),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--model",
                    "openai:gpt-4o",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0

    finally:
        test_file.unlink()




def test_translate_with_warnings(cli_runner):
    """Test translation with cost warnings."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    # Create cost estimate with warnings
    estimate = CostEstimate(
        estimated_tokens=100_000,
        estimated_cost=3.0,
        estimated_time=600.0,
        warnings=["Large document detected (100,000 tokens)"],
    )

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=estimate),
            patch("tinbox.cli.console") as mock_console,
            patch("typer.confirm", return_value=False),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--model",
                    "openai:gpt-4o",
                ],
            )
            assert result.exit_code == 1
            # Check if the cancellation message was printed
            mock_console.print.assert_any_call("\nTranslation cancelled.")

    finally:
        test_file.unlink()




def test_translate_with_output_file(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
):
    """Test translation with output file."""
    input_file = Path("input.txt")
    output_file = Path("output.txt")
    input_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=mock_translation_result,
            ),
            patch(
                "tinbox.cli.load_document",
                new_callable=AsyncMock,
                return_value=mock_document_content,
            ),
            patch("tinbox.cli.create_translator", return_value=mock_model_interface),
            patch("tinbox.cli.console"),
            ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(input_file),
                    "--output",
                    str(output_file),
                    "--to",
                    "es",
                    "--model",
                    "openai:gpt-4o",
                ],
            )
            assert result.exit_code == 0
            assert output_file.exists()
            assert output_file.read_text() == mock_translation_result.text

    finally:
        input_file.unlink()
        if output_file.exists():
            output_file.unlink()


def test_translate_invalid_file(cli_runner):
    """Test translation with non-existent file."""
    result = cli_runner.invoke(
        app,
        [
            "translate",
            "nonexistent.txt",
            "--to",
            "es",
        ],
    )
    assert result.exit_code != 0


def test_translate_verbose_mode(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
):
    """Test translation in verbose mode."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=mock_translation_result,
            ),
            patch(
                "tinbox.cli.load_document",
                new_callable=AsyncMock,
                return_value=mock_document_content,
            ),
            patch("tinbox.cli.create_translator", return_value=mock_model_interface),
            patch("tinbox.cli.console"),
            ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--model",
                    "openai:gpt-4o",
                    "--verbose",
                ],
            )
            assert result.exit_code == 0

    finally:
        test_file.unlink()




def test_translate_force_mode(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
):
    """Test translation in force mode."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=mock_translation_result,
            ),
            patch(
                "tinbox.cli.load_document",
                new_callable=AsyncMock,
                return_value=mock_document_content,
            ),
            patch("tinbox.cli.create_translator", return_value=mock_model_interface),
            patch("tinbox.cli.console"),
            ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--model",
                    "openai:gpt-4o",
                    "--force",
                ],
            )
            assert result.exit_code == 0

    finally:
        test_file.unlink()




def test_translate_max_cost(cli_runner):
    """Test translation with max cost threshold."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    # Create cost estimate exceeding threshold
    estimate = CostEstimate(
        estimated_tokens=100_000,
        estimated_cost=3.0,
        estimated_time=600.0,
        warnings=[
            "Estimated cost ($3.00) exceeds maximum threshold ($1.00)",
        ],
    )

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=estimate),
            patch("tinbox.cli.console") as mock_console,
            patch("typer.confirm", return_value=False),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--model",
                    "openai:gpt-4o",
                    "--max-cost",
                    "1.0",
                ],
            )
            assert result.exit_code == 1
            # Check if the cancellation message was printed
            mock_console.print.assert_any_call("\nTranslation cancelled.")

    finally:
        test_file.unlink()



def test_cli_context_aware_default(cli_runner, tmp_path):
    """Test that context-aware is the default algorithm."""
    input_file = tmp_path / "test.txt"
    input_file.write_text("Test content")
    
    with patch("tinbox.cli.estimate_cost") as mock_estimate, \
         patch("tinbox.cli.load_document") as mock_load, \
         patch("tinbox.cli.create_translator") as mock_translator, \
         patch("tinbox.cli.translate_document") as mock_translate, \
         patch("tinbox.cli.console"):
        
        # Setup mocks
        mock_estimate.return_value = CostEstimate(
            estimated_tokens=100,
            estimated_cost=0.01,
            estimated_time=10.0,
            warnings=[]
        )
        mock_load.return_value = DocumentContent(
            pages=["Test content"],
            content_type="text/plain",
            metadata={}
        )
        # Mock translate_document to return a proper result object
        mock_translate.return_value = TranslationResponse(
            text="Translated text",
            tokens_used=100,
            cost=0.01,
            time_taken=5.0
        )
        
        # Don't specify algorithm - should default to context-aware
        result = cli_runner.invoke(app, [
            "translate",
            str(input_file),
            "--model", "openai:gpt-4o",
            "--force",
        ])
        
        assert result.exit_code == 0
        
        # Verify default algorithm is context-aware
        mock_estimate.assert_called_once()
        call_args = mock_estimate.call_args
        assert call_args.kwargs["algorithm"] == "context-aware"


def test_translate_reasoning_effort_default(cli_runner, tmp_path):
    """Test that reasoning_effort defaults to minimal."""
    input_file = tmp_path / "test.txt"
    input_file.write_text("Test content")
    
    with patch("tinbox.cli.estimate_cost") as mock_estimate, \
         patch("tinbox.cli.load_document") as mock_load, \
         patch("tinbox.cli.create_translator") as mock_translator, \
         patch("tinbox.cli.translate_document") as mock_translate, \
         patch("tinbox.cli.console"):
        
        # Setup mocks
        mock_estimate.return_value = CostEstimate(
            estimated_tokens=100,
            estimated_cost=0.01,
            estimated_time=10.0,
            warnings=[]
        )
        mock_load.return_value = DocumentContent(
            pages=["Test content"],
            content_type="text/plain",
            metadata={}
        )
        mock_translate.return_value = TranslationResponse(
            text="Translated text",
            tokens_used=100,
            cost=0.01,
            time_taken=5.0
        )
        
        # Don't specify reasoning_effort - should default to minimal
        result = cli_runner.invoke(app, [
            "translate",
            str(input_file),
            "--model", "openai:gpt-4o",
            "--force",
        ])
        
        assert result.exit_code == 0
        
        # Verify default reasoning_effort is minimal
        mock_estimate.assert_called_once()
        call_args = mock_estimate.call_args
        assert call_args.kwargs["reasoning_effort"] == "minimal"


def test_translate_reasoning_effort_high(cli_runner, tmp_path):
    """Test translation with high reasoning effort."""
    input_file = tmp_path / "test.txt"
    input_file.write_text("Test content")
    
    with patch("tinbox.cli.estimate_cost") as mock_estimate, \
         patch("tinbox.cli.load_document") as mock_load, \
         patch("tinbox.cli.create_translator") as mock_translator, \
         patch("tinbox.cli.translate_document") as mock_translate, \
         patch("tinbox.cli.console"):
        
        # Setup mocks
        mock_estimate.return_value = CostEstimate(
            estimated_tokens=100,
            estimated_cost=0.01,
            estimated_time=10.0,
            warnings=[]
        )
        mock_load.return_value = DocumentContent(
            pages=["Test content"],
            content_type="text/plain",
            metadata={}
        )
        mock_translate.return_value = TranslationResponse(
            text="Translated text",
            tokens_used=100,
            cost=0.01,
            time_taken=5.0
        )
        
        result = cli_runner.invoke(app, [
            "translate",
            str(input_file),
            "--model", "openai:gpt-4o",
            "--reasoning-effort", "high",
            "--force",
        ])
        
        assert result.exit_code == 0
        
        # Verify reasoning_effort is passed correctly
        mock_estimate.assert_called_once()
        call_args = mock_estimate.call_args
        assert call_args.kwargs["reasoning_effort"] == "high"


def test_translate_invalid_reasoning_effort(cli_runner, tmp_path):
    """Test translation with invalid reasoning effort."""
    input_file = tmp_path / "test.txt"
    input_file.write_text("Test content")
    
    result = cli_runner.invoke(app, [
        "translate",
        str(input_file),
        "--model", "openai:gpt-4o",
        "--reasoning-effort", "invalid",
    ])
    
    assert result.exit_code == 1
    assert "Invalid reasoning effort" in result.stdout


def test_translate_with_checkpoints_cleanup_success(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
    tmp_path,
):
    """Test that checkpoint cleanup is called after successful translation."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text("Test content")

    # Mock checkpoint manager
    mock_checkpoint_manager = MagicMock(spec=CheckpointManager)
    mock_checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    with (
        patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
        patch(
            "tinbox.cli.translate_document",
            new_callable=AsyncMock,
            return_value=mock_translation_result,
        ),
        patch(
            "tinbox.cli.load_document",
            new_callable=AsyncMock,
            return_value=mock_document_content,
        ),
        patch("tinbox.cli.create_translator", return_value=mock_model_interface),
        patch("tinbox.cli.CheckpointManager", return_value=mock_checkpoint_manager),
        patch("tinbox.cli.console"),
    ):
        result = cli_runner.invoke(
            app,
            [
                "translate",
                str(input_file),
                "--output",
                str(output_file),
                "--to",
                "es",
                "--model",
                "openai:gpt-4o",
                "--checkpoint-dir",
                str(tmp_path / "checkpoints"),
            ],
        )
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Verify checkpoint cleanup was called after successful translation
        mock_checkpoint_manager.cleanup_old_checkpoints.assert_called_once_with(
            input_file
        )


def test_translate_with_checkpoints_no_cleanup_without_checkpoint_dir(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
    tmp_path,
):
    """Test that checkpoint cleanup is NOT called when no checkpoint directory is specified."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Test content")

    with (
        patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
        patch(
            "tinbox.cli.translate_document",
            new_callable=AsyncMock,
            return_value=mock_translation_result,
        ),
        patch(
            "tinbox.cli.load_document",
            new_callable=AsyncMock,
            return_value=mock_document_content,
        ),
        patch("tinbox.cli.create_translator", return_value=mock_model_interface),
        patch("tinbox.cli.CheckpointManager") as mock_checkpoint_class,
        patch("tinbox.cli.console"),
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
                # No --checkpoint-dir specified
            ],
        )
        
        assert result.exit_code == 0
        
        # Verify CheckpointManager was never instantiated
        mock_checkpoint_class.assert_not_called()


def test_translate_with_checkpoints_cleanup_called_after_output_written(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
    tmp_path,
):
    """Test that checkpoint cleanup is called after output file is written, not before."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text("Test content")

    # Mock checkpoint manager
    mock_checkpoint_manager = MagicMock(spec=CheckpointManager)
    mock_checkpoint_manager.cleanup_old_checkpoints = AsyncMock()

    # Track call order
    call_order = []
    
    def track_write(*args, **kwargs):
        call_order.append("write")
        # Actually write the file
        output_file.write_text(mock_translation_result.text)
    
    async def track_cleanup(*args, **kwargs):
        call_order.append("cleanup")

    with (
        patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
        patch(
            "tinbox.cli.translate_document",
            new_callable=AsyncMock,
            return_value=mock_translation_result,
        ),
        patch(
            "tinbox.cli.load_document",
            new_callable=AsyncMock,
            return_value=mock_document_content,
        ),
        patch("tinbox.cli.create_translator", return_value=mock_model_interface),
        patch("tinbox.cli.CheckpointManager", return_value=mock_checkpoint_manager),
        patch("tinbox.cli.console"),
    ):
        # Mock the handler.write method to track when it's called
        with patch("tinbox.cli.create_handler") as mock_create_handler:
            mock_handler = MagicMock()
            mock_handler.write = MagicMock(side_effect=track_write)
            mock_create_handler.return_value = mock_handler
            
            # Mock cleanup to track when it's called
            mock_checkpoint_manager.cleanup_old_checkpoints.side_effect = track_cleanup
            
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(input_file),
                    "--output",
                    str(output_file),
                    "--to",
                    "es",
                    "--model",
                    "openai:gpt-4o",
                    "--checkpoint-dir",
                    str(tmp_path / "checkpoints"),
                ],
            )
            
            assert result.exit_code == 0
            
            # Verify that output was written before cleanup
            assert call_order == ["write", "cleanup"]
            
            # Verify cleanup was called
            mock_checkpoint_manager.cleanup_old_checkpoints.assert_called_once_with(
                input_file
            )
