"""Tests for output handlers."""

import json
from datetime import datetime

import pytest

from tinbox.core.output import (
    OutputFormat,
    TranslationMetadata,
    TranslationOutput,
    create_handler,
)
from tinbox.core.types import FileType, ModelType, TranslationResult


@pytest.fixture
def sample_output(tmp_path):
    """Create a sample translation output."""
    return TranslationOutput(
        metadata=TranslationMetadata(
            source_lang="en",
            target_lang="es",
            model=ModelType.ANTHROPIC,
            algorithm="page",
            input_file=tmp_path / "test.txt",
            input_file_type=FileType.TXT,
            timestamp=datetime(2024, 3, 21, 14, 30),
        ),
        result=TranslationResult(
            text="Translated text",
            tokens_used=1500,
            cost=0.045,
            time_taken=12.5,
        ),
        warnings=["Large document detected"],
        errors=[],
    )


def test_text_output(sample_output, tmp_path, capsys):
    """Test plain text output handler."""
    output_file = tmp_path / "output.txt"
    handler = create_handler(OutputFormat.TEXT)

    # Test file output
    handler.write(sample_output, output_file)
    assert output_file.read_text() == "Translated text"

    # Test stdout output (captured by pytest)
    handler.write(sample_output)
    captured = capsys.readouterr()
    assert captured.out.strip() == "Translated text"


def test_json_output(sample_output, tmp_path, capsys):
    """Test JSON output handler."""
    output_file = tmp_path / "output.json"
    handler = create_handler(OutputFormat.JSON)

    # Test file output
    handler.write(sample_output, output_file)
    data = json.loads(output_file.read_text())

    # Verify structure
    assert "metadata" in data
    assert "result" in data
    assert "warnings" in data
    assert "errors" in data

    # Verify content
    assert data["metadata"]["source_lang"] == "en"
    assert data["metadata"]["target_lang"] == "es"
    assert data["metadata"]["model"] == "anthropic"
    assert data["result"]["text"] == "Translated text"
    assert data["result"]["tokens_used"] == 1500
    assert data["warnings"] == ["Large document detected"]
    assert data["errors"] == []

    # Test stdout output (captured by pytest)
    handler.write(sample_output)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["result"]["text"] == "Translated text"


def test_markdown_output(sample_output, tmp_path, capsys):
    """Test Markdown output handler."""
    output_file = tmp_path / "output.md"
    handler = create_handler(OutputFormat.MARKDOWN)

    # Test file output
    handler.write(sample_output, output_file)
    content = output_file.read_text()

    # Verify structure and content
    assert "# Translation Results" in content
    assert "## Metadata" in content
    assert "## Translation" in content
    assert "## Statistics" in content
    assert "## Warnings" in content
    assert "## Errors" in content

    # Verify specific content
    assert "Source Language: en" in content
    assert "Target Language: es" in content
    assert "Model: anthropic" in content
    assert "```text\nTranslated text\n```" in content
    assert "Tokens Used: 1,500" in content
    assert "Cost: $0.0450" in content
    assert "Time Taken: 12.5s" in content
    assert "Large document detected" in content
    assert "[None]" in content  # No errors

    # Test stdout output
    handler.write(sample_output)
    captured = capsys.readouterr()
    assert "# Translation Results" in captured.out
    assert "Translated text" in captured.out


def test_markdown_output_with_errors(tmp_path):
    """Test Markdown output with errors."""
    output = TranslationOutput(
        metadata=TranslationMetadata(
            source_lang="en",
            target_lang="es",
            model=ModelType.ANTHROPIC,
            algorithm="page",
            input_file=tmp_path / "test.txt",
            input_file_type=FileType.TXT,
        ),
        result=TranslationResult(
            text="Partial translation",
            tokens_used=100,
            cost=0.003,
            time_taken=1.0,
        ),
        warnings=["Network latency detected"],
        errors=["Failed to translate page 2"],
    )

    # Test file output with errors
    handler = create_handler(OutputFormat.MARKDOWN)
    output_file = tmp_path / "output.md"
    handler.write(output, output_file)
    content = output_file.read_text()

    # Verify error section
    assert "## Errors" in content
    assert "Failed to translate page 2" in content
    assert "Network latency detected" in content


def test_unsupported_format():
    """Test handling of unsupported output format."""
    with pytest.raises(ValueError, match="Unsupported output format"):
        create_handler("invalid")  # Invalid format


def test_output_with_missing_data(tmp_path):
    """Test output handling with missing optional data."""
    # Create output with minimal data
    output = TranslationOutput(
        metadata=TranslationMetadata(
            source_lang="en",
            target_lang="es",
            model=ModelType.ANTHROPIC,
            algorithm="page",
            input_file=tmp_path / "test.txt",
            input_file_type=FileType.TXT,
        ),
        result=TranslationResult(
            text="Translated text",
            tokens_used=100,
            cost=0.003,
            time_taken=1.0,
        ),
    )

    # Test JSON output
    handler = create_handler(OutputFormat.JSON)
    output_file = tmp_path / "output.json"
    handler.write(output, output_file)
    data = json.loads(output_file.read_text())

    # Verify optional fields are handled
    assert "warnings" in data
    assert "errors" in data
    assert data["warnings"] == []
    assert data["errors"] == []

    # Test Markdown output
    handler = create_handler(OutputFormat.MARKDOWN)
    output_file = tmp_path / "output.md"
    handler.write(output, output_file)
    content = output_file.read_text()

    # Verify optional sections are handled
    assert "## Warnings" not in content  # No warnings section if empty
    assert "## Errors" in content
    assert "[None]" in content  # Shows [None] for empty errors
