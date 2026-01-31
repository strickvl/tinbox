"""Output handlers for different formats."""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

from pydantic import BaseModel, Field

from tinbox.core.types import FileType, ModelType, TranslationResult
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class OutputFormat(str, Enum):
    """Supported output formats."""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class TranslationMetadata(BaseModel):
    """Metadata about the translation process."""

    source_lang: str
    target_lang: str
    model: ModelType
    algorithm: str
    input_file: Path
    input_file_type: FileType
    timestamp: datetime = Field(default_factory=datetime.now)


class TranslationOutput(BaseModel):
    """Complete translation output including metadata."""

    metadata: TranslationMetadata
    result: TranslationResult
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class OutputHandler(Protocol):
    """Protocol for output handlers."""

    def write(
        self,
        output: TranslationOutput,
        file: Optional[Path] = None,
    ) -> None:
        """Write the translation output.

        Args:
            output: The translation output to write
            file: Optional file to write to. If None, writes to stdout.
        """
        ...


class JSONOutputHandler:
    """Handler for JSON output format."""

    def write(
        self,
        output: TranslationOutput,
        file: Optional[Path] = None,
    ) -> None:
        """Write translation output as JSON.

        Args:
            output: The translation output to write
            file: Optional file to write to. If None, writes to stdout.
        """
        # Convert to JSON-serializable dict
        data = output.model_dump(
            mode="json",
            exclude_none=True,
        )

        # Convert Path objects to strings
        data["metadata"]["input_file"] = str(data["metadata"]["input_file"])
        if "output_file" in data["metadata"]:
            data["metadata"]["output_file"] = str(data["metadata"]["output_file"])

        # Format JSON with indentation
        json_str = json.dumps(data, indent=2)

        if file:
            file.write_text(json_str, encoding="utf-8")
        else:
            print(json_str)


class TextOutputHandler:
    """Handler for plain text output format."""

    def write(
        self,
        output: TranslationOutput,
        file: Optional[Path] = None,
    ) -> None:
        """Write translation output as plain text.

        Args:
            output: The translation output to write
            file: Optional file to write to. If None, writes to stdout.
        """
        # Just write the translated text
        if file:
            file.write_text(output.result.text, encoding="utf-8")
        else:
            print(output.result.text)


class MarkdownOutputHandler:
    """Handler for Markdown output format."""

    def write(
        self,
        output: TranslationOutput,
        file: Optional[Path] = None,
    ) -> None:
        """Write translation output as Markdown.

        Args:
            output: The translation output to write
            file: Optional file to write to. If None, writes to stdout.
        """
        # Build Markdown content
        md_lines = [
            "# Translation Results\n",
            "## Metadata",
            f"- Source Language: {output.metadata.source_lang}",
            f"- Target Language: {output.metadata.target_lang}",
            f"- Model: {output.metadata.model.value}",
            f"- Algorithm: {output.metadata.algorithm}",
            f"- Input File: {output.metadata.input_file.name}",
            f"- File Type: {output.metadata.input_file_type.value}",
            f"- Timestamp: {output.metadata.timestamp.isoformat()}\n",
            "## Translation",
            "```text",
            output.result.text,
            "```\n",
            "## Statistics",
            f"- Tokens Used: {output.result.tokens_used:,}",
            f"- Cost: ${output.result.cost:.4f}",
            f"- Time Taken: {output.result.time_taken:.1f}s\n",
        ]

        # Add warnings if any
        if output.warnings:
            md_lines.extend(
                [
                    "## Warnings",
                    *[f"- {warning}" for warning in output.warnings],
                    "",
                ]
            )

        # Add errors if any
        if output.errors:
            md_lines.extend(
                [
                    "## Errors",
                    *[f"- {error}" for error in output.errors],
                    "",
                ]
            )
        else:
            md_lines.extend(
                [
                    "## Errors",
                    "[None]",
                    "",
                ]
            )

        # Join lines with newlines
        md_content = "\n".join(md_lines)

        if file:
            file.write_text(md_content, encoding="utf-8")
        else:
            print(md_content)


def create_handler(format: OutputFormat) -> OutputHandler:
    """Create an output handler for the specified format.

    Args:
        format: The desired output format

    Returns:
        An appropriate output handler

    Raises:
        ValueError: If the format is not supported
    """
    handlers = {
        OutputFormat.TEXT: TextOutputHandler(),
        OutputFormat.JSON: JSONOutputHandler(),
        OutputFormat.MARKDOWN: MarkdownOutputHandler(),
    }

    handler = handlers.get(format)
    if not handler:
        raise ValueError(f"Unsupported output format: {format}")

    return handler
