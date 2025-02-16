"""Command-line interface for Tinbox."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from tinbox.core import FileType, ModelType, TranslationConfig, translate_document
from tinbox.core.cost import estimate_cost
from tinbox.core.processor import load_document
from tinbox.core.translation.interface import ModelInterface
from tinbox.core.output import (
    OutputFormat,
    TranslationMetadata,
    TranslationOutput,
    create_handler,
)
from tinbox.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="tinbox",
    help="A CLI tool for translating documents using LLMs",
    add_completion=False,
)
console = Console()
logger = get_logger()


def version_callback(value: bool) -> None:
    """Print version information and exit."""
    if value:
        from tinbox import __version__

        console.print(f"Tinbox version: {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version information and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set the logging level.",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output logs in JSON format.",
    ),
) -> None:
    """Tinbox - A CLI Translation Tool using LLMs."""
    configure_logging(level=log_level, json=json_logs)


def display_cost_estimate(estimate, model: ModelType) -> None:
    """Display cost estimate in a rich format."""
    table = Table(title="Cost Estimate", show_header=False)
    table.add_column("Metric", style="bold blue")
    table.add_column("Value")

    table.add_row("Estimated Tokens", f"{estimate.estimated_tokens:,}")
    if model != ModelType.OLLAMA:
        table.add_row("Estimated Cost", f"${estimate.estimated_cost:.2f}")
    table.add_row("Estimated Time", f"{estimate.estimated_time / 60:.1f} minutes")
    table.add_row("Cost Level", estimate.cost_level.value.title())

    console.print(table)

    if estimate.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in estimate.warnings:
            console.print(f"• {warning}")


@app.command()
def translate(
    input_file: Path = typer.Argument(
        ...,
        help="The input file to translate.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="The output file. If not specified, prints to stdout.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        help="Output format (text, json, or markdown).",
    ),
    source_lang: str = typer.Option(
        None,
        "--from",
        "-f",
        help="Source language code (e.g., 'en', 'zh'). Defaults to auto-detect.",
    ),
    target_lang: str = typer.Option(
        "en",
        "--to",
        "-t",
        help="Target language code. Defaults to 'en' (English).",
    ),
    model: ModelType = typer.Option(
        ModelType.CLAUDE_3_SONNET,
        "--model",
        "-m",
        help="The model to use for translation.",
    ),
    algorithm: str = typer.Option(
        "page",
        "--algorithm",
        "-a",
        help="Translation algorithm to use: 'page' or 'sliding-window'.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Estimate cost and tokens without performing translation.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip warnings and proceed with translation.",
    ),
    max_cost: Optional[float] = typer.Option(
        None,
        "--max-cost",
        help="Maximum cost threshold in USD.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed progress information.",
    ),
) -> None:
    """Translate a document using LLMs."""
    try:
        # Determine file type
        file_type = FileType(input_file.suffix.lstrip(".").lower())

        # Get cost estimate
        estimate = estimate_cost(
            input_file,
            model,
            max_cost=max_cost,
        )

        # Display cost estimate
        console.print("\n[bold]Translation Plan[/bold]")
        display_cost_estimate(estimate, model)

        # In dry-run mode, exit after showing estimate
        if dry_run:
            return

        # Check for warnings and confirm if needed
        if not force and estimate.warnings:
            console.print(
                "\n[yellow]Warning:[/yellow] This translation has potential issues."
            )
            proceed = typer.confirm("Do you want to proceed?")
            if not proceed:
                console.print("\nTranslation cancelled.")
                raise typer.Exit(1)

        # Create configuration
        config = TranslationConfig(
            source_lang=source_lang or "auto",
            target_lang=target_lang,
            model=model,
            algorithm=algorithm,
            input_file=input_file,
            output_file=output_file,
            verbose=verbose,
            max_cost=max_cost,
            force=force,
        )

        # Load document
        content = asyncio.run(load_document(input_file))

        # Initialize model interface
        translator = ModelInterface()

        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Run translation
            result = asyncio.run(
                translate_document(
                    content=content,
                    config=config,
                    translator=translator,
                    progress=progress,
                )
            )

        # Create translation output with metadata
        output = TranslationOutput(
            metadata=TranslationMetadata(
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                model=config.model,
                algorithm=config.algorithm,
                input_file=config.input_file,
                input_file_type=file_type,
            ),
            result=result,
            warnings=estimate.warnings,
        )

        # Get appropriate output handler
        handler = create_handler(output_format)
        handler.write(output, output_file)

        # Show final statistics (only for text output)
        if output_format == OutputFormat.TEXT:
            table = Table(title="Translation Statistics", show_header=False)
            table.add_column("Metric", style="bold blue")
            table.add_column("Value")

            table.add_row("Time Taken", f"{result.time_taken:.1f} seconds")
            table.add_row("Tokens Used", f"{result.tokens_used:,}")
            if model != ModelType.OLLAMA:
                table.add_row("Final Cost", f"${result.cost:.4f}")

            console.print("\n", table)

    except Exception as e:
        logger.exception("Translation failed")
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
