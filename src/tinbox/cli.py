"""Command-line interface for Tinbox."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from tinbox.core import (
    FileType,
    ModelType,
    TranslationConfig,
    TranslationResult,
    translate_document,
)
from tinbox.core.translation.checkpoint import CheckpointManager
from tinbox.core.cost import estimate_cost
from tinbox.core.processor import load_document
from tinbox.core.translation import create_translator
from tinbox.core.translation.glossary import GlossaryManager
from tinbox.core.output import (
    OutputFormat,
    TranslationMetadata,
    TranslationOutput,
    create_handler,
)
from tinbox.utils.logging import configure_logging, get_logger
from tinbox.core.progress import CurrentCostColumn, EstimatedCostColumn

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
    table.add_row("Estimated Time", f"{estimate.estimated_time / 60:.1f} minutes" if estimate.estimated_time > 60 else "<1 minute")
    table.add_row("Cost Level", estimate.cost_level.value.title())

    console.print(table)

    if estimate.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in estimate.warnings:
            console.print(f"• {warning}")


def parse_model_spec(model_spec: str) -> tuple[ModelType, str]:
    """Parse a model specification string.

    Args:
        model_spec: Model specification (e.g., 'openai:gpt-4o', 'anthropic:claude-3-sonnet', 'ollama:mistral-small')

    Returns:
        Tuple of (ModelType, str) for the provider and model name.

    Raises:
        ValueError: If the model specification is invalid
    """
    if ":" not in model_spec:
        raise ValueError(
            "Invalid model specification. Use format 'provider:model' "
            "(e.g., 'openai:gpt-4o', 'anthropic:claude-3-sonnet', 'ollama:mistral-small')"
        )

    provider, model = model_spec.split(":", 1)
    try:
        return ModelType(provider.lower()), model
    except ValueError:
        raise ValueError(
            f"Unknown model provider: {provider}. "
            f"Supported providers: {', '.join(m.value for m in ModelType)}"
        )


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
        "-F",
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
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model to use (e.g., 'openai:gpt-4o', 'anthropic:claude-3-sonnet', 'ollama:mistral-small').",
    ),
    algorithm: Optional[str] = typer.Option(
        None,
        "--algorithm",
        "-a",
        help="Translation algorithm: 'page', 'sliding-window', or 'context-aware'. Auto-selects based on file type if not specified (PDF→page, text→context-aware).",
    ),
    context_size: Optional[int] = typer.Option(
        2000,
        "--context-size",
        help="Target chunk size for context-aware algorithm (characters).",
    ),
    custom_split_token: Optional[str] = typer.Option(
        None,
        "--split-token",
        help="Custom token to split text on (context-aware only).",
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
    checkpoint_dir: Optional[Path] = typer.Option(
        None,
        "--checkpoint-dir",
        help="Directory to store translation checkpoints for resuming interrupted translations.",
    ),
    checkpoint_frequency: int = typer.Option(
        1,
        "--checkpoint-frequency",
        help="Save checkpoint every N pages/chunks (default: 1).",
    ),
    use_glossary: bool = typer.Option(
        False,
        "--glossary",
        help="Enable glossary for consistent term translations.",
    ),
    glossary_file: Optional[Path] = typer.Option(
        None,
        "--glossary-file",
        help="Path to existing glossary file (JSON format) to load initial terms from.",
    ),
    save_glossary: Optional[Path] = typer.Option(
        None,
        "--save-glossary",
        help="Path to save the updated glossary after translation.",
    ),
    reasoning_effort: str = typer.Option(
        "minimal",
        "--reasoning-effort",
        help="Model reasoning effort level (minimal, low, medium, high). Higher levels improve quality but increase cost and time significantly.",
    ),
) -> None:
    """Translate a document using LLMs."""
    try:
        # Validate reasoning effort parameter
        valid_reasoning_efforts = ["minimal", "low", "medium", "high"]
        if reasoning_effort not in valid_reasoning_efforts:
            raise ValueError(
                f"Invalid reasoning effort '{reasoning_effort}'. "
                f"Valid options: {', '.join(valid_reasoning_efforts)}"
            )

        # Parse model specification
        model_type, model_name = parse_model_spec(model)

        # Determine file type
        file_type = FileType(input_file.suffix.lstrip(".").lower())

        # Auto-select algorithm based on file type, or validate user's choice
        if algorithm is None:
            # Auto-select: PDF needs 'page' algorithm, text files use 'context-aware'
            effective_algorithm = "page" if file_type == FileType.PDF else "context-aware"
        else:
            effective_algorithm = algorithm
            # Validate: PDF files only work with 'page' algorithm
            if file_type == FileType.PDF and algorithm != "page":
                console.print(
                    f"\n[red]Error:[/red] PDF files only support the 'page' algorithm.\n"
                    f"PDFs are processed as images and cannot use '{algorithm}' algorithm.\n"
                    f"Please use: tinbox translate {input_file} --algorithm page --model {model}"
                )
                raise typer.Exit(1)

        # Get cost estimate
        estimate = estimate_cost(
            input_file,
            model_type,
            algorithm=effective_algorithm,
            max_cost=max_cost,
            use_glossary=use_glossary,
            reasoning_effort=reasoning_effort,
        )

        # Display cost estimate
        console.print("\n[bold]Translation Plan[/bold]")
        display_cost_estimate(estimate, model_type)

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

        # Create translation config
        config = TranslationConfig(
            source_lang=source_lang or "auto",
            target_lang=target_lang,
            model=model_type,
            model_name=model_name,
            algorithm=effective_algorithm,
            input_file=input_file,
            output_file=output_file,
            force=force,
            max_cost=max_cost,
            verbose=verbose,
            context_size=context_size,
            custom_split_token=custom_split_token,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            use_glossary=use_glossary,
            reasoning_effort=reasoning_effort,
        )

        # Load document
        content = asyncio.run(load_document(input_file))

        # Initialize model interface
        translator = create_translator(config)

        # Create checkpoint manager if checkpoint directory is specified
        checkpoint_manager = None
        if config.checkpoint_dir:
            checkpoint_manager = CheckpointManager(config)

        # Initialize glossary manager
        glossary_manager = None
        if use_glossary:
            if glossary_file:
                glossary_manager = GlossaryManager.load_from_file(glossary_file)
            else:
                glossary_manager = GlossaryManager()

        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            CurrentCostColumn(),
            EstimatedCostColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Run translation
            response = asyncio.run(
                translate_document(
                    content=content,
                    config=config,
                    translator=translator,
                    progress=progress,
                    checkpoint_manager=checkpoint_manager,
                    glossary_manager=glossary_manager,
                )
            )

            # Convert TranslationResponse to TranslationResult
            result = TranslationResult(
                text=response.text,
                tokens_used=response.tokens_used,
                cost=response.cost,
                time_taken=response.time_taken,
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

        # Save glossary if requested
        if glossary_manager and save_glossary:
            glossary_manager.save_to_file(save_glossary)

        # Clean up checkpoints after successful output
        if checkpoint_manager:
            asyncio.run(checkpoint_manager.cleanup_old_checkpoints(config.input_file))

        # Show final statistics (only for text output)
        if output_format == OutputFormat.TEXT:
            table = Table(title="Translation Statistics", show_header=False)
            table.add_column("Metric", style="bold blue")
            table.add_column("Value")

            table.add_row("Time Taken", f"{result.time_taken:.1f} seconds")
            table.add_row("Tokens Used", f"{result.tokens_used:,}")
            if model_type != ModelType.OLLAMA:
                table.add_row("Final Cost", f"${result.cost:.4f}")

            console.print("\n", table)

    except Exception as e:
        logger.exception("Translation failed")
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
