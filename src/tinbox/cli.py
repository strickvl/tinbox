"""Command-line interface for Tinbox."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from tinbox.core import (
    FileType,
    ModelType,
    TranslationConfig,
    TranslationResult,
    translate_document,
)
from tinbox.core.cost import estimate_cost, is_local_model
from tinbox.core.doctor import run_doctor_checks
from tinbox.core.output import (
    OutputFormat,
    TranslationMetadata,
    TranslationOutput,
    create_handler,
)
from tinbox.core.processor import load_document
from tinbox.core.progress import CurrentCostColumn, EstimatedCostColumn
from tinbox.core.review import (
    ReviewBundle,
    load_review_bundle,
    save_review_bundle,
    validate_new_output_path,
    write_review_report,
)
from tinbox.core.review_matches import load_matches, save_matches, unchecked_matches
from tinbox.core.translation import create_translator
from tinbox.core.translation.checkpoint import CheckpointManager
from tinbox.core.translation.glossary import GlossaryManager
from tinbox.core.types import TranslationProtocol
from tinbox.utils.language import LanguageError, validate_language_pair
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
    version: bool | None = typer.Option(
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


def display_cost_estimate(estimate, model: ModelType, *, local: bool = False) -> None:
    """Display cost estimate in a rich format."""
    table = Table(title="Cost Estimate", show_header=False)
    table.add_column("Metric", style="bold blue")
    table.add_column("Value")

    table.add_row("Estimated Tokens", f"{estimate.estimated_tokens:,}")
    if is_local_model(model, local=local):
        table.add_row("API Charges", "None (local model)")
    else:
        table.add_row("Estimated Cost", f"${estimate.estimated_cost:.2f}")
    table.add_row(
        "Estimated Time",
        f"{estimate.estimated_time / 60:.1f} minutes"
        if estimate.estimated_time > 60
        else "<1 minute",
    )
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
    output_file: Path | None = typer.Option(
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
    save_review_input: Path | None = typer.Option(
        None,
        "--save-review-input",
        help="Save source/translation chunk pairs to a new JSON file for offline review. Fresh .txt context-aware translations only.",
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
    algorithm: str | None = typer.Option(
        None,
        "--algorithm",
        "-a",
        help="Translation algorithm: 'page', 'sliding-window', or 'context-aware'. Defaults to page for PDF/TranslateGemma, context-aware for other text.",
    ),
    translation_protocol: TranslationProtocol = typer.Option(
        TranslationProtocol.CHAT,
        "--translation-protocol",
        help="Use chat JSON responses or TranslateGemma plain-text completions.",
    ),
    local_model: bool = typer.Option(
        False,
        "--local",
        help="The endpoint runs a local model with no API charges (not a paid API proxy).",
    ),
    context_size: int | None = typer.Option(
        2000,
        "--context-size",
        help="Target chunk size for context-aware algorithm (characters).",
    ),
    custom_split_token: str | None = typer.Option(
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
    max_cost: float | None = typer.Option(
        None,
        "--max-cost",
        help="Maximum cost threshold in USD.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed progress information.",
    ),
    checkpoint_dir: Path | None = typer.Option(
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
    glossary_file: Path | None = typer.Option(
        None,
        "--glossary-file",
        help="Path to existing glossary file (JSON format) to load initial terms from.",
    ),
    save_glossary: Path | None = typer.Option(
        None,
        "--save-glossary",
        help="Path to save the updated glossary after translation.",
    ),
    reasoning_effort: str = typer.Option(
        "minimal",
        "--reasoning-effort",
        help="Model reasoning effort level (minimal, low, medium, high). Higher levels improve quality but increase cost and time significantly.",
    ),
    pdf_dpi: int = typer.Option(
        200,
        "--pdf-dpi",
        help="DPI for PDF rasterization (PDF files only). Higher values = better quality but more tokens. Default: 200.",
    ),
) -> None:
    """Translate a document using LLMs."""
    try:
        # Validate and normalize language codes early (before cost estimation)
        try:
            effective_source = source_lang or "auto"
            effective_source, effective_target = validate_language_pair(
                effective_source, target_lang
            )
        except LanguageError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        # Validate reasoning effort parameter
        valid_reasoning_efforts = ("minimal", "low", "medium", "high")
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
            effective_algorithm = (
                "page"
                if file_type == FileType.PDF
                or translation_protocol == TranslationProtocol.TRANSLATEGEMMA
                else "context-aware"
            )
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

        if effective_algorithm not in ("page", "sliding-window", "context-aware"):
            raise ValueError(f"Invalid translation algorithm '{effective_algorithm}'")

        # Create translation config (use normalized language codes)
        config = TranslationConfig(
            source_lang=effective_source,
            target_lang=effective_target,
            model=model_type,
            model_name=model_name,
            translation_protocol=translation_protocol,
            local_model=local_model,
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
            preserve_review_pairs=save_review_input is not None,
        )

        if save_review_input is not None:
            translation_paths = [input_file]
            translation_paths.extend(
                path for path in (output_file, save_glossary) if path is not None
            )
            if checkpoint_dir is not None:
                translation_paths.append(
                    checkpoint_dir / f"{input_file.stem}_checkpoint.json"
                )
            protected_paths = [*translation_paths]
            if glossary_file is not None:
                protected_paths.append(glossary_file)
            validate_new_output_path(save_review_input, protected_paths)
            for index, path in enumerate(translation_paths):
                for other in translation_paths[:index]:
                    if path.resolve() == other.resolve() or (
                        path.exists() and other.exists() and path.samefile(other)
                    ):
                        raise ValueError(
                            "Review export requires separate source, translation, "
                            f"saved glossary and checkpoint paths: {path} and {other}"
                        )

        # Get cost estimate
        if (
            translation_protocol == TranslationProtocol.TRANSLATEGEMMA
            and not os.environ.get("OPENAI_BASE_URL")
        ):
            raise ValueError("Set OPENAI_BASE_URL to your TranslateGemma server")

        estimate = estimate_cost(
            input_file,
            model_type,
            algorithm=effective_algorithm,
            max_cost=max_cost,
            use_glossary=use_glossary,
            reasoning_effort=reasoning_effort,
            local=local_model,
        )

        # Display cost estimate
        console.print("\n[bold]Translation Plan[/bold]")
        display_cost_estimate(estimate, model_type, local=local_model)

        # In dry-run mode, exit after showing estimate
        if dry_run:
            return

        # Check for warnings and confirm if needed
        if not force and estimate.warnings and estimate.requires_confirmation:
            console.print(
                "\n[yellow]Review the token and cost estimates above before proceeding.[/yellow]"
            )
            proceed = typer.confirm("Do you want to proceed?")
            if not proceed:
                console.print("\nTranslation cancelled.")
                raise typer.Exit(1)

        # Load document (pass DPI setting for PDF files)
        source_sha256 = (
            hashlib.sha256(input_file.read_bytes()).hexdigest()
            if save_review_input is not None
            else None
        )
        processor_settings = {"dpi": pdf_dpi} if file_type == FileType.PDF else None
        content = asyncio.run(
            load_document(input_file, processor_settings=processor_settings)
        )

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

            # Convert TranslationResponse to TranslationResult (including failure info)
            result = TranslationResult(
                text=response.text,
                tokens_used=response.tokens_used,
                cost=response.cost,
                time_taken=response.time_taken,
                failed_pages=response.failed_pages,
                page_errors=response.page_errors,
                warnings=response.warnings,
            )

        # Combine warnings from estimate and translation
        all_warnings = estimate.warnings + response.warnings

        # Build error messages from failed pages
        errors = [
            f"Failed to translate page {p}: {response.page_errors.get(p, 'Unknown error')}"
            for p in response.failed_pages
        ]

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
            warnings=all_warnings,
            errors=errors,
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

        if save_review_input is not None:
            try:
                if not response.review_pairs or source_sha256 is None:
                    raise ValueError("Translation did not return review pairs")
                bundle = ReviewBundle(
                    source_name=input_file.name,
                    source_sha256=source_sha256,
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    provider=config.model.value,
                    model_name=config.model_name,
                    context_size=config.context_size or 2000,
                    custom_split_token=config.custom_split_token,
                    use_glossary=config.use_glossary,
                    reasoning_effort=config.reasoning_effort,
                    pairs=response.review_pairs,
                )
                save_review_bundle(bundle, save_review_input)
            except (OSError, ValueError) as e:
                console.print(
                    f"[red]Review bundle could not be saved:[/red] {escape(str(e))}\n"
                    "Translation output has already been written "
                    + (
                        f"to {escape(str(output_file))}."
                        if output_file is not None
                        else "to the console."
                    )
                )
                sys.exit(1)
            console.print(
                f"Review input saved to {escape(str(save_review_input))} "
                f"({len(bundle.pairs)} chunks)."
            )

        # Show final statistics (only for text output)
        if output_format == OutputFormat.TEXT:
            table = Table(title="Translation Statistics", show_header=False)
            table.add_column("Metric", style="bold blue")
            table.add_column("Value")

            table.add_row("Time Taken", f"{result.time_taken:.1f} seconds")
            table.add_row("Tokens Used", f"{result.tokens_used:,}")
            if not is_local_model(model_type, local=local_model):
                table.add_row("Final Cost", f"${result.cost:.4f}")

            console.print("\n", table)

    except Exception as e:
        logger.exception("Translation failed")
        console.print(f"[red]Error: {e!s}[/red]")
        sys.exit(1)


@app.command()
def prepare_matches(
    review_input: Path = typer.Argument(
        ...,
        help="A JSON bundle created with translate --save-review-input.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="New JSON matching draft. Every chunk starts unchecked.",
    ),
) -> None:
    """Prepare an offline matching draft for manual curation, without model calls."""
    try:
        validate_new_output_path(output, [review_input])
        bundle = load_review_bundle(review_input)
        save_matches(unchecked_matches(bundle), output, bundle)
    except (OSError, ValueError) as e:
        console.print(f"[red]Could not prepare matching draft:[/red] {escape(str(e))}")
        raise typer.Exit(1) from e
    console.print(
        f"Matching draft saved to {escape(str(output))}. "
        "Every chunk is unchecked; curate sentence spans before marking matches. "
        "No automatic alignment or quality assessment was performed."
    )


@app.command()
def review(
    review_input: Path = typer.Argument(
        ...,
        help="A JSON bundle created with translate --save-review-input.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="New HTML report file. Its parent directory must already exist.",
    ),
    matches: Path | None = typer.Option(
        None,
        "--matches",
        help="Optional curated sentence-matching JSON for this exact review bundle.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Create an offline side-by-side report for human review, without model calls."""
    try:
        protected_paths = [review_input]
        if matches is not None:
            protected_paths.append(matches)
        validate_new_output_path(output, protected_paths)
        bundle = load_review_bundle(review_input)
        matching = load_matches(matches, bundle) if matches is not None else None
        write_review_report(bundle, output, matches=matching)
    except (OSError, ValueError) as e:
        console.print(f"[red]Could not create review report:[/red] {escape(str(e))}")
        raise typer.Exit(1) from e
    console.print(
        f"Review report saved to {escape(str(output))} ({len(bundle.pairs)} chunks). "
        "Open it in a browser. No automated quality assessment was performed."
    )


@app.command()
def doctor(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output report as JSON.",
    ),
) -> None:
    """Check Tinbox setup and diagnose common issues.

    Runs diagnostic checks for:
    - System tools (poppler for PDF processing)
    - Python packages (pdf2image, python-docx, pillow)
    - API keys (OpenAI, Anthropic, Google)
    - Local model support (Ollama)
    """
    report = run_doctor_checks()

    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return

    # Group checks by category
    categories: dict[str, list] = {}
    for check in report.checks:
        if check.category not in categories:
            categories[check.category] = []
        categories[check.category].append(check)

    console.print("\n[bold]Tinbox Diagnostic Report[/bold]\n")

    for category, checks in categories.items():
        table = Table(title=category, show_header=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Details")

        for check in checks:
            if check.ok:
                status = "[green]OK[/green]"
            elif check.required:
                status = "[red]MISSING[/red]"
            else:
                status = "[yellow]OPTIONAL[/yellow]"
            details = escape(check.details or "")
            if not check.ok and check.hint:
                details = f"{details}\n[dim]{escape(check.hint)}[/dim]"
            table.add_row(check.name, status, details)

        console.print(table)
        console.print()

    # Summary
    if report.all_ok:
        console.print("[green]All checks passed![/green] Tinbox is ready to use.\n")
    elif report.required_ok:
        console.print(
            "[yellow]Core checks passed.[/yellow] Some optional features may be unavailable.\n"
            "Follow the hints above to enable the optional features you need.\n"
        )
    else:
        console.print(
            "[red]Some required checks failed.[/red] Please fix the issues above.\n"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
