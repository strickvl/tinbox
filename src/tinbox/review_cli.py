"""Commands that build and check translation review files.

They are registered on the main `tinbox` app in ``tinbox.cli``.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from tinbox.core.review import (
    load_review_bundle,
    validate_new_output_path,
    write_review_report,
)
from tinbox.core.review_annotations import (
    DEFAULT_MODEL,
    load_annotations,
    save_annotations,
)
from tinbox.core.review_annotations import (
    annotate_review as annotate_groups,
)
from tinbox.core.review_matches import load_matches, save_matches, unchecked_matches

console = Console()


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


def annotate_review(
    review_input: Path = typer.Argument(
        ...,
        help="A JSON bundle created with translate --save-review-input.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    matches: Path = typer.Option(
        ...,
        "--matches",
        help="Curated sentence matches. Only matched groups are sent to TypeSafe.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="New annotation JSON file, including partial results if a request fails.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Exact TypeSafe model version, not an alias such as jev-latest.",
    ),
    timeout: float = typer.Option(
        30.0,
        "--timeout",
        help="Positive, finite timeout in seconds for each request. No retries.",
    ),
    max_groups: int = typer.Option(
        100,
        "--max-groups",
        min=1,
        help="Refuse runs exceeding this number of matched groups before any request.",
    ),
) -> None:
    """Send matched text to TypeSafe for optional review hints; never edit translations."""
    try:
        validate_new_output_path(output, [review_input, matches])
        bundle = load_review_bundle(review_input)
        matching = load_matches(matches, bundle)
        matched_count = sum(
            group.status == "matched"
            for chunk in matching.chunks
            for group in chunk.groups
        )
        console.print(
            f"Matched groups eligible: {matched_count}. "
            "Unchecked groups will not be sent to TypeSafe."
        )
        annotations = annotate_groups(
            bundle,
            matching,
            api_key=os.environ.get("TYPESAFE_API_KEY", ""),
            model_name=model,
            timeout=timeout,
            max_groups=max_groups,
        )
        save_annotations(annotations, output, bundle, matching)
    except (OSError, ValueError) as e:
        console.print(f"[red]Could not annotate review:[/red] {escape(str(e))}")
        raise typer.Exit(1) from e
    counts = {
        status: sum(group.status == status for group in annotations.groups)
        for status in ("completed", "skipped_unchecked", "error", "not_attempted")
    }
    console.print(
        f"Annotations saved to {escape(str(output))}. "
        f"{counts['completed']} assessed, {counts['skipped_unchecked']} unchecked skipped, "
        f"{counts['error']} failed, {counts['not_attempted']} not attempted. "
        "These are limited review hints, not a translation quality verdict."
    )
    if counts["error"] or counts["not_attempted"]:
        failed = next(
            (row for row in annotations.groups if row.status == "error"), None
        )
        if failed is not None and failed.error is not None:
            console.print(
                f"First failure: chunk {failed.unit_id}, group {failed.group_index}: "
                f"{failed.error.replace('_', ' ')}."
            )
        console.print(
            "[yellow]Review stopped after a request failed or was interrupted. "
            "Completed results are saved; no automatic retry was made.[/yellow]"
        )
        raise typer.Exit(1)


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
    annotations: Path | None = typer.Option(
        None,
        "--annotations",
        help="Optional saved TypeSafe annotations. Requires --matches; runs offline.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Create an offline side-by-side report for human review, without model calls."""
    try:
        if annotations is not None and matches is None:
            raise ValueError("--annotations requires the original --matches file")
        protected_paths = [review_input]
        if matches is not None:
            protected_paths.append(matches)
        if annotations is not None:
            protected_paths.append(annotations)
        validate_new_output_path(output, protected_paths)
        bundle = load_review_bundle(review_input)
        matching = load_matches(matches, bundle) if matches is not None else None
        assessment = (
            load_annotations(annotations, bundle, matching)
            if annotations is not None and matching is not None
            else None
        )
        write_review_report(bundle, output, matches=matching, annotations=assessment)
    except (OSError, ValueError) as e:
        console.print(f"[red]Could not create review report:[/red] {escape(str(e))}")
        raise typer.Exit(1) from e
    notice = (
        "Saved TypeSafe review hints are shown; report generation made no model calls."
        if annotations is not None
        else "No automated quality assessment was performed."
    )
    console.print(
        f"Review report saved to {escape(str(output))} ({len(bundle.pairs)} chunks). "
        f"Open it in a browser. {notice}"
    )
