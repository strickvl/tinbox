"""Preserved translation chunks and offline reports for human review."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from tinbox.core.review_annotations import AnnotationGroup, ReviewAnnotations
    from tinbox.core.review_matches import ChunkMatches, ReviewMatches


class ReviewPair(BaseModel):
    """One source chunk and the exact translation returned for it."""

    unit_id: int = Field(ge=1)
    source: str
    translation: str

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class ReviewBundle(BaseModel):
    """Versioned input to an offline review report, independent of provider SDKs."""

    schema_version: Literal[1] = 1
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    source_name: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_lang: str
    target_lang: str
    provider: str
    model_name: str
    algorithm: Literal["context-aware"] = "context-aware"
    context_size: int = Field(gt=0)
    custom_split_token: str | None = None
    use_glossary: bool = False
    reasoning_effort: str = "minimal"
    pairs: list[ReviewPair] = Field(min_length=1)

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("created_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_pair_order(self) -> Self:
        if any(pair.unit_id != index for index, pair in enumerate(self.pairs, 1)):
            raise ValueError("Pair unit IDs must be ordered consecutively from 1")
        return self


def validate_new_output_path(path: Path, protected_paths: Sequence[Path] = ()) -> None:
    """Require a new artifact path without replacing inputs or existing files."""
    resolved = path.resolve()
    if any(resolved == protected.resolve() for protected in protected_paths):
        raise ValueError(f"Review output must use a separate path: {path}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Review output already exists: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(
            f"Review output directory does not exist: {path.parent}"
        )
    if not path.parent.is_dir():
        raise NotADirectoryError(
            f"Review output parent is not a directory: {path.parent}"
        )


def _write_new_text(path: Path, text: str) -> None:
    validate_new_output_path(path)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
        # Linking a complete file publishes it atomically and refuses to overwrite.
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_review_bundle(bundle: ReviewBundle, path: Path) -> None:
    """Save a new UTF-8 JSON bundle without replacing an existing artifact."""
    _write_new_text(path, bundle.model_dump_json(indent=2) + "\n")


def load_review_bundle(path: Path) -> ReviewBundle:
    """Validate an on-disk bundle before it enters the report renderer."""
    bundle = ReviewBundle.model_validate_json(path.read_text(encoding="utf-8"))
    if not {"schema_version", "created_at"} <= bundle.model_fields_set:
        raise ValueError("Review bundle must include schema_version and created_at")
    return bundle


def _render_pair(
    source: str, translation: str, source_lang: str, target_lang: str
) -> str:
    return f"""<div class="pair">
    <div class="column"><h3>Source <span>{source_lang}</span></h3>
      <div class="chunk-text" lang="{source_lang}" dir="auto">{escape(source)}</div></div>
    <div class="column"><h3>Translation <span>{target_lang}</span></h3>
      <div class="chunk-text" lang="{target_lang}" dir="auto">{escape(translation)}</div></div>
  </div>"""


def _render_matched_chunk(
    pair: ReviewPair,
    matches: ChunkMatches,
    source_lang: str,
    target_lang: str,
    annotations: dict[tuple[int, int], AnnotationGroup] | None = None,
    threshold: float = 0.70,
) -> str:
    groups = []
    for index, group in enumerate(matches.groups, 1):
        label = "Matched for comparison" if group.status == "matched" else "Unchecked"
        source = "".join(pair.source[span.start : span.end] for span in group.source)
        translation = "".join(
            pair.translation[span.start : span.end] for span in group.translation
        )
        reason = (
            f'<p class="match-reason" dir="auto">{escape(group.reason)}</p>'
            if group.reason
            else ""
        )
        missing = []
        if not group.source:
            missing.append("No source text in this region.")
        if not group.translation:
            missing.append("No translation text in this region.")
        missing_note = (
            f'<p class="match-reason">{" ".join(missing)}</p>' if missing else ""
        )
        annotation = (
            _render_annotation(annotations[(pair.unit_id, index)], threshold)
            if annotations is not None
            else ""
        )
        groups.append(f"""<div class="match-group {group.status}" id="chunk-{pair.unit_id}-group-{index}">
    <h3 class="match-heading"><a href="#chunk-{pair.unit_id}-group-{index}">Group {index}</a> · {label}</h3>
    {reason}{missing_note}
    {_render_pair(source, translation, source_lang, target_lang)}
    {annotation}
  </div>""")
    original = _render_pair(pair.source, pair.translation, source_lang, target_lang)
    return f"""<section class="chunk" id="chunk-{pair.unit_id}">
  <h2><a href="#chunk-{pair.unit_id}">Chunk {pair.unit_id}</a></h2>
  {"\n".join(groups)}
  <details class="original-chunk">
    <summary>Full original chunk pair</summary>
    {original}
  </details>
</section>"""


def _render_annotation(row: AnnotationGroup, threshold: float) -> str:
    from tinbox.core.review_annotations import annotation_scores

    scores = annotation_scores(row)
    if row.status == "completed":
        concern = any(score >= threshold for score in scores.values())
        label = "TypeSafe concern" if concern else "No concern flagged by these checks"
        state = "concern" if concern else "completed"
        labels = {
            "participants": "Participant roles",
            "certainty": "Certainty",
            "quantifiers": "Quantifiers",
        }
        score_list = "".join(
            f"<div><dt>{label}</dt><dd>{scores[key]!r}</dd></div>"
            for key, label in labels.items()
        )
        body = f'<dl class="annotation-scores">{score_list}</dl>'
    else:
        state = row.status
        label = {
            "skipped_unchecked": "Not assessed: unchecked match",
            "error": "Request failed",
            "not_attempted": "Not attempted",
        }[row.status]
        errors = {
            "authentication_error": "Authentication failed. Check TYPESAFE_API_KEY.",
            "rate_limited": "TypeSafe limited this request. Try again later.",
            "service_error": "TypeSafe returned a service error.",
            "transport_error": "The request could not complete; its remote outcome may be unknown.",
            "invalid_response": "TypeSafe returned a response Tinbox could not validate.",
            "interrupted_outcome_unknown": "The request was interrupted; its remote outcome is unknown.",
        }
        body = (
            f'<p dir="auto">{escape(errors[row.error])} ({escape(row.error)})</p>'
            if row.error
            else ""
        )
    return f"""<aside class="annotation {state}" aria-label="TypeSafe annotation">
      <p class="annotation-label">{label}</p>{body}
    </aside>"""


def _render_annotation_summary(annotations: ReviewAnnotations) -> str:
    from tinbox.core.review_annotations import annotation_scores

    completed = sum(row.status == "completed" for row in annotations.groups)
    flagged = sum(
        any(score >= annotations.threshold for score in annotation_scores(row).values())
        for row in annotations.groups
        if row.status == "completed"
    )
    skipped = sum(row.status == "skipped_unchecked" for row in annotations.groups)
    errors = sum(row.status == "error" for row in annotations.groups)
    error_label = "request error" if errors == 1 else "request errors"
    not_attempted = sum(row.status == "not_attempted" for row in annotations.groups)
    definitions = escape(
        json.dumps(annotations.questions, ensure_ascii=False, indent=2)
    )
    return f"""<section class="annotation-summary" aria-label="TypeSafe review summary">
      <h2>Optional TypeSafe checks</h2>
      <p><strong>{completed} completed · {flagged} flagged · {skipped} skipped unchecked · {errors} {error_label} · {not_attempted} not attempted.</strong></p>
      <p>Scores estimate the probability that a participant role, certainty, or quantifier changed.
        Higher scores suggest a possible change, not its severity. A score at or above {annotations.threshold!r} flags a concern.
        This experimental threshold is not a translation quality score or a pass result.
        These limited checks can miss errors. Review the text yourself even when no concern is flagged.</p>
      <details>
        <summary>TypeSafe model, timing, and check definitions</summary>
        <dl>
          <div><dt>Model</dt><dd dir="auto">{escape(annotations.model_name)}</dd></div>
          <div><dt>Started (UTC)</dt><dd>{annotations.created_at.isoformat()}</dd></div>
          <div><dt>Completed (UTC)</dt><dd>{annotations.completed_at.isoformat()}</dd></div>
          <div><dt>Concern threshold</dt><dd>{annotations.threshold!r}</dd></div>
        </dl>
        <pre class="check-definitions" dir="auto">{definitions}</pre>
      </details>
    </section>"""


def write_review_report(
    bundle: ReviewBundle,
    path: Path,
    *,
    matches: ReviewMatches | None = None,
    annotations: ReviewAnnotations | None = None,
) -> None:
    """Write a self-contained HTML report without calling a model or modifying text."""
    source_lang = escape(bundle.source_lang, quote=True)
    target_lang = escape(bundle.target_lang, quote=True)
    if matches is not None:
        from tinbox.core.review_matches import validate_matches

        validate_matches(bundle, matches)
    annotation_rows = None
    annotation_summary = ""
    if annotations is not None:
        from tinbox.core.review_annotations import validate_annotations

        if matches is None:
            raise ValueError("TypeSafe annotations require their matching file")
        validate_annotations(bundle, matches, annotations)
        annotation_rows = {
            (row.unit_id, row.group_index): row for row in annotations.groups
        }
        annotation_summary = _render_annotation_summary(annotations)
    assessment_notice = (
        "Saved TypeSafe annotations appear separately below each group. "
        "They are limited checks for human review, not a quality verdict."
        if annotations is not None
        else "No automated quality assessment has been performed."
    )
    sections = []
    for index, pair in enumerate(bundle.pairs):
        if matches is not None:
            sections.append(
                _render_matched_chunk(
                    pair,
                    matches.chunks[index],
                    source_lang,
                    target_lang,
                    annotation_rows,
                    annotations.threshold if annotations is not None else 0.70,
                )
            )
            continue
        sections.append(f"""<section class="chunk" id="chunk-{pair.unit_id}">
  <h2><a href="#chunk-{pair.unit_id}">Chunk {pair.unit_id}</a></h2>
  {_render_pair(pair.source, pair.translation, source_lang, target_lang)}
</section>""")
    notice = f"""<strong>{len(bundle.pairs)} paired chunks.</strong> Read each source chunk beside its translation.
      These are the chunks used during translation, not aligned sentences.
      {assessment_notice}"""
    details = [
        ("Source file", bundle.source_name),
        ("Languages", f"{bundle.source_lang} → {bundle.target_lang}"),
        ("Model", f"{bundle.provider}:{bundle.model_name}"),
        ("Created (UTC)", bundle.created_at.strftime("%Y-%m-%d %H:%M:%S")),
        ("Algorithm", bundle.algorithm),
        ("Target chunk size", f"{bundle.context_size:,} characters"),
        ("Glossary", "Enabled" if bundle.use_glossary else "Disabled"),
        ("Reasoning effort", bundle.reasoning_effort),
    ]
    if matches is not None:
        matched_count = sum(
            group.status == "matched"
            for chunk in matches.chunks
            for group in chunk.groups
        )
        unchecked_count = sum(
            group.status == "unchecked"
            for chunk in matches.chunks
            for group in chunk.groups
        )
        provenance = {
            "unreviewed": "Unreviewed draft; no sentence matches have been selected.",
            "assistant-curated": "Assistant-curated matches; not human-verified.",
            "human-reviewed": "Human-reviewed matches; this is not a quality verdict.",
        }[matches.provenance]
        group_label = "group" if matched_count == 1 else "groups"
        region_label = "region" if unchecked_count == 1 else "regions"
        notice = f"""<strong>{matched_count} matched {group_label} · {unchecked_count} unchecked {region_label}.</strong>
      Matched groups are selected for comparison, not certified as correct translations.
      Unchecked regions remain visible. {provenance}
      {assessment_notice}"""
        details.extend(
            [
                ("Match provenance", provenance),
                (
                    "Matches created (UTC)",
                    matches.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            ]
        )
    if bundle.custom_split_token is not None:
        details.append(
            ("Split token (overrides chunk size)", bundle.custom_split_token)
        )
    metadata = "\n".join(
        f'<div><dt>{escape(label)}</dt><dd dir="auto">{escape(value)}</dd></div>'
        for label, value in details
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
  <title>Translation review: {escape(bundle.source_name)}</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #182c37; background: #f4f6f7; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 2rem 1.25rem 4rem; }}
    main {{ max-width: 90rem; margin: 0 auto; }}
    header {{ margin: 0 0 2rem; }}
    .eyebrow {{ color: #386576; font-size: .8rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
    h1 {{ margin: .5rem 0; font-size: clamp(1.8rem, 4vw, 2.6rem); line-height: 1.2; }}
    .filename {{ overflow-wrap: anywhere; color: #425866; margin-top: .5rem; }}
    .notice {{ background: #e5eff3; border-left: 4px solid #547f90; padding: 1rem 1.25rem; line-height: 1.6; }}
    details {{ margin-top: 1.25rem; }}
    summary {{ cursor: pointer; font-weight: 600; padding: .4rem 0; }}
    dl {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1.25rem; }}
    dt {{ color: #425866; font-size: .8rem; }}
    dd {{ margin: .25rem 0 0; overflow-wrap: anywhere; white-space: pre-wrap; }}
    .fingerprint {{ font-size: .85rem; overflow-wrap: anywhere; }}
    .chunk {{ background: #fff; border: 1px solid #d7e0e4; border-radius: .6rem; margin: 0 0 1.5rem; overflow: hidden; break-inside: avoid; }}
    h2 {{ margin: 0; padding: .8rem 1.25rem; background: #edf2f4; font-size: .9rem; }}
    a {{ color: inherit; text-decoration: none; }}
    a:hover, a:focus-visible {{ text-decoration: underline; }}
    .pair {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }}
    .column {{ min-width: 0; padding: 1.25rem; }}
    .column + .column {{ border-left: 1px solid #d7e0e4; }}
    h3 {{ margin: 0 0 1rem; font-size: .8rem; color: #425866; text-transform: uppercase; letter-spacing: .05em; }}
    h3 span {{ font-weight: 400; margin-left: .5rem; }}
    .chunk-text {{ font-family: inherit; font-size: 1rem; line-height: 1.85; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; unicode-bidi: plaintext; text-align: start; }}
    .match-group + .match-group {{ border-top: 1px solid #d7e0e4; }}
    .match-heading {{ margin: 0; padding: .8rem 1.25rem; background: #f4f7f8; text-transform: none; letter-spacing: 0; }}
    .unchecked .match-heading {{ background: #faf4e5; }}
    .match-reason {{ margin: 0; padding: .6rem 1.25rem 0; color: #425866; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .original-chunk {{ margin: 0; border-top: 1px solid #d7e0e4; }}
    .original-chunk summary {{ padding: .8rem 1.25rem; color: #425866; }}
    .annotation-summary {{ margin-top: 1.25rem; border: 1px solid #d7e0e4; border-radius: .6rem; padding: 1rem 1.25rem; background: #fff; line-height: 1.6; }}
    .annotation-summary h2 {{ padding: 0; background: transparent; font-size: 1rem; }}
    .annotation {{ margin: 0 1.25rem 1.25rem; padding: .8rem 1rem; background: #f4f7f8; border-left: 3px solid #8c9ca5; line-height: 1.6; }}
    .annotation.concern {{ background: #fff2dd; border-color: #a56d16; }}
    .annotation.error {{ background: #fbeeed; border-color: #a65049; }}
    .annotation p {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .annotation-label {{ font-weight: 600; }}
    .annotation-scores {{ margin: .5rem 0 0; grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr)); gap: .5rem; }}
    .annotation-scores dd {{ font-variant-numeric: tabular-nums; }}
    .check-definitions {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: .85rem; }}
    footer {{ color: #425866; font-size: .85rem; line-height: 1.6; }}
    @media (max-width: 44rem) {{
      body {{ padding: 1.25rem .75rem 2rem; }}
      .pair {{ grid-template-columns: minmax(0, 1fr); }}
      .column + .column {{ border-left: 0; border-top: 1px solid #d7e0e4; }}
    }}
    @media print {{ body {{ padding: 0; background: white; }} details {{ display: none; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">Tinbox · Review only</p>
    <h1>Translation review</h1>
    <p class="filename" dir="auto">{escape(bundle.source_name)}</p>
    <p class="notice">{notice}</p>
    {annotation_summary}
    <details>
      <summary>Translation details</summary>
      <dl>{metadata}</dl>
      <p class="fingerprint">Original source file SHA-256: <code>{escape(bundle.source_sha256)}</code></p>
      <p class="fingerprint">Review bundle schema: {bundle.schema_version}</p>
    </details>
  </header>
  {"\n".join(sections)}
  <footer>This report preserves the saved source chunks and translations for human review.
    Chunking can change source whitespace or omit split tokens. Context sent alongside a chunk is not shown here.
    The report runs offline, makes no model requests, and does not change your translation.</footer>
</main>
</body>
</html>
"""
    _write_new_text(path, html)
