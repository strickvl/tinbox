"""Preserved translation chunks and offline reports for human review."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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


def write_review_report(bundle: ReviewBundle, path: Path) -> None:
    """Write a self-contained HTML report without calling a model or modifying text."""
    source_lang = escape(bundle.source_lang, quote=True)
    target_lang = escape(bundle.target_lang, quote=True)
    sections = []
    for pair in bundle.pairs:
        sections.append(f"""<section class="chunk" id="chunk-{pair.unit_id}">
  <h2><a href="#chunk-{pair.unit_id}">Chunk {pair.unit_id}</a></h2>
  <div class="pair">
    <div class="column"><h3>Source <span>{source_lang}</span></h3>
      <div class="chunk-text" lang="{source_lang}" dir="auto">{escape(pair.source)}</div></div>
    <div class="column"><h3>Translation <span>{target_lang}</span></h3>
      <div class="chunk-text" lang="{target_lang}" dir="auto">{escape(pair.translation)}</div></div>
  </div>
</section>""")
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
    <p class="notice"><strong>{len(bundle.pairs)} paired chunks.</strong> Read each source chunk beside its translation.
      These are the chunks used during translation, not aligned sentences.
      No automated quality assessment has been performed.</p>
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
