"""Explicit sentence groups attached to an unchanged translation review bundle."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
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
    from tinbox.core.review import ReviewBundle


class TextSpan(BaseModel):
    """A half-open range of Python Unicode code points in one saved chunk."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("Text span end must be greater than start")
        return self


class MatchGroup(BaseModel):
    """Corresponding sentence spans or text explicitly left unchecked."""

    status: Literal["matched", "unchecked"]
    source: list[TextSpan]
    translation: list[TextSpan]
    reason: str = ""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "matched" and (not self.source or not self.translation):
            raise ValueError("Matched groups must include source and translation spans")
        if self.status == "unchecked" and not self.reason.strip():
            raise ValueError("Unchecked groups must include a nonblank reason")
        return self


class ChunkMatches(BaseModel):
    """Ordered groups within one preserved source/translation pair."""

    unit_id: int = Field(ge=1)
    groups: list[MatchGroup] = Field(min_length=1)

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class ReviewMatches(BaseModel):
    """A versioned matching sidecar; matched does not mean translation quality passed."""

    schema_version: Literal[1] = 1
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: Literal["unreviewed", "assistant-curated", "human-reviewed"]
    chunks: list[ChunkMatches] = Field(min_length=1)

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
    def validate_provenance(self) -> Self:
        if self.provenance == "unreviewed" and any(
            group.status == "matched" for chunk in self.chunks for group in chunk.groups
        ):
            raise ValueError(
                "Matched groups require assistant-curated or human-reviewed provenance"
            )
        return self


def bundle_digest(bundle: ReviewBundle) -> str:
    """Hash all bundle fields, independent of JSON indentation and key order."""
    canonical = json.dumps(
        bundle.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_matches(bundle: ReviewBundle, matches: ReviewMatches) -> None:
    """Require an exact bundle identity and complete, ordered coverage of both sides."""
    if matches.bundle_sha256 != bundle_digest(bundle):
        raise ValueError(
            "Sentence matches belong to a different or changed review bundle"
        )
    if [chunk.unit_id for chunk in matches.chunks] != [
        pair.unit_id for pair in bundle.pairs
    ]:
        raise ValueError(
            "Sentence matches must include every chunk exactly once in order"
        )
    for pair, chunk in zip(bundle.pairs, matches.chunks, strict=True):
        for side, text, spans in (
            (
                "source",
                pair.source,
                [s for group in chunk.groups for s in group.source],
            ),
            (
                "translation",
                pair.translation,
                [s for group in chunk.groups for s in group.translation],
            ),
        ):
            end = 0
            for span in spans:
                if span.start != end or span.end > len(text):
                    raise ValueError(
                        f"Chunk {pair.unit_id} {side} spans must cover the text in order "
                        "without gaps, overlaps, or out-of-range offsets"
                    )
                end = span.end
            if end != len(text):
                raise ValueError(
                    f"Chunk {pair.unit_id} {side} spans must cover the entire text; "
                    "represent unmatched text in unchecked groups"
                )


def load_matches(path: Path, bundle: ReviewBundle) -> ReviewMatches:
    """Validate a saved sidecar before using any of its text references."""
    matches = ReviewMatches.model_validate_json(path.read_text(encoding="utf-8"))
    if not {"schema_version", "created_at", "provenance"} <= matches.model_fields_set:
        raise ValueError(
            "Sentence matches must include schema_version, created_at, and provenance"
        )
    validate_matches(bundle, matches)
    return matches


def save_matches(matches: ReviewMatches, path: Path, bundle: ReviewBundle) -> None:
    """Publish a valid sidecar atomically without replacing an existing artifact."""
    from tinbox.core.review import _write_new_text

    validate_matches(bundle, matches)
    _write_new_text(path, matches.model_dump_json(indent=2) + "\n")


def unchecked_matches(bundle: ReviewBundle) -> ReviewMatches:
    """Start with all text visible and unchecked, without inferring sentence matches."""
    return ReviewMatches(
        bundle_sha256=bundle_digest(bundle),
        provenance="unreviewed",
        chunks=[
            ChunkMatches(
                unit_id=pair.unit_id,
                groups=[
                    MatchGroup(
                        status="unchecked",
                        source=[TextSpan(start=0, end=len(pair.source))]
                        if pair.source
                        else [],
                        translation=[TextSpan(start=0, end=len(pair.translation))]
                        if pair.translation
                        else [],
                        reason="Needs manual sentence matching.",
                    )
                ],
            )
            for pair in bundle.pairs
        ],
    )
