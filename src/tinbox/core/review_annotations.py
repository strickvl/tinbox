"""Bounded, opt-in TypeSafe checks over explicitly matched translation text."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from typing import Any, Literal, Self
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from tinbox.core.review import ReviewBundle, _write_new_text
from tinbox.core.review_matches import (
    MatchGroup,
    ReviewMatches,
    bundle_digest,
    validate_matches,
)
from tinbox.core.review_questions import QUESTIONS

DEFAULT_MODEL = "jev-1.14.0"
REVIEW_THRESHOLD = 0.70
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_REQUEST_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024

AnnotationError = Literal[
    "authentication_error",
    "rate_limited",
    "service_error",
    "transport_error",
    "invalid_response",
    "interrupted_outcome_unknown",
]


class AnnotationGroup(BaseModel):
    """One attempted, skipped, or unattempted group, without a quality verdict."""

    unit_id: int = Field(ge=1)
    group_index: int = Field(ge=1)
    status: Literal["completed", "error", "not_attempted", "skipped_unchecked"]
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    error: AnnotationError | None = None
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", revalidate_instances="always"
    )

    @field_validator("started_at", "completed_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status in ("skipped_unchecked", "not_attempted"):
            if any(
                value is not None
                for value in (
                    self.request,
                    self.response,
                    self.error,
                    self.started_at,
                    self.completed_at,
                )
            ):
                raise ValueError(
                    "Unattempted groups cannot contain a request or result"
                )
            return self
        if self.request is None or self.started_at is None or self.completed_at is None:
            raise ValueError("Attempted groups require a request and both timestamps")
        if self.completed_at < self.started_at:
            raise ValueError("Annotation completion cannot precede its start")
        if self.status == "completed":
            if self.response is None or self.error is not None:
                raise ValueError(
                    "Completed annotations require a response and no error"
                )
        elif self.response is not None or self.error is None:
            raise ValueError("Failed annotations require an error and no response")
        return self


class ReviewAnnotations(BaseModel):
    """An audit record bound to both the exact bundle and its matching decisions."""

    schema_version: Literal[1] = 1
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: AwareDatetime
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matches_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name: str = Field(min_length=1)
    threshold: float = Field(default=REVIEW_THRESHOLD, ge=0.70, le=0.70)
    questions: dict[str, Any]
    groups: list[AnnotationGroup] = Field(min_length=1)

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", revalidate_instances="always"
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("created_at", "completed_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("Annotation model must be a nonblank model name")
        if value.endswith("-latest"):
            raise ValueError("Annotation model must use an exact version, not -latest")
        return value

    @model_validator(mode="after")
    def validate_metadata(self) -> Self:
        if self.completed_at < self.created_at:
            raise ValueError("Run completion cannot precede its start")
        if self.questions != QUESTIONS:
            raise ValueError("Annotations must record the exact supported questions")
        return self


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def matches_digest(matches: ReviewMatches) -> str:
    """Hash every matching field, including provenance, reasons and span boundaries."""
    return hashlib.sha256(_json_bytes(matches.model_dump(mode="json"))).hexdigest()


def _request_for(
    bundle: ReviewBundle,
    unit_index: int,
    group: MatchGroup,
    model_name: str,
) -> dict[str, Any]:
    pair = bundle.pairs[unit_index]
    return {
        "model": model_name,
        "state": {
            "source_text": "".join(
                pair.source[span.start : span.end] for span in group.source
            ),
            "translation": "".join(
                pair.translation[span.start : span.end] for span in group.translation
            ),
            "source_language": bundle.source_lang,
            "target_language": bundle.target_lang,
        },
        "questions": deepcopy(QUESTIONS),
    }


def _response_scores(response: dict[str, Any], model_name: str) -> dict[str, float]:
    if response.get("model") != model_name:
        raise ValueError("Annotation response model does not match the requested model")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(QUESTIONS):
        raise ValueError(
            "Annotation response must contain exactly the expected answers"
        )
    scores = {}
    for name, answer in answers.items():
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ValueError("Annotation answers must use the noul type")
        value = answer.get("noul")
        if (
            type(value) not in (int, float)
            or not 0 <= value <= 1
            or not math.isfinite(value)
        ):
            raise ValueError(
                "Annotation scores must be finite numbers from zero to one"
            )
        scores[name] = float(value)
    usage = response.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("input_tokens", "output_tokens")
    ):
        raise ValueError(
            "Annotation usage must contain nonnegative integer token counts"
        )
    _json_bytes(response)
    return scores


def annotation_scores(row: AnnotationGroup) -> dict[str, float]:
    """Return recorded probabilities only for a successfully completed request."""
    if row.status != "completed":
        return {}
    assert row.request is not None and row.response is not None
    return _response_scores(row.response, row.request["model"])


def validate_annotations(
    bundle: ReviewBundle, matches: ReviewMatches, annotations: ReviewAnnotations
) -> None:
    """Reject stale, incomplete, or inconsistent annotation sidecars."""
    annotations = ReviewAnnotations.model_validate(annotations)
    validate_matches(bundle, matches)
    if annotations.bundle_sha256 != bundle_digest(bundle):
        raise ValueError("Annotations belong to a different or changed review bundle")
    if annotations.matches_sha256 != matches_digest(matches):
        raise ValueError("Annotations belong to different or changed sentence matches")
    expected = [
        (unit_index, chunk.unit_id, group_index, group)
        for unit_index, chunk in enumerate(matches.chunks)
        for group_index, group in enumerate(chunk.groups, 1)
    ]
    if [(row.unit_id, row.group_index) for row in annotations.groups] != [
        (unit_id, group_index) for _, unit_id, group_index, _ in expected
    ]:
        raise ValueError("Annotations must include every group exactly once in order")
    failed = False
    previous_end = annotations.created_at
    for row, (unit_index, _, _, group) in zip(
        annotations.groups, expected, strict=True
    ):
        if group.status == "unchecked":
            if row.status != "skipped_unchecked":
                raise ValueError("Unchecked matches must never be annotated")
            continue
        if row.status == "skipped_unchecked":
            raise ValueError("Matched groups cannot be marked skipped_unchecked")
        if failed:
            if row.status != "not_attempted":
                raise ValueError(
                    "Annotation requests must stop after the first failure"
                )
            continue
        if row.status == "not_attempted":
            raise ValueError("Unattempted matches require a preceding failure")
        if row.request != _request_for(
            bundle, unit_index, group, annotations.model_name
        ):
            raise ValueError("Annotation request differs from its exact matched text")
        assert row.started_at is not None and row.completed_at is not None
        if row.started_at < previous_end or row.completed_at > annotations.completed_at:
            raise ValueError("Annotation timestamps must follow the recorded run order")
        previous_end = row.completed_at
        if row.status == "completed":
            annotation_scores(row)
        else:
            failed = True


def _reject_constant(value: str) -> None:
    raise ValueError("Nonfinite JSON number")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _decode_json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except RecursionError:
        raise ValueError("JSON nesting exceeds the supported depth") from None
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    _json_bytes(value)
    return value


def load_annotations(
    path: Path, bundle: ReviewBundle, matches: ReviewMatches
) -> ReviewAnnotations:
    """Load an offline annotation artifact without contacting TypeSafe."""
    data = path.read_bytes()
    raw = _decode_json(data)
    if not set(ReviewAnnotations.model_fields) <= raw.keys():
        raise ValueError("Annotations must explicitly record all schema metadata")
    annotations = ReviewAnnotations.model_validate_json(data)
    validate_annotations(bundle, matches, annotations)
    return annotations


def save_annotations(
    annotations: ReviewAnnotations,
    path: Path,
    bundle: ReviewBundle,
    matches: ReviewMatches,
) -> None:
    """Publish an annotation artifact atomically, never replacing an existing file."""
    validate_annotations(bundle, matches, annotations)
    _write_new_text(path, annotations.model_dump_json(indent=2) + "\n")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _send_request(body: bytes, api_key: str, timeout: float) -> dict[str, Any]:
    request = Request(
        ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("Annotation response exceeded the size limit")
    return _decode_json(data)


def annotate_review(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    *,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    max_groups: int = 100,
) -> ReviewAnnotations:
    """Check matched groups once each; retain partial work and stop on any failure."""
    validate_matches(bundle, matches)
    try:
        timeout_value = float(timeout) if type(timeout) in (float, int) else math.nan
    except OverflowError:
        timeout_value = math.nan
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("Annotation timeout must be a finite positive number")
    if type(max_groups) is not int or max_groups < 1:
        raise ValueError("Annotation group limit must be a positive integer")
    if not isinstance(model_name, str):
        raise ValueError("Annotation model must be a nonblank model name")
    ReviewAnnotations.validate_model_name(model_name)
    requests = [
        _request_for(bundle, unit_index, group, model_name)
        if group.status == "matched"
        else None
        for unit_index, chunk in enumerate(matches.chunks)
        for group in chunk.groups
    ]
    count = sum(request is not None for request in requests)
    if count > max_groups:
        raise ValueError(f"Matched groups exceed the annotation limit of {max_groups}")
    if count and (
        not isinstance(api_key, str)
        or not api_key
        or any(not 33 <= ord(character) <= 126 for character in api_key)
    ):
        raise ValueError("A valid TYPESAFE_API_KEY is required for matched groups")
    bodies = [
        _json_bytes(request) if request is not None else None for request in requests
    ]
    if any(body is not None and len(body) > MAX_REQUEST_BYTES for body in bodies):
        raise ValueError(
            f"A matched group exceeds the {MAX_REQUEST_BYTES}-byte request limit"
        )
    created_at = datetime.now(UTC)
    rows = [
        AnnotationGroup(
            unit_id=chunk.unit_id,
            group_index=group_index,
            status="skipped_unchecked"
            if group.status == "unchecked"
            else "not_attempted",
        )
        for chunk in matches.chunks
        for group_index, group in enumerate(chunk.groups, 1)
    ]
    for index, (request, body) in enumerate(zip(requests, bodies, strict=True)):
        if request is None or body is None:
            continue
        started_at = datetime.now(UTC)
        response = None
        error: AnnotationError | None = None
        try:
            response = _send_request(body, api_key, timeout_value)
            _response_scores(response, model_name)
        except KeyboardInterrupt:
            error = "interrupted_outcome_unknown"
        except HTTPError as failure:
            if failure.code in (401, 403):
                error = "authentication_error"
            elif failure.code in (429, 529):
                error = "rate_limited"
            else:
                error = "service_error"
        except (URLError, OSError, HTTPException):
            error = "transport_error"
        except (ValueError, TypeError, RecursionError, OverflowError):
            error = "invalid_response"
        rows[index] = AnnotationGroup(
            unit_id=rows[index].unit_id,
            group_index=rows[index].group_index,
            status="error" if error else "completed",
            request=request,
            response=response if error is None else None,
            error=error,
            started_at=started_at,
            completed_at=datetime.now(UTC),
        )
        if error is not None:
            break
    annotations = ReviewAnnotations(
        created_at=created_at,
        completed_at=datetime.now(UTC),
        bundle_sha256=bundle_digest(bundle),
        matches_sha256=matches_digest(matches),
        model_name=model_name,
        questions=deepcopy(QUESTIONS),
        groups=rows,
    )
    validate_annotations(bundle, matches, annotations)
    return annotations
