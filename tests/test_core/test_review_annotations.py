"""Offline evidence for bounded TypeSafe requests and exact artifact binding."""

import io
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
from pydantic import ValidationError

from tinbox.core import review_annotations as annotations_module
from tinbox.core.review import ReviewBundle, ReviewPair
from tinbox.core.review_annotations import (
    DEFAULT_MODEL,
    ENDPOINT,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    QUESTIONS,
    AnnotationGroup,
    ReviewAnnotations,
    annotate_review,
    annotation_scores,
    load_annotations,
    save_annotations,
    validate_annotations,
)
from tinbox.core.review_matches import (
    ChunkMatches,
    MatchGroup,
    ReviewMatches,
    TextSpan,
    bundle_digest,
    unchecked_matches,
)


@pytest.fixture
def bundle() -> ReviewBundle:
    return ReviewBundle(
        source_name="private-filename.txt",
        source_sha256="ab" * 32,
        source_lang="de",
        target_lang="ar",
        provider="openai",
        model_name="translation-model",
        context_size=100,
        pairs=[
            ReviewPair(
                unit_id=1,
                source="PRIVATE HEADING\nA. B. C.",
                translation="سري\nأ. ب. ج.",
            )
        ],
    )


@pytest.fixture
def matches(bundle: ReviewBundle) -> ReviewMatches:
    source_parts = ["PRIVATE HEADING\n", "A. ", "B. ", "C."]
    translation_parts = ["سري\n", "أ. ", "ب. ", "ج."]
    source_start = translation_start = 0
    groups = []
    for index, (source, translation) in enumerate(
        zip(source_parts, translation_parts, strict=True)
    ):
        groups.append(
            MatchGroup(
                status="unchecked" if index == 0 else "matched",
                source=[TextSpan(start=source_start, end=source_start + len(source))],
                translation=[
                    TextSpan(
                        start=translation_start,
                        end=translation_start + len(translation),
                    )
                ],
                reason="PRIVATE MATCHING NOTE",
            )
        )
        source_start += len(source)
        translation_start += len(translation)
    return ReviewMatches(
        bundle_sha256=bundle_digest(bundle),
        provenance="assistant-curated",
        chunks=[ChunkMatches(unit_id=1, groups=groups)],
    )


@pytest.fixture
def response() -> dict[str, Any]:
    return {
        "model": DEFAULT_MODEL,
        "answers": {
            "participants": {"type": "noul", "noul": 0.8, "extra": "kept"},
            "certainty": {"type": "noul", "noul": 0.5},
            "quantifiers": {"type": "noul", "noul": 0},
        },
        "usage": {"input_tokens": 123, "output_tokens": 12, "extra": 3},
        "request_id": "provider-metadata-kept",
    }


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]) -> Mock:
    opener = Mock()
    opener.open.side_effect = lambda *args, **kwargs: io.BytesIO(
        json.dumps(response).encode("utf-8")
    )
    monkeypatch.setattr(annotations_module, "build_opener", Mock(return_value=opener))
    return opener.open


def test_sends_exact_matched_text_only_and_keeps_full_response(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    response: dict[str, Any],
    network: Mock,
) -> None:
    original_bundle = bundle.model_dump_json()
    original_matches = matches.model_dump_json()
    result = annotate_review(bundle, matches, api_key="secret-key", timeout=8.5)

    assert network.call_count == 3
    assert [row.status for row in result.groups] == [
        "skipped_unchecked",
        "completed",
        "completed",
        "completed",
    ]
    assert [row.group_index for row in result.groups] == [1, 2, 3, 4]
    for call, source, translation, row in zip(
        network.call_args_list,
        ["A. ", "B. ", "C."],
        ["أ. ", "ب. ", "ج."],
        result.groups[1:],
        strict=True,
    ):
        request = call.args[0]
        assert request.full_url == ENDPOINT
        assert request.method == "POST"
        assert request.get_header("Authorization") == "Bearer secret-key"
        assert call.kwargs == {"timeout": 8.5}
        assert json.loads(request.data) == {
            "model": DEFAULT_MODEL,
            "questions": QUESTIONS,
            "state": {
                "source_text": source,
                "translation": translation,
                "source_language": "de",
                "target_language": "ar",
            },
        }
        assert "PRIVATE" not in request.data.decode()
        assert "private-filename" not in request.data.decode()
        assert row.response == response
        assert annotation_scores(row) == {
            "participants": 0.8,
            "certainty": 0.5,
            "quantifiers": 0.0,
        }
    assert annotation_scores(result.groups[0]) == {}
    assert "secret-key" not in result.model_dump_json()
    assert result.created_at.tzinfo == result.completed_at.tzinfo == UTC
    assert bundle.model_dump_json() == original_bundle
    assert matches.model_dump_json() == original_matches


def test_all_unchecked_needs_no_key_or_network(
    bundle: ReviewBundle, network: Mock
) -> None:
    result = annotate_review(bundle, unchecked_matches(bundle), api_key="")
    assert [row.status for row in result.groups] == ["skipped_unchecked"]
    assert result.groups[0].request is None
    network.assert_not_called()


@pytest.mark.parametrize(
    "timeout", [0, -1, float("inf"), float("nan"), True, "3", 10**400]
)
def test_invalid_timeout_fails_before_network(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock, timeout: Any
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        annotate_review(bundle, matches, api_key="key", timeout=timeout)
    network.assert_not_called()


@pytest.mark.parametrize("max_groups", [0, -1, 2, 2.5, True, "3"])
def test_group_limit_fails_before_network(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock, max_groups: Any
) -> None:
    with pytest.raises(ValueError, match="limit"):
        annotate_review(bundle, matches, api_key="key", max_groups=max_groups)
    network.assert_not_called()


@pytest.mark.parametrize(
    "key", ["", "secret\nkey", "secret key", "secret\x00key", "سري", None]
)
def test_invalid_key_fails_safely_before_network(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock, key: Any
) -> None:
    with pytest.raises(ValueError, match="valid TYPESAFE_API_KEY") as error:
        annotate_review(bundle, matches, api_key=key)
    assert "secret" not in str(error.value)
    network.assert_not_called()


@pytest.mark.parametrize("model", ["", "  ", " jev-1.14.0", None])
def test_invalid_model_fails_before_network(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock, model: Any
) -> None:
    with pytest.raises(ValueError, match="nonblank model"):
        annotate_review(bundle, matches, api_key="key", model_name=model)
    network.assert_not_called()


def test_oversized_later_group_prevents_all_calls(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock
) -> None:
    huge_source = bundle.pairs[0].source + "Z" * MAX_REQUEST_BYTES
    changed_bundle = bundle.model_copy(
        update={"pairs": [bundle.pairs[0].model_copy(update={"source": huge_source})]}
    )
    groups = matches.chunks[0].groups.copy()
    groups[-1] = groups[-1].model_copy(
        update={
            "source": [TextSpan(start=groups[-1].source[0].start, end=len(huge_source))]
        }
    )
    changed_matches = matches.model_copy(
        update={
            "bundle_sha256": bundle_digest(changed_bundle),
            "chunks": [ChunkMatches(unit_id=1, groups=groups)],
        }
    )
    with pytest.raises(ValueError, match="request limit"):
        annotate_review(changed_bundle, changed_matches, api_key="key")
    network.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (HTTPError(ENDPOINT, 429, "secret-key", Message(), None), "rate_limited"),
        (HTTPError(ENDPOINT, 529, "secret-key", Message(), None), "rate_limited"),
        (
            HTTPError(ENDPOINT, 401, "secret-key", Message(), None),
            "authentication_error",
        ),
        (
            HTTPError(ENDPOINT, 403, "secret-key", Message(), None),
            "authentication_error",
        ),
        (HTTPError(ENDPOINT, 500, "secret-key", Message(), None), "service_error"),
        (URLError("secret-key"), "transport_error"),
        (TimeoutError("secret-key"), "transport_error"),
        (OSError("secret-key"), "transport_error"),
        (KeyboardInterrupt("secret-key"), "interrupted_outcome_unknown"),
    ],
)
def test_failure_retains_completed_work_and_stops_without_retries_or_key_leak(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    response: dict[str, Any],
    network: Mock,
    failure: BaseException,
    category: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    network.side_effect = [io.BytesIO(json.dumps(response).encode()), failure]
    result = annotate_review(bundle, matches, api_key="secret-key")
    assert network.call_count == 2
    assert [row.status for row in result.groups] == [
        "skipped_unchecked",
        "completed",
        "error",
        "not_attempted",
    ]
    assert result.groups[2].error == category
    assert result.groups[2].request is not None
    assert result.groups[2].response is None
    assert result.groups[3].request is None
    path = tmp_path / "annotations.json"
    save_annotations(result, path, bundle, matches)
    assert load_annotations(path, bundle, matches) == result
    assert "secret-key" not in path.read_text()
    assert "secret-key" not in str(capsys.readouterr())


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"\xff",
        b"[]",
        b'{"model":"a","model":"b"}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1e999}',
        b" " * (MAX_RESPONSE_BYTES + 1),
    ],
)
def test_bad_response_bytes_stop_safely(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock, payload: bytes
) -> None:
    network.side_effect = [io.BytesIO(payload)]
    result = annotate_review(bundle, matches, api_key="key")
    assert result.groups[1].error == "invalid_response"
    assert result.groups[2].status == "not_attempted"
    network.assert_called_once()


@pytest.mark.parametrize(
    ("location", "value"),
    [
        (("model",), "jev-other"),
        (("model",), None),
        (("answers",), {}),
        (("answers", "unexpected"), {"type": "noul", "noul": 0.1}),
        (("answers", "participants"), []),
        (("answers", "participants", "type"), "score"),
        (("answers", "participants", "noul"), True),
        (("answers", "participants", "noul"), "0.5"),
        (("answers", "participants", "noul"), -0.1),
        (("answers", "participants", "noul"), 1.1),
        (("answers", "participants", "noul"), 10**400),
        (("answers", "participants", "noul"), float("nan")),
        (("usage",), None),
        (("usage", "input_tokens"), -1),
        (("usage", "input_tokens"), True),
        (("usage", "input_tokens"), 1.0),
        (("usage", "output_tokens"), "12"),
        (("extra",), float("inf")),
    ],
)
def test_malformed_response_stops_before_other_calls(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    response: dict[str, Any],
    network: Mock,
    location: tuple[str, ...],
    value: Any,
) -> None:
    changed = deepcopy(response)
    parent = changed
    for key in location[:-1]:
        parent = parent[key]
    parent[location[-1]] = value
    network.side_effect = [io.BytesIO(json.dumps(changed).encode())]
    result = annotate_review(bundle, matches, api_key="key")
    assert result.groups[1].error == "invalid_response"
    network.assert_called_once()


def test_explicit_model_requires_exact_returned_name(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    response: dict[str, Any],
    network: Mock,
) -> None:
    response["model"] = "jev-1.13.0"
    result = annotate_review(bundle, matches, api_key="key", model_name="jev-1.13.0")
    assert result.model_name == "jev-1.13.0"
    assert all(row.status == "completed" for row in result.groups[1:])


def test_redirects_are_disabled_in_the_actual_opener(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock
) -> None:
    annotate_review(bundle, matches, api_key="key")
    opener_factory = annotations_module.build_opener
    assert isinstance(opener_factory, Mock)
    handler = opener_factory.call_args.args[0]
    for destination in ("https://attacker.invalid", ENDPOINT):
        assert (
            handler.redirect_request(
                Request(ENDPOINT), None, 307, "redirect", {}, destination
            )
            is None
        )


def test_artifact_roundtrip_and_no_overwrite(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock, tmp_path: Path
) -> None:
    result = annotate_review(bundle, matches, api_key="key")
    path = tmp_path / "annotations.json"
    save_annotations(result, path, bundle, matches)
    before = path.read_bytes()
    assert load_annotations(path, bundle, matches) == result
    with pytest.raises(FileExistsError):
        save_annotations(result, path, bundle, matches)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    assert network.call_count == 3


@pytest.mark.parametrize("metadata", list(ReviewAnnotations.model_fields))
def test_saved_artifact_requires_explicit_metadata(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    network: Mock,
    tmp_path: Path,
    metadata: str,
) -> None:
    result = annotate_review(bundle, matches, api_key="key").model_dump(mode="json")
    del result[metadata]
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="explicitly record"):
        load_annotations(path, bundle, matches)


def test_changes_to_bundle_or_any_matching_field_invalidate_annotations(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock
) -> None:
    result = annotate_review(bundle, matches, api_key="key")
    changed_bundle = bundle.model_copy(update={"source_name": "renamed.txt"})
    rebound_matches = matches.model_copy(
        update={"bundle_sha256": bundle_digest(changed_bundle)}
    )
    with pytest.raises(ValueError, match="changed review bundle"):
        validate_annotations(changed_bundle, rebound_matches, result)
    for update in (
        {"provenance": "human-reviewed"},
        {"created_at": matches.created_at + timedelta(seconds=1)},
        {
            "chunks": [
                ChunkMatches(
                    unit_id=1,
                    groups=[
                        group.model_copy(update={"reason": "changed reason"})
                        for group in matches.chunks[0].groups
                    ],
                )
            ]
        },
    ):
        with pytest.raises(ValueError, match="changed sentence matches"):
            validate_annotations(bundle, matches.model_copy(update=update), result)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "reorder", "request", "scores", "huge_score"]
)
def test_tampered_group_artifact_is_rejected(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    network: Mock,
    tmp_path: Path,
    mutation: str,
) -> None:
    data = annotate_review(bundle, matches, api_key="key").model_dump(mode="json")
    if mutation == "missing":
        data["groups"].pop()
    elif mutation == "duplicate":
        data["groups"].append(data["groups"][-1])
    elif mutation == "reorder":
        data["groups"].reverse()
    elif mutation == "request":
        data["groups"][1]["request"]["state"]["source_text"] = "Wrong sentence"
    elif mutation == "scores":
        data["groups"][1]["response"]["answers"]["participants"]["noul"] = 2
    else:
        data["groups"][1]["response"]["answers"]["participants"]["noul"] = 10**400
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_annotations(path, bundle, matches)


def test_unchecked_cannot_be_annotated_and_matched_cannot_be_silently_skipped(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock
) -> None:
    result = annotate_review(bundle, matches, api_key="key")
    rows = result.groups.copy()
    rows[0] = rows[1].model_copy(update={"group_index": 1})
    with pytest.raises(ValueError, match="Unchecked matches"):
        validate_annotations(
            bundle, matches, result.model_copy(update={"groups": rows})
        )
    rows = result.groups.copy()
    rows[1] = AnnotationGroup(unit_id=1, group_index=2, status="skipped_unchecked")
    with pytest.raises(ValueError, match="Matched groups"):
        validate_annotations(
            bundle, matches, result.model_copy(update={"groups": rows})
        )


def test_no_result_is_valid_after_a_failure(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    network: Mock,
    response: dict[str, Any],
) -> None:
    network.side_effect = [URLError("offline")]
    result = annotate_review(bundle, matches, api_key="key")
    rows = result.groups.copy()
    rows[2] = rows[1].model_copy(update={"group_index": 3})
    with pytest.raises(ValueError, match="stop after the first failure"):
        validate_annotations(
            bundle, matches, result.model_copy(update={"groups": rows})
        )


@pytest.mark.parametrize("status", ["skipped_unchecked", "not_attempted"])
def test_unattempted_states_cannot_have_request_or_response(status: str) -> None:
    with pytest.raises(ValidationError, match="cannot contain"):
        AnnotationGroup.model_validate(
            {
                "unit_id": 1,
                "group_index": 1,
                "status": status,
                "request": {"model": DEFAULT_MODEL},
            }
        )


def test_timestamps_require_aware_values_and_consistent_order() -> None:
    with pytest.raises(ValidationError):
        AnnotationGroup(
            unit_id=1,
            group_index=1,
            status="error",
            request={},
            error="transport_error",
            started_at=datetime(2026, 10, 22),
            completed_at=datetime(2026, 10, 22, tzinfo=UTC),
        )


def test_latest_model_alias_is_rejected_before_network(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock
) -> None:
    with pytest.raises(ValueError, match="exact version"):
        annotate_review(bundle, matches, api_key="key", model_name="jev-latest")
    network.assert_not_called()


def test_grouped_spans_keep_exact_whitespace_in_one_request(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock
) -> None:
    first, second, third, fourth = matches.chunks[0].groups
    combined = MatchGroup(
        status="matched",
        source=second.source + third.source,
        translation=[
            TextSpan(start=second.translation[0].start, end=third.translation[0].end)
        ],
    )
    grouped = matches.model_copy(
        update={"chunks": [ChunkMatches(unit_id=1, groups=[first, combined, fourth])]}
    )
    result = annotate_review(bundle, grouped, api_key="key")
    assert network.call_count == 2
    request = result.groups[1].request
    assert request is not None
    assert request["state"]["source_text"] == "A. B. "
    assert request["state"]["translation"] == "أ. ب. "


def test_stale_matching_input_fails_before_network(
    bundle: ReviewBundle, matches: ReviewMatches, network: Mock
) -> None:
    changed = bundle.model_copy(update={"source_name": "renamed.txt"})
    with pytest.raises(ValueError, match="changed review bundle"):
        annotate_review(changed, matches, api_key="key")
    network.assert_not_called()


@pytest.mark.parametrize("threshold", [0.69, 0.71, "0.7", float("nan")])
def test_offline_threshold_is_fixed_and_not_coerced(
    bundle: ReviewBundle,
    matches: ReviewMatches,
    network: Mock,
    tmp_path: Path,
    threshold: Any,
) -> None:
    data = annotate_review(bundle, matches, api_key="key").model_dump(mode="json")
    data["threshold"] = threshold
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_annotations(path, bundle, matches)
