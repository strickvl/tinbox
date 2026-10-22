"""Coverage and identity checks for explicit review sentence matches."""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinbox.core.review import ReviewBundle, ReviewPair
from tinbox.core.review_matches import (
    ChunkMatches,
    MatchGroup,
    ReviewMatches,
    TextSpan,
    bundle_digest,
    load_matches,
    save_matches,
    unchecked_matches,
    validate_matches,
)


@pytest.fixture
def bundle() -> ReviewBundle:
    return ReviewBundle(
        created_at=datetime(2026, 10, 22, 10, 30, tzinfo=UTC),
        source_name="example.txt",
        source_sha256="ab" * 32,
        source_lang="en",
        target_lang="de",
        provider="openai",
        model_name="test-model",
        context_size=100,
        pairs=[ReviewPair(unit_id=1, source="One. Two.", translation="Eins. Zwei.")],
    )


def test_unchecked_template_keeps_all_text_visible(bundle: ReviewBundle) -> None:
    matches = unchecked_matches(bundle)
    validate_matches(bundle, matches)

    assert matches.provenance == "unreviewed"
    assert matches.created_at.tzinfo == UTC
    assert matches.chunks == [
        ChunkMatches(
            unit_id=1,
            groups=[
                MatchGroup(
                    status="unchecked",
                    source=[TextSpan(start=0, end=9)],
                    translation=[TextSpan(start=0, end=11)],
                    reason="Needs manual sentence matching.",
                )
            ],
        )
    ]


@pytest.mark.parametrize(
    ("source", "translation"), [("", ""), ("", "Text"), ("Text", "")]
)
def test_unchecked_template_accepts_empty_sides(
    bundle: ReviewBundle, source: str, translation: str
) -> None:
    empty = bundle.model_copy(
        update={
            "pairs": [ReviewPair(unit_id=1, source=source, translation=translation)]
        }
    )
    matches = unchecked_matches(empty)
    validate_matches(empty, matches)
    group = matches.chunks[0].groups[0]
    assert bool(group.source) == bool(source)
    assert bool(group.translation) == bool(translation)


@pytest.mark.parametrize(
    ("source", "translation", "source_spans", "translation_spans"),
    [
        ("One and two.", "Eins. Zwei.", [(0, 12)], [(0, 6), (6, 11)]),
        ("One. Two.", "Eins und zwei.", [(0, 5), (5, 9)], [(0, 14)]),
        (
            "One. Two.",
            "Eins. Zwei.",
            [(0, 5), (5, 9)],
            [(0, 6), (6, 11)],
        ),
    ],
)
def test_grouped_matches_preserve_split_and_merged_sentences(
    bundle: ReviewBundle,
    source: str,
    translation: str,
    source_spans: list[tuple[int, int]],
    translation_spans: list[tuple[int, int]],
) -> None:
    grouped = bundle.model_copy(
        update={
            "pairs": [ReviewPair(unit_id=1, source=source, translation=translation)]
        }
    )
    matches = ReviewMatches(
        bundle_sha256=bundle_digest(grouped),
        provenance="human-reviewed",
        chunks=[
            ChunkMatches(
                unit_id=1,
                groups=[
                    MatchGroup(
                        status="matched",
                        source=[TextSpan(start=s, end=e) for s, e in source_spans],
                        translation=[
                            TextSpan(start=s, end=e) for s, e in translation_spans
                        ],
                    )
                ],
            )
        ],
    )
    validate_matches(grouped, matches)
    assert len(matches.chunks[0].groups[0].source) == len(source_spans)
    assert len(matches.chunks[0].groups[0].translation) == len(translation_spans)


def test_unicode_offsets_are_code_points_and_keep_whitespace(
    bundle: ReviewBundle, tmp_path: Path
) -> None:
    unicode_bundle = bundle.model_copy(
        update={
            "source_lang": "de",
            "target_lang": "ar",
            "pairs": [
                ReviewPair(
                    unit_id=1,
                    source="Ä. 👩🏽‍💻!\n",
                    translation="أ. مُ!\n",
                )
            ],
        }
    )
    matches = ReviewMatches(
        bundle_sha256=bundle_digest(unicode_bundle),
        provenance="assistant-curated",
        chunks=[
            ChunkMatches(
                unit_id=1,
                groups=[
                    MatchGroup(
                        status="matched",
                        source=[TextSpan(start=0, end=3)],
                        translation=[TextSpan(start=0, end=3)],
                    ),
                    MatchGroup(
                        status="unchecked",
                        source=[TextSpan(start=3, end=9)],
                        translation=[TextSpan(start=3, end=7)],
                        reason="No clear correspondence.",
                    ),
                ],
            )
        ],
    )
    path = tmp_path / "matches.json"
    save_matches(matches, path, unicode_bundle)
    loaded = load_matches(path, unicode_bundle)
    assert loaded == matches
    pair = unicode_bundle.pairs[0]
    assert pair.source[3:9] == "👩🏽‍💻!\n"
    assert pair.translation[3:7] == "مُ!\n"
    assert loaded.chunks[0].groups[1].status == "unchecked"
    assert len(list(tmp_path.iterdir())) == 1


def test_one_sided_omission_remains_explicitly_unchecked(bundle: ReviewBundle) -> None:
    omitted = bundle.model_copy(
        update={
            "pairs": [ReviewPair(unit_id=1, source="One. Two.", translation="Eins.")]
        }
    )
    matches = ReviewMatches(
        bundle_sha256=bundle_digest(omitted),
        provenance="human-reviewed",
        chunks=[
            ChunkMatches(
                unit_id=1,
                groups=[
                    MatchGroup(
                        status="matched",
                        source=[TextSpan(start=0, end=5)],
                        translation=[TextSpan(start=0, end=5)],
                    ),
                    MatchGroup(
                        status="unchecked",
                        source=[TextSpan(start=5, end=9)],
                        translation=[],
                        reason="No corresponding translation found.",
                    ),
                ],
            )
        ],
    )
    validate_matches(omitted, matches)
    assert matches.chunks[0].groups[1].source == [TextSpan(start=5, end=9)]


@pytest.mark.parametrize("side", ["source", "translation"])
@pytest.mark.parametrize(
    "ranges",
    [
        [],
        [(1, 3), (3, 5)],
        [(0, 2), (3, 5)],
        [(0, 3), (2, 5)],
        [(0, 3), (0, 3)],
        [(0, 5), (3, 4)],
        [(0, 3), (3, 99)],
        [(0, 3)],
    ],
)
def test_incomplete_overlapping_crossing_or_out_of_bounds_spans_are_rejected(
    bundle: ReviewBundle, side: str, ranges: list[tuple[int, int]]
) -> None:
    data = unchecked_matches(bundle).model_dump()
    data["chunks"][0]["groups"][0][side] = [
        {"start": start, "end": end} for start, end in ranges
    ]
    matches = ReviewMatches.model_validate(data)
    with pytest.raises(ValueError, match=f"Chunk 1 {side} spans must cover"):
        validate_matches(bundle, matches)


def test_coverage_is_checked_across_groups(bundle: ReviewBundle) -> None:
    matches = unchecked_matches(bundle)
    matches.chunks[0].groups.append(matches.chunks[0].groups[0])
    with pytest.raises(ValueError, match="without gaps, overlaps"):
        validate_matches(bundle, matches)


@pytest.mark.parametrize("ids", [[1], [2, 1], [1, 1], [1, 2, 3]])
def test_requires_all_original_chunks_once_in_order(
    bundle: ReviewBundle, ids: list[int]
) -> None:
    two_chunks = bundle.model_copy(
        update={
            "pairs": [
                *bundle.pairs,
                ReviewPair(unit_id=2, source="Three.", translation="Drei."),
            ]
        }
    )
    original = unchecked_matches(two_chunks)
    matches = original.model_copy(
        update={
            "chunks": [
                original.chunks[0].model_copy(update={"unit_id": i}) for i in ids
            ]
        }
    )
    with pytest.raises(ValueError, match="every chunk exactly once in order"):
        validate_matches(two_chunks, matches)


def test_digest_includes_translation_and_every_metadata_field(
    bundle: ReviewBundle,
) -> None:
    original = bundle_digest(bundle)
    data = bundle.model_dump()
    mutations = {
        "created_at": bundle.created_at + timedelta(seconds=1),
        "source_name": "different.txt",
        "source_sha256": "cd" * 32,
        "source_lang": "fr",
        "target_lang": "ar",
        "provider": "anthropic",
        "model_name": "other-model",
        "context_size": 101,
        "custom_split_token": "|",
        "use_glossary": True,
        "reasoning_effort": "high",
        "pairs": [ReviewPair(unit_id=1, source="One. Two.", translation="Changed.")],
    }
    for field, value in mutations.items():
        changed = ReviewBundle.model_validate({**data, field: value})
        assert bundle_digest(changed) != original, field
        with pytest.raises(ValueError, match="different or changed review bundle"):
            validate_matches(changed, unchecked_matches(bundle))


def test_digest_ignores_json_format_and_key_order(bundle: ReviewBundle) -> None:
    json_data = json.loads(bundle.model_dump_json())
    reversed_data = dict(reversed(list(json_data.items())))
    same_bundle = ReviewBundle.model_validate_json(json.dumps(reversed_data, indent=4))
    assert bundle_digest(bundle) == bundle_digest(same_bundle)


@pytest.mark.parametrize("field", ["schema_version", "created_at", "provenance"])
def test_load_requires_explicit_headers(
    bundle: ReviewBundle, tmp_path: Path, field: str
) -> None:
    data = json.loads(unchecked_matches(bundle).model_dump_json())
    del data[field]
    path = tmp_path / "matches.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=r"include|provenance"):
        load_matches(path, bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", "1"),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("created_at", "2026-10-22T10:30:00"),
        ("bundle_sha256", "AB" * 32),
        ("bundle_sha256", "not a hash"),
        ("provenance", "automated"),
        ("chunks", []),
        ("unexpected", "field"),
    ],
)
def test_invalid_sidecar_fields_are_rejected(
    bundle: ReviewBundle, tmp_path: Path, field: str, value: object
) -> None:
    data = json.loads(unchecked_matches(bundle).model_dump_json())
    data[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_matches(path, bundle)


@pytest.mark.parametrize(
    "value",
    [
        {"start": -1, "end": 2},
        {"start": 2, "end": 2},
        {"start": 2, "end": 1},
        {"start": 0, "end": 0},
        {"start": True, "end": 2},
        {"start": 0, "end": "2"},
        {"start": 0, "end": 2.0},
        {"start": 0, "end": 2, "text": "ab"},
    ],
)
def test_span_boundaries_are_strict(value: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TextSpan.model_validate(value)


@pytest.mark.parametrize(
    "value",
    [
        {"status": "matched", "source": [], "translation": []},
        {
            "status": "matched",
            "source": [{"start": 0, "end": 1}],
            "translation": [],
        },
        {
            "status": "matched",
            "source": [],
            "translation": [{"start": 0, "end": 1}],
        },
        {"status": "unchecked", "source": [], "translation": []},
        {"status": "unchecked", "source": [], "translation": [], "reason": " \n"},
        {"status": "approved", "source": [], "translation": []},
    ],
)
def test_status_requirements_are_enforced(value: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MatchGroup.model_validate(value)


def test_timestamps_are_aware_and_normalized(bundle: ReviewBundle) -> None:
    data = unchecked_matches(bundle).model_dump()
    data["created_at"] = datetime(
        2026, 10, 22, 12, 30, tzinfo=timezone(timedelta(hours=2))
    )
    matches = ReviewMatches.model_validate(data)
    assert matches.created_at == datetime(2026, 10, 22, 10, 30, tzinfo=UTC)
    assert matches.created_at.tzinfo == UTC


def test_matched_groups_require_explicit_curation_provenance(
    bundle: ReviewBundle,
) -> None:
    data = unchecked_matches(bundle).model_dump()
    data["chunks"][0]["groups"][0]["status"] = "matched"
    with pytest.raises(
        ValidationError, match="require assistant-curated or human-reviewed"
    ):
        ReviewMatches.model_validate(data)


def test_save_never_overwrites_and_stale_matches_do_not_create_a_file(
    bundle: ReviewBundle, tmp_path: Path
) -> None:
    path = tmp_path / "matches.json"
    matches = unchecked_matches(bundle)
    save_matches(matches, path, bundle)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_matches(matches, path, bundle)
    assert path.read_bytes() == original

    stale = matches.model_copy(update={"bundle_sha256": "00" * 32})
    with pytest.raises(ValueError, match="different or changed review bundle"):
        save_matches(stale, tmp_path / "stale.json", bundle)
    assert not (tmp_path / "stale.json").exists()
    assert len(list(tmp_path.iterdir())) == 1


def test_load_rejects_validly_shaped_but_incomplete_coverage(
    bundle: ReviewBundle, tmp_path: Path
) -> None:
    data = json.loads(unchecked_matches(bundle).model_dump_json())
    data["chunks"][0]["groups"][0]["source"][0]["end"] = 3
    path = tmp_path / "matches.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="must cover the entire text"):
        load_matches(path, bundle)
