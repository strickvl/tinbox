"""Tests for saved translation pairs and offline human review reports."""

import json
import os
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tinbox.core.review import (
    ReviewBundle,
    ReviewPair,
    load_review_bundle,
    save_review_bundle,
    validate_new_output_path,
    write_review_report,
)


@pytest.fixture
def bundle() -> ReviewBundle:
    return ReviewBundle(
        created_at=datetime(2026, 10, 22, 10, 30, tzinfo=UTC),
        source_name="example.txt",
        source_sha256="ab" * 32,
        source_lang="en",
        target_lang="ar",
        provider="openai",
        model_name="test-model",
        context_size=100,
        pairs=[
            ReviewPair(
                unit_id=1, source="\n  First line.\n", translation="\n أول سطر.\n"
            ),
            ReviewPair(unit_id=2, source="Second chunk.", translation="الجزء الثاني."),
        ],
    )


def test_round_trip_preserves_unicode_whitespace_and_metadata(
    bundle: ReviewBundle, tmp_path: Path
) -> None:
    path = tmp_path / "review.json"
    save_review_bundle(bundle, path)

    assert load_review_bundle(path) == bundle
    assert "أول سطر" in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text())["schema_version"] == 1
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", "1"),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("source_sha256", "AB" * 32),
        ("source_sha256", "bad hash"),
        ("source_name", 123),
        ("context_size", "100"),
        ("context_size", 0),
        ("context_size", True),
        ("created_at", "2026-10-22T10:30:00"),
        ("algorithm", "page"),
        ("use_glossary", "false"),
        ("unexpected_field", "ignored?"),
        ("pairs", []),
        ("pairs", [{"unit_id": 2, "source": "a", "translation": "b"}]),
        ("pairs", [{"unit_id": "1", "source": "a", "translation": "b"}]),
        ("pairs", [{"unit_id": 1, "source": 2, "translation": "b"}]),
        (
            "pairs",
            [{"unit_id": 1, "source": "a", "translation": "b", "score": 0.9}],
        ),
        (
            "pairs",
            [
                {"unit_id": 1, "source": "a", "translation": "b"},
                {"unit_id": 1, "source": "c", "translation": "d"},
            ],
        ),
    ],
)
def test_load_rejects_invalid_bundles(
    bundle: ReviewBundle, tmp_path: Path, field: str, value: object
) -> None:
    data = json.loads(bundle.model_dump_json())
    data[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_review_bundle(path)


def test_timestamps_are_aware_and_normalized_to_utc(bundle: ReviewBundle) -> None:
    data = bundle.model_dump()
    data["created_at"] = datetime(
        2026, 10, 22, 12, 30, tzinfo=timezone(timedelta(hours=2))
    )
    assert ReviewBundle.model_validate(data).created_at == bundle.created_at
    assert ReviewBundle.model_validate(data).created_at.tzinfo == UTC
    del data["created_at"]
    assert ReviewBundle.model_validate(data).created_at.tzinfo == UTC


@pytest.mark.parametrize("field", ["schema_version", "created_at"])
def test_load_requires_recorded_version_and_timestamp(
    bundle: ReviewBundle, tmp_path: Path, field: str
) -> None:
    data = json.loads(bundle.model_dump_json())
    del data[field]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="must include schema_version and created_at"):
        load_review_bundle(path)


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.chunks: list[str] = []
        self.in_chunk = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if attributes.get("class") == "chunk-text":
            self.in_chunk = True
            self.chunks.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            self.in_chunk = False

    def handle_data(self, data: str) -> None:
        if self.in_chunk:
            self.chunks[-1] += data


def test_html_shows_exact_pairs_and_offline_review_limitations(
    bundle: ReviewBundle, tmp_path: Path
) -> None:
    path = tmp_path / "report.html"
    write_review_report(bundle, path)
    html = path.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(html)

    assert parser.chunks == [
        "\n  First line.\n",
        "\n أول سطر.\n",
        "Second chunk.",
        "الجزء الثاني.",
    ]
    chunk_attributes = [
        attrs for _, attrs in parser.elements if attrs.get("class") == "chunk-text"
    ]
    assert [attrs["lang"] for attrs in chunk_attributes] == ["en", "ar", "en", "ar"]
    assert all(attrs["dir"] == "auto" for attrs in chunk_attributes)
    assert "Review only" in html
    assert "2 paired chunks" in html
    assert "not aligned sentences" in html
    assert "No automated quality assessment" in html
    assert "white-space: pre-wrap" in html
    assert "@media (max-width:" in html
    assert "grid-template-columns: minmax(0, 1fr);" in html
    assert "Content-Security-Policy" in html
    assert "default-src 'none'" in html
    assert all(
        tag not in {"script", "link", "img", "iframe"} for tag, _ in parser.elements
    )
    assert all("src" not in attrs for _, attrs in parser.elements)
    assert all(
        href.startswith("#")
        for _, attrs in parser.elements
        if (href := attrs.get("href")) is not None
    )


def test_html_escapes_text_and_metadata(bundle: ReviewBundle, tmp_path: Path) -> None:
    hostile = '</div><script>alert("bad")</script><img src="https://example.com"> &'
    data = bundle.model_dump()
    data.update(
        source_name=hostile,
        source_lang='en" onclick="bad',
        target_lang='ar" onclick="bad',
        provider=hostile,
        model_name=hostile,
        custom_split_token=hostile,
        reasoning_effort=hostile,
        pairs=[ReviewPair(unit_id=1, source=hostile, translation=hostile)],
    )
    path = tmp_path / "escaped.html"
    write_review_report(ReviewBundle.model_validate(data), path)
    html = path.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(html)

    assert parser.chunks == [hostile, hostile]
    assert "&lt;script&gt;" in html
    assert all(tag not in {"script", "img"} for tag, _ in parser.elements)
    assert all("onclick" not in attrs for _, attrs in parser.elements)
    assert "Split token (overrides chunk size)" in html


@pytest.mark.parametrize("writer", [save_review_bundle, write_review_report])
def test_existing_output_is_never_overwritten(writer, bundle, tmp_path) -> None:
    path = tmp_path / "existing"
    path.write_text("keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        writer(bundle, path)
    assert path.read_text() == "keep me"
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("target_exists", [True, False])
def test_output_symlinks_are_rejected(tmp_path: Path, target_exists: bool) -> None:
    target = tmp_path / "target"
    if target_exists:
        target.write_text("keep me")
    path = tmp_path / "symlink"
    path.symlink_to(target)

    with pytest.raises(FileExistsError):
        validate_new_output_path(path)
    assert path.is_symlink()
    assert target.exists() == target_exists


def test_output_path_must_differ_from_protected_paths(tmp_path: Path) -> None:
    protected = tmp_path / "future-translation.txt"
    with pytest.raises(ValueError, match="separate path"):
        validate_new_output_path(tmp_path / "." / protected.name, [protected])


def test_output_parent_must_be_an_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        validate_new_output_path(tmp_path / "missing" / "output")
    parent_file = tmp_path / "file"
    parent_file.write_text("a file")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        validate_new_output_path(parent_file / "output")


def test_failed_publish_leaves_no_partial_file_or_temp_file(bundle, tmp_path) -> None:
    path = tmp_path / "review.json"
    with patch("tinbox.core.review.os.link", side_effect=OSError("disk error")):
        with pytest.raises(OSError, match="disk error"):
            save_review_bundle(bundle, path)
    assert list(tmp_path.iterdir()) == []


def test_failed_write_leaves_no_partial_file_or_temp_file(bundle, tmp_path) -> None:
    data = bundle.model_dump()
    data["pairs"] = [ReviewPair(unit_id=1, source="\ud800", translation="text")]
    with pytest.raises(UnicodeEncodeError):
        write_review_report(ReviewBundle.model_validate(data), tmp_path / "report.html")
    assert list(tmp_path.iterdir()) == []


def test_output_created_during_write_is_not_replaced(bundle, tmp_path) -> None:
    path = tmp_path / "review.json"
    link = os.link

    def competing_write(source: Path, destination: Path) -> None:
        destination.write_text("someone else's result")
        link(source, destination)

    with patch("tinbox.core.review.os.link", side_effect=competing_write):
        with pytest.raises(FileExistsError):
            save_review_bundle(bundle, path)
    assert path.read_text() == "someone else's result"
    assert len(list(tmp_path.iterdir())) == 1
