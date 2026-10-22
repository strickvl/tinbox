"""Offline report behavior for saved TypeSafe outcomes."""

from copy import deepcopy
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tinbox.core.review import ReviewBundle, ReviewPair, write_review_report
from tinbox.core.review_annotations import (
    AnnotationGroup,
    ReviewAnnotations,
    matches_digest,
)
from tinbox.core.review_matches import (
    ChunkMatches,
    MatchGroup,
    ReviewMatches,
    TextSpan,
    bundle_digest,
    unchecked_matches,
)
from tinbox.core.review_questions import QUESTIONS


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.texts: list[str] = []
        self.labels: list[str] = []
        self.in_text = False
        self.in_label = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if tag == "div" and attributes.get("class") == "chunk-text":
            self.in_text = True
            self.texts.append("")
        if tag == "p" and attributes.get("class") == "annotation-label":
            self.in_label = True
            self.labels.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            self.in_text = False
        if tag == "p":
            self.in_label = False

    def handle_data(self, data: str) -> None:
        if self.in_text:
            self.texts[-1] += data
        if self.in_label:
            self.labels[-1] += data


@pytest.fixture
def review() -> tuple[ReviewBundle, ReviewMatches, ReviewAnnotations]:
    source = [
        "Heading\n",
        '<script>alert("source")</script> She may return.\n',
        "All three people left. ",
        "The clerk spoke. ",
        "She thanked him.\n",
    ]
    translation = [
        "عنوان\n",
        '<img src="https://example.com"> قد تعود.\n',
        "غادر شخصان. ",
        "تحدث الكاتب. ",
        "شكرته.\n",
    ]
    timestamp = datetime(2026, 10, 22, 19, 30, tzinfo=UTC)
    bundle = ReviewBundle(
        created_at=timestamp,
        source_name="story.txt",
        source_sha256="ab" * 32,
        source_lang="en",
        target_lang="ar",
        provider="openai",
        model_name="translation-model",
        context_size=2000,
        pairs=[
            ReviewPair(
                unit_id=1, source="".join(source), translation="".join(translation)
            )
        ],
    )
    groups = []
    source_offset = translation_offset = 0
    for index, (original, translated) in enumerate(
        zip(source, translation, strict=True)
    ):
        groups.append(
            MatchGroup(
                status="unchecked" if index == 0 else "matched",
                source=[
                    TextSpan(start=source_offset, end=source_offset + len(original))
                ],
                translation=[
                    TextSpan(
                        start=translation_offset,
                        end=translation_offset + len(translated),
                    )
                ],
                reason="Non-sentential heading." if index == 0 else "",
            )
        )
        source_offset += len(original)
        translation_offset += len(translated)
    matches = ReviewMatches(
        created_at=timestamp,
        bundle_sha256=bundle_digest(bundle),
        provenance="assistant-curated",
        chunks=[ChunkMatches(unit_id=1, groups=groups)],
    )
    model = 'jev-test<script>alert("model")</script>'
    rows = [AnnotationGroup(unit_id=1, group_index=1, status="skipped_unchecked")]
    for index in range(1, 4):
        request = {
            "model": model,
            "state": {
                "source_text": source[index],
                "translation": translation[index],
                "source_language": "en",
                "target_language": "ar",
            },
            "questions": deepcopy(QUESTIONS),
        }
        response = None
        if index < 3:
            scores = (
                {"participants": 0.01, "certainty": 0.6999, "quantifiers": 0.02}
                if index == 1
                else {"participants": 0.03, "certainty": 0.04, "quantifiers": 0.70}
            )
            response = {
                "model": model,
                "answers": {
                    key: {"type": "noul", "noul": value}
                    for key, value in scores.items()
                },
                "usage": {"input_tokens": 20, "output_tokens": 10},
                "extra_provider_data": '<script>alert("response")</script>',
            }
        rows.append(
            AnnotationGroup(
                unit_id=1,
                group_index=index + 1,
                status="completed" if index < 3 else "error",
                request=request,
                response=response,
                error=None if index < 3 else "transport_error",
                started_at=timestamp,
                completed_at=timestamp,
            )
        )
    rows.append(AnnotationGroup(unit_id=1, group_index=5, status="not_attempted"))
    annotations = ReviewAnnotations(
        created_at=timestamp,
        completed_at=timestamp,
        bundle_sha256=bundle_digest(bundle),
        matches_sha256=matches_digest(matches),
        model_name=model,
        questions=deepcopy(QUESTIONS),
        groups=rows,
    )
    return bundle, matches, annotations


def test_report_distinguishes_outcomes_without_a_quality_pass(
    review: tuple[ReviewBundle, ReviewMatches, ReviewAnnotations], tmp_path: Path
) -> None:
    bundle, matches, annotations = review
    path = tmp_path / "report.html"

    write_review_report(bundle, path, matches=matches, annotations=annotations)

    html = path.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(html)
    assert parser.labels == [
        "Not assessed: unchecked match",
        "No concern flagged by these checks",
        "TypeSafe concern",
        "Request failed",
        "Not attempted",
    ]
    assert (
        "2 completed · 1 flagged · 1 skipped unchecked · 1 request error · 1 not attempted"
        in html
    )
    assert "transport_error" in html
    assert "4 matched groups · 1 unchecked region" in html
    assert "No automated quality assessment has been performed." not in html
    assert "These limited checks can miss errors." in html
    assert "not a translation quality score or a pass result" in html
    assert html.count("<dt>Participant roles</dt>") == 2
    assert html.count("<dt>Certainty</dt>") == 2
    assert html.count("<dt>Quantifiers</dt>") == 2
    assert "<dd>0.6999</dd>" in html
    assert "<dd>0.7</dd>" in html


def test_report_preserves_text_and_escapes_annotation_metadata(
    review: tuple[ReviewBundle, ReviewMatches, ReviewAnnotations], tmp_path: Path
) -> None:
    bundle, matches, annotations = review
    before = [item.model_dump_json() for item in review]
    path = tmp_path / "report.html"

    write_review_report(bundle, path, matches=matches, annotations=annotations)

    html = path.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(html)
    pair = bundle.pairs[0]
    expected = []
    for group in matches.chunks[0].groups:
        expected.extend(
            [
                "".join(pair.source[span.start : span.end] for span in group.source),
                "".join(
                    pair.translation[span.start : span.end]
                    for span in group.translation
                ),
            ]
        )
    assert parser.texts == [*expected, pair.source, pair.translation]
    assert all(
        tag not in {"script", "img", "iframe", "link"} for tag, _ in parser.elements
    )
    assert "jev-test&lt;script&gt;" in html
    assert "extra_provider_data" not in html
    assert "2026-10-22T19:30:00+00:00" in html
    assert "Concern threshold" in html
    assert "Assess only participant identity and roles." in html
    assert "Full original chunk pair" in html
    assert "default-src 'none'" in html
    assert all(
        attrs.get("dir") == "auto"
        for _, attrs in parser.elements
        if attrs.get("class") == "chunk-text"
    )
    assert [item.model_dump_json() for item in review] == before


def test_annotations_require_matches_before_creating_report(
    review: tuple[ReviewBundle, ReviewMatches, ReviewAnnotations], tmp_path: Path
) -> None:
    bundle, _, annotations = review
    with pytest.raises(ValueError, match="require their matching file"):
        write_review_report(bundle, tmp_path / "report.html", annotations=annotations)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("changed_input", ["bundle", "matches"])
def test_report_rejects_stale_annotations_before_creating_file(
    review: tuple[ReviewBundle, ReviewMatches, ReviewAnnotations],
    tmp_path: Path,
    changed_input: str,
) -> None:
    bundle, matches, annotations = review
    if changed_input == "bundle":
        bundle = bundle.model_copy(update={"source_name": "changed.txt"})
        matches = matches.model_copy(update={"bundle_sha256": bundle_digest(bundle)})
    else:
        matches = matches.model_copy(update={"provenance": "human-reviewed"})
    with pytest.raises(ValueError, match="Annotations belong to"):
        write_review_report(
            bundle, tmp_path / "report.html", matches=matches, annotations=annotations
        )
    assert list(tmp_path.iterdir()) == []


def test_report_rejects_invalid_scores_before_creating_file(
    review: tuple[ReviewBundle, ReviewMatches, ReviewAnnotations], tmp_path: Path
) -> None:
    bundle, matches, annotations = review
    response = annotations.groups[1].response
    assert response is not None
    response["answers"]["certainty"]["noul"] = float("nan")
    with pytest.raises(ValueError, match="finite numbers"):
        write_review_report(
            bundle, tmp_path / "report.html", matches=matches, annotations=annotations
        )
    assert list(tmp_path.iterdir()) == []


def test_all_unchecked_report_has_no_scores_or_completed_assessments(
    review: tuple[ReviewBundle, ReviewMatches, ReviewAnnotations], tmp_path: Path
) -> None:
    bundle, _, annotations = review
    matches = unchecked_matches(bundle)
    annotations = annotations.model_copy(
        update={
            "matches_sha256": matches_digest(matches),
            "groups": [
                AnnotationGroup(unit_id=1, group_index=1, status="skipped_unchecked")
            ],
        }
    )
    path = tmp_path / "unchecked.html"

    write_review_report(bundle, path, matches=matches, annotations=annotations)

    html = path.read_text(encoding="utf-8")
    assert (
        "0 completed · 0 flagged · 1 skipped unchecked · 0 request errors · 0 not attempted"
        in html
    )
    assert "Not assessed: unchecked match" in html
    assert '<dl class="annotation-scores">' not in html
    assert "No concern flagged by these checks" not in html


@pytest.mark.parametrize("use_matches", [False, True])
def test_annotations_are_opt_in_and_default_reports_are_unchanged(
    review: tuple[ReviewBundle, ReviewMatches, ReviewAnnotations],
    tmp_path: Path,
    use_matches: bool,
) -> None:
    bundle, matches, _ = review
    selected = matches if use_matches else None
    implicit = tmp_path / "implicit.html"
    explicit = tmp_path / "explicit.html"
    write_review_report(bundle, implicit, matches=selected)
    write_review_report(bundle, explicit, matches=selected, annotations=None)

    assert implicit.read_bytes() == explicit.read_bytes()
    html = implicit.read_text(encoding="utf-8")
    assert "TypeSafe" not in html
    assert "No automated quality assessment has been performed." in html
    assert '<aside class="annotation' not in html
