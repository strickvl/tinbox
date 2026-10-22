"""Observable HTML behavior for curated sentence comparisons."""

from html.parser import HTMLParser
from pathlib import Path

import pytest

from tinbox.core.review import ReviewBundle, ReviewPair, write_review_report
from tinbox.core.review_matches import (
    ChunkMatches,
    MatchGroup,
    ReviewMatches,
    TextSpan,
    bundle_digest,
    unchecked_matches,
)


class MatchedReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.texts: list[str] = []
        self.in_text = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if tag == "div" and attributes.get("class") == "chunk-text":
            self.in_text = True
            self.texts.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            self.in_text = False

    def handle_data(self, data: str) -> None:
        if self.in_text:
            self.texts[-1] += data


@pytest.fixture
def bundle() -> ReviewBundle:
    return ReviewBundle(
        source_name="story.txt",
        source_sha256="ab" * 32,
        source_lang="en",
        target_lang="ar",
        provider="openai",
        model_name="test-model",
        context_size=2000,
        pairs=[
            ReviewPair(
                unit_id=1,
                source=(
                    "\nAnna left. She waved.\nShe might return, but nobody knows. "
                ),
                translation=(
                    "\n غادرت آنا ولوّحت بيدها. "
                    "قد تعود. لكن لا أحد يعلم.\n👋 ملاحظة إضافية."
                ),
            )
        ],
    )


@pytest.fixture
def matches(bundle: ReviewBundle) -> ReviewMatches:
    pair = bundle.pairs[0]
    source_breaks = [0, len("\nAnna left. "), len("\nAnna left. She waved.\n")]
    translation_breaks = [
        0,
        len("\n غادرت آنا ولوّحت بيدها. "),
        len("\n غادرت آنا ولوّحت بيدها. قد تعود. "),
        pair.translation.index("👋"),
    ]
    return ReviewMatches(
        bundle_sha256=bundle_digest(bundle),
        provenance="assistant-curated",
        chunks=[
            ChunkMatches(
                unit_id=1,
                groups=[
                    MatchGroup(
                        status="matched",
                        source=[
                            TextSpan(start=source_breaks[0], end=source_breaks[1]),
                            TextSpan(start=source_breaks[1], end=source_breaks[2]),
                        ],
                        translation=[TextSpan(start=0, end=translation_breaks[1])],
                        reason="Two source sentences combined in the translation.",
                    ),
                    MatchGroup(
                        status="matched",
                        source=[TextSpan(start=source_breaks[2], end=len(pair.source))],
                        translation=[
                            TextSpan(
                                start=translation_breaks[1], end=translation_breaks[2]
                            ),
                            TextSpan(
                                start=translation_breaks[2], end=translation_breaks[3]
                            ),
                        ],
                        reason="One source sentence split into two sentences.",
                    ),
                    MatchGroup(
                        status="unchecked",
                        source=[],
                        translation=[
                            TextSpan(
                                start=translation_breaks[3], end=len(pair.translation)
                            )
                        ],
                        reason="No corresponding source text identified.",
                    ),
                ],
            )
        ],
    )


def test_report_preserves_grouped_sentences_unchecked_text_and_originals(
    bundle: ReviewBundle, matches: ReviewMatches, tmp_path: Path
) -> None:
    before_bundle = bundle.model_dump_json()
    before_matches = matches.model_dump_json()
    path = tmp_path / "matched.html"

    write_review_report(bundle, path, matches=matches)

    html = path.read_text(encoding="utf-8")
    parser = MatchedReportParser()
    parser.feed(html)
    assert parser.texts == [
        "\nAnna left. She waved.\n",
        "\n غادرت آنا ولوّحت بيدها. ",
        "She might return, but nobody knows. ",
        "قد تعود. لكن لا أحد يعلم.\n",
        "",
        "👋 ملاحظة إضافية.",
        bundle.pairs[0].source,
        bundle.pairs[0].translation,
    ]
    assert "2 matched groups · 1 unchecked region" in html
    assert html.count("Matched for comparison") == 2
    assert "No source text in this region." in html
    assert "No corresponding source text identified." in html
    assert "Assistant-curated matches; not human-verified." in html
    assert "No automated quality assessment has been performed." in html
    assert "Full original chunk pair" in html
    assert any(
        tag == "details" and attrs.get("class") == "original-chunk"
        for tag, attrs in parser.elements
    )
    assert bundle.model_dump_json() == before_bundle
    assert matches.model_dump_json() == before_matches


def test_matched_report_retains_rtl_offline_and_responsive_behavior(
    bundle: ReviewBundle, matches: ReviewMatches, tmp_path: Path
) -> None:
    path = tmp_path / "matched.html"
    write_review_report(bundle, path, matches=matches)
    html = path.read_text(encoding="utf-8")
    parser = MatchedReportParser()
    parser.feed(html)
    text_elements = [
        attrs for _, attrs in parser.elements if attrs.get("class") == "chunk-text"
    ]
    assert [attrs["lang"] for attrs in text_elements] == ["en", "ar"] * 4
    assert all(attrs["dir"] == "auto" for attrs in text_elements)
    assert "white-space: pre-wrap" in html
    assert "unicode-bidi: plaintext" in html
    assert "@media (max-width:" in html
    assert "grid-template-columns: minmax(0, 1fr);" in html
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


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [
        ("unreviewed", "Unreviewed draft; no sentence matches have been selected."),
        ("human-reviewed", "Human-reviewed matches; this is not a quality verdict."),
    ],
)
def test_report_distinguishes_match_provenance(
    bundle: ReviewBundle, tmp_path: Path, provenance: str, expected: str
) -> None:
    data = unchecked_matches(bundle).model_dump()
    data["provenance"] = provenance
    path = tmp_path / "draft.html"
    write_review_report(bundle, path, matches=ReviewMatches.model_validate(data))
    html = path.read_text(encoding="utf-8")

    assert expected in html
    assert "0 matched groups · 1 unchecked region" in html
    assert "Matched for comparison" not in html
    assert "Needs manual sentence matching." in html


def test_match_reasons_and_text_cannot_inject_html(
    bundle: ReviewBundle, tmp_path: Path
) -> None:
    hostile = '</div><script>alert("bad")</script><img src="https://example.com"> &'
    data = bundle.model_dump()
    data["source_lang"] = 'en" onclick="bad'
    data["pairs"] = [ReviewPair(unit_id=1, source=hostile, translation=hostile)]
    bundle = ReviewBundle.model_validate(data)
    matches = unchecked_matches(bundle)
    matches.chunks[0].groups[0] = MatchGroup(
        status="unchecked",
        source=[TextSpan(start=0, end=len(hostile))],
        translation=[TextSpan(start=0, end=len(hostile))],
        reason=hostile,
    )
    path = tmp_path / "escaped.html"
    write_review_report(bundle, path, matches=matches)
    html = path.read_text(encoding="utf-8")
    parser = MatchedReportParser()
    parser.feed(html)

    assert parser.texts == [hostile] * 4
    assert "&lt;script&gt;" in html
    assert all(tag not in {"script", "img"} for tag, _ in parser.elements)
    assert all("onclick" not in attrs for _, attrs in parser.elements)


def test_report_rejects_stale_matches_before_creating_file(
    bundle: ReviewBundle, matches: ReviewMatches, tmp_path: Path
) -> None:
    changed = bundle.model_copy(update={"model_name": "another-model"})
    with pytest.raises(ValueError, match="different or changed review bundle"):
        write_review_report(changed, tmp_path / "invalid.html", matches=matches)

    assert list(tmp_path.iterdir()) == []


def test_unchecked_source_without_translation_remains_visible(
    bundle: ReviewBundle, tmp_path: Path
) -> None:
    data = bundle.model_dump()
    data["pairs"] = [
        ReviewPair(unit_id=1, source="Unmatched source text.\n", translation="")
    ]
    bundle = ReviewBundle.model_validate(data)
    path = tmp_path / "unmatched.html"
    write_review_report(bundle, path, matches=unchecked_matches(bundle))
    html = path.read_text(encoding="utf-8")
    parser = MatchedReportParser()
    parser.feed(html)

    assert parser.texts == ["Unmatched source text.\n", ""] * 2
    assert "No translation text in this region." in html
    assert "0 matched groups · 1 unchecked region" in html


def test_report_rejects_gaps_before_creating_file(
    bundle: ReviewBundle, tmp_path: Path
) -> None:
    matches = unchecked_matches(bundle)
    matches.chunks[0].groups[0].source[0] = TextSpan(
        start=1, end=len(bundle.pairs[0].source)
    )
    with pytest.raises(ValueError, match="without gaps"):
        write_review_report(bundle, tmp_path / "invalid.html", matches=matches)

    assert list(tmp_path.iterdir()) == []
