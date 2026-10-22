"""Reproduce assistant-curated German matches for one saved pilot bundle."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from tinbox.core.review import load_review_bundle
from tinbox.core.review_matches import (
    ChunkMatches,
    MatchGroup,
    ReviewMatches,
    TextSpan,
    bundle_digest,
    save_matches,
)

RUN_DIR = Path(__file__).parent / "runs" / "2026-10-22-elara-cli-03"
EXPECTED_DIGEST = "269397f0735ad700ec6474a50fbdd86be0cb495dd0506ec08a920f162df34793"
CURATED_AT = datetime(2026, 10, 22, 19, 30, 15, tzinfo=UTC)

# These paired boundaries were selected by reading the saved bilingual text.
# A quoted question and its attribution are one unit, despite punctuation inside it.
BOUNDARIES: list[list[tuple[str, str]]] = [
    [
        (
            "# The Girl Who Couldn\u2019t Understand",
            "# Das Mädchen, das nicht verstehen konnte",
        ),
        ("there lived a girl named Elara.", "lebte ein Mädchen namens Elara."),
        ("followed her everywhere.", "das sie überallhin verfolgte."),
        ("She could not understand anyone.", "Sie konnte niemanden verstehen."),
        ("wind in the chimney.", "des Windes im Schornstein."),
        ("the bleating of sheep.", "das Blöken der Schafe herüber."),
        ("drops of rain on stone.", "wie Regentropfen auf Stein."),
        ("lost in a sea of sounds.", "in einem Meer von Geräuschen."),
        (
            "The villagers thought she was strange.",
            "Die Dorfbewohner hielten sie für seltsam.",
        ),
        ("sometimes they grew impatient.", "wurden manchmal ungeduldig."),
        (
            "“She does not listen,” they muttered.",
            "„Sie hört nicht zu“, murmelten sie.",
        ),
        ("could not catch the meaning.", "die Bedeutung einfach nicht erfassen."),
    ],
    [
        ("the edge of the village.", "am Rand des Dorfes."),
        (
            "the water reflected the first stars.",
            "das Wasser spiegelte die ersten Sterne wider.",
        ),
        ("anchor her in silence.", "in der Stille verankern."),
        ("rose from the river mist.", "aus dem Flussnebel empor."),
        ("yet glowing with pale light.", "doch mit blassem Licht leuchtend."),
        ("every word ever spoken.", "jedes jemals gesprochene Wort gesehen."),
        ("“why do you sigh so deeply?”", "„warum seufzt du so tief?“"),
        ("Elara blinked.", "Elara blinzelte."),
        ("she understood someone perfectly.", "verstand sie jemanden vollkommen."),
        (
            "“You can speak to me?” she gasped.",
            "„Du kannst mit mir sprechen?“ hauchte sie.",
        ),
        ("the ghostly figure replied.", "antwortete die geisterhafte Gestalt."),
        ("but to the longing of hearts.", "sondern an die Sehnsucht der Herzen."),
        ("You may call me Lume.”", "Du darfst mich Lume nennen.“"),
        (
            "warmth spread through her chest.",
            "wie sich Wärme in ihrer Brust ausbreitete.",
        ),
        ("but I cannot grasp their words.", "aber ich kann ihre Worte nicht fassen."),
        ("They blur together.", "Sie verschwimmen ineinander."),
        ("Can you help me?”", "Kannst du mir helfen?“"),
    ],
    [
        ("like mist around her shoulders.", "wie Nebel um ihre Schultern."),
        ("“I can,” Lume said.", "„Ich kann“, sagte Lume."),
        ("in the way you need.", "die du brauchst."),
        (
            "But magic is never given without purpose.",
            "Aber Magie wird nie ohne Zweck gegeben.",
        ),
        (
            "You must promise something in return.”",
            "Du musst etwas im Gegenzug versprechen.“",
        ),
        (
            "her heart already leaned forward eagerly.",
            "ihr Herz bereits eifrig nach vorne geneigt war.",
        ),
    ],
]


def take_span(text: str, start: int, marker: str) -> TextSpan:
    """End at an explicit curated marker, retaining following whitespace."""
    marker_start = text.index(marker, start)
    end = marker_start + len(marker)
    if text.find(marker, end) != -1:
        raise ValueError(f"Curated marker is not unique: {marker!r}")
    while end < len(text) and text[end].isspace():
        end += 1
    return TextSpan(start=start, end=end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="New matches JSON path; never overwritten"
    )
    args = parser.parse_args()
    bundle = load_review_bundle(RUN_DIR / "de.review.json")
    digest = bundle_digest(bundle)
    if digest != EXPECTED_DIGEST:
        raise ValueError(
            "This curation applies only to the original German pilot bundle"
        )
    chunks: list[ChunkMatches] = []
    for pair, boundaries in zip(bundle.pairs, BOUNDARIES, strict=True):
        groups: list[MatchGroup] = []
        source_end = translation_end = 0
        for index, (source_marker, translation_marker) in enumerate(boundaries):
            source_span = take_span(pair.source, source_end, source_marker)
            translation_span = take_span(
                pair.translation, translation_end, translation_marker
            )
            heading = pair.unit_id == 1 and index == 0
            groups.append(
                MatchGroup(
                    status="unchecked" if heading else "matched",
                    source=[source_span],
                    translation=[translation_span],
                    reason=(
                        "Document heading, not a sentence; left outside sentence review."
                        if heading
                        else ""
                    ),
                )
            )
            source_end, translation_end = source_span.end, translation_span.end
        chunks.append(ChunkMatches(unit_id=pair.unit_id, groups=groups))
    matches = ReviewMatches(
        created_at=CURATED_AT,
        bundle_sha256=digest,
        provenance="assistant-curated",
        chunks=chunks,
    )
    save_matches(matches, args.output, bundle)
    print(args.output)


if __name__ == "__main__":
    main()
