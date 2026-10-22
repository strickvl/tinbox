"""Reproduce assistant-curated Arabic correspondence for the recorded pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal

from tinbox.core.review import ReviewBundle, load_review_bundle
from tinbox.core.review_matches import (
    ChunkMatches,
    MatchGroup,
    ReviewMatches,
    TextSpan,
    bundle_digest,
    save_matches,
)

RUN_DIR = Path(__file__).parent / "runs" / "2026-10-22-elara-cli-03"
BUNDLE_SHA256 = "cff1b76373d670fa0a5f1b369d293f64606760c08bd13b2bf4d90fdbe125687d"
CURATED_AT = datetime(2026, 10, 22, 19, 30, 25, tzinfo=UTC)


@dataclass(frozen=True)
class Boundaries:
    source: tuple[str, ...]
    translation: tuple[str, ...]
    status: Literal["matched", "unchecked"] = "matched"
    reason: str = ""


def match(source: str, translation: str) -> Boundaries:
    return Boundaries((source,), (translation,))


GROUPS: dict[int, tuple[Boundaries, ...]] = {
    1: (
        Boundaries(
            ("# The Girl Who",),
            ("# الفتاة التي لم تستطع الفهم",),
            status="unchecked",
            reason="Non-sentential heading; excluded from sentence matching.",
        ),
        match("Once upon a time,", "في قديم الزمان،"),
        match("She was bright-eyed,", "كانت ذات عيون لامعة،"),
        match("She could not understand anyone.", "لم تستطع فهم أي شخص."),
        match("When the baker spoke to her,", "عندما كان الخباز يتحدث إليها،"),
        match("When the shepherd called", "وعندما كان الراعي ينادي"),
        match("Even her own mother", "حتى جمل والدتها"),
        match("Elara nodded and smiled,", "كانت إلارا تهز رأسها وتبتسم،"),
        match("The villagers thought", "كان القرويون يظنون"),
        match("They whispered,", "كانوا يهمسون،"),
        match("“She does not listen,”", '"إنها لا تستمع"،'),
        match("But Elara *did* listen,", "لكن إلارا *كانت* تستمع،"),
    ),
    2: (
        match("One evening,", "في مساء أحد الأيام،"),
        match("The sun had gone down,", "كانت الشمس قد غابت،"),
        match("She held a small pebble", "كانت تمسك بحجر صغير"),
        match("That was when a soft shimmer", "حينها ارتفعت وميض ناعم"),
        match("A shape gathered", "تشكل شكل"),
        match("It looked like both a man", "بدا كأنه رجل وسحابة"),
        match("“Child,” said the figure,", '"يا طفلة،" قال الشكل،'),
        match("Elara blinked.", "رمشت إلارا."),
        match("For the first time in her life,", "لأول مرة في حياتها،"),
        match("“You can speak to me?”", '"هل يمكنك التحدث إلي؟"'),
        match("“I am a translator of meanings,”", '"أنا مترجم المعاني،"'),
        match("“A genie of tongues,", '"جني الألسنة،'),
        match("You may call me Lume.”", 'يمكنك مناداتي لوم."'),
        match("Elara felt warmth", "شعرت إلارا بالدفء"),
        match("“Everyone speaks to me,", '"الجميع يتحدثون إلي،'),
        match("They blur together.", "إنها تتداخل معاً."),
        match("Can you help me?”", 'هل يمكنك مساعدتي؟"'),
    ),
    3: (
        match("The genie drifted closer,", "اقترب الجني،"),
        Boundaries(
            (
                "“I can,” Lume said.",
                "“I can hear their words,",
                "But magic is never given",
                "You must promise something",
            ),
            (
                'قال لوم، "أستطيع ذلك.',
                "أستطيع سماع كلماتهم،",
                "لكن السحر لا يُعطى",
                "يجب أن تعدي بشيء",
            ),
            reason=(
                "Grouped quoted turn: Arabic moves Lume's speaker attribution "
                "before the speech and uses one quotation across four sentences."
            ),
        ),
        match("“What must I promise?”", '"بماذا يجب أن أعد؟"'),
    ),
}


def spans_at_boundaries(text: str, markers: tuple[str, ...]) -> dict[str, TextSpan]:
    """Resolve authored boundaries, without inferring any sentence correspondence."""
    if any(text.count(marker) != 1 for marker in markers):
        raise ValueError("Each curated boundary must occur exactly once in its chunk")
    starts = [text.index(marker) for marker in markers]
    if starts[0] != 0 or any(a >= b for a, b in pairwise(starts)):
        raise ValueError("Curated boundaries must start at zero and follow text order")
    ends = [*starts[1:], len(text)]
    return {
        marker: TextSpan(start=start, end=end)
        for marker, start, end in zip(markers, starts, ends, strict=True)
    }


def curate(bundle: ReviewBundle) -> ReviewMatches:
    if bundle_digest(bundle) != BUNDLE_SHA256:
        raise ValueError(
            "These curated boundaries only apply to the recorded ar bundle"
        )
    chunks = []
    for pair in bundle.pairs:
        boundaries = GROUPS[pair.unit_id]
        source = spans_at_boundaries(
            pair.source,
            tuple(marker for group in boundaries for marker in group.source),
        )
        translation = spans_at_boundaries(
            pair.translation,
            tuple(marker for group in boundaries for marker in group.translation),
        )
        chunks.append(
            ChunkMatches(
                unit_id=pair.unit_id,
                groups=[
                    MatchGroup(
                        status=group.status,
                        source=[source[marker] for marker in group.source],
                        translation=[
                            translation[marker] for marker in group.translation
                        ],
                        reason=group.reason,
                    )
                    for group in boundaries
                ],
            )
        )
    return ReviewMatches(
        created_at=CURATED_AT,
        bundle_sha256=BUNDLE_SHA256,
        provenance="assistant-curated",
        chunks=chunks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_review_bundle(RUN_DIR / "ar.review.json")
    matches = curate(bundle)
    save_matches(matches, args.output, bundle)
    print(f"Saved assistant-curated sentence matches to {args.output}")


if __name__ == "__main__":
    main()
