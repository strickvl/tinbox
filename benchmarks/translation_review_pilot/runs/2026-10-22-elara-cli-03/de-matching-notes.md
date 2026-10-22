# German sentence correspondence

The saved German translation has 34 clear sentence correspondences across three chunks. The heading remains unchecked because it is not a sentence. No ambiguous correspondence was found in this excerpt. This is assistant-curated correspondence, not human verification or a translation quality assessment.

| Chunk | Matched groups | Matched source spans | Matched translation spans | Unchecked groups |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11 | 11 | 11 | 1 heading |
| 2 | 17 | 17 | 17 | 0 |
| 3 | 6 | 6 | 6 | 0 |
| Total | 34 | 34 | 34 | 1 |

All matched groups are 1:1 in this translation. Including the unchecked heading, the sidecar contains 35 source spans and 35 translation spans. Both sides cover every saved character exactly once, including spaces, paragraph breaks, Markdown markers, and quotation marks. Trailing whitespace belongs to the preceding span.

## Boundary decisions

- Dialogue and its attribution remain one sentence unit. For example, source “You can speak to me?” plus “she gasped.” corresponds to German “Du kannst mit mir sprechen?” plus “hauchte sie.” The question mark inside the quotation does not create a separate attribution-only match.
- The interrupted “Child” / “Kind” question remains one unit, including its middle attribution. The final “What must I promise?” question likewise includes Elara's attribution and following clause on both sides.
- Multiple sentences within one speech are separate matches when their correspondence is clear. “A genie of tongues…” is a complete intended utterance fragment that corresponds to “Ein Dschinn der Zungen…”. Its opening quotation mark remains in that span; the quotation's closing mark remains with the following sentence.
- Wording choices such as “translator” becoming “Übersetzerin” do not change the sentence correspondence. A matched label does not approve that wording or resolve fidelity questions.

## Reproduction and limits

`curate_de.py` contains explicit paired boundary markers selected after reading the actual saved source and German translation. It does not infer matches from equal sentence counts or use a generic sentence splitter. The script requires the canonical bundle digest `269397f0735ad700ec6474a50fbdd86be0cb495dd0506ec08a920f162df34793`, so these decisions cannot silently be applied to another translation.

From the repository root, reproduce the mapping into a new file:

```bash
uv run --locked --all-extras python benchmarks/translation_review_pilot/curate_de.py /tmp/elara-de.matches.json
```

The output path must not exist. The fixed curation timestamp makes the same mapping byte-for-byte reproducible. `save_matches` and `load_matches` validate bundle identity, ordered chunk coverage, and complete span coverage. The original source, translation, and review bundle are not modified.

All 35 rendered text pairs were read back from their saved spans after generation. This small literary excerpt does not demonstrate reliability on omitted, reordered, merged, or split sentences, and it contains no actual 1:N or N:1 correspondence. The assistant-curated labels still need bilingual human review before serving as evaluation ground truth. No TypeSafe calls, annotations, or quality scores were added.
