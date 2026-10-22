# Arabic sentence correspondence

The actual Arabic CLI translation has 31 matched groups containing 34 source spans and 34 translation spans. One additional group contains the two non-sentential headings and is explicitly unchecked. Every character in all three saved chunk pairs, including whitespace, is covered once and in order.

| Chunk | Matched groups | Matched source spans | Matched translation spans | Unchecked groups |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11 | 11 | 11 | 1 heading |
| 2 | 17 | 17 | 17 | 0 |
| 3 | 3 | 6 | 6 | 0 |
| Total | 31 | 34 | 34 | 1 heading |

These are assistant-curated correspondences, not human-verified labels or automated alignment results. They say which text should be compared, not whether the translation is faithful, grammatical or good. No TypeSafe checks were run. Translation wording and punctuation remain exactly as returned by Tinbox.

## Dialogue and grouping

Question marks inside quotations are not treated as an automatic boundary before their speaker attribution. For example, the source question “You can speak to me?” stays with “she gasped,” and the Arabic question stays with its following attribution. The same rule keeps Elara's final question together with the following narrative clause. Arabic question marks (`؟`), English curly quotes and Arabic output's straight quotes were inspected directly.

Chunk 3, group 2 contains four source sentence spans and four translation sentence spans in one matched group. English introduces Lume between two quotations: “I can,” Lume said. “I can hear their words...” Arabic places the attribution before a single quotation spanning the entire turn: `قال لوم، "أستطيع ذلك. أستطيع سماع كلماتهم..."`. Grouping the complete turn keeps the moved attribution and different quotation boundaries together. The four sentence boundaries remain recorded inside the group.

The other 30 matched groups each contain one source span and one translation span. This excerpt did not require a one-to-many or many-to-one group. No uncertain prose correspondence remained after direct inspection; the heading is excluded because it is not a sentence, not because its counterpart is unclear. This small, orderly excerpt does not test how reliably an aligner would abstain on genuinely ambiguous, omitted or reordered material.

## Reproduce and verify

From the repository root, choose a new output path:

```bash
uv run --no-sync --locked --all-extras python benchmarks/translation_review_pilot/curate_ar.py \
  --output /tmp/elara-ar-regenerated.matches.json
```

The script uses explicitly authored boundary markers and group memberships. It does not zip independently split sentence lists or infer semantic correspondence from equal counts. The canonical review bundle SHA-256 must equal `cff1b76373d670fa0a5f1b369d293f64606760c08bd13b2bf4d90fdbe125687d`; changed bundles are rejected. The curation timestamp is fixed so regeneration is byte-for-byte reproducible. `save_matches` checks complete, ordered text coverage and publishes a new file without overwriting an existing artifact.

The saved sidecar was loaded and validated, and every group was inspected as exact source and Arabic substrings. Regeneration produced identical bytes. Report HTML text was also compared against those exact substrings after HTML entity decoding. This is a text-preservation check, not browser layout validation.
