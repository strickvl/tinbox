# Translation review pilot

This pilot uses real Tinbox CLI translations into German and Arabic, followed by assistant-curated sentence matching and offline reports. The initial translation and matching stage made no TypeSafe calls. A later opt-in annotation run, recorded below, adds limited review hints without changing either translation.

The input is the first 12 double-newline-separated paragraphs of `examples/elara_story.txt`, including the heading and final double newline. The excerpt has 2,180 characters and 2,230 UTF-8 bytes. Each run saves its own exact source, the original example's fingerprint and the excerpt range.

## Reproduce

Run from the repository root with the locked development environment installed:

```bash
uv run --no-sync --locked --all-extras python benchmarks/translation_review_pilot/run.py \
  --run-name my-fresh-pilot --check

uv run --no-sync --locked --all-extras python benchmarks/translation_review_pilot/run.py \
  --run-name my-fresh-pilot
```

`--check` validates the excerpt without writing files or using credentials or network access. The live command requires `OPENAI_API_KEY` in the repository `.env` or existing process environment. It reads only that key from `.env`, does not execute shell assignments or expand variables, and removes API credentials for other providers from the translation subprocess environment. The API key is never included in the manifest or command line. The standard OpenAI endpoint is selected explicitly.

Each language runs the real `tinbox translate` command with `openai:gpt-4.1-mini-2025-04-14`, context-aware mode, a target chunk size of 1,000 characters, `--force`, `--max-cost 1`, a text output, and a review bundle. The runner does not change Tinbox's provider behavior. Each subprocess has a 180-second deadline. Tinbox and SDK retries can occur inside that deadline, but the runner never retries a failed language or an entire run. A failed run stops before starting the next language and retains its logs and any existing output.

Every live run requires a fresh directory name under `runs/`. Existing runs cannot be overwritten or resumed. Inspect the failure before intentionally starting another run.

## Saved evidence

Each run directory contains:

- `source.txt`: the exact input excerpt.
- `manifest.json`: source selection and hashes, Git revision, source code and lockfile fingerprints, package versions, configuration, commands, progress, durations and verification results.
- `de.txt` and `ar.txt`: unmodified CLI translation output, when successful.
- `de.review.json` and `ar.review.json`: preserved source/translation chunks, when successful.
- `de.cli.log` and `ar.cli.log`: captured CLI output, including errors and statistics.

The runner checks that the source hash in each review bundle matches the saved input and that concatenating its translation chunks exactly reproduces the output file. Chunking can change source whitespace, so concatenated source chunks are not asserted to reproduce the original bytes.

CLI token totals come from provider response usage. CLI costs use Tinbox's approximate provider-wide rates and are rounded to four decimal places. They are not model-specific charges or invoices. CLI time is also rounded; the manifest separately records subprocess wall time.

Sentence matching and report generation are separate offline steps. They do not rerun translation or change these outputs.

## Recorded attempts

- `2026-10-22-elara-cli-01` stopped during a local CLI import while concurrent sentence-matching implementation was incomplete. No provider call started.
- `2026-10-22-elara-cli-02` reached OpenAI and received HTTP 400 because Tinbox explicitly requested 100,000 completion tokens but the selected model allows at most 32,768. No translation completed; Arabic was not attempted.
- `2026-10-22-elara-cli-03` completed both languages with the same CLI arguments after a small provider fix omitted the default `max_tokens` value. The model server now supplies its own output limit. The manifest records the changed code fingerprints.

These failed attempts are retained as evidence. They are not benchmark cases or translation-quality failures.

The successful run completed on 22 October 2026 at 21:28 CEST (Europe/Amsterdam):

| Target | Paired chunks | Provider-reported tokens | Subprocess wall time | Tinbox approximate cost |
| --- | ---: | ---: | ---: | ---: |
| German | 3 | 3,346 | 6.99 seconds | $0.0099 |
| Arabic | 3 | 3,363 | 7.04 seconds | $0.0100 |

Both bundles passed the source fingerprint and exact translation concatenation checks. These six paired chunks establish that the real CLI can preserve text for inspection. They do not measure translation quality or sentence-matching accuracy.

## Sentence matching results

Open the [German report](runs/2026-10-22-elara-cli-03/de.matched.html) or [Arabic report](runs/2026-10-22-elara-cli-03/ar.matched.html). Both retain the complete original chunk pairs in expandable sections. The matching files select exact substrings of the saved translations; they do not rewrite them.

| Target | Matched groups | Matched source / translation spans | Unchecked regions |
| --- | ---: | ---: | --- |
| German | 34 | 34 / 34 | 1 non-sentential heading |
| Arabic | 31 | 34 / 34 | 1 non-sentential heading |

The Arabic report keeps Lume's four-sentence speech together because the translation moves its speaker attribution and changes quotation boundaries. Its four source spans and four translation spans form one comparison group. All other matched groups in both languages contain one span per side. Detailed boundary decisions are recorded in the [German notes](runs/2026-10-22-elara-cli-03/de-matching-notes.md) and [Arabic notes](runs/2026-10-22-elara-cli-03/ar-matching-notes.md).

No genuinely ambiguous prose correspondence was found in this short excerpt. The headings are unchecked because they are not sentences. This pilot therefore does not establish how an automatic aligner would perform or abstain on omissions, reordering or ambiguous references. Tinbox does not include an automatic aligner: the scripts below reproduce decisions selected by reading the actual bilingual text, and require these exact bundles. The labels remain assistant-curated and need bilingual human review before use as evaluation ground truth.

### Reproduce the offline steps

The successful run includes `de.draft.matches.json` and `ar.draft.matches.json`, produced by the actual `prepare-matches` command. Every chunk in these initial drafts is unchecked. Their curated counterparts are `de.matches.json` and `ar.matches.json`.

To prepare a new draft or render the saved curation, choose unused output paths:

```bash
pilot_run=benchmarks/translation_review_pilot/runs/2026-10-22-elara-cli-03

uv run --no-sync --locked --all-extras tinbox prepare-matches \
  "$pilot_run/de.review.json" --output /tmp/elara-de-draft.matches.json

uv run --no-sync --locked --all-extras tinbox review \
  "$pilot_run/de.review.json" --matches "$pilot_run/de.matches.json" \
  --output /tmp/elara-de-matched.html

uv run --no-sync --locked --all-extras tinbox review \
  "$pilot_run/ar.review.json" --matches "$pilot_run/ar.matches.json" \
  --output /tmp/elara-ar-matched.html
```

The authored mappings can also be regenerated byte for byte:

```bash
uv run --no-sync --locked --all-extras python \
  benchmarks/translation_review_pilot/curate_de.py /tmp/elara-de-curated.matches.json

uv run --no-sync --locked --all-extras python \
  benchmarks/translation_review_pilot/curate_ar.py --output /tmp/elara-ar-curated.matches.json
```

These commands require no provider credentials and make no model requests. The [translation review guide](../../docs/translation-review.md) explains the matching format and how to leave uncertain regions unchecked.

## Verification and next step

Both reports were produced through the real CLI with provider credentials removed from the subprocess environment. All 76 German and 70 Arabic HTML text blocks, including the original chunk pairs, were parsed and compared with their exact saved substrings. Every character is covered once in order. Original source, translation and bundle fingerprints still match the live-run manifest. The evidence is saved in [report-verification.json](runs/2026-10-22-elara-cli-03/report-verification.json).

The full repository suite passed with 570 tests, together with Ruff lint, Ruff formatting and ty. Tests exercise grouped sentences, one-sided unchecked text, stale matching files and complete coverage. These tests establish structural behavior; they do not measure semantic matching accuracy. Browser layout was not visually verified in this run.

The pilot also exposed and verified a provider fix: ordinary OpenAI-compatible requests now omit an unspecified output-token limit instead of always requesting 100,000 tokens. Explicit limits still work, and TranslateGemma keeps its 2,048-token default.

Before interpreting annotation results, a bilingual reviewer should inspect these groups and add examples with genuinely ambiguous correspondence.

## Opt-in TypeSafe annotation run

The separate annotation run completed on 22 October 2026 at 22:07 CEST (Europe/Amsterdam). Open the [annotated German report](runs/2026-10-22-elara-cli-03/de.annotated.html) or [annotated Arabic report](runs/2026-10-22-elara-cli-03/ar.annotated.html). The original unannotated reports above remain intact.

Each matched group received one request to `jev-1.14.0` with the three fixed checks for participant roles, certainty and quantifiers. The requests contained only the group's exact source and translation spans and their language codes. Both unchecked headings were excluded. There were no automatic retries.

| Target | Completed requests | Unchecked groups skipped | Request failures | Concerns at 0.70 | Input / output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| German | 34 | 1 | 0 | 0 | 31,586 / 1,836 |
| Arabic | 31 | 1 | 0 | 0 | 29,431 / 1,674 |

These are provider-reported usage totals for the successful requests. No claim about billed cost is inferred from them. The complete two-language run, including offline rendering and text verification, took 20.65 seconds in this execution; this is one observed run, not a performance benchmark.

No score reached the fixed `0.70` concern threshold. The highest individual scores were `0.24` for German and `0.39` for Arabic. These outcomes are **No concern flagged by these checks**, not a translation-quality pass. The excerpt has no adjudicated quality labels and no planted errors. It cannot measure accuracy or establish that the texts are error-free.

`de.annotations.json` and `ar.annotations.json` preserve exact requests, complete successful responses, model, timestamps and usage. [annotation-run.json](runs/2026-10-22-elara-cli-03/annotation-run.json) records the CLI commands, code fingerprints and verification results. Every annotation was validated against its bundle and matching file. All 76 German and 70 Arabic HTML text blocks still match the original reports exactly, and the original source, translation, bundle and matching file hashes are unchanged. No credentials are stored in these artifacts.

The live commands used `--timeout 20 --max-groups 40` for each language. To run again, explicitly export `TYPESAFE_API_KEY` as described in the [guide](../../docs/translation-review.md#add-optional-typesafe-annotations), then choose new output paths. Each rerun makes new paid requests; there is no automatic resume:

```bash
pilot_run=benchmarks/translation_review_pilot/runs/2026-10-22-elara-cli-03

tinbox annotate-review "$pilot_run/de.review.json" --matches "$pilot_run/de.matches.json" \
  --timeout 20 --max-groups 40 --output /tmp/elara-de-new.annotations.json

tinbox annotate-review "$pilot_run/ar.review.json" --matches "$pilot_run/ar.matches.json" \
  --timeout 20 --max-groups 40 --output /tmp/elara-ar-new.annotations.json
```

Rendering the saved annotations requires no API key or network access:

```bash
tinbox review "$pilot_run/de.review.json" --matches "$pilot_run/de.matches.json" \
  --annotations "$pilot_run/de.annotations.json" --output /tmp/elara-de-annotated.html

tinbox review "$pilot_run/ar.review.json" --matches "$pilot_run/ar.matches.json" \
  --annotations "$pilot_run/ar.annotations.json" --output /tmp/elara-ar-annotated.html
```

The annotation implementation passed the full 678-test suite, Ruff lint and formatting, and ty. An independent review found no actionable issues, and an import check succeeded with optional provider SDKs and `httpx` unavailable. Browser layout was not visually checked in this run; text preservation and outcome labels were checked programmatically.
