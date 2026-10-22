# Translation review

[README](../README.md) · [CLI reference](cli.md#review-commands) · [Workflows](workflows.md#review-a-text-translation)

Tinbox can preserve each source chunk beside the translation returned for that chunk. The `review` command turns this saved JSON bundle into a standalone HTML report for human inspection. An optional matching file selects smaller sentence groups to compare while keeping ambiguous regions unchecked.

Review is opt-in. It does not change the translated text or make corrections. A separate, optional TypeSafe command checks matched groups for changes to participant roles, certainty, and quantifiers. Saved annotations can appear in the report, but they do not certify translation quality.

## German translation example

From a source checkout, this command translates the example story and saves the pairs:

```bash
tinbox translate ./examples/elara_story.txt --from en --to de \
  --model openai:gpt-4o --algorithm context-aware \
  --output elara.de.txt --save-review-input elara.de.review.json
```

The translation uses the selected provider as usual, including its SDK, API key, and charges. In this example, OpenAI needs `OPENAI_API_KEY` and Tinbox's `openai` extra. Context-aware mode is already the default for ordinary `.txt` translations, so `--algorithm context-aware` is optional here.

The command writes two files:

- `elara.de.txt` contains the normal translation.
- `elara.de.review.json` contains the ordered source and translation pairs plus the translation settings.

The report command reads the JSON bundle:

```bash
tinbox review elara.de.review.json --output elara.de.review.html
```

Open `elara.de.review.html` in a browser. Every chunk appears beside its translation in document order. On narrow screens, the two columns stack. Whitespace and line breaks remain visible, and text that resembles HTML appears as text.

## Arabic translation example

Arabic uses the same workflow with separate output paths:

```bash
tinbox translate ./examples/elara_story.txt --from en --to ar \
  --model openai:gpt-4o --context-size 1500 \
  --output elara.ar.txt --save-review-input elara.ar.review.json

tinbox review elara.ar.review.json --output elara.ar.review.html
```

The report supports right-to-left text. Source and translation remain in separate columns, so an English source can be read beside an Arabic translation.

## Offline reports

`tinbox review` reads the saved bundle and, when supplied, its matching and annotation files. It does not translate again, call an AI provider, or require the original source file. The base Tinbox installation is sufficient to generate a report. No provider SDK or API key is needed for this command, including when it displays saved TypeSafe annotations.

The HTML file contains its own styles and needs no external assets or network connection. The report shows all chunks without a translation quality verdict or a pass/fail result. A report records text for inspection; it does not certify translation quality.

Command help is available locally:

```bash
tinbox translate --help
tinbox prepare-matches --help
tinbox annotate-review --help
tinbox review --help
```

## Compare sentence groups

Sentence matches are optional and separate from the saved translation. Tinbox does not automatically align sentences in this version. A person or an assistant must read both texts and select the groups. Equal sentence counts or similar punctuation alone do not establish a match.

First, create an unchecked draft from an existing bundle:

```bash
tinbox prepare-matches elara.de.review.json --output elara.de.matches.json
```

The draft covers every source and translation chunk and labels every group `unchecked`. Its provenance is `unreviewed`. Preparing the draft runs offline and makes no model requests.

Next, edit a copy of the matching file to select sentence groups. Keep the saved review bundle unchanged. A group can pair one source sentence with one translation sentence, one with several, or several with one. Leave any uncertain correspondence as `unchecked` and record the reason. An unchecked group can contain text on just one side when no corresponding text can be identified on the other side.

Set the file's provenance to `assistant-curated` when an assistant selects the matches, or `human-reviewed` after a person reviews them. Assistant-curated matches have not been verified by a human. Neither provenance value certifies translation quality, and Tinbox cannot verify who reviewed a file. An `unreviewed` draft cannot contain groups marked `matched`.

Render the curated file with its original bundle:

```bash
tinbox review elara.de.review.json --matches elara.de.curated.matches.json \
  --output elara.de.matched.html
```

The report labels each group **Matched for comparison** or **Unchecked**, shows its recorded reason, and counts both categories. Every chunk also has a **Full original chunk pair** section that you can expand. Unchecked text remains visible; it is never treated as a successful review. This command runs no quality checks. To display saved TypeSafe results, explicitly add the annotation file as described below.

For Arabic, use the same commands with the Arabic bundle and separate output paths. The report keeps the source and translation in separate columns and detects each text's writing direction.

The [real-document pilot](../benchmarks/translation_review_pilot/README.md) includes saved German and Arabic translations, assistant-curated matching files, HTML reports, reproduction commands and observed limitations. It demonstrates the offline workflow on a short excerpt from the example story.

### Exact text ranges

The matching file refers to text in the bundle through `start` and `end` offsets. These are Python string indices measured in Unicode code points, with an inclusive start and exclusive end: `text[start:end]`. They are not UTF-8 byte offsets or JavaScript UTF-16 offsets. For example, `TextSpan(start=0, end=len(text))` selects an entire string. A combining mark is a separate code point.

Each chunk has a `unit_id` and ordered `groups`. Each group contains a `status`, lists of `source` and `translation` spans, and a `reason`. Multiple spans on one side represent grouped sentences. A missing side uses an empty list, not an empty span, and is permitted only for an unchecked group. Record a nonempty reason for every unchecked group.

Together, the groups must cover every character on both sides exactly once, including spaces and line breaks. Include whitespace with an adjacent sentence rather than dropping it. Span ranges must stay within their original chunk, in order, without gaps or overlaps. Crossed matches and sentence reordering cannot be represented as separate matches; keep the affected region unchecked instead. Every chunk must be present in the same order as the bundle.

The matching file uses its own `schema_version: 1`. Its `bundle_sha256` binds it to the complete canonical review bundle, including metadata and text. This differs from the bundle's `source_sha256`, which identifies the original input file's bytes. JSON indentation and key order do not change the canonical fingerprint. Editing bundle text or metadata does, so generate a new draft and review the matches again after such an edit.

The loader checks ranges, coverage, order, and the bundle fingerprint before rendering. Those checks prove that references are structurally valid. They do not prove that the selected sentences correspond in meaning.

## Add optional TypeSafe annotations

Use TypeSafe after curating the sentence matches. The annotation command sends each matched group to TypeSafe and saves the responses in a new JSON file. It does not send unchecked groups, filenames, or curation reasons. Each request contains only the selected spans; the rest of the chunk is not added as context. The matching file controls which exact source and translation spans leave your machine.

Put `TYPESAFE_API_KEY=...` in a local, gitignored `.env` file, or export that variable through your usual secret-management workflow. Tinbox reads the environment and does not load `.env` files automatically. To load a trusted shell-compatible `.env` file explicitly in bash or zsh:

```bash
set -a
source .env
set +a
```

Sourcing executes shell code, so use only a file you control. Do not paste the key into reports or commit it. No TypeSafe SDK or additional Tinbox extra is needed.

Create an annotation file from the original bundle and curated matches:

```bash
tinbox annotate-review elara.de.review.json --matches elara.de.curated.matches.json \
  --output elara.de.annotations.json
```

This is the network step and uses TypeSafe's API. For every matched group, Tinbox sends the exact selected source text and translation, their language codes, and all three fixed questions in one request. The model defaults to `jev-1.14.0`. The checks ask whether the translation changes:

- **Participant roles:** who acts, experiences a state, speaks, or receives an action, including pronoun referents and possessors.
- **Certainty:** whether a claim is tentative, likely, possible, or definite, including the direction of uncertainty.
- **Quantifiers:** quantities, numerical bounds, frequency, or the scope of quantified claims and negation.

Requests run sequentially, with no automatic retries. `--model` selects a different pinned TypeSafe model; names ending in `-latest` are rejected because they can change between runs. `--timeout` sets a positive finite timeout in seconds (default `30`), and `--max-groups` sets the maximum number of matched groups (default `100`). Tinbox rejects an over-limit run before sending any requests. Each serialized request is limited to 16 KiB and each response to 1 MiB. An entirely unchecked matching file produces only skipped records and requires neither an API key nor network access.

Render the saved results with the same bundle and matches:

```bash
tinbox review elara.de.review.json --matches elara.de.curated.matches.json \
  --annotations elara.de.annotations.json --output elara.de.annotated.html
```

This command runs offline and requires no API key. For Arabic, use the Arabic bundle and its matching file with separate annotation and HTML paths. Text direction and the original source and translation spans are preserved.

### Read the annotations

Matching status and TypeSafe outcomes are separate. **Matched for comparison** identifies the selected text; it does not say whether TypeSafe assessed that text or whether its translation is correct.

Completed checks show all three stored scores. A score is TypeSafe's estimated probability that the answer to that change-detection question is yes. It is not a severity measure or a translation quality percentage. The report labels a group **TypeSafe concern** when any score is at least `0.70`; otherwise it says **No concern flagged by these checks**. The threshold is experimental. These checks can miss errors, so both outcomes still need human review.

Unchecked groups show **Not assessed: unchecked match**. A failed request shows **Request failed** with a safe error category. Groups that were never submitted show **Not attempted**. The report counts completed groups, flagged groups, skipped groups, request errors, and groups not attempted. Expand **TypeSafe model, timing, and check definitions** to inspect the recorded setup.

### Saved annotations and partial failures

The annotation file uses `schema_version: 1`. It records the requested model, timestamps, threshold, fixed questions, and an outcome for every group. Completed groups retain the exact request and complete response, including returned model and usage, so an interesting result can be reproduced or prepared for feedback. API keys, authorization headers, and raw failed HTTP response bodies are not stored.

Canonical fingerprints bind annotations to both the complete bundle and the matching file. The loader also validates requests against their selected spans and checks response structure and scores. Changing either input's text or metadata makes the annotations stale. Use the original files or generate new annotations after reviewing the changed matches. Indentation and JSON key order do not affect canonical fingerprints.

The command stops at the first request failure, saves completed results and the failed request's safe error category, and marks later matched groups **Not attempted**. It exits with status `1` after saving that partial annotation file. A high concern score alone does not cause a failing exit status. Timeouts can leave the remote request's outcome unknown, which is why Tinbox does not retry automatically. Rendering the partial file remains available offline. Rerunning annotation requires a new output path and repeats the requests; it does not resume from the previous file.

## Supported translations

Pair export has these limits:

- The input must be a nonempty `.txt` file.
- The algorithm must be `context-aware`, whether selected by default or explicitly.
- The translation must start fresh. A run that resumes saved translation chunks cannot export review pairs.
- PDF, DOCX, `page`, `sliding-window`, and TranslateGemma translations are not supported for pair export.

Without `--save-review-input`, normal translation and checkpoint behavior remain unchanged. The existing text, JSON, and Markdown output formats also remain unchanged when pair export is enabled. The review bundle is a separate file.

`--context-size`, `--split-token`, glossary options, and reasoning settings still apply to the translation. `--dry-run` estimates the translation and writes neither a review bundle nor a report.

## What a pair means

Tinbox records a pair at each successful model response. The source is the actual current chunk sent for translation, and the translation is the returned text for that chunk. Nearby source text and an earlier translation may help the model as context, but that context is not added to the pair's source.

A chunk can contain several sentences or paragraphs. A pair does not claim that individual sentences align. Tinbox does not split the final translation and try to match those pieces to the source afterward.

The bundle preserves the chunks used by the translation algorithm, not the original file's byte layout. For example, the existing custom split-token behavior removes the delimiter before translation. The original source filename and SHA-256 fingerprint identify the input file separately.

## Saved bundle

The UTF-8 JSON bundle uses `schema_version: 1`. The `pairs` list contains units with a consecutive, one-based `unit_id`, a `source` string, and a `translation` string. Their order is the translation order.

The bundle records these metadata fields:

| Field | Meaning |
| --- | --- |
| `created_at` | Bundle creation time with a UTC offset. |
| `source_name`, `source_sha256` | Original source filename and SHA-256 fingerprint of its bytes. |
| `source_lang`, `target_lang` | Configured language codes. An automatically detected source remains `auto` in the metadata. |
| `provider`, `model_name` | Selected model provider and model name. |
| `algorithm` | `context-aware` for this schema version. |
| `context_size` | Target chunk size in characters. |
| `custom_split_token` | Custom delimiter, or `null` when none is set. A delimiter takes precedence over the target chunk size. |
| `use_glossary` | Whether the translation uses a glossary. Glossary entries are not included. |
| `reasoning_effort` | Configured reasoning setting. |

The bundle does not store API keys.

For example, a unit can contain:

```json
{
  "unit_id": 1,
  "source": "Elara opened the door.",
  "translation": "Elara öffnete die Tür."
}
```

This is a unit illustration, not a complete bundle. Use the bundle produced by `--save-review-input` as the input to `tinbox review`. Malformed bundles and unsupported schema versions produce an error instead of a partial report.

## File handling and recovery

Review JSON, matching JSON, annotation JSON, and HTML destinations must be new files, and their parent directories must already exist. Tinbox refuses to overwrite an existing review file. Choose different paths for the source, translated output, review bundle, matching file, annotations, report, and any glossary files. `--force` applies to translation cost warnings and does not override review-file checks.

Tinbox writes the normal translation and any requested glossary before the review bundle. If bundle writing fails after translation, the command reports the error and leaves those outputs in place. A missing bundle cannot be reconstructed from the final translation alone.

If HTML report generation fails, fix the destination or bundle problem and rerun `tinbox review`. A valid saved bundle is enough; there is no need to pay for another translation.

Common problems have these remedies:

| Problem | Remedy |
| --- | --- |
| A review destination already exists | Choose a new filename. `--force` does not replace review files. |
| A destination's parent directory is missing | Create the directory before running the command. |
| Output paths overlap | Choose distinct paths for the source, translation, saved glossary, and review bundle. Do not use the checkpoint filename for an output. |
| Pair export encounters a saved checkpoint | Omit `--checkpoint-dir` or choose a new checkpoint directory to start fresh. A fresh run translates the document again and incurs the provider's normal charges. |
| The input or algorithm is unsupported | Use a nonempty `.txt` input with context-aware mode, or omit `--save-review-input` for an ordinary translation. |
| The bundle is malformed or has an unsupported schema version | Use an intact bundle produced by the supported Tinbox version. |
| The matching file belongs to a different bundle | Use its original bundle, or prepare a new draft from the intended bundle and review the matches again. |
| Match ranges leave gaps, overlap, or appear out of order | Restore complete, ordered coverage on both sides, including whitespace. Keep ambiguous regions unchecked. |
| An unchecked group has no reason | Record why its correspondence remains unchecked. |
| An unreviewed draft contains matched groups | Set accurate provenance after curating or reviewing the groups, or leave them unchecked. |
| `annotate-review` cannot find a TypeSafe key | Export `TYPESAFE_API_KEY` in the command's environment. Tinbox does not load `.env` automatically. |
| The annotation command exceeds a request or group limit | Review smaller groups or raise `--max-groups` deliberately. Payload size limits cannot be bypassed with that option. |
| An annotation request fails | Inspect the saved partial results and safe error category. Fix the cause before rerunning with a new output path; requests are not automatically retried or resumed. |
| Annotations belong to a different bundle or matching file | Use their original inputs, or generate a new annotation file for the reviewed matches. |
| `review --annotations` has no matching file | Supply the exact matching file used by `annotate-review` with `--matches`. |

A fresh translation can save checkpoints while exporting pairs. If that run is interrupted, resuming it with pair export is rejected because the saved checkpoint does not preserve trustworthy source/translation pairs. Ordinary checkpoint resume remains available when pair export is omitted.

## Content and sharing

The JSON bundle and HTML report contain copies of the source chunks and translated text. Annotation JSON contains matched text in its saved requests and the TypeSafe responses. The matching file contains ranges, provenance, and reasons, which may also quote document text. They are local, readable files, not encrypted storage. Sharing a file shares its content and recorded metadata.

Saving pairs adds no provider request beyond the normal translation. Preparing a matching draft and generating or viewing the report send no content to TypeSafe or another service. Only the explicit `annotate-review` step sends the selected matched text to TypeSafe. If you give a bundle to an external assistant to curate matches, that is a separate action that shares its contents with that assistant.
