# Translation workflows

Start with the [quick start](../README.md#quick-start) and [provider setup](providers.md). The examples use the model from the quick start:

```bash
export TINBOX_MODEL="openai:gpt-6.1-sol"
```

You can substitute a compatible configured `provider:model` value. The filenames below are placeholders for your own documents. Each translation command makes model requests unless it includes `--dry-run`.

## Translate a PDF

Install the [PDF prerequisites](providers.md#pdf-requirements) and choose a model and endpoint that accept images. Tinbox rasterizes each PDF page and sends the image for translation. It automatically selects `page`, the only algorithm supported for PDFs.

```bash
tinbox translate document.pdf --model "$TINBOX_MODEL" --from ar --to en \
  --output translated.txt
```

The default rasterization resolution is 200 DPI. To change it, add `--pdf-dpi 300`. Higher resolution produces larger images and can increase token use. Check the result on a short document before processing a large one.

The output contains translated text. Tinbox does not export a translated PDF or preserve DOCX layout.

## Choose an output format

Use `--output` or redirect stdout to write a clean file. The translation plan, progress, statistics, and logs go to stderr, so stdout carries only the selected output, including when you select JSON.

Choose one of these formats:

- `text` (default): translated text only.
- `json`: structured metadata, translation results, statistics, warnings, and errors. The translated text is in `result.text`.
- `markdown`: a report with metadata, the translation inside a text code block, statistics, warnings, and errors.

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en \
  --format json --output translated.json
```

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en \
  --format markdown --output translated.md
```

The format follows `--format`, not the output filename extension. Create the output's parent directory first.

For PDFs, inspect the result for `[TRANSLATION_FAILED: Page N]` markers. A run can write partial results when some pages fail. JSON also records failures in `result.failed_pages`, `result.page_errors`, and `errors`.

With `context-aware` and `sliding-window`, a failed chunk or a `--max-cost` stop ends the run, but Tinbox still writes every chunk translated so far, followed by a `[TRANSLATION_FAILED: Chunks N-M of M]` marker (`Chunk M of M` when only the last one is missing) with the reason, and exits with code 1. JSON records the stop in `warnings` and `errors`. If no chunk was translated, the run fails without output as before.

JSON `metadata` records the provider in `model`, plus `model_name`, `input_tokens`, `output_tokens`, and `tinbox_version`.

## Review a text translation

Save source and translation chunks during a fresh context-aware `.txt` translation, then inspect them in an offline HTML report:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --from en --to de \
  --output translated.txt --save-review-input translated.review.json

tinbox review translated.review.json --output translated.review.html
```

The report keeps the translation unchanged. The [translation review guide](translation-review.md) covers German and Arabic examples, manual sentence matching, visible unchecked regions, and optional TypeSafe annotations for matched groups. Matching does not certify translation quality. Only `annotate-review` sends matched text to TypeSafe; preparing a matching draft and rendering saved annotations run offline. Use new output paths for review files, and keep their document contents in mind before sharing them.

## Split long text

Text and DOCX inputs use `context-aware` by default with the normal chat protocol. Set `--context-size` to a positive target size in characters, not tokens:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en \
  --context-size 3000 --output translated.txt
```

Tinbox splits near paragraph, sentence, or other text boundaries. Each request also includes nearby source text and the previous translation, so the chunk size is not the complete request size or the model's context limit.

If your document has explicit section markers, split on a literal string instead:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en \
  --split-token '[[SECTION]]' --output translated.txt
```

Put `[[SECTION]]` between sections in the input. Custom splitting ignores the target chunk size, drops empty chunks, and removes the marker. Tinbox concatenates translated chunks without inserting a replacement separator, so retain any required whitespace around your markers. `--split-token` applies only to `context-aware`.

## Reuse a glossary

Create `glossary.json` with an `entries` object that maps source terms to their translations:

```json
{
  "entries": {
    "كتاب": "book",
    "مكتبة": "library"
  }
}
```

Load it with `--glossary-file`, which turns the glossary on:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --from ar --to en \
  --glossary-file glossary.json \
  --save-glossary updated-glossary.json --output translated.txt
```

The model receives the current terms and can return new entries for later chunks. It cannot change a translation that came from `--glossary-file`, even by returning the term with different capitalisation or spacing: those entries are kept as written, including when a run resumes from a checkpoint. Terms the model adds can be updated by later chunks. `--save-glossary` writes the loaded and added terms together. To build a glossary from scratch, pass `--glossary` or only `--save-glossary`.

A glossary file that is not valid JSON, or whose `entries` is not an object of strings, stops the run before the cost estimate. A missing file is logged as a warning, and a missing file, an empty file, or `{}` starts an empty glossary, so the same path can be passed to both flags to start a new glossary. Glossaries require the chat protocol and are not supported by TranslateGemma.

## Resume an interrupted translation

Use a dedicated checkpoint directory for each document and configuration:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en \
  --checkpoint-dir .checkpoints/document --checkpoint-frequency 1 \
  --output translated.txt
```

After an interruption, rerun the same command. Tinbox automatically loads a matching checkpoint and reuses saved translations. There is no separate CLI resume flag. The default frequency saves after each completed page or chunk; a larger value can repeat more work after interruption.

Keep the input and all translation settings unchanged. Before resuming, Tinbox compares the checkpoint with the current run: a SHA-256 of the input file, the language pair, provider, model name, algorithm, protocol, local-model flag, and the chunking settings the algorithm uses (`--context-size` and `--split-token` for `context-aware`, window and overlap sizes for `sliding-window`). If anything differs, or the checkpoint was written by Tinbox 0.5.1 or older, it logs a warning naming the mismatch and translates from the beginning, overwriting the old checkpoint. The endpoint, glossary, reasoning effort, and PDF DPI are not compared.

Checkpoint filenames use only the input's stem, such as `document_checkpoint.json`. Two inputs with the same stem, such as `document.txt` and `document.pdf`, would take turns overwriting each other's checkpoint, so give each document its own directory. Use a new checkpoint directory to start over.

Tinbox deletes the checkpoint after writing a complete translation and any requested glossary. When a page-based translation finishes with failed pages, it still writes the output with `TRANSLATION_FAILED` placeholders, but it keeps the checkpoint and exits with code 1. Before exiting, Tinbox saves any completed pages that `--checkpoint-frequency` had not saved yet, so rerunning the same command retries only the failed pages and reuses the completed ones. When a `context-aware` or `sliding-window` run stops early, Tinbox saves the checkpoint at the stopping point, keeps it, and exits with code 1; rerun the same command to continue from the first untranslated chunk. After a `--max-cost` stop, rerun with a larger `--max-cost` instead: the saved cost already exceeds the old threshold, so the same command stops again before sending a request. Inspect the output before treating the document as complete.

Pair export with `--save-review-input` requires a fresh translation and rejects checkpoint resume. Omit that option for ordinary resume, or use a new checkpoint directory and new review output path to translate again. See [review recovery](translation-review.md#file-handling-and-recovery).

## Preview and limit estimated cost

Preview the plan without making translation requests:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en --dry-run
```

The estimate uses document-size heuristics and each model's list price from the [genai-prices](https://github.com/pydantic/genai-prices) catalog bundled with Tinbox; nothing is fetched at run time. For a model the catalog does not know, such as one behind a custom `openai:` endpoint, the estimate warns `Price for <model> unknown, using <provider> default` and uses a provider default rate. Set the real price in USD per 1M tokens with `--price-in` and `--price-out`:

```bash
tinbox translate document.txt --model openai:my-proxy-model --to en \
  --price-in 0.50 --price-out 1.50 --dry-run
```

Token counts, cost, and time are approximate, and a dry run does not test credentials or model compatibility.

To refuse the run when the estimate exceeds a threshold and stop when Tinbox's accumulated cost exceeds it:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en \
  --max-cost 1.00 --checkpoint-dir .checkpoints/document \
  --output translated.txt
```

`--max-cost` is not a billing cap. If the initial estimate for a cloud model already exceeds the threshold, Tinbox refuses to start unless you pass `--force`; local models skip this pre-flight check. During translation, Tinbox checks its calculated cost before each request, counting cost resumed from a checkpoint, so the request that crosses the threshold still completes and the threshold can be exceeded by one request. That cost includes replies the provider bills but Tinbox rejects. Once the threshold is exceeded, Tinbox stops sending requests, writes what it has translated with `TRANSLATION_FAILED` markers for the rest, and exits with code 1. To continue, rerun with a larger `--max-cost`. `--max-cost 0` is a real threshold: a cloud model with any estimated cost is refused before it starts, and with `--force` Tinbox stops after the first request. `--force` skips the confirmation prompt and the pre-flight refusal but does not disable the runtime threshold check.

Tinbox asks for confirmation before translating only when the estimate reaches $1.00 on a cloud model or when `--reasoning-effort` is above `minimal`. Short documents and local models at the default `minimal` effort start without a prompt. Scripts without a terminal that hit the prompt exit with code 1; pass `--force` there.

Start with the default `--reasoning-effort minimal`. For GPT-6.1 Sol and Claude Sonnet 5.5, this requests `low` effort; other model mappings are listed in the [CLI reference](cli.md#algorithms-and-models). Increase effort only if the model supports it and the result benefits from it. Higher effort can increase time and cost. TranslateGemma rejects non-default reasoning settings.

See the [CLI reference](cli.md) for all options and the [provider guide](providers.md) for endpoint configuration.
