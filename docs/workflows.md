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

Enable the glossary explicitly when loading or saving it:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --from ar --to en \
  --glossary --glossary-file glossary.json \
  --save-glossary updated-glossary.json --output translated.txt
```

`--glossary-file` and `--save-glossary` do not enable `--glossary` themselves. To build a glossary during translation, omit `--glossary-file` and retain the other flags. The model receives the current terms and can return new entries for subsequent chunks.

Check the logs if expected terms are missing: an absent or malformed glossary file is logged, and translation continues with an empty glossary. Glossaries require the chat protocol and are not supported by TranslateGemma.

## Resume an interrupted translation

Use a dedicated checkpoint directory for each document and configuration:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en \
  --checkpoint-dir .checkpoints/document --checkpoint-frequency 1 \
  --output translated.txt
```

After an interruption, rerun the same command. Tinbox automatically loads a matching checkpoint and reuses saved translations. There is no separate CLI resume flag. The default frequency saves after each completed page or chunk; a larger value can repeat more work after interruption.

Keep the input and all translation settings unchanged. Checkpoints check the language pair, provider, algorithm, protocol, and local-model flag, but do not validate input contents, the exact model name, endpoint, chunk settings, or PDF DPI. A changed configuration can therefore reuse unsuitable results.

Checkpoint filenames use only the input's stem, such as `document_checkpoint.json`. Separate directories prevent collisions between inputs such as `document.txt` and `document.pdf`. Use a new checkpoint directory to start over.

Tinbox deletes the checkpoint after writing output and any requested glossary. This cleanup also occurs when a page-based translation writes partial results with failed pages. Inspect the output before treating the document as complete.

Pair export with `--save-review-input` requires a fresh translation and rejects checkpoint resume. Omit that option for ordinary resume, or use a new checkpoint directory and new review output path to translate again. See [review recovery](translation-review.md#file-handling-and-recovery).

## Preview and limit estimated cost

Preview the plan without making translation requests:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en --dry-run
```

The estimate uses document-size heuristics and fixed provider-level prices. It does not look up the selected model's current price. Token counts, cost, and time are approximate, and a dry run does not test credentials or model compatibility.

To prompt when the estimate exceeds a threshold and stop when Tinbox's accumulated cost exceeds it:

```bash
tinbox translate document.txt --model "$TINBOX_MODEL" --to en \
  --max-cost 1.00 --checkpoint-dir .checkpoints/document \
  --output translated.txt
```

`--max-cost` is not a billing cap. The initial estimate can trigger a confirmation prompt rather than refuse the run. During translation, Tinbox checks its calculated cost after a completed request, so the threshold can be exceeded. Those calculations use fixed provider-level prices too. `--force` skips confirmation prompts but does not disable the runtime threshold check.

Start with the default `--reasoning-effort minimal`. For GPT-6.1 Sol and Claude Sonnet 5.5, this requests `low` effort; other model mappings are listed in the [CLI reference](cli.md#algorithms-and-models). Increase effort only if the model supports it and the result benefits from it. Higher effort can increase time and cost. TranslateGemma rejects non-default reasoning settings.

See the [CLI reference](cli.md) for all options and the [provider guide](providers.md) for endpoint configuration.
