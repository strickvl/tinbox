# Changelog

## 0.6.0 (2026-11-01)

### Cost

- Price each model by name. Tinbox now uses the [genai-prices](https://github.com/pydantic/genai-prices) catalog bundled with the package, including long-prompt tiers and dated price changes, instead of one September 2025 rate per provider that priced every Anthropic model as Sonnet 4. Unknown model names fall back to a provider default (checked 1 November 2026) and the estimate warns `Price for <model> unknown, using <provider> default`.
- Add `--price-in` and `--price-out` to `translate`, in USD per 1M tokens, for custom or OpenAI-compatible endpoints. They override the catalog for the estimate, recorded cost, and `--max-cost`.
- Count replies the provider bills but Tinbox rejects, such as a truncated reply or invalid JSON, toward recorded cost and `--max-cost`. Previously they cost nothing in Tinbox's accounting.
- Ask for confirmation only when the estimated cost reaches $1.00 on a cloud model or `--reasoning-effort` is above `minimal`. Short documents and local models at the default `minimal` effort start without a prompt. When confirmation is needed and no terminal is attached, `tinbox translate` exits with code 1 and names `--force` instead of waiting on a prompt that cannot be answered.
- Refuse to start a cloud model when the estimate already exceeds `--max-cost`; `--force` overrides the refusal. Local models skip this check.
- Treat `--max-cost 0` as a real threshold. Previously `0` disabled the check. A cloud model with any estimated cost is now refused before it starts unless you pass `--force`, which then allows one request.
- Check `--max-cost` before each request, counting cost resumed from a checkpoint. Previously the check ran after each request, so every rerun after a cost stop sent one more request before stopping again. After a cost stop, Tinbox now says to rerun with a larger `--max-cost` instead of rerunning the same command.

### Failures, retries, and resume

- Resume from a checkpoint only when it matches the current run. Checkpoints now record a SHA-256 of the input file, the model name, and the chunking settings, and Tinbox compares them before reusing saved chunks. Previously a rerun with a different `--context-size`, an edited input, or a different file with the same name silently spliced old translations onto new text and dropped the rest. Mismatches, and checkpoints written by 0.5.1 or older, are logged and the translation starts from the beginning.
- Keep the checkpoint and exit with code 1 when a page-based translation finishes with failed pages. The output is still written with `TRANSLATION_FAILED` placeholders, and rerunning the same command retries only the failed pages: before exiting, Tinbox saves completed pages that `--checkpoint-frequency` had not saved yet. Previously the run exited 0 and deleted the checkpoint, so retrying one page meant paying for every page again.
- Keep paid-for work when a run stops early. A failed chunk or a `--max-cost` stop in `context-aware` or `sliding-window` now writes the chunks translated so far, ends the output with a `TRANSLATION_FAILED` marker, saves and keeps any checkpoint, and exits with code 1. A `--max-cost` stop in page mode marks the remaining pages as skipped instead of discarding the translated pages. Previously all three raised an error and wrote nothing.
- Stop retrying requests that cannot succeed. A bad API key, a missing permission, or an OpenAI `insufficient_quota` 429 now fails the page immediately, marks the remaining pages as skipped, and stops sending requests. Previously each page retried for about six minutes. Transient rate limits still retry, now honouring a numeric `retry-after` header (capped at 120 s), and the last failure surfaces the provider's own error instead of a `RetryError` wrapper.
- Stop on more account-limit errors. Anthropic's 400 "credit balance is too low", and Gemini's daily-quota, zero-quota, spending-cap, and 402 errors, now stop the run the same way an OpenAI `insufficient_quota` error does. Previously Tinbox retried each page for about six minutes on a Gemini daily-quota 429, and kept sending a failing request for every remaining page on Anthropic. Gemini's per-minute 429s are still retried.

### Inputs and glossary

- Read tables, headers, footers, footnotes, and endnotes from DOCX files, in reading order. Previously only body paragraphs were read, so table contents were never translated and nothing said they were missing. Each table row becomes one line with cells separated by ` | `, and merged cells appear once. Paragraphs are now separated by a blank line, so `context-aware` and `sliding-window` can split DOCX text at paragraph breaks. First-page and even-page headers and footers are read only while that option is on in Word. Cost estimates count the same text, and checkpoints saved before this change are not resumed, because the chunks they recorded no longer line up.
- Convert PDF pages to images one page at a time. Tinbox still prepares every page before the first request, but it now keeps only the compressed PNGs instead of holding every decoded page at once. Previously a 300-page PDF at 200 DPI needed about 3.5 GB of memory; in testing, peak memory for a 63-page PDF fell from 3.2 GB to 160 MB.
- Turn the glossary on when `--glossary-file` or `--save-glossary` is passed. Previously both were ignored without `--glossary`. Translations loaded from `--glossary-file` are now kept as written: the model can add terms but no longer overwrites yours, even with different capitalisation or spacing. An unreadable or malformed glossary file now stops the run before the cost estimate instead of being replaced by an empty glossary. A missing or empty file, or `{}`, still starts an empty glossary.

### Output

- Record `model_name`, `input_tokens`, `output_tokens`, and `tinbox_version` in JSON `metadata`. Existing keys are unchanged; `model` still holds the provider.

### Changes that can affect scripts

- `tinbox translate` now exits with code 1 when any page or chunk fails, or when `--max-cost` stops the run. The partial output is still written.
- A run that needs confirmation with no terminal attached exits with code 1; pass `--force` in scripts that should proceed regardless of cost.
- `--max-cost 0` now refuses any cloud run with a non-zero estimate.
- Checkpoints written by earlier versions are not resumed; the run starts from the beginning.
- `--glossary-file` and `--save-glossary` now turn the glossary on, and a malformed glossary file stops the run.
- Tinbox now depends on `genai-prices` for model prices.

### Upgrade

```bash
uv tool upgrade tinbox
tinbox --version
```

When changing extras or Python versions, or replacing a GitHub installation, use `uv tool install --force --python 3.13 "tinbox[all]>=0.6.0"`.

## 0.5.1 (2026-10-29)

- Accept Latin as a source or target language: `la`, `lat`, or `latin` (#47).
- Ignore invisible characters, such as zero-width spaces, in `--from` and `--to` language codes (#47). A command pasted from a web page or chat app could carry one, and `--from fr` then failed with `Unsupported language code: fr`. That error now escapes non-ASCII characters, so a look-alike letter shows up as, for example, `'f\u0433'`.

### Upgrade

```bash
uv tool upgrade tinbox
tinbox --version
```

When changing extras or Python versions, or replacing a GitHub installation, use `uv tool install --force --python 3.13 "tinbox[all]>=0.5.1"`.

## 0.5.0 (2026-10-28)

- Read `.md`, `.markdown`, `.rst`, `.org`, `.adoc`, `.asciidoc`, and `.text` files as plain text, the same way as `.txt` (#43). Unsupported extensions now name the supported ones instead of failing with `'md' is not a valid FileType`.
- Print the translation plan, cost estimate, progress, statistics, and errors from `tinbox translate` to stderr (#44). Stdout now carries only the translation, so `tinbox translate notes.md ... > notes.de.md` writes a clean file. Scripts that parsed those tables from stdout should read stderr instead.
- Send the overlap between `sliding-window` windows as context the model must not translate, instead of translating it twice. Keep whitespace between windows, and skip incompatible checkpoints saved by older versions (#41).
- Keep whitespace when `smart_text_split` splits at the end of a window (#42).

### Upgrade

```bash
uv tool upgrade tinbox
tinbox --version
```

When changing extras or Python versions, or replacing a GitHub installation, use `uv tool install --force --python 3.13 "tinbox[all]>=0.5.0"`.

## 0.4.0 (2026-10-26)

- Split long `.txt` inputs for `--translation-protocol translategemma` at paragraph and sentence breaks, or at `--split-token`, and send each chunk as its own request (#39). TranslateGemma now defaults to `context-aware` splitting instead of one request per file; each chunk is sent without neighbouring text because the model's fixed prompt has no place for it. Use `--algorithm page` for the previous behaviour. Review pair export is rejected for this protocol.
- Support Anthropic SDK 1.x by sending `temperature` in the request body; a fresh 0.3.0 install resolves to Anthropic 1.x, which removed that argument (#35). Test against OpenAI SDK 3.x (#34).
- Remove retired model names from model parsing and validation (#37, #38).
- Add `just` recipes for checks and tests, and fail CI when a Python file in `src/` or `tests/` exceeds 1,000 lines (#32).

### Upgrade

```bash
uv tool upgrade tinbox
tinbox --version
```

When changing extras or Python versions, or replacing a GitHub installation, use `uv tool install --force --python 3.13 "tinbox[all]>=0.4.0"`.

## 0.3.0 (2026-10-22)

- Preserve source and translation pairs with `translate --save-review-input` for fresh, context-aware `.txt` translations. Generate offline side-by-side HTML reports with `tinbox review`, including right-to-left text support.
- Add `prepare-matches` for unchecked sentence-group drafts. Curated matching files preserve every character and keep ambiguous regions unchecked; Tinbox does not automatically align sentences.
- Add opt-in `annotate-review` checks through TypeSafe for participant roles, certainty, and quantifiers. Only matched groups are sent. Save responses for offline reports, skip unchecked groups, and retain partial results after a request failure.
- Keep all review features optional and review-only. TypeSafe scores can miss errors and do not certify translation quality or change the translation. See the [translation review guide](docs/translation-review.md) and [German and Arabic pilot](benchmarks/translation_review_pilot/README.md).
- Rewrite the README and add provider setup, translation workflows, CLI reference, and contributing guides with current model examples.
- Use supported reasoning settings and request parameters for GPT-6 and current Claude models. Reduce the default chat output budget to 16,384 tokens, including for Gemini and other compatible endpoints.
- Require Anthropic SDK 0.88.0 or newer for current Claude request parameters.
- Refresh locked runtime and development dependencies, including security updates, and add GitHub Actions trusted publishing to PyPI.

### Upgrade

For an existing PyPI tool installation with the desired provider extras and Python 3.12 or newer:

```bash
uv tool upgrade tinbox
tinbox --version
```

When changing extras or Python versions, or replacing a GitHub installation, use `uv tool install --force --python 3.13 "tinbox[all]>=0.3.0"`. TypeSafe annotations need `TYPESAFE_API_KEY` in the environment, with no additional SDK or extra. Offline review reports need no API key.

## 0.2.0 (2026-10-22)

- Fix structured translation requests to use the OpenAI SDK's `chat.completions.parse()` method, with regression coverage through the actual SDK (#14).
- Add explicit oMLX/TranslateGemma support through `--translation-protocol translategemma`, using the model's translation prompt and `/v1/completions` plain-text responses. Correct the duplicate start token found during live testing (#15).
- Add `--local` for zero API charges in estimates and recorded usage, and clarify approximate context-overhead notices (#15).
- Fix sliding-window translation coverage, including zero-overlap windows (#5), and treat PDF dependencies as optional in `tinbox doctor` (#16).
- Replace LiteLLM with direct, optional OpenAI and Anthropic SDKs. Ollama, Gemini, and custom OpenAI-compatible endpoints use the `openai` extra (#11).
- Standardize development and CI on locked uv dependencies, Ruff, and ty. CI tests Python 3.12 and 3.13 with Poppler installed, and Dependabot checks Python and GitHub Actions dependencies weekly (#17).

### Upgrade

Python 3.12 or newer is required. For an existing PyPI tool installation with the desired extras and a supported interpreter:

```bash
uv tool upgrade tinbox
```

When upgrading from 0.1.0, switching from a GitHub installation, or changing Python versions, reinstall with explicit provider extras:

```bash
uv tool install --force --python 3.13 "tinbox[all]>=0.2.0"
tinbox --version
```

For OpenAI, Ollama, Gemini, or oMLX alone, use `tinbox[openai]` instead of `tinbox[all]`. Provider SDKs are no longer installed by the base package.

### TranslateGemma scope

The initial protocol supports short `.txt` inputs with an explicit source language and the automatically selected `page` algorithm. Context-aware/sliding-window algorithms, glossary, reasoning options, and image/PDF inputs are not supported. Each file is one request; keep inputs within the model's documented 2K-token input context, which Tinbox does not enforce. Output is capped at 2,048 tokens.

Live CLI validation passed with oMLX 0.7.2 and TranslateGemma 4B, including German ↔ English and text/JSON/Markdown exports. The 27B variant has not been tested live. See the [usage instructions](https://github.com/strickvl/tinbox#translategemma-with-omlx) and [validation notes](https://github.com/strickvl/tinbox/blob/main/plans/omlx-live-validation.md).
