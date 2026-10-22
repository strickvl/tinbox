# CLI reference

[README](../README.md) · [Providers](providers.md) · [Workflows](workflows.md) · [Translation review](translation-review.md)

## Commands

```text
tinbox [GLOBAL_OPTIONS] COMMAND
tinbox translate [OPTIONS] INPUT_FILE
tinbox prepare-matches REVIEW_INPUT --output MATCHES_JSON
tinbox annotate-review REVIEW_INPUT --matches MATCHES_JSON --output ANNOTATIONS_JSON [OPTIONS]
tinbox review REVIEW_INPUT --output REPORT_HTML [OPTIONS]
tinbox doctor [--json]
```

`INPUT_FILE` must be an existing `.txt`, `.pdf`, or `.docx` file. `REVIEW_INPUT` is a saved review bundle from `translate --save-review-input`. Run `tinbox --help` or add `--help` after any command for the installed version's help.

Global options go before the command:

| Option | Default | Meaning |
| --- | --- | --- |
| `--version`, `-v` | Off | Print the installed version and exit |
| `--log-level`, `-l` | `INFO` | Set the logging level, such as `DEBUG` or `ERROR` |
| `--json`, `-j` | Off | Format logs as JSON; does not change translation output |
| `--help` | | Show help and exit |

`tinbox doctor` reports dependency, API-key presence, and local tool checks. `tinbox doctor --json` changes the diagnostic report to JSON. It does not test authentication or run a translation. See [what doctor checks](providers.md#troubleshooting).

## Translation options

### Input and output

| Option | Default | Meaning |
| --- | --- | --- |
| `--model`, `-m` | Required | `PROVIDER:MODEL_ID`; providers are `openai`, `anthropic`, `gemini`, and `ollama` |
| `--from`, `-f` | Auto-detect | Source language code or accepted alias |
| `--to`, `-t` | `en` | Target language code or accepted alias |
| `--output`, `-o` | Terminal | Write the translation or report to this file |
| `--format`, `-F` | `text` | `text`, `json`, or `markdown`; selected independently of the output filename |
| `--save-review-input` | None | Write paired source and translation chunks to a new JSON file; requires a fresh context-aware `.txt` translation |
| `--verbose` | Off | Accepted but currently has no effect; use global `--log-level DEBUG` for diagnostic logs |
| `--help` | | Show translation help and exit |

Use `--output` to get a clean file. Terminal output also includes the translation plan and progress. JSON contains metadata, results, warnings, and errors. Markdown is a report with the translation in a text code block, not a recreation of the source document's formatting.

### Algorithms and models

| Option | Default | Meaning |
| --- | --- | --- |
| `--algorithm`, `-a` | Automatic | `page` for PDFs and TranslateGemma; `context-aware` for other inputs. `sliding-window` is also accepted for text |
| `--context-size` | `2000` | Target chunk size in characters for `context-aware` |
| `--split-token` | None | Split context-aware text on this marker instead of using the target chunk size |
| `--pdf-dpi` | `200` | Resolution used to convert PDF pages to images |
| `--translation-protocol` | `chat` | `chat` for the normal JSON response protocol, or `translategemma` for plain-text completions |
| `--reasoning-effort` | `minimal` | `minimal`, `low`, `medium`, or `high`; support depends on the provider and model |

PDFs only support `page`. The `page` algorithm sends one request per page; for a text file it sends the whole file. `context-aware` sends text chunks with surrounding context. `sliding-window` uses overlapping text windows; the CLI does not expose its window or overlap sizes. See [chunking examples](workflows.md#split-long-text).

The default `minimal` setting maps to the supported minimum for the current models:

| Model | Meaning of `minimal` |
| --- | --- |
| GPT-6.1 Sol and GPT-6 Astra | `low` reasoning effort |
| GPT-6 Sol and GPT-6 Luna | `none` reasoning effort |
| Claude Sonnet, Opus, and Haiku 5.5; Fable 5.1 | Adaptive thinking with `low` effort |
| Other chat models | No explicit reasoning or thinking parameter |

For those GPT-6 and Claude models, `low`, `medium`, and `high` map to the corresponding provider effort. Other OpenAI-compatible models receive `reasoning_effort`; other Anthropic models receive thinking budgets of 1,024, 4,096, and 8,192 tokens. Model support varies. Higher effort can increase time and cost. The default chat output budget is 16,384 tokens per request, including reasoning tokens where the provider counts them toward that limit.

TranslateGemma requires an `openai:` model, `OPENAI_BASE_URL`, a `.txt` file, an explicit source language, `page`, and the default reasoning setting. It rejects glossary use and `--split-token`; `--context-size` has no effect in this mode. See [TranslateGemma setup](providers.md#translategemma-with-omlx).

### Costs and confirmation

| Option | Default | Meaning |
| --- | --- | --- |
| `--dry-run` | Off | Show approximate tokens, cost, and time without contacting a model or writing a translation |
| `--max-cost` | None | Stop when recorded cost exceeds this USD threshold, checked after a request completes |
| `--local` | Off | Treat the endpoint as a local model with zero API charges; Ollama is already treated this way |
| `--force` | Off | Skip the estimate confirmation prompt; does not disable validation or the runtime cost threshold |

Costs use fixed provider-level rates. A request can exceed `--max-cost` before Tinbox stops, and estimates do not guarantee the provider's bill. `--local` changes accounting only; it does not start a server or select an endpoint. See [cost workflows](workflows.md#preview-and-limit-estimated-cost).

### Checkpoints and glossaries

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint-dir` | None | Enable saved progress in this directory and automatically attempt to resume |
| `--checkpoint-frequency` | `1` | Save progress every N pages or chunks |
| `--glossary` | Off | Enable glossary instructions and updates |
| `--glossary-file` | None | Load a glossary JSON file; requires `--glossary` |
| `--save-glossary` | None | Save the glossary after translation; requires `--glossary` |

See [resume instructions and limits](workflows.md#resume-an-interrupted-translation) and the [glossary format](workflows.md#reuse-a-glossary).

## Review commands

Review is opt-in and never changes translations. Pair export supports only fresh context-aware `.txt` translations; PDF, DOCX, other algorithms, and checkpoint resume are excluded. All review output paths must be new files with existing parent directories. See the [translation review guide](translation-review.md) for examples, matching-file curation, supported inputs, and recovery.

### Prepare matching drafts

`tinbox prepare-matches REVIEW_INPUT --output MATCHES_JSON` creates an offline draft with every chunk marked unchecked. `--output` (alias `-o`) is required. This command does not align sentences or call a model. A person or assistant must select matching spans, record the provenance, and leave ambiguous correspondence unchecked.

### Annotate matched groups

`tinbox annotate-review` sends only matched source and translation spans, their language codes, and three fixed questions to TypeSafe. It skips unchecked groups. This is a separate network step that requires `TYPESAFE_API_KEY` when matched groups are present; Tinbox does not automatically load `.env` files.

| Option | Default | Meaning |
| --- | --- | --- |
| `--matches` | Required | Curated matching JSON bound to the supplied review bundle |
| `--output`, `-o` | Required | New annotation JSON file, including partial results if a request fails |
| `--model` | `jev-1.14.0` | Pinned TypeSafe model name; names ending in `-latest` are rejected |
| `--timeout` | `30` | Positive finite timeout in seconds per request |
| `--max-groups` | `100` | Maximum matched groups; over-limit runs are rejected before any request |

The checks look for changes to participant roles, certainty, and quantifiers. They provide review hints, not quality certification. Requests run sequentially without retries or resume. A request failure stops the run, saves partial results, and exits with status `1`; a concern score alone does not cause a failing exit status. Rerunning with a new output path repeats requests. See [TypeSafe setup, limits, and outcomes](translation-review.md#add-optional-typesafe-annotations).

### Render reports

`tinbox review` generates a standalone HTML report entirely offline, including when displaying saved TypeSafe annotations. No API key or provider SDK is needed.

| Option | Default | Meaning |
| --- | --- | --- |
| `--output`, `-o` | Required | New HTML report file |
| `--matches` | None | Curated matching JSON for this exact review bundle |
| `--annotations` | None | Saved TypeSafe annotation JSON; requires the original `--matches` file |

Without `--matches`, the report displays the original chunk pairs. With matches, it displays smaller comparison groups, visible unchecked regions, and expandable original pairs. Annotations add stored outcomes without submitting new requests. Input fingerprints prevent rendering matches or annotations against changed inputs.

## Languages

Tinbox accepts these codes, subject to the selected model's language capabilities:

```text
ar  bn  cs  da  de  el  en  es  fi  fr  he  hi  id  it
ja  ko  nl  pl  pt  ru  sv  th  tr  uk  vi  zh  zh-tw
```

Common aliases such as `english`, `german`, `french`, and `chinese-traditional` are accepted. Omit `--from` or use `--from auto` for source-language detection with the chat protocol. Explicit source and target languages must differ.
