# CLI reference

[README](../README.md) · [Providers](providers.md) · [Workflows](workflows.md)

## Commands

```text
tinbox [GLOBAL_OPTIONS] COMMAND
tinbox translate [OPTIONS] INPUT_FILE
tinbox doctor [--json]
```

`INPUT_FILE` must be an existing `.txt`, `.pdf`, or `.docx` file. Run `tinbox --help`, `tinbox translate --help`, or `tinbox doctor --help` for the installed version's help.

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

## Languages

Tinbox accepts these codes, subject to the selected model's language capabilities:

```text
ar  bn  cs  da  de  el  en  es  fi  fr  he  hi  id  it
ja  ko  nl  pl  pt  ru  sv  th  tr  uk  vi  zh  zh-tw
```

Common aliases such as `english`, `german`, `french`, and `chinese-traditional` are accepted. Omit `--from` or use `--from auto` for source-language detection with the chat protocol. Explicit source and target languages must differ.
