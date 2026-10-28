# Tinbox

**Translate documents from your terminal with cloud or local language models.**

[![PyPI](https://img.shields.io/pypi/v/tinbox)](https://pypi.org/project/tinbox/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/tinbox/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Tinbox reads PDFs, Word documents, and text files, then writes translations as text, JSON, or Markdown reports. It splits long text into chunks, can carry context and glossary terms between requests, and saves progress when you enable checkpoints.

[Quick start](#quick-start) · [Providers](docs/providers.md) · [Workflows](docs/workflows.md) · [CLI reference](docs/cli.md) · [Translation review](docs/translation-review.md) · [Contributing](docs/contributing.md)

## Quick start

This example translates a short English note into German with [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol). Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and have an OpenAI API key ready. Tinbox requires Python 3.12 or newer.

Install Tinbox 0.5.0 or newer with OpenAI support:

```bash
uv tool install --python 3.13 "tinbox[openai]>=0.5.0"
```

Set your key, create a sample file, and translate it. These examples use a Bash or Zsh shell and the default OpenAI endpoint; unset `OPENAI_BASE_URL` if you previously configured another server.

```bash
export OPENAI_API_KEY="your-openai-api-key"
printf 'The library opens at nine.\n' > note.txt

tinbox translate note.txt --from en --to de \
  --model openai:gpt-6.1-sol --output note.de.txt

cat note.de.txt
```

Tinbox shows a translation plan, progress, and statistics in the terminal on stderr. `note.de.txt` contains only the translated text, for example:

```text
Die Bibliothek öffnet um neun.
```

The wording can vary. Translation calls your provider and may incur API charges. To inspect the plan without contacting a model, add `--dry-run` to the translation command; it will not create an output file.

Prefer a local model? Use [Gemma 4 with Ollama](docs/providers.md#connect-to-ollama), or see [provider setup](docs/providers.md) for Claude, Gemini, and custom endpoints. The guide also covers installation from PyPI and upgrades.

## What can I translate?

| Input | How Tinbox reads it | What to expect |
| --- | --- | --- |
| `.txt`, `.md`, `.markdown`, `.rst`, `.org`, `.adoc`, `.asciidoc`, `.text` | Reads plain text and splits it into chunks by default | Context-aware translation for longer text; markup is passed to the model, which is asked to keep it |
| `.pdf` | Converts each page to an image | Requires Poppler, PDF dependencies, and a vision-capable model |
| `.docx` | Extracts paragraph text | Table contents are omitted; Word formatting and page layout are not preserved |

All inputs produce **text, JSON, or a Markdown report**. Tinbox does not export translated PDF or Word files. Translation quality and language coverage depend on the model; the CLI accepts a [defined set of language codes](docs/cli.md#languages).

## Common tasks

| I want to… | Guide |
| --- | --- |
| Translate a PDF | [Install PDF dependencies](docs/providers.md#pdf-requirements), then [translate the document](docs/workflows.md#translate-a-pdf) |
| Keep terminology consistent | [Load and save a glossary](docs/workflows.md#reuse-a-glossary) |
| Continue an interrupted job | [Save progress and resume](docs/workflows.md#resume-an-interrupted-translation) |
| Inspect source and translation side by side | [Opt-in translation review](docs/translation-review.md), with curated sentence matches and optional TypeSafe checks |
| Save metadata or inspect failed pages | [Choose an output format](docs/workflows.md#choose-an-output-format) |
| Preview costs or change chunk size | [Translation workflows](docs/workflows.md) |

### Review a text translation

For a fresh context-aware `.txt` translation, preserve the source and translation chunks and render an offline report:

```bash
tinbox translate document.txt --from en --to de --model openai:gpt-6.1-sol \
  --output document.de.txt --save-review-input document.de.review.json

tinbox review document.de.review.json --output document.de.review.html
```

The [translation review guide](docs/translation-review.md) explains how to curate sentence matches, leave ambiguous regions unchecked, and optionally send matched groups to TypeSafe for participant, certainty, and quantifier checks. Reports preserve the translated text; matching and TypeSafe hints do not certify its quality. Only the explicit `annotate-review` command sends text to TypeSafe. Review files contain document text and require new output paths.

### TranslateGemma with oMLX

For a local TranslateGemma server, use the dedicated [TranslateGemma setup](docs/providers.md#translategemma-with-omlx). It supports text files with an explicit source language and splits long files into chunks; it does not support PDFs or glossaries, and it sends no neighbouring context with each chunk.

## Limits worth knowing

- **Costs are approximate.** Tinbox uses fixed provider rates, not live model prices. `--max-cost` checks recorded cost after requests complete and is not a billing cap.
- **Model compatibility varies.** The [provider guide](docs/providers.md#configure-a-cloud-provider) lists current examples and request settings. A dry run checks local options and estimates, not provider access or translation quality.
- **Check results before relying on them.** A PDF translation can contain failed pages. JSON and Markdown reports include error details; see [output formats](docs/workflows.md#choose-an-output-format).

## Help and development

```bash
tinbox --help
tinbox translate --help
tinbox doctor
```

Read the [CLI reference](docs/cli.md), [setup troubleshooting](docs/providers.md#troubleshooting), or [contributing guide](docs/contributing.md). For release and upgrade notes, see the [changelog](CHANGELOG.md).

Licensed under the [MIT License](LICENSE).
