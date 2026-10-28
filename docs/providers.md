# Set up providers and document support

[Back to README](../README.md) · [Translation workflows](workflows.md) · [CLI reference](cli.md)

## Install the extras you need

Tinbox requires Python 3.12 or later. The translation review workflow requires Tinbox 0.3.0 or newer. Current provider SDKs and TranslateGemma long-file splitting require 0.4.0 or newer. Markdown and other plain-text inputs, and translation-only stdout, require 0.5.0 or newer. Install all extras with:

```bash
uv tool install --python 3.13 "tinbox[all]>=0.5.0"
```

For a smaller installation, choose one command for your provider and input files:

```bash
uv tool install --python 3.13 "tinbox[openai]>=0.5.0"
uv tool install --python 3.13 "tinbox[anthropic]>=0.5.0"
uv tool install --python 3.13 "tinbox[openai,pdf]>=0.5.0"
```

The `openai` extra supports OpenAI, Gemini, Ollama, and custom OpenAI-compatible endpoints, including oMLX. The `anthropic` extra supports Claude. The `pdf` extra adds `pdf2image`; PDF processing also needs Poppler, described below. DOCX support is included in the base package, and a `docx` extra remains available. A base-only installation does not include either provider SDK.

To update an existing PyPI installation that already has the right extras and Python version, run:

```bash
uv tool upgrade tinbox
```

When changing extras or Python versions, upgrading from 0.1.0, or replacing a GitHub installation, reinstall explicitly:

```bash
uv tool install --force --python 3.13 "tinbox[all]>=0.5.0"
tinbox --version
```

Version 0.2.0 replaces LiteLLM with optional provider SDKs. A base-only upgrade does not install those SDKs. For a source checkout, see [Contributing](contributing.md).

## Configure a cloud provider

The model IDs below were checked against official documentation on 22 October 2026. These cloud models and the Ollama example have not been live-tested with Tinbox during this update.

| Provider | Current model example |
| --- | --- |
| OpenAI | [`gpt-6.1-sol`](https://developers.openai.com/api/docs/models/gpt-6.1-sol) |
| Anthropic | [`claude-sonnet-5-5`](https://platform.claude.com/docs/en/models/sonnet-5-5/overview) |
| Gemini | [`gemini-3.8-flash`](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) |
| Ollama | [`gemma4:e2b`](https://ollama.com/library/gemma4/tags) |

The commands below translate an existing `document.txt` into German. Replace that filename with your input path. Add `--dry-run` to estimate the work without calling a model. A successful dry run does not validate credentials, model support, or translation quality. Cost estimates use fixed provider rates, not each model's current price.

### OpenAI

Install `tinbox[openai]` and set `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY="your-openai-api-key"
tinbox translate document.txt --to de --model openai:gpt-6.1-sol
```

The OpenAI SDK uses its default endpoint unless `OPENAI_BASE_URL` is set. If you previously configured a custom endpoint, unset that variable before using OpenAI directly.

For `gpt-6.1-sol`, Tinbox maps its default `minimal` reasoning setting to `low`, omits sampling parameters such as `temperature`, and sends the output budget as `max_completion_tokens`. This follows [OpenAI's parameter requirements](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.1-sol). The CLI's `low`, `medium`, and `high` settings pass through unchanged.

### Anthropic

Install `tinbox[anthropic]` and set `ANTHROPIC_API_KEY`:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
tinbox translate document.txt --to de --model anthropic:claude-sonnet-5-5
```

Tinbox uses the Anthropic SDK directly. For `claude-sonnet-5-5`, it requests adaptive thinking without a temperature setting. The default `minimal` reasoning setting maps to `low` effort; `medium` and `high` map to the corresponding provider efforts.

### Gemini

Install `tinbox[openai]` and set `GOOGLE_API_KEY`:

```bash
export GOOGLE_API_KEY="your-google-api-key"
tinbox translate document.txt --to de --model gemini:gemini-3.8-flash
```

Tinbox sends Gemini requests through the OpenAI SDK to `https://generativelanguage.googleapis.com/v1beta/openai/`. The `gemini:` provider uses this fixed endpoint, independently of `OPENAI_BASE_URL`.

## Connect to Ollama

Install `tinbox[openai]`. With your Ollama server running, download Gemma 4 E2B and translate an existing `document.txt`:

```bash
ollama pull gemma4:e2b
tinbox translate document.txt --to de --model ollama:gemma4:e2b
```

Tinbox connects to `http://localhost:11434/v1` with a placeholder key and needs no cloud API key. The `ollama:` provider uses this fixed endpoint and records zero API cost automatically. To use another address, configure an `openai:` endpoint as described next.

## Connect to an OpenAI-compatible endpoint

Install `tinbox[openai]`. Set `OPENAI_BASE_URL` to the server's API base URL, including its version path, and `OPENAI_API_KEY` to its key. Use a nonempty placeholder key only if the server does not require authentication.

For example, to use OpenRouter:

```bash
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="your-openrouter-api-key"
tinbox translate document.txt --to de --model openai:MODEL_ID
```

Replace `MODEL_ID` with the exact model ID accepted by the endpoint. The default `chat` protocol expects chat completions with Tinbox's JSON translation response. An OpenAI-compatible URL alone does not guarantee that a model supports this protocol or Tinbox's request parameters.

For a local model with no API charges, add `--local`:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local"
tinbox translate document.txt --to de --model openai:MODEL_ID --local
```

`--local` sets estimated and recorded API cost to zero. It does not start a server, redirect requests, or verify where inference runs. Do not use it for a localhost proxy that forwards requests to a paid provider. Without `--local`, custom `openai:` endpoints use Tinbox's approximate OpenAI provider rates, which may differ from the endpoint's actual prices. Token and time estimates are approximate too.

## TranslateGemma with oMLX

Start an oMLX server with a TranslateGemma model and install `tinbox[openai]`. This example uses the 4B model from the repository's [recorded live validation](../plans/omlx-live-validation.md). Replace the base URL, key, and model ID with your server's values. The exact model ID may differ from the example.

Create a short German `.txt` input and select the translation-specific protocol:

```bash
printf 'Guten Morgen! Die rote Katze schläft auf dem Sofa.\n' > german.txt
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local"

tinbox translate german.txt --from de --to en \
  --model openai:mlx-community--translategemma-4b-it-4bit \
  --translation-protocol translategemma --local \
  --format text --output translated.txt
```

This protocol renders TranslateGemma's translation prompt and sends it to `/v1/completions`. It reads plain text rather than a JSON translation response. Text, JSON, and Markdown exports remain available.

Keep these constraints in mind before preparing input:

- Supply an explicit source language with `--from`; auto-detection is unsupported.
- Use `.txt` input. Tinbox selects the `context-aware` algorithm by default and splits the file into chunks of about `--context-size` characters (default 2,000) at paragraph and sentence breaks, or at `--split-token`. Each chunk becomes one request. Unlike the chat protocol, Tinbox sends no previous or next chunk with it, because TranslateGemma's fixed prompt has no place for that text. Use `--algorithm page` to send the whole file as one request.
- Keep each chunk within TranslateGemma's documented 2K-token input context. Tinbox counts characters, not tokens, and does not enforce that limit. Lower `--context-size`, for example to 1000, for Chinese, Japanese, or Korean text, where one character can take one token or more.
- Output is capped at 2,048 tokens. A response that ends because it hits the output limit is reported as a failure.
- The sliding-window algorithm, glossary use, non-minimal reasoning effort, review pair export, and image or PDF inputs are rejected.

The linked validation notes record short translations with oMLX 0.7.2 and `mlx-community/translategemma-4b-it-4bit`. They do not establish current server availability, long-document quality, or live support for the 27B variant.

## PDF requirements

Install the `pdf` extra alongside your provider, or use `tinbox[all]`. Install Poppler separately:

```bash
# macOS
brew install poppler

# Ubuntu or Debian
sudo apt-get install poppler-utils
```

On Windows, install a Poppler distribution and add its executable directory to `PATH`. Confirm that both `pdfinfo` and `pdftoppm` are available. Tinbox renders PDF pages as images, so choose a model and endpoint that accept image input. See [Translation workflows](workflows.md) for PDF commands.

## Troubleshooting

```bash
tinbox doctor
```

The doctor checks installed Python packages, provider SDKs, key presence, `pdfinfo`, and the Ollama executable. It does not authenticate keys, contact model servers, or check whether a model is installed. Missing optional dependencies produce installation hints without failing the command. PDF translation still requires Poppler and `pdf2image` even when the doctor exits successfully.

If translation fails, check the relevant setup:

- **Missing SDK:** reinstall with `openai`, `anthropic`, or `all` extras in the same environment as the `tinbox` command.
- **Authentication or model error:** check the key, endpoint, and exact model ID. For custom endpoints, confirm that `OPENAI_BASE_URL` includes the API version path.
- **Output-token limit error:** Tinbox's default chat output budget is 16,384 tokens. A model or endpoint with a lower limit may reject the request even for a short input. The CLI has no option to reduce that budget. Choose an endpoint that accepts it; TranslateGemma uses its separate 2,048-token cap.
- **Ollama connection error:** confirm that the server is running at `localhost:11434` and that the requested model is available.
- **PDF error:** confirm that `pdf2image`, `pdfinfo`, and `pdftoppm` are available, and that the model accepts images.
- **TranslateGemma template or JSON error:** use `--translation-protocol translategemma`, an `openai:` model, `OPENAI_BASE_URL`, and an explicit `--from`.
- **Incomplete TranslateGemma response:** lower `--context-size` and retry.

Use `tinbox translate --help` for option details or consult the [CLI reference](cli.md).
