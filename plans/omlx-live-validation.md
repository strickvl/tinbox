# Live oMLX validation

Validated on 22 October 2026 for PR #15. All three real CLI translation checks passed. This was a protocol smoke test with short inputs, not a translation-quality evaluation or a test of the reporter's 27B model.

## Environment

- Mac with Apple M5 Pro, 24 GB memory, macOS 26.6.1.
- Python 3.12.13, OpenAI SDK 1.109.1, MLX 0.32.4.
- [oMLX v0.7.2](https://github.com/jundot/omlx/releases/tag/v0.7.2), commit `23b671f64f0a64f532e38d692a6c480ea704418d`.
- [mlx-community/translategemma-4b-it-4bit](https://huggingface.co/mlx-community/translategemma-4b-it-4bit), revision `5788ec08c047f3f2e17808101b8d9566ac930d58`.
- Loopback server on port 18973, one concurrent request, a 4 GB process memory cap, SSD cache disabled, headless mode, and Hugging Face cache discovery disabled.

The server, dependencies, model, configuration, and caches were placed under a temporary directory. Two oMLX startup bookkeeping functions were redirected to temporary paths because they otherwise ignore `--base-path`: cluster-launch cleanup and the cluster Python shim. Inference and tokenizer code were unchanged. No application, persistent service, or user oMLX configuration was installed.

After validation, the server was stopped and the downloaded model, temporary server environment, and caches were removed. The small input/output evidence files were retained separately.

## CLI results

Each invocation used the real CLI, OpenAI SDK, HTTP server, and model. No HTTP transport or generated response was mocked.

| Direction and format | Input | Output |
| --- | --- | --- |
| German → English, text | `Guten Morgen! Die rote Katze schläft auf dem Sofa.` | `Good morning! The red cat is sleeping on the sofa.` |
| English → German, JSON | `The blue bicycle is beside the library.` | `Das blaue Fahrrad steht neben der Bibliothek.` |
| German → English, Markdown | `Hallo, Welt! Heute scheint die Sonne.` | `Hello, world! Today the sun is shining.` |

The text case also preserved two leading spaces and the final newline. The JSON case recorded 92 tokens and cost `0.0`, with no failed pages or errors. The Markdown export recorded 94 tokens and `$0.0000` API cost. The CLI displayed `API Charges: None (local model)`.

The command pattern was:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:18973/v1
export OPENAI_API_KEY=your-local-server-key

tinbox translate input.txt --from de --to en \
  --model openai:mlx-community--translategemma-4b-it-4bit \
  --translation-protocol translategemma --local \
  --format text --output translated.txt
```

A control request to `/v1/chat/completions` using a generic system/user prompt returned HTTP 400: `Chat template error: Conversations must start with a user prompt.` The explicit TranslateGemma route instead used `/v1/completions` and succeeded.

## Start-token correction

The initial implementation included the official template's literal `<bos>`. The real oMLX tokenizer adds its own start token when it encodes a completion prompt, so that request began with token IDs `[2, 2, 105, 2364, 107]`.

Omitting the literal `<bos>` produces `[2, 105, 2364, 107, 3048]`, matching the official template's token sequence exactly. The renderer and real-SDK request fixture were corrected before the live generation checks. The regression failed before the correction and passed afterward. An independent check against the downloaded tokenizer confirmed exact token-sequence parity for all 729 pairs of Tinbox's 27 supported languages.

All 330 automated tests, Ruff lint, and formatting checks also passed after the correction. The 27B quantization, long documents, context-aware translation, glossary, and image inputs were not validated by this run; the initial protocol mode continues to reject unsupported features.
