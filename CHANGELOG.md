# Changelog

## Unreleased

- Rewrite the README and add provider setup, translation workflows, CLI reference, and contributing guides with current model examples.
- Use supported reasoning settings and request parameters for GPT-6 and current Claude models. Reduce the default chat output budget to 16,384 tokens, including for Gemini and other compatible endpoints.
- Require Anthropic SDK 0.88.0 or newer for current Claude request parameters.

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
