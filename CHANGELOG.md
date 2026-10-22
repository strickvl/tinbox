# Changelog

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
