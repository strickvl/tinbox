# Repository Guidelines

## Project Structure & Module Organization
- `src/tinbox/`: application code (CLI in `cli.py`, core logic in `core/`, shared helpers in `utils/`).
- `tests/`: pytest suite, organized by feature area (e.g., `tests/test_core/`, `tests/test_cli.py`).
- `examples/`: runnable sample inputs and scripts.
- `plans/`: design notes and proposal drafts.
- `scripts/`: repository maintenance checks run in CI.

## Build, Test, and Development Commands
- `uv sync --locked --all-extras`: install the project, all provider/document extras, and development tools from `uv.lock` into `.venv`.
- `uv run --locked --all-extras tinbox --help`: inspect the CLI.
- Checks run through [just](https://just.systems) recipes in `justfile`; each recipe wraps the locked `uv run --locked --all-extras` command. CI calls the same recipes.
- `just lint`: lint code and check import sorting with Ruff.
- `just format-check`: check formatting; `just format` formats files in place.
- `just types`: type-check application code and tests with ty; warnings fail the check.
- `just file-length`: fail if any Python file in `src/` or `tests/` is over 1000 lines; split the file rather than raising the limit.
- `just check`: run lint, format-check, types, and file-length together.
- `just test`: run the full test suite; extra arguments go to pytest, e.g. `just test tests/test_core/test_cost.py -v` or `just test --cov-report=html`.
- Use `uv add` for runtime dependencies and `uv add --dev` for development tools. Include corresponding `uv.lock` changes whenever dependencies change.
- Dependabot checks Python and GitHub Actions dependencies weekly with a seven-day cooldown for routine updates; security updates bypass the cooldown. Minor/patch updates are grouped, while major updates remain separate.

## Coding Style & Naming Conventions
- Python 3.12+ with src-layout imports (`src/tinbox/...`).
- Use Ruff for formatting (line length 88), linting, and import sorting; use ty for type checking. Keep Python files at or under 1000 lines. Configuration lives in `pyproject.toml`.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.

## Testing Guidelines
- Framework: pytest with async support (`pytest-asyncio`).
- Naming: tests live under `tests/`, files named `test_*.py`, functions `test_*`.
- Prefer unit tests in `tests/test_core/` and integration tests in `tests/test_cli.py`.
- Install Poppler before running the PDF tests: `brew install poppler` on macOS or `sudo apt-get install poppler-utils` on Ubuntu/Debian. Confirm both `pdfinfo` and `pdftoppm` are available so PDF tests run.
- Provider tests mock SDK clients and need no live API credentials.
- CI runs Ruff, ty, and the file-length check once and pytest on Python 3.12 and 3.13, using the same `just` recipes above.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, sentence case (e.g., "Update readme", "Adjust recommended reasoning level").
- PRs should include a concise description, test commands run, and any relevant sample inputs/outputs.
- Include screenshots only when changing user-facing CLI output or docs formatting.

## Security & Configuration Tips
- Use environment variables for API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`); never commit secrets.

## Provider and Document Dependencies
- Translation uses the OpenAI and Anthropic SDKs directly. Ollama and Gemini use OpenAI-compatible endpoints and require the `openai` extra.
- Provider SDKs and `pdf2image` are optional at runtime; preserve lazy imports and helpful installation errors when those extras are absent.
- Pillow provides image processing for PDF translation.
