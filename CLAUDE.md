# Notes for Claude

## Project Structure & Module Organization

- `src/tinbox/`: application code (CLI in `cli.py`, core logic in `core/`, shared helpers in `utils/`)
- `tests/`: pytest suite, organized by feature area (e.g., `tests/test_core/`, `tests/test_cli.py`)
- `examples/`: runnable sample inputs and scripts
- `plans/`: design notes and proposal drafts

## Environment Setup

- Use `uv pip install` instead of regular pip for package installation
- Editable install with dev tooling: `uv pip install -e ".[dev]"`
- Editable install with all extras (PDF/DOCX): `uv pip install -e ".[all]"`

## Common Commands

Use `tinbox --help` to see all available options. Examples:

- `tinbox translate --to de --model openai:gpt-5-2025-08-07 ./examples/elara_story.txt` — translate with a specific model
- `tinbox --to es document.pdf` — translate a PDF to Spanish
- `tinbox --from zh --to en document.docx` — translate a Word document from Chinese to English
- `pytest` — run the full test suite
- `pytest tests/test_core/test_cost.py -v` — run a focused test file
- `pytest --cov=tinbox --cov-report=html` — generate coverage report

## Coding Style & Naming Conventions

- Python 3.9+ with src-layout imports (`src/tinbox/...`)
- Format with Black (line length 88) and sort imports with isort (Black profile)
- Lint with Ruff; type-check with mypy (strict settings in `pyproject.toml`)
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`

## Testing Guidelines

- Framework: pytest with async support (`pytest-asyncio`)
- Naming: tests live under `tests/`, files named `test_*.py`, functions `test_*`
- Prefer unit tests in `tests/test_core/` and integration tests in `tests/test_cli.py`

## Commit & Pull Request Guidelines

- Commit messages are short, imperative, sentence case (e.g., "Update readme", "Adjust recommended reasoning level")
- PRs should include a concise description, test commands run, and any relevant sample inputs/outputs

## Security & Configuration Tips

- Use environment variables for API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`); never commit secrets
- PDF translation requires system `poppler` (`brew install poppler` on macOS)

## Project-Specific Information

- Pillow is required for image processing in the PDF processor and LiteLLM translator
