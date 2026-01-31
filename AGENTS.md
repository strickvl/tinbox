# Repository Guidelines

## Project Structure & Module Organization
Tinbox is a Python CLI. Core code lives in `src/tinbox/` with the entry point in
`src/tinbox/cli.py`. Translation logic is in `src/tinbox/core/` (processors,
translation algorithms, and cost tracking), and shared helpers are in
`src/tinbox/utils/`. Tests are under `tests/` with unit suites in
`tests/test_core/`, `tests/test_utils/`, and fixtures in `tests/data/`. Example
usage lives in `examples/`.

## Build, Test, and Development Commands
- Editable install (preferred): `uv pip install -e .`
- Dev extras (linting/testing tools): `uv pip install -e .[dev]`
- Optional extras: `uv pip install -e .[pdf]` or `.[docx]` (or `.[all]`)
- Run the CLI: `tinbox --help` or `tinbox --to es document.pdf`
- Run tests: `pytest`
- Run a single test file: `pytest tests/test_cli.py`

## Coding Style & Naming Conventions
- Python 3.9+ with type hints; untyped defs are disallowed by `mypy`.
- Indentation: 4 spaces; max line length: 88 (Black/Ruff/Isort aligned).
- Lint/format tools: `ruff check src tests`, `black src tests`, `isort src tests`.
- Naming: modules `snake_case.py`, classes `CapWords`, functions `snake_case`,
  tests and files start with `test_`.

## Testing Guidelines
Pytest is the test runner with `pytest-asyncio` enabled. Coverage is configured
with `--cov=tinbox` via `pyproject.toml`. Prefer focused unit tests and name test
functions `test_<behavior>()`.

## Commit & Pull Request Guidelines
Commit history favors short, imperative messages (e.g., `docs: ...`, `Fix: ...`,
`Add ...`). Keep commits scoped to a single change when possible. For PRs,
include a concise summary, how you tested (`pytest`, specific files), and link
relevant issues. If behavior changes, add example CLI commands/output in the PR
description.

## Agent-Specific Notes
Use `uv pip install` instead of `pip`. Pillow is required for PDF processing and
the LiteLLM translator; ensure it is installed when working on PDF features.
