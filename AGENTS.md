# Agent Guidelines for Tinbox

This repository contains the source code for **Tinbox**, a command line tool for translating documents using LLMs. The following guidelines apply to any changes across the project.

## Environment Setup
- Use `uv pip install` for installing packages. For editable installs run `uv pip install -e .[dev]`.
- The codebase targets Python 3.9+.

## Code Style
- Format Python files with **black** using the configuration in `pyproject.toml` (line length 88).
- Order imports with **isort** (profile `black`).
- Run **ruff** to lint the codebase.
- Perform static type checking with **mypy**. The project uses a strict configuration, so all functions should be typed.
- Include docstrings for public functions and classes.

## Testing
- Run `pytest` before committing. The tests live under `tests/` and make use of `pytest-asyncio`.

## CLI Usage
- Run `tinbox --help` to view available commands.
- Example usage: `tinbox --to es document.pdf` translates a PDF to Spanish.

## Additional Notes
- Some processors require `Pillow` for image handling.
