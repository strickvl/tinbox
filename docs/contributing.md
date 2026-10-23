# Contributing

[README](../README.md) · [CLI reference](cli.md)

## Set up a checkout

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and install the locked dependencies:

```bash
git clone https://github.com/strickvl/tinbox.git
cd tinbox
uv sync --locked --all-extras
uv run --locked --all-extras tinbox --help
```

The checkout uses Python 3.12 from `.python-version`. `uv sync` installs all provider and document extras plus development tools into `.venv`. Install [Poppler](providers.md#pdf-requirements) and confirm both `pdfinfo` and `pdftoppm` are available before running PDF tests.

## Run checks

Install [just](https://just.systems/man/en/packages.html) (for example `brew install just`), then run the checks through the recipes in `justfile`:

```bash
just check
just test
```

`just check` runs Ruff linting, the Ruff format check, ty, and a file-length check that fails if any Python file in `src/` or `tests/` is over 1000 lines. Run `just` to list every recipe, and `just format` to format files in place.

CI calls the same recipes. It runs Ruff, ty, and the file-length check, then tests on Python 3.12 and 3.13 with Poppler installed. Type-checker warnings fail the check. Provider tests mock SDK clients and need no live API credentials.

For a focused test run or coverage report:

```bash
just test tests/test_cli.py
just test tests/test_core/test_cost.py -v
just test --cov-report=html
```

## Find the code

| Location | Contents |
| --- | --- |
| `src/tinbox/cli.py` | Commands, options, and translation flow |
| `src/tinbox/core/processor/` | Text, PDF, and DOCX input readers |
| `src/tinbox/core/translation/` | Provider adapters, algorithms, checkpoints, and glossaries |
| `src/tinbox/core/cost.py` and `output.py` | Cost estimates and output formats |
| `tests/` | CLI and core behavior tests |
| `examples/` | Sample text, PDF, and processing script |

Use `uv add` for runtime dependencies or `uv add --dev` for development tools, and include the resulting `uv.lock` changes. Preserve optional provider imports and installation hints. Follow the [repository guidelines](../AGENTS.md) for style and tests.

Keep pull requests focused. Describe the user-visible change and the checks you ran. For documentation changes, check examples against `tinbox translate --help` and the implementation; a successful dry run verifies local validation and estimation, not provider compatibility or translation quality.
