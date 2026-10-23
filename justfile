uv := "uv run --locked --all-extras"

# List available recipes
default:
    @just --list

# Lint code and check import sorting
lint:
    {{uv}} ruff check src tests scripts

# Check formatting without changing files
format-check:
    {{uv}} ruff format --check src tests scripts

# Format code in place
format:
    {{uv}} ruff format src tests scripts

# Type-check application code and tests
types:
    {{uv}} ty check

# Fail if any Python file in src/ or tests/ is over 1000 lines
file-length:
    {{uv}} python scripts/check_file_length.py

# Run the test suite; extra arguments go to pytest
test *args:
    {{uv}} pytest {{args}}

# Run every check CI runs except the tests
check: lint format-check types file-length
