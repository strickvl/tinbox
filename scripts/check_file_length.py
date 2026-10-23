"""Fail if any Python file under src/ or tests/ is longer than MAX_LINES."""

import sys
from pathlib import Path

MAX_LINES = 1000
ROOTS = ("src", "tests")


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    too_long = [
        (path.relative_to(repo_root), n)
        for root in ROOTS
        for path in sorted((repo_root / root).rglob("*.py"))
        if (n := count_lines(path)) > MAX_LINES
    ]
    for path, n in too_long:
        print(f"{path}: {n} lines (max {MAX_LINES}); split it into smaller modules")
    return 1 if too_long else 0


if __name__ == "__main__":
    sys.exit(main())
