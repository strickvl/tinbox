"""Run a bounded real-CLI translation pilot without TypeSafe or alignment calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = Path(__file__).resolve().parent
ORIGIN = ROOT / "examples/elara_story.txt"
MODEL = "openai:gpt-4.1-mini-2025-04-14"
ENDPOINT = "https://api.openai.com/v1"
DEADLINE_SECONDS = 180


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def read_openai_key() -> tuple[str, str]:
    dotenv = ROOT / ".env"
    lines = dotenv.read_text(encoding="utf-8").splitlines() if dotenv.exists() else []
    for line in lines:
        match = re.fullmatch(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*?)\s*", line)
        if match:
            value = match.group(1)
            if value.startswith(("'", '"')) and value[-1:] == value[:1]:
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if value and not any(char.isspace() for char in value):
                return value, "repository .env OPENAI_API_KEY"
            raise ValueError("OPENAI_API_KEY in .env is empty or malformed")
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value, "existing process environment OPENAI_API_KEY"
    raise ValueError("OPENAI_API_KEY was not found in .env or the process environment")


def source_excerpt() -> bytes:
    original = ORIGIN.read_text(encoding="utf-8")
    excerpt = "\n\n".join(original.split("\n\n")[:12]) + "\n\n"
    if len(excerpt) != 2180 or not original.startswith(excerpt):
        raise ValueError("Elara example changed; review the pilot excerpt selection")
    return excerpt.encode("utf-8")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def command(run_dir: Path, language: str) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "--locked",
        "--all-extras",
        "tinbox",
        "translate",
        str(run_dir / "source.txt"),
        "--from",
        "en",
        "--to",
        language,
        "--model",
        MODEL,
        "--algorithm",
        "context-aware",
        "--context-size",
        "1000",
        "--force",
        "--max-cost",
        "1",
        "--output",
        str(run_dir / f"{language}.txt"),
        "--save-review-input",
        str(run_dir / f"{language}.review.json"),
    ]


def run_cli(
    arguments: list[str], environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    with subprocess.Popen(
        arguments,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=DEADLINE_SECONDS)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return subprocess.CompletedProcess(
            arguments, process.returncode, stdout, stderr
        )


def parse_statistics(log: str) -> dict[str, int | float | str]:
    tokens = re.search(r"Tokens Used\s*[│|]\s*([\d,]+)", log)
    cost = re.search(r"Final Cost\s*[│|]\s*\$([\d.]+)", log)
    duration = re.search(r"Time Taken\s*[│|]\s*([\d.]+) seconds", log)
    if not (tokens and cost and duration):
        raise ValueError("Completed CLI output did not contain translation statistics")
    return {
        "tokens_used": int(tokens.group(1).replace(",", "")),
        "tinbox_cost_usd_rounded": float(cost.group(1)),
        "tinbox_time_seconds_rounded": float(duration.group(1)),
        "cost_note": (
            "Tinbox estimate using approximate provider-wide rates, rounded to four "
            "decimal places by the CLI; not a model-specific invoice."
        ),
    }


def verify_outputs(run_dir: Path, language: str) -> dict[str, Any]:
    bundle_path = run_dir / f"{language}.review.json"
    translation_path = run_dir / f"{language}.txt"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    translation = translation_path.read_text(encoding="utf-8")
    source_hash = sha256((run_dir / "source.txt").read_bytes())
    if bundle["source_sha256"] != source_hash:
        raise ValueError("Bundle source fingerprint differs from saved input")
    if "".join(pair["translation"] for pair in bundle["pairs"]) != translation:
        raise ValueError(
            "Concatenated pair translations differ from translation output"
        )
    if [pair["unit_id"] for pair in bundle["pairs"]] != list(
        range(1, len(bundle["pairs"]) + 1)
    ):
        raise ValueError("Bundle chunk IDs are not consecutive")
    return {
        "chunk_count": len(bundle["pairs"]),
        "translation_sha256": sha256(translation_path.read_bytes()),
        "review_bundle_sha256": sha256(bundle_path.read_bytes()),
        "source_fingerprint_verified": True,
        "pair_translation_concatenation_verified": True,
    }


def run(run_name: str) -> Path:
    excerpt = source_excerpt()
    api_key, credential_source = read_openai_key()
    run_dir = PILOT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "source.txt").write_bytes(excerpt)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": now(),
        "source": {
            "origin": str(ORIGIN.relative_to(ROOT)),
            "origin_sha256": sha256(ORIGIN.read_bytes()),
            "selection": "First 12 double-newline-separated paragraphs, including heading",
            "character_range": [0, 2180],
            "byte_range": [0, len(excerpt)],
            "final_separator_included": True,
            "saved_sha256": sha256(excerpt),
        },
        "git": {
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes())
            for path in [
                Path(__file__).resolve(),
                *sorted((ROOT / "src").rglob("*.py")),
                ROOT / "uv.lock",
            ]
        },
        "python_version": platform.python_version(),
        "packages": {
            name: version(name) for name in ("tinbox", "openai", "pydantic", "tenacity")
        },
        "model": MODEL,
        "endpoint": ENDPOINT,
        "credential_source": credential_source,
        "context_size": 1000,
        "deadline_seconds_per_language": DEADLINE_SECONDS,
        "runner_retries": 0,
        "provider_retry_note": "Unmodified Tinbox and OpenAI SDK retry settings apply inside the subprocess deadline.",
        "typesafe_calls": 0,
        "alignment_calls": 0,
        "languages": {},
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith("API_KEY")
        and key not in {"OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}
    }
    environment.update(
        {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": ENDPOINT,
            "UV_CACHE_DIR": "/private/tmp/tinbox-typesafe-uv-cache",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "COLUMNS": "160",
        }
    )
    try:
        for language in ("de", "ar"):
            record: dict[str, Any] = {
                "status": "running",
                "started_at": now(),
                "command": command(run_dir, language),
            }
            manifest["languages"][language] = record
            write_manifest(manifest_path, manifest)
            start = time.monotonic()
            try:
                result = run_cli(record["command"], environment)
                log = (result.stdout + result.stderr).replace(api_key, "[REDACTED]")
                (run_dir / f"{language}.cli.log").write_text(log, encoding="utf-8")
                record.update(
                    {
                        "elapsed_seconds": time.monotonic() - start,
                        "completed_at": now(),
                        "returncode": result.returncode,
                    }
                )
                if result.returncode:
                    raise RuntimeError(
                        f"{language} translation CLI failed; inspect {language}.cli.log"
                    )
                record.update(verify_outputs(run_dir, language))
                record.update(parse_statistics(log))
                record["status"] = "completed"
                write_manifest(manifest_path, manifest)
                print(
                    f"{language}: {record['chunk_count']} chunks, {record['tokens_used']} tokens",
                    flush=True,
                )
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or b"") + (exc.stderr or b"")
                log = output.decode("utf-8", errors="replace").replace(
                    api_key, "[REDACTED]"
                )
                (run_dir / f"{language}.cli.log").write_text(log, encoding="utf-8")
                record.update(
                    {"status": "timed_out", "elapsed_seconds": time.monotonic() - start}
                )
                raise RuntimeError(
                    f"{language} translation exceeded {DEADLINE_SECONDS}s; no rerun attempted"
                ) from exc
            except Exception:
                record["status"] = "failed"
                raise
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc).replace(api_key, "[REDACTED]")
        raise
    finally:
        manifest["completed_at"] = now()
        write_manifest(manifest_path, manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        required=True,
        help="A fresh directory name; existing runs are never overwritten.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check excerpt selection only, without credentials, writes, or network calls.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.run_name):
        parser.error(
            "run-name must contain only letters, digits, underscores, or hyphens"
        )
    if args.check:
        excerpt = source_excerpt()
        print(
            f"Excerpt: {len(excerpt.decode('utf-8'))} characters, {len(excerpt)} bytes; sha256={sha256(excerpt)}"
        )
        return
    print(run(args.run_name))


if __name__ == "__main__":
    main()
