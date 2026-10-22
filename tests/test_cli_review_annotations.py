"""Opt-in network review and offline report workflows."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from tinbox.cli import app
from tinbox.core import review_annotations
from tinbox.core.review import ReviewBundle, ReviewPair, save_review_bundle
from tinbox.core.review_matches import (
    ChunkMatches,
    MatchGroup,
    ReviewMatches,
    TextSpan,
    bundle_digest,
    save_matches,
    unchecked_matches,
)


@pytest.fixture
def review_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(
        "tinbox.cli.create_translator",
        Mock(side_effect=AssertionError("Review must never translate")),
    )
    bundle = ReviewBundle(
        source_name="story.txt",
        source_sha256="ab" * 32,
        source_lang="en",
        target_lang="de",
        provider="openai",
        model_name="fixture",
        context_size=1000,
        pairs=[
            ReviewPair(
                unit_id=1,
                source="Private heading. She called. He listened.",
                translation="Private Überschrift. Sie rief. Er hörte zu.",
            )
        ],
    )
    source = bundle.pairs[0].source
    target = bundle.pairs[0].translation
    source_edges = [0, source.index("She"), source.index("He"), len(source)]
    target_edges = [0, target.index("Sie"), target.index("Er"), len(target)]
    matches = ReviewMatches(
        bundle_sha256=bundle_digest(bundle),
        provenance="assistant-curated",
        chunks=[
            ChunkMatches(
                unit_id=1,
                groups=[
                    MatchGroup(
                        status="unchecked" if i == 0 else "matched",
                        source=[
                            TextSpan(start=source_edges[i], end=source_edges[i + 1])
                        ],
                        translation=[
                            TextSpan(start=target_edges[i], end=target_edges[i + 1])
                        ],
                        reason="Heading is not a sentence" if i == 0 else "",
                    )
                    for i in range(3)
                ],
            )
        ],
    )
    bundle_path = tmp_path / "review.json"
    matches_path = tmp_path / "matches.json"
    save_review_bundle(bundle, bundle_path)
    save_matches(matches, matches_path, bundle)
    return bundle, matches, bundle_path, matches_path


def install_service(monkeypatch: pytest.MonkeyPatch, *, score: float = 0.8):
    requests: list[dict] = []

    def request(body: bytes, api_key: str, timeout: float) -> dict:
        payload = json.loads(body)
        assert api_key == "test-key"
        assert timeout == 30.0
        requests.append(payload)
        return {
            "model": payload["model"],
            "answers": {
                name: {"type": "noul", "noul": score if name == "participants" else 0.1}
                for name in ("participants", "certainty", "quantifiers")
            },
            "usage": {"input_tokens": 100, "output_tokens": 12},
            "extra_provider_metadata": {"retained": True},
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(review_annotations, "_send_request", request)
    return requests


def annotate_args(bundle_path: Path, matches_path: Path, output: Path) -> list[str]:
    return [
        "annotate-review",
        str(bundle_path),
        "--matches",
        str(matches_path),
        "--output",
        str(output),
    ]


def test_annotate_only_matched_groups_and_render_without_network(
    review_files, monkeypatch: pytest.MonkeyPatch
):
    bundle, matches, bundle_path, matches_path = review_files
    requests = install_service(monkeypatch)
    output = bundle_path.with_name("annotations.json")
    original = (bundle_path.read_bytes(), matches_path.read_bytes())
    result = CliRunner().invoke(app, annotate_args(bundle_path, matches_path, output))
    assert result.exit_code == 0, result.output
    assert "2 assessed, 1 unchecked skipped" in " ".join(result.output.split())
    assert [item["state"]["source_text"] for item in requests] == [
        "She called. ",
        "He listened.",
    ]
    assert all("Private" not in json.dumps(item) for item in requests)
    saved = review_annotations.load_annotations(output, bundle, matches)
    assert [row.status for row in saved.groups] == [
        "skipped_unchecked",
        "completed",
        "completed",
    ]
    assert saved.groups[1].response is not None
    assert saved.groups[1].response["extra_provider_metadata"] == {"retained": True}

    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.setattr(
        review_annotations,
        "_send_request",
        Mock(side_effect=AssertionError("Rendering must be offline")),
    )
    report_path = bundle_path.with_name("annotated.html")
    result = CliRunner().invoke(
        app,
        [
            "review",
            str(bundle_path),
            "--matches",
            str(matches_path),
            "--annotations",
            str(output),
            "-o",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "report generation made no model calls" in " ".join(result.output.split())
    assert "No automated quality assessment was performed" not in " ".join(
        result.output.split()
    )
    html = report_path.read_text()
    assert "TypeSafe concern" in html
    assert "Not assessed: unchecked match" in html
    assert "Private Überschrift." in html
    assert (bundle_path.read_bytes(), matches_path.read_bytes()) == original


def test_all_unchecked_needs_no_key_and_makes_no_requests(
    review_files, monkeypatch: pytest.MonkeyPatch
):
    bundle, _, bundle_path, _ = review_files
    matches_path = bundle_path.with_name("unchecked.json")
    save_matches(unchecked_matches(bundle), matches_path, bundle)
    output = bundle_path.with_name("annotations.json")
    network = Mock(side_effect=AssertionError("Unchecked groups must never be sent"))
    monkeypatch.setattr(review_annotations, "_send_request", network)
    result = CliRunner().invoke(app, annotate_args(bundle_path, matches_path, output))
    assert result.exit_code == 0, result.output
    assert "0 assessed, 1 unchecked skipped" in " ".join(result.output.split())
    network.assert_not_called()


def test_missing_key_and_existing_destination_fail_before_network(
    review_files, monkeypatch: pytest.MonkeyPatch
):
    _, _, bundle_path, matches_path = review_files
    output = bundle_path.with_name("annotations.json")
    network = Mock(side_effect=AssertionError("Preflight must fail before network"))
    monkeypatch.setattr(review_annotations, "_send_request", network)
    result = CliRunner().invoke(app, annotate_args(bundle_path, matches_path, output))
    assert result.exit_code == 1
    assert "TYPESAFE_API_KEY" in " ".join(result.output.split())
    assert not output.exists()
    output.write_text("keep me")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    result = CliRunner().invoke(app, annotate_args(bundle_path, matches_path, output))
    assert result.exit_code == 1
    assert output.read_text() == "keep me"
    network.assert_not_called()


def test_annotations_require_the_matching_file(review_files):
    _, _, bundle_path, _ = review_files
    annotations = bundle_path.with_name("annotations.json")
    annotations.write_text("{}")
    output = bundle_path.with_name("report.html")
    result = CliRunner().invoke(
        app,
        [
            "review",
            str(bundle_path),
            "--annotations",
            str(annotations),
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "--annotations requires" in " ".join(result.output.split())
    assert not output.exists()


def test_changed_matching_rejects_annotations_before_report(
    review_files, monkeypatch: pytest.MonkeyPatch
):
    _, _, bundle_path, matches_path = review_files
    install_service(monkeypatch)
    annotations = bundle_path.with_name("annotations.json")
    result = CliRunner().invoke(
        app, annotate_args(bundle_path, matches_path, annotations)
    )
    assert result.exit_code == 0, result.output
    matching = json.loads(matches_path.read_text())
    matching["chunks"][0]["groups"][0]["reason"] = "A changed decision"
    matches_path.write_text(json.dumps(matching))
    output = bundle_path.with_name("report.html")
    result = CliRunner().invoke(
        app,
        [
            "review",
            str(bundle_path),
            "--matches",
            str(matches_path),
            "--annotations",
            str(annotations),
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert not output.exists()


def test_partial_failure_saves_evidence_and_returns_nonzero(
    review_files, monkeypatch: pytest.MonkeyPatch
):
    bundle, matches, bundle_path, matches_path = review_files
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    network = Mock(side_effect=TimeoutError("test-key must not escape"))
    monkeypatch.setattr(review_annotations, "_send_request", network)
    output = bundle_path.with_name("annotations.json")
    result = CliRunner().invoke(app, annotate_args(bundle_path, matches_path, output))
    assert result.exit_code == 1, result.output
    assert "Annotations saved" in " ".join(result.output.split())
    assert "1 failed, 1 not attempted" in " ".join(result.output.split())
    assert "test-key" not in " ".join(result.output.split()) + output.read_text()
    saved = review_annotations.load_annotations(output, bundle, matches)
    assert [row.status for row in saved.groups] == [
        "skipped_unchecked",
        "error",
        "not_attempted",
    ]
    network.assert_called_once()


def test_group_limit_fails_before_any_request(
    review_files, monkeypatch: pytest.MonkeyPatch
):
    _, _, bundle_path, matches_path = review_files
    requests = install_service(monkeypatch)
    output = bundle_path.with_name("annotations.json")
    result = CliRunner().invoke(
        app, [*annotate_args(bundle_path, matches_path, output), "--max-groups", "1"]
    )
    assert result.exit_code == 1
    assert not output.exists()
    assert requests == []
