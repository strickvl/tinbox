"""Offline matching draft and report workflows."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from tinbox.cli import app
from tinbox.core.review import ReviewBundle, ReviewPair, save_review_bundle
from tinbox.core.review_matches import (
    ChunkMatches,
    MatchGroup,
    TextSpan,
    load_matches,
    save_matches,
    unchecked_matches,
)


@pytest.fixture
def saved_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "tinbox.cli.create_translator",
        Mock(side_effect=AssertionError("Offline matching must not call a provider")),
    )
    bundle = ReviewBundle(
        source_name="example.txt",
        source_sha256="ab" * 32,
        source_lang="en",
        target_lang="ar",
        provider="openai",
        model_name="fixture",
        context_size=1000,
        pairs=[
            ReviewPair(
                unit_id=1,
                source="Elara waited. Lume listened.",
                translation="انتظرت إلارا. أصغى لومي.",
            ),
            ReviewPair(unit_id=2, source="She called her.", translation="نادت عليها."),
        ],
    )
    path = tmp_path / "review.json"
    save_review_bundle(bundle, path)
    return bundle, path


def test_prepare_matches_is_offline_and_starts_unchecked(saved_bundle):
    bundle, path = saved_bundle
    draft_path = path.with_name("matches.json")
    result = CliRunner().invoke(
        app, ["prepare-matches", str(path), "--output", str(draft_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Every chunk is unchecked" in result.output
    draft = load_matches(draft_path, bundle)
    assert draft.provenance == "unreviewed"
    assert all(
        group.status == "unchecked" for chunk in draft.chunks for group in chunk.groups
    )
    report = path.with_name("unchecked.html")
    result = CliRunner().invoke(
        app, ["review", str(path), "--matches", str(draft_path), "-o", str(report)]
    )
    assert result.exit_code == 0, result.output
    assert "Unchecked" in report.read_text(encoding="utf-8")


def test_curated_matches_and_ambiguous_regions_render_offline(saved_bundle):
    bundle, path = saved_bundle
    draft = unchecked_matches(bundle)
    source = bundle.pairs[0].source
    translation = bundle.pairs[0].translation
    source_boundary = source.index("Lume")
    translation_boundary = translation.index("أصغى")
    first_chunk = ChunkMatches(
        unit_id=1,
        groups=[
            MatchGroup(
                status="matched",
                source=[TextSpan(start=0, end=source_boundary)],
                translation=[TextSpan(start=0, end=translation_boundary)],
            ),
            MatchGroup(
                status="matched",
                source=[TextSpan(start=source_boundary, end=len(source))],
                translation=[
                    TextSpan(start=translation_boundary, end=len(translation))
                ],
            ),
        ],
    )
    matches = draft.model_copy(
        update={
            "provenance": "assistant-curated",
            "chunks": [first_chunk, draft.chunks[1]],
        }
    )
    matches_path = path.with_name("curated.json")
    save_matches(matches, matches_path, bundle)
    original = path.read_bytes()
    report = path.with_name("curated.html")
    result = CliRunner().invoke(
        app, ["review", str(path), "--matches", str(matches_path), "-o", str(report)]
    )
    assert result.exit_code == 0, result.output
    html = report.read_text(encoding="utf-8")
    assert "Matched for comparison" in html
    assert "Unchecked" in html
    assert "انتظرت إلارا." in html
    assert "She called her." in html
    assert path.read_bytes() == original


def test_stale_matches_do_not_create_a_report(saved_bundle):
    bundle, path = saved_bundle
    matches_path = path.with_name("matches.json")
    save_matches(unchecked_matches(bundle), matches_path, bundle)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pairs"][0]["translation"] = "A changed translation."
    path.write_text(json.dumps(data), encoding="utf-8")
    report = path.with_name("report.html")
    result = CliRunner().invoke(
        app, ["review", str(path), "--matches", str(matches_path), "-o", str(report)]
    )
    assert result.exit_code == 1
    assert "Could not create review report" in result.output
    assert not report.exists()


@pytest.mark.parametrize("command", ["prepare-matches", "review"])
def test_matching_workflow_never_overwrites_an_input(saved_bundle, command):
    bundle, path = saved_bundle
    matches_path = path.with_name("matches.json")
    save_matches(unchecked_matches(bundle), matches_path, bundle)
    original_bundle = path.read_bytes()
    original_matches = matches_path.read_bytes()
    args = [command, str(path), "-o", str(matches_path)]
    if command == "review":
        args += ["--matches", str(matches_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert path.read_bytes() == original_bundle
    assert matches_path.read_bytes() == original_matches


def test_invalid_matching_json_does_not_create_report(saved_bundle):
    _, path = saved_bundle
    matches_path = path.with_name("matches.json")
    matches_path.write_text("{", encoding="utf-8")
    report = path.with_name("report.html")
    result = CliRunner().invoke(
        app, ["review", str(path), "--matches", str(matches_path), "-o", str(report)]
    )
    assert result.exit_code == 1
    assert not report.exists()
