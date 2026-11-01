"""Tests that a checkpoint only resumes the same source text split the same way."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tinbox.core.translation.checkpoint import CheckpointManager, TranslationState
from tinbox.core.types import ModelType, TranslationConfig

SOURCE_TEXT = "First sentence. Second sentence. Third sentence."


def make_config(tmp_path: Path, **overrides: object) -> TranslationConfig:
    input_file = tmp_path / "book.txt"
    if not input_file.exists():
        input_file.write_text(SOURCE_TEXT)
    fields: dict[str, object] = {
        "source_lang": "en",
        "target_lang": "fr",
        "model": ModelType.OPENAI,
        "model_name": "gpt-4o",
        "algorithm": "context-aware",
        "input_file": input_file,
        "checkpoint_dir": tmp_path / "checkpoints",
        "resume_from_checkpoint": True,
    }
    return TranslationConfig.model_validate(fields | overrides)


async def save_checkpoint(config: TranslationConfig) -> Path:
    await CheckpointManager(config).save(
        TranslationState(
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            algorithm=config.algorithm,
            completed_pages=[1, 2],
            failed_pages=[],
            translated_chunks={1: "Première phrase.", 2: "Deuxième phrase."},
            token_usage=20,
            cost=0.002,
            time_taken=1.0,
        )
    )
    assert config.checkpoint_dir is not None
    return config.checkpoint_dir / "book_checkpoint.json"


def load_with_logger(config: TranslationConfig) -> tuple[CheckpointManager, MagicMock]:
    manager = CheckpointManager(config)
    manager._logger = MagicMock()
    return manager, manager._logger


async def test_unchanged_run_resumes_and_records_settings(tmp_path):
    config = make_config(tmp_path, context_size=100)
    checkpoint_path = await save_checkpoint(config)

    state = await CheckpointManager(config).load()

    assert state is not None
    assert state.translated_chunks == {1: "Première phrase.", 2: "Deuxième phrase."}
    saved = json.loads(checkpoint_path.read_text())["config"]
    assert saved["checkpoint_format_version"] == 2
    assert len(saved["source_sha256"]) == 64
    assert saved["input_path"] == str(config.input_file.resolve())
    assert saved["model_name"] == "gpt-4o"
    assert saved["context_size"] == 100
    assert saved["custom_split_token"] is None
    assert saved["window_size"] == config.window_size
    assert saved["overlap_size"] == config.overlap_size


async def test_edited_source_is_rejected(tmp_path):
    config = make_config(tmp_path)
    await save_checkpoint(config)
    config.input_file.write_text(SOURCE_TEXT + " A new fourth sentence.")

    manager, log = load_with_logger(config)

    assert await manager.load() is None
    mismatched = log.warning.call_args.kwargs["mismatched"]
    assert set(mismatched) == {"source_sha256"}


async def test_other_file_with_same_name_is_rejected(tmp_path):
    """sub/book.txt shares book.txt's checkpoint name but not its content."""
    config = make_config(tmp_path)
    await save_checkpoint(config)
    other_file = tmp_path / "sub" / "book.txt"
    other_file.parent.mkdir()
    other_file.write_text("A completely different document.")

    assert (
        await CheckpointManager(
            config.model_copy(update={"input_file": other_file})
        ).load()
        is None
    )


async def test_moved_file_with_same_content_resumes(tmp_path):
    config = make_config(tmp_path)
    await save_checkpoint(config)
    moved = tmp_path / "moved" / "book.txt"
    moved.parent.mkdir()
    moved.write_text(SOURCE_TEXT)

    assert (
        await CheckpointManager(config.model_copy(update={"input_file": moved})).load()
        is not None
    )


@pytest.mark.parametrize(
    ("algorithm", "field", "before", "after"),
    [
        ("context-aware", "context_size", 100, 300),
        ("context-aware", "custom_split_token", None, "|||"),
        ("sliding-window", "window_size", 2000, 500),
        ("sliding-window", "overlap_size", 200, 50),
        ("page", "model_name", "gpt-4o", "gpt-4o-mini"),
        ("context-aware", "model_name", "gpt-4o", "gpt-4o-mini"),
    ],
)
async def test_changed_setting_is_rejected_and_named(
    tmp_path, algorithm, field, before, after
):
    await save_checkpoint(make_config(tmp_path, algorithm=algorithm, **{field: before}))
    manager, log = load_with_logger(
        make_config(tmp_path, algorithm=algorithm, **{field: after})
    )

    assert await manager.load() is None
    mismatched = log.warning.call_args.kwargs["mismatched"]
    assert mismatched == {field: {"checkpoint": before, "current": after}}


@pytest.mark.parametrize(
    ("algorithm", "field", "before", "after"),
    [
        ("page", "context_size", 100, 300),
        ("page", "window_size", 2000, 500),
        ("context-aware", "window_size", 2000, 500),
        ("sliding-window", "context_size", 100, 300),
    ],
)
async def test_setting_the_algorithm_ignores_still_resumes(
    tmp_path, algorithm, field, before, after
):
    await save_checkpoint(make_config(tmp_path, algorithm=algorithm, **{field: before}))

    state = await CheckpointManager(
        make_config(tmp_path, algorithm=algorithm, **{field: after})
    ).load()

    assert state is not None


async def test_checkpoint_from_older_tinbox_is_rejected(tmp_path):
    config = make_config(tmp_path)
    checkpoint_path = await save_checkpoint(config)
    data = json.loads(checkpoint_path.read_text())
    data["config"] = {
        key: data["config"][key]
        for key in (
            "source_lang",
            "target_lang",
            "model",
            "algorithm",
            "translation_protocol",
            "local_model",
        )
    }
    checkpoint_path.write_text(json.dumps(data))

    manager, log = load_with_logger(config)

    assert await manager.load() is None
    assert "older tinbox version" in log.warning.call_args.args[0]


@pytest.mark.parametrize(("saved", "current"), [(None, 2000), (2000, None)])
async def test_omitted_context_size_matches_its_default(tmp_path, saved, current):
    """The algorithm splits with 2000 when context_size is None, so both match."""
    await save_checkpoint(make_config(tmp_path, context_size=saved))

    state = await CheckpointManager(make_config(tmp_path, context_size=current)).load()

    assert state is not None
