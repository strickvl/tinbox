"""Tests that a stopped translation keeps paid-for work and counts billed failures."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import translate_document
from tinbox.core.translation.checkpoint import CheckpointManager
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import ModelType, TranslationConfig

REJECTED = "Invalid finish reason from model: length, expected 'stop'"


def _config(algorithm: str, **overrides) -> TranslationConfig:
    settings = {
        "source_lang": "en",
        "target_lang": "fr",
        "model": ModelType.OPENAI,
        "model_name": "gpt-6.1-sol",
        "algorithm": algorithm,
        "input_file": Path("test.txt"),
        # Three windows or chunks: "Aaaa. ", "Bbbb. ", "Cccc."
        "window_size": 6,
        "overlap_size": 0,
        "context_size": 6,
    }
    return TranslationConfig.model_validate(settings | overrides)


def _content(algorithm: str) -> DocumentContent:
    pages = (
        ["Aaaa.", "Bbbb.", "Cccc."] if algorithm == "page" else ["Aaaa. Bbbb. Cccc."]
    )
    return DocumentContent(pages=pages, content_type="text/plain")


def _translator(fail_on: int | None = None, billed: float = 0.5) -> AsyncMock:
    """Each successful call costs 1.0; call `fail_on` is billed, then rejected."""
    calls = 0

    async def translate(request: TranslationRequest) -> TranslationResponse:
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise TranslationError(
                REJECTED, cost=billed, input_tokens=100, output_tokens=50
            )
        assert isinstance(request.content, str)
        return TranslationResponse(
            text=f"<{request.content.strip()}>",
            tokens_used=15,
            cost=1.0,
            time_taken=0.1,
            input_tokens=10,
            output_tokens=5,
        )

    translator = AsyncMock(spec=ModelInterface)
    translator.translate = AsyncMock(side_effect=translate)
    return translator


@pytest.mark.parametrize("algorithm", ["sliding-window", "context-aware"])
async def test_failed_chunk_keeps_earlier_chunks(algorithm):
    result = await translate_document(
        _content(algorithm), _config(algorithm), _translator(fail_on=2)
    )

    assert result.text == (
        "<Aaaa.>\n\n[TRANSLATION_FAILED: Chunks 2-3 of 3]\n"
        f"Reason: {REJECTED}\n[/TRANSLATION_FAILED]"
    )
    assert result.failed_chunks == [2, 3]
    assert result.warnings == [
        f"Translation incomplete: chunks 2-3 of 3 not translated. Reason: {REJECTED}"
    ]
    # The rejected reply was billed, so it counts toward cost and tokens
    assert result.cost == pytest.approx(1.5)
    assert result.tokens_used == 15 + 150
    assert (result.input_tokens, result.output_tokens) == (110, 55)


async def test_stop_saves_work_the_checkpoint_frequency_skipped():
    manager = AsyncMock(spec=CheckpointManager)
    manager.load.return_value = None
    config = _config(
        "context-aware", checkpoint_dir=Path("checkpoints"), checkpoint_frequency=5
    )

    await translate_document(
        _content("context-aware"),
        config,
        _translator(fail_on=3),
        checkpoint_manager=manager,
    )

    # No periodic save ran before chunk 3 failed, so the stop saves chunks 1-2
    manager.save.assert_awaited_once()
    state = manager.save.await_args.args[0]
    assert state.translated_chunks == {1: "<Aaaa.>", 2: "<Bbbb.>"}
    assert state.cost == pytest.approx(2.5)
    assert (state.input_tokens, state.output_tokens) == (120, 60)


@pytest.mark.parametrize("algorithm", ["sliding-window", "context-aware"])
async def test_zero_max_cost_stops_after_first_request(algorithm):
    translator = _translator()
    result = await translate_document(
        _content(algorithm), _config(algorithm, max_cost=0.0), translator
    )

    assert translator.translate.await_count == 1
    assert result.failed_chunks == [2, 3]
    assert result.text.startswith("<Aaaa.>\n\n[TRANSLATION_FAILED: Chunks 2-3 of 3]")
    assert "exceeded maximum cost of $0.00" in result.warnings[0]


async def test_cap_reached_on_last_chunk_returns_full_text_with_warning():
    result = await translate_document(
        _content("context-aware"), _config("context-aware", max_cost=2.5), _translator()
    )

    assert result.text == "<Aaaa.><Bbbb.><Cccc.>"
    assert result.failed_chunks == []
    assert result.warnings == [
        "Translation cost of $3.0000 exceeded maximum cost of $2.50"
    ]


async def test_stopped_run_returns_no_review_pairs():
    config = _config("context-aware", preserve_review_pairs=True)
    result = await translate_document(
        _content("context-aware"), config, _translator(fail_on=2)
    )

    assert result.failed_chunks == [2, 3]
    assert result.review_pairs is None


async def test_first_chunk_failure_raises_with_billed_cost():
    with pytest.raises(TranslationError, match="Translation failed") as exc_info:
        await translate_document(
            _content("context-aware"), _config("context-aware"), _translator(fail_on=1)
        )
    assert exc_info.value.cost == pytest.approx(0.5)


async def test_failed_page_counts_billed_cost():
    result = await translate_document(
        _content("page"), _config("page"), _translator(fail_on=2)
    )

    assert result.failed_pages == [2]
    assert result.cost == pytest.approx(2.5)
    assert (result.input_tokens, result.output_tokens) == (120, 60)


async def test_zero_max_cost_skips_remaining_pages():
    translator = _translator()
    result = await translate_document(
        _content("page"), _config("page", max_cost=0.0), translator
    )

    assert translator.translate.await_count == 1
    assert result.failed_pages == [2, 3]
    assert result.page_errors[2] == (
        "Skipped: Translation cost of $1.0000 exceeded maximum cost of $0.00"
    )
    assert result.text.startswith("<Aaaa.>\n\n[TRANSLATION_FAILED: Page 2]")
