"""Tests that page mode labels, skips and saves pages by their real PDF number."""

from pathlib import Path
from unittest.mock import AsyncMock

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import translate_document
from tinbox.core.translation.checkpoint import CheckpointManager
from tinbox.core.translation.interface import (
    ModelInterface,
    ProviderAccessError,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import ModelType, TranslationConfig


def _config(**overrides) -> TranslationConfig:
    settings = {
        "source_lang": "en",
        "target_lang": "fr",
        "model": ModelType.OPENAI,
        "model_name": "gpt-6.1-sol",
        "algorithm": "page",
        "input_file": Path("doc.pdf"),
        "pages": [2, 3, 7],
    }
    return TranslationConfig.model_validate(settings | overrides)


CONTENT = DocumentContent(
    pages=["two", "three", "seven"], content_type="text/plain", page_numbers=[2, 3, 7]
)


def _translator(error_on: str | None = None, error: Exception | None = None):
    async def translate(request: TranslationRequest) -> TranslationResponse:
        if request.content == error_on and error is not None:
            raise error
        return TranslationResponse(
            text=f"<{request.content}>", tokens_used=3, cost=0.1, time_taken=0.0
        )

    translator = AsyncMock(spec=ModelInterface)
    translator.translate.side_effect = translate
    return translator


async def test_failed_page_uses_its_pdf_page_number():
    response = await translate_document(
        CONTENT, _config(), _translator("three", TranslationError("bad reply"))
    )

    assert response.failed_pages == [3]
    assert response.text == (
        "<two>\n\n[TRANSLATION_FAILED: Page 3]\nReason: bad reply\n"
        "[/TRANSLATION_FAILED]\n\n<seven>"
    )


async def test_access_error_skips_the_remaining_selected_pages():
    response = await translate_document(
        CONTENT, _config(), _translator("three", ProviderAccessError("no credit"))
    )

    assert response.failed_pages == [3, 7]
    assert response.page_errors[7] == "Skipped: no credit"
    assert "<two>" in response.text


async def test_checkpoint_keys_pages_by_number_and_resumes(tmp_path):
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF fake")
    config = _config(
        input_file=source,
        checkpoint_dir=tmp_path / "ck",
        resume_from_checkpoint=True,
    )
    first = await translate_document(
        CONTENT,
        config,
        _translator("seven", TranslationError("bad reply")),
        checkpoint_manager=CheckpointManager(config),
    )
    assert first.failed_pages == [7]
    state = await CheckpointManager(config).load()
    assert state is not None
    assert state.translated_chunks == {2: "<two>", 3: "<three>"}

    retry = _translator()
    second = await translate_document(
        CONTENT, config, retry, checkpoint_manager=CheckpointManager(config)
    )

    assert second.failed_pages == []
    assert second.text == "<two>\n\n<three>\n\n<seven>"
    assert [c.args[0].content for c in retry.translate.await_args_list] == ["seven"]
