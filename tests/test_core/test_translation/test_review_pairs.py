"""Behavior of opt-in pair preservation during context-aware translation."""

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import translate_document
from tinbox.core.translation.checkpoint import CheckpointManager, TranslationState
from tinbox.core.translation.glossary import GlossaryManager
from tinbox.core.translation.interface import (
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import GlossaryEntry, ModelType, TranslationConfig


class RecordingTranslator:
    def __init__(self, translations: list[str]) -> None:
        self.translations = translations
        self.requests: list[TranslationRequest] = []

    async def validate_model(self) -> bool:
        return True

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        translation = self.translations[len(self.requests)]
        self.requests.append(request)
        return TranslationResponse(
            text=translation,
            tokens_used=10,
            cost=0.001,
            time_taken=0.1,
            glossary_updates=[GlossaryEntry(term="letter", translation="Brief")],
        )


def review_config(tmp_path: Path, **kwargs: object) -> TranslationConfig:
    return TranslationConfig.model_validate(
        {
            "source_lang": "en",
            "target_lang": "de",
            "model": ModelType.OPENAI,
            "model_name": "gpt-4.1-mini",
            "algorithm": "context-aware",
            "input_file": tmp_path / "source.txt",
            "preserve_review_pairs": True,
            **kwargs,
        }
    )


@pytest.mark.parametrize("custom_split", [False, True])
async def test_pairs_preserve_each_request_and_response_without_context(
    tmp_path: Path, custom_split: bool
) -> None:
    source_chunks = ["First letter.\n\n", "Second letter."]
    text = "|||".join(source_chunks) if custom_split else "".join(source_chunks)
    translations = [" Erster Brief.\n\n", "Zweiter Brief.\n"]
    translator = RecordingTranslator(translations)
    config = review_config(
        tmp_path,
        context_size=15,
        custom_split_token="|||" if custom_split else None,
        use_glossary=True,
    )
    glossary = GlossaryManager()

    response = await translate_document(
        DocumentContent(pages=[text], content_type="text/plain"),
        config,
        translator,
        glossary_manager=glossary,
    )

    assert [request.content for request in translator.requests] == source_chunks
    assert response.review_pairs is not None
    assert [pair.model_dump() for pair in response.review_pairs] == [
        {"unit_id": 1, "source": source_chunks[0], "translation": translations[0]},
        {"unit_id": 2, "source": source_chunks[1], "translation": translations[1]},
    ]
    assert response.text == " Erster Brief.\n\nZweiter Brief.\n"
    assert response.tokens_used == 20
    assert response.cost == 0.002
    assert translator.requests[0].context is not None
    assert "Second letter." in translator.requests[0].context
    assert translator.requests[1].context is not None
    assert "First letter." in translator.requests[1].context
    assert translations[0] in translator.requests[1].context
    assert translator.requests[1].glossary is not None
    assert translator.requests[1].glossary.entries == {"letter": "Brief"}


async def test_pair_capture_is_disabled_by_default(tmp_path: Path) -> None:
    config = TranslationConfig(
        source_lang="en",
        target_lang="ar",
        model=ModelType.OPENAI,
        model_name="gpt-4.1-mini",
        algorithm="context-aware",
        input_file=tmp_path / "source.txt",
        custom_split_token="|||",
    )
    translator = RecordingTranslator(["رسالة أولى.\n", "رسالة ثانية."])

    response = await translate_document(
        DocumentContent(pages=["First.|||Second."], content_type="text/plain"),
        config,
        translator,
    )

    assert response.review_pairs is None
    assert response.text == "رسالة أولى.\nرسالة ثانية."  # noqa: RUF001 (Arabic text)
    assert len(translator.requests) == 2


@pytest.mark.parametrize(
    ("suffix", "algorithm"),
    [
        (".txt", "page"),
        (".txt", "sliding-window"),
        (".pdf", "context-aware"),
        (".docx", "context-aware"),
    ],
)
def test_pair_capture_rejects_unsupported_inputs(
    tmp_path: Path,
    suffix: str,
    algorithm: Literal["page", "sliding-window", "context-aware"],
) -> None:
    with pytest.raises(ValidationError, match=r"\.txt files.*context-aware"):
        review_config(
            tmp_path, input_file=tmp_path / f"source{suffix}", algorithm=algorithm
        )


@pytest.mark.parametrize("completed_chunks", [1, 2])
async def test_pair_capture_rejects_partial_and_complete_resume_before_requests(
    tmp_path: Path, completed_chunks: int
) -> None:
    config = review_config(
        tmp_path, checkpoint_dir=tmp_path / "checkpoints", custom_split_token="|||"
    )
    manager = CheckpointManager(config)
    await manager.save(
        TranslationState(
            source_lang="en",
            target_lang="de",
            algorithm="context-aware",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={index + 1: "Saved" for index in range(completed_chunks)},
            token_usage=10,
            cost=0.001,
            time_taken=0.1,
        )
    )
    translator = RecordingTranslator([])

    with pytest.raises(
        TranslationError,
        match=r"fresh translation.*new checkpoint directory.*omit --checkpoint-dir",
    ):
        await translate_document(
            DocumentContent(pages=["First.|||Second."], content_type="text/plain"),
            config,
            translator,
            checkpoint_manager=manager,
        )

    assert translator.requests == []
    checkpoint = await manager.load()
    assert checkpoint is not None
    assert len(checkpoint.translated_chunks) == completed_chunks


async def test_pair_capture_allows_fresh_checkpoint_saving(tmp_path: Path) -> None:
    config = review_config(
        tmp_path, checkpoint_dir=tmp_path / "checkpoints", custom_split_token="|||"
    )
    manager = CheckpointManager(config)
    translator = RecordingTranslator(["Erste.", "Zweite."])

    response = await translate_document(
        DocumentContent(pages=["First.|||Second."], content_type="text/plain"),
        config,
        translator,
        checkpoint_manager=manager,
    )

    assert response.review_pairs is not None
    assert len(response.review_pairs) == 2
    assert response.text == "Erste.Zweite."
    checkpoint = await manager.load()
    assert checkpoint is not None
    assert checkpoint.translated_chunks == {1: "Erste.", 2: "Zweite."}


@pytest.mark.parametrize("text", ["", "||||||"])
async def test_pair_capture_rejects_no_chunks_before_requests(
    tmp_path: Path, text: str
) -> None:
    translator = RecordingTranslator([])

    with pytest.raises(TranslationError, match="at least one non-empty source chunk"):
        await translate_document(
            DocumentContent(pages=[text], content_type="text/plain"),
            review_config(tmp_path, custom_split_token="|||"),
            translator,
        )

    assert translator.requests == []


async def test_empty_input_without_capture_keeps_existing_behavior(
    tmp_path: Path,
) -> None:
    translator = RecordingTranslator([])

    response = await translate_document(
        DocumentContent(pages=[""], content_type="text/plain"),
        review_config(tmp_path, preserve_review_pairs=False),
        translator,
    )

    assert response.text == ""
    assert response.review_pairs is None
    assert translator.requests == []
