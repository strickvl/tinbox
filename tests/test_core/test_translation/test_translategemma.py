"""TranslateGemma protocol checks using the real SDK over mocked HTTP."""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import httpx2
import pytest
from openai import AsyncOpenAI
from typer.testing import CliRunner

from tinbox.cli import app
from tinbox.core.translation.checkpoint import CheckpointManager, TranslationState
from tinbox.core.translation.interface import TranslationError, TranslationRequest
from tinbox.core.translation.openai import OpenAITranslator
from tinbox.core.types import (
    ModelType,
    TranslationConfig,
    TranslationProtocol,
)

GERMAN_TO_ENGLISH_PROMPT = (
    "<start_of_turn>user\n"
    "You are a professional German (de) to English (en) translator. Your goal is to "
    "accurately convey the meaning and nuances of the original German text while "
    "adhering to English grammar, vocabulary, and cultural sensitivities.\n"
    "Produce only the English translation, without any additional explanations or "
    "commentary. Please translate the following German text into English:\n\n\n"
    "Hallo, Welt!<end_of_turn>\n<start_of_turn>model\n"
)


def completion_response(
    text: str = "Hello, world!", finish_reason: str = "stop"
) -> dict:
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 0,
        "model": "local-alias",
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,finish_reason",
    [("Hello, world!", "stop"), ("", "stop"), ("Partial", "length")],
)
async def test_translategemma_real_sdk(text: str, finish_reason: str):
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        requests.append(body)
        assert body["prompt"] == GERMAN_TO_ENGLISH_PROMPT
        assert "messages" not in body
        assert "response_format" not in body
        assert "reasoning_effort" not in body
        assert body["max_tokens"] == 2048
        return httpx2.Response(200, json=completion_response(text, finish_reason))

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    ) as client:
        translator = OpenAITranslator()
        translator._clients[ModelType.OPENAI] = client
        request = TranslationRequest(
            source_lang="de",
            target_lang="en",
            content="  Hallo, Welt!\n",
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "local-alias"},
            translation_protocol=TranslationProtocol.TRANSLATEGEMMA,
            local_model=True,
        )
        if not text or finish_reason != "stop":
            with pytest.raises(
                TranslationError, match=r"No content|Invalid finish reason"
            ):
                await translator.translate(request)
        else:
            response = await translator.translate(request)
            assert response.text == "  Hello, world!\n"
            assert response.tokens_used == 110
            assert response.cost == 0
            assert response.glossary_updates == []
    assert len(requests) == 1


@pytest.mark.parametrize(
    "arguments,suffix,message",
    [
        ([], ".txt", "explicit source language"),
        (["--from", "de", "--algorithm", "context-aware"], ".txt", "--algorithm page"),
        (["--from", "de", "--glossary"], ".txt", "glossary or reasoning"),
        (
            ["--from", "de", "--reasoning-effort", "high"],
            ".txt",
            "glossary or reasoning",
        ),
        (["--from", "de"], ".pdf", ".txt files"),
        (["--from", "de", "--split-token", "|||"], ".txt", "--split-token"),
    ],
)
def test_translategemma_cli_rejects_unsupported_options(
    tmp_path: Path, monkeypatch, arguments: list[str], suffix: str, message: str
):
    source = tmp_path / f"source{suffix}"
    source.write_text("Hallo, Welt!")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    create_translator = Mock()
    monkeypatch.setattr("tinbox.cli.create_translator", create_translator)
    result = CliRunner().invoke(
        app,
        [
            "translate",
            str(source),
            "--model",
            "openai:local-alias",
            "--translation-protocol",
            "translategemma",
            "--dry-run",
            *arguments,
        ],
    )
    assert result.exit_code == 1
    assert message in result.stdout
    create_translator.assert_not_called()


def test_translategemma_cli_with_real_sdk(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("  Hallo, Welt!\n")
    output = tmp_path / "translated.txt"
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.url.path == "/v1/completions"
        assert json.loads(request.content)["prompt"] == GERMAN_TO_ENGLISH_PROMPT
        return httpx2.Response(200, json=completion_response())

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(
        "tinbox.core.translation.openai.AsyncOpenAI", lambda **kwargs: client
    )
    try:
        result = CliRunner().invoke(
            app,
            [
                "translate",
                str(source),
                "--from",
                "de",
                "--to",
                "en",
                "--model",
                "openai:local-alias",
                "--translation-protocol",
                "translategemma",
                "--local",
                "--output",
                str(output),
            ],
        )
    finally:
        asyncio.run(client.close())
    assert result.exit_code == 0, result.stdout
    assert output.read_text() == "  Hello, world!\n"
    assert len(requests) == 1
    assert "None (local model)" in result.stdout
    assert "Estimated Cost" not in result.stdout
    assert "Final Cost" not in result.stdout
    assert "potential issues" not in result.stdout


def test_ollama_context_notice_does_not_require_confirmation(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.txt"
    source.write_text("Hallo, Welt!")
    output = tmp_path / "translated.txt"

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"translation":"Hello, world!"}',
                        },
                    }
                ],
            },
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    )
    monkeypatch.setattr(
        "tinbox.core.translation.openai.AsyncOpenAI", lambda **kwargs: client
    )
    try:
        result = CliRunner().invoke(
            app,
            [
                "translate",
                str(source),
                "--model",
                "ollama:test",
                "--output",
                str(output),
            ],
        )
    finally:
        asyncio.run(client.close())
    assert result.exit_code == 0, result.stdout
    assert output.read_text() == "Hello, world!"
    assert "Do you want to proceed?" not in result.stdout
    assert "approximate" in result.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"translation_protocol": TranslationProtocol.TRANSLATEGEMMA},
        {"local_model": True},
    ],
)
async def test_checkpoint_rejects_different_protocol_or_pricing(
    tmp_path: Path, changed: dict
):
    config = TranslationConfig(
        source_lang="de",
        target_lang="en",
        model=ModelType.OPENAI,
        model_name="local-alias",
        algorithm="page",
        input_file=tmp_path / "source.txt",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    manager = CheckpointManager(config)
    await manager.save(
        TranslationState(
            source_lang="de",
            target_lang="en",
            algorithm="page",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={1: "Hello"},
            token_usage=10,
            cost=1.0,
            time_taken=1.0,
        )
    )
    other = CheckpointManager(config.model_copy(update=changed))
    assert await other.load() is None
