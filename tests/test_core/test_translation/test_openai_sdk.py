"""Exercise translation requests through the real OpenAI SDK without network I/O."""

import json
from typing import Literal

import httpx
import pytest
from openai import AsyncOpenAI

from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.translation.openai import OpenAITranslator
from tinbox.core.types import Glossary, GlossaryEntry, ModelType, TranslationProtocol


def chat_response() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"translation": "Hello, world!"}',
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


@pytest.mark.parametrize(
    "model", [ModelType.OPENAI, ModelType.OLLAMA, ModelType.GEMINI]
)
@pytest.mark.parametrize("use_glossary", [False, True])
@pytest.mark.asyncio
async def test_translation_through_real_sdk(model: ModelType, use_glossary: bool):
    requests: list[dict] = []
    translated = {"translation": "Hello, world!"}
    if use_glossary:
        translated["glossary_extension"] = [{"term": "Welt", "translation": "world"}]

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(translated),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        translator = OpenAITranslator()
        translator._clients[model] = client
        response = await translator.translate(
            TranslationRequest(
                source_lang="de",
                target_lang="en",
                content="  Hallo, Welt!\n",
                content_type="text/plain",
                model=model,
                model_params={"model_name": "test-model"},
                glossary=Glossary() if use_glossary else None,
            )
        )

    assert response.text == "  Hello, world!\n"
    assert response.tokens_used == 20
    assert response.glossary_updates == (
        [GlossaryEntry(term="Welt", translation="world")] if use_glossary else []
    )
    assert len(requests) == 1
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[0]["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize(
    "model_name,default_effort",
    [
        ("gpt-6.1-sol", "low"),
        ("gpt-6-astra", "low"),
        ("gpt-6-sol", "none"),
        ("gpt-6-luna", "none"),
        ("gpt-6.1-sol-2026-10-20", "low"),
    ],
)
@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
@pytest.mark.parametrize("max_tokens", [None, 8192])
@pytest.mark.parametrize("local_model", [False, True])
@pytest.mark.asyncio
async def test_gpt6_request_contract(
    model_name: str,
    default_effort: str,
    effort: Literal["minimal", "low", "medium", "high"],
    max_tokens: int | None,
    local_model: bool,
):
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response())

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        translator = (
            OpenAITranslator()
            if max_tokens is None
            else OpenAITranslator(max_tokens=max_tokens)
        )
        translator._clients[ModelType.OPENAI] = client
        response = await translator.translate(
            TranslationRequest(
                source_lang="de",
                target_lang="en",
                content="Hallo, Welt!",
                content_type="text/plain",
                model=ModelType.OPENAI,
                model_params={
                    "model_name": model_name,
                    "top_p": 0.9,
                    "logprobs": True,
                    "top_logprobs": 2,
                },
                reasoning_effort=effort,
                local_model=local_model,
            )
        )

    assert response.text == "Hello, world!"
    assert response.tokens_used == 20
    if local_model:
        assert response.cost == 0.0
    else:
        assert response.cost > 0.0
    assert len(requests) == 1
    body = requests[0]
    assert body["model"] == model_name
    assert body["max_completion_tokens"] == (
        16384 if max_tokens is None else max_tokens
    )
    assert "max_tokens" not in body
    expected_effort = default_effort if effort == "minimal" else effort
    assert body["reasoning_effort"] == expected_effort
    if expected_effort == "none":
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.9
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 2
    else:
        assert "temperature" not in body
        assert "top_p" not in body
        assert "logprobs" not in body
        assert "top_logprobs" not in body
    assert body["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize(
    "provider,model_name",
    [
        (ModelType.GEMINI, "gemini-3.8-flash"),
        (ModelType.OPENAI, "gpt-4o"),
        (ModelType.OPENAI, "custom-gpt-6.1-sol"),
        (ModelType.OPENAI, "gpt-6.1-sol-custom"),
        (ModelType.OLLAMA, "gemma4:e2b"),
    ],
)
@pytest.mark.parametrize("max_tokens", [None, 8192])
@pytest.mark.asyncio
async def test_standard_and_compatible_chat_request_contract(
    provider: ModelType, model_name: str, max_tokens: int | None
):
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response())

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        translator = (
            OpenAITranslator()
            if max_tokens is None
            else OpenAITranslator(max_tokens=max_tokens)
        )
        translator._clients[provider] = client
        response = await translator.translate(
            TranslationRequest(
                source_lang="de",
                target_lang="en",
                content="Hallo, Welt!",
                content_type="text/plain",
                model=provider,
                model_params={"model_name": model_name},
            )
        )

    assert response.text == "Hello, world!"
    assert len(requests) == 1
    body = requests[0]
    assert body["max_tokens"] == (16384 if max_tokens is None else max_tokens)
    assert "max_completion_tokens" not in body
    assert body["temperature"] == 0.3
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("max_tokens,expected_limit", [(16384, 2048), (1024, 1024)])
@pytest.mark.asyncio
async def test_translategemma_keeps_completion_token_cap(
    max_tokens: int, expected_limit: int
):
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-test",
                "object": "text_completion",
                "created": 0,
                "model": "translategemma",
                "choices": [
                    {"index": 0, "text": "Hello, world!", "finish_reason": "stop"}
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        translator = OpenAITranslator(max_tokens=max_tokens)
        translator._clients[ModelType.OPENAI] = client
        response = await translator.translate(
            TranslationRequest(
                source_lang="de",
                target_lang="en",
                content="Hallo, Welt!",
                content_type="text/plain",
                model=ModelType.OPENAI,
                model_params={"model_name": "translategemma"},
                translation_protocol=TranslationProtocol.TRANSLATEGEMMA,
                local_model=True,
            )
        )

    assert response.text == "Hello, world!"
    assert len(requests) == 1
    body = requests[0]
    assert body["max_tokens"] == expected_limit
    assert body["temperature"] == 0.3
    assert body["stop"] == ["<end_of_turn>", "<eos>"]
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body
