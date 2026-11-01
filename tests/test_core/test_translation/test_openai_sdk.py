"""Exercise translation requests through the real OpenAI SDK without network I/O."""

import json
from typing import Literal

import httpx2
import pytest
from openai import AsyncOpenAI

from tinbox.core.translation.interface import (
    ProviderAccessError,
    TranslationError,
    TranslationRequest,
)
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
@pytest.mark.parametrize("max_tokens", [None, 4096])
@pytest.mark.asyncio
async def test_translation_through_real_sdk(
    model: ModelType, use_glossary: bool, max_tokens: int | None
):
    requests: list[dict] = []
    translated = {"translation": "Hello, world!"}
    if use_glossary:
        translated["glossary_extension"] = [{"term": "Welt", "translation": "world"}]

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body.get("max_tokens", 0) > 32768:
            return httpx2.Response(
                400,
                json={"error": {"message": "max_tokens exceeds the model limit"}},
            )
        return httpx2.Response(
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
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    ) as client:
        translator = (
            OpenAITranslator()
            if max_tokens is None
            else OpenAITranslator(max_tokens=max_tokens)
        )
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
    assert requests[0]["max_tokens"] == (16384 if max_tokens is None else max_tokens)
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

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=chat_response())

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
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

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=chat_response())

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
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

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/completions"
        requests.append(json.loads(request.content))
        return httpx2.Response(
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
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
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


def _gemini_error(status: int, message: str, details: list[dict]) -> list[dict]:
    """Gemini's OpenAI-compatible endpoint wraps its error object in a list."""
    return [
        {
            "error": {
                "code": status,
                "message": message,
                "status": "RESOURCE_EXHAUSTED",
                "details": details,
            }
        }
    ]


GEMINI_DAILY_QUOTA = _gemini_error(
    429,
    "You exceeded your current quota, please check your plan and billing details. "
    "Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 50, model: gemini-2.5-pro\nPlease retry in 34.1s.",
    [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": "generativelanguage.googleapis.com/"
                    "generate_content_free_tier_requests",
                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                }
            ],
        },
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
    ],
)
GEMINI_PER_MINUTE = _gemini_error(
    429,
    "You exceeded your current quota. Please retry in 11.7s.",
    [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}
            ],
        }
    ],
)


@pytest.mark.parametrize(
    ("status", "payload", "message"),
    [
        (
            401,
            {"error": {"message": "Incorrect API key", "code": "invalid_api_key"}},
            "Incorrect API key",
        ),
        (
            403,
            {"error": {"message": "Project does not have access", "code": None}},
            "Project does not have access",
        ),
        (
            429,
            {
                "error": {
                    "message": "You exceeded your current quota",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            },
            "You exceeded your current quota",
        ),
        (429, GEMINI_DAILY_QUOTA, "PerDay"),
        (
            429,
            _gemini_error(429, "Quota exceeded for metric: x, limit: 0", []),
            "limit: 0",
        ),
        (
            400,
            _gemini_error(400, "Your project has exceeded its spending cap.", []),
            "spending cap",
        ),
        (402, {"error": {"message": "Prepay credits are depleted"}}, "depleted"),
    ],
)
@pytest.mark.asyncio
async def test_access_errors_raise_provider_access_error_without_retry(
    status: int, payload: object, message: str
):
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(status, json=payload)

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    ) as client:
        translator = OpenAITranslator()
        translator._clients[ModelType.OPENAI] = client
        with pytest.raises(ProviderAccessError, match=message):
            await translator.translate(
                TranslationRequest(
                    source_lang="de",
                    target_lang="en",
                    content="Hallo, Welt!",
                    content_type="text/plain",
                    model=ModelType.OPENAI,
                    model_params={"model_name": "gpt-5.5"},
                )
            )

    assert len(requests) == 1


@pytest.mark.parametrize(
    ("finish_reason", "content", "error"),
    [
        ("length", '{"translation": "Bonj', "Invalid finish reason from model: length"),
        ("content_filter", None, "Invalid finish reason from model: content_filter"),
        ("stop", '{"translation": "Bonj', "Invalid JSON response format"),
    ],
)
@pytest.mark.parametrize("glossary", [None, Glossary()])
@pytest.mark.asyncio
async def test_rejected_reply_reports_billed_cost(
    finish_reason: str, content: str | None, error: str, glossary: Glossary | None
):
    """The provider bills a reply Tinbox rejects, so the error carries its cost."""
    reply = chat_response()
    reply["choices"][0]["finish_reason"] = finish_reason
    reply["choices"][0]["message"]["content"] = content
    reply["usage"] = {
        "prompt_tokens": 100_000,
        "completion_tokens": 16_384,
        "total_tokens": 116_384,
    }

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=reply)

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    ) as client:
        translator = OpenAITranslator()
        translator._clients[ModelType.OPENAI] = client
        with pytest.raises(TranslationError, match=error) as exc_info:
            await translator.translate(
                TranslationRequest(
                    source_lang="en",
                    target_lang="fr",
                    content="Hello",
                    content_type="text/plain",
                    model=ModelType.OPENAI,
                    model_params={"model_name": "gpt-6.1-sol"},
                    glossary=glossary,
                )
            )

    # gpt-6.1-sol short-prompt list price: $2 input, $10 output per 1M tokens
    assert exc_info.value.cost == pytest.approx((100_000 * 2 + 16_384 * 10) / 1_000_000)
    assert (exc_info.value.input_tokens, exc_info.value.output_tokens) == (
        100_000,
        16_384,
    )
