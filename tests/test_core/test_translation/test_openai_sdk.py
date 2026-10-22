"""Exercise translation requests through the real OpenAI SDK without network I/O."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.translation.openai import OpenAITranslator
from tinbox.core.types import Glossary, GlossaryEntry, ModelType


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

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body.get("max_tokens", 0) > 32768:
            return httpx.Response(
                400,
                json={"error": {"message": "max_tokens exceeds the model limit"}},
            )
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
    if max_tokens is None:
        assert "max_tokens" not in requests[0]
    else:
        assert requests[0]["max_tokens"] == max_tokens
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
