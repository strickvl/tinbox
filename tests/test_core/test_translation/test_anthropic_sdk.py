"""Exercise Claude requests through the real SDK without network I/O."""

import json
from typing import Literal

import httpx
import pytest
from anthropic import AsyncAnthropic

from tinbox.core.translation.anthropic import AnthropicTranslator
from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.types import ModelType


@pytest.mark.parametrize(
    ("model_name", "effort", "thinking", "output_effort", "temperature"),
    [
        ("claude-sonnet-5-5", "minimal", {"type": "adaptive"}, "low", None),
        ("claude-sonnet-5-5", "low", {"type": "adaptive"}, "low", None),
        ("claude-sonnet-5-5", "medium", {"type": "adaptive"}, "medium", None),
        ("claude-sonnet-5-5", "high", {"type": "adaptive"}, "high", None),
        ("claude-opus-5-5", "minimal", {"type": "adaptive"}, "low", None),
        ("claude-haiku-5-5", "minimal", {"type": "adaptive"}, "low", None),
        ("claude-fable-5-1", "high", {"type": "adaptive"}, "high", None),
        ("claude-sonnet-4-5", "minimal", None, None, 0.3),
        (
            "claude-sonnet-4-5",
            "high",
            {"type": "enabled", "budget_tokens": 8192},
            None,
            None,
        ),
    ],
)
@pytest.mark.parametrize("max_tokens", [None, 12288])
@pytest.mark.asyncio
async def test_translation_through_real_sdk(
    model_name: str,
    effort: Literal["minimal", "low", "medium", "high"],
    thinking: dict | None,
    output_effort: str | None,
    temperature: float | None,
    max_tokens: int | None,
):
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        body = json.loads(request.content)
        requests.append(body)
        if thinking is None:
            assert "thinking" not in body
        else:
            assert body["thinking"] == thinking
        if output_effort is None:
            assert "output_config" not in body
        else:
            assert body["output_config"] == {"effort": output_effort}
        if temperature is None:
            assert "temperature" not in body
        else:
            assert body["temperature"] == temperature
        assert "top_p" not in body
        assert "top_k" not in body
        assert body["max_tokens"] == (16384 if max_tokens is None else max_tokens)
        assert body["model"] == model_name
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "test"},
                    {"type": "text", "text": '{"translation": "Hello, world!"}'},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
        )

    async with AsyncAnthropic(
        api_key="test-key",
        base_url="http://test.invalid",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        translator = (
            AnthropicTranslator()
            if max_tokens is None
            else AnthropicTranslator(max_tokens=max_tokens)
        )
        translator._client = client
        response = await translator.translate(
            TranslationRequest(
                source_lang="de",
                target_lang="en",
                content="  Hallo, Welt!\n",
                content_type="text/plain",
                model=ModelType.ANTHROPIC,
                model_params={"model_name": model_name},
                reasoning_effort=effort,
            )
        )

    assert response.text == "  Hello, world!\n"
    assert response.tokens_used == 20
    assert len(requests) == 1
