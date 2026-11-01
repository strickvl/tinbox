"""Tests for the shared rate-limit retry policy."""

import httpx2
import pytest
from openai import RateLimitError

from tinbox.core.translation._shared import build_retry_decorator, is_quota_exhausted
from tinbox.utils.logging import get_logger

TRANSIENT = {"message": "Rate limit reached", "type": "requests", "code": None}
NO_CREDIT = {
    "message": "You exceeded your current quota",
    "type": "insufficient_quota",
    "code": "insufficient_quota",
}


def rate_limit_error(body: dict, headers: dict | None = None) -> RateLimitError:
    request = httpx2.Request("POST", "http://test.invalid/v1/chat/completions")
    response = httpx2.Response(
        429, headers=headers or {}, json={"error": body}, request=request
    )
    return RateLimitError("rate limited", response=response, body=body)


def run_with_failures(errors: list[RateLimitError]) -> tuple[int, list[float], object]:
    """Call a decorated function that raises ``errors`` in turn, then succeeds."""
    calls = 0
    sleeps: list[float] = []

    def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls <= len(errors):
            raise errors[calls - 1]
        return "ok"

    decorated = build_retry_decorator(RateLimitError, get_logger(__name__))(flaky)
    try:
        result: object = decorated.retry_with(sleep=sleeps.append)()
    except RateLimitError as e:
        result = e
    return calls, sleeps, result


def test_transient_rate_limit_is_retried_with_exponential_wait():
    calls, sleeps, result = run_with_failures([rate_limit_error(TRANSIENT)] * 2)

    assert result == "ok"
    assert calls == 3
    assert sleeps == [4, 4]


@pytest.mark.parametrize(("header", "expected"), [("7", 7.0), ("999", 120.0)])
def test_retry_after_header_sets_the_wait(header: str, expected: float):
    error = rate_limit_error(TRANSIENT, headers={"retry-after": header})
    calls, sleeps, result = run_with_failures([error])

    assert result == "ok"
    assert calls == 2
    assert sleeps == [expected]


def test_quota_exhausted_is_not_retried():
    error = rate_limit_error(NO_CREDIT)
    calls, sleeps, result = run_with_failures([error])

    assert calls == 1
    assert sleeps == []
    assert result is error


def test_last_attempt_reraises_the_original_error():
    errors = [rate_limit_error(TRANSIENT) for _ in range(10)]
    calls, sleeps, result = run_with_failures(errors)

    assert calls == 10
    assert len(sleeps) == 9
    assert result is errors[-1]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (NO_CREDIT, True),
        ({"error": {"code": "insufficient_quota"}}, True),
        ({"type": "insufficient_quota"}, True),
        (TRANSIENT, False),
        ("not a dict", False),
    ],
)
def test_is_quota_exhausted_reads_code_and_body(body: object, expected: bool):
    request = httpx2.Request("POST", "http://test.invalid/v1/chat/completions")
    response = httpx2.Response(429, request=request)
    error = RateLimitError("rate limited", response=response, body=body)

    assert is_quota_exhausted(error) is expected
