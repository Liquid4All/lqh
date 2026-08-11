"""Regression tests for transient-upstream-rejection handling in the client.

The backend proxies to pooled upstream models. A transient pool-side rejection
surfaces as a 400 ``request rejected by upstream model`` rather than a 5xx. That
shape must be retried (and treated as reconnectable in auto mode), while a
genuinely malformed-request 400 must still fail fast.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    RateLimitError,
)

from lqh.client import (
    chat_with_retry,
    describe_api_error,
    is_transient_upstream_error,
)


def _bad_request(body: dict) -> BadRequestError:
    req = httpx.Request("POST", "https://api.lqh.ai/v1/chat/completions")
    resp = httpx.Response(400, request=req, json=body)
    return BadRequestError(message=str(body), response=resp, body=body.get("error"))


def _status_error(status: int, message: str = "upstream model did not respond") -> APIStatusError:
    req = httpx.Request("POST", "https://api.lqh.ai/v1/chat/completions")
    body = {"error": {"code": status, "message": message, "type": "upstream_timeout"}}
    resp = httpx.Response(status, request=req, json=body)
    return APIStatusError(message=message, response=resp, body=body["error"])


UPSTREAM_REJECTION = {
    "error": {
        "code": 400,
        "message": "request rejected by upstream model",
        "type": "invalid_request_error",
    }
}
MALFORMED_REQUEST = {
    "error": {
        "code": 400,
        "message": "invalid 'messages': missing role",
        "type": "invalid_request_error",
    }
}


class TestIsTransientUpstreamError:
    def test_detects_upstream_rejection(self) -> None:
        assert is_transient_upstream_error(_bad_request(UPSTREAM_REJECTION)) is True

    def test_ignores_genuine_malformed_request(self) -> None:
        assert is_transient_upstream_error(_bad_request(MALFORMED_REQUEST)) is False

    def test_ignores_non_status_error(self) -> None:
        assert is_transient_upstream_error(ValueError("boom")) is False


class TestDescribeApiError:
    """The label the TUI puts in front of the user when a turn stalls."""

    def test_names_status_and_backend_message(self) -> None:
        got = describe_api_error(_status_error(502))
        assert got.startswith("HTTP 502")
        assert "upstream model did not respond" in got

    def test_truncates_a_long_message(self) -> None:
        got = describe_api_error(_status_error(500, "x" * 500))
        assert len(got) < 200
        assert got.endswith("...")

    def test_falls_back_to_the_exception_name(self) -> None:
        assert "ValueError" in describe_api_error(ValueError("boom"))


class TestChatWithRetry:
    @pytest.mark.asyncio
    async def test_retries_transient_upstream_then_succeeds(self) -> None:
        client = AsyncMock()
        sentinel = object()
        client.chat.completions.create = AsyncMock(
            side_effect=[_bad_request(UPSTREAM_REJECTION), sentinel]
        )
        result = await chat_with_retry(client, max_retries=3, model="x", messages=[])
        assert result is sentinel
        assert client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_gateway_statuses(self) -> None:
        """502/504 are the shapes users actually hit; both must be replayed."""
        for status in (502, 503, 504, 408, 500):
            client = AsyncMock()
            sentinel = object()
            client.chat.completions.create = AsyncMock(
                side_effect=[_status_error(status), sentinel]
            )
            result = await chat_with_retry(
                client, max_retries=1, model="x", messages=[]
            )
            assert result is sentinel, f"{status} was not retried"

    @pytest.mark.asyncio
    async def test_notifies_on_each_retry(self) -> None:
        """A silent retry ladder is what made a 502 feel like a hang."""
        seen: list[tuple[str, int, int, float]] = []

        async def on_retry(detail: str, attempt: int, total: int, wait: float) -> None:
            seen.append((detail, attempt, total, wait))

        client = AsyncMock()
        sentinel = object()
        client.chat.completions.create = AsyncMock(
            side_effect=[_status_error(502), _status_error(502), sentinel]
        )
        result = await chat_with_retry(
            client, max_retries=3, on_retry=on_retry, model="x", messages=[]
        )
        assert result is sentinel
        assert len(seen) == 2
        assert all("502" in detail for detail, *_ in seen)
        assert [attempt for _, attempt, _, _ in seen] == [1, 2]
        assert all(total == 4 for *_, total, _ in seen)

    @pytest.mark.asyncio
    async def test_broken_notifier_does_not_break_the_retry(self) -> None:
        async def on_retry(detail: str, attempt: int, total: int, wait: float) -> None:
            raise RuntimeError("surface is gone")

        client = AsyncMock()
        sentinel = object()
        client.chat.completions.create = AsyncMock(
            side_effect=[_status_error(502), sentinel]
        )
        result = await chat_with_retry(
            client, max_retries=1, on_retry=on_retry, model="x", messages=[]
        )
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_no_notification_when_the_call_succeeds(self) -> None:
        seen: list[str] = []

        async def on_retry(detail: str, attempt: int, total: int, wait: float) -> None:
            seen.append(detail)

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=object())
        await chat_with_retry(client, on_retry=on_retry, model="x", messages=[])
        assert seen == []

    @pytest.mark.asyncio
    async def test_malformed_request_fails_fast(self) -> None:
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            side_effect=_bad_request(MALFORMED_REQUEST)
        )
        with pytest.raises(BadRequestError):
            await chat_with_retry(client, max_retries=3, model="x", messages=[])
        # No retry: a malformed request can never succeed on replay.
        assert client.chat.completions.create.await_count == 1


# ---------------------------------------------------------------------------
# Ladder exhaustion + backoff schedule
#
# Neither the default 429 branch nor the connection branch had any coverage:
# every retry test here succeeds on call 2, and the only test reaching a
# terminal raise takes the non-retryable fast path without entering the loop.
# A restructure that failed to advance `attempt` in those branches would loop
# forever, so the failure mode was a HUNG SUITE rather than a red test.
# ---------------------------------------------------------------------------


async def test_rate_limit_exhausts_the_ladder_and_reraises() -> None:
    client = MagicMock()
    exc = RateLimitError(
        "rate limited",
        response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
        body=None,
    )
    client.chat.completions.create = AsyncMock(side_effect=exc)

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(RateLimitError):
            await asyncio.wait_for(
                chat_with_retry(client, max_retries=2, model="x", messages=[]),
                timeout=10,
            )

    assert client.chat.completions.create.await_count == 3  # max_retries + 1


async def test_connection_error_exhausts_the_ladder_and_reraises() -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=APIConnectionError(request=httpx.Request("POST", "http://x"))
    )

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(
                chat_with_retry(client, max_retries=2, model="x", messages=[]),
                timeout=10,
            )

    assert client.chat.completions.create.await_count == 3


async def test_backoff_schedule_is_exponential_from_one_second() -> None:
    """Pins 2**attempt. The parallel rate-limit branch uses its own exponent,
    so nothing else would catch drift here."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=APIConnectionError(request=httpx.Request("POST", "http://x"))
    )

    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    with patch("asyncio.sleep", new=_sleep):
        with pytest.raises(APIConnectionError):
            await chat_with_retry(client, max_retries=3, model="x", messages=[])

    assert slept == [1, 2, 4]
