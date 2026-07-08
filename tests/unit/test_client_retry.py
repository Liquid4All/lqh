"""Regression tests for transient-upstream-rejection handling in the client.

The backend proxies to pooled upstream models. A transient pool-side rejection
surfaces as a 400 ``request rejected by upstream model`` rather than a 5xx. That
shape must be retried (and treated as reconnectable in auto mode), while a
genuinely malformed-request 400 must still fail fast.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError

from lqh.client import chat_with_retry, is_transient_upstream_error


def _bad_request(body: dict) -> BadRequestError:
    req = httpx.Request("POST", "https://api.lqh.ai/v1/chat/completions")
    resp = httpx.Response(400, request=req, json=body)
    return BadRequestError(message=str(body), response=resp, body=body.get("error"))


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
    async def test_malformed_request_fails_fast(self) -> None:
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            side_effect=_bad_request(MALFORMED_REQUEST)
        )
        with pytest.raises(BadRequestError):
            await chat_with_retry(client, max_retries=3, model="x", messages=[])
        # No retry: a malformed request can never succeed on replay.
        assert client.chat.completions.create.await_count == 1
