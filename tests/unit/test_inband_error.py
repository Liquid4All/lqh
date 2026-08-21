"""In-band error detection for heartbeat-committed 200 responses.

Once the backend's keepalive heartbeat has committed a 200 for a slow call,
upstream failures arrive as a 200 whose body is the OpenAI error envelope
with the true status in ``error.code``. ``create_client`` installs a wrapper
that re-raises those as the matching ``APIStatusError`` subclass so retry
ladders behave as if the real status had been on the wire.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import APIStatusError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletion

from lqh.client import (
    RETRYABLE_STATUS,
    _inband_status_error,
    _install_default_max_tokens,
    _install_inband_error_check,
    chat_with_retry,
)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key="test-key", base_url="https://api.lqh.ai/v1", max_retries=0)


def _error_completion(err: dict) -> ChatCompletion:
    # The SDK parses a 200 error body leniently (construct_type keeps unknown
    # fields, missing required fields become None) — mirror that shape.
    return ChatCompletion.construct(error=err)


class TestInbandStatusError:
    def test_status_from_code_and_subclass_mapping(self):
        client = _client()
        exc = _inband_status_error(client, {"code": 429, "message": "slow down", "type": "rate_limit_error"})
        assert isinstance(exc, RateLimitError)
        assert exc.status_code == 429

        exc = _inband_status_error(client, {"code": 500, "message": "boom"})
        assert isinstance(exc, InternalServerError)
        assert exc.status_code == 500

    def test_missing_or_bogus_code_falls_back_to_502(self):
        client = _client()
        for err in ({"message": "no code"}, {"code": "nope"}, {"code": 200}):
            exc = _inband_status_error(client, err)
            assert isinstance(exc, APIStatusError)
            assert exc.status_code == 502

    def test_524_yields_plain_status_error(self):
        exc = _inband_status_error(_client(), {"code": 524, "message": "edge timeout"})
        assert isinstance(exc, APIStatusError)
        assert exc.status_code == 524


class TestCreateCheckedWrapper:
    async def test_error_body_raises_with_embedded_code(self):
        client = _client()
        client.chat.completions.create = AsyncMock(
            return_value=_error_completion(
                {"code": 502, "message": "upstream timeout, please retry", "type": "upstream_timeout"}
            )
        )
        _install_inband_error_check(client)

        with pytest.raises(APIStatusError) as excinfo:
            await client.chat.completions.create(model="small", messages=[])
        assert excinfo.value.status_code == 502
        assert "upstream timeout" in str(excinfo.value)

    async def test_inband_429_raises_rate_limit_error(self):
        client = _client()
        client.chat.completions.create = AsyncMock(
            return_value=_error_completion({"code": 429, "message": "rate limit reached", "type": "rate_limit_error"})
        )
        _install_inband_error_check(client)

        with pytest.raises(RateLimitError):
            await client.chat.completions.create(model="small", messages=[])

    async def test_error_next_to_real_choices_passes_through(self):
        resp = SimpleNamespace(
            error={"code": 500, "message": "partial provider hiccup"},
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        )
        client = _client()
        client.chat.completions.create = AsyncMock(return_value=resp)
        _install_inband_error_check(client)

        assert await client.chat.completions.create(model="small", messages=[]) is resp

    async def test_clean_response_passes_through_and_patches_compose(self):
        resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
        inner = AsyncMock(return_value=resp)
        client = _client()
        client.chat.completions.create = inner
        # Same installation order as create_client.
        _install_default_max_tokens(client)
        _install_inband_error_check(client)

        out = await client.chat.completions.create(model="small", messages=[])
        assert out is resp
        assert inner.call_args.kwargs["max_tokens"] > 0


class TestRetryOn524:
    async def test_chat_with_retry_retries_524(self):
        assert 524 in RETRYABLE_STATUS
        import httpx

        err = APIStatusError(
            "edge timeout",
            response=httpx.Response(524, request=httpx.Request("POST", "https://api.lqh.ai/v1")),
            body=None,
        )
        ok = object()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=[err, ok])))
        )
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await chat_with_retry(client, max_retries=1, model="small", messages=[])
        assert result is ok
