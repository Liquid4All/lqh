"""Asynchronous completions in the client (backend/CLI_API.md, "Async completions").

A long orchestration turn is handed to the server (``X-LQH-Async: 1``), which
answers 202 + an id, and the client long-polls for the result instead of
holding one HTTP connection open for half an hour.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, NotFoundError, RateLimitError
from openai.types.chat import ChatCompletion

from lqh.client import (
    ASYNC_HEADER,
    POLL_MAX_CONSECUTIVE_FAILURES,
    POLL_WAIT_S,
    AsyncCompletionHooks,
    CompletionCancelledError,
    CompletionLostError,
    CompletionPollExhausted,
    cancel_completion,
    capture_api_metrics,
    chat_with_retry,
    create_client,
    describe_api_error,
    is_pending_resumable_error,
)


def _client() -> AsyncOpenAI:
    return create_client("test-key", "https://api.lqh.ai/v1", max_retries=0)


def _completion(content: str = "done") -> ChatCompletion:
    return ChatCompletion.construct(
        id="chatcmpl-1", object="chat.completion", created=1, model="orchestration:15",
        choices=[SimpleNamespace(index=0, finish_reason="stop",
                                 message=SimpleNamespace(role="assistant", content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )


def _running(cid: str = "lqhc_" + "a" * 32, tokens: int = 0, elapsed_ms: int = 0, poll_after_ms: int = 0) -> ChatCompletion:
    return ChatCompletion.construct(
        id=cid, object="chat.completion.async", status="running",
        tokens_so_far=tokens, tokens_estimated=True, elapsed_ms=elapsed_ms, poll_after_ms=poll_after_ms,
    )


def _raw(status_code: int, parsed: Any) -> SimpleNamespace:
    return SimpleNamespace(status_code=status_code, headers=httpx.Headers(), parse=lambda: parsed)


def _status_error(status: int, cls: type = APIStatusError) -> APIStatusError:
    resp = httpx.Response(status, request=httpx.Request("GET", "https://api.lqh.ai/v1/chat/completions/x"))
    return cls("boom", response=resp, body=None)


def _conn_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("GET", "https://api.lqh.ai/v1"))


def _wire(client: AsyncOpenAI, *, post: list[Any], polls: list[Any]) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Script the POST (raw responses or exceptions) and the poll GETs."""
    def _side_effects(items: list[Any]):
        async def _fn(*args: Any, **kwargs: Any) -> Any:
            item = items.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return _fn
    raw_create = AsyncMock(side_effect=_side_effects(post))
    client.chat.completions.with_raw_response.create = raw_create  # type: ignore[method-assign]
    get = AsyncMock(side_effect=_side_effects(polls))
    client.get = get  # type: ignore[method-assign]
    delete = AsyncMock(return_value=None)
    client.delete = delete  # type: ignore[method-assign]
    return raw_create, get, delete


async def test_sync_mode_never_sends_the_async_header():
    client = _client()
    create = AsyncMock(return_value=_completion())
    client.chat.completions.create = create  # type: ignore[method-assign]
    raw_create = AsyncMock()
    client.chat.completions.with_raw_response.create = raw_create  # type: ignore[method-assign]

    resp = await chat_with_retry(client, model="judge:small", messages=[])

    assert resp.choices[0].message.content == "done"
    raw_create.assert_not_called()
    assert ASYNC_HEADER not in str(create.call_args)


async def test_fast_completion_returns_inline(monkeypatch):
    client = _client()
    raw_create, get, _ = _wire(client, post=[_raw(200, _completion("quick"))], polls=[])
    started: list[str] = []
    hooks = AsyncCompletionHooks(on_started=AsyncMock(side_effect=lambda cid: started.append(cid)))
    log: list[dict] = []
    token = capture_api_metrics(log)
    try:
        resp = await chat_with_retry(client, async_mode=True, async_hooks=hooks, model="orchestration:15", messages=[])
    finally:
        pass
    assert resp.choices[0].message.content == "quick"
    assert raw_create.call_args.kwargs["extra_headers"][ASYNC_HEADER] == "1"
    get.assert_not_called()
    assert started == []
    assert log[-1]["phase"] == "post"


async def test_202_then_polls_until_done(monkeypatch):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    cid = "lqhc_" + "b" * 32
    raw_create, get, _ = _wire(
        client,
        post=[_raw(202, _running(cid))],
        polls=[_raw(202, _running(cid, tokens=1200, elapsed_ms=61000, poll_after_ms=1000)),
               _raw(202, _running(cid, tokens=5400, elapsed_ms=120000, poll_after_ms=30000)),
               _raw(200, _completion("finally"))],
    )
    started: list[str] = []
    progress: list[tuple[int, float]] = []
    hooks = AsyncCompletionHooks(
        on_started=AsyncMock(side_effect=lambda c: started.append(c)),
        on_progress=lambda t, e: progress.append((t, e)),
    )
    log: list[dict] = []
    capture_api_metrics(log)

    resp = await chat_with_retry(client, async_mode=True, async_hooks=hooks, model="orchestration:15", messages=[])

    assert resp.choices[0].message.content == "finally"
    assert started == [cid]
    assert progress == [(1200, 61.0), (5400, 120.0)]
    assert get.call_count == 3
    path, kwargs = get.call_args.args[0], get.call_args.kwargs
    assert path == f"/chat/completions/{cid}"
    assert kwargs["options"]["params"] == {"wait": POLL_WAIT_S}
    assert kwargs["options"]["timeout"].read >= 45
    sleeps = [c.args[0] for c in __import__("lqh.client", fromlist=["asyncio"]).asyncio.sleep.call_args_list]
    assert sleeps == [1.0, 10.0]  # hint honoured, then clamped to POLL_AFTER_MAX_S
    assert log[-1]["phase"] == "poll" and log[-1]["polls"] == 3 and log[-1]["completion_id"] == cid


async def test_inband_running_envelope_on_200_is_treated_as_async(monkeypatch):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    cid = "lqhc_" + "c" * 32
    _wire(client, post=[_raw(200, _running(cid))], polls=[_raw(200, _completion("ok"))])
    resp = await chat_with_retry(client, async_mode=True, model="orchestration:15", messages=[])
    assert resp.choices[0].message.content == "ok"


async def test_poll_failures_retry_the_poll_not_the_post(monkeypatch):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    cid = "lqhc_" + "d" * 32
    rl = _status_error(429, RateLimitError)
    rl.response.headers["Retry-After"] = "7"
    raw_create, get, _ = _wire(
        client,
        post=[_raw(202, _running(cid))],
        polls=[_status_error(502), _conn_error(), httpx.ReadTimeout("slow"), rl, _raw(200, _completion("ok"))],
    )
    retries: list[tuple[str, int, int, float]] = []

    async def on_poll_retry(desc: str, n: int, total: int, wait: float) -> None:
        retries.append((desc, n, total, wait))

    resp = await chat_with_retry(
        client, async_mode=True, async_hooks=AsyncCompletionHooks(on_poll_retry=on_poll_retry),
        model="orchestration:15", messages=[],
    )
    assert resp.choices[0].message.content == "ok"
    assert raw_create.call_count == 1
    assert get.call_count == 5
    assert [r[1] for r in retries] == [1, 2, 3, 4]
    assert all(r[2] == POLL_MAX_CONSECUTIVE_FAILURES for r in retries)
    assert retries[0][3] == 1.0 and retries[1][3] == 2.0 and retries[2][3] == 4.0
    assert retries[3][3] == 7.0  # Retry-After honoured


async def test_too_many_poll_failures_raise_exhausted_without_spending_attempts(monkeypatch):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    cid = "lqhc_" + "e" * 32
    raw_create, get, _ = _wire(
        client, post=[_raw(202, _running(cid))],
        polls=[_status_error(503) for _ in range(POLL_MAX_CONSECUTIVE_FAILURES + 1)],
    )
    with pytest.raises(CompletionPollExhausted) as exc:
        await chat_with_retry(client, async_mode=True, max_retries=3, model="orchestration:15", messages=[])
    assert exc.value.completion_id == cid
    assert isinstance(exc.value, APIConnectionError)
    assert raw_create.call_count == 1  # no re-send: the turn is still running server-side
    assert is_pending_resumable_error(exc.value)
    assert "lost contact" in describe_api_error(exc.value)


@pytest.mark.parametrize("lost", [
    lambda: _status_error(404, NotFoundError),
    lambda: _status_error(410),
    lambda: _raw(200, ChatCompletion.construct(id="lqhc_" + "f" * 32, status="lost")),
])
async def test_lost_completion_is_resent_as_one_attempt(monkeypatch, lost):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    cid = "lqhc_" + "f" * 32
    raw_create, get, _ = _wire(
        client,
        post=[_raw(202, _running(cid)), _raw(200, _completion("second try"))],
        polls=[lost()],
    )
    lost_ids: list[str] = []
    notices: list[tuple[str, int, int, float]] = []

    async def on_lost(c: str) -> None:
        lost_ids.append(c)

    async def on_retry(desc: str, attempt: int, total: int, wait: float) -> None:
        notices.append((desc, attempt, total, wait))

    resp = await chat_with_retry(
        client, async_mode=True, max_retries=1, on_retry=on_retry,
        async_hooks=AsyncCompletionHooks(on_lost=on_lost), model="orchestration:15", messages=[],
    )
    assert resp.choices[0].message.content == "second try"
    assert lost_ids == [cid]
    assert raw_create.call_count == 2
    assert notices and "lost" in notices[0][0] and notices[0][3] == 0.0

    # With no attempts left the loss escapes as a reconnectable error.
    raw_create2, _, _ = _wire(client, post=[_raw(202, _running(cid))], polls=[_status_error(410)])
    with pytest.raises(CompletionLostError) as exc:
        await chat_with_retry(client, async_mode=True, max_retries=0, model="orchestration:15", messages=[])
    assert isinstance(exc.value, APIConnectionError)
    assert not is_pending_resumable_error(exc.value)


async def test_resume_id_polls_first_and_falls_back_to_a_fresh_post(monkeypatch):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    old = "lqhc_" + "0" * 32
    raw_create, get, _ = _wire(client, post=[], polls=[_raw(200, _completion("resumed"))])
    resp = await chat_with_retry(client, async_mode=True, resume_id=old, model="orchestration:15", messages=[])
    assert resp.choices[0].message.content == "resumed"
    raw_create.assert_not_called()
    assert get.call_args.args[0] == f"/chat/completions/{old}"

    raw_create, get, _ = _wire(client, post=[_raw(200, _completion("fresh"))], polls=[_status_error(404, NotFoundError)])
    resp = await chat_with_retry(client, async_mode=True, resume_id=old, model="orchestration:15", messages=[])
    assert resp.choices[0].message.content == "fresh"
    assert raw_create.call_count == 1 and get.call_count == 1


async def test_poll_inband_error_and_non_retryable_status_raise(monkeypatch):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    cid = "lqhc_" + "1" * 32
    _wire(client, post=[_raw(202, _running(cid))],
          polls=[_raw(200, ChatCompletion.construct(error={"code": 402, "message": "out of credits"}))])
    with pytest.raises(APIStatusError) as exc:
        await chat_with_retry(client, async_mode=True, model="orchestration:15", messages=[])
    assert exc.value.status_code == 402

    lost_hook = AsyncMock()
    _wire(client, post=[_raw(202, _running(cid))], polls=[_status_error(401)])
    with pytest.raises(APIStatusError) as exc:
        await chat_with_retry(client, async_mode=True, async_hooks=AsyncCompletionHooks(on_lost=lost_hook),
                              model="orchestration:15", messages=[])
    assert exc.value.status_code == 401
    lost_hook.assert_not_called()


async def test_cancelled_poll_propagates_and_does_not_delete(monkeypatch):
    import asyncio

    client = _client()
    cid = "lqhc_" + "2" * 32
    never = asyncio.Event()

    async def hang(*a: Any, **k: Any) -> Any:
        await never.wait()

    _wire(client, post=[_raw(202, _running(cid))], polls=[])
    client.get = AsyncMock(side_effect=hang)  # type: ignore[method-assign]
    task = asyncio.create_task(chat_with_retry(client, async_mode=True, model="orchestration:15", messages=[]))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.delete.assert_not_called()  # type: ignore[attr-defined]


async def test_cancel_completion_is_best_effort():
    client = _client()
    client.delete = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await cancel_completion(client, "lqhc_" + "3" * 32) is True
    assert client.delete.call_args.args[0] == "/chat/completions/lqhc_" + "3" * 32
    client.delete = AsyncMock(side_effect=_status_error(404, NotFoundError))  # type: ignore[method-assign]
    assert await cancel_completion(client, "lqhc_" + "3" * 32) is False


def test_is_pending_resumable_error_table():
    assert is_pending_resumable_error(_conn_error())
    assert is_pending_resumable_error(_status_error(429, RateLimitError))
    assert is_pending_resumable_error(_status_error(502))
    assert not is_pending_resumable_error(_status_error(400))
    assert not is_pending_resumable_error(_status_error(401))
    assert not is_pending_resumable_error(ValueError("x"))


# ---------------------------------------------------------------------------
# Review fixes: terminal results on the poll endpoint, cancellation, and
# idempotent submission.
# ---------------------------------------------------------------------------


def _terminal_error(status: int, cls: type = APIStatusError, state: str = "error") -> APIStatusError:
    exc = _status_error(status, cls)
    exc.response.headers["X-LQH-Completion-Status"] = state
    return exc


async def test_terminal_error_on_poll_goes_to_the_ladder_not_the_poll_loop(monkeypatch):
    """A 429/502 that IS the completion's result must not be retried as a poll."""
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    cid = "lqhc_" + "4" * 32
    raw_create, get, _ = _wire(
        client,
        post=[_raw(202, _running(cid)), _raw(200, _completion("second"))],
        polls=[_terminal_error(429, RateLimitError)],
    )
    lost: list[str] = []

    async def on_lost(c: str) -> None:
        lost.append(c)

    resp = await chat_with_retry(
        client, async_mode=True, max_retries=1, async_hooks=AsyncCompletionHooks(on_lost=on_lost),
        model="orchestration:15", messages=[],
    )
    assert resp.choices[0].message.content == "second"
    assert get.call_count == 1  # not polled again
    assert lost == [cid]  # record dropped
    assert raw_create.call_count == 2
    first = raw_create.call_args_list[0].kwargs["extra_headers"]["X-LQH-Request-Id"]
    second = raw_create.call_args_list[1].kwargs["extra_headers"]["X-LQH-Request-Id"]
    assert first != second, "a re-send after a terminal result must use a fresh request id"

    # With no attempts left the terminal status escapes as itself.
    _wire(client, post=[_raw(202, _running(cid))], polls=[_terminal_error(502)])
    with pytest.raises(APIStatusError) as exc:
        await chat_with_retry(client, async_mode=True, max_retries=0, model="orchestration:15", messages=[])
    assert exc.value.status_code == 502


@pytest.mark.parametrize("cancelled", [
    lambda: _terminal_error(410, state="cancelled"),
    lambda: (lambda e: (setattr(e, "body", {"error": {"type": "completion_cancelled", "code": 410}}), e)[1])(_status_error(410)),
    lambda: _raw(200, ChatCompletion.construct(id="lqhc_" + "5" * 32, status="cancelled")),
])
async def test_cancelled_completion_is_terminal_and_not_resent(monkeypatch, cancelled):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    cid = "lqhc_" + "5" * 32
    raw_create, _, _ = _wire(client, post=[_raw(202, _running(cid))], polls=[cancelled()])
    lost: list[str] = []

    async def on_lost(c: str) -> None:
        lost.append(c)

    with pytest.raises(CompletionCancelledError) as exc:
        await chat_with_retry(client, async_mode=True, max_retries=3,
                              async_hooks=AsyncCompletionHooks(on_lost=on_lost), model="orchestration:15", messages=[])
    assert exc.value.status_code == 410 and exc.value.completion_id == cid
    assert raw_create.call_count == 1, "a cancelled turn must never be re-sent"
    assert lost == [cid]
    assert not is_pending_resumable_error(exc.value)
    assert "cancelled" in describe_api_error(exc.value)


async def test_request_id_is_persisted_before_the_post_and_reused_only_without_a_response(monkeypatch):
    monkeypatch.setattr("lqh.client.asyncio.sleep", AsyncMock())
    client = _client()
    submitted: list[str] = []
    order: list[str] = []

    async def on_submitting(rid: str) -> None:
        submitted.append(rid)
        order.append("persist")

    async def create(*a: Any, **k: Any) -> Any:
        order.append("post:" + k["extra_headers"]["X-LQH-Request-Id"])
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    # Connection error (no response) -> same id; HTTP error -> new id; success ends it.
    script: list[Any] = [_conn_error(), _status_error(503), _raw(200, _completion("ok"))]
    client.chat.completions.with_raw_response.create = AsyncMock(side_effect=create)  # type: ignore[method-assign]
    client.get = AsyncMock()  # type: ignore[method-assign]
    resp = await chat_with_retry(client, async_mode=True, max_retries=3,
                                 async_hooks=AsyncCompletionHooks(on_submitting=on_submitting),
                                 model="orchestration:15", messages=[])
    assert resp.choices[0].message.content == "ok"
    assert all(s.startswith("lqhr_") for s in submitted)
    assert submitted[0] == submitted[1], "no response: the retry must reuse the id so the server can attach"
    assert submitted[2] != submitted[1], "an HTTP error proves the server rejected it: rotate"
    assert order[0] == "persist" and order[1].startswith("post:" + submitted[0])

    # A caller-supplied request id (from a persisted record) is used first.
    script = [_raw(200, _completion("attached"))]
    submitted.clear()
    await chat_with_retry(client, async_mode=True, request_id="lqhr_persisted0001",
                          async_hooks=AsyncCompletionHooks(on_submitting=on_submitting),
                          model="orchestration:15", messages=[])
    assert submitted == ["lqhr_persisted0001"]
