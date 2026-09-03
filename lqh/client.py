from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai._constants import RAW_RESPONSE_HEADER
from openai._types import NoneType
from openai.types.chat import ChatCompletion

from lqh.config import default_api_base_url

# Default ceiling for completion length when a caller doesn't supply one.
# Pipelines call ``client.chat.completions.create(...)`` directly without
# max_tokens, and the API's own default is too low for thread translations
# and other longer outputs — generations get silently truncated. 8192 is
# comfortably above any realistic single-response need on api.lqh.ai.
# Override per-process via ``LQH_DEFAULT_MAX_TOKENS=N``.
DEFAULT_MAX_TOKENS = int(os.environ.get("LQH_DEFAULT_MAX_TOKENS", "16384"))

logger = logging.getLogger(__name__)

# Diagnostics: when set (via capture_api_metrics), every chat_with_retry attempt
# appends a record describing what happened. Used by the E2E benchmark harness
# to surface per-attempt timing / errors / finish_reason in scores.json so we
# can post-hoc diagnose hangs and mysterious timeouts without capturing raw
# transcripts.
_api_metrics_log: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("lqh_api_metrics_log", default=None)
)


def capture_api_metrics(log: list[dict[str, Any]]) -> contextvars.Token:
    """Enable per-attempt metrics capture for the current async context.

    Every ``chat_with_retry`` attempt (success or failure) inside the enclosing
    ``with``/context appends a record to ``log``. Each record contains keys:
      - attempt: int (0-based)
      - duration_s: float — how long this attempt actually took
      - error: str | None — exception class name if the attempt failed
      - error_msg: str | None — short exception message (truncated)
      - status_code: int | None — HTTP status if APIStatusError
      - finish_reason: str | None — OpenAI finish_reason on success
      - prompt_tokens: int | None
      - completion_tokens: int | None
      - tool_call_count: int — number of tool calls in the returned response
      - has_content: bool — whether the assistant message had any text

    Returns a Token that callers can pass to ``_api_metrics_log.reset(token)``
    when they want to stop capturing (usually in a finally).
    """
    return _api_metrics_log.set(log)


def _record_attempt(entry: dict[str, Any]) -> None:
    log = _api_metrics_log.get()
    if log is not None:
        log.append(entry)


def create_client(
    api_key: str,
    base_url: str | None = None,
    max_retries: int | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at api.lqh.ai.

    Installs a default ``max_tokens`` on chat completions so that callers
    that don't pass one (notably user-authored data-gen pipelines) don't
    get truncated by the API's lower default. Callers that pass an explicit
    ``max_tokens`` or ``max_completion_tokens`` are untouched.

    *max_retries* overrides the SDK's own retry layer, which by default
    replays every 5xx twice before raising. That layer is invisible: it
    cannot tell the user anything, and on a long reasoning turn each of its
    attempts can spend minutes, so a caller that wants to narrate its retries
    (``chat_with_retry`` with ``on_retry``) should pass 0 and own them.
    """
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url if base_url is not None else default_api_base_url(),
        timeout=300.0,
        # Opt in to the backend's keepalive heartbeat for slow completions.
        # The backend only ever commits an early 200 (whitespace keepalives,
        # errors in-band) for requests carrying this header, because only a
        # client with the in-band error check below can read that shape.
        default_headers={"X-LQH-Heartbeat": "1"},
        **({} if max_retries is None else {"max_retries": max_retries}),
    )
    _install_default_max_tokens(client)
    _install_inband_error_check(client)
    return client


def _install_default_max_tokens(client: AsyncOpenAI) -> None:
    """Patch ``client.chat.completions.create`` to inject a default max_tokens."""
    completions = client.chat.completions
    original = completions.create

    async def create_with_default(*args: Any, **kwargs: Any) -> Any:
        if "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS
        return await original(*args, **kwargs)

    completions.create = create_with_default  # type: ignore[method-assign]


def _install_inband_error_check(client: AsyncOpenAI) -> None:
    """Patch ``create`` to surface backend errors delivered inside a 200.

    Once the backend's keepalive heartbeat has committed a 200 (a slow call
    outlasting the edge proxy's read-timeout window), it can no longer change
    the HTTP status — an upstream failure then arrives as a 200 whose body is
    the OpenAI error envelope with the true status in ``error.code``. The SDK
    parses that leniently into a ChatCompletion with no choices and the
    envelope kept as an ``error`` extra field. Re-raise it as the matching
    APIStatusError subclass so every retry ladder above behaves exactly as if
    the real status had been on the wire.
    """
    completions = client.chat.completions
    original = completions.create

    async def create_checked(*args: Any, **kwargs: Any) -> Any:
        resp = await original(*args, **kwargs)
        err = _extract_inband_error(resp)
        if err is not None:
            raise _inband_status_error(client, err)
        return resp

    completions.create = create_checked  # type: ignore[method-assign]


def _extract_inband_error(resp: Any) -> dict[str, Any] | None:
    """Return the error envelope if *resp* is an in-band error, else None.

    Only an ``error`` WITHOUT choices is the in-band shape; some providers
    attach a partial ``error`` next to a real completion. Tolerates the SDK
    representing the unknown member as either a plain dict or a model.
    """
    try:
        err = getattr(resp, "error", None)
        if err is None or getattr(resp, "choices", None):
            return None
        if isinstance(err, dict):
            return err
        dump = getattr(err, "model_dump", None)
        if callable(dump):
            dumped = dump()
            if isinstance(dumped, dict):
                return dumped
    except Exception:  # detection must never break a healthy response
        logger.debug("in-band error detection failed", exc_info=True)
    return None


# Public status→exception mapping, mirroring the SDK's own dispatch. Kept
# explicit (rather than calling the SDK's private ``_make_status_error``) so
# an in-band 429 reliably becomes RateLimitError — which ``chat_with_retry``
# handles on its own Retry-After branch — across SDK versions.
_STATUS_ERROR_CLASSES: dict[int, type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}


def _inband_status_error(
    client: AsyncOpenAI,
    err: dict[str, Any],
    *,
    method: str = "POST",
    path: str = "/chat/completions",
) -> APIStatusError:
    """Build the APIStatusError an in-band error body would have been.

    *method* and *path* name the call this error stands in for. The synthetic
    request is what tells the user which endpoint failed
    (``describe_api_error``), so it must be the real one — a poll that fails
    is a GET on the completion, not the POST that submitted it.
    """
    code = err.get("code")
    status = code if isinstance(code, int) and 400 <= code <= 599 else 502
    message = str(err.get("message") or "upstream error")
    response = httpx.Response(
        status,
        request=httpx.Request(method, str(client.base_url).rstrip("/") + path),
    )
    cls = _STATUS_ERROR_CLASSES.get(
        status, InternalServerError if status >= 500 else APIStatusError
    )
    exc = cls(message, response=response, body=err)
    # Subclasses pin status_code as a class-level literal; make sure the
    # instance carries the actual embedded status (e.g. 524 on
    # InternalServerError) so retry ladders see the truth.
    exc.status_code = status  # type: ignore[misc]
    return exc


def is_transient_upstream_error(exc: object) -> bool:
    """True for a 400 that is really a transient upstream-model rejection.

    The backend proxies to pooled upstream models. When a provider transiently
    rejects a request (capacity, a spurious content flag, a hiccup handing the
    payload to the model) it surfaces as a 400 with a message like
    ``request rejected by upstream model`` rather than the 5xx you'd expect.
    Unlike a genuinely malformed-request 400, re-sending the same payload can
    succeed — often against a different pool member — so we treat this shape as
    transient and retry it.
    """
    if not isinstance(exc, APIStatusError) or exc.status_code != 400:
        return False
    parts = [str(getattr(exc, "message", "") or ""), str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    text = " ".join(parts).lower()
    return "rejected by upstream" in text or "upstream model" in text


# Awaited before each backoff sleep: (description, attempt, total_attempts,
# wait_seconds). ``attempt`` is 1-based and counts the attempt that just failed.
OnRetry = Callable[[str, int, int, float], Awaitable[None]] | None

# HTTP statuses worth re-sending the same payload for. 502 covers both an
# upstream model that ran out of time and a proxy that could not reach the
# API; 504/408 are the same story told by an intermediary. 524 is the edge
# proxy giving up on a silent origin — retryable, though a request that
# inherently needs longer than the edge window will 524 every time (the
# backend's keepalive heartbeat exists to prevent that). 429 is absent
# because ``RateLimitError`` is handled on its own branch (it honours
# Retry-After). This must stay a SUBSET of the statuses
# ``lqh.tui.app._is_reconnectable_error`` accepts — that function decides the
# same question one layer up, and a status retried here but not there would
# strand a turn the outer ladder refuses to resume.
RETRYABLE_STATUS = frozenset({408, 500, 502, 503, 504, 524})


def _endpoint_of_error(exc: BaseException) -> str:
    """``METHOD /path`` of the call that failed, or "" when unknown.

    The CLI talks to a dozen backend endpoints in one turn (completions, the
    async poll, jobs, artifacts, snapshots). A notice that says only
    "HTTP 502" leaves the user — and whoever reads their bug report — with no
    way to tell which one broke, so name it whenever the exception carries a
    request. Both openai's ``APIError`` and httpx's transport errors hang one
    off ``.request``.
    """
    try:
        # httpx's transport errors raise from ``.request`` when it was never
        # attached, so a plain getattr default is not enough here.
        req = getattr(exc, "request", None)
        if req is None:
            return ""
        return f"{req.method} {req.url.path}"
    except Exception:
        return ""


def describe_api_error(exc: BaseException) -> str:
    """One short line naming what went wrong, for a user-facing notice."""
    if isinstance(exc, CompletionLostError):
        return "the running response was lost on the server"
    if isinstance(exc, CompletionCancelledError):
        return "the running response was cancelled"
    if isinstance(exc, CompletionPollExhausted):
        return (
            f"lost contact with the running response after "
            f"{POLL_MAX_CONSECUTIVE_FAILURES} polls"
        )
    where = _endpoint_of_error(exc)
    if isinstance(exc, APIStatusError):
        detail = ""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            detail = str(body.get("message") or "")
        if not detail:
            detail = str(getattr(exc, "message", "") or "")
        detail = detail.strip().replace("\n", " ")
        if len(detail) > 160:
            detail = detail[:157] + "..."
        head = f"HTTP {exc.status_code}" + (f" on {where}" if where else "")
        return head + (f": {detail}" if detail else "")
    if isinstance(exc, APIConnectionError):
        return "connection to api.lqh.ai failed" + (f" on {where}" if where else "")
    base = f"{type(exc).__name__}: {str(exc)[:160]}"
    return base + (f" (on {where})" if where else "")


# ---------------------------------------------------------------------------
# Asynchronous completions (backend/CLI_API.md, "Async completions").
#
# Orchestration turns can reason for 30+ minutes. Instead of holding one HTTP
# connection open that long, the client opts in with ``X-LQH-Async: 1``: the
# server answers the normal body if the model finishes within its inline
# window, otherwise ``202 {id, status: running}`` while the generation keeps
# running server-side, and the client long-polls
# ``GET /chat/completions/{id}?wait=30`` until the result is ready. A dropped
# connection, a sleeping laptop or a CLI restart then costs nothing: the id
# is all that is needed to pick the turn back up.
# ---------------------------------------------------------------------------

ASYNC_HEADER = "X-LQH-Async"
# Client-chosen idempotency id for one submission. Persisted BEFORE the POST
# so a connection drop inside the server's inline window (the generation may
# already be running) is re-sent with the same id and attaches to that
# completion instead of starting a second paid one.
REQUEST_ID_HEADER = "X-LQH-Request-Id"
# Poll/attach responses name the completion's own state here; on an error
# status it tells a terminal result apart from a failing poll endpoint.
COMPLETION_STATUS_HEADER = "X-LQH-Completion-Status"
# Seconds the server may hold one poll open before answering 202 again.
POLL_WAIT_S = 30
# httpx read timeout for a poll: comfortably above POLL_WAIT_S so a full
# long-poll never trips the client side.
POLL_READ_TIMEOUT_S = 50.0
POLL_CONNECT_TIMEOUT_S = 10.0
# Delay between polls when the server's ``poll_after_ms`` hint is missing,
# and the cap applied to the hint.
POLL_AFTER_DEFAULT_S = 2.0
POLL_AFTER_MAX_S = 10.0
# Consecutive failed polls (timeouts, 5xx, connection errors) before giving
# up on THIS poll loop. The completion is still running server-side, so the
# caller can keep the id and resume later; ~20 attempts with a 30s backoff
# cap is roughly ten minutes of a fully unreachable API.
POLL_MAX_CONSECUTIVE_FAILURES = 20
POLL_BACKOFF_MAX_S = 30.0
CANCEL_TIMEOUT_S = 10.0


def new_request_id() -> str:
    """A fresh submission id (see REQUEST_ID_HEADER)."""
    return "lqhr_" + uuid.uuid4().hex


@dataclass
class AsyncCompletionHooks:
    """Callbacks for the async poll loop; every field is optional.

    ``on_submitting(request_id)`` fires before every POST — persist the id
    so a submission whose response never arrived can be re-sent with it.
    ``on_started(id)`` fires once when the server accepted the turn for
    background execution — persist the id there so the turn can be resumed.
    ``on_progress(tokens_so_far, elapsed_s)`` fires on every progress poll.
    ``on_lost(id)`` fires when the server no longer knows the id (expired,
    restarted); the request is about to be re-sent as a fresh turn.
    ``on_poll_retry`` narrates poll failures the way ``on_retry`` narrates
    request failures.
    """
    on_started: Callable[[str], Awaitable[None]] | None = None
    on_progress: Callable[[int, float], None] | None = None
    on_lost: Callable[[str], Awaitable[None]] | None = None
    on_poll_retry: OnRetry = None
    on_submitting: Callable[[str], Awaitable[None]] | None = None


class CompletionCancelledError(APIStatusError):
    """The running completion was cancelled (by another process/device).

    Terminal and NOT retryable: re-sending would pay for a turn the user
    stopped on purpose. Surfaces as a 410 so no ladder treats it as
    transient.
    """

    def __init__(self, completion_id: str, base_url: str) -> None:
        response = httpx.Response(410, request=httpx.Request("GET", base_url))
        super().__init__(
            f"completion {completion_id} was cancelled", response=response,
            body={"code": 410, "type": "completion_cancelled"},
        )
        self.status_code = 410  # type: ignore[misc]
        self.completion_id = completion_id


class CompletionLostError(APIConnectionError):
    """The server no longer has the running completion (404/410).

    Subclasses APIConnectionError so every existing retry / reconnect ladder
    treats it as transient: the right move is to re-send the request.
    """

    def __init__(self, completion_id: str, request: httpx.Request) -> None:
        super().__init__(
            message=f"completion {completion_id} was lost on the server", request=request,
        )
        self.completion_id = completion_id


class CompletionPollExhausted(APIConnectionError):
    """Too many consecutive poll failures; the completion may still be running.

    Callers should keep the completion id (it is still valid server-side)
    and resume polling once connectivity is back.
    """

    def __init__(self, completion_id: str, last_error: BaseException, request: httpx.Request) -> None:
        super().__init__(
            message=f"lost contact with completion {completion_id}: {last_error}", request=request,
        )
        self.completion_id = completion_id
        self.last_error = last_error


def _poll_url(completion_id: str) -> str:
    return f"/chat/completions/{completion_id}"


def _extract_async_handle(status_code: int, resp: Any) -> str | None:
    """Return the completion id when *resp* is the 202 'running' envelope.

    The SDK parses the envelope leniently into a ChatCompletion with the
    extra fields kept; a real completion has ``choices``. The status-code
    check is primary; the in-band shape (a heartbeat-committed 200 carrying
    the envelope) is the fallback.
    """
    try:
        if getattr(resp, "choices", None):
            return None
        cid = getattr(resp, "id", None)
        if not isinstance(cid, str) or not cid:
            return None
        if status_code == 202:
            return cid
        if getattr(resp, "status", None) == "running":
            return cid
    except Exception:
        logger.debug("async handle detection failed", exc_info=True)
    return None


def _completion_state(exc: BaseException) -> str | None:
    """The completion's own state named on an error response, if any."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    try:
        return headers.get(COMPLETION_STATUS_HEADER) or None
    except Exception:
        return None


def _error_type(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("type") or "")
    return ""


@dataclass
class _SubmissionState:
    """The request id in flight across retries of one chat_with_retry call.

    Kept while a POST produced no response at all (the server may be
    generating under it); dropped as soon as any response — 200, 202 or an
    HTTP error — proves the server has seen or rejected that submission.
    """
    request_id: str | None = None


async def _submit_async(
    client: AsyncOpenAI,
    kwargs: dict[str, Any],
    state: _SubmissionState,
    hooks: AsyncCompletionHooks | None,
) -> tuple[ChatCompletion | None, str | None]:
    """POST with the async headers. Returns (completion, None) or (None, id)."""
    extra = dict(kwargs.get("extra_headers") or {})
    extra[ASYNC_HEADER] = "1"
    if state.request_id is None:
        state.request_id = new_request_id()
    if hooks is not None and hooks.on_submitting is not None:
        await hooks.on_submitting(state.request_id)
    extra[REQUEST_ID_HEADER] = state.request_id
    call_kwargs = {k: v for k, v in kwargs.items() if k != "extra_headers"}
    try:
        raw = await client.chat.completions.with_raw_response.create(  # type: ignore[arg-type]
            **call_kwargs, extra_headers=extra,
        )
    except APIConnectionError:
        # No response: the server may or may not have started the turn.
        # Keep the id so the retry attaches instead of duplicating.
        raise
    except APIStatusError:
        state.request_id = None
        raise
    state.request_id = None
    parsed = raw.parse()
    err = _extract_inband_error(parsed)
    if err is not None:
        raise _inband_status_error(client, err)
    handle = _extract_async_handle(raw.status_code, parsed)
    if handle is not None:
        return None, handle
    return parsed, None


def _poll_options() -> dict[str, Any]:
    return {
        "params": {"wait": POLL_WAIT_S},
        "headers": {RAW_RESPONSE_HEADER: "true"},
        "timeout": httpx.Timeout(POLL_READ_TIMEOUT_S, connect=POLL_CONNECT_TIMEOUT_S),
    }


async def _poll_completion(
    client: AsyncOpenAI,
    completion_id: str,
    *,
    hooks: AsyncCompletionHooks | None,
    entry_base: dict[str, Any],
) -> tuple[ChatCompletion, int]:
    """Long-poll until the completion finishes. Returns (completion, polls).

    Raises ``CompletionLostError`` when the server no longer has the id,
    ``CompletionPollExhausted`` after ``POLL_MAX_CONSECUTIVE_FAILURES``
    consecutive transport/5xx failures, and any non-retryable status as-is.
    """
    hooks = hooks or AsyncCompletionHooks()
    request = httpx.Request("GET", str(client.base_url))
    failures = 0
    polls = 0
    last_exc: BaseException | None = None

    async def _failed(exc: BaseException, wait: float) -> None:
        nonlocal failures, last_exc
        failures += 1
        last_exc = exc
        _record_attempt({
            **entry_base, "phase": "poll", "completion_id": completion_id, "polls": polls,
            "error": type(exc).__name__, "error_msg": str(exc)[:200],
            "status_code": getattr(exc, "status_code", None),
        })
        if failures > POLL_MAX_CONSECUTIVE_FAILURES:
            raise CompletionPollExhausted(completion_id, exc, request)
        notify = hooks.on_poll_retry
        if notify is not None:
            try:
                await notify(describe_api_error(exc), failures, POLL_MAX_CONSECUTIVE_FAILURES, wait)
            except Exception:
                logger.debug("on_poll_retry callback failed", exc_info=True)
        await asyncio.sleep(wait)

    while True:
        polls += 1
        try:
            raw = await client.get(_poll_url(completion_id), cast_to=ChatCompletion, options=_poll_options())
        except (APITimeoutError, httpx.TimeoutException) as exc:
            await _failed(exc, min(2.0 ** (failures), POLL_BACKOFF_MAX_S))
            continue
        except APIStatusError as exc:
            # A status with the completion's own state attached is the
            # turn's terminal result, not the poll endpoint failing: hand it
            # to the caller's ladder (which may re-send, with a fresh
            # request id) and drop the record.
            state = _completion_state(exc)
            if state == "cancelled" or _error_type(exc) == "completion_cancelled":
                if hooks.on_lost is not None:
                    await hooks.on_lost(completion_id)
                raise CompletionCancelledError(completion_id, str(client.base_url)) from exc
            if state == "error":
                if hooks.on_lost is not None:
                    await hooks.on_lost(completion_id)
                raise
            if exc.status_code in (404, 410):
                if hooks.on_lost is not None:
                    await hooks.on_lost(completion_id)
                raise CompletionLostError(completion_id, request) from exc
            if isinstance(exc, RateLimitError):
                try:
                    wait = _parse_retry_after(exc)
                except Exception:
                    wait = None
                await _failed(exc, min(wait if wait is not None else 2.0, POLL_BACKOFF_MAX_S))
                continue
            if exc.status_code in RETRYABLE_STATUS:
                await _failed(exc, min(2.0 ** (failures), POLL_BACKOFF_MAX_S))
                continue
            raise
        except APIConnectionError as exc:
            await _failed(exc, min(2.0 ** (failures), POLL_BACKOFF_MAX_S))
            continue
        parsed = raw.parse()
        err = _extract_inband_error(parsed)
        if err is not None:
            raise _inband_status_error(
                client, err, method="GET", path=_poll_url(completion_id),
            )
        status_code = getattr(raw, "status_code", 200)
        if not getattr(parsed, "choices", None):
            inband = getattr(parsed, "status", None)
            if inband == "cancelled":
                if hooks.on_lost is not None:
                    await hooks.on_lost(completion_id)
                raise CompletionCancelledError(completion_id, str(client.base_url))
            if inband == "lost" or status_code in (404, 410):
                if hooks.on_lost is not None:
                    await hooks.on_lost(completion_id)
                raise CompletionLostError(completion_id, request)
        if _extract_async_handle(status_code, parsed) is not None:
            failures = 0
            tokens = getattr(parsed, "tokens_so_far", None)
            elapsed_ms = getattr(parsed, "elapsed_ms", None)
            if hooks.on_progress is not None:
                try:
                    hooks.on_progress(
                        int(tokens) if isinstance(tokens, (int, float)) else 0,
                        float(elapsed_ms) / 1000.0 if isinstance(elapsed_ms, (int, float)) else 0.0,
                    )
                except Exception:
                    logger.debug("on_progress callback failed", exc_info=True)
            hint = getattr(parsed, "poll_after_ms", None)
            delay = (
                float(hint) / 1000.0 if isinstance(hint, (int, float)) else POLL_AFTER_DEFAULT_S
            )
            await asyncio.sleep(max(0.0, min(delay, POLL_AFTER_MAX_S)))
            continue
        return parsed, polls


async def _run_async_attempt(
    client: AsyncOpenAI,
    kwargs: dict[str, Any],
    resume_id: str | None,
    state: _SubmissionState,
    hooks: AsyncCompletionHooks | None,
    entry: dict[str, Any],
) -> ChatCompletion:
    """One attempt in async mode: resume a pending id, else submit and poll."""
    if resume_id:
        try:
            resp, polls = await _poll_completion(client, resume_id, hooks=hooks, entry_base=entry)
        except CompletionLostError:
            logger.info("pending completion %s is gone; sending a fresh request", resume_id)
        else:
            entry.update({"phase": "poll", "completion_id": resume_id, "polls": polls, "resumed": True})
            return resp
    resp, handle = await _submit_async(client, kwargs, state, hooks)
    if resp is not None:
        entry.update({"phase": "post"})
        return resp
    assert handle is not None
    if hooks is not None and hooks.on_started is not None:
        await hooks.on_started(handle)
    resp, polls = await _poll_completion(client, handle, hooks=hooks, entry_base=entry)
    entry.update({"phase": "poll", "completion_id": handle, "polls": polls})
    return resp


async def cancel_completion(
    client: AsyncOpenAI, completion_id: str, *, timeout: float = CANCEL_TIMEOUT_S,
) -> bool:
    """Best-effort DELETE of a running completion. Never raises."""
    try:
        await asyncio.wait_for(
            client.delete(_poll_url(completion_id), cast_to=NoneType, options={"timeout": timeout}),
            timeout=timeout + 1.0,
        )
        return True
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("cancel of completion %s failed", completion_id, exc_info=True)
        return False


def is_pending_resumable_error(exc: BaseException) -> bool:
    """True when a failed async turn may still be running server-side.

    The caller should then KEEP its persisted completion id so the next
    attempt (a reconnect, /resume, or the next loop iteration) polls it
    instead of re-sending — re-sending would pay for the turn twice.
    """
    if isinstance(exc, (CompletionLostError, CompletionCancelledError)):
        return False
    if isinstance(exc, (CompletionPollExhausted, APIConnectionError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code in RETRYABLE_STATUS


MAX_RATE_LIMIT_WAITS = 5
# Ceiling on a single honoured Retry-After when nobody above us owns a
# deadline. Capping the wait COUNT alone does not bound the time: a server
# reporting Retry-After: 3600 would otherwise sleep five hours inside one call.
RATE_LIMIT_WAIT_CAP_S = 30.0


async def chat_with_retry(
    client: AsyncOpenAI,
    max_retries: int = 3,
    on_retry: OnRetry = None,
    rate_limits_are_free: bool = False,
    max_rate_limit_waits: int | None = MAX_RATE_LIMIT_WAITS,
    *,
    async_mode: bool = False,
    resume_id: str | None = None,
    request_id: str | None = None,
    async_hooks: AsyncCompletionHooks | None = None,
    **kwargs: object,
) -> ChatCompletion:
    """Call chat completions with retry logic for transient errors.

    *async_mode* sends ``X-LQH-Async: 1`` and, when the server answers 202,
    long-polls the completion to its result (see ``AsyncCompletionHooks``).
    Only callers prepared for that shape opt in — the agent's orchestration
    turns — which is why this is a per-call flag rather than a client-wide
    header. *resume_id* skips the POST on the first attempt and polls an
    already-running completion instead; if the server has lost it the
    request is sent fresh within the same attempt. *request_id* re-sends a
    submission whose response never arrived (see REQUEST_ID_HEADER): the
    server attaches it to the completion that POST started, if any.

    Retries on:
      - 429 (rate limit): honours Retry-After header, falls back to 2^attempt seconds.
      - ``RETRYABLE_STATUS`` / connection errors: exponential backoff up to
        *max_retries*.
    All other errors are raised immediately.

    *rate_limits_are_free* stops a 429 from spending an attempt. A rate limit
    is the server pacing us, not a failure of this request, so on a short
    ladder a brief 429 burst otherwise drops the call outright — the caller
    waited out the backoff and then gave up anyway. Off by default because a
    caller with no outer bound could then wait indefinitely; callers that own
    a deadline (see ``lqh/golden.py``) pass True.

    *max_rate_limit_waits* bounds how many such waits are allowed, and each is
    clamped to ``RATE_LIMIT_WAIT_CAP_S`` — together they stop a missing or
    misreported ``Retry-After`` from hanging a caller that has no deadline of
    its own. Pass ``None`` when you DO own one (an ``asyncio.timeout`` around
    the call): the deadline is then the only thing that decides when to give
    up, the server's number is honoured as sent, and a counter would
    otherwise abandon a request the server was about to serve with most of
    the budget unspent.

    *on_retry* is awaited before each backoff sleep with ``(description,
    attempt, total_attempts, wait_seconds)``. Retrying silently is what made
    these failures feel like a hang: a 502 on a long reasoning turn can burn
    minutes per attempt, and without this the caller has nothing to show for
    it. Callers that have no user to tell (pipelines, scoring) leave it None.

    When metrics capture is enabled via ``capture_api_metrics``, records each
    attempt (success or failure) with timing, error type, and response shape.
    """
    total_attempts = max_retries + 1
    submission = _SubmissionState(request_id=request_id)

    async def _notify(
        exc: BaseException, attempt: int, wait: float,
        *, total_override: int | None = None,
    ) -> None:
        if on_retry is None:
            return
        try:
            await on_retry(
                describe_api_error(exc), attempt + 1,
                total_override if total_override is not None else total_attempts,
                wait,
            )
        except Exception:  # a broken notifier must not eat the retry
            logger.debug("on_retry callback failed", exc_info=True)

    attempt = 0
    rate_limit_waits = 0
    while attempt <= max_retries:
        start = time.monotonic()
        entry: dict[str, Any] = {"attempt": attempt}
        try:
            if async_mode:
                resp: ChatCompletion = await _run_async_attempt(
                    client, dict(kwargs), resume_id if attempt == 0 else None,
                    submission, async_hooks, entry,
                )
            else:
                resp = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
            # Record successful attempt with response shape
            try:
                choice = resp.choices[0] if resp.choices else None
                msg = choice.message if choice else None
                entry.update({
                    "duration_s": round(time.monotonic() - start, 3),
                    "error": None,
                    "error_msg": None,
                    "status_code": None,
                    "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
                    "tool_call_count": len(msg.tool_calls or []) if msg else 0,
                    "has_content": bool(msg.content) if msg else False,
                })
            except Exception:
                pass
            _record_attempt(entry)
            return resp
        except CompletionLostError as exc:
            # The server dropped a running completion (restart, expiry). The
            # request is re-sent right away as its own attempt — nothing to
            # wait for, and the history is unchanged.
            entry.update({
                "duration_s": round(time.monotonic() - start, 3),
                "error": "CompletionLostError", "error_msg": str(exc)[:200],
                "status_code": None, "completion_id": exc.completion_id,
            })
            _record_attempt(entry)
            if attempt >= max_retries:
                raise
            logger.warning("chat_with_retry: completion lost on attempt %d, re-sending", attempt)
            await _notify(exc, attempt, 0.0)
        except CompletionPollExhausted as exc:
            # Connectivity, not the request, is what failed; the turn is
            # still running server-side. Spending more attempts here would
            # re-send and pay twice — surface it and let the caller resume
            # the poll once it can reach the API again.
            entry.update({
                "duration_s": round(time.monotonic() - start, 3),
                "error": "CompletionPollExhausted", "error_msg": str(exc)[:200],
                "status_code": None, "completion_id": exc.completion_id,
            })
            _record_attempt(entry)
            raise
        except RateLimitError as exc:
            entry.update({
                "duration_s": round(time.monotonic() - start, 3),
                "error": "RateLimitError", "error_msg": str(exc)[:200],
                "status_code": 429,
            })
            _record_attempt(entry)
            try:
                retry_after = _parse_retry_after(exc)
            except Exception:
                # Header parsing must never be what kills a request.
                retry_after = None
            if rate_limits_are_free:
                # The server is pacing us, not rejecting this request, so the
                # wait doesn't cost an attempt. Bounded by the caller's own
                # deadline; the counter is only here in case there isn't one.
                if (
                    max_rate_limit_waits is not None
                    and rate_limit_waits >= max_rate_limit_waits
                ):
                    raise
                rate_limit_waits += 1
                wait = (
                    retry_after if retry_after is not None
                    else 2 ** (rate_limit_waits - 1)
                )
                if max_rate_limit_waits is not None:
                    # Counted mode means nobody above us owns a deadline, so
                    # the wait itself has to be bounded too.
                    wait = min(wait, RATE_LIMIT_WAIT_CAP_S)
                logger.warning(
                    "chat_with_retry: 429 (wait %d, attempt not spent), sleeping %.1fs",
                    rate_limit_waits, wait,
                )
                # Reported as the wait number, not the attempt: the attempt is
                # deliberately frozen here, and showing "attempt 1 of 2" five
                # times in a row reads as a stuck ladder.
                await _notify(
                    exc, rate_limit_waits - 1,
                    wait,
                    total_override=max_rate_limit_waits or rate_limit_waits,
                )
                await asyncio.sleep(wait)
                continue
            if attempt >= max_retries:
                raise
            wait = retry_after if retry_after is not None else 2**attempt
            logger.warning("chat_with_retry: 429 on attempt %d, sleeping %.1fs", attempt, wait)
            await _notify(exc, attempt, wait)
            await asyncio.sleep(wait)
        except APIConnectionError as exc:
            entry.update({
                "duration_s": round(time.monotonic() - start, 3),
                "error": "APIConnectionError", "error_msg": str(exc)[:200],
                "status_code": None,
            })
            _record_attempt(entry)
            if attempt >= max_retries:
                raise
            wait = 2**attempt
            logger.warning("chat_with_retry: connection error on attempt %d, sleeping %.1fs", attempt, wait)
            await _notify(exc, attempt, wait)
            await asyncio.sleep(wait)
        except APIStatusError as exc:
            entry.update({
                "duration_s": round(time.monotonic() - start, 3),
                "error": f"APIStatusError({exc.status_code})",
                "error_msg": str(exc)[:200],
                "status_code": exc.status_code,
            })
            _record_attempt(entry)
            retryable = (
                exc.status_code in RETRYABLE_STATUS
                or is_transient_upstream_error(exc)
            )
            if retryable and attempt < max_retries:
                wait = 2**attempt
                logger.warning("chat_with_retry: %d on attempt %d, sleeping %.1fs", exc.status_code, attempt, wait)
                await _notify(exc, attempt, wait)
                await asyncio.sleep(wait)
            else:
                raise
        attempt += 1

    # Should be unreachable, but keeps the type checker happy.
    raise RuntimeError("Exceeded max retries")


def _parse_retry_after(exc: RateLimitError) -> float | None:
    """Try to extract a Retry-After value (in seconds) from the error."""
    headers = getattr(exc, "response", None)
    if headers is not None:
        raw = headers.headers.get("retry-after") or headers.headers.get("Retry-After")
        if raw is not None:
            try:
                return float(raw)
            except (ValueError, TypeError):
                return None
    return None
