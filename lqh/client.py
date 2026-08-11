from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    RateLimitError,
)
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
        **({} if max_retries is None else {"max_retries": max_retries}),
    )
    _install_default_max_tokens(client)
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
# API; 504/408 are the same story told by an intermediary. 429 is absent
# because ``RateLimitError`` is handled on its own branch (it honours
# Retry-After). This must stay a SUBSET of the statuses
# ``lqh.tui.app._is_reconnectable_error`` accepts — that function decides the
# same question one layer up, and a status retried here but not there would
# strand a turn the outer ladder refuses to resume.
RETRYABLE_STATUS = frozenset({408, 500, 502, 503, 504})


def describe_api_error(exc: BaseException) -> str:
    """One short line naming what went wrong, for a user-facing notice."""
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
        return f"HTTP {exc.status_code}" + (f": {detail}" if detail else "")
    if isinstance(exc, APIConnectionError):
        return "connection to api.lqh.ai failed"
    return f"{type(exc).__name__}: {str(exc)[:160]}"


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
    **kwargs: object,
) -> ChatCompletion:
    """Call chat completions with retry logic for transient errors.

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
            resp: ChatCompletion = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
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
