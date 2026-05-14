from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
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
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at api.lqh.ai.

    Installs a default ``max_tokens`` on chat completions so that callers
    that don't pass one (notably user-authored data-gen pipelines) don't
    get truncated by the API's lower default. Callers that pass an explicit
    ``max_tokens`` or ``max_completion_tokens`` are untouched.
    """
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url if base_url is not None else default_api_base_url(),
        timeout=300.0,
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


async def chat_with_retry(
    client: AsyncOpenAI,
    max_retries: int = 3,
    **kwargs: object,
) -> ChatCompletion:
    """Call chat completions with retry logic for transient errors.

    Retries on:
      - 429 (rate limit): honours Retry-After header, falls back to 2^attempt seconds.
      - 502/503/connection errors: exponential backoff up to *max_retries*.
    All other errors are raised immediately.

    When metrics capture is enabled via ``capture_api_metrics``, records each
    attempt (success or failure) with timing, error type, and response shape.
    """
    for attempt in range(max_retries + 1):
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
            if attempt >= max_retries:
                raise
            retry_after = _parse_retry_after(exc)
            wait = retry_after if retry_after is not None else 2**attempt
            logger.warning("chat_with_retry: 429 on attempt %d, sleeping %.1fs", attempt, wait)
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
            await asyncio.sleep(wait)
        except APIStatusError as exc:
            entry.update({
                "duration_s": round(time.monotonic() - start, 3),
                "error": f"APIStatusError({exc.status_code})",
                "error_msg": str(exc)[:200],
                "status_code": exc.status_code,
            })
            _record_attempt(entry)
            if exc.status_code in (502, 503) and attempt < max_retries:
                wait = 2**attempt
                logger.warning("chat_with_retry: %d on attempt %d, sleeping %.1fs", exc.status_code, attempt, wait)
                await asyncio.sleep(wait)
            else:
                raise

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
