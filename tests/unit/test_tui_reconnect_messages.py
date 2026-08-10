"""What the TUI tells the user when a turn is interrupted by a 502.

Feedback #49 was not "retries are broken" — the retry ladder worked. It was
that the ladder said nothing useful: no indication of how to recover, and no
answer to whether the work the model had already done was lost.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from openai import APIStatusError

from lqh.tui.app import LqhApp


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LqhApp:
    monkeypatch.setenv("HOME", str(tmp_path))
    return LqhApp(tmp_path)


def _gateway_error() -> APIStatusError:
    req = httpx.Request("POST", "https://api.lqh.ai/v1/chat/completions")
    body = {"error": {"message": "upstream model did not respond within 240s"}}
    resp = httpx.Response(502, request=req, json=body)
    return APIStatusError(message=body["error"]["message"], response=resp, body=body["error"])


async def test_failed_reconnect_says_what_survived_and_how_to_resume(
    app: LqhApp, monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []
    saves: list[int] = []

    async def fake_emit(text: str) -> None:
        emitted.append(text)

    monkeypatch.setattr(app, "_emit", fake_emit)
    monkeypatch.setattr(app, "_save_session", lambda: saves.append(len(emitted)))
    # No real backoff sleeps.
    monkeypatch.setattr(app, "_reconnect_backoffs", (0.0, 0.0))

    async def always_fails() -> None:
        raise _gateway_error()

    ok = await app._run_agent_with_reconnect(always_fails, always_fails)

    assert ok is False
    blob = "\n".join(emitted)
    # The retry notices name the failure instead of just "Connection interrupted".
    assert "502" in blob
    # The final message answers both halves of the user's question.
    final = emitted[-1]
    assert "/reconnect" in final
    assert "Kept:" in final and "Lost:" in final
    # And the session is on disk BEFORE we claim it is.
    assert saves and saves[0] == len(emitted) - 1


async def test_a_recoverable_turn_emits_no_failure_notice(
    app: LqhApp, monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []

    async def fake_emit(text: str) -> None:
        emitted.append(text)

    monkeypatch.setattr(app, "_emit", fake_emit)
    monkeypatch.setattr(app, "_reconnect_backoffs", (0.0, 0.0))

    calls = {"n": 0}

    async def fails_once() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _gateway_error()

    ok = await app._run_agent_with_reconnect(fails_once, fails_once)

    assert ok is True
    assert app._pending_reconnect is None
    assert len(emitted) == 1
    # The renderer wraps and colours, so match on words, not on the sentence.
    assert "Resuming" in emitted[0]
    assert "saved" in emitted[0]


def test_tui_agent_uses_the_nested_retry_depth(app: LqhApp) -> None:
    """The TUI's own ladder is the outer one; the agent's must stay shallow."""
    from lqh.agent import ORCHESTRATION_API_RETRIES_NESTED
    from lqh.session import Session

    app._session = Session.create(app.project_dir)
    agent = app._create_agent()
    assert agent.api_retries == ORCHESTRATION_API_RETRIES_NESTED


def test_describe_error_handles_a_missing_exception(app: LqhApp) -> None:
    assert app._describe_error(None) == "unknown error"
    assert "502" in app._describe_error(_gateway_error())
