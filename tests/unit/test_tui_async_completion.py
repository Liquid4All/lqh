"""TUI behaviour around server-side (async) orchestration turns."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lqh.session import Session
from lqh.tui.app import AgentInterrupted, LqhApp
from lqh.tui.status_bar import StatusBar


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LqhApp:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("lqh.auth.get_token", lambda: "test-token")
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: "test-token")
    instance = LqhApp(tmp_path)
    emitted: list[str] = []

    async def _emit(text: str) -> None:
        emitted.append(text)

    instance._emit = _emit  # type: ignore[method-assign]
    instance._emitted = emitted  # type: ignore[attr-defined]
    instance._invalidate = lambda: None  # type: ignore[method-assign]
    instance._session = Session.create(tmp_path)
    instance._agent = instance._create_agent()
    return instance


def _pending(agent) -> dict:
    return {"id": "lqhc_" + "a" * 32, "model": agent.orchestration_model, "kind": "turn",
            "started_at": "now", "turn_seq": agent.session.last_seq}


async def _interrupt(app: LqhApp, shutdown: bool) -> None:
    started = asyncio.Event()

    async def action() -> None:
        started.set()
        await asyncio.Event().wait()

    async def runner() -> None:
        await app._run_interruptible(action)

    task = asyncio.create_task(runner())
    await started.wait()
    app._interrupt_requested = True
    app._shutdown_requested = shutdown
    app._agent_task.cancel()
    with pytest.raises(AgentInterrupted):
        await task


async def test_ctrl_c_cancels_the_server_side_turn(app: LqhApp) -> None:
    app._agent.cancel_pending_completion = AsyncMock()
    await _interrupt(app, shutdown=False)
    app._agent.cancel_pending_completion.assert_awaited_once()


async def test_quit_keeps_the_server_side_turn_running(app: LqhApp) -> None:
    app._agent.cancel_pending_completion = AsyncMock()
    await _interrupt(app, shutdown=True)
    app._agent.cancel_pending_completion.assert_not_awaited()


async def test_resume_pending_completion_drives_the_agent(app: LqhApp) -> None:
    app._session.add_message({"role": "user", "content": "hi"})
    app._session.set_pending_completion(_pending(app._agent))
    app._agent.continue_after_interruption = AsyncMock()

    await app._resume_pending_completion()

    app._agent.continue_after_interruption.assert_awaited()
    assert any("still running" in text for text in app._emitted)
    assert app._session.messages[-1]["role"] == "user"  # no extra message appended


async def test_resume_pending_completion_is_a_noop_without_a_valid_record(app: LqhApp) -> None:
    app._agent.continue_after_interruption = AsyncMock()
    await app._resume_pending_completion()
    app._agent.continue_after_interruption.assert_not_awaited()

    # Stale record (history moved on) is dropped, not resumed.
    app._session.add_message({"role": "user", "content": "hi"})
    rec = _pending(app._agent)
    rec["turn_seq"] -= 1
    app._session.set_pending_completion(rec)
    app._agent._get_client = lambda: object()
    await app._resume_pending_completion()
    app._agent.continue_after_interruption.assert_not_awaited()
    assert app._session.pending_completion is None


def test_status_bar_shows_server_progress() -> None:
    bar = StatusBar()
    bar.start_spinning()
    bar.set_thinking_progress(41203, 552.0)
    text = "".join(t for _, t in bar.get_formatted_text())
    assert "thinking... 41,203 tok (9m12s)" in text
    bar.stop_spinning()
    bar.start_spinning()
    text = "".join(t for _, t in bar.get_formatted_text())
    assert "tok" not in text and "thinking..." in text


def test_interrupted_offer_mentions_the_running_response(project_dir: Path) -> None:
    session = Session.create(project_dir)
    session.add_message({"role": "user", "content": "hi"})
    session.set_pending_completion({"id": "lqhc_" + "b" * 32, "model": "m", "kind": "turn",
                                    "started_at": "now", "turn_seq": session.last_seq})
    rows = Session.list_sessions(project_dir)
    assert rows[0]["pending_completion"] is True


def test_reconnectable_errors_include_the_async_ones() -> None:
    import httpx
    from lqh.client import CompletionLostError, CompletionPollExhausted

    req = httpx.Request("GET", "https://api.lqh.ai/v1")
    assert LqhApp._is_reconnectable_error(CompletionLostError("lqhc_" + "c" * 32, req))
    assert LqhApp._is_reconnectable_error(CompletionPollExhausted("lqhc_" + "c" * 32, RuntimeError("x"), req))
