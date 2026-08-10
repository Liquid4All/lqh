"""Tests for `lqh --resume ID` and the resume hint printed on exit.

The flag adopts a prior conversation at startup instead of the fresh
session; the hint prints the id needed to get back in. Both reuse the
existing session machinery (Session.resolve_id / _adopt_session).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lqh.session import Session
from lqh.tui.app import LqhApp


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: str | None) -> LqhApp:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("lqh.auth.get_token", lambda: "test-token")
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: "test-token")
    instance = LqhApp(tmp_path, resume_session_id=resume)
    emitted: list[str] = []

    async def _emit(text: str) -> None:
        emitted.append(text)

    instance._emit = _emit  # type: ignore[method-assign]
    instance._emitted = emitted  # type: ignore[attr-defined]
    instance._invalidate = lambda: None  # type: ignore[method-assign]
    instance._session = Session.create(tmp_path)
    return instance


def _prior_session(project_dir: Path) -> Session:
    session = Session.create(project_dir)
    session.add_message({"role": "user", "content": "earlier question"})
    session.add_message({"role": "assistant", "content": "earlier answer"})
    return session


async def test_resume_flag_adopts_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _prior_session(tmp_path)
    app = _app(tmp_path, monkeypatch, prior.id)

    assert await app._resume_requested_session() is True
    assert app._session.id == prior.id
    assert app._agent is not None
    assert app._agent.session is app._session
    contents = [m.get("content") for m in app._agent.session.messages]
    assert contents == ["earlier question", "earlier answer"]
    # History is replayed into scrollback.
    joined = "".join(app._emitted)
    assert "earlier question" in joined
    assert f"Resumed session {prior.id[:8]}" in joined


async def test_resume_flag_accepts_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _prior_session(tmp_path)
    app = _app(tmp_path, monkeypatch, prior.id[:8])

    assert await app._resume_requested_session() is True
    assert app._session.id == prior.id


async def test_resume_flag_unknown_id_keeps_fresh_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prior_session(tmp_path)
    app = _app(tmp_path, monkeypatch, "zzzzzzzz")
    fresh_id = app._session.id

    assert await app._resume_requested_session() is False
    assert app._session.id == fresh_id
    assert "No conversation" in "".join(app._emitted)


async def test_resume_flag_refuses_session_owned_by_live_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _prior_session(tmp_path)
    app = _app(tmp_path, monkeypatch, prior.id)
    fresh_id = app._session.id
    monkeypatch.setattr(Session, "claim_active", lambda self: False)

    assert await app._resume_requested_session() is False
    assert app._session.id == fresh_id
    assert "active in another process" in "".join(app._emitted)


async def test_resume_flag_survives_corrupt_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Damaged meta.json must start a new conversation, not crash startup.

    Session.load int()s the token counts, so a non-numeric value raises
    ValueError — outside the FileNotFoundError/OSError family.
    """
    import json

    prior = _prior_session(tmp_path)
    meta_path = tmp_path / ".lqh" / "conversations" / prior.id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["prompt_tokens"] = "not-a-number"
    meta_path.write_text(json.dumps(meta))

    app = _app(tmp_path, monkeypatch, prior.id)
    fresh_id = app._session.id

    assert await app._resume_requested_session() is False
    assert app._session.id == fresh_id
    assert "ValueError" in "".join(app._emitted)


async def test_no_resume_flag_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prior_session(tmp_path)
    app = _app(tmp_path, monkeypatch, None)
    fresh_id = app._session.id

    assert await app._resume_requested_session() is False
    assert app._session.id == fresh_id
    assert app._emitted == []


async def test_startup_announces_and_holds_input_while_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay is announced before the startup refresh, not after it.

    The refresh (job scan + cloud snapshot) sits between the prompt appearing
    and the conversation being replayed; without a marker the user starts
    typing into what looks like an idle prompt.
    """
    prior = _prior_session(tmp_path)
    app = _app(tmp_path, monkeypatch, prior.id)
    locked_during_refresh: list[bool] = []

    async def _refresh() -> None:
        locked_during_refresh.append(app._processing)

    app._refresh_startup_state = _refresh  # type: ignore[method-assign]

    assert await app._restore_startup_state() is True

    joined = "".join(app._emitted)
    assert "Loading previous conversation" in joined
    assert joined.index("Loading previous conversation") < joined.index(
        "earlier question"
    )
    # Input is held for the whole window and released once history is on screen.
    assert locked_during_refresh == [True]
    assert app._processing is False


async def test_startup_without_resume_is_silent_and_unlocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch, None)

    async def _refresh() -> None:
        assert app._processing is False

    app._refresh_startup_state = _refresh  # type: ignore[method-assign]

    assert await app._restore_startup_state() is False
    assert app._emitted == []
    assert app._processing is False


async def test_startup_releases_input_when_resume_blows_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash in the startup refresh must not leave the prompt locked."""
    prior = _prior_session(tmp_path)
    app = _app(tmp_path, monkeypatch, prior.id)

    async def _refresh() -> None:
        raise RuntimeError("boom")

    app._refresh_startup_state = _refresh  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await app._restore_startup_state()
    assert app._processing is False


async def test_quitting_during_the_load_skips_the_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C while the conversation loads must not dump it on the way out.

    Once the application is gone _emit writes straight to stdout, so replaying
    then floods the user's real terminal after they asked to leave.
    """
    prior = _prior_session(tmp_path)
    app = _app(tmp_path, monkeypatch, prior.id)

    async def _refresh() -> None:
        app._shutdown_requested = True

    app._refresh_startup_state = _refresh  # type: ignore[method-assign]

    assert await app._restore_startup_state() is True
    joined = "".join(app._emitted)
    assert "earlier question" not in joined
    assert "earlier answer" not in joined


async def test_input_typed_while_held_is_refused_but_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enter during the load must not eat what the user typed.

    Goes through validate_and_handle (not _on_accept directly) because that
    is where prompt_toolkit resets the buffer unless the handler keeps it.
    """
    import asyncio

    from prompt_toolkit.buffer import Buffer

    app = _app(tmp_path, monkeypatch, None)
    app._lock_input()
    buffer = Buffer(accept_handler=app._on_accept)
    buffer.text = "hello"

    buffer.validate_and_handle()
    await asyncio.sleep(0)

    printed = "".join(app._emitted)
    assert "Please wait" in printed
    # Input held at startup has nothing to interrupt — don't offer the keys.
    assert "Esc" not in printed
    assert buffer.text == "hello"
    assert app._input_queue.empty()


async def test_wait_message_offers_interrupt_during_an_agent_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from prompt_toolkit.buffer import Buffer

    app = _app(tmp_path, monkeypatch, None)
    app._lock_input()
    turn = asyncio.create_task(asyncio.sleep(60))
    app._agent_task = turn
    try:
        buffer = Buffer(accept_handler=app._on_accept)
        buffer.text = "hello"
        buffer.validate_and_handle()
        await asyncio.sleep(0)
    finally:
        turn.cancel()

    assert "Esc / Ctrl+C to interrupt" in "".join(app._emitted)


async def test_resume_hint_prints_full_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch, None)
    app._session.add_message({"role": "user", "content": "some work"})

    await app._emit_resume_hint()

    printed = "".join(app._emitted)
    assert "lqh --resume" in printed
    assert app._session.id in printed


async def test_resume_hint_silent_for_empty_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch, None)

    await app._emit_resume_hint()

    assert app._emitted == []
