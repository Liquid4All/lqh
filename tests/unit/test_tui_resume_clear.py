"""Tests for the TUI /clear, /resume, and interrupted-resume flows.

Flipped from the Phase 0 characterization suite (see PERSISTENCY_PLAN.md):
/clear and /resume now refresh the ephemeral project context, /resume
resolves selections positionally, and session lifecycle states are
marked/repaired.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lqh.session import Session, sessions_dir
from lqh.tui.app import LqhApp


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
    return instance


async def test_clear_starts_fresh_session(app: LqhApp) -> None:
    old_id = app._session.id
    app._session.add_message({"role": "user", "content": "before clear"})

    await app._handle_command("/clear")

    assert app._session.id != old_id
    assert app._session.messages == []
    assert app._agent is not None
    assert app._agent.session is app._session


async def test_clear_marks_old_session_completed(app: LqhApp) -> None:
    app._session.add_message({"role": "user", "content": "before clear"})

    await app._handle_command("/clear")

    listed = Session.list_sessions(app.project_dir)
    assert listed[0]["state"] == "completed"


async def test_clear_refreshes_project_context(app: LqhApp) -> None:
    """Flipped from Phase 0: /clear re-runs the same context preparation
    as startup — the new conversation immediately sees the current spec,
    as an ephemeral prefix (nothing persisted)."""
    (app.project_dir / "SPEC.md").write_text("# spec\nimportant content\n")

    await app._handle_command("/clear")

    injected = [m["content"] for m in app._agent.context_messages]
    assert any("important content" in c for c in injected)
    assert app._agent.session.messages == []


def _write_session(project_dir: Path, created_at: str, first_msg: str) -> Session:
    session = Session.create(project_dir)
    session.created_at = created_at
    session.add_message({"role": "user", "content": first_msg})
    return session


def _option_for(index: int, listed: list[dict]) -> str:
    info = listed[index]
    return f"{index + 1}. {info.get('created_at', '?')} - {info.get('preview', '(empty)')[:60]}"


async def test_resume_loads_selected_session(app: LqhApp) -> None:
    _write_session(app.project_dir, "2026-01-01T00:00:00+00:00", "old work")
    newer = _write_session(
        app.project_dir, "2026-06-01T00:00:00+00:00", "recent work"
    )

    listed = Session.list_sessions(app.project_dir)
    app._wait_for_user_response = AsyncMock(  # type: ignore[method-assign]
        return_value=_option_for(0, listed)
    )

    await app._do_resume()

    assert app._session.id == newer.id
    assert [m["content"] for m in app._session.messages] == ["recent work"]
    assert app._session.state == "active"


async def test_resume_with_identical_previews_resolves_positionally(
    app: LqhApp,
) -> None:
    """Flipped from Phase 0: options are index-prefixed, so two sessions
    with identical timestamp+preview are still distinguishable and the
    user's actual pick wins."""
    for _ in range(2):
        _write_session(
            app.project_dir, "2026-03-01T00:00:00+00:00", "duplicate task"
        )

    listed = Session.list_sessions(app.project_dir)
    assert len(listed) == 2
    app._wait_for_user_response = AsyncMock(  # type: ignore[method-assign]
        return_value=_option_for(1, listed)
    )

    await app._do_resume()

    assert app._session.id == listed[1]["id"]


async def test_resume_unmatched_selection_resumes_nothing(app: LqhApp) -> None:
    _write_session(app.project_dir, "2026-01-01T00:00:00+00:00", "old work")
    before = app._session.id
    app._wait_for_user_response = AsyncMock(  # type: ignore[method-assign]
        return_value="something the options never contained"
    )

    await app._do_resume()

    assert app._session.id == before


async def test_resume_refreshes_context_without_touching_history(
    app: LqhApp,
) -> None:
    """Flipped from Phase 0: the resumed conversation is restored verbatim,
    and the *current* project state (spec edited since) arrives as the
    agent's ephemeral context prefix."""
    session = _write_session(
        app.project_dir, "2026-01-01T00:00:00+00:00", "old work"
    )
    (app.project_dir / "SPEC.md").write_text("# spec\nedited yesterday\n")

    listed = Session.list_sessions(app.project_dir)
    app._wait_for_user_response = AsyncMock(  # type: ignore[method-assign]
        return_value=_option_for(0, listed)
    )

    await app._do_resume()

    assert app._session.id == session.id
    # Stored history: verbatim, no injected turns.
    assert [m["content"] for m in app._session.messages] == ["old work"]
    # Fresh context: ephemeral prefix on the agent.
    contents = [m["content"] for m in app._agent.context_messages]
    assert any("edited yesterday" in c for c in contents)


async def test_offer_interrupted_resume_restores_session(app: LqhApp) -> None:
    interrupted = _write_session(
        app.project_dir, "2026-05-01T00:00:00+00:00", "crashed work"
    )
    meta_path = sessions_dir(app.project_dir) / interrupted.id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["state"] = "interrupted"
    meta_path.write_text(json.dumps(meta))

    app._wait_for_user_response = AsyncMock(  # type: ignore[method-assign]
        return_value="Resume interrupted session: crashed work "
        f"({meta['updated_at']})"
    )

    await app._offer_interrupted_resume()

    assert app._session.id == interrupted.id
    assert app._session.state == "active"
    assert [m["content"] for m in app._session.messages] == ["crashed work"]


async def test_offer_interrupted_resume_declines_cleanly(app: LqhApp) -> None:
    interrupted = _write_session(
        app.project_dir, "2026-05-01T00:00:00+00:00", "crashed work"
    )
    meta_path = sessions_dir(app.project_dir) / interrupted.id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["state"] = "interrupted"
    meta_path.write_text(json.dumps(meta))

    before = app._session.id
    app._wait_for_user_response = AsyncMock(  # type: ignore[method-assign]
        return_value="Start a new session"
    )

    await app._offer_interrupted_resume()

    assert app._session.id == before


async def test_startup_refresh_syncs_remote_state_before_signals(
    app: LqhApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-shot startup scan runs BEFORE context/signals are built, so
    a cloud job that went terminal while LQH was closed is signaled as
    finished — not reported as still running from stale cloud_state.json."""
    from lqh.signals import load_seen_states, record_seen_states

    (app.project_dir / "SPEC.md").write_text("# spec\n")
    run = app.project_dir / "runs" / "cloud_sft"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"type": "sft"}))
    (run / "remote_job.json").write_text(
        json.dumps({"job_id": "j1", "backend": "cloud", "remote_name": "lqh-cloud"})
    )
    # Stale on-disk state: still "running" from before the CLI closed.
    (run / "cloud_state.json").write_text(
        json.dumps({"job_id": "j1", "status": "running"})
    )
    record_seen_states(app.project_dir, {"cloud_sft": "running"})

    async def fake_scan(manager):
        # Simulates sync_progress: the backend reports the job finished,
        # and the scan persists that to cloud_state.json.
        (run / "cloud_state.json").write_text(
            json.dumps({"job_id": "j1", "status": "completed"})
        )
        return [("cloud_sft", "completed", None, "lqh-cloud")]

    monkeypatch.setattr(app._supervisor, "scan_jobs", fake_scan)
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: None)
    app._agent = app._create_agent()

    await app._refresh_startup_state()
    await app._agent.prepare_context()

    contents = [m["content"] for m in app._agent.context_messages]
    signal_blocks = [c for c in contents if c.startswith("⚡ Attention signals")]
    assert len(signal_blocks) == 1
    assert "runs/cloud_sft → completed" in signal_blocks[0]
    # Logged out without a cache: unavailability is also signaled.
    assert "cloud state is unavailable" in signal_blocks[0]
    # Recorded as seen: the next open stays quiet about this run.
    assert load_seen_states(app.project_dir)["cloud_sft"] == "completed"


async def test_clear_and_resume_retain_one_shot_signals(
    app: LqhApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """finished-while-away signals consume the job_seen baseline when the
    TUI records it at startup — /clear (and /resume) must re-inject the
    same signals, not recompute an already-consumed diff into silence."""
    from lqh.signals import record_seen_states

    (app.project_dir / "SPEC.md").write_text("# spec\n")
    run = app.project_dir / "runs" / "sft_v1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"type": "sft"}))
    (run / "progress.jsonl").write_text(json.dumps({"status": "completed"}) + "\n")
    record_seen_states(app.project_dir, {"sft_v1": "running"})

    async def fake_scan(manager):
        return []

    monkeypatch.setattr(app._supervisor, "scan_jobs", fake_scan)
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: None)
    app._agent = app._create_agent()
    await app._refresh_startup_state()
    await app._agent.prepare_context()

    def signal_block(agent) -> str:
        blocks = [
            m["content"] for m in agent.context_messages
            if m["content"].startswith("⚡ Attention signals")
        ]
        return blocks[0] if blocks else ""

    assert "runs/sft_v1 → completed" in signal_block(app._agent)

    await app._handle_command("/clear")
    assert "runs/sft_v1 → completed" in signal_block(app._agent)


async def test_failed_startup_refresh_is_flagged(
    app: LqhApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    (app.project_dir / "SPEC.md").write_text("# spec\n")
    run = app.project_dir / "runs" / "sft_v1"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")
    (run / "pid").write_text("1")  # "alive" per kill(1, 0) → shown running

    async def failing_scan(manager):
        raise TimeoutError("remote unreachable")

    monkeypatch.setattr(app._supervisor, "scan_jobs", failing_scan)
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: None)
    app._agent = app._create_agent()
    await app._refresh_startup_state()
    await app._agent.prepare_context()

    contents = [m["content"] for m in app._agent.context_messages]
    block = [c for c in contents if c.startswith("⚡ Attention signals")][0]
    assert "startup job-state refresh failed" in block


async def test_offer_interrupted_resume_noop_without_interrupted(
    app: LqhApp,
) -> None:
    _write_session(app.project_dir, "2026-05-01T00:00:00+00:00", "done work")
    app._session.mark_state("completed")
    responder = AsyncMock()
    app._wait_for_user_response = responder  # type: ignore[method-assign]

    await app._offer_interrupted_resume()

    responder.assert_not_awaited()