"""Tests for the TUI /compact command (feedback #98).

/compact runs the agent's ordinary non-destructive compaction pass on
demand: the raw transcript is untouched and only the working view shrinks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lqh.session import Session
from lqh.tui.app import LqhApp
from lqh.tui.commands import COMMANDS


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


def test_compact_is_a_registered_command() -> None:
    assert any(cmd.name == "/compact" for cmd in COMMANDS)


async def test_compact_runs_the_agent_pass(app: LqhApp) -> None:
    app._agent._compact_context = AsyncMock(return_value=True)

    await app._handle_command("/compact")

    app._agent._compact_context.assert_awaited_once_with(force=True)
    assert not any("Unknown command" in text for text in app._emitted)


async def test_compact_reports_when_nothing_was_compacted(app: LqhApp) -> None:
    app._agent._compact_context = AsyncMock(return_value=False)

    await app._handle_command("/compact")

    assert any("Nothing to compact" in text for text in app._emitted)


async def test_compact_keeps_the_raw_transcript(app: LqhApp) -> None:
    """The pass is derived-only — /compact must never drop stored history."""
    for i in range(8):
        app._session.add_message({"role": "user", "content": f"msg {i}"})
    raw_before = app._session.read_log()

    async def _fake_compact(*, force: bool = False) -> bool:
        app._session.set_compacted_view(
            "summary text",
            covers_to_seq=app._session.log_entries()[3][0],
            model="test",
        )
        return True

    app._agent._compact_context = _fake_compact  # type: ignore[assignment]

    await app._handle_command("/compact")

    assert app._session.read_log() == raw_before
    assert len(app._session.messages) < len(raw_before)
