"""`lqh feedback` — the headless counterpart of the TUI's /feedback.

Exactly one JSON result on stdout, the project's most recent conversation
attached by default, and a login requirement before anything is sent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lqh.cli import main
from lqh.session import Session


def _run(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []

    async def _send(message, context, session_id):
        calls.append((message, context, session_id))

    monkeypatch.setattr("lqh.auth.send_feedback", _send)
    monkeypatch.setattr("lqh.auth.get_token", lambda: "test-token")
    return calls


def _session_with_history(project_dir: Path) -> Session:
    session = Session.create(project_dir)
    session.add_message({"role": "user", "content": "train a model"})
    session.add_message({"role": "assistant", "content": "starting"})
    return session


def test_submits_message_with_latest_session(
    tmp_path: Path, monkeypatch, capsys, sent
) -> None:
    session = _session_with_history(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert _run(["lqh", "feedback", "the run crashed"], monkeypatch) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["status"] == "submitted"
    assert out["session_id"] == session.id

    message, context, session_id = sent[0]
    assert message == "the run crashed"
    assert session_id == session.id
    assert [m["content"] for m in context] == ["train a model", "starting"]


def test_no_context_sends_message_alone(
    tmp_path: Path, monkeypatch, capsys, sent
) -> None:
    _session_with_history(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert _run(["lqh", "feedback", "flag bug", "--no-context"], monkeypatch) == 0
    out = json.loads(capsys.readouterr().out)
    assert "session_id" not in out  # omitted when nothing was attached
    assert sent[0] == ("flag bug", [], None)


def test_session_prefix_selects_conversation(
    tmp_path: Path, monkeypatch, capsys, sent
) -> None:
    older = _session_with_history(tmp_path)
    _session_with_history(tmp_path)  # newer — would win by default

    assert _run(
        ["lqh", "feedback", "x", "--project", str(tmp_path),
         "--session", older.id[:8]],
        monkeypatch,
    ) == 0
    assert sent[0][2] == older.id


def test_unknown_session_is_a_usage_error(
    tmp_path: Path, monkeypatch, capsys, sent
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _run(["lqh", "feedback", "x", "--session", "nope"], monkeypatch) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert not sent


def test_message_from_stdin(tmp_path: Path, monkeypatch, capsys, sent) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("piped report\n"))
    assert _run(["lqh", "feedback", "-"], monkeypatch) == 0
    assert sent[0][0] == "piped report"


def test_empty_message_rejected(tmp_path: Path, monkeypatch, capsys, sent) -> None:
    monkeypatch.chdir(tmp_path)
    assert _run(["lqh", "feedback", "   "], monkeypatch) == 2
    assert not sent


def test_requires_login(tmp_path: Path, monkeypatch, capsys, sent) -> None:
    monkeypatch.setattr("lqh.auth.get_token", lambda: None)
    monkeypatch.chdir(tmp_path)
    assert _run(["lqh", "feedback", "x"], monkeypatch) == 4
    out = json.loads(capsys.readouterr().out)
    assert "lqh login" in out["message"]
    assert not sent


def test_message_and_message_file_conflict(
    tmp_path: Path, monkeypatch, capsys, sent
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("from file")
    assert _run(
        ["lqh", "feedback", "inline", "--message-file", str(tmp_path / "note.txt")],
        monkeypatch,
    ) == 2
    assert "not both" in json.loads(capsys.readouterr().out)["message"]
    assert not sent


def test_session_and_no_context_conflict(
    tmp_path: Path, monkeypatch, capsys, sent
) -> None:
    session = _session_with_history(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _run(
        ["lqh", "feedback", "x", "--session", session.id, "--no-context"],
        monkeypatch,
    ) == 2
    assert not sent
