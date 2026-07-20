"""`lqh project continue|fork` — headless copy resolution."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lqh.cli_cmds.project_cmd import cmd_project
from lqh.project_identity import detect_copy, ensure_identity


def _ns(action: str) -> argparse.Namespace:
    return argparse.Namespace(command="project", project_command=action)


def _make_copy(tmp_path: Path) -> Path:
    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    copy = tmp_path / "proj_copy"
    shutil.copytree(original, copy)
    return copy


def test_continue_records_decision(tmp_path: Path, monkeypatch, capsys) -> None:
    copy = _make_copy(tmp_path)
    monkeypatch.chdir(copy)
    assert cmd_project(_ns("continue")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["action"] == "continue"
    assert payload["project_uuid"]
    assert detect_copy(copy) == "same"


def test_fork_mints_new_identity(tmp_path: Path, monkeypatch, capsys) -> None:
    original = tmp_path / "proj"
    copy = _make_copy(tmp_path)
    original_id = json.loads(
        (original / ".lqh" / "project.json").read_text()
    )["project_id"]
    monkeypatch.chdir(copy)
    assert cmd_project(_ns("fork")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "fork"
    assert payload["project_uuid"] != original_id
    assert payload["forked_from"] == original_id
    assert detect_copy(copy) == "same"


def test_continue_on_non_copy_is_noop(tmp_path: Path, monkeypatch, capsys) -> None:
    ensure_identity(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cmd_project(_ns("continue")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no_copy_detected"


def test_fork_on_non_copy_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    ensure_identity(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cmd_project(_ns("fork")) == 2
    assert "not an unresolved copy" in capsys.readouterr().err


def test_corrupt_identity_exit_5(tmp_path: Path, monkeypatch, capsys) -> None:
    ensure_identity(tmp_path)
    (tmp_path / ".lqh" / "project.json").write_text("garbage")
    monkeypatch.chdir(tmp_path)
    assert cmd_project(_ns("continue")) == 5
    assert "corrupt" in capsys.readouterr().err
