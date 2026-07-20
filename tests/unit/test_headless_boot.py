"""headless_boot: the shared identity/copy/session startup contract."""

from __future__ import annotations

import shutil
from pathlib import Path

from lqh.headless import headless_boot
from lqh.project_identity import ensure_identity
from lqh.session import Session


def test_fresh_dir_gets_identity(tmp_path: Path) -> None:
    boot = headless_boot(tmp_path)
    assert boot.identity_error is None
    assert boot.copy_status == "same"
    assert boot.identity is not None and boot.identity["project_id"]
    assert (tmp_path / ".lqh" / "project.json").exists()


def test_corrupt_identity_surfaced_not_replaced(tmp_path: Path) -> None:
    ensure_identity(tmp_path)
    identity_path = tmp_path / ".lqh" / "project.json"
    identity_path.write_text("garbage")
    boot = headless_boot(tmp_path)
    assert boot.identity_error is not None
    assert boot.identity is None
    # Never auto-replaced.
    assert identity_path.read_text() == "garbage"


def test_copy_detected(tmp_path: Path) -> None:
    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    copy = tmp_path / "proj_copy"
    shutil.copytree(original, copy)
    boot = headless_boot(copy)
    assert boot.identity_error is None
    assert boot.copy_status == "copied"


def test_repair_sessions_toggle(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(
        Session, "repair_states", classmethod(lambda cls, p: calls.append(p))
    )
    headless_boot(tmp_path, repair_sessions=False)
    assert calls == []
    headless_boot(tmp_path)
    assert calls == [tmp_path]
