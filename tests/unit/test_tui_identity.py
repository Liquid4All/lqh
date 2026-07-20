"""TUI-level tests for the Phase 3 identity flows.

Covers what the pure ``lqh.project_identity`` unit tests cannot: the
startup wiring in ``LqhApp.run()`` (auto-mode copy abort, identity-first
ordering, corrupt-identity surfacing) and the late ``/login`` migration
hook.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lqh.project_identity import ensure_identity
from lqh.tui.app import LqhApp


def _stub_lifecycle(instance: LqhApp, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Neutralize the heavyweight parts of run(): the prompt_toolkit app,
    network calls, and telemetry teardown. Returns the emit log."""
    emitted: list[str] = []

    async def _emit(text: str) -> None:
        emitted.append(str(text))

    async def _noop_async(*args, **kwargs) -> None:
        return None

    instance._emit = _emit  # type: ignore[method-assign]
    instance._invalidate = lambda: None  # type: ignore[method-assign]
    instance._start_application_task = (  # type: ignore[method-assign]
        lambda: asyncio.get_event_loop().create_task(asyncio.sleep(0))
    )
    instance._show_update_notice = _noop_async  # type: ignore[method-assign]
    instance._refresh_hf_status = _noop_async  # type: ignore[method-assign]
    instance._finish_telemetry = _noop_async  # type: ignore[method-assign]
    instance._telemetry_heartbeat = _noop_async  # type: ignore[method-assign]
    instance._start_telemetry_flush = lambda: None  # type: ignore[method-assign]
    return emitted


async def test_auto_mode_aborts_on_unresolved_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--auto must never silently reuse a copied project's identity (and
    with it the original's cloud namespace): it refuses to start and
    tells the user to decide interactively."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: None)

    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    copy = tmp_path / "proj_copy"
    shutil.copytree(original, copy)

    app = LqhApp(copy, auto_mode=True)
    emitted = _stub_lifecycle(app, monkeypatch)
    pipeline = AsyncMock()
    app._run_auto_mode = pipeline  # type: ignore[method-assign]

    await app.run()

    joined = "\n".join(emitted)
    assert "COPY of another project" in joined
    assert "Run lqh" in joined and "interactively" in joined
    pipeline.assert_not_awaited()
    # The identity was NOT silently continued or forked.
    identity = json.loads((copy / ".lqh" / "project.json").read_text())
    original_identity = json.loads(
        (original / ".lqh" / "project.json").read_text()
    )
    assert identity["project_id"] == original_identity["project_id"]
    assert identity["copy_decision"] is None
    assert identity["last_seen_path"] == str(original.resolve())


async def test_identity_created_before_any_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity creation is the FIRST unconditional startup step — a
    fresh auto-mode run that ends immediately still leaves one behind."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: None)

    project = tmp_path / "fresh"
    project.mkdir()
    app = LqhApp(project, auto_mode=True)
    _stub_lifecycle(app, monkeypatch)
    app._run_auto_mode = AsyncMock()  # type: ignore[method-assign]

    await app.run()

    assert (project / ".lqh" / "project.json").exists()


async def test_corrupt_identity_is_surfaced_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: None)

    project = tmp_path / "proj"
    (project / ".lqh").mkdir(parents=True)
    (project / ".lqh" / "project.json").write_text("{{{ corrupt")

    app = LqhApp(project, auto_mode=True)
    emitted = _stub_lifecycle(app, monkeypatch)
    app._run_auto_mode = AsyncMock()  # type: ignore[method-assign]

    await app.run()

    joined = "\n".join(emitted)
    assert "Project identity problem" in joined
    assert "NOT auto-replaced" in joined
    # The corrupt file is untouched.
    assert (project / ".lqh" / "project.json").read_text() == "{{{ corrupt"


async def test_late_login_runs_identity_migration_and_snapshot_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting logged out and running /login later must still perform
    the one-time cloud identity migration and refresh the snapshot —
    not leave the session on the legacy basename key."""
    monkeypatch.setenv("HOME", str(tmp_path))
    app = LqhApp(tmp_path)
    emitted: list[str] = []

    async def _emit(text: str) -> None:
        emitted.append(str(text))

    app._emit = _emit  # type: ignore[method-assign]
    app._invalidate = lambda: None  # type: ignore[method-assign]

    async def fake_login(on_user_code):
        return {"email": "user@example.com"}

    migrate = AsyncMock(return_value="stable-id")
    refresh = AsyncMock()
    monkeypatch.setattr("lqh.tui.app.login_device_code", fake_login)
    monkeypatch.setattr(
        "lqh.project_identity.migrate_cloud_identity", migrate
    )
    app._refresh_cloud_snapshot = refresh  # type: ignore[method-assign]

    await app._do_login()

    migrate.assert_awaited_once_with(tmp_path)
    refresh.assert_awaited_once()
    assert any("Logged in" in line for line in emitted)
