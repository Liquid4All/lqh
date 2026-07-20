"""Shared headless boot: the identity/copy/session contract (CLI_PLAN §4.8).

The non-interactive prefix of the TUI's startup sequence, extracted so
headless surfaces (`lqh tool call`, `lqh project`, later `lqh run`)
honor the same invariants:

1. ``ensure_identity`` first, unconditionally — no command may run cloud
   operations in a project without a stable identity. A corrupt identity
   file is surfaced, never silently replaced.
2. ``detect_copy`` next — an unresolved copy must block cloud/mutating
   work (the caller decides how; the TUI prompts, the CLI exits 5).
3. ``Session.repair_states`` — sessions left "active" by a dead process
   become "interrupted" so both surfaces see truthful session state.

This module must not import the TUI or telemetry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Advisory marker for "an agent loop is running in this project"
# (CLI_PLAN §7). Best-effort: concurrent READ-ONLY work alongside a live
# loop is supported; a second loop / concurrent mutating calls get a
# warning, not a hard failure.
_LOOP_MARKER = Path(".lqh") / "agent_loop.json"


def live_loop_owner(project_dir: Path) -> int | None:
    """Pid of a LIVE agent loop registered for this project (≠ us), or None."""
    from lqh.session import _pid_alive

    try:
        marker = json.loads((project_dir / _LOOP_MARKER).read_text())
    except (OSError, ValueError):
        return None
    pid = marker.get("pid")
    if pid == os.getpid():
        return None
    if _pid_alive(pid, marker.get("pid_start")):
        return int(pid)
    return None


def claim_loop(project_dir: Path) -> None:
    """Register this process as the project's running agent loop (best-effort)."""
    from lqh.fsio import atomic_write_json
    from lqh.session import _pid_start_time

    pid = os.getpid()
    try:
        atomic_write_json(project_dir / _LOOP_MARKER, {
            "pid": pid,
            "pid_start": _pid_start_time(pid),
        })
    except OSError:
        pass


def release_loop(project_dir: Path) -> None:
    """Drop the loop marker iff this process owns it (best-effort)."""
    path = project_dir / _LOOP_MARKER
    try:
        marker = json.loads(path.read_text())
        if marker.get("pid") == os.getpid():
            path.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


@dataclass(frozen=True)
class BootStatus:
    identity: dict | None  # identity record; None when identity_error is set
    copy_status: str  # "same" | "moved" | "copied" ("same" on identity error)
    identity_error: str | None  # "<ExcType>: <msg>", matching the TUI's format


def headless_boot(project_dir: Path, *, repair_sessions: bool = True) -> BootStatus:
    identity: dict | None = None
    copy_status = "same"
    identity_error: str | None = None
    try:
        from lqh.project_identity import detect_copy, ensure_identity

        identity, _ = ensure_identity(project_dir)
        copy_status = detect_copy(project_dir)
    except Exception as exc:
        identity_error = f"{type(exc).__name__}: {exc}"

    if repair_sessions:
        try:
            from lqh.session import Session

            Session.repair_states(project_dir)
        except Exception:
            pass

    return BootStatus(
        identity=identity,
        copy_status=copy_status,
        identity_error=identity_error,
    )
