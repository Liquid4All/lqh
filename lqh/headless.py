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

from dataclasses import dataclass
from pathlib import Path


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
