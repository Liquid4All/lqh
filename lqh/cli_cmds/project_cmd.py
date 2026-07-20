"""`lqh project continue|fork` — resolve a copied project headlessly.

Thin wrappers over `record_continue_decision` / `fork_identity` so a
pure-headless harness is never forced into the TUI to unblock itself
after copying a project directory (CLI_PLAN §4.8).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_project(args: argparse.Namespace) -> int:
    action = args.project_command
    project_dir = Path.cwd()

    from lqh.headless import headless_boot

    boot = headless_boot(project_dir, repair_sessions=False)
    if boot.identity_error:
        print(
            "Project identity file is corrupt and will NOT be auto-replaced: "
            f"{boot.identity_error}\nRestore .lqh/project.json (e.g. from a "
            "backup or the original project directory) — do NOT delete it; "
            "a fresh identity would disconnect this project's cloud history.",
            file=sys.stderr,
        )
        return 5

    if boot.copy_status != "copied":
        if action == "continue":
            # Idempotent no-op so scripted retries are safe.
            print(json.dumps({
                "schema_version": 1, "ok": True, "action": "continue",
                "status": "no_copy_detected",
            }))
            return 0
        print(
            "This project is not an unresolved copy — forking would detach "
            "it from its cloud history for no reason. Nothing to do.",
            file=sys.stderr,
        )
        return 2

    from lqh.project_identity import (
        fork_identity,
        project_uuid,
        record_continue_decision,
    )

    try:
        if action == "continue":
            record_continue_decision(project_dir)
            payload = {
                "schema_version": 1, "ok": True, "action": "continue",
                "project_uuid": project_uuid(project_dir),
            }
        else:
            identity = fork_identity(project_dir)
            payload = {
                "schema_version": 1, "ok": True, "action": "fork",
                "project_uuid": identity.get("project_id"),
                "forked_from": identity.get("forked_from"),
            }
    except Exception as e:  # noqa: BLE001
        print(
            f"Failed to record the decision: {type(e).__name__}: {e}\n"
            "Fix the .lqh directory and retry; you will be asked again.",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(payload))
    return 0
