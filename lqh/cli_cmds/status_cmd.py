"""`lqh status [--json]` — project state at a glance (CLI_PLAN §4.8.5).

Serializes the same attention signals the agent sees at startup
(lqh/signals.py) plus the run-directory scan. Local-first: remote/cloud
runs are polled best-effort with a bounded timeout; on failure the
signals say states may be stale instead of pretending freshness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def _gather(project_dir: Path) -> dict:
    from lqh.jobs import JobSupervisor
    from lqh.signals import collect_signals
    from lqh.snapshot import read_cached_snapshot
    from lqh.subprocess_manager import SubprocessManager

    supervisor = JobSupervisor(project_dir)
    jobs_refreshed = True
    try:
        snapshots = await asyncio.wait_for(
            supervisor.scan_jobs(SubprocessManager()), timeout=20.0
        )
    except Exception:
        snapshots = []
        jobs_refreshed = False

    run_states = {r: s for r, s, _, _ in snapshots if s != "unknown"}
    snapshot = read_cached_snapshot(project_dir)
    signals = collect_signals(
        project_dir,
        snapshot=snapshot,
        snapshot_fresh=False,
        run_states=run_states or None,
        jobs_refreshed=jobs_refreshed,
    )
    return {
        "schema_version": 1,
        "runs": [
            {"name": name, "state": state, "error": error, "remote": remote}
            for name, state, error, remote in snapshots
        ],
        "signals": [{"kind": s.kind, "text": s.text} for s in signals],
        "jobs_refreshed": jobs_refreshed,
    }


def cmd_status(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    payload = asyncio.run(_gather(project_dir))
    if args.json_out:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if not payload["runs"]:
        print("No runs.")
    else:
        width = max(len(r["name"]) for r in payload["runs"])
        for run in payload["runs"]:
            location = f" @{run['remote']}" if run["remote"] else ""
            error = f" — {run['error']}" if run["error"] else ""
            print(f"{run['name']:<{width}}  {run['state']}{location}{error}")
    if payload["signals"]:
        print()
        for signal in payload["signals"]:
            print(f"⚠ [{signal['kind']}] {signal['text']}")
    if not payload["jobs_refreshed"]:
        print("\n(remote run states could not be refreshed)", file=sys.stderr)
    return 0
