"""Subprocess lifecycle management for training and inference runs.

Runs in the main lqh process.  Never imports torch or transformers.
Communicates with subprocesses exclusively via the filesystem.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lqh.train.progress import read_latest_progress, read_progress

__all__ = [
    "SubprocessManager",
    "RunStatus",
]


@dataclass
class RunStatus:
    """Snapshot of a training/inference run's state."""

    state: str  # "running" | "completed" | "failed" | "unknown"
    step: int | None = None
    loss: float | None = None
    lr: float | None = None
    epoch: float | None = None
    error: str | None = None
    eval_scores: dict[str, Any] = field(default_factory=dict)


class SubprocessManager:
    """Spawn and monitor ``python -m lqh.train`` / ``python -m lqh.infer``
    subprocesses.

    The manager is intentionally stateless between lqh restarts — it
    discovers run state from the filesystem (PID file + progress.jsonl).
    """

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(
        self,
        run_dir: Path,
        config: dict[str, Any],
        *,
        module: str = "lqh.train",
        project_dir: Path | None = None,
    ) -> int:
        """Write *config* to ``config.json`` and spawn the subprocess.

        Returns the subprocess PID.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")

        cwd = project_dir or run_dir.parent.parent  # runs/<name> → project root

        stdout_f = open(run_dir / "stdout.log", "w")
        stderr_f = open(run_dir / "stderr.log", "w")

        proc = subprocess.Popen(
            [sys.executable, "-m", module, str(config_path)],
            cwd=str(cwd),
            # Inherit env so CUDA_VISIBLE_DEVICES, HF_TOKEN, etc. pass through.
            env=os.environ.copy(),
            # Detach stdio so the subprocess doesn't block on the TUI.
            stdin=subprocess.DEVNULL,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,  # Don't die when TUI exits.
        )

        # Close our handles — the subprocess inherited them via fork.
        stdout_f.close()
        stderr_f.close()

        # The subprocess also writes its own PID (in __main__.py),
        # but we write it here too for immediate availability.
        (run_dir / "pid").write_text(str(proc.pid))

        return proc.pid

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def is_alive(self, run_dir: Path) -> bool:
        """Check whether the subprocess is still running."""
        pid = self._read_pid(run_dir)
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we can't signal it — treat as alive.
            return True

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def read_progress(self, run_dir: Path, last_n: int = 5) -> list[dict[str, Any]]:
        """Return the last *last_n* progress entries."""
        return read_progress(run_dir, last_n=last_n)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, run_dir: Path) -> RunStatus:
        """Build a composite status from the PID file and progress log."""
        latest = read_latest_progress(run_dir)

        # Terminal states from progress.jsonl
        if latest and latest.get("status") == "completed":
            return RunStatus(state="completed", **self._extract_metrics(latest))
        if latest and latest.get("status") == "failed":
            return RunStatus(
                state="failed",
                error=latest.get("error"),
                **self._extract_metrics(latest),
            )

        # Process-level check
        if self.is_alive(run_dir):
            return RunStatus(state="running", **self._extract_metrics(latest))

        # Process is gone but no terminal status in progress.jsonl
        if latest:
            return RunStatus(
                state="failed",
                error="Process exited without writing final status",
                **self._extract_metrics(latest),
            )

        return RunStatus(state="unknown")

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self, run_dir: Path, timeout: float = 10.0) -> bool:
        """Send SIGTERM, wait, then SIGKILL if needed.  Returns True if stopped."""
        pid = self._read_pid(run_dir)
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.5)

        # Still alive — force kill.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True

    # ------------------------------------------------------------------
    # List active runs
    # ------------------------------------------------------------------

    def list_runs(self, project_dir: Path) -> list[tuple[str, RunStatus]]:
        """Return ``(run_name, status)`` for every directory under ``runs/``."""
        runs_dir = project_dir / "runs"
        if not runs_dir.is_dir():
            return []
        results: list[tuple[str, RunStatus]] = []
        for entry in sorted(runs_dir.iterdir()):
            if entry.is_dir() and (entry / "config.json").exists():
                results.append((entry.name, self.get_status(entry)))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_pid(run_dir: Path) -> int | None:
        pid_file = run_dir / "pid"
        if not pid_file.exists():
            return None
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    @staticmethod
    def _extract_metrics(entry: dict[str, Any] | None) -> dict[str, Any]:
        if entry is None:
            return {}
        return {
            k: entry[k]
            for k in ("step", "loss", "lr", "epoch")
            if k in entry
        }
