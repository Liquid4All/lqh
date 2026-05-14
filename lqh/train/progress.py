"""File-protocol helpers for subprocess → main-process communication.

All functions here operate on the filesystem only — no sockets, no pipes.
The subprocess writes; the main process reads.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Writing (subprocess side)
# ---------------------------------------------------------------------------


def write_progress(
    run_dir: Path,
    *,
    step: int,
    loss: float | None = None,
    lr: float | None = None,
    epoch: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a progress line to ``progress.jsonl``."""
    entry: dict[str, Any] = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if loss is not None:
        entry["loss"] = loss
    if lr is not None:
        entry["lr"] = lr
    if epoch is not None:
        entry["epoch"] = epoch
    if extra:
        entry.update(extra)

    _append_jsonl(run_dir / "progress.jsonl", entry)


def write_status(
    run_dir: Path,
    status: str,
    *,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a terminal status line (``completed`` or ``failed``)."""
    entry: dict[str, Any] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        entry["error"] = error
    if extra:
        entry.update(extra)

    _append_jsonl(run_dir / "progress.jsonl", entry)


def write_eval_request(checkpoint_dir: Path) -> None:
    """Signal to the main process that predictions are ready to score."""
    payload = {
        "status": "ready",
        "predictions": "predictions.parquet",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (checkpoint_dir / "eval_request.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def write_iter_request(iter_dir: Path) -> None:
    """Signal that an on-policy iteration's predictions are ready."""
    payload = {
        "status": "ready",
        "predictions": "predictions.parquet",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (iter_dir / "iter_request.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# Waiting (subprocess side — blocks until main process writes a file)
# ---------------------------------------------------------------------------


def wait_for_file(
    path: Path,
    *,
    poll_interval: float = 2.0,
    timeout: float = 3600.0,
) -> Path:
    """Block until *path* exists and is non-empty, then return it.

    Raises ``TimeoutError`` if the file does not appear within *timeout*
    seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return path
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for {path} after {timeout}s")


# ---------------------------------------------------------------------------
# Reading (main-process side)
# ---------------------------------------------------------------------------


def read_progress(run_dir: Path, last_n: int = 10) -> list[dict[str, Any]]:
    """Read the last *last_n* lines of ``progress.jsonl``.

    Returns an empty list if the file does not exist yet.
    """
    progress_file = run_dir / "progress.jsonl"
    if not progress_file.exists():
        return []

    lines: list[str] = []
    try:
        with open(progress_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    except OSError:
        return []

    # Keep only last N
    lines = lines[-last_n:]
    result: list[dict[str, Any]] = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def read_latest_progress(run_dir: Path) -> dict[str, Any] | None:
    """Return the most recent progress entry, or ``None``."""
    entries = read_progress(run_dir, last_n=1)
    return entries[0] if entries else None


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
