"""File-protocol helpers for subprocess → main-process communication.

All functions here operate on the filesystem only — no sockets, no pipes.
The subprocess writes; the main process reads.

Cloud sandboxes are a special case: the host can't tail a file inside
the sandbox's volume in real time. So when ``LQH_JOB_ID`` is set in the
process env (which the cloud runner injects via the sandbox env),
every progress and status write is ALSO echoed to stdout as a single
line prefixed with ``LQH_EVENT_JSON:``. The cloud runner's stdout
parser (``cloud.parseSentinel``) pulls these out, converts them to
SSE events, and persists them to ``cloud_job_events`` so the live
stream reaches the lqh CLI. Local and SSH-direct runs don't set
``LQH_JOB_ID``, so the sentinel path is a no-op there — file behavior
is unchanged.
"""

from __future__ import annotations

import json
import os
import sys
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
    """Append a progress line to ``progress.jsonl`` (and mirror to stdout
    as an LQH_EVENT_JSON sentinel when running in a cloud sandbox)."""
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

    # Emit sentinel first so the cloud SSE path is robust even if
    # the local file write later fails (disk full, permissions, etc).
    # Local + SSH runs treat this as a no-op.
    _emit_sentinel("progress", entry)
    _append_jsonl(run_dir / "progress.jsonl", entry)


def write_status(
    run_dir: Path,
    status: str,
    *,
    error: str | None = None,
    oom: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a terminal status line (``completed`` or ``failed``).

    ``oom=True`` stamps an ``oom`` flag on the event. This is the
    cooperative out-of-memory signal the backend relies on: exit 137 is
    SIGKILL for both preemption and OOM, so the backend cannot tell them
    apart from the exit code alone. Emitting this flag *before* the
    process dies lets the per-lease telemetry classify the lease as
    ``oom`` (vs ``preempted``) and lets batch auto-tuning self-heal
    (GPU_TYPE.md Design §3.5, §6).
    """
    entry: dict[str, Any] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        entry["error"] = error
    if oom:
        entry["oom"] = True
    if extra:
        entry.update(extra)

    _emit_sentinel("status", entry)
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
    """Append one JSON line. Uses ``_json_default`` to coerce non-JSON
    types (numpy/torch scalars, custom objects in trainer hooks) so
    a stray value in ``extra`` never crashes a training run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=_json_default) + "\n")


# The env var the cloud runner injects to mark "you are inside a
# cloud sandbox". Other backends (local, SSH-direct) leave it unset.
_CLOUD_ENV_MARKER = "LQH_JOB_ID"

# stdout sentinel prefix recognized by the backend's cloud runner
# (parseSentinel). Any line starting with this prefix is parsed as a
# structured event; everything else is treated as a log line.
_SENTINEL_PREFIX = "LQH_EVENT_JSON:"


def _emit_sentinel(kind: str, payload: dict[str, Any]) -> None:
    """If running in a cloud sandbox, echo one structured event to
    stdout for the cloud runner to pick up.

    Silent in every other context (local training, SSH-direct, tests).
    Designed to be a strict superset of the file-based protocol — the
    file is always written; the sentinel is the additional cloud-path
    signal.

    Failures are swallowed: a broken stdout (closed during shutdown,
    encoding error on a quirky payload) must NEVER take down a
    training run. The host-side parser tolerates malformed sentinels
    by logging them and moving on.
    """
    if not os.environ.get(_CLOUD_ENV_MARKER):
        return
    try:
        line = _SENTINEL_PREFIX + " " + json.dumps(
            {"kind": kind, "payload": payload},
            default=_json_default,
        )
        # Use print rather than sys.stdout.write so we get auto-flush
        # behavior on line buffering. flush=True forces it even on
        # block-buffered stdout (which is what the sandbox stdout pipe
        # typically is).
        print(line, flush=True)
    except Exception:
        # Don't let telemetry break the run.
        return


def _json_default(obj: Any) -> Any:
    """Fallback for objects json.dumps can't serialize natively.

    Mostly catches numpy / torch scalars that creep into the
    `extra` dict in DPO/SFT trainer hooks. Returns str(obj) as a
    last resort so the sentinel still parses on the host side.
    """
    # The most common offenders have .item() (numpy/torch 0-d arrays).
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(obj)
