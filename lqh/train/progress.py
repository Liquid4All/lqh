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
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lqh.progress import (
    RUN_ATTEMPT_ENV,
    append_jsonl as _append_jsonl,
    emit_sentinel as _emit_sentinel,
    format_event_oneline,
    read_jsonl_tail,
)


def begin_run_attempt(run_dir: Path) -> str:
    """Start (or inherit) one append-only progress attempt."""
    attempt_id = os.environ.get(RUN_ATTEMPT_ENV) or uuid.uuid4().hex
    os.environ[RUN_ATTEMPT_ENV] = attempt_id
    write_status(run_dir, "running", extra={"attempt_id": attempt_id})
    return attempt_id


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
    emit_cloud: bool = True,
) -> None:
    """Append a progress line to ``progress.jsonl`` (and mirror to stdout
    as an LQH_EVENT_JSON sentinel when running in a cloud sandbox)."""
    entry: dict[str, Any] = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if attempt_id := os.environ.get(RUN_ATTEMPT_ENV):
        entry["attempt_id"] = attempt_id
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
    if emit_cloud:
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
    if attempt_id := os.environ.get(RUN_ATTEMPT_ENV):
        entry["attempt_id"] = attempt_id
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
    error_path: Path | None = None,
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
        if error_path is not None and error_path.exists():
            try:
                payload = json.loads(error_path.read_text())
                message = payload.get("error", "upstream work failed")
            except Exception:
                message = "upstream work failed"
            raise RuntimeError(str(message))
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for {path} after {timeout}s")


# ---------------------------------------------------------------------------
# Reading (main-process side)
# ---------------------------------------------------------------------------


def read_progress(run_dir: Path, last_n: int = 10) -> list[dict[str, Any]]:
    """Read the last *last_n* lines of ``progress.jsonl``.

    Returns an empty list if the file does not exist yet.
    """
    return read_jsonl_tail(run_dir / "progress.jsonl", last_n=last_n)


def read_latest_progress(run_dir: Path) -> dict[str, Any] | None:
    """Return the most recent progress entry, or ``None``."""
    entries = read_progress(run_dir, last_n=1)
    return entries[0] if entries else None


def read_latest_status(run_dir: Path) -> dict[str, Any] | None:
    """Return the newest status row even if later telemetry was appended."""
    for row in reversed(read_progress(run_dir, last_n=4096)):
        if "status" in row:
            return row
    return None


def read_current_attempt_id(run_dir: Path) -> str | None:
    """Return the active attempt tag from recent status or v1 rows.

    Unlike the startup status row, every current v1 event carries this tag, so
    discovery does not degrade after a fixed number of progress updates.
    """
    for row in reversed(read_progress(run_dir, last_n=64)):
        value = row.get("attempt_id")
        if isinstance(value, str) and value:
            return value
    return None


def read_latest_metrics(run_dir: Path) -> dict[str, Any] | None:
    """Return the newest row carrying trainer metrics.

    Trainer v1 and legacy rows may both carry these fields. Searching by
    content keeps callers independent of which transport produced the newest
    metric-bearing observation.
    """
    metric_keys = ("step", "loss", "lr", "epoch")
    for row in reversed(read_progress(run_dir, last_n=4096)):
        if any(key in row for key in metric_keys):
            return row
    return None


def format_progress_oneline(
    latest: dict[str, Any] | None,
    *,
    history: list[dict[str, Any]] | None = None,
    observed_at: float | None = None,
) -> tuple[str, int | None]:
    """Compact one-line progress for the status bar, plus a percent (or None).

    Distinct from the verbose tool-output renderer
    (``handlers._format_latest_sweep_progress``): this targets the single-line
    status bar, so it stays terse. Defensive about missing fields — a partial
    progress row never raises.

    Returns ``("", None)`` when there's nothing meaningful to show.
    """
    if not latest:
        return ("", None)

    # The common v1 event already carries an authoritative whole-job fraction.
    # Keep all legacy branches below so runs created by older clients remain
    # readable during upgrades and reconnects.
    if "overall_fraction" in latest:
        return format_event_oneline(
            latest,
            history=history or (),
            observed_at=observed_at,
        )

    def _pct(step: object, total: object) -> int | None:
        if isinstance(step, int) and isinstance(total, int) and total > 0:
            return max(0, min(100, round(step / total * 100)))
        return None

    phase = latest.get("phase")
    # Sweep child step takes precedence: it's the live inner-loop progress.
    if isinstance(phase, str) and phase == "sweep_config_progress":
        step = latest.get("child_step", latest.get("step"))
        total = latest.get("child_max_steps")
        idx = latest.get("config_index")
        n = latest.get("n_configs")
        config_pos = ""
        if isinstance(idx, int) and isinstance(n, int) and n > 0:
            config_pos = f"{idx + 1}/{n} · "
        pct = _pct(step, total)
        if isinstance(step, int) and isinstance(total, int) and total > 0:
            return (f"{config_pos}step {step}/{total} ({pct}%)", pct)
        if isinstance(step, int):
            return (f"{config_pos}step {step}", None)
        return (config_pos.rstrip(" ·"), None)

    # Plain (non-sweep) run.
    step = latest.get("step")
    total = latest.get("max_steps")
    pct = _pct(step, total)
    if isinstance(step, int) and isinstance(total, int) and total > 0:
        return (f"step {step}/{total} ({pct}%)", pct)
    if isinstance(step, int):
        epoch = latest.get("epoch")
        if isinstance(epoch, (int, float)):
            return (f"step {step} · epoch {epoch:.2f}", None)
        return (f"step {step}", None)
    return ("", None)
