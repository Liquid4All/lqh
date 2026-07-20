"""Concurrency-safe reservation of dataset output names.

The no-overwrite guards (PERSISTENCY_PLAN.md R5) check for an existing
``data.parquet`` — but a plain existence check races: two CLIs can both
pass it and then generate into the same directory. This module adds an
atomic check-and-claim under a cross-process lock:

* ``claim_output`` refuses when finalized data exists (unless
  ``overwrite``) or when another LIVE process holds a claim on the same
  name. It then records this process's claim.
* ``release_output`` drops the claim when the work finishes (either
  outcome) — and a claim whose pid is dead is ignored anyway, so a
  crashed producer never blocks the name forever.

Claims protect the local-generation window only. Cloud jobs outlive the
submitting CLI (their pid dies while the job runs); their overlap is
governed by the submit-time existence guard plus the download-side
newest-submission-wins policy in the TUI finalizer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from lqh.fsio import atomic_write_json, file_lock

logger = logging.getLogger(__name__)

_CLAIM_FILE = ".lqh_claim.json"

# Names claimed by THIS process (pid claims cannot distinguish two
# concurrent tasks inside one CLI — e.g. two agent tool calls racing).
_LOCAL_CLAIMS: set[tuple[str, str]] = set()
_LOCAL_CLAIMS_LOCK = threading.Lock()


def _claims_lock(project_dir: Path) -> Path:
    return project_dir / ".lqh" / "dataset_claims.lock"


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def existing_output(project_dir: Path, output_dataset: str) -> str | None:
    """Name of the file that makes this LOGICAL output already exist.

    Not just ``data.parquet``: a dataset directory holding ANY parquet
    (train/validation splits, imports) or a ``manifest.json`` is spoken
    for — generating into it would mix unrelated artifacts and replace
    the directory's provenance.
    """
    dataset_dir = project_dir / "datasets" / output_dataset
    try:
        parquets = sorted(p.name for p in dataset_dir.glob("*.parquet"))
    except OSError:
        parquets = []
    if parquets:
        return parquets[0]
    if (dataset_dir / "manifest.json").exists():
        return "manifest.json"
    return None


def pending_cloud_job(project_dir: Path, output_dataset: str) -> str | None:
    """Run name of a pending cloud data-gen job targeting this dataset.

    The durable ``.lqh_data_gen.json`` marker exists from submission
    until finalization consumes it, so it identifies logical outputs
    that are already spoken for even though no data.parquet exists yet.
    """
    runs_dir = project_dir / "runs"
    if not runs_dir.is_dir():
        return None
    for marker_path in runs_dir.glob("*/.lqh_data_gen.json"):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if marker.get("output_dataset") == output_dataset:
            return marker_path.parent.name
    return None


# Backwards-compatible alias (pre-round-6 internal name).
_pending_cloud_job = pending_cloud_job


def overwrite_refusal(
    project_dir: Path, output_dataset: str, *, overwrite: bool = False
) -> str | None:
    """Read-only immutability check (no lock, no side effects).

    Used for fast refusal before permission prompts; ``claim_output``
    re-checks the same conditions atomically under the lock.

    ``overwrite=True`` here means the HANDLER already routed this
    request through the human confirmation gate (which covers both an
    existing logical output and a pending cloud job targeting it) —
    the model's bare ``overwrite=true`` argument is never passed down
    without that round-trip.
    """
    existing = existing_output(project_dir, output_dataset)
    if existing and not overwrite:
        return (
            f"datasets/{output_dataset}/{existing} already exists — "
            "refusing to overwrite an existing dataset output (generation "
            "is expensive). Either use a new versioned name (e.g. "
            f"'{output_dataset}_v2') to keep the old data, or — only "
            "after confirming with the user that the existing data "
            "should be destroyed — retry with overwrite=true."
        )
    if not overwrite:
        pending_run = pending_cloud_job(project_dir, output_dataset)
        if pending_run:
            return (
                f"datasets/{output_dataset}/ is already the target of the "
                f"pending cloud data-gen job '{pending_run}' — its output "
                "will land there when it finishes. Use a different output "
                "name, or wait/cancel that job first."
            )
    return None


def claim_output(
    project_dir: Path, output_dataset: str, *, overwrite: bool = False
) -> str | None:
    """Atomically claim ``datasets/<output_dataset>`` for this process.

    Returns None on success, or a human-readable refusal reason. The
    same process may re-claim its own name (the permission-prompt
    round-trip re-invokes the handler).
    """
    dataset_dir = project_dir / "datasets" / output_dataset
    local_key = (str(project_dir), output_dataset)
    try:
        with file_lock(_claims_lock(project_dir)):
            refusal = overwrite_refusal(
                project_dir, output_dataset, overwrite=overwrite
            )
            if refusal:
                return refusal
            # In-process guard: pid claims cannot tell two concurrent
            # tasks in the SAME CLI apart. (The permission-prompt
            # round-trip releases before re-invoking, so re-claiming
            # only collides on genuinely concurrent work.)
            with _LOCAL_CLAIMS_LOCK:
                if local_key in _LOCAL_CLAIMS:
                    return (
                        f"datasets/{output_dataset}/ is currently being "
                        "written by another task in this session — use a "
                        "different output name or wait for it to finish."
                    )
            claim_path = dataset_dir / _CLAIM_FILE
            if claim_path.exists():
                try:
                    claim = json.loads(claim_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    claim = {}
                holder = claim.get("pid")
                if holder != os.getpid() and _pid_alive(holder):
                    return (
                        f"datasets/{output_dataset}/ is currently being written "
                        f"by another lqh process (pid {holder}) — use a "
                        "different output name or wait for it to finish."
                    )
            dataset_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(claim_path, {
                "pid": os.getpid(),
                "claimed_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            })
            with _LOCAL_CLAIMS_LOCK:
                _LOCAL_CLAIMS.add(local_key)
            return None
    except OSError as exc:
        # Locking infrastructure failure: FAIL CLOSED. An unlocked
        # fallback would restore the exact cross-process overwrite race
        # this claim exists to remove — for immutable expensive outputs,
        # refusing until the reservation can actually be established is
        # the safe direction.
        logger.warning("dataset claim infrastructure failed", exc_info=True)
        return (
            f"cannot reserve datasets/{output_dataset}/ — the claim lock "
            f"is unavailable ({type(exc).__name__}: {exc}). Refusing to "
            "generate without overwrite protection; fix the .lqh/ "
            "directory permissions (or disk) and retry."
        )


def release_output(project_dir: Path, output_dataset: str) -> None:
    """Drop this process's claim (best-effort; called on every outcome)."""
    with _LOCAL_CLAIMS_LOCK:
        _LOCAL_CLAIMS.discard((str(project_dir), output_dataset))
    claim_path = project_dir / "datasets" / output_dataset / _CLAIM_FILE
    try:
        with file_lock(_claims_lock(project_dir)):
            try:
                claim = json.loads(claim_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            if claim.get("pid") == os.getpid():
                claim_path.unlink(missing_ok=True)
    except OSError:
        pass
