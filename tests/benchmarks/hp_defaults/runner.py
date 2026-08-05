"""Launching cell jobs, on LQH Cloud or on a local GPU.

One **job** is one cell's grid (or a chunk of it) run as a single
``lqh.train.sweep`` with ``eval_all`` on: the sweep trains each config and
judge-scores it in the same sandbox, and every result comes back on that
config's row in ``sweep_summary.json``.

Cloud is the real path — 48 cells fanned out concurrently is the difference
between a day and a month. Local exists so the whole pipeline can be smoke
-tested end to end without spending GPU budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lqh.subprocess_manager import SubprocessManager

logger = logging.getLogger(__name__)

# The cloud runner caps train_sft_sweep at 720 minutes and a timeout loses
# every config in the job, not just the one that ran over.
DEFAULT_CHUNK_SIZE = 6


@dataclass
class JobSpec:
    cell_id: str
    chunk_index: int
    run_name: str
    launch_config: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.cell_id}#{self.chunk_index}"


def build_launch_config(
    *,
    base_model: str,
    dataset_rel: str,
    eval_rel: str,
    scorer_rel: str,
    grid_override: list[dict],
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """The sweep payload for one cell chunk.

    Deliberately mirrors what ``handle_start_training`` builds, minus the
    hyperparameters the grid is going to override anyway, so the study
    measures the configuration the product actually ships.
    """
    from lqh.train import defaults

    recommended = defaults.recommended(run_type="sft", lora=True)
    training = recommended.training_config()
    # The grid owns these two.
    training.pop("learning_rate", None)
    training.pop("num_epochs", None)

    base_config: dict[str, Any] = {
        "type": "sft",
        "base_model": base_model,
        "dataset": dataset_rel,
        "eval_dataset": eval_rel,
        "scorer": scorer_rel,
        "max_new_tokens": max_new_tokens,
        # Per-config eval is what eval_all does; the ordinary checkpoint eval
        # would duplicate it at every save.
        "eval_on_checkpoints": False,
        "training": training,
        "lora": recommended.lora,
        "manifest": ["base_model", "dataset", "eval_dataset", "scorer"],
    }
    return {
        "type": "sweep",
        "base_config": base_config,
        "grid_override": grid_override,
        # The study needs a real metric for EVERY config, not just the winner.
        "eval_all": True,
        # ...which makes a separate eval of the winner pure duplication.
        "eval_best": False,
    }


# What the study needs back from a finished job. The sweep leaderboard is the
# only place per-config judge scores exist off the sandbox volume.
RESULT_ARTIFACTS = ("sweep_summary.json", "runs.jsonl")


def sweep_summary_path(workdir: Path, run_name: str) -> Path:
    return workdir / "runs" / run_name / "sweep_summary.json"


async def fetch_result_artifacts(run_dir: Path) -> list[str]:
    """Download the sweep leaderboard from the artifact store into *run_dir*.

    ``sync_progress`` translates the SSE event stream into local files
    (progress.jsonl, stdout.log, status.json, artifacts.json) — it does **not**
    pull artifact bytes down from R2. Those are uploaded by the sandbox's
    publisher and stay there until something asks for them. So a cloud job can
    complete perfectly and still leave no ``sweep_summary.json`` on the laptop,
    which reads downstream as a job that produced nothing.

    Returns the relpaths actually fetched, so the caller can tell "the file
    isn't there" from "the file was never published".
    """
    from lqh.artifacts import BackendArtifactStore

    try:
        listing = json.loads((run_dir / "artifacts.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    entries = listing.get("artifacts")
    if not isinstance(entries, list):
        return []

    store = BackendArtifactStore()
    fetched: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        relpath = entry.get("relpath")
        artifact_id = entry.get("artifact_id")
        if relpath not in RESULT_ARTIFACTS or not artifact_id:
            continue
        try:
            await store.download(artifact_id, run_dir / relpath)
            fetched.append(relpath)
        except Exception as exc:  # noqa: BLE001 — one missing file is not fatal
            logger.warning("could not fetch %s for %s: %s", relpath, run_dir.name, exc)
    return fetched


def read_rows(workdir: Path, run_name: str) -> list[dict[str, Any]]:
    """Per-config rows from a finished job, or [] if it has none yet."""
    path = sweep_summary_path(workdir, run_name)
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = summary.get("rows")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def job_complete(workdir: Path, run_name: str, expected_configs: int) -> bool:
    """Whether a previous attempt already produced every config's row.

    Resume is not a nicety here: a 48-cell study will lose jobs to preemption,
    and re-running a completed cell wastes an hour of GPU each time.
    """
    rows = read_rows(workdir, run_name)
    return len(rows) >= expected_configs


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


async def run_local(
    spec: JobSpec, *, workdir: Path, timeout: float,
) -> None:
    from ..base_vs_instruct.eval_local import _await_run

    run_dir = workdir / "runs" / spec.run_name
    manager = SubprocessManager()
    manager.start(
        run_dir, spec.launch_config, module="lqh.train.sweep", project_dir=workdir,
    )
    await _await_run(manager, run_dir, timeout=timeout, label=f"sweep:{spec.label}")


# ---------------------------------------------------------------------------
# Cloud
# ---------------------------------------------------------------------------


async def run_cloud(
    spec: JobSpec,
    *,
    workdir: Path,
    timeout: float,
    poll_interval: float = 30.0,
) -> None:
    """Submit one cell chunk to LQH Cloud and wait for it to finish.

    Artifacts land locally through the ordinary SSE sync, so afterwards
    ``read_rows`` sees the same ``sweep_summary.json`` a local run would write.
    """
    from lqh.remote.backend import RemoteConfig
    from lqh.remote.cloud import CloudBackend

    run_dir = workdir / "runs" / spec.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = RemoteConfig(
        name="cloud",
        type="cloud",
        hostname="api.lqh.ai",  # informational; CloudBackend hits api_root()
        remote_root="cloud:lqh",
    )
    backend = CloudBackend(cfg, workdir)
    job_id = await backend.submit_run(
        str(run_dir), spec.launch_config, module="lqh.train.sweep",
    )
    logger.info("%s: submitted cloud job %s", spec.label, job_id)

    waited = 0.0
    while waited < timeout:
        # sync_progress consumes the SSE stream and returns on a terminal
        # event or an idle timeout; it is idempotent, so looping on it is the
        # intended way to ride out a disconnect.
        try:
            await backend.sync_progress(str(run_dir), str(run_dir))
        except Exception as exc:  # noqa: BLE001 — a blip must not kill the study
            logger.warning("%s: progress sync blipped: %s", spec.label, exc)

        status = await backend.poll_status(job_id)
        if status.state == "completed":
            fetched = await fetch_result_artifacts(run_dir)
            if "sweep_summary.json" not in fetched:
                raise RuntimeError(
                    f"{spec.label}: cloud job {job_id} completed but published no "
                    "sweep_summary.json. The sandbox image's lqh predates "
                    "eval_all / the sweep-leaderboard artifact — publish the "
                    "current lqh and rebuild the Modal training image before "
                    "running the study (see CLAUDE.md 'push to production')."
                )
            logger.info("%s: cloud job completed", spec.label)
            return
        if status.state == "failed":
            raise RuntimeError(
                f"{spec.label}: cloud job {job_id} failed: "
                f"{status.error or 'no error recorded'}"
            )
        await asyncio.sleep(poll_interval)
        waited += poll_interval

    raise TimeoutError(
        f"{spec.label}: cloud job {job_id} did not finish within {timeout:.0f}s"
    )


async def run_job(
    spec: JobSpec, *, workdir: Path, compute: str, timeout: float,
) -> None:
    if compute == "cloud":
        await run_cloud(spec, workdir=workdir, timeout=timeout)
    elif compute == "local":
        await run_local(spec, workdir=workdir, timeout=timeout)
    else:
        raise SystemExit(f"unknown compute target {compute!r}; use cloud or local")
