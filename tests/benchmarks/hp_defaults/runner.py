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


def is_auth_error(exc: BaseException) -> bool:
    """Whether a submit failure is a credentials problem rather than a job one.

    Auth is a precondition of the whole study, not a property of one cell:
    retrying the other 47 cells against the same rejected token just burns
    wall-clock and produces 47 identical tracebacks.
    """
    from lqh.remote.cloud import CloudError

    return isinstance(exc, CloudError) and (
        "401" in str(exc) or "403" in str(exc)
    )


class CloudAuthError(SystemExit):
    """Raised when the credentials cannot submit cloud jobs."""


class RetryableJobError(RuntimeError):
    """A job that failed for a reason unrelated to its configuration.

    Resubmitting the identical chunk is pointless for an OOM or a bad config
    and correct for an interruption, so the two are separate exception types.
    """


class EmptyResultError(RetryableJobError):
    """The job reported success but published no sweep leaderboard.

    Observed on 2026-08-06 in stage A: two sandboxes went quiet mid-eval, and
    on the next backend restart the job pump reattached and immediately wrote
    ``exit_code=0, status=completed`` for both. Nothing was ever published, yet
    the study was billed the full GPU time. A job that really finishes writes
    the leaderboard *before* exiting, so "completed with no artifacts" is an
    orphan wearing a success label — retry it rather than trusting the status.
    """


class InfraJobError(RetryableJobError):
    """The job failed and the backend classified the cause as infrastructure."""


def _auth_help(detail: str) -> str:
    return (
        f"cloud authentication failed ({detail}).\n\n"
        "The most common cause is LQH_DEBUG_API_KEY being set: it is accepted "
        "on /v1/* (so data generation works fine) but carries no user or org, "
        "and cloud jobs need both for ownership and billing — so every submit "
        "is rejected. Check with:\n"
        "    echo $LQH_DEBUG_API_KEY\n"
        "and if it is set, unset it so the logged-in credentials are used:\n"
        "    unset LQH_DEBUG_API_KEY\n\n"
        "Otherwise the CLI token is missing or expired — run `lqh` and /login."
    )


async def check_cloud_auth(workdir: Path) -> None:
    """Verify the credentials can submit cloud jobs, before spending anything.

    The planner endpoint validates auth and creates nothing, so this costs one
    round trip. It runs BEFORE data generation on purpose: a study that
    generates four tasks' worth of data for hours and only then discovers it
    cannot submit has wasted the expensive part for no result.
    """
    from lqh.remote.backend import RemoteConfig
    from lqh.remote.cloud import CloudBackend, CloudError

    backend = CloudBackend(
        RemoteConfig(
            name="cloud", type="cloud",
            hostname="api.lqh.ai", remote_root="cloud:lqh",
        ),
        workdir,
    )
    try:
        await backend.plan_job("train_sft", base_model="LiquidAI/LFM2.5-350M")
    except CloudError as exc:
        if is_auth_error(exc):
            raise CloudAuthError(_auth_help(str(exc))) from exc
        # A planner error that is not about auth (an unknown kind, a backend
        # blip) should not block the study — submit will surface it properly.
        logger.warning("cloud preflight could not reach the planner: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cloud preflight skipped: %s", exc)


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
    train_rows: int | None = None,
) -> dict[str, Any]:
    """The sweep payload for one cell chunk.

    Deliberately mirrors what ``handle_start_training`` builds, minus the
    hyperparameters the grid is going to override anyway, so the study
    measures the configuration the product actually ships.

    ``train_rows`` is the cell's dataset size, and it goes into the config as
    ``dataset_rows.train_effective`` so ``lqh.train.sweep`` re-derives each grid
    point's batch for **its own** epoch count, exactly as a standalone product
    run would. That is what keeps the epochs axis interpretable: every config
    aims at the same optimizer-step target, so the axis measures epochs rather
    than update count, and the learning-rate axis stays clean because the batch
    depends only on epochs.

    Stage A (hpd-stageA) ran BEFORE that fix: it derived one batch from the
    3-epoch default and let the grid override epochs without resizing it, so its
    1- and 2-epoch configs took ~1/3 and ~2/3 of the updates they should have.
    Its epoch conclusion is therefore confounded and must not be quoted; its
    learning-rate conclusion is not (all LRs shared one epoch count, hence one
    batch).
    """
    from lqh.train import defaults

    recommended = defaults.recommended(
        run_type="sft",
        lora=True,
        train_rows=train_rows,
        num_epochs=defaults.DEFAULT_SFT_EPOCHS if train_rows else None,
    )
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
    if train_rows:
        # What lqh.train.sweep._rederive_sft_batch reads to resize each grid
        # point's batch for its own epoch count (the product does the same at
        # submission). Without it every config inherits the 3-epoch batch and
        # the epochs axis becomes an update-count axis.
        base_config["dataset_rows"] = {
            "train": train_rows,
            "train_effective": train_rows,
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


def _published_anything(run_dir: Path) -> int:
    try:
        listing = json.loads((run_dir / "artifacts.json").read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    entries = listing.get("artifacts")
    return len(entries) if isinstance(entries, list) else 0


def _no_result_message(spec: JobSpec, job_id: str, run_dir: Path) -> str:
    """Why a 'completed' job left nothing behind — the two causes look alike.

    Distinguishing them from the client costs one already-synced file: an
    orphan publishes *nothing*, while an out-of-date sandbox image still
    publishes checkpoints and logs and only misses the leaderboard.
    """
    published = _published_anything(run_dir)
    if published == 0:
        return (
            f"{spec.label}: cloud job {job_id} reported completed but published "
            "no artifacts at all — not even logs. A sandbox that finishes "
            "normally registers its artifacts before exiting, so this is an "
            "infrastructure orphan mislabelled as success (a backend restart "
            "reattaching to a sandbox it can no longer observe writes "
            "exit_code=0). Retrying the chunk."
        )
    return (
        f"{spec.label}: cloud job {job_id} published {published} artifacts but no "
        "sweep_summary.json. The sandbox image's lqh most likely predates "
        "eval_all / the sweep-leaderboard artifact — check the job log for the "
        "'sweep: eval_all on' banner, and if it is missing, publish the current "
        "lqh and rebuild the Modal training image (CLAUDE.md 'push to "
        "production')."
    )


def reset_run_dir(workdir: Path, run_name: str) -> None:
    """Clear one job's synced state so the chunk can be resubmitted in place.

    The run dir has to keep its name — ``_collect`` finds a cell's rows by run
    name — so a retry reuses it and only the previous attempt's bookkeeping is
    removed. cloud_state.json is the important one: sync_progress returns
    immediately when the state it loads is already terminal.
    """
    run_dir = workdir / "runs" / run_name
    for name in ("cloud_state.json", "artifacts.json", "status.json",
                 "progress.jsonl", "stdout.log", "stderr.log"):
        (run_dir / name).unlink(missing_ok=True)


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
                raise EmptyResultError(_no_result_message(spec, job_id, run_dir))
            logger.info("%s: cloud job completed", spec.label)
            return
        if status.state == "failed":
            detail = (
                f"{spec.label}: cloud job {job_id} failed: "
                f"{status.error or 'no error recorded'}"
            )
            if (status.failure or {}).get("infra"):
                raise InfraJobError(f"{detail} (classified infrastructure)")
            raise RuntimeError(detail)
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
