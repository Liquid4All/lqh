"""In-sandbox scoring helpers for the DPO + SFT-sweep cloud pathways.

When a training subprocess runs on a SSH backend, the laptop-side
``RemoteRunWatcher`` (lqh/remote/watcher.py) is the one scoring
predictions, generating golden trajectories, and writing
``preferences.parquet`` (DPO) / ``eval_result.json`` (per-checkpoint
SFT eval) back to the run directory.

When the same subprocess runs in a cloud sandbox, that ping-pong
breaks: ``CloudBackend.sync_file_to_remote`` raises ``NotImplementedError``
because the sandbox filesystem isn't reachable from the laptop. The
fix is to do the scoring *inside the sandbox*, using the scoped
``LQH_API_TOKEN`` injected at submit time.

This module is the in-sandbox half of that. The trainer calls into
``score_dpo_iter_inline()`` after writing ``predictions.parquet`` +
``iter_request.json``, which then runs ``run_scoring`` + ``generate_golden``
synchronously, writing ``preferences.parquet`` directly. The
``wait_for_file(preferences.parquet)`` in the trainer's main loop
then succeeds immediately.

Detection: ``is_cloud_mode()`` returns true only when both
``LQH_API_TOKEN`` and ``LQH_BASE_URL`` are set in the environment.
The cloud launcher injects both; SSH backends
inject neither (laptop watcher does the work).

If we're not in cloud mode, the helpers are no-ops — the laptop
watcher path keeps working unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "is_cloud_mode",
    "score_dpo_iter_inline",
    "score_held_out_eval_inline",
    "score_run_eval_inline",
]


def is_cloud_mode() -> bool:
    """True iff this subprocess is running inside a backend-managed
    sandbox with a scoped API token to do its own scoring.

    Both vars must be present — LQH_API_TOKEN alone could mean a
    user passed their own bearer for a custom local script; we want
    the explicit "backend booted us" signal which is the combination
    with LQH_BASE_URL.
    """
    return bool(os.environ.get("LQH_API_TOKEN")) and bool(os.environ.get("LQH_BASE_URL"))


def _make_client():
    """Build an AsyncOpenAI client pointed at the backend API.

    Defers the import so non-cloud trainers don't pay the openai
    import cost at module load.
    """
    from openai import AsyncOpenAI

    base = os.environ["LQH_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return AsyncOpenAI(
        base_url=base,
        api_key=os.environ["LQH_API_TOKEN"],
    )


def _resolve_scorer_path(config: dict[str, Any], project_dir: Path) -> Path | None:
    """The DPO config carries ``scorer`` as a project-relative path
    (e.g. ``scorers/translation.md``). In the sandbox the trainer's
    PWD is the bundle root, which IS the project root — so the
    relative path resolves against the bundle. Caller passes
    ``Path.cwd()`` (or the explicit run-dir's ``inputs`` dir) as
    ``project_dir``. Returns None when no scorer is configured.
    """
    scorer_rel = config.get("scorer")
    if not scorer_rel:
        return None
    p = (project_dir / scorer_rel).resolve()
    if not p.exists():
        logger.warning("scorer file not found at %s", p)
        return None
    return p


async def _score_iter_async(
    iter_dir: Path,
    config: dict[str, Any],
    project_dir: Path,
) -> None:
    """Async core of score_dpo_iter_inline — does the work without
    swallowing exceptions. Caller is responsible for catching.

    Mirrors RemoteRunWatcher._process_iteration (lqh/remote/watcher.py)
    so the result on disk is byte-identical to the laptop-driven path.
    """
    predictions_path = iter_dir / "predictions.parquet"
    if not predictions_path.exists():
        logger.warning("no predictions.parquet in %s; skipping inline scoring", iter_dir)
        return
    scorer_path = _resolve_scorer_path(config, project_dir)
    if scorer_path is None:
        logger.warning("no scorer configured; skipping inline scoring for %s", iter_dir)
        return

    from lqh.golden import generate_golden
    from lqh.scoring import run_scoring

    client = _make_client()
    try:
        # Step 1: judge-score predictions
        await run_scoring(
            dataset_path=predictions_path,
            scorer_path=scorer_path,
            output_dir=iter_dir,
            client=client,
            run_inference=False,
        )
        # Step 2: assemble preferences. generate_golden writes
        # preferences.parquet into iter_dir, which is what the
        # trainer's wait_for_file() call is watching for.
        await generate_golden(
            predictions_path=predictions_path,
            scores_path=iter_dir / "results.parquet",
            dataset_path=config.get("dataset", ""),
            config=config,
            client=client,
            output_dir=iter_dir,
        )
    finally:
        # AsyncOpenAI's underlying httpx client holds open connections
        # until close(); explicit close avoids the "Unclosed client
        # session" asyncio warning at process exit.
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass


def score_dpo_iter_inline(
    iter_dir: Path,
    config: dict[str, Any],
    project_dir: Path,
) -> bool:
    """Synchronously score one DPO iteration + assemble preferences.

    Returns True if the inline scoring actually ran (cloud mode), in
    which case ``iter_dir/preferences.parquet`` is now on disk and the
    trainer's wait_for_file() will succeed immediately. Returns False
    when we're not in cloud mode — the laptop watcher path remains
    authoritative, and the trainer should block on the file as usual.

    Failures are logged and re-raised so the trainer can decide what
    to do (today: it'll be caught by the trainer's existing
    wait_for_file timeout path, which surfaces as "no preferences
    received this iter — converged or upstream failure").
    """
    if not is_cloud_mode():
        return False
    logger.info("inline scoring DPO iter at %s (cloud mode)", iter_dir)
    asyncio.run(_score_iter_async(iter_dir, config, project_dir))
    return True


async def _score_held_out_async(
    iter_dir: Path,
    config: dict[str, Any],
    project_dir: Path,
) -> dict[str, Any] | None:
    """Score the per-iter held-out eval predictions. Returns the
    summary dict (mean/median/std) or None when nothing was scored.
    """
    eval_pred_path = iter_dir / "eval_predictions.parquet"
    if not eval_pred_path.exists():
        return None
    scorer_path = _resolve_scorer_path(config, project_dir)
    if scorer_path is None:
        return None

    from lqh.scoring import run_scoring

    client = _make_client()
    try:
        # Put the eval results in a sibling dir so they don't collide
        # with the iter's main results.parquet (which is the score
        # over the on-policy predictions, not the held-out set).
        eval_out_dir = iter_dir / "held_out_eval"
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        await run_scoring(
            dataset_path=eval_pred_path,
            scorer_path=scorer_path,
            output_dir=eval_out_dir,
            client=client,
            run_inference=False,
        )
        summary_path = eval_out_dir / "summary.json"
        if summary_path.exists():
            try:
                return json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                return None
        return None
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass


async def _score_run_eval_async(
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Score the sweep's eval-of-best predictions and write
    eval_result.json + results.parquet at the run-dir level. Mirrors
    the laptop watcher's _check_eval_requests flow.
    """
    preds = run_dir / "predictions.parquet"
    if not preds.exists():
        return None
    # The scorer path here is relative to the bundle root (the
    # trainer's PWD), same convention as the DPO iter.
    scorer_path = _resolve_scorer_path(config, Path.cwd())
    if scorer_path is None:
        return None

    from lqh.scoring import run_scoring

    client = _make_client()
    try:
        await run_scoring(
            dataset_path=preds,
            scorer_path=scorer_path,
            output_dir=run_dir,
            client=client,
            run_inference=False,
        )
        summary_path = run_dir / "summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                summary = {}
        # Drop an eval_result.json next to the predictions so the
        # publisher classifies it as an eval_result artifact (see
        # lqh/remote/publish.py:_resolve_candidates).
        (run_dir / "eval_result.json").write_text(
            json.dumps({
                "summary": summary,
                "predictions_path": str(preds.relative_to(run_dir)),
            }, indent=2) + "\n"
        )
        return summary or None
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass


def score_run_eval_inline(
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Synchronously score eval-of-best predictions at the run-dir
    level. Called by ``lqh.train.sweep`` after ``_run_eval_of_best``
    symlinks ``predictions.parquet`` + ``eval_request.json`` up.

    Returns the judge summary dict on success, or None when:
      * not in cloud mode (laptop watcher handles it),
      * no scorer configured,
      * no predictions on disk.
    """
    if not is_cloud_mode():
        return None
    logger.info("inline scoring eval-of-best at %s (cloud mode)", run_dir)
    return asyncio.run(_score_run_eval_async(run_dir, config))


def score_held_out_eval_inline(
    iter_dir: Path,
    config: dict[str, Any],
    project_dir: Path,
) -> dict[str, Any] | None:
    """Score the per-iteration held-out eval (eval_predictions.parquet).

    Returns the summary dict on success, or None when no eval was
    scored (not in cloud mode, no scorer, or no eval predictions on
    disk). The caller decides whether to use the score for an
    early-abort decision.
    """
    if not is_cloud_mode():
        return None
    logger.info("inline scoring held-out eval at %s (cloud mode)", iter_dir)
    return asyncio.run(_score_held_out_async(iter_dir, config, project_dir))
