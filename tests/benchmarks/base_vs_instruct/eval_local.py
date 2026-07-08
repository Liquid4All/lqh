"""The benchmark's single eval primitive: LOCAL inference + judge scoring.

This is deliberately the *only* way the benchmark measures a model, so the
three reported points (baseline, best-SFT, best-DPO) are apples-to-apples:

  1. spawn ``python -m lqh.infer`` on the model + eval set (local GPU; the
     base/instruct/checkpoint distinction is transparent to
     ``load_for_inference``), which strips each trailing assistant turn,
     generates a prediction, and writes ``predictions.parquet``;
  2. call ``lqh.scoring.run_scoring`` over those predictions with the task's
     judge rubric (``run_inference=False`` — the assistant turn already IS the
     prediction). The judge is an API call (``judge:<size>``); only the GPU
     inference is local.

This is exactly what the ``start_local_eval`` tool does, minus the TUI watcher
that would otherwise drive the scoring step — here the script drives it.

The system prompt is baked into the eval parquet rows by the datagen pipeline,
so we don't re-apply it here; ``lqh.infer`` already uses the row's own system
message.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from lqh.scoring import run_scoring
from lqh.subprocess_manager import SubprocessManager

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    mean: float | None
    n_scored: int
    predictions_path: Path
    scores_dir: Path


def _stderr_tail(run_dir: Path, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    """Return the tail of a run's stderr.log for inclusion in error messages.

    Subprocess failures (a crashed sweep child, an infer OOM, a bad config)
    write their traceback to stderr.log; the one-line status.error rarely
    carries it. Surfacing the tail turns "see stderr.log" — useless once the
    run is on a remote box and the logs are gone — into an actionable error.
    """
    log_path = run_dir / "stderr.log"
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return ""
    text = text.strip()
    if not text:
        return ""
    lines = text.splitlines()[-max_lines:]
    tail = "\n".join(lines)
    if len(tail) > max_chars:
        tail = "…" + tail[-max_chars:]
    return f"\n--- {log_path} (tail) ---\n{tail}\n--- end ---"


async def _await_run(
    manager: SubprocessManager,
    run_dir: Path,
    *,
    timeout: float,
    poll: float = 3.0,
    label: str = "run",
) -> None:
    """Poll a subprocess run dir until terminal; raise on failure/timeout.

    On failure/timeout the raised error carries the tail of stderr.log so the
    actual traceback is visible at the call site (and in the benchmark report),
    not buried in a log file on a possibly-remote machine.
    """
    waited = 0.0
    while waited < timeout:
        status = manager.get_status(run_dir)
        if status.state == "completed":
            return
        if status.state == "failed":
            raise RuntimeError(
                f"{label} failed: "
                f"{status.error or 'no status.error recorded'}"
                f"{_stderr_tail(run_dir)}"
            )
        # "unknown" right after launch is normal (no progress yet); keep waiting
        # as long as the process is alive or hasn't been started long.
        if status.state == "unknown" and waited > poll * 3 and not manager.is_alive(run_dir):
            # Process gone with no terminal status — treat as failure.
            raise RuntimeError(
                f"{label} exited without a terminal status"
                f"{_stderr_tail(run_dir)}"
            )
        await asyncio.sleep(poll)
        waited += poll
    if manager.is_alive(run_dir):
        manager.stop(run_dir)
    raise TimeoutError(
        f"{label} did not finish within {timeout:.0f}s ({run_dir})"
        f"{_stderr_tail(run_dir)}"
    )


async def eval_local(
    *,
    workdir: Path,
    run_name: str,
    model_path: str,
    eval_parquet: Path,
    scorer_path: Path,
    client: AsyncOpenAI,
    judge_size: str = "small",
    max_new_tokens: int = 512,
    infer_timeout: float = 3600.0,
    resume: bool = True,
) -> EvalResult:
    """Run local inference for *model_path* on *eval_parquet*, then judge-score.

    *model_path* may be a HF id (``LiquidAI/LFM2.5-350M``) or a local model
    directory (``runs/<sweep>/model``); ``load_for_inference`` resolves both.
    """
    run_dir = workdir / "runs" / run_name
    predictions = run_dir / "predictions.parquet"
    scores_dir = run_dir / "scores"
    summary_path = scores_dir / "summary.json"

    # --- resume: reuse a completed score if present ---
    if resume and summary_path.exists():
        mean, n = _read_summary(summary_path)
        logger.info("eval %s: reusing cached score mean=%s", run_name, mean)
        return EvalResult(mean, n, predictions, scores_dir)

    # --- inference (skip if predictions already there) ---
    if not (resume and predictions.exists()):
        manager = SubprocessManager()
        config = {
            "type": "infer",
            "base_model": model_path,
            "dataset": str(eval_parquet.resolve()),
            "max_new_tokens": max_new_tokens,
            "manifest": ["base_model", "dataset"],
        }
        manager.start(run_dir, config, module="lqh.infer", project_dir=workdir)
        await _await_run(manager, run_dir, timeout=infer_timeout, label=f"infer:{run_name}")
        if not predictions.exists():
            raise RuntimeError(
                f"infer:{run_name} completed but wrote no predictions.parquet "
                f"({run_dir})"
            )

    # --- judge scoring ---
    scores_dir.mkdir(parents=True, exist_ok=True)
    await run_scoring(
        dataset_path=predictions,
        scorer_path=scorer_path,
        output_dir=scores_dir,
        client=client,
        model_size=judge_size,
        run_inference=False,
    )
    mean, n = _read_summary(summary_path)
    logger.info("eval %s: mean=%s over %d scored", run_name, mean, n)
    return EvalResult(mean, n, predictions, scores_dir)


def _read_summary(summary_path: Path) -> tuple[float | None, int]:
    """Pull ``scores.mean`` + scored count from a run_scoring summary.json."""
    if not summary_path.exists():
        return None, 0
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, 0
    scores = data.get("scores")
    mean = None
    if isinstance(scores, dict) and scores.get("mean") is not None:
        try:
            mean = float(scores["mean"])
        except (TypeError, ValueError):
            mean = None
    n = int(data.get("num_scored", data.get("num_samples", 0)) or 0)
    return mean, n
