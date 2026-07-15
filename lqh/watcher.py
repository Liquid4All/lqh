"""Async file watcher for training run orchestration.

Runs as an ``asyncio.Task`` in the main lqh process event loop alongside
the agent.  Polls the filesystem for signal files written by training
subprocesses and responds by running scoring, generating golden
trajectories, or assembling preference pairs.

Never imports torch or transformers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Protocol

from lqh.config import default_api_base_url
from lqh.subprocess_manager import SubprocessManager
from lqh.train.progress import (
    read_current_attempt_id,
    read_latest_metrics,
    read_latest_status,
)

logger = logging.getLogger(__name__)

_SCORE_SUCCESS = "success"
_SCORE_NOT_READY = "not_ready"
_SCORE_FAILED = "failed"
_SCORE_RETRY_DELAYS = (10.0, 60.0)
_PAYLOAD_HANDOFF_TIMEOUT = 15 * 60.0

__all__ = ["RunWatcher", "WatcherCallbacks"]


class WatcherCallbacks(Protocol):
    """Callbacks from the watcher to the TUI / agent."""

    def on_training_progress(
        self,
        run_name: str,
        step: int | None,
        loss: float | None,
        lr: float | None,
        epoch: float | None,
    ) -> None: ...

    def on_training_completed(self, run_name: str) -> None: ...

    def on_training_failed(self, run_name: str, error: str | None) -> None: ...

    def on_eval_scored(
        self,
        run_name: str,
        checkpoint: str,
        mean_score: float,
    ) -> None: ...

    def on_iter_scored(
        self,
        run_name: str,
        iteration: str,
        mean_score: float,
    ) -> None: ...


class _NullCallbacks:
    """Default no-op callbacks."""

    def on_training_progress(self, *a: Any, **kw: Any) -> None:
        pass

    def on_training_completed(self, *a: Any, **kw: Any) -> None:
        pass

    def on_training_failed(self, *a: Any, **kw: Any) -> None:
        pass

    def on_eval_scored(self, *a: Any, **kw: Any) -> None:
        pass

    def on_iter_scored(self, *a: Any, **kw: Any) -> None:
        pass


class RunWatcher:
    """Watch a training run directory and respond to subprocess signals.

    For SFT runs:
      - Detects ``checkpoints/step_N/eval_request.json``
      - Scores predictions via the API judge
      - Writes ``eval_result.json``

    For DPO runs:
      - Detects ``iterations/iter_NNN/iter_request.json``
      - Scores predictions, generates golden trajectories,
        assembles preference pairs
      - Writes ``preferences.parquet``
      - Scores ``eval_predictions.parquet`` on the fixed held-out set

    Parameters
    ----------
    run_dir : Path
        The run directory (``runs/<run_name>/``).
    config : dict
        The run's config.json contents.
    project_dir : Path
        Project root directory.
    api_key : str
        API key for scoring calls.
    api_base_url : str
        Base URL for the API.
    callbacks : WatcherCallbacks, optional
        Callbacks for TUI integration.
    poll_interval : float
        Seconds between filesystem polls.
    """

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        project_dir: Path,
        api_key: str,
        api_base_url: str | None = None,
        callbacks: WatcherCallbacks | None = None,
        poll_interval: float = 3.0,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.project_dir = project_dir
        self.api_key = api_key
        self.api_base_url = api_base_url if api_base_url is not None else default_api_base_url()
        self.callbacks: Any = callbacks or _NullCallbacks()
        self.poll_interval = poll_interval
        self.run_name = run_dir.name

        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._manager = SubprocessManager()

        # Track which requests we've already processed
        self._processed_eval_requests: set[str] = set()
        self._processed_iter_requests: set[str] = set()
        self._processed_held_out_requests: set[str] = set()
        self._scoring_attempts: dict[str, int] = {}
        self._iteration_attempts: dict[str, int] = {}
        self._retry_not_before: dict[str, float] = {}
        self._not_ready_since: dict[str, float] = {}

    async def start(self) -> None:
        """Start the watcher as a background asyncio task."""
        self._stop.clear()
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Signal the watcher to stop and wait for it."""
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._task = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _watch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._update_progress()
                await self._check_eval_requests()
                await self._check_iter_requests()
                self._check_completion()
            except Exception:
                logger.exception("Watcher error for run %s", self.run_name)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                break  # stop was set
            except asyncio.TimeoutError:
                pass  # normal poll interval

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def _update_progress(self) -> None:
        latest = read_latest_metrics(self.run_dir)
        if latest and "step" in latest:
            self.callbacks.on_training_progress(
                self.run_name,
                step=latest.get("step"),
                loss=latest.get("loss"),
                lr=latest.get("lr"),
                epoch=latest.get("epoch"),
            )

    def _check_completion(self) -> None:
        latest = read_latest_status(self.run_dir)
        if latest and latest.get("status") == "completed":
            if self._has_pending_scoring_requests():
                return
            self.callbacks.on_training_completed(self.run_name)
            self._stop.set()
        elif latest and latest.get("status") in {"failed", "interrupted"}:
            self.callbacks.on_training_failed(
                self.run_name, latest.get("error")
            )
            self._stop.set()
        elif not self._manager.is_alive(self.run_dir):
            # A startup "running" row is non-terminal. If the process was
            # hard-killed before it could append a terminal row, liveness is
            # still authoritative.
            self.callbacks.on_training_failed(
                self.run_name, "Process exited without writing final status"
            )
            self._stop.set()

    # ------------------------------------------------------------------
    # SFT: checkpoint eval requests
    # ------------------------------------------------------------------

    async def _check_eval_requests(self) -> None:
        """Score any unprocessed eval_request.json files.

        Inference runs (``type: infer``) write the request at the run root
        next to predictions.parquet; training runs write one per checkpoint
        under ``checkpoints/step_N/``. Both are handled here.
        """
        # Run-root request: emitted by `python -m lqh.infer` for one-shot eval.
        request_file = self.run_dir / "eval_request.json"
        result_file = self.run_dir / "eval_result.json"
        error_file = self.run_dir / "eval_error.json"
        key = str(self.run_dir)
        if (
            request_file.exists()
            and not result_file.exists()
            and not error_file.exists()
            and key not in self._processed_eval_requests
            and self._retry_due(key)
        ):
            self._processed_eval_requests.add(key)
            outcome = await self._score_checkpoint(self.run_dir)
            self._handle_scoring_outcome(
                key, self.run_dir, outcome, preference=False,
            )

        # Per-checkpoint requests from training runs.
        checkpoints_dir = self.run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return

        for cp_dir in sorted(checkpoints_dir.iterdir()):
            if not cp_dir.is_dir():
                continue
            request_file = cp_dir / "eval_request.json"
            result_file = cp_dir / "eval_result.json"
            error_file = cp_dir / "eval_error.json"
            key = str(cp_dir)

            if (
                request_file.exists()
                and not result_file.exists()
                and not error_file.exists()
                and key not in self._processed_eval_requests
                and self._retry_due(key)
            ):
                self._processed_eval_requests.add(key)
                outcome = await self._score_checkpoint(cp_dir)
                self._handle_scoring_outcome(
                    key, cp_dir, outcome, preference=False,
                )

    def _retry_due(self, key: str) -> bool:
        return time.monotonic() >= self._retry_not_before.get(key, 0.0)

    def _current_attempt_id(self) -> str | None:
        return read_current_attempt_id(self.run_dir)

    def _handle_scoring_outcome(
        self,
        key: str,
        output_dir: Path,
        outcome: str,
        *,
        preference: bool,
    ) -> None:
        processed = (
            self._processed_iter_requests
            if preference
            else self._processed_eval_requests
        )
        attempts_by_key = (
            self._iteration_attempts if preference else self._scoring_attempts
        )
        if outcome == _SCORE_SUCCESS:
            attempts_by_key.pop(key, None)
            self._retry_not_before.pop(key, None)
            self._not_ready_since.pop(key, None)
            return

        # A request marker can arrive before its larger parquet payload over
        # SSH/rsync. That is a handoff state, not a failed judge attempt.
        processed.discard(key)
        if outcome == _SCORE_NOT_READY:
            first_seen = self._not_ready_since.setdefault(key, time.monotonic())
            if time.monotonic() - first_seen >= _PAYLOAD_HANDOFF_TIMEOUT:
                from lqh.progress import write_error_marker

                processed.add(key)
                marker = (
                    output_dir / "preference_error.json"
                    if preference
                    else output_dir / "eval_error.json"
                )
                write_error_marker(
                    marker,
                    "scoring payload did not arrive within 15 minutes",
                )
                self._not_ready_since.pop(key, None)
                self._retry_not_before.pop(key, None)
            return

        self._not_ready_since.pop(key, None)

        attempts = attempts_by_key.get(key, 0) + 1
        attempts_by_key[key] = attempts
        if attempts < 3:
            self._retry_not_before[key] = (
                time.monotonic() + _SCORE_RETRY_DELAYS[attempts - 1]
            )
            return

        from lqh.progress import write_error_marker

        processed.add(key)
        marker = (
            output_dir / "preference_error.json"
            if preference
            else output_dir / "eval_error.json"
        )
        label = "preference scoring" if preference else "judge scoring"
        write_error_marker(marker, f"{label} failed after 3 attempts")
        self._retry_not_before.pop(key, None)
        self._not_ready_since.pop(key, None)

    def _has_pending_scoring_requests(self) -> bool:
        from lqh.progress import has_pending_final_result

        return has_pending_final_result(self.run_dir, self.config)

    async def _score_checkpoint(self, checkpoint_dir: Path) -> str:
        """Score predictions from a checkpoint and write eval_result.json."""
        predictions_path = checkpoint_dir / "predictions.parquet"
        if not predictions_path.exists():
            logger.warning("No predictions.parquet in %s", checkpoint_dir)
            return _SCORE_NOT_READY

        effective_config = (
            self.config.get("base_config", {})
            if self.config.get("type") == "sweep"
            else self.config
        )
        scorer_path = effective_config.get("scorer")
        if not scorer_path:
            logger.warning("No scorer configured, skipping eval for %s", checkpoint_dir)
            return _SCORE_SUCCESS

        scoring_completed = False
        try:
            from lqh.client import create_client
            from lqh.progress import (
                OBSERVER_PROGRESS_FILE,
                ProgressReporter,
                final_scoring_context,
            )
            from lqh.scoring import score_predictions_by_source

            client = create_client(self.api_key, self.api_base_url)

            # A run-root request is the final result stage. Checkpoint-local
            # requests are internal training diagnostics and must not replace
            # the whole-job headline.
            headline_config = dict(effective_config)
            if self.config.get("type") == "sweep":
                headline_config["progress_task_kind"] = "training_sweep"
            scoring_context = final_scoring_context(
                checkpoint_dir, headline_config,
            )
            scoring_attempt = self._scoring_attempts.get(
                str(checkpoint_dir), 0,
            ) + 1
            scoring_phase = f"scoring_attempt_{scoring_attempt}"
            reporter = None
            if scoring_context is not None:
                reporter = ProgressReporter(
                    task_kind=scoring_context.task_kind,
                    label=self.run_name,
                    run_dir=scoring_context.progress_dir,
                    file_name=OBSERVER_PROGRESS_FILE,
                    attempt_id=self._current_attempt_id(),
                )
                reporter.update(
                    phase=scoring_phase,
                    phase_label="judging results",
                    completed=0,
                    unit="samples", overall_fraction=scoring_context.start,
                    detail=(
                        f"retry {scoring_attempt}/3"
                        if scoring_attempt > 1 else None
                    ),
                    force=True,
                )

            def on_progress(completed: int, total: int) -> None:
                if reporter is not None:
                    reporter.update(
                        phase=scoring_phase, phase_label="judging results",
                        completed=completed, total=total, unit="samples",
                        overall_fraction=(
                            scoring_context.start
                            + (1.0 - scoring_context.start)
                            * completed / max(total, 1)
                        ),
                        detail=(
                            f"retry {scoring_attempt}/3"
                            if scoring_attempt > 1 else None
                        ),
                        force=completed == total,
                    )

            # Per-source scoring: scores each eval source separately and
            # writes eval_result.json with a per_source breakdown plus a
            # macro-average headline (scores.mean). Single-source predictions
            # collapse to one group — identical to the legacy behaviour.
            payload = await score_predictions_by_source(
                predictions_path=predictions_path,
                scorer_path=self.project_dir / scorer_path,
                output_dir=checkpoint_dir,
                client=client,
                on_progress=on_progress,
            )
            scoring_completed = True
            if reporter is not None:
                num_scored = int(payload.get("num_scored", 0))
                total_attempted = num_scored + int(payload.get("num_failed", 0))
                if num_scored > 0:
                    reporter.update(
                        phase="completed", phase_label="results ready",
                        completed=total_attempted, total=total_attempted,
                        unit="samples", overall_fraction=1.0,
                        result_ready=True, force=True,
                    )
                else:
                    # run_scoring may have written a zero-score result. It is
                    # not a successful handoff and must remain retryable.
                    (checkpoint_dir / "eval_result.json").unlink(missing_ok=True)
                    return _SCORE_FAILED

            mean_score = payload.get("scores", {}).get("mean")
            if mean_score is not None:
                try:
                    self.callbacks.on_eval_scored(
                        self.run_name,
                        checkpoint_dir.name,
                        mean_score,
                    )
                except Exception:
                    # Notification failures must not invalidate a paid,
                    # durably-written judge result.
                    logger.exception(
                        "Eval callback failed for %s", checkpoint_dir,
                    )
                logger.info(
                    "Scored %s/%s: mean=%.2f",
                    self.run_name, checkpoint_dir.name, mean_score,
                )
            return _SCORE_SUCCESS

        except Exception:
            logger.exception("Failed to score %s", checkpoint_dir)
            if not scoring_completed:
                (checkpoint_dir / "eval_result.json").unlink(missing_ok=True)
            return _SCORE_FAILED

    # ------------------------------------------------------------------
    # DPO: iteration requests
    # ------------------------------------------------------------------

    async def _check_iter_requests(self) -> None:
        """Check for unprocessed iter_request.json files in iterations."""
        effective_config = (
            self.config.get("base_config", {})
            if self.config.get("type") == "sweep"
            else self.config
        )
        if effective_config.get("type") not in ("on_policy_dpo", "dpo"):
            return

        for iter_dir in self._iteration_dirs():
            request_file = iter_dir / "iter_request.json"
            preferences_file = iter_dir / "preferences.parquet"
            error_file = iter_dir / "preference_error.json"
            key = str(iter_dir)

            if (
                request_file.exists()
                and not preferences_file.exists()
                and not error_file.exists()
                and key not in self._processed_iter_requests
                and self._retry_due(key)
            ):
                self._processed_iter_requests.add(key)
                outcome = await self._process_iteration(iter_dir)
                self._handle_scoring_outcome(
                    key, iter_dir, outcome, preference=True,
                )

            held_out_ready = iter_dir / "eval_predictions_ready.json"
            held_out_summary = iter_dir / "held_out_eval" / "summary.json"
            held_out_error = iter_dir / "held_out_eval" / "eval_error.json"
            held_out_key = f"{iter_dir}:held_out"
            if (
                held_out_ready.exists()
                and not held_out_summary.exists()
                and not held_out_error.exists()
                and held_out_key not in self._processed_held_out_requests
                and self._retry_due(held_out_key)
            ):
                self._processed_held_out_requests.add(held_out_key)
                outcome = await self._score_dpo_held_out(iter_dir)
                if outcome == _SCORE_SUCCESS:
                    self._processed_held_out_requests.discard(held_out_key)
                    self._retry_not_before.pop(held_out_key, None)
                else:
                    # Reuse the ordinary eval retry machinery, but keep its
                    # bookkeeping set in sync with this request class.
                    self._processed_eval_requests.add(held_out_key)
                    self._handle_scoring_outcome(
                        held_out_key,
                        iter_dir / "held_out_eval",
                        outcome,
                        preference=False,
                    )
                    self._processed_held_out_requests.discard(held_out_key)

    def _iteration_dirs(self) -> list[Path]:
        """Return standalone and sweep-child DPO iteration directories."""
        roots = [self.run_dir / "iterations"]
        if self.config.get("type") == "sweep":
            roots.extend(sorted(self.run_dir.glob("sweep_*/iterations")))
        result: list[Path] = []
        for root in roots:
            if root.is_dir():
                result.extend(
                    directory
                    for directory in sorted(root.iterdir())
                    if directory.is_dir()
                )
        return result

    async def _score_dpo_held_out(self, iter_dir: Path) -> str:
        """Judge a DPO iteration on its fixed held-out prompt set."""
        predictions = iter_dir / "eval_predictions.parquet"
        if not predictions.exists():
            return _SCORE_NOT_READY
        effective_config = (
            self.config.get("base_config", {})
            if self.config.get("type") == "sweep"
            else self.config
        )
        scorer = effective_config.get("scorer")
        if not scorer:
            return _SCORE_FAILED
        try:
            from lqh.client import create_client
            from lqh.scoring import run_scoring

            output = iter_dir / "held_out_eval"
            output.mkdir(parents=True, exist_ok=True)
            result = await run_scoring(
                dataset_path=predictions,
                scorer_path=self.project_dir / scorer,
                output_dir=output,
                client=create_client(self.api_key, self.api_base_url),
                model_size=str(
                    effective_config.get("preference_judge_size", "small")
                ),
                run_inference=False,
            )
            return _SCORE_SUCCESS if result.scored else _SCORE_FAILED
        except Exception:
            logger.exception("Failed to score held-out DPO iteration %s", iter_dir)
            return _SCORE_FAILED

    async def _process_iteration(self, iter_dir: Path) -> str:
        """Score predictions, generate golden trajectories, assemble preferences."""
        effective_config = (
            self.config.get("base_config", {})
            if self.config.get("type") == "sweep"
            else self.config
        )
        predictions_path = iter_dir / "predictions.parquet"
        if not predictions_path.exists():
            logger.warning("No predictions.parquet in %s", iter_dir)
            return _SCORE_NOT_READY

        scorer_path = effective_config.get("scorer")
        if not scorer_path:
            logger.warning("No scorer configured for DPO iteration")
            from lqh.progress import write_error_marker

            write_error_marker(
                iter_dir / "preference_error.json",
                "preference scoring cannot start: no scorer configured",
            )
            return _SCORE_SUCCESS

        try:
            from lqh.client import create_client
            from lqh.golden import generate_golden
            from lqh.progress import (
                DEFAULT_DPO_ITERATIONS,
                OBSERVER_PROGRESS_FILE,
                ProgressReporter,
                dpo_judging_fraction,
                dpo_preferences_ready_fraction,
                nonnegative_int,
                training_end_for,
            )
            from lqh.scoring import run_scoring

            client = create_client(self.api_key, self.api_base_url)
            try:
                iteration = int(iter_dir.name.rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                iteration = 0
            n_iterations = nonnegative_int(
                effective_config.get("num_iterations"), DEFAULT_DPO_ITERATIONS,
            )
            training_end = training_end_for(effective_config)
            reporter = ProgressReporter(
                task_kind="dpo", label=self.run_name, run_dir=self.run_dir,
                file_name=OBSERVER_PROGRESS_FILE,
                attempt_id=self._current_attempt_id(),
            )

            def on_progress(completed: int, total: int) -> None:
                reporter.update(
                    phase="preference_scoring",
                    phase_label=f"judging preferences {iteration + 1}/{n_iterations}",
                    completed=completed, total=total, unit="samples",
                    overall_fraction=dpo_judging_fraction(
                        iteration, n_iterations, completed, total, training_end,
                    ),
                    force=completed == total,
                )

            # Step 1: Score predictions
            judge_size = str(effective_config.get("preference_judge_size", "small"))
            score_result = await run_scoring(
                dataset_path=predictions_path,
                scorer_path=self.project_dir / scorer_path,
                output_dir=iter_dir,
                client=client,
                model_size=judge_size,
                run_inference=False,
                on_progress=on_progress,
            )
            if score_result.scored == 0:
                logger.warning("All preference judge calls failed for %s", iter_dir)
                return _SCORE_FAILED

            if score_result.mean_score is not None:
                self.callbacks.on_iter_scored(
                    self.run_name,
                    iter_dir.name,
                    score_result.mean_score,
                )

            reporter.update(
                phase="preference_assembly",
                phase_label=f"assembling preferences {iteration + 1}/{n_iterations}",
                completed=0, total=1, unit="stage",
                overall_fraction=dpo_judging_fraction(
                    iteration, n_iterations, 1, 1, training_end,
                ),
                force=True,
            )

            # Step 2: Generate golden trajectories + assemble preferences
            chosen_scores = None
            if (
                effective_config.get("selection")
                and effective_config.get("golden_source", "dataset") == "dataset"
            ):
                from lqh.golden import load_or_score_chosen_scores

                cache_value = effective_config.get("chosen_scores_cache_path")
                cache_path = (
                    Path(cache_value)
                    if isinstance(cache_value, str) and cache_value
                    else self.run_dir / "chosen_scores.parquet"
                )
                if not cache_path.is_absolute():
                    cache_path = self.project_dir / cache_path
                chosen_scores = await load_or_score_chosen_scores(
                    dataset_spec=effective_config.get("dataset", ""),
                    scorer_path=self.project_dir / scorer_path,
                    project_dir=self.project_dir,
                    client=client,
                    cache_path=cache_path,
                    model_size=judge_size,
                )

            await generate_golden(
                predictions_path=predictions_path,
                scores_path=iter_dir / "results.parquet",
                dataset_path=effective_config.get("dataset", ""),
                config=effective_config,
                client=client,
                output_dir=iter_dir,
                chosen_scores=chosen_scores,
            )
            reporter.update(
                phase="preference_ready",
                phase_label=f"preferences ready {iteration + 1}/{n_iterations}",
                completed=1, total=1, unit="stage",
                overall_fraction=dpo_preferences_ready_fraction(
                    iteration, n_iterations, training_end,
                ),
                force=True,
            )

            logger.info(
                "Processed DPO iteration %s/%s: mean=%.2f",
                self.run_name,
                iter_dir.name,
                score_result.mean_score or 0,
            )
            return _SCORE_SUCCESS

        except Exception:
            logger.exception("Failed to process iteration %s", iter_dir)
            return _SCORE_FAILED
