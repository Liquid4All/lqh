"""Async file watcher for training run orchestration.

Runs as an ``asyncio.Task`` in the main lqh process event loop alongside
the agent.  Polls the filesystem for signal files written by training
subprocesses and responds by running scoring, generating golden
trajectories, or assembling preference pairs.

Never imports torch or transformers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from lqh.config import default_api_base_url
from lqh.subprocess_manager import SubprocessManager
from lqh.train.progress import read_latest_progress

logger = logging.getLogger(__name__)

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
        latest = read_latest_progress(self.run_dir)
        if latest and "step" in latest:
            self.callbacks.on_training_progress(
                self.run_name,
                step=latest.get("step"),
                loss=latest.get("loss"),
                lr=latest.get("lr"),
                epoch=latest.get("epoch"),
            )

    def _check_completion(self) -> None:
        latest = read_latest_progress(self.run_dir)
        if latest:
            if latest.get("status") == "completed":
                self.callbacks.on_training_completed(self.run_name)
                self._stop.set()
            elif latest.get("status") == "failed":
                self.callbacks.on_training_failed(
                    self.run_name, latest.get("error")
                )
                self._stop.set()
        elif not self._manager.is_alive(self.run_dir):
            # Process died without writing status
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
        key = str(self.run_dir)
        if (
            request_file.exists()
            and not result_file.exists()
            and key not in self._processed_eval_requests
        ):
            self._processed_eval_requests.add(key)
            await self._score_checkpoint(self.run_dir)

        # Per-checkpoint requests from training runs.
        checkpoints_dir = self.run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return

        for cp_dir in sorted(checkpoints_dir.iterdir()):
            if not cp_dir.is_dir():
                continue
            request_file = cp_dir / "eval_request.json"
            result_file = cp_dir / "eval_result.json"
            key = str(cp_dir)

            if (
                request_file.exists()
                and not result_file.exists()
                and key not in self._processed_eval_requests
            ):
                self._processed_eval_requests.add(key)
                await self._score_checkpoint(cp_dir)

    async def _score_checkpoint(self, checkpoint_dir: Path) -> None:
        """Score predictions from a checkpoint and write eval_result.json."""
        predictions_path = checkpoint_dir / "predictions.parquet"
        if not predictions_path.exists():
            logger.warning("No predictions.parquet in %s", checkpoint_dir)
            return

        scorer_path = self.config.get("scorer")
        if not scorer_path:
            logger.warning("No scorer configured, skipping eval for %s", checkpoint_dir)
            return

        try:
            from lqh.client import create_client
            from lqh.scoring import run_scoring

            client = create_client(self.api_key, self.api_base_url)

            result = await run_scoring(
                dataset_path=predictions_path,
                scorer_path=self.project_dir / scorer_path,
                output_dir=checkpoint_dir,
                client=client,
                run_inference=False,
            )

            # Write eval_result.json
            eval_result = {
                "num_scored": result.scored,
                "num_failed": result.failed,
                "scores": {
                    "mean": result.mean_score,
                    "median": result.median_score,
                },
            }
            (checkpoint_dir / "eval_result.json").write_text(
                json.dumps(eval_result, indent=2) + "\n"
            )

            if result.mean_score is not None:
                self.callbacks.on_eval_scored(
                    self.run_name,
                    checkpoint_dir.name,
                    result.mean_score,
                )
                logger.info(
                    "Scored %s/%s: mean=%.2f",
                    self.run_name, checkpoint_dir.name, result.mean_score,
                )

        except Exception:
            logger.exception("Failed to score %s", checkpoint_dir)

    # ------------------------------------------------------------------
    # DPO: iteration requests
    # ------------------------------------------------------------------

    async def _check_iter_requests(self) -> None:
        """Check for unprocessed iter_request.json files in iterations."""
        if self.config.get("type") not in ("on_policy_dpo", "dpo"):
            return

        iterations_dir = self.run_dir / "iterations"
        if not iterations_dir.exists():
            return

        for iter_dir in sorted(iterations_dir.iterdir()):
            if not iter_dir.is_dir():
                continue
            request_file = iter_dir / "iter_request.json"
            preferences_file = iter_dir / "preferences.parquet"
            key = str(iter_dir)

            if (
                request_file.exists()
                and not preferences_file.exists()
                and key not in self._processed_iter_requests
            ):
                self._processed_iter_requests.add(key)
                await self._process_iteration(iter_dir)

    async def _process_iteration(self, iter_dir: Path) -> None:
        """Score predictions, generate golden trajectories, assemble preferences."""
        predictions_path = iter_dir / "predictions.parquet"
        if not predictions_path.exists():
            logger.warning("No predictions.parquet in %s", iter_dir)
            return

        scorer_path = self.config.get("scorer")
        if not scorer_path:
            logger.warning("No scorer configured for DPO iteration")
            return

        try:
            from lqh.client import create_client
            from lqh.golden import generate_golden
            from lqh.scoring import run_scoring

            client = create_client(self.api_key, self.api_base_url)

            # Step 1: Score predictions
            score_result = await run_scoring(
                dataset_path=predictions_path,
                scorer_path=self.project_dir / scorer_path,
                output_dir=iter_dir,
                client=client,
                run_inference=False,
            )

            if score_result.mean_score is not None:
                self.callbacks.on_iter_scored(
                    self.run_name,
                    iter_dir.name,
                    score_result.mean_score,
                )

            # Step 2: Generate golden trajectories + assemble preferences
            await generate_golden(
                predictions_path=predictions_path,
                scores_path=iter_dir / "results.parquet",
                dataset_path=self.config.get("dataset", ""),
                config=self.config,
                client=client,
                output_dir=iter_dir,
            )

            logger.info(
                "Processed DPO iteration %s/%s: mean=%.2f",
                self.run_name,
                iter_dir.name,
                score_result.mean_score or 0,
            )

        except Exception:
            logger.exception("Failed to process iteration %s", iter_dir)
