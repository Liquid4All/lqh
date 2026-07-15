"""Remote-aware run watcher.

Extends ``RunWatcher`` with sync-from-remote before each poll cycle and
push-to-remote after scoring/golden generation.  The core scoring and
DPO orchestration logic is inherited unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lqh.config import default_api_base_url
from lqh.remote.backend import RemoteBackend
from lqh.train.progress import read_latest_progress, read_latest_status
from lqh.watcher import RunWatcher, WatcherCallbacks

logger = logging.getLogger(__name__)

__all__ = ["RemoteRunWatcher"]


class RemoteRunWatcher(RunWatcher):
    """Watch a remote training run by syncing files before each poll.

    The workflow each cycle:

    1. **Pull** from remote — ``sync_progress()`` copies progress.jsonl
       and signal files (eval_request.json, iter_request.json,
       predictions.parquet) to the local mirror.

    2. **Check + Score** — inherited methods from ``RunWatcher`` read
       the local mirror, detect new requests, score via API, and write
       results (eval_result.json, preferences.parquet) to the local
       mirror.

    3. **Push** results back to the remote so the subprocess can pick
       them up (eval_result.json for SFT, preferences.parquet for DPO).

    4. **Check completion** — using ``backend.is_job_alive()`` instead
       of the local PID check.
    """

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        project_dir: Path,
        api_key: str,
        backend: RemoteBackend,
        remote_run_dir: str,
        job_id: str,
        api_base_url: str | None = None,
        callbacks: WatcherCallbacks | None = None,
        poll_interval: float = 30.0,
    ) -> None:
        super().__init__(
            run_dir=run_dir,
            config=config,
            project_dir=project_dir,
            api_key=api_key,
            api_base_url=api_base_url if api_base_url is not None else default_api_base_url(),
            callbacks=callbacks,
            poll_interval=poll_interval,
        )
        self._backend = backend
        self._remote_run_dir = remote_run_dir
        self._job_id = job_id

        # Track which result files we've already pushed to the remote
        self._pushed_eval_results: set[str] = set()
        self._pushed_preferences: set[str] = set()
        self._pushed_preference_errors: set[str] = set()
        self._pushed_held_out_results: set[str] = set()

    # ------------------------------------------------------------------
    # Override the main loop
    # ------------------------------------------------------------------

    async def _watch_loop(self) -> None:
        # Cloud backends inject LQH_API_TOKEN into the sandbox so the
        # trainer can do its own scoring + golden assembly via
        # lqh.train.cloud_score. When that path is active, the laptop
        # must NOT also try to score (it would race the sandbox and
        # also fail to push results back — CloudBackend.sync_file_to_remote
        # raises NotImplementedError). The check goes by backend
        # identity: a backend that reports a non-empty `inline_scoring`
        # attribute as True is doing its own scoring. SSH backends
        # don't set it; CloudBackend sets it to True.
        sandbox_scores_inline = bool(getattr(self._backend, "inline_scoring", False))

        while not self._stop.is_set():
            try:
                # 1. Pull from remote
                await self._sync_from_remote()

                # 2. Check progress + scoring (inherited).
                #    Skip the scoring half when the sandbox is doing
                #    it itself — we still want progress updates so the
                #    TUI continues to display live status.
                self._update_progress()
                if not sandbox_scores_inline:
                    await self._check_eval_requests()
                    await self._check_iter_requests()

                # 3. Push results back to remote. Also skipped in the
                #    sandbox-scores-inline path — there's nothing to
                #    push, and the backend would reject it anyway.
                if not sandbox_scores_inline:
                    await self._push_results()

                # 4. Check completion (remote-aware)
                await self._check_completion_remote()

            except Exception:
                logger.exception(
                    "Remote watcher error for run %s", self.run_name,
                )

            try:
                import asyncio
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.poll_interval,
                )
                break  # stop was set
            except asyncio.TimeoutError:
                pass  # normal poll interval

    # ------------------------------------------------------------------
    # Remote sync
    # ------------------------------------------------------------------

    async def _sync_from_remote(self) -> None:
        """Pull progress and signal files from the remote."""
        try:
            await self._backend.sync_progress(
                self._remote_run_dir, str(self.run_dir),
            )
        except Exception:
            logger.warning(
                "Failed to sync progress from remote for %s",
                self.run_name,
                exc_info=True,
            )

    async def _push_results(self) -> None:
        """Push scoring results back to the remote.

        After the inherited ``_check_eval_requests()`` writes
        ``eval_result.json`` or ``_check_iter_requests()`` writes
        ``preferences.parquet``, we detect those new local files and
        rsync them to the corresponding remote paths.
        """
        await self._push_eval_results()
        await self._push_preferences()
        await self._push_preference_errors()
        await self._push_held_out_results()

    async def _push_eval_results(self) -> None:
        """Push new eval_result.json files for SFT checkpoint scoring."""
        checkpoints_dir = self.run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return

        for cp_dir in sorted(checkpoints_dir.iterdir()):
            if not cp_dir.is_dir():
                continue
            result_file = cp_dir / "eval_result.json"
            key = str(result_file)
            if result_file.exists() and key not in self._pushed_eval_results:
                self._pushed_eval_results.add(key)
                remote_path = (
                    f"{self._remote_run_dir}/checkpoints/"
                    f"{cp_dir.name}/eval_result.json"
                )
                try:
                    await self._backend.sync_file_to_remote(
                        str(result_file), remote_path,
                    )
                    logger.info(
                        "Pushed eval_result.json for %s/%s",
                        self.run_name, cp_dir.name,
                    )
                except Exception:
                    logger.warning(
                        "Failed to push eval_result.json for %s/%s",
                        self.run_name, cp_dir.name,
                        exc_info=True,
                    )
                    # Remove from pushed so we retry next cycle
                    self._pushed_eval_results.discard(key)

    async def _push_preferences(self) -> None:
        """Push new preferences.parquet files for DPO iterations."""
        effective = (
            self.config.get("base_config", {})
            if self.config.get("type") == "sweep"
            else self.config
        )
        if effective.get("type") not in ("on_policy_dpo", "dpo"):
            return
        for iter_dir in self._iteration_dirs():
            prefs_file = iter_dir / "preferences.parquet"
            key = str(prefs_file)
            if prefs_file.exists() and key not in self._pushed_preferences:
                self._pushed_preferences.add(key)
                relative = prefs_file.relative_to(self.run_dir).as_posix()
                remote_path = f"{self._remote_run_dir}/{relative}"
                try:
                    await self._backend.sync_file_to_remote(
                        str(prefs_file), remote_path,
                    )
                    logger.info(
                        "Pushed preferences.parquet for %s/%s",
                        self.run_name, iter_dir.name,
                    )
                except Exception:
                    logger.warning(
                        "Failed to push preferences.parquet for %s/%s",
                        self.run_name, iter_dir.name,
                        exc_info=True,
                    )
                    self._pushed_preferences.discard(key)

    async def _push_preference_errors(self) -> None:
        """Wake an SSH DPO trainer after terminal preference-scoring failure."""
        for iter_dir in self._iteration_dirs():
            error_file = iter_dir / "preference_error.json"
            key = str(error_file)
            if not error_file.exists() or key in self._pushed_preference_errors:
                continue
            self._pushed_preference_errors.add(key)
            relative = error_file.relative_to(self.run_dir).as_posix()
            remote_path = f"{self._remote_run_dir}/{relative}"
            try:
                await self._backend.sync_file_to_remote(
                    str(error_file), remote_path,
                )
            except Exception:
                self._pushed_preference_errors.discard(key)
                logger.warning(
                    "Failed to push preference_error.json for %s/%s",
                    self.run_name, iter_dir.name, exc_info=True,
                )

    async def _push_held_out_results(self) -> None:
        """Push fixed-validation judge summaries back to an SSH trainer."""
        for iter_dir in self._iteration_dirs():
            summary = iter_dir / "held_out_eval" / "summary.json"
            key = str(summary)
            if not summary.exists() or key in self._pushed_held_out_results:
                continue
            self._pushed_held_out_results.add(key)
            relative = summary.relative_to(self.run_dir).as_posix()
            remote_path = f"{self._remote_run_dir}/{relative}"
            try:
                await self._backend.sync_file_to_remote(str(summary), remote_path)
            except Exception:
                self._pushed_held_out_results.discard(key)
                logger.warning(
                    "Failed to push held-out summary for %s/%s",
                    self.run_name,
                    iter_dir.name,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Remote-aware completion check
    # ------------------------------------------------------------------

    async def _check_completion_remote(self) -> None:
        """Check for job completion using the remote backend.

        Handles the race where the remote subprocess finishes between our
        sync_from_remote() at the start of the cycle and is_job_alive()
        at the end: we resync once more before declaring "no terminal
        status" so late-arriving progress + predictions don't get stranded.
        """
        latest = read_latest_status(self.run_dir)
        if latest:
            if latest.get("status") == "completed":
                if (
                    not bool(getattr(self._backend, "inline_scoring", False))
                    and self._has_pending_scoring_requests()
                ):
                    return
                self.callbacks.on_training_completed(self.run_name)
                self._stop.set()
                return
            if latest.get("status") in {"failed", "interrupted"}:
                self.callbacks.on_training_failed(
                    self.run_name, latest.get("error"),
                )
                self._stop.set()
                return

        # Check if the remote process is still alive
        try:
            alive = await self._backend.is_job_alive(self._job_id)
        except Exception:
            logger.warning(
                "Failed to check remote job liveness for %s",
                self.run_name,
                exc_info=True,
            )
            return  # Don't declare dead on transient SSH failure

        if alive:
            return  # still running

        # Remote process is gone but our mirror has no terminal status.
        # Almost certainly a race: the subprocess wrote
        # predictions.parquet + eval_request.json + status=completed and
        # exited *after* our cycle-start sync but *before* this check.
        # Resync once more, run a final scoring pass, and re-read
        # progress before deciding it actually failed.
        try:
            await self._sync_from_remote()
            await self._check_eval_requests()
        except Exception:
            logger.warning(
                "Final resync failed for %s", self.run_name, exc_info=True,
            )

        latest = read_latest_status(self.run_dir)
        if latest and latest.get("status") == "completed":
            if (
                not bool(getattr(self._backend, "inline_scoring", False))
                and self._has_pending_scoring_requests()
            ):
                return
            self.callbacks.on_training_completed(self.run_name)
            self._stop.set()
            return
        if latest and latest.get("status") in {"failed", "interrupted"}:
            self.callbacks.on_training_failed(
                self.run_name, latest.get("error"),
            )
            self._stop.set()
            return

        # Remote process is gone and post-resync still no terminal status.
        # Declare failure unconditionally — if progress.jsonl was never
        # written at all (early crash) we'd otherwise loop forever.
        self.callbacks.on_training_failed(
            self.run_name,
            "Remote process exited without writing final status",
        )
        self._stop.set()
