"""Headless background-job supervisor (CLI_PLAN §4.5, phase 4).

Extracted from ``LqhApp`` so the TUI and headless surfaces (`lqh run`,
`lqh tool call training_status --wait`) share ONE implementation of:

- the job registry (``BackgroundTaskRegistry`` + last-state tracking),
- the periodic scan/watch loop that detects ``running → terminal``
  transitions, finalizes runs (manifests, cloud data-gen downloads),
  spawns scoring/sync watchers, and records completion notices,
- parking (``wait_for_runs``): suspend until a watched run is terminal.

UI and telemetry side effects stay in the caller via ``SupervisorHooks``
— every hook is optional, so a bare ``JobSupervisor(project_dir)`` is a
fully functional headless supervisor. This module must not import the
TUI or telemetry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from lqh.project_identity import cloud_project_key as _ckey
from lqh.background_tasks import BackgroundTask, BackgroundTaskRegistry

if TYPE_CHECKING:
    from lqh.subprocess_manager import SubprocessManager

logger = logging.getLogger("lqh.jobs")

# How often the watch loop rescans runs/.
JOB_POLL_INTERVAL_SEC = 60.0
# A wall-clock gap larger than poll_interval * this factor means the
# machine slept / lost connection — the caller may want to tell the user.
SLEEP_GAP_FACTOR = 2.0
# Bounded grace for the scoring watcher to write eval_result.json after an
# infer run goes terminal, so a parked agent wakes to readable results.
SCORING_GRACE_SEC = 180.0
# A running job that has emitted no progress for this long is very
# likely a stalled sandbox. Deliberately generous: base-model download,
# batch-size calibration, checkpoint evaluation and DPO rollouts all
# have legitimately long quiet stretches, and a false alarm here costs
# the user real GPU time if they act on it.
STALL_NOTICE_MINUTES = 60.0
# A job that has never reported ANY progress gets longer still: cold
# start, image pull and a multi-GB model download all happen before the
# first event, and on a busy provider that can legitimately take a while.
STALL_NOTICE_SILENT_MINUTES = 90.0
# A reattached run cannot emit progress at all, so its frozen file is
# not evidence of a stall — but "cannot prove it is stuck" is not "is
# fine forever". This much quieter threshold still catches a reattached
# job that is genuinely wedged, without warning on every long run that
# outlived a deploy.
STALL_NOTICE_REATTACHED_MINUTES = 240.0
# Marker text the cloud backend appends to progress.jsonl when a
# reattached pump takes over a running sandbox.
_STALE_PROGRESS_MARKER = "progress may be stale"


# Keys that mean a row is REAL progress from the workload, as opposed to
# the lifecycle rows the cloud event translator mirrors into the same
# file (it appends a status=running row the moment the job starts).
# Treating any non-empty progress.jsonl as "has reported progress" put
# every cloud job on the 60-minute threshold, so the 90-minute
# silent-start allowance never applied to the runs it was written for.
_PROGRESS_KEYS = ("step", "completed", "overall_fraction", "phase")


def _has_real_progress(run_dir: Path) -> bool:
    try:
        from lqh.progress import read_jsonl_tail

        rows = read_jsonl_tail(run_dir / "progress.jsonl", last_n=64)
    except Exception:  # noqa: BLE001 — advisory path
        return False
    return any(
        any(k in row for k in _PROGRESS_KEYS) for row in rows if isinstance(row, dict)
    )


def _progress_is_known_stale(run_dir: Path) -> bool:
    """Whether the newest progress row is the reattach staleness note."""
    try:
        from lqh.progress import read_jsonl_tail

        rows = read_jsonl_tail(run_dir / "progress.jsonl", last_n=8)
    except Exception:  # noqa: BLE001 — advisory path, never fatal
        return False
    for row in reversed(rows):
        detail = row.get("detail")
        if isinstance(detail, str) and _STALE_PROGRESS_MARKER in detail:
            return True
        # Any later real progress row means the marker is history.
        if isinstance(row.get("overall_fraction"), (int, float)):
            return False
    return False

# CloudBackend maps backend "cancelled" → "failed" already; "cancelled" is
# listed defensively so a path that surfaces the raw status still
# finalizes (marker consumption, notification).
TERMINAL_STATES = {"completed", "failed", "cancelled"}


@dataclass
class SupervisorHooks:
    """Optional side-effect hooks. All default to None (headless no-op)."""

    # Registry repaint trigger (TUI: invalidate).
    on_registry_change: Callable[[], None] | None = None
    # The watch loop resumed after a sleep/connection gap.
    on_gap: Callable[[], Awaitable[None]] | None = None
    # A completion notice was recorded for run_name (already in
    # pending_completions). TUI: push into the input queue + status bar.
    on_notice: Callable[[str, str, str], None] | None = None  # (run, text, state)
    # A run was observed running (after registry registration). TUI:
    # rehydrate telemetry job record + progress refresh.
    on_running: Callable[[str, str | None], None] | None = None
    # A run left the running set. TUI: clear progress caches.
    on_terminal: Callable[[str], None] | None = None
    # Whether the caller holds a job record that warrants a completion
    # notification even without an observed running→terminal transition
    # (TUI: persisted telemetry job record). Args: (run_name,
    # first_observation) — first_observation is True when the supervisor
    # has no prior observed state for the run.
    has_job_record: Callable[[str, bool], bool] | None = None
    # Telemetry mirror of a training/eval completion (the supervisor
    # already wrote the manifest + project-log event).
    on_record_completion: Callable[[str, str, str | None, str | None], None] | None = None
    # Telemetry mirror of a cloud data-gen terminal outcome.
    # (outcome "succeeded"|"failed", workflow_id, marker dict)
    on_data_gen_terminal: Callable[[str, str, dict], None] | None = None


class JobSupervisor:
    """Owns background-run supervision for one project directory."""

    def __init__(
        self,
        project_dir: Path,
        *,
        hooks: SupervisorHooks | None = None,
        poll_interval: float = JOB_POLL_INTERVAL_SEC,
    ) -> None:
        self.project_dir = project_dir
        self.hooks = hooks or SupervisorHooks()
        self.poll_interval = poll_interval
        self.tasks = BackgroundTaskRegistry(
            on_change=self.hooks.on_registry_change
        )
        # Last observed state per run; the running→terminal transition detector.
        self.job_last_state: dict[str, str] = {}
        # Completion notice text keyed by run, popped by parking.
        self.pending_completions: dict[str, str] = {}
        # Live RunWatcher/RemoteRunWatcher per run.
        self.run_watchers: dict[str, Any] = {}
        # Cloud data-gen runs whose download gave up this session.
        self.data_gen_gave_up: set[str] = set()
        # Computed interruption taxonomy per run (lqh.remote.failure),
        # captured on the poll that observed the terminal state and used
        # to write the recovery directive into the completion notice.
        self.job_failure: dict[str, dict[str, Any]] = {}
        # Runs we've already warned about stalling, so the notice fires
        # once per run per process rather than every poll.
        self.stall_notified: set[str] = set()
        # The advisory text per stalled run, folded into that run's
        # completion notice so a parked agent hears about it too.
        self.stall_advisories: dict[str, str] = {}
        # Per-run verdict from finalize_eval_hf_run, consumed by the
        # watch loop to keep the recorded state consistent with the
        # notice: "ok" | "missing_result" | "unverified".
        self.eval_hf_verdicts: dict[str, str] = {}
        # Wake signal for wait_for_runs; set on every recorded completion.
        self._completion_signal = asyncio.Event()
        # Set after the first scan so a fresh process can wait for the
        # registry to reflect reality before deciding "nothing running".
        self._primed = asyncio.Event()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_started(
        self, task_id: str, kind: str, label: str, remote: str | None,
    ) -> None:
        """A tool just submitted a job that will notify later."""
        self.tasks.register(BackgroundTask(
            task_id=task_id,
            kind=kind,
            label=label,
            state="running",
            remote=remote,
        ))
        # Seed the watcher's last-known state so a short run that finishes
        # before the first scan is still seen as a running -> terminal
        # transition (otherwise no completion is recorded and parking waits
        # for the full safety interval). Also drop any stale completion left
        # over from an earlier run that reused this name.
        self.job_last_state[task_id] = "running"
        self.pending_completions.pop(task_id, None)

    def ensure_task_registered(self, run_name: str, remote: str | None) -> None:
        """Register a task for a live run if not already in the registry.

        Used by the watch loop to recover after a restart, since
        handler-driven eager registration only fires on the original
        submission.
        """
        if any(t.task_id == run_name for t in self.tasks.snapshot()):
            return
        config_path = self.project_dir / "runs" / run_name / "config.json"
        kind = "train"
        if config_path.exists():
            try:
                run_type = json.loads(config_path.read_text()).get("type", "")
                kind = "eval" if run_type in {"infer", "eval_hf"} else "train"
            except Exception:
                pass
        self.tasks.register(BackgroundTask(
            task_id=run_name,
            kind=kind,
            label=run_name,
            state="running",
            remote=remote,
        ))

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    async def scan_jobs(
        self, manager: "SubprocessManager",
    ) -> list[tuple[str, str, str | None, str | None]]:
        """Return ``(run_name, state, error, remote_name)`` for every run dir.

        Local runs are queried via ``SubprocessManager.get_status`` (PID +
        progress.jsonl). Remote runs (``remote_job.json`` present) are
        probed over SSH via the backend's ``poll_status``.
        """
        runs_dir = self.project_dir / "runs"
        if not runs_dir.is_dir():
            return []

        results: list[tuple[str, str, str | None, str | None]] = []
        for entry in sorted(runs_dir.iterdir()):
            if not entry.is_dir() or not (entry / "config.json").exists():
                continue

            remote_meta = entry / "remote_job.json"
            if remote_meta.exists():
                try:
                    meta = json.loads(remote_meta.read_text())
                    from lqh.project_identity import marker_is_foreign

                    if marker_is_foreign(self.project_dir, meta):
                        # Run dir copied in from another project — never
                        # poll/watch someone else's job from here.
                        logger.warning(
                            "runs/%s: remote_job.json belongs to another "
                            "project identity; skipping", entry.name,
                        )
                        continue
                    state, error = await self.poll_remote(entry, meta)
                    results.append((entry.name, state, error, meta["remote_name"]))
                except Exception:
                    results.append((entry.name, "unknown", None, None))
                continue

            status = manager.get_status(entry)
            results.append((entry.name, status.state, status.error, None))

        return results

    def _record_terminal_locally(self, run_dir: Path, status: Any) -> None:
        """Mirror a backend-reported terminal state into the run dir.

        Best-effort and idempotent: the startup signals read run-dir
        files, not the backend, so a terminal the event stream never
        carried has to land here or it is invisible to a fresh process.
        """
        try:
            from lqh.remote.cloud import _CloudState

            path = run_dir / "cloud_state.json"
            state = _CloudState.load(path)
            if state is not None and state.status == status.state:
                return
            if state is not None:
                state.status = status.state
                state.save(path)
        except Exception:  # noqa: BLE001 — advisory bookkeeping
            pass
        # progress.jsonl is the other source observe_run_states consults
        # (SSH runs have no cloud_state.json, and cloud runs may have a
        # stale one), and it is what progress_terminal_state reads.
        try:
            from lqh.progress import read_jsonl_tail

            for row in read_jsonl_tail(run_dir / "progress.jsonl", last_n=16):
                if row.get("status") in ("completed", "failed", "cancelled"):
                    return
            with (run_dir / "progress.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "status": status.state,
                    "error": status.error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
        except Exception:  # noqa: BLE001
            pass

    async def poll_remote(
        self, run_dir: Path, meta: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Sync and poll a remote/cloud job. Returns (state, error)."""
        backend = self.make_remote_backend(meta)
        if backend is None:
            return ("unknown", None)
        remote_run_dir = meta.get("remote_run_dir")
        if remote_run_dir:
            await backend.sync_progress(str(remote_run_dir), str(run_dir))
        status = await backend.poll_status(str(meta["job_id"]))
        # Reconcile LOCAL state with what the backend just told us.
        #
        # The orphan reconciler resolves a vanished sandbox by updating
        # the cloud_jobs row; it appends no terminal cloud_job_events
        # entry, so sync_progress has nothing to mirror and this run dir
        # still looks "running" on disk. observe_run_states reads exactly
        # that (cloud_state.json / progress.jsonl), so a session opened
        # after an orphan failure got NEITHER the finished-while-away
        # signal nor the infra_failure one — the two things built to
        # surface it.
        if status.state in ("completed", "failed", "cancelled"):
            self._record_terminal_locally(run_dir, status)
        # getattr, not attribute access: scan_jobs swallows exceptions
        # into state="unknown", so a backend whose JobStatus predates
        # this field must not silently blind the watcher.
        if getattr(status, "failure", None):
            self.job_failure[run_dir.name] = status.failure
            # On disk ONLY for a run that actually failed. A completed
            # run also carries a taxonomy — that is how the success
            # notice knows to mention "interrupted twice, elapsed time is
            # higher" — but writing cloud_failure.json for it leaves a
            # durable file whose name says the run failed, for a run that
            # succeeded. The in-memory copy above is enough: the
            # completion notice is formatted in this same process.
            #
            # lqh/signals.py reads the file to raise a startup signal in
            # a session opened AFTER the failure, which is exactly the
            # case that needs it.
            if status.state in ("failed", "cancelled"):
                try:
                    (run_dir / "cloud_failure.json").write_text(
                        json.dumps(status.failure, indent=2) + "\n"
                    )
                except OSError:
                    pass
        return (status.state, status.error)

    def make_remote_backend(self, meta: dict[str, Any]) -> Any | None:
        """Build the backend described by remote_job.json."""
        remote_name = str(meta.get("remote_name", ""))
        backend_name = str(meta.get("backend", ""))

        try:
            from lqh.remote.compute import is_cloud, ssh_remote_name
            if backend_name == "cloud" or is_cloud(remote_name):
                from lqh.remote.backend import RemoteConfig
                from lqh.remote.cloud import CloudBackend

                cfg = RemoteConfig(
                    name="cloud",
                    type="cloud",
                    hostname="api.lqh.ai",
                    remote_root="cloud:lqh",
                )
                return CloudBackend(cfg, self.project_dir)

            from lqh.remote.config import get_remote
            from lqh.remote.ssh_direct import SSHDirectBackend

            ssh_name = ssh_remote_name(remote_name) or remote_name
            remote_config = get_remote(self.project_dir, ssh_name)
            if remote_config is None or remote_config.type != "ssh_direct":
                return None
            return SSHDirectBackend(remote_config, self.project_dir)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Run metadata helpers
    # ------------------------------------------------------------------

    def run_type(self, run_name: str) -> str:
        """The run's config ``type`` ("sft", "data_gen", ...), "" if unknown."""
        try:
            config = json.loads(
                (self.project_dir / "runs" / run_name / "config.json").read_text()
            )
            return str(config.get("type", ""))
        except Exception:
            return ""

    def data_gen_pending(self, run_name: str) -> bool:
        """Whether a cloud data-gen run still owes finalization.

        The marker is written by the submit handler and removed by
        ``finalize_data_gen_run`` — durable across restarts.
        """
        return (
            self.project_dir / "runs" / run_name / ".lqh_data_gen.json"
        ).exists()

    def results_pending(self, run_name: str) -> bool:
        """Whether a successful process still owes its useful eval result."""
        from lqh.progress import has_pending_final_result

        run_dir = self.project_dir / "runs" / run_name
        config_path = run_dir / "config.json"
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            return False
        return has_pending_final_result(run_dir, config)

    # ------------------------------------------------------------------
    # Watch loop
    # ------------------------------------------------------------------

    async def watch_loop(self) -> None:
        """Periodically scan background runs and record completion notices.

        Detects ``running → completed/failed`` transitions per run; the
        first observation of any run is silent — already-terminal runs at
        startup don't fire.
        """
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        last_wall_time = time.time()

        first_scan = True
        while True:
            if not first_scan:
                try:
                    await asyncio.sleep(self.poll_interval)
                except asyncio.CancelledError:
                    return

            now = time.time()
            if (
                not first_scan
                and now - last_wall_time > self.poll_interval * SLEEP_GAP_FACTOR
                and self.hooks.on_gap is not None
            ):
                await self.hooks.on_gap()
            last_wall_time = now
            first_scan = False

            try:
                snapshots = await self.scan_jobs(manager)
            except Exception:
                # Never let scan errors kill the watcher.
                self._primed.set()
                continue

            # Cull watchers whose runs have finished.
            for name in [
                n for n, w in self.run_watchers.items() if not w.is_running
            ]:
                self.run_watchers.pop(name, None)
                self.tasks.unregister(name)

            for run_name, state, error, remote in snapshots:
                if state == "unknown":
                    # Transient SSH/FS hiccup — don't update last_state,
                    # retry next tick.
                    continue
                if state == "completed" and self.results_pending(run_name):
                    # The process has exited, but the user-facing job has
                    # not: inference/judging still owes its final result
                    # artifact. Demote BEFORE the manifest write below, or
                    # the manifest would be finalized without the final
                    # evaluation artifact and never rewritten.
                    state = "running"
                # Every terminal run gets a finalization manifest exactly
                # once — INDEPENDENT of the notification gates below, so a
                # run first observed terminal after a restart (no
                # running→terminal transition) is still traceable.
                if state in TERMINAL_STATES:
                    run_dir = self.project_dir / "runs" / run_name
                    if not (run_dir / "manifest.json").exists():
                        try:
                            from lqh.manifest import write_run_manifest

                            written = write_run_manifest(
                                self.project_dir, run_dir,
                                state=state, error=error,
                            )
                        except Exception:
                            written = None
                        if written is None:
                            # Surfaced through the activity log — the run
                            # is terminal but not traceable.
                            try:
                                from lqh.project_log import append_event

                                append_event(
                                    self.project_dir, "manifest_write_failed",
                                    f"Provenance manifest could not be written for runs/{run_name}",
                                    run_name=run_name,
                                )
                            except Exception:
                                pass
                prev = self.job_last_state.get(run_name)
                # A caller-held job record (e.g. a persisted telemetry
                # workflow) warrants a completion notification even
                # without an observed running→terminal transition.
                pending_record = (
                    self.hooks.has_job_record is not None
                    and self.hooks.has_job_record(run_name, prev is None)
                )
                # Cloud data-gen runs leave a durable marker at submit; it
                # survives restarts, so a job first observed already
                # terminal still gets finalized (download + notification).
                # Runs whose download gave up this session are excluded
                # until a restart clears the set.
                needs_finalize = (
                    self.data_gen_pending(run_name)
                    and run_name not in self.data_gen_gave_up
                )
                if state in TERMINAL_STATES and (
                    prev == "running" or pending_record or needs_finalize
                ):
                    text: str | None
                    if self.run_type(run_name) == "data_gen":
                        # Cloud data-gen: pull the dataset artifact into
                        # datasets/<name>/ before telling the agent, so
                        # the follow-up (scoring) finds the file locally.
                        # None = transient download failure; the marker is
                        # kept and the next scan retries silently.
                        text = await self.finalize_data_gen_run(
                            run_name, state, error,
                        )
                    elif self.run_type(run_name) == "eval_hf":
                        # Cloud HF eval: completion is only real when the
                        # sandbox published eval_result.json — gate the
                        # success notice on the artifact manifest and
                        # pull the result file down for local reads.
                        text = await self.finalize_eval_hf_run(
                            run_name, state, error, remote,
                        )
                        if (
                            state == "completed"
                            and self.eval_hf_verdicts.pop(run_name, "ok")
                            == "missing_result"
                        ):
                            # Keep the recorded state (manifest, project
                            # log, telemetry hooks) consistent with the
                            # failure-styled notice — a completed-without-
                            # result eval is a failure everywhere, not
                            # just in the message text. "unverified"
                            # (artifact API unreachable) deliberately
                            # stays completed: absence wasn't proven.
                            state = "failed"
                            error = error or "no eval_result.json artifact was published"
                        else:
                            self.eval_hf_verdicts.pop(run_name, None)
                    else:
                        text = self.format_completion_message(
                            run_name, state, error, remote,
                        )
                    if text is not None:
                        # Record the notice keyed by run so parking
                        # delivers it only to the run actually being
                        # waited on.
                        self.record_completion_notice(run_name, text)
                        if self.hooks.on_notice is not None:
                            self.hooks.on_notice(run_name, text, state)
                        if self.run_type(run_name) != "data_gen":
                            # data_gen writes its own project-log event in
                            # finalize_data_gen_run; the generic recorder
                            # would mislabel it training_completed.
                            self.record_completion(run_name, state, error, remote)
                            if self.hooks.on_record_completion is not None:
                                self.hooks.on_record_completion(
                                    run_name, state, error, remote,
                                )
                        self.tasks.unregister(run_name)
                # Keep the registry in sync with live state. The handler
                # eagerly registers on submission; this branch is the
                # fallback for jobs discovered after a restart.
                if state == "running":
                    self.ensure_task_registered(run_name, remote)
                    if self.hooks.on_running is not None:
                        self.hooks.on_running(run_name, remote)
                    self.maybe_notify_stall(run_name)
                elif state in TERMINAL_STATES:
                    self.stall_notified.discard(run_name)
                    self.tasks.unregister(run_name)
                    if self.hooks.on_terminal is not None:
                        self.hooks.on_terminal(run_name)
                self.job_last_state[run_name] = state

                # Ensure a scoring/sync watcher is attached to runs that may
                # still need work: live runs (rsync + score during run) AND
                # finished runs that don't yet have eval_result.json (handles
                # fast remote inferences that finish before our scan tick,
                # and completed-but-unscored runs after a restart).
                if run_name in self.run_watchers:
                    continue
                if state in ("failed", "cancelled"):
                    continue
                if self.run_type(run_name) == "data_gen":
                    # Data-gen runs have no predictions to score mid-run;
                    # their only follow-up is the dataset download handled
                    # in finalize_data_gen_run above.
                    continue
                run_dir = self.project_dir / "runs" / run_name
                if state == "completed" and (run_dir / "eval_result.json").exists():
                    continue
                try:
                    await self.spawn_run_watcher(run_name, remote)
                except Exception:
                    pass

            # Persist observed states so the next startup can report
            # "finished while LQH was closed" (see lqh/signals.py).
            try:
                from lqh.signals import record_seen_states

                record_seen_states(
                    self.project_dir,
                    {
                        run: st
                        for run, st, _, _ in snapshots
                        if st != "unknown"
                    },
                )
            except Exception:
                pass

            self._primed.set()

    # ------------------------------------------------------------------
    # Stall detection
    # ------------------------------------------------------------------

    def stalled_minutes(self, run_name: str) -> tuple[float, bool] | None:
        """``(idle_minutes, saw_progress)`` for a run, or None if unknown.

        ``saw_progress`` False means the run has never reported anything
        and the clock runs from submission instead — a job wedged during
        model download or setup produces no progress event at all, which
        is exactly the case a "last progress was N minutes ago" check
        would miss forever.
        """
        run_dir = self.project_dir / "runs" / run_name
        progress = run_dir / "progress.jsonl"
        reattached = _progress_is_known_stale(run_dir)
        try:
            stat = progress.stat()
            if stat.st_size > 0:
                idle = max(0.0, (time.time() - stat.st_mtime) / 60.0)
                # A reattached pump never re-streams trainer stdout
                # (lqh/remote/cloud.py: _append_stale_progress_marker
                # says so in the row it appends), so the file freezes
                # whether or not the job is healthy. Report it as
                # "never reported progress", which routes it to the much
                # longer threshold instead of exempting it forever.
                #
                # And a file containing only the translator's lifecycle
                # rows is not progress either: the cloud path writes a
                # status=running row at submit, so "non-empty" was true
                # for every cloud job from second one.
                return idle, (not reattached) and _has_real_progress(run_dir)
        except OSError:
            pass
        for anchor in (run_dir / "remote_job.json", run_dir / "config.json"):
            try:
                return max(0.0, (time.time() - anchor.stat().st_mtime) / 60.0), False
            except OSError:
                continue
        return None

    def maybe_notify_stall(self, run_name: str) -> None:
        """Push a one-shot advisory when a running job stops moving.

        Deliberately NOT a completion notice: it never enters
        ``pending_completions``, so ``wait_for_runs`` keeps blocking and
        a parked auto-mode agent stays parked at zero LLM cycles. The
        job is still registered as running and nothing has been
        cancelled — this is information, not a state change.
        """
        if run_name in self.stall_notified:
            return
        # Durable across restarts: both collections are in-memory, so a
        # CLI restart used to re-notify about the same stalled run and
        # an advisory raised before the restart could never be folded
        # into the completion notice that arrived after it.
        stamp = self.project_dir / "runs" / run_name / ".lqh_stall_notice.json"
        if stamp.exists():
            self.stall_notified.add(run_name)
            try:
                saved = json.loads(stamp.read_text())
                if isinstance(saved.get("text"), str):
                    self.stall_advisories[run_name] = saved["text"]
            except (OSError, ValueError):
                pass
            return
        observed = self.stalled_minutes(run_name)
        if observed is None:
            return
        idle, saw_progress = observed
        if saw_progress:
            threshold = STALL_NOTICE_MINUTES
        elif _progress_is_known_stale(self.project_dir / "runs" / run_name):
            threshold = STALL_NOTICE_REATTACHED_MINUTES
        else:
            threshold = STALL_NOTICE_SILENT_MINUTES
        if idle < threshold:
            return
        self.stall_notified.add(run_name)
        stage = ""
        try:
            from lqh.train.progress import read_latest_metrics

            latest = read_latest_metrics(self.project_dir / "runs" / run_name) or {}
            label = latest.get("phase_label") or latest.get("phase")
            if label:
                stage = f" (last: {label})"
        except Exception:  # noqa: BLE001 — advisory text only
            pass
        what = (
            f"has produced no progress for {int(idle)} minutes{stage}"
            if saw_progress
            else f"has not reported any progress in the {int(idle)} minutes "
            "since it was submitted"
        )
        text = (
            f"[System: training run {run_name} {what}. It is still registered as "
            "running and nothing has been cancelled — this usually means a "
            "stalled sandbox. Tell the user in one sentence and offer "
            f"stop_training(run_name='{run_name}') if they would rather stop "
            "paying for it. Otherwise keep waiting: do NOT call training_status "
            "in a loop, and do not start a second run of the same job.]"
        )
        # Durable, because the immediate hook reaches the three modes
        # very unevenly:
        #
        #   interactive TUI — queued as input, seen on the next turn;
        #   auto mode       — parked in wait_for_runs, which explicitly
        #                     drops [System: ...] input
        #                     (tui/app.py _pause_for_input);
        #   headless run    — surfaced on the NDJSON event stream for
        #                     the wrapping harness, but the agent itself
        #                     is mid-task with nothing consuming input.
        #
        # So in two of three modes the agent only learns about it at
        # wake, attached to the completion it does receive. That is
        # bounded rather than unbounded — a cloud job always terminates,
        # at its wall-clock cap if nothing else — but it is a real
        # latency, not a delivery guarantee.
        self.stall_advisories[run_name] = text
        try:
            (self.project_dir / "runs" / run_name / ".lqh_stall_notice.json").write_text(
                json.dumps({
                    "text": text,
                    "at": datetime.now(timezone.utc).isoformat(),
                }) + "\n"
            )
        except OSError:
            pass
        if self.hooks.on_notice is not None:
            self.hooks.on_notice(run_name, text, "stalled")

    # ------------------------------------------------------------------
    # Completion recording / formatting
    # ------------------------------------------------------------------

    def format_completion_message(
        self, run_name: str, state: str, error: str | None, remote: str | None,
    ) -> str:
        location = f" on remote '{remote}'" if remote else ""
        # status derives the remote from the run's remote_job.json, so the
        # call never needs a remote argument.
        status_call = f"training_status(run_name='{run_name}')"
        # The recovery directive is COMPUTED from the job's lease history
        # (lqh/remote/failure.py) rather than left to the agent to recall:
        # an interruption can land at any point in a session, including
        # after a compaction dropped whatever guidance it once read.
        from lqh.remote.failure import (
            completion_note,
            describe_failure,
            failure_from_dict,
        )

        # A run that went quiet for an hour and then ended is a different
        # story from one that ended promptly, and the agent parked in
        # wait_for_runs never saw the live advisory (see maybe_notify_stall).
        stall = self.stall_advisories.pop(run_name, None)
        stall_note = (
            " Note: this run stopped reporting progress well before it ended "
            "(the harness flagged it as stalled), so some of the elapsed GPU "
            "time produced nothing."
            if stall
            else ""
        )
        failure = failure_from_dict(self.job_failure.get(run_name))
        if failure is None:
            failure_marker = (
                self.project_dir / "runs" / run_name / "cloud_failure.json"
            )
            if failure_marker.exists():
                try:
                    failure = failure_from_dict(json.loads(failure_marker.read_text()))
                except (OSError, json.JSONDecodeError):
                    failure = None
        if state == "completed":
            run_dir = self.project_dir / "runs" / run_name
            scoring_failed = any(path.exists() for path in (
                run_dir / "eval_error.json",
                run_dir / "checkpoints" / "final" / "eval_error.json",
            ))
            if scoring_failed:
                error_message = "final evaluation failed"
                for marker in (
                    run_dir / "eval_error.json",
                    run_dir / "checkpoints" / "final" / "eval_error.json",
                ):
                    if marker.exists():
                        try:
                            payload = json.loads(marker.read_text())
                            error_message = str(payload.get("error", error_message))
                        except (OSError, json.JSONDecodeError):
                            pass
                        break
                return (
                    f"[System: run {run_name} finished{location}, but its final "
                    f"evaluation failed: {error_message}. Call {status_call} to inspect "
                    "the completed model and scoring error, then decide whether to retry.]"
                )
            return (
                f"[System: training run {run_name} completed successfully{location}."
                f"{completion_note(failure)}{stall_note} "
                f"Call {status_call} now to read final details, then continue with "
                "the natural next step.]"
            )
        err_part = f": {error}" if error else "."
        diagnosis = describe_failure(failure, run_name)
        if not diagnosis:
            # Unclassifiable (no snapshot, or a backend that predates the
            # recovery fields) — today's message, unchanged.
            return (
                f"[System: training run {run_name} failed{location}{err_part}{stall_note} "
                f"Call {status_call} now to read final details, then explain the failure "
                "and the natural recovery step.]"
            )
        return (
            f"[System: training run {run_name} failed{location}{err_part} {diagnosis}"
            f"{stall_note} "
            f"Call {status_call} now for the full attempt history, then follow the "
            "recovery step above and explain it to the user in one or two "
            "sentences. Do not call this a generic transient issue and do not "
            "retry the same job shape blindly.]"
        )

    def eval_hf_result_artifact(self, run_name: str) -> dict | None:
        """The artifacts.json entry for the run's eval_result.json, or
        None. The manifest is appended from the sandbox publisher's
        artifact events (remote/cloud.py), so an entry here means the
        result actually made it to storage.
        """
        path = self.project_dir / "runs" / run_name / "artifacts.json"
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        entries = manifest.get("artifacts") if isinstance(manifest, dict) else None
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if (
                isinstance(entry, dict)
                and entry.get("kind") == "eval_result"
                and entry.get("relpath") == "eval_result.json"
                and entry.get("artifact_id")
            ):
                return entry
        return None

    def _cloud_job_id(self, run_name: str) -> str:
        """The backend job UUID for a run, from remote_job.json ('' if absent)."""
        meta_path = self.project_dir / "runs" / run_name / "remote_job.json"
        try:
            return str(json.loads(meta_path.read_text()).get("job_id") or "")
        except (OSError, json.JSONDecodeError):
            return ""

    def _run_artifact_entry(self, run_name: str, relpath: str) -> dict | None:
        """The artifacts.json entry with the given relpath, or None."""
        path = self.project_dir / "runs" / run_name / "artifacts.json"
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        entries = manifest.get("artifacts") if isinstance(manifest, dict) else None
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if (
                isinstance(entry, dict)
                and entry.get("relpath") == relpath
                and entry.get("artifact_id")
            ):
                return entry
        return None

    async def _download_run_results_parquet(self, run_name: str) -> None:
        """Best-effort local copy of the run's results.parquet (per-sample
        scores + judge reasoning), so get_eval_failures / failure-analysis
        browsing works right after a cloud eval. Falls back to the backend
        artifact API when the SSE-fed manifest missed the publish event.
        Never raises — the aggregate notice must not fail over a missing
        per-sample file.
        """
        run_dir = self.project_dir / "runs" / run_name
        target = run_dir / "results.parquet"
        if target.exists():
            return
        entry = self._run_artifact_entry(run_name, "results.parquet")
        artifact_id = entry.get("artifact_id") if entry else None
        try:
            from lqh.artifacts import BackendArtifactStore

            store = BackendArtifactStore()
            if artifact_id is None:
                job_id = self._cloud_job_id(run_name)
                if not job_id:
                    return
                handles = await store.list_for_project(
                    _ckey(self.project_dir), kind="predictions", job_id=job_id,
                )
                for handle in handles:
                    if handle.r2_key.endswith("results.parquet"):
                        artifact_id = handle.id
                        break
            if artifact_id is None:
                return
            await store.download(str(artifact_id), target)
        except Exception:
            return

    async def resolve_eval_hf_result_artifact(
        self, run_name: str,
    ) -> tuple[dict | None, bool]:
        """(entry, verified) for the run's eval_result.json artifact.

        The local artifacts.json manifest only sees artifact events that
        were streamed over SSE — after a backend restart the reattached
        pump never replays them (Modal reattach doesn't re-stream
        stdout), so a missing local entry is NOT proof of failure. When
        the manifest has no entry, ask the backend artifact API before
        rendering a verdict. verified=False means the API couldn't be
        reached and the absence is inconclusive.
        """
        entry = self.eval_hf_result_artifact(run_name)
        if entry is not None:
            return entry, True
        job_id = self._cloud_job_id(run_name)
        if not job_id:
            # Never reached the backend → nothing can have been published.
            return None, True
        try:
            from lqh.artifacts import BackendArtifactStore

            handles = await BackendArtifactStore().list_for_project(
                _ckey(self.project_dir), kind="eval_result", job_id=job_id,
            )
        except Exception:
            return None, False
        for handle in handles:
            if handle.r2_key.endswith("eval_result.json"):
                entry = {
                    "artifact_id": handle.id,
                    "kind": "eval_result",
                    "relpath": "eval_result.json",
                }
                # Backfill the manifest so later checks stay local.
                try:
                    from lqh.remote.cloud import _append_artifact_manifest

                    _append_artifact_manifest(
                        self.project_dir / "runs" / run_name / "artifacts.json",
                        entry,
                    )
                except Exception:
                    pass
                return entry, True
        return None, True

    async def finalize_eval_hf_run(
        self, run_name: str, state: str, error: str | None, remote: str | None,
    ) -> str | None:
        """Completion notice for a cloud eval_hf run.

        Unlike training runs, an eval_hf run's only real output is
        eval_result.json — a "completed" state without that artifact is
        a failure and must be reported as one (belt-and-braces: the
        backend's completion gate should already have flipped such jobs
        to failed). When the artifact exists, best-effort download it so
        training_status and follow-up reads find the file locally.
        Records the verdict in self.eval_hf_verdicts so the watch loop
        keeps the recorded state consistent with the notice.
        """
        if state != "completed":
            self.eval_hf_verdicts[run_name] = "ok"
            return self.format_completion_message(run_name, state, error, remote)

        location = f" on remote '{remote}'" if remote else ""
        status_call = f"training_status(run_name='{run_name}')"
        run_dir = self.project_dir / "runs" / run_name
        entry, verified = await self.resolve_eval_hf_result_artifact(run_name)
        if entry is None and not verified:
            # Inconclusive: the backend says completed and we couldn't
            # reach the artifact API to double-check. Don't claim
            # failure on a network blip — keep completed with a caveat.
            # Deliberate trade-off: against a pre-gate backend this
            # leaves a false-completion window, but flipping to failed
            # here would manufacture false FAILURES on every transient
            # API hiccup against gated backends — the far more common
            # case. The caveat text routes the user to verify.
            self.eval_hf_verdicts[run_name] = "unverified"
            return (
                f"[System: eval run {run_name} completed{location}, but the "
                "result artifact could not be verified (artifact API "
                f"unreachable). Call {status_call} to confirm and fetch "
                f"runs/{run_name}/eval_result.json via the artifacts tool.]"
            )
        if entry is None:
            self.eval_hf_verdicts[run_name] = "missing_result"
            return (
                f"[System: eval run {run_name} reported completed{location}, but "
                "no eval_result.json artifact was published — treat it as failed. "
                f"Call {status_call} and check stdout.log / eval_error.json for "
                "the scoring or publish error, then decide whether to retry.]"
            )
        self.eval_hf_verdicts[run_name] = "ok"
        # Per-sample scores for get_eval_failures / failure analysis.
        await self._download_run_results_parquet(run_name)

        result_path = run_dir / "eval_result.json"
        score_part = ""
        if not result_path.exists():
            try:
                from lqh.artifacts import BackendArtifactStore

                await BackendArtifactStore().download(
                    str(entry["artifact_id"]), result_path,
                )
            except Exception:
                # The result exists in storage; only the local copy is
                # missing. Not worth a failure-styled notice.
                return (
                    f"[System: eval run {run_name} completed{location}. "
                    "Downloading eval_result.json failed; use the artifacts "
                    f"tool or {status_call} to read the scores.]"
                )
        try:
            summary = json.loads(result_path.read_text())
            scores = summary.get("scores") if isinstance(summary, dict) else None
            mean = scores.get("mean") if isinstance(scores, dict) else None
            if mean is not None:
                score_part = (
                    f" Judge mean {float(mean):.3f} over "
                    f"{int(summary.get('num_scored') or 0)} scored samples."
                )
                dist = summary.get("score_distribution")
                pct = dist.get("percentiles") if isinstance(dist, dict) else None
                if isinstance(pct, dict):
                    score_part += (
                        f" p10/p50/p90 = {pct['p10']:.1f}/{pct['p50']:.1f}"
                        f"/{pct['p90']:.1f}."
                    )
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass
        return (
            f"[System: eval run {run_name} completed{location}.{score_part} "
            f"Results in runs/{run_name}/eval_result.json; call {status_call} "
            "for details, then continue with the natural next step.]"
        )

    def record_completion(
        self, run_name: str, state: str, error: str | None, remote: str | None,
    ) -> None:
        """Mirror the transition in the manifest + project log.

        Telemetry mirroring stays with the caller (``on_record_completion``).
        """
        from lqh.project_log import append_event

        # Finalization manifest for the run itself: traces checkpoints/
        # eval results back to their spec revision, base model, and
        # dataset composition. Best-effort.
        try:
            from lqh.manifest import write_run_manifest

            write_run_manifest(
                self.project_dir,
                self.project_dir / "runs" / run_name,
                state=state,
                error=error,
            )
        except Exception:
            pass

        event = "training_completed" if state == "completed" else "training_failed"
        if state == "completed":
            desc = f"Run {run_name} completed"
            if remote:
                desc += f" on remote '{remote}'"
        else:
            desc = f"Run {run_name} failed"
            if remote:
                desc += f" on remote '{remote}'"
            if error:
                desc += f": {error}"
        kwargs: dict = {"run_name": run_name}
        if remote:
            kwargs["remote"] = remote
        if error:
            kwargs["error"] = error
        try:
            append_event(self.project_dir, event, desc, **kwargs)
        except Exception:
            pass

    async def finalize_data_gen_run(
        self, run_name: str, state: str, error: str | None,
    ) -> str | None:
        """Terminal handling for a cloud data-gen run.

        On success, download the run's ``dataset`` artifact into
        ``datasets/<output_dataset>/data.parquet`` so local flows
        (scoring, training bundles) proceed exactly as after a local
        run, then return the completion notice for the agent. The
        submit-time marker is consumed on every outcome EXCEPT a
        transient download failure — there it survives (with a retry
        counter) so the next watcher scan retries automatically, and
        ``None`` is returned to suppress a premature notification.
        """
        run_dir = self.project_dir / "runs" / run_name
        try:
            config = json.loads((run_dir / "config.json").read_text())
        except Exception:
            config = {}

        marker_path = run_dir / ".lqh_data_gen.json"
        marker: dict[str, Any] = {}
        try:
            marker = json.loads(marker_path.read_text())
        except Exception:
            pass
        workflow_id = str(marker.get("workflow_id") or uuid.uuid4())

        def _consume_marker() -> None:
            marker_path.unlink(missing_ok=True)

        output_dataset = str(config.get("output_dataset") or run_name)
        # The submit handler validates this, but the config file on disk
        # is not trusted for path construction — a separator or ".."
        # would escape datasets/.
        if Path(output_dataset).name != output_dataset or output_dataset in (".", ".."):
            output_dataset = run_name

        def _emit_terminal(outcome: str) -> None:
            if self.hooks.on_data_gen_terminal is not None:
                self.hooks.on_data_gen_terminal(outcome, workflow_id, marker)

        recovered_note = ""
        recovered_artifact_id: str | None = None
        if state != "completed":
            # A backend restart mid-job loses the event pump; the orphan
            # reconciler used to label such jobs failed even when they
            # finished and published. The backend now checks for the
            # dataset before deciding, but stay defensive here too: if
            # the dataset artifact exists, recover it instead of
            # reporting a bogus failure.
            try:
                from lqh.artifacts import BackendArtifactStore as _Store

                job_id = str(marker.get("job_id") or "")
                if job_id:
                    # Server-side job_id filter: a newest-N scan would
                    # miss the artifact once the project accumulates
                    # more datasets than one page.
                    for handle in await _Store().list_for_project(
                        _ckey(self.project_dir), kind="dataset", job_id=job_id,
                    ):
                        recovered_artifact_id = handle.id
                        break
            except Exception:
                recovered_artifact_id = None
            if recovered_artifact_id is None:
                _consume_marker()
                _emit_terminal("failed")
                verb = "was cancelled" if state == "cancelled" else "failed"
                err_part = f": {error}" if error else "."
                # Route through the same taxonomy as training and eval.
                # This branch used to assert "the pipeline error" for
                # every terminal, so an orphaned or interrupted data-gen
                # sandbox was reported to the agent as the user's
                # pipeline being broken — the one diagnosis it certainly
                # was not.
                from lqh.remote.failure import describe_failure, failure_from_dict

                directive = describe_failure(
                    failure_from_dict(self.job_failure.get(run_name)), run_name,
                )
                if not directive:
                    directive = (
                        f"Check runs/{run_name}/stderr.log (or the job's log "
                        "artifacts) for the pipeline error, fix it, validate "
                        "locally, and resubmit."
                    )
                return (
                    f"[System: cloud data-gen run {run_name} {verb}{err_part} "
                    f"{directive}]"
                )
            recovered_note = (
                " (the job was reported failed — likely a backend restart — "
                "but its dataset was published, so it was recovered)"
            )

        # A prior transient download failure schedules the next attempt;
        # until then stay silent (no network work, no notification).
        retry_after = marker.get("retry_after")
        if isinstance(retry_after, (int, float)) and time.time() < float(retry_after):
            return None

        counts = ""
        try:
            status = json.loads((run_dir / "status.json").read_text())
            if "succeeded" in status:
                counts = f" ({status['succeeded']}/{status.get('total', '?')} samples ok)"
        except Exception:
            pass
        if not counts:
            # The SSE status mirror overwrites status.json with state-only
            # payloads; the sandbox's summary progress row carries the
            # real sample counts.
            try:
                lines = (run_dir / "progress.jsonl").read_text().splitlines()
                for line in reversed(lines[-100:]):
                    row = json.loads(line)
                    if "succeeded" in row:
                        counts = f" ({row['succeeded']}/{row.get('total', '?')} samples ok)"
                        break
            except Exception:
                pass

        # Preferred source: the artifact SSE event mirrored into
        # artifacts.json by sync_progress. Fallback: list the project's
        # dataset artifacts and match on this run's job id (covers a
        # missed event after a long disconnect).
        artifact_id: str | None = recovered_artifact_id
        if artifact_id is None:
            try:
                manifest = json.loads((run_dir / "artifacts.json").read_text())
                for entry in manifest.get("artifacts", []):
                    if entry.get("kind") == "dataset" and entry.get("artifact_id"):
                        artifact_id = str(entry["artifact_id"])
                        break
            except Exception:
                pass

        dest = self.project_dir / "datasets" / output_dataset / "data.parquet"
        sidecar_path = dest.parent / ".lqh_source.json"
        replaced = dest.exists()

        # Overwrite policy: the NEWEST SUBMISSION wins among cloud jobs
        # (a sidecar written on every download records which submission
        # produced the local file), and anything the user generated
        # locally after this submit is never clobbered (mtime check —
        # applies only when no sidecar attributes the file to a job).
        submitted_at = marker.get("submitted_at")
        if replaced and isinstance(submitted_at, (int, float)):
            sidecar: dict[str, Any] = {}
            try:
                loaded = json.loads(sidecar_path.read_text())
                if isinstance(loaded, dict):
                    sidecar = loaded
            except Exception:
                pass
            prior_submitted = sidecar.get("submitted_at")
            downloaded_at = sidecar.get("downloaded_at")
            local_kept_msg = (
                f"[System: cloud data-gen run {run_name} completed{counts}, but "
                f"datasets/{output_dataset}/data.parquet was regenerated locally "
                "after the job was submitted, so the local file was kept. The "
                "cloud dataset remains available via the artifacts tool if you "
                "want it instead.]"
            )
            if isinstance(prior_submitted, (int, float)):
                if float(prior_submitted) > float(submitted_at):
                    _consume_marker()
                    _emit_terminal("succeeded")
                    return (
                        f"[System: cloud data-gen run {run_name} completed{counts}, "
                        f"but datasets/{output_dataset}/data.parquet already holds "
                        "the result of a NEWER submission, so it was kept. This "
                        "run's dataset remains available via the artifacts tool.]"
                    )
                # We are the newer submission — but the file may have been
                # modified/regenerated locally SINCE its cloud download
                # (local runs also delete the sidecar, this is the belt for
                # in-place edits): local work wins over any cloud job.
                if (
                    isinstance(downloaded_at, (int, float))
                    and dest.stat().st_mtime > float(downloaded_at) + 2.0
                ):
                    _consume_marker()
                    _emit_terminal("succeeded")
                    return local_kept_msg
                # else: overwrite below.
            elif dest.stat().st_mtime > float(submitted_at):
                _consume_marker()
                _emit_terminal("succeeded")
                return local_kept_msg

        try:
            from lqh.artifacts import BackendArtifactStore

            store = BackendArtifactStore()
            if artifact_id is None:
                job_id = str(marker.get("job_id") or "")
                if not job_id:
                    meta_path = run_dir / "remote_job.json"
                    if meta_path.exists():
                        job_id = str(json.loads(meta_path.read_text()).get("job_id") or "")
                if job_id:
                    # Server-side job_id filter (see comment on the
                    # recovery path above).
                    for handle in await store.list_for_project(
                        _ckey(self.project_dir), kind="dataset", job_id=job_id,
                    ):
                        artifact_id = handle.id
                        break
            if artifact_id is None:
                raise RuntimeError("no dataset artifact registered for this job")
            await store.download(artifact_id, dest)
        except Exception as exc:
            attempts = int(marker.get("download_attempts", 0) or 0) + 1
            if attempts < 8 and marker_path.exists():
                # Transient (network blip, brief API/R2 outage, listing
                # lag): keep the marker with a bumped counter and an
                # exponential backoff, and stay silent — a later watcher
                # scan retries the whole finalization. Eight attempts
                # with growing gaps ride out multi-minute outages that
                # three back-to-back scans could not.
                marker["download_attempts"] = attempts
                marker["retry_after"] = time.time() + min(900.0, 60.0 * (2 ** min(attempts, 4)))
                try:
                    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
                except OSError:
                    pass
                return None
            # Give up FOR THIS SESSION: notify once with the manual
            # recovery path, park the run in data_gen_gave_up, and keep
            # the marker (attempts reset) so a restart retries the
            # download automatically. Generation itself succeeded — the
            # workflow closes as such.
            self.data_gen_gave_up.add(run_name)
            _emit_terminal("succeeded")
            if marker_path.exists():
                marker["download_attempts"] = 0
                marker.pop("retry_after", None)
                marker["workflow_closed"] = True
                try:
                    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
                except OSError:
                    pass
            return (
                f"[System: cloud data-gen run {run_name} completed{counts}, but "
                f"downloading the dataset failed after {attempts} attempts: {exc}. "
                f"Use the artifacts tool to locate the run's dataset artifact and "
                f"download it to datasets/{output_dataset}/data.parquet, then "
                "continue (a restart will also retry the download).]"
            )

        # Attribute the local file to this submission so a concurrent
        # job targeting the same dataset can apply newest-submission-wins.
        try:
            sidecar_path.write_text(json.dumps({
                "job_id": marker.get("job_id"),
                "run_name": run_name,
                "submitted_at": submitted_at,
                # Lets a later cloud completion detect that the file was
                # modified locally AFTER this download (mtime check) and
                # keep the local version.
                "downloaded_at": time.time(),
            }, indent=2) + "\n")
        except OSError:
            pass

        # Finalization manifest for the downloaded dataset (provenance for
        # summary, spec-drift signals, and future sessions). Spec/pipeline
        # hashes come from the submission marker — the job ran the
        # submitted revisions, not whatever is on disk now.
        try:
            from lqh.manifest import write_dataset_manifest

            rows: int | None = None
            try:
                import pyarrow.parquet as _pq

                rows = _pq.read_metadata(dest).num_rows
            except Exception:
                pass
            manifest_ok = write_dataset_manifest(
                self.project_dir,
                dest.parent,
                purpose=str(marker.get("purpose") or "unspecified"),
                rows=rows,
                pipeline_path=marker.get("script_path"),
                pipeline_hash=marker.get("pipeline_hash"),
                spec_sha256=marker.get("spec_sha256"),
                run_name=run_name,
                job_id=str(marker.get("job_id") or "") or None,
                cloud_artifact_id=artifact_id,
            ) is not None
        except Exception:
            manifest_ok = False
        if not manifest_ok:
            # Surface it: the artifact exists but is not traceable.
            try:
                from lqh.project_log import append_event

                append_event(
                    self.project_dir, "manifest_write_failed",
                    f"Provenance manifest could not be written for datasets/{output_dataset}",
                    run_name=run_name,
                )
            except Exception:
                pass

        try:
            from lqh.project_log import append_event

            append_event(
                self.project_dir,
                "data_gen_completed",
                f"Cloud data gen {run_name} completed; dataset at datasets/{output_dataset}/",
                run_name=run_name,
                output_dataset=output_dataset,
            )
        except Exception:
            pass
        _consume_marker()
        _emit_terminal("succeeded")

        replaced_note = " (replaced the existing local file)" if replaced else ""
        manifest_note = (
            "" if manifest_ok else
            " ⚠️ Its provenance manifest could not be written — the dataset "
            "is not traceable to its spec/pipeline revision."
        )
        return (
            f"[System: cloud data-gen run {run_name} completed{counts}{recovered_note}. "
            f"The dataset was downloaded to datasets/{output_dataset}/data.parquet"
            f"{replaced_note}{manifest_note} — continue with the natural next step "
            "(e.g. scoring or training).]"
        )

    # ------------------------------------------------------------------
    # Scoring/sync watchers
    # ------------------------------------------------------------------

    async def spawn_run_watcher(self, run_name: str, remote: str | None) -> None:
        """Start a RunWatcher (or RemoteRunWatcher) for an active run.

        The watcher rsyncs predictions back from the remote (when applicable),
        invokes the API judge over predictions.parquet, writes eval_result.json,
        and self-stops when the run reaches a terminal state.
        """
        from lqh.auth import get_token
        from lqh.config import load_config

        run_dir = self.project_dir / "runs" / run_name
        config_path = run_dir / "config.json"
        if not config_path.exists():
            return
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            return

        api_key = get_token() or ""
        api_base_url = load_config().api_base_url

        if remote:
            from lqh.remote.watcher import RemoteRunWatcher

            meta_file = run_dir / "remote_job.json"
            if not meta_file.exists():
                return
            meta = json.loads(meta_file.read_text())
            backend = self.make_remote_backend(meta)
            if backend is None:
                return

            watcher = RemoteRunWatcher(
                run_dir=run_dir,
                config=config,
                project_dir=self.project_dir,
                api_key=api_key,
                api_base_url=api_base_url,
                backend=backend,
                remote_run_dir=meta["remote_run_dir"],
                job_id=str(meta["job_id"]),
            )
        else:
            from lqh.watcher import RunWatcher

            watcher = RunWatcher(
                run_dir=run_dir,
                config=config,
                project_dir=self.project_dir,
                api_key=api_key,
                api_base_url=api_base_url,
            )

        await watcher.start()
        self.run_watchers[run_name] = watcher

    async def stop_watchers(self) -> None:
        """Stop every scoring/sync watcher (used at shutdown)."""
        for watcher in list(self.run_watchers.values()):
            try:
                await watcher.stop()
            except Exception:
                pass
        self.run_watchers.clear()

    # ------------------------------------------------------------------
    # Parking
    # ------------------------------------------------------------------

    async def wait_primed(self, timeout: float = 30.0) -> None:
        """Wait until the first scan has populated the registry."""
        try:
            await asyncio.wait_for(self._primed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    def record_completion_notice(self, run_name: str, text: str) -> None:
        """Record a completion notice and wake any parked waiter.

        The watch loop calls this on a running→terminal transition; tests
        and alternative producers can use it to simulate one.
        """
        self.pending_completions[run_name] = text
        self._completion_signal.set()

    def take_pending_completion(self, run_names: list[str]) -> str | None:
        """Pop the recorded completion notice for the first finished wanted run.

        Returns ``None`` if none of ``run_names`` has a pending completion.
        Matching by run name keeps a stale notice for some other run from
        being handed back as this run's completion.
        """
        for name in run_names:
            text = self.pending_completions.pop(name, None)
            if text is not None:
                return text
        return None

    async def wait_for_results(self, run_names: list[str]) -> None:
        """Bounded wait for watcher-scored eval/infer runs to write results.

        A ``type: infer`` run reaches a terminal state when inference
        finishes, but ``eval_result.json`` is written afterwards by the
        scoring watcher. Without this, a parked agent could wake and read
        a completed-but-unscored run. Training runs (and runs with no
        pending scoring) return immediately; the wait is capped by
        ``SCORING_GRACE_SEC`` so a failed/stuck scorer can never hang the
        agent.
        """

        def _pending() -> list[str]:
            out: list[str] = []
            for name in run_names:
                run_dir = self.project_dir / "runs" / name
                config_path = run_dir / "config.json"
                if not config_path.exists():
                    continue
                try:
                    run_type = json.loads(config_path.read_text()).get("type", "")
                except Exception:
                    continue
                # Training eval lands per-checkpoint during the run; only
                # inference-style runs score after going terminal.
                if run_type not in ("infer", "eval_hf"):
                    continue
                if (run_dir / "eval_result.json").exists():
                    continue
                if run_type == "eval_hf":
                    # Cloud eval_hf scores sandbox-side; the local file
                    # arrives via the finalize download. Wait only while
                    # a published result artifact makes that download
                    # possible — a run that never published one won't
                    # produce the file no matter how long we park.
                    if self.eval_hf_result_artifact(name) is not None:
                        out.append(name)
                    continue
                # Only wait when scoring is actually in flight or pending —
                # otherwise (e.g. a failed run with no predictions) return
                # now.
                scoring_possible = (
                    name in self.run_watchers
                    or (run_dir / "predictions.parquet").exists()
                    or (run_dir / "eval_request.json").exists()
                )
                if scoring_possible:
                    out.append(name)
            return out

        loop = asyncio.get_event_loop()
        deadline = loop.time() + SCORING_GRACE_SEC
        while _pending():
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(2.0, remaining))

    async def wait_for_runs(
        self, run_names: list[str] | None, recheck_interval: float = 600.0,
    ) -> str | None:
        """Park until a watched run reaches a terminal state.

        Returns the ``[System: ...]`` completion notification once a run
        finishes, or ``None`` when nothing relevant is running so the
        caller should just use the status it already has. The wake signal
        is the completion the watch loop records; ``recheck_interval`` is
        only an internal safety re-check cadence (never surfaced).
        """

        def _running_targets() -> list[str]:
            # The registry is the authoritative "is it running" view: it is
            # populated eagerly at launch (register_started) and cleared by
            # the watch loop on a terminal transition.
            running = [
                t.label for t in self.tasks.snapshot() if t.state == "running"
            ]
            if run_names:
                wanted = set(run_names)
                running = [r for r in running if r in wanted]
            return running

        # The runs this call is responsible for: the explicit request, else
        # everything currently running. Captured once so a completion is
        # matched to the run even after it leaves the running set.
        wanted = list(run_names) if run_names else _running_targets()

        # A wanted run may have already finished — before we parked, or
        # before the very first scan for a fast run. Deliver its completion
        # now rather than parking for the full safety interval.
        done = self.take_pending_completion(wanted)
        if done is not None:
            await self.wait_for_results(wanted)
            return done

        if not _running_targets():
            return None

        while True:
            self._completion_signal.clear()
            # Re-check after clearing so a completion recorded between the
            # check above and the clear cannot be missed.
            done = self.take_pending_completion(wanted)
            if done is not None:
                await self.wait_for_results(wanted)
                return done
            try:
                await asyncio.wait_for(
                    self._completion_signal.wait(), timeout=recheck_interval
                )
            except asyncio.TimeoutError:
                pass
            done = self.take_pending_completion(wanted)
            if done is not None:
                await self.wait_for_results(wanted)
                return done
            # No wanted run is done. Stop only if none of them is still
            # running (its completion was missed before we parked);
            # otherwise keep parking silently.
            if not _running_targets():
                return None
