"""A running job that stops moving must be reported, not waited on
forever — and reporting it must not break the anti-polling contract.

The user in feedback #37 watched "step 50/72" for 70+ minutes with no
way to tell a working job from a hung one. The fix is a pushed advisory;
the constraint is that it stays an advisory: `wait_for_runs` must still
return only on a terminal state, or a parked auto-mode agent would wake
up and start burning LLM cycles on a job that is merely quiet.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from lqh.jobs import (
    STALL_NOTICE_MINUTES,
    STALL_NOTICE_REATTACHED_MINUTES,
    STALL_NOTICE_SILENT_MINUTES,
    JobSupervisor,
    SupervisorHooks,
)


def _run_with_progress(project: Path, name: str, *, age_minutes: float) -> Path:
    run_dir = project / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"type": "sft"}) + "\n")
    progress = run_dir / "progress.jsonl"
    progress.write_text(
        json.dumps({"phase": "training", "phase_label": "training", "step": 50}) + "\n"
    )
    old = time.time() - age_minutes * 60
    os.utime(progress, (old, old))
    return run_dir


def _supervisor(project: Path) -> tuple[JobSupervisor, list[tuple[str, str, str]]]:
    notices: list[tuple[str, str, str]] = []
    hooks = SupervisorHooks(
        on_notice=lambda run, text, state: notices.append((run, text, state))
    )
    return JobSupervisor(project, hooks=hooks), notices


def test_no_notice_while_progress_is_fresh(tmp_path: Path) -> None:
    _run_with_progress(tmp_path, "run_1", age_minutes=5)
    sup, notices = _supervisor(tmp_path)

    sup.maybe_notify_stall("run_1")

    assert notices == []


def test_a_freshly_submitted_run_is_not_stalled(tmp_path: Path) -> None:
    # A job still downloading its base model is slow, not stuck.
    run_dir = tmp_path / "runs" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}\n")
    sup, notices = _supervisor(tmp_path)

    idle, saw_progress = sup.stalled_minutes("run_1")
    assert saw_progress is False
    assert idle < 1
    sup.maybe_notify_stall("run_1")
    assert notices == []


def test_a_run_wedged_before_its_first_progress_event_still_trips(
    tmp_path: Path,
) -> None:
    """The case a last-progress-timestamp check can never see.

    A sandbox stuck in setup — image pull, model download, a wedged
    calibration probe — emits nothing at all, so there is no progress
    mtime to age. Anchor on submission instead, with a longer threshold
    because cold starts are legitimately slow.
    """
    import os

    run_dir = tmp_path / "runs" / "run_1"
    run_dir.mkdir(parents=True)
    config = run_dir / "config.json"
    config.write_text("{}\n")
    old = time.time() - (STALL_NOTICE_SILENT_MINUTES + 30) * 60
    os.utime(config, (old, old))
    sup, notices = _supervisor(tmp_path)

    sup.maybe_notify_stall("run_1")

    assert len(notices) == 1
    assert "has not reported any progress in the" in notices[0][1]

    # ...but not before the longer silent threshold.
    other = tmp_path / "runs" / "run_2"
    other.mkdir(parents=True)
    cfg2 = other / "config.json"
    cfg2.write_text("{}\n")
    recent = time.time() - (STALL_NOTICE_MINUTES + 5) * 60
    os.utime(cfg2, (recent, recent))
    sup.maybe_notify_stall("run_2")
    assert len(notices) == 1


def test_notice_fires_once_per_run(tmp_path: Path) -> None:
    _run_with_progress(tmp_path, "run_1", age_minutes=STALL_NOTICE_MINUTES + 15)
    sup, notices = _supervisor(tmp_path)

    sup.maybe_notify_stall("run_1")
    sup.maybe_notify_stall("run_1")
    sup.maybe_notify_stall("run_1")

    assert len(notices) == 1
    run, text, state = notices[0]
    assert run == "run_1"
    assert state == "stalled"
    assert "no progress for 75 minutes" in text
    assert "stop_training(run_name='run_1')" in text
    # The advisory must not undo the anti-polling contract it sits next to.
    assert "do NOT call training_status in a loop" in text
    assert "do not start a second run" in text


def test_notice_names_the_last_reported_phase(tmp_path: Path) -> None:
    _run_with_progress(tmp_path, "run_1", age_minutes=STALL_NOTICE_MINUTES + 1)
    sup, notices = _supervisor(tmp_path)

    sup.maybe_notify_stall("run_1")

    assert "(last: training)" in notices[0][1]


@pytest.mark.asyncio
async def test_stall_notice_does_not_wake_a_parked_agent(tmp_path: Path) -> None:
    """The regression guard: a stall is an advisory, not a completion."""
    _run_with_progress(tmp_path, "run_1", age_minutes=STALL_NOTICE_MINUTES + 5)
    sup, notices = _supervisor(tmp_path)
    from lqh.tui.background_tasks import BackgroundTask

    sup.tasks.register(
        BackgroundTask(task_id="run_1", kind="train", label="run_1", state="running")
    )

    sup.maybe_notify_stall("run_1")

    assert len(notices) == 1
    # Nothing was queued for parking...
    assert sup.pending_completions == {}
    # ...so a waiting agent stays asleep rather than spending a turn on a
    # job that is still registered as running.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(sup.wait_for_runs(["run_1"]), timeout=0.25)


def test_stall_reaches_a_parked_agent_via_the_completion_notice(
    tmp_path: Path,
) -> None:
    """The delivery gap the live hook cannot close.

    Auto mode parks inside wait_for_runs and the TUI's input pump drops
    `[System: ...]` messages while an agent turn is active, so the live
    advisory reaches nobody there. Folding it into the completion notice
    means the parked agent still learns the run was wedged — at the one
    moment it is listening.
    """
    _run_with_progress(tmp_path, "run_1", age_minutes=STALL_NOTICE_MINUTES + 5)
    sup, _ = _supervisor(tmp_path)

    sup.maybe_notify_stall("run_1")
    message = sup.format_completion_message("run_1", "failed", "exit code 1", "cloud")

    assert "flagged it as stalled" in message
    # Consumed once — the next message for the same run is clean.
    assert "flagged it as stalled" not in sup.format_completion_message(
        "run_1", "failed", "exit code 1", "cloud",
    )


def test_terminal_state_clears_the_stall_flag(tmp_path: Path) -> None:
    _run_with_progress(tmp_path, "run_1", age_minutes=STALL_NOTICE_MINUTES + 5)
    sup, _ = _supervisor(tmp_path)

    sup.maybe_notify_stall("run_1")
    assert "run_1" in sup.stall_notified

    sup.stall_notified.discard("run_1")
    sup.maybe_notify_stall("run_1")
    assert "run_1" in sup.stall_notified


def test_a_reattached_run_uses_the_quiet_threshold(tmp_path: Path) -> None:
    """After a backend restart the reattached pump does not re-stream
    trainer stdout, so progress.jsonl freezes by design and the backend
    says so in the row it appends. Treating that frozen file as evidence
    would warn on every healthy multi-hour run that outlived a deploy —
    and offer to cancel it. But "cannot prove it is stuck" is not "is
    fine forever": it moves to a much longer threshold, not an exemption.
    """
    sup, notices = _supervisor(tmp_path)
    run_dir = _run_with_progress(tmp_path, "sft_003", age_minutes=0.0)
    progress = run_dir / "progress.jsonl"
    with progress.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "phase": "training",
                    "overall_fraction": 0.4,
                    "detail": (
                        "progress may be stale — backend restarted mid-run; "
                        "sample counts freeze until the job finishes"
                    ),
                }
            )
            + "\n"
        )
    # Well past the ordinary thresholds, nowhere near the reattached one.
    old = time.time() - (STALL_NOTICE_SILENT_MINUTES + 30) * 60
    os.utime(progress, (old, old))
    sup.maybe_notify_stall("sft_003")
    assert notices == []

    # ...but a reattached run that has been silent for four hours is
    # worth mentioning; it is no longer explainable by the reattach.
    older = time.time() - (STALL_NOTICE_REATTACHED_MINUTES + 30) * 60
    os.utime(progress, (older, older))
    sup.maybe_notify_stall("sft_003")
    assert len(notices) == 1
    assert notices[0][2] == "stalled"


def test_progress_after_a_reattach_marker_restores_stall_detection(
    tmp_path: Path,
) -> None:
    """The marker only means "frozen from here". A real progress row
    after it means the stream is alive again and the ordinary rule
    applies."""
    sup, _ = _supervisor(tmp_path)
    run_dir = _run_with_progress(tmp_path, "sft_004", age_minutes=0.0)
    progress = run_dir / "progress.jsonl"
    with progress.open("a") as fh:
        fh.write(json.dumps({"overall_fraction": 0.4, "detail":
                             "progress may be stale — backend restarted mid-run"}) + "\n")
        fh.write(json.dumps({"overall_fraction": 0.5, "phase": "training"}) + "\n")
    old = time.time() - (STALL_NOTICE_MINUTES + 10) * 60
    os.utime(progress, (old, old))

    observed = sup.stalled_minutes("sft_004")
    assert observed is not None
    idle, saw_progress = observed
    assert saw_progress is True
    assert idle > STALL_NOTICE_MINUTES


def test_headless_run_installs_a_notice_hook() -> None:
    """`lqh run` wired on_running but not on_notice, so a stall advisory
    reached nothing at all in the mode most likely to be unattended.
    It is now on the NDJSON event stream the harness reads.
    """
    import inspect

    from lqh.cli_cmds import run_cmd

    src = inspect.getsource(run_cmd._run_async)
    assert "on_notice=" in src
    assert '"job_notice"' in src


def test_headless_reads_signals_after_the_first_backend_poll() -> None:
    """cloud_failure.json is written by the supervisor's poll. Building
    the agent's context first meant a job that failed while the CLI was
    closed produced its infra_failure signal one session late — and
    headless has no second context refresh.
    """
    import inspect

    from lqh.cli_cmds import run_cmd

    src = inspect.getsource(run_cmd._run_async)
    primed = src.index("await supervisor.wait_primed()")
    prepared = src.index("await agent.prepare_context()")
    assert primed < prepared, "context must be built after the first scan"

    # The DIFF must also be computed after the scan: observe_run_states
    # reads whatever the last sync left on disk, so computing it first
    # compares stale local state and misses every job that finished
    # remotely while the CLI was closed.
    diffed = src.index("finished_while_away_signals(")
    assert primed < diffed, "the diff must be computed against refreshed state"
    # ...against a baseline snapshotted BEFORE the scan, because the
    # supervisor's first scan records a new one.
    baseline = src.index("load_seen_states(project_dir)")
    assert baseline < primed, "the baseline must be captured before the scan"
    assert "diff_signals=" in src


def test_a_lifecycle_row_is_not_progress(tmp_path: Path) -> None:
    """The cloud event translator mirrors a status=running row into
    progress.jsonl the moment the job starts. Treating any non-empty
    file as "has reported progress" put every cloud job on the 60-minute
    threshold, so the 90-minute silent-start allowance — written for
    exactly these runs — never applied to one.
    """
    sup, notices = _supervisor(tmp_path)
    run_dir = tmp_path / "runs" / "sft_003"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"type": "sft"}) + "\n")
    progress = run_dir / "progress.jsonl"
    progress.write_text(json.dumps({"status": "running", "timestamp": "t"}) + "\n")
    old = time.time() - (STALL_NOTICE_MINUTES + 10) * 60
    os.utime(progress, (old, old))

    observed = sup.stalled_minutes("sft_003")
    assert observed is not None
    _, saw_progress = observed
    assert saw_progress is False, "a lifecycle row is not workload progress"
    sup.maybe_notify_stall("sft_003")
    assert notices == [], "still inside the silent-start allowance"

    older = time.time() - (STALL_NOTICE_SILENT_MINUTES + 10) * 60
    os.utime(progress, (older, older))
    sup.maybe_notify_stall("sft_003")
    assert len(notices) == 1


def test_the_stall_notice_survives_a_restart(tmp_path: Path) -> None:
    """Both collections are in-memory. A restart re-notified about the
    same run, and an advisory raised before the restart could never be
    folded into the completion notice that arrived after it."""
    sup, notices = _supervisor(tmp_path)
    _run_with_progress(tmp_path, "sft_003", age_minutes=STALL_NOTICE_MINUTES + 10)
    sup.maybe_notify_stall("sft_003")
    assert len(notices) == 1

    # A fresh process over the same project directory.
    restarted, later_notices = _supervisor(tmp_path)
    restarted.maybe_notify_stall("sft_003")
    assert later_notices == [], "the same stall must not be announced twice"
    # ...and the advisory is still available to the completion notice.
    assert "sft_003" in restarted.stall_advisories
    assert "no progress" in restarted.stall_advisories["sft_003"]
