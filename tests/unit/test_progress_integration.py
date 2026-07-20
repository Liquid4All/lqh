from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from lqh.progress import (
    OBSERVER_PROGRESS_FILE,
    ProgressEvent,
    ProgressReporter,
    estimate_eta_seconds,
    final_scoring_context,
    format_event_oneline,
    read_progress_events,
    relay_cloud_sentinel,
    write_progress_event,
)
from lqh.train.progress import read_latest_metrics, write_progress, write_status
from lqh.subprocess_manager import SubprocessManager
from lqh.tui.app import LqhApp
from lqh.tui.background_tasks import BackgroundTask
from lqh.watcher import RunWatcher


def _register(app: LqhApp, name: str = "run") -> None:
    app._tasks.register(BackgroundTask(name, "train", name, "running"))


def test_v1_progress_wins_over_later_legacy_row(tmp_path) -> None:
    run = tmp_path / "runs" / "run"
    write_progress_event(run, ProgressEvent(
        task_kind="sft", label="run", phase="training",
        phase_label="training SFT", completed=21, total=100,
        unit="steps", overall_fraction=0.21,
    ))
    write_progress(run, step=82, extra={"max_steps": 100})

    app = LqhApp(tmp_path)
    _register(app)
    app._update_task_progress("run")

    task = app._tasks.snapshot()[0]
    assert "21%" in (task.progress or "")
    assert "82%" not in (task.progress or "")


def test_scoring_retry_replaces_prior_attempt_high_water_mark(tmp_path) -> None:
    run = tmp_path / "runs" / "run"
    write_progress_event(run, ProgressEvent(
        task_kind="evaluation", label="run", phase="scoring_attempt_1",
        phase_label="judging results", completed=50, total=100,
        unit="samples", overall_fraction=0.975,
    ))
    write_progress_event(run, ProgressEvent(
        task_kind="evaluation", label="run", phase="scoring_attempt_2",
        phase_label="judging results", completed=0, total=100,
        unit="samples", overall_fraction=0.95, detail="retry 2/3",
    ))

    app = LqhApp(tmp_path)
    _register(app)
    app._update_task_progress("run")

    progress = app._tasks.snapshot()[0].progress or ""
    assert "0/100" in progress
    assert "retry 2/3" in progress


def test_result_ready_overrides_scoring_attempt_selection(tmp_path) -> None:
    run = tmp_path / "runs" / "run"
    write_progress_event(run, ProgressEvent(
        task_kind="evaluation", label="run", phase="scoring_attempt_1",
        phase_label="judging results", completed=99, total=100,
        unit="samples", overall_fraction=0.999,
    ))
    write_progress_event(run, ProgressEvent(
        task_kind="evaluation", label="run", phase="completed",
        phase_label="results ready", completed=100, total=100,
        unit="samples", overall_fraction=1.0, result_ready=True,
    ))

    app = LqhApp(tmp_path)
    _register(app)
    app._update_task_progress("run")

    progress = app._tasks.snapshot()[0].progress or ""
    assert "results ready" in progress
    assert "100%" in progress


def test_new_run_attempt_can_regress_from_stale_high_water_mark(tmp_path) -> None:
    run = tmp_path / "runs" / "run"
    run.mkdir(parents=True)
    write_progress_event(run, ProgressEvent(
        task_kind="sft", label="run", phase="training",
        phase_label="old training", overall_fraction=0.99,
        attempt_id="old",
    ))
    with (run / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps({
            "status": "running", "attempt_id": "new",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
    write_progress_event(run, ProgressEvent(
        task_kind="sft", label="run", phase="training",
        phase_label="new training", completed=1, total=10,
        overall_fraction=0.1, attempt_id="new",
    ))

    app = LqhApp(tmp_path)
    _register(app)
    app._update_task_progress("run")

    progress = app._tasks.snapshot()[0].progress or ""
    assert "new training" in progress
    assert "10%" in progress


def test_new_attempt_setup_window_hides_prior_attempt_progress(tmp_path) -> None:
    run = tmp_path / "runs" / "run"
    run.mkdir(parents=True)
    write_progress_event(run, ProgressEvent(
        task_kind="sft", label="run", phase="completed",
        phase_label="old results ready", overall_fraction=1.0,
        result_ready=True, attempt_id="old",
    ))
    with (run / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps({
            "status": "running", "attempt_id": "new",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")

    app = LqhApp(tmp_path)
    app._tasks.register(BackgroundTask(
        "run", "train", "run", "running", progress="old results ready · 100%",
    ))
    app._update_task_progress("run")

    assert app._tasks.snapshot()[0].progress is None


def test_observer_progress_survives_remote_file_replacement(tmp_path) -> None:
    run = tmp_path / "runs" / "run"
    write_progress_event(run, ProgressEvent(
        task_kind="training", label="run", phase="inference",
        phase_label="evaluating", overall_fraction=0.95,
    ))
    write_progress_event(
        run,
        ProgressEvent(
            task_kind="training", label="run", phase="completed",
            phase_label="results ready", overall_fraction=1,
            result_ready=True,
        ),
        file_name=OBSERVER_PROGRESS_FILE,
    )

    # Simulate the next SSH rsync replacing the producer-owned file.
    (run / "progress.jsonl").write_text(json.dumps(
        ProgressEvent(
            task_kind="training", label="run", phase="inference",
            phase_label="evaluating", overall_fraction=0.90,
        ).as_payload()
    ) + "\n")

    rows = read_progress_events(run)
    assert any(row.get("result_ready") for row in rows)
    app = LqhApp(tmp_path)
    _register(app)
    app._update_task_progress("run")
    assert "100%" in (app._tasks.snapshot()[0].progress or "")


def test_results_pending_requires_actionable_handoff(tmp_path) -> None:
    run = tmp_path / "runs" / "eval"
    run.mkdir(parents=True)
    app = LqhApp(tmp_path)

    (run / "config.json").write_text(json.dumps({"type": "infer"}))
    assert app._supervisor.results_pending("eval") is False

    (run / "config.json").write_text(json.dumps({
        "type": "infer", "scorer": "evals/scorer.md",
    }))
    assert app._supervisor.results_pending("eval") is False

    (run / "eval_request.json").write_text("{}")
    (run / "predictions.parquet").write_bytes(b"predictions")
    assert app._supervisor.results_pending("eval") is True

    (run / "eval_error.json").write_text("{}")
    assert app._supervisor.results_pending("eval") is False


def test_running_status_does_not_mask_dead_process(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    write_status(run, "running")
    failures: list[str | None] = []

    class Callbacks:
        def on_training_failed(self, _name, error):
            failures.append(error)

    watcher = RunWatcher(
        run_dir=run, config={"type": "sft"}, project_dir=tmp_path,
        api_key="token", callbacks=Callbacks(),
    )
    watcher._manager.is_alive = lambda _run: False  # type: ignore[method-assign]

    watcher._check_completion()

    assert failures == ["Process exited without writing final status"]
    assert watcher._stop.is_set()


@pytest.mark.asyncio
async def test_terminal_eval_error_is_not_retried_after_restart(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "eval_request.json").write_text("{}")
    (run / "predictions.parquet").write_bytes(b"predictions")
    (run / "eval_error.json").write_text('{"error":"terminal"}\n')
    watcher = RunWatcher(
        run_dir=run,
        config={"type": "infer", "scorer": "evals/scorer.md"},
        project_dir=tmp_path,
        api_key="token",
    )
    called = False

    async def score(_path):
        nonlocal called
        called = True
        return "success"

    watcher._score_checkpoint = score  # type: ignore[method-assign]
    await watcher._check_eval_requests()

    assert called is False


def test_v1_cloud_event_carries_training_metrics(tmp_path) -> None:
    reporter = ProgressReporter(
        task_kind="sft", label="run", run_dir=tmp_path, min_interval=0,
    )
    reporter.update(
        phase="training", phase_label="training", completed=4, total=10,
        overall_fraction=0.4, step=4, loss=0.25, lr=1e-5, epoch=1.5,
    )

    metrics = read_latest_metrics(tmp_path)
    assert metrics is not None
    assert metrics["step"] == 4
    assert metrics["loss"] == 0.25

@pytest.mark.asyncio
async def test_watcher_retries_scoring_then_writes_terminal_error(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "eval_request.json").write_text("{}")
    (run / "predictions.parquet").write_bytes(b"predictions")
    watcher = RunWatcher(
        run_dir=run,
        config={"type": "infer", "scorer": "evals/scorer.md"},
        project_dir=tmp_path,
        api_key="token",
    )
    attempts = 0

    async def fail(_path):
        nonlocal attempts
        attempts += 1
        return "failed"

    watcher._score_checkpoint = fail  # type: ignore[method-assign]
    await watcher._check_eval_requests()
    watcher._retry_not_before[str(run)] = 0
    await watcher._check_eval_requests()
    assert not (run / "eval_error.json").exists()
    watcher._retry_not_before[str(run)] = 0
    await watcher._check_eval_requests()

    assert attempts == 3
    assert (run / "eval_error.json").exists()
    assert watcher._has_pending_scoring_requests() is False


@pytest.mark.asyncio
async def test_dpo_iteration_failure_is_capped(tmp_path) -> None:
    run = tmp_path / "run"
    iter_dir = run / "iterations" / "iter_000"
    iter_dir.mkdir(parents=True)
    (iter_dir / "iter_request.json").write_text("{}")
    (iter_dir / "predictions.parquet").write_bytes(b"predictions")
    watcher = RunWatcher(
        run_dir=run,
        config={"type": "dpo", "scorer": "evals/scorer.md"},
        project_dir=tmp_path,
        api_key="token",
    )
    attempts = 0

    async def fail(_path):
        nonlocal attempts
        attempts += 1
        return "failed"

    watcher._process_iteration = fail  # type: ignore[method-assign]
    for _ in range(3):
        await watcher._check_iter_requests()
        watcher._retry_not_before[str(iter_dir)] = 0

    assert attempts == 3
    assert (iter_dir / "preference_error.json").exists()


@pytest.mark.asyncio
async def test_missing_scoring_payload_does_not_consume_retry(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "eval_request.json").write_text("{}")
    watcher = RunWatcher(
        run_dir=run,
        config={"type": "infer", "scorer": "evals/scorer.md"},
        project_dir=tmp_path,
        api_key="token",
    )

    await watcher._check_eval_requests()

    assert watcher._scoring_attempts == {}
    assert not (run / "eval_error.json").exists()

    # A permanently missing payload still reaches a terminal state instead
    # of pinning a completed run forever.
    watcher._not_ready_since[str(run)] -= 15 * 60 + 1
    await watcher._check_eval_requests()
    assert (run / "eval_error.json").exists()


def test_intermediate_checkpoint_has_no_headline_scoring_context(tmp_path) -> None:
    intermediate = tmp_path / "run" / "checkpoints" / "step_10"
    final = tmp_path / "run" / "checkpoints" / "final"
    config = {"type": "sft"}

    assert final_scoring_context(intermediate, config) is None
    context = final_scoring_context(final, config)
    assert context is not None
    assert context.progress_dir == tmp_path / "run"
    assert context.start == 0.95


@pytest.mark.asyncio
async def test_cloud_intermediate_checkpoint_emits_no_headline_sentinel(
    tmp_path, monkeypatch, capsys,
) -> None:
    from lqh.train import cloud_score
    import lqh.scoring as scoring

    checkpoint = tmp_path / "run" / "checkpoints" / "step_10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "predictions.parquet").write_bytes(b"predictions")
    scorer = tmp_path / "scorer.md"
    scorer.write_text("score")

    class Client:
        async def close(self):
            return None

    async def fake_score(**_kwargs):
        return {"num_scored": 1, "num_failed": 0, "scores": {"mean": 8}}

    monkeypatch.setenv("LQH_JOB_ID", "cloud-job")
    monkeypatch.setattr(cloud_score, "_resolve_scorer_path", lambda *_: scorer)
    monkeypatch.setattr(cloud_score, "_make_client", Client)
    monkeypatch.setattr(scoring, "score_predictions_by_source", fake_score)

    await cloud_score._score_run_eval_async(
        checkpoint,
        {"type": "sft", "scorer": str(scorer)},
    )

    assert "LQH_EVENT_JSON:" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_all_failed_cloud_scoring_never_reports_results_ready(
    tmp_path, monkeypatch, capsys,
) -> None:
    from lqh.train import cloud_score
    import lqh.scoring as scoring

    run = tmp_path / "run"
    run.mkdir()
    (run / "predictions.parquet").write_bytes(b"predictions")
    scorer = tmp_path / "scorer.md"
    scorer.write_text("score")

    class Client:
        async def close(self):
            return None

    async def fake_score(**_kwargs):
        return {"num_scored": 0, "num_failed": 3, "scores": {"mean": 0}}

    monkeypatch.setenv("LQH_JOB_ID", "cloud-job")
    monkeypatch.setattr(cloud_score, "_resolve_scorer_path", lambda *_: scorer)
    monkeypatch.setattr(cloud_score, "_make_client", Client)
    monkeypatch.setattr(scoring, "score_predictions_by_source", fake_score)

    await cloud_score._score_run_eval_async(
        run, {"type": "infer", "scorer": str(scorer)},
    )

    assert '"result_ready": true' not in capsys.readouterr().out
    assert (run / "eval_error.json").exists()


def test_status_detection_ignores_later_progress_rows(tmp_path) -> None:
    from lqh.train.progress import write_status

    run = tmp_path / "run"
    run.mkdir()
    write_status(run, "completed")
    write_progress_event(run, ProgressEvent(
        task_kind="sft", label="run", phase="late",
        phase_label="late telemetry", overall_fraction=0.9,
    ))

    assert SubprocessManager().get_status(run).state == "completed"


def test_status_metrics_ignore_later_v1_telemetry(tmp_path) -> None:
    from lqh.train.progress import write_progress, write_status

    run = tmp_path / "run"
    run.mkdir()
    write_progress(run, step=42, loss=0.25, lr=1e-5, epoch=1.5)
    write_progress_event(run, ProgressEvent(
        task_kind="sft", label="run", phase="training",
        phase_label="training", overall_fraction=0.42,
    ))
    write_status(run, "completed")

    status = SubprocessManager().get_status(run)
    assert status.step == 42
    assert status.loss == 0.25
    assert status.lr == 1e-5
    assert status.epoch == 1.5


def test_interrupted_status_preserves_handoff_error(tmp_path) -> None:
    from lqh.train.progress import write_status

    run = tmp_path / "run"
    run.mkdir()
    write_status(run, "interrupted", error="preference judge failed")

    status = SubprocessManager().get_status(run)
    assert status.state == "failed"
    assert status.error == "preference judge failed"


def test_wait_for_file_stops_on_upstream_error(tmp_path) -> None:
    from lqh.train.progress import wait_for_file

    error = tmp_path / "preference_error.json"
    error.write_text('{"error":"judge failed"}\n')
    with pytest.raises(RuntimeError, match="judge failed"):
        wait_for_file(
            tmp_path / "preferences.parquet",
            error_path=error,
            poll_interval=0.001,
            timeout=1,
        )

def test_sweep_child_fraction_is_not_divided_by_training_budget(tmp_path) -> None:
    from lqh.train.sweep import _ChildProgressContext, _forward_child_progress_row

    run = tmp_path / "run"
    ctx = _ChildProgressContext(
        parent_run_dir=run,
        config_id="cfg",
        config_index=1,
        n_configs=4,
        training_end=0.90,
    )
    _forward_child_progress_row(ctx, ProgressEvent(
        task_kind="sft", label="child", phase="inference",
        phase_label="final eval", overall_fraction=0.95,
    ).as_payload())
    row = read_progress_events(run)[-1]
    assert row["overall_fraction"] == pytest.approx(0.90 * (1 + 0.95) / 4)


def test_nan_does_not_ratchet_reporter_to_end() -> None:
    seen: list[ProgressEvent] = []
    reporter = ProgressReporter(
        task_kind="sft", label="run", callback=seen.append, min_interval=0,
    )
    reporter.update(
        phase="training", phase_label="training", overall_fraction=0.2,
    )
    reporter.update(
        phase="training", phase_label="training", overall_fraction=float("nan"),
    )
    reporter.update(
        phase="training", phase_label="training", overall_fraction=0.3,
    )
    assert [event.overall_fraction for event in seen] == [0.2, 0.2, 0.3]


def test_eta_window_handles_four_hz_producer() -> None:
    start = datetime.now(timezone.utc) - timedelta(seconds=30)
    rows = [
        ProgressEvent(
            task_kind="data_gen", label="gen", phase="generation",
            phase_label="generating", overall_fraction=i / 200,
            timestamp=(start + timedelta(seconds=i * 0.25)).isoformat(),
        )
        for i in range(100)
    ]
    now = (start + timedelta(seconds=24.75)).timestamp()
    assert estimate_eta_seconds(rows, now=now) is not None


def test_background_eta_uses_local_observation_age_despite_clock_skew() -> None:
    remote_start = datetime.now(timezone.utc) - timedelta(hours=2)
    rows = [
        ProgressEvent(
            task_kind="sft", label="run", phase="training",
            phase_label="training", overall_fraction=i / 10,
            timestamp=(remote_start + timedelta(seconds=i * 5)).isoformat(),
        )
        for i in range(1, 7)
    ]
    line, _ = format_event_oneline(
        rows[-1], history=rows,
        observed_at=datetime.now(timezone.utc).timestamp(),
    )
    assert "ETA" in line

    from lqh.train.progress import format_progress_oneline

    wrapped, _ = format_progress_oneline(
        rows[-1].as_payload(),
        history=[row.as_payload() for row in rows],
    )
    assert "ETA" not in wrapped

    observed, _ = format_progress_oneline(
        rows[-1].as_payload(),
        history=[row.as_payload() for row in rows],
        observed_at=datetime.now(timezone.utc).timestamp(),
    )
    assert "ETA" in observed


def test_callback_type_error_is_not_probed_twice() -> None:
    calls = 0

    def broken(_event: ProgressEvent) -> None:
        nonlocal calls
        calls += 1
        raise TypeError("inside callback")

    reporter = ProgressReporter(
        task_kind="data_gen", label="gen", callback=broken, min_interval=0,
    )
    reporter.update(
        phase="generation", phase_label="generating", overall_fraction=0.1,
    )
    assert calls == 1


def test_three_parameter_event_callback_is_not_misclassified() -> None:
    received: list[ProgressEvent] = []

    def callback(event, _extra=None, _log=None):
        received.append(event)

    ProgressReporter(
        task_kind="data_gen", label="gen", callback=callback, min_interval=0,
    ).update(
        phase="generation", phase_label="generating", overall_fraction=0.1,
    )
    assert isinstance(received[0], ProgressEvent)


def test_zero_total_does_not_bypass_throttle() -> None:
    import time

    received: list[ProgressEvent] = []
    reporter = ProgressReporter(
        task_kind="evaluation", label="empty", callback=received.append,
        min_interval=1000,
    )
    reporter._last_emit = time.monotonic()
    reporter.update(
        phase="scoring", phase_label="scoring", completed=0, total=0,
        overall_fraction=0,
    )
    assert received == []


@pytest.mark.asyncio
async def test_refresh_loop_survives_one_broken_task(tmp_path) -> None:
    import asyncio

    app = LqhApp(tmp_path)
    app._tasks.register(BackgroundTask("broken", "train", "broken", "running"))
    app._tasks.register(BackgroundTask("healthy", "train", "healthy", "running"))
    seen: list[str] = []

    def refresh(task_id: str) -> None:
        if task_id == "broken":
            raise ValueError("malformed progress")
        seen.append(task_id)

    app._update_task_progress = refresh  # type: ignore[method-assign]
    app._ensure_progress_refresh_task()
    await asyncio.sleep(0.02)
    assert seen == ["healthy"]
    assert app._progress_refresh_task is not None
    assert not app._progress_refresh_task.done()
    app._progress_refresh_task.cancel()
    try:
        await app._progress_refresh_task
    except asyncio.CancelledError:
        pass


def test_cloud_child_sentinel_relay(monkeypatch, capsys) -> None:
    line = 'LQH_EVENT_JSON: {"kind":"progress","payload":{"x":1}}\n'
    monkeypatch.setenv("LQH_JOB_ID", "job")
    assert relay_cloud_sentinel(line) is True
    assert line.strip() in capsys.readouterr().out

    monkeypatch.delenv("LQH_JOB_ID")
    assert relay_cloud_sentinel(line) is False
