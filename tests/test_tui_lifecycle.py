"""Regression tests for TUI background lifecycle behavior."""

from __future__ import annotations

import asyncio
import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lqh.tui.app import LqhApp


class _FakeAgent:
    def __init__(self) -> None:
        self.process_calls: list[str] = []
        self.continue_calls = 0
        self.fail_process = True
        self.fail_continue = False

    async def process_user_input(self, text: str) -> None:
        self.process_calls.append(text)
        if self.fail_process:
            self.fail_process = False
            raise OSError(errno.ENETDOWN, "network is down")

    async def continue_after_interruption(self) -> None:
        self.continue_calls += 1
        if self.fail_continue:
            raise OSError(errno.ENETDOWN, "network is down")


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LqhApp:
    instance = LqhApp(tmp_path)
    instance._reconnect_backoffs = (0.0,)
    emitted: list[str] = []

    async def _emit(text: str) -> None:
        emitted.append(text)

    instance._emit = _emit  # type: ignore[method-assign]
    instance._emitted = emitted  # type: ignore[attr-defined]
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: "test-token")
    return instance


async def test_transient_agent_failure_retries_without_duplicate_message(app: LqhApp) -> None:
    agent = _FakeAgent()
    app._agent = agent  # type: ignore[assignment]

    await app._handle_message("train the model")

    assert agent.process_calls == ["train the model"]
    assert agent.continue_calls == 1
    assert app._pending_reconnect is None


async def test_reconnect_command_resumes_pending_turn(app: LqhApp) -> None:
    agent = _FakeAgent()
    agent.fail_continue = True
    app._agent = agent  # type: ignore[assignment]

    await app._handle_message("train the model")

    assert agent.process_calls == ["train the model"]
    assert agent.continue_calls == 1
    assert app._pending_reconnect is not None

    agent.fail_continue = False
    await app._handle_command("/reconnect")

    assert agent.process_calls == ["train the model"]
    assert agent.continue_calls == 2
    assert app._pending_reconnect is None


async def test_reconnect_command_no_pending_operation(app: LqhApp) -> None:
    await app._handle_command("/reconnect")

    emitted = getattr(app, "_emitted")
    assert any("No reconnect is pending" in text for text in emitted)


async def test_scan_jobs_syncs_cloud_remote_before_polling(tmp_path: Path) -> None:
    project = tmp_path
    run_dir = project / "runs" / "cloud_run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"type": "infer"}) + "\n")
    (run_dir / "remote_job.json").write_text(json.dumps({
        "job_id": "job-1",
        "remote_name": "cloud",
        "remote_run_dir": "cloud:job-1",
        "backend": "cloud",
    }) + "\n")

    class FakeBackend:
        def __init__(self) -> None:
            self.synced: list[tuple[str, str]] = []
            self.polled: list[str] = []

        async def sync_progress(self, remote_run_dir: str, local_run_dir: str) -> None:
            self.synced.append((remote_run_dir, local_run_dir))

        async def poll_status(self, job_id: str):
            self.polled.append(job_id)
            return SimpleNamespace(state="completed", error=None)

    backend = FakeBackend()
    app = LqhApp(project)
    app._make_remote_backend = lambda _meta: backend  # type: ignore[method-assign]

    snapshots = await app._scan_jobs(SimpleNamespace())

    assert snapshots == [("cloud_run", "completed", None, "cloud")]
    assert backend.synced == [("cloud:job-1", str(run_dir))]
    assert backend.polled == ["job-1"]


def test_completion_message_tells_agent_to_check_status(tmp_path: Path) -> None:
    app = LqhApp(tmp_path)

    message = app._format_completion_message("run_1", "completed", None, "cloud")

    # The suggested call carries no remote arg — status derives it from
    # the run's remote_job.json. The remote still appears as readable
    # context ("on remote 'cloud'").
    assert "training_status(run_name='run_1')" in message
    assert "remote=" not in message
    assert "on remote 'cloud'" in message
    assert "continue with the natural next step" in message


# ---------------------------------------------------------------------------
# Auto-mode transparent waiting (_await_background)
# ---------------------------------------------------------------------------


def _register_running(app: LqhApp, name: str) -> None:
    from lqh.tui.background_tasks import BackgroundTask

    app._tasks.register(
        BackgroundTask(task_id=name, kind="train", label=name, state="running")
    )


async def test_await_background_returns_none_when_nothing_running(tmp_path: Path) -> None:
    app = LqhApp(tmp_path)
    # No tasks registered -> nothing to wait for.
    result = await app._await_background(["run_1"], timeout=5.0)
    assert result is None


async def test_await_background_returns_none_when_target_not_running(tmp_path: Path) -> None:
    app = LqhApp(tmp_path)
    _register_running(app, "other_run")
    # Asking about run_1, but only other_run is active -> don't park.
    result = await app._await_background(["run_1"], timeout=5.0)
    assert result is None


def _simulate_watch_completion(app: LqhApp, name: str, notice: str) -> None:
    """Mirror what _watch_jobs does on a running -> terminal transition:
    record the per-run notice, wake the park via the queue, unregister."""
    app._pending_completions[name] = notice
    app._input_queue.put_nowait(notice)
    app._tasks.unregister(name)


async def test_await_background_wakes_on_completion_message(tmp_path: Path) -> None:
    app = LqhApp(tmp_path)
    _register_running(app, "run_1")

    # Simulate _watch_jobs signalling completion while the agent parks.
    notice = "[System: training run run_1 completed successfully.]"

    async def _producer() -> None:
        await asyncio.sleep(0.01)
        _simulate_watch_completion(app, "run_1", notice)

    producer = asyncio.create_task(_producer())
    result = await app._await_background(["run_1"], timeout=5.0)
    await producer

    assert result == notice


async def test_await_background_delivers_completion_that_finished_before_park(
    tmp_path: Path,
) -> None:
    # Fast run: it finished (watcher recorded the notice and unregistered it)
    # before the agent called training_status. The park must deliver it at
    # once instead of waiting out the safety interval.
    app = LqhApp(tmp_path)
    notice = "[System: training run fast_run completed successfully.]"
    app._pending_completions["fast_run"] = notice  # watcher already saw it

    result = await app._await_background(["fast_run"], timeout=5.0)
    assert result == notice
    # The pending entry is consumed, not left to leak to a later run.
    assert "fast_run" not in app._pending_completions


async def test_await_background_ignores_stale_completion_for_other_run(
    tmp_path: Path,
) -> None:
    # A completion for run_a is left over in the queue/registry. While waiting
    # for run_b, the park must NOT hand back run_a's notice as run_b's.
    app = LqhApp(tmp_path)
    _register_running(app, "run_b")
    stale = "[System: training run run_a completed successfully.]"
    app._pending_completions["run_a"] = stale
    app._input_queue.put_nowait(stale)  # stale wake nudge

    park = asyncio.create_task(app._await_background(["run_b"], timeout=0.02))
    await asyncio.sleep(0.1)  # consumes the stale nudge, keeps parking
    assert not park.done()
    assert app._pending_completions.get("run_a") == stale  # untouched

    # run_b finishes for real -> that is what gets delivered.
    notice_b = "[System: training run run_b completed successfully.]"
    _simulate_watch_completion(app, "run_b", notice_b)
    result = await asyncio.wait_for(park, timeout=1.0)
    assert result == notice_b


async def test_await_background_parks_silently_until_terminal(tmp_path: Path) -> None:
    app = LqhApp(tmp_path)
    _register_running(app, "run_1")

    # With a tiny re-check interval and nothing pushed, the park must NOT
    # return a heartbeat — it keeps waiting silently while the run is alive
    # (zero LLM cycles; progress is shown in the status bar instead).
    park = asyncio.create_task(app._await_background(["run_1"], timeout=0.02))
    await asyncio.sleep(0.1)  # several internal re-check cycles
    assert not park.done()

    # Simulate the run going terminal without a queued message (drained):
    # _watch_jobs would unregister it; the next re-check then returns None.
    app._tasks.unregister("run_1")
    result = await asyncio.wait_for(park, timeout=1.0)
    assert result is None


def test_on_background_task_started_seeds_running_state(tmp_path: Path) -> None:
    # Eager registration must seed _job_last_state so a run that finishes
    # before the first watcher scan still counts as a running -> terminal
    # transition (otherwise its completion is never recorded).
    app = LqhApp(tmp_path)
    # A stale completion from an earlier run that reused this name must be
    # cleared, so it can't be delivered as the new run's completion.
    app._pending_completions["run_x"] = "[System: old run_x finished.]"

    app._on_background_task_started("run_x", "train", "run_x", None)

    assert app._job_last_state["run_x"] == "running"
    assert "run_x" not in app._pending_completions


# ---------------------------------------------------------------------------
# Wake-once-results-arrive for eval/infer runs (_wait_for_results)
# ---------------------------------------------------------------------------


def _make_run(tmp_path: Path, name: str, run_type: str, **files: str) -> Path:
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"type": run_type}) + "\n")
    for fname, body in files.items():
        (run_dir / fname.replace("__", ".")).write_text(body)
    return run_dir


async def test_wait_for_results_returns_immediately_for_training(tmp_path: Path) -> None:
    app = LqhApp(tmp_path)
    _make_run(tmp_path, "sft_v1", "sft")
    # Training runs never block here even without eval_result.json.
    await asyncio.wait_for(app._wait_for_results(["sft_v1"]), timeout=1.0)


async def test_wait_for_results_skips_infer_with_no_pending_scoring(tmp_path: Path) -> None:
    app = LqhApp(tmp_path)
    # Infer run, no predictions / eval_request / watcher -> nothing to wait for.
    _make_run(tmp_path, "eval_1", "infer")
    await asyncio.wait_for(app._wait_for_results(["eval_1"]), timeout=1.0)


async def test_wait_for_results_waits_until_eval_result_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = LqhApp(tmp_path)
    run_dir = _make_run(
        tmp_path, "eval_1", "infer", predictions__parquet="x",
    )

    async def _writer() -> None:
        await asyncio.sleep(0.05)
        (run_dir / "eval_result.json").write_text(json.dumps({"mean": 7.0}))

    writer = asyncio.create_task(_writer())
    # Grace window comfortably longer than the writer delay.
    monkeypatch.setattr("lqh.tui.app.SCORING_GRACE_SEC", 5.0)
    await asyncio.wait_for(app._wait_for_results(["eval_1"]), timeout=3.0)
    await writer
    assert (run_dir / "eval_result.json").exists()


async def test_wait_for_results_respects_grace_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = LqhApp(tmp_path)
    # Pending scoring that never completes -> bounded by the grace window.
    _make_run(tmp_path, "eval_1", "infer", predictions__parquet="x")
    monkeypatch.setattr("lqh.tui.app.SCORING_GRACE_SEC", 0.1)
    # Must return despite eval_result.json never appearing.
    await asyncio.wait_for(app._wait_for_results(["eval_1"]), timeout=1.0)
