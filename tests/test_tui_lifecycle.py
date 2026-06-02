"""Regression tests for TUI background lifecycle behavior."""

from __future__ import annotations

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
