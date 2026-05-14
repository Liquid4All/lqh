"""Test RemoteRunWatcher sync/push cycle with a mock backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lqh.remote.backend import JobStatus, RemoteBackend, RemoteConfig
from lqh.remote.watcher import RemoteRunWatcher


class MockBackend:
    """Minimal mock RemoteBackend for watcher tests."""

    def __init__(self) -> None:
        self.sync_progress_calls: list[tuple[str, str]] = []
        self.pushed_files: list[tuple[str, str]] = []
        self.alive = True
        self.config = RemoteConfig(
            name="test", type="ssh_direct", hostname="h", remote_root="/r",
        )

    async def sync_progress(self, remote_run_dir: str, local_run_dir: str) -> None:
        self.sync_progress_calls.append((remote_run_dir, local_run_dir))

    async def sync_file_to_remote(self, local_path: str, remote_path: str) -> None:
        self.pushed_files.append((local_path, remote_path))

    async def sync_file_from_remote(self, remote_path: str, local_path: str) -> None:
        pass

    async def is_job_alive(self, job_id: str) -> bool:
        return self.alive

    async def setup(self) -> str:
        return "ok"

    async def submit_run(self, *a: Any, **kw: Any) -> str:
        return "1234"

    async def poll_status(self, job_id: str) -> JobStatus:
        return JobStatus(state="running", pid=int(job_id))

    async def teardown(self, job_id: str) -> None:
        pass


class MockCallbacks:
    """Capture watcher callbacks."""

    def __init__(self) -> None:
        self.progress: list[dict[str, Any]] = []
        self.completed: list[str] = []
        self.failed: list[tuple[str, str | None]] = []

    def on_training_progress(self, run_name: str, **kw: Any) -> None:
        self.progress.append({"run_name": run_name, **kw})

    def on_training_completed(self, run_name: str) -> None:
        self.completed.append(run_name)

    def on_training_failed(self, run_name: str, error: str | None) -> None:
        self.failed.append((run_name, error))

    def on_eval_scored(self, run_name: str, checkpoint: str, mean_score: float) -> None:
        pass

    def on_iter_scored(self, run_name: str, iteration: str, mean_score: float) -> None:
        pass


class TestRemoteRunWatcher:
    """Test the RemoteRunWatcher sync/push/completion cycle."""

    def _make_watcher(
        self, run_dir: Path, backend: MockBackend, callbacks: MockCallbacks,
    ) -> RemoteRunWatcher:
        config = {"type": "sft", "scorer": "evals/scorers/test.md"}
        return RemoteRunWatcher(
            run_dir=run_dir,
            config=config,
            project_dir=run_dir.parent.parent,
            api_key="test-key",
            backend=backend,
            remote_run_dir="/remote/runs/test_run",
            job_id="1234",
            callbacks=callbacks,
            poll_interval=0.1,
        )

    def test_instantiation(self, tmp_path: Path):
        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)
        backend = MockBackend()
        callbacks = MockCallbacks()
        watcher = self._make_watcher(run_dir, backend, callbacks)
        assert watcher.run_name == "test_run"
        assert watcher._job_id == "1234"

    @pytest.mark.asyncio
    async def test_detects_completion_from_progress(self, tmp_path: Path):
        """Watcher should detect completed state from progress.jsonl."""
        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)

        # Write a completed progress entry
        (run_dir / "progress.jsonl").write_text(
            json.dumps({"status": "completed", "step": 100, "timestamp": "t"}) + "\n"
        )

        backend = MockBackend()
        callbacks = MockCallbacks()
        watcher = self._make_watcher(run_dir, backend, callbacks)

        # Run one cycle of the check
        await watcher._sync_from_remote()
        watcher._update_progress()
        await watcher._check_completion_remote()

        assert "test_run" in callbacks.completed

    @pytest.mark.asyncio
    async def test_detects_dead_process(self, tmp_path: Path):
        """Watcher should detect when remote process dies."""
        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)

        # Write some progress but no terminal status
        (run_dir / "progress.jsonl").write_text(
            json.dumps({"step": 50, "loss": 2.0, "timestamp": "t"}) + "\n"
        )

        backend = MockBackend()
        backend.alive = False  # Process is dead
        callbacks = MockCallbacks()
        watcher = self._make_watcher(run_dir, backend, callbacks)

        await watcher._sync_from_remote()
        watcher._update_progress()
        await watcher._check_completion_remote()

        assert len(callbacks.failed) == 1
        assert "without writing final status" in (callbacks.failed[0][1] or "")

    @pytest.mark.asyncio
    async def test_push_eval_results(self, tmp_path: Path):
        """Watcher should push eval_result.json files to remote."""
        run_dir = tmp_path / "runs" / "test_run"
        cp_dir = run_dir / "checkpoints" / "step_500"
        cp_dir.mkdir(parents=True)

        # Simulate a scored checkpoint
        (cp_dir / "eval_result.json").write_text(
            json.dumps({"scores": {"mean": 7.5}}) + "\n"
        )

        backend = MockBackend()
        callbacks = MockCallbacks()
        watcher = self._make_watcher(run_dir, backend, callbacks)

        await watcher._push_eval_results()

        assert len(backend.pushed_files) == 1
        local_path, remote_path = backend.pushed_files[0]
        assert "eval_result.json" in local_path
        assert "step_500/eval_result.json" in remote_path

    @pytest.mark.asyncio
    async def test_push_preferences(self, tmp_path: Path):
        """Watcher should push preferences.parquet files for DPO."""
        run_dir = tmp_path / "runs" / "test_run"
        iter_dir = run_dir / "iterations" / "iter_000"
        iter_dir.mkdir(parents=True)

        (iter_dir / "preferences.parquet").write_bytes(b"parquet-data")

        backend = MockBackend()
        callbacks = MockCallbacks()
        config = {"type": "on_policy_dpo", "scorer": "evals/scorers/test.md"}
        watcher = RemoteRunWatcher(
            run_dir=run_dir,
            config=config,
            project_dir=tmp_path,
            api_key="test-key",
            backend=backend,
            remote_run_dir="/remote/runs/test_run",
            job_id="1234",
            callbacks=callbacks,
            poll_interval=0.1,
        )

        await watcher._push_preferences()

        assert len(backend.pushed_files) == 1
        _, remote_path = backend.pushed_files[0]
        assert "iter_000/preferences.parquet" in remote_path

    @pytest.mark.asyncio
    async def test_no_double_push(self, tmp_path: Path):
        """Files should only be pushed once."""
        run_dir = tmp_path / "runs" / "test_run"
        cp_dir = run_dir / "checkpoints" / "step_500"
        cp_dir.mkdir(parents=True)
        (cp_dir / "eval_result.json").write_text("{}")

        backend = MockBackend()
        callbacks = MockCallbacks()
        watcher = self._make_watcher(run_dir, backend, callbacks)

        await watcher._push_eval_results()
        await watcher._push_eval_results()  # second call

        assert len(backend.pushed_files) == 1  # only pushed once
