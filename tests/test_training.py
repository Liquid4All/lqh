"""Tests for the fine-tuning infrastructure.

Unit tests (no GPU) verify:
    - Progress file protocol (write/read)
    - Subprocess manager (lifecycle, status parsing)
    - Data utils (parquet ChatML conversions)
    - Sync backend protocol
    - Tool handler validation (start_training, training_status, stop_training)
    - Golden trajectory assembly

GPU tests (require torch + CUDA) verify:
    - SFT training loop end-to-end with checkpoint eval
    - DPO training loop with mock preferences
    - Local inference subprocess
    - Full tool → subprocess → watcher → scoring pipeline

GPU-gated tests opt in via ``@pytest.mark.gpu`` and skip automatically
when no CUDA device is visible (see ``tests/conftest.py``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A throw-away run directory at ``<tmp>/runs/test_run``."""
    path = tmp_path / "runs" / "test_run"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """``<tmp>/runs`` for tests that create multiple sibling runs."""
    path = tmp_path / "runs"
    path.mkdir(parents=True)
    return path


# ---------------------------------------------------------------------------
# Progress protocol
# ---------------------------------------------------------------------------


class TestProgressProtocol:
    """File-based progress protocol."""

    def test_write_and_read_progress(self, run_dir: Path) -> None:
        from lqh.train.progress import (
            read_latest_progress,
            read_progress,
            write_progress,
        )

        write_progress(run_dir, step=10, loss=2.5, lr=1e-5, epoch=0.5)
        write_progress(run_dir, step=20, loss=2.1, lr=9e-6, epoch=1.0)

        entries = read_progress(run_dir)
        assert len(entries) == 2
        assert entries[0]["step"] == 10
        assert entries[1]["step"] == 20
        assert entries[0]["loss"] == pytest.approx(2.5)

        latest = read_latest_progress(run_dir)
        assert latest is not None
        assert latest["step"] == 20

    def test_read_progress_empty(self, run_dir: Path) -> None:
        from lqh.train.progress import read_latest_progress, read_progress

        assert read_progress(run_dir) == []
        assert read_latest_progress(run_dir) is None

    def test_write_status(self, run_dir: Path) -> None:
        from lqh.train.progress import read_latest_progress, write_status

        write_status(run_dir, "completed")
        latest = read_latest_progress(run_dir)
        assert latest["status"] == "completed"

    def test_write_status_failed_with_error(self, run_dir: Path) -> None:
        from lqh.train.progress import read_latest_progress, write_status

        write_status(run_dir, "failed", error="CUDA OOM")
        latest = read_latest_progress(run_dir)
        assert latest["status"] == "failed"
        assert latest["error"] == "CUDA OOM"

    def test_write_eval_request(self, run_dir: Path) -> None:
        from lqh.train.progress import write_eval_request

        cp_dir = run_dir / "checkpoints" / "step_100"
        cp_dir.mkdir(parents=True)
        write_eval_request(cp_dir)

        req = json.loads((cp_dir / "eval_request.json").read_text())
        assert req["status"] == "ready"
        assert req["predictions"] == "predictions.parquet"

    def test_write_iter_request(self, run_dir: Path) -> None:
        from lqh.train.progress import write_iter_request

        iter_dir = run_dir / "iterations" / "iter_000"
        iter_dir.mkdir(parents=True)
        write_iter_request(iter_dir)

        req = json.loads((iter_dir / "iter_request.json").read_text())
        assert req["status"] == "ready"

    def test_read_progress_last_n(self, run_dir: Path) -> None:
        from lqh.train.progress import read_progress, write_progress

        for i in range(20):
            write_progress(run_dir, step=i, loss=float(i))

        entries = read_progress(run_dir, last_n=3)
        assert len(entries) == 3
        assert entries[0]["step"] == 17
        assert entries[-1]["step"] == 19

    def test_wait_for_file_exists(self, run_dir: Path) -> None:
        from lqh.train.progress import wait_for_file

        target = run_dir / "result.json"
        target.write_text('{"ok": true}')

        assert wait_for_file(target, poll_interval=0.01, timeout=1.0) == target

    def test_wait_for_file_timeout(self, run_dir: Path) -> None:
        from lqh.train.progress import wait_for_file

        with pytest.raises(TimeoutError):
            wait_for_file(run_dir / "nonexistent.json", poll_interval=0.01, timeout=0.05)


# ---------------------------------------------------------------------------
# Subprocess manager
# ---------------------------------------------------------------------------


class TestSubprocessManager:
    """Subprocess lifecycle management."""

    def test_start_writes_config_and_pid(self, tmp_path: Path, runs_dir: Path) -> None:
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        run = runs_dir / "test_001"

        config = {"type": "sft", "base_model": "test", "dataset": "test"}
        # ``python -m time`` exits cleanly; this is the smallest harmless target.
        pid = manager.start(run, config, module="time", project_dir=tmp_path)

        assert (run / "config.json").exists()
        assert (run / "pid").exists()
        assert isinstance(pid, int)
        assert pid > 0
        assert json.loads((run / "config.json").read_text())["type"] == "sft"

    def test_get_status_unknown_no_progress(self, runs_dir: Path) -> None:
        from lqh.subprocess_manager import SubprocessManager

        run = runs_dir / "empty_run"
        run.mkdir()
        assert SubprocessManager().get_status(run).state == "unknown"

    def test_get_status_completed(self, runs_dir: Path) -> None:
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import write_progress, write_status

        run = runs_dir / "done_run"
        run.mkdir()
        write_progress(run, step=100, loss=0.5)
        write_status(run, "completed")

        assert SubprocessManager().get_status(run).state == "completed"

    def test_get_status_failed(self, runs_dir: Path) -> None:
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import write_status

        run = runs_dir / "failed_run"
        run.mkdir()
        write_status(run, "failed", error="CUDA OOM")

        status = SubprocessManager().get_status(run)
        assert status.state == "failed"
        assert status.error == "CUDA OOM"

    def test_list_runs(self, tmp_path: Path, runs_dir: Path) -> None:
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import write_status

        for name in ("sft_001", "dpo_001"):
            run = runs_dir / name
            run.mkdir()
            (run / "config.json").write_text('{"type": "sft"}')
            write_status(run, "completed")

        runs = SubprocessManager().list_runs(tmp_path)
        names = [r[0] for r in runs]
        assert set(names) == {"sft_001", "dpo_001"}

    def test_list_runs_empty(self, tmp_path: Path) -> None:
        from lqh.subprocess_manager import SubprocessManager

        assert SubprocessManager().list_runs(tmp_path) == []

    def test_is_alive_no_pid(self, runs_dir: Path) -> None:
        from lqh.subprocess_manager import SubprocessManager

        run = runs_dir / "no_pid"
        run.mkdir()
        assert SubprocessManager().is_alive(run) is False

    def test_is_alive_dead_pid(self, runs_dir: Path) -> None:
        from lqh.subprocess_manager import SubprocessManager

        run = runs_dir / "dead_pid"
        run.mkdir()
        (run / "pid").write_text("999999999")
        assert SubprocessManager().is_alive(run) is False


# ---------------------------------------------------------------------------
# Data utils
# ---------------------------------------------------------------------------


class TestDataUtils:
    """Parquet ChatML conversion utilities."""

    def test_load_chatml_dataset(
        self,
        tmp_path: Path,
        write_chatml_parquet,
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.data_utils import load_chatml_dataset

        path = write_chatml_parquet(tmp_path / "data.parquet", sample_conversations(3))
        loaded = load_chatml_dataset(path)
        assert len(loaded) == 3
        assert loaded[0][0]["role"] == "user"
        assert loaded[0][1]["role"] == "assistant"

    def test_chatml_to_sft_dataset(
        self, sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.data_utils import chatml_to_sft_dataset

        sft = chatml_to_sft_dataset(sample_conversations(2))
        assert len(sft) == 2
        assert "messages" in sft[0]
        assert sft[0]["messages"][0]["role"] == "user"

    def test_chatml_to_dpo_dataset(self) -> None:
        from lqh.train.data_utils import chatml_to_dpo_dataset

        dpo = chatml_to_dpo_dataset([{
            "prompt": [{"role": "user", "content": "hello"}],
            "chosen": "good response",
            "rejected": "bad response",
        }])
        assert len(dpo) == 1
        assert {"prompt", "chosen", "rejected"} <= dpo[0].keys()

    def test_load_preferences_parquet(self, tmp_path: Path) -> None:
        from lqh.train.data_utils import load_preferences_parquet

        path = tmp_path / "preferences.parquet"
        pq.write_table(
            pa.table({
                "prompt": [json.dumps([{"role": "user", "content": "hi"}])],
                "chosen": ["good"],
                "rejected": ["bad"],
            }),
            path,
        )

        loaded = load_preferences_parquet(path)
        assert len(loaded) == 1
        assert isinstance(loaded[0]["prompt"], list)
        assert loaded[0]["chosen"] == "good"
        assert loaded[0]["rejected"] == "bad"


# ---------------------------------------------------------------------------
# Sync backend
# ---------------------------------------------------------------------------


class TestSyncBackend:
    """Sync protocol and ``LocalSync``."""

    async def test_local_sync_is_noop(self, tmp_path: Path) -> None:
        from lqh.sync import LocalSync

        sync = LocalSync()
        await sync.push([], tmp_path)
        await sync.pull(tmp_path, ["*.json"], tmp_path)

    def test_resolve_manifest(self, tmp_path: Path) -> None:
        from lqh.sync import resolve_manifest

        (tmp_path / "datasets").mkdir()
        (tmp_path / "datasets" / "data.parquet").write_text("dummy")

        paths = resolve_manifest(
            {
                "dataset": "datasets/data.parquet",
                "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",  # not local
                "manifest": ["dataset", "base_model"],
            },
            tmp_path,
        )
        assert len(paths) == 1
        assert paths[0].name == "data.parquet"

    def test_resolve_manifest_empty(self, tmp_path: Path) -> None:
        from lqh.sync import resolve_manifest

        assert resolve_manifest({}, tmp_path) == []


# ---------------------------------------------------------------------------
# Training tool validation
# ---------------------------------------------------------------------------


@pytest.fixture
def training_workspace(
    tmp_path: Path,
    write_chatml_parquet,
    sample_conversations: Callable[[int], list],
) -> Path:
    """Project dir with dataset, scorer, and auto-allow permissions."""
    ds_dir = tmp_path / "datasets" / "test_ds"
    write_chatml_parquet(ds_dir / "data.parquet", sample_conversations(5))

    scorer_dir = tmp_path / "evals" / "scorers"
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "test.md").write_text("Score 1-10.")

    lqh_dir = tmp_path / ".lqh"
    lqh_dir.mkdir(parents=True)
    (lqh_dir / "permissions.json").write_text(json.dumps({"project_allow_all": True}))

    return tmp_path


@pytest.fixture
def stub_torch_available():
    """Patch the torch availability probe so handlers reach validation."""
    with patch("lqh.tools.handlers._check_torch_available", return_value=None) as p:
        yield p


class TestTrainingToolValidation:
    """Training tool handlers validate inputs correctly."""

    async def test_start_training_missing_dataset(
        self, training_workspace: Path, stub_torch_available,
    ) -> None:
        from lqh.tools.handlers import handle_start_training

        result = await handle_start_training(
            training_workspace,
            type="sft",
            base_model="test-model",
            dataset="datasets/nonexistent",
        )
        assert "not found" in result.content

    async def test_start_training_missing_eval_dataset(
        self, training_workspace: Path, stub_torch_available,
    ) -> None:
        from lqh.tools.handlers import handle_start_training

        result = await handle_start_training(
            training_workspace,
            type="sft",
            base_model="test-model",
            dataset="datasets/test_ds",
            eval_dataset="datasets/nonexistent_eval",
        )
        assert "not found" in result.content

    async def test_start_training_missing_scorer(
        self, training_workspace: Path, stub_torch_available,
    ) -> None:
        from lqh.tools.handlers import handle_start_training

        result = await handle_start_training(
            training_workspace,
            type="sft",
            base_model="test-model",
            dataset="datasets/test_ds",
            scorer="evals/scorers/nonexistent.md",
        )
        assert "not found" in result.content

    async def test_training_status_no_runs(self, training_workspace: Path) -> None:
        from lqh.tools.handlers import handle_training_status

        result = await handle_training_status(training_workspace)
        assert "No training runs" in result.content

    async def test_training_status_nonexistent_run(self, training_workspace: Path) -> None:
        from lqh.tools.handlers import handle_training_status

        result = await handle_training_status(training_workspace, run_name="nonexistent")
        assert "not found" in result.content

    async def test_stop_training_nonexistent_run(self, training_workspace: Path) -> None:
        from lqh.tools.handlers import handle_stop_training

        result = await handle_stop_training(training_workspace, run_name="nonexistent")
        assert "not found" in result.content

    async def test_stop_training_not_running(self, training_workspace: Path) -> None:
        from lqh.tools.handlers import handle_stop_training
        from lqh.train.progress import write_status

        run = training_workspace / "runs" / "done_run"
        run.mkdir(parents=True)
        (run / "config.json").write_text('{"type": "sft"}')
        write_status(run, "completed")

        result = await handle_stop_training(training_workspace, run_name="done_run")
        assert "not currently running" in result.content

    async def test_training_status_with_completed_run(
        self, training_workspace: Path,
    ) -> None:
        from lqh.tools.handlers import handle_training_status
        from lqh.train.progress import write_progress, write_status

        run = training_workspace / "runs" / "sft_001"
        run.mkdir(parents=True)
        (run / "config.json").write_text('{"type": "sft"}')
        write_progress(run, step=100, loss=0.5, lr=1e-5, epoch=3.0)
        write_status(run, "completed")

        result = await handle_training_status(training_workspace, run_name="sft_001")
        assert "completed" in result.content
        assert "sft_001" in result.content

    async def test_start_local_eval_missing_model(
        self, training_workspace: Path, stub_torch_available,
    ) -> None:
        from lqh.tools.handlers import handle_start_local_eval

        result = await handle_start_local_eval(
            training_workspace,
            model_path="runs/nonexistent/model",
            dataset="datasets/test_ds",
            scorer="evals/scorers/test.md",
        )
        assert "not found" in result.content

    def test_next_run_name_generation(self, training_workspace: Path) -> None:
        from lqh.tools.handlers import _next_run_name

        assert _next_run_name(training_workspace, "sft") == "sft_001"

        (training_workspace / "runs" / "sft_001").mkdir(parents=True)
        (training_workspace / "runs" / "sft_002").mkdir(parents=True)

        assert _next_run_name(training_workspace, "sft") == "sft_003"
        assert _next_run_name(training_workspace, "dpo") == "dpo_001"


# ---------------------------------------------------------------------------
# Golden trajectory assembly
# ---------------------------------------------------------------------------


@pytest.fixture
def predictions_and_scores(tmp_path: Path) -> Callable[[list[float]], tuple[Path, Path, Path]]:
    """Factory writing matched ``predictions.parquet`` + ``results.parquet``.

    Returns ``(output_dir, predictions_path, results_path)``.
    """
    output_dir = tmp_path / "iter_000"
    output_dir.mkdir(parents=True)

    def _factory(scores: list[float]) -> tuple[Path, Path, Path]:
        n = len(scores)
        convos = [
            [
                {"role": "user", "content": f"prompt {i}"},
                {"role": "assistant", "content": f"bad response {i}"},
            ]
            for i in range(n)
        ]
        pred_path = output_dir / "predictions.parquet"
        pq.write_table(
            pa.table({
                "sample_index": list(range(n)),
                "messages": [json.dumps(c) for c in convos],
            }),
            pred_path,
        )

        scores_path = output_dir / "results.parquet"
        pq.write_table(
            pa.table({
                "sample_index": list(range(n)),
                "score": scores,
                "reasoning": [f"reason {i}" for i in range(n)],
                "messages": [json.dumps(c) for c in convos],
            }),
            scores_path,
        )

        return output_dir, pred_path, scores_path

    return _factory


class TestGoldenAssembly:
    """Golden trajectory generation and preference pair assembly."""

    async def test_golden_from_dataset(
        self, tmp_path: Path, predictions_and_scores, write_chatml_parquet,
    ) -> None:
        """``golden_source='dataset'`` pulls from training data."""
        from lqh.golden import generate_golden

        output_dir, pred_path, scores_path = predictions_and_scores([3.0, 8.0, 2.0])

        ds_path = write_chatml_parquet(
            tmp_path / "train_data.parquet",
            [
                [{"role": "user", "content": "prompt 0"}, {"role": "assistant", "content": "good response 0"}],
                [{"role": "user", "content": "prompt 1"}, {"role": "assistant", "content": "good response 1"}],
                [{"role": "user", "content": "prompt 2"}, {"role": "assistant", "content": "good response 2"}],
            ],
        )

        await generate_golden(
            predictions_path=pred_path,
            scores_path=scores_path,
            dataset_path=str(ds_path),
            config={
                "golden_source": "dataset",
                "rejection_threshold": 6.0,
                "dataset": str(ds_path),
            },
            client=MagicMock(),
            output_dir=output_dir,
        )

        # Preferences for samples 0 and 2 (scores 3.0 and 2.0).
        prefs_path = output_dir / "preferences.parquet"
        assert prefs_path.exists()
        assert len(pq.read_table(str(prefs_path))) == 2

    async def test_no_low_scorers_writes_empty_preferences(
        self, predictions_and_scores,
    ) -> None:
        """When all scores are above threshold, write empty preferences."""
        from lqh.golden import generate_golden

        output_dir, pred_path, scores_path = predictions_and_scores([9.0, 8.0, 7.0])

        await generate_golden(
            predictions_path=pred_path,
            scores_path=scores_path,
            dataset_path="",
            config={"golden_source": "dataset", "rejection_threshold": 6.0},
            client=MagicMock(),
            output_dir=output_dir,
        )

        prefs_path = output_dir / "preferences.parquet"
        assert prefs_path.exists()
        assert len(pq.read_table(str(prefs_path))) == 0


# ===========================================================================
# GPU tests
# ===========================================================================


@pytest.mark.gpu
class TestSFTEndToEnd:
    """End-to-end SFT training test with a tiny model and dataset.

    Runs a real training loop for a few steps, verifies checkpoints,
    progress output, and the eval loop.
    """

    @pytest.fixture
    def workspace(
        self,
        tmp_path: Path,
        write_chatml_parquet,
        sample_conversations: Callable[[int], list],
    ) -> dict[str, Path]:
        run = tmp_path / "runs" / "sft_test"
        run.mkdir(parents=True)
        write_chatml_parquet(
            tmp_path / "datasets" / "train" / "data.parquet",
            sample_conversations(10),
        )
        write_chatml_parquet(
            tmp_path / "datasets" / "eval" / "data.parquet",
            sample_conversations(3),
        )
        return {"project": tmp_path, "run": run}

    def test_sft_training_loop(self, workspace: dict[str, Path]) -> None:
        """Run SFT for a few steps and verify outputs."""
        from lqh.train.progress import read_progress
        from lqh.train.sft import sft_loop

        project, run = workspace["project"], workspace["run"]
        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(project / "datasets" / "train" / "data.parquet"),
            "eval_dataset": str(project / "datasets" / "eval" / "data.parquet"),
            "eval_on_checkpoints": True,
            "lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj"],
            },
            "training": {
                "num_epochs": 1,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-5,
                "logging_steps": 1,
                "save_steps": 5,
                "gradient_checkpointing": False,
                "bf16": True,
                "max_seq_length": 128,
                "dataloader_num_workers": 0,
            },
        }

        sft_loop(run, config)

        entries = read_progress(run, last_n=100)
        assert entries
        assert entries[-1].get("status") == "completed"

        model_dir = run / "model"
        assert model_dir.exists()
        assert (model_dir / "config.json").exists()

        final_cp = run / "checkpoints" / "final"
        if final_cp.exists():
            assert (final_cp / "predictions.parquet").exists()
            assert (final_cp / "eval_request.json").exists()


@pytest.mark.gpu
class TestInferSubprocess:
    """Inference subprocess end-to-end."""

    def test_infer_subprocess(
        self,
        tmp_path: Path,
        write_chatml_parquet,
        sample_conversations: Callable[[int], list],
    ) -> None:
        run_dir = tmp_path / "runs" / "infer_test"
        run_dir.mkdir(parents=True)
        eval_path = write_chatml_parquet(
            tmp_path / "datasets" / "eval" / "data.parquet",
            sample_conversations(3),
        )

        config = {
            "type": "infer",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(eval_path),
            "max_new_tokens": 32,
        }
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))

        result = subprocess.run(
            [sys.executable, "-m", "lqh.infer", str(config_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=tmp_path,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (run_dir / "predictions.parquet").exists()
        assert (run_dir / "eval_request.json").exists()
        assert (run_dir / "pid").exists()

        table = pq.read_table(str(run_dir / "predictions.parquet"))
        assert len(table) == 3
        assert {"messages", "sample_index"} <= set(table.column_names)

        last_entry = json.loads(
            (run_dir / "progress.jsonl").read_text().strip().split("\n")[-1]
        )
        assert last_entry["status"] == "completed"


@pytest.mark.gpu
class TestSubprocessManagerWithRealProcess:
    """``SubprocessManager`` with a real training subprocess."""

    def test_start_and_monitor_training(
        self,
        tmp_path: Path,
        write_chatml_parquet,
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.subprocess_manager import SubprocessManager

        write_chatml_parquet(
            tmp_path / "datasets" / "train" / "data.parquet",
            sample_conversations(5),
        )

        manager = SubprocessManager()
        run = tmp_path / "runs" / "sft_real"
        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(tmp_path / "datasets" / "train" / "data.parquet"),
            "lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj"],
            },
            "training": {
                "num_epochs": 1,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "logging_steps": 1,
                "save_steps": 999,
                "gradient_checkpointing": False,
                "bf16": True,
                "max_seq_length": 64,
                "dataloader_num_workers": 0,
            },
        }

        pid = manager.start(run, config, project_dir=tmp_path)
        assert pid > 0

        time.sleep(2)
        status = manager.get_status(run)
        assert status.state in ("running", "completed", "failed", "unknown")

        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            status = manager.get_status(run)
            if status.state in ("completed", "failed"):
                break
            time.sleep(5)

        assert status.state == "completed", f"Training failed: {status.error}"
        assert manager.read_progress(run)

        runs = manager.list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0][0] == "sft_real"


@pytest.mark.gpu
class TestSFTWithEvalLoop:
    """Full SFT → checkpoint eval → scoring pipeline."""

    def test_sft_with_checkpoint_eval(
        self,
        tmp_path: Path,
        write_chatml_parquet,
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.sft import sft_loop

        write_chatml_parquet(
            tmp_path / "datasets" / "train" / "data.parquet",
            sample_conversations(10),
        )
        write_chatml_parquet(
            tmp_path / "datasets" / "eval" / "data.parquet",
            sample_conversations(3),
        )
        scorer_dir = tmp_path / "evals" / "scorers"
        scorer_dir.mkdir(parents=True)
        (scorer_dir / "test.md").write_text(
            "Score the response quality from 1 to 10.\n"
            "10 = perfect, 1 = completely wrong."
        )

        run = tmp_path / "runs" / "sft_eval_test"
        run.mkdir(parents=True)

        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(tmp_path / "datasets" / "train" / "data.parquet"),
            "eval_dataset": str(tmp_path / "datasets" / "eval" / "data.parquet"),
            "scorer": "evals/scorers/test.md",
            "eval_on_checkpoints": True,
            "lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj"],
            },
            "training": {
                "num_epochs": 1,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-5,
                "logging_steps": 1,
                "save_steps": 2,
                "gradient_checkpointing": False,
                "bf16": True,
                "max_seq_length": 64,
                "dataloader_num_workers": 0,
            },
        }

        sft_loop(run, config)

        assert (run / "checkpoints").exists()
        final_cp = run / "checkpoints" / "final"
        assert final_cp.exists()
        assert (final_cp / "predictions.parquet").exists()
        assert (final_cp / "eval_request.json").exists()

        pred = pq.read_table(str(final_cp / "predictions.parquet"))
        assert len(pred) == 3
        assert {"sample_index", "messages"} <= set(pred.column_names)

        for i in range(len(pred)):
            msgs = json.loads(pred.column("messages")[i].as_py())
            assert msgs
            assert msgs[-1]["role"] == "assistant"
            assert len(msgs[-1]["content"]) > 0

        assert (run / "model").exists()


@pytest.mark.gpu
class TestDPOEndToEnd:
    """End-to-end on-policy DPO test.

    Runs the full ping-pong loop:
    1. ``dpo_loop`` generates predictions (in a thread)
    2. Main thread detects ``iter_request.json``, builds mock preferences
    3. ``dpo_loop`` picks up ``preferences.parquet``, runs DPO step
    4. Verify iteration artifacts and final model
    """

    def test_dpo_one_iteration(
        self,
        tmp_path: Path,
        write_chatml_parquet,
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.dpo import dpo_loop
        from lqh.train.progress import read_progress

        write_chatml_parquet(
            tmp_path / "datasets" / "eval" / "data.parquet",
            sample_conversations(5),
        )
        write_chatml_parquet(
            tmp_path / "datasets" / "train" / "data.parquet",
            sample_conversations(5),
        )
        run = tmp_path / "runs" / "dpo_test"
        run.mkdir(parents=True)

        config = {
            "type": "on_policy_dpo",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(tmp_path / "datasets" / "train" / "data.parquet"),
            "eval_dataset": str(tmp_path / "datasets" / "eval" / "data.parquet"),
            "num_iterations": 1,
            "dpo_beta": 0.1,
            "golden_source": "dataset",
            "rejection_threshold": 6.0,
            "lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj"],
            },
            "training": {
                "per_device_batch_size": 2,
                "learning_rate": 5e-6,
                "gradient_checkpointing": False,
                "bf16": True,
                "max_seq_length": 128,
                "dataloader_num_workers": 0,
            },
        }

        dpo_error: list[Exception] = []

        def run_dpo() -> None:
            try:
                dpo_loop(run, config)
            except Exception as e:
                dpo_error.append(e)

        thread = threading.Thread(target=run_dpo, daemon=True)
        thread.start()

        iter_dir = run / "iterations" / "iter_000"
        request_file = iter_dir / "iter_request.json"

        deadline = time.monotonic() + 300  # 5 min timeout for model loading
        while time.monotonic() < deadline and not request_file.exists():
            time.sleep(1)

        assert request_file.exists(), (
            "iter_request.json never appeared (DPO generation timed out)"
        )

        predictions_path = iter_dir / "predictions.parquet"
        assert predictions_path.exists()
        pred_table = pq.read_table(str(predictions_path))
        assert len(pred_table) == 5

        # Build mock preferences: training data as chosen, model output as rejected.
        train_convos = sample_conversations(5)
        preferences = []
        for i in range(len(pred_table)):
            pred_msgs = json.loads(pred_table.column("messages")[i].as_py())
            prompt = [m for m in pred_msgs if m["role"] != "assistant"]
            preferences.append({
                "prompt": json.dumps(prompt),
                "chosen": train_convos[i][-1]["content"],
                "rejected": pred_msgs[-1]["content"] if pred_msgs else "bad",
            })

        pq.write_table(
            pa.table({
                "prompt": [p["prompt"] for p in preferences],
                "chosen": [p["chosen"] for p in preferences],
                "rejected": [p["rejected"] for p in preferences],
            }),
            iter_dir / "preferences.parquet",
        )

        thread.join(timeout=300)
        assert not thread.is_alive(), "DPO thread should have finished"
        if dpo_error:
            raise dpo_error[0]

        dpo_result = iter_dir / "dpo_result.json"
        assert dpo_result.exists()
        metrics = json.loads(dpo_result.read_text())
        assert metrics["iteration"] == 0
        assert metrics["num_preferences"] == 5

        entries = read_progress(run, last_n=100)
        assert entries
        assert entries[-1].get("status") == "completed"

        model_dir = run / "model"
        assert model_dir.exists()
        assert (model_dir / "config.json").exists()
