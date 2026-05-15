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
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chatml_parquet(path: Path, conversations: list[list[dict]], *, num: int | None = None) -> None:
    """Write a parquet file with ChatML conversations."""
    if num is not None:
        # Repeat conversations to reach num samples
        while len(conversations) < num:
            conversations = conversations + conversations
        conversations = conversations[:num]

    messages = [json.dumps(conv) for conv in conversations]
    table = pa.table(
        {"messages": messages},
        schema=pa.schema([pa.field("messages", pa.string())]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _sample_conversations(n: int = 5) -> list[list[dict]]:
    """Generate n sample ChatML conversations."""
    convos = []
    for i in range(n):
        convos.append([
            {"role": "user", "content": f"Convert {i}.mp4 to mp3"},
            {"role": "assistant", "content": f"ffmpeg -i {i}.mp4 {i}.mp3"},
        ])
    return convos


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Unit tests: Progress protocol
# ---------------------------------------------------------------------------


class TestProgressProtocol(unittest.TestCase):
    """Test the file-based progress protocol."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir) / "runs" / "test_run"
        self.run_dir.mkdir(parents=True)

    def test_write_and_read_progress(self) -> None:
        from lqh.train.progress import read_latest_progress, read_progress, write_progress

        write_progress(self.run_dir, step=10, loss=2.5, lr=1e-5, epoch=0.5)
        write_progress(self.run_dir, step=20, loss=2.1, lr=9e-6, epoch=1.0)

        entries = read_progress(self.run_dir)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["step"], 10)
        self.assertEqual(entries[1]["step"], 20)
        self.assertAlmostEqual(entries[0]["loss"], 2.5)

        latest = read_latest_progress(self.run_dir)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["step"], 20)

    def test_read_progress_empty(self) -> None:
        from lqh.train.progress import read_latest_progress, read_progress

        entries = read_progress(self.run_dir)
        self.assertEqual(entries, [])

        latest = read_latest_progress(self.run_dir)
        self.assertIsNone(latest)

    def test_write_status(self) -> None:
        from lqh.train.progress import read_latest_progress, write_status

        write_status(self.run_dir, "completed")
        latest = read_latest_progress(self.run_dir)
        self.assertEqual(latest["status"], "completed")

    def test_write_status_failed_with_error(self) -> None:
        from lqh.train.progress import read_latest_progress, write_status

        write_status(self.run_dir, "failed", error="CUDA OOM")
        latest = read_latest_progress(self.run_dir)
        self.assertEqual(latest["status"], "failed")
        self.assertEqual(latest["error"], "CUDA OOM")

    def test_write_eval_request(self) -> None:
        from lqh.train.progress import write_eval_request

        cp_dir = self.run_dir / "checkpoints" / "step_100"
        cp_dir.mkdir(parents=True)
        write_eval_request(cp_dir)

        req = json.loads((cp_dir / "eval_request.json").read_text())
        self.assertEqual(req["status"], "ready")
        self.assertEqual(req["predictions"], "predictions.parquet")

    def test_write_iter_request(self) -> None:
        from lqh.train.progress import write_iter_request

        iter_dir = self.run_dir / "iterations" / "iter_000"
        iter_dir.mkdir(parents=True)
        write_iter_request(iter_dir)

        req = json.loads((iter_dir / "iter_request.json").read_text())
        self.assertEqual(req["status"], "ready")

    def test_read_progress_last_n(self) -> None:
        from lqh.train.progress import read_progress, write_progress

        for i in range(20):
            write_progress(self.run_dir, step=i, loss=float(i))

        entries = read_progress(self.run_dir, last_n=3)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["step"], 17)
        self.assertEqual(entries[-1]["step"], 19)

    def test_wait_for_file_exists(self) -> None:
        from lqh.train.progress import wait_for_file

        target = self.run_dir / "result.json"
        target.write_text('{"ok": true}')

        result = wait_for_file(target, poll_interval=0.01, timeout=1.0)
        self.assertEqual(result, target)

    def test_wait_for_file_timeout(self) -> None:
        from lqh.train.progress import wait_for_file

        target = self.run_dir / "nonexistent.json"
        with self.assertRaises(TimeoutError):
            wait_for_file(target, poll_interval=0.01, timeout=0.05)


# ---------------------------------------------------------------------------
# Unit tests: Subprocess manager
# ---------------------------------------------------------------------------


class TestSubprocessManager(unittest.TestCase):
    """Test subprocess lifecycle management."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)
        self.runs_dir = self.project_dir / "runs"
        self.runs_dir.mkdir(parents=True)

    def test_start_writes_config_and_pid(self) -> None:
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        run_dir = self.runs_dir / "test_001"

        # Spawn a harmless subprocess
        config = {"type": "sft", "base_model": "test", "dataset": "test"}
        # Use a script that just sleeps briefly
        pid = manager.start(
            run_dir,
            config,
            module="time",  # python -m time will just exit quickly
            project_dir=self.project_dir,
        )

        self.assertTrue((run_dir / "config.json").exists())
        self.assertTrue((run_dir / "pid").exists())
        self.assertIsInstance(pid, int)
        self.assertGreater(pid, 0)

        # Config should be valid JSON
        written_config = json.loads((run_dir / "config.json").read_text())
        self.assertEqual(written_config["type"], "sft")

    def test_get_status_unknown_no_progress(self) -> None:
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        run_dir = self.runs_dir / "empty_run"
        run_dir.mkdir()

        status = manager.get_status(run_dir)
        self.assertEqual(status.state, "unknown")

    def test_get_status_completed(self) -> None:
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import write_progress, write_status

        manager = SubprocessManager()
        run_dir = self.runs_dir / "done_run"
        run_dir.mkdir()

        write_progress(run_dir, step=100, loss=0.5)
        write_status(run_dir, "completed")

        status = manager.get_status(run_dir)
        self.assertEqual(status.state, "completed")

    def test_get_status_failed(self) -> None:
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import write_status

        manager = SubprocessManager()
        run_dir = self.runs_dir / "failed_run"
        run_dir.mkdir()

        write_status(run_dir, "failed", error="CUDA OOM")

        status = manager.get_status(run_dir)
        self.assertEqual(status.state, "failed")
        self.assertEqual(status.error, "CUDA OOM")

    def test_list_runs(self) -> None:
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import write_status

        manager = SubprocessManager()

        # Create two runs
        for name in ("sft_001", "dpo_001"):
            run_dir = self.runs_dir / name
            run_dir.mkdir()
            (run_dir / "config.json").write_text('{"type": "sft"}')
            write_status(run_dir, "completed")

        runs = manager.list_runs(self.project_dir)
        self.assertEqual(len(runs), 2)
        names = [r[0] for r in runs]
        self.assertIn("sft_001", names)
        self.assertIn("dpo_001", names)

    def test_list_runs_empty(self) -> None:
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        runs = manager.list_runs(self.project_dir)
        self.assertEqual(runs, [])

    def test_is_alive_no_pid(self) -> None:
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        run_dir = self.runs_dir / "no_pid"
        run_dir.mkdir()

        self.assertFalse(manager.is_alive(run_dir))

    def test_is_alive_dead_pid(self) -> None:
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        run_dir = self.runs_dir / "dead_pid"
        run_dir.mkdir()
        # Write a PID that almost certainly doesn't exist
        (run_dir / "pid").write_text("999999999")

        self.assertFalse(manager.is_alive(run_dir))


# ---------------------------------------------------------------------------
# Unit tests: Data utils
# ---------------------------------------------------------------------------


class TestDataUtils(unittest.TestCase):
    """Test parquet ChatML conversion utilities."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def test_load_chatml_dataset(self) -> None:
        from lqh.train.data_utils import load_chatml_dataset

        path = Path(self.tmpdir) / "data.parquet"
        convos = _sample_conversations(3)
        _make_chatml_parquet(path, convos)

        loaded = load_chatml_dataset(path)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0][0]["role"], "user")
        self.assertEqual(loaded[0][1]["role"], "assistant")

    def test_chatml_to_sft_dataset(self) -> None:
        from lqh.train.data_utils import chatml_to_sft_dataset

        convos = _sample_conversations(2)
        sft_data = chatml_to_sft_dataset(convos)

        self.assertEqual(len(sft_data), 2)
        self.assertIn("messages", sft_data[0])
        self.assertIsInstance(sft_data[0]["messages"], list)
        self.assertEqual(sft_data[0]["messages"][0]["role"], "user")

    def test_chatml_to_dpo_dataset(self) -> None:
        from lqh.train.data_utils import chatml_to_dpo_dataset

        preferences = [
            {
                "prompt": [{"role": "user", "content": "hello"}],
                "chosen": "good response",
                "rejected": "bad response",
            }
        ]
        dpo_data = chatml_to_dpo_dataset(preferences)
        self.assertEqual(len(dpo_data), 1)
        self.assertIn("prompt", dpo_data[0])
        self.assertIn("chosen", dpo_data[0])
        self.assertIn("rejected", dpo_data[0])

    def test_load_preferences_parquet(self) -> None:
        from lqh.train.data_utils import load_preferences_parquet

        path = Path(self.tmpdir) / "preferences.parquet"
        table = pa.table({
            "prompt": [json.dumps([{"role": "user", "content": "hi"}])],
            "chosen": ["good"],
            "rejected": ["bad"],
        })
        pq.write_table(table, path)

        loaded = load_preferences_parquet(path)
        self.assertEqual(len(loaded), 1)
        self.assertIsInstance(loaded[0]["prompt"], list)
        self.assertEqual(loaded[0]["chosen"], "good")
        self.assertEqual(loaded[0]["rejected"], "bad")


# ---------------------------------------------------------------------------
# Unit tests: Sync backend
# ---------------------------------------------------------------------------


class TestSyncBackend(unittest.TestCase):
    """Test sync protocol and LocalSync."""

    def test_local_sync_is_noop(self) -> None:
        from lqh.sync import LocalSync

        sync = LocalSync()
        # Should not raise
        asyncio.run(sync.push([], Path("/tmp")))
        asyncio.run(sync.pull(Path("/tmp"), ["*.json"], Path("/tmp")))

    def test_resolve_manifest(self) -> None:
        from lqh.sync import resolve_manifest

        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "datasets").mkdir()
        (tmpdir / "datasets" / "data.parquet").write_text("dummy")

        config = {
            "dataset": "datasets/data.parquet",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",  # not local
            "manifest": ["dataset", "base_model"],
        }

        paths = resolve_manifest(config, tmpdir)
        # Only the local file should be resolved
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].name == "data.parquet")

    def test_resolve_manifest_empty(self) -> None:
        from lqh.sync import resolve_manifest

        paths = resolve_manifest({}, Path("/tmp"))
        self.assertEqual(paths, [])


# ---------------------------------------------------------------------------
# Unit tests: Tool handler validation
# ---------------------------------------------------------------------------


class TestTrainingToolValidation(unittest.TestCase):
    """Test training tool handlers validate inputs correctly."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)

        # Create a dataset
        ds_dir = self.project_dir / "datasets" / "test_ds"
        ds_dir.mkdir(parents=True)
        _make_chatml_parquet(ds_dir / "data.parquet", _sample_conversations(5))

        # Create a scorer
        scorer_dir = self.project_dir / "evals" / "scorers"
        scorer_dir.mkdir(parents=True)
        (scorer_dir / "test.md").write_text("Score 1-10.")

        # Create permissions that auto-allow
        lqh_dir = self.project_dir / ".lqh"
        lqh_dir.mkdir(parents=True)
        (lqh_dir / "permissions.json").write_text(json.dumps({"project_allow_all": True}))

    @patch("lqh.tools.handlers._check_torch_available", return_value=None)
    def test_start_training_missing_dataset(self, _mock: MagicMock) -> None:
        from lqh.tools.handlers import handle_start_training

        result = asyncio.run(handle_start_training(
            self.project_dir,
            type="sft",
            base_model="test-model",
            dataset="datasets/nonexistent",
        ))
        self.assertIn("not found", result.content)

    @patch("lqh.tools.handlers._check_torch_available", return_value=None)
    def test_start_training_missing_eval_dataset(self, _mock: MagicMock) -> None:
        from lqh.tools.handlers import handle_start_training

        result = asyncio.run(handle_start_training(
            self.project_dir,
            type="sft",
            base_model="test-model",
            dataset="datasets/test_ds",
            eval_dataset="datasets/nonexistent_eval",
        ))
        self.assertIn("not found", result.content)

    @patch("lqh.tools.handlers._check_torch_available", return_value=None)
    def test_start_training_missing_scorer(self, _mock: MagicMock) -> None:
        from lqh.tools.handlers import handle_start_training

        result = asyncio.run(handle_start_training(
            self.project_dir,
            type="sft",
            base_model="test-model",
            dataset="datasets/test_ds",
            scorer="evals/scorers/nonexistent.md",
        ))
        self.assertIn("not found", result.content)

    def test_training_status_no_runs(self) -> None:
        from lqh.tools.handlers import handle_training_status

        result = asyncio.run(handle_training_status(self.project_dir))
        self.assertIn("No training runs", result.content)

    def test_training_status_nonexistent_run(self) -> None:
        from lqh.tools.handlers import handle_training_status

        result = asyncio.run(handle_training_status(
            self.project_dir,
            run_name="nonexistent",
        ))
        self.assertIn("not found", result.content)

    def test_stop_training_nonexistent_run(self) -> None:
        from lqh.tools.handlers import handle_stop_training

        result = asyncio.run(handle_stop_training(
            self.project_dir,
            run_name="nonexistent",
        ))
        self.assertIn("not found", result.content)

    def test_stop_training_not_running(self) -> None:
        from lqh.tools.handlers import handle_stop_training
        from lqh.train.progress import write_status

        # Create a completed run
        run_dir = self.project_dir / "runs" / "done_run"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text('{"type": "sft"}')
        write_status(run_dir, "completed")

        result = asyncio.run(handle_stop_training(
            self.project_dir,
            run_name="done_run",
        ))
        self.assertIn("not currently running", result.content)

    def test_training_status_with_completed_run(self) -> None:
        from lqh.tools.handlers import handle_training_status
        from lqh.train.progress import write_progress, write_status

        run_dir = self.project_dir / "runs" / "sft_001"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text('{"type": "sft"}')
        write_progress(run_dir, step=100, loss=0.5, lr=1e-5, epoch=3.0)
        write_status(run_dir, "completed")

        result = asyncio.run(handle_training_status(
            self.project_dir,
            run_name="sft_001",
        ))
        self.assertIn("completed", result.content)
        self.assertIn("sft_001", result.content)

    @patch("lqh.tools.handlers._check_torch_available", return_value=None)
    def test_start_local_eval_missing_model(self, _mock: MagicMock) -> None:
        from lqh.tools.handlers import handle_start_local_eval

        result = asyncio.run(handle_start_local_eval(
            self.project_dir,
            model_path="runs/nonexistent/model",
            dataset="datasets/test_ds",
            scorer="evals/scorers/test.md",
        ))
        self.assertIn("not found", result.content)

    def test_next_run_name_generation(self) -> None:
        from lqh.tools.handlers import _next_run_name

        # No runs dir yet
        name = _next_run_name(self.project_dir, "sft")
        self.assertEqual(name, "sft_001")

        # Create some runs
        (self.project_dir / "runs" / "sft_001").mkdir(parents=True)
        (self.project_dir / "runs" / "sft_002").mkdir(parents=True)

        name = _next_run_name(self.project_dir, "sft")
        self.assertEqual(name, "sft_003")

        name = _next_run_name(self.project_dir, "dpo")
        self.assertEqual(name, "dpo_001")


# ---------------------------------------------------------------------------
# Unit tests: Golden trajectory assembly
# ---------------------------------------------------------------------------


class TestGoldenAssembly(unittest.TestCase):
    """Test golden trajectory generation and preference pair assembly."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir) / "iter_000"
        self.output_dir.mkdir(parents=True)

    def _make_predictions_and_scores(self, scores: list[float]) -> tuple[Path, Path]:
        """Create predictions.parquet and results.parquet with given scores."""
        n = len(scores)
        convos = []
        for i in range(n):
            convos.append([
                {"role": "user", "content": f"prompt {i}"},
                {"role": "assistant", "content": f"bad response {i}"},
            ])

        pred_path = self.output_dir / "predictions.parquet"
        pred_table = pa.table({
            "sample_index": list(range(n)),
            "messages": [json.dumps(c) for c in convos],
        })
        pq.write_table(pred_table, pred_path)

        scores_path = self.output_dir / "results.parquet"
        score_table = pa.table({
            "sample_index": list(range(n)),
            "score": scores,
            "reasoning": [f"reason {i}" for i in range(n)],
            "messages": [json.dumps(c) for c in convos],
        })
        pq.write_table(score_table, scores_path)

        return pred_path, scores_path

    def test_golden_from_dataset(self) -> None:
        """Test golden_source='dataset' pulls from training data."""
        from lqh.golden import generate_golden

        pred_path, scores_path = self._make_predictions_and_scores([3.0, 8.0, 2.0])

        # Create "training dataset" with better responses
        ds_path = Path(self.tmpdir) / "train_data.parquet"
        train_convos = [
            [{"role": "user", "content": "prompt 0"}, {"role": "assistant", "content": "good response 0"}],
            [{"role": "user", "content": "prompt 1"}, {"role": "assistant", "content": "good response 1"}],
            [{"role": "user", "content": "prompt 2"}, {"role": "assistant", "content": "good response 2"}],
        ]
        _make_chatml_parquet(ds_path, train_convos)

        config = {
            "golden_source": "dataset",
            "rejection_threshold": 6.0,
            "dataset": str(ds_path),
        }

        asyncio.run(generate_golden(
            predictions_path=pred_path,
            scores_path=scores_path,
            dataset_path=str(ds_path),
            config=config,
            client=MagicMock(),
            output_dir=self.output_dir,
        ))

        # Should have preferences for samples 0 and 2 (scores 3.0 and 2.0)
        prefs_path = self.output_dir / "preferences.parquet"
        self.assertTrue(prefs_path.exists())

        prefs_table = pq.read_table(str(prefs_path))
        self.assertEqual(len(prefs_table), 2)

    def test_no_low_scorers_writes_empty_preferences(self) -> None:
        """When all scores are above threshold, write empty preferences."""
        from lqh.golden import generate_golden

        pred_path, scores_path = self._make_predictions_and_scores([9.0, 8.0, 7.0])

        config = {
            "golden_source": "dataset",
            "rejection_threshold": 6.0,
        }

        asyncio.run(generate_golden(
            predictions_path=pred_path,
            scores_path=scores_path,
            dataset_path="",
            config=config,
            client=MagicMock(),
            output_dir=self.output_dir,
        ))

        prefs_path = self.output_dir / "preferences.parquet"
        self.assertTrue(prefs_path.exists())
        prefs_table = pq.read_table(str(prefs_path))
        self.assertEqual(len(prefs_table), 0)


# ===========================================================================
# GPU tests: require torch + CUDA
# ===========================================================================


@unittest.skipUnless(_has_cuda(), "Requires CUDA GPU")
class TestSFTEndToEnd(unittest.TestCase):
    """End-to-end SFT training test with a tiny model and dataset.

    Runs a real training loop for a few steps, verifies checkpoints,
    progress output, and the eval loop.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)
        self.run_dir = self.project_dir / "runs" / "sft_test"
        self.run_dir.mkdir(parents=True)

        # Create a tiny training dataset (10 samples)
        ds_dir = self.project_dir / "datasets" / "train"
        _make_chatml_parquet(
            ds_dir / "data.parquet",
            _sample_conversations(10),
        )

        # Create a tiny eval dataset (3 samples)
        eval_dir = self.project_dir / "datasets" / "eval"
        _make_chatml_parquet(
            eval_dir / "data.parquet",
            _sample_conversations(3),
        )

    def test_sft_training_loop(self) -> None:
        """Run SFT for a few steps and verify outputs."""
        from lqh.train.progress import read_progress
        from lqh.train.sft import sft_loop

        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(self.project_dir / "datasets" / "train" / "data.parquet"),
            "eval_dataset": str(self.project_dir / "datasets" / "eval" / "data.parquet"),
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

        sft_loop(self.run_dir, config)

        # Verify progress was written
        entries = read_progress(self.run_dir, last_n=100)
        self.assertGreater(len(entries), 0)

        # Should have a completed status
        last = entries[-1]
        self.assertEqual(last.get("status"), "completed")

        # Verify final model was saved
        model_dir = self.run_dir / "model"
        self.assertTrue(model_dir.exists())
        self.assertTrue((model_dir / "config.json").exists())

        # Verify final checkpoint eval was triggered
        final_cp = self.run_dir / "checkpoints" / "final"
        if final_cp.exists():
            self.assertTrue((final_cp / "predictions.parquet").exists())
            self.assertTrue((final_cp / "eval_request.json").exists())


@unittest.skipUnless(_has_cuda(), "Requires CUDA GPU")
class TestInferSubprocess(unittest.TestCase):
    """Test the inference subprocess end-to-end."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir) / "runs" / "infer_test"
        self.run_dir.mkdir(parents=True)

        # Create eval dataset
        eval_dir = Path(self.tmpdir) / "datasets" / "eval"
        _make_chatml_parquet(
            eval_dir / "data.parquet",
            _sample_conversations(3),
        )
        self.eval_path = eval_dir / "data.parquet"

    def test_infer_subprocess(self) -> None:
        """Spawn lqh.infer as a real subprocess and verify outputs."""
        config = {
            "type": "infer",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(self.eval_path),
            "max_new_tokens": 32,
        }
        config_path = self.run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))

        result = subprocess.run(
            [sys.executable, "-m", "lqh.infer", str(config_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=self.tmpdir,
        )

        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

        # Verify outputs
        self.assertTrue((self.run_dir / "predictions.parquet").exists())
        self.assertTrue((self.run_dir / "eval_request.json").exists())
        self.assertTrue((self.run_dir / "pid").exists())

        # Verify predictions have correct shape
        table = pq.read_table(str(self.run_dir / "predictions.parquet"))
        self.assertEqual(len(table), 3)
        self.assertIn("messages", table.column_names)
        self.assertIn("sample_index", table.column_names)

        # Verify progress shows completed
        progress_file = self.run_dir / "progress.jsonl"
        self.assertTrue(progress_file.exists())
        lines = progress_file.read_text().strip().split("\n")
        last_entry = json.loads(lines[-1])
        self.assertEqual(last_entry["status"], "completed")


@unittest.skipUnless(_has_cuda(), "Requires CUDA GPU")
class TestSubprocessManagerWithRealProcess(unittest.TestCase):
    """Test SubprocessManager with a real training subprocess."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)

        # Create dataset
        ds_dir = self.project_dir / "datasets" / "train"
        _make_chatml_parquet(
            ds_dir / "data.parquet",
            _sample_conversations(5),
        )

    def test_start_and_monitor_training(self) -> None:
        """Start a real training subprocess and monitor it."""
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        run_dir = self.project_dir / "runs" / "sft_real"

        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(self.project_dir / "datasets" / "train" / "data.parquet"),
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

        pid = manager.start(run_dir, config, project_dir=self.project_dir)
        self.assertGreater(pid, 0)

        # Should be alive initially
        time.sleep(2)
        # It might still be loading or might have finished quickly
        status = manager.get_status(run_dir)
        self.assertIn(status.state, ("running", "completed", "failed", "unknown"))

        # Wait for completion (max 5 minutes)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            status = manager.get_status(run_dir)
            if status.state in ("completed", "failed"):
                break
            time.sleep(5)

        # Check final status
        self.assertEqual(status.state, "completed", f"Training failed: {status.error}")

        # Should have progress entries
        entries = manager.read_progress(run_dir)
        self.assertGreater(len(entries), 0)

        # Should show up in list_runs
        runs = manager.list_runs(self.project_dir)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][0], "sft_real")


@unittest.skipUnless(_has_cuda(), "Requires CUDA GPU")
class TestSFTWithEvalLoop(unittest.TestCase):
    """Test the full SFT → checkpoint eval → scoring pipeline.

    This test:
    1. Starts SFT training with checkpoint saving
    2. The training subprocess generates eval predictions at checkpoints
    3. Verifies that eval_request.json and predictions.parquet are created
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)

        # Training data
        train_dir = self.project_dir / "datasets" / "train"
        _make_chatml_parquet(
            train_dir / "data.parquet",
            _sample_conversations(10),
        )

        # Eval data
        eval_dir = self.project_dir / "datasets" / "eval"
        _make_chatml_parquet(
            eval_dir / "data.parquet",
            _sample_conversations(3),
        )

        # Scorer
        scorer_dir = self.project_dir / "evals" / "scorers"
        scorer_dir.mkdir(parents=True)
        (scorer_dir / "test.md").write_text(
            "Score the response quality from 1 to 10.\n"
            "10 = perfect, 1 = completely wrong."
        )

    def test_sft_with_checkpoint_eval(self) -> None:
        """Run SFT with save_steps=2 and verify checkpoint eval outputs."""
        from lqh.train.sft import sft_loop

        run_dir = self.project_dir / "runs" / "sft_eval_test"
        run_dir.mkdir(parents=True)

        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(self.project_dir / "datasets" / "train" / "data.parquet"),
            "eval_dataset": str(self.project_dir / "datasets" / "eval" / "data.parquet"),
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

        sft_loop(run_dir, config)

        # Verify checkpoint structure
        checkpoints_dir = run_dir / "checkpoints"
        self.assertTrue(checkpoints_dir.exists())

        # Should have the final checkpoint with eval
        final_cp = checkpoints_dir / "final"
        self.assertTrue(final_cp.exists(), "Final checkpoint dir should exist")
        self.assertTrue(
            (final_cp / "predictions.parquet").exists(),
            "Final checkpoint should have predictions",
        )
        self.assertTrue(
            (final_cp / "eval_request.json").exists(),
            "Final checkpoint should have eval_request.json",
        )

        # Verify predictions have correct structure
        pred_table = pq.read_table(str(final_cp / "predictions.parquet"))
        self.assertEqual(len(pred_table), 3)  # 3 eval samples
        self.assertIn("sample_index", pred_table.column_names)
        self.assertIn("messages", pred_table.column_names)

        # Verify each prediction has a model-generated assistant response
        for i in range(len(pred_table)):
            msgs = json.loads(pred_table.column("messages")[i].as_py())
            self.assertGreater(len(msgs), 0)
            self.assertEqual(msgs[-1]["role"], "assistant")
            self.assertGreater(len(msgs[-1]["content"]), 0)

        # Verify final model saved
        model_dir = run_dir / "model"
        self.assertTrue(model_dir.exists())


@unittest.skipUnless(_has_cuda(), "Requires CUDA GPU")
class TestDPOEndToEnd(unittest.TestCase):
    """End-to-end on-policy DPO test.

    Runs the full ping-pong loop:
    1. dpo_loop generates predictions (in a thread)
    2. Main thread detects iter_request.json, builds mock preferences
    3. dpo_loop picks up preferences.parquet, runs DPO step
    4. Verify iteration artifacts and final model
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)
        self.run_dir = self.project_dir / "runs" / "dpo_test"
        self.run_dir.mkdir(parents=True)

        # Eval dataset (used for generation prompts)
        eval_dir = self.project_dir / "datasets" / "eval"
        _make_chatml_parquet(
            eval_dir / "data.parquet",
            _sample_conversations(5),
        )

        # Training dataset (used as golden source)
        train_dir = self.project_dir / "datasets" / "train"
        _make_chatml_parquet(
            train_dir / "data.parquet",
            _sample_conversations(5),
        )

    def test_dpo_one_iteration(self) -> None:
        """Run DPO for 1 iteration with mock preference assembly."""
        import threading

        from lqh.train.dpo import dpo_loop
        from lqh.train.progress import read_progress

        config = {
            "type": "on_policy_dpo",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(self.project_dir / "datasets" / "train" / "data.parquet"),
            "eval_dataset": str(self.project_dir / "datasets" / "eval" / "data.parquet"),
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

        # Run dpo_loop in a thread so we can supply preferences from main
        dpo_error: list[Exception] = []

        def run_dpo() -> None:
            try:
                dpo_loop(self.run_dir, config)
            except Exception as e:
                dpo_error.append(e)

        thread = threading.Thread(target=run_dpo, daemon=True)
        thread.start()

        # Wait for iter_000/iter_request.json to appear
        iter_dir = self.run_dir / "iterations" / "iter_000"
        request_file = iter_dir / "iter_request.json"

        deadline = time.monotonic() + 300  # 5 min timeout for model loading
        while time.monotonic() < deadline:
            if request_file.exists():
                break
            time.sleep(1)
        else:
            self.fail("iter_request.json never appeared (DPO generation timed out)")

        # Verify predictions were written
        predictions_path = iter_dir / "predictions.parquet"
        self.assertTrue(predictions_path.exists(), "predictions.parquet should exist")
        pred_table = pq.read_table(str(predictions_path))
        self.assertEqual(len(pred_table), 5)  # 5 eval samples

        # Build mock preferences: use predictions as rejected, training data as chosen
        # This simulates what the watcher+golden module would do
        train_convos = _sample_conversations(5)
        preferences = []
        for i in range(len(pred_table)):
            pred_msgs = json.loads(pred_table.column("messages")[i].as_py())
            # Extract the user prompt
            prompt = [m for m in pred_msgs if m["role"] != "assistant"]
            # Rejected = model's current response
            rejected = pred_msgs[-1]["content"] if pred_msgs else "bad"
            # Chosen = training data response
            chosen = train_convos[i][-1]["content"]

            preferences.append({
                "prompt": json.dumps(prompt),
                "chosen": chosen,
                "rejected": rejected,
            })

        prefs_table = pa.table({
            "prompt": [p["prompt"] for p in preferences],
            "chosen": [p["chosen"] for p in preferences],
            "rejected": [p["rejected"] for p in preferences],
        })
        prefs_path = iter_dir / "preferences.parquet"
        pq.write_table(prefs_table, prefs_path)
        print(f"Wrote {len(preferences)} preference pairs to {prefs_path}")

        # Wait for DPO to complete
        thread.join(timeout=300)
        self.assertFalse(thread.is_alive(), "DPO thread should have finished")

        if dpo_error:
            raise dpo_error[0]

        # Verify DPO step metrics
        dpo_result = iter_dir / "dpo_result.json"
        self.assertTrue(dpo_result.exists(), "dpo_result.json should exist")
        metrics = json.loads(dpo_result.read_text())
        self.assertEqual(metrics["iteration"], 0)
        self.assertEqual(metrics["num_preferences"], 5)

        # Verify progress
        entries = read_progress(self.run_dir, last_n=100)
        self.assertGreater(len(entries), 0)
        last = entries[-1]
        self.assertEqual(last.get("status"), "completed")

        # Verify final model saved
        model_dir = self.run_dir / "model"
        self.assertTrue(model_dir.exists(), "Final model should be saved")
        self.assertTrue((model_dir / "config.json").exists())

        print(f"DPO test passed: 1 iteration, {len(preferences)} preferences, model saved")


if __name__ == "__main__":
    unittest.main()
