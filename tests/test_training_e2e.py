"""End-to-end training test: download a real dataset and fine-tune LFM2.5-1.2B.

Requires:
  - CUDA GPU
  - pip install lqh[train]
  - Internet access (downloads dataset + model from HuggingFace)

Run directly:
    python -m tests.test_training_e2e

Or via unittest:
    python -m pytest tests/test_training_e2e.py -v -s
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_URL = (
    "https://huggingface.co/datasets/mlech26l/shell-helper/resolve/main/"
    "data/shell_helper_dataset_many_opt.parquet?download=true"
)

# We use a small subset for eval so the test doesn't take forever
EVAL_SIZE = 10
TRAIN_MAX = 200  # cap training samples to keep it fast


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _download_dataset(dest: Path) -> Path:
    """Download the shell-helper parquet and return its path."""
    import urllib.request

    raw_path = dest / "shell_helper_raw.parquet"
    if not raw_path.exists():
        print(f"Downloading dataset to {raw_path}...")
        urllib.request.urlretrieve(DATASET_URL, raw_path)
        print(f"Downloaded {raw_path.stat().st_size / 1024:.0f} KB")
    return raw_path


def _convert_to_chatml(raw_path: Path, output_path: Path, max_samples: int | None = None) -> int:
    """Convert HF chat dataset to lqh ChatML parquet format.

    The shell-helper dataset has a 'messages' column that is already a list
    of {role, content} dicts (HF chat format). We JSON-encode each
    conversation into a single string column, which is lqh's standard format.
    """
    table = pq.read_table(str(raw_path))
    n = len(table)
    if max_samples and n > max_samples:
        table = table.slice(0, max_samples)
        n = max_samples

    messages_col = table.column("messages")
    chatml_strings = []

    for i in range(n):
        raw = messages_col[i].as_py()
        # HF stores messages as list of structs — convert to list of dicts
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], dict):
                chatml_strings.append(json.dumps(raw))
            else:
                # Struct format: convert
                chatml_strings.append(json.dumps([dict(m) for m in raw]))
        else:
            chatml_strings.append(json.dumps(raw))

    out_table = pa.table(
        {"messages": chatml_strings},
        schema=pa.schema([pa.field("messages", pa.string())]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_path)
    return n


@unittest.skipUnless(_has_cuda(), "Requires CUDA GPU")
class TestTrainingE2EShellHelper(unittest.TestCase):
    """Download the shell-helper dataset and fine-tune LFM2.5-1.2B-Instruct.

    This test:
    1. Downloads the parquet from HuggingFace
    2. Converts to lqh ChatML format
    3. Splits into train/eval
    4. Runs SFT for 1 epoch via the subprocess manager
    5. Verifies the model is saved and training completed
    6. Runs inference on a few prompts to verify the model generates output
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="lqh_e2e_")
        cls.project_dir = Path(cls.tmpdir)
        print(f"\nE2E test workspace: {cls.project_dir}")

        # Download
        raw_path = _download_dataset(cls.project_dir)

        # Convert and split
        train_dir = cls.project_dir / "datasets" / "shell_helper"
        eval_dir = cls.project_dir / "datasets" / "shell_helper_eval"

        # Read raw, split into train and eval
        raw_table = pq.read_table(str(raw_path))
        total = len(raw_table)
        print(f"Raw dataset: {total} samples")

        # Use first EVAL_SIZE as eval, rest as train (capped)
        eval_table = raw_table.slice(0, EVAL_SIZE)
        train_start = EVAL_SIZE
        train_end = min(total, train_start + TRAIN_MAX)
        train_table = raw_table.slice(train_start, train_end - train_start)

        # Convert eval
        n_eval = _convert_to_chatml_from_table(eval_table, eval_dir / "data.parquet")
        print(f"Eval dataset: {n_eval} samples")

        # Convert train
        n_train = _convert_to_chatml_from_table(train_table, train_dir / "data.parquet")
        print(f"Train dataset: {n_train} samples")

        cls.train_path = train_dir / "data.parquet"
        cls.eval_path = eval_dir / "data.parquet"

    def test_sft_full_epoch(self) -> None:
        """Train LFM2.5-1.2B on shell-helper for 1 epoch."""
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import read_progress

        run_dir = self.project_dir / "runs" / "e2e_sft"

        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(self.train_path),
            "eval_dataset": str(self.eval_path),
            "eval_on_checkpoints": True,
            "lora": {
                "enabled": True,
                "r": 16,
                "alpha": 32,
                "dropout": 0.02,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "in_proj", "out_proj", "w1", "w2", "w3",
                ],
            },
            "training": {
                "num_epochs": 1,
                "per_device_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.1,
                "logging_steps": 5,
                "save_steps": 50,
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 512,
                "dataloader_num_workers": 2,
            },
        }

        # Start training via subprocess manager
        manager = SubprocessManager()
        pid = manager.start(run_dir, config, project_dir=self.project_dir)
        print(f"\nTraining started (PID {pid})")

        # Wait for completion (max 20 minutes)
        deadline = time.monotonic() + 1200
        last_step = -1
        while time.monotonic() < deadline:
            status = manager.get_status(run_dir)

            # Print progress updates
            if status.step is not None and status.step != last_step:
                last_step = status.step
                loss_str = f"loss={status.loss:.4f}" if status.loss else ""
                print(f"  Step {status.step} {loss_str}")

            if status.state in ("completed", "failed"):
                break
            time.sleep(5)

        # Dump stderr on failure
        if status.state == "failed":
            stderr_log = run_dir / "stderr.log"
            if stderr_log.exists():
                print(f"\n--- stderr.log ---\n{stderr_log.read_text()[-2000:]}")

        self.assertEqual(
            status.state, "completed",
            f"Training failed: {status.error}",
        )
        print(f"Training completed at step {status.step}")

        # Verify model saved
        model_dir = run_dir / "model"
        self.assertTrue(model_dir.exists(), "Final model dir should exist")
        self.assertTrue(
            (model_dir / "config.json").exists(),
            "Model config.json should exist",
        )

        # Verify progress has multiple entries
        entries = read_progress(run_dir, last_n=100)
        training_entries = [e for e in entries if "step" in e and "status" not in e]
        self.assertGreater(
            len(training_entries), 0,
            "Should have logged training steps",
        )

        # Verify final checkpoint eval was produced
        final_cp = run_dir / "checkpoints" / "final"
        if final_cp.exists():
            self.assertTrue(
                (final_cp / "predictions.parquet").exists(),
                "Final checkpoint should have predictions",
            )
            self.assertTrue(
                (final_cp / "eval_request.json").exists(),
                "Final checkpoint should have eval_request",
            )

            # Verify predictions are sensible
            pred_table = pq.read_table(str(final_cp / "predictions.parquet"))
            self.assertEqual(len(pred_table), EVAL_SIZE)

            # Check model actually generated something
            for i in range(len(pred_table)):
                msgs = json.loads(pred_table.column("messages")[i].as_py())
                self.assertEqual(msgs[-1]["role"], "assistant")
                response = msgs[-1]["content"]
                self.assertGreater(
                    len(response), 0,
                    f"Sample {i}: model should generate a non-empty response",
                )

        print("\nRunning post-training inference check...")
        self._verify_inference(model_dir)

    def _verify_inference(self, model_dir: Path) -> None:
        """Load the fine-tuned model and verify it generates shell commands."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        prompts = [
            "convert video.mp4 to mp3",
            "list all .py files recursively",
            "find files larger than 100MB",
        ]

        model.eval()
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=128,
                    do_sample=False,
                )
            response = tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
            print(f"  Prompt: {prompt}")
            print(f"  Response: {response[:200]}")
            print()

            # The model should generate *something*
            self.assertGreater(len(response.strip()), 0)

        del model
        torch.cuda.empty_cache()


def _convert_to_chatml_from_table(table: pa.Table, output_path: Path) -> int:
    """Convert a PyArrow table with HF chat messages to lqh ChatML parquet."""
    n = len(table)
    messages_col = table.column("messages")
    chatml_strings = []

    for i in range(n):
        raw = messages_col[i].as_py()
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], dict):
                chatml_strings.append(json.dumps(raw))
            else:
                chatml_strings.append(json.dumps([dict(m) for m in raw]))
        else:
            chatml_strings.append(json.dumps(raw))

    out_table = pa.table(
        {"messages": chatml_strings},
        schema=pa.schema([pa.field("messages", pa.string())]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_path)
    return n


@unittest.skipUnless(_has_cuda(), "Requires CUDA GPU")
class TestTrainingE2ECrashRecovery(unittest.TestCase):
    """Verify that a subprocess crash (e.g. CUDA OOM) is detected and reported.

    We provoke a crash by requesting an absurdly large batch size and
    sequence length that will exceed GPU memory. The test verifies:
    1. The subprocess exits with a non-zero status
    2. The main process detects the failure via progress.jsonl or PID check
    3. stderr.log contains the error details
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="lqh_crash_")
        self.project_dir = Path(self.tmpdir)

        # Create a small dataset
        train_dir = self.project_dir / "datasets" / "crash_test"
        train_dir.mkdir(parents=True)
        convos = []
        for i in range(20):
            convos.append([
                {"role": "user", "content": f"Convert file_{i}.mp4 to mp3"},
                {"role": "assistant", "content": f"ffmpeg -i file_{i}.mp4 file_{i}.mp3"},
            ])
        table = pa.table(
            {"messages": [json.dumps(c) for c in convos]},
            schema=pa.schema([pa.field("messages", pa.string())]),
        )
        pq.write_table(table, train_dir / "data.parquet")
        self.train_path = train_dir / "data.parquet"

    def test_oom_crash_detected(self) -> None:
        """Subprocess should crash with OOM and main process should detect it."""
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        run_dir = self.project_dir / "runs" / "oom_crash"

        # Absurd config: huge batch size + long sequences + no gradient checkpointing
        # This should OOM on any reasonable GPU
        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(self.train_path),
            "lora": {"enabled": False},  # full fine-tune = more memory
            "training": {
                "num_epochs": 1,
                "per_device_batch_size": 512,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "logging_steps": 1,
                "save_steps": 9999,
                "gradient_checkpointing": False,
                "bf16": True,
                "max_seq_length": 8192,
                "dataloader_num_workers": 0,
            },
        }

        pid = manager.start(run_dir, config, project_dir=self.project_dir)
        print(f"\nCrash test started (PID {pid})")

        # Wait for the subprocess to die (should be fast — OOM during first step)
        deadline = time.monotonic() + 180  # 3 min max
        while time.monotonic() < deadline:
            status = manager.get_status(run_dir)
            if status.state in ("completed", "failed"):
                break
            if not manager.is_alive(run_dir):
                # Give it a moment for progress.jsonl to be flushed
                time.sleep(2)
                status = manager.get_status(run_dir)
                break
            time.sleep(3)

        print(f"Subprocess state: {status.state}")
        if status.error:
            print(f"Error: {status.error}")

        # The subprocess should have failed
        self.assertEqual(status.state, "failed", "Subprocess should have crashed")

        # stderr.log should contain the error
        stderr_log = run_dir / "stderr.log"
        self.assertTrue(stderr_log.exists(), "stderr.log should exist")
        stderr_content = stderr_log.read_text()
        print(f"\n--- stderr.log (last 1000 chars) ---\n{stderr_content[-1000:]}")

        # Should contain some indication of the crash (OOM, CUDA, RuntimeError, etc.)
        error_indicators = [
            "CUDA out of memory",
            "OutOfMemoryError",
            "RuntimeError",
            "Error",
            "Traceback",
        ]
        found = any(indicator in stderr_content for indicator in error_indicators)
        self.assertTrue(
            found,
            f"stderr should contain an error indicator, got:\n{stderr_content[-500:]}",
        )

        # The model directory should NOT exist (training never completed)
        model_dir = run_dir / "model"
        self.assertFalse(
            model_dir.exists(),
            "Model dir should not exist after a crash",
        )

        print("Crash correctly detected and reported.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
