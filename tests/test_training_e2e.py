"""End-to-end training test: download a real dataset and fine-tune LFM2.5-1.2B.

Requires:
    - CUDA GPU
    - ``pip install lqh[train]``
    - Internet access (downloads dataset + model from HuggingFace)

Run directly::

    python -m tests.test_training_e2e

Or via pytest::

    pytest tests/test_training_e2e.py -v -s

Both tests are gated on ``@pytest.mark.gpu``; ``conftest.py`` auto-skips
them when no CUDA device is visible.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


DATASET_URL = (
    "https://huggingface.co/datasets/mlech26l/shell-helper/resolve/main/"
    "data/shell_helper_dataset_many_opt.parquet?download=true"
)

# A small subset for eval so the test doesn't take forever.
EVAL_SIZE = 10
TRAIN_MAX = 200  # cap training samples to keep it fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_dataset(dest: Path) -> Path:
    """Download the shell-helper parquet and return its path."""
    raw_path = dest / "shell_helper_raw.parquet"
    if not raw_path.exists():
        print(f"Downloading dataset to {raw_path}...")
        urllib.request.urlretrieve(DATASET_URL, raw_path)
        print(f"Downloaded {raw_path.stat().st_size / 1024:.0f} KB")
    return raw_path


def _convert_to_chatml_from_table(table: pa.Table, output_path: Path) -> int:
    """Convert a PyArrow table with HF chat messages to lqh ChatML parquet."""
    n = len(table)
    messages_col = table.column("messages")
    chatml_strings: list[str] = []

    for i in range(n):
        raw = messages_col[i].as_py()
        if isinstance(raw, list) and raw:
            first = raw[0]
            if isinstance(first, dict):
                chatml_strings.append(json.dumps(raw))
            else:
                chatml_strings.append(json.dumps([dict(m) for m in raw]))
        else:
            chatml_strings.append(json.dumps(raw))

    pq.write_table(
        pa.table(
            {"messages": chatml_strings},
            schema=pa.schema([pa.field("messages", pa.string())]),
        ),
        output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return n


# ---------------------------------------------------------------------------
# Shell-helper SFT E2E
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shell_helper_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Download + slice the shell-helper dataset once per module run."""
    project = tmp_path_factory.mktemp("lqh_e2e")
    print(f"\nE2E test workspace: {project}")

    raw_path = _download_dataset(project)
    raw_table = pq.read_table(str(raw_path))
    total = len(raw_table)
    print(f"Raw dataset: {total} samples")

    train_dir = project / "datasets" / "shell_helper"
    eval_dir = project / "datasets" / "shell_helper_eval"

    eval_table = raw_table.slice(0, EVAL_SIZE)
    train_end = min(total, EVAL_SIZE + TRAIN_MAX)
    train_table = raw_table.slice(EVAL_SIZE, train_end - EVAL_SIZE)

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    n_eval = _convert_to_chatml_from_table(eval_table, eval_dir / "data.parquet")
    n_train = _convert_to_chatml_from_table(train_table, train_dir / "data.parquet")
    print(f"Eval: {n_eval}  Train: {n_train}")

    return {
        "project": project,
        "train_path": train_dir / "data.parquet",
        "eval_path": eval_dir / "data.parquet",
    }


@pytest.mark.gpu
class TestTrainingE2EShellHelper:
    """Download the shell-helper dataset and fine-tune LFM2.5-1.2B-Instruct."""

    def test_sft_full_epoch(self, shell_helper_workspace: dict[str, Path]) -> None:
        """Train LFM2.5-1.2B on shell-helper for 1 epoch."""
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import read_progress

        project = shell_helper_workspace["project"]
        run = project / "runs" / "e2e_sft"

        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(shell_helper_workspace["train_path"]),
            "eval_dataset": str(shell_helper_workspace["eval_path"]),
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

        manager = SubprocessManager()
        pid = manager.start(run, config, project_dir=project)
        print(f"\nTraining started (PID {pid})")

        deadline = time.monotonic() + 1200  # 20 min cap
        last_step = -1
        while time.monotonic() < deadline:
            status = manager.get_status(run)

            if status.step is not None and status.step != last_step:
                last_step = status.step
                loss_str = f"loss={status.loss:.4f}" if status.loss else ""
                print(f"  Step {status.step} {loss_str}")

            if status.state in ("completed", "failed"):
                break
            time.sleep(5)

        if status.state == "failed":
            stderr_log = run / "stderr.log"
            if stderr_log.exists():
                print(f"\n--- stderr.log ---\n{stderr_log.read_text()[-2000:]}")

        assert status.state == "completed", f"Training failed: {status.error}"
        print(f"Training completed at step {status.step}")

        model_dir = run / "model"
        assert model_dir.exists()
        assert (model_dir / "config.json").exists()

        entries = read_progress(run, last_n=100)
        training_entries = [e for e in entries if "step" in e and "status" not in e]
        assert training_entries, "Should have logged training steps"

        final_cp = run / "checkpoints" / "final"
        if final_cp.exists():
            assert (final_cp / "predictions.parquet").exists()
            assert (final_cp / "eval_request.json").exists()

            pred_table = pq.read_table(str(final_cp / "predictions.parquet"))
            assert len(pred_table) == EVAL_SIZE

            for i in range(len(pred_table)):
                msgs = json.loads(pred_table.column("messages")[i].as_py())
                assert msgs[-1]["role"] == "assistant"
                assert len(msgs[-1]["content"]) > 0

        print("\nRunning post-training inference check...")
        self._verify_inference(model_dir)

    @staticmethod
    def _verify_inference(model_dir: Path) -> None:
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
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids, max_new_tokens=128, do_sample=False,
                )
            response = tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True,
            )
            print(f"  Prompt: {prompt}")
            print(f"  Response: {response[:200]}\n")

            assert len(response.strip()) > 0

        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Crash-recovery test
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestTrainingE2ECrashRecovery:
    """Verify that a subprocess crash (e.g. CUDA OOM) is detected and reported.

    We provoke a crash by requesting an absurdly large batch size and
    sequence length that will exceed GPU memory.  The test verifies:

    1. The subprocess exits with a non-zero status.
    2. The main process detects the failure via ``progress.jsonl`` or
       a PID check.
    3. ``stderr.log`` contains the error details.
    """

    @pytest.fixture
    def crash_workspace(
        self, tmp_path: Path, write_chatml_parquet,
    ) -> dict[str, Path]:
        convos = [
            [
                {"role": "user", "content": f"Convert file_{i}.mp4 to mp3"},
                {"role": "assistant", "content": f"ffmpeg -i file_{i}.mp4 file_{i}.mp3"},
            ]
            for i in range(20)
        ]
        train_path = write_chatml_parquet(
            tmp_path / "datasets" / "crash_test" / "data.parquet", convos,
        )
        return {"project": tmp_path, "train_path": train_path}

    def test_oom_crash_detected(self, crash_workspace: dict[str, Path]) -> None:
        """Subprocess should crash with OOM and main process should detect it."""
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        run = crash_workspace["project"] / "runs" / "oom_crash"

        # Absurd config: huge batch size + long sequences + no gradient checkpointing.
        # Should OOM on any reasonable GPU.
        config = {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset": str(crash_workspace["train_path"]),
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

        pid = manager.start(run, config, project_dir=crash_workspace["project"])
        print(f"\nCrash test started (PID {pid})")

        deadline = time.monotonic() + 180  # 3 min max
        while time.monotonic() < deadline:
            status = manager.get_status(run)
            if status.state in ("completed", "failed"):
                break
            if not manager.is_alive(run):
                time.sleep(2)  # let progress.jsonl flush
                status = manager.get_status(run)
                break
            time.sleep(3)

        print(f"Subprocess state: {status.state}")
        if status.error:
            print(f"Error: {status.error}")

        assert status.state == "failed", "Subprocess should have crashed"

        stderr_log = run / "stderr.log"
        assert stderr_log.exists()
        stderr_content = stderr_log.read_text()
        print(f"\n--- stderr.log (last 1000 chars) ---\n{stderr_content[-1000:]}")

        indicators = (
            "CUDA out of memory",
            "OutOfMemoryError",
            "RuntimeError",
            "Error",
            "Traceback",
        )
        assert any(i in stderr_content for i in indicators), (
            f"stderr should contain an error indicator, got:\n{stderr_content[-500:]}"
        )

        assert not (run / "model").exists(), "Model dir should not exist after a crash"
