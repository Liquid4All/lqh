"""End-to-end remote training test: SFT on toka via SSHDirectBackend.

Downloads a small dataset, syncs to remote, submits SFT training,
monitors progress, and verifies completion.

Usage::

    pytest tests/remote/test_remote_training_e2e.py --remote-host=toka -v -s
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import pytest_asyncio

from lqh.remote.backend import RemoteConfig
from lqh.remote.ssh_direct import SSHDirectBackend
from lqh.remote.ssh_helpers import ssh_run
from lqh.train.progress import read_latest_progress, read_progress

DATASET_URL = (
    "https://huggingface.co/datasets/mlech26l/shell-helper/resolve/main/"
    "data/shell_helper_dataset_many_opt.parquet?download=true"
)

TRAIN_SAMPLES = 50  # small for speed
EVAL_SAMPLES = 5
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"


def _download_and_prepare(project_dir: Path) -> tuple[Path, Path]:
    """Download shell-helper dataset, convert to ChatML, split train/eval."""
    import urllib.request

    raw_path = project_dir / "shell_helper_raw.parquet"
    if not raw_path.exists():
        print(f"  Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, raw_path)
        print(f"  Downloaded {raw_path.stat().st_size / 1024:.0f} KB")

    raw_table = pq.read_table(str(raw_path))
    total = len(raw_table)
    print(f"  Raw dataset: {total} samples")

    # Convert to ChatML format
    def convert_slice(table: pa.Table, out_path: Path) -> int:
        messages_col = table.column("messages")
        chatml_strings = []
        for i in range(len(table)):
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out_table, out_path)
        return len(table)

    eval_dir = project_dir / "datasets" / "shell_helper_eval"
    train_dir = project_dir / "datasets" / "shell_helper_train"

    n_eval = convert_slice(
        raw_table.slice(0, EVAL_SAMPLES),
        eval_dir / "data.parquet",
    )
    n_train = convert_slice(
        raw_table.slice(EVAL_SAMPLES, TRAIN_SAMPLES),
        train_dir / "data.parquet",
    )
    print(f"  Train: {n_train}, Eval: {n_eval}")

    return train_dir / "data.parquet", eval_dir / "data.parquet"


@pytest_asyncio.fixture
async def remote_training_env(
    remote_host: str, tmp_path: Path,
) -> tuple[SSHDirectBackend, Path, Path, Path, str]:
    """Set up a full remote training environment.

    Returns (backend, project_dir, train_parquet, eval_parquet, remote_root).
    """
    from uuid import uuid4

    remote_root = f"/tmp/lqh-e2e-train-{uuid4().hex[:8]}"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".lqh").mkdir()
    (project_dir / "runs").mkdir()

    config = RemoteConfig(
        name="e2e-train",
        type="ssh_direct",
        hostname=remote_host,
        remote_root=remote_root,
    )
    backend = SSHDirectBackend(config, project_dir)

    # Bootstrap remote
    print(f"\n[setup] Bootstrapping remote at {remote_root}...")
    log = await backend.setup()
    print(log)

    # Prepare dataset locally
    print("[setup] Preparing dataset...")
    train_path, eval_path = _download_and_prepare(project_dir)

    yield backend, project_dir, train_path, eval_path, remote_root

    # Cleanup
    print(f"\n[teardown] Cleaning up {remote_root}...")
    await ssh_run(remote_host, f"rm -rf {remote_root}", timeout=60.0)


class TestRemoteSFTTraining:
    """Full SFT training cycle on a remote GPU host."""

    @pytest.mark.asyncio
    async def test_remote_sft(
        self, remote_training_env: tuple, remote_host: str,
    ):
        backend, project_dir, train_path, eval_path, remote_root = remote_training_env

        run_name = "e2e_remote_sft"
        run_dir = project_dir / "runs" / run_name
        run_dir.mkdir(parents=True)

        config = {
            "type": "sft",
            "base_model": MODEL_ID,
            "dataset": str(train_path),
            "eval_dataset": str(eval_path),
            "eval_on_checkpoints": False,  # skip eval for speed
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
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.1,
                "logging_steps": 2,
                "save_steps": 999,  # no checkpoints, just final
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 512,
                "dataloader_num_workers": 0,
            },
        }

        # Submit
        print(f"\n[test] Submitting SFT job to {remote_host}...")
        job_id = await backend.submit_run(str(run_dir), config)
        print(f"[test] Job submitted, PID: {job_id}")

        remote_run_dir = f"{remote_root}/runs/{run_name}"

        # Poll for completion
        deadline = time.monotonic() + 600  # 10 minute timeout
        last_step = -1
        final_state = "unknown"

        while time.monotonic() < deadline:
            # Sync progress from remote
            try:
                await backend.sync_progress(remote_run_dir, str(run_dir))
            except Exception as e:
                print(f"  [sync error: {e}]")

            # Read local mirror
            latest = read_latest_progress(run_dir)
            if latest:
                step = latest.get("step")
                if step is not None and step != last_step:
                    last_step = step
                    loss = latest.get("loss")
                    loss_str = f"loss={loss:.4f}" if loss else ""
                    print(f"  Step {step} {loss_str}")

                status = latest.get("status")
                if status in ("completed", "failed"):
                    final_state = status
                    if status == "failed":
                        print(f"  FAILED: {latest.get('error')}")
                    break
            else:
                # Also check if process is still alive
                alive = await backend.is_job_alive(job_id)
                if not alive and last_step >= 0:
                    # Process died, check one more time for final status
                    await backend.sync_progress(remote_run_dir, str(run_dir))
                    latest = read_latest_progress(run_dir)
                    if latest and latest.get("status"):
                        final_state = latest["status"]
                    else:
                        final_state = "failed"
                    break

            await asyncio.sleep(5)

        # On failure, dump remote stderr
        if final_state != "completed":
            print("\n--- Remote stderr.log ---")
            stdout, _, _ = await ssh_run(
                remote_host,
                f"tail -50 {remote_run_dir}/stderr.log 2>/dev/null",
                timeout=10.0,
            )
            print(stdout)
            print("--- End stderr ---\n")

        assert final_state == "completed", (
            f"Training did not complete (state={final_state}). "
            f"Last step: {last_step}"
        )

        print(f"[test] Training completed at step {last_step}")

        # Verify progress entries
        entries = read_progress(run_dir, last_n=100)
        training_entries = [
            e for e in entries if "step" in e and "status" not in e
        ]
        assert len(training_entries) > 0, "Should have logged training steps"
        print(f"[test] {len(training_entries)} training steps logged")

        # Verify model was saved on remote
        stdout, _, rc = await ssh_run(
            remote_host,
            f"test -f {remote_run_dir}/model/config.json && echo yes",
            timeout=10.0,
        )
        assert "yes" in stdout, "Final model config.json should exist on remote"
        print("[test] Model saved on remote ✓")

        # Verify loss decreased
        if len(training_entries) >= 3:
            first_loss = training_entries[0].get("loss")
            last_loss = training_entries[-1].get("loss")
            if first_loss and last_loss:
                print(f"[test] Loss: {first_loss:.4f} → {last_loss:.4f}")
                # Not asserting decrease since 1 epoch on 50 samples is noisy,
                # but log it for visibility
