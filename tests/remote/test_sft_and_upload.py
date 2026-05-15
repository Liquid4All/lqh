"""E2E test: SFT training (200 samples) + upload model to HuggingFace.

Trains a small SFT model on a remote GPU, then uploads the resulting
model directly from the remote to HuggingFace (no local download).

Usage::

    pytest tests/remote/test_sft_and_upload.py --remote-host=toka -v -s

Requires:
    - SSH access to a GPU host (--remote-host or LQH_TEST_REMOTE_HOST)
    - HF_TOKEN set locally (will be configured on remote if missing)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import pytest_asyncio

from lqh.remote.backend import RemoteConfig
from lqh.remote.bootstrap import check_hf_token, configure_hf_token
from lqh.remote.ssh_direct import SSHDirectBackend
from lqh.remote.ssh_helpers import ssh_run
from lqh.train.progress import read_latest_progress, read_progress

DATASET_URL = (
    "https://huggingface.co/datasets/mlech26l/shell-helper/resolve/main/"
    "data/shell_helper_dataset_many_opt.parquet?download=true"
)

TRAIN_SAMPLES = 200
EVAL_SAMPLES = 10
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
HF_REPO_ID = "mlech26l/test_lfm"


def _download_and_prepare(project_dir: Path) -> tuple[Path, Path]:
    """Download shell-helper dataset, convert to ChatML, split train/eval."""
    import urllib.request

    raw_path = project_dir / "shell_helper_raw.parquet"
    if not raw_path.exists():
        print("  Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, raw_path)
        print(f"  Downloaded {raw_path.stat().st_size / 1024:.0f} KB")

    raw_table = pq.read_table(str(raw_path))
    print(f"  Raw dataset: {len(raw_table)} samples")

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


async def _ensure_remote_hf_token(hostname: str, remote_root: str) -> None:
    """Ensure HF_TOKEN is available on the remote.

    Checks the remote first.  If not found, copies the local HF_TOKEN
    to the remote's .env file.  Fails if no local token either.
    """
    has_token = await check_hf_token(hostname, remote_root)
    if has_token:
        print("  HF_TOKEN already configured on remote ✓")
        return

    local_token = os.environ.get("HF_TOKEN")
    if not local_token:
        pytest.fail(
            "HF_TOKEN not set on remote and not available locally. "
            "Set the HF_TOKEN environment variable to proceed."
        )

    print("  HF_TOKEN not found on remote, configuring from local...")
    await configure_hf_token(hostname, remote_root, local_token)
    print("  HF_TOKEN configured on remote ✓")


@pytest_asyncio.fixture
async def remote_env(
    remote_host: str, tmp_path: Path,
) -> tuple[SSHDirectBackend, Path, Path, Path, str]:
    """Set up a remote training environment.

    Returns (backend, project_dir, train_parquet, eval_parquet, remote_root).
    """
    from uuid import uuid4

    remote_root = f"/tmp/lqh-e2e-sft-upload-{uuid4().hex[:8]}"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".lqh").mkdir()
    (project_dir / "runs").mkdir()

    config = RemoteConfig(
        name="e2e-sft-upload",
        type="ssh_direct",
        hostname=remote_host,
        remote_root=remote_root,
    )
    backend = SSHDirectBackend(config, project_dir)

    # Bootstrap
    print(f"\n[setup] Bootstrapping remote at {remote_root}...")
    log = await backend.setup()
    print(log)

    # Ensure HF_TOKEN is on the remote
    await _ensure_remote_hf_token(remote_host, remote_root)

    # Prepare dataset
    print("[setup] Preparing dataset...")
    train_path, eval_path = _download_and_prepare(project_dir)

    yield backend, project_dir, train_path, eval_path, remote_root

    # Cleanup
    print(f"\n[teardown] Cleaning up {remote_root}...")
    await ssh_run(remote_host, f"rm -rf {remote_root}", timeout=60.0)


async def _wait_for_training(
    backend: SSHDirectBackend,
    run_dir: Path,
    remote_run_dir: str,
    job_id: str,
    remote_host: str,
    timeout_s: int = 900,
) -> str:
    """Poll until training completes or fails. Returns final state."""
    deadline = time.monotonic() + timeout_s
    last_step = -1

    while time.monotonic() < deadline:
        try:
            await backend.sync_progress(remote_run_dir, str(run_dir))
        except Exception as e:
            print(f"  [sync error: {e}]")

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
                if status == "failed":
                    print(f"  FAILED: {latest.get('error')}")
                return status
        else:
            alive = await backend.is_job_alive(job_id)
            if not alive and last_step >= 0:
                await backend.sync_progress(remote_run_dir, str(run_dir))
                latest = read_latest_progress(run_dir)
                if latest and latest.get("status"):
                    return latest["status"]
                return "failed"

        await asyncio.sleep(5)

    return "timeout"


async def _upload_model_from_remote(
    hostname: str,
    remote_root: str,
    remote_model_dir: str,
    repo_id: str,
    private: bool = True,
) -> str:
    """Upload a model to HuggingFace directly from the remote machine.

    Uses the remote's venv (which has huggingface_hub installed) and
    HF_TOKEN (from .env).  Returns the HF repo URL.
    """
    activate = f"source {remote_root}/.lqh-env/bin/activate"
    env_source = (
        f"[ -f {remote_root}/.lqh-env/.env ] && "
        f"set -a && source {remote_root}/.lqh-env/.env && set +a"
    )

    # Python one-liner to upload via HfApi — more reliable than CLI
    # for handling private repos and commit messages
    upload_script = (
        "from huggingface_hub import HfApi; "
        "api = HfApi(); "
        f"api.create_repo('{repo_id}', repo_type='model', private={private}, exist_ok=True); "
        f"url = api.upload_folder(folder_path='{remote_model_dir}', repo_id='{repo_id}', "
        f"repo_type='model', commit_message='SFT model upload from lqh e2e test'); "
        "print(f'UPLOAD_URL={{url}}')"
    )

    cmd = f"{activate} && {env_source} && python -c \"{upload_script}\""

    print(f"  Uploading model to {repo_id} from remote...")
    stdout, stderr, rc = await ssh_run(hostname, cmd, timeout=600.0)

    if rc != 0:
        raise RuntimeError(
            f"Remote HF upload failed (exit {rc}):\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )

    # Extract URL from output
    for line in stdout.splitlines():
        if "UPLOAD_URL=" in line:
            url = line.split("UPLOAD_URL=", 1)[1].strip()
            print(f"  Upload complete: {url}")
            return url

    # If no URL found, construct it
    url = f"https://huggingface.co/{repo_id}"
    print(f"  Upload complete: {url}")
    return url


class TestSFTAndUpload:
    """SFT training (200 samples) + upload model to HuggingFace."""

    @pytest.mark.asyncio
    async def test_sft_train_and_upload(
        self, remote_env: tuple, remote_host: str,
    ):
        backend, project_dir, train_path, eval_path, remote_root = remote_env

        # ---------------------------------------------------------------
        # Phase 1: SFT training
        # ---------------------------------------------------------------
        run_name = "e2e_sft_upload"
        run_dir = project_dir / "runs" / run_name
        run_dir.mkdir(parents=True)

        config = {
            "type": "sft",
            "base_model": MODEL_ID,
            "dataset": str(train_path),
            "eval_dataset": str(eval_path),
            "eval_on_checkpoints": False,
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
                "gradient_accumulation_steps": 2,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.1,
                "logging_steps": 5,
                "save_steps": 999,  # no intermediate checkpoints
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 512,
                "dataloader_num_workers": 0,
            },
        }

        print(f"\n[train] Submitting SFT job ({TRAIN_SAMPLES} samples)...")
        job_id = await backend.submit_run(str(run_dir), config)
        print(f"[train] Job submitted, PID: {job_id}")

        remote_run_dir = f"{remote_root}/runs/{run_name}"

        final_state = await _wait_for_training(
            backend, run_dir, remote_run_dir, job_id, remote_host,
        )

        # On failure, dump stderr for debugging
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
            f"Training did not complete (state={final_state})"
        )
        print("[train] Training completed ✓")

        # Verify model exists on remote
        stdout, _, rc = await ssh_run(
            remote_host,
            f"test -f {remote_run_dir}/model/config.json && echo yes",
            timeout=10.0,
        )
        assert "yes" in stdout, "Model config.json should exist on remote"

        # Log training stats
        entries = read_progress(run_dir, last_n=200)
        training_entries = [e for e in entries if "step" in e and "status" not in e]
        if len(training_entries) >= 2:
            first_loss = training_entries[0].get("loss")
            last_loss = training_entries[-1].get("loss")
            if first_loss and last_loss:
                print(f"[train] Loss: {first_loss:.4f} → {last_loss:.4f} "
                      f"({len(training_entries)} steps)")

        # ---------------------------------------------------------------
        # Phase 2: Upload model to HuggingFace from remote
        # ---------------------------------------------------------------
        print(f"\n[upload] Uploading model to {HF_REPO_ID}...")
        remote_model_dir = f"{remote_run_dir}/model"

        url = await _upload_model_from_remote(
            hostname=remote_host,
            remote_root=remote_root,
            remote_model_dir=remote_model_dir,
            repo_id=HF_REPO_ID,
            private=True,
        )

        print(f"[upload] Model available at: {url}")

        # Verify the repo exists on HF
        verify_cmd = (
            f"source {remote_root}/.lqh-env/bin/activate && "
            f"[ -f {remote_root}/.lqh-env/.env ] && "
            f"set -a && source {remote_root}/.lqh-env/.env && set +a && "
            f"python -c \""
            f"from huggingface_hub import HfApi; "
            f"api = HfApi(); "
            f"info = api.repo_info('{HF_REPO_ID}', repo_type='model'); "
            f"print(f'REPO_OK id={{info.id}} private={{info.private}}')"
            f"\""
        )
        stdout, stderr, rc = await ssh_run(
            remote_host, verify_cmd, timeout=30.0,
        )
        assert rc == 0 and "REPO_OK" in stdout, (
            f"Failed to verify HF repo: stdout={stdout}, stderr={stderr}"
        )
        assert "private=True" in stdout, "Repo should be private"
        print("[upload] Verified on HuggingFace ✓")
