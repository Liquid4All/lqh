"""Full pipeline E2E for tool calling: data gen -> baseline eval -> remote train -> post-train eval.

Exercises the complete lqh workflow with tool-calling data:
1. Generate training + eval data with tool calls
2. Score baseline model via API (before training)
3. Train on remote GPU (SFT with LoRA)
4. Run inference with trained model on remote
5. Score post-training predictions locally
6. Compare before vs after scores

Requires:
  - --remote-host (SSH-accessible GPU host)
  - LQH API access (for data gen + scoring)
  - Internet (model download on remote)

Usage::

    pytest tests/remote/test_tool_calling_e2e.py --remote-host=toka -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import pytest_asyncio

from lqh.remote.backend import RemoteConfig
from lqh.remote.ssh_direct import SSHDirectBackend
from lqh.remote.ssh_helpers import ssh_run
from lqh.train.progress import read_latest_progress

logger = logging.getLogger(__name__)

# -- Tool calling pipeline + scorer --

_SPEC = """\
# Specification: Tool Calling Agent

## Overview
Build a tool-calling agent that correctly selects and invokes tools
based on user queries.

## Available Tools
- get_weather: Get current weather for a location
- search_products: Search product catalog
- check_order_status: Check order status by ID
- schedule_appointment: Schedule an appointment
- translate_text: Translate text to a target language

## Requirements
1. Correctly identify which tool to call based on user intent
2. Provide accurate arguments to the tool
3. Summarize tool results in a natural, friendly response
"""

_PIPELINE_PATH = Path(__file__).parent / "tool_calling_e2e_pipeline.py"

_SCORER = """\
# Scorer: Tool Calling Quality (Complex)

## Task Description
The assistant has 6 tools available (search_flights, book_hotel, convert_currency,
schedule_meeting, analyze_data, send_notification). Each tool has multiple required
parameters with specific formats.

## Scoring Criteria
Evaluate both tool selection AND argument accuracy:

### Tool Selection (40%)
- Did the assistant pick the correct tool for the user's request?
- With 6 tools and overlapping domains, this requires understanding intent.

### Argument Accuracy (40%)
- Are ALL required parameters present?
- Are dates in YYYY-MM-DD format (not natural language)?
- Are times in HH:MM 24h format?
- Are airport/currency codes correct (IATA codes like SFO not city names)?
- Are numeric values extracted correctly from the query?
- Are array parameters (participants, metrics) properly structured?

### Response Quality (20%)
- Does the final response correctly summarize the tool output?
- Is it natural and helpful?

## Scoring Scale
- **9-10**: Correct tool, all arguments accurate with proper formats, great response
- **7-8**: Correct tool, most arguments right but 1 minor format issue (e.g. date format)
- **5-6**: Correct tool, but multiple argument issues or missing optional params that were mentioned
- **3-4**: Wrong tool selected, OR correct tool but most arguments wrong
- **1-2**: No tool call, completely wrong tool, or garbled arguments

## Critical Failures (automatic score <= 3)
- Wrong tool selected
- Missing more than 1 required parameter
- Date still in natural language (not YYYY-MM-DD)
- Airport codes replaced with city names
"""

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When the user asks a question that requires a tool, call the appropriate tool "
    "with the correct arguments. Pay careful attention to parameter formats: "
    "dates as YYYY-MM-DD, times as HH:MM, airport IATA codes, currency codes, etc. "
    "After receiving the result, explain it to the user."
)

TRAIN_SAMPLES = 500
EVAL_SAMPLES = 30
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
INFERENCE_MODEL = "lfm2.5-1.2b-instruct"


async def _setup_project(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text(_SPEC)
    (project_dir / ".lqh").mkdir(exist_ok=True)

    dg = project_dir / "data_gen"
    dg.mkdir(exist_ok=True)
    import shutil
    shutil.copy(_PIPELINE_PATH, dg / "tool_calling_v1.py")

    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "tool_calling_v1.md").write_text(_SCORER)


async def _generate_data(
    project_dir: Path,
    num_train: int,
    num_eval: int,
) -> tuple[Path, Path]:
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    pipeline_path = project_dir / "data_gen" / "tool_calling_v1.py"

    train_dir = project_dir / "datasets" / "tool_calling_train"
    print(f"  Generating {num_train} training samples...")
    result = await run_pipeline(
        script_path=pipeline_path,
        num_samples=num_train,
        output_dir=train_dir,
        client=client,
        concurrency=5,
        max_retries=5,
    )
    print(f"  Training: {result.succeeded}/{result.total} succeeded")

    eval_dir = project_dir / "datasets" / "tool_calling_eval"
    print(f"  Generating {num_eval} eval samples...")
    result = await run_pipeline(
        script_path=pipeline_path,
        num_samples=num_eval,
        output_dir=eval_dir,
        client=client,
        concurrency=5,
        max_retries=5,
    )
    print(f"  Eval: {result.succeeded}/{result.total} succeeded")

    return train_dir / "data.parquet", eval_dir / "data.parquet"


async def _run_baseline_eval(project_dir: Path) -> float:
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_scoring

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    eval_path = project_dir / "datasets" / "tool_calling_eval" / "data.parquet"
    scorer_path = project_dir / "evals" / "scorers" / "tool_calling_v1.md"
    output_dir = project_dir / "evals" / "runs" / "baseline"

    result = await run_scoring(
        dataset_path=eval_path,
        scorer_path=scorer_path,
        output_dir=output_dir,
        client=client,
        run_inference=True,
        inference_model=INFERENCE_MODEL,
        inference_system_prompt=_SYSTEM_PROMPT,
    )
    return result.mean_score


async def _run_remote_eval(
    project_dir: Path,
    backend: SSHDirectBackend,
    remote_root: str,
    remote_host: str,
    model_remote: str,
    eval_run_name: str,
) -> float:
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_scoring

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    eval_run_dir = project_dir / "runs" / eval_run_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    eval_parquet = project_dir / "datasets" / "tool_calling_eval" / "data.parquet"

    infer_config = {
        "type": "infer",
        "base_model": model_remote,
        "dataset": str(eval_parquet),
        "system_prompt": _SYSTEM_PROMPT,
        "manifest": ["dataset"],
    }

    print(f"  Submitting inference job ({eval_run_name}) on remote...")
    job_id = await backend.submit_run(str(eval_run_dir), infer_config, module="lqh.infer")
    print(f"  Inference job PID: {job_id}")

    remote_eval_run = f"{remote_root}/runs/{eval_run_name}"

    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            await backend.sync_progress(remote_eval_run, str(eval_run_dir))
        except Exception:
            pass

        latest = read_latest_progress(eval_run_dir)
        if latest and latest.get("status") == "completed":
            break
        if latest and latest.get("status") == "failed":
            raise RuntimeError(f"Inference failed: {latest.get('error', 'unknown')}")

        alive = await backend.is_job_alive(job_id)
        if not alive:
            await backend.sync_progress(remote_eval_run, str(eval_run_dir))
            latest = read_latest_progress(eval_run_dir)
            if latest and latest.get("status") == "completed":
                break
            stderr_out, _, _ = await ssh_run(
                remote_host, f"tail -20 {remote_eval_run}/stderr.log 2>/dev/null",
                timeout=10.0,
            )
            raise RuntimeError(f"Inference process died.\nstderr: {stderr_out}")

        await asyncio.sleep(3)
    else:
        raise RuntimeError("Inference timed out")

    print("  Inference completed, pulling predictions...")

    predictions_remote = f"{remote_eval_run}/predictions.parquet"
    predictions_local = eval_run_dir / "predictions.parquet"
    await backend.sync_file_from_remote(predictions_remote, str(predictions_local))

    if not predictions_local.exists():
        raise RuntimeError("predictions.parquet not found after sync")

    print(f"  Predictions: {pq.read_metadata(str(predictions_local)).num_rows} samples")

    scorer_path = project_dir / "evals" / "scorers" / "tool_calling_v1.md"
    output_dir = project_dir / "evals" / "runs" / eval_run_name

    result = await run_scoring(
        dataset_path=predictions_local,
        scorer_path=scorer_path,
        output_dir=output_dir,
        client=client,
        run_inference=False,
    )
    return result.mean_score


@pytest_asyncio.fixture
async def pipeline_env(remote_host: str, tmp_path: Path):
    from uuid import uuid4

    remote_root = f"/tmp/lqh-toolcall-e2e-{uuid4().hex[:8]}"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    await _setup_project(project_dir)

    remote_config = RemoteConfig(
        name="toolcall-test",
        type="ssh_direct",
        hostname=remote_host,
        remote_root=remote_root,
    )
    backend = SSHDirectBackend(remote_config, project_dir)

    print(f"\n[setup] Bootstrapping remote at {remote_root}...")
    log = await backend.setup()
    for line in log.splitlines():
        if line.strip():
            print(f"  {line}")

    yield project_dir, backend, remote_root

    print(f"\n[teardown] Cleaning up {remote_root}...")
    await ssh_run(remote_host, f"rm -rf {remote_root}", timeout=60.0)


class TestToolCallingPipelineE2E:
    """Full pipeline: tool-calling data gen -> baseline eval -> remote train -> post-train eval."""

    @pytest.mark.asyncio
    async def test_tool_calling_datagen_train_eval(
        self,
        pipeline_env: tuple,
        remote_host: str,
    ):
        project_dir, backend, remote_root = pipeline_env

        # ---- Step 1: Generate data ----
        print("\n[1/7] Generating training + eval data...")
        train_path, eval_path = await _generate_data(
            project_dir, TRAIN_SAMPLES, EVAL_SAMPLES,
        )
        assert train_path.exists()
        assert eval_path.exists()
        train_rows = pq.read_metadata(str(train_path)).num_rows
        eval_rows = pq.read_metadata(str(eval_path)).num_rows
        print(f"  Train: {train_rows} rows, Eval: {eval_rows} rows")

        # Verify tools column
        train_table = pq.read_table(str(train_path))
        assert "tools" in train_table.column_names, "Training data missing tools column"

        # ---- Step 2: Baseline eval via API ----
        print("\n[2/7] Baseline eval (API inference + scoring)...")
        baseline_api_score = await _run_baseline_eval(project_dir)
        print(f"  Baseline (API): {baseline_api_score:.2f}/10")

        # ---- Step 3: Baseline eval via local HF inference on remote ----
        # This is the apples-to-apples comparison with post-training
        print(f"\n[3/7] Baseline eval (local HF inference on {remote_host})...")
        baseline_local_score = await _run_remote_eval(
            project_dir, backend, remote_root, remote_host,
            MODEL_ID,  # base model from HF Hub
            "baseline_local",
        )
        print(f"  Baseline (local HF): {baseline_local_score:.2f}/10")

        # ---- Step 4: Train with normal LR ----
        print(f"\n[4/7] Training on {remote_host} (lr=2e-5)...")
        last_step, trained_model = await self._train_on_remote(
            project_dir, backend, remote_root, remote_host,
            train_path, eval_path,
            run_name="sft_tool_calling",
            learning_rate=2e-5,
        )
        print(f"  Training completed at step {last_step}")

        # ---- Step 5: Post-training eval (normal LR) ----
        print(f"\n[5/7] Post-training eval (lr=2e-5)...")
        post_normal_score = await _run_remote_eval(
            project_dir, backend, remote_root, remote_host,
            trained_model, "post_training_normal",
        )
        print(f"  Post-training (normal): {post_normal_score:.2f}/10")

        # ---- Step 6: Train with tiny LR (sanity check) ----
        print(f"\n[6/7] Training on {remote_host} (lr=1e-8, sanity check)...")
        last_step_tiny, trained_model_tiny = await self._train_on_remote(
            project_dir, backend, remote_root, remote_host,
            train_path, eval_path,
            run_name="sft_tiny_lr",
            learning_rate=1e-8,
        )
        print(f"  Training completed at step {last_step_tiny}")

        # ---- Step 7: Post-training eval (tiny LR) ----
        print(f"\n[7/7] Post-training eval (lr=1e-8, should match baseline)...")
        post_tiny_score = await _run_remote_eval(
            project_dir, backend, remote_root, remote_host,
            trained_model_tiny, "post_training_tiny_lr",
        )
        print(f"  Post-training (tiny lr): {post_tiny_score:.2f}/10")

        # ---- Summary ----
        delta_normal = post_normal_score - baseline_local_score
        delta_tiny = post_tiny_score - baseline_local_score
        print(f"\n{'='*60}")
        print(f"Tool Calling E2E Results:")
        print(f"  Baseline (API):              {baseline_api_score:.2f}/10")
        print(f"  Baseline (local HF):         {baseline_local_score:.2f}/10")
        print(f"  Post-training (lr=2e-5):     {post_normal_score:.2f}/10")
        print(f"  Post-training (lr=1e-8):     {post_tiny_score:.2f}/10")
        print(f"")
        print(f"  Delta (normal lr):           {delta_normal:+.2f}")
        print(f"  Delta (tiny lr):             {delta_tiny:+.2f}")
        print(f"{'='*60}")

        assert 1.0 <= baseline_api_score <= 10.0
        assert 1.0 <= baseline_local_score <= 10.0
        assert 1.0 <= post_normal_score <= 10.0
        assert 1.0 <= post_tiny_score <= 10.0

        # Tiny LR should produce nearly identical score to baseline
        assert abs(post_tiny_score - baseline_local_score) <= 2.0, (
            f"Tiny LR training should not significantly change scores: "
            f"baseline={baseline_local_score:.2f}, tiny_lr={post_tiny_score:.2f}"
        )

        summary = {
            "train_samples": train_rows,
            "eval_samples": eval_rows,
            "baseline_api_score": round(baseline_api_score, 2),
            "baseline_local_score": round(baseline_local_score, 2),
            "post_training_normal_score": round(post_normal_score, 2),
            "post_training_tiny_lr_score": round(post_tiny_score, 2),
            "delta_normal": round(delta_normal, 2),
            "delta_tiny": round(delta_tiny, 2),
            "model": MODEL_ID,
            "remote_host": remote_host,
        }
        (project_dir / "tool_calling_e2e_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    async def _train_on_remote(
        self,
        project_dir: Path,
        backend: SSHDirectBackend,
        remote_root: str,
        remote_host: str,
        train_path: Path,
        eval_path: Path,
        *,
        run_name: str,
        learning_rate: float,
    ) -> tuple[int, str]:
        run_dir = project_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

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
                "num_epochs": 3,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": learning_rate,
                "warmup_ratio": 0.1,
                "logging_steps": 10,
                "save_steps": 999,
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 2048,
                "dataloader_num_workers": 0,
            },
        }

        job_id = await backend.submit_run(str(run_dir), config)
        print(f"  Training job PID: {job_id}")

        remote_run_dir = f"{remote_root}/runs/{run_name}"
        deadline = time.monotonic() + 600
        last_step = -1
        final_state = "unknown"

        while time.monotonic() < deadline:
            try:
                await backend.sync_progress(remote_run_dir, str(run_dir))
            except Exception:
                pass

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
                    break

            if not await backend.is_job_alive(job_id) and last_step >= 0:
                await backend.sync_progress(remote_run_dir, str(run_dir))
                latest = read_latest_progress(run_dir)
                final_state = latest.get("status", "failed") if latest else "failed"
                break

            await asyncio.sleep(5)

        if final_state != "completed":
            stderr_out, _, _ = await ssh_run(
                remote_host,
                f"tail -30 {remote_run_dir}/stderr.log 2>/dev/null",
                timeout=10.0,
            )
            print(f"\n  --- stderr ---\n{stderr_out}\n  --- end ---")

        assert final_state == "completed", f"Training failed (state={final_state})"

        trained_model_remote = f"{remote_run_dir}/model"
        stdout, _, rc = await ssh_run(
            remote_host,
            f"test -f {trained_model_remote}/config.json && echo yes",
            timeout=10.0,
        )
        assert "yes" in stdout, "Model not found on remote"

        return last_step, trained_model_remote
