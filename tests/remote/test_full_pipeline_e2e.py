"""Full pipeline E2E: data gen → baseline eval → remote train → post-train eval.

This test exercises the complete lqh workflow:
1. Generate training + eval data using the pipeline engine
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

    pytest tests/remote/test_full_pipeline_e2e.py --remote-host=toka -v -s
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

# -- Reuse the translation scenario assets from tests/e2e/scenarios.py --

_SPEC = """\
# Specification: Multi-Language Translation

## Overview
Translate input text into 5 languages: German, French, Spanish, English, and Chinese.
Output as a JSON object with keys: de, fr, es, en, zh.

## Input Format
- **Type**: Plain text, 1-5 sentences
- **Language**: Any language (auto-detected)

## Output Format
- **Type**: JSON object
- **Keys**: de, fr, es, en, zh

## Requirements
1. All 5 target languages must be present
2. Translations must be accurate and natural
3. Preserve proper nouns, numbers
"""

_PIPELINE = '''\
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step
import json
import random
import liquidrandom

class TranslationPipeline(Pipeline):
    """Generate translation training samples."""

    SAMPLE_TYPES = [
        "casual message", "formal email", "technical sentence",
        "idiomatic expression", "short phrase",
    ]

    async def generate(self, client, input=None) -> Conversation:
        self.persona = liquidrandom.persona()
        self.sample_type = random.choice(self.SAMPLE_TYPES)
        self.seed = f"{self.persona.name}-{self.sample_type}"

        await self._generate_source(client)
        await self._generate_translations(client)

        return [
            ChatMLMessage("user", self.source_text),
            ChatMLMessage("assistant", self.translations_json),
        ]

    @step(retries=3)
    async def _generate_source(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Write a short {self.sample_type} (1-2 sentences) that "
                    f"a {self.persona.brief()} would write. "
                    f"Output ONLY the text, nothing else."
                ),
            }],
        )
        self.source_text = resp.choices[0].message.content.strip()
        if len(self.source_text) < 5:
            raise GenerationError("Source text too short")

    @step(retries=3)
    async def _generate_translations(self, client):
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[
                {"role": "system", "content": "Translate into all 5 languages. Return ONLY JSON with keys: de, fr, es, en, zh."},
                {"role": "user", "content": self.source_text},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        if not raw:
            raise GenerationError("Empty translation response")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise GenerationError(f"Expected dict, got {type(data).__name__}")
        required = {"de", "fr", "es", "en", "zh"}
        if not required.issubset(data.keys()):
            raise GenerationError(f"Missing keys: {required - set(data.keys())}")
        self.translations_json = json.dumps(data, ensure_ascii=False)
'''

_SCORER = """\
# Scorer: Multi-Language Translation Quality

## Task Description
The model translates input text into 5 languages (de, fr, es, en, zh) as JSON.

## Conversation Format
```
[User]
<the input text to translate>

[Assistant]
<the model's translation response — this is what you score>
```

## Scoring Scale
- **9-10**: All 5 translations present, accurate, natural, valid JSON
- **7-8**: All translations present with minor quality issues
- **5-6**: Valid JSON but some translations significantly inaccurate
- **3-4**: Missing language keys, or multiple translations are wrong
- **1-2**: Not valid JSON, or most translations are missing/wrong

## Critical Failures (automatic score <= 3)
- Output is not valid JSON
- More than 1 language key missing
"""

_SYSTEM_PROMPT = (
    "Translate the following text into German, French, Spanish, English, "
    "and Chinese. Format your response as a JSON object with the keys "
    '"de", "fr", "es", "en", and "zh". Output ONLY the JSON object, '
    "nothing else."
)

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "de": {"type": "string"},
                "fr": {"type": "string"},
                "es": {"type": "string"},
                "en": {"type": "string"},
                "zh": {"type": "string"},
            },
            "required": ["de", "fr", "es", "en", "zh"],
            "additionalProperties": False,
        },
    },
}

# Sizes kept small for speed
TRAIN_SAMPLES = 80
EVAL_SAMPLES = 15
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
INFERENCE_MODEL = "lfm2.5-1.2b-instruct"


async def _setup_project(project_dir: Path) -> None:
    """Create project structure with spec, pipeline, scorer, prompt."""
    (project_dir / "SPEC.md").write_text(_SPEC)
    (project_dir / ".lqh").mkdir(exist_ok=True)

    dg = project_dir / "data_gen"
    dg.mkdir(exist_ok=True)
    (dg / "translation_v1.py").write_text(_PIPELINE)

    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "translation_v1.md").write_text(_SCORER)

    prompts = project_dir / "prompts"
    prompts.mkdir(exist_ok=True)
    (prompts / "translation_v0.md").write_text(_SYSTEM_PROMPT)
    (prompts / "translation.schema.json").write_text(
        json.dumps(_RESPONSE_FORMAT, indent=2)
    )


async def _generate_data(
    project_dir: Path,
    num_train: int,
    num_eval: int,
) -> tuple[Path, Path]:
    """Generate training and eval datasets using the pipeline engine."""
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    pipeline_path = project_dir / "data_gen" / "translation_v1.py"

    # Generate training data
    train_dir = project_dir / "datasets" / "translation_train"
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

    # Generate eval data
    eval_dir = project_dir / "datasets" / "translation_eval"
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


async def _run_baseline_eval(
    project_dir: Path,
) -> float:
    """Score the base model via API inference. Returns mean score."""
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_scoring

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    eval_path = project_dir / "datasets" / "translation_eval" / "data.parquet"
    scorer_path = project_dir / "evals" / "scorers" / "translation_v1.md"
    output_dir = project_dir / "evals" / "runs" / "baseline"

    result = await run_scoring(
        dataset_path=eval_path,
        scorer_path=scorer_path,
        output_dir=output_dir,
        client=client,
        run_inference=True,
        inference_model=INFERENCE_MODEL,
        inference_system_prompt=_SYSTEM_PROMPT,
        inference_response_format=_RESPONSE_FORMAT,
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
    """Run inference with a model on remote, pull predictions, score locally.

    Returns mean score.
    """
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_scoring

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    # Submit inference job on remote
    eval_run_dir = project_dir / "runs" / eval_run_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    eval_parquet = project_dir / "datasets" / "translation_eval" / "data.parquet"

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

    # Wait for inference to complete
    deadline = time.monotonic() + 300  # 5 min timeout
    while time.monotonic() < deadline:
        try:
            await backend.sync_progress(remote_eval_run, str(eval_run_dir))
        except Exception:
            pass

        latest = read_latest_progress(eval_run_dir)
        if latest and latest.get("status") == "completed":
            break
        if latest and latest.get("status") == "failed":
            err = latest.get("error", "unknown")
            raise RuntimeError(f"Inference failed: {err}")

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

    # Pull predictions.parquet from remote
    predictions_remote = f"{remote_eval_run}/predictions.parquet"
    predictions_local = eval_run_dir / "predictions.parquet"
    await backend.sync_file_from_remote(predictions_remote, str(predictions_local))

    if not predictions_local.exists():
        raise RuntimeError("predictions.parquet not found after sync")

    print(f"  Predictions: {pq.read_metadata(str(predictions_local)).num_rows} samples")

    # Score locally via API judge
    scorer_path = project_dir / "evals" / "scorers" / "translation_v1.md"
    output_dir = project_dir / "evals" / "runs" / "post_training"

    result = await run_scoring(
        dataset_path=predictions_local,
        scorer_path=scorer_path,
        output_dir=output_dir,
        client=client,
        run_inference=False,  # already have predictions
    )
    return result.mean_score


@pytest_asyncio.fixture
async def pipeline_env(
    remote_host: str, tmp_path: Path,
):
    """Full pipeline environment: project + remote backend."""
    from uuid import uuid4

    remote_root = f"/tmp/lqh-pipeline-e2e-{uuid4().hex[:8]}"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Setup project
    await _setup_project(project_dir)

    # Setup remote
    remote_config = RemoteConfig(
        name="pipeline-test",
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

    # Cleanup
    print(f"\n[teardown] Cleaning up {remote_root}...")
    await ssh_run(remote_host, f"rm -rf {remote_root}", timeout=60.0)


class TestFullPipelineE2E:
    """Full pipeline: data gen → baseline eval → remote train → post-train eval."""

    @pytest.mark.asyncio
    async def test_datagen_train_eval(
        self,
        pipeline_env: tuple,
        remote_host: str,
    ):
        project_dir, backend, remote_root = pipeline_env

        # ---- Step 1: Generate data ----
        print("\n[1/7] Generating training + eval data via pipeline...")
        train_path, eval_path = await _generate_data(
            project_dir, TRAIN_SAMPLES, EVAL_SAMPLES,
        )
        assert train_path.exists()
        assert eval_path.exists()
        train_rows = pq.read_metadata(str(train_path)).num_rows
        eval_rows = pq.read_metadata(str(eval_path)).num_rows
        print(f"  Train: {train_rows} rows, Eval: {eval_rows} rows")

        # ---- Step 2: Baseline eval via API ----
        print("\n[2/7] Baseline eval (API inference + scoring)...")
        baseline_api_score = await _run_baseline_eval(project_dir)
        print(f"  Baseline (API): {baseline_api_score:.2f}/10")

        # ---- Step 3: Baseline eval via local HF inference on remote ----
        # This gives us an apples-to-apples comparison with post-training eval
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
            run_name="sft_normal",
            learning_rate=2e-5,
        )
        print(f"  Training completed at step {last_step}")

        # ---- Step 5: Post-training eval (normal LR) ----
        print("\n[5/7] Post-training eval (normal lr=2e-5)...")
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
        print("\n[7/7] Post-training eval (tiny lr=1e-8, should match baseline)...")
        post_tiny_score = await _run_remote_eval(
            project_dir, backend, remote_root, remote_host,
            trained_model_tiny, "post_training_tiny_lr",
        )
        print(f"  Post-training (tiny lr): {post_tiny_score:.2f}/10")

        # ---- Summary ----
        print(f"\n{'='*60}")
        print("Score comparison:")
        print(f"  Baseline (API):              {baseline_api_score:.2f}/10")
        print(f"  Baseline (local HF):         {baseline_local_score:.2f}/10")
        print(f"  Post-training (lr=2e-5):     {post_normal_score:.2f}/10")
        print(f"  Post-training (lr=1e-8):     {post_tiny_score:.2f}/10")
        print("")
        delta_normal = post_normal_score - baseline_local_score
        delta_tiny = post_tiny_score - baseline_local_score
        print(f"  Delta (normal lr):           {delta_normal:+.2f}")
        print(f"  Delta (tiny lr):             {delta_tiny:+.2f}")
        print(f"{'='*60}")

        # Assertions
        assert 1.0 <= baseline_api_score <= 10.0
        assert 1.0 <= baseline_local_score <= 10.0
        assert 1.0 <= post_normal_score <= 10.0
        assert 1.0 <= post_tiny_score <= 10.0

        # Tiny LR should produce nearly identical score to baseline
        assert abs(post_tiny_score - baseline_local_score) <= 2.0, (
            f"Tiny LR training should not significantly change scores: "
            f"baseline={baseline_local_score:.2f}, tiny_lr={post_tiny_score:.2f}"
        )

        # Write summary
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
        (project_dir / "pipeline_e2e_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print("\n  Summary saved to pipeline_e2e_summary.json")

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
        """Train on remote, return (last_step, remote_model_path)."""
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
                "num_epochs": 1,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": learning_rate,
                "warmup_ratio": 0.1,
                "logging_steps": 5,
                "save_steps": 999,
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 512,
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

        assert final_state == "completed", f"Training {run_name} failed (state={final_state})"

        trained_model_remote = f"{remote_run_dir}/model"
        stdout, _, rc = await ssh_run(
            remote_host,
            f"test -f {trained_model_remote}/config.json && echo yes",
            timeout=10.0,
        )
        assert "yes" in stdout, f"Model not found on remote for {run_name}"

        return last_step, trained_model_remote
