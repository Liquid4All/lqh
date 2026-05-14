"""Experiment: Does 1k-sample fine-tuning meaningfully improve translation scores?

Usage::

    python -m tests.remote.experiment_larger_training --remote-host=toka
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq


# ---- Config ----
TRAIN_SAMPLES = 1000
EVAL_SAMPLES = 30
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
INFERENCE_MODEL = "lfm2.5-1.2b-instruct"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

SYSTEM_PROMPT = (
    "Translate the following text into German, French, Spanish, English, "
    "and Chinese. Format your response as a JSON object with the keys "
    '"de", "fr", "es", "en", and "zh". Output ONLY the JSON object, '
    "nothing else."
)

RESPONSE_FORMAT = {
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

SCORER_CONTENT = """\
# Scorer: Multi-Language Translation Quality

## Task Description
The model translates input text into 5 languages (de, fr, es, en, zh) as JSON.

## Conversation Format
```
[User]
<the input text to translate>

[Assistant]
<the model's translation response>
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

PIPELINE_CODE = '''\
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step
import json
import random
import liquidrandom

class TranslationPipeline(Pipeline):
    SAMPLE_TYPES = [
        "casual message", "formal email", "technical sentence",
        "idiomatic expression", "short phrase", "multi-sentence paragraph",
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

    @step(retries=5)
    async def _generate_source(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Write a short {self.sample_type} (1-3 sentences) that "
                    f"a {self.persona.brief()} would write. "
                    f"Output ONLY the text, nothing else."
                ),
            }],
        )
        if not resp.choices:
            raise GenerationError("Empty choices in source response")
        content = resp.choices[0].message.content
        if not content:
            raise GenerationError("Empty source response")
        self.source_text = content.strip()
        if len(self.source_text) < 5:
            raise GenerationError("Source text too short")

    @step(retries=5)
    async def _generate_translations(self, client):
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[
                {"role": "system", "content": "Translate into all 5 languages. Return ONLY JSON with keys: de, fr, es, en, zh."},
                {"role": "user", "content": self.source_text},
            ],
            response_format={"type": "json_object"},
        )
        if not resp.choices:
            raise GenerationError("Empty choices in translation response")
        content = resp.choices[0].message.content
        if not content:
            raise GenerationError("Empty translation response")
        raw = content.strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise GenerationError(f"Expected dict, got {type(data).__name__}")
        required = {"de", "fr", "es", "en", "zh"}
        if not required.issubset(data.keys()):
            raise GenerationError(f"Missing keys: {required - set(data.keys())}")
        self.translations_json = json.dumps(data, ensure_ascii=False)
'''


async def main(remote_host: str, train_samples: int = TRAIN_SAMPLES, eval_samples: int = EVAL_SAMPLES) -> None:
    import tempfile

    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline
    from lqh.remote.backend import RemoteConfig
    from lqh.remote.bootstrap import bootstrap_remote
    from lqh.remote.ssh_direct import SSHDirectBackend
    from lqh.remote.ssh_helpers import ssh_run
    from lqh.scoring import run_scoring
    from lqh.train.progress import read_latest_progress, read_progress

    remote_root = f"/tmp/lqh-experiment-{uuid4().hex[:8]}"
    tmpdir = Path(tempfile.mkdtemp(prefix="lqh_exp_"))
    project_dir = tmpdir / "project"
    project_dir.mkdir()
    (project_dir / ".lqh").mkdir()
    (project_dir / "runs").mkdir()

    print(f"Project dir: {project_dir}")
    print(f"Remote root: {remote_root}")
    print(f"Remote host: {remote_host}")
    print(f"Config: {train_samples} train, {eval_samples} eval, {NUM_EPOCHS} epochs, lr={LEARNING_RATE}")
    print()

    # Setup project files
    dg = project_dir / "data_gen"
    dg.mkdir()
    (dg / "translation_v1.py").write_text(PIPELINE_CODE)
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True)
    (scorers / "translation_v1.md").write_text(SCORER_CONTENT)

    # Setup remote
    print("=" * 60)
    print("[1/6] Bootstrapping remote...")
    print("=" * 60)
    config = RemoteConfig(
        name="experiment",
        type="ssh_direct",
        hostname=remote_host,
        remote_root=remote_root,
    )
    backend = SSHDirectBackend(config, project_dir)
    log = await backend.setup()
    print(log)

    try:
        # Generate data
        print()
        print("=" * 60)
        print(f"[2/6] Generating {train_samples} train + {eval_samples} eval samples...")
        print("=" * 60)
        api_config = load_config()
        token = require_token()
        client = create_client(token, api_config.api_base_url)
        pipeline_path = dg / "translation_v1.py"

        train_dir = project_dir / "datasets" / "translation_train"
        t0 = time.monotonic()
        result = await run_pipeline(
            script_path=pipeline_path,
            num_samples=train_samples,
            output_dir=train_dir,
            client=client,
            concurrency=10,
            max_retries=5,
        )
        t_train = time.monotonic() - t0
        print(f"  Training data: {result.succeeded}/{result.total} in {t_train:.0f}s")

        eval_dir = project_dir / "datasets" / "translation_eval"
        result = await run_pipeline(
            script_path=pipeline_path,
            num_samples=eval_samples,
            output_dir=eval_dir,
            client=client,
            concurrency=10,
            max_retries=5,
        )
        print(f"  Eval data: {result.succeeded}/{result.total}")

        train_path = train_dir / "data.parquet"
        eval_path = eval_dir / "data.parquet"
        train_rows = pq.read_metadata(str(train_path)).num_rows
        eval_rows = pq.read_metadata(str(eval_path)).num_rows
        print(f"  Final: {train_rows} train rows, {eval_rows} eval rows")

        # Baseline eval via API
        print()
        print("=" * 60)
        print("[3/6] Baseline eval (API inference)...")
        print("=" * 60)
        scorer_path = scorers / "translation_v1.md"
        baseline_api_dir = project_dir / "evals" / "runs" / "baseline_api"
        result = await run_scoring(
            dataset_path=eval_path,
            scorer_path=scorer_path,
            output_dir=baseline_api_dir,
            client=client,
            run_inference=True,
            inference_model=INFERENCE_MODEL,
            inference_system_prompt=SYSTEM_PROMPT,
            inference_response_format=RESPONSE_FORMAT,
        )
        baseline_api = result.mean_score
        print(f"  Baseline (API): {baseline_api:.2f}/10")

        # Baseline eval via local HF on remote
        print()
        print("=" * 60)
        print("[4/6] Baseline eval (local HF on remote)...")
        print("=" * 60)
        baseline_local = await _remote_eval(
            project_dir, backend, remote_root, remote_host, client,
            MODEL_ID, "baseline_local", eval_path, scorer_path,
        )
        print(f"  Baseline (local HF): {baseline_local:.2f}/10")

        # Train
        print()
        print("=" * 60)
        print(f"[5/6] Training ({train_samples} samples, {NUM_EPOCHS} epochs, lr={LEARNING_RATE})...")
        print("=" * 60)
        run_name = "sft_1k"
        run_dir = project_dir / "runs" / run_name
        run_dir.mkdir(parents=True)

        train_config = {
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
                "num_epochs": NUM_EPOCHS,
                "per_device_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": LEARNING_RATE,
                "warmup_ratio": 0.1,
                "logging_steps": 10,
                "save_steps": 999,
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 512,
                "dataloader_num_workers": 2,
            },
        }

        t0 = time.monotonic()
        job_id = await backend.submit_run(str(run_dir), train_config)
        print(f"  Job PID: {job_id}")
        remote_run_dir = f"{remote_root}/runs/{run_name}"

        deadline = time.monotonic() + 3600  # 1 hour
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
                    lr = latest.get("lr")
                    epoch = latest.get("epoch")
                    parts = [f"Step {step}"]
                    if loss is not None:
                        parts.append(f"loss={loss:.4f}")
                    if lr is not None:
                        parts.append(f"lr={lr:.2e}")
                    if epoch is not None:
                        parts.append(f"epoch={epoch:.2f}")
                    print(f"  {' | '.join(parts)}")

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

        t_train = time.monotonic() - t0

        if final_state != "completed":
            stderr_out, _, _ = await ssh_run(
                remote_host, f"tail -30 {remote_run_dir}/stderr.log 2>/dev/null",
                timeout=10.0,
            )
            print(f"\n  --- stderr ---\n{stderr_out}\n  --- end ---")
            raise RuntimeError(f"Training failed: {final_state}")

        print(f"  Completed in {t_train:.0f}s ({last_step} steps)")

        # Post-training eval
        print()
        print("=" * 60)
        print("[6/6] Post-training eval...")
        print("=" * 60)
        trained_model = f"{remote_run_dir}/model"
        post_score = await _remote_eval(
            project_dir, backend, remote_root, remote_host, client,
            trained_model, "post_training", eval_path, scorer_path,
        )
        print(f"  Post-training: {post_score:.2f}/10")

        # Results
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        delta = post_score - baseline_local
        print(f"  Train samples:       {train_rows}")
        print(f"  Eval samples:        {eval_rows}")
        print(f"  Epochs:              {NUM_EPOCHS}")
        print(f"  Training steps:      {last_step}")
        print(f"  Training time:       {t_train:.0f}s")
        print(f"")
        print(f"  Baseline (API):      {baseline_api:.2f}/10")
        print(f"  Baseline (local HF): {baseline_local:.2f}/10")
        print(f"  Post-training:       {post_score:.2f}/10")
        print(f"  Delta:               {delta:+.2f}")
        print(f"  Improvement:         {'YES' if delta > 0.5 else 'marginal' if delta > 0 else 'NO'}")
        print("=" * 60)

        summary = {
            "train_samples": train_rows,
            "eval_samples": eval_rows,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "training_steps": last_step,
            "training_time_s": round(t_train, 1),
            "baseline_api": round(baseline_api, 2),
            "baseline_local": round(baseline_local, 2),
            "post_training": round(post_score, 2),
            "delta": round(delta, 2),
            "model": MODEL_ID,
            "remote_host": remote_host,
        }
        out_path = project_dir / "experiment_results.json"
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nResults saved to {out_path}")

    finally:
        print(f"\nCleaning up {remote_root}...")
        await ssh_run(remote_host, f"rm -rf {remote_root}", timeout=60.0)


async def _remote_eval(
    project_dir: Path,
    backend,
    remote_root: str,
    remote_host: str,
    client,
    model_remote: str,
    eval_run_name: str,
    eval_path: Path,
    scorer_path: Path,
) -> float:
    """Run inference on remote, pull predictions, score locally."""
    from lqh.remote.ssh_helpers import ssh_run
    from lqh.scoring import run_scoring
    from lqh.train.progress import read_latest_progress

    eval_run_dir = project_dir / "runs" / eval_run_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    infer_config = {
        "type": "infer",
        "base_model": model_remote,
        "dataset": str(eval_path),
        "system_prompt": SYSTEM_PROMPT,
        "manifest": ["dataset"],
    }

    job_id = await backend.submit_run(str(eval_run_dir), infer_config, module="lqh.infer")
    print(f"  Inference PID: {job_id}")

    remote_eval_run = f"{remote_root}/runs/{eval_run_name}"
    # Use a progress-based deadline: reset whenever we see new progress.
    # Initial 600s covers model loading; after that each progress update
    # resets the clock so any dataset size works without guessing a timeout.
    last_step_seen = -1
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            await backend.sync_progress(remote_eval_run, str(eval_run_dir))
        except Exception:
            pass
        latest = read_latest_progress(eval_run_dir)
        if latest and latest.get("status") == "completed":
            break
        if latest and latest.get("status") == "failed":
            raise RuntimeError(f"Inference failed: {latest.get('error')}")
        if latest:
            step = latest.get("step", -1)
            if step > last_step_seen:
                last_step_seen = step
                deadline = time.monotonic() + 600  # reset: still making progress
        if not await backend.is_job_alive(job_id):
            await backend.sync_progress(remote_eval_run, str(eval_run_dir))
            latest = read_latest_progress(eval_run_dir)
            if latest and latest.get("status") == "completed":
                break
            stderr_out, _, _ = await ssh_run(
                remote_host, f"tail -20 {remote_eval_run}/stderr.log 2>/dev/null",
                timeout=10.0,
            )
            raise RuntimeError(f"Inference died.\nstderr: {stderr_out}")
        await asyncio.sleep(3)
    else:
        raise RuntimeError("Inference timed out (no progress for 600s)")

    predictions_local = eval_run_dir / "predictions.parquet"
    await backend.sync_file_from_remote(
        f"{remote_eval_run}/predictions.parquet", str(predictions_local),
    )
    print(f"  Predictions: {pq.read_metadata(str(predictions_local)).num_rows} samples")

    output_dir = project_dir / "evals" / "runs" / eval_run_name
    result = await run_scoring(
        dataset_path=predictions_local,
        scorer_path=scorer_path,
        output_dir=output_dir,
        client=client,
        run_inference=False,
    )
    return result.mean_score


if __name__ == "__main__":
    host = None
    n_train: int | None = None
    n_eval: int | None = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--remote-host="):
            host = arg.split("=", 1)[1]
        elif arg == "--remote-host" and i < len(sys.argv) - 1:
            host = sys.argv[i + 1]
        elif arg.startswith("--train-samples="):
            n_train = int(arg.split("=", 1)[1])
        elif arg == "--train-samples" and i < len(sys.argv) - 1:
            n_train = int(sys.argv[i + 1])
        elif arg.startswith("--eval-samples="):
            n_eval = int(arg.split("=", 1)[1])
        elif arg == "--eval-samples" and i < len(sys.argv) - 1:
            n_eval = int(sys.argv[i + 1])
    if not host:
        import os
        host = os.environ.get("LQH_TEST_REMOTE_HOST")
    if not host:
        print("Usage: python -m tests.remote.experiment_larger_training --remote-host=toka [--train-samples=1000] [--eval-samples=200]")
        sys.exit(1)
    asyncio.run(main(host, train_samples=n_train or TRAIN_SAMPLES, eval_samples=n_eval or EVAL_SAMPLES))
