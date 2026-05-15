"""Experiment: SFT then DPO (validate -> scale -> polish pipeline).

Full workflow:
  1. Generate data
  2. Baseline eval
  3. SFT training
  4. Post-SFT eval
  5. DPO on top of SFT model
  6. Post-DPO eval

Usage::

    python -m tests.remote.experiment_sft_then_dpo --remote-host=toka
    python -m tests.remote.experiment_sft_then_dpo --remote-host=toka --train-samples=500 --eval-samples=200
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
TRAIN_SAMPLES = 200
EVAL_SAMPLES = 200
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"

# SFT config
SFT_EPOCHS = 3
SFT_LEARNING_RATE = 2e-5

# DPO config
DPO_ITERATIONS = 3
DPO_BETA = 0.1
DPO_LEARNING_RATE = 5e-6

SYSTEM_PROMPT = (
    "Translate the following text into German, French, Spanish, English, "
    "and Chinese. Format your response as a JSON object with the keys "
    '"de", "fr", "es", "en", and "zh". Output ONLY the JSON object, '
    "nothing else."
)

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
                    f"Write a short {self.sample_type} (1-2 sentences) that "
                    f"a {self.persona.brief()} would write. "
                    f"Output ONLY the text, nothing else."
                ),
            }],
        )
        if not resp.choices:
            raise GenerationError("Empty choices in source response")
        content = resp.choices[0].message.content
        if not content:
            raise GenerationError("Empty response")
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
            raise GenerationError("Empty response")
        raw = content.strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise GenerationError(f"Expected dict, got {type(data).__name__}")
        required = {"de", "fr", "es", "en", "zh"}
        if not required.issubset(data.keys()):
            raise GenerationError(f"Missing keys: {required - set(data.keys())}")
        self.translations_json = json.dumps(data, ensure_ascii=False)
'''

LORA_CONFIG = {
    "enabled": True,
    "r": 16,
    "alpha": 32,
    "dropout": 0.02,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj", "w1", "w2", "w3",
    ],
}


async def main(
    remote_host: str,
    train_samples: int = TRAIN_SAMPLES,
    eval_samples: int = EVAL_SAMPLES,
) -> None:
    import tempfile

    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline
    from lqh.golden import generate_golden
    from lqh.remote.backend import RemoteConfig
    from lqh.remote.ssh_direct import SSHDirectBackend
    from lqh.remote.ssh_helpers import ssh_run
    from lqh.scoring import run_scoring
    from lqh.train.progress import read_latest_progress

    remote_root = f"/tmp/lqh-sft-dpo-exp-{uuid4().hex[:8]}"
    tmpdir = Path(tempfile.mkdtemp(prefix="lqh_sft_dpo_exp_"))
    project_dir = tmpdir / "project"
    project_dir.mkdir()
    (project_dir / ".lqh").mkdir()
    (project_dir / "runs").mkdir()

    print(f"Project dir: {project_dir}")
    print(f"Remote root: {remote_root}")
    print(f"Config: {train_samples} train, {eval_samples} eval")
    print(f"  SFT: {SFT_EPOCHS} epochs, lr={SFT_LEARNING_RATE}")
    print(f"  DPO: {DPO_ITERATIONS} iterations, beta={DPO_BETA}, lr={DPO_LEARNING_RATE}")
    print()

    # Setup project files
    dg = project_dir / "data_gen"
    dg.mkdir()
    (dg / "translation_v1.py").write_text(PIPELINE_CODE)
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True)
    scorer_path = scorers / "translation_v1.md"
    scorer_path.write_text(SCORER_CONTENT)

    # API client
    api_config = load_config()
    token = require_token()
    client = create_client(token, api_config.api_base_url)

    # Bootstrap remote
    print("=" * 60)
    print("[1/8] Bootstrapping remote...")
    print("=" * 60)
    remote_config = RemoteConfig(
        name="sft-dpo-exp", type="ssh_direct", hostname=remote_host, remote_root=remote_root,
    )
    backend = SSHDirectBackend(remote_config, project_dir)
    log = await backend.setup()
    print(log)

    try:
        # ---- Generate data ----
        print()
        print("=" * 60)
        print(f"[2/8] Generating {train_samples} train + {eval_samples} eval samples...")
        print("=" * 60)
        pipeline_path = dg / "translation_v1.py"

        train_dir = project_dir / "datasets" / "translation_train"
        result = await run_pipeline(
            script_path=pipeline_path, num_samples=train_samples,
            output_dir=train_dir, client=client, concurrency=10, max_retries=5,
        )
        print(f"  Training data: {result.succeeded}/{result.total}")

        eval_dir = project_dir / "datasets" / "translation_eval"
        result = await run_pipeline(
            script_path=pipeline_path, num_samples=eval_samples,
            output_dir=eval_dir, client=client, concurrency=10, max_retries=5,
        )
        print(f"  Eval data: {result.succeeded}/{result.total}")

        train_path = train_dir / "data.parquet"
        eval_path = eval_dir / "data.parquet"

        # ---- Baseline eval ----
        print()
        print("=" * 60)
        print("[3/8] Baseline eval (local HF on remote)...")
        print("=" * 60)
        baseline_score = await _remote_eval(
            project_dir, backend, remote_root, remote_host, client,
            MODEL_ID, "baseline", eval_path, scorer_path,
        )
        print(f"  Baseline: {baseline_score:.2f}/10")

        # ---- SFT training ----
        print()
        print("=" * 60)
        print(f"[4/8] SFT training ({train_samples} samples, {SFT_EPOCHS} epochs)...")
        print("=" * 60)

        sft_run_name = "sft_translation"
        sft_run_dir = project_dir / "runs" / sft_run_name
        sft_run_dir.mkdir(parents=True)

        sft_config = {
            "type": "sft",
            "base_model": MODEL_ID,
            "dataset": str(train_path),
            "eval_dataset": str(eval_path),
            "eval_on_checkpoints": False,
            "lora": LORA_CONFIG,
            "training": {
                "num_epochs": SFT_EPOCHS,
                "per_device_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": SFT_LEARNING_RATE,
                "warmup_ratio": 0.1,
                "logging_steps": 10,
                "save_steps": 999,
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 512,
                "dataloader_num_workers": 2,
            },
        }

        t0_sft = time.monotonic()
        job_id = await backend.submit_run(str(sft_run_dir), sft_config)
        print(f"  Job PID: {job_id}")
        remote_sft_dir = f"{remote_root}/runs/{sft_run_name}"

        last_step, final_state = await _wait_for_training(
            backend, remote_sft_dir, sft_run_dir, job_id, remote_host,
        )
        t_sft = time.monotonic() - t0_sft

        if final_state != "completed":
            stderr_out, _, _ = await ssh_run(
                remote_host, f"tail -30 {remote_sft_dir}/stderr.log 2>/dev/null",
                timeout=10.0,
            )
            print(f"\n  --- stderr ---\n{stderr_out}\n  --- end ---")
            raise RuntimeError(f"SFT training failed: {final_state}")

        print(f"  SFT completed in {t_sft:.0f}s ({last_step} steps)")

        # ---- Post-SFT eval ----
        print()
        print("=" * 60)
        print("[5/8] Post-SFT eval...")
        print("=" * 60)
        sft_model_remote = f"{remote_sft_dir}/model"
        post_sft_score = await _remote_eval(
            project_dir, backend, remote_root, remote_host, client,
            sft_model_remote, "post_sft", eval_path, scorer_path,
        )
        print(f"  Post-SFT: {post_sft_score:.2f}/10")

        # ---- DPO on top of SFT ----
        print()
        print("=" * 60)
        print(f"[6/8] DPO training ({DPO_ITERATIONS} iterations on SFT model)...")
        print("=" * 60)

        dpo_run_name = "dpo_translation"
        dpo_run_dir = project_dir / "runs" / dpo_run_name
        dpo_run_dir.mkdir(parents=True)

        dpo_config = {
            "type": "on_policy_dpo",
            "base_model": sft_model_remote,  # start from SFT model
            "dataset": str(train_path),
            # DPO generates on the training dataset (preference_dataset defaults
            # to dataset when not set).  No eval_dataset needed here.
            "system_prompt": SYSTEM_PROMPT,
            "num_iterations": DPO_ITERATIONS,
            "dpo_beta": DPO_BETA,
            "golden_source": "dataset",
            "rejection_threshold": 6.0,
            "scorer": str(scorer_path.relative_to(project_dir)),
            "lora": LORA_CONFIG,
            "training": {
                "per_device_batch_size": 2,
                "learning_rate": DPO_LEARNING_RATE,
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 512,
                "dataloader_num_workers": 0,
            },
        }

        t0_dpo = time.monotonic()
        job_id = await backend.submit_run(str(dpo_run_dir), dpo_config)
        print(f"  Job PID: {job_id}")
        remote_dpo_dir = f"{remote_root}/runs/{dpo_run_name}"

        # DPO ping-pong
        iteration_scores: list[float] = []
        for iteration in range(DPO_ITERATIONS):
            iter_name = f"iter_{iteration:03d}"
            local_iter_dir = dpo_run_dir / "iterations" / iter_name
            local_iter_dir.mkdir(parents=True, exist_ok=True)
            remote_iter_dir = f"{remote_dpo_dir}/iterations/{iter_name}"

            print(f"\n  --- Iteration {iteration + 1}/{DPO_ITERATIONS} ---")

            # Wait for predictions
            print("  Waiting for predictions...")
            last_step_seen = -1
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                try:
                    await backend.sync_progress(remote_dpo_dir, str(dpo_run_dir))
                except Exception:
                    pass

                request_file = local_iter_dir / "iter_request.json"
                predictions_file = local_iter_dir / "predictions.parquet"
                if request_file.exists() and predictions_file.exists():
                    break

                latest = read_latest_progress(dpo_run_dir)
                if latest:
                    step = latest.get("step", -1)
                    if step > last_step_seen:
                        last_step_seen = step
                        deadline = time.monotonic() + 600

                if not await backend.is_job_alive(job_id):
                    await backend.sync_progress(remote_dpo_dir, str(dpo_run_dir))
                    if request_file.exists() and predictions_file.exists():
                        break
                    latest = read_latest_progress(dpo_run_dir)
                    err = latest.get("error", "unknown") if latest else "process died"
                    stderr_out, _, _ = await ssh_run(
                        remote_host, f"tail -20 {remote_dpo_dir}/stderr.log 2>/dev/null",
                        timeout=10.0,
                    )
                    raise RuntimeError(
                        f"DPO process died during iteration {iteration}.\n"
                        f"Error: {err}\nstderr: {stderr_out}"
                    )

                await asyncio.sleep(3)
            else:
                raise RuntimeError(f"Timeout waiting for iteration {iteration} predictions (no progress for 600s)")

            n_preds = pq.read_metadata(str(predictions_file)).num_rows
            print(f"  Got {n_preds} predictions, scoring...")

            # Score predictions
            score_result = await run_scoring(
                dataset_path=predictions_file,
                scorer_path=scorer_path,
                output_dir=local_iter_dir,
                client=client,
                run_inference=False,
            )
            iter_score = score_result.mean_score
            iteration_scores.append(iter_score)
            print(f"  Iteration {iteration} score: {iter_score:.2f}/10")

            # Generate golden + assemble preferences
            await generate_golden(
                predictions_path=predictions_file,
                scores_path=local_iter_dir / "results.parquet",
                dataset_path=str(train_path),
                config=dpo_config,
                client=client,
                output_dir=local_iter_dir,
            )

            prefs_path = local_iter_dir / "preferences.parquet"
            if prefs_path.exists():
                n_prefs = pq.read_metadata(str(prefs_path)).num_rows
                print(f"  {n_prefs} preference pairs assembled")
            else:
                print("  WARNING: preferences.parquet not created!")
                break

            # Push preferences back to remote
            remote_prefs = f"{remote_iter_dir}/preferences.parquet"
            await backend.sync_file_to_remote(str(prefs_path), remote_prefs)
            print("  Preferences synced to remote, DPO step running...")

        # Wait for DPO to complete
        print("\n  Waiting for DPO to finish...")
        deadline = time.monotonic() + 600
        final_state = "unknown"
        while time.monotonic() < deadline:
            try:
                await backend.sync_progress(remote_dpo_dir, str(dpo_run_dir))
            except Exception:
                pass
            latest = read_latest_progress(dpo_run_dir)
            if latest:
                status = latest.get("status")
                if status in ("completed", "failed", "interrupted"):
                    final_state = status
                    break
            if not await backend.is_job_alive(job_id):
                await backend.sync_progress(remote_dpo_dir, str(dpo_run_dir))
                latest = read_latest_progress(dpo_run_dir)
                final_state = latest.get("status", "failed") if latest else "failed"
                break
            await asyncio.sleep(3)

        t_dpo = time.monotonic() - t0_dpo

        if final_state not in ("completed", "interrupted"):
            stderr_out, _, _ = await ssh_run(
                remote_host, f"tail -30 {remote_dpo_dir}/stderr.log 2>/dev/null",
                timeout=10.0,
            )
            print(f"\n  --- stderr ---\n{stderr_out}\n  --- end ---")
            raise RuntimeError(f"DPO training failed: {final_state}")

        print(f"  DPO completed in {t_dpo:.0f}s")

        # ---- Post-DPO eval ----
        print()
        print("=" * 60)
        print("[7/8] Post-DPO eval...")
        print("=" * 60)
        dpo_model_remote = f"{remote_dpo_dir}/model"
        post_dpo_score = await _remote_eval(
            project_dir, backend, remote_root, remote_host, client,
            dpo_model_remote, "post_dpo", eval_path, scorer_path,
        )
        print(f"  Post-DPO: {post_dpo_score:.2f}/10")

        # ---- Results ----
        print()
        print("=" * 60)
        print("[8/8] RESULTS")
        print("=" * 60)
        sft_delta = post_sft_score - baseline_score
        dpo_delta = post_dpo_score - post_sft_score
        total_delta = post_dpo_score - baseline_score
        print(f"  Train samples:       {train_samples}")
        print(f"  Eval samples:        {eval_samples}")
        print(f"  SFT epochs:          {SFT_EPOCHS}")
        print(f"  SFT time:            {t_sft:.0f}s")
        print(f"  DPO iterations:      {DPO_ITERATIONS}")
        print(f"  DPO time:            {t_dpo:.0f}s")
        print("")
        print(f"  Baseline (local HF): {baseline_score:.2f}/10")
        print(f"  Post-SFT:            {post_sft_score:.2f}/10  (delta: {sft_delta:+.2f})")
        for i, s in enumerate(iteration_scores):
            print(f"    DPO iter {i} score:  {s:.2f}/10")
        print(f"  Post-DPO:            {post_dpo_score:.2f}/10  (delta vs SFT: {dpo_delta:+.2f})")
        print("")
        print(f"  Total improvement:   {total_delta:+.2f}")
        print(f"  Pipeline:            {'YES' if total_delta > 0 else 'NO'}")
        print("=" * 60)

        summary = {
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "sft_epochs": SFT_EPOCHS,
            "sft_learning_rate": SFT_LEARNING_RATE,
            "sft_time_s": round(t_sft, 1),
            "dpo_iterations": DPO_ITERATIONS,
            "dpo_beta": DPO_BETA,
            "dpo_learning_rate": DPO_LEARNING_RATE,
            "dpo_time_s": round(t_dpo, 1),
            "baseline_score": round(baseline_score, 2),
            "post_sft_score": round(post_sft_score, 2),
            "sft_delta": round(sft_delta, 2),
            "iteration_scores": [round(s, 2) for s in iteration_scores],
            "post_dpo_score": round(post_dpo_score, 2),
            "dpo_delta": round(dpo_delta, 2),
            "total_delta": round(total_delta, 2),
            "model": MODEL_ID,
            "remote_host": remote_host,
        }
        out_path = project_dir / "sft_dpo_experiment_results.json"
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nResults saved to {out_path}")

    finally:
        print(f"\nCleaning up {remote_root}...")
        await ssh_run(remote_host, f"rm -rf {remote_root}", timeout=60.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_training(
    backend, remote_run_dir: str, local_run_dir: Path, job_id: str, remote_host: str,
) -> tuple[int, str]:
    """Poll for SFT training completion. Returns (last_step, final_state)."""
    from lqh.train.progress import read_latest_progress

    deadline = time.monotonic() + 3600
    last_step = -1
    final_state = "unknown"

    while time.monotonic() < deadline:
        try:
            await backend.sync_progress(remote_run_dir, str(local_run_dir))
        except Exception:
            pass

        latest = read_latest_progress(local_run_dir)
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
            await backend.sync_progress(remote_run_dir, str(local_run_dir))
            latest = read_latest_progress(local_run_dir)
            final_state = latest.get("status", "failed") if latest else "failed"
            break

        await asyncio.sleep(5)

    return last_step, final_state


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
                deadline = time.monotonic() + 600
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
    n = pq.read_metadata(str(predictions_local)).num_rows
    print(f"  Predictions: {n} samples")

    output_dir = project_dir / "evals" / "runs" / eval_run_name
    result = await run_scoring(
        dataset_path=predictions_local, scorer_path=scorer_path,
        output_dir=output_dir, client=client, run_inference=False,
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
        print("Usage: python -m tests.remote.experiment_sft_then_dpo --remote-host=toka [--train-samples=200] [--eval-samples=200]")
        sys.exit(1)
    asyncio.run(main(host, train_samples=n_train or TRAIN_SAMPLES, eval_samples=n_eval or EVAL_SAMPLES))
