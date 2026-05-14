"""Experiment: does SFT improve the model on this task, or hurt it?

Given a project that already has a spec, a data-gen pipeline, and
*pre-judged* train/eval datasets, this script validates the full
zero-shot → SFT → post-SFT loop and reports whether SFT actually
helped.

Steps::

    1. Zero-shot eval: run `lqh.infer` on the base model with the eval
       dataset, then score with the API judge.
    2. SFT:           run `lqh.train` on the train dataset.
    3. Post-SFT eval: run `lqh.infer` on the trained checkpoint with
       the same eval dataset, then score the same way.

Reports baseline mean/median, post-SFT mean/median, and the delta.
Exits non-zero if SFT regressed by more than ``--regression-threshold``
(default -0.5 mean-score points) — that is the guardrail.

The script reuses ``lqh.infer`` and ``lqh.train`` as subprocesses (same
path that ``start_local_eval`` and ``start_training`` go through), and
``lqh.scoring.run_scoring`` for the judge. So scores are directly
comparable to what the agent reports.

Usage::

    python -m tests.remote.experiment_sft_no_regression \\
        --project-dir example_project \\
        --train-dataset datasets/business_chat_translation_v1_filtered/data.parquet \\
        --eval-dataset datasets/business_chat_translation_v1_eval/data.parquet \\
        --scorer evals/scorers/business_chat_translation_v1.md \\
        --system-prompt prompts/business_chat_translation_v0.md \\
        --schema prompts/business_chat_translation.schema.json \\
        --exp-name sft_regression_v1

To save GPU time during iteration, ``--skip-baseline``, ``--skip-train``,
and ``--skip-sft-eval`` reuse already-completed steps from a previous run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"


@dataclass
class EvalReport:
    label: str
    run_dir: Path
    mean: float
    median: float
    scored: int
    failed: int


def _read_text(p: Path | None) -> str | None:
    return p.read_text(encoding="utf-8") if p is not None else None


def _load_schema_envelope(p: Path | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _build_infer_config(
    *,
    base_model: str,
    eval_dataset: Path,
    scorer: Path,
    system_prompt: str | None,
    schema_envelope: dict[str, Any] | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "type": "infer",
        "base_model": base_model,
        "dataset": str(eval_dataset.resolve()),
        "scorer": str(scorer.resolve()),
        "max_new_tokens": max_new_tokens,
        "manifest": ["base_model", "dataset", "scorer"],
    }
    if system_prompt is not None:
        cfg["system_prompt"] = system_prompt
    if schema_envelope is not None:
        cfg["response_format"] = schema_envelope
    return cfg


def _build_sft_config(
    *,
    base_model: str,
    train_dataset: Path,
    num_epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
) -> dict[str, Any]:
    return {
        "type": "sft",
        "base_model": base_model,
        "dataset": str(train_dataset.resolve()),
        "training": {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
        },
        "lora": {
            "enabled": True,
            "r": lora_r,
            "alpha": lora_alpha,
        },
    }


def _run_subprocess(module: str, config_path: Path) -> None:
    """Run ``python -m <module> <config>`` and stream its output."""
    print(f"\n=== {module} {config_path} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", module, str(config_path)],
        cwd=str(Path.cwd()),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{module} exited with status {proc.returncode}; see output above."
        )


async def _score_predictions(
    predictions_parquet: Path,
    scorer_path: Path,
    output_dir: Path,
    *,
    judge_size: str,
    concurrency: int,
) -> tuple[float, float, int, int]:
    """Run the API judge over ``predictions_parquet``. Returns mean/median/scored/failed."""
    from lqh.auth import get_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_scoring

    api_key = os.environ.get("LQH_API_KEY") or get_token() or ""
    if not api_key:
        raise RuntimeError("No lqh API key (set LQH_API_KEY or run `lqh /login`).")
    client = create_client(api_key, load_config().api_base_url)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = await run_scoring(
        dataset_path=predictions_parquet,
        scorer_path=scorer_path,
        output_dir=output_dir,
        client=client,
        model_size=judge_size,
        concurrency=concurrency,
        run_inference=False,
    )
    return result.mean_score, result.median_score, result.scored, result.failed


async def _run_eval_stage(
    label: str,
    run_dir: Path,
    *,
    base_model: str,
    eval_dataset: Path,
    scorer: Path,
    system_prompt: str | None,
    schema_envelope: dict[str, Any] | None,
    max_new_tokens: int,
    judge_size: str,
    judge_concurrency: int,
    skip_inference: bool,
) -> EvalReport:
    """Inference (skippable) + scoring for a single eval stage."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    predictions = run_dir / "predictions.parquet"

    if skip_inference and predictions.exists():
        print(f"[{label}] reusing existing predictions at {predictions}")
    else:
        config = _build_infer_config(
            base_model=base_model,
            eval_dataset=eval_dataset,
            scorer=scorer,
            system_prompt=system_prompt,
            schema_envelope=schema_envelope,
            max_new_tokens=max_new_tokens,
        )
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        _run_subprocess("lqh.infer", config_path)

    if not predictions.exists():
        raise RuntimeError(f"[{label}] inference produced no predictions.parquet")

    print(f"[{label}] scoring predictions…", flush=True)
    mean, median, scored, failed = await _score_predictions(
        predictions, scorer, run_dir / "scoring",
        judge_size=judge_size, concurrency=judge_concurrency,
    )
    return EvalReport(
        label=label, run_dir=run_dir,
        mean=mean, median=median, scored=scored, failed=failed,
    )


def _train_stage(
    run_dir: Path,
    *,
    base_model: str,
    train_dataset: Path,
    num_epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    skip: bool,
) -> Path:
    """Run SFT (skippable). Returns the path to the trained model dir."""
    model_dir = run_dir / "model"
    if skip and model_dir.exists():
        print(f"[sft] reusing existing checkpoint at {model_dir}")
        return model_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = _build_sft_config(
        base_model=base_model,
        train_dataset=train_dataset,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    _run_subprocess("lqh.train", config_path)

    if not model_dir.exists():
        raise RuntimeError(f"SFT produced no model at {model_dir}")
    return model_dir


def _print_summary(
    baseline: EvalReport,
    sft: EvalReport,
    *,
    threshold: float,
) -> int:
    delta = sft.mean - baseline.mean
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  baseline  mean={baseline.mean:6.3f}  median={baseline.median:5.1f}  "
          f"scored={baseline.scored}/{baseline.scored + baseline.failed}")
    print(f"  post-SFT  mean={sft.mean:6.3f}  median={sft.median:5.1f}  "
          f"scored={sft.scored}/{sft.scored + sft.failed}")
    print(f"  delta     {delta:+.3f}")

    print()
    if delta < threshold:
        print(f"❌ FAIL — SFT regressed beyond threshold ({delta:+.3f} < {threshold:+.3f})")
        return 1
    if delta < 0:
        print(f"⚠️  WARN — SFT regressed slightly ({delta:+.3f}); within threshold {threshold:+.3f}")
        return 0
    print(f"✅ PASS — SFT improved by {delta:+.3f}")
    return 0


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--project-dir", type=Path, default=Path("example_project"),
                   help="Project root (used to resolve relative paths and place run dirs)")
    p.add_argument("--train-dataset", type=Path, required=True,
                   help="Path (relative to --project-dir) to the train parquet")
    p.add_argument("--eval-dataset", type=Path, required=True,
                   help="Path (relative to --project-dir) to the eval parquet")
    p.add_argument("--scorer", type=Path, required=True,
                   help="Path (relative to --project-dir) to the scorer .md")
    p.add_argument("--system-prompt", type=Path, default=None,
                   help="Optional system-prompt .md (relative to --project-dir)")
    p.add_argument("--schema", type=Path, default=None,
                   help="Optional JSON-schema envelope (relative to --project-dir)")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help=f"HF base-model id (default {DEFAULT_BASE_MODEL})")
    p.add_argument("--exp-name", default="sft_regression",
                   help="Run-dir prefix; runs land under <project>/runs/<exp_name>_*")

    # Training overrides
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=8192)

    # Judge config
    p.add_argument("--judge-size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--judge-concurrency", type=int, default=5)

    # Pass/fail
    p.add_argument("--regression-threshold", type=float, default=-0.5,
                   help="Fail if SFT mean is more than this many points BELOW baseline. "
                        "Default -0.5 (mild regression tolerated, larger regression fails).")

    # Skip flags for incremental dev
    p.add_argument("--skip-baseline", action="store_true",
                   help="Reuse existing baseline predictions if present.")
    p.add_argument("--skip-train", action="store_true",
                   help="Reuse existing SFT checkpoint if present.")
    p.add_argument("--skip-sft-eval", action="store_true",
                   help="Reuse existing post-SFT predictions if present.")
    args = p.parse_args()

    project = args.project_dir.resolve()
    if not project.is_dir():
        print(f"--project-dir does not exist: {project}", file=sys.stderr)
        return 2

    def in_project(p: Path) -> Path:
        return (project / p).resolve() if not p.is_absolute() else p

    train_ds = in_project(args.train_dataset)
    eval_ds = in_project(args.eval_dataset)
    scorer = in_project(args.scorer)
    sys_prompt_path = in_project(args.system_prompt) if args.system_prompt else None
    schema_path = in_project(args.schema) if args.schema else None

    for path in (train_ds, eval_ds, scorer):
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    system_prompt = _read_text(sys_prompt_path)
    schema_envelope = _load_schema_envelope(schema_path)

    runs_root = project / "runs"
    baseline_dir = runs_root / f"{args.exp_name}_baseline_eval"
    sft_dir = runs_root / f"{args.exp_name}_sft"
    sft_eval_dir = runs_root / f"{args.exp_name}_sft_eval"

    print(f"project:    {project}")
    print(f"train:      {train_ds.relative_to(project)}")
    print(f"eval:       {eval_ds.relative_to(project)}")
    print(f"scorer:     {scorer.relative_to(project)}")
    print(f"base model: {args.base_model}")
    print(f"runs:       {baseline_dir.name}, {sft_dir.name}, {sft_eval_dir.name}")

    # 1. Baseline eval
    baseline = await _run_eval_stage(
        "baseline", baseline_dir,
        base_model=args.base_model,
        eval_dataset=eval_ds,
        scorer=scorer,
        system_prompt=system_prompt,
        schema_envelope=schema_envelope,
        max_new_tokens=args.max_new_tokens,
        judge_size=args.judge_size,
        judge_concurrency=args.judge_concurrency,
        skip_inference=args.skip_baseline,
    )
    print(f"[baseline] mean={baseline.mean:.3f} median={baseline.median:.1f}")

    # 2. SFT
    sft_model_dir = _train_stage(
        sft_dir,
        base_model=args.base_model,
        train_dataset=train_ds,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        skip=args.skip_train,
    )

    # 3. Post-SFT eval
    sft = await _run_eval_stage(
        "post-sft", sft_eval_dir,
        base_model=str(sft_model_dir),
        eval_dataset=eval_ds,
        scorer=scorer,
        system_prompt=system_prompt,
        schema_envelope=schema_envelope,
        max_new_tokens=args.max_new_tokens,
        judge_size=args.judge_size,
        judge_concurrency=args.judge_concurrency,
        skip_inference=args.skip_sft_eval,
    )
    print(f"[post-sft] mean={sft.mean:.3f} median={sft.median:.1f}")

    return _print_summary(baseline, sft, threshold=args.regression_threshold)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
