"""Measure GRPO's incremental value on fresh data with a matched SFT control.

Fork of ``tests.benchmarks.dpo_value`` with the DPO arm replaced by GRPO
(GRPO plan Phase 4). Per training seed:

  1. SFT sweep on the SFT split → the baseline checkpoint.
  2. Continued-SFT grid on the fresh RL prompt pool — **the control that
     matters**: same starting checkpoint, same fresh data, supervised
     objective. If this matches GRPO, the bottleneck was data, not the
     algorithm.
  3. GRPO arm(s) on the same RL pool, starting from the same SFT
     checkpoint: ``grpo_rank`` (group-rank judge + pointwise anchor +
     guards — the proposal), and optional ablations ``grpo_pointwise``
     (pointwise-only reward) and ``guards_only``.

Environments: SFT/continued-SFT/eval run in the dev environment like
every other benchmark (local subprocesses). The GRPO trainer CANNOT run
there — it needs vLLM + trl 1.10 (see GRPO_IMPLEMENTATION.md) — so it
runs as a subprocess of ``--grpo-python``, a dedicated venv mirroring
the production grpo image:

    uv venv ~/grpo-venv -p 3.12
    uv pip install --python ~/grpo-venv/bin/python vllm==0.26.0 \
        trl==1.10.0 "peft>=0.15" accelerate datasets pyarrow \
        "openai>=1.0" "httpx>=0.27" "prompt_toolkit>=3.0" "rich>=13.0" \
        "pillow>=10.0" "packaging>=24.0" "huggingface_hub>=0.20" \
        "lm-format-enforcer>=0.11" hf_transfer
    uv pip install --python ~/grpo-venv/bin/python --no-deps -e <lqh_py>

Scoring: final-test evals run under the training judge (``--judge-size``,
default small) AND under ``--robustness-judge-size`` (default medium) —
a gain that evaporates when the judge changes is reward hacking, not
learning (plan gate G3).

Gates (per seed, same bar DPO was held to): GRPO−SFT ≥ +0.3 with the
paired 95% interval excluding zero, GRPO ≥ continued-SFT, and the
robustness-judge delta positive.

Usage (pilot):
    uv run python -m tests.benchmarks.grpo_value.run \
        --sft-train-size 2000 --rl-train-size 400 \
        --validation-size 60 --test-size 100 --seeds 17 \
        --grid-size tiny --grpo-max-steps 60

Full run:
    uv run python -m tests.benchmarks.grpo_value.run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lqh.auth import api_root, get_token
from lqh.client import create_client
from lqh.scoring import is_scoring_error
from lqh.subprocess_manager import SubprocessManager

from tests.benchmarks.base_vs_instruct.eval_local import eval_local
from tests.benchmarks.base_vs_instruct.run import (
    MODELS,
    _base_config,
    _dataset_ready,
    _generate_filtered_split,
    _run_sweep,
)
from tests.benchmarks.base_vs_instruct.tasks import resolve_tasks

from ..dpo_value.stats import paired_bootstrap
from ..dpo_value.voice_metrics import voice_metrics

logger = logging.getLogger("grpo_value")

GRPO_ARMS = ("grpo_rank", "grpo_pointwise", "guards_only", "grpo_probe")

# Reward-weight profiles per arm (see lqh.train.reward.build_reward_funcs:
# a zero weight skips the judge calls entirely, so ablations don't bill
# for signal they can't use).
_ARM_REWARDS: dict[str, dict[str, float]] = {
    "grpo_rank": {"rank_weight": 1.0, "absolute_weight": 0.2, "guard_weight": 1.0},
    "grpo_pointwise": {"rank_weight": 0.0, "absolute_weight": 1.0, "guard_weight": 1.0},
    "guards_only": {"rank_weight": 0.0, "absolute_weight": 0.0, "guard_weight": 1.0},
    # Same reward as grpo_rank; only the update budget differs
    # (--probe-lr / --probe-beta). Separates "no exploitable signal"
    # from "defaults too conservative to move the policy" when the main
    # arm reads null with tiny training KL.
    "grpo_probe": {"rank_weight": 1.0, "absolute_weight": 0.2, "guard_weight": 1.0},
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh-data GRPO vs continued-SFT value benchmark",
    )
    parser.add_argument("--task", default="voice_satisfaction")
    parser.add_argument("--model", default="1.2B-Instruct")
    parser.add_argument("--sft-train-size", type=int, default=10_000)
    parser.add_argument("--rl-train-size", type=int, default=2_000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seeds", default="17,29,41")
    parser.add_argument("--grid-size", choices=["tiny", "small"], default="small")
    parser.add_argument("--judge-size", choices=["small", "medium", "large"], default="small")
    parser.add_argument(
        "--robustness-judge-size", choices=["small", "medium", "large", "none"],
        default="medium",
        help="second judge for the final-test robustness check (G3); "
        "'none' disables it.",
    )
    parser.add_argument(
        "--arms", default="grpo_rank",
        help=f"comma list of GRPO arms to run: {', '.join(GRPO_ARMS)}",
    )
    parser.add_argument("--filter-threshold", type=float, default=7.0)
    parser.add_argument("--overgen-factor", type=float, default=1.6)
    parser.add_argument("--datagen-concurrency", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sweep-timeout", type=float, default=48 * 3600)
    parser.add_argument("--eval-timeout", type=float, default=3600)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--no-resume", action="store_true")
    # --- GRPO arm execution ---
    parser.add_argument(
        "--grpo-python", default="~/grpo-venv/bin/python",
        help="python interpreter of the vllm+trl-1.10 venv the GRPO "
        "trainer runs under (the dev env cannot run it).",
    )
    parser.add_argument("--grpo-max-steps", type=int, default=300)
    parser.add_argument("--grpo-num-generations", type=int, default=8)
    parser.add_argument("--grpo-temperature", type=float, default=0.3)
    parser.add_argument("--grpo-max-completion-length", type=int, default=512)
    parser.add_argument("--grpo-learning-rate", type=float, default=2e-6)
    parser.add_argument("--grpo-per-device-batch", type=int, default=8)
    parser.add_argument("--grpo-grad-accum", type=int, default=8)
    parser.add_argument("--grpo-timeout", type=float, default=8 * 3600)
    parser.add_argument(
        "--probe-lr", type=float, default=1e-5,
        help="learning rate for the grpo_probe arm (only used when "
        "'grpo_probe' is in --arms).",
    )
    parser.add_argument(
        "--probe-beta", type=float, default=0.001,
        help="KL beta for the grpo_probe arm.",
    )
    parser.add_argument(
        "--from-base", action="store_true",
        help="signal-seeking mode: run GRPO straight off the raw model "
        "(skipping the SFT/continued arms) and compare against the raw "
        "model. Faster read on 'does the reward move the policy', at the "
        "cost of not answering the product SFT→GRPO question.",
    )
    parser.add_argument(
        "--reference-model", default="",
        help="from-base mode only: path to an existing SFT checkpoint "
        "evaluated as a reference column (e.g. the seed's completed "
        "sweep winner).",
    )
    args = parser.parse_args(argv)
    for name in ("sft_train_size", "rl_train_size", "validation_size", "test_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        args.seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    except ValueError as exc:
        parser.error(f"--seeds must be comma-separated integers: {exc}")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in args.arms if a not in GRPO_ARMS]
    if unknown:
        parser.error(f"unknown arm(s) {unknown}; valid: {', '.join(GRPO_ARMS)}")
    total = args.grpo_per_device_batch * args.grpo_grad_accum
    if total % args.grpo_num_generations:
        parser.error(
            f"--grpo-per-device-batch × --grpo-grad-accum ({total}) must be "
            f"divisible by --grpo-num-generations ({args.grpo_num_generations})"
        )
    args.grpo_python = str(Path(args.grpo_python).expanduser())
    if not Path(args.grpo_python).exists():
        parser.error(
            f"--grpo-python not found: {args.grpo_python} — build the GRPO "
            "venv first (see this module's docstring / GRPO_IMPLEMENTATION.md)"
        )
    return args


def _resolve_model(value: str) -> tuple[str, str]:
    if value in MODELS:
        return value, MODELS[value]
    if "/" in value:
        return value.rsplit("/", 1)[-1], value
    raise SystemExit(f"unknown model {value!r}; use one of {', '.join(MODELS)} or a HF id")


async def _ensure_splits(
    *, workdir: Path, task: Any, sizes: dict[str, int], client: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, str], str]:
    scorer_rel = f"scorers/{task.name}.md"
    scorer_path = workdir / scorer_rel
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(task.scorer_md)

    paths: dict[str, str] = {}
    for split, size in sizes.items():
        rel = f"datasets/{task.name}_{split}/data.parquet"
        out_dir = (workdir / rel).parent
        # dedup.json marks a split that was deliberately shrunk by
        # _dedupe_splits — without it, a resume would see the
        # under-target parquet, regenerate the split, and reintroduce
        # the very duplicates the dedupe removed.
        deduped = (out_dir / "dedup.json").exists() and (workdir / rel).exists()
        if not deduped and not (
            not args.no_resume and _dataset_ready(workdir / rel, size)
        ):
            await _generate_filtered_split(
                script_path=task.pipeline_path,
                scorer_path=scorer_path,
                target=size,
                out_dir=out_dir,
                client=client,
                concurrency=args.datagen_concurrency,
                threshold=args.filter_threshold,
                overgen_factor=args.overgen_factor,
                label=f"{task.name}_{split}",
                judge_size=args.judge_size,
            )
        paths[split] = rel
    _dedupe_splits(workdir, paths)
    return paths, scorer_rel


def _prompt_key(value: Any) -> str:
    messages = json.loads(value) if isinstance(value, str) else value
    prompt = (
        messages[:-1]
        if messages and messages[-1].get("role") == "assistant"
        else messages
    )
    return json.dumps(prompt, sort_keys=True)


def _dedupe_splits(workdir: Path, paths: dict[str, str]) -> None:
    """Enforce exact-prompt disjointness by DROPPING duplicates from later
    splits (dict order: training splits first, test last — so test ends up
    clean of any prompt a model trained on).

    At pilot scale exact collisions never happened and dpo_value could
    afford to hard-fail; at 10k+ samples the pipeline's prompt space
    collides occasionally by construction. A few dropped rows cost a
    sliver of statistical power; a hard failure costs the whole datagen
    spend. Still fails on gross overlap (>10% of a split), which would
    mean the generator is degenerate, not unlucky.
    """
    seen: set[str] = set()
    for split, rel in paths.items():
        path = workdir / rel
        table = pq.read_table(path)
        keep: list[int] = []
        for i, value in enumerate(table["messages"].to_pylist()):
            key = _prompt_key(value)
            if key not in seen:
                seen.add(key)
                keep.append(i)
        dropped = len(table) - len(keep)
        if dropped:
            if dropped > 0.1 * len(table):
                raise RuntimeError(
                    f"dataset leakage: {dropped}/{len(table)} rows of {split} "
                    "duplicate earlier splits — generator looks degenerate, "
                    "not unlucky; refusing to shrink it silently"
                )
            pq.write_table(table.take(keep), path)
            logger.info(
                "dedupe: dropped %d duplicate prompt(s) from %s (%d rows remain)",
                dropped, split, len(keep),
            )
        (path.parent / "dedup.json").write_text(
            json.dumps({"rows": len(keep), "dropped": dropped}) + "\n"
        )


def _score_vector(scores_dir: Path) -> dict[int, float]:
    table = pq.read_table(scores_dir / "results.parquet")
    result: dict[int, float] = {}
    for index, score, reasoning in zip(
        table["sample_index"].to_pylist(),
        table["score"].to_pylist(),
        table["reasoning"].to_pylist(),
        strict=True,
    ):
        if score is not None and not is_scoring_error(reasoning or ""):
            result[int(index)] = float(score)
    return result


def _comparison(
    treatment: dict[int, float], control: dict[int, float],
    args: argparse.Namespace, *, seed: int,
) -> dict[str, Any]:
    return asdict(paired_bootstrap(
        treatment, control, samples=args.bootstrap_samples, seed=seed,
    ))


def _grpo_config(
    *, arm: str, base_model: str, dataset_rel: str, scorer_rel: str,
    args: argparse.Namespace, seed: int,
) -> dict[str, Any]:
    learning_rate = (
        args.probe_lr if arm == "grpo_probe" else args.grpo_learning_rate
    )
    cfg = {
        "type": "grpo",
        "base_model": base_model,
        "dataset": dataset_rel,
        # Unused (no in-training eval — the benchmark evals externally),
        # but keeps the config schema consistent with SFT/DPO runs.
        "eval_dataset": dataset_rel,
        "scorer": scorer_rel,
        "training": {
            "learning_rate": learning_rate,
            "per_device_batch_size": args.grpo_per_device_batch,
            "gradient_accumulation_steps": args.grpo_grad_accum,
            "logging_steps": 5,
            "max_seq_length": 2048,
            "seed": seed,
        },
        "grpo": {
            "num_generations": args.grpo_num_generations,
            "max_steps": args.grpo_max_steps,
            "max_completion_length": args.grpo_max_completion_length,
            "temperature": args.grpo_temperature,
            "save_steps": max(10, args.grpo_max_steps // 10),
        },
        "reward": {
            "judge_size": args.judge_size,
            "concurrency": 16,
            **_ARM_REWARDS[arm],
        },
    }
    if arm == "grpo_probe":
        cfg["grpo"]["beta"] = args.probe_beta
    return cfg


async def _run_grpo(
    *, workdir: Path, run_name: str, config: dict[str, Any],
    grpo_python: str, timeout: float, resume: bool,
) -> Path:
    """Run one GRPO training as a subprocess of the GRPO venv.

    Mirrors ``_run_sweep``'s contract (returns the winner model dir) but
    without SubprocessManager.start — the trainer must run under
    *grpo_python*, not this interpreter. Status/progress still flow
    through the standard run-dir files, so ``get_status`` works.
    """
    run_dir = workdir / "runs" / run_name
    model_dir = run_dir / "model-lora"
    manager = SubprocessManager()
    if resume and model_dir.exists() \
            and manager.get_status(run_dir).state == "completed":
        logger.info("grpo %s: reusing completed run", run_name)
        return model_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log_path = run_dir / "stdout.log"
    logger.info("grpo %s: launching under %s", run_name, grpo_python)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [grpo_python, "-u", "-m", "lqh.train", str(run_dir / "config.json")],
            cwd=workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),  # carries LQH_API_TOKEN / LQH_BASE_URL
        )
        deadline = time.monotonic() + timeout
        state = "unknown"
        while time.monotonic() < deadline:
            await asyncio.sleep(10)
            state = manager.get_status(run_dir).state
            if state in ("completed", "failed"):
                break
            if proc.poll() is not None and state not in ("completed", "failed"):
                # Give trailing status writes a moment, then re-read.
                await asyncio.sleep(5)
                state = manager.get_status(run_dir).state
                break
        else:
            proc.kill()
            raise RuntimeError(f"grpo:{run_name} timed out after {timeout:.0f}s")
    if state != "completed":
        tail = ""
        try:
            tail = "".join(log_path.read_text().splitlines(keepends=True)[-30:])
        except OSError:
            pass
        raise RuntimeError(
            f"grpo:{run_name} ended in state {state!r} "
            f"(exit={proc.poll()}):\n{tail}"
        )
    if not model_dir.exists():
        raise RuntimeError(f"grpo:{run_name} completed but no {model_dir}")
    return model_dir


async def _evaluate(
    *, workdir: Path, name: str, model: str, paths: dict[str, str],
    scorer_rel: str, client: Any, args: argparse.Namespace,
    judge_size: str, split: str = "test",
) -> tuple[float | None, dict[int, float], dict[str, Any]]:
    result = await eval_local(
        workdir=workdir,
        run_name=name,
        model_path=model,
        eval_parquet=workdir / paths[split],
        scorer_path=workdir / scorer_rel,
        client=client,
        judge_size=judge_size,
        max_new_tokens=args.max_new_tokens,
        infer_timeout=args.eval_timeout,
        resume=not args.no_resume,
    )
    task_metrics = (
        voice_metrics(result.predictions_path, workdir / paths[split])
        if args.task == "voice_satisfaction" and split == "test" else {}
    )
    return result.mean, _score_vector(result.scores_dir), task_metrics


async def _evaluate_arm(
    *, prefix: str, name: str, model_path: str, workdir: Path,
    paths: dict[str, str], scorer_rel: str, client: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    robustness = (
        args.robustness_judge_size
        if args.robustness_judge_size != "none" else None
    )
    mean, scores, metrics = await _evaluate(
        workdir=workdir, name=f"{prefix}__{name}_test",
        model=model_path, paths=paths, scorer_rel=scorer_rel,
        client=client, args=args, judge_size=args.judge_size,
    )
    out = {"mean": mean, "scores": scores, "metrics": metrics}
    if robustness:
        r_mean, r_scores, _ = await _evaluate(
            workdir=workdir, name=f"{prefix}__{name}_test_{robustness}",
            model=model_path, paths=paths, scorer_rel=scorer_rel,
            client=client, args=args, judge_size=robustness,
        )
        out["robust_mean"] = r_mean
        out["robust_scores"] = r_scores
    return out


async def _run_seed_from_base(
    *, seed: int, workdir: Path, task: Any, model_key: str, hf_id: str,
    paths: dict[str, str], scorer_rel: str, client: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Signal-seeking mode (--from-base): GRPO straight off the raw model,
    compared against the raw model itself and (optionally) a reference
    SFT checkpoint. Skips SFT/continued arms entirely.

    Not the product question — the product flow is SFT→GRPO — but with a
    strong SFT baseline every arm fights for scraps; from the raw model
    there is real headroom, so "does the reward move the policy at all?"
    gets a fast, unambiguous read.
    """
    prefix = f"{task.name}__{model_key}__base_seed{seed}"
    resume = not args.no_resume
    common_eval = dict(
        prefix=prefix, workdir=workdir, paths=paths, scorer_rel=scorer_rel,
        client=client, args=args,
    )
    arms: dict[str, dict[str, Any]] = {
        "baseline": await _evaluate_arm(
            name="baseline", model_path=hf_id, **common_eval,
        ),
    }
    if args.reference_model:
        arms["sft_reference"] = await _evaluate_arm(
            name="sft_reference",
            model_path=str(Path(args.reference_model).expanduser().resolve()),
            **common_eval,
        )
    for arm in args.arms:
        grpo_model = await _run_grpo(
            workdir=workdir, run_name=f"{prefix}__{arm}",
            config=_grpo_config(
                arm=arm, base_model=hf_id,
                dataset_rel=paths["rl_train"], scorer_rel=scorer_rel,
                args=args, seed=seed,
            ),
            grpo_python=args.grpo_python,
            timeout=args.grpo_timeout, resume=resume,
        )
        arms[arm] = await _evaluate_arm(
            name=arm, model_path=str(grpo_model.resolve()), **common_eval,
        )

    comparisons: dict[str, Any] = {}
    for i, arm in enumerate(args.arms):
        comparisons[f"{arm}_minus_baseline"] = _comparison(
            arms[arm]["scores"], arms["baseline"]["scores"],
            args, seed=seed + 10 + i,
        )
        if arms[arm].get("robust_scores") is not None \
                and arms["baseline"].get("robust_scores") is not None:
            comparisons[f"{arm}_minus_baseline_robust"] = _comparison(
                arms[arm]["robust_scores"], arms["baseline"]["robust_scores"],
                args, seed=seed + 30 + i,
            )
    if "sft_reference" in arms:
        comparisons["sft_reference_minus_baseline"] = _comparison(
            arms["sft_reference"]["scores"], arms["baseline"]["scores"],
            args, seed=seed + 50,
        )
    return {
        "seed": seed,
        "means": {name: data["mean"] for name, data in arms.items()},
        "robust_means": {
            name: data.get("robust_mean") for name, data in arms.items()
        },
        "task_metrics": {name: data["metrics"] for name, data in arms.items()},
        "comparisons": comparisons,
    }


async def _run_seed(
    *, seed: int, workdir: Path, task: Any, model_key: str, hf_id: str,
    paths: dict[str, str], scorer_rel: str, client: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prefix = f"{task.name}__{model_key}__seed{seed}"
    resume = not args.no_resume
    common_sweep = dict(
        workdir=workdir, grid_size=args.grid_size,
        timeout=args.sweep_timeout, resume=resume,
    )

    async def evaluate_arm(name: str, model_path: str) -> dict[str, Any]:
        return await _evaluate_arm(
            prefix=prefix, name=name, model_path=model_path, workdir=workdir,
            paths=paths, scorer_rel=scorer_rel, client=client, args=args,
        )

    # 1) SFT baseline.
    sft_config = _base_config(
        run_type="sft", base_model=hf_id,
        dataset_rel=paths["sft_train"], eval_rel=paths["validation"],
        scorer_rel=scorer_rel, train_size=args.sft_train_size,
    )
    sft_config["training"]["seed"] = seed
    sft_model, _ = await _run_sweep(
        run_name=f"{prefix}__sft", base_config=sft_config, **common_sweep,
    )
    arms: dict[str, dict[str, Any]] = {
        "sft": await evaluate_arm("sft", str(sft_model.resolve())),
    }

    # 2) Continued-SFT control on the fresh RL pool.
    continued_config = _base_config(
        run_type="sft", base_model=str(sft_model.resolve()),
        dataset_rel=paths["rl_train"], eval_rel=paths["validation"],
        scorer_rel=scorer_rel, train_size=args.rl_train_size,
    )
    continued_config["training"]["seed"] = seed
    continued_grid = [
        {"id": f"continued_sft_lr{lr:g}_e{epochs}",
         "overrides": {"training": {"learning_rate": lr, "num_epochs": epochs}}}
        for lr in (5e-6, 1e-5, 2e-5)
        for epochs in (1, 2)
    ]
    continued_model, _ = await _run_sweep(
        run_name=f"{prefix}__continued_sft", base_config=continued_config,
        grid_override=continued_grid, **common_sweep,
    )
    arms["continued_sft"] = await evaluate_arm(
        "continued_sft", str(continued_model.resolve()),
    )

    # 3) GRPO arm(s) on the same fresh pool, same starting checkpoint.
    for arm in args.arms:
        grpo_model = await _run_grpo(
            workdir=workdir, run_name=f"{prefix}__{arm}",
            config=_grpo_config(
                arm=arm, base_model=str(sft_model.resolve()),
                dataset_rel=paths["rl_train"], scorer_rel=scorer_rel,
                args=args, seed=seed,
            ),
            grpo_python=args.grpo_python,
            timeout=args.grpo_timeout, resume=resume,
        )
        arms[arm] = await evaluate_arm(arm, str(grpo_model.resolve()))

    comparisons: dict[str, Any] = {
        "continued_sft_minus_sft": _comparison(
            arms["continued_sft"]["scores"], arms["sft"]["scores"],
            args, seed=seed + 1,
        ),
    }
    for i, arm in enumerate(args.arms):
        comparisons[f"{arm}_minus_sft"] = _comparison(
            arms[arm]["scores"], arms["sft"]["scores"], args, seed=seed + 10 + i,
        )
        comparisons[f"{arm}_minus_continued_sft"] = _comparison(
            arms[arm]["scores"], arms["continued_sft"]["scores"],
            args, seed=seed + 20 + i,
        )
        if arms[arm].get("robust_scores") is not None \
                and arms["sft"].get("robust_scores") is not None:
            comparisons[
                f"{arm}_minus_sft_{args.robustness_judge_size}_judge"
            ] = _comparison(
                arms[arm]["robust_scores"], arms["sft"]["robust_scores"],
                args, seed=seed + 30 + i,
            )

    return {
        "seed": seed,
        "means": {name: data["mean"] for name, data in arms.items()},
        "robust_means": {
            name: data.get("robust_mean") for name, data in arms.items()
        },
        "task_metrics": {name: data["metrics"] for name, data in arms.items()},
        "comparisons": comparisons,
    }


def _render_from_base_report(
    meta: dict[str, Any], rows: list[dict[str, Any]],
) -> str:
    arms: list[str] = meta["arms"]
    has_ref = any("sft_reference" in row["means"] for row in rows)
    header = ["Seed", "Raw model"] + (["SFT ref"] if has_ref else []) + arms \
        + [f"{a}−raw [95% CI]" for a in arms]
    lines = [
        "# GRPO from-base signal trial",
        "",
        f"Task: `{meta['task']}`; model: `{meta['model']}` (raw, no SFT); "
        f"seeds: `{meta['seeds']}`; train judge: `{meta['judge_size']}`; "
        f"robustness judge: `{meta['robustness_judge_size']}`.",
        "",
        "Signal-seeking mode: GRPO trains directly on the raw model, so any "
        "reward signal has real headroom to show up. This answers 'does the "
        "reward move the policy', NOT the product SFT→GRPO question.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---:|" * len(header),
    ]
    for row in rows:
        means = row["means"]
        cells = [str(row["seed"]), f"{means['baseline']:.2f}"]
        if has_ref:
            ref = means.get("sft_reference")
            cells.append(f"{ref:.2f}" if ref is not None else "—")
        for a in arms:
            cells.append(f"{means[a]:.2f}" if means.get(a) is not None else "—")
        for a in arms:
            c = row["comparisons"][f"{a}_minus_baseline"]
            cells.append(
                f"{c['mean']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Robustness-judge deltas (arm − raw, second judge): "
        + "; ".join(
            f"seed {row['seed']} {a}: {row['comparisons'][k]['mean']:+.2f}"
            for row in rows for a in arms
            if (k := f"{a}_minus_baseline_robust") in row["comparisons"]
        )
        + ".",
    ]
    return "\n".join(lines) + "\n"


def _render_report(meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    arms: list[str] = meta["arms"]
    primary = arms[0] if arms else "grpo_rank"
    lines = [
        "# Fresh-data GRPO value benchmark",
        "",
        f"Task: `{meta['task']}`; model: `{meta['model']}`; seeds: `{meta['seeds']}`; "
        f"train judge: `{meta['judge_size']}`; robustness judge: "
        f"`{meta['robustness_judge_size']}`.",
        "",
        "All values are final-test judge scores. Confidence intervals are paired "
        "bootstrap 95% intervals over identical test sample IDs.",
        "",
        "| Seed | SFT | Continued SFT | " + " | ".join(arms)
        + f" | {primary}-SFT [95% CI] | {primary}-continued [95% CI] |",
        "|---:" + "|---:" * (3 + len(arms)) + "|---:|",
    ]
    for row in rows:
        means = row["means"]
        gs = row["comparisons"][f"{primary}_minus_sft"]
        gc = row["comparisons"][f"{primary}_minus_continued_sft"]
        arm_cells = " | ".join(
            f"{means[a]:.2f}" if means.get(a) is not None else "—" for a in arms
        )
        lines.append(
            f"| {row['seed']} | {means['sft']:.2f} | {means['continued_sft']:.2f} "
            f"| {arm_cells} | {gs['mean']:+.2f} [{gs['ci_low']:+.2f}, {gs['ci_high']:+.2f}] "
            f"| {gc['mean']:+.2f} [{gc['ci_low']:+.2f}, {gc['ci_high']:+.2f}] |"
        )
    demonstrated = []
    for row in rows:
        gs = row["comparisons"][f"{primary}_minus_sft"]
        gc = row["comparisons"][f"{primary}_minus_continued_sft"]
        robust_key = next(
            (k for k in row["comparisons"] if k.startswith(f"{primary}_minus_sft_")),
            None,
        )
        robust_ok = (
            row["comparisons"][robust_key]["mean"] > 0 if robust_key else True
        )
        demonstrated.append(
            gs["mean"] >= 0.3 and gs["ci_low"] > 0
            and gc["mean"] >= 0 and robust_ok
        )
    gains = [row["comparisons"][f"{primary}_minus_sft"]["mean"] for row in rows]
    lines.extend([
        "",
        f"Mean {primary}-SFT delta across seeds: `{sum(gains) / len(gains):+.2f}`.",
        "",
        "Criterion per seed: gain >= +0.3 with the paired interval excluding "
        "zero, not behind the continued-SFT control, and (when enabled) a "
        "positive delta under the robustness judge. "
        f"Result: `{sum(demonstrated)}/{len(demonstrated)}` seeds pass.",
        "",
        "The continued-SFT column is the extra-data control: same starting "
        "checkpoint, same fresh prompt pool, supervised objective.",
    ])
    if meta["task"] == "voice_satisfaction":
        lines.extend([
            "",
            "## Voice-satisfaction diagnostics",
            "",
            "| Seed | Stage | JSON valid | Score direction | Frustration miss | Failure tags exact |",
            "|---:|---|---:|---:|---:|---:|",
        ])
        for row in rows:
            for stage, metrics in row["task_metrics"].items():
                if not metrics:
                    continue
                lines.append(
                    f"| {row['seed']} | {stage} | {metrics['json_valid_rate']:.1%} "
                    f"| {metrics['score_direction_accuracy']:.1%} "
                    f"| {metrics['frustration_miss_rate']:.1%} "
                    f"| {metrics['failure_tags_exact_rate']:.1%} |"
                )
    return "\n".join(lines) + "\n"


async def _main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = get_token()
    if not token:
        raise SystemExit("not authenticated; run lqh /login or set LQH_API_TOKEN")
    root = api_root()
    client = create_client(token, root + "/v1")
    # Inline self-scoring for training subprocesses + the judge channel for
    # the GRPO reward engine (cloud-mode client in the grpo venv).
    os.environ["LQH_API_TOKEN"] = token
    os.environ["LQH_BASE_URL"] = root

    run_name = args.run_name or f"grpo-value-{time.strftime('%Y%m%d-%H%M%S')}"
    workdir = Path(args.workdir).expanduser() if args.workdir else Path(
        f"~/.lqh-grpo-value/{run_name}"
    ).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    task = resolve_tasks([args.task])[0]
    model_key, hf_id = _resolve_model(args.model)
    sizes = {
        "sft_train": args.sft_train_size,
        "rl_train": args.rl_train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    paths, scorer_rel = await _ensure_splits(
        workdir=workdir, task=task, sizes=sizes, client=client, args=args,
    )

    seed_runner = _run_seed_from_base if args.from_base else _run_seed
    report_name = "report_from_base.md" if args.from_base else "report.md"
    renderer = _render_from_base_report if args.from_base else _render_report
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        logger.info("starting seed %d", seed)
        rows.append(await seed_runner(
            seed=seed, workdir=workdir, task=task, model_key=model_key,
            hf_id=hf_id, paths=paths, scorer_rel=scorer_rel,
            client=client, args=args,
        ))
        meta = {
            "run_name": run_name,
            "task": task.name,
            "model": hf_id,
            "seeds": args.seeds,
            "sizes": sizes,
            "grid_size": args.grid_size,
            "judge_size": args.judge_size,
            "robustness_judge_size": args.robustness_judge_size,
            "arms": args.arms,
            "grpo": {
                "max_steps": args.grpo_max_steps,
                "num_generations": args.grpo_num_generations,
                "temperature": args.grpo_temperature,
                "max_completion_length": args.grpo_max_completion_length,
                "learning_rate": args.grpo_learning_rate,
                "completions_per_step":
                    args.grpo_per_device_batch * args.grpo_grad_accum,
            },
        }
        results_name = (
            "results_from_base.json" if args.from_base else "results.json"
        )
        (workdir / results_name).write_text(
            json.dumps({"meta": meta, "seeds": rows}, indent=2,
                       default=str) + "\n"
        )
        (workdir / report_name).write_text(renderer(meta, rows))
    logger.info("report: %s", workdir / report_name)
    print((workdir / report_name).read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
