"""Exploration study: production-RL GRPO hyperparameters vs the shipped ones.

Motivated by the LFM-release GRPO settings used in production (Slack,
2026-08-14): rollouts at temperature 1.0 / top_p 1.0 (no min_p, no
repetition penalty, top_k off), KL beta in {0.001, 0.01, 0}, and the
warning that anything below T=0.7 "messes up the exploration". Their LR
grid (2e-8..2e-6) is FULL fine-tuning; we train LoRA, so the comparable
move is testing a LARGER lr than our measured 1e-5, not a smaller one.

Every arm uses the pure sampling profile (top_p=1.0, min_p=0, rep=1.0)
and varies temperature / lr / KL against the shipped LFM low-temperature
discipline (T=0.3, min_p=0.05, rep=1.05) whose results are already on
disk in the same workdir:

  raw 1.2B 4.48 · from-base GRPO@T=0.3 4.82 (+0.34) · SFT 7.59 ·
  SFT+GRPO@T=0.3 7.62 (+0.03) · continued-SFT 7.73     (RESULTS.md)

Two deliverable metrics:
  1. baseline vs SFT vs GRPO      — the base_* arms (raw-model start)
  2. SFT vs SFT+GRPO              — the sft_* arms (SFT-winner start)

Designed to run split across two GPU boxes: ``--mode train`` runs (and
test-evals) a subset of arms on each box against a shared-layout workdir;
after rsyncing the run dirs together, ``--mode report`` on one box
computes all paired comparisons against the SAME cached baseline/SFT
score vectors and renders the report. Single seed (17) by design — this
is a screening study; the winning config needs 2 more seeds before any
default moves (same provenance bar as RESULTS.md).

Usage (toka):
    uv run python -m tests.benchmarks.grpo_value.exploration_study \
        --workdir ~/.lqh-grpo-value/full1 \
        --sft-model <seed17 sft winner model dir> \
        --mode train --arms base_t10,base_t07,base_t10_hot,base_t10_nokl
    # lambda: --arms sft_t10,sft_t07,sft_t10_hot,sft_t10_nokl
    # then, with lambda's runs/ rsynced back: --mode report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from lqh.auth import api_root, get_token
from lqh.client import create_client

from tests.benchmarks.base_vs_instruct.tasks import resolve_tasks

from .run import (
    _ARM_REWARDS,
    _comparison,
    _ensure_splits,
    _evaluate_arm,
    _resolve_model,
    _run_grpo,
    _score_vector,
)

logger = logging.getLogger("grpo_exploration")

# Fernando's rollout profile, common to every arm. top_k is already off
# (vLLM default); max_response_length 16k is irrelevant for this
# short-output task (completions capped at 512 like every prior run).
PURE_SAMPLING = {"top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0}

# start: "base" (raw model — metric 1) or "sft" (SFT winner — metric 2).
# Optional keys: "groups" (prompt groups per optimizer step; default 8 —
# realized as gradient_accumulation_steps since micro-batch stays 8),
# "steps" (max_steps override), "train_judge" (training-reward judge
# size override; evals stay on the standard judges).
ARMS: dict[str, dict[str, Any]] = {
    "base_t10":      {"start": "base", "temperature": 1.0, "lr": 1e-5, "beta": 0.001},
    "base_t07":      {"start": "base", "temperature": 0.7, "lr": 1e-5, "beta": 0.001},
    "base_t10_hot":  {"start": "base", "temperature": 1.0, "lr": 4e-5, "beta": 0.001},
    "base_t10_nokl": {"start": "base", "temperature": 1.0, "lr": 1e-5, "beta": 0.0},
    "sft_t10":       {"start": "sft",  "temperature": 1.0, "lr": 1e-5, "beta": 0.001},
    "sft_t07":       {"start": "sft",  "temperature": 0.7, "lr": 1e-5, "beta": 0.001},
    "sft_t10_hot":   {"start": "sft",  "temperature": 1.0, "lr": 4e-5, "beta": 0.001},
    "sft_t10_nokl":  {"start": "sft",  "temperature": 1.0, "lr": 1e-5, "beta": 0.0},
    # Batch-scale probe (production runs 128-256 groups/step vs our 8;
    # "small batch sizes updates the model too frequently"): 4x the
    # groups at 1/4 the steps — SAME 19.2k-completion rollout/judge
    # budget as base_t10, so the comparison isolates update granularity.
    "base_t10_bs32": {
        "start": "base", "temperature": 1.0, "lr": 1e-5, "beta": 0.001,
        "groups": 32, "steps": 75,
    },
    # Judge-fidelity probe: judge:medium as the TRAINING reward (the
    # from-base result made judge fidelity the measured ceiling).
    "base_t10_jm": {
        "start": "base", "temperature": 1.0, "lr": 1e-5, "beta": 0.001,
        "train_judge": "medium",
    },
    # Fidelity-vs-cost curve: judge:large as the training reward.
    "base_t10_jl": {
        "start": "base", "temperature": 1.0, "lr": 1e-5, "beta": 0.001,
        "train_judge": "large",
    },
    # Does judge fidelity reopen the post-SFT null? Best-known SFT
    # config (T=1.0, beta=0 — the KL anchor is wrong for continuation)
    # with the medium training judge.
    "sft_t10_nokl_jm": {
        "start": "sft", "temperature": 1.0, "lr": 1e-5, "beta": 0.0,
        "train_judge": "medium",
    },
}

# Cached eval run names of the shipped-sampling (T=0.3) counterparts in
# the same workdir, quoted in the report when present. grpo_probe IS the
# lr 1e-5 / beta 0.001 from-base arm (RESULTS.md); seed17 grpo_rank is
# the from-SFT main-table arm.
T03_REFERENCE_EVALS = {
    "base_t03_ref": "{task}__{model}__base_seed{seed}__grpo_probe_test",
    "sft_t03_ref": "{task}__{model}__seed{seed}__grpo_rank_test",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=["train", "report"], default="train")
    parser.add_argument(
        "--arms", default="",
        help=f"comma list (train mode): {', '.join(ARMS)}",
    )
    parser.add_argument(
        "--sft-model", default="",
        help="seed's SFT winner model dir — start point of sft_* arms and "
        "the sft_reference eval column.",
    )
    parser.add_argument("--task", default="voice_satisfaction")
    parser.add_argument("--model", default="1.2B-Instruct")
    parser.add_argument("--seed", type=int, default=17)
    # Split sizes must MATCH the workdir's existing splits or
    # _ensure_splits would regenerate data — defaults mirror full1.
    parser.add_argument("--sft-train-size", type=int, default=10_000)
    parser.add_argument("--rl-train-size", type=int, default=2_000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--judge-size", default="small")
    parser.add_argument("--robustness-judge-size", default="medium")
    parser.add_argument("--filter-threshold", type=float, default=7.0)
    parser.add_argument("--overgen-factor", type=float, default=1.6)
    parser.add_argument("--datagen-concurrency", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-timeout", type=float, default=3600)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--grpo-python", default="~/grpo-venv/bin/python")
    parser.add_argument("--grpo-max-steps", type=int, default=300)
    parser.add_argument("--grpo-timeout", type=float, default=8 * 3600)
    args = parser.parse_args(argv)
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s) {unknown}; valid: {', '.join(ARMS)}")
    if args.mode == "train" and not args.arms:
        parser.error("--mode train needs --arms")
    needs_sft = args.mode == "report" or any(
        ARMS[a]["start"] == "sft" for a in args.arms
    )
    if needs_sft and not args.sft_model:
        parser.error("--sft-model is required for sft_* arms and report mode")
    if args.sft_model:
        args.sft_model = str(Path(args.sft_model).expanduser().resolve())
        if not Path(args.sft_model).exists():
            parser.error(f"--sft-model not found: {args.sft_model}")
    args.grpo_python = str(Path(args.grpo_python).expanduser())
    if args.mode == "train" and not Path(args.grpo_python).exists():
        parser.error(f"--grpo-python not found: {args.grpo_python}")
    return args


def _arm_config(
    *, name: str, spec: dict[str, Any], base_model: str, dataset_rel: str,
    scorer_rel: str, args: argparse.Namespace,
) -> dict[str, Any]:
    max_steps = int(spec.get("steps", args.grpo_max_steps))
    return {
        "type": "grpo",
        "base_model": base_model,
        "dataset": dataset_rel,
        "eval_dataset": dataset_rel,  # unused; schema consistency
        "scorer": scorer_rel,
        "training": {
            "learning_rate": spec["lr"],
            "per_device_batch_size": 8,
            # groups/step == grad_accum: micro-batch is 8 completions =
            # one whole group at G=8, so accumulation counts groups.
            "gradient_accumulation_steps": int(spec.get("groups", 8)),
            "logging_steps": 5,
            "max_seq_length": 2048,
            "seed": args.seed,
        },
        "grpo": {
            "num_generations": 8,
            "max_steps": max_steps,
            "max_completion_length": 512,
            "temperature": spec["temperature"],
            "beta": spec["beta"],
            **PURE_SAMPLING,
            "save_steps": max(10, max_steps // 10),
        },
        "reward": {
            "judge_size": spec.get("train_judge", args.judge_size),
            "concurrency": 16,
            **_ARM_REWARDS["grpo_rank"],
        },
    }


def _eval_prefix(task: Any, model_key: str, seed: int) -> str:
    # Deliberately the from-base trial's prefix so the baseline and
    # sft_reference test evals cached by that trial are reused verbatim —
    # paired comparisons then share the exact same control vectors.
    return f"{task.name}__{model_key}__base_seed{seed}"


async def _train(args: argparse.Namespace, ctx: dict[str, Any]) -> None:
    for name in args.arms:
        spec = ARMS[name]
        base_model = ctx["hf_id"] if spec["start"] == "base" else args.sft_model
        config = _arm_config(
            name=name, spec=spec, base_model=base_model,
            dataset_rel=ctx["paths"]["rl_train"],
            scorer_rel=ctx["scorer_rel"], args=args,
        )
        logger.info(
            "arm %s: start=%s T=%s lr=%g beta=%g", name, spec["start"],
            spec["temperature"], spec["lr"], spec["beta"],
        )
        model_dir = await _run_grpo(
            workdir=ctx["workdir"], run_name=f"exp{args.seed}__{name}",
            config=config, grpo_python=args.grpo_python,
            timeout=args.grpo_timeout, resume=not args.no_resume,
        )
        result = await _evaluate_arm(
            prefix=ctx["prefix"], name=name,
            model_path=str(model_dir.resolve()), workdir=ctx["workdir"],
            paths=ctx["paths"], scorer_rel=ctx["scorer_rel"],
            client=ctx["client"], args=args,
        )
        logger.info(
            "arm %s: test mean=%s robust=%s", name,
            result.get("mean"), result.get("robust_mean"),
        )


def _load_cached_scores(
    workdir: Path, run_name: str,
) -> tuple[float | None, dict[int, float]] | None:
    scores_dir = workdir / "runs" / run_name / "scores"
    if not (scores_dir / "results.parquet").exists():
        return None
    scores = _score_vector(scores_dir)
    if not scores:
        return None
    return sum(scores.values()) / len(scores), scores


async def _report(args: argparse.Namespace, ctx: dict[str, Any]) -> None:
    workdir, prefix, client = ctx["workdir"], ctx["prefix"], ctx["client"]
    common = dict(
        prefix=prefix, workdir=workdir, paths=ctx["paths"],
        scorer_rel=ctx["scorer_rel"], client=client, args=args,
    )
    # Controls — cache hits when the from-base trial ran in this workdir.
    evals: dict[str, dict[str, Any]] = {
        "baseline": await _evaluate_arm(
            name="baseline", model_path=ctx["hf_id"], **common,
        ),
        "sft_reference": await _evaluate_arm(
            name="sft_reference", model_path=args.sft_model, **common,
        ),
    }
    done: list[str] = []
    for name in ARMS:
        model_dir = workdir / "runs" / f"exp{args.seed}__{name}" / "model-lora"
        if not model_dir.exists():
            logger.info("arm %s: no trained model in workdir — skipped", name)
            continue
        evals[name] = await _evaluate_arm(
            name=name, model_path=str(model_dir.resolve()), **common,
        )
        done.append(name)

    comparisons: dict[str, Any] = {
        "sft_reference_minus_baseline": _comparison(
            evals["sft_reference"]["scores"], evals["baseline"]["scores"],
            args, seed=args.seed + 50,
        ),
    }
    for i, name in enumerate(done):
        control = "baseline" if ARMS[name]["start"] == "base" else "sft_reference"
        comparisons[f"{name}_minus_{control}"] = _comparison(
            evals[name]["scores"], evals[control]["scores"],
            args, seed=args.seed + 100 + i,
        )
        if evals[name].get("robust_scores") and evals[control].get("robust_scores"):
            comparisons[f"{name}_minus_{control}_robust"] = _comparison(
                evals[name]["robust_scores"], evals[control]["robust_scores"],
                args, seed=args.seed + 200 + i,
            )

    t03: dict[str, float] = {}
    for label, template in T03_REFERENCE_EVALS.items():
        cached = _load_cached_scores(workdir, template.format(
            task=ctx["task"].name, model=ctx["model_key"], seed=args.seed,
        ))
        if cached:
            t03[label] = cached[0]

    out = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": ctx["task"].name, "model": ctx["hf_id"],
            "seed": args.seed, "max_steps": args.grpo_max_steps,
            "sampling": PURE_SAMPLING,
            "arms": {n: ARMS[n] for n in done},
            "judge_size": args.judge_size,
            "robustness_judge_size": args.robustness_judge_size,
            "t03_references": t03,
        },
        "means": {n: d["mean"] for n, d in evals.items()},
        "robust_means": {n: d.get("robust_mean") for n, d in evals.items()},
        "task_metrics": {n: d.get("metrics") for n, d in evals.items()},
        "comparisons": comparisons,
    }
    (workdir / "exploration_results.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    report = _render(out, done)
    (workdir / "exploration_report.md").write_text(report)
    print(report)


def _fmt_cmp(c: dict[str, Any]) -> str:
    return f"{c['mean']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]"


def _render(out: dict[str, Any], done: list[str]) -> str:
    meta, means, cmps = out["meta"], out["means"], out["comparisons"]
    t03 = meta["t03_references"]
    lines = [
        "# GRPO exploration study (production-RL sampling profile)",
        "",
        f"Task `{meta['task']}`, model `{meta['model']}`, seed {meta['seed']}, "
        f"{meta['max_steps']} steps, G=8, 64 completions/step; sampling "
        "top_p=1.0, min_p=0, repetition_penalty=1.0 on every arm; judge "
        f"`{meta['judge_size']}` (robustness `{meta['robustness_judge_size']}`). "
        "Single-seed screening — CIs are paired bootstrap over test sample IDs.",
        "",
    ]

    def section(title: str, start: str, control: str, control_label: str) -> None:
        arms = [n for n in done if ARMS[n]["start"] == start]
        if not arms:
            return
        lines.extend([
            f"## {title}",
            "",
            f"Controls: {control_label}",
            "",
            "| Arm | T | lr | β | Test | Δ vs control [95% CI] | Δ robust judge |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for n in arms:
            spec = ARMS[n]
            c = cmps.get(f"{n}_minus_{control}")
            cr = cmps.get(f"{n}_minus_{control}_robust")
            mean = means.get(n)
            lines.append(
                f"| {n} | {spec['temperature']} | {spec['lr']:g} | "
                f"{spec['beta']:g} | "
                + (f"{mean:.2f}" if mean is not None else "—") + " | "
                + (_fmt_cmp(c) if c else "—") + " | "
                + (f"{cr['mean']:+.2f}" if cr else "—") + " |"
            )
        lines.append("")

    section(
        "Metric 1 — baseline vs SFT vs GRPO (from raw model)",
        "base", "baseline",
        f"raw model {means.get('baseline', float('nan')):.2f}; "
        f"SFT reference {means.get('sft_reference', float('nan')):.2f} "
        f"({_fmt_cmp(cmps['sft_reference_minus_baseline'])} vs raw)"
        + (f"; shipped-sampling T=0.3 GRPO {t03['base_t03_ref']:.2f}"
           if "base_t03_ref" in t03 else ""),
    )
    section(
        "Metric 2 — SFT vs SFT+GRPO",
        "sft", "sft_reference",
        f"SFT winner {means.get('sft_reference', float('nan')):.2f}"
        + (f"; shipped-sampling T=0.3 SFT+GRPO {t03['sft_t03_ref']:.2f}"
           if "sft_t03_ref" in t03 else ""),
    )
    lines.extend([
        "Robustness column: Δ vs the same control under the second judge — "
        "a gain that flips sign there is reward hacking, not learning.",
        "",
        "Not varied here (follow-ups): group-count per step (production uses "
        "128–256 groups/step vs our 8), num_iterations, judge fidelity.",
    ])
    return "\n".join(lines) + "\n"


async def _main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    token = get_token()
    if not token:
        raise SystemExit("not authenticated; run lqh /login or set LQH_API_TOKEN")
    root = api_root()
    client = create_client(token, root + "/v1")
    os.environ["LQH_API_TOKEN"] = token
    os.environ["LQH_BASE_URL"] = root

    workdir = Path(args.workdir).expanduser()
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
    ctx = {
        "workdir": workdir, "task": task, "model_key": model_key,
        "hf_id": hf_id, "paths": paths, "scorer_rel": scorer_rel,
        "client": client, "prefix": _eval_prefix(task, model_key, args.seed),
    }
    if args.mode == "train":
        await _train(args, ctx)
    else:
        await _report(args, ctx)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
