"""Measure DPO's incremental value on fresh data with a matched SFT control."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lqh.auth import api_root, get_token
from lqh.client import create_client
from lqh.scoring import is_scoring_error

from tests.benchmarks.base_vs_instruct.eval_local import eval_local
from tests.benchmarks.base_vs_instruct.run import (
    MODELS,
    _base_config,
    _dataset_ready,
    _generate_filtered_split,
    _run_sweep,
)
from tests.benchmarks.base_vs_instruct.tasks import resolve_tasks

from .stats import paired_bootstrap
from .voice_metrics import voice_metrics

logger = logging.getLogger("dpo_value")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh-data DPO vs continued-SFT value benchmark",
    )
    parser.add_argument("--task", default="voice_satisfaction")
    parser.add_argument("--model", default="1.2B-Instruct")
    parser.add_argument("--sft-train-size", type=int, default=10_000)
    parser.add_argument("--dpo-train-size", type=int, default=2_000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seeds", default="17,29,41")
    parser.add_argument("--grid-size", choices=["tiny", "small"], default="small")
    parser.add_argument("--judge-size", choices=["small", "medium", "large"], default="small")
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
    args = parser.parse_args(argv)
    for name in ("sft_train_size", "dpo_train_size", "validation_size", "test_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        args.seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    except ValueError as exc:
        parser.error(f"--seeds must be comma-separated integers: {exc}")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    return args


def _resolve_model(value: str) -> tuple[str, str]:
    if value in MODELS:
        return value, MODELS[value]
    if "/" in value:
        return value.rsplit("/", 1)[-1], value
    raise SystemExit(f"unknown model {value!r}; use one of {', '.join(MODELS)} or a HF id")


async def _ensure_splits(
    *,
    workdir: Path,
    task: Any,
    sizes: dict[str, int],
    client: Any,
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
        if not (not args.no_resume and _dataset_ready(workdir / rel, size)):
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
    _assert_disjoint(workdir, paths)
    return paths, scorer_rel


def _assert_disjoint(workdir: Path, paths: dict[str, str]) -> None:
    """Fail rather than silently leak exact examples across benchmark splits."""
    seen: dict[str, str] = {}
    for split, rel in paths.items():
        table = pq.read_table(workdir / rel, columns=["messages"])
        for value in table["messages"].to_pylist():
            messages = json.loads(value) if isinstance(value, str) else value
            prompt = (
                messages[:-1]
                if messages and messages[-1].get("role") == "assistant"
                else messages
            )
            key = json.dumps(prompt, sort_keys=True)
            prior = seen.setdefault(key, split)
            if prior != split:
                raise RuntimeError(
                    f"dataset leakage: an identical sample appears in {prior} and {split}"
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
    treatment: dict[int, float],
    control: dict[int, float],
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    value = paired_bootstrap(
        treatment,
        control,
        samples=args.bootstrap_samples,
        seed=seed,
    )
    return asdict(value)


async def _evaluate(
    *,
    workdir: Path,
    name: str,
    model: Path,
    paths: dict[str, str],
    scorer_rel: str,
    client: Any,
    args: argparse.Namespace,
) -> tuple[float | None, dict[int, float], dict[str, float | int]]:
    result = await eval_local(
        workdir=workdir,
        run_name=name,
        model_path=str(model.resolve()),
        eval_parquet=workdir / paths["test"],
        scorer_path=workdir / scorer_rel,
        client=client,
        judge_size=args.judge_size,
        max_new_tokens=args.max_new_tokens,
        infer_timeout=args.eval_timeout,
        resume=not args.no_resume,
    )
    task_metrics = (
        voice_metrics(result.predictions_path, workdir / paths["test"])
        if args.task == "voice_satisfaction" else {}
    )
    return result.mean, _score_vector(result.scores_dir), task_metrics


async def _run_seed(
    *,
    seed: int,
    workdir: Path,
    task: Any,
    model_key: str,
    hf_id: str,
    paths: dict[str, str],
    scorer_rel: str,
    client: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prefix = f"{task.name}__{model_key}__seed{seed}"
    common = dict(
        workdir=workdir,
        grid_size=args.grid_size,
        timeout=args.sweep_timeout,
        resume=not args.no_resume,
    )

    sft_config = _base_config(
        run_type="sft",
        base_model=hf_id,
        dataset_rel=paths["sft_train"],
        eval_rel=paths["validation"],
        scorer_rel=scorer_rel,
        train_size=args.sft_train_size,
    )
    sft_config["training"]["seed"] = seed
    sft_model, _ = await _run_sweep(
        run_name=f"{prefix}__sft", base_config=sft_config, **common,
    )
    sft_mean, sft_scores, sft_metrics = await _evaluate(
        workdir=workdir,
        name=f"{prefix}__sft_test",
        model=sft_model,
        paths=paths,
        scorer_rel=scorer_rel,
        client=client,
        args=args,
    )

    continued_config = _base_config(
        run_type="sft",
        base_model=str(sft_model.resolve()),
        dataset_rel=paths["dpo_train"],
        eval_rel=paths["validation"],
        scorer_rel=scorer_rel,
        train_size=args.dpo_train_size,
    )
    continued_config["training"]["seed"] = seed
    continued_model, _ = await _run_sweep(
        run_name=f"{prefix}__continued_sft",
        base_config=continued_config,
        **common,
    )
    continued_mean, continued_scores, continued_metrics = await _evaluate(
        workdir=workdir,
        name=f"{prefix}__continued_sft_test",
        model=continued_model,
        paths=paths,
        scorer_rel=scorer_rel,
        client=client,
        args=args,
    )

    dpo_config = _base_config(
        run_type="on_policy_dpo",
        base_model=str(sft_model.resolve()),
        dataset_rel=paths["dpo_train"],
        eval_rel=paths["test"],
        scorer_rel=scorer_rel,
        train_size=args.dpo_train_size,
    )
    # The fixed validation set selects configs and iterations. Remove final
    # test from every sweep child; the winner alone is evaluated below.
    dpo_config.pop("eval_dataset", None)
    dpo_config["manifest"] = [
        item for item in dpo_config["manifest"] if item != "eval_dataset"
    ]
    dpo_config["held_out_eval_dataset"] = paths["validation"]
    if "held_out_eval_dataset" not in dpo_config["manifest"]:
        dpo_config["manifest"].append("held_out_eval_dataset")
    dpo_config["preference_judge_size"] = args.judge_size
    dpo_config["selection"]["min_pairs_per_iter"] = 50
    dpo_config["training"].update({
        "seed": seed,
        "data_seed": seed,
        "dpo_min_optimizer_steps": 50,
    })
    dpo_model, _ = await _run_sweep(
        run_name=f"{prefix}__dpo", base_config=dpo_config, **common,
    )
    dpo_mean, dpo_scores, dpo_metrics = await _evaluate(
        workdir=workdir,
        name=f"{prefix}__dpo_test",
        model=dpo_model,
        paths=paths,
        scorer_rel=scorer_rel,
        client=client,
        args=args,
    )

    comparisons = {
        "dpo_minus_sft": _comparison(dpo_scores, sft_scores, args, seed=seed),
        "continued_sft_minus_sft": _comparison(
            continued_scores, sft_scores, args, seed=seed + 1,
        ),
        "dpo_minus_continued_sft": _comparison(
            dpo_scores, continued_scores, args, seed=seed + 2,
        ),
    }
    return {
        "seed": seed,
        "means": {
            "sft": sft_mean,
            "continued_sft": continued_mean,
            "dpo": dpo_mean,
        },
        "task_metrics": {
            "sft": sft_metrics,
            "continued_sft": continued_metrics,
            "dpo": dpo_metrics,
        },
        "comparisons": comparisons,
    }


def _render_report(meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Fresh-data DPO value benchmark",
        "",
        f"Task: `{meta['task']}`; model: `{meta['model']}`; seeds: `{meta['seeds']}`.",
        "",
        "All values are final-test judge scores. Confidence intervals are paired "
        "bootstrap 95% intervals over identical test sample IDs.",
        "",
        "| Seed | SFT | Continued SFT | DPO | DPO-SFT [95% CI] | DPO-continued [95% CI] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        means = row["means"]
        ds = row["comparisons"]["dpo_minus_sft"]
        dc = row["comparisons"]["dpo_minus_continued_sft"]
        lines.append(
            f"| {row['seed']} | {means['sft']:.2f} | {means['continued_sft']:.2f} "
            f"| {means['dpo']:.2f} | {ds['mean']:+.2f} "
            f"[{ds['ci_low']:+.2f}, {ds['ci_high']:+.2f}] | {dc['mean']:+.2f} "
            f"[{dc['ci_low']:+.2f}, {dc['ci_high']:+.2f}] |"
        )
    gains = [row["comparisons"]["dpo_minus_sft"]["mean"] for row in rows]
    demonstrated = [
        row["comparisons"]["dpo_minus_sft"]["mean"] >= 0.3
        and row["comparisons"]["dpo_minus_sft"]["ci_low"] > 0
        for row in rows
    ]
    lines.extend([
        "",
        f"Mean DPO-SFT delta across training seeds: `{sum(gains) / len(gains):+.2f}`.",
        "",
        "Criterion: a seed demonstrates DPO gain only when DPO-SFT is at least "
        "+0.3 and its paired interval excludes zero. "
        f"Result: `{sum(demonstrated)}/{len(demonstrated)}` seeds pass.",
        "",
        "The continued-SFT column is the extra-data control: it starts from the "
        "same SFT checkpoint and trains on the same fresh chosen-response pool as DPO.",
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
            for stage in ("sft", "continued_sft", "dpo"):
                metrics = row["task_metrics"][stage]
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
    os.environ["LQH_API_TOKEN"] = token
    os.environ["LQH_BASE_URL"] = root

    run_name = args.run_name or f"dpo-value-{time.strftime('%Y%m%d-%H%M%S')}"
    workdir = Path(args.workdir).expanduser() if args.workdir else Path(
        f"~/.lqh-dpo-value/{run_name}"
    ).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    task = resolve_tasks([args.task])[0]
    model_key, hf_id = _resolve_model(args.model)
    sizes = {
        "sft_train": args.sft_train_size,
        "dpo_train": args.dpo_train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    paths, scorer_rel = await _ensure_splits(
        workdir=workdir,
        task=task,
        sizes=sizes,
        client=client,
        args=args,
    )

    rows = []
    for seed in args.seeds:
        logger.info("starting seed %d", seed)
        rows.append(await _run_seed(
            seed=seed,
            workdir=workdir,
            task=task,
            model_key=model_key,
            hf_id=hf_id,
            paths=paths,
            scorer_rel=scorer_rel,
            client=client,
            args=args,
        ))
        meta = {
            "run_name": run_name,
            "task": task.name,
            "model": hf_id,
            "seeds": args.seeds,
            "sizes": sizes,
            "grid_size": args.grid_size,
            "judge_size": args.judge_size,
            "preference_generation": "greedy",
            "selection": {
                "top_quantile": 1.0,
                "min_gap": 1.0,
                "min_pairs_per_iter": 50,
            },
            "dpo_effective_batch": 16,
        }
        (workdir / "results.json").write_text(
            json.dumps({"meta": meta, "seeds": rows}, indent=2) + "\n"
        )
        (workdir / "report.md").write_text(_render_report(meta, rows))
    logger.info("report: %s", workdir / "report.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
