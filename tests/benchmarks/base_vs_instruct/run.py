"""Base-vs-Instruct fine-tuning benchmark — orchestrator.

Runs, for each (task, model): baseline local eval -> SFT sweep -> eval best
SFT -> DPO sweep (on best SFT) -> eval best DPO, then writes a result table.

Compute is **local GPU**: training and inference run as local subprocesses
(`lqh.train.sweep`, `lqh.infer`). The judge is always an API call. Because a
standalone script has no TUI watcher, we set ``LQH_API_TOKEN`` + ``LQH_BASE_URL``
in the environment so the training subprocess self-scores inline
(`lqh.train.cloud_score.is_cloud_mode`) — this is mandatory for on-policy DPO,
which builds preference pairs from judge-scored rollouts every iteration.

Usage (smoke):
    uv run python -m tests.benchmarks.base_vs_instruct.run \\
        --tasks translation --models 350M-Instruct,350M-Base \\
        --train-size 100 --eval-size 20 --grid-size tiny

Usage (full):
    uv run python -m tests.benchmarks.base_vs_instruct.run \\
        --train-size 20000 --eval-size 400 --grid-size small

See ``README.md`` for the full flag list and cost/time notes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lqh.auth import api_root, get_token
from lqh.client import create_client
from lqh.engine import run_pipeline
from lqh.scoring import run_data_filter
from lqh.subprocess_manager import SubprocessManager

from .eval_local import _await_run, eval_local
from .report import ModelResult, write_report
from .tasks import Task, resolve_tasks

logger = logging.getLogger("bvi")


def _suppress_noisy_http_logs() -> None:
    """Keep long benchmark logs focused on benchmark progress.

    This suppresses library log records only; HTTP/API exceptions still raise
    normally and are handled by the caller.
    """
    for name in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)

# Friendly key -> HuggingFace id. The 350M instruct variant has no -Instruct
# suffix upstream (see BASE_VS_INSTRUCT.md).
#
# The LFM2.5-* rows are the newer generation; the LFM2-* rows are the previous
# generation, included so the benchmark also answers "how fine-tuneable is the
# older model vs the newer one?". LFM2 ships only instruct repos (no separately
# published -Base), so just the two instruct ids are added. NOTE: the two
# generations may have been saved with different transformers versions (the
# v4→v5 transition) — ``preflight.py`` verifies every id loads under the
# installed transformers before the run spends any GPU time.
MODELS: dict[str, str] = {
    "350M-Instruct": "LiquidAI/LFM2.5-350M",
    "350M-Base": "LiquidAI/LFM2.5-350M-Base",
    "1.2B-Instruct": "LiquidAI/LFM2.5-1.2B-Instruct",
    "1.2B-Base": "LiquidAI/LFM2.5-1.2B-Base",
    # Previous generation (instruct only) — finetuneability comparison.
    "LFM2-1.2B": "LiquidAI/LFM2-1.2B",
    "LFM2-350M": "LiquidAI/LFM2-350M",
}

# LoRA target modules — mirrors handler/handle_start_training so the sweep
# trains the same adapter shape the product uses.
_LORA = {
    "enabled": True,
    "r": 32,
    "alpha": 64,
    "dropout": 0.02,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj", "w1", "w2", "w3",
    ],
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="base_vs_instruct",
        description="Compare base vs instruct LFM2.5 variants as fine-tuning bases.",
    )
    p.add_argument(
        "--models", default="",
        help="comma list of model keys (default: all). Keys: "
        + ", ".join(MODELS),
    )
    p.add_argument(
        "--tasks", default="",
        help="comma list: translation, extraction, classification, "
        "messy_extraction, style_rewrite, voice_satisfaction (default: all).",
    )
    p.add_argument("--train-size", type=int, default=200, help="train samples per task")
    p.add_argument("--eval-size", type=int, default=40, help="eval samples per task")
    p.add_argument(
        "--grid-size", choices=["tiny", "small"], default="tiny",
        help="sweep grid size (tiny=3 configs smoke, small=6 configs).",
    )
    p.add_argument("--skip-dpo", action="store_true", help="run SFT only, no DPO stage")
    p.add_argument(
        "--dpo-train-size", type=int, default=1000,
        help="prompt count for the DPO stage (a slice of the train set). DPO "
        "regenerates rollouts on ALL its prompts every iteration, so this is "
        "kept small and decoupled from --train-size (SFT uses the full set). "
        "Capped at --train-size. Set to 0 to skip DPO and run the SFT "
        "comparison only.",
    )
    p.add_argument(
        "--judge-size", choices=["small", "medium", "large"], default="small",
        help="judge model size for eval scoring.",
    )
    p.add_argument(
        "--workdir", default="",
        help="scratch project dir (default ~/.lqh-bvi-<run-name>).",
    )
    p.add_argument("--run-name", default="", help="run name (default: timestamped).")
    p.add_argument(
        "--no-resume", action="store_true",
        help="recompute every stage even if outputs exist.",
    )
    p.add_argument(
        "--skip-preflight", action="store_true",
        help="skip the transformers v4/v5 model-compatibility preflight "
        "(by default each model's config + chat template are verified to load "
        "under the installed transformers before any GPU time is spent).",
    )
    p.add_argument("--datagen-concurrency", type=int, default=100)
    p.add_argument(
        "--no-filter", action="store_true",
        help="skip scorer-based quality filtering of generated datasets "
        "(use raw pipeline output as-is).",
    )
    p.add_argument(
        "--filter-threshold", type=float, default=7.0,
        help="keep generated samples scoring >= this (0-10) against the task "
        "scorer; lower scorers are dropped and regenerated. Default 7.0.",
    )
    p.add_argument(
        "--overgen-factor", type=float, default=1.6,
        help="over-generate by this factor before filtering, to absorb the "
        "expected drop rate. Default 1.6.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512, help="eval generation cap")
    p.add_argument(
        "--sweep-timeout", type=float, default=48 * 3600,
        help="per-sweep wall-clock timeout (seconds).",
    )
    p.add_argument(
        "--eval-timeout", type=float, default=3600,
        help="per local-eval inference timeout (seconds).",
    )
    args = p.parse_args(argv)
    if args.dpo_train_size < 0:
        p.error("--dpo-train-size must be >= 0")
    return args


def _resolve_models(spec: str) -> list[tuple[str, str]]:
    if not spec.strip():
        return list(MODELS.items())
    out: list[tuple[str, str]] = []
    for raw in spec.split(","):
        key = raw.strip()
        if not key:
            continue
        if key in MODELS:
            out.append((key, MODELS[key]))
        elif "/" in key:  # raw HF id
            out.append((key.split("/")[-1], key))
        else:
            raise SystemExit(
                f"unknown model {key!r}; choose from {', '.join(MODELS)} or pass a HF id"
            )
    return out


def _dataset_ready(data_parquet: Path, want_rows: int) -> bool:
    rows = _dataset_rows(data_parquet)
    return rows is not None and rows >= want_rows


def _dataset_rows(data_parquet: Path) -> int | None:
    if not data_parquet.exists():
        return None
    try:
        return pq.read_metadata(data_parquet).num_rows
    except Exception:
        return None


def _status_completed(run_dir: Path) -> bool:
    return SubprocessManager().get_status(run_dir).state == "completed"


def _sweep_timing(run_dir: Path) -> dict:
    """Read per-config and total training duration from sweep_summary.json."""
    summary_path = run_dir / "sweep_summary.json"
    if not summary_path.exists():
        return {"sweep_s": None, "training_runs": []}
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"sweep_s": None, "training_runs": []}

    runs = []
    for row in summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        elapsed = row.get("elapsed_s")
        try:
            elapsed_s = float(elapsed) if elapsed is not None else None
        except (TypeError, ValueError):
            elapsed_s = None
        runs.append({
            "config_id": row.get("config_id"),
            "elapsed_s": elapsed_s,
            "rc": row.get("rc"),
            "primary": row.get("primary"),
            "collapsed": row.get("collapsed", False),
            "winner": (
                isinstance(summary.get("winner"), dict)
                and row.get("config_id") == summary["winner"].get("config_id")
            ),
        })

    sweep_s = sum(
        r["elapsed_s"] for r in runs
        if isinstance(r.get("elapsed_s"), (int, float))
    )
    return {
        "sweep_s": sweep_s if runs else None,
        "training_runs": runs,
    }


def _ensure_dpo_dataset(workdir: Path, task: Task, n: int, *, resume: bool) -> str:
    """Materialize the DPO prompt set as the first *n* rows of the train set.

    On-policy DPO regenerates rollouts on every prompt each iteration, so it
    must run on a bounded slice — not the full (possibly 20k) SFT train set.
    Slicing the existing train parquet avoids any extra generation. Returns the
    workdir-relative path to the sliced data.parquet."""
    import pyarrow.parquet as _pq

    src = workdir / f"datasets/{task.name}_train/data.parquet"
    dst_dir = workdir / f"datasets/{task.name}_dpo"
    dst = dst_dir / "data.parquet"
    rel = f"datasets/{task.name}_dpo/data.parquet"
    if resume and _dataset_ready(dst, n):
        return rel
    table = _pq.read_table(src)
    dst_dir.mkdir(parents=True, exist_ok=True)
    _pq.write_table(table.slice(0, min(n, table.num_rows)), dst)
    logger.info("dpo dataset %s: sliced %d rows from train", rel, min(n, table.num_rows))
    return rel


# SFT/DPO batch shape. Start from a large real micro-batch so each optimizer
# step does substantial GPU work; ``training.auto_batch`` probes down to the
# largest safe value if 256 does not fit.
_SFT_PER_DEVICE_BATCH = 256
_SFT_GRAD_ACCUM = 1
_DPO_PER_DEVICE_BATCH = 256
_DPO_GRAD_ACCUM = 1

# Smallest train set for which on-policy DPO can compute its sweep proxy.
# Measured pair-yield is ~0.13 for a strong SFT model; the proxy needs ≥10
# held-out preference pairs in iter_000 (split_train_eval min_eval=10) while
# leaving enough to train on. Below this we auto-skip DPO (see
# _run_model_on_task) rather than fail. DPO's real value is at much larger
# train sizes anyway.
_DPO_MIN_TRAIN_SIZE = 400
# Conservative on-policy preference-pair yield per train prompt per iter,
# observed across model strengths (strong models yield the least). Used to size
# the held-out eval split so iter_000 clears the floor of 10.
_DPO_PAIR_YIELD = 0.13


def _sft_step_schedule(train_size: int, *, min_epochs: int = 2,
                       target_evals: int = 4, max_eval_steps: int = 200) -> dict:
    """Pick eval/save/logging cadence that actually fires on this dataset.

    ``sft.py`` evals/saves on a STEP schedule (default 50) and uses
    ``load_best_model_at_end`` keyed on ``eval_loss``. On small datasets the
    default 50 is larger than the total optimizer-step count, so eval never
    runs → no ``eval_loss`` → the sweep proxy is empty and every config is
    marked "failed" (and the empty best-checkpoint can crash the child). We
    size ``eval_steps`` so eval fires ~``target_evals`` times even for the
    shortest config in the grid (``min_epochs`` epochs).
    """
    import math

    eff_batch = _SFT_PER_DEVICE_BATCH * _SFT_GRAD_ACCUM
    steps_per_epoch = max(1, math.ceil(train_size / eff_batch))
    min_total_steps = steps_per_epoch * min_epochs
    eval_steps = max(1, min(max_eval_steps, min_total_steps // target_evals or 1))
    return {
        "per_device_batch_size": _SFT_PER_DEVICE_BATCH,
        "gradient_accumulation_steps": _SFT_GRAD_ACCUM,
        "eval_steps": eval_steps,
        "save_steps": eval_steps,  # must match eval cadence for load_best
        "logging_steps": max(1, min(eval_steps, 10)),
    }


def _base_config(
    *,
    run_type: str,
    base_model: str,
    dataset_rel: str,
    eval_rel: str,
    scorer_rel: str,
    train_size: int,
) -> dict:
    """Build the SFT/DPO base config the sweep wraps. The grid overrides
    learning_rate / num_epochs (SFT) or learning_rate / dpo_beta (DPO)."""
    training: dict = {
        "learning_rate": 2e-5 if run_type == "sft" else 5e-6,
        "max_seq_length": 2048,
    }
    cfg: dict = {
        "type": run_type,
        "base_model": base_model,
        "dataset": dataset_rel,
        "eval_dataset": eval_rel,
        "eval_on_checkpoints": True,
        "scorer": scorer_rel,
        "training": training,
        "lora": dict(_LORA),
        "manifest": ["base_model", "dataset", "eval_dataset", "scorer"],
    }
    if run_type == "sft":
        training["num_epochs"] = 3
        # Cadence sized to the dataset so eval (the sweep's selection signal)
        # actually fires. The grid's smallest config is 2 epochs.
        training.update(_sft_step_schedule(train_size, min_epochs=2))
        training["effective_batch_size"] = _SFT_PER_DEVICE_BATCH * _SFT_GRAD_ACCUM
        training["auto_batch"] = True
    else:  # on_policy_dpo
        cfg["num_iterations"] = 5
        cfg["dpo_beta"] = 0.1
        cfg["golden_source"] = "dataset"
        training.update({
            "per_device_batch_size": _DPO_PER_DEVICE_BATCH,
            "gradient_accumulation_steps": _DPO_GRAD_ACCUM,
            "effective_batch_size": _DPO_PER_DEVICE_BATCH * _DPO_GRAD_ACCUM,
            "auto_batch": True,
        })
        # The DPO sweep selects on `eval_ce_chosen_mean`, computed over a
        # held-out split of the per-iteration PREFERENCE pairs — and the sweep
        # reads ITER_000 specifically. split_train_eval has a hard floor
        # (min_eval=10): if round(n_pairs * ratio) < 10 the eval split is EMPTY
        # → no chosen-CE summary → proxy null → no winner. On-policy yield is as
        # low as ~0.13 pairs per train prompt for a strong model (its greedy
        # rollout usually matches the gold), so a small train set simply can't
        # produce 10 held-out pairs at any ratio — that case is auto-skipped in
        # _run_model_on_task (train_size < _DPO_MIN_TRAIN_SIZE). Here we size the
        # ratio off the conservative yield, aiming for 14 eval pairs (margin
        # over the floor) so iter_000 clears it even at the low end of variance.
        est_prefs = max(1, int(train_size * _DPO_PAIR_YIELD))
        cfg["training"]["eval_split_ratio"] = round(min(0.5, max(0.1, 14.0 / est_prefs)), 3)
    return cfg


async def _run_sweep(
    *,
    workdir: Path,
    run_name: str,
    base_config: dict,
    grid_size: str,
    timeout: float,
    resume: bool,
) -> tuple[Path, dict]:
    """Launch a local sweep subprocess, wait for completion, return winner dir."""
    run_dir = workdir / "runs" / run_name
    # The sweep materializes its winner under the source dir name: ``model``
    # for full/merged checkpoints, ``model-lora`` for adapter-only LoRA
    # winners (see sweep.py:_materialize_best_model / _winner_model_dir).
    # Accept either so adapter sweeps aren't misreported as failed.
    def _winner_model_dir() -> Path | None:
        for name in ("model", "model-lora"):
            p = run_dir / name
            if p.exists():
                return p
        return None

    existing = _winner_model_dir()
    if resume and existing is not None and _status_completed(run_dir):
        logger.info("sweep %s: reusing completed winner", run_name)
        return existing, _sweep_timing(run_dir)

    launch = {
        "type": "sweep",
        "base_config": base_config,
        "grid_size": grid_size,
        # We run our own eval_local on the winner for the reported number, so
        # skip the sweep's inline eval-of-best. Winner selection still uses the
        # validated in-training proxy (eval_loss / eval_ce_chosen_mean).
        "eval_best": False,
    }
    SubprocessManager().start(run_dir, launch, module="lqh.train.sweep", project_dir=workdir)
    await _await_run(SubprocessManager(), run_dir, timeout=timeout, label=f"sweep:{run_name}")
    model_dir = _winner_model_dir()
    if model_dir is None:
        raise RuntimeError(
            f"sweep:{run_name} completed but no winner model at "
            f"{run_dir}/(model|model-lora) "
            f"(all configs may have failed/collapsed; see {run_dir}/stderr.log)"
        )
    return model_dir, _sweep_timing(run_dir)


async def _generate_filtered_split(
    *, script_path: Path, scorer_path: Path, target: int, out_dir: Path,
    client, concurrency: int, threshold: float, overgen_factor: float,
    label: str, judge_size: str = "small", max_rounds: int = 4,
) -> None:
    """Generate a split, score every sample's gold against the task scorer, and
    keep only samples at/above *threshold*, regenerating until *target* clear.

    Writes ``out_dir/data.parquet`` with exactly *target* high-quality rows.
    Over-generates by *overgen_factor* up front, then tops up using the
    observed keep-rate. Raises if it cannot reach *target* within *max_rounds*.
    """
    tmp_root = out_dir.parent / f"_filt_tmp_{out_dir.name}"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    kept_tables: list[pa.Table] = []
    kept = 0
    total_generated = 0
    keep_rate = 1.0 / max(overgen_factor, 1.0)  # initial drop-rate estimate
    for round_i in range(1, max_rounds + 1):
        if kept >= target:
            break
        need = target - kept
        gen_n = max(need, math.ceil(need / max(keep_rate, 0.05)))
        # Cap per-round blow-up: if the keep-rate is pathologically low, prefer
        # failing across rounds with a clear message over one giant round.
        gen_n = min(gen_n, target * 4)
        raw_dir = tmp_root / f"raw_{round_i}"
        filt_dir = tmp_root / f"filt_{round_i}"
        logger.info(
            "datagen %s: round %d — generating %d (need %d more) ...",
            label, round_i, gen_n, need,
        )
        gen = await run_pipeline(
            script_path=script_path, num_samples=gen_n, output_dir=raw_dir,
            client=client, concurrency=concurrency,
        )
        fr = await run_data_filter(
            input_path=raw_dir / "data.parquet", scorer_path=scorer_path,
            output_dataset_dir=filt_dir, client=client,
            threshold=threshold, model_size=judge_size, concurrency=concurrency,
        )
        kept_tbl = pq.read_table(filt_dir / "data.parquet")
        total_generated += gen.succeeded
        if kept_tbl.num_rows:
            kept_tables.append(kept_tbl)
            kept += kept_tbl.num_rows
        observed = (kept_tbl.num_rows / gen.succeeded) if gen.succeeded else 0.0
        if observed > 0:
            keep_rate = max(0.05, observed)
        logger.info(
            "datagen %s: round %d kept %d/%d (rate=%.2f, mean=%.2f); total %d/%d",
            label, round_i, kept_tbl.num_rows, gen.succeeded, observed,
            fr.mean_score, kept, target,
        )

    if kept < target:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise RuntimeError(
            f"datagen {label}: only {kept}/{target} samples cleared the "
            f"score>={threshold} filter after {max_rounds} rounds. Lower "
            "--filter-threshold or raise --overgen-factor."
        )

    full = pa.concat_tables(kept_tables).slice(0, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(full, out_dir / "data.parquet")
    shutil.rmtree(tmp_root, ignore_errors=True)
    logger.info(
        "datagen %s: wrote %d filtered rows (generated %d, kept %d, drop %d)",
        label, target, total_generated, kept, total_generated - kept,
    )


async def _ensure_datasets(
    *, workdir: Path, task: Task, train_size: int, eval_size: int,
    client, concurrency: int, resume: bool,
    filter_threshold: float | None, overgen_factor: float,
    judge_size: str = "small",
) -> tuple[str, str, str]:
    """Generate train + eval datasets for a task (once, shared by all models).

    When *filter_threshold* is not None, each split is over-generated and then
    scorer-filtered (keep score >= threshold) down to the exact target count;
    otherwise the raw pipeline output is used as-is.

    Returns (dataset_rel, eval_rel, scorer_rel) — paths relative to workdir."""
    scorer_rel = f"scorers/{task.name}.md"
    scorer_path = workdir / scorer_rel
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(task.scorer_md)

    specs = [
        (f"datasets/{task.name}_train", train_size),
        (f"datasets/{task.name}_eval", eval_size),
    ]
    for rel, n in specs:
        out_dir = workdir / rel
        data_parquet = out_dir / "data.parquet"
        if resume and _dataset_ready(data_parquet, n):
            logger.info("datagen %s: reusing %d rows", rel, n)
            continue

        if filter_threshold is None:
            logger.info("datagen %s: generating %d samples (no filter) ...", rel, n)
            res = await run_pipeline(
                script_path=task.pipeline_path,
                num_samples=n,
                output_dir=out_dir,
                client=client,
                concurrency=concurrency,
            )
            logger.info(
                "datagen %s: %d ok / %d failed", rel, res.succeeded, res.failed,
            )
        else:
            await _generate_filtered_split(
                script_path=task.pipeline_path,
                scorer_path=scorer_path,
                target=n,
                out_dir=out_dir,
                client=client,
                concurrency=concurrency,
                threshold=filter_threshold,
                overgen_factor=overgen_factor,
                judge_size=judge_size,
                label=rel,
            )

        rows = _dataset_rows(data_parquet) or 0
        if rows < n:
            raise RuntimeError(
                f"datagen produced only {rows}/{n} rows for {rel}. Refusing to "
                "run a benchmark on a partial dataset."
            )

    return (
        f"datasets/{task.name}_train/data.parquet",
        f"datasets/{task.name}_eval/data.parquet",
        scorer_rel,
    )


async def _run_model_on_task(
    *, workdir: Path, task: Task, model_key: str, hf_id: str,
    dataset_rel: str, eval_rel: str, scorer_rel: str, args, client,
) -> ModelResult:
    resume = not args.no_resume
    eval_parquet = workdir / eval_rel
    scorer_path = workdir / scorer_rel
    tag = f"{task.name}__{model_key}"
    result = ModelResult(task=task.name, model=model_key)

    eval_kwargs = dict(
        workdir=workdir, eval_parquet=eval_parquet, scorer_path=scorer_path,
        client=client, judge_size=args.judge_size,
        max_new_tokens=args.max_new_tokens, infer_timeout=args.eval_timeout,
        resume=resume,
    )

    # 1) Baseline (untrained model)
    try:
        r = await eval_local(run_name=f"{tag}__baseline", model_path=hf_id, **eval_kwargs)
        result.baseline, result.n_eval = r.mean, r.n_scored
    except Exception as exc:
        logger.exception("baseline failed for %s", tag)
        result.notes.append(f"baseline failed: {exc}")
        return result  # can't meaningfully continue without a baseline anchor

    # 2) SFT sweep + eval of winner
    try:
        sft_model, sft_timing = await _run_sweep(
            workdir=workdir, run_name=f"{tag}__sft",
            base_config=_base_config(
                run_type="sft", base_model=hf_id,
                dataset_rel=dataset_rel, eval_rel=eval_rel, scorer_rel=scorer_rel,
                train_size=args.train_size),
            grid_size=args.grid_size, timeout=args.sweep_timeout, resume=resume,
        )
        result.sft_sweep_s = sft_timing["sweep_s"]
        result.sft_training_runs = sft_timing["training_runs"]
        r = await eval_local(
            run_name=f"{tag}__sft_eval", model_path=str(sft_model.resolve()), **eval_kwargs)
        result.sft = r.mean
    except Exception as exc:
        sft_timing = _sweep_timing(workdir / "runs" / f"{tag}__sft")
        result.sft_sweep_s = sft_timing["sweep_s"]
        result.sft_training_runs = sft_timing["training_runs"]
        logger.exception("SFT stage failed for %s", tag)
        result.notes.append(f"sft failed: {exc}")
        return result

    # 3) DPO sweep (on best SFT) + eval of winner.
    # Auto-skip below _DPO_MIN_TRAIN_SIZE: on-policy DPO yields only ~0.13
    # preference pairs per train prompt for a well-trained model (a near-perfect
    # rollout rarely disagrees with the gold), so a small train set can't
    # produce the ≥10 held-out pairs the sweep proxy (eval_ce_chosen_mean)
    # needs in iter_000. Running it anyway just wastes GPU and fails. DPO is
    # meaningful at scale, where pairs are plentiful.
    if args.skip_dpo:
        return result
    if args.dpo_train_size == 0:
        result.notes.append("dpo skipped: dpo_train_size=0 (SFT-only comparison requested)")
        return result
    dpo_size = min(args.dpo_train_size, args.train_size)
    if dpo_size < _DPO_MIN_TRAIN_SIZE:
        result.notes.append(
            f"dpo skipped: dpo_train_size={dpo_size} < {_DPO_MIN_TRAIN_SIZE} "
            f"(on-policy preference yield too low to compute the sweep proxy; "
            f"raise --dpo-train-size/--train-size or pass --skip-dpo)"
        )
        return result
    try:
        dpo_dataset_rel = _ensure_dpo_dataset(workdir, task, dpo_size, resume=resume)
        dpo_model, dpo_timing = await _run_sweep(
            workdir=workdir, run_name=f"{tag}__dpo",
            # Use the SFT winner's actual materialized path — it may be a
            # merged ``model`` dir OR an adapter-only ``model-lora`` dir
            # (SFT trains LoRA with merge=False). Hardcoding ``model`` here
            # made every DPO config crash on load when the winner was an
            # adapter. load_for_training(merge_before_attach=True) consumes
            # either layout (dpo.py).
            base_config=_base_config(
                run_type="on_policy_dpo", base_model=str(sft_model.resolve()),
                dataset_rel=dpo_dataset_rel, eval_rel=eval_rel, scorer_rel=scorer_rel,
                train_size=dpo_size),
            grid_size=args.grid_size, timeout=args.sweep_timeout, resume=resume,
        )
        result.dpo_sweep_s = dpo_timing["sweep_s"]
        result.dpo_training_runs = dpo_timing["training_runs"]
        r = await eval_local(
            run_name=f"{tag}__dpo_eval", model_path=str(dpo_model.resolve()), **eval_kwargs)
        result.dpo = r.mean
    except Exception as exc:
        dpo_timing = _sweep_timing(workdir / "runs" / f"{tag}__dpo")
        result.dpo_sweep_s = dpo_timing["sweep_s"]
        result.dpo_training_runs = dpo_timing["training_runs"]
        logger.exception("DPO stage failed for %s", tag)
        result.notes.append(f"dpo failed: {exc}")

    return result


async def _main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _suppress_noisy_http_logs()

    token = get_token()
    if not token:
        raise SystemExit("not authenticated — run `lqh` and /login, or set LQH_API_TOKEN.")
    root = api_root()
    client = create_client(token, root + "/v1")

    # The contract that makes the training subprocess self-score inline (no TUI
    # watcher in a standalone run). Mandatory for on-policy DPO.
    os.environ["LQH_API_TOKEN"] = token
    os.environ["LQH_BASE_URL"] = root

    run_name = args.run_name or f"bvi-{time.strftime('%Y%m%d-%H%M%S')}"
    workdir = Path(args.workdir).expanduser() if args.workdir \
        else Path(f"~/.lqh-bvi/{run_name}").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    models = _resolve_models(args.models)
    tasks = resolve_tasks([t for t in args.tasks.split(",") if t.strip()] or None)
    resume = not args.no_resume

    # Transformers v4/v5 compatibility preflight. The benchmark mixes model
    # generations (older LFM2 + newer LFM2.5) that may have been saved with
    # different transformers versions; verify every id actually loads under the
    # installed transformers (config + chat template, no weights) BEFORE
    # spending datagen/GPU time, so an unsupported architecture surfaces here
    # with an actionable message instead of crashing a sweep child hours in.
    if not args.skip_preflight:
        from .preflight import run_preflight

        compat = run_preflight(models)
        bad = [c for c in compat if not c.ok]
        if bad:
            detail = "\n".join(
                f"  - {c.key} ({c.hf_id}): {'; '.join(c.notes) or c.error or 'incompatible'}"
                for c in bad
            )
            raise SystemExit(
                "model compatibility preflight failed for "
                f"{len(bad)} model(s) under the installed transformers:\n{detail}\n"
                "Upgrade transformers (target >=5,<6), drop the offending "
                "model(s) from --models, or pass --skip-preflight to override."
            )

    meta = {
        "run_name": run_name, "workdir": str(workdir),
        "train_size": args.train_size, "eval_size": args.eval_size,
        "grid_size": args.grid_size, "judge_size": args.judge_size,
        "skip_dpo": args.skip_dpo or args.dpo_train_size == 0, "compute": "local-gpu",
        "dpo_train_size": min(args.dpo_train_size, args.train_size),
        "filter_threshold": (None if args.no_filter else args.filter_threshold),
        "overgen_factor": args.overgen_factor,
        "models": [k for k, _ in models], "tasks": [t.name for t in tasks],
    }
    logger.info("workdir: %s", workdir)
    logger.info("models: %s", ", ".join(k for k, _ in models))
    logger.info("tasks:  %s", ", ".join(t.name for t in tasks))
    dpo_size = min(args.dpo_train_size, args.train_size)
    if not args.skip_dpo and args.dpo_train_size > 0 and dpo_size < _DPO_MIN_TRAIN_SIZE:
        logger.warning(
            "DPO will be SKIPPED: dpo_train_size=%d < %d. On-policy preference "
            "yield is too low at this scale to compute the DPO sweep proxy "
            "(a strong model produces only ~%.0f%% pairs/prompt). Raise "
            "--dpo-train-size to >= %d for DPO, or pass --skip-dpo to silence "
            "this. SFT results are unaffected.",
            dpo_size, _DPO_MIN_TRAIN_SIZE, _DPO_PAIR_YIELD * 100,
            _DPO_MIN_TRAIN_SIZE,
        )
    if not args.skip_dpo and args.dpo_train_size == 0:
        logger.info("DPO will be SKIPPED: dpo_train_size=0 (SFT-only comparison requested).")

    results: list[ModelResult] = []
    for task in tasks:
        dataset_rel, eval_rel, scorer_rel = await _ensure_datasets(
            workdir=workdir, task=task, train_size=args.train_size,
            eval_size=args.eval_size, client=client,
            concurrency=args.datagen_concurrency, resume=resume,
            filter_threshold=(None if args.no_filter else args.filter_threshold),
            overgen_factor=args.overgen_factor,
            judge_size=args.judge_size,
        )
        for model_key, hf_id in models:
            logger.info("=== task=%s model=%s (%s) ===", task.name, model_key, hf_id)
            res = await _run_model_on_task(
                workdir=workdir, task=task, model_key=model_key, hf_id=hf_id,
                dataset_rel=dataset_rel, eval_rel=eval_rel, scorer_rel=scorer_rel,
                args=args, client=client,
            )
            results.append(res)
            # Write the report incrementally so a long run is inspectable mid-flight.
            jpath, mpath = write_report(workdir / "report", meta, results)

    logger.info("done. report: %s", mpath)
    print("\n" + mpath.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
