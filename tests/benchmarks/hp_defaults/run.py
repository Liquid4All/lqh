"""Default-hyperparameter calibration study — orchestrator.

Answers: **what learning rate and epoch count should LQH ship as its SFT
defaults, and does one setting work everywhere?**

For every (task × dataset size × model) cell it trains the whole HP grid and
judge-scores every config, then reports the config with the lowest mean regret
against each cell's oracle — plus, for each dimension, whether a per-level
default would beat the global one by more than the measured noise.

This exists because the defaults became load-bearing: SFT no longer sweeps by
default, so the first run after a dataset is ready IS the defaults. See
``README.md`` for the staged protocol and cost.

Smoke (local GPU, no cloud spend):

    uv run python -m tests.benchmarks.hp_defaults.run --compute local \\
        --tasks translation --models 350M-Instruct --sizes 100 \\
        --grid-points lr=2e-5:e1,lr=1e-4:e1 --eval-size 20 --yes

Stage A (screen the full grid on the anchor cells):

    uv run python -m tests.benchmarks.hp_defaults.run --anchors-only --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pyarrow.parquet as pq

from lqh.auth import api_root, get_token
from lqh.client import create_client

from ..base_vs_instruct.tasks import resolve_tasks
from ..shared.datagen import ensure_split
from . import runner
from .analyze import Observation
from .cells import Cell, MODELS, resolve_cells
from .grid import GridPoint, chunk_points, parse_points, replicate_grid, study_grid
from .report import write_report

logger = logging.getLogger("hp_defaults")

# One resubmit for a chunk lost to infrastructure. Two identical losses are
# not bad luck, and an unbounded retry on a systematically broken cell would
# spend the study's budget on a config that can never report.
_MAX_CHUNK_ATTEMPTS = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hp_defaults",
        description="Calibrate LQH's default SFT hyperparameters.",
    )
    p.add_argument("--tasks", default="", help="comma list (default: all four).")
    p.add_argument("--models", default="", help=f"comma list of {', '.join(MODELS)}.")
    p.add_argument("--sizes", default="", help="comma list of train sizes.")
    p.add_argument(
        "--anchors-only", action="store_true",
        help="run only a balanced subset covering every level of every "
             "dimension — the stage-A screen.",
    )
    p.add_argument(
        "--grid-points", default="",
        help="explicit grid, e.g. 'lr=2e-5:e2,lr=1e-4:e1' (default: the full "
             "study grid). Use this for stage B with the stage-A finalists.",
    )
    p.add_argument(
        "--replicate-seeds", default="",
        help="comma list of extra training seeds to re-run the grid under, "
             "e.g. '1,2,3'. This is what measures the noise floor; without it "
             "the study cannot tell a real per-dimension gap from variance.",
    )
    p.add_argument("--eval-size", type=int, default=400)
    p.add_argument("--judge-size", choices=["small", "medium", "large"], default="small")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--compute", choices=["cloud", "local"], default="cloud",
        help="cloud fans cells out in parallel; local runs them one at a time "
             "on this machine's GPU (smoke only).",
    )
    p.add_argument("--max-concurrent-jobs", type=int, default=8)
    p.add_argument(
        "--chunk-size", type=int, default=runner.DEFAULT_CHUNK_SIZE,
        help="configs per job. Smaller chunks fan out wider and bound the "
             "blast radius of a job timeout.",
    )
    p.add_argument("--job-timeout", type=float, default=12 * 3600)
    p.add_argument("--datagen-concurrency", type=int, default=100)
    p.add_argument("--filter-threshold", type=float, default=7.0)
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--overgen-factor", type=float, default=1.6)
    p.add_argument("--workdir", default="")
    p.add_argument("--run-name", default="")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument(
        "--yes", action="store_true",
        help="skip the cost confirmation. Required for non-interactive runs.",
    )
    p.add_argument(
        "--analyze-only", action="store_true",
        help="re-run the analysis over an existing workdir without training.",
    )
    return p.parse_args(argv)


def _suppress_noisy_http_logs() -> None:
    for name in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)


# ---------------------------------------------------------------------------
# Datasets — generated once per task, sliced to each size
# ---------------------------------------------------------------------------


async def _ensure_task_datasets(
    *, workdir: Path, task, sizes: list[int], eval_size: int, client,
    args: argparse.Namespace,
) -> dict[int, str]:
    """Generate the largest split once, then take nested prefixes.

    Sizes are **prefixes of the same dataset**, not independent draws. That
    makes dataset size a clean nested factor: a difference between the 500-row
    and 8000-row cells is about volume, not about the 500 rows happening to be
    easier. Independent draws would confound the two.
    """
    scorer_rel = f"scorers/{task.name}.md"
    scorer_path = workdir / scorer_rel
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    scorer_path.write_text(task.scorer_md)

    resume = not args.no_resume
    threshold = None if args.no_filter else args.filter_threshold
    biggest = max(sizes)

    full_dir = workdir / f"datasets/{task.name}_train_full"
    await ensure_split(
        out_dir=full_dir, pipeline_path=task.pipeline_path, scorer_path=scorer_path,
        n=biggest, client=client, concurrency=args.datagen_concurrency,
        resume=resume, filter_threshold=threshold,
        overgen_factor=args.overgen_factor, judge_size=args.judge_size,
        label=f"{task.name}_train_full",
    )
    # One eval set per task, shared by every cell of that task, so judge
    # scores are comparable across sizes and models.
    await ensure_split(
        out_dir=workdir / f"datasets/{task.name}_eval",
        pipeline_path=task.pipeline_path, scorer_path=scorer_path,
        n=eval_size, client=client, concurrency=args.datagen_concurrency,
        resume=resume, filter_threshold=threshold,
        overgen_factor=args.overgen_factor, judge_size=args.judge_size,
        label=f"{task.name}_eval",
    )

    full_table = pq.read_table(full_dir / "data.parquet")
    out: dict[int, str] = {}
    for size in sorted(sizes):
        rel = f"datasets/{task.name}_train_n{size}"
        target = workdir / rel
        parquet = target / "data.parquet"
        if not (resume and parquet.exists()
                and pq.read_metadata(parquet).num_rows == size):
            target.mkdir(parents=True, exist_ok=True)
            pq.write_table(full_table.slice(0, size), parquet)
            logger.info("dataset %s: sliced %d rows from the full split", rel, size)
        out[size] = f"{rel}/data.parquet"
    return out


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------


def _job_specs(
    cell: Cell, *, dataset_rel: str, eval_rel: str, scorer_rel: str,
    points: list[GridPoint], chunk_size: int, max_new_tokens: int,
) -> list[runner.JobSpec]:
    specs = []
    for index, chunk in enumerate(chunk_points(points, chunk_size)):
        specs.append(runner.JobSpec(
            cell_id=cell.id,
            chunk_index=index,
            run_name=f"{cell.id}__c{index}",
            launch_config=runner.build_launch_config(
                base_model=cell.hf_id,
                dataset_rel=dataset_rel,
                eval_rel=eval_rel,
                scorer_rel=scorer_rel,
                grid_override=[p.to_override() for p in chunk],
                max_new_tokens=max_new_tokens,
                train_rows=cell.train_size,
            ),
        ))
    return specs


async def _run_cell(
    cell: Cell, specs: list[runner.JobSpec], *, workdir: Path,
    args: argparse.Namespace, semaphore: asyncio.Semaphore,
) -> list[str]:
    """Run every chunk of one cell. Returns notes about anything skipped."""
    notes: list[str] = []
    for spec in specs:
        expected = len(spec.launch_config["grid_override"])
        if not args.no_resume and runner.job_complete(workdir, spec.run_name, expected):
            logger.info("%s: reusing %d completed configs", spec.label, expected)
            continue
        for attempt in range(1, _MAX_CHUNK_ATTEMPTS + 1):
            async with semaphore:
                logger.info(
                    "%s: launching %d configs%s", spec.label, expected,
                    f" (attempt {attempt})" if attempt > 1 else "",
                )
                try:
                    await runner.run_job(
                        spec, workdir=workdir, compute=args.compute,
                        timeout=args.job_timeout,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    if runner.is_auth_error(exc):
                        # Not this cell's problem — the credentials are wrong
                        # for every cell. Marching through the remaining 47
                        # produces 47 identical tracebacks and an empty report.
                        raise runner.CloudAuthError(
                            runner._auth_help(str(exc))
                        ) from exc
                    retryable = isinstance(exc, runner.RetryableJobError)
                    if retryable and attempt < _MAX_CHUNK_ATTEMPTS:
                        # Preemption and silently-orphaned jobs say nothing
                        # about the config, and losing a chunk costs six cells'
                        # worth of the panel the analysis needs to stay
                        # balanced. Resubmit once; a second identical loss is
                        # evidence of something real, so stop there.
                        logger.warning("%s: %s", spec.label, exc)
                        runner.reset_run_dir(workdir, spec.run_name)
                        continue
                    # One dead chunk must not sink the study. Its configs
                    # simply go unmeasured, the analysis excludes them from the
                    # balanced panel, and the report says so.
                    logger.exception("%s failed", spec.label)
                    notes.append(f"{spec.label}: {exc}")
                    break
    return notes


def _collect(cell: Cell, specs: list[runner.JobSpec], workdir: Path) -> list[Observation]:
    """Read one cell's per-config rows into observations."""
    out: list[Observation] = []
    for spec in specs:
        for row in runner.read_rows(workdir, spec.run_name):
            overrides = row.get("overrides", {}).get("training", {})
            config_id = _canonical_config_id(row.get("config_id", ""))
            out.append(Observation(
                cell_id=cell.id,
                config_id=config_id,
                dimensions=cell.dimensions(),
                judge_mean=_number(row.get("judge_mean")),
                eval_loss=_number(row.get("primary")),
                num_epochs=_int(overrides.get("num_epochs")),
                best_epoch=_number(row.get("best_epoch")),
                elapsed_s=_number(row.get("elapsed_s")),
                judge_num_scored=_int(row.get("judge_num_scored")) or 0,
                judge_std=_number(row.get("judge_std")),
                seed=_int(overrides.get("seed")),
                note=row.get("judge_skipped"),
            ))
    return out


def _canonical_config_id(config_id: str) -> str:
    """Strip the seed suffix so replicates group under one config.

    ``lr5e-5_e2_s3`` and ``lr5e-5_e2`` are the same hyperparameters; the seed
    is the thing being varied to measure noise, not part of the config.
    """
    parts = config_id.split("_")
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        return "_".join(parts[:-1])
    return config_id


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _int(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _confirm_cost(cells: list[Cell], n_points: int, args) -> None:
    total_runs = len(cells) * n_points
    logger.info(
        "plan: %d cells × %d configs = %d training runs (+ one judge eval each)",
        len(cells), n_points, total_runs,
    )
    if args.compute == "cloud":
        # A rough order-of-magnitude figure. Real cost depends on dataset size
        # and model; the point is that nobody launches this by accident.
        low = total_runs * 6 / 60 * 9
        high = total_runs * 20 / 60 * 9
        logger.info(
            "estimated cloud GPU cost: roughly $%.0f–$%.0f (A100-80GB, billed)",
            low, high,
        )
    if not args.yes:
        raise SystemExit(
            "refusing to launch without --yes. Review the plan above; add "
            "--yes to proceed, or narrow it with --tasks/--models/--sizes/"
            "--anchors-only."
        )


async def _main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _suppress_noisy_http_logs()

    cells = resolve_cells(
        tasks=args.tasks, sizes=args.sizes, models=args.models,
        anchors_only=args.anchors_only,
    )
    if not cells:
        raise SystemExit("no cells selected")

    points = parse_points(args.grid_points) if args.grid_points else study_grid()
    seeds = tuple(int(s) for s in args.replicate_seeds.split(",") if s.strip())
    if seeds:
        points = points + replicate_grid(points, seeds)

    run_name = args.run_name or f"hpd-{time.strftime('%Y%m%d-%H%M%S')}"
    workdir = (
        Path(args.workdir).expanduser() if args.workdir
        else Path(f"~/.lqh-hp-defaults/{run_name}").expanduser()
    )
    workdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_name": run_name,
        "workdir": str(workdir),
        "compute": args.compute,
        "judge_size": args.judge_size,
        "eval_size": args.eval_size,
        "filter_threshold": None if args.no_filter else args.filter_threshold,
        "n_cells": len(cells),
        "n_configs": len(points),
        "config_ids": [p.id for p in points],
        "replicate_seeds": list(seeds),
        "cells": [c.id for c in cells],
    }

    if not args.analyze_only:
        _confirm_cost(cells, len(points), args)

    token = get_token()
    if not token:
        raise SystemExit("not authenticated — run `lqh` and /login, or set LQH_API_TOKEN.")
    root = api_root()
    client = create_client(token, root + "/v1")
    # The contract that makes a training subprocess self-score inline: there is
    # no TUI watcher behind a standalone script, and eval_all needs the judge.
    os.environ["LQH_API_TOKEN"] = token
    os.environ["LQH_BASE_URL"] = root

    task_names = sorted({c.task for c in cells})
    if not args.skip_preflight and not args.analyze_only:
        from ..base_vs_instruct.preflight import run_preflight

        model_keys = sorted({c.model_key for c in cells})
        compat = run_preflight([(k, MODELS[k][0]) for k in model_keys])
        bad = [c for c in compat if not c.ok]
        if bad:
            detail = "\n".join(
                f"  - {c.key} ({c.hf_id}): {'; '.join(c.notes) or c.error}"
                for c in bad
            )
            raise SystemExit(
                f"model compatibility preflight failed:\n{detail}\n"
                "Drop the offending model(s) or pass --skip-preflight."
            )

    # Prove we can submit BEFORE generating anything. Data generation for four
    # tasks at scale runs for hours; discovering a rejected token afterwards
    # spends the expensive part for nothing.
    if not args.analyze_only and args.compute == "cloud":
        await runner.check_cloud_auth(workdir)

    tasks = {t.name: t for t in resolve_tasks(task_names)}
    datasets: dict[str, dict[int, str]] = {}
    if not args.analyze_only:
        for name in task_names:
            sizes = sorted({c.train_size for c in cells if c.task == name})
            datasets[name] = await _ensure_task_datasets(
                workdir=workdir, task=tasks[name], sizes=sizes,
                eval_size=args.eval_size, client=client, args=args,
            )

    specs_by_cell: dict[str, list[runner.JobSpec]] = {}
    for cell in cells:
        dataset_rel = (
            datasets.get(cell.task, {}).get(cell.train_size)
            or f"datasets/{cell.task}_train_n{cell.train_size}/data.parquet"
        )
        specs_by_cell[cell.id] = _job_specs(
            cell,
            dataset_rel=dataset_rel,
            eval_rel=f"datasets/{cell.task}_eval/data.parquet",
            scorer_rel=f"scorers/{cell.task}.md",
            points=points,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
        )

    notes: list[str] = []
    if not args.analyze_only:
        semaphore = asyncio.Semaphore(max(1, args.max_concurrent_jobs))
        running = [
            asyncio.create_task(
                _run_cell(cell, specs_by_cell[cell.id], workdir=workdir,
                          args=args, semaphore=semaphore)
            )
            for cell in cells
        ]
        try:
            for task in asyncio.as_completed(running):
                notes.extend(await task)
                # Rewrite the report after every cell so a long study is
                # inspectable while it is still running.
                observations = [
                    o for cell in cells
                    for o in _collect(cell, specs_by_cell[cell.id], workdir)
                ]
                write_report(workdir / "report", meta, observations, notes)
        except runner.CloudAuthError:
            # Stop the fleet rather than let the remaining cells each rediscover
            # the same rejected token.
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise

    observations = [
        o for cell in cells for o in _collect(cell, specs_by_cell[cell.id], workdir)
    ]
    _, md_path = write_report(workdir / "report", meta, observations, notes)
    logger.info("done. report: %s", md_path)
    print("\n" + md_path.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
