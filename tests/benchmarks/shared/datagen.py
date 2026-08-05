"""Quality-gated dataset generation for benchmark orchestrators.

Extracted from ``base_vs_instruct/run.py`` so ``hp_defaults`` can reuse it
rather than growing a second copy that drifts.

The core idea is the scorer filter: a benchmark that trains on raw pipeline
output measures the pipeline's bad days as much as the thing under test. Each
generated sample's gold is judged against the task scorer, everything below the
threshold is dropped, and the shortfall is regenerated until the target count is
reached — so the returned split always has exactly the requested number of rows,
all of them above the bar.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lqh.engine import run_pipeline
from lqh.scoring import run_data_filter

logger = logging.getLogger(__name__)


def dataset_rows(data_parquet: Path) -> int | None:
    """Row count from parquet metadata, or None if unreadable."""
    if not data_parquet.exists():
        return None
    try:
        return pq.read_metadata(data_parquet).num_rows
    except Exception:  # noqa: BLE001 — a corrupt file is "not ready", not fatal
        return None


def dataset_ready(data_parquet: Path, want_rows: int) -> bool:
    rows = dataset_rows(data_parquet)
    return rows is not None and rows >= want_rows


async def generate_filtered_split(
    *,
    script_path: Path,
    scorer_path: Path,
    target: int,
    out_dir: Path,
    client,
    concurrency: int,
    threshold: float,
    overgen_factor: float,
    label: str,
    judge_size: str = "small",
    max_rounds: int = 4,
) -> None:
    """Generate a split, keep only samples scoring >= *threshold*, top up.

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
            "the filter threshold or raise the over-generation factor."
        )

    full = pa.concat_tables(kept_tables).slice(0, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(full, out_dir / "data.parquet")
    shutil.rmtree(tmp_root, ignore_errors=True)
    logger.info(
        "datagen %s: wrote %d filtered rows (generated %d, kept %d, drop %d)",
        label, target, total_generated, kept, total_generated - kept,
    )


async def ensure_split(
    *,
    out_dir: Path,
    pipeline_path: Path,
    scorer_path: Path,
    n: int,
    client,
    concurrency: int,
    resume: bool = True,
    filter_threshold: float | None,
    overgen_factor: float = 1.6,
    judge_size: str = "small",
    label: str | None = None,
) -> Path:
    """Ensure ``out_dir/data.parquet`` holds at least *n* rows.

    With *filter_threshold* set, the split is scorer-filtered down to exactly
    *n* rows; otherwise raw pipeline output is used as-is. Returns the parquet
    path. Raises rather than returning a short dataset — a benchmark run on a
    partial split silently measures the wrong thing.
    """
    label = label or out_dir.name
    data_parquet = out_dir / "data.parquet"
    if resume and dataset_ready(data_parquet, n):
        logger.info("datagen %s: reusing %d rows", label, n)
        return data_parquet

    if filter_threshold is None:
        logger.info("datagen %s: generating %d samples (no filter) ...", label, n)
        res = await run_pipeline(
            script_path=pipeline_path, num_samples=n, output_dir=out_dir,
            client=client, concurrency=concurrency,
        )
        logger.info("datagen %s: %d ok / %d failed", label, res.succeeded, res.failed)
    else:
        await generate_filtered_split(
            script_path=pipeline_path,
            scorer_path=scorer_path,
            target=n,
            out_dir=out_dir,
            client=client,
            concurrency=concurrency,
            threshold=filter_threshold,
            overgen_factor=overgen_factor,
            judge_size=judge_size,
            label=label,
        )

    rows = dataset_rows(data_parquet) or 0
    if rows < n:
        raise RuntimeError(
            f"datagen produced only {rows}/{n} rows for {label}. Refusing to "
            "run a benchmark on a partial dataset."
        )
    return data_parquet
