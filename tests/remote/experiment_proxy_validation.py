"""Validate val_loss as a proxy for judge score across a hyperparameter sweep.

Workflow:
  1. Load a frozen dataset (filtered ChatML for --mode sft, preferences
     parquet for --mode dpo).
  2. For each config in the chosen grid:
       a. Train (subprocess: ``python -m lqh.train <cfg>``).
       b. Read final eval metrics from ``eval_history.json`` /
          ``progress.jsonl`` (val_loss, and for DPO eval_rewards/margins
          + accuracies).
       c. Run held-out inference on the same eval set.
       d. Score predictions with the judge (``lqh.scoring.run_scoring``).
       e. Record (config, val_metrics, judge_mean, judge_median).
  3. Emit ``correlation_report.md`` with Pearson/Spearman correlations,
     top-K agreement, and a verdict.

Reuses helpers and patterns from ``experiment_e2e_pipeline.py``:
  ``_run_subprocess``, ``_build_infer_config``, ``run_scoring``, the
  ``TaskConfig`` loader. Each per-config run is independent — no
  on-policy ping-pong, no held-out scoring loop.

Usage:
  python -m tests.remote.experiment_proxy_validation \\
      --mode dpo \\
      --task ar_to_de \\
      --base-model <path-to-sft-merged-model> \\
      --preferences <path-to-preferences.parquet> \\
      --eval-dataset <path-to-eval-filtered.parquet> \\
      --out-dir results/proxy_validation/dpo_<timestamp> \\
      --grid small
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Reuse e2e helpers: TaskConfig loader, infer-config builder.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.remote.experiment_e2e_pipeline import (  # noqa: E402
    TaskConfig,
    _build_infer_config,
    _run_subprocess,
    load_task_config,
)


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DPOConfig:
    lr: float
    beta: float
    epochs: int = 1

    def name(self) -> str:
        return f"dpo_lr{self.lr:g}_b{self.beta:g}_e{self.epochs}"


@dataclass(frozen=True)
class SFTConfig:
    lr: float
    epochs: int

    def name(self) -> str:
        return f"sft_lr{self.lr:g}_e{self.epochs}"


def dpo_grid(size: str) -> list[DPOConfig]:
    # lr range spans the n10000-collapse-relevant axis: 1e-6 (calm) up to
    # 2e-5 (spicy). beta=0.1 is the current default; 0.05 weakens the
    # KL anchor and should be more prone to collapse at high lr.
    small = [
        DPOConfig(lr=1e-6, beta=0.10),
        DPOConfig(lr=1e-6, beta=0.05),
        DPOConfig(lr=5e-6, beta=0.10),
        DPOConfig(lr=5e-6, beta=0.05),
        DPOConfig(lr=2e-5, beta=0.10),
        DPOConfig(lr=2e-5, beta=0.05),
    ]
    if size == "tiny":
        return [small[0], small[2], small[5]]
    if size == "small":
        return small
    if size == "medium":
        # add a 3rd epoch budget and a higher lr extreme
        extras = [
            DPOConfig(lr=5e-5, beta=0.10),
            DPOConfig(lr=5e-5, beta=0.05),
            DPOConfig(lr=5e-6, beta=0.10, epochs=2),
            DPOConfig(lr=5e-6, beta=0.05, epochs=2),
            DPOConfig(lr=2e-5, beta=0.10, epochs=2),
            DPOConfig(lr=2e-5, beta=0.05, epochs=2),
        ]
        return small + extras
    raise ValueError(f"unknown grid size: {size}")


def sft_grid(size: str) -> list[SFTConfig]:
    small = [
        SFTConfig(lr=5e-6, epochs=2),
        SFTConfig(lr=5e-6, epochs=3),
        SFTConfig(lr=2e-5, epochs=2),
        SFTConfig(lr=2e-5, epochs=3),
        SFTConfig(lr=5e-5, epochs=2),
        SFTConfig(lr=5e-5, epochs=3),
    ]
    if size == "tiny":
        return [small[0], small[2], small[4]]
    if size == "small":
        return small
    if size == "medium":
        extras = [
            SFTConfig(lr=1e-5, epochs=2),
            SFTConfig(lr=1e-5, epochs=3),
            SFTConfig(lr=1e-4, epochs=2),
            SFTConfig(lr=1e-4, epochs=3),
            SFTConfig(lr=2e-5, epochs=4),
            SFTConfig(lr=5e-5, epochs=4),
        ]
        return small + extras
    raise ValueError(f"unknown grid size: {size}")


# --------------------------------------------------------------------------
# Per-config training
# --------------------------------------------------------------------------


def train_one_dpo(
    cfg: DPOConfig,
    base_model: str,
    preferences_path: Path,
    run_dir: Path,
    cfg_e2e: TaskConfig,
) -> dict[str, Any]:
    """Train DPO once on a frozen preference set. Returns final eval metrics."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pre-seed iter_000/preferences.parquet and set
    # skip_generation_if_preferences_exist=True so the DPO loop
    # bypasses on-policy generation and goes straight to the
    # optimization step.
    iter0_dir = run_dir / "iterations" / "iter_000"
    iter0_dir.mkdir(parents=True, exist_ok=True)
    pref_target = iter0_dir / "preferences.parquet"
    if pref_target.exists() or pref_target.is_symlink():
        pref_target.unlink()
    try:
        pref_target.symlink_to(preferences_path.resolve())
    except OSError:
        import shutil
        shutil.copyfile(preferences_path, pref_target)

    config = {
        "type": "on_policy_dpo",
        "base_model": base_model,
        # preference_dataset is irrelevant here — generation is
        # skipped because preferences.parquet is pre-seeded. We still
        # need to point it at SOME valid parquet for the initial
        # load_chatml_dataset call at module top; use the
        # preferences file itself (the load is forgiving — it only
        # reads the "messages" column shape, which we won't use).
        "preference_dataset": str(preferences_path.resolve()),
        "skip_generation_if_preferences_exist": True,
        "num_iterations": 1,
        "dpo_beta": cfg.beta,
        "training": {
            "learning_rate": cfg.lr,
            "per_device_batch_size": 1,
            "gradient_checkpointing": True,
            "bf16": True,
            "max_seq_length": 2048,
            "dpo_num_epochs": cfg.epochs,
            "eval_split_ratio": 0.1,
            "dpo_eval_steps": 10,
        },
        "lora": {
            "enabled": True,
            "r": 32,
            "alpha": 64,
        },
    }
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2) + "\n")

    _run_subprocess("lqh.train", cfg_path)

    # Read final eval metrics from per-iter eval_history.json.
    eval_hist_path = iter0_dir / "eval_history.json"
    final_eval: dict[str, Any] = {}
    if eval_hist_path.exists():
        try:
            entries = json.loads(eval_hist_path.read_text())
            for entry in reversed(entries):
                if "eval_loss" in entry:
                    final_eval = entry
                    break
        except Exception:
            pass

    # Also merge in chosen-CE metrics (written by _ChosenCECallback).
    ce_path = iter0_dir / "chosen_ce_summary.json"
    if ce_path.exists():
        try:
            ce = json.loads(ce_path.read_text())
            final_eval.update(ce)
        except Exception:
            pass

    return final_eval


def train_one_sft(
    cfg: SFTConfig,
    base_model: str,
    train_dataset: Path,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "type": "sft",
        "base_model": base_model,
        "dataset": str(train_dataset.resolve()),
        "training": {
            "num_epochs": cfg.epochs,
            "learning_rate": cfg.lr,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "eval_split_ratio": 0.1,
            "eval_steps": 50,
            "logging_steps": 25,
        },
        "lora": {
            "enabled": True,
            "r": 32,
            "alpha": 64,
        },
    }
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2) + "\n")

    _run_subprocess("lqh.train", cfg_path)

    eval_hist_path = run_dir / "eval_history.json"
    final_eval: dict[str, Any] = {}
    if eval_hist_path.exists():
        try:
            entries = json.loads(eval_hist_path.read_text())
            for entry in reversed(entries):
                if "eval_loss" in entry:
                    final_eval = entry
                    break
        except Exception:
            pass
    return final_eval


# --------------------------------------------------------------------------
# Per-config judge eval
# --------------------------------------------------------------------------


async def judge_score(
    cfg_e2e: TaskConfig,
    model_dir: Path,
    eval_dataset: Path,
    out_dir: Path,
    *,
    judge_size: str,
    judge_concurrency: int,
) -> dict[str, Any]:
    """Run inference + judge scoring on the eval set. Reuse e2e helpers."""
    from lqh.auth import get_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_scoring

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    predictions = out_dir / "predictions.parquet"

    infer_cfg = _build_infer_config(cfg_e2e, str(model_dir.resolve()), eval_dataset)
    cfg_path.write_text(json.dumps(infer_cfg, indent=2) + "\n")
    _run_subprocess("lqh.infer", cfg_path)

    if not predictions.exists():
        raise RuntimeError(f"inference produced no predictions at {predictions}")

    api_key = os.environ.get("LQH_API_KEY") or get_token() or ""
    client = create_client(api_key, load_config().api_base_url)
    result = await run_scoring(
        dataset_path=predictions,
        scorer_path=cfg_e2e.scorer_path,
        output_dir=out_dir / "scoring",
        client=client,
        model_size=judge_size,
        concurrency=judge_concurrency,
        run_inference=False,
    )
    return {
        "mean": result.mean_score,
        "median": result.median_score,
        "scored": result.scored,
        "failed": result.failed,
    }


# --------------------------------------------------------------------------
# Correlation analysis
# --------------------------------------------------------------------------


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sx * sy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation. Uses average ranks for ties."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        result = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based average rank
            for k in range(i, j + 1):
                result[order[k]] = avg
            i = j + 1
        return result

    return pearson(ranks(xs), ranks(ys))


def verdict(r: float | None, top1_match: bool) -> tuple[str, str]:
    """Return (emoji, text) for the headline verdict."""
    if r is None:
        return "❓", "Not enough data points for correlation."
    r_abs = abs(r)
    if r_abs >= 0.7 and top1_match:
        return "✅", f"r={r:+.3f} and top-1 picks match. Proxy validated; use eval_loss in sweep harness."
    if r_abs >= 0.4 or top1_match:
        return "⚠", f"r={r:+.3f} (top-1 {'match' if top1_match else 'miss'}). Proxy is weak — use as filter only; still judge-eval top candidates."
    return "❌", f"r={r:+.3f} and top-1 miss. Proxy is unreliable on this dataset; fall back to per-config judge eval."


def _correlations(
    rows: list[dict[str, Any]], proxy_key: str, *, lower_is_better: bool
) -> dict[str, Any]:
    """Compute pearson/spearman + top-K agreement of one proxy vs judge_mean."""
    pairs = [(r[proxy_key], r["judge_mean"]) for r in rows
             if r.get(proxy_key) is not None and r.get("judge_mean") is not None]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson": None, "spearman": None,
                "top1_match": False, "top3_overlap": 0}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    r = pearson(xs, ys)
    rs = spearman(xs, ys)
    # Top-K: best config by judge vs best by proxy (low or high depending on metric)
    by_judge = sorted(rows, key=lambda r: r.get("judge_mean") or float("-inf"), reverse=True)
    by_proxy = sorted(
        rows,
        key=lambda r: (r.get(proxy_key) if r.get(proxy_key) is not None else float("inf")),
        reverse=not lower_is_better,
    )
    top1_match = (by_judge[0]["config_id"] == by_proxy[0]["config_id"])
    top3_judge = {r["config_id"] for r in by_judge[:3]}
    top3_proxy = {r["config_id"] for r in by_proxy[:3]}
    return {
        "n": len(pairs),
        "pearson": r,
        "spearman": rs,
        "top1_match": top1_match,
        "top3_overlap": len(top3_judge & top3_proxy),
    }


def write_report(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    mode: str,
    grid_size: str,
) -> None:
    """Sort by judge_mean desc, compute correlations, write markdown + json."""
    valid = [r for r in rows if r.get("judge_mean") is not None and r.get("eval_loss") is not None]
    judge_vals = [r["judge_mean"] for r in valid]
    loss_vals = [r["eval_loss"] for r in valid]

    # Primary proxy candidate: DPO eval_loss. Lower is better → expect negative r.
    r_loss = pearson(loss_vals, judge_vals)
    sp_loss = spearman(loss_vals, judge_vals)

    # DPO eval_rewards/margins — expect positive r (higher margin = better)
    margins_valid = [r for r in valid if r.get("margins") is not None]
    r_margins: float | None = None
    sp_margins: float | None = None
    if margins_valid:
        m_judge = [r["judge_mean"] for r in margins_valid]
        margins_vals = [r["margins"] for r in margins_valid]
        r_margins = pearson(margins_vals, m_judge)
        sp_margins = spearman(margins_vals, m_judge)

    # NEW: chosen-CE proxies. Lower CE = higher P(chosen) = closer to
    # generating that good response → expect NEGATIVE r with judge.
    ce_corrs = {
        "ce_chosen_mean": _correlations(valid, "ce_chosen_mean", lower_is_better=True),
        "ce_chosen_p90":  _correlations(valid, "ce_chosen_p90",  lower_is_better=True),
        "ce_chosen_p95":  _correlations(valid, "ce_chosen_p95",  lower_is_better=True),
        "ce_chosen_max":  _correlations(valid, "ce_chosen_max",  lower_is_better=True),
        "ce_chosen_delta_ref": _correlations(valid, "ce_chosen_delta_ref", lower_is_better=True),
        "ce_abs_margin":  _correlations(valid, "ce_abs_margin",  lower_is_better=False),
    }

    # Top-K agreement: best config by judge_mean vs best by (low) eval_loss.
    if valid:
        by_judge = sorted(valid, key=lambda r: r["judge_mean"], reverse=True)
        by_loss = sorted(valid, key=lambda r: r["eval_loss"])
        top1_match = by_judge[0]["config_id"] == by_loss[0]["config_id"]
        top3_judge = {r["config_id"] for r in by_judge[:3]}
        top3_loss = {r["config_id"] for r in by_loss[:3]}
        top3_overlap = len(top3_judge & top3_loss)
    else:
        by_judge = []
        top1_match = False
        top3_overlap = 0

    # Verdict uses the BEST available proxy: pick whichever candidate
    # has the highest |pearson r| with the right sign + top-1 match.
    proxy_candidates: list[tuple[str, float | None, bool, int, str]] = [
        ("eval_loss (DPO loss)", r_loss, top1_match, top3_overlap, "negative"),
    ]
    for key, c in ce_corrs.items():
        sign = "positive" if key == "ce_abs_margin" else "negative"
        proxy_candidates.append((key, c["pearson"], c["top1_match"], c["top3_overlap"], sign))

    def _score(cand: tuple[str, float | None, bool, int, str]) -> float:
        _, r, t1, t3, sign = cand
        if r is None:
            return -1.0
        # Penalize wrong sign (a negative-expected proxy with positive r is broken)
        sign_ok = (r <= 0 and sign == "negative") or (r >= 0 and sign == "positive")
        if not sign_ok:
            return -abs(r)  # negative score for wrong-sign proxies
        return abs(r) + 0.1 * t3 + (0.05 if t1 else 0)

    best_proxy = max(proxy_candidates, key=_score)
    best_name, best_r, best_t1, best_t3, _ = best_proxy
    emoji, summary = verdict(best_r, best_t1)
    summary = f"best proxy: **{best_name}** ({summary})"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "correlation_data.json").write_text(
        json.dumps({
            "mode": mode,
            "grid_size": grid_size,
            "n_configs": len(rows),
            "n_valid": len(valid),
            "pearson_eval_loss_vs_judge": r_loss,
            "spearman_eval_loss_vs_judge": sp_loss,
            "pearson_margins_vs_judge": r_margins,
            "spearman_margins_vs_judge": sp_margins,
            "ce_correlations": ce_corrs,
            "best_proxy": best_name,
            "top1_match_eval_loss": top1_match,
            "top3_overlap_eval_loss": top3_overlap,
            "rows": rows,
        }, indent=2, default=str) + "\n"
    )

    def _fmt(v: float | None) -> str:
        return f"{v:+.3f}" if v is not None else "n/a"

    lines: list[str] = []
    lines.append(f"# Proxy-validation report — mode={mode}, grid={grid_size}\n")
    lines.append(f"- timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- configs run: {len(rows)} (valid: {len(valid)})\n")

    lines.append("## Verdict\n")
    lines.append(f"{emoji} {summary}\n")

    lines.append("## Correlations (all proxy candidates vs judge_mean)\n")
    lines.append("| proxy | pearson r | spearman ρ | top-1 | top-3 | expected sign |")
    lines.append("|---|---:|---:|:--:|:--:|:--:|")
    lines.append(
        f"| eval_loss (DPO loss) | {_fmt(r_loss)} | {_fmt(sp_loss)} "
        f"| {'✓' if top1_match else '✗'} | {top3_overlap}/3 | negative |"
    )
    if r_margins is not None:
        margins_corr = _correlations(valid, "margins", lower_is_better=False)
        lines.append(
            f"| eval_rewards/margins | {_fmt(r_margins)} | {_fmt(sp_margins)} "
            f"| {'✓' if margins_corr['top1_match'] else '✗'} "
            f"| {margins_corr['top3_overlap']}/3 | positive |"
        )
    for key, c in ce_corrs.items():
        if c["pearson"] is None:
            continue
        sign = "positive" if key == "ce_abs_margin" else "negative"
        lines.append(
            f"| {key} | {_fmt(c['pearson'])} | {_fmt(c['spearman'])} "
            f"| {'✓' if c['top1_match'] else '✗'} | {c['top3_overlap']}/3 | {sign} |"
        )
    lines.append("")
    lines.append(f"- top-3 overlap: **{top3_overlap}/3**")
    lines.append("")

    lines.append("## Per-config results (sorted by judge_mean desc)\n")
    if mode == "dpo":
        lines.append("| config | lr | β | epochs | eval_loss | margins | CE(ch) | CE(ch)p90 | Δref | judge_mean | judge_med | t (s) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| config | lr | epochs | eval_loss | judge_mean | judge_median | elapsed (s) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")

    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get("judge_mean") is None, -(r.get("judge_mean") or 0.0)),
    )
    for r in rows_sorted:
        hp = r["hyperparams"]
        jm = r.get("judge_mean")
        jmed = r.get("judge_median")
        elapsed = r.get("elapsed_s")
        elapsed_s = f"{elapsed:.0f}" if elapsed is not None else "—"
        eloss = r.get("eval_loss")
        eloss_s = f"{eloss:.4f}" if eloss is not None else "—"
        jm_s = f"{jm:.3f}" if jm is not None else "—"
        jmed_s = f"{jmed:.1f}" if jmed is not None else "—"
        if mode == "dpo":
            m = r.get("margins")
            m_s = f"{m:+.3f}" if m is not None else "—"
            ce_mean = r.get("ce_chosen_mean")
            ce_p90 = r.get("ce_chosen_p90")
            ce_dref = r.get("ce_chosen_delta_ref")
            ce_mean_s = f"{ce_mean:.3f}" if ce_mean is not None else "—"
            ce_p90_s = f"{ce_p90:.3f}" if ce_p90 is not None else "—"
            ce_dref_s = f"{ce_dref:+.3f}" if ce_dref is not None else "—"
            lines.append(
                f"| {r['config_id']} | {hp['lr']:g} | {hp['beta']:g} | "
                f"{hp['epochs']} | {eloss_s} | {m_s} | {ce_mean_s} | "
                f"{ce_p90_s} | {ce_dref_s} | {jm_s} | {jmed_s} | {elapsed_s} |"
            )
        else:
            lines.append(
                f"| {r['config_id']} | {hp['lr']:g} | {hp['epochs']} | "
                f"{eloss_s} | {jm_s} | {jmed_s} | {elapsed_s} |"
            )
    lines.append("")

    if valid:
        lines.append("## ASCII scatter (DPO eval_loss vs judge_mean)\n")
        lines.append("```")
        lines.append("eval_loss → judge_mean")
        for r in sorted(valid, key=lambda r: r["eval_loss"]):
            lines.append(
                f"  {r['config_id']:24s}  loss={r['eval_loss']:.4f}  "
                f"judge={r['judge_mean']:.3f}"
            )
        lines.append("```\n")

        # Second scatter: chosen-CE proxies, if present
        ce_rows = [r for r in valid if r.get("ce_chosen_p90") is not None]
        if ce_rows:
            lines.append("## ASCII scatter (CE_chosen_p90 vs judge_mean)\n")
            lines.append("```")
            lines.append("ce_p90 → judge_mean   (lower CE = higher P(chosen) = better)")
            for r in sorted(ce_rows, key=lambda r: r["ce_chosen_p90"]):
                lines.append(
                    f"  {r['config_id']:24s}  ce_p90={r['ce_chosen_p90']:.3f}  "
                    f"ce_mean={r.get('ce_chosen_mean', 0):.3f}  "
                    f"Δref={r.get('ce_chosen_delta_ref') or 0:+.3f}  "
                    f"judge={r['judge_mean']:.3f}"
                )
            lines.append("```\n")

    (out_dir / "correlation_report.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


async def run_dpo_validation(args: argparse.Namespace) -> None:
    cfg_e2e = load_task_config(
        args.task,
        override_train=None,
        override_eval=None,
        config_name="default",
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = dpo_grid(args.grid)
    rows: list[dict[str, Any]] = []

    print(f"\n{'='*70}")
    print(f"DPO proxy validation — {len(grid)} configs, task={args.task}")
    print(f"  base_model:  {args.base_model}")
    print(f"  preferences: {Path(args.preferences).relative_to(REPO_ROOT) if Path(args.preferences).is_absolute() else args.preferences}")
    print(f"  eval_set:    {Path(args.eval_dataset).relative_to(REPO_ROOT) if Path(args.eval_dataset).is_absolute() else args.eval_dataset}")
    print(f"  out_dir:     {out_dir.relative_to(REPO_ROOT) if out_dir.is_absolute() else out_dir}")
    print(f"{'='*70}\n", flush=True)

    for i, dcfg in enumerate(grid):
        config_id = dcfg.name()
        cfg_dir = out_dir / config_id
        print(f"\n[{i+1}/{len(grid)}] {config_id}", flush=True)
        t0 = time.time()
        row: dict[str, Any] = {
            "config_id": config_id,
            "hyperparams": asdict(dcfg),
        }
        try:
            final_eval = train_one_dpo(
                dcfg, args.base_model, Path(args.preferences), cfg_dir, cfg_e2e
            )
            row["eval_loss"] = final_eval.get("eval_loss")
            row["margins"] = final_eval.get("eval_rewards/margins")
            row["accuracies"] = final_eval.get("eval_rewards/accuracies")
            row["chosen_reward"] = final_eval.get("eval_rewards/chosen")
            row["rejected_reward"] = final_eval.get("eval_rewards/rejected")
            # Chosen-CE proxies (generation-aligned, dpo-collapse-aware):
            row["ce_chosen_mean"] = final_eval.get("eval_ce_chosen_mean")
            row["ce_chosen_p90"] = final_eval.get("eval_ce_chosen_p90")
            row["ce_chosen_p95"] = final_eval.get("eval_ce_chosen_p95")
            row["ce_chosen_max"] = final_eval.get("eval_ce_chosen_max")
            row["ce_rejected_mean"] = final_eval.get("eval_ce_rejected_mean")
            row["ce_abs_margin"] = final_eval.get("eval_ce_abs_margin")
            row["ce_chosen_delta_ref"] = final_eval.get("eval_ce_chosen_delta_ref")
            row["ref_ce_chosen_mean"] = final_eval.get("ref_ce_chosen_mean")

            # judge eval on the held-out set
            model_dir = cfg_dir / "model"
            if not model_dir.exists():
                raise RuntimeError(f"trained model missing at {model_dir}")
            judge_dir = cfg_dir / "judge_eval"
            judge = await judge_score(
                cfg_e2e, model_dir, Path(args.eval_dataset), judge_dir,
                judge_size=args.judge_size,
                judge_concurrency=args.judge_concurrency,
            )
            row["judge_mean"] = judge["mean"]
            row["judge_median"] = judge["median"]
            row["judge_scored"] = judge["scored"]
            row["judge_failed"] = judge["failed"]
        except Exception as exc:
            row["error"] = str(exc)
            print(f"  ❌ {config_id} failed: {exc}", flush=True)
        row["elapsed_s"] = time.time() - t0
        rows.append(row)

        # Write progress as we go (so a crash doesn't lose data).
        (out_dir / "runs.jsonl").open("a").write(json.dumps(row, default=str) + "\n")
        write_report(rows, out_dir, mode="dpo", grid_size=args.grid)

    print(f"\nDone. Report at {out_dir / 'correlation_report.md'}")


async def run_sft_validation(args: argparse.Namespace) -> None:
    cfg_e2e = load_task_config(
        args.task,
        override_train=None,
        override_eval=None,
        config_name="default",
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = sft_grid(args.grid)
    rows: list[dict[str, Any]] = []

    print(f"\n{'='*70}")
    print(f"SFT proxy validation — {len(grid)} configs, task={args.task}")
    print(f"  base_model:   {args.base_model}")
    print(f"  train_data:   {args.train_dataset}")
    print(f"  eval_set:     {args.eval_dataset}")
    print(f"  out_dir:      {out_dir}")
    print(f"{'='*70}\n", flush=True)

    for i, scfg in enumerate(grid):
        config_id = scfg.name()
        cfg_dir = out_dir / config_id
        print(f"\n[{i+1}/{len(grid)}] {config_id}", flush=True)
        t0 = time.time()
        row: dict[str, Any] = {
            "config_id": config_id,
            "hyperparams": asdict(scfg),
        }
        try:
            final_eval = train_one_sft(
                scfg, args.base_model, Path(args.train_dataset), cfg_dir
            )
            row["eval_loss"] = final_eval.get("eval_loss")

            model_dir = cfg_dir / "model"
            if not model_dir.exists():
                raise RuntimeError(f"trained model missing at {model_dir}")
            judge_dir = cfg_dir / "judge_eval"
            judge = await judge_score(
                cfg_e2e, model_dir, Path(args.eval_dataset), judge_dir,
                judge_size=args.judge_size,
                judge_concurrency=args.judge_concurrency,
            )
            row["judge_mean"] = judge["mean"]
            row["judge_median"] = judge["median"]
            row["judge_scored"] = judge["scored"]
            row["judge_failed"] = judge["failed"]
        except Exception as exc:
            row["error"] = str(exc)
            print(f"  ❌ {config_id} failed: {exc}", flush=True)
        row["elapsed_s"] = time.time() - t0
        rows.append(row)

        (out_dir / "runs.jsonl").open("a").write(json.dumps(row, default=str) + "\n")
        write_report(rows, out_dir, mode="sft", grid_size=args.grid)

    print(f"\nDone. Report at {out_dir / 'correlation_report.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sft", "dpo"), required=True)
    p.add_argument("--task", required=True,
                   help="Task name under tests/e2e_projects/ (provides scorer + system prompt + schema)")
    p.add_argument("--base-model", required=True,
                   help="HF model id or local checkpoint path")
    p.add_argument("--preferences",
                   help="(dpo only) path to preferences.parquet")
    p.add_argument("--train-dataset",
                   help="(sft only) path to filtered ChatML parquet")
    p.add_argument("--eval-dataset", required=True,
                   help="held-out eval parquet for judge scoring")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--grid", choices=("tiny", "small", "medium"), default="small")
    p.add_argument("--judge-size", default="small")
    p.add_argument("--judge-concurrency", type=int, default=8)
    args = p.parse_args()

    if args.mode == "dpo":
        if not args.preferences:
            p.error("--preferences is required for --mode dpo")
        asyncio.run(run_dpo_validation(args))
    else:
        if not args.train_dataset:
            p.error("--train-dataset is required for --mode sft")
        asyncio.run(run_sft_validation(args))


if __name__ == "__main__":
    main()
