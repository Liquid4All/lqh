"""Result-table rendering for the base-vs-instruct benchmark.

Consumes the flat list of per-(task, model) records the orchestrator
accumulates and emits ``results.json`` (machine) + ``report.md`` (human). The
markdown carries the headline question — does the base-vs-instruct ranking
hold across tasks — as a per-task table plus a cross-task aggregate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelResult:
    """One model's three eval points on one task. Means are 0-10 judge scores
    (or None when that stage was skipped / failed / produced no scored rows)."""

    task: str
    model: str
    n_eval: int = 0
    baseline: float | None = None
    sft: float | None = None
    dpo: float | None = None
    sft_sweep_s: float | None = None
    dpo_sweep_s: float | None = None
    sft_training_runs: list[dict[str, Any]] = field(default_factory=list)
    dpo_training_runs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def sft_delta(self) -> float | None:
        return _delta(self.sft, self.baseline)

    @property
    def dpo_delta(self) -> float | None:
        return _delta(self.dpo, self.baseline)

    @property
    def best(self) -> float | None:
        vals = [v for v in (self.baseline, self.sft, self.dpo) if v is not None]
        return max(vals) if vals else None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def _fmt_delta(v: float | None) -> str:
    return "—" if v is None else f"{v:+.2f}"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{int(seconds)}s"


def write_report(
    out_dir: Path,
    meta: dict[str, Any],
    results: list[ModelResult],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    md_path = out_dir / "report.md"

    json_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "results": [
                    {
                        **asdict(r),
                        "sft_delta": r.sft_delta,
                        "dpo_delta": r.dpo_delta,
                        "best": r.best,
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    md_path.write_text(_render_md(meta, results))
    return json_path, md_path


def _render_md(meta: dict[str, Any], results: list[ModelResult]) -> str:
    tasks = _ordered_unique(r.task for r in results)
    models = _ordered_unique(r.model for r in results)
    by_key = {(r.task, r.model): r for r in results}

    lines: list[str] = []
    lines.append("# Base vs Instruct — fine-tuning benchmark")
    lines.append("")
    lines.append(f"Run: `{meta.get('run_name', '?')}`  ")
    lines.append(
        f"train_size={meta.get('train_size')}  eval_size={meta.get('eval_size')}  "
        f"grid={meta.get('grid_size')}  judge={meta.get('judge_size')}  "
        f"dpo={'on' if not meta.get('skip_dpo') else 'off'}  "
        f"compute={meta.get('compute', 'local')}"
    )
    lines.append("")
    lines.append(
        "Scores are mean judge ratings (0–10) on the held-out eval set, from "
        "**local** inference of the model weights (not the API). Higher is "
        "better. Δ columns are improvement over each model's own baseline."
    )
    lines.append("")

    # --- per-task tables ---
    for task in tasks:
        lines.append(f"## Task: {task}")
        lines.append("")
        lines.append(
            "| Model | Baseline | Best SFT | Best DPO | ΔSFT | ΔDPO | "
            "Best | SFT sweep | DPO sweep |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for model in models:
            r = by_key.get((task, model))
            if r is None:
                continue
            lines.append(
                f"| {model} | {_fmt(r.baseline)} | {_fmt(r.sft)} | {_fmt(r.dpo)} "
                f"| {_fmt_delta(r.sft_delta)} | {_fmt_delta(r.dpo_delta)} "
                f"| {_fmt(r.best)} | {_fmt_duration(r.sft_sweep_s)} "
                f"| {_fmt_duration(r.dpo_sweep_s)} |"
            )
        lines.append("")
        winner = _best_final(by_key, task, models)
        gainer = _best_gainer(by_key, task, models)
        if winner:
            lines.append(
                f"- **Highest final score:** `{winner[0]}` at {winner[1]:.2f}."
            )
        if gainer:
            lines.append(
                f"- **Largest lift from fine-tuning:** `{gainer[0]}` "
                f"({gainer[1]:+.2f} over its baseline)."
            )
        lines.append("")

    # --- cross-task aggregate (mean of best-achieved across tasks) ---
    lines.append("## Cross-task aggregate")
    lines.append("")
    lines.append(
        "Mean across tasks of each model's *best achieved* score (max of "
        "baseline/SFT/DPO per task) and its lift over baseline."
    )
    lines.append("")
    lines.append("| Model | Mean baseline | Mean best | Mean lift |")
    lines.append("|---|---|---|---|")
    for model in models:
        rs = [by_key[(t, model)] for t in tasks if (t, model) in by_key]
        mean_base = _mean([r.baseline for r in rs])
        mean_best = _mean([r.best for r in rs])
        mean_lift = _mean([_delta(r.best, r.baseline) for r in rs])
        lines.append(
            f"| {model} | {_fmt(mean_base)} | {_fmt(mean_best)} "
            f"| {_fmt_delta(mean_lift)} |"
        )
    lines.append("")

    # --- notes ---
    noted = [(r, n) for r in results for n in r.notes]
    if noted:
        lines.append("## Notes")
        lines.append("")
        for r, n in noted:
            # Multi-line notes (e.g. a failure carrying a stderr tail) render
            # as a summary bullet + fenced block so the detail survives intact
            # instead of being mangled into a one-line list item.
            if "\n" in n:
                summary, _, detail = n.partition("\n")
                lines.append(f"- `{r.task}/{r.model}`: {summary}")
                lines.append("")
                lines.append("  ```")
                lines.extend(f"  {ln}" for ln in detail.splitlines())
                lines.append("  ```")
            else:
                lines.append(f"- `{r.task}/{r.model}`: {n}")
        lines.append("")

    timed = [
        r for r in results
        if r.sft_training_runs or r.dpo_training_runs
    ]
    if timed:
        lines.append("## Training Duration Details")
        lines.append("")
        for r in timed:
            if r.sft_training_runs:
                parts = [
                    f"{x.get('config_id', '?')}={_fmt_duration(x.get('elapsed_s'))}"
                    for x in r.sft_training_runs
                ]
                lines.append(f"- `{r.task}/{r.model}` SFT configs: " + ", ".join(parts))
            if r.dpo_training_runs:
                parts = [
                    f"{x.get('config_id', '?')}={_fmt_duration(x.get('elapsed_s'))}"
                    for x in r.dpo_training_runs
                ]
                lines.append(f"- `{r.task}/{r.model}` DPO configs: " + ", ".join(parts))
        lines.append("")

    return "\n".join(lines)


def _best_final(by_key, task, models) -> tuple[str, float] | None:
    cands = [
        (m, by_key[(task, m)].best)
        for m in models
        if (task, m) in by_key and by_key[(task, m)].best is not None
    ]
    return max(cands, key=lambda x: x[1]) if cands else None


def _best_gainer(by_key, task, models) -> tuple[str, float] | None:
    cands = []
    for m in models:
        r = by_key.get((task, m))
        if r is None:
            continue
        lift = _delta(r.best, r.baseline)
        if lift is not None:
            cands.append((m, lift))
    return max(cands, key=lambda x: x[1]) if cands else None


def _mean(vals: list[float | None]) -> float | None:
    present = [v for v in vals if v is not None]
    return sum(present) / len(present) if present else None


def _ordered_unique(it) -> list[str]:
    seen: dict[str, None] = {}
    for x in it:
        seen.setdefault(x, None)
    return list(seen)
