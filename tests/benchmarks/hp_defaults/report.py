"""Report rendering for the hyperparameter-defaults study.

Emits ``results.json`` (machine, carries every observation so the analysis can
be re-run without touching a GPU) and ``report.md`` (human).

The markdown leads with the two things the study exists to answer — which
single default to ship, and whether any dimension needs its own — and ends with
a paste-ready snippet for ``lqh/train/defaults.py``, so acting on the result is
a copy rather than a translation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .analyze import (
    Observation,
    conditional_defaults,
    epoch_report,
    noise_floor,
    proxy_validation,
    recommend,
    widest_panel,
)


def build_results(
    meta: dict[str, Any],
    observations: Sequence[Observation],
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the full analysis and return the JSON payload."""
    configs, cells = widest_panel(observations)
    rec = recommend(observations, configs=configs, cells=cells)
    floor = noise_floor(observations)
    conditional = (
        conditional_defaults(
            observations, global_config=rec.config_id,
            configs=configs, cells=cells, noise=floor,
        )
        if rec.config_id else []
    )
    return {
        "meta": dict(meta),
        "notes": list(notes),
        "n_observations": len(observations),
        "n_scored": sum(1 for o in observations if o.scored),
        "recommendation": rec.as_dict(),
        "noise_floor": floor.as_dict(),
        "conditional_defaults": [c.as_dict() for c in conditional],
        "proxy_validation": proxy_validation(observations, cells=cells).as_dict(),
        "epochs": epoch_report(observations).as_dict(),
        "observations": [asdict(o) for o in observations],
    }


def write_report(
    out_dir: Path,
    meta: dict[str, Any],
    observations: Sequence[Observation],
    notes: Sequence[str] = (),
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = build_results(meta, observations, notes)
    json_path = out_dir / "results.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    md_path.write_text(render_markdown(results))
    return json_path, md_path


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(results: dict[str, Any]) -> str:
    meta = results["meta"]
    rec = results["recommendation"]
    floor = results["noise_floor"]

    lines: list[str] = [
        "# Default hyperparameters — calibration study",
        "",
        f"Run `{meta.get('run_name', '?')}` · compute `{meta.get('compute', '?')}` "
        f"· judge `{meta.get('judge_size', '?')}`",
        f"{results['n_scored']}/{results['n_observations']} measurements scored.",
        "",
    ]

    if not rec.get("config_id"):
        lines += [
            "## No recommendation",
            "",
            "No configuration was measured on a comparable set of cells. Check "
            "`notes` below and `results.json` for what failed.",
            "",
        ]
        lines += _render_notes(results)
        return "\n".join(lines) + "\n"

    lines += _render_recommendation(rec, floor)
    lines += _render_ranking(rec)
    lines += _render_conditional(results, rec)
    lines += _render_proxy(results)
    lines += _render_epochs(results)
    lines += _render_snippet(rec, meta)
    lines += _render_notes(results)
    return "\n".join(lines) + "\n"


def _render_recommendation(rec: dict[str, Any], floor: dict[str, Any]) -> list[str]:
    ci = ""
    if rec.get("ci_low") is not None:
        ci = f" (95% CI {rec['ci_low']:.3f}–{rec['ci_high']:.3f})"
    lines = [
        "## Recommended default",
        "",
        f"**`{rec['config_id']}`** — mean regret "
        f"**{rec['mean_regret']:.3f}** judge points{ci}, over "
        f"{rec['n_panel_cells']} cells and {len(rec['compared_configs'])} configs.",
        "",
        "Regret is how much judge score this config gives up against the best "
        "config in the same cell. It is the price of using one default "
        "everywhere instead of tuning per project — and therefore the number "
        "that says whether skipping the sweep on a first run is defensible.",
        "",
    ]
    basis = floor.get("basis")
    if basis == "seed_replicates":
        groups = floor["n_replicate_groups"]
        lines += [
            f"Noise floor: **{floor['value']:.3f}** judge points, from "
            f"{groups} seed-replicate {'group' if groups == 1 else 'groups'} "
            f"(mean range {_fmt(floor.get('seed_range'))}, pooled sd "
            f"{_fmt(floor.get('seed_std'))}). Differences smaller than this are "
            "not real.",
            "",
        ]
    elif basis == "judge_sem":
        lines += [
            f"> **Noise floor is a lower bound only: {floor['value']:.3f} judge "
            f"points** (2 × median judge SEM "
            f"{_fmt(floor.get('median_judge_sem'))}). No seed replicates were "
            "run, so this captures the judge's sampling error but **not** "
            "training variance — usually the larger term. A per-dimension "
            "split that clears this floor may still be noise. Re-run with "
            "`--replicate-seeds 1,2,3` before acting on one.",
            "",
        ]
    else:
        lines += [
            "> **No noise floor at all.** Neither seed replicates nor judge "
            "score spread were available, so every split below rests on the "
            "bootstrap CI alone. Re-run with `--replicate-seeds 1,2,3`.",
            "",
        ]
    if rec.get("excluded_configs"):
        lines += [
            f"Excluded from the comparison (not measured on these cells): "
            f"{', '.join('`' + c + '`' for c in rec['excluded_configs'])}.",
            "",
        ]
    return lines


def _render_ranking(rec: dict[str, Any]) -> list[str]:
    lines = [
        "## All configs",
        "",
        "| config | mean regret | worst regret | mean rank | wins | cells | mean train s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rec.get("ranking", []):
        lines.append(
            f"| `{row['config_id']}` | {row['mean_regret']:.3f} | "
            f"{row['max_regret']:.3f} | {row['mean_rank']:.2f} | {row['wins']} | "
            f"{row['n_cells']} | {_fmt(row.get('mean_elapsed_s'), '.0f')} |"
        )
    lines.append("")
    lines.append(
        "Worst-case regret matters as much as the mean for a default: a config "
        "that is usually fine and occasionally catastrophic is a bad default "
        "even with a good average."
    )
    lines.append("")
    return lines


def _render_conditional(results: dict[str, Any], rec: dict[str, Any]) -> list[str]:
    findings = results.get("conditional_defaults", [])
    lines = [
        "## Does any dimension need its own default?",
        "",
        "`gain` is what a per-level default would buy over the global one, in "
        "judge points. A split is only recommended when the gain clears the "
        "measured noise floor **and** its bootstrap CI excludes zero — a rule "
        "nobody can reproduce is worse than no rule.",
        "",
    ]
    split = [f for f in findings if f["split_recommended"]]
    if split:
        lines.append(
            "**Verdict: separate defaults are justified for "
            + ", ".join(f"`{f['dimension']}`" for f in split)
            + ".**"
        )
    else:
        lines.append(
            f"**Verdict: one default (`{rec['config_id']}`) is enough.** No "
            "dimension showed a per-level gain that survives the noise floor "
            "and the CI."
        )
    lines += ["", "| dimension | level | best here | gain | 95% CI | cells | real? |",
              "|---|---|---|---:|---|---:|---|"]
    for finding in findings:
        if not finding["levels"]:
            lines.append(
                f"| `{finding['dimension']}` | — | — | — | — | — | too few cells |"
            )
        for level in finding["levels"]:
            lines.append(
                f"| `{finding['dimension']}` | {level['level']} | "
                f"`{level['best_config']}` | {level['gain']:+.3f} | "
                f"{level['ci_low']:+.3f}–{level['ci_high']:+.3f} | "
                f"{level['n_cells']} | {'**yes**' if level['beats_noise'] else 'no'} |"
            )
    lines.append("")
    return lines


def _render_proxy(results: dict[str, Any]) -> list[str]:
    proxy = results.get("proxy_validation", {})
    if not proxy.get("n_cells"):
        return []
    return [
        "## Proxy check (free by-product)",
        "",
        "The shipped sweep picks its winner on `eval_loss` alone. Because this "
        "study judged every config, it can check that assumption directly.",
        "",
        f"- Cells analysed: **{proxy['n_cells']}**",
        f"- Mean Spearman ρ(eval_loss, judge): **{_fmt(proxy.get('mean_spearman'))}** "
        "(strongly negative is what we want)",
        f"- Top-1 agreement (lowest eval_loss == highest judge): "
        f"**{_fmt_pct(proxy.get('top1_hit_rate'))}**",
        "",
    ]


def _render_epochs(results: dict[str, Any]) -> list[str]:
    epochs = results.get("epochs", {})
    if not epochs.get("n_runs"):
        return []
    lines = [
        "## Is the epochs axis doing anything?",
        "",
        "Training already keeps the best checkpoint by eval loss "
        "(`load_best_model_at_end`), so a 3-epoch run can save an epoch-1 "
        "model. Where that happens, `num_epochs` is a ceiling, not a tuned "
        "quantity — and the product's sweep grid could drop the axis.",
        "",
        f"- Runs with epoch data: **{epochs['n_runs']}**",
        f"- Saved a checkpoint from before the last epoch: "
        f"**{_fmt_pct(epochs.get('early_stop_rate'))}**",
        "",
        "| num_epochs | runs | stopped early | mean best epoch |",
        "|---:|---:|---:|---:|",
    ]
    for row in epochs.get("by_num_epochs", []):
        lines.append(
            f"| {row['num_epochs']} | {row['n_runs']} | {row['n_stopped_early']} | "
            f"{row['mean_best_epoch']:.2f} |"
        )
    lines.append("")
    return lines


def _render_snippet(rec: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    lr, epochs = _parse_config_id(rec["config_id"])
    return [
        "## Install this",
        "",
        "Paste into `lqh/train/defaults.py` and update `PROVENANCE` in the "
        "same commit, then re-run `tests/unit/test_train_defaults.py` — the "
        "parity test will fail until it is updated to match, which is the "
        "point: a shipped default should never move silently.",
        "",
        "```python",
        f"PROVENANCE = (",
        f'    "hp_defaults study {meta.get("run_name", "?")}: '
        f'{rec["config_id"]}, mean regret {rec["mean_regret"]:.3f} judge points "',
        f'    "over {rec["n_panel_cells"]} cells "',
        f'    "(task x dataset size x base/instruct x model size)."',
        f")",
        "",
        f"# in recommended():",
        f"learning_rate = {lr if lr is not None else '...'}   # was 2e-5",
        f"num_epochs = {epochs if epochs is not None else '...'}",
        "```",
        "",
    ]


def _render_notes(results: dict[str, Any]) -> list[str]:
    notes = results.get("notes") or []
    if not notes:
        return []
    lines = ["## Jobs that did not complete", "",
             "Their configs are excluded from the comparison above.", ""]
    lines += [f"- {n}" for n in notes]
    lines.append("")
    return lines


def _parse_config_id(config_id: str) -> tuple[str | None, int | None]:
    """Pull lr / epochs back out of an id like ``lr5e-05_e2``."""
    lr = epochs = None
    for part in config_id.split("_"):
        if part.startswith("lr"):
            lr = part[2:]
        elif part.startswith("e") and part[1:].isdigit():
            epochs = int(part[1:])
    return lr, epochs


def _fmt(value: Any, spec: str = ".3f") -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return format(value, spec)


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value * 100:.0f}%"
