"""Aggregate benchmark report: models x categories matrix."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tests.e2e.benchmark.scoring import BenchmarkScore


def generate_aggregate_report(
    scores: list[BenchmarkScore],
    output_dir: Path,
) -> Path:
    """Generate the aggregate benchmark report (markdown + JSON).

    Returns the path to the markdown report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "benchmark_report.md"
    json_path = output_dir / "benchmark_report.json"

    md = _render_markdown(scores)
    md_path.write_text(md, encoding="utf-8")

    json_data = _render_json(scores)
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return md_path


def _render_markdown(scores: list[BenchmarkScore]) -> str:
    lines: list[str] = []

    # Collect models and categories
    models = sorted(set(s.model for s in scores))
    categories = sorted(set(s.category for s in scores))

    # Group scores
    by_model_cat: dict[tuple[str, str], list[BenchmarkScore]] = defaultdict(list)
    for s in scores:
        by_model_cat[(s.model, s.category)].append(s)

    lines.append("# Orchestration Benchmark Report")
    lines.append("")
    lines.append(f"**Models tested:** {', '.join(f'`{m}`' for m in models)}")
    lines.append(f"**Categories:** {len(categories)}")
    lines.append(f"**Total runs:** {len(scores)}")
    total_duration = sum(s.duration_seconds for s in scores)
    lines.append(f"**Total duration:** {total_duration / 3600:.1f} hours")
    lines.append("")

    # === Models x Categories Matrix ===
    lines.append("## Models x Categories Matrix")
    lines.append("")

    # Header row
    header = "| Category |"
    sep = "|----------|"
    for m in models:
        header += f" `{m}` |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    # Category rows
    model_totals: dict[str, list[float]] = defaultdict(list)
    model_catastrophic: dict[str, int] = defaultdict(int)

    for cat in categories:
        row = f"| {cat} |"
        for m in models:
            cat_scores = by_model_cat.get((m, cat), [])
            if not cat_scores:
                row += " - |"
                continue
            avg = sum(s.composite_score for s in cat_scores) / len(cat_scores)
            has_catastrophic = any(s.is_catastrophic_failure for s in cat_scores)
            flag = " :warning:" if has_catastrophic else ""
            row += f" {avg:.0f}{flag} |"
            model_totals[m].append(avg)
            if has_catastrophic:
                model_catastrophic[m] += sum(1 for s in cat_scores if s.is_catastrophic_failure)
        lines.append(row)

    # Overall row
    row = "| **Overall** |"
    for m in models:
        totals = model_totals.get(m, [])
        if totals:
            overall = sum(totals) / len(totals)
            row += f" **{overall:.0f}** |"
        else:
            row += " - |"
    lines.append(row)
    lines.append("")

    # === Worst-Case Breakdown ===
    lines.append("## Worst-Case Breakdown")
    lines.append("")
    for m in models:
        catastrophic = [s for s in scores if s.model == m and s.is_catastrophic_failure]
        if catastrophic:
            lines.append(f"### `{m}` ({len(catastrophic)} catastrophic failures)")
            for s in catastrophic:
                details = ""
                if s.category == "error_recovery":
                    details = f" (fix_attempts={s.category_details.get('fix_attempts', '?')})"
                elif s.category == "datagen_pipeline":
                    details = f" (ran={s.category_details.get('pipeline_ran_successfully', False)})"
                elif s.category == "next_steps":
                    details = f" (expected={s.category_details.get('expected_next_step', '?')}, actual={s.category_details.get('actual_next_step', '?')})"
                lines.append(f"- `{s.category}/{s.scenario_name}`: score={s.composite_score:.0f}{details}")
            lines.append("")
        else:
            lines.append(f"### `{m}` (no catastrophic failures)")
            lines.append("")

    # === Per-Scenario Detail ===
    lines.append("## Per-Scenario Detail")
    lines.append("")

    for cat in categories:
        lines.append(f"### {cat}")
        lines.append("")

        # Get all scenarios in this category
        cat_scenario_names = sorted(set(
            s.scenario_name for s in scores if s.category == cat
        ))

        header = "| Scenario |"
        sep = "|----------|"
        for m in models:
            header += f" `{m}` |"
            sep += "---------|"
        lines.append(header)
        lines.append(sep)

        for sname in cat_scenario_names:
            row = f"| {sname} |"
            for m in models:
                matching = [s for s in scores if s.model == m and s.scenario_name == sname]
                if not matching:
                    row += " - |"
                    continue
                s = matching[0]
                flag = " :x:" if s.is_catastrophic_failure else ""
                row += f" {s.composite_score:.0f}{flag} |"
            lines.append(row)
        lines.append("")

    # === Recommendation ===
    lines.append("## Recommendation")
    lines.append("")

    if model_totals:
        # Best overall
        best_overall = max(model_totals.items(), key=lambda x: sum(x[1]) / len(x[1]))
        lines.append(f"**Best overall:** `{best_overall[0]}` (avg {sum(best_overall[1]) / len(best_overall[1]):.0f}/100)")

        # Fewest catastrophic failures
        if model_catastrophic:
            fewest_catastrophic = min(model_catastrophic.items(), key=lambda x: x[1])
            lines.append(f"**Fewest catastrophic failures:** `{fewest_catastrophic[0]}` ({fewest_catastrophic[1]} failures)")
        else:
            lines.append("**No catastrophic failures detected across any model.**")

        # Check if they differ
        if model_catastrophic:
            best_cat = min(model_catastrophic.items(), key=lambda x: x[1])
            if best_cat[0] != best_overall[0]:
                lines.append("")
                lines.append(
                    f"> Note: `{best_overall[0]}` has the highest average score but "
                    f"`{best_cat[0]}` has fewer catastrophic failures. "
                    f"If avoiding worst-case failures is critical, prefer `{best_cat[0]}`."
                )

    lines.append("")
    return "\n".join(lines)


def _render_json(scores: list[BenchmarkScore]) -> dict:
    models = sorted(set(s.model for s in scores))
    categories = sorted(set(s.category for s in scores))

    by_model_cat: dict[tuple[str, str], list[BenchmarkScore]] = defaultdict(list)
    for s in scores:
        by_model_cat[(s.model, s.category)].append(s)

    matrix = {}
    for m in models:
        matrix[m] = {}
        for cat in categories:
            cat_scores = by_model_cat.get((m, cat), [])
            if cat_scores:
                matrix[m][cat] = {
                    "avg_composite": round(sum(s.composite_score for s in cat_scores) / len(cat_scores), 1),
                    "catastrophic_failures": sum(1 for s in cat_scores if s.is_catastrophic_failure),
                    "scenarios": len(cat_scores),
                }

    return {
        "models": models,
        "categories": categories,
        "total_runs": len(scores),
        "total_duration_hours": round(sum(s.duration_seconds for s in scores) / 3600, 2),
        "matrix": matrix,
        "scores": [s.to_dict() for s in scores],
    }
