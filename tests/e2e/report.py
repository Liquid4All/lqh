"""Generate human-readable reports from E2E test results."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tests.e2e.harness import E2EResult

REPORTS_DIR = Path(__file__).parent / "reports"


def generate_report(result: E2EResult) -> Path:
    """Generate a Markdown report and JSON export for an E2E result.

    Returns the path to the Markdown report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_tag = result.orchestration_model.replace(":", "_")
    base = f"{result.scenario_name}_{model_tag}_{timestamp}"

    md_path = REPORTS_DIR / f"{base}.md"
    json_path = REPORTS_DIR / f"{base}.json"

    # Generate Markdown
    md = _render_markdown(result)
    md_path.write_text(md, encoding="utf-8")

    # Generate JSON
    json_data = _render_json(result)
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return md_path


def _render_markdown(result: E2EResult) -> str:
    lines: list[str] = []

    lines.append(f"# E2E Test Report: {result.scenario_name}")
    lines.append("")

    # --- Executive summary ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Orchestration model | `{result.orchestration_model}` |")
    lines.append(f"| Duration | {result.duration_seconds:.1f}s |")
    lines.append(f"| User turns | {result.total_turns} |")
    lines.append(f"| Tool calls | {result.total_tool_calls} |")
    lines.append(f"| Skills loaded | {', '.join(result.skills_loaded) or 'none'} |")
    lines.append(f"| Errors | {len(result.errors)} |")

    artifacts = result.artifacts
    lines.append(f"| Artifacts created | {len(artifacts)} |")
    lines.append(f"| SPEC.md | {'✅' if 'SPEC.md' in artifacts else '❌'} |")

    scorer_exists = any("evals/scorers" in p for p in artifacts)
    lines.append(f"| Scorer | {'✅' if scorer_exists else '❌'} |")

    pa = result.pipeline_attempts
    if pa["total"] > 0:
        lines.append(f"| Pipeline runs | {pa['succeeded']}/{pa['total']} succeeded, {pa['failed']} failed |")

    cs = result.context_stats
    if cs and cs.turns:
        s = cs.summary()
        lines.append(f"| Peak prompt tokens | {s['peak_prompt_tokens']:,} |")
        lines.append(f"| Total tokens used | {s['total_tokens']:,} |")
        lines.append(f"| Context compactions | {s['compactions']} |")

    lines.append("")

    # Scenario
    lines.append("## Scenario")
    lines.append(f"> {result.scenario_description}")
    lines.append("")

    # Errors
    if result.errors:
        lines.append("## Errors")
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    # Warnings (non-fatal issues like single compactions)
    if getattr(result, "warnings", None):
        lines.append("## Warnings")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Tool usage breakdown
    lines.append("## Tool Usage")
    lines.append("")
    tool_counts: dict[str, int] = {}
    for t in result.transcript:
        if t.role == "tool_call" and t.tool_name:
            tool_counts[t.tool_name] = tool_counts.get(t.tool_name, 0) + 1
    if tool_counts:
        lines.append("| Tool | Calls |")
        lines.append("|------|-------|")
        for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{name}` | {count} |")
    lines.append("")

    # Context stats
    if result.context_stats and result.context_stats.turns:
        lines.append("## Context Usage")
        lines.append("")
        lines.append(result.context_stats.format_report())
        lines.append("")

    # Transcript
    lines.append("## Conversation Transcript")
    lines.append("")

    turn_num = 0
    for rec in result.transcript:
        if rec.role == "user":
            turn_num += 1
            lines.append(f"### Turn {turn_num} — User")
            lines.append(f"> {rec.content}")
            lines.append("")
        elif rec.role == "agent":
            lines.append("### Agent")
            # Truncate long agent messages
            content = rec.content
            if len(content) > 1000:
                content = content[:1000] + "\n\n*[truncated]*"
            lines.append(content)
            lines.append("")
        elif rec.role == "tool_call":
            lines.append(f"**🔧 Tool:** `{rec.tool_name}`")
            if rec.tool_args:
                args_str = json.dumps(rec.tool_args, ensure_ascii=False)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                lines.append(f"```json\n{args_str}\n```")
            lines.append("")
        elif rec.role == "tool_result":
            content = rec.content
            if len(content) > 500:
                content = content[:500] + "\n*[truncated]*"
            lines.append(f"<details><summary>Result ({rec.tool_name})</summary>\n\n```\n{content}\n```\n</details>")
            lines.append("")
        elif rec.role == "ask_user_q":
            lines.append(f"**❓ Agent asks (ask_user):** {rec.content}")
            lines.append("")
        elif rec.role == "ask_user_a":
            lines.append(f"**💬 Simulated user:** {rec.content}")
            lines.append("")
        elif rec.role == "chat_q":
            lines.append(f"**💬❓ Agent asks (chat):** {rec.content}")
            lines.append("")
        elif rec.role == "chat_a":
            lines.append(f"**💬 Simulated user (chat reply):** {rec.content}")
            lines.append("")
        elif rec.role == "skill_loaded":
            lines.append(f"**⚡ Skill loaded:** `{rec.content}`")
            lines.append("")

    # Artifacts
    lines.append("## Artifacts Created")
    lines.append("")

    for path, content in sorted(artifacts.items()):
        lines.append(f"### {path}")
        if content.startswith("<binary"):
            lines.append(f"*{content}*")
        else:
            preview = content
            if len(preview) > 2000:
                preview = preview[:2000] + "\n\n*[truncated]*"
            ext = path.rsplit(".", 1)[-1] if "." in path else ""
            lines.append(f"```{ext}\n{preview}\n```")
        lines.append("")

    return "\n".join(lines)


def _render_json(result: E2EResult) -> dict:
    return {
        "scenario": result.scenario_name,
        "description": result.scenario_description,
        "orchestration_model": result.orchestration_model,
        "context_stats": result.context_stats.summary() if result.context_stats else {},
        "pipeline_attempts": result.pipeline_attempts,
        "duration_seconds": round(result.duration_seconds, 2),
        "total_turns": result.total_turns,
        "total_tool_calls": result.total_tool_calls,
        "skills_loaded": result.skills_loaded,
        "errors": result.errors,
        "tools_called": sorted(result.tools_called()),
        "artifacts": {k: v[:5000] for k, v in result.artifacts.items()},
        "transcript": [
            {
                "role": rec.role,
                "content": rec.content[:2000],
                "tool_name": rec.tool_name,
            }
            for rec in result.transcript
        ],
    }
