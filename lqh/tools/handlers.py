"""Tool execution dispatch and handlers for lqh agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from lqh.skills import list_available_skills, load_skill_content


# Truncation threshold: ~40,000 chars (~10k tokens)
TRUNCATION_THRESHOLD = 40_000


@dataclass
class ToolResult:
    """Result from a tool execution."""
    content: str
    requires_user_input: bool = False
    question: str | None = None
    options: list[str] | None = None
    multi_select: bool = False
    show_file_path: str | None = None
    skill_content: str | None = None
    # Auto-mode signals. The agent loop checks these after each tool call.
    exit_auto_mode: bool = False
    auto_status: str | None = None  # "success" | "failure"
    auto_reason: str | None = None
    auto_stage: str | None = None
    auto_stage_note: str | None = None


def _validate_path(project_dir: Path, rel_path: str) -> Path:
    """Validate and resolve a path within the project directory."""
    resolved = (project_dir / rel_path).resolve()
    project_resolved = project_dir.resolve()
    try:
        resolved.relative_to(project_resolved)
    except ValueError as exc:
        raise ValueError(f"Path '{rel_path}' is outside the project directory") from exc
    return resolved


def _truncate_content(content: str, offset: int = 0) -> tuple[str, bool]:
    """Truncate content if it exceeds the threshold."""
    lines = content.split("\n")
    total_lines = len(lines)

    if offset > 0:
        lines = lines[offset:]

    result = "\n".join(lines)
    if len(result) <= TRUNCATION_THRESHOLD:
        return result, False

    # Truncate
    truncated = ""
    line_count = 0
    for line in lines:
        if len(truncated) + len(line) + 1 > TRUNCATION_THRESHOLD:
            break
        truncated += line + "\n"
        line_count += 1

    shown_start = offset + 1
    shown_end = offset + line_count
    footer = (
        f"\n[truncated: showing lines {shown_start}-{shown_end} of {total_lines} "
        f"total lines. Use offset={shown_end} to continue reading.]"
    )
    return truncated + footer, True


def _parquet_metadata(path: Path) -> tuple[int | None, int]:
    """Read parquet file metadata (row count) without loading data into memory."""
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        return meta.num_rows, path.stat().st_size
    except Exception:
        return None, path.stat().st_size


def _format_score_distribution(scores_path: Path) -> str:
    """Build a short distribution summary of judge scores for a tool result.

    Reads ``scores_path`` (a parquet with a ``score`` column written by
    ``run_scoring`` or ``run_data_filter``) and returns 4-6 lines of
    quantiles and a coarse histogram. The agent reads this in its
    conversation context and can reason about whether the data is
    bimodal, uniformly mediocre, or has a strong mode at the top —
    information that mean/median alone hide.

    Returns ``""`` if the parquet is missing, has no rows, or has no
    score column. The caller appends the result to its tool output.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return ""
    if not scores_path.exists():
        return ""
    try:
        table = pq.read_table(str(scores_path))
        if "score" not in table.column_names:
            return ""
        scores = [s for s in table.column("score").to_pylist() if s is not None and s > 0]
    except Exception:
        return ""
    if not scores:
        return ""

    scores = sorted(scores)
    n = len(scores)

    def _q(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return scores[idx]

    # Coarse histogram: scores are integers 1-10. We bucket by floor.
    buckets = {b: 0 for b in range(1, 11)}
    for s in scores:
        b = max(1, min(10, int(s)))
        buckets[b] += 1

    # Render as a horizontal mini-histogram. Width scales to max bucket.
    max_count = max(buckets.values()) if buckets else 1
    bar_width = 24
    lines: list[str] = []
    lines.append("  Score distribution (n={}):".format(n))
    lines.append(
        "    p10={:.1f}  p25={:.1f}  p50={:.1f}  p75={:.1f}  p90={:.1f}".format(
            _q(0.10), _q(0.25), _q(0.50), _q(0.75), _q(0.90)
        )
    )
    for b in range(10, 0, -1):
        c = buckets[b]
        if c == 0:
            continue
        bar = "█" * max(1, int(round(c / max_count * bar_width)))
        pct = 100.0 * c / n
        lines.append(f"    {b:>2} | {bar:<{bar_width}}  {c:>5}  ({pct:4.1f}%)")
    return "\n".join(lines)


def _fmt_size(size: int) -> str:
    """Format a file size as a human-readable string."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


async def handle_summary(project_dir: Path, **kwargs: Any) -> ToolResult:
    """Give a summary of the project state."""
    parts: list[str] = []
    parts.append(f"## Project: {project_dir.name}")
    parts.append(f"**Directory:** {project_dir}\n")

    # Check for SPEC.md
    spec = project_dir / "SPEC.md"
    if spec.exists():
        stat = spec.stat()
        parts.append(f"- **SPEC.md**: {stat.st_size} bytes, modified {datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()}")
    else:
        parts.append("- **SPEC.md**: not found (new project)")

    # Other specs
    other_specs = project_dir / "other_specs"
    if other_specs.is_dir():
        specs = list(other_specs.iterdir())
        if specs:
            parts.append(f"- **other_specs/**: {len(specs)} file(s)")
            for s in specs[:10]:
                parts.append(f"  - {s.name}")

    # Data gen pipelines
    data_gen = project_dir / "data_gen"
    if data_gen.is_dir():
        scripts = list(data_gen.glob("*.py"))
        parts.append(f"- **data_gen/**: {len(scripts)} pipeline(s)")
        for s in scripts[:10]:
            parts.append(f"  - {s.name}")

    # Datasets
    datasets_dir = project_dir / "datasets"
    if datasets_dir.is_dir():
        datasets = sorted(
            [d for d in datasets_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        parts.append(f"- **datasets/**: {len(datasets)} dataset(s)")
        for d in datasets[:15]:
            parquet_files = list(d.glob("*.parquet"))
            if not parquet_files:
                parts.append(f"  - {d.name}: (empty)")
                continue
            # Use parquet metadata for fast row count without loading data
            ds_info = []
            for pf in parquet_files:
                row_count, file_size = _parquet_metadata(pf)
                if row_count is not None:
                    ds_info.append(f"{pf.name}: {row_count:,} rows, {_fmt_size(file_size)}")
                else:
                    ds_info.append(f"{pf.name}: {_fmt_size(pf.stat().st_size)}")
            is_draft = d.name.endswith("_draft")
            is_eval = d.name.endswith("_eval")
            label = " (draft)" if is_draft else " (eval)" if is_eval else ""

            # Check for co-located scores
            scores_file = d / "scores.parquet"
            score_info = ""
            if scores_file.exists():
                try:
                    import pyarrow.parquet as pq
                    st = pq.read_table(scores_file, columns=["score"])
                    score_vals = [s.as_py() for s in st.column("score") if s.as_py() and s.as_py() > 0]
                    if score_vals:
                        avg = sum(score_vals) / len(score_vals)
                        score_info = f", scored ✓ (avg {avg:.1f}/10)"
                    else:
                        score_info = ", scored ✓"
                except Exception:
                    score_info = ", scored ✓"

            mtime = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            parts.append(f"  - {d.name}{label}: {', '.join(ds_info)}{score_info} [{mtime}]")

    # Runs
    runs_dir = project_dir / "runs"
    if runs_dir.is_dir():
        runs = [d for d in runs_dir.iterdir() if d.is_dir()]
        parts.append(f"- **runs/**: {len(runs)} run(s)")
        for r in runs[:10]:
            parts.append(f"  - {r.name}")

    # Evals
    evals_dir = project_dir / "evals"
    if evals_dir.is_dir():
        # Scorers
        scorers_dir = evals_dir / "scorers"
        if scorers_dir.is_dir():
            scorer_files = list(scorers_dir.glob("*.md"))
            if scorer_files:
                parts.append(f"- **evals/scorers/**: {len(scorer_files)} scorer(s)")
                for sf in scorer_files[:10]:
                    parts.append(f"  - {sf.name}")

        # Eval runs
        runs_dir_evals = evals_dir / "runs"
        if runs_dir_evals.is_dir():
            eval_runs = sorted(
                [d for d in runs_dir_evals.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if eval_runs:
                parts.append(f"- **evals/runs/**: {len(eval_runs)} run(s)")
                for er in eval_runs[:10]:
                    summary_file = er / "summary.json"
                    if summary_file.exists():
                        try:
                            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                            scores = summary_data.get("scores", {})
                            mean = scores.get("mean", "?")
                            n = summary_data.get("num_samples", "?")
                            parts.append(f"  - {er.name}: mean {mean}/10 ({n} samples)")
                        except (json.JSONDecodeError, KeyError):
                            parts.append(f"  - {er.name}")
                    else:
                        parts.append(f"  - {er.name} (no summary)")

    # Recent conversations
    convos_dir = project_dir / ".lqh" / "conversations"
    if convos_dir.is_dir():
        sessions = sorted(convos_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        parts.append(f"\n- **Conversations**: {len(sessions)} session(s)")
        for s in sessions[:5]:
            try:
                first_line = s.read_text().split("\n")[0]
                meta = json.loads(first_line)
                preview = meta.get("preview", "")[:60]
                parts.append(f"  - {meta.get('created_at', '?')}: {preview}")
            except (json.JSONDecodeError, IndexError):
                parts.append(f"  - {s.stem}")

    return ToolResult(content="\n".join(parts))


async def handle_list_files(project_dir: Path, *, path: str = ".", **kwargs: Any) -> ToolResult:
    """List files and directories within the project."""
    target = _validate_path(project_dir, path)
    if not target.exists():
        # Make the error actionable: otherwise reasoning models have been
        # observed calling list_files on the same missing path repeatedly,
        # reasoning themselves into the same wrong answer. Telling them the
        # natural next step (create_file, which auto-creates parents) breaks
        # the loop.
        return ToolResult(content=(
            f"Path '{path}' does not exist yet. "
            f"This is normal for a fresh project. If you want to write a "
            f"file under this path, just call create_file with the full "
            f"path — parent directories are created automatically. "
            f"Do NOT call list_files on this path again; the answer will "
            f"be the same until something creates it."
        ))
    if not target.is_dir():
        return ToolResult(content=f"Error: '{path}' is not a directory")

    entries: list[str] = []
    for item in sorted(target.iterdir()):
        if item.name.startswith(".") and item.name != ".lqh":
            continue
        stat = item.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        if item.is_dir():
            entries.append(f"  {item.name}/  (dir)  {mtime}")
        else:
            size = stat.st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            entries.append(f"  {item.name}  {size_str}  {mtime}")

    if not entries:
        return ToolResult(content=f"Directory '{path}' is empty")

    header = f"Contents of {path}/ ({len(entries)} items):\n"
    return ToolResult(content=header + "\n".join(entries))


async def handle_read_file(
    project_dir: Path,
    *,
    path: str,
    offset: int = 0,
    limit: int | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Read file contents with truncation support."""
    target = _validate_path(project_dir, path)
    if not target.exists():
        return ToolResult(content=f"Error: file '{path}' does not exist")
    if target.is_dir():
        return ToolResult(content=f"Error: '{path}' is a directory, use list_files instead")

    # Handle parquet files
    if target.suffix == ".parquet":
        return await _read_parquet(target)

    # Read text file
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(content=f"Error: '{path}' is not a text file")

    lines = text.split("\n")
    total_lines = len(lines)

    if offset > 0:
        lines = lines[offset:]
    if limit is not None:
        lines = lines[:limit]

    content = "\n".join(lines)
    content, truncated = _truncate_content(content)

    if not truncated and offset == 0:
        header = f"File: {path} ({total_lines} lines)\n\n"
    else:
        start = offset + 1
        end = offset + len(content.split("\n"))
        header = f"File: {path} (showing lines {start}-{end} of {total_lines})\n\n"

    return ToolResult(content=header + content)


async def _read_parquet(path: Path) -> ToolResult:
    """Read a parquet file and render as text."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return ToolResult(content="Error: pyarrow not installed")

    table = pq.read_table(path)
    total_rows = len(table)
    schema_str = str(table.schema)

    # Show first 20 rows
    preview_rows = min(20, total_rows)
    preview = table.slice(0, preview_rows).to_pandas().to_string()

    content = (
        f"Parquet file: {path.name}\n"
        f"Total rows: {total_rows}\n\n"
        f"Schema:\n{schema_str}\n\n"
        f"First {preview_rows} rows:\n{preview}"
    )

    if total_rows > preview_rows:
        content += f"\n\n[Showing {preview_rows} of {total_rows} rows. Use offset={preview_rows} to see more.]"

    return ToolResult(content=content)


async def handle_create_file(project_dir: Path, *, path: str, content: str, **kwargs: Any) -> ToolResult:
    """Create a new file. Fails if it already exists."""
    target = _validate_path(project_dir, path)
    if target.exists():
        return ToolResult(content=f"Error: file '{path}' already exists. Use write_file to overwrite.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    lines = content.count("\n") + 1
    return ToolResult(content=f"✅ Created {path} ({lines} lines, {len(content):,} chars)")


async def handle_write_file(project_dir: Path, *, path: str, content: str, **kwargs: Any) -> ToolResult:
    """Write/overwrite a file."""
    target = _validate_path(project_dir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    lines = content.count("\n") + 1
    return ToolResult(content=f"✅ Wrote {path} ({lines} lines, {len(content):,} chars)")


async def handle_edit_file(
    project_dir: Path,
    *,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    **kwargs: Any,
) -> ToolResult:
    """Edit a file by string replacement."""
    target = _validate_path(project_dir, path)
    if not target.exists():
        return ToolResult(content=f"Error: file '{path}' does not exist")

    text = target.read_text(encoding="utf-8")

    if old_string not in text:
        return ToolResult(content=f"Error: old_string not found in '{path}'")

    if not replace_all:
        count = text.count(old_string)
        if count > 1:
            return ToolResult(
                content=f"Error: old_string found {count} times in '{path}'. "
                "Use replace_all=true or provide a more specific string."
            )
        text = text.replace(old_string, new_string, 1)
    else:
        text = text.replace(old_string, new_string)

    target.write_text(text, encoding="utf-8")
    return ToolResult(content=f"✅ Edited {path}")


async def handle_run_data_gen_pipeline(
    project_dir: Path,
    *,
    script_path: str,
    num_samples: int,
    output_dataset: str,
    validation_instructions: str | None = None,
    samples_per_item: int = 1,
    **kwargs: Any,
) -> ToolResult:
    """Execute a data generation pipeline. Requires user permission."""
    from lqh.tools.permissions import check_permission

    target = _validate_path(project_dir, script_path)
    if not target.exists():
        return ToolResult(content=f"Error: script '{script_path}' does not exist")

    # Pre-validate imports before executing
    try:
        source = target.read_text(encoding="utf-8")
        bad_imports = [
            ("from data_gen.", "from data_gen."),
            ("from data_gen import", "from data_gen import"),
            ("from pipeline import", "from pipeline import"),
            ("import pipeline\n", "import pipeline"),
        ]
        for pattern, display in bad_imports:
            if pattern in source:
                return ToolResult(
                    content=(
                        f"Error: Pipeline has incorrect import: `{display}`\n"
                        f"Fix: use `from lqh.pipeline import Pipeline, ChatMLMessage, Conversation`\n"
                        f"All pipeline imports must come from `lqh.pipeline`, not `data_gen` or `pipeline`."
                    )
                )
    except OSError:
        pass

    # Check if we already have permission
    if not check_permission(project_dir, script_path):
        # Need to ask for permission - this will be handled by the agent loop
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            question=(
                f"The agent wants to execute the pipeline script:\n"
                f"  {script_path}\n"
                f"  Samples: {num_samples}\n"
                f"  Output: datasets/{output_dataset}/\n\n"
                f"Allow execution?"
            ),
            options=[
                "Execute once, ask again next time",
                "Execute and don't ask again for this file",
                "Execute and don't ask again for this project",
                "Do not execute",
            ],
        )

    # Execute the pipeline (pass through any callbacks from kwargs)
    return await _execute_pipeline(
        project_dir, script_path, num_samples, output_dataset, validation_instructions,
        samples_per_item=samples_per_item,
        on_pipeline_progress=kwargs.get("on_pipeline_progress"),
        on_pipeline_done=kwargs.get("on_pipeline_done"),
    )


async def _execute_pipeline(
    project_dir: Path,
    script_path: str,
    num_samples: int,
    output_dataset: str,
    validation_instructions: str | None,
    *,
    samples_per_item: int = 1,
    on_pipeline_progress: Callable | None = None,
    on_pipeline_done: Callable | None = None,
) -> ToolResult:
    """Actually execute the pipeline after permission is granted."""
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline

    concurrency = min(5, num_samples)

    # Signal start immediately
    if on_pipeline_progress:
        on_pipeline_progress(0, num_samples, concurrency)

    try:
        config = load_config()
        token = require_token()
        client = create_client(token, config.api_base_url)

        target = _validate_path(project_dir, script_path)
        output_dir = project_dir / "datasets" / output_dataset

        val_text: str | None = None
        if validation_instructions:
            val_path = _validate_path(project_dir, validation_instructions)
            val_text = val_path.read_text(encoding="utf-8")

        def on_progress(completed: int, total: int) -> None:
            if on_pipeline_progress:
                on_pipeline_progress(completed, total, concurrency)

        result = await run_pipeline(
            script_path=target,
            num_samples=num_samples,
            output_dir=output_dir,
            client=client,
            concurrency=concurrency,
            samples_per_item=samples_per_item,
            validation_instructions=val_text,
            on_progress=on_progress,
        )

        from lqh.project_log import append_event, file_hash_prefix

        append_event(
            project_dir,
            "data_gen_completed",
            f"Generated {output_dataset} ({result.succeeded}/{result.total} ok)",
            script_path=script_path,
            script_hash=file_hash_prefix(project_dir / script_path),
            output_dataset=output_dataset,
            num_samples=num_samples,
            succeeded=result.succeeded,
            failed=result.failed,
        )

        return ToolResult(
            content=(
                f"✅ Pipeline completed\n"
                f"  Samples: {result.succeeded}/{result.total} succeeded"
                + (f", {result.failed} failed" if result.failed else "")
                + f"\n  Output:  {result.output_path}"
            )
        )
    except Exception as e:
        import traceback

        from lqh.project_log import append_event, file_hash_prefix

        append_event(
            project_dir,
            "data_gen_failed",
            f"Pipeline {script_path} failed: {type(e).__name__}: {e}",
            script_path=script_path,
            script_hash=file_hash_prefix(project_dir / script_path),
            output_dataset=output_dataset,
            num_samples=num_samples,
            error=f"{type(e).__name__}: {e}",
        )

        tb = traceback.format_exc()
        err_str = str(e)
        hint = ""
        if "list() takes no keyword arguments" in err_str or "Conversation(" in tb:
            hint = (
                "\n\nHint: Conversation is a type alias for list[ChatMLMessage], not a class. "
                "Return a plain list:\n"
                "  return [ChatMLMessage('system', '...'), ChatMLMessage('user', '...'), ChatMLMessage('assistant', '...')]"
            )
        elif "unexpected keyword argument 'input'" in err_str:
            hint = (
                "\n\nHint: The generate() method must accept input as a parameter:\n"
                "  async def generate(self, client, input=None) -> Conversation:"
            )
        return ToolResult(content=f"❌ Pipeline failed: {type(e).__name__}: {e}{hint}\n\n{tb}")
    finally:
        if on_pipeline_done:
            on_pipeline_done()


async def handle_ask_user(
    *, question: str, options: list[str] | None = None, multi_select: bool = False, **kwargs: Any,
) -> ToolResult:
    """Present a question to the user. Handled specially by the agent loop."""
    return ToolResult(
        content="",
        requires_user_input=True,
        question=question,
        options=options,
        multi_select=multi_select,
    )


async def handle_show_file(project_dir: Path, *, path: str, **kwargs: Any) -> ToolResult:
    """Show a file to the user in scrollable view. Returns truncated version to agent."""
    target = _validate_path(project_dir, path)
    if not target.exists():
        return ToolResult(content=f"Error: file '{path}' does not exist")

    # Parquet files: open interactive dataset viewer via TUI callback
    if target.suffix == ".parquet":
        return ToolResult(
            content=f"[Opening interactive dataset viewer for {path}]",
            show_file_path=path,
        )

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(content=f"Error: '{path}' is not a text file")

    lines = text.split("\n")
    total_lines = len(lines)

    # For the agent context, return a summary
    preview_lines = min(50, total_lines)
    preview = "\n".join(lines[:preview_lines])

    summary = f"Displayed {path} to user ({total_lines} lines)"
    if total_lines > preview_lines:
        summary += f"\nFirst {preview_lines} lines:\n{preview}\n[... {total_lines - preview_lines} more lines]"
    else:
        summary += f"\n{preview}"

    return ToolResult(content=summary, show_file_path=path)


async def handle_get_eval_failures(
    project_dir: Path,
    *,
    eval_run: str,
    threshold: float = 6.0,
    min_failures: int = 5,
    max_failures: int = 15,
    **kwargs: Any,
) -> ToolResult:
    """Extract and format failure cases from an eval run."""
    run_dir = _validate_path(project_dir, eval_run)
    results_path = run_dir / "results.parquet"
    if not results_path.exists():
        return ToolResult(content=f"Error: no results.parquet in '{eval_run}'")

    from lqh.scoring import extract_failures

    failures, scoring_errors = extract_failures(
        results_path,
        threshold=threshold,
        min_failures=min_failures,
        max_failures=max_failures,
    )

    if not failures and not scoring_errors:
        return ToolResult(content="No failure cases found. All samples scored above threshold.")

    import pyarrow.parquet as pq_mod

    total = pq_mod.read_metadata(results_path).num_rows

    parts: list[str] = []

    if failures:
        parts.append(
            f"## Failure Cases ({len(failures)} of {total} samples, threshold < {threshold})\n"
        )
        for f in failures:
            parts.append(f"### Sample {f['sample_index']} — Score: {f['score']:.1f}/10")
            parts.append(f"**Judge reasoning:** {f['reasoning']}")
            for msg in f["messages"]:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 500:
                    content = content[:500] + "..."
                parts.append(f"**{role}:** {content}")
            parts.append("")
    else:
        parts.append(
            f"## Failure Cases (0 of {total} samples below threshold {threshold})\n"
        )

    if scoring_errors:
        parts.append("")
        parts.append(
            f"## Scoring Errors ({len(scoring_errors)} samples — could NOT be scored, "
            f"do NOT count as model failures)"
        )
        parts.append(
            "These samples hit a judge-API or parse error. Their score of 0.0 is a "
            "placeholder, not a quality verdict. Re-run scoring on the dataset to get "
            "real scores for them.\n"
        )
        for e in scoring_errors:
            err = e["reasoning"]
            if len(err) > 300:
                err = err[:300] + "..."
            parts.append(f"- **Sample {e['sample_index']}** — {err}")
        parts.append("")

    content = "\n".join(parts)
    content, _ = _truncate_content(content)
    return ToolResult(content=content)


async def handle_list_models(**kwargs: Any) -> ToolResult:
    """List available LFM models from the API."""
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config

    try:
        config = load_config()
        token = require_token()
        client = create_client(token, config.api_base_url)
    except Exception as e:
        return ToolResult(content=f"Error: {e}")

    try:
        models_response = await client.models.list()
        lines = ["Available Liquid Foundation Models:\n"]
        lines.append(f"{'Model ID':<30} {'Display Name':<35} {'Context':<10} {'HuggingFace ID'}")
        lines.append("-" * 110)
        for m in models_response.data:
            model_id = m.id
            display_name = getattr(m, "display_name", m.id)
            context_length = getattr(m, "context_length", "?")
            hf_id = getattr(m, "huggingface_id", "")
            lines.append(f"{model_id:<30} {display_name:<35} {str(context_length):<10} {hf_id}")

        lines.append("")
        lines.append("Additionally, these pool/utility models are always available:")
        lines.append("  small, medium, large          — default model from each size pool")
        lines.append("  random:<size>                  — random model from pool (different each request)")
        lines.append("  random:<size>:<seed>           — deterministic model from pool")
        lines.append("  judge:small, judge:medium, judge:large — dedicated scoring models")
        lines.append("  orchestration                  — frontier agent model with tool calling")

        return ToolResult(content="\n".join(lines))
    except Exception as e:
        return ToolResult(content=f"Error listing models: {e}")


async def handle_list_skills(**kwargs: Any) -> ToolResult:
    """List all available skills/modes."""
    skills = list_available_skills()
    lines = ["Available skills:\n"]
    for s in skills:
        lines.append(f"  {s['command']:12s} {s['description']}")
    return ToolResult(content="\n".join(lines))


async def handle_load_skill(*, skill_name: str, **kwargs: Any) -> ToolResult:
    """Load a skill's SKILL.md into the conversation."""
    try:
        content = load_skill_content(skill_name)
        return ToolResult(
            content=f"⚡ Skill loaded: {skill_name}",
            skill_content=content,
        )
    except FileNotFoundError as e:
        return ToolResult(content=f"Error: {e}")


async def handle_run_scoring(
    project_dir: Path,
    *,
    dataset: str,
    scorer: str,
    mode: str,
    run_name: str | None = None,
    model_size: str = "small",
    inference_model: str | None = None,
    inference_system_prompt: str | None = None,
    system_prompt_path: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Run scoring on a dataset using LLM-as-judge."""
    dataset_dir = _validate_path(project_dir, dataset)
    data_path = dataset_dir / "data.parquet"
    if not data_path.exists():
        return ToolResult(content=f"Error: no data.parquet in '{dataset}'")

    scorer_path = _validate_path(project_dir, scorer)
    if not scorer_path.exists():
        return ToolResult(content=f"Error: scorer '{scorer}' does not exist")

    # Resolve system_prompt_path -> inference_system_prompt if needed
    if system_prompt_path and not inference_system_prompt:
        prompt_file = _validate_path(project_dir, system_prompt_path)
        if not prompt_file.exists():
            return ToolResult(content=f"Error: prompt file '{system_prompt_path}' does not exist")
        inference_system_prompt = prompt_file.read_text(encoding="utf-8")

    # Auto-discover response_format schema from prompt path
    # e.g., prompts/translation_v0.md → prompts/translation.schema.json
    inference_response_format = None
    response_format_path = kwargs.get("response_format_path")
    if response_format_path:
        schema_file = _validate_path(project_dir, response_format_path)
        if not schema_file.exists():
            return ToolResult(content=f"Error: schema file '{response_format_path}' does not exist")
        inference_response_format = json.loads(schema_file.read_text(encoding="utf-8"))
    elif system_prompt_path:
        # Auto-discover: prompts/translation_v0.md → prompts/translation.schema.json
        prompt_stem = Path(system_prompt_path).stem  # "translation_v0"
        task_name = prompt_stem.rsplit("_v", 1)[0]   # "translation"
        auto_schema = Path(system_prompt_path).parent / f"{task_name}.schema.json"
        full_auto = project_dir / auto_schema
        if full_auto.exists():
            inference_response_format = json.loads(full_auto.read_text(encoding="utf-8"))

    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config

    try:
        config = load_config()
        token = require_token()
        client = create_client(token, config.api_base_url)
    except Exception as e:
        return ToolResult(content=f"Error: {e}")

    on_progress = kwargs.get("on_pipeline_progress")

    def _progress(completed: int, total: int) -> None:
        if on_progress:
            on_progress(completed, total, min(5, total))

    try:
        if mode == "data_quality":
            from lqh.scoring import run_data_scoring

            result = await run_data_scoring(
                dataset_dir=dataset_dir,
                scorer_path=scorer_path,
                client=client,
                model_size=model_size,
                on_progress=_progress,
            )

            from lqh.project_log import append_event

            append_event(
                project_dir,
                "scoring_completed",
                f"Scored {dataset} (data_quality) mean={result.mean_score:.1f} median={result.median_score:.1f}",
                dataset=dataset,
                scorer=scorer,
                mode="data_quality",
                mean_score=round(result.mean_score, 2),
                median_score=round(result.median_score, 2),
            )

            distribution = _format_score_distribution(data_path.parent / "scores.parquet")
            return ToolResult(
                content=(
                    f"✅ Data quality scoring complete\n"
                    f"  Dataset: {dataset}\n"
                    f"  Scored: {result.scored}/{result.total}"
                    + (
                        f" ({result.failed} judge errors — could not be scored, "
                        f"not counted in mean/median)" if result.failed else ""
                    )
                    + f"\n  Mean score: {result.mean_score:.1f}/10"
                    f"\n  Median score: {result.median_score:.1f}/10"
                    + (f"\n{distribution}" if distribution else "")
                    + f"\n  Output: {dataset}/scores.parquet"
                )
            )

        elif mode == "model_eval":
            if not run_name:
                return ToolResult(content="Error: run_name is required for mode='model_eval'")
            if not inference_model:
                return ToolResult(
                    content="Error: inference_model is required for mode='model_eval'. "
                    "Use list_models to discover available models."
                )

            from lqh.scoring import JUDGE_MODELS, run_scoring

            output_dir = project_dir / "evals" / "runs" / run_name
            if output_dir.exists():
                return ToolResult(
                    content=f"Error: eval run '{run_name}' already exists. Use a different name."
                )

            debug_mode = os.environ.get("LQH_DEBUG", "").lower() in ("1", "true", "yes")
            result = await run_scoring(
                dataset_path=data_path,
                scorer_path=scorer_path,
                output_dir=output_dir,
                client=client,
                model_size=model_size,
                run_inference=True,
                inference_model=inference_model,
                inference_system_prompt=inference_system_prompt,
                inference_response_format=inference_response_format,
                on_progress=_progress,
                debug=debug_mode,
            )

            # Write config.json
            scoring_model = JUDGE_MODELS.get(model_size, f"judge:{model_size}")
            config_data: dict[str, Any] = {
                "eval_dataset": dataset,
                "scorer": scorer,
                "mode": mode,
                "scoring_model": scoring_model,
                "inference_model": inference_model,
            }
            if inference_system_prompt:
                config_data["inference_system_prompt"] = inference_system_prompt
            if system_prompt_path:
                config_data["system_prompt_path"] = system_prompt_path
            (output_dir / "config.json").write_text(
                json.dumps(config_data, indent=2), encoding="utf-8"
            )

            from lqh.project_log import append_event

            append_event(
                project_dir,
                "scoring_completed",
                f"Scored {dataset} (model_eval, run={run_name}) mean={result.mean_score:.1f} median={result.median_score:.1f}",
                dataset=dataset,
                scorer=scorer,
                mode="model_eval",
                run_name=run_name,
                mean_score=round(result.mean_score, 2),
                median_score=round(result.median_score, 2),
            )

            distribution = _format_score_distribution(output_dir / "results.parquet")
            return ToolResult(
                content=(
                    f"✅ Model evaluation complete\n"
                    f"  Dataset: {dataset}\n"
                    f"  Scored: {result.scored}/{result.total}"
                    + (
                        f" ({result.failed} judge errors — could not be scored, "
                        f"not counted in mean/median; re-run to score them)"
                        if result.failed else ""
                    )
                    + f"\n  Mean score: {result.mean_score:.1f}/10"
                    f"\n  Median score: {result.median_score:.1f}/10"
                    + (f"\n{distribution}" if distribution else "")
                    + f"\n  Results: evals/runs/{run_name}/"
                )
            )
        else:
            return ToolResult(content=f"Error: unknown mode '{mode}'. Use 'data_quality' or 'model_eval'.")

    except Exception as e:
        import traceback

        from lqh.project_log import append_event

        append_event(
            project_dir,
            "scoring_failed",
            f"Scoring failed on {dataset}: {type(e).__name__}: {e}",
            dataset=dataset,
            scorer=scorer,
            mode=mode,
            error=f"{type(e).__name__}: {e}",
        )

        tb = traceback.format_exc()
        return ToolResult(content=f"❌ Scoring failed: {type(e).__name__}: {e}\n\n{tb}")
    finally:
        on_done = kwargs.get("on_pipeline_done")
        if on_done:
            on_done()


# ---------------------------------------------------------------------------
# Hugging Face Hub helpers
# ---------------------------------------------------------------------------

HF_MAPPINGS_FILE = ".lqh/hf.json"


def _get_hf_token() -> str:
    """Return HF_TOKEN from environment or raise with instructions."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Export HF_TOKEN=hf_... or set it in your shell. "
            "Get a token at https://huggingface.co/settings/tokens"
        )
    return token


def _get_hf_api():
    """Create an authenticated HfApi instance."""
    from huggingface_hub import HfApi

    token = _get_hf_token()
    return HfApi(token=token)


def _load_hf_mappings(project_dir: Path) -> dict:
    """Load HF repo mappings from .lqh/hf.json."""
    path = project_dir / HF_MAPPINGS_FILE
    if not path.exists():
        return {"mappings": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"mappings": []}


def _save_hf_mapping(
    project_dir: Path,
    local_path: str,
    repo_id: str,
    repo_type: str,
    split: str | None = None,
) -> None:
    """Add or update a mapping in .lqh/hf.json."""
    data = _load_hf_mappings(project_dir)
    mappings = data.get("mappings", [])

    # Update existing or append
    found = False
    for m in mappings:
        if m.get("local_path") == local_path and m.get("repo_id") == repo_id:
            m["repo_type"] = repo_type
            if split:
                m["split"] = split
            m["last_synced"] = datetime.now(tz=timezone.utc).isoformat()
            found = True
            break

    if not found:
        entry: dict[str, Any] = {
            "local_path": local_path,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "last_synced": datetime.now(tz=timezone.utc).isoformat(),
        }
        if split:
            entry["split"] = split
        mappings.append(entry)

    data["mappings"] = mappings
    path = project_dir / HF_MAPPINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Hugging Face Hub handlers
# ---------------------------------------------------------------------------


async def handle_hf_repo_info(
    *, repo_id: str | None = None, repo_type: str = "dataset", **kwargs: Any,
) -> ToolResult:
    """Get info about a HF repo or the authenticated user."""
    try:
        api = _get_hf_api()
    except ValueError as e:
        return ToolResult(content=f"Error: {e}")

    try:
        if repo_id is None:
            # whoami
            info = api.whoami()
            username = info.get("name", "unknown")
            orgs = [o.get("name", "?") for o in info.get("orgs", [])]
            auth = info.get("auth", {})
            access_type = auth.get("accessToken", {}).get("type", "unknown")
            lines = [
                f"🤗 Authenticated as: **{username}**",
                f"  Token type: {access_type}",
            ]
            if orgs:
                lines.append(f"  Organizations: {', '.join(orgs)}")
            return ToolResult(content="\n".join(lines))
        else:
            info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
            lines = [
                f"🤗 Repo: **{repo_id}** ({repo_type})",
                f"  Private: {info.private}",
                f"  Last modified: {info.last_modified}",
            ]
            if hasattr(info, "card_data") and info.card_data:
                lines.append(f"  Card data: {info.card_data}")
            siblings = info.siblings or []
            if siblings:
                lines.append(f"  Files ({len(siblings)}):")
                for s in siblings[:20]:
                    lines.append(f"    - {s.rfilename}")
                if len(siblings) > 20:
                    lines.append(f"    ... and {len(siblings) - 20} more")
            return ToolResult(content="\n".join(lines))
    except Exception as e:
        return ToolResult(content=f"Error: {e}")


async def handle_hf_pull(
    project_dir: Path,
    *,
    repo_id: str,
    local_path: str | None = None,
    split: str | None = None,
    subset: str | None = None,
    files: list[str] | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Download a dataset from HF Hub to local storage."""
    token = os.environ.get("HF_TOKEN")  # optional for public repos

    # Determine local output path
    if local_path is None:
        # username/my-dataset -> my-dataset
        repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        local_path = f"datasets/{repo_name}"

    target = _validate_path(project_dir, local_path)
    target.mkdir(parents=True, exist_ok=True)

    try:
        if files:
            # Download specific files
            from huggingface_hub import hf_hub_download

            downloaded = []
            for f in files:
                out = hf_hub_download(
                    repo_id=repo_id,
                    filename=f,
                    repo_type="dataset",
                    local_dir=str(target),
                    token=token,
                )
                downloaded.append(out)

            _save_hf_mapping(project_dir, local_path, repo_id, "dataset", split)
            return ToolResult(
                content=(
                    f"✅ Downloaded {len(downloaded)} file(s) to {local_path}/\n"
                    + "\n".join(f"  - {Path(d).name}" for d in downloaded)
                )
            )
        else:
            # Download full dataset via datasets library
            import datasets as ds_lib

            load_kwargs: dict[str, Any] = {"path": repo_id, "trust_remote_code": False}
            if token:
                load_kwargs["token"] = token
            if split:
                load_kwargs["split"] = split
            if subset:
                load_kwargs["name"] = subset

            dataset = ds_lib.load_dataset(**load_kwargs)

            # dataset can be DatasetDict (multiple splits) or Dataset (single split)
            if isinstance(dataset, ds_lib.DatasetDict):
                total_rows = 0
                split_info = []
                for split_name, split_ds in dataset.items():
                    out_path = target / f"{split_name}.parquet"
                    split_ds.to_parquet(str(out_path))
                    total_rows += len(split_ds)
                    split_info.append(f"  - {split_name}: {len(split_ds):,} rows -> {split_name}.parquet")

                _save_hf_mapping(project_dir, local_path, repo_id, "dataset")
                return ToolResult(
                    content=(
                        f"✅ Downloaded {repo_id} to {local_path}/ ({total_rows:,} rows total)\n"
                        + "\n".join(split_info)
                    )
                )
            else:
                # Single split
                out_path = target / "data.parquet"
                dataset.to_parquet(str(out_path))

                _save_hf_mapping(project_dir, local_path, repo_id, "dataset", split)
                return ToolResult(
                    content=(
                        f"✅ Downloaded {repo_id} to {local_path}/data.parquet "
                        f"({len(dataset):,} rows)"
                    )
                )

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            hint = " This may be a private repo — make sure HF_TOKEN is set with appropriate permissions."
        elif "404" in error_msg or "not found" in error_msg.lower():
            hint = " Check the repo ID and whether the repo exists."
        else:
            hint = ""
        return ToolResult(content=f"Error downloading {repo_id}: {e}{hint}")


async def handle_hf_push(
    project_dir: Path,
    *,
    local_path: str,
    repo_id: str | None = None,
    private: bool = True,
    split: str = "train",
    subset: str | None = None,
    commit_message: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Push a local dataset to HF Hub. Requires permission."""
    # Check HF token first
    try:
        api = _get_hf_api()
    except ValueError as e:
        return ToolResult(content=f"Error: {e}")

    # Validate local path
    target = _validate_path(project_dir, local_path)
    if not target.exists():
        return ToolResult(content=f"Error: '{local_path}' does not exist")

    # Find parquet file(s)
    if target.is_dir():
        parquet_files = list(target.glob("*.parquet"))
        if not parquet_files:
            return ToolResult(content=f"Error: no .parquet files in '{local_path}'")
        # Use data.parquet if it exists, otherwise first parquet
        data_parquet = target / "data.parquet"
        parquet_path = data_parquet if data_parquet.exists() else parquet_files[0]
    elif target.suffix == ".parquet":
        parquet_path = target
    else:
        return ToolResult(content=f"Error: '{local_path}' is not a parquet file or directory containing one")

    # Auto-generate repo_id if not provided
    if repo_id is None:
        try:
            info = api.whoami()
            username = info.get("name", "unknown")
        except Exception as e:
            return ToolResult(content=f"Error getting HF username: {e}")

        project_name = project_dir.name
        dataset_name = target.name if target.is_dir() else target.parent.name
        repo_id = f"{username}/{project_name}-{dataset_name}"

    # Check permission
    from lqh.tools.permissions import check_hf_permission

    if not check_hf_permission(project_dir, repo_id):
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            question=(
                f"The agent wants to push data to Hugging Face Hub:\n"
                f"  Local: {local_path}\n"
                f"  Repo:  {repo_id}\n"
                f"  Private: {private}\n"
                f"  Split: {split}\n\n"
                f"Allow push?"
            ),
            options=[
                "Push once, ask again next time",
                "Push and don't ask again for this repo",
                "Push and don't ask again for this project",
                "Do not push",
            ],
        )

    # Execute the push
    return await _execute_hf_push(
        project_dir, parquet_path, local_path, repo_id, private, split, subset, commit_message, api,
    )


async def _execute_hf_push(
    project_dir: Path,
    parquet_path: Path,
    local_path: str,
    repo_id: str,
    private: bool,
    split: str,
    subset: str | None,
    commit_message: str | None,
    api,
) -> ToolResult:
    """Actually push to HF after permission is granted."""
    try:
        import datasets as ds_lib

        # Create repo if needed
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

        # Load and push
        dataset = ds_lib.Dataset.from_parquet(str(parquet_path))

        push_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "split": split,
            "private": private,
        }
        if subset:
            push_kwargs["config_name"] = subset
        if commit_message:
            push_kwargs["commit_message"] = commit_message
        else:
            push_kwargs["commit_message"] = f"Push {split} split ({len(dataset):,} rows)"

        dataset.push_to_hub(**push_kwargs)

        # Save mapping
        _save_hf_mapping(project_dir, local_path, repo_id, "dataset", split)

        url = f"https://huggingface.co/datasets/{repo_id}"
        visibility = "private" if private else "public"
        return ToolResult(
            content=(
                f"✅ Pushed to HF Hub\n"
                f"  Repo:  {repo_id} ({visibility})\n"
                f"  Split: {split}\n"
                f"  Rows:  {len(dataset):,}\n"
                f"  URL:   {url}"
            )
        )

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg:
            hint = " Your HF_TOKEN may not have write access. Check token permissions at https://huggingface.co/settings/tokens"
        else:
            hint = ""
        return ToolResult(content=f"Error pushing to {repo_id}: {e}{hint}")


# ---------------------------------------------------------------------------
# Training tools
# ---------------------------------------------------------------------------


def _check_torch_available() -> str | None:
    """Return an error message if torch is not importable, else None."""
    try:
        import torch  # noqa: F401

        return None
    except ImportError:
        return (
            "Training requires the 'train' optional dependencies.\n"
            "Install them with: pip install lqh[train]"
        )


def _next_run_name(project_dir: Path, prefix: str) -> str:
    """Generate the next sequential run name (e.g. sft_001, sft_002)."""
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return f"{prefix}_001"
    existing = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not existing:
        return f"{prefix}_001"
    nums = []
    for name in existing:
        suffix = name[len(prefix) + 1:]
        try:
            nums.append(int(suffix))
        except ValueError:
            continue
    next_num = max(nums, default=0) + 1
    return f"{prefix}_{next_num:03d}"


async def handle_start_training(
    project_dir: Path,
    *,
    type: str,
    base_model: str,
    dataset: str,
    eval_dataset: str | None = None,
    scorer: str | None = None,
    run_name: str | None = None,
    lora: bool = True,
    num_epochs: int = 3,
    learning_rate: float | None = None,
    num_iterations: int = 5,
    dpo_beta: float = 0.1,
    golden_source: str = "dataset",
    remote: str | None = None,
    enable_sweep: bool = True,
    grid_size: str = "small",
    **kwargs: Any,
) -> ToolResult:
    """Start a training subprocess.

    Sweep behaviour
    ---------------
    By default ``enable_sweep=True``: instead of a single training run, we
    sweep a small hyperparameter grid (see ``lqh.train.sweep``) and pick
    the best config by a cheap, validated in-training proxy:

    - SFT: ``eval_loss`` (Pearson r = −0.90 with judge_mean on ar_to_de).
    - DPO: ``eval_ce_chosen_mean`` (Spearman ρ = −1.0). DPO's own
      ``eval_loss`` is NOT used — it's broken as a selector (correlates
      with judge in the wrong direction because driving DPO loss down
      can mean policy collapse).

    Rationale: training itself is cheap, data generation and judge eval
    are expensive. Sweeping ~6 configs costs ~2-3× single-config wall
    time and reliably finds a better config than zero-shot defaults.

    Pass ``enable_sweep=false`` to fall back to single-config behaviour
    only when the user explicitly asks for it (e.g. "just train one
    config with lr=2e-5"). Specific ``learning_rate``/``num_epochs``/
    ``dpo_beta`` values supplied by the agent are honoured under
    ``enable_sweep=false``; under sweep they are overridden by the grid.
    """
    from lqh.tools.permissions import check_permission

    # Check torch + GPU only when running locally; remote execution has its
    # own venv (provisioned by remote_setup) and its own GPUs.
    if remote is None:
        err = _check_torch_available()
        if err:
            return ToolResult(content=f"❌ {err}")

        try:
            import torch

            if not torch.cuda.is_available():
                return ToolResult(
                    content="⚠️ No CUDA GPU detected. Training requires a GPU."
                )
            gpu_info = ", ".join(
                f"{torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())
            )
        except Exception:
            gpu_info = "unknown"
    else:
        gpu_info = f"remote ({remote})"

    # Validate dataset paths
    ds_path = _validate_path(project_dir, dataset)
    data_parquet = ds_path / "data.parquet"
    if not data_parquet.exists():
        return ToolResult(content=f"Error: dataset not found at {dataset}/data.parquet")

    eval_parquet_path: str | None = None
    if eval_dataset:
        eval_ds_path = _validate_path(project_dir, eval_dataset)
        eval_parquet = eval_ds_path / "data.parquet"
        if not eval_parquet.exists():
            return ToolResult(
                content=f"Error: eval dataset not found at {eval_dataset}/data.parquet"
            )
        eval_parquet_path = str(eval_parquet)

    scorer_path: str | None = None
    if scorer:
        scorer_resolved = _validate_path(project_dir, scorer)
        if not scorer_resolved.exists():
            return ToolResult(content=f"Error: scorer not found at {scorer}")
        scorer_path = scorer

    # Generate run name
    if not run_name:
        prefix = "sft" if type == "sft" else "dpo"
        run_name = _next_run_name(project_dir, prefix)

    run_dir = project_dir / "runs" / run_name

    if run_dir.exists() and (run_dir / "config.json").exists():
        return ToolResult(content=f"Error: run '{run_name}' already exists")

    # Build config
    default_lr = 2e-5 if type == "sft" else 5e-6
    lr = learning_rate if learning_rate is not None else default_lr

    config: dict[str, Any] = {
        "type": type,
        "base_model": base_model,
        "dataset": str(data_parquet),
        "training": {
            "learning_rate": lr,
            "max_seq_length": 2048,
        },
        "lora": {
            "enabled": lora,
            "r": 32,
            "alpha": 64,
            "dropout": 0.02,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj", "out_proj", "w1", "w2", "w3",
            ],
        },
        "manifest": ["base_model", "dataset"],
    }

    if eval_parquet_path:
        config["eval_dataset"] = eval_parquet_path
        config["eval_on_checkpoints"] = True
        config["manifest"].append("eval_dataset")
    if scorer_path:
        config["scorer"] = scorer_path
        config["manifest"].append("scorer")

    if type == "sft":
        config["training"]["num_epochs"] = num_epochs
    elif type in ("on_policy_dpo", "dpo"):
        config["num_iterations"] = num_iterations
        config["dpo_beta"] = dpo_beta
        config["golden_source"] = golden_source

    # Permission check
    perm_key = f"training:{run_name}"
    if not check_permission(project_dir, perm_key):
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            question=(
                f"The agent wants to start a {type.upper()} training run:\n"
                f"  Run:       {run_name}\n"
                f"  Model:     {base_model}\n"
                f"  Dataset:   {dataset}\n"
                f"  GPU:       {gpu_info}\n\n"
                f"Allow execution?"
            ),
            options=[
                "Start training",
                "Do not start training",
            ],
        )

    on_bg_started = kwargs.get("on_background_task_started")

    # Build the launch payload. Sweep wraps the base config + grid spec;
    # single-config sends the base config directly. The remote backend
    # ships either payload identically — sweep just looks like a different
    # subprocess type to the watcher.
    if enable_sweep:
        launch_config: dict[str, Any] = {
            "type": "sweep",
            "base_config": config,
            "grid_size": grid_size,
        }
        launch_module = "lqh.train.sweep"
    else:
        launch_config = config
        launch_module = "lqh.train"

    if remote:
        return await _execute_start_training_remote(
            project_dir, run_dir, launch_config, run_name, remote,
            kwargs.get("api_key", ""),
            on_bg_started=on_bg_started,
            module=launch_module,
        )
    return await _execute_start_training(
        project_dir, run_dir, launch_config, run_name,
        on_bg_started=on_bg_started,
        module=launch_module,
    )


async def _execute_start_training_remote(
    project_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    run_name: str,
    remote_name: str,
    api_key: str,
    *,
    on_bg_started: Callable[[str, str, str, str | None], None] | None = None,
    module: str = "lqh.train",
) -> ToolResult:
    """Start training on a remote backend."""
    from lqh.remote.config import get_remote
    from lqh.remote.ssh_direct import SSHDirectBackend

    remote_config = get_remote(project_dir, remote_name)
    if remote_config is None:
        return ToolResult(
            content=f"Error: remote '{remote_name}' not found. Use remote_list to see configured remotes."
        )

    if remote_config.type == "ssh_slurm":
        return ToolResult(content="Error: SSH+Slurm backend is not yet implemented.")

    backend = SSHDirectBackend(remote_config, project_dir)
    remote_run_dir = f"{remote_config.remote_root}/runs/{run_name}"

    try:
        job_id = await backend.submit_run(str(run_dir), config, module=module)
    except Exception as e:
        return ToolResult(content=f"Error launching remote training: {e}")

    if on_bg_started is not None:
        on_bg_started(run_name, "train", run_name, remote_name)

    from lqh.project_log import append_event

    # When sweep is enabled the launch config is wrapped:
    # {"type": "sweep", "base_config": {real config}}. Unwrap one
    # level so the event log records the actual run_type/base_model.
    inner = config.get("base_config", config) if config.get("type") == "sweep" else config

    append_event(
        project_dir,
        "training_started",
        f"Started {inner.get('type', 'training')} run {run_name} on remote '{remote_name}' (job {job_id})",
        run_name=run_name,
        run_type=inner.get("type", "unknown"),
        base_model=inner.get("base_model", ""),
        remote=remote_name,
    )

    return ToolResult(
        content=(
            f"🚀 Remote training started on '{remote_name}'\n"
            f"  Run:      {run_name}\n"
            f"  Type:     {config['type']}\n"
            f"  Job ID:   {job_id}\n"
            f"  Host:     {remote_config.hostname}\n"
            f"  Dir:      {remote_run_dir}\n\n"
            f"Use training_status(remote='{remote_name}') to monitor progress."
        )
    )


async def _execute_start_training(
    project_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    run_name: str,
    *,
    on_bg_started: Callable[[str, str, str, str | None], None] | None = None,
    module: str = "lqh.train",
) -> ToolResult:
    """Actually start the training subprocess after permission is granted.

    ``module`` is ``"lqh.train"`` for a single-config run or
    ``"lqh.train.sweep"`` for a hyperparameter sweep. The sweep
    subprocess writes the same progress/PID files so SubprocessManager
    treats it identically.
    """
    from lqh.subprocess_manager import SubprocessManager

    manager = SubprocessManager()

    pid = manager.start(run_dir, config, module=module, project_dir=project_dir)

    if on_bg_started is not None:
        on_bg_started(run_name, "train", run_name, None)

    from lqh.project_log import append_event

    inner = config.get("base_config", config) if config.get("type") == "sweep" else config

    append_event(
        project_dir,
        "training_started",
        f"Started {inner.get('type', 'training')} run {run_name} (PID {pid})",
        run_name=run_name,
        run_type=inner.get("type", "unknown"),
        base_model=inner.get("base_model", ""),
    )

    return ToolResult(
        content=(
            f"🚀 Training started\n"
            f"  Run:    {run_name}\n"
            f"  Type:   {config.get('type', 'unknown')}\n"
            f"  PID:    {pid}\n"
            f"  Dir:    runs/{run_name}/\n\n"
            f"Use training_status to monitor progress."
        )
    )


async def handle_training_status(
    project_dir: Path,
    *,
    run_name: str | None = None,
    remote: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Check training run status.

    Per-run with explicit remote: dispatch to the remote handler. Single
    run without a remote: read local mirror + (if remote_job.json exists)
    poll the corresponding remote — local PIDs aren't comparable across
    machines. List mode (no run_name): same per-run rule applied to every
    runs/<name>/ entry.
    """
    if remote and run_name:
        return await _training_status_remote(project_dir, run_name, remote)

    from lqh.subprocess_manager import SubprocessManager

    manager = SubprocessManager()

    if run_name:
        run_dir = _validate_path(project_dir, f"runs/{run_name}")
        if not run_dir.exists():
            return ToolResult(content=f"Error: run '{run_name}' not found")
        meta = _read_remote_meta(run_dir)
        if meta is not None:
            return await _training_status_remote(
                project_dir, run_name, meta["remote_name"],
            )
        status = manager.get_status(run_dir)
        return ToolResult(content=_format_status(run_name, status, run_dir))

    runs_dir = project_dir / "runs"
    if not runs_dir.is_dir():
        return ToolResult(content="No training runs found.")

    parts: list[str] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir() or not (entry / "config.json").exists():
            continue
        meta = _read_remote_meta(entry)
        if meta is not None:
            remote_status = await _training_status_remote(
                project_dir, entry.name, meta["remote_name"],
            )
            parts.append(remote_status.content)
        else:
            status = manager.get_status(entry)
            parts.append(_format_status(entry.name, status, entry))

    if not parts:
        return ToolResult(content="No training runs found.")
    return ToolResult(content="\n\n".join(parts))


def _read_remote_meta(run_dir: Path) -> dict[str, Any] | None:
    """Return remote_job.json contents if the run was launched on a remote."""
    meta_file = run_dir / "remote_job.json"
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text())
    except Exception:
        return None


async def _training_status_remote(
    project_dir: Path,
    run_name: str,
    remote_name: str,
) -> ToolResult:
    """Check status of a remote training run."""
    from lqh.remote.config import get_remote
    from lqh.remote.ssh_direct import SSHDirectBackend

    remote_config = get_remote(project_dir, remote_name)
    if remote_config is None:
        return ToolResult(content=f"Error: remote '{remote_name}' not found.")

    backend = SSHDirectBackend(remote_config, project_dir)
    run_dir = project_dir / "runs" / run_name

    # Read job_id from local metadata
    meta_file = run_dir / "remote_job.json"
    if not meta_file.exists():
        return ToolResult(content=f"Error: no remote job metadata for run '{run_name}'.")

    meta = json.loads(meta_file.read_text())
    job_id = meta["job_id"]
    remote_run_dir = meta["remote_run_dir"]

    try:
        # Sync progress first
        await backend.sync_progress(remote_run_dir, str(run_dir))
        status = await backend.poll_status(job_id)
    except Exception as e:
        return ToolResult(content=f"Error checking remote status: {e}")

    state_emoji = {
        "running": "🏃", "completed": "✅", "failed": "❌",
        "waiting_for_scoring": "⏳", "unknown": "❓",
    }
    emoji = state_emoji.get(status.state, "❓")
    lines = [f"{emoji} **{run_name}** — {status.state} (remote: {remote_name})"]
    if status.current_step is not None:
        lines.append(f"  Step: {status.current_step}")
    if status.error:
        lines.append(f"  Error: {status.error}")

    # Also show local mirror progress if available
    from lqh.train.progress import read_latest_progress
    latest = read_latest_progress(run_dir)
    if latest:
        if latest.get("loss") is not None:
            lines.append(f"  Loss: {latest['loss']:.4f}")
        if latest.get("lr") is not None:
            lines.append(f"  LR:   {latest['lr']:.2e}")

    return ToolResult(content="\n".join(lines))


def _format_status(run_name: str, status: Any, run_dir: Path) -> str:
    """Format a RunStatus as a readable string."""
    state_emoji = {
        "running": "🏃",
        "completed": "✅",
        "failed": "❌",
        "unknown": "❓",
    }
    emoji = state_emoji.get(status.state, "❓")
    lines = [f"{emoji} **{run_name}** — {status.state}"]

    if status.step is not None:
        lines.append(f"  Step: {status.step}")
    if status.loss is not None:
        lines.append(f"  Loss: {status.loss:.4f}")
    if status.lr is not None:
        lines.append(f"  LR:   {status.lr:.2e}")
    if status.epoch is not None:
        lines.append(f"  Epoch: {status.epoch:.2f}")
    if status.error:
        lines.append(f"  Error: {status.error}")

    # SFT/checkpoint eval results
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        eval_results = []
        for cp_dir in sorted(checkpoints_dir.iterdir()):
            result_file = cp_dir / "eval_result.json"
            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                    mean_score = result.get("scores", {}).get("mean")
                    if mean_score is not None:
                        eval_results.append(f"    {cp_dir.name}: mean={mean_score:.2f}")
                except (json.JSONDecodeError, OSError):
                    pass
        if eval_results:
            lines.append("  Eval scores:")
            lines.extend(eval_results)

    # Chosen-pool ceiling — the harness scores the training set once
    # upfront and stashes the mean here. The model can't exceed this
    # on the same judge, so it's the most useful "is there room left?"
    # signal when deciding whether to keep tuning hyperparams or scale
    # data instead.
    chosen_summary = run_dir / "chosen_pool_summary.json"
    if chosen_summary.exists():
        try:
            payload = json.loads(chosen_summary.read_text())
            mean = payload.get("mean")
            if mean is not None:
                lines.append(
                    f"  Chosen-pool ceiling: {mean:.2f} — model can't "
                    f"exceed this on the same judge."
                )
        except (json.JSONDecodeError, OSError):
            pass

    # DPO iter stats — preference_stats.json (selection funnel +
    # gap distribution) and held_out_eval.json (per-iter eval delta
    # vs baseline). Both written by the harness; surfacing them here
    # so the agent can see whether DPO has signal and whether the
    # held-out trajectory is healthy without reading files manually.
    iterations_dir = run_dir / "iterations"
    if iterations_dir.exists():
        iter_lines = _format_dpo_iter_stats(iterations_dir)
        if iter_lines:
            lines.append("  DPO iterations:")
            lines.extend(iter_lines)

    # If an early_abort.json was written by the harness, surface it.
    abort = run_dir / "early_abort.json"
    if abort.exists():
        try:
            payload = json.loads(abort.read_text())
            reason = payload.get("reason", "regression past threshold")
            lines.append(f"  ⚠️  Early-abort signaled: {reason}")
        except (json.JSONDecodeError, OSError):
            lines.append("  ⚠️  Early-abort signaled (unparseable)")

    # Sweep summary (when handle_start_training was invoked with the
    # default enable_sweep=True). We deliberately surface only the
    # validated proxy here:
    #   - For SFT: eval_loss (Pearson r=-0.90 with judge_mean).
    #   - For DPO: eval_ce_chosen_mean and eval_ce_chosen_delta_ref
    #     (Spearman ρ=-1.0). DPO eval_loss and eval_rewards/margins are
    #     NOT shown — they correlate with judge in the wrong direction
    #     and would mislead the agent into picking a collapsed config.
    #     See lqh/train/sweep.py for the experiment that established this.
    sweep_lines = _format_sweep_summary(run_dir)
    if sweep_lines:
        lines.extend(sweep_lines)

    return "\n".join(lines)


def _format_sweep_summary(run_dir: Path) -> list[str]:
    """Render the per-config table for a hyperparameter sweep, if present.

    DPO val_loss and eval_rewards/margins are intentionally NOT surfaced
    (they are wrong-signed proxies — see ``lqh/train/sweep.py``).
    """
    summary_path = run_dir / "sweep_summary.json"
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    rows = payload.get("rows") or []
    if not rows:
        return []

    mode = payload.get("mode", "sft")
    proxy_key = payload.get("proxy_key", "eval_loss")
    winner = payload.get("winner") or {}
    winner_id = winner.get("config_id")
    n_done = payload.get("n_completed", len(rows))
    n_total = payload.get("n_configs", len(rows))

    out: list[str] = []
    header = f"  Sweep: {n_done}/{n_total} configs · proxy={proxy_key}"
    if winner_id:
        primary = winner.get("primary")
        primary_s = f"{primary:.4f}" if isinstance(primary, (int, float)) else "—"
        header += f" · best={winner_id} ({proxy_key}={primary_s})"
    out.append(header)

    # Sort by primary asc (best first), collapsed/failed configs at the bottom.
    def _sort_key(r: dict[str, Any]) -> tuple[int, float]:
        p = r.get("primary")
        is_bad = r.get("collapsed") or p is None
        return (1 if is_bad else 0, p if isinstance(p, (int, float)) else float("inf"))

    for r in sorted(rows, key=_sort_key):
        cid = r.get("config_id", "?")
        ov = r.get("overrides", {}) or {}
        # Pull just the hyperparam knobs the user cares about, regardless
        # of where they live in the nested override dict.
        tr = ov.get("training") or {}
        hp_bits: list[str] = []
        lr = tr.get("learning_rate")
        if lr is not None:
            hp_bits.append(f"lr={lr:g}")
        ep = tr.get("num_epochs")
        if ep is not None:
            hp_bits.append(f"epochs={ep}")
        beta = ov.get("dpo_beta")
        if beta is not None:
            hp_bits.append(f"β={beta:g}")
        hp_str = " ".join(hp_bits) or "(no overrides)"

        primary = r.get("primary")
        primary_s = f"{primary:.4f}" if isinstance(primary, (int, float)) else "—"
        marker = " ← winner" if cid == winner_id else ""
        if r.get("collapsed"):
            marker = " ⚠ collapsed"
        elif r.get("rc") not in (0, None):
            marker = f" ✗ failed (rc={r.get('rc')})"

        if mode == "sft":
            out.append(f"    {cid} · {hp_str} · eval_loss={primary_s}{marker}")
        else:
            # DPO row: CE-mean + Δref. Hide DPO eval_loss and margins.
            dref = r.get("eval_ce_chosen_delta_ref")
            p90 = r.get("eval_ce_chosen_p90")
            extras: list[str] = []
            extras.append(f"CE(ch)_mean={primary_s}")
            if isinstance(p90, (int, float)):
                extras.append(f"p90={p90:.3f}")
            if isinstance(dref, (int, float)):
                extras.append(f"Δref={dref:+.3f}")
            out.append(f"    {cid} · {hp_str} · " + " ".join(extras) + marker)
    return out


def _format_dpo_iter_stats(iterations_dir: Path) -> list[str]:
    """Build per-iter lines for DPO runs.

    For each iter dir, reads preference_stats.json (selection funnel +
    gap p10/p50/p90) and held_out_eval.json (mean + Δ vs baseline if
    present). Returns one line per iter, formatted compactly. Returns
    [] if no iter dirs or no usable data.
    """
    iter_lines: list[str] = []
    for iter_dir in sorted(iterations_dir.iterdir()):
        if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
            continue
        # Selection funnel + gap distribution
        kept_str = ""
        gap_str = ""
        prefs_path = iter_dir / "preference_stats.json"
        if prefs_path.exists():
            try:
                stats = json.loads(prefs_path.read_text())
                kept = stats.get("kept")
                pairs_total = stats.get("pairs_with_both_scored") or stats.get("total_predictions")
                if kept is not None and pairs_total:
                    kept_str = f"{kept}/{pairs_total} pairs"
                gp50 = stats.get("qualifying_gap_p50") or stats.get("gap_p50")
                gp90 = stats.get("qualifying_gap_p90") or stats.get("gap_p90")
                if gp50 is not None and gp90 is not None:
                    gap_str = f"gap p50={gp50:.1f}, p90={gp90:.1f}"
                if stats.get("skipped_reason"):
                    gap_str = (gap_str + " ⚠️ skipped: " + stats["skipped_reason"]).strip()
            except (json.JSONDecodeError, OSError):
                pass
        # Held-out eval
        held_str = ""
        held_path = iter_dir / "held_out_eval.json"
        if held_path.exists():
            try:
                held = json.loads(held_path.read_text())
                mean = held.get("mean")
                delta = held.get("delta_vs_baseline")
                if mean is not None and delta is not None:
                    held_str = f"held-out mean={mean:.2f} (Δ {delta:+.2f})"
                elif mean is not None:
                    held_str = f"held-out mean={mean:.2f}"
            except (json.JSONDecodeError, OSError):
                pass

        # Skip empty iter dirs
        if not (kept_str or gap_str or held_str):
            continue
        parts: list[str] = []
        if kept_str:
            parts.append(kept_str)
        if gap_str:
            parts.append(gap_str)
        if held_str:
            parts.append("→ " + held_str)
        iter_lines.append(f"    {iter_dir.name}: " + "  ".join(parts))
    return iter_lines


async def handle_stop_training(
    project_dir: Path,
    *,
    run_name: str,
    remote: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Stop a training subprocess."""
    if remote:
        return await _stop_training_remote(project_dir, run_name, remote)

    from lqh.subprocess_manager import SubprocessManager

    run_dir = _validate_path(project_dir, f"runs/{run_name}")
    if not run_dir.exists():
        return ToolResult(content=f"Error: run '{run_name}' not found")

    manager = SubprocessManager()
    if not manager.is_alive(run_dir):
        return ToolResult(content=f"Run '{run_name}' is not currently running.")

    stopped = manager.stop(run_dir)
    if stopped:
        from lqh.project_log import append_event

        append_event(
            project_dir,
            "training_stopped",
            f"Stopped training run {run_name}",
            run_name=run_name,
        )
        return ToolResult(content=f"🛑 Training run '{run_name}' stopped.")
    else:
        return ToolResult(content=f"Failed to stop run '{run_name}'.")


async def _stop_training_remote(
    project_dir: Path,
    run_name: str,
    remote_name: str,
) -> ToolResult:
    """Stop a remote training run."""
    from lqh.remote.config import get_remote
    from lqh.remote.ssh_direct import SSHDirectBackend

    remote_config = get_remote(project_dir, remote_name)
    if remote_config is None:
        return ToolResult(content=f"Error: remote '{remote_name}' not found.")

    run_dir = project_dir / "runs" / run_name
    meta_file = run_dir / "remote_job.json"
    if not meta_file.exists():
        return ToolResult(content=f"Error: no remote job metadata for run '{run_name}'.")

    meta = json.loads(meta_file.read_text())
    job_id = meta["job_id"]

    backend = SSHDirectBackend(remote_config, project_dir)
    try:
        await backend.teardown(job_id)
    except Exception as e:
        return ToolResult(content=f"Error stopping remote run: {e}")

    from lqh.project_log import append_event

    append_event(
        project_dir,
        "training_stopped",
        f"Stopped remote training run {run_name} on '{remote_name}'",
        run_name=run_name,
        remote=remote_name,
    )
    return ToolResult(content=f"🛑 Remote training run '{run_name}' stopped on '{remote_name}'.")


def _resolve_eval_extras(
    project_dir: Path,
    *,
    system_prompt_path: str | None,
    response_format_path: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Read the system-prompt file and (auto-)discover the response_format schema.

    Mirrors the discovery logic in handle_run_scoring so passing
    system_prompt_path to start_local_eval auto-picks up the matching
    prompts/<task>.schema.json file. Returns (system_prompt_text, schema_dict).
    """
    system_prompt: str | None = None
    if system_prompt_path:
        prompt_file = _validate_path(project_dir, system_prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"system_prompt_path '{system_prompt_path}' does not exist"
            )
        system_prompt = prompt_file.read_text(encoding="utf-8")

    schema_dict: dict[str, Any] | None = None
    if response_format_path:
        schema_file = _validate_path(project_dir, response_format_path)
        if not schema_file.exists():
            raise FileNotFoundError(
                f"response_format_path '{response_format_path}' does not exist"
            )
        schema_dict = json.loads(schema_file.read_text(encoding="utf-8"))
    elif system_prompt_path:
        # Auto-discover: prompts/translation_v0.md → prompts/translation.schema.json
        prompt_stem = Path(system_prompt_path).stem
        task_name = prompt_stem.rsplit("_v", 1)[0]
        auto_schema = Path(system_prompt_path).parent / f"{task_name}.schema.json"
        full_auto = project_dir / auto_schema
        if full_auto.exists():
            schema_dict = json.loads(full_auto.read_text(encoding="utf-8"))

    return system_prompt, schema_dict


async def handle_start_local_eval(
    project_dir: Path,
    *,
    model_path: str,
    dataset: str,
    scorer: str,
    run_name: str | None = None,
    remote: str | None = None,
    system_prompt_path: str | None = None,
    response_format_path: str | None = None,
    max_new_tokens: int = 4096,
    **kwargs: Any,
) -> ToolResult:
    """Start a local inference subprocess for model evaluation."""
    on_bg_started = kwargs.get("on_background_task_started")
    if remote:
        return await _start_local_eval_remote(
            project_dir, model_path, dataset, scorer, run_name, remote,
            system_prompt_path=system_prompt_path,
            response_format_path=response_format_path,
            max_new_tokens=max_new_tokens,
            on_bg_started=on_bg_started,
        )

    # Check torch
    err = _check_torch_available()
    if err:
        return ToolResult(content=f"❌ {err}")

    # Validate paths
    model_dir = _validate_path(project_dir, model_path)
    if not model_dir.exists():
        return ToolResult(content=f"Error: model not found at {model_path}")

    ds_path = _validate_path(project_dir, dataset)
    data_parquet = ds_path / "data.parquet"
    if not data_parquet.exists():
        return ToolResult(content=f"Error: dataset not found at {dataset}/data.parquet")

    scorer_resolved = _validate_path(project_dir, scorer)
    if not scorer_resolved.exists():
        return ToolResult(content=f"Error: scorer not found at {scorer}")

    try:
        system_prompt, schema_dict = _resolve_eval_extras(
            project_dir,
            system_prompt_path=system_prompt_path,
            response_format_path=response_format_path,
        )
    except FileNotFoundError as e:
        return ToolResult(content=f"Error: {e}")

    # Generate run name
    if not run_name:
        run_name = _next_run_name(project_dir, "local_eval")

    eval_run_dir = project_dir / "runs" / run_name

    # Build infer config
    config: dict[str, Any] = {
        "type": "infer",
        "base_model": str(model_dir),
        "dataset": str(data_parquet),
        "scorer": scorer,
        "max_new_tokens": max_new_tokens,
        "manifest": ["base_model", "dataset", "scorer"],
    }
    if system_prompt is not None:
        config["system_prompt"] = system_prompt
    if schema_dict is not None:
        config["response_format"] = schema_dict

    from lqh.subprocess_manager import SubprocessManager

    manager = SubprocessManager()
    pid = manager.start(eval_run_dir, config, module="lqh.infer", project_dir=project_dir)

    if on_bg_started is not None:
        on_bg_started(run_name, "eval", run_name, None)

    return ToolResult(
        content=(
            f"🔍 Local eval started\n"
            f"  Run:     {run_name}\n"
            f"  Model:   {model_path}\n"
            f"  PID:     {pid}\n"
            f"  Dir:     runs/{run_name}/\n\n"
            f"Predictions will be scored automatically when ready."
        )
    )


async def _start_local_eval_remote(
    project_dir: Path,
    model_path: str,
    dataset: str,
    scorer: str,
    run_name: str | None,
    remote_name: str,
    *,
    system_prompt_path: str | None = None,
    response_format_path: str | None = None,
    max_new_tokens: int = 4096,
    on_bg_started: Callable[[str, str, str, str | None], None] | None = None,
) -> ToolResult:
    """Start inference on a remote backend."""
    from lqh.remote.config import get_remote
    from lqh.remote.ssh_direct import SSHDirectBackend

    remote_config = get_remote(project_dir, remote_name)
    if remote_config is None:
        return ToolResult(content=f"Error: remote '{remote_name}' not found.")

    if remote_config.type == "ssh_slurm":
        return ToolResult(content="Error: SSH+Slurm backend is not yet implemented.")

    # Validate local paths
    ds_path = _validate_path(project_dir, dataset)
    data_parquet = ds_path / "data.parquet"
    if not data_parquet.exists():
        return ToolResult(content=f"Error: dataset not found at {dataset}/data.parquet")

    scorer_resolved = _validate_path(project_dir, scorer)
    if not scorer_resolved.exists():
        return ToolResult(content=f"Error: scorer not found at {scorer}")

    try:
        system_prompt, schema_dict = _resolve_eval_extras(
            project_dir,
            system_prompt_path=system_prompt_path,
            response_format_path=response_format_path,
        )
    except FileNotFoundError as e:
        return ToolResult(content=f"Error: {e}")

    if not run_name:
        run_name = _next_run_name(project_dir, "remote_eval")

    run_dir = project_dir / "runs" / run_name
    config: dict[str, Any] = {
        "type": "infer",
        "base_model": model_path,
        "dataset": str(data_parquet),
        "scorer": scorer,
        "max_new_tokens": max_new_tokens,
        "manifest": ["base_model", "dataset", "scorer"],
    }
    if system_prompt is not None:
        config["system_prompt"] = system_prompt
    if schema_dict is not None:
        config["response_format"] = schema_dict

    backend = SSHDirectBackend(remote_config, project_dir)
    try:
        job_id = await backend.submit_run(str(run_dir), config, module="lqh.infer")
    except Exception as e:
        return ToolResult(content=f"Error launching remote inference: {e}")

    if on_bg_started is not None:
        on_bg_started(run_name, "eval", run_name, remote_name)

    return ToolResult(
        content=(
            f"🔍 Remote eval started on '{remote_name}'\n"
            f"  Run:     {run_name}\n"
            f"  Model:   {model_path}\n"
            f"  Job ID:  {job_id}\n"
            f"  Host:    {remote_config.hostname}\n\n"
            f"Predictions will be scored automatically when ready."
        )
    )


# ------------------------------------------------------------------
# Remote management tools
# ------------------------------------------------------------------


async def handle_remote_list(project_dir: Path, **kwargs: Any) -> ToolResult:
    """List global machines and project bindings."""
    from lqh.remote.config import load_bindings, load_machines

    machines = load_machines()
    bindings = load_bindings(project_dir)

    if not machines and not bindings:
        return ToolResult(
            content="No remotes configured. Use remote_add to add a machine."
        )

    lines: list[str] = []

    # Show all global machines and whether they're bound to this project
    if machines:
        lines.append("**Available machines** (global):\n")
        for name, m in machines.items():
            bound = bindings.get(name)
            status = "✅ bound" if bound else "— not bound"
            lines.append(
                f"  {name}  [{status}]\n"
                f"    Type:     {m.type}\n"
                f"    Host:     {m.hostname}"
            )
            if m.gpu_ids is not None:
                lines.append(f"    GPUs:     {m.gpu_ids}")
            if bound:
                lines.append(f"    Root:     {bound.remote_root}")
                lines.append(
                    f"    HF token: {'✅' if bound.hf_token_configured else '❌'}"
                )
                if bound.gpu_ids is not None:
                    lines.append(f"    GPUs (project override): {bound.gpu_ids}")
            lines.append("")

    # Warn about orphan bindings (machine deleted globally)
    orphans = [n for n in bindings if n not in machines]
    if orphans:
        lines.append(
            f"⚠️  Orphan bindings (machine removed globally): {', '.join(orphans)}"
        )

    return ToolResult(content="\n".join(lines))


async def handle_remote_add(
    project_dir: Path,
    *,
    name: str,
    type: str,
    hostname: str,
    gpu_ids: list[int] | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Add a global machine definition."""
    from lqh.remote.backend import RemoteMachine
    from lqh.remote.config import add_machine

    machine = RemoteMachine(
        name=name,
        type=type,
        hostname=hostname,
        gpu_ids=gpu_ids,
    )
    try:
        add_machine(machine)
    except ValueError as e:
        return ToolResult(content=f"Error: {e}")

    return ToolResult(
        content=(
            f"✅ Machine '{name}' added globally.\n"
            f"  Type: {type}\n"
            f"  Host: {hostname}\n"
            + (f"  GPUs: {gpu_ids}\n" if gpu_ids else "")
            + f"\nUse remote_bind(name='{name}', remote_root='...') to bind "
            f"it to this project."
        )
    )


async def handle_remote_bind(
    project_dir: Path,
    *,
    name: str,
    remote_root: str,
    gpu_ids: list[int] | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Bind a global machine to the current project."""
    from lqh.remote.backend import ProjectBinding
    from lqh.remote.config import add_binding, get_machine

    machine = get_machine(name)
    if machine is None:
        return ToolResult(
            content=(
                f"Error: machine '{name}' not found globally. "
                f"Use remote_add to create it first."
            )
        )

    # Resolve "~" / "$HOME" against the remote user's home so persisted paths
    # are absolute. Keeps config rewrites and Python path opens working,
    # since neither expand "~" the way a login shell does.
    if remote_root.startswith("~") or "$HOME" in remote_root or "$home" in remote_root:
        from lqh.remote.ssh_helpers import ssh_run

        try:
            stdout, stderr, rc = await ssh_run(
                machine.hostname, f"echo {remote_root}", timeout=10.0,
            )
        except Exception as e:
            return ToolResult(
                content=f"Error resolving '{remote_root}' on {machine.hostname}: {e}"
            )
        if rc != 0:
            return ToolResult(
                content=(
                    f"Error resolving '{remote_root}' on {machine.hostname}: "
                    f"{stderr.strip() or 'ssh exited with code ' + str(rc)}"
                )
            )
        resolved = stdout.strip()
        if not resolved or not resolved.startswith("/"):
            return ToolResult(
                content=(
                    f"Error: could not resolve '{remote_root}' to an absolute path "
                    f"on {machine.hostname} (got: {resolved!r})"
                )
            )
        remote_root = resolved

    binding = ProjectBinding(
        name=name,
        remote_root=remote_root,
        gpu_ids=gpu_ids,
    )
    try:
        add_binding(project_dir, binding)
    except ValueError as e:
        return ToolResult(content=f"Error: {e}")

    return ToolResult(
        content=(
            f"✅ Machine '{name}' bound to this project.\n"
            f"  Host: {machine.hostname}\n"
            f"  Root: {remote_root}\n\n"
            f"Run remote_setup(name='{name}') to provision the environment."
        )
    )


async def handle_remote_remove(
    project_dir: Path,
    *,
    name: str,
    **kwargs: Any,
) -> ToolResult:
    """Unbind a remote from the current project."""
    from lqh.remote.config import remove_binding

    try:
        remove_binding(project_dir, name)
    except KeyError:
        return ToolResult(content=f"Error: remote '{name}' not bound to this project.")

    return ToolResult(
        content=(
            f"✅ Remote '{name}' unbound from this project.\n"
            f"The global machine definition is kept."
        )
    )


async def handle_remote_remove_machine(
    project_dir: Path,
    *,
    name: str,
    **kwargs: Any,
) -> ToolResult:
    """Remove a machine globally."""
    from lqh.remote.config import remove_machine

    try:
        remove_machine(name)
    except KeyError:
        return ToolResult(content=f"Error: machine '{name}' not found globally.")

    return ToolResult(content=f"✅ Machine '{name}' removed globally.")


async def handle_remote_setup(
    project_dir: Path,
    *,
    name: str,
    **kwargs: Any,
) -> ToolResult:
    """Provision a remote environment."""
    from lqh.remote.config import get_remote
    from lqh.remote.ssh_direct import SSHDirectBackend
    from lqh.remote.ssh_helpers import ssh_check

    remote_config = get_remote(project_dir, name)
    if remote_config is None:
        return ToolResult(content=f"Error: remote '{name}' not found.")

    if remote_config.type == "ssh_slurm":
        return ToolResult(content="Error: SSH+Slurm backend is not yet implemented.")

    # Check SSH connectivity first
    reachable = await ssh_check(remote_config.hostname)
    if not reachable:
        return ToolResult(
            content=(
                f"Error: cannot reach {remote_config.hostname} via SSH. "
                f"Check that SSH public key auth is configured and the host "
                f"is reachable."
            )
        )

    backend = SSHDirectBackend(remote_config, project_dir)
    try:
        log = await backend.setup()
    except Exception as e:
        return ToolResult(content=f"Error during setup: {e}")

    # Update config to mark HF token as configured if it was
    remote_config.hf_token_configured = True
    from lqh.remote.config import add_remote
    add_remote(project_dir, remote_config)

    return ToolResult(content=f"✅ Remote '{name}' provisioned.\n\n{log}")


async def handle_remote_status(
    project_dir: Path,
    *,
    name: str,
    **kwargs: Any,
) -> ToolResult:
    """Query a remote machine's GPU utilization and running processes."""
    from lqh.remote.config import get_machine
    from lqh.remote.gpu import query_gpu_status
    from lqh.remote.ssh_helpers import ssh_check, ssh_run

    machine = get_machine(name)
    if machine is None:
        return ToolResult(content=f"Error: machine '{name}' not found globally.")

    hostname = machine.hostname

    # Check SSH connectivity first
    reachable = await ssh_check(hostname)
    if not reachable:
        return ToolResult(
            content=(
                f"❌ Cannot reach **{name}** ({hostname}) via SSH.\n"
                f"Check that SSH public key auth is configured and the host "
                f"is reachable."
            )
        )

    lines = [f"**Remote status: {name}** ({hostname})\n"]

    # lqh version drift check — compares the install_hash sentinel written
    # by remote_setup against the current local source. If they differ,
    # signal the agent to re-run remote_setup before launching new jobs.
    from lqh.remote.bootstrap import (
        compute_local_lqh_hash,
        read_remote_lqh_hash,
        short_hash,
    )
    from lqh.remote.config import get_binding

    binding = get_binding(project_dir, name)
    local_hash = compute_local_lqh_hash()
    if binding is not None:
        remote_hash = await read_remote_lqh_hash(hostname, binding.remote_root)
        if remote_hash is None:
            lines.append(
                "📦 **lqh code:** ❓ no install_hash on remote — "
                "predates this check or never set up. Run `remote_setup` "
                "to update."
            )
        elif local_hash and remote_hash != local_hash:
            lines.append(
                f"📦 **lqh code:** ⚠️ OUTDATED on remote "
                f"(remote {short_hash(remote_hash)} vs local "
                f"{short_hash(local_hash)}). Run `remote_setup(name='{name}')` "
                f"to push the latest code; jobs launched now will run the "
                f"older lqh version."
            )
        else:
            lines.append(
                f"📦 **lqh code:** ✅ in sync ({short_hash(local_hash) if local_hash else 'pypi'})"
            )
        lines.append("")

    # GPU status
    gpus = await query_gpu_status(hostname)
    if gpus:
        lines.append(f"🖥️  **GPUs:** {len(gpus)} detected\n")
        for gpu in gpus:
            bar_len = 20
            used_blocks = round(gpu.gpu_utilization_pct / 100 * bar_len)
            bar = "█" * used_blocks + "░" * (bar_len - used_blocks)
            temp_str = f" {gpu.temperature_c}°C" if gpu.temperature_c is not None else ""
            lines.append(
                f"  GPU {gpu.index}: {gpu.name}\n"
                f"    Utilization: [{bar}] {gpu.gpu_utilization_pct}%{temp_str}\n"
                f"    Memory:      {gpu.memory_used_mib}/{gpu.memory_total_mib} MiB "
                f"({gpu.memory_utilization_pct}% used, "
                f"{gpu.memory_free_mib} MiB free)"
            )
    else:
        lines.append("🖥️  **GPUs:** none detected")

    # HF_TOKEN status
    lines.append("")
    # Check for HF_TOKEN in shell environment
    hf_stdout, _, hf_rc = await ssh_run(hostname, "echo $HF_TOKEN", timeout=10.0)
    if hf_rc == 0 and hf_stdout.strip():
        lines.append("🤗 **HF_TOKEN:** ✅ set in environment")
    else:
        # Also check if any project binding has it configured
        from lqh.remote.config import get_binding
        binding = get_binding(project_dir, name)
        if binding and binding.hf_token_configured:
            lines.append("🤗 **HF_TOKEN:** ✅ configured in project .env")
        else:
            lines.append("🤗 **HF_TOKEN:** ❌ not set")

    # Training processes
    lines.append("")
    # Look for python processes that look like training (lqh.train, lqh.infer,
    # torch, transformers, etc.)
    proc_cmd = (
        "ps aux | grep -E 'lqh\\.(train|infer)|transformers|torch\\.distributed' "
        "| grep -v grep"
    )
    stdout, _, rc = await ssh_run(hostname, proc_cmd, timeout=10.0)
    if rc == 0 and stdout.strip():
        proc_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        lines.append(f"⚙️  **Training processes:** {len(proc_lines)} found\n")
        for pl in proc_lines[:10]:  # cap at 10 to avoid flooding
            # Show user, PID, %CPU, %MEM, and command (trimmed)
            parts = pl.split(None, 10)
            if len(parts) >= 11:
                lines.append(
                    f"  PID {parts[1]}  CPU {parts[2]}%  MEM {parts[3]}%  "
                    f"{parts[10][:80]}"
                )
            else:
                lines.append(f"  {pl[:120]}")
    else:
        lines.append("⚙️  **Training processes:** none running")

    return ToolResult(content="\n".join(lines))


# Tool name -> handler mapping
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


async def handle_list_user_data(project_dir: Path, **kwargs: Any) -> ToolResult:
    """Report user-brought data in the project directory.

    Scans ``seed_data/``, any folder containing image files directly under
    the project root, and top-level JSONL/CSV/Parquet files.  Returns a
    concise textual summary the agent can fold into SPEC.md.
    """
    lines: list[str] = []

    # 1. seed_data/
    seed_dir = project_dir / "seed_data"
    if seed_dir.is_dir():
        entries = []
        for p in sorted(seed_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".jsonl", ".csv", ".txt"):
                continue
            try:
                if p.suffix.lower() == ".txt":
                    n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
                elif p.suffix.lower() == ".jsonl":
                    n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
                else:  # csv
                    n = max(0, sum(1 for _ in p.read_text(encoding="utf-8").splitlines()) - 1)
            except OSError:
                n = -1
            entries.append(f"  - {p.name} ({n} rows)")
        if entries:
            lines.append("seed_data/:")
            lines.extend(entries)
            lines.append(
                "  Use: `lqh.sources.seed_data(\"<stem>\")` in your pipeline."
            )

    # 2. image folders at project root
    image_folders: list[tuple[str, int, list[str]]] = []
    for p in sorted(project_dir.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name in {
            "datasets", "data_gen", "evals", "runs", "seed_data", "other_specs",
        }:
            continue
        # Count images (non-recursive first, then recursive if none)
        flat = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS]
        if flat:
            image_folders.append((p.name, len(flat), []))
            continue
        # Check for subfolders with images
        subs = [s for s in p.iterdir() if s.is_dir()]
        total = 0
        labels: list[str] = []
        for s in subs:
            n = sum(1 for f in s.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
            if n > 0:
                total += n
                labels.append(s.name)
        if total > 0:
            image_folders.append((p.name, total, sorted(labels)))
    if image_folders:
        lines.append("image folders:")
        for name, n, labels in image_folders:
            suffix = f" (subfolders: {', '.join(labels)})" if labels else ""
            lines.append(f"  - {name}/ ({n} images){suffix}")
        lines.append(
            "  Use: `lqh.sources.image_folder(\"<folder>\", include_subfolder_label=True)` "
            "when subfolders carry labels."
        )

    # 3. Top-level data files (JSONL/CSV/Parquet)
    data_files: list[tuple[str, str, int]] = []
    for p in sorted(project_dir.iterdir()):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in (".jsonl", ".csv", ".parquet"):
            continue
        try:
            if suffix == ".parquet":
                import pyarrow.parquet as pq
                n = pq.read_metadata(p).num_rows
            elif suffix == ".jsonl":
                n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
            else:  # csv
                n = max(0, sum(1 for _ in p.read_text(encoding="utf-8").splitlines()) - 1)
        except Exception:
            n = -1
        data_files.append((p.name, suffix, n))
    if data_files:
        lines.append("data files (project root):")
        for name, suffix, n in data_files:
            lines.append(f"  - {name} ({n} rows, {suffix[1:]})")
        lines.append(
            "  Use: `lqh.sources.prompts(\"<file>\")` for prompt lists, "
            "`lqh.sources.parquet(\"<file>\")` / `lqh.sources.jsonl(\"<file>\")` for arbitrary rows."
        )

    if not lines:
        return ToolResult(
            content=(
                "No user-brought data detected.\n"
                "Looked for: seed_data/, image folders at project root, "
                "top-level .jsonl/.csv/.parquet files.\n"
                "This is a synthetic-generation project — use liquidrandom for seeding."
            )
        )

    return ToolResult(content="\n".join(lines))


async def handle_run_data_filter(
    project_dir: Path,
    *,
    input_path: str,
    scorer_path: str,
    output_dataset: str,
    threshold: float = 6.0,
    model_size: str = "small",
    **kwargs: Any,
) -> ToolResult:
    """Score a user-brought dataset and emit a filtered subset."""
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_data_filter

    input_abs = _validate_path(project_dir, input_path)
    scorer_abs = _validate_path(project_dir, scorer_path)
    if not input_abs.exists():
        return ToolResult(content=f"Error: input '{input_path}' does not exist")
    if not scorer_abs.exists():
        return ToolResult(content=f"Error: scorer '{scorer_path}' does not exist")

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    output_dir = project_dir / "datasets" / output_dataset

    def on_progress(completed: int, total: int) -> None:
        cb = kwargs.get("on_pipeline_progress")
        if cb:
            cb(completed, total, 5)

    try:
        result = await run_data_filter(
            input_path=input_abs,
            scorer_path=scorer_abs,
            output_dataset_dir=output_dir,
            client=client,
            threshold=threshold,
            model_size=model_size,
            on_progress=on_progress,
        )
    except Exception as exc:
        return ToolResult(content=f"❌ run_data_filter failed: {type(exc).__name__}: {exc}")
    finally:
        on_done = kwargs.get("on_pipeline_done")
        if on_done:
            on_done()

    distribution = _format_score_distribution(output_dir / "scores.parquet")
    return ToolResult(
        content=(
            f"✅ Filtered dataset written\n"
            f"  Input:     {input_path} ({result.total} rows)\n"
            f"  Threshold: {threshold} (judge: {model_size})\n"
            f"  Kept:      {result.kept} / {result.total} ({result.kept / max(result.total, 1):.0%})\n"
            f"  Dropped:   {result.dropped}\n"
            f"  Failed:    {result.failed}\n"
            f"  Mean score: {result.mean_score:.2f}\n"
            + (f"\n{distribution}\n" if distribution else "")
            + f"  Output:    datasets/{output_dataset}/ (data.parquet, scores.parquet, summary.json)"
        )
    )


async def handle_exit_auto_mode(
    *, status: str, reason: str, **kwargs: Any,
) -> ToolResult:
    """Terminate auto mode. Only meaningful when the agent runs in auto mode."""
    status_norm = (status or "").strip().lower()
    if status_norm not in ("success", "failure"):
        return ToolResult(
            content=(
                f"Error: status must be 'success' or 'failure', got {status!r}. "
                "Call exit_auto_mode again with a valid status."
            ),
        )
    return ToolResult(
        content=f"Exiting auto mode: {status_norm} — {reason}",
        exit_auto_mode=True,
        auto_status=status_norm,
        auto_reason=reason,
    )


async def handle_set_auto_stage(
    *, stage: str, note: str | None = None, **kwargs: Any,
) -> ToolResult:
    """Report the current pipeline stage to the auto-mode TUI."""
    stage_norm = (stage or "").strip()
    if not stage_norm:
        return ToolResult(content="Error: stage must be a non-empty string.")
    msg = f"Stage set: {stage_norm}"
    if note:
        msg += f" — {note}"
    return ToolResult(
        content=msg,
        auto_stage=stage_norm,
        auto_stage_note=note,
    )


TOOL_HANDLERS: dict[str, Callable[..., Awaitable[ToolResult]]] = {
    "summary": handle_summary,
    "list_files": handle_list_files,
    "list_user_data": handle_list_user_data,
    "read_file": handle_read_file,
    "create_file": handle_create_file,
    "write_file": handle_write_file,
    "edit_file": handle_edit_file,
    "run_data_gen_pipeline": handle_run_data_gen_pipeline,
    "run_data_filter": handle_run_data_filter,
    "run_scoring": handle_run_scoring,
    "get_eval_failures": handle_get_eval_failures,
    "ask_user": handle_ask_user,
    "show_file": handle_show_file,
    "list_models": handle_list_models,
    "list_skills": handle_list_skills,
    "load_skill": handle_load_skill,
    "hf_push": handle_hf_push,
    "hf_pull": handle_hf_pull,
    "hf_repo_info": handle_hf_repo_info,
    "start_training": handle_start_training,
    "training_status": handle_training_status,
    "stop_training": handle_stop_training,
    "start_local_eval": handle_start_local_eval,
    "remote_list": handle_remote_list,
    "remote_add": handle_remote_add,
    "remote_bind": handle_remote_bind,
    "remote_remove": handle_remote_remove,
    "remote_remove_machine": handle_remote_remove_machine,
    "remote_setup": handle_remote_setup,
    "remote_status": handle_remote_status,
    "exit_auto_mode": handle_exit_auto_mode,
    "set_auto_stage": handle_set_auto_stage,
}


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    project_dir: Path,
    **extra_kwargs: Any,
) -> ToolResult:
    """Dispatch a tool call to the appropriate handler."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return ToolResult(content=f"Error: unknown tool '{tool_name}'")

    # Tools that don't need project_dir
    if tool_name in (
        "ask_user", "list_skills", "list_models", "hf_repo_info",
        "exit_auto_mode", "set_auto_stage",
    ):
        return await handler(**arguments)
    if tool_name == "load_skill":
        return await handler(**arguments)

    # Pass extra kwargs (e.g. pipeline callbacks) through to the handler
    return await handler(project_dir, **arguments, **extra_kwargs)
