"""Tool execution dispatch and handlers for lqh agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lqh.project_identity import cloud_project_key as _ckey
from typing import Any, Callable, Awaitable

from lqh.skills import list_available_skills, load_skill_content
from lqh.tools.permissions import PERMISSIONS_FILE, PermissionContext
from lqh.update_check import install_extras_command

logger = logging.getLogger(__name__)


# Truncation threshold: ~40,000 chars (~10k tokens)
TRUNCATION_THRESHOLD = 40_000


# Sentinel content value: the tool produced a one-time secret that must be
# delivered to the user out-of-band (never into the conversation). The agent
# loop intercepts this (like PERMISSION_REQUIRED) and replaces the result with
# the redacted message before anything is persisted. See SecretDelivery.
SECRET_DELIVERY_REQUIRED = "SECRET_DELIVERY_REQUIRED"


@dataclass
class SecretDelivery:
    """A one-time secret to hand to the user out-of-band.

    Carried on a transient ``ToolResult`` whose ``content`` is the
    ``SECRET_DELIVERY_REQUIRED`` sentinel. The agent loop shows ``display`` to
    the user (TUI panel) and/or appends ``payload`` to ``.env``, then returns a
    *new* ``ToolResult`` whose content is ``redacted`` — so the plaintext never
    reaches ``session.messages`` (and therefore never the local JSONL log nor
    the backend payload capture).
    """
    payload: str          # plaintext secret — out-of-band only, never logged
    display: str          # full TUI message incl. the secret + "copy now" warning
    redacted: str         # message that lands in the conversation (no secret)
    env_var: str          # env var name for the optional .env append
    env_comment: str | None = None  # comment line written above the .env entry


# Error taxonomy for structured tool failures (CLI_PLAN §5.3).
# "conflict" covers overwrite-guard / already-exists refusals.
ERROR_KINDS = frozenset({
    "auth", "permission", "config", "validation",
    "not_found", "conflict", "upstream", "runtime",
})


@dataclass
class ToolResult:
    """Result from a tool execution."""
    content: str
    requires_user_input: bool = False
    question: str | None = None
    options: list[str] | None = None
    multi_select: bool = False
    # For PERMISSION_REQUIRED results, the exact permission key to grant on
    # approval (e.g. "training:<run_name>"). Lets the agent grant only the
    # specific action the user approved instead of a project-wide flag.
    permission_key: str | None = None
    show_file_path: str | None = None
    # Optional instruction the agent attaches to the dataset viewer, shown
    # as a banner above the data (e.g. "Review these samples for tone").
    show_file_message: str | None = None
    skill_content: str | None = None
    # Set with content==SECRET_DELIVERY_REQUIRED to hand a one-time secret to
    # the user out-of-band. Never serialized into the conversation.
    secret: SecretDelivery | None = None
    # Auto-mode signals. The agent loop checks these after each tool call.
    exit_auto_mode: bool = False
    auto_status: str | None = None  # "success" | "failure"
    auto_reason: str | None = None
    auto_stage: str | None = None
    auto_stage_note: str | None = None
    # True only after a downstream evaluation/training launch has actually
    # been accepted. Permission and compute-picker sentinels deliberately
    # leave this false so pipeline readiness cannot complete prematurely.
    workflow_launched: bool = False
    # Structured outcome (CLI_PLAN §5.3). None = legacy/unclassified —
    # downstream consumers fall back to the "Error:"/"❌" prefix sniff.
    # Interactive sentinels (PERMISSION_REQUIRED etc.) are not failures
    # and keep ok=None.
    ok: bool | None = None
    error_kind: str | None = None  # member of ERROR_KINDS
    retryable: bool = False
    details: dict[str, Any] | None = None

    @classmethod
    def fail(
        cls,
        kind: str,
        content: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ToolResult":
        """Build a classified failure. ``content`` keeps the exact prose the
        agent sees; the structured fields ride alongside for headless
        consumers."""
        if kind not in ERROR_KINDS:
            raise ValueError(f"unknown error_kind {kind!r}")
        return cls(
            content=content,
            ok=False,
            error_kind=kind,
            retryable=retryable,
            details=details,
            **kwargs,
        )


def _validate_path(project_dir: Path, rel_path: str) -> Path:
    """Validate and resolve a path within the project directory."""
    resolved = (project_dir / rel_path).resolve()
    project_resolved = project_dir.resolve()
    try:
        resolved.relative_to(project_resolved)
    except ValueError as exc:
        raise ValueError(f"Path '{rel_path}' is outside the project directory") from exc
    return resolved


def _validate_writable_path(project_dir: Path, rel_path: str) -> Path:
    """_validate_path plus a deny on the CLI's own state directory.

    ``.lqh/`` holds security-relevant state the agent must not author:
    permissions.json (user consent grants, including cloud spend) and
    data_gen_validation.json (the handler-enforced cloud-validation
    gate, whose source_paths/needs_hf feed bundle contents and HF-token
    injection). Letting the model write there would turn every
    "handler-enforced, not prompt-trusted" gate into a prompt-trusted
    one.
    """
    resolved = _validate_path(project_dir, rel_path)
    rel = resolved.relative_to(project_dir.resolve())
    if rel.parts and rel.parts[0] == ".lqh":
        raise ValueError(
            f"Path '{rel_path}' is inside .lqh/ — CLI-internal state is not "
            "writable through file tools"
        )
    return resolved


def _resolve_training_sources(
    project_dir: Path,
    spec: "str | list[Any]",
    *,
    kind: str,
    allow_repeat: bool,
) -> "tuple[list[dict[str, Any]], list[Path], str | None]":
    """Validate one or more dataset sources and resolve them to canonical
    config entries.

    *spec* is the agent-facing form: a single dataset-DIRECTORY path, a list
    of such paths, or (train only) a list of ``{"path", "repeat"}`` objects.
    Each directory must contain ``data.parquet``.

    Returns ``(entries, resolved_parquet_paths, error)``. On the first failure
    *error* is a human-readable string and the other two values are empty.
    Each entry is ``{"path": <project-rel data.parquet>, "repeat": int,
    "source": <label>}`` (``repeat`` omitted when *allow_repeat* is False).
    """
    from lqh.train.data_utils import normalize_sources

    try:
        raw = normalize_sources(spec, allow_repeat=allow_repeat)
    except ValueError as exc:
        return [], [], f"Error: invalid {kind}: {exc}"

    entries: list[dict[str, Any]] = []
    resolved: list[Path] = []
    project_resolved = project_dir.resolve()
    for i, src in enumerate(raw, start=1):
        try:
            ds_path = _validate_path(project_dir, src["path"])
        except ValueError as exc:
            return [], [], f"Error: {kind} source {i}: {exc}"
        data_parquet = ds_path / "data.parquet"
        if not data_parquet.exists():
            return [], [], (
                f"Error: {kind} source {i} not found at {src['path']}/data.parquet"
            )
        rel = data_parquet.relative_to(project_resolved).as_posix()
        entry: dict[str, Any] = {"path": rel}
        if allow_repeat:
            entry["repeat"] = src["repeat"]
        entries.append(entry)
        resolved.append(data_parquet.resolve())

    # Derive stable, disambiguated source labels from the resolved parquet
    # paths (parent dir name) — reuse normalize_sources' labelling so it
    # matches what load_eval_sources/load_chatml_datasets derive downstream.
    labels = normalize_sources([e["path"] for e in entries], allow_repeat=False)
    for entry, lab in zip(entries, labels):
        entry["source"] = lab["source"]

    return entries, resolved, None


def _sources_to_config(entries: "list[dict[str, Any]]") -> "str | list[dict[str, Any]]":
    """Canonical config value for a resolved source list.

    A single source with no over-sampling collapses to a bare path string —
    byte-for-byte the legacy single-dataset config shape, so existing configs,
    bundles, and downstream readers are unaffected. Multi-source (or a
    ``repeat`` > 1) keeps the full list of ``{"path", ...}`` entries.
    """
    if len(entries) == 1 and entries[0].get("repeat", 1) == 1:
        return entries[0]["path"]
    return entries


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
    from lqh.scoring import (
        format_score_distribution_text,
        is_scoring_error,
        score_distribution_stats,
    )

    try:
        table = pq.read_table(str(scores_path))
        if "score" not in table.column_names:
            return ""
        raw_scores = table.column("score").to_pylist()
        if "reasoning" in table.column_names:
            # 0 is the rubric's worst grade, not an error marker — exclude
            # only samples the judge actually failed to score.
            reasons = table.column("reasoning").to_pylist()
            scores = [
                s for s, r in zip(raw_scores, reasons)
                if s is not None and not is_scoring_error(r)
            ]
        else:
            # No reasoning column to tell errors apart — fall back to the
            # legacy proxy of dropping the 0.0 error placeholders.
            scores = [s for s in raw_scores if s is not None and s > 0]
    except Exception:
        return ""

    dist = score_distribution_stats(scores)
    if dist is None:
        return ""
    return format_score_distribution_text(dist)


def _fmt_size(size: int) -> str:
    """Format a file size as a human-readable string."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def _eval_spec_hash(project_dir: Path) -> str | None:
    """Submission-time spec hash for eval/infer configs (R6 traceability)."""
    from lqh.project_meta import compute_spec_sha256

    return compute_spec_sha256(project_dir)


def _summarize_datasets(project_dir: Path) -> list[str]:
    datasets_dir = project_dir / "datasets"
    if not datasets_dir.is_dir():
        return []
    datasets = sorted(
        [d for d in datasets_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    lines = [f"- **datasets/**: {len(datasets)} dataset(s)"]
    for d in datasets[:15]:
        parquet_files = [
            p for p in d.glob("*.parquet") if p.name != "scores.parquet"
        ]
        if not parquet_files:
            lines.append(f"  - {d.name}: (empty)")
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

        # Provenance from existing sidecars (best-effort). manifest.json
        # (Phase 4 finalization manifests) is authoritative when present.
        provenance = ""
        manifest_path = d / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                purpose = manifest.get("purpose")
                if purpose:
                    provenance += f", {purpose}"
                parent = manifest.get("parent_dataset") or manifest.get("parent")
                if parent:
                    provenance += f", supplements {Path(str(parent)).name}"
                derived = manifest.get("derived_from")
                if derived:
                    origin = Path(str(derived))
                    origin_name = (
                        origin.parent.name
                        if origin.suffix == ".parquet" and origin.parent.name
                        else origin.name
                    )
                    provenance += f", filtered from {origin_name}"
                recorded_spec = manifest.get("spec_sha256") or manifest.get("spec_hash")
                if recorded_spec:
                    from lqh.project_meta import compute_spec_sha256

                    current_spec = compute_spec_sha256(project_dir)
                    if current_spec and recorded_spec != current_spec:
                        provenance += ", built against an OLDER spec"
                    elif current_spec:
                        provenance += ", spec ✓"
            except Exception:
                pass
        filter_summary = d / "summary.json"
        if filter_summary.exists():
            try:
                fs = json.loads(filter_summary.read_text(encoding="utf-8"))
                kept, total = fs.get("kept"), fs.get("total")
                threshold = fs.get("threshold")
                if kept is not None and total:
                    provenance += f", filtered {kept}/{total}"
                    if threshold is not None:
                        provenance += f" @ ≥{threshold}"
            except Exception:
                pass
        source_sidecar = d / ".lqh_source.json"
        if source_sidecar.exists():
            try:
                src = json.loads(source_sidecar.read_text(encoding="utf-8"))
                origin = src.get("run_name") or src.get("job_id")
                if origin:
                    provenance += f", cloud output of {origin}"
            except Exception:
                pass

        mtime = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"  - {d.name}{label}: {', '.join(ds_info)}{score_info}{provenance} [{mtime}]"
        )
    if len(datasets) > 15:
        lines.append(
            f"  …{len(datasets) - 15} older datasets not shown (use list_files datasets/)"
        )
    return lines


def _summarize_prompts(project_dir: Path) -> list[str]:
    prompts_dir = project_dir / "prompts"
    if not prompts_dir.is_dir():
        return []
    prompt_files = sorted(
        list(prompts_dir.glob("*.md")) + list(prompts_dir.glob("*.schema.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not prompt_files:
        return []
    lines = [f"- **prompts/**: {len(prompt_files)} file(s)"]
    for p in prompt_files[:10]:
        lines.append(f"  - {p.name}")
    if len(prompt_files) > 10:
        lines.append(
            f"  …{len(prompt_files) - 10} more not shown (use list_files prompts/)"
        )
    return lines


def _progress_terminal(run_dir: Path) -> tuple[str, str | None] | None:
    """Last terminal status row from progress.jsonl: (state, error) or None."""
    path = run_dir / "progress.jsonl"
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = row.get("status")
        if status in ("completed", "failed", "cancelled", "interrupted"):
            state = "failed" if status == "interrupted" else status
            return state, row.get("error")
    return None


def _dataset_display_name(path_str: str) -> str:
    """Dataset name for display: canonical paths end in data.parquet, and
    'data: data.parquet' tells the reader nothing — use the dataset dir."""
    p = Path(path_str)
    if p.suffix == ".parquet" and p.parent.name not in ("", "."):
        return p.parent.name
    return p.name


def _dataset_entry_name(value: Any) -> str | None:
    """Display name of one training-dataset entry (string path or dict form)."""
    if isinstance(value, str) and value:
        return _dataset_display_name(value)
    if isinstance(value, dict):
        for key in ("path", "dataset", "dataset_path", "name"):
            inner = value.get(key)
            if isinstance(inner, str) and inner:
                name = _dataset_display_name(inner)
                repeat = value.get("repeat") or value.get("repeats")
                return f"{name}×{repeat}" if repeat else name
    return None


def _run_updated_at(run_dir: Path) -> float:
    """Best-known last-activity time: progress/status files beat dir mtime.

    Appending to progress.jsonl does not touch the directory mtime, so
    sorting by dir mtime alone would order active runs as stale.
    """
    times = [run_dir.stat().st_mtime]
    for name in ("progress.jsonl", "status.json", "cloud_state.json"):
        try:
            times.append((run_dir / name).stat().st_mtime)
        except OSError:
            pass
    return max(times)


def _run_status_line(run_dir: Path) -> str:
    """One line of semantic status for a training/eval/inference run."""
    from lqh.subprocess_manager import SubprocessManager

    config: dict[str, Any] = {}
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    remote_job = run_dir / "remote_job.json"
    submit_intent = run_dir / "submit_intent.json"
    if remote_job.exists():
        # Cloud submissions stamp `"backend": "cloud"`; SSH metadata has
        # the same job_id/remote_name fields but no backend marker.
        remote_kind = "remote"
        try:
            rj = json.loads(remote_job.read_text(encoding="utf-8"))
            remote_kind = "cloud" if rj.get("backend") == "cloud" else "ssh"
        except Exception:
            pass
        status: str | None = None
        error: str | None = None
        cloud_state = run_dir / "cloud_state.json"
        if cloud_state.exists():
            try:
                cs = json.loads(cloud_state.read_text(encoding="utf-8"))
                if cs.get("status") in ("completed", "failed", "cancelled"):
                    status = cs.get("status")
                    error = cs.get("error")
            except Exception:
                pass
        terminal = _progress_terminal(run_dir)
        if status is None:
            # SSH runs (and stale cloud state): the synced progress log
            # carries the terminal verdict and failure reason.
            if terminal:
                status, error = terminal
            else:
                status = "running (as of last sync)"
        elif error is None and terminal:
            # cloud_state.json records the terminal status but not the
            # failure reason — the replayed progress row carries it.
            error = terminal[1]
        status_desc = f"{remote_kind}, {status}"
        if error:
            status_desc += f" — {str(error)[:80]}"
    elif submit_intent.exists():
        # Idempotency marker without an accepted job: fate unknown.
        status_desc = "submitted, fate unknown (submit_intent.json present)"
    else:
        st = SubprocessManager().get_status(run_dir)
        status_desc = st.state
        if st.step is not None:
            status_desc += f" @ step {st.step}"
            if st.loss is not None:
                status_desc += f", loss {st.loss:.4g}"
        if st.state == "failed" and st.error:
            status_desc += f" — {str(st.error)[:80]}"

    # Sweep configs nest the model/data facts under base_config.
    base_config = config.get("base_config")
    if not isinstance(base_config, dict):
        base_config = {}

    def _cfg(key: str) -> Any:
        value = config.get(key)
        return value if value is not None else base_config.get(key)

    extras = []
    base_model = _cfg("base_model") or _cfg("model")
    if base_model:
        extras.append(str(base_model))
    for key in ("datasets", "dataset", "dataset_path", "data_path", "eval_dataset"):
        value = _cfg(key)
        if isinstance(value, list) and value:
            names = [n for n in (_dataset_entry_name(v) for v in value[:3]) if n]
            if names:
                extras.append(f"data: {', '.join(names)}")
            break
        name = _dataset_entry_name(value)
        if name:
            extras.append(f"data: {name}")
            break
    checkpoints_dir = run_dir / "checkpoints"
    try:
        if checkpoints_dir.is_dir() and any(checkpoints_dir.iterdir()):
            extras.append("ckpt ✓")
    except OSError:
        pass
    spec_note = _manifest_spec_note(run_dir, run_dir.parent.parent)
    if spec_note:
        extras.append(spec_note)
    suffix = f" ({'; '.join(extras)})" if extras else ""
    return f"{run_dir.name}: {status_desc}{suffix}"


def _manifest_spec_note(artifact_dir: Path, project_dir: Path) -> str | None:
    """Spec match/mismatch marker from an artifact's manifest, if any."""
    try:
        manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        recorded = manifest.get("spec_sha256") or manifest.get("spec_hash")
        if not recorded:
            return None
        from lqh.project_meta import compute_spec_sha256

        current = compute_spec_sha256(project_dir)
        if not current:
            return None
        return "spec ✓" if recorded == current else "built against an OLDER spec"
    except Exception:
        return None


def _summarize_runs(project_dir: Path) -> list[str]:
    runs_dir = project_dir / "runs"
    if not runs_dir.is_dir():
        return []
    runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=_run_updated_at,
        reverse=True,
    )
    lines = [f"- **runs/**: {len(runs)} run(s)"]
    for r in runs[:10]:
        try:
            lines.append(f"  - {_run_status_line(r)}")
        except Exception:
            lines.append(f"  - {r.name}")
    if len(runs) > 10:
        lines.append(
            f"  …{len(runs) - 10} older runs not shown (use list_files runs/)"
        )
    return lines


def _summarize_cloud(project_dir: Path) -> list[str]:
    """Cloud facts from the cached snapshot only — never touches the network."""
    from lqh.snapshot import read_cached_snapshot

    wrapper = read_cached_snapshot(project_dir)
    if wrapper is None:
        return []
    snap = wrapper.get("snapshot") or {}
    if not isinstance(snap, dict):
        snap = {}
    # NOTE: an empty core snapshot must NOT short-circuit — wrapper-level
    # deployments/artifacts exist independently (e.g. the project endpoint
    # 404s but live deployments were fetched).
    fetched = wrapper.get("fetched_at") or "unknown time"
    # The summary tool never fetches — this is always the cached view, and
    # the label must say so (a snapshot cached before going offline would
    # otherwise read as current).
    lines = [
        f"\n- **Cloud** (cached snapshot from {fetched} — may lag live "
        "state; verify with training_status/list_deployments/artifacts):"
    ]

    jobs = snap.get("jobs") or snap.get("recent_jobs") or []
    if isinstance(jobs, list) and jobs:
        lines.append(f"  - {len(jobs)} recent cloud job(s):")
        for job in jobs[:5]:
            if not isinstance(job, dict):
                continue
            job_id = job.get("job_id") or job.get("id") or "?"
            status = job.get("status") or "?"
            kind = job.get("kind") or job.get("purpose") or ""
            kind_label = f" {kind}" if kind else ""
            lines.append(f"    - {job_id}{kind_label}: {status}")
        if len(jobs) > 5:
            lines.append(f"    …{len(jobs) - 5} more not shown")
        if wrapper.get("jobs_truncated"):
            lines.append(
                "    …job list incomplete (client cap reached or paging "
                "interrupted) — older jobs are not shown here"
            )

    spend = snap.get("lifetime_spend_micros")
    if isinstance(spend, (int, float)) and spend > 0:
        lines.append(f"  - lifetime cloud spend: ${spend / 1_000_000:.2f}")

    best = snap.get("best_checkpoint")
    if isinstance(best, dict) and best:
        best_id = best.get("artifact_id") or best.get("id") or best.get("name")
        if best_id:
            lines.append(f"  - selected best checkpoint: {best_id}")

    stale_sections = wrapper.get("stale_sections") or []

    artifacts = wrapper.get("artifacts")
    if "artifacts" in stale_sections and not artifacts:
        # A stale section with nothing carried forward must not vanish
        # silently — absence of data is not absence of artifacts.
        lines.append(
            "  - artifact list unavailable (last refresh failed and no "
            "older data was cached)"
        )
    if isinstance(artifacts, list) and artifacts:
        stale_note = (
            " (STALE — last refresh failed, carried from an older snapshot)"
            if "artifacts" in stale_sections else ""
        )
        lines.append(f"  - {len(artifacts)} cloud artifact(s):{stale_note}")
        for art in artifacts[:5]:
            if not isinstance(art, dict):
                continue
            art_id = art.get("artifact_id") or art.get("id") or "?"
            kind = art.get("kind") or "?"
            name = art.get("name") or art.get("logical_name") or ""
            name_label = f" {name}" if name else ""
            lines.append(f"    - {art_id} [{kind}]{name_label}")
        if len(artifacts) > 5:
            lines.append(f"    …{len(artifacts) - 5} more not shown (use the artifacts tool)")
        if wrapper.get("artifacts_truncated"):
            lines.append(
                "    …artifact list incomplete (client cap reached or "
                "paging interrupted) — older artifacts are not shown here"
            )

    # Deployments live at the wrapper top level (fetched separately from
    # the project snapshot); the in-snapshot key is a fallback.
    deployments = wrapper.get("deployments")
    if not isinstance(deployments, list):
        deployments = snap.get("deployments")
    if "deployments" in stale_sections and not deployments:
        lines.append(
            "  - deployment state unavailable (last refresh failed and no "
            "older data was cached)"
        )
    if isinstance(deployments, list) and deployments:
        stale_note = (
            " (STALE — last refresh failed, carried from an older snapshot)"
            if "deployments" in stale_sections else ""
        )
        lines.append(f"  - {len(deployments)} deployment(s):{stale_note}")
        for dep in deployments[:5]:
            if not isinstance(dep, dict):
                continue
            name = dep.get("name") or dep.get("deployment_id") or dep.get("id") or "?"
            status = dep.get("status") or "?"
            lines.append(f"    - {name}: {status}")
        if len(deployments) > 5:
            lines.append(
                f"    …{len(deployments) - 5} more not shown (use list_deployments)"
            )

    unattributed = wrapper.get("unattributed_deployments")
    if isinstance(unattributed, list) and unattributed:
        # These rows carry no project attribution at all — they may or
        # may not belong here, so they are never presented as this
        # project's deployments.
        lines.append(
            f"  - {len(unattributed)} deployment(s) NOT attributed to any "
            "project (may belong to another project; verify with "
            "list_deployments before redeploying):"
        )
        for dep in unattributed[:3]:
            if not isinstance(dep, dict):
                continue
            name = dep.get("name") or dep.get("deployment_id") or dep.get("id") or "?"
            status = dep.get("status") or "?"
            lines.append(f"    - {name}: {status}")
        if len(unattributed) > 3:
            lines.append(f"    …{len(unattributed) - 3} more not shown")

    if len(lines) == 1:
        return []  # nothing beyond the header — omit the section
    return lines


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

    # Agent notes (prose handoff, see NOTES.md convention)
    notes = project_dir / "NOTES.md"
    if notes.exists():
        stat = notes.stat()
        parts.append(
            f"- **NOTES.md**: {stat.st_size} bytes, modified "
            f"{datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()}"
        )

    # Other specs
    other_specs = project_dir / "other_specs"
    if other_specs.is_dir():
        specs = list(other_specs.iterdir())
        if specs:
            parts.append(f"- **other_specs/**: {len(specs)} file(s)")
            for s in specs[:10]:
                parts.append(f"  - {s.name}")
            if len(specs) > 10:
                parts.append(f"  …{len(specs) - 10} more not shown (use list_files other_specs/)")

    # Data gen pipelines
    data_gen = project_dir / "data_gen"
    if data_gen.is_dir():
        scripts = list(data_gen.glob("*.py"))
        parts.append(f"- **data_gen/**: {len(scripts)} pipeline(s)")
        for s in scripts[:10]:
            parts.append(f"  - {s.name}")
        if len(scripts) > 10:
            parts.append(f"  …{len(scripts) - 10} more not shown (use list_files data_gen/)")

    parts.extend(_summarize_datasets(project_dir))
    parts.extend(_summarize_prompts(project_dir))
    parts.extend(_summarize_runs(project_dir))

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
                if len(scorer_files) > 10:
                    parts.append(
                        f"  …{len(scorer_files) - 10} more not shown (use list_files evals/scorers/)"
                    )

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
                    spec_note = _manifest_spec_note(er, project_dir)
                    spec_suffix = f" ({spec_note})" if spec_note else ""
                    summary_file = er / "summary.json"
                    if summary_file.exists():
                        # Broad except: a malformed artifact (e.g.
                        # {"scores": null}) must degrade to a bare name,
                        # never abort the whole summary/startup path.
                        try:
                            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                            scores = summary_data.get("scores") or {}
                            if not isinstance(scores, dict):
                                scores = {}
                            mean = scores.get("mean", "?")
                            n = summary_data.get("num_samples", "?")
                            parts.append(f"  - {er.name}: mean {mean}/10 ({n} samples){spec_suffix}")
                        except Exception:
                            parts.append(f"  - {er.name}{spec_suffix}")
                    else:
                        parts.append(f"  - {er.name} (no summary){spec_suffix}")
                if len(eval_runs) > 10:
                    parts.append(
                        f"  …{len(eval_runs) - 10} older eval runs not shown "
                        "(use list_files evals/runs/)"
                    )

    parts.extend(_summarize_cloud(project_dir))

    # Recent conversations (covers both the v2 directory format and
    # unmigrated legacy single-file sessions).
    convos_dir = project_dir / ".lqh" / "conversations"
    if convos_dir.is_dir():
        from lqh.session import Session

        sessions = Session.list_sessions(project_dir)
        if sessions:
            parts.append(f"\n- **Conversations**: {len(sessions)} session(s)")
            for s in sessions[:5]:
                preview = s.get("preview", "")[:60]
                state = s.get("state", "")
                state_label = f" [{state}]" if state and state != "completed" else ""
                parts.append(
                    f"  - {s.get('created_at', '?')}: {preview}{state_label}"
                )
            if len(sessions) > 5:
                parts.append(
                    f"  …{len(sessions) - 5} older session(s) not shown (use /resume to browse)"
                )

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
    target = _validate_writable_path(project_dir, path)
    if target.exists():
        return ToolResult(content=f"Error: file '{path}' already exists. Use write_file to overwrite.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    lines = content.count("\n") + 1
    return ToolResult(content=f"✅ Created {path} ({lines} lines, {len(content):,} chars)")


async def handle_write_file(project_dir: Path, *, path: str, content: str, **kwargs: Any) -> ToolResult:
    """Write/overwrite a file."""
    target = _validate_writable_path(project_dir, path)
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
    target = _validate_writable_path(project_dir, path)
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
    purpose: str = "unspecified",
    execution: str = "local",
    timeout_minutes: int = 720,
    overwrite: bool = False,
    parent_dataset: str | None = None,
    _permissions: PermissionContext | None = None,
    _overwrite_consent: bool = False,
    **kwargs: Any,
) -> ToolResult:
    """Execute a data generation pipeline. Requires user permission.

    ``execution="cloud"`` submits a background CPU cloud job instead of
    running in-process (CLOUD_OFFLOAD_PLAN.md §2), gated on a prior
    successful local run of the same pipeline version plus user consent.
    ``_permissions`` is internal: the agent loop passes an invocation-scoped
    ``PermissionContext`` grant when re-invoking after the user approved the
    corresponding prompt, so a one-time grant works without persisting
    anything; headless surfaces pass full consent.
    """
    perms = _permissions or PermissionContext()

    if execution not in ("local", "cloud"):
        return ToolResult.fail(
            "validation",
            f"Error: execution must be 'local' or 'cloud', got {execution!r}",
        )
    if num_samples <= 0:
        return ToolResult.fail(
            "validation",
            f"Error: num_samples must be positive, got {num_samples}",
        )
    if samples_per_item <= 0:
        return ToolResult.fail(
            "validation",
            f"Error: samples_per_item must be positive, got {samples_per_item}",
        )
    # Mirror the backend picker's clamp so the consent prompt's cost cap
    # matches what actually gets scheduled.
    timeout_minutes = max(10, min(int(timeout_minutes or 720), 1440))
    # output_dataset becomes a path component (datasets/<name>/, and the
    # cloud run dir name) — require a plain directory name so it can't
    # escape the project layout or hide runs from the watcher.
    if (
        not output_dataset
        or output_dataset in (".", "..")
        or "/" in output_dataset
        or "\\" in output_dataset
    ):
        return ToolResult.fail(
            "validation",
            (
                f"Error: output_dataset must be a plain name (no path separators), "
                f"got {output_dataset!r}"
            ),
        )

    # Fast read-only immutability refusal BEFORE the permission prompt so
    # the agent gets immediate feedback; the atomic claim below re-checks
    # under the cross-process lock right before work starts.
    from lqh.dataset_guard import overwrite_refusal

    early_refusal = overwrite_refusal(project_dir, output_dataset, overwrite=overwrite)
    if early_refusal:
        return ToolResult.fail("conflict", f"Error: {early_refusal}")

    target = _validate_path(project_dir, script_path)
    if not target.exists():
        return ToolResult.fail("not_found", f"Error: script '{script_path}' does not exist")
    # Pipelines must sit directly under data_gen/: the engine derives the
    # project root as script_path.parent.parent, so a script anywhere else
    # resolves source() against the wrong directory (and the cloud bundle
    # layout + import pre-scan assume the same location).
    rel_parts = target.relative_to(project_dir.resolve()).parts
    if len(rel_parts) != 2 or rel_parts[0] != "data_gen" or not rel_parts[1].endswith(".py"):
        return ToolResult.fail(
            "validation",
            (
                f"Error: pipeline scripts must be .py files directly under data_gen/ "
                f"(got '{script_path}'). Move the script to data_gen/<name>.py and retry."
            ),
        )

    # Pre-validate imports before executing
    try:
        source = target.read_text(encoding="utf-8")
        bad_imports = [
            ("from data_gen.", "from data_gen."),
            ("from data_gen import", "from data_gen import"),
            ("import data_gen.", "import data_gen."),
            ("from pipeline import", "from pipeline import"),
            ("import pipeline\n", "import pipeline"),
        ]
        for pattern, display in bad_imports:
            if pattern in source:
                return ToolResult.fail(
                    "validation",
                    (
                        f"Error: Pipeline has incorrect import: `{display}`\n"
                        f"Fix: use `from lqh.pipeline import Pipeline, ChatMLMessage, Conversation`\n"
                        f"All pipeline imports must come from `lqh.pipeline`, not `data_gen` or `pipeline`."
                    ),
                )
    except OSError:
        pass

    # Check if we already have permission (script-execution domain;
    # applies to both execution targets — cloud runs the same script).
    if not perms.allows_script(project_dir, script_path):
        # Need to ask for permission - this will be handled by the agent loop
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            permission_key=f"script:{script_path}",
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

    # Expensive outputs are immutable by default (PERSISTENCY_PLAN.md R5).
    # claim_output is an atomic check-and-reserve under a cross-process
    # lock: it refuses when finalized data exists (a leftover
    # data.partial.jsonl does NOT bypass this — resume only applies while
    # no data.parquet exists) and when another live process is currently
    # generating into the same name.
    from lqh.dataset_guard import (
        claim_output,
        existing_output,
        pending_cloud_job,
        release_output,
    )

    dataset_dir = project_dir / "datasets" / output_dataset
    existing_name = existing_output(project_dir, output_dataset)
    pending_run = pending_cloud_job(project_dir, output_dataset)
    if overwrite and (existing_name or pending_run) and not _overwrite_consent:
        # overwrite=true from the model is a REQUEST, not consent —
        # destroying data (or racing a pending cloud job's output slot)
        # needs an explicit human yes (the agent loop relays this prompt
        # and re-invokes with the consent flag).
        if existing_name:
            question = (
                f"The agent wants to OVERWRITE datasets/{output_dataset}/ — "
                f"the existing {existing_name} (and any other files in the "
                "dataset directory) will be destroyed and regenerated. Data "
                "generation is expensive and this cannot be undone. Allow?"
            )
        else:
            question = (
                f"datasets/{output_dataset}/ is the pending output of cloud "
                f"data-gen job '{pending_run}'. The agent wants to start "
                "ANOTHER writer for the same dataset name — the two outputs "
                "will race (newest submission wins). Allow anyway?"
            )
        return ToolResult(
            content="OVERWRITE_CONFIRMATION_REQUIRED",
            requires_user_input=True,
            question=question,
            options=[
                "Yes, destroy/replace and regenerate this dataset",
                "No, keep the existing data",
            ],
        )

    refusal = claim_output(project_dir, output_dataset, overwrite=overwrite)
    if refusal:
        return ToolResult.fail("conflict", f"Error: {refusal}")

    if overwrite and existing_name:
        # Confirmed overwrite: drop EVERY artifact describing the OLD
        # contents — all parquet splits plus provenance/score sidecars —
        # so the regenerated dataset can't mix with or misdescribe them.
        try:
            stale_files = list(dataset_dir.glob("*.parquet"))
        except OSError:
            stale_files = []
        stale_files += [
            dataset_dir / name
            for name in ("manifest.json", "summary.json", ".lqh_source.json")
        ]
        for stale in stale_files:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        if execution == "cloud":
            # After submission the cloud job outlives this process; the
            # download-side newest-submission-wins policy governs overlap
            # from here, so the pid-scoped claim is released either way.
            return await _submit_cloud_data_gen(
                project_dir,
                script_path=script_path,
                num_samples=num_samples,
                output_dataset=output_dataset,
                validation_instructions=validation_instructions,
                samples_per_item=samples_per_item,
                purpose=purpose,
                timeout_minutes=timeout_minutes,
                permissions=perms,
                on_bg_started=kwargs.get("on_background_task_started"),
                hf_donate=kwargs.get("_hf_donate"),
            )

        # Execute the pipeline (pass through any callbacks from kwargs)
        return await _execute_pipeline(
            project_dir, script_path, num_samples, output_dataset, validation_instructions,
            samples_per_item=samples_per_item,
            purpose=purpose,
            parent_dataset=parent_dataset,
            on_pipeline_progress=kwargs.get("on_pipeline_progress"),
            on_pipeline_done=kwargs.get("on_pipeline_done"),
            legacy_progress_callback=bool(kwargs.get("legacy_progress_callback", True)),
        )
    finally:
        release_output(project_dir, output_dataset)


async def _fetch_billed_rate_usd(field: str) -> float | None:
    """A billed $/hr rate field from GET /v1/cloud/pricing; None if
    unreachable or absent (e.g. an older backend without the field).

    Fetched live so consent prompts can't drift from operator overrides
    of the rate or margin envvars; callers fall back to the defaults
    with an explicit "at default rates" caveat.
    """
    try:
        import httpx

        from lqh.auth import api_root, get_token

        token = get_token()
        if not token:
            return None
        async with httpx.AsyncClient(base_url=api_root(), timeout=5.0) as client:
            resp = await client.get(
                "/v1/cloud/pricing",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                return None
            micros = resp.json().get(field)
            if isinstance(micros, (int, float)) and micros > 0:
                return float(micros) / 1e6
            return None
    except Exception:
        return None


async def _fetch_data_gen_rate_usd() -> float | None:
    """Billed data_gen $/hr; None if unreachable."""
    return await _fetch_billed_rate_usd("data_gen_cpu_rate_billed_micros_per_hour")


async def _fetch_eval_hf_rate_usd() -> float | None:
    """Billed eval_hf GPU $/hr; None if unreachable."""
    return await _fetch_billed_rate_usd("eval_hf_gpu_rate_billed_micros_per_hour")


async def _submit_cloud_data_gen(
    project_dir: Path,
    *,
    script_path: str,
    num_samples: int,
    output_dataset: str,
    validation_instructions: str | None,
    samples_per_item: int,
    purpose: str,
    timeout_minutes: int = 720,
    permissions: PermissionContext,
    on_bg_started: Callable[[str, str, str, str | None], None] | None = None,
    hf_donate: bool | None = None,
) -> ToolResult:
    """Submit the pipeline as a background cloud CPU job.

    Two gates, in order (both after the script-execution permission
    handled by the caller):

    1. Correctness gate — a successful LOCAL run of this exact pipeline
       version (content hash) must be on record. Handler-enforced, not
       prompt-trusted: an agent edit to the file re-arms it.
    2. Consent gate — the user approves the submit (sample count + cost
       estimate) unless the invocation or the project carries a grant.
    """
    from lqh.data_gen_validation import check_validation

    target = _validate_path(project_dir, script_path)
    # Canonical project-relative form: the sandbox resolves script_path
    # against the extracted bundle root, so an absolute (even
    # inside-project) path in config.json would break there.
    script_path = target.relative_to(project_dir.resolve()).as_posix()

    # validation_instructions becomes a bundle-manifest entry — validate
    # it like every other path (a model-supplied absolute or ../ path
    # would otherwise upload an arbitrary readable local file) and store
    # it project-relative so the sandbox finds it under inputs/.
    if validation_instructions:
        try:
            val_resolved = _validate_path(project_dir, validation_instructions)
        except ValueError as e:
            return ToolResult.fail("validation", f"Error: validation_instructions: {e}")
        if not val_resolved.exists():
            return ToolResult.fail(
                "not_found",
                f"Error: validation_instructions file "
                f"'{validation_instructions}' does not exist",
            )
        validation_instructions = val_resolved.relative_to(
            project_dir.resolve()
        ).as_posix()

    record = check_validation(project_dir, target)
    if record is None:
        return ToolResult.fail(
            "config",
            (
                "VALIDATION_REQUIRED: cloud execution is locked until this exact "
                "pipeline version and its recorded local inputs have succeeded locally.\n"
                f"Run `run_data_gen_pipeline` with execution='local' first — a draft "
                f"(num_samples=3, purpose='smoke') to check correctness, then an "
                f"inspection batch (num_samples≈20, purpose='inspection') to review "
                f"quality — and retry execution='cloud' afterwards.\n"
                f"Note: any edit to {script_path} re-arms this gate."
            ),
        )

    # The gate hashes code, not data — a recorded seed input deleted or
    # moved since validation would silently vanish from the bundle
    # (resolve_manifest skips missing paths) and only fail in the paid
    # sandbox. Catch it here instead.
    missing_sources = [s for s in record.source_paths if not (project_dir / s).exists()]
    if missing_sources:
        return ToolResult.fail(
            "config",
            (
                "VALIDATION_REQUIRED: recorded source inputs no longer exist: "
                + ", ".join(missing_sources[:5])
                + (" …" if len(missing_sources) > 5 else "")
                + "\nRestore them or re-run the pipeline locally (which re-records "
                "its inputs), then retry execution='cloud'."
            ),
        )

    total_calls = num_samples * max(1, samples_per_item)
    if not permissions.allows_cloud_data_gen(project_dir):
        rate_usd = await _fetch_data_gen_rate_usd()
        if rate_usd is not None:
            rate_note = f"≈ ${rate_usd:.2f}/hr"
        else:
            rate_usd = 1.0
            rate_note = "≈ $1/hr at default rates"
        hours = timeout_minutes / 60
        inputs_line = ""
        if record.source_paths:
            shown = record.source_paths[:5]
            more = len(record.source_paths) - len(shown)
            inputs_line = (
                f"  Inputs:  {', '.join(shown)}"
                + (f" …and {more} more files" if more > 0 else "")
                + " (uploaded with the job)\n"
            )
        hf_line = ""
        if record.needs_hf:
            # Be explicit that a credential leaves the machine with the
            # job — this is a consent prompt, not a changelog. The
            # donation itself is consented separately (hf_donate, below);
            # this line only sets expectations about what the pipeline
            # needs, since it streams a Hugging Face dataset.
            from lqh.hf_token import (
                DONATE_ALWAYS,
                DONATE_NEVER,
                hf_token_origin,
                resolve_hf_donate_decision,
            )

            origin = hf_token_origin(project_dir)
            if origin is not None and origin.donation_enabled:
                # Three states, not two: the donation question is normally
                # already answered up front, and promising a prompt that
                # will not come is how a user ends up waiting for one.
                decision = resolve_hf_donate_decision(project_dir)
                if decision == DONATE_ALWAYS:
                    outcome = (
                        f"the token from {origin.label} is sent with it "
                        "(you allowed this up front)"
                    )
                elif decision == DONATE_NEVER:
                    outcome = (
                        f"the token from {origin.label} is NOT sent (you declined "
                        "up front) — a private dataset needs a stored account "
                        "token or the job will fail"
                    )
                else:
                    outcome = (
                        f"you'll be asked whether to send the token from "
                        f"{origin.label} with it"
                    )
                hf_line = (
                    "  HF:      pipeline streams a Hugging Face dataset — "
                    f"{outcome} (available to the trusted pipeline Python "
                    "process)\n"
                )
            else:
                hf_line = (
                    "  HF:      pipeline streams a Hugging Face dataset — no local "
                    "token found; private datasets need one (or a stored account "
                    "token) or the job will fail. Any stored token used is "
                    "available to the trusted pipeline Python process\n"
                )
        else:
            # needs_hf only sees lqh.sources.hf_dataset. A pipeline
            # reaching HF another way still gets the donation question,
            # so say so here rather than letting the next prompt arrive
            # unannounced.
            from lqh.hf_token import hf_disclosure_line

            hf_line = hf_disclosure_line(project_dir)
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            permission_key=f"cloud_data_gen:{script_path}",
            question=(
                f"The agent wants to run this data-gen pipeline in the cloud:\n"
                f"  Script:  {script_path} (validated locally: "
                f"{record.succeeded}/{record.num_samples} ok)\n"
                f"  Samples: {num_samples}"
                + (f" × {samples_per_item} per item (≈{total_calls} generations)"
                   if samples_per_item > 1 else "")
                + f"\n  Output:  datasets/{output_dataset}/ (auto-downloads on completion)\n"
                + (f"  Rubric:  {validation_instructions} (uploaded with the job)\n"
                   if validation_instructions else "")
                + inputs_line
                + hf_line
                # Billed by wall-clock at the flat rate — fetched live
                # from /v1/cloud/pricing so operator overrides of the
                # rate/margin can't make this prompt lie; the fallback
                # figures say "at default rates". The timeout is the
                # hard cost cap for the compute part.
                + f"  Compute: billed by wall-clock, {rate_note} — an "
                f"8-hour overnight run bills ≈ ${8 * rate_usd:.0f}; hard cap "
                f"≈ ${hours * rate_usd:.0f} at the {hours:g}-hour timeout. "
                "LLM tokens are billed as usual. The backend allows at most "
                f"{total_calls * 10} LLM requests for this job (10 per requested "
                "output); the expected count shown above assumes one request per output.\n\n"
                "Submit the cloud job?"
            ),
            options=[
                "Submit to cloud (this time)",
                "Submit and don't ask again for this project",
                "Do not submit",
            ],
        )

    # HF donation. Runs after the submit consent above so the user isn't
    # asked about a credential for a job they then decline.
    #
    # Deliberately NOT gated on record.needs_hf, unlike the stored-token
    # injection. needs_hf is an observation of `lqh.sources.hf_dataset`
    # specifically, confirmed against the script text — a deliberately
    # narrow signal, because it decides whether to hand a job the user's
    # ACCOUNT credential without asking. A pipeline reaching HF through
    # datasets.load_dataset, huggingface_hub, or any wrapper indirect
    # enough to defeat the text check is invisible to it, and those
    # pipelines were silently denied the donation too: they worked
    # locally (the resolver finds the token) and 401'd in the cloud.
    #
    # Donation is the opposite kind of decision — explicit, per-job, and
    # declinable — so it does not need a detector to be right. Training
    # and eval already offer it on every cloud submit; data-gen was the
    # only workflow where a detector stood between the user and the
    # question, and the cost of it being wrong was a paid failure.
    donate_hf, hf_prompt = _resolve_hf_donation(
        project_dir, permissions, hf_donate, "data-gen"
    )
    if hf_prompt is not None:
        return hf_prompt

    from datetime import datetime, timezone

    from lqh.remote.backend import RemoteConfig
    from lqh.remote.cloud import CloudBackend, CloudError

    # Random suffix: second-resolution timestamps collide under rapid
    # double-submits, which would share a run dir and clobber its state.
    run_name = "data_gen_{}_{}_{}".format(
        output_dataset,
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        __import__("uuid").uuid4().hex[:6],
    )
    run_dir = project_dir / "runs" / run_name

    config: dict[str, Any] = {
        "kind": "data_gen",
        "type": "data_gen",
        "script_path": script_path,
        "num_samples": num_samples,
        "samples_per_item": samples_per_item,
        # Total work is num_samples × samples_per_item — sizing off
        # num_samples alone would run a 1-item × 100-variant iterate-N×
        # pipeline at concurrency 1 (and cloud bills wall-clock).
        "concurrency": min(100, max(1, num_samples * max(1, samples_per_item))),
        "output_dataset": output_dataset,
        "validation_instructions": validation_instructions,
        # Inputs recorded during the validated local run — the bundle
        # builder resolves each manifest key's value(s) to files/dirs.
        # Pipelines are self-contained single files (sibling imports are
        # unsupported and fail locally), so no code beyond script_path
        # ships.
        "source_paths": record.source_paths,
        "manifest": ["script_path", "validation_instructions", "source_paths"],
        # The backend injects the user's stored HF token into a data_gen
        # sandbox only when the pipeline actually uses HF. Observed
        # during the validated local run (lqh.sources.hf_dataset ran) —
        # not guessed from source text, so wrappers work and unrelated
        # string matches don't leak the token.
        "needs_hf": record.needs_hf,
        # Wall-clock cap; the backend picker clamps to [10, 1440].
        "timeout_minutes": timeout_minutes,
    }

    from lqh.telemetry import active_telemetry
    telemetry = active_telemetry()
    workflow_id = str(__import__("uuid").uuid4())
    if purpose not in {"smoke", "inspection", "validation", "training", "failures", "probe", "imported", "unspecified"}:
        purpose = "unspecified"
    if telemetry:
        await telemetry.run_deferred(telemetry.record_generation_attempt)
        await telemetry.run_deferred(telemetry.event, "data_generation_started", {
            "workflow_kind": "data_generation", "purpose": purpose,
            "requested_count": num_samples, "execution_target": "cloud",
        }, workflow_id)

    cfg = RemoteConfig(
        name="cloud",
        type="cloud",
        hostname="api.lqh.ai",  # informational; CloudBackend hits api_root()
        remote_root="cloud:lqh",
    )
    backend = CloudBackend(cfg, project_dir)
    # Provenance captured BEFORE submission — these are the revisions the
    # bundle is built from. Re-hashing after the network round-trip would
    # attribute the job to whatever the files contain at completion of
    # the POST, not what was uploaded (an external edit during bundle
    # construction/upload would poison the marker and every manifest
    # derived from it). The spec hash additionally rides in the config
    # itself so the sandbox records the same revision.
    from lqh.project_log import file_hash_prefix as _hash_prefix
    from lqh.project_meta import compute_spec_sha256 as _spec_hash

    pre_submit_pipeline_hash = _hash_prefix(project_dir / script_path, n=12)
    pre_submit_spec_sha256 = _spec_hash(project_dir)
    if pre_submit_spec_sha256 and not config.get("spec_sha256"):
        config["spec_sha256"] = pre_submit_spec_sha256
    try:
        job_id = await backend.submit_run(
            str(run_dir), config,
            module="lqh.remote.data_gen",
            telemetry_workflow_id=workflow_id,
            donate_hf_token=donate_hf,
        )
    except CloudError as e:
        if telemetry:
            await telemetry.run_deferred(telemetry.event, "data_generation_failed", {
                "workflow_kind": "data_generation", "purpose": purpose,
                "execution_target": "cloud", "outcome": "failed",
                "requested_count": num_samples,
            }, workflow_id)
        return ToolResult.fail("upstream", f"Error submitting cloud data-gen job: {e}")
    except Exception as e:
        if telemetry:
            await telemetry.run_deferred(telemetry.event, "data_generation_failed", {
                "workflow_kind": "data_generation", "purpose": purpose,
                "execution_target": "cloud", "outcome": "failed",
                "requested_count": num_samples,
            }, workflow_id)
        return ToolResult.fail(
            "upstream",
            f"Error submitting cloud data-gen job: {type(e).__name__}: {e}",
        )

    # Durable finalization marker: the TUI watcher downloads the dataset
    # and notifies when it sees this file on a terminal job — including
    # after a TUI restart where the running→terminal transition was
    # never observed. Also carries the workflow id so the completion
    # telemetry closes the workflow opened above.
    # Provenance uses the PRE-SUBMISSION captures above: the job runs
    # the submitted pipeline against the submitted spec, regardless of
    # local edits made during the upload or while it executes.
    from lqh.project_identity import project_uuid as _project_uuid

    marker = {
        "workflow_id": workflow_id,
        "output_dataset": output_dataset,
        "purpose": purpose,
        "script_path": script_path,
        "pipeline_hash": pre_submit_pipeline_hash,
        "spec_sha256": pre_submit_spec_sha256,
        "job_id": job_id,
        # Owning project identity — a marker copied into another
        # project's tree is recognizably foreign.
        "owner_project_id": _project_uuid(project_dir),
        # Lets the finalizer refuse to clobber a dataset regenerated
        # locally AFTER this submit (older job finishing later).
        "submitted_at": time.time(),
    }
    marker_warning = ""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ".lqh_data_gen.json").write_text(json.dumps(marker, indent=2) + "\n")
    except OSError as e:
        # The job is running and remote_job.json is in place (submit_run
        # cancels on ITS persistence failures), so the watcher still
        # tracks it — only the auto-download after a TUI restart is
        # degraded. Report it rather than failing a live submission.
        marker_warning = (
            f"\n⚠️ Could not write the finalization marker ({e}); if the TUI "
            "restarts before the job completes, download the dataset via "
            "the artifacts tool."
        )

    if on_bg_started is not None:
        on_bg_started(run_name, "data_gen", run_name, "cloud")

    from lqh.project_log import append_event, file_hash_prefix

    append_event(
        project_dir,
        "data_gen_submitted",
        f"Submitted cloud data gen for {output_dataset} (job {job_id})",
        script_path=script_path,
        script_hash=file_hash_prefix(project_dir / script_path),
        output_dataset=output_dataset,
        num_samples=num_samples,
        job_id=job_id,
        run_name=run_name,
    )

    return ToolResult(
        content=(
            f"☁️ Cloud data-gen job started\n"
            f"  Run:     {run_name}\n"
            f"  Job ID:  {job_id}\n"
            f"  Samples: {num_samples}"
            + (f" × {samples_per_item} per item" if samples_per_item > 1 else "")
            + f"\n  Output:  datasets/{output_dataset}/ (downloads automatically "
            "when the job completes)\n\n"
            "The job runs in the background — never poll. You'll get a system "
            "notification once the dataset has been downloaded; only then does "
            f"datasets/{output_dataset}/ exist locally. In auto mode, if your "
            f"next step needs the dataset, call "
            f"training_status(run_name='{run_name}') — it parks until the job "
            "finishes and the dataset is downloaded."
            + marker_warning
            + _submit_advisories(backend)
        ),
        workflow_launched=True,
    )


async def _execute_pipeline(
    project_dir: Path,
    script_path: str,
    num_samples: int,
    output_dataset: str,
    validation_instructions: str | None,
    *,
    samples_per_item: int = 1,
    purpose: str = "unspecified",
    parent_dataset: str | None = None,
    on_pipeline_progress: Callable | None = None,
    on_pipeline_done: Callable | None = None,
    legacy_progress_callback: bool = True,
) -> ToolResult:
    """Actually execute the pipeline after permission is granted."""
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline
    from lqh.progress import ProgressReporter

    # Total work is num_samples × samples_per_item (iterate-N× mode).
    concurrency = min(100, max(1, num_samples * max(1, samples_per_item)))
    from lqh.telemetry import active_telemetry
    telemetry = active_telemetry()
    workflow_id = str(__import__("uuid").uuid4())
    started_mono = time.monotonic()
    if telemetry:
        _enabled, consent_epoch, started_active, _account_key = telemetry.state_snapshot()
    else:
        started_active, consent_epoch = 0.0, -1
    # This is only a cheap scheduling gate. Each queued telemetry mutation
    # re-validates durable consent on the ordered worker before writing, so a
    # slow in-flight flush cannot make a timed-out result suppress the entire
    # workflow's telemetry.
    telemetry_started = bool(
        telemetry and telemetry.cached_consent_active(consent_epoch)
    )
    if purpose not in {"smoke", "inspection", "validation", "training", "failures", "probe", "imported", "unspecified"}:
        purpose = "unspecified"
    if telemetry_started and telemetry:
        await telemetry.run_deferred(telemetry.record_generation_attempt)
        await telemetry.run_deferred(telemetry.event, "data_generation_started", {
            "workflow_kind":"data_generation", "purpose":purpose,
            "requested_count":num_samples, "execution_target":"local",
        }, workflow_id)

    reporter = ProgressReporter(
        task_kind="data_gen",
        label="Data generation",
        callback=on_pipeline_progress,
        legacy_callback=legacy_progress_callback,
    )
    reporter.update(
        phase="generation", phase_label="generating", completed=0,
        total=num_samples, unit="samples", overall_fraction=0,
        concurrency=concurrency, force=True,
    )

    try:
        config = load_config()
        token = require_token()
        client = create_client(token, config.api_base_url)

        target = _validate_path(project_dir, script_path)
        from lqh.data_gen_validation import pipeline_digest

        pre_run_pipeline_digest = pipeline_digest(target)
        # Provenance hashes are captured BEFORE the run: if SPEC.md or the
        # pipeline is edited while a long generation runs, the manifest
        # must attribute the artifact to the inputs it was built from.
        from lqh.project_log import file_hash_prefix as _hash_prefix
        from lqh.project_meta import compute_spec_sha256 as _spec_hash

        pre_run_spec_sha256 = _spec_hash(project_dir)
        pre_run_pipeline_hash = _hash_prefix(target, n=12)
        output_dir = project_dir / "datasets" / output_dataset

        val_text: str | None = None
        if validation_instructions:
            val_path = _validate_path(project_dir, validation_instructions)
            val_text = val_path.read_text(encoding="utf-8")

        def on_progress(done: int, total: int) -> None:
            # `done` is samples in hand, not attempts finished — it advances
            # only on success (see run_pipeline's on_progress contract).
            reporter.update(
                phase="generation", phase_label="generating",
                completed=done, total=total, unit="samples",
                overall_fraction=done / max(total, 1),
                concurrency=concurrency,
            )

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
        if result.succeeded <= 0:
            if telemetry_started and telemetry and telemetry.cached_consent_active(consent_epoch):
                await telemetry.run_deferred(telemetry.event, "data_generation_failed", {
                    "workflow_kind":"data_generation", "purpose":purpose,
                    "execution_target":"local", "outcome":"failed",
                    "wall_duration_ms":int((time.monotonic()-started_mono)*1000),
                    "active_duration_ms":int(max(telemetry.state_snapshot()[2]-started_active, 0)*1000),
                    "requested_count":num_samples, "succeeded_count":0,
                    "failed_count":result.failed, "sample_count":result.total,
                    "sample_failures":result.sample_failures,
                }, workflow_id)

            from lqh.project_log import append_event, file_hash_prefix

            append_event(
                project_dir,
                "data_gen_failed",
                f"Pipeline {script_path} produced no successful samples",
                script_path=script_path,
                script_hash=file_hash_prefix(project_dir / script_path),
                output_dataset=output_dataset,
                num_samples=num_samples,
                sample_failures=result.sample_failures,
                error="no successful samples",
            )
            reporter.update(
                phase="failed", phase_label="no samples generated",
                completed=result.total, total=result.total, unit="samples",
                overall_fraction=1.0, force=True,
            )
            return ToolResult.fail("runtime", (
                "❌ Pipeline failed: no samples were generated successfully\n"
                f"  Samples: 0/{result.total} succeeded, {result.failed} failed"
            ))

        if telemetry_started and telemetry and telemetry.cached_consent_active(consent_epoch):
            if result.succeeded > 0:
                await telemetry.run_deferred(telemetry.record_generation_succeeded, output_dir)
            await telemetry.run_deferred(telemetry.event, "data_generation_completed", {
                "workflow_kind":"data_generation", "purpose":purpose,
                "execution_target":"local", "outcome":"succeeded",
                "wall_duration_ms":int((time.monotonic()-started_mono)*1000),
                "active_duration_ms":int(max(telemetry.state_snapshot()[2]-started_active, 0)*1000),
                "requested_count":num_samples,"succeeded_count":result.succeeded,
                # failed_count is the shortfall (what the caller didn't get);
                # sample_failures is every permanent per-sample failure,
                # including ones an over-commit spare made up for. Only the
                # latter tracks pipeline reliability.
                "failed_count":result.failed,"sample_count":result.total,
                "sample_failures":result.sample_failures,
            }, workflow_id)

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
            sample_failures=result.sample_failures,
        )
        reporter.update(
            phase="completed", phase_label="dataset ready",
            # The rows produced, not the rows asked for, so a short run
            # reads "58/60" rather than claiming the full count. (The
            # fraction is pinned to 1.0 by result_ready — the *job* is
            # done; the counts are what carry the shortfall.)
            completed=result.succeeded, total=result.total, unit="samples",
            overall_fraction=1.0, result_ready=True, force=True,
        )

        # This dataset is now locally produced: drop the cloud-download
        # sidecar (if any) so a still-running cloud job targeting the
        # same name applies its "was this modified locally?" guard
        # against the fresh file instead of attributing it to an old
        # download and clobbering it.
        try:
            (output_dir / ".lqh_source.json").unlink(missing_ok=True)
        except OSError:
            pass

        # Finalization manifest: provenance for the summary tool, spec-
        # drift signals, and future sessions. Hashes were captured before
        # the run; a failed write is surfaced in the result, not hidden.
        from lqh.manifest import write_dataset_manifest

        manifest_written = write_dataset_manifest(
            project_dir,
            output_dir,
            purpose=purpose,
            rows=result.succeeded,
            pipeline_path=script_path,
            pipeline_hash=pre_run_pipeline_hash,
            spec_sha256=pre_run_spec_sha256,
            parent_dataset=parent_dataset,
            source_paths=[str(p) for p in (result.source_paths or [])],
            provenance_note=(
                f"resumed run — source recording covers only the final "
                f"process ({result.resumed_samples} samples carried over)"
                if result.resumed_samples > 0 else None
            ),
        ) is not None
        manifest_warning = (
            "" if manifest_written else
            "\n⚠️ Provenance manifest could not be written — this dataset is "
            "not traceable to its spec/pipeline revision (check disk/logs)."
        )

        # A successful local run validates this pipeline version for
        # cloud submission and records which lqh.sources inputs it read
        # (the cloud bundle manifest) — UNLESS the run resumed from a
        # partial file (its source recording covers only this process,
        # so the manifest would be incomplete).
        # Best-effort — never fail the run over gate bookkeeping.
        validation_note = ""
        if result.resumed_samples > 0:
            validation_note = (
                "\nℹ️ Not validated for cloud execution: this run resumed "
                f"{result.resumed_samples} samples from an interrupted run, so its "
                "input recording is incomplete. Run once uninterrupted to unlock "
                "execution='cloud'."
            )
        elif pipeline_digest(target) != pre_run_pipeline_digest:
            validation_note = (
                "\n⚠️ Not validated for cloud execution: the pipeline file changed "
                "while it was running. Run the current version once unchanged to "
                "unlock execution='cloud'."
            )
        else:
            try:
                from lqh.data_gen_validation import record_validation

                record_validation(
                    project_dir, target,
                    num_samples=num_samples,
                    succeeded=result.succeeded,
                    failed=result.failed,
                    source_paths=result.source_paths,
                    needs_hf=result.used_hf,
                )
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "failed to record data-gen validation", exc_info=True
                )

        cloud_tip = ""
        if num_samples >= 500:
            cloud_tip = (
                "\n\n💡 Runs this size can go to the cloud instead: "
                "execution='cloud' submits a background CPU job (fire-and-forget, "
                "dataset auto-downloads on completion)."
            )

        return ToolResult(
            content=(
                f"✅ Pipeline completed\n"
                f"  Samples: {result.succeeded}/{result.total} succeeded"
                + (f", {result.failed} failed" if result.failed else "")
                + f"\n  Output:  {result.output_path}"
                + manifest_warning
                + validation_note
                + cloud_tip
            ),
            ok=True,
        )
    except asyncio.CancelledError:
        if telemetry_started and telemetry and telemetry.cached_consent_active(consent_epoch):
            await telemetry.run_deferred(telemetry.event, "data_generation_failed", {
                "workflow_kind":"data_generation", "purpose":purpose,
                "execution_target":"local", "outcome":"cancelled",
                "wall_duration_ms":int((time.monotonic()-started_mono)*1000),
                "active_duration_ms":int(max(telemetry.state_snapshot()[2]-started_active, 0)*1000),
                "requested_count":num_samples,
            }, workflow_id)
        raise
    except Exception as e:
        import traceback

        if telemetry_started and telemetry and telemetry.cached_consent_active(consent_epoch):
            await telemetry.run_deferred(telemetry.event, "data_generation_failed", {
                "workflow_kind":"data_generation", "purpose":purpose,
                "execution_target":"local", "outcome":"failed",
                "wall_duration_ms":int((time.monotonic()-started_mono)*1000),
                "active_duration_ms":int(max(telemetry.state_snapshot()[2]-started_active, 0)*1000),
                "requested_count":num_samples,
            }, workflow_id)

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
        return ToolResult.fail("runtime", f"❌ Pipeline failed: {type(e).__name__}: {e}{hint}\n\n{tb}")
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


async def handle_compute_set(
    project_dir: Path,
    *,
    value: str | None = None,
    scope: str = "global",
    **kwargs: Any,
) -> ToolResult:
    """Persist the user's default compute target.

    Parameters
    ----------
    value : str | None
        ``"cloud"`` for LQH Cloud, ``"ssh:<name>"`` for a previously-bound
        SSH remote, ``"local"`` for in-process training on this machine
        (requires a local CUDA GPU), or empty string to clear. When omitted, the handler
        reports the current resolved compute target instead of writing
        anything — so an agent that calls ``compute_set`` with no args
        gets a useful answer instead of a TypeError.
    scope : str
        ``"global"`` writes ``~/.lqh/config.json`` (default — affects every
        project). ``"project"`` writes ``<project>/.lqh/compute.json``
        (overrides the global default for this project only).
    """
    from lqh.remote.compute import (
        load_global_default,
        load_project_default,
        resolve_compute,
        save_global_default,
        save_project_default,
    )

    # No value supplied → "show current". This is the friendly answer
    # for the model when it forgets the value arg (previously raised
    # TypeError, surfaced to the user as an opaque internal error).
    if value is None:
        resolved = resolve_compute(project_dir)
        proj = load_project_default(project_dir)
        glob = load_global_default()
        lines = [f"Current compute target: **{resolved}**"]
        lines.append(f"  • global default: {glob or '(unset → LQH Cloud)'}")
        lines.append(f"  • project default: {proj or '(unset)'}")
        lines.append(
            "Pass `value='cloud'` or `value='ssh:<name>'` to change it; "
            "`value=''` to clear."
        )
        return ToolResult(content="\n".join(lines))

    if scope not in ("global", "project"):
        return ToolResult.fail("validation", f"Error: scope must be 'global' or 'project', got {scope!r}")

    value = value.strip()
    if value == "":
        # Clear.
        if scope == "global":
            save_global_default(None)
        else:
            save_project_default(project_dir, None)
        return ToolResult(content=f"Cleared default compute ({scope}).")

    # Validate the shape — clearer to fail here than at /train time.
    if value not in ("cloud", "local") and not value.startswith("ssh:"):
        return ToolResult.fail("validation", (
            f"Error: value must be 'cloud', 'local', or 'ssh:<remote_name>', "
            f"got {value!r}."
        ))

    if scope == "global":
        save_global_default(value)
        return ToolResult(content=f"✅ Default compute set to '{value}' (global).")
    save_project_default(project_dir, value)
    return ToolResult(content=f"✅ Default compute set to '{value}' for this project.")


async def handle_show_file(
    project_dir: Path, *, path: str, message: str | None = None, **kwargs: Any,
) -> ToolResult:
    """Show a file to the user in scrollable view. Returns truncated version to agent."""
    target = _validate_path(project_dir, path)
    if not target.exists():
        return ToolResult(content=f"Error: file '{path}' does not exist")

    # Dataset files: open interactive dataset viewer via TUI callback
    if target.suffix.lower() in (".parquet", ".jsonl", ".json"):
        return ToolResult(
            content=f"[Opening interactive dataset viewer for {path}]",
            show_file_path=path,
            show_file_message=message,
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


def _render_message_content(content: Any, max_chars: int) -> str:
    """Render a ChatML message's content for tool output, truncated.

    String content is clamped to *max_chars*. Multimodal list content (VLM
    samples) renders text parts and replaces image parts with a size
    placeholder — a base64 data-URL must never be interpolated into agent
    context, where a single image would blow the advertised char cap by
    orders of magnitude. The final clamp applies to the joined text either
    way.
    """
    if isinstance(content, str):
        rendered = content
    elif isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                pieces.append(str(part))
                continue
            ptype = part.get("type")
            if ptype == "text":
                pieces.append(str(part.get("text") or ""))
            elif ptype == "image_url":
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                pieces.append(f"[image: {len(str(url or '')):,} chars]")
            else:
                pieces.append(f"[{ptype or 'part'}]")
        rendered = " ".join(p for p in pieces if p)
    elif content is None:
        rendered = ""
    else:
        rendered = str(content)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "..."
    return rendered


def _find_artifact_manifest(start_dir: Path) -> tuple[Path, Path] | None:
    """Locate the run's artifacts.json for *start_dir*.

    ``start_dir`` may be the run root or a nested eval dir inside it
    (``runs/<name>/checkpoints/final``) — the manifest always lives at the
    run root, so walk up a few levels. Returns ``(manifest_path, run_root)``
    or None.
    """
    d = start_dir
    for _ in range(4):
        manifest = d / "artifacts.json"
        if manifest.exists():
            return manifest, d
        if d.parent == d:
            break
        d = d.parent
    return None


async def _fetch_run_artifact(target: Path) -> bool:
    """Best-effort download of one published run file from the artifact
    store, keyed by the run's artifacts.json manifest (relpath relative to
    the run root, so nested files like checkpoints/final/results.parquet
    resolve too). Returns True when the file exists locally afterwards;
    never raises."""
    if target.exists():
        return True
    found = _find_artifact_manifest(target.parent)
    if found is None:
        return False
    manifest_path, run_root = found
    try:
        relpath = target.relative_to(run_root).as_posix()
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("artifacts") if isinstance(manifest, dict) else None
        artifact_id = next(
            (
                e.get("artifact_id")
                for e in (entries or [])
                if isinstance(e, dict)
                and e.get("relpath") == relpath
                and e.get("artifact_id")
            ),
            None,
        )
        if not artifact_id:
            return False
        from lqh.artifacts import BackendArtifactStore

        target.parent.mkdir(parents=True, exist_ok=True)
        await BackendArtifactStore().download(str(artifact_id), target)
    except Exception:
        return False
    return target.exists()


async def _fetch_results_parquet_artifact(run_dir: Path) -> bool:
    """Best-effort download of *run_dir*'s results.parquet (per-sample
    scores + judge reasoning) from the artifact store."""
    return await _fetch_run_artifact(run_dir / "results.parquet")


async def _hydrate_run_eval_artifacts(run_dir: Path) -> None:
    """Pull a completed cloud run's eval + training-metric outputs into the
    local mirror.

    Cloud sync records artifact descriptors, not contents — without this,
    a finished cloud sweep can have its results published while
    training_status still shows no final score. Covers both layouts:
    run-root (sweep eval-of-best, standalone evals) and checkpoints/final/
    (non-sweep SFT). Best-effort; never raises.
    """
    for rel in (
        "eval_result.json",
        "results.parquet",
        "checkpoints/final/eval_result.json",
        "checkpoints/final/results.parquet",
        # HF log_history: steps, loss trajectory, token accuracy. Published as
        # a "metrics" artifact but never pulled, so the training-health block
        # was empty for every cloud run — i.e. for every real run.
        "eval_history.json",
        # A sweep's per-config training output is in sweep_<id>/, so the
        # leaderboard has to come down first to learn which child won.
        "sweep_summary.json",
    ):
        await _fetch_run_artifact(run_dir / rel)

    # Second pass, sweeps only: the winner's own training metrics + config
    # (which carries the swept learning rate the base config omits).
    if not (run_dir / "eval_history.json").exists():
        if config_id := _sweep_winner_config_id(run_dir):
            for rel in (
                f"sweep_{config_id}/eval_history.json",
                f"sweep_{config_id}/config.json",
            ):
                await _fetch_run_artifact(run_dir / rel)


async def handle_get_eval_failures(
    project_dir: Path,
    *,
    eval_run: str,
    threshold: float = 6.0,
    min_failures: int = 5,
    max_failures: int = 15,
    score_min: float | None = None,
    score_max: float | None = None,
    sort: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    seed: int | None = None,
    sample_indices: list[int] | None = None,
    max_chars_per_message: int | None = None,
    export_path: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Inspect scored samples of an eval run.

    Default mode extracts failures (below-threshold + bottom-N padding);
    passing any browse parameter (score_min/score_max/sort/limit/offset/
    sample_indices) switches to browse mode — a filtered, paged slice of
    the scored samples. The legacy contract is byte-identical when no
    browse parameter is given.
    """
    run_dir = _validate_path(project_dir, eval_run)
    results_path = run_dir / "results.parquet"
    if not results_path.exists():
        # Cloud runs publish results.parquet as an artifact; sessions that
        # missed the completion download (or predate it) can still fetch
        # it on demand from the run's artifacts.json manifest.
        await _fetch_results_parquet_artifact(run_dir)
    if not results_path.exists():
        hint = ""
        if (run_dir / "remote_job.json").exists():
            hint = (
                " This is a cloud run — its per-sample results may not have "
                "been published (older lqh versions did not publish "
                "results.parquet). Check the `artifacts` tool for this run."
            )
        return ToolResult.fail(
            "not_found", f"Error: no results.parquet in '{eval_run}'.{hint}"
        )

    browse_mode = any(
        v is not None
        for v in (score_min, score_max, sort, limit, offset, sample_indices)
    )
    # Truncation limit applies in both modes; clamp to keep tool output sane.
    max_chars = max(50, min(int(max_chars_per_message or 500), 4000))

    total_matching = 0
    if browse_mode:
        from lqh.scoring import browse_results

        eff_limit = max(1, min(int(limit or 15), 25))
        eff_offset = max(0, int(offset or 0))
        eff_sort = sort if sort in ("asc", "desc", "random") else "asc"
        failures, scoring_errors, total_matching = browse_results(
            results_path,
            score_min=score_min,
            score_max=score_max,
            sort=eff_sort,
            limit=eff_limit,
            offset=eff_offset,
            seed=int(seed or 0),
            sample_indices=sample_indices,
        )
    else:
        from lqh.scoring import extract_failures

        failures, scoring_errors = extract_failures(
            results_path,
            threshold=threshold,
            min_failures=min_failures,
            max_failures=max_failures,
        )

    if browse_mode and not failures:
        note = ""
        if scoring_errors:
            note = (
                f"\nℹ️ {len(scoring_errors)} sample(s) hit judge/scoring errors "
                "and are excluded — call without browse filters to list them."
            )
        return ToolResult(content="No samples match the given filters." + note)
    if not failures and not scoring_errors:
        return ToolResult(content="No failure cases found. All samples scored above threshold.")

    export_note = ""
    if export_path:
        # Durable, untruncated export (the feedback/ workflow): full
        # messages plus the origin of each case — which eval run and
        # which model produced it. Confined to feedback/ and never
        # overwrites: an errant path must not be able to replace
        # SPEC.md, NOTES.md, or an artifact.
        normalized = export_path.replace("\\", "/")
        if not normalized.startswith("feedback/") or ".." in normalized.split("/"):
            return ToolResult.fail("validation", (
                f"Error: export_path must live under feedback/ "
                f"(got {export_path!r}) — e.g. 'feedback/eval_failures_v1.jsonl'."
            ))
        export_abs = _validate_path(project_dir, export_path)
        if export_abs.exists():
            return ToolResult.fail("conflict", (
                f"Error: {export_path} already exists — exports never "
                "overwrite. Pick a new file name."
            ))
        model_origin: dict[str, Any] = {}
        try:
            run_config = json.loads(
                (run_dir / "config.json").read_text(encoding="utf-8")
            )
            model_origin = {
                k: run_config[k]
                for k in (
                    "hf_repo", "revision", "base_model",
                    "inference_model", "model_path", "type",
                )
                if run_config.get(k) is not None
            }
        except Exception:
            pass
        # Legacy exports carry the threshold; browse exports carry the
        # filter that produced the selection instead.
        if browse_mode:
            selection_meta: dict[str, Any] = {"filter": {
                k: v for k, v in (
                    ("score_min", score_min),
                    ("score_max", score_max),
                    ("sort", sort),
                    ("limit", limit),
                    ("offset", offset),
                    ("seed", seed),
                    ("sample_indices", sample_indices),
                ) if v is not None
            }}
        else:
            selection_meta = {"threshold": threshold}
        try:
            export_abs.parent.mkdir(parents=True, exist_ok=True)
            from datetime import datetime as _dt, timezone as _tz

            exported_at = _dt.now(_tz.utc).isoformat(timespec="seconds")
            with open(export_abs, "w", encoding="utf-8") as f:
                for failure in failures:
                    f.write(json.dumps({
                        "sample_index": failure["sample_index"],
                        "score": failure["score"],
                        "reasoning": failure["reasoning"],
                        "messages": failure["messages"],
                        "eval_run": eval_run,
                        "model": model_origin,
                        **selection_meta,
                        "exported_at": exported_at,
                        "scoring_error": False,
                    }, ensure_ascii=False) + "\n")
                # Scoring errors are part of the legacy failure export only;
                # a browse export is exactly the browsed selection.
                if not browse_mode:
                    for err in scoring_errors:
                        f.write(json.dumps({
                            "sample_index": err["sample_index"],
                            "score": None,
                            "reasoning": err["reasoning"],
                            "messages": err.get("messages"),
                            "eval_run": eval_run,
                            "model": model_origin,
                            "exported_at": exported_at,
                            "scoring_error": True,
                        }, ensure_ascii=False) + "\n")
            exported_errors = 0 if browse_mode else len(scoring_errors)
            export_note = (
                f"\n💾 Exported {len(failures)} sample(s)"
                + (f" + {exported_errors} scoring error(s)" if exported_errors else "")
                + f" (full, untruncated) to {export_path}"
            )
        except OSError as exc:
            export_note = f"\n⚠️ Export to {export_path} failed: {exc}"

    import pyarrow.parquet as pq_mod

    total = pq_mod.read_metadata(results_path).num_rows

    parts: list[str] = []

    if browse_mode:
        if sample_indices is not None:
            parts.append(
                f"## Samples ({len(failures)} of {len(sample_indices)} "
                f"requested indices found)\n"
            )
        else:
            filter_bits: list[str] = []
            if score_min is not None and score_max is not None:
                filter_bits.append(f"score in [{score_min:g}, {score_max:g}]")
            elif score_min is not None:
                filter_bits.append(f"score ≥ {score_min:g}")
            elif score_max is not None:
                filter_bits.append(f"score ≤ {score_max:g}")
            else:
                filter_bits.append("all scores")
            filter_bits.append(f"sort={eff_sort}")
            first = eff_offset + 1
            last = eff_offset + len(failures)
            parts.append(
                f"## Samples {first}–{last} of {total_matching} matching "
                f"({', '.join(filter_bits)})\n"
            )
    elif failures:
        parts.append(
            f"## Failure Cases ({len(failures)} of {total} samples, threshold < {threshold})\n"
        )
    else:
        parts.append(
            f"## Failure Cases (0 of {total} samples below threshold {threshold})\n"
        )

    for f in failures:
        parts.append(f"### Sample {f['sample_index']} — Score: {f['score']:.1f}/10")
        parts.append(f"**Judge reasoning:** {f['reasoning']}")
        for msg in f["messages"]:
            role = msg.get("role", "?")
            rendered = _render_message_content(msg.get("content", ""), max_chars)
            parts.append(f"**{role}:** {rendered}")
        parts.append("")

    if browse_mode:
        if sample_indices is None and eff_offset + len(failures) < total_matching:
            remaining = total_matching - (eff_offset + len(failures))
            parts.append(
                f"[{remaining} more matching — use "
                f"offset={eff_offset + len(failures)} to continue]"
            )
        if scoring_errors:
            parts.append(
                f"ℹ️ {len(scoring_errors)} sample(s) hit judge/scoring errors and "
                "are excluded here — call without browse filters to list them."
            )
    elif scoring_errors:
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

    content = "\n".join(parts) + export_note
    content, _ = _truncate_content(content)
    return ToolResult(content=content)


async def handle_list_models(**kwargs: Any) -> ToolResult:
    """List the Liquid model catalog plus the baseline/judge pool models.

    The Liquid catalog is a local constant (lqh.models) — the old
    router.liquid.ai listing API has been retired (see MODELS.md). Liquid
    checkpoints are evaluated via the HuggingFace inference path
    (eval_hf_model / start_local_eval), not via run_scoring mode='model_eval'.
    """
    from lqh.models import format_catalog

    lines = [format_catalog()]
    lines.append("")
    lines.append("To evaluate a Liquid checkpoint, use the HuggingFace inference path:")
    lines.append("  eval_hf_model     — cloud eval of a HuggingFace repo id / revision")
    lines.append("                      (for a catalog model above, pass training_method='full';")
    lines.append("                      'lora' is only for adapter repos and needs base_model)")
    lines.append("  start_local_eval  — local or SSH-remote GPU eval of a checkpoint dir")
    lines.append("")
    lines.append("These pool/utility models are baselines/judges served by the API and")
    lines.append("can be used as inference_model in run_scoring mode='model_eval':")
    lines.append("  small, medium, large          — default model from each size pool")
    lines.append("  random:<size>                  — random model from pool (different each request)")
    lines.append("  random:<size>:<seed>           — deterministic model from pool")
    lines.append("  judge:small, judge:medium, judge:large — dedicated scoring models")
    lines.append("  orchestration                  — agent-grade model with tool calling")
    lines.append("")
    lines.append("Pool names are not model identities: the platform maps each pool to")
    lines.append("a concrete model based on the task, cost, complexity and other")
    lines.append("factors, and which model that is is not exposed and can change.")
    lines.append("Only the Liquid catalog above names specific models.")

    return ToolResult(content="\n".join(lines))


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
        return ToolResult.fail("not_found", f"Error: no data.parquet in '{dataset}'")

    scorer_path = _validate_path(project_dir, scorer)
    if not scorer_path.exists():
        return ToolResult.fail("not_found", f"Error: scorer '{scorer}' does not exist")

    # Resolve system_prompt_path -> inference_system_prompt if needed
    if system_prompt_path and not inference_system_prompt:
        prompt_file = _validate_path(project_dir, system_prompt_path)
        if not prompt_file.exists():
            return ToolResult.fail("not_found", f"Error: prompt file '{system_prompt_path}' does not exist")
        inference_system_prompt = prompt_file.read_text(encoding="utf-8")

    # Auto-discover response_format schema from prompt path
    # e.g., prompts/translation_v0.md → prompts/translation.schema.json
    inference_response_format = None
    response_format_path = kwargs.get("response_format_path")
    if response_format_path:
        schema_file = _validate_path(project_dir, response_format_path)
        if not schema_file.exists():
            return ToolResult.fail("not_found", f"Error: schema file '{response_format_path}' does not exist")
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
        # max_retries=0: scoring owns its own retry ladder and bounds each
        # sample with a deadline. The SDK's invisible replay layer would sit
        # underneath both and multiply every timeout by three.
        client = create_client(token, config.api_base_url, max_retries=0)
    except Exception as e:
        return ToolResult.fail("auth", f"Error: {e}")

    from lqh.progress import ProgressReporter

    on_progress = kwargs.get("on_pipeline_progress")
    progress_kind = "zero_shot_eval" if mode == "model_eval" else "evaluation"
    progress_label = "Zero-shot evaluation" if mode == "model_eval" else "Data scoring"
    reporter = ProgressReporter(
        task_kind=progress_kind,
        label=progress_label,
        callback=on_progress,
        legacy_callback=bool(kwargs.get("legacy_progress_callback", True)),
    )
    reporter.update(
        phase="setup", phase_label="preparing evaluation",
        overall_fraction=0, unit="samples", force=True,
    )

    def _progress(completed: int, total: int) -> None:
        reporter.update(
            phase="evaluation", phase_label="evaluating",
            completed=completed, total=total, unit="samples",
            overall_fraction=completed / max(total, 1),
            concurrency=min(100, total), force=completed == total,
        )

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

            # Record the scoring pass on the dataset's manifest (no-op if
            # the dataset has none — annotation never invents provenance).
            from lqh.manifest import annotate_manifest

            annotate_manifest(
                dataset_dir,
                scored_by=scorer,
                scored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                score_mean=round(result.mean_score, 2),
            )
            reporter.update(
                phase="completed",
                phase_label=(
                    "scores ready" if result.scored > 0 else "no valid scores"
                ),
                completed=result.total, total=result.total, unit="samples",
                overall_fraction=1.0,
                result_ready=result.scored > 0, force=True,
            )

            distribution = _format_score_distribution(data_path.parent / "scores.parquet")
            from lqh.scoring import failure_warning

            warning = failure_warning(result.failed, result.total)
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
                    + (f"\n{warning}" if warning else "")
                    + (f"\n{distribution}" if distribution else "")
                    + f"\n  Output: {dataset}/scores.parquet"
                )
            )

        elif mode == "model_eval":
            if not run_name:
                return ToolResult.fail("validation", "Error: run_name is required for mode='model_eval'")
            if not inference_model:
                return ToolResult.fail(
                    "validation",
                    "Error: inference_model is required for mode='model_eval'. "
                    "Use list_models to discover available models.",
                )

            # Liquid checkpoints can no longer be evaluated through the API:
            # the router.liquid.ai inference API has been retired (see MODELS.md).
            # Redirect to the HuggingFace inference path. Pool/baseline names
            # (small/medium/large/orchestration) still run via the API here.
            from lqh.models import is_liquid_model_name

            if is_liquid_model_name(inference_model):
                return ToolResult.fail(
                    "validation",
                    (
                        f"Error: Liquid model '{inference_model}' cannot be evaluated via "
                        "run_scoring mode='model_eval' — the router.liquid.ai API has been "
                        "retired (see MODELS.md). To evaluate a Liquid checkpoint, use the "
                        "HuggingFace inference path instead:\n"
                        "  - eval_hf_model  — cloud eval of a HuggingFace repo id / revision\n"
                        "  - start_local_eval — local or SSH-remote GPU eval of a checkpoint dir\n"
                        "run_scoring mode='model_eval' remains available for the pool "
                        "baselines (small / medium / large / orchestration)."
                    ),
                )

            from lqh.scoring import JUDGE_MODELS, run_scoring

            output_dir = project_dir / "evals" / "runs" / run_name
            try:
                # Atomic claim — a bare exists() check races a second CLI.
                output_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                # An interrupted run leaves the directory it claimed, holding
                # partial artifacts and no real ones. "Use a different name"
                # would strand scores the user already paid for, so say what
                # is actually in there.
                from lqh.scoring import PARTIAL_SUFFIX

                interrupted = (
                    not (output_dir / "summary.json").exists()
                    and (output_dir / f"summary{PARTIAL_SUFFIX}.json").exists()
                )
                if interrupted:
                    return ToolResult.fail(
                        "conflict",
                        f"Error: eval run '{run_name}' was interrupted and left "
                        f"partial results.\n"
                        f"  evals/runs/{run_name}/summary{PARTIAL_SUFFIX}.json "
                        "holds the samples that were scored before it stopped — "
                        "read it before discarding them.\n"
                        f"  To redo the run: delete evals/runs/{run_name}/ or "
                        "pick a different run_name.",
                    )
                return ToolResult.fail(
                    "conflict",
                    f"Error: eval run '{run_name}' already exists. Use a different name.",
                )
            except OSError as exc:
                return ToolResult.fail(
                    "runtime",
                    f"Error: cannot create evals/runs/{run_name}: {exc}",
                )

            # Spec provenance is captured BEFORE the eval runs: a spec
            # edit during a long evaluation must not attribute the
            # output to the completion-time spec it never ran under.
            from lqh.project_meta import compute_spec_sha256 as _spec_hash2

            pre_eval_spec_sha256 = _spec_hash2(project_dir)
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
                "spec_sha256": pre_eval_spec_sha256,
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

            # Finalization manifest for the eval run (reads the config.json
            # and summary.json just written above). A write failure is
            # surfaced in the result — silently missing provenance would
            # read as "this eval predates manifests".
            from lqh.manifest import write_run_manifest

            _eval_manifest = write_run_manifest(
                project_dir, output_dir, state="completed"
            )
            _eval_manifest_warn = (
                "\n⚠️ Provenance manifest could not be written for this eval "
                "run (check disk/logs)."
                if _eval_manifest is None else ""
            )
            reporter.update(
                phase="completed",
                phase_label=(
                    "evaluation ready" if result.scored > 0 else "evaluation failed"
                ),
                completed=result.total, total=result.total, unit="samples",
                overall_fraction=1.0,
                result_ready=result.scored > 0, force=True,
            )

            distribution = _format_score_distribution(output_dir / "results.parquet")
            from lqh.scoring import failure_warning

            warning = failure_warning(result.failed, result.total)
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
                    + (f"\n{warning}" if warning else "")
                    + (f"\n{distribution}" if distribution else "")
                    + f"\n  Results: evals/runs/{run_name}/"
                    + _eval_manifest_warn
                )
            )
        else:
            return ToolResult.fail("validation", f"Error: unknown mode '{mode}'. Use 'data_quality' or 'model_eval'.")

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
        return ToolResult.fail("runtime", f"❌ Scoring failed: {type(e).__name__}: {e}\n\n{tb}")
    finally:
        on_done = kwargs.get("on_pipeline_done")
        if on_done:
            on_done()


# ---------------------------------------------------------------------------
# Hugging Face Hub helpers
# ---------------------------------------------------------------------------

HF_MAPPINGS_FILE = ".lqh/hf.json"

# Options for the HF-donation consent prompt. Matched by substring in the
# agent loop's grant site, so keep the distinguishing words ("don't ask
# again", "without") stable.
#
# Yes and no are deliberately symmetric: both have a durable form. Without
# the durable no, a user who declines gets asked again at every cloud
# submit — five sites, so a single pipeline re-asks several times — and
# the only way out was LQH_HF_DONATE=0, which means leaving the session to
# set an env var.
HF_DONATE_OPTIONAL_OPTIONS = [
    "Yes — send my HF token with this job",
    "Yes, and don't ask again for this project",
    "No — run this job without it",
    "No, and don't ask again for this project",
]

# Same four answers, but the decline options cannot claim the job still
# runs: a transfer or a GGUF push with no token anywhere is rejected by
# the backend.
HF_DONATE_REQUIRED_OPTIONS = [
    "Yes — send my HF token with this job",
    "Yes, and don't ask again for this project",
    "No — don't send it (needs a token stored via /hf_login)",
    "No, and don't ask again for this project",
]

# Back-compat alias; the optional wording is the common case.
HF_DONATE_OPTIONS = HF_DONATE_OPTIONAL_OPTIONS


def _is_cloud_target(remote: str) -> bool:
    """Whether a resolved compute target is LQH Cloud."""
    from lqh.remote.compute import is_cloud

    return is_cloud(remote)


def _eval_hf_disclosure(project_dir: Path) -> str:
    """HF disclosure line for the cloud-eval consent prompt ("" when none)."""
    from lqh.hf_token import hf_disclosure_line

    return hf_disclosure_line(project_dir, indent="  ")


def _submit_advisories(backend: Any) -> str:
    """Render a cloud backend's post-submit advisories, or "".

    submit_run returns only a job id (the RemoteBackend contract), so
    non-fatal notes — a credential-like file left out of the upload, a
    donated token the backend could not make restart-safe — ride on the
    backend object. Dropping them is how a user ends up debugging a paid
    sandbox failure with no idea a file was excluded.
    """
    warnings = getattr(backend, "last_submit_warnings", None) or []
    if not warnings:
        return ""
    return "\n" + "\n".join(f"  \u26a0\ufe0f {w}" for w in warnings)


def _resolve_hf_donation(
    project_dir: Path,
    permissions: PermissionContext | None,
    hf_donate: bool | None,
    job_label: str,
    *,
    token_required: bool = False,
) -> tuple[bool, ToolResult | None]:
    """Decide whether to send a locally-found HF token with a cloud job.

    Returns ``(donate, prompt)``. A non-None ``prompt`` means the caller
    must return it unchanged — the user has to answer first.

    Note what this gate does NOT do: block the job. Declining donation
    submits the job without a token, which is why the decline answer
    travels back as ``_hf_donate=False`` rather than as an absent grant
    (an absent grant is indistinguishable from "not asked yet" and would
    re-prompt forever).

    ``token_required`` changes only the copy, and it has to. For most
    jobs a decline is genuinely free — the job runs, it just can't reach
    gated repos. But an HF *upload* (a transfer, or a GGUF conversion
    with a push target) cannot run at all without a credential: the
    backend rejects it outright when neither an inline nor an account
    token exists. Telling those users "declining still runs the job" and
    then handing them a 400 is a promise the workflow can't keep, so
    they get told what declining actually costs.

    The prompt only fires when a token was actually found, so users
    without one never see it — and normally it does not fire at all,
    because the question is put up front (interactive startup, see
    ``TuiApp._settle_hf_donation``) and the standing answer is what this
    function reads. What is left here is the fallback for the cases the
    up-front question cannot cover: a token that appeared mid-session, a
    startup prompt the user quit out of, and the headless surfaces.

    Provenance only in this function — no plaintext. See lqh.hf_token.
    """
    if hf_donate is False:
        return False, None

    from lqh.hf_token import DONATE_ALWAYS, hf_token_origin, resolve_hf_donate_decision

    origin = hf_token_origin(project_dir)
    if origin is None or not origin.donation_enabled:
        # Nothing to donate, or LQH_HF_DONATE=0. Either way: no prompt,
        # no donation, job proceeds.
        return False, None

    # Default rather than skip when no invocation context was supplied:
    # a tool's first call normally carries no _permissions at all.
    perms = permissions or PermissionContext()

    # An invocation-scoped grant is checked FIRST, ahead of anything on
    # disk. `--allow-hf-donate` (or the agent loop re-invoking after a
    # "yes") is the user saying yes about this run, right now; a stored
    # "no" from some earlier session must not veto it. The env opt-out
    # still wins over both — it was checked above.
    if "hf_donate" in perms.grants:
        return True, None

    # A standing answer settles it silently, in both directions. Asking
    # again mid-pipeline is the complaint this path exists to remove: a
    # credential question arriving between two stages of a run the user
    # is watching reads as "something needs my token *now*", when in fact
    # nothing here does. Both durable per-project answers written by the
    # "don't ask again" options are read here, so answering either one
    # mid-pipeline silences every later submit in the same run.
    decision = resolve_hf_donate_decision(project_dir)
    if decision is not None:
        return decision == DONATE_ALWAYS, None

    where = origin.label
    if origin.path:
        where += f" ({origin.path})"
    # Both "don't ask again" answers are written to disk, and there is no
    # in-product toggle to undo them — so the prompt has to name the file,
    # the way the startup question does. Being told a decision is durable
    # without being told where it lives is how a user ends up unable to
    # take it back.
    durable_note = (
        f"Either \"don't ask again\" answer is recorded in "
        f"{PERMISSIONS_FILE} — change it there any time, or set "
        f"LQH_HF_DONATE=0 to stop offering this on every project."
    )
    if token_required:
        # This workflow uploads to HF. Without a token there is nothing
        # to run, so "declining still runs the job" would be a lie.
        decline_note = (
            "This job UPLOADS to Hugging Face, so it needs a token. Declining "
            "works only if you already stored one with /hf_login — otherwise "
            "the job is rejected and nothing is charged. \"No, and don't ask "
            "again\" also makes that the standing answer for this project — "
            "later upload jobs then behave the same way, without asking. "
            + durable_note
        )
    else:
        decline_note = (
            "Declining still runs the job — without this token (any token you "
            "stored with /hf_login is unaffected). \"No, and don't ask again\" "
            "also makes that the standing answer for this project, so no later "
            "job in this run asks again. "
            + durable_note
        )
    extra = ""
    if origin.is_hub_cache:
        # This token was created for the Hub CLI, not for us. Say so.
        extra = (
            "\nThis is the token `huggingface-cli login` saved on this machine — "
            "you didn't set it up for LQH specifically."
        )
    return False, ToolResult(
        content="PERMISSION_REQUIRED",
        requires_user_input=True,
        permission_key=f"hf_donate:{job_label}",
        question=(
            f"Send your Hugging Face token with this {job_label} job?\n"
            f"  Source:  {where}\n"
            f"  Used to: read gated/private models and datasets inside the sandbox\n"
            # Precise about retention: the backend holds it encrypted,
            # scoped to this job, so a replacement worker after a restart
            # still has it — and deletes it when the job ends. Saying
            # "not stored" would be false.
            f"  Kept:    encrypted for this job only, then deleted — it is not "
            f"added to your LQH account{extra}\n"
            # Precedence matters and is easy to get wrong from the outside:
            # a donated token REPLACES the account one for this job, so a
            # user with both could unknowingly run under a different (or
            # weaker) credential than they expect.
            f"  Note:    if you have a token stored via /hf_login, this one "
            f"takes precedence for this job\n\n"
            # Not "the job won't have HF access": an account token stored
            # via /hf_login still applies, and claiming otherwise would
            # push people into donating unnecessarily.
            + decline_note
        ),
        options=HF_DONATE_OPTIONAL_OPTIONS if not token_required else HF_DONATE_REQUIRED_OPTIONS,
    )


def _get_hf_token(project_dir: Path | None = None) -> str:
    """Return a local HF token or raise with instructions.

    Shares :mod:`lqh.hf_token` with the cloud donate path so a token in a
    project ``.env`` (or from ``huggingface-cli login``) works the same for
    local tools as it does for a cloud submit — otherwise a user would find
    that cloud fine-tuning works while ``push`` insists the env var is unset.
    The token stays on this machine here; nothing is sent to LQH.
    """
    from lqh.hf_token import local_hf_token

    token = local_hf_token(project_dir)
    if not token:
        raise ValueError(
            "No Hugging Face credentials on this machine. This tool talks to "
            "the Hub from here, so it needs a local token — looked at: HF_TOKEN "
            "and HUGGING_FACE_HUB_TOKEN in the environment, .env.local/.env in "
            "the project, and ~/.cache/huggingface/token (huggingface-cli "
            "login).\n"
            "Get one at https://huggingface.co/settings/tokens, then either "
            "export HF_TOKEN=... or add HF_TOKEN=... to the project .env.\n"
            "Note: a token stored with /hf_login lives on the LQH backend and is "
            "used by CLOUD jobs only — it is deliberately never sent back to "
            "this machine, so it cannot serve this call."
        )
    return token


def _get_hf_api(project_dir: Path | None = None):
    """Create an authenticated HfApi instance."""
    from huggingface_hub import HfApi

    token = _get_hf_token(project_dir)
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
        api = _get_hf_api(kwargs.get("project_dir"))
    except ValueError as e:
        return ToolResult.fail("auth", f"Error: {e}")

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
        return ToolResult.fail("upstream", f"Error: {e}")


# ----------------------------------------------------------------------
# Unified pull / push over the location URI grammar (hf: / lqh: / local).
# Thin wrappers over the HF handlers and the artifact store; the scheme
# is always explicit (see lqh.tools.uri).
# ----------------------------------------------------------------------


async def handle_pull(
    project_dir: Path, *, source: str, dest: str | None = None, **kwargs: Any,
) -> ToolResult:
    """Download from hf: or lqh: into local storage."""
    from lqh.tools.uri import parse_location, LocationError

    try:
        loc = parse_location(source)
    except LocationError as e:
        return ToolResult.fail("validation", f"Error: {e}")

    if loc.scheme == "hf":
        return await handle_hf_pull(
            project_dir, repo_id=loc.value, local_path=dest, revision=loc.revision,
        )
    if loc.scheme == "lqh":
        return await _pull_lqh_artifact(project_dir, loc.value, dest)
    return ToolResult.fail(
        "validation",
        (
            f"Error: pull source must be 'hf:owner/repo' or 'lqh:<artifact_id>'; "
            f"got a local path {source!r}. Local files are already on disk — use "
            "read_file / list_files instead."
        ),
    )


async def _pull_lqh_artifact(project_dir: Path, artifact_id: str, dest: str | None) -> ToolResult:
    from lqh.artifacts import ArtifactError, BackendArtifactStore

    rel = dest or f"artifacts/{artifact_id}"
    try:
        target = _validate_path(project_dir, rel)
    except ValueError as e:
        return ToolResult.fail("validation", f"Error: {e}")

    store = BackendArtifactStore()
    try:
        await store.download(artifact_id, target)
    except ArtifactError as e:
        return ToolResult.fail("upstream", f"Error downloading lqh:{artifact_id}: {e}")
    except Exception as e:  # noqa: BLE001 - surface any client error to the agent
        return ToolResult.fail("upstream", f"Error downloading lqh:{artifact_id}: {e}")

    size = target.stat().st_size if target.exists() else 0
    return ToolResult(
        content=(
            f"✅ Downloaded lqh:{artifact_id} -> {rel} ({size:,} bytes). "
            "Checkpoints arrive as a .tar.gz; extract before use."
        ),
    )


async def handle_push(
    project_dir: Path, *, source: str, dest: str, private: bool = True, **kwargs: Any,
) -> ToolResult:
    """Push a local path or an lqh: artifact to a Hugging Face repo.

    A local source uploads directly. An lqh: source (an R2 artifact) is
    transferred to HF by a short CPU-only cloud sandbox — bytes never
    round-trip through this laptop.
    """
    from lqh.tools.uri import parse_location, LocationError

    try:
        src = parse_location(source)
        dst = parse_location(dest)
    except LocationError as e:
        return ToolResult.fail("validation", f"Error: {e}")

    if dst.scheme != "hf":
        return ToolResult.fail(
            "validation",
            f"Error: push destination must be 'hf:owner/repo'; got {dest!r}",
        )

    if src.scheme == "local":
        return await handle_hf_push(
            project_dir, local_path=src.value, repo_id=dst.value, private=private,
            **{k: kwargs[k] for k in ("_permissions",) if k in kwargs},
        )
    if src.scheme == "lqh":
        return await _push_lqh_to_hf(
            project_dir, src.value, dst.value, private,
            permissions=kwargs.get("_permissions"),
            hf_donate=kwargs.get("_hf_donate"),
        )
    return ToolResult.fail(
        "validation",
        (
            f"Error: push source must be a local path or 'lqh:<artifact_id>'; "
            f"got {source!r}"
        ),
    )


async def _push_lqh_to_hf(
    project_dir: Path,
    artifact_id: str,
    target_repo: str,
    private: bool,
    *,
    permissions: PermissionContext | None = None,
    hf_donate: bool | None = None,
) -> ToolResult:
    """Submit a CPU-only transfer job that copies an R2 artifact to HF.

    Gated on the same hf_push domain as the local-file push path. This
    writes to the user's HuggingFace account, so "the bytes start in R2
    rather than on disk" is not a reason to skip consent — and donating a
    token sharpens it, since the agent could otherwise reach any repo the
    token can.
    """
    from lqh.remote.transfer import submit_transfer

    perms = permissions or PermissionContext()
    if not perms.allows_hf_push(project_dir, target_repo):
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            permission_key=f"hf_push:{target_repo}",
            question=(
                f"The agent wants to upload a checkpoint to your HuggingFace account:\n"
                f"  Artifact: lqh:{artifact_id}\n"
                f"  Repo:     {target_repo} ({'private' if private else 'PUBLIC'})\n\n"
                f"Allow the upload?"
            ),
            options=[
                "Upload",
                "Upload and don't ask again for this repo",
                "Do not upload",
            ],
        )

    donate_hf, hf_prompt = _resolve_hf_donation(
        project_dir, perms, hf_donate, "transfer", token_required=True
    )
    if hf_prompt is not None:
        return hf_prompt

    try:
        job_id = await submit_transfer(
            project_id=_ckey(project_dir),
            source_artifact_id=artifact_id,
            target_hf_repo=target_repo,
            private=private,
            project_dir=project_dir,
            donate_hf_token=donate_hf,
        )
    except Exception as e:  # noqa: BLE001 - surface clearly to the agent
        return ToolResult.fail("upstream", f"Error starting transfer of lqh:{artifact_id}: {e}")
    return ToolResult(
        content=(
            f"🚚 Transferring lqh:{artifact_id} → hf:{target_repo} via a CPU sandbox "
            f"(job {job_id}). The checkpoint is uploaded from cloud storage directly; check "
            "training_status or the artifact's hf_repo once it completes."
        ),
    )


async def handle_gguf_convert(
    project_dir: Path,
    *,
    artifact_id: str,
    quant_types: list[str],
    target_hf_repo: str | None = None,
    private: bool = True,
    include_f16: bool = False,
    base_model: str | None = None,
    artifact_format: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Submit a CPU-only cloud job that converts an LQH checkpoint to GGUF
    and quantizes it into the requested types."""
    from lqh.remote.gguf_convert import submit_gguf

    if not quant_types:
        return ToolResult.fail("validation", "Error: quant_types must list at least one type.")

    perms = kwargs.get("_permissions") or PermissionContext()
    # A push writes to the user's HF account — same gate as any other
    # upload. Conversion without a push target touches nothing of theirs.
    if target_hf_repo and not perms.allows_hf_push(project_dir, target_hf_repo):
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            permission_key=f"hf_push:{target_hf_repo}",
            question=(
                f"The agent wants to upload GGUF files to your HuggingFace account:\n"
                f"  Artifact: lqh:{artifact_id}\n"
                f"  Quants:   {', '.join(quant_types)}\n"
                f"  Repo:     {target_hf_repo} ({'private' if private else 'PUBLIC'})\n\n"
                f"Allow the upload?"
            ),
            options=[
                "Upload",
                "Upload and don't ask again for this repo",
                "Do not upload",
            ],
        )

    # Offer when the sandbox can actually use the token, which mirrors
    # the backend's two conditions: a push target, or a LoRA merge that
    # downloads a possibly-gated base.
    #
    # `artifact_format="full"` is the one case the CLI can rule out
    # locally — the backend treats it as authoritative (it overrides
    # lineage and the filename convention), so a full checkpoint with no
    # push target neither downloads nor uploads anything and the token
    # has no purpose. Every other combination stays permissive: a LoRA's
    # base is normally resolved from server-side lineage, so the CLI
    # commonly sees neither `base_model` nor a format and must not skip
    # the token in exactly that case — that would hand the user a paid
    # job that 401s.
    #
    # This matters most AFTER a "don't ask again" grant, where an
    # unnecessary donation is silent.
    donate_hf = False
    if target_hf_repo or artifact_format != "full":
        donate_hf, hf_prompt = _resolve_hf_donation(
            project_dir, perms, kwargs.get("_hf_donate"), "gguf",
            # Only a push is unconditional: converting a public base
            # needs no credential, so the token is genuinely optional
            # there and the decline copy must not claim otherwise.
            token_required=bool(target_hf_repo),
        )
        if hf_prompt is not None:
            return hf_prompt

    try:
        job_id = await submit_gguf(
            project_id=_ckey(project_dir),
            source_artifact_id=artifact_id,
            quant_types=quant_types,
            target_hf_repo=target_hf_repo,
            private=private,
            include_f16=include_f16,
            base_model=base_model,
            artifact_format=artifact_format,
            project_dir=project_dir,
            donate_hf_token=donate_hf,
        )
    except Exception as e:  # noqa: BLE001 - surface clearly to the agent
        return ToolResult.fail("upstream", f"Error starting gguf conversion of lqh:{artifact_id}: {e}")

    quants = ", ".join(quant_types)
    push = f" and pushing to hf:{target_hf_repo}" if target_hf_repo else ""
    return ToolResult(
        content=(
            f"🧱 Converting lqh:{artifact_id} → GGUF ({quants}){push} via a CPU sandbox "
            f"(job {job_id}). Each quant is converted from cloud storage directly and smoke-tested; "
            "the produced .gguf files register as new artifacts (kind 'gguf'). Check "
            "training_status for progress, then 'artifacts' (action=list) to download them."
            + (" The HF push uses whichever token applied to this job — the one "
               "you approved sending, or the one stored on your account."
               if target_hf_repo else "")
        ),
    )


async def handle_artifacts(
    project_dir: Path,
    *,
    action: str = "list",
    artifact_id: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    **kwargs: Any,
) -> ToolResult:
    """List / pin / unpin / delete artifacts registered for this project."""
    from lqh.artifacts import ArtifactError, BackendArtifactStore

    store = BackendArtifactStore()
    act = (action or "list").lower().strip()

    try:
        if act == "list":
            handles = await store.list_for_project(
                _ckey(project_dir), kind=kind, limit=limit,
            )
            if not handles:
                return ToolResult(content="No artifacts registered for this project.")
            lines = [f"Artifacts for project '{project_dir.name}':"]
            for h in handles:
                flags = []
                if h.pinned:
                    flags.append("📌 pinned")
                if h.checkpoint_role:
                    flags.append(h.checkpoint_role)
                if h.expires_at:
                    flags.append(f"expires {h.expires_at}")
                elif not h.pinned:
                    flags.append("never expires")
                suffix = f"  ({', '.join(flags)})" if flags else ""
                size_mb = h.size_bytes / 1_000_000
                lines.append(f"  - {h.id}  {h.kind}  {size_mb:.1f} MB{suffix}")
            return ToolResult(content="\n".join(lines))

        if not artifact_id:
            return ToolResult.fail("validation", f"Error: action '{act}' requires artifact_id")

        if act == "pin":
            await store.pin(artifact_id)
            return ToolResult(content=f"📌 Pinned {artifact_id} — exempt from auto-expiry.")
        if act == "unpin":
            await store.unpin(artifact_id)
            return ToolResult(content=f"Unpinned {artifact_id} — per-kind expiry re-armed.")
        if act == "delete":
            await store.delete(artifact_id)
            return ToolResult(content=f"Deleted {artifact_id} (stored bytes purged on the next retention tick).")
        return ToolResult.fail("validation", f"Error: unknown action '{act}' (use list/pin/unpin/delete)")
    except ArtifactError as e:
        return ToolResult.fail("upstream", f"Error: {e}")
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail("upstream", f"Error: {e}")


# ----------------------------------------------------------------------
# Inference deployments + keys (LQH Cloud serving).
# Thin clients over the backend's /v1/deployments and /v1/inference-keys
# endpoints; deployed models are served OpenAI-compatible at
# https://inference.lqh.ai/v1 with the deployment name as the model id.
# ----------------------------------------------------------------------

_INFERENCE_ENDPOINT = "https://inference.lqh.ai/v1"


def _fmt_usd_micros(micros: Any) -> str:
    """Format a micros amount (margin already applied by the backend) as dollars."""
    if micros is None:
        return "$?"
    dollars = micros / 1_000_000
    if dollars != 0 and abs(dollars) < 1:
        return f"${dollars:.3f}"
    return f"${dollars:,.2f}"


def _fmt_count(value: Any, default: str = "0") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return default


def _fmt_float(value: Any, fmt: str, default: str = "n/a") -> str:
    if value is None:
        return default
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return default


def _api_error_message(status: int, data: Any) -> str:
    """Pull a human-readable message out of a backend error body."""
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str) and err:
            return err
        if data.get("message"):
            return str(data["message"])
    return f"HTTP {status}"


async def _backend_json(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """Authenticated JSON request against the backend (api_root() + /v1 path)."""
    import httpx

    from lqh.auth import api_root, require_token

    token = require_token()
    async with httpx.AsyncClient(base_url=api_root(), timeout=60.0) as client:
        r = await client.request(
            method,
            path,
            json=json_body,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
    try:
        data = r.json()
    except Exception:
        data = {"message": r.text[:300]}
    return r.status_code, data


def _deployment_gpu(dep: dict[str, Any]) -> str:
    gpu_type = dep.get("gpu_type") or "?"
    count = dep.get("gpu_count") or 1
    return f"{count}x {gpu_type}"


async def handle_push_to_production(
    project_dir: Path,
    *,
    artifact_id: str,
    name: str,
    tier: str = "debug",
    gpu_type: str | None = None,
    min_containers: int | None = None,
    max_containers: int | None = None,
    project_id: str | None = None,
    artifact_format: str | None = None,
    base_model: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Deploy a checkpoint artifact as a serving endpoint on LQH Cloud."""
    # Deployment attribution ALWAYS uses the resolved stable key — a
    # model-supplied project_id (e.g. a stale basename) would create the
    # deployment outside this project's namespace. The parameter is no
    # longer on the tool surface; a stray value is ignored.
    if project_id and project_id != _ckey(project_dir):
        logger.warning(
            "push_to_production: ignoring supplied project_id %r (using %r)",
            project_id, _ckey(project_dir),
        )
    body: dict[str, Any] = {
        "name": name,
        "artifact_id": artifact_id,
        "tier": tier,
        "project_id": _ckey(project_dir),
    }
    if gpu_type:
        body["gpu_type"] = gpu_type
    if min_containers is not None:
        body["min_containers"] = min_containers
    if max_containers is not None:
        body["max_containers"] = max_containers
    if artifact_format:
        body["artifact_format"] = artifact_format
    if base_model:
        body["base_model"] = base_model

    try:
        status, data = await _backend_json("POST", "/v1/deployments", json_body=body)
    except Exception as e:  # noqa: BLE001 - surface clearly to the agent
        return ToolResult.fail("upstream", f"Error: {e}")

    if status == 402:
        return ToolResult.fail(
            "permission",
            (
                "❌ Out of credits — the deployment was not created. The org has "
                "insufficient credits to run a GPU deployment; top up and retry."
            ),
        )
    if status == 409:
        return ToolResult.fail(
            "conflict",
            (
                f"❌ Deployment name '{name}' is already taken. Pick a different "
                "name (list_deployments shows the existing ones) and retry."
            ),
        )
    if status not in (200, 201):
        return ToolResult.fail(
            "upstream",
            f"Error creating deployment: {_api_error_message(status, data)}",
        )

    dep = data
    return ToolResult(
        content=(
            f"🚀 Deployment created\n"
            f"  ID:       {dep.get('id')}\n"
            f"  Name:     {dep.get('name')}\n"
            f"  Status:   {dep.get('status')} (LoRA checkpoints auto-merge first: "
            f"pending → merging → deploying → running; full fine-tunes skip merging)\n"
            f"  Tier:     {dep.get('tier')}\n"
            f"  GPU:      {_deployment_gpu(dep)}\n"
            f"  Est. cost: {_fmt_usd_micros(dep.get('billed_per_hour_estimate'))}/hr while running\n"
            f"\n"
            f"Once status is 'running', the model is served OpenAI-compatible at:\n"
            f"  Endpoint: {_INFERENCE_ENDPOINT}\n"
            f"  Model:    {dep.get('name')}\n"
            f"\n"
            f"Authentication needs an inference key — create one with "
            f"create_inference_key. Track progress with get_deployment."
        )
    )


async def handle_list_deployments(project_dir: Path, **kwargs: Any) -> ToolResult:
    """List all inference deployments with status and cost."""
    try:
        status, data = await _backend_json("GET", "/v1/deployments")
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail("upstream", f"Error: {e}")
    if status != 200:
        return ToolResult.fail(
            "upstream",
            f"Error listing deployments: {_api_error_message(status, data)}",
        )

    deployments = data.get("deployments") or []
    if not deployments:
        return ToolResult(
            content=(
                "No deployments. Use push_to_production to deploy a trained "
                "checkpoint artifact."
            )
        )

    lines = [f"Deployments ({len(deployments)}):"]
    for dep in deployments:
        lines.append(
            f"  - {dep.get('name')}  [{dep.get('status')}]  tier={dep.get('tier')}  "
            f"gpu={_deployment_gpu(dep)}  "
            f"{_fmt_usd_micros(dep.get('billed_per_hour_estimate'))}/hr est  "
            f"billed to date {_fmt_usd_micros(dep.get('billed_cost_micros'))}"
        )
        lines.append(f"      id: {dep.get('id')}")
        if dep.get("error"):
            lines.append(f"      ⚠️ error: {dep['error']}")
    lines.append("")
    lines.append(f"Endpoint: {_INFERENCE_ENDPOINT} (model = deployment name)")
    return ToolResult(content="\n".join(lines))


async def handle_get_deployment(
    project_dir: Path, *, deployment_id: str, **kwargs: Any,
) -> ToolResult:
    """Show one deployment plus its current-period usage summary."""
    try:
        status, dep = await _backend_json("GET", f"/v1/deployments/{deployment_id}")
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail("upstream", f"Error: {e}")
    if status != 200:
        return ToolResult.fail(
            "upstream",
            f"Error fetching deployment: {_api_error_message(status, dep)}",
        )

    lines = [
        f"Deployment {dep.get('name')} ({dep.get('id')}):",
        f"  Status:    {dep.get('status')} (desired: {dep.get('desired_status')})",
        f"  Tier:      {dep.get('tier')}",
        f"  Base model: {dep.get('base_model')}",
        f"  GPU:       {_deployment_gpu(dep)}  "
        f"(containers {dep.get('min_containers')}-{dep.get('max_containers')}"
        + (f", replicas {dep.get('replicas')}" if dep.get("replicas") is not None else "")
        + ")",
        f"  Est. cost: {_fmt_usd_micros(dep.get('billed_per_hour_estimate'))}/hr",
        f"  Billed to date: {_fmt_usd_micros(dep.get('billed_cost_micros'))} "
        f"({_fmt_count(dep.get('gpu_seconds'))} GPU-seconds)",
        f"  Created:   {dep.get('created_at')}",
    ]
    if dep.get("error"):
        lines.append(f"  ⚠️ Error:  {dep['error']}")
    lines.append(f"  Endpoint:  {_INFERENCE_ENDPOINT}  (model = '{dep.get('name')}')")

    # Usage summary is best-effort — the deployment view is still useful
    # if the usage endpoint fails.
    try:
        ustatus, usage = await _backend_json(
            "GET",
            f"/v1/deployments/{deployment_id}/usage",
            params={"range": "current_period"},
        )
    except Exception as e:  # noqa: BLE001
        ustatus, usage = 0, {"message": str(e)}
    if ustatus == 200:
        totals = usage.get("totals") or {}
        lines.append("")
        lines.append("Usage (current period):")
        lines.append(
            f"  Requests:  {_fmt_count(totals.get('requests'))} "
            f"({_fmt_count(totals.get('errors'))} errors)"
        )
        lines.append(
            f"  Tokens:    {_fmt_count(totals.get('input_tokens'))} in / "
            f"{_fmt_count(totals.get('output_tokens'))} out"
        )
        lines.append(
            f"  Latency:   avg TTFT {_fmt_float(totals.get('avg_ttft_ms'), '.0f')} ms, "
            f"avg duration {_fmt_float(totals.get('avg_duration'), '.2f')} s"
        )
        lines.append(
            f"  GPU cost:  {_fmt_usd_micros(usage.get('billed_gpu_cost_micros'))} "
            f"({_fmt_count(usage.get('gpu_seconds'))} GPU-seconds)"
        )
    else:
        lines.append("")
        lines.append(f"(usage unavailable: {_api_error_message(ustatus, usage)})")
    return ToolResult(content="\n".join(lines))


async def _deployment_action(deployment_id: str, action: str, emoji: str) -> ToolResult:
    try:
        status, dep = await _backend_json(
            "POST", f"/v1/deployments/{deployment_id}/{action}",
        )
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail("upstream", f"Error: {e}")
    if status != 200:
        return ToolResult.fail(
            "upstream",
            f"Error on {action}: {_api_error_message(status, dep)}",
        )
    return ToolResult(
        content=(
            f"{emoji} Deployment '{dep.get('name')}' {action} requested — "
            f"status: {dep.get('status')} (desired: {dep.get('desired_status')}). "
            f"Billed to date: {_fmt_usd_micros(dep.get('billed_cost_micros'))}. "
            f"Check with get_deployment."
        )
    )


async def handle_stop_deployment(
    project_dir: Path, *, deployment_id: str, **kwargs: Any,
) -> ToolResult:
    """Stop a running deployment (GPU billing stops)."""
    return await _deployment_action(deployment_id, "stop", "🛑")


async def handle_restart_deployment(
    project_dir: Path, *, deployment_id: str, **kwargs: Any,
) -> ToolResult:
    """Restart a stopped deployment (GPU billing resumes)."""
    return await _deployment_action(deployment_id, "restart", "🔄")


async def handle_create_inference_key(
    project_dir: Path,
    *,
    name: str,
    deployment_ids: list[str] | None = None,
    all_deployments: bool | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Create an inference API key; the plaintext is shown exactly once."""
    body: dict[str, Any] = {"name": name}
    if deployment_ids:
        body["deployment_ids"] = deployment_ids
        if all_deployments:
            body["all_deployments"] = True
    else:
        # No explicit scope → grant access to all deployments.
        body["all_deployments"] = True if all_deployments is None else all_deployments

    try:
        status, data = await _backend_json("POST", "/v1/inference-keys", json_body=body)
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail("upstream", f"Error: {e}")
    if status == 403:
        return ToolResult.fail(
            "permission",
            (
                "❌ The org has reached its inference-key cap. Revoke an unused "
                "key (list_inference_keys / revoke_inference_key) and retry."
            ),
        )
    if status not in (200, 201):
        return ToolResult.fail(
            "upstream",
            f"Error creating inference key: {_api_error_message(status, data)}",
        )

    key = data.get("key", "")
    key_name = data.get("name") or name
    key_id = data.get("id")
    prefix = (key[:12] + "…") if len(key) > 12 else key

    # Usage examples are identical for both messages; only the display embeds
    # the real key (out-of-band), the redacted one keeps the placeholder.
    usage = (
        f"Usage (OpenAI-compatible, model = deployment name):\n"
        f"  from openai import OpenAI\n"
        f"  client = OpenAI(base_url=\"{_INFERENCE_ENDPOINT}\", "
        f"api_key=\"<the key>\")\n"
        f"  client.chat.completions.create(model=\"<deployment-name>\", "
        f"messages=[...])"
    )
    display = (
        f"🔑 Inference key created: {key_name} (id {key_id})\n"
        f"\n"
        f"  {key}\n"
        f"\n"
        f"⚠️ Copy it now — this is the ONLY time the plaintext key is shown. "
        f"It cannot be retrieved again.\n"
        f"\n"
        f"  curl {_INFERENCE_ENDPOINT}/chat/completions \\\n"
        f"    -H 'Authorization: Bearer {key}' \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\"model\": \"<deployment-name>\", "
        f"\"messages\": [{{\"role\": \"user\", \"content\": \"Hi\"}}]}}'\n"
        f"\n"
        f"{usage}"
    )
    redacted = (
        f"🔑 Inference key \"{key_name}\" (id {key_id}, prefix {prefix}) created "
        f"and delivered to the user. The plaintext is not available to you — it "
        f"was shown once and is not retrievable.\n"
        f"\n"
        f"{usage}"
    )
    return ToolResult(
        content=SECRET_DELIVERY_REQUIRED,
        requires_user_input=True,
        secret=SecretDelivery(
            payload=key,
            display=display,
            redacted=redacted,
            env_var="LQH_INFERENCE_KEY",
            env_comment=f'# LQH inference key "{key_name}" (id {key_id})',
        ),
        options=["Continue (hide key)", "Continue & append to .env"],
    )


async def handle_list_inference_keys(project_dir: Path, **kwargs: Any) -> ToolResult:
    """List inference API keys (no plaintext — only prefixes)."""
    try:
        status, data = await _backend_json("GET", "/v1/inference-keys")
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail("upstream", f"Error: {e}")
    if status != 200:
        return ToolResult.fail(
            "upstream",
            f"Error listing inference keys: {_api_error_message(status, data)}",
        )

    keys = data.get("keys") or []
    if not keys:
        return ToolResult(
            content="No inference keys. Create one with create_inference_key."
        )
    lines = [f"Inference keys ({len(keys)}):"]
    for k in keys:
        if k.get("all_deployments"):
            scope = "all deployments"
        else:
            ids = k.get("deployment_ids") or []
            scope = f"{len(ids)} deployment(s)"
        flags = []
        if k.get("revoked_at") or k.get("revoked"):
            flags.append("REVOKED")
        if k.get("expires_at"):
            flags.append(f"expires {k['expires_at']}")
        suffix = f"  ({', '.join(flags)})" if flags else ""
        lines.append(
            f"  - {k.get('name')}  {k.get('prefix')}…  {scope}{suffix}"
        )
        lines.append(f"      id: {k.get('id')}")
    return ToolResult(content="\n".join(lines))


async def handle_revoke_inference_key(
    project_dir: Path, *, key_id: str, **kwargs: Any,
) -> ToolResult:
    """Revoke an inference API key immediately."""
    try:
        status, data = await _backend_json(
            "POST", f"/v1/inference-keys/{key_id}/revoke",
        )
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail("upstream", f"Error: {e}")
    if status != 200:
        return ToolResult.fail(
            "upstream",
            f"Error revoking key: {_api_error_message(status, data)}",
        )
    return ToolResult(
        content=(
            f"🗑️ Revoked inference key '{data.get('name')}' ({data.get('id')}). "
            "Requests using it will now fail; create a new key with "
            "create_inference_key if access is needed again."
        )
    )


def _resolve_hf_pull_repo_type(api, repo_id: str, explicit: str | None) -> tuple[str | None, str | None]:
    """Determine repo_type for hf_pull. Returns (repo_type, error_message)."""
    if explicit is not None:
        if explicit not in ("dataset", "model"):
            return None, f"invalid repo_type '{explicit}' (must be 'dataset' or 'model')"
        return explicit, None

    from huggingface_hub.errors import RepositoryNotFoundError

    for candidate in ("model", "dataset"):
        try:
            api.repo_info(repo_id=repo_id, repo_type=candidate)
            return candidate, None
        except RepositoryNotFoundError:
            continue
        except Exception as e:
            return None, f"failed to query Hub for '{repo_id}': {e}"
    return None, f"repo '{repo_id}' not found on the Hub as either a model or a dataset"


async def handle_hf_pull(
    project_dir: Path,
    *,
    repo_id: str,
    repo_type: str | None = None,
    local_path: str | None = None,
    split: str | None = None,
    subset: str | None = None,
    files: list[str] | None = None,
    revision: str | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> ToolResult:
    """Download a dataset or model from HF Hub to local storage."""
    from lqh.hf_token import local_hf_token

    token = local_hf_token(project_dir)  # optional for public repos

    try:
        api = _get_hf_api(project_dir)
    except ValueError as e:
        return ToolResult.fail("auth", f"Error: {e}")

    resolved_type, err = _resolve_hf_pull_repo_type(api, repo_id, repo_type)
    if err is not None:
        return ToolResult.fail("validation", f"Error: {err}")
    repo_type = resolved_type

    repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    if local_path is None:
        local_path = f"{'datasets' if repo_type == 'dataset' else 'models'}/{repo_name}"

    target = _validate_path(project_dir, local_path)
    # Dataset immutability applies to imports too: refuse to clobber an
    # existing local dataset's parquet files. Recursive — a requested
    # nested file like data/train.parquet must not slip past a top-level
    # glob. overwrite=true from the model is a REQUEST, not consent:
    # replacing existing data goes through the same human confirmation
    # round-trip as data generation (auto mode always declines).
    if repo_type == "dataset":
        existing_parquet = sorted(
            str(p.relative_to(target)) for p in target.rglob("*.parquet")
        ) if target.is_dir() else []
        if existing_parquet and not overwrite:
            return ToolResult.fail("conflict", (
                f"Error: {local_path}/ already contains "
                f"{', '.join(existing_parquet[:3])} — refusing to overwrite "
                "an existing dataset. Pull into a different local_path "
                "(e.g. a versioned name), or pass overwrite=true only "
                "after the user confirmed replacing it."
            ))
        if existing_parquet and overwrite and not kwargs.get("_overwrite_consent"):
            return ToolResult(
                content="OVERWRITE_CONFIRMATION_REQUIRED",
                requires_user_input=True,
                question=(
                    f"The agent wants to pull {repo_id} into {local_path}/, "
                    f"which already contains {', '.join(existing_parquet[:3])}"
                    f"{'…' if len(existing_parquet) > 3 else ''}. Existing "
                    "files may be replaced and this cannot be undone. Allow?"
                ),
                options=[
                    "Yes, replace the existing dataset files",
                    "No, keep the existing data",
                ],
            )
    target.mkdir(parents=True, exist_ok=True)

    try:
        if files:
            from huggingface_hub import hf_hub_download

            downloaded = []
            for f in files:
                out = hf_hub_download(
                    repo_id=repo_id,
                    filename=f,
                    repo_type=repo_type,
                    local_dir=str(target),
                    token=token,
                    revision=revision,
                )
                downloaded.append(out)

            _save_hf_mapping(project_dir, local_path, repo_id, repo_type, split)
            _mf = None
            if repo_type == "dataset":
                from lqh.manifest import write_dataset_manifest

                _mf = write_dataset_manifest(
                    project_dir, target,
                    purpose="imported",
                    source_paths=[
                        f"hf://{repo_id}/{f}" + (f"@{revision}" if revision else "")
                        for f in files
                    ],
                )
            _mf_warn = (
                "\n⚠️ Provenance manifest could not be written (check disk/logs)."
                if repo_type == "dataset" and _mf is None else ""
            )
            return ToolResult(
                content=(
                    f"✅ Downloaded {len(downloaded)} file(s) from {repo_id} ({repo_type}) to {local_path}/\n"
                    + "\n".join(f"  - {Path(d).name}" for d in downloaded)
                    + _mf_warn
                )
            )

        if repo_type == "model":
            if split or subset:
                return ToolResult.fail(
                    "validation",
                    "Error: split/subset are dataset-only options; omit them for model pulls.",
                )

            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(target),
                token=token,
                revision=revision,
            )
            _save_hf_mapping(project_dir, local_path, repo_id, "model")

            file_count = sum(
                1 for p in target.rglob("*")
                if p.is_file() and not any(part.startswith(".") for part in p.relative_to(target).parts)
            )
            return ToolResult(
                content=(
                    f"✅ Downloaded model {repo_id} to {local_path}/ ({file_count} files)\n"
                    f"  Use this path as base_model in training configs or as the eval target."
                )
            )

        # Dataset path: download full dataset via datasets library
        import datasets as ds_lib

        load_kwargs: dict[str, Any] = {"path": repo_id, "trust_remote_code": False}
        if token:
            load_kwargs["token"] = token
        if split:
            load_kwargs["split"] = split
        if subset:
            load_kwargs["name"] = subset
        if revision:
            load_kwargs["revision"] = revision

        dataset = ds_lib.load_dataset(**load_kwargs)

        from lqh.manifest import write_dataset_manifest

        if isinstance(dataset, ds_lib.DatasetDict):
            total_rows = 0
            split_info = []
            for split_name, split_ds in dataset.items():
                out_path = target / f"{split_name}.parquet"
                split_ds.to_parquet(str(out_path))
                total_rows += len(split_ds)
                split_info.append(f"  - {split_name}: {len(split_ds):,} rows -> {split_name}.parquet")

            _save_hf_mapping(project_dir, local_path, repo_id, "dataset")
            _mf = write_dataset_manifest(
                project_dir, target,
                purpose="imported",
                rows=total_rows,
                source_paths=[f"hf://{repo_id}" + (f"@{revision}" if revision else "")],
            )
            return ToolResult(
                content=(
                    f"✅ Downloaded {repo_id} to {local_path}/ ({total_rows:,} rows total)\n"
                    + "\n".join(split_info)
                    + ("\n⚠️ Provenance manifest could not be written (check disk/logs)." if _mf is None else "")
                )
            )

        out_path = target / "data.parquet"
        dataset.to_parquet(str(out_path))

        _save_hf_mapping(project_dir, local_path, repo_id, "dataset", split)
        _mf = write_dataset_manifest(
            project_dir, target,
            purpose="imported",
            rows=len(dataset),
            source_paths=[f"hf://{repo_id}" + (f"@{revision}" if revision else "")],
        )
        return ToolResult(
            content=(
                f"✅ Downloaded {repo_id} to {local_path}/data.parquet "
                f"({len(dataset):,} rows)"
                + ("\n⚠️ Provenance manifest could not be written (check disk/logs)." if _mf is None else "")
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
        return ToolResult.fail("upstream", f"Error downloading {repo_id}: {e}{hint}")


_MODEL_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.ckpt", "*.pt", "*.pth")


def _detect_hf_repo_type(target: Path) -> tuple[str | None, list[str], list[str]]:
    """Inspect a folder and decide whether it looks like a dataset or a model.

    Returns (repo_type, parquet_files, model_files). repo_type is None when
    detection is ambiguous (both sets non-empty) or empty (neither found).
    """
    parquet_files = [p.name for p in target.glob("*.parquet")]
    model_files: list[str] = []
    if (target / "config.json").exists():
        model_files.append("config.json")
    for pattern in _MODEL_WEIGHT_GLOBS:
        model_files.extend(p.name for p in target.glob(pattern))

    if parquet_files and not model_files:
        return "dataset", parquet_files, model_files
    if model_files and not parquet_files:
        return "model", parquet_files, model_files
    return None, parquet_files, model_files


async def handle_hf_push(
    project_dir: Path,
    *,
    local_path: str,
    repo_type: str | None = None,
    repo_id: str | None = None,
    private: bool = True,
    split: str = "train",
    subset: str | None = None,
    commit_message: str | None = None,
    _permissions: PermissionContext | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Push a local dataset or model checkpoint to HF Hub. Requires permission."""
    # Check HF token first
    try:
        api = _get_hf_api(project_dir)
    except ValueError as e:
        return ToolResult.fail("auth", f"Error: {e}")

    # Validate local path
    target = _validate_path(project_dir, local_path)
    if not target.exists():
        return ToolResult.fail("not_found", f"Error: '{local_path}' does not exist")
    if not target.is_dir():
        return ToolResult.fail(
            "validation",
            f"Error: '{local_path}' is not a directory. hf_push expects a folder containing either parquet files (dataset) or model files (config.json + weights).",
        )

    if repo_type is not None and repo_type not in ("dataset", "model"):
        return ToolResult.fail("validation", f"Error: invalid repo_type '{repo_type}' (must be 'dataset' or 'model')")

    detected, parquet_files, model_files = _detect_hf_repo_type(target)

    if repo_type is None:
        if detected is None:
            if parquet_files and model_files:
                return ToolResult.fail(
                    "validation",
                    (
                        f"Error: '{local_path}' contains both parquet files and model files — "
                        f"cannot auto-detect repo type. Pass repo_type='dataset' or repo_type='model' to disambiguate."
                    ),
                )
            return ToolResult.fail(
                "validation",
                (
                    f"Error: '{local_path}' is not recognizable as a dataset or model folder. "
                    f"Expected either .parquet files or HF-style model files "
                    f"(config.json, *.safetensors, *.bin, *.ckpt, *.pt, *.pth)."
                ),
            )
        repo_type = detected
    else:
        # Validate explicit override against what we found.
        if repo_type == "dataset" and not parquet_files:
            return ToolResult.fail(
                "validation",
                f"Error: repo_type='dataset' but no .parquet files found in '{local_path}'.",
            )
        if repo_type == "model" and not model_files:
            return ToolResult.fail(
                "validation",
                (
                    f"Error: repo_type='model' but '{local_path}' has no model files "
                    f"(config.json, *.safetensors, *.bin, *.ckpt, *.pt, *.pth)."
                ),
            )

    # Auto-generate repo_id if not provided
    if repo_id is None:
        try:
            info = api.whoami()
            username = info.get("name", "unknown")
        except Exception as e:
            return ToolResult.fail("auth", f"Error getting HF username: {e}")

        project_name = project_dir.name
        repo_id = f"{username}/{project_name}-{target.name}"

    # Check permission
    if not (_permissions or PermissionContext()).allows_hf_push(project_dir, repo_id):
        details = f"  Split: {split}\n" if repo_type == "dataset" else ""
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            permission_key=f"hf_push:{repo_id}",
            question=(
                f"The agent wants to push to Hugging Face Hub:\n"
                f"  Local: {local_path}\n"
                f"  Repo:  {repo_id} ({repo_type})\n"
                f"  Private: {private}\n"
                f"{details}"
                f"\nAllow push?"
            ),
            options=[
                "Push once, ask again next time",
                "Push and don't ask again for this repo",
                "Push and don't ask again for this project",
                "Do not push",
            ],
        )

    # Dispatch
    if repo_type == "dataset":
        # Use data.parquet if it exists, otherwise first parquet
        data_parquet = target / "data.parquet"
        parquet_path = data_parquet if data_parquet.exists() else target / parquet_files[0]
        return await _execute_hf_push_dataset(
            project_dir, target, parquet_path, local_path, repo_id, private, split, subset, commit_message, api,
        )
    return await _execute_hf_push_model(
        project_dir, target, local_path, repo_id, private, commit_message, api,
    )


async def _execute_hf_push_dataset(
    project_dir: Path,
    target: Path,
    parquet_path: Path,
    local_path: str,
    repo_id: str,
    private: bool,
    split: str,
    subset: str | None,
    commit_message: str | None,
    api,
) -> ToolResult:
    """Push a parquet dataset (and optional README.md) to HF Hub."""
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

        # Dataset.push_to_hub does not pick up a user-authored README.md, so
        # upload it separately if present.
        readme_path = target / "README.md"
        readme_note = ""
        if readme_path.is_file():
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or "Update README.md",
            )
            readme_note = "\n  README: uploaded"

        # Save mapping
        _save_hf_mapping(project_dir, local_path, repo_id, "dataset", split)

        url = f"https://huggingface.co/datasets/{repo_id}"
        visibility = "private" if private else "public"
        return ToolResult(
            content=(
                f"✅ Pushed dataset to HF Hub\n"
                f"  Repo:  {repo_id} ({visibility})\n"
                f"  Split: {split}\n"
                f"  Rows:  {len(dataset):,}"
                f"{readme_note}\n"
                f"  URL:   {url}"
            )
        )

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg:
            hint = " Your HF_TOKEN may not have write access. Check token permissions at https://huggingface.co/settings/tokens"
        else:
            hint = ""
        return ToolResult.fail("upstream", f"Error pushing to {repo_id}: {e}{hint}")


def _looks_like_hub_id(value: str) -> bool:
    """Hub ids are ``owner/name``; local paths usually contain ``/`` and a
    file separator (``..``, ``./``, drive letter, or an actual existing
    path). This is a heuristic, not a verifier — the worst case is the
    user gets a clear error from the HF SDK when the id doesn't resolve.
    """
    if not value or value.startswith((".", "/", "~")) or ":" in value:
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts) and not Path(value).exists()


def _prepare_adapter_for_upload(
    target: Path, repo_id: str,
) -> tuple[bool, str | None]:
    """If ``target`` is a PEFT adapter dir, normalise its metadata for
    a clean HF Hub upload.

    Returns ``(is_adapter, base_model_id)``. When ``is_adapter`` is True:
      - validates that ``adapter_config.json`` carries a hub-shaped
        ``base_model_name_or_path`` (if it's a sandbox-local path the
        upload would dangle; we raise so the caller surfaces a clear
        error)
      - writes a minimal README.md tagging the upload as a peft adapter
        if one isn't already present.
    For merged dirs returns ``(False, None)`` and does nothing.
    """
    cfg_path = target / "adapter_config.json"
    if not cfg_path.is_file():
        return False, None

    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{cfg_path} is not valid JSON ({exc}); cannot push adapter"
        ) from exc
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise RuntimeError(
            f"{cfg_path} has no 'base_model_name_or_path'; cannot push "
            f"adapter without naming the base model. Edit the file to "
            f"set base_model_name_or_path to a hub id."
        )
    if not _looks_like_hub_id(base):
        raise RuntimeError(
            f"{cfg_path}: base_model_name_or_path={base!r} doesn't look "
            f"like a hub id (owner/name). The adapter would dangle on "
            f"HF Hub. Edit the file to point at the published base."
        )

    readme = target / "README.md"
    if not readme.exists():
        readme.write_text(
            "---\n"
            "library_name: peft\n"
            f"base_model: {base}\n"
            "tags:\n"
            "- peft\n"
            "- lora\n"
            "---\n\n"
            f"# {repo_id}\n\n"
            "LoRA adapter trained with [lqh](https://github.com/Liquid4All/lqh).\n\n"
            "## Loading\n\n"
            "```python\n"
            "from transformers import AutoModelForCausalLM\n"
            "from peft import PeftModel\n\n"
            f'base = AutoModelForCausalLM.from_pretrained("{base}")\n'
            f'model = PeftModel.from_pretrained(base, "{repo_id}")\n'
            "```\n"
        )
    return True, base


async def _execute_hf_push_model(
    project_dir: Path,
    target: Path,
    local_path: str,
    repo_id: str,
    private: bool,
    commit_message: str | None,
    api,
) -> ToolResult:
    """Push a model checkpoint folder (weights, config, tokenizer, README) to HF Hub.

    Adapter dirs (containing ``adapter_config.json``) get their
    base-model metadata validated and a peft-tagged README synthesized
    when one isn't already present, so a downstream consumer can find
    the base model and load via ``PeftModel.from_pretrained``.
    """
    try:
        is_adapter, base_model = _prepare_adapter_for_upload(target, repo_id)
    except RuntimeError as exc:
        return ToolResult.fail("runtime", f"Error preparing adapter for upload: {exc}")

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        api.upload_folder(
            folder_path=str(target),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message or f"Push checkpoint from {target.name}",
        )

        _save_hf_mapping(project_dir, local_path, repo_id, "model")

        # Count files for the summary (top-level + nested, excluding hidden dirs).
        file_count = sum(
            1 for p in target.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(target).parts)
        )
        has_readme = (target / "README.md").is_file()

        url = f"https://huggingface.co/{repo_id}"
        visibility = "private" if private else "public"
        adapter_note = f"\n  Kind:   PEFT adapter (base: {base_model})" if is_adapter else ""
        return ToolResult(
            content=(
                f"✅ Pushed model to HF Hub\n"
                f"  Repo:   {repo_id} ({visibility})"
                f"{adapter_note}\n"
                f"  Files:  {file_count}"
                f"{' (incl. README.md)' if has_readme else ''}\n"
                f"  URL:    {url}"
            )
        )

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg:
            hint = " Your HF_TOKEN may not have write access. Check token permissions at https://huggingface.co/settings/tokens"
        else:
            hint = ""
        return ToolResult.fail("upstream", f"Error pushing to {repo_id}: {e}{hint}")


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
            f"Install them with: {install_extras_command('train')}"
        )


def _claim_run_name(
    project_dir: Path, run_name: str | None, prefix: str
) -> tuple[str | None, str | None]:
    """Atomically reserve ``runs/<name>`` for a new run.

    check-then-create races: two CLIs can both pass an existence check
    and then truncate each other's config/logs via ``exist_ok=True``.
    The claim here is the mkdir itself (``exist_ok=False`` — atomic at
    the filesystem level). Auto-generated names retry on collision (the
    other CLI took the next number first); explicit names surface an
    error instead.

    Returns ``(claimed_name, None)`` on success, ``(None, error)`` on
    failure. On success the (empty) run directory exists — downstream
    ``mkdir(exist_ok=True)`` calls are then benign.
    """
    explicit = run_name is not None
    for _ in range(50):
        name = run_name if explicit else _next_run_name(project_dir, prefix)
        try:
            (project_dir / "runs" / name).mkdir(parents=True, exist_ok=False)
            return name, None
        except FileExistsError:
            if explicit:
                return None, (
                    f"run '{name}' already exists — run names must be "
                    "unique (an existing run's config/logs would be "
                    "overwritten). Pick a different run_name or omit it "
                    "for an auto-generated one."
                )
            continue  # racing CLI claimed this number; take the next
        except OSError as exc:
            return None, f"cannot create runs/{name}: {exc}"
    return None, f"could not allocate a unique '{prefix}_*' run name"


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


# Max bring-your-own-compute remotes to show in the project compute
# picker. Extras stay reachable via the "Something else" option.
_MAX_PICKER_REMOTES = 5

# Sentinel ToolResult.content that tells the agent loop to run the
# one-time project compute picker (see lqh/agent.py). Returned by the
# launch handlers when a project has a real compute choice to make but
# hasn't yet pinned a target.
COMPUTE_PICK_REQUIRED = "COMPUTE_PICK_REQUIRED"

# The picker decides the project's compute target for all GPU work
# (training and eval), so the question is phrased generically.
COMPUTE_PICK_QUESTION = "Where should this project run GPU work (training & eval)?"


def _local_gpu_available() -> bool:
    """True iff torch is importable and a CUDA GPU is visible locally.

    Gates the "Local (this machine)" compute-picker option — there is no
    point offering in-process training on a laptop without a usable GPU.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _compute_pick_options(project_dir: Path) -> list[str] | None:
    """Return compute-picker option labels, or None when no pick is needed.

    The compute target is a fixed, per-project decision — not a per-call
    parameter. We only prompt when the project hasn't chosen, no global
    default is set, AND the project actually has a choice to make: at
    least one bring-your-own-compute (SSH) remote is bound, or a local
    CUDA GPU is available for in-process training. Otherwise LQH Cloud is
    the silent default and no dialog is shown.
    """
    from lqh.remote.compute import load_global_default, load_project_default
    from lqh.remote.config import load_remotes

    if load_project_default(project_dir) or load_global_default():
        return None
    remotes = load_remotes(project_dir)
    local_ok = _local_gpu_available()
    if not remotes and not local_ok:
        return None
    options = ["LQH Cloud (recommended)"]
    if local_ok:
        options.append("Local (this machine)")
    for cfg in list(remotes.values())[:_MAX_PICKER_REMOTES]:
        options.append(f"{cfg.name} — {cfg.hostname}")
    options.append("Something else (set up a different remote)")
    return options


def _resolve_compute_target(project_dir: Path) -> str | None:
    """Resolve the project's pinned compute target for a launch.

    The target is fixed per project (see lqh.remote.compute); there is no
    per-call override. Returns ``"cloud"`` or ``"ssh:<name>"`` for remote
    execution, or ``None`` for the in-process local-GPU path — a persisted
    ``"local"`` pin (e.g. chosen via the picker on a GPU box) maps to
    ``None`` so the caller takes its local branch.
    """
    from lqh.remote.compute import resolve_compute

    target = resolve_compute(project_dir)
    return None if target == "local" else target


def _budget_base_model(project_dir: Path, name: str) -> str:
    """Resolve a local checkpoint path to its underlying base model for
    the inference-budget check.

    Follows ``lineage.json`` (written by the trainers) or
    ``adapter_config.json``, then falls back to the owning run's
    config.json (sweep-unwrapped), up to 4 hops — lineage may chain
    through several continuation runs. Non-path names and unresolvable
    paths return unchanged; :func:`lqh.models.check_budget_for_model`
    then applies its own parse rules (unparsable names fail open).
    """
    current = name
    for _ in range(4):
        try:
            path = project_dir / current
            if not path.is_dir():
                return current
        except (OSError, ValueError):
            return current
        nxt: str | None = None
        for fname, key in (
            ("lineage.json", "base_model"),
            ("adapter_config.json", "base_model_name_or_path"),
        ):
            try:
                value = json.loads(
                    (path / fname).read_text(encoding="utf-8")
                ).get(key)
            except (OSError, ValueError):
                continue
            if isinstance(value, str) and value:
                nxt = value
                break
        if nxt is None:
            # model[-lora]/ and checkpoints/<n>/ dirs sit one or two
            # levels below the run root that holds config.json.
            for parent in (path.parent, path.parent.parent):
                try:
                    cfg = json.loads(
                        (parent / "config.json").read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    continue
                if not isinstance(cfg, dict):
                    continue
                inner = (
                    cfg.get("base_config", cfg)
                    if cfg.get("type") == "sweep"
                    else cfg
                )
                value = inner.get("base_model") if isinstance(inner, dict) else None
                if isinstance(value, str) and value:
                    nxt = value
                    break
        if nxt is None or nxt == current:
            return current
        current = nxt
    return current


async def handle_start_training(
    project_dir: Path,
    *,
    type: str,
    base_model: str,
    dataset: str | list[Any],
    eval_dataset: str | list[Any] | None = None,
    scorer: str | None = None,
    disable_scoring: bool = False,
    run_name: str | None = None,
    lora: bool = True,
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    num_iterations: int = 5,
    dpo_beta: float = 0.1,
    golden_source: str = "dataset",
    enable_sweep: bool | None = None,
    grid_size: str = "small",
    override_budget: bool = False,
    _permissions: PermissionContext | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Start a training subprocess.

    Sweep behaviour
    ---------------
    ``enable_sweep`` defaults to None, which resolves per run type
    (``lqh.train.defaults.sweeps_by_default``):

    - **SFT: no sweep.** One run at the recommended defaults from
      ``lqh/train/defaults.py``. A sweep trains the grid sequentially inside a
      single job, so it costs hours on the very first run after a dataset is
      ready — the wrong trade when the defaults are already validated. Sweep
      later, once data volume and model size are settled and the remaining
      gains are small enough to be worth searching for.
    - **DPO: sweep.** DPO is far more sensitive to learning rate and beta, and
      its defaults are not covered by the SFT calibration study, so paying for
      the search is still the safer default there.

    Winners are picked by a cheap, validated in-training proxy:

    - SFT: ``eval_loss`` (Pearson r = −0.90 with judge_mean on ar_to_de).
    - DPO: fixed held-out judge score, with chosen-response CE retained as a
      catastrophic-collapse veto. DPO's own ``eval_loss`` is NOT used — it
      can improve by suppressing rejected likelihood while task quality falls.

    DPO sweeps are deliberately judge-backed and therefore more expensive than
    SFT sweeps. This avoids ranking configurations on incomparable on-policy
    preference subsets.

    Explicit ``enable_sweep=true/false`` always wins. ``learning_rate`` /
    ``num_epochs`` / ``dpo_beta`` supplied by the agent are honoured when not
    sweeping; under a sweep they are overridden by the grid.

    Eval / scoring contract
    -----------------------
    ``dataset`` and ``eval_dataset`` are strictly separated for both SFT
    and DPO: ``dataset`` is the only source of training prompts (SFT trains
    on it; DPO generates on-policy rollouts from it), and ``eval_dataset``
    is held-out — used only for evaluation, never to generate training
    data.

    ``eval_dataset`` is mandatory and must resolve to a DIFFERENT path than
    ``dataset`` (the call is rejected otherwise). For SFT it is the sweep's
    selection signal (held-out val_loss) and the judge eval-of-best set.
    For DPO it is the fixed judge-scored validation set shared by all configs
    and iterations. Preference-pair chosen CE is still measured at every
    iteration, but is used only to veto clear collapse. The focused benchmark
    keeps an additional untouched final-test split outside this tool call.

    ``scorer`` must be an explicit decision: pass the project's
    default/current scorer, or set ``disable_scoring=True`` (only when the
    user explicitly asks not to score). The call is rejected when neither
    is provided, so a missing judge score is never a silent omission.

    ``disable_scoring`` is SFT-only — it skips the final judge eval while
    training still proceeds on the val_loss proxy. **DPO rejects it**:
    on-policy DPO builds its preference pairs from scored rollouts every
    iteration, so a scorer is mandatory for DPO to run at all.
    """
    # Compute target is fixed per project — there is no per-call override.
    # When the project has a real choice to make (a BYOC remote and/or a
    # local GPU) but hasn't pinned a target yet, defer to the one-time
    # picker driven by the agent loop (see lqh/agent.py). This never fires
    # for cloud-only projects (silent default) or once a choice has been
    # persisted.
    pick_options = _compute_pick_options(project_dir)
    if pick_options is not None:
        return ToolResult(
            content=COMPUTE_PICK_REQUIRED,
            requires_user_input=True,
            question=COMPUTE_PICK_QUESTION,
            options=pick_options,
        )

    remote = _resolve_compute_target(project_dir)

    is_grpo = type in ("grpo", "on_policy_grpo")
    # GRPO runs only on LQH Cloud: the distinct `grpo` image carries the
    # vLLM + newer-TRL runtime that the local/SSH training environments do
    # not (and cannot — trl<1.1 is pinned there; see GRPO_IMPLEMENTATION.md).
    if is_grpo and not (remote and _is_cloud_target(remote)):
        return ToolResult.fail(
            "config",
            (
                "Error: GRPO training runs only on LQH Cloud — it needs the "
                "dedicated grpo image (vLLM rollout engine + TRL 1.10), which "
                "local and SSH environments do not have. This project's compute "
                f"target is {'local' if remote is None else remote!r}."
            ),
        )

    # Check torch + GPU only when running locally; remote execution has its
    # own venv (provisioned by remote_setup) and its own GPUs.
    if remote is None:
        err = _check_torch_available()
        if err:
            return ToolResult.fail("config", f"❌ {err}")

        try:
            import torch

            if not torch.cuda.is_available():
                return ToolResult.fail(
                    "config",
                    "⚠️ No CUDA GPU detected. Training requires a GPU.",
                )
            gpu_info = ", ".join(
                f"{torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())
            )
        except Exception:
            gpu_info = "unknown"
    else:
        gpu_info = f"remote ({remote})"

    # Inference-budget guard (SPEC.md `**Budget**:` line). Enforced here —
    # not just in skill prose — so a direct CLI call or an agent slip can't
    # silently train past a pin or cap. `override_budget=True` is the
    # explicit escape hatch and must only be passed after the user agreed.
    # Zero-shot *evaluation* of larger models stays unrestricted: gauging
    # headroom above the budget is part of the routing playbook.
    budget_violation = None
    try:
        from lqh.models import check_budget_for_model

        spec_text = (project_dir / "SPEC.md").read_text(encoding="utf-8")
        # Continuation runs (e.g. DPO on runs/sft_001/model-lora) pass a
        # checkpoint PATH — the budget constrains the underlying model,
        # so resolve the lineage before checking. A pinned budget must
        # not reject continuing from a checkpoint OF the pinned model,
        # and a size cap must see through the path to the real size.
        effective_base = _budget_base_model(project_dir, base_model)
        budget_violation = check_budget_for_model(spec_text, effective_base)
        if budget_violation and effective_base != base_model:
            budget_violation += (
                f" (base_model '{base_model}' resolves to '{effective_base}' "
                "via its checkpoint lineage.)"
            )
    except OSError:
        pass
    if budget_violation and not override_budget:
        return ToolResult.fail(
            "permission",
            (
                f"Error: {budget_violation} Training past the budget needs the "
                "user's explicit consent: ask_user first, and only on a yes "
                "re-call start_training with override_budget=true. To change "
                "the budget itself, update the '**Budget**:' line in SPEC.md "
                "with the user."
            ),
        )

    # Validate dataset source(s). A single string or a list of sources to
    # combine; train sources may carry an integer `repeat` over-sampling
    # factor. Resolves to canonical {"path", "repeat", "source"} entries.
    dataset_sources, train_resolved, ds_err = _resolve_training_sources(
        project_dir, dataset, kind="dataset", allow_repeat=True
    )
    if ds_err:
        return ToolResult.fail("validation", ds_err)

    # eval_dataset is mandatory: the sweep needs a held-out signal to pick its
    # winner, and the judge eval-of-best needs rollouts to score. (The tool
    # schema marks it required; this guards non-schema callers.)
    if not eval_dataset:
        return ToolResult.fail(
            "validation",
            (
                "Error: eval_dataset is required. Pass the project's held-out eval "
                "set (e.g. 'datasets/<name>_eval'). It is the signal used to select "
                "the sweep winner and the set the best checkpoint is judge-scored on."
            ),
        )

    eval_sources, eval_resolved, eval_err = _resolve_training_sources(
        project_dir, eval_dataset, kind="eval_dataset", allow_repeat=False
    )
    if eval_err:
        return ToolResult.fail("validation", eval_err)

    # Reject duplicate eval sources — scoring the same set twice would
    # double-count it in the macro-average.
    eval_seen: set[str] = set()
    for p in eval_resolved:
        key = str(p)
        if key in eval_seen:
            return ToolResult.fail(
                "validation",
                (
                    "Error: eval_dataset lists the same source twice "
                    f"({p.name}). Each eval source must be distinct so the "
                    "macro-average weights them once each."
                ),
            )
        eval_seen.add(key)

    # dataset and eval_dataset must be DISTINCT. Evaluating on the training
    # prompts is exactly the leak the train/eval split exists to prevent —
    # reject any overlap between a train source and an eval source.
    overlap = {str(p) for p in train_resolved} & eval_seen
    if overlap:
        names = ", ".join(sorted(Path(p).as_posix() for p in overlap))
        return ToolResult.fail(
            "validation",
            (
                "Error: eval_dataset must be different from dataset — these "
                f"source(s) appear in both: {names}. Evaluating on the training "
                "prompts leaks train into eval. Pass separate held-out eval "
                "set(s) (e.g. 'datasets/<name>_eval')."
            ),
        )

    # On-policy DPO builds its preference pairs by judge-scoring generated
    # rollouts every iteration, so a scorer is mandatory — scoring cannot be
    # disabled the way it can for SFT (where it only gates the final eval).
    if type in ("on_policy_dpo", "dpo") and disable_scoring:
        return ToolResult.fail(
            "validation",
            (
                "Error: scoring cannot be disabled for DPO. On-policy DPO assembles "
                "its preference pairs from scored rollouts each iteration, so a scorer "
                "is required — pass `scorer=<path>` (the project's default/best scorer)."
            ),
        )

    # GRPO's scorer IS the reward: the judge ranks every rollout group
    # against it. Without a scorer there is no training signal at all.
    if is_grpo and disable_scoring:
        return ToolResult.fail(
            "validation",
            (
                "Error: scoring cannot be disabled for GRPO — the scorer is the "
                "reward function (every rollout group is judge-ranked against "
                "it). Pass `scorer=<path>` (the project's default/best scorer)."
            ),
        )

    # No GRPO sweeps in v1 (single run at the measured defaults; parallel
    # GRPO is judge-RPM hostile and the colocated vLLM engine cannot share
    # a GPU across configs).
    if is_grpo and enable_sweep:
        return ToolResult.fail(
            "validation",
            (
                "Error: GRPO does not support sweeps — it runs ONCE at the "
                "measured defaults (see the rl skill). Omit enable_sweep."
            ),
        )

    # Scoring must be an explicit decision: pass a scorer, or opt out via
    # disable_scoring. Silently omitting the scorer would degrade eval-of-best
    # to proxy-only with no judge score — a common, quiet failure mode.
    if not scorer and not disable_scoring:
        return ToolResult.fail(
            "validation",
            (
                "Error: no scorer provided. The best checkpoint needs a scorer to get "
                "a real judge score. Pass `scorer=<path>` set to the project's "
                "default/current scorer (the one under evals/scorers/ used for the "
                "baseline eval), or — only if the user explicitly asked not to score — "
                "set disable_scoring=true."
            ),
        )

    scorer_path: str | None = None
    if scorer:
        scorer_resolved = _validate_path(project_dir, scorer)
        if not scorer_resolved.exists():
            return ToolResult.fail("not_found", f"Error: scorer not found at {scorer}")
        scorer_path = scorer

    # Vision-language (LFM-VL) bases switch the run into the vision path:
    # AutoProcessor + image collation in the subprocess, the Liquid VLM
    # LoRA recipe, and conservative batch defaults (the text calibration
    # probe is skipped for vision). SFT-only for now.
    from lqh.models import is_vlm_model_name

    is_vision = is_vlm_model_name(base_model)
    if is_vision and type != "sft":
        return ToolResult.fail(
            "validation",
            (
                f"Error: {type} is not supported for vision-language models yet — "
                f"only SFT is. Train {base_model} with type='sft'."
            ),
        )

    # Select the name used in the permission prompt without claiming it yet.
    # The agent re-invokes this handler after the user approves the launch; a
    # pre-consent mkdir would make that second invocation collide with its own
    # empty directory. The approved auto-generated name is pinned by the agent
    # on re-invocation, then atomically claimed immediately before submission.
    run_prefix = "sft" if type == "sft" else ("grpo" if is_grpo else "dpo")
    requested_run_name = run_name or None
    run_name = requested_run_name or _next_run_name(project_dir, run_prefix)
    run_dir = project_dir / "runs" / run_name

    # Reject an already-taken explicit name before asking the user to approve a
    # launch that cannot succeed. The later atomic claim remains authoritative
    # and closes the check-then-create race.
    if requested_run_name and run_dir.exists():
        return ToolResult.fail(
            "conflict",
            (
                f"Error: run '{run_name}' already exists — run names must be "
                "unique (an existing run's config/logs would be overwritten). "
                "Pick a different run_name or omit it for an auto-generated one."
            ),
        )

    from lqh.project_meta import compute_spec_sha256

    # Dataset scale is a first-class input for post-eval routing (the
    # failure_analysis skill decides "scale data vs model" from it), so
    # record row counts in the run config where training_status can
    # surface them. It also sizes the LoRA batch below, which is why this
    # runs before the defaults are resolved. Parquet metadata only — no data
    # is loaded.
    train_rows = 0
    train_rows_effective = 0
    for src_entry, src_path in zip(dataset_sources, train_resolved):
        rows = _parquet_metadata(src_path)[0] or 0
        train_rows += rows
        train_rows_effective += rows * (src_entry.get("repeat") or 1)
    eval_rows = sum((_parquet_metadata(p)[0] or 0) for p in eval_resolved)

    # Build config. Every hyperparameter default comes from one place
    # (lqh/train/defaults.py) so the hp_defaults calibration study can update
    # them with a data-only edit; explicit tool arguments still win.
    from lqh.train import defaults as hp_defaults

    recommended = hp_defaults.recommended(
        run_type=type,
        lora=lora,
        modality="vision" if is_vision else "text",
        base_model=base_model,
        # Effective (repeat-weighted) rows and the caller's epoch override are
        # what the run actually trains on — they set the batch so the run takes
        # enough optimizer steps to learn.
        train_rows=train_rows_effective,
        num_epochs=num_epochs,
    )
    lr = learning_rate if learning_rate is not None else recommended.learning_rate
    epochs = num_epochs if num_epochs is not None else recommended.num_epochs

    config: dict[str, Any] = {
        "type": type,
        "base_model": base_model,
        "dataset": _sources_to_config(dataset_sources),
        "dataset_rows": {
            "train": train_rows,
            "train_effective": train_rows_effective,
            "eval": eval_rows,
        },
        # num_samples doubles as the manifest's sample count (see
        # write_run_manifest, which copies it only when present).
        "num_samples": train_rows,
        # Spec revision at submission time: checkpoints trained from this
        # config stay traceable to the spec they were built against even
        # after SPEC.md changes mid-run (PERSISTENCY_PLAN.md R6).
        "spec_sha256": compute_spec_sha256(project_dir),
        # Recommended defaults, with the caller's explicit overrides applied.
        # training_config() carries num_epochs for SFT and omits it for DPO.
        "training": {
            **recommended.training_config(),
            "learning_rate": lr,
        },
        "lora": recommended.lora,
        "manifest": ["base_model", "dataset"],
    }
    if epochs is not None:
        config["training"]["num_epochs"] = epochs
    if is_vision:
        config["modality"] = "vision"
        # Per-image token budget for the processor. Effective text budget
        # is roughly max_seq_length − n_images × max_image_tokens.
        config["training"]["max_image_tokens"] = hp_defaults.VISION_MAX_IMAGE_TOKENS

    if eval_sources:
        config["eval_dataset"] = _sources_to_config(eval_sources)
        config["eval_on_checkpoints"] = True
        config["manifest"].append("eval_dataset")
        if type in ("on_policy_dpo", "dpo"):
            # DPO quality selection must use a fixed prompt set shared across
            # configs and iterations. Projects with a dedicated DPO validation
            # split may override this config field; the public tool's safe
            # default is its required held-out eval dataset.
            config["held_out_eval_dataset"] = _sources_to_config(eval_sources)
    if scorer_path:
        config["scorer"] = scorer_path
        config["manifest"].append("scorer")

    if type in ("on_policy_dpo", "dpo"):
        config["num_iterations"] = num_iterations
        config["dpo_beta"] = dpo_beta
        config["golden_source"] = golden_source
        config["dpo_early_abort_delta"] = 0.5
        config["dpo_plateau_patience"] = 2
        config["dpo_min_held_out_improvement"] = 0.05
        config["training"]["dpo_step_aware_batch"] = True
        config["training"]["dpo_target_optimizer_steps_per_iter"] = 30
        # Dataset gold is useful only when it is verified to beat the policy
        # rollout under the same judge.  The scoring paths cache chosen scores
        # once and activate the gap selector when this block is present.
        config["selection"] = {
            "top_quantile": 1.0,
            "min_gap": 1.0,
            "min_pairs_per_iter": 50,
        }

    # Human-readable summary of the (possibly multiple) training sources,
    # e.g. "datasets/type_a + datasets/type_b (×3)".
    def _summarize(entry: dict[str, Any]) -> str:
        d = Path(entry["path"]).parent.as_posix()
        rep = entry.get("repeat", 1)
        return f"{d} (×{rep})" if rep and rep > 1 else d

    dataset_summary = " + ".join(_summarize(e) for e in dataset_sources)
    eval_summary = " + ".join(Path(e["path"]).parent.as_posix() for e in eval_sources)

    # Cloud bundles are tarred to disk and uploaded (large ones via a
    # presigned direct-to-storage PUT) — still warn before shipping a
    # very large dataset (image datasets inflate fast: base64 data-URLs
    # inside the messages column) since the upload takes bandwidth and
    # the server caps staged bundles at 2 GiB.
    size_warning = ""
    try:
        total_bytes = sum(p.stat().st_size for p in (*train_resolved, *eval_resolved) if p.exists())
        if remote and total_bytes > 1 << 30:
            size_warning = (
                f"\n  ⚠️ Datasets total {total_bytes / (1 << 30):.1f} GB — the cloud "
                "bundle upload may be slow, and bundles over 2 GB are refused. "
                "Consider fewer/smaller samples"
                + (" or smaller images (max_dim)." if is_vision else ".")
            )
    except OSError:
        pass

    # Permission check. Training has its own permission domain (see
    # permissions.check_training_permission) so approving a run never grants
    # arbitrary pipeline/script execution.
    perm_key = f"training:{run_name}"
    if not (_permissions or PermissionContext()).allows_training(project_dir, run_name):
        # Only on the cloud branch: this prompt is shared with local and
        # SSH runs, where "sent with this job" would simply be false.
        hf_line = ""
        if remote and _is_cloud_target(remote):
            from lqh.hf_token import hf_disclosure_line

            hf_line = hf_disclosure_line(project_dir, indent="  ")
        # GRPO's dominant cost after GPU time is judge traffic: the reward
        # channel makes ~(groups + completions) judge calls per optimizer
        # step. Surface the estimate so approving the run is an informed
        # spend decision, not a surprise on the invoice.
        grpo_line = ""
        if is_grpo:
            from lqh.train import defaults as _hp

            _steps = _hp.GRPO_MAX_STEPS
            _per_step = 64 + 64 // _hp.GRPO_NUM_GENERATIONS
            grpo_line = (
                f"  Reward:    ~{_steps * _per_step / 1000:.0f}k judge calls "
                f"over {_steps} steps (billed; judge tier per the rl skill)\n"
            )
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            permission_key=perm_key,
            question=(
                f"The agent wants to start a {type.upper()} training run:\n"
                f"  Run:       {run_name}\n"
                f"  Model:     {base_model}\n"
                f"  Dataset:   {dataset_summary}\n"
                f"  Eval:      {eval_summary}\n"
                + grpo_line
                + f"  GPU:       {gpu_info}{size_warning}\n"
                + hf_line
                + "\nAllow execution?"
            ),
            options=[
                "Start training",
                "Do not start training",
            ],
        )

    # HF donation, cloud only — on a local or SSH run nothing leaves the
    # machine through us, so there is nothing to consent to. Asked before
    # the run name is claimed below: a prompt returns and re-invokes, and
    # claiming first would strand a run directory on every round trip.
    donate_hf = False
    if remote and _is_cloud_target(remote):
        donate_hf, hf_prompt = _resolve_hf_donation(
            project_dir, _permissions, kwargs.get("_hf_donate"), "training"
        )
        if hf_prompt is not None:
            return hf_prompt

    # Consent has been established for this exact name. Reserve it atomically
    # only now, so permission prompts and declines never leave ghost run
    # directories. Passing the selected name explicitly also makes a race fail
    # closed instead of silently launching a different, unapproved name.
    claimed_run_name, claim_err = _claim_run_name(
        project_dir, run_name, run_prefix
    )
    if claim_err:
        return ToolResult.fail("conflict", f"Error: {claim_err}")
    run_name = claimed_run_name
    run_dir = project_dir / "runs" / run_name

    on_bg_started = kwargs.get("on_background_task_started")

    # Build the launch payload. Sweep wraps the base config + grid spec;
    # single-config sends the base config directly. The remote backend
    # ships either payload identically — sweep just looks like a different
    # subprocess type to the watcher.
    #
    # None means "decide by run type": SFT trains once at the validated
    # defaults, DPO still sweeps. An explicit boolean from the agent wins.
    if enable_sweep is None:
        enable_sweep = hp_defaults.sweeps_by_default(type)
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
            donate_hf_token=donate_hf,
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
    donate_hf_token: bool = False,
) -> ToolResult:
    """Start training on a remote backend.

    Routes to ``CloudBackend`` when ``remote_name == "cloud"`` (or the
    legacy ``"ssh:cloud"`` form); otherwise looks up an SSH remote by
    name and uses ``SSHDirectBackend``.
    """
    from lqh.remote.compute import is_cloud, ssh_remote_name
    from lqh.remote.backend import RemoteConfig

    # --- Cloud path ---
    if is_cloud(remote_name):
        from lqh.remote.cloud import CloudBackend

        cfg = RemoteConfig(
            name="cloud",
            type="cloud",
            hostname="api.lqh.ai",  # informational; CloudBackend hits api_root()
            remote_root="cloud:lqh",
        )
        backend = CloudBackend(cfg, project_dir)
        try:
            job_id = await backend.submit_run(
                str(run_dir), config, module=module,
                donate_hf_token=donate_hf_token,
            )
        except Exception as e:
            return ToolResult.fail("upstream", f"Error launching cloud training: {e}")

        if on_bg_started is not None:
            on_bg_started(run_name, "train", run_name, "cloud")

        from lqh.project_log import append_event
        inner = config.get("base_config", config) if config.get("type") == "sweep" else config
        append_event(
            project_dir,
            "training_started",
            f"Started {inner.get('type', 'training')} run {run_name} on LQH Cloud (job {job_id})",
            run_name=run_name,
            run_type=inner.get("type", "unknown"),
            base_model=inner.get("base_model", ""),
            remote="cloud",
        )
        data_line = _training_data_line(config)
        return ToolResult(
            content=(
                f"🚀 Cloud training submitted\n"
                f"  Run:     {run_name}\n"
                f"  Type:    {config.get('type', 'unknown')}\n"
                + (f"  Data:    {data_line}\n" if data_line else "")
                + f"  Job ID:  {job_id}\n\n"
                f"Backend: LQH Cloud (api.lqh.ai). Use training_status to monitor progress."
                + _submit_advisories(backend)
            ),
            workflow_launched=True,
        )

    # --- SSH path (existing behavior) ---
    from lqh.remote.config import get_remote
    from lqh.remote.ssh_direct import SSHDirectBackend

    ssh_name = ssh_remote_name(remote_name) or remote_name
    remote_config = get_remote(project_dir, ssh_name)
    if remote_config is None:
        return ToolResult.fail(
            "config",
            f"Error: remote '{ssh_name}' not found. Use remote_list to see configured remotes.",
        )

    if remote_config.type == "ssh_slurm":
        return ToolResult.fail("config", "Error: SSH+Slurm backend is not yet implemented.")

    backend = SSHDirectBackend(remote_config, project_dir)
    remote_run_dir = f"{remote_config.remote_root}/runs/{run_name}"

    try:
        job_id = await backend.submit_run(str(run_dir), config, module=module)
    except Exception as e:
        return ToolResult.fail("upstream", f"Error launching remote training: {e}")

    if on_bg_started is not None:
        on_bg_started(run_name, "train", run_name, ssh_name)

    from lqh.project_log import append_event

    # When sweep is enabled the launch config is wrapped:
    # {"type": "sweep", "base_config": {real config}}. Unwrap one
    # level so the event log records the actual run_type/base_model.
    inner = config.get("base_config", config) if config.get("type") == "sweep" else config

    append_event(
        project_dir,
        "training_started",
        f"Started {inner.get('type', 'training')} run {run_name} on remote '{ssh_name}' (job {job_id})",
        run_name=run_name,
        run_type=inner.get("type", "unknown"),
        base_model=inner.get("base_model", ""),
        remote=ssh_name,
    )

    data_line = _training_data_line(config)
    return ToolResult(
        content=(
            f"🚀 Remote training started on '{ssh_name}'\n"
            f"  Run:      {run_name}\n"
            f"  Type:     {config['type']}\n"
            + (f"  Data:     {data_line}\n" if data_line else "")
            + f"  Job ID:   {job_id}\n"
            f"  Host:     {remote_config.hostname}\n"
            f"  Dir:      {remote_run_dir}\n\n"
            f"Use training_status(run_name='{run_name}') to monitor progress."
        ),
        workflow_launched=True,
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

    data_line = _training_data_line(config)
    return ToolResult(
        content=(
            f"🚀 Training started\n"
            f"  Run:    {run_name}\n"
            f"  Type:   {config.get('type', 'unknown')}\n"
            + (f"  Data:   {data_line}\n" if data_line else "")
            + f"  PID:    {pid}\n"
            f"  Dir:    runs/{run_name}/\n\n"
            f"Use training_status to monitor progress."
        ),
        workflow_launched=True,
    )


async def handle_training_status(
    project_dir: Path,
    *,
    run_name: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Check training run status.

    The compute target is derived per-run from the run's persisted
    ``remote_job.json`` (written at launch) — never from a caller
    argument. A run with that metadata polls the corresponding remote
    (local PIDs aren't comparable across machines); a run without it is
    a local subprocess. List mode (no run_name) applies the same rule to
    every runs/<name>/ entry.
    """
    from lqh.subprocess_manager import SubprocessManager

    manager = SubprocessManager()

    if run_name:
        run_dir = _validate_path(project_dir, f"runs/{run_name}")
        if not run_dir.exists():
            return ToolResult.fail("not_found", f"Error: run '{run_name}' not found")
        meta = _read_remote_meta(run_dir)
        if meta is not None:
            result = await _training_status_remote(
                project_dir, run_name, meta["remote_name"],
            )
            # Cloud data-gen: the job reaching "completed" is not the
            # end of the story — the dataset download happens in the
            # background watcher afterwards (the marker is consumed once
            # it lands). Without this note, an interactive agent seeing
            # "completed" could proceed to scoring before the file exists.
            if (run_dir / ".lqh_data_gen.json").exists():
                result.content += (
                    "\n⏳ Dataset download pending — wait for the completion "
                    "notification before using the dataset locally."
                )
            return result
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
    """Check status of a remote training run.

    Branches on ``remote_name``: ``"cloud"`` (or the legacy
    ``"ssh:cloud"``) routes through ``CloudBackend``; anything else
    is treated as an SSH remote.
    """
    from lqh.remote.compute import is_cloud

    run_dir = project_dir / "runs" / run_name

    meta_file = run_dir / "remote_job.json"
    if not meta_file.exists():
        return ToolResult.fail("not_found", f"Error: no remote job metadata for run '{run_name}'.")
    meta = json.loads(meta_file.read_text())
    job_id = meta["job_id"]
    remote_run_dir = meta["remote_run_dir"]

    if is_cloud(remote_name):
        from lqh.remote.backend import RemoteConfig
        from lqh.remote.cloud import CloudBackend

        cfg = RemoteConfig(
            name="cloud",
            type="cloud",
            hostname="api.lqh.ai",
            remote_root="cloud:lqh",
        )
        backend = CloudBackend(cfg, project_dir)
        display_remote = "LQH Cloud"
    else:
        from lqh.remote.compute import ssh_remote_name
        from lqh.remote.config import get_remote
        from lqh.remote.ssh_direct import SSHDirectBackend

        ssh_name = ssh_remote_name(remote_name) or remote_name
        remote_config = get_remote(project_dir, ssh_name)
        if remote_config is None:
            return ToolResult.fail("config", f"Error: remote '{ssh_name}' not found.")
        backend = SSHDirectBackend(remote_config, project_dir)
        display_remote = ssh_name

    try:
        # Sync progress first
        await backend.sync_progress(remote_run_dir, str(run_dir))
        status = await backend.poll_status(job_id)
    except Exception as e:
        return ToolResult.fail("upstream", _format_training_status_error(e))

    state_emoji = {
        "running": "🏃", "completed": "✅", "failed": "❌",
        "waiting_for_scoring": "⏳", "unknown": "❓",
    }
    emoji = state_emoji.get(status.state, "❓")
    lines = [f"{emoji} **{run_name}** — {status.state} (remote: {display_remote})"]
    if status.current_step is not None:
        lines.append(f"  Step: {status.current_step}")

    # Cloud jobs: fetch the snapshot BEFORE the error line so the
    # diagnosis can sit right under it — a bare provider error ("exit
    # code 124", "orphaned: ...") tells the reader nothing about what to
    # do next. Best-effort, never fails status.
    snap: dict[str, Any] | None = None
    if display_remote == "LQH Cloud":
        try:
            snap = await backend.job_snapshot(job_id)
        except Exception:  # noqa: BLE001
            snap = None

    if status.error:
        lines.append(f"  Error: {status.error}")
    if snap is not None:
        from lqh.remote.failure import attempt_lines, diagnosis_line

        # Diagnosis (what class of failure), then the compute line, then
        # the lease history — "preempted 2×, resumed" has to be legible
        # to the user and to the agent without reading the raw error.
        lines.extend(diagnosis_line(snap, status.error))
        lines.extend(_format_cloud_resource_lines(snap))
        lines.extend(attempt_lines(snap))

    # Also show local mirror progress if available
    from lqh.train.progress import read_latest_metrics
    latest = read_latest_metrics(run_dir)
    latest_sweep_lines = _format_latest_sweep_progress(latest)
    if latest_sweep_lines:
        lines.extend(latest_sweep_lines)
    elif latest:
        if latest.get("loss") is not None:
            lines.append(f"  Loss: {latest['loss']:.4f}")
        if latest.get("lr") is not None:
            lines.append(f"  LR:   {latest['lr']:.2e}")
    if progress_line := _unified_progress_line(run_dir):
        lines.append(f"  Progress: {progress_line}")
    if data_line := _run_config_data_line(run_dir):
        lines.append(f"  Data: {data_line}")

    # Cloud sync only records artifact descriptors — on a finished run,
    # pull the eval outputs (aggregates + per-sample results) into the
    # local mirror so the final-eval block and get_eval_failures work.
    if status.state == "completed":
        await _hydrate_run_eval_artifacts(run_dir)

    # Final eval from the local mirror (empty until eval_result.json syncs).
    lines.extend(_format_final_eval_block(run_dir))
    # Steps / loss trajectory / token accuracy / LR — how the run trained, not
    # just what it scored. The half of the diagnosis a judge mean can't give.
    lines.extend(_format_training_health_block(run_dir))

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

    iterations_dir = run_dir / "iterations"
    if iterations_dir.exists():
        iter_lines = _format_dpo_iter_stats(iterations_dir)
        if iter_lines:
            lines.append("  DPO iterations:")
            lines.extend(iter_lines)

    abort = run_dir / "early_abort.json"
    if abort.exists():
        try:
            payload = json.loads(abort.read_text())
            reason = payload.get("reason", "regression past threshold")
            lines.append(f"  ⚠️  Early-abort signaled: {reason}")
        except (json.JSONDecodeError, OSError):
            lines.append("  ⚠️  Early-abort signaled (unparseable)")

    sweep_lines = _format_sweep_summary(run_dir)
    if sweep_lines:
        lines.extend(sweep_lines)

    return ToolResult(content="\n".join(lines))


def _format_cloud_resource_lines(snap: dict[str, Any]) -> list[str]:
    """One 'Compute:' line from a cloud job snapshot: GPU, elapsed vs.
    the wall-clock limit, and the billed hard cap. Tolerates any field
    being absent (older backends)."""
    resource = snap.get("resource") or {}
    parts: list[str] = []
    if gpu := resource.get("gpu_type"):
        parts.append(f"{gpu} GPU")
    timeout = resource.get("timeout_minutes")
    elapsed_min: int | None = None
    started = snap.get("started_at")
    if isinstance(started, str) and started:
        try:
            from datetime import datetime, timezone

            t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
            ended = snap.get("ended_at")
            t1 = (
                datetime.fromisoformat(ended.replace("Z", "+00:00"))
                if isinstance(ended, str) and ended
                else datetime.now(timezone.utc)
            )
            elapsed_min = max(0, int((t1 - t0).total_seconds() // 60))
        except ValueError:
            pass
    if isinstance(timeout, (int, float)) and timeout > 0:
        if elapsed_min is not None:
            parts.append(f"{elapsed_min}/{int(timeout)} min used")
        else:
            parts.append(f"{int(timeout)} min limit")
    cap = resource.get("worst_case_cost_billed_micros")
    if isinstance(cap, (int, float)) and cap > 0:
        parts.append(f"hard cap ≈ ${float(cap) / 1e6:.2f}")
    return [f"  Compute: {' · '.join(parts)}"] if parts else []


_TRAINING_STATUS_RATE_LIMIT_HINT = (
    "LQH is already watching this training run in the background. Do not poll "
    "training_status again; if you need to wait for completion, end the "
    "conversation without emitting another tool call. The session will wake "
    "automatically when the watcher observes completion."
)


def _is_http_429_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "429" in msg
        or "rate limit" in msg.lower()
        or "too many requests" in msg.lower()
    )


def _format_training_status_error(exc: Exception) -> str:
    content = f"Error checking remote status: {exc}"
    if _is_http_429_error(exc):
        content = f"{content}\n\n{_TRAINING_STATUS_RATE_LIMIT_HINT}"
    return content


def _training_data_line(config: dict[str, Any]) -> str:
    """``train 2,000 rows (eff. 6,000) · eval 300 rows`` from a run config's
    ``dataset_rows`` (absent in configs written before it existed → "")."""
    inner = (
        config.get("base_config", config)
        if config.get("type") == "sweep"
        else config
    )
    rows = inner.get("dataset_rows") or {}
    train = rows.get("train")
    if not train:
        return ""
    line = f"train {train:,} rows"
    eff = rows.get("train_effective") or train
    if eff != train:
        line += f" (eff. {eff:,})"
    if rows.get("eval"):
        line += f" · eval {rows['eval']:,} rows"
    return line


def _run_config_data_line(run_dir: Path) -> str:
    """The :func:`_training_data_line` for a run directory's config.json."""
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(cfg, dict):
        return ""
    return _training_data_line(cfg)


def _format_final_eval_block(run_dir: Path) -> list[str]:
    """Render the run-root ``eval_result.json`` — the eval-of-best result of a
    training run, or the headline of a standalone eval run (``eval_hf_model`` /
    ``start_local_eval``); both live under ``runs/<name>/``.

    Includes the score distribution when the file carries one. Older files
    (and cloud workers on an older lqh) lack ``score_distribution`` — render
    whatever is present, never fail status over it.
    """
    result_file = run_dir / "eval_result.json"
    if not result_file.exists():
        # Non-sweep SFT writes its final eval under checkpoints/final/
        # instead of the run root — fall back so single-run training
        # still gets the full final-eval block.
        result_file = run_dir / "checkpoints" / "final" / "eval_result.json"
    if not result_file.exists():
        return []
    try:
        result = json.loads(result_file.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    scores = result.get("scores") or {}
    mean = scores.get("mean")
    if mean is None:
        return []

    per_source = result.get("per_source") or {}
    parts = [f"mean={mean:.2f}"]
    if len(per_source) > 1:
        parts[0] = f"mean={mean:.2f} (macro)"
        weighted = result.get("scores_weighted_mean")
        if weighted is not None:
            parts.append(f"weighted={weighted:.2f}")
    num_scored = result.get("num_scored")
    if num_scored is not None:
        scored_part = f"scored={num_scored}"
        if result.get("num_failed"):
            scored_part += f" ({result['num_failed']} failed)"
        parts.append(scored_part)

    lines = [f"  Final eval: {', '.join(parts)}"]
    # The headline is the number people act on, so the caveat travels with it
    # rather than living only in eval_result.json.
    eval_warning = result.get("failure_warning")
    if isinstance(eval_warning, str) and eval_warning:
        lines.append(f"    {eval_warning.strip()}")
    if len(per_source) > 1:
        for label in sorted(per_source):
            entry = per_source[label]
            src_mean = entry.get("scores", {}).get("mean")
            if src_mean is None:
                continue
            attempted = entry.get("num_attempted")
            scored_n = entry.get("num_scored")
            # _score_stats([]) reports mean 0.0, so a source nothing could be
            # scored on must not print as "mean=0.00" — that is an invented
            # score, and the worst one available.
            if scored_n == 0:
                lines.append(
                    f"    {label}: no usable scores"
                    + (f" (0/{attempted})" if isinstance(attempted, int) else "")
                )
                continue
            line = f"    {label}: mean={src_mean:.2f}"
            # A source averaged over 3 of its 100 rows carries the same weight
            # in the macro mean as one averaged over all 100.
            if isinstance(attempted, int) and isinstance(scored_n, int) and (
                scored_n < attempted
            ):
                line += f" (scored {scored_n}/{attempted})"
            lines.append(line)

    dist = result.get("score_distribution")
    if isinstance(dist, dict):
        try:
            from lqh.scoring import format_score_distribution_text

            lines.extend(
                "  " + ln
                for ln in format_score_distribution_text(dist).splitlines()
            )
        except Exception:  # noqa: BLE001 — malformed dist must not break status
            pass
    return lines


def _run_config_training_block(run_dir: Path) -> dict[str, Any]:
    """A run's ``config.json`` → ``training`` block ({} when unreadable).

    Unwraps a sweep's ``base_config`` the same way :func:`_training_data_line`
    does. Note that a sweep's base block carries the *pre-grid* hyperparameters
    — the grid overrides learning rate and epochs per config — so for a sweep,
    read the winner's own sub-dir instead (see :func:`_sweep_winner_config_id`).
    """
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(cfg, dict):
        return {}
    inner = cfg.get("base_config", cfg) if cfg.get("type") == "sweep" else cfg
    training = inner.get("training") if isinstance(inner, dict) else None
    return training if isinstance(training, dict) else {}


def _sweep_winner_config_id(run_dir: Path) -> str | None:
    """The winning config id from ``sweep_summary.json``, or None.

    A sweep's per-config training output lives in ``sweep_<config_id>/``; the
    run root has no ``eval_history.json`` of its own, so anything reading
    training metrics off a swept run has to go through the winner.
    """
    try:
        summary = json.loads((run_dir / "sweep_summary.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    winner = summary.get("winner") if isinstance(summary, dict) else None
    config_id = winner.get("config_id") if isinstance(winner, dict) else None
    return config_id if isinstance(config_id, str) and config_id else None


def _training_metrics_dir(run_dir: Path) -> Path:
    """Where this run's ``eval_history.json`` lives.

    The run root for a single training run; the winner's ``sweep_<id>/`` for a
    sweep (each grid point trains in its own child dir). Falls back to the run
    root when there is no identifiable winner, which yields an empty health
    block rather than a wrong one.
    """
    if (run_dir / "eval_history.json").exists():
        return run_dir
    config_id = _sweep_winner_config_id(run_dir)
    if config_id:
        candidate = run_dir / f"sweep_{config_id}"
        if candidate.is_dir():
            return candidate
    return run_dir


def _format_training_health_block(run_dir: Path) -> list[str]:
    """Render the mechanical training signals from ``eval_history.json``.

    A judge mean alone cannot distinguish "the dataset is bad" from "the run
    took 21 optimizer steps at a learning rate too low to move the adapter" —
    and that distinction is the difference between a $35 model step-up and a
    40-minute rerun. All of this is already dumped by ``lqh/train/sft.py``
    (HF Trainer's ``log_history``); the only thing missing was showing it, so
    diagnosing a flat run meant paging through raw ``stderr.log`` tqdm output.

    For a sweep this reports the *winner's* child run — the config the run
    actually produced a checkpoint from, and the only one whose hyperparameters
    are worth reading.

    Best-effort: any missing field is simply omitted, and an unreadable or
    unexpected file yields no lines rather than an error.
    """
    metrics_dir = _training_metrics_dir(run_dir)
    try:
        history = json.loads((metrics_dir / "eval_history.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = [r for r in history if isinstance(r, dict)] if isinstance(history, list) else []
    if not rows:
        return []

    def _num(row: dict[str, Any], key: str) -> float | None:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _first(key: str) -> float | None:
        return next((v for r in rows if (v := _num(r, key)) is not None), None)

    def _last(key: str) -> float | None:
        return next((v for r in reversed(rows) if (v := _num(r, key)) is not None), None)

    steps = max((int(v) for r in rows if (v := _num(r, "step")) is not None), default=0)
    first_loss, last_loss = _first("loss"), _last("loss")
    if last_loss is None:
        # A run that logged no per-step loss row still has the end-of-training
        # summary row, whose train_loss is the mean over the run.
        last_loss = _last("train_loss")
    eval_loss = _last("eval_loss")
    token_acc = _last("eval_mean_token_accuracy")
    if token_acc is None:
        token_acc = _last("mean_token_accuracy")
    # For a sweep, the LR that matters is the winning grid point's, which lives
    # in the child's own config.json. Deliberately NO fallback to the run root:
    # a sweep's base_config carries the *pre-grid* learning rate, so falling
    # back would report a plausible wrong number — and the skill tells the agent
    # to retrain at 5x this value. Omitting it is the safe failure.
    lr = _run_config_training_block(metrics_dir).get("learning_rate")

    parts: list[str] = []
    if steps:
        parts.append(f"{steps} steps")
    if first_loss is not None and last_loss is not None and first_loss != last_loss:
        parts.append(f"loss {first_loss:.2f} → {last_loss:.2f}")
    elif last_loss is not None:
        parts.append(f"loss {last_loss:.2f}")
    if eval_loss is not None:
        parts.append(f"eval_loss {eval_loss:.2f}")
    if token_acc is not None:
        parts.append(f"token_acc {token_acc * 100:.0f}%")
    if isinstance(lr, (int, float)) and not isinstance(lr, bool):
        parts.append(f"lr {lr:.1e}")
    if not parts:
        return []

    lines = [f"  Training health: {' · '.join(parts)}"]
    from lqh.train.defaults import SFT_MIN_HEALTHY_OPTIMIZER_STEPS

    if steps and steps < SFT_MIN_HEALTHY_OPTIMIZER_STEPS:
        # Deliberately names levers `start_training` actually accepts: the
        # batch is derived, not a parameter, so "lower the batch" would be a
        # tool call that silently changes nothing.
        lines.append(
            f"    ⚠️  Only {steps} optimizer updates — too few to conclude "
            f"anything about the data. Retrain with more rows (or a higher "
            f"num_epochs on the same data) before changing the dataset or the "
            f"model size."
        )
    return lines


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

    from lqh.train.progress import read_latest_metrics
    latest = read_latest_metrics(run_dir)
    latest_sweep_lines = _format_latest_sweep_progress(latest)
    if latest_sweep_lines:
        lines.extend(latest_sweep_lines)
    else:
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
    if progress_line := _unified_progress_line(run_dir):
        lines.append(f"  Progress: {progress_line}")
    if data_line := _run_config_data_line(run_dir):
        lines.append(f"  Data: {data_line}")

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
                        line = f"    {cp_dir.name}: mean={mean_score:.2f}"
                        # Compact distribution hint per checkpoint (the full
                        # histogram is reserved for the final-eval block).
                        dist = result.get("score_distribution")
                        pct = (
                            dist.get("percentiles")
                            if isinstance(dist, dict) else None
                        )
                        if isinstance(pct, dict) and all(
                            k in pct for k in ("p10", "p50", "p90")
                        ):
                            line += (
                                f"  p10/p50/p90="
                                f"{pct['p10']:.1f}/{pct['p50']:.1f}/{pct['p90']:.1f}"
                            )
                        # A checkpoint mean taken over a thinned sample set is
                        # not comparable to its neighbours, and these lines are
                        # read side by side to pick a checkpoint.
                        if result.get("num_failed"):
                            line += f"  ({result['num_failed']} failed)"
                        eval_results.append(line)
                        # When multiple eval sources were scored, the headline
                        # mean is a macro-average — show the per-source breakdown.
                        per_source = result.get("per_source") or {}
                        if len(per_source) > 1:
                            for label in sorted(per_source):
                                entry = per_source[label]
                                src_mean = entry.get("scores", {}).get("mean")
                                if src_mean is None:
                                    continue
                                scored_n = entry.get("num_scored")
                                attempted_n = entry.get("num_attempted")
                                # _score_stats([]) reports mean 0.0, so a
                                # source nothing could be scored on would
                                # otherwise print as "mean=0.00" — the
                                # strongest negative signal there is, invented.
                                # These lines get read side by side to pick a
                                # checkpoint.
                                if scored_n == 0:
                                    eval_results.append(
                                        f"      {label}: no usable scores"
                                        + (
                                            f" (0/{attempted_n})"
                                            if isinstance(attempted_n, int)
                                            else ""
                                        )
                                    )
                                    continue
                                src_line = f"      {label}: mean={src_mean:.2f}"
                                if (
                                    isinstance(scored_n, int)
                                    and isinstance(attempted_n, int)
                                    and scored_n < attempted_n
                                ):
                                    src_line += f" (scored {scored_n}/{attempted_n})"
                                eval_results.append(src_line)
                except (json.JSONDecodeError, OSError):
                    pass
        if eval_results:
            lines.append("  Eval scores:")
            lines.extend(eval_results)

    # Final eval (run-root eval_result.json): eval-of-best for training
    # runs, headline for standalone eval runs — with score distribution.
    lines.extend(_format_final_eval_block(run_dir))
    lines.extend(_format_training_health_block(run_dir))

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

    # Sweep summary (present only when the run was launched as a sweep —
    # every DPO run, and SFT runs where the caller asked for one). We
    # deliberately surface only the validated selection metric here:
    #   - For SFT: eval_loss (Pearson r=-0.90 with judge_mean).
    #   - For DPO: the held-out judge score, plus eval_ce_chosen_delta_ref
    #     as the collapse veto. DPO eval_loss and eval_rewards/margins are
    #     NOT shown — they correlate with judge in the wrong direction
    #     and would mislead the agent into picking a collapsed config.
    #     See lqh/train/sweep.py for the experiment that established this.
    sweep_lines = _format_sweep_summary(run_dir)
    if sweep_lines:
        lines.extend(sweep_lines)

    return "\n".join(lines)


def _unified_progress_line(run_dir: Path) -> str:
    """Render the latest current-attempt v1 percentage/ETA for tool output."""
    from lqh.progress import (
        format_event_oneline,
        read_progress_events,
        select_display_event,
    )
    from lqh.train.progress import read_current_attempt_id

    rows = [
        row for row in read_progress_events(run_dir, last_n=256)
        if isinstance(row.get("overall_fraction"), (int, float))
    ]
    attempt_id = read_current_attempt_id(run_dir)
    if isinstance(attempt_id, str) and attempt_id:
        rows = [row for row in rows if row.get("attempt_id") == attempt_id]
    if not rows:
        return ""
    latest = select_display_event(rows)
    if latest is None:
        return ""
    phase_rows = [row for row in rows if row.get("phase") == latest.get("phase")]
    observed_candidates: list[float] = []
    for name in ("progress.jsonl", "observer_progress.jsonl"):
        try:
            observed_candidates.append((run_dir / name).stat().st_ctime)
        except OSError:
            pass
    observed_at = max(observed_candidates) if observed_candidates else None
    line, _ = format_event_oneline(
        latest, history=phase_rows, observed_at=observed_at,
    )
    return line


def _format_latest_sweep_progress(latest: dict[str, Any] | None) -> list[str]:
    """Render the live sweep row from progress.jsonl, if the latest row is one."""
    if not latest:
        return []
    phase = latest.get("phase")
    if not isinstance(phase, str) or not phase.startswith("sweep_"):
        return []

    config_id = latest.get("config_id")
    config_label = f" · {config_id}" if isinstance(config_id, str) and config_id else ""
    idx = latest.get("config_index")
    total = latest.get("n_configs")
    position = ""
    if isinstance(idx, int) and isinstance(total, int) and total > 0:
        position = f" {idx + 1}/{total}"
    elif isinstance(total, int) and total > 0:
        position = f" {total} configs"

    if phase == "sweep_start":
        proxy = latest.get("proxy_key")
        proxy_label = f" · proxy={proxy}" if isinstance(proxy, str) and proxy else ""
        return [f"  Sweep: starting{position}{proxy_label}"]

    if phase == "sweep_config_start":
        return [f"  Sweep: running config{position}{config_label}"]

    if phase == "sweep_config_progress":
        step = latest.get("child_step", latest.get("step"))
        max_steps = latest.get("child_max_steps")
        step_label = ""
        if isinstance(step, int):
            if isinstance(max_steps, int) and max_steps > 0:
                step_label = f" · step {step}/{max_steps}"
            else:
                step_label = f" · step {step}"
        metric_bits: list[str] = []
        loss = latest.get("child_loss", latest.get("loss"))
        if isinstance(loss, (int, float)):
            metric_bits.append(f"loss={loss:.4f}")
        eval_loss = latest.get("child_eval_loss")
        if isinstance(eval_loss, (int, float)):
            metric_bits.append(f"eval_loss={eval_loss:.4f}")
        lr = latest.get("child_lr", latest.get("lr"))
        if isinstance(lr, (int, float)):
            metric_bits.append(f"lr={lr:.2e}")
        epoch = latest.get("child_epoch", latest.get("epoch"))
        if isinstance(epoch, (int, float)):
            metric_bits.append(f"epoch={epoch:.2f}")
        metrics = f" · {' '.join(metric_bits)}" if metric_bits else ""
        return [f"  Sweep: config{position}{config_label}{step_label}{metrics}"]

    if phase == "sweep_config_done":
        primary = latest.get("primary")
        primary_label = (
            f" · proxy={primary:.4f}"
            if isinstance(primary, (int, float))
            else ""
        )
        return [f"  Sweep: completed config{position}{config_label}{primary_label}"]

    return []


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
            # DPO row: held-out judge score (the selection metric) + the
            # chosen-CE drift that acts as the collapse veto. DPO eval_loss and
            # reward margins stay hidden — they can look great on a collapsed
            # model. `primary` is the NEGATED judge mean so that lower-is-better
            # holds for every mode; show the judge score itself, right way up.
            dref = r.get("eval_ce_chosen_delta_ref")
            judge = r.get("held_out_judge_mean")
            extras: list[str] = []
            if isinstance(judge, (int, float)):
                extras.append(f"judge={judge:.3f}")
            elif isinstance(primary, (int, float)):
                extras.append(f"judge={-primary:.3f}")
            else:
                extras.append("judge=—")
            if isinstance(dref, (int, float)):
                extras.append(f"CE(ch)Δref={dref:+.3f}")
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
                    # `kept` is the pre-generation selection. What actually
                    # reaches training is pairs_written, which is smaller
                    # whenever golden generation dropped a response or a pair
                    # was identical/duplicate. Showing only `kept` overstates
                    # the training set.
                    written = stats.get("pairs_written")
                    if isinstance(written, int) and written != kept:
                        kept_str += f" → {written} written"
                gp50 = stats.get("qualifying_gap_p50") or stats.get("gap_p50")
                gp90 = stats.get("qualifying_gap_p90") or stats.get("gap_p90")
                if gp50 is not None and gp90 is not None:
                    gap_str = f"gap p50={gp50:.1f}, p90={gp90:.1f}"
                if stats.get("skipped_reason"):
                    gap_str = (gap_str + " ⚠️ skipped: " + stats["skipped_reason"]).strip()
                missing = stats.get("golden_missing")
                if isinstance(missing, int) and missing > 0:
                    gap_str = (
                        gap_str
                        + f" ⚠️ {missing} golden response(s) missing"
                    ).strip()
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
    **kwargs: Any,
) -> ToolResult:
    """Stop a training subprocess.

    Whether the run is remote is derived from its persisted
    ``remote_job.json`` (written at launch), not from a caller argument.
    """
    from lqh.subprocess_manager import SubprocessManager

    run_dir = _validate_path(project_dir, f"runs/{run_name}")
    if not run_dir.exists():
        return ToolResult.fail("not_found", f"Error: run '{run_name}' not found")

    meta = _read_remote_meta(run_dir)
    if meta is not None:
        return await _stop_training_remote(project_dir, run_name, meta["remote_name"])

    manager = SubprocessManager()
    if not manager.is_alive(run_dir):
        return ToolResult.fail("conflict", f"Run '{run_name}' is not currently running.")

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
        return ToolResult.fail("runtime", f"Failed to stop run '{run_name}'.")


async def _stop_training_remote(
    project_dir: Path,
    run_name: str,
    remote_name: str,
) -> ToolResult:
    """Stop a remote training run.

    Branches on ``remote_name``: ``"cloud"`` routes through
    ``CloudBackend``; anything else is treated as an SSH remote.
    """
    from lqh.remote.compute import is_cloud

    run_dir = project_dir / "runs" / run_name
    meta_file = run_dir / "remote_job.json"
    if not meta_file.exists():
        return ToolResult.fail("not_found", f"Error: no remote job metadata for run '{run_name}'.")

    meta = json.loads(meta_file.read_text())
    job_id = meta["job_id"]

    if is_cloud(remote_name):
        from lqh.remote.backend import RemoteConfig
        from lqh.remote.cloud import CloudBackend

        cfg = RemoteConfig(
            name="cloud",
            type="cloud",
            hostname="api.lqh.ai",
            remote_root="cloud:lqh",
        )
        backend = CloudBackend(cfg, project_dir)
        remote_name = "LQH Cloud"
    else:
        from lqh.remote.compute import ssh_remote_name
        from lqh.remote.config import get_remote
        from lqh.remote.ssh_direct import SSHDirectBackend

        ssh_name = ssh_remote_name(remote_name) or remote_name
        remote_config = get_remote(project_dir, ssh_name)
        if remote_config is None:
            return ToolResult.fail("config", f"Error: remote '{ssh_name}' not found.")
        remote_name = ssh_name
        backend = SSHDirectBackend(remote_config, project_dir)

    try:
        await backend.teardown(job_id)
    except Exception as e:
        return ToolResult.fail("upstream", f"Error stopping remote run: {e}")

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
    dataset: str | list[Any],
    scorer: str,
    run_name: str | None = None,
    system_prompt_path: str | None = None,
    response_format_path: str | None = None,
    max_new_tokens: int = 4096,
    **kwargs: Any,
) -> ToolResult:
    """Start a local inference subprocess for model evaluation."""
    from lqh.remote.compute import ssh_remote_name

    on_bg_started = kwargs.get("on_background_task_started")

    # Compute target is fixed per project — same one-time picker as
    # training. If the project has a real choice (a BYOC remote and/or a
    # local GPU) but hasn't pinned a target, defer to the picker.
    pick_options = _compute_pick_options(project_dir)
    if pick_options is not None:
        return ToolResult(
            content=COMPUTE_PICK_REQUIRED,
            requires_user_input=True,
            question=COMPUTE_PICK_QUESTION,
            options=pick_options,
        )

    # Eval runs on the project's pinned SSH remote when there is one;
    # otherwise it runs locally in-process. Cloud eval of LQH-trained
    # checkpoints isn't wired yet (the artifact-aware cloud eval path is
    # a gap — eval_hf_model only accepts HF repos), so a cloud-pinned
    # project falls back to the local path rather than erroring; to
    # evaluate a cloud-trained checkpoint, push it via hf_push and use
    # eval_hf_model instead.
    target = _resolve_compute_target(project_dir)
    ssh_name = ssh_remote_name(target) if target else None
    if ssh_name:
        return await _start_local_eval_remote(
            project_dir, model_path, dataset, scorer, run_name, target,
            system_prompt_path=system_prompt_path,
            response_format_path=response_format_path,
            max_new_tokens=max_new_tokens,
            on_bg_started=on_bg_started,
        )

    # Check torch
    err = _check_torch_available()
    if err:
        return ToolResult.fail("config", f"❌ {err}")

    # Validate paths
    model_dir = _validate_path(project_dir, model_path)
    if not model_dir.exists():
        return ToolResult.fail("not_found", f"Error: model not found at {model_path}")

    # Eval dataset(s) — one path or a list of held-out sources. Multiple
    # sources are scored separately and combined into a macro-average (each
    # source weighted equally), same as the training eval-of-best.
    eval_sources, eval_resolved, ds_err = _resolve_training_sources(
        project_dir, dataset, kind="dataset", allow_repeat=False
    )
    if ds_err:
        return ToolResult.fail("validation", ds_err)

    scorer_resolved = _validate_path(project_dir, scorer)
    if not scorer_resolved.exists():
        return ToolResult.fail("not_found", f"Error: scorer not found at {scorer}")

    try:
        system_prompt, schema_dict = _resolve_eval_extras(
            project_dir,
            system_prompt_path=system_prompt_path,
            response_format_path=response_format_path,
        )
    except FileNotFoundError as e:
        return ToolResult.fail("not_found", f"Error: {e}")

    run_name, claim_err = _claim_run_name(project_dir, run_name or None, "local_eval")
    if claim_err:
        return ToolResult.fail("conflict", f"Error: {claim_err}")

    eval_run_dir = project_dir / "runs" / run_name

    # Build infer config
    config: dict[str, Any] = {
        "type": "infer",
        "spec_sha256": _eval_spec_hash(project_dir),
        "base_model": str(model_dir),
        "dataset": _sources_to_config(eval_sources),
        "scorer": scorer,
        "num_samples": sum((_parquet_metadata(path)[0] or 0) for path in eval_resolved),
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
        ),
        workflow_launched=True,
    )


async def handle_eval_hf_model(
    project_dir: Path,
    *,
    repo: str,
    eval_dataset: str,
    scorer: str,
    revision: str = "main",
    training_method: str = "lora",
    base_model: str | None = None,
    system_prompt_path: str | None = None,
    judge_size: str = "small",
    run_name: str | None = None,
    max_new_tokens: int = 4096,
    timeout_minutes: int = 120,
    _permissions: PermissionContext | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Submit an eval_hf cloud job — runs ``lqh.infer.eval_hf`` in a
    GPU sandbox (backend-implemented) to evaluate any HF checkpoint
    against this project's eval set + scorer.

    Cloud-only: HF download + GPU inference + judge scoring all happen
    sandbox-side using the scoped LQH_API_TOKEN. SSH backends are not
    a supported route in v1 — they'd need their own HF-download +
    scoped-token plumbing that doesn't exist yet, and the use case
    (evaluate someone else's HF model without locally training)
    naturally lives on managed compute.
    """
    on_bg_started = kwargs.get("on_background_task_started")

    # --- Validate inputs ---
    if training_method not in ("lora", "full"):
        return ToolResult.fail(
            "validation",
            f"Error: training_method must be 'lora' or 'full', got {training_method!r}",
        )
    if training_method == "lora" and not base_model:
        return ToolResult.fail(
            "validation",
            "Error: base_model is required when training_method='lora'",
        )
    if judge_size not in ("small", "medium", "large"):
        return ToolResult.fail(
            "validation",
            f"Error: judge_size must be small/medium/large, got {judge_size!r}",
        )
    # Same clamp the backend picker applies ([10, the 24h sandbox max])
    # so the consent prompt and the submitted job agree.
    timeout_minutes = max(10, min(int(timeout_minutes or 120), 1440))

    # Eval dataset(s) — one path or a list of held-out sources, scored
    # separately and macro-averaged sandbox-side.
    eval_sources, eval_resolved, ds_err = _resolve_training_sources(
        project_dir, eval_dataset, kind="eval_dataset", allow_repeat=False
    )
    if ds_err:
        return ToolResult.fail("validation", ds_err)

    scorer_resolved = _validate_path(project_dir, scorer)
    if not scorer_resolved.exists():
        return ToolResult.fail("not_found", f"Error: scorer not found at {scorer}")

    try:
        system_prompt, schema_dict = _resolve_eval_extras(
            project_dir,
            system_prompt_path=system_prompt_path,
            response_format_path=None,
        )
    except FileNotFoundError as e:
        return ToolResult.fail("not_found", f"Error: {e}")

    num_samples = sum((_parquet_metadata(path)[0] or 0) for path in eval_resolved)

    from lqh.remote.backend import RemoteConfig
    from lqh.remote.cloud import CloudBackend

    cfg = RemoteConfig(
        name="cloud",
        type="cloud",
        hostname="api.lqh.ai",
        remote_root="cloud:lqh",
    )
    backend = CloudBackend(cfg, project_dir)

    # Dry-run the backend planner so consent covers the ACTUAL worst
    # case: large models get upsized off the default L4 to a pricier
    # GPU, and consenting to an L4-rate "hard cap" that the planner then
    # exceeds is not consent. Also fails fast on known no-fit models —
    # the real submit would reject them anyway.
    plan: dict[str, Any] | None = None
    try:
        plan_config: dict[str, Any] = {
            "hf_repo": repo,
            "timeout_minutes": timeout_minutes,
        }
        plan = await backend.plan_job(
            "eval_hf",
            base_model=base_model if training_method == "lora" else None,
            config=plan_config,
        )
    except Exception:  # noqa: BLE001 — older backend / network: estimate instead
        plan = None
    if plan is not None and plan.get("fits") is False:
        return ToolResult.fail(
            "validation",
            f"Error: this model fits no supported GPU — "
            f"{plan.get('no_fit_reason') or 'no fitting GPU available'}",
        )

    # Consent gate — GPU wall-clock spend needs the user's sign-off
    # (same shape as the cloud data-gen consent).
    if not (_permissions or PermissionContext()).allows_cloud_eval_hf(project_dir):
        hours = timeout_minutes / 60
        if plan is not None and plan.get("gpu_type"):
            cap_micros = plan.get("worst_case_cost_billed_micros")
            cap_part = (
                f"hard cap ≈ ${float(cap_micros) / 1e6:.2f} at the "
                f"{hours:g}-hour timeout"
                if isinstance(cap_micros, (int, float)) and cap_micros > 0
                else f"capped by the {hours:g}-hour timeout"
            )
            size_caveat = (
                " Model size is unknown, so the GPU choice is not "
                "size-validated."
                if "not size-validated" in str(plan.get("selection_reason") or "")
                else ""
            )
            compute_line = (
                f"  Compute: {plan['gpu_type']} GPU billed by wall-clock — "
                f"{cap_part}.{size_caveat} Judge LLM tokens are billed as "
                "usual.\n\n"
            )
        else:
            # Planner preview unavailable (older backend, network): fall
            # back to the default-GPU rate with an explicit caveat.
            rate_usd = await _fetch_eval_hf_rate_usd()
            if rate_usd is not None:
                rate_note = f"≈ ${rate_usd:.2f}/hr"
            else:
                rate_usd = 2.4
                rate_note = "≈ $2.40/hr at default rates"
            compute_line = (
                f"  Compute: single GPU billed by wall-clock, {rate_note} — "
                f"hard cap ≈ ${hours * rate_usd:.0f} at the {hours:g}-hour "
                f"timeout, assuming the default GPU. Large models may be "
                f"placed on a bigger (pricier) GPU; the actual GPU and cap "
                f"appear in the job status. Judge LLM tokens are billed as "
                f"usual.\n\n"
            )
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
            permission_key=f"cloud_eval_hf:{repo}",
            question=(
                f"The agent wants to evaluate an HF checkpoint on LQH Cloud:\n"
                f"  Model:   {repo}@{revision} ({training_method}"
                + (f", base {base_model}" if training_method == "lora" else "")
                + ")\n"
                f"  Eval:    {num_samples} samples, judge:{judge_size}, "
                f"max {max_new_tokens} tokens/sample\n"
                + compute_line
                + _eval_hf_disclosure(project_dir)
                + "Submit the cloud eval?"
            ),
            options=[
                "Submit to cloud (this time)",
                "Submit and don't ask again for this project",
                "Do not submit",
            ],
        )

    # HF donation. Evaluating your own PRIVATE checkpoint is the headline
    # trial workflow, and lqh.infer.eval_hf reads HF_TOKEN in the sandbox
    # to fetch it — so this is the path where a missing token most often
    # costs someone a paid failed run. Asked before the run name is
    # claimed: a prompt returns and re-invokes, and claiming first would
    # strand a run directory each round trip.
    donate_hf, hf_prompt = _resolve_hf_donation(
        project_dir, _permissions, kwargs.get("_hf_donate"), "eval"
    )
    if hf_prompt is not None:
        return hf_prompt

    run_name, claim_err = _claim_run_name(project_dir, run_name or None, "eval_hf")
    if claim_err:
        return ToolResult.fail("conflict", f"Error: {claim_err}")
    run_dir = project_dir / "runs" / run_name

    # --- Build sandbox config ---
    # The sandbox cd's to the bundle root, so the dataset / scorer
    # paths in the config must be relative paths inside the bundle.
    # We pass them as the user gave them (project-relative); the
    # manifest list below tells build_bundle which on-disk files to
    # ship under those same paths.
    config: dict[str, Any] = {
        "type": "eval_hf",
        "spec_sha256": _eval_spec_hash(project_dir),
        "hf_repo": repo,
        "revision": revision,
        "training_method": training_method,
        "eval_dataset": _sources_to_config(eval_sources),
        "scorer": scorer,
        "judge_size": judge_size,
        "max_new_tokens": max_new_tokens,
        # Read by the backend picker (clamped there too) — the job's
        # wall-clock and hard compute-cost cap.
        "timeout_minutes": timeout_minutes,
        "num_samples": num_samples,
        # manifest tells lqh.remote.bundle.resolve_manifest which keys
        # in this config name files to include in the bundle. The hf
        # repo itself is downloaded sandbox-side via snapshot_download
        # — it's NOT in the manifest.
        "manifest": ["eval_dataset", "scorer"],
    }
    if training_method == "lora":
        config["base_model"] = base_model
    if system_prompt is not None:
        config["system_prompt"] = system_prompt
    if schema_dict is not None:
        config["response_format"] = schema_dict
    # sglang-engine debug/tuning knobs (ISSUE 4 P1) — not in the tool
    # schema, but forwardable programmatically (parity tests, ad-hoc
    # ops). eval_hf.py passes them through to the engine dispatcher;
    # none participate in the resume digest.
    for knob in ("generation_concurrency", "sglang_extra_args", "force_hf_engine"):
        if (v := kwargs.get(knob)) is not None:
            config[knob] = v
    if system_prompt_path:
        # Also include the source file in the bundle so the
        # artifact_lineage row can pin it (the publisher records the
        # config alongside the eval artifacts).
        config["system_prompt_path"] = system_prompt_path
        config["manifest"].append("system_prompt_path")

    # --- Submit to LQH Cloud (backend constructed above, pre-consent) ---
    try:
        job_id = await backend.submit_run(
            str(run_dir), config, module="lqh.infer.eval_hf",
            donate_hf_token=donate_hf,
        )
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail("upstream", f"Error submitting eval_hf job: {e}")

    if on_bg_started is not None:
        on_bg_started(run_name, "eval", run_name, "cloud")

    from lqh.project_log import append_event

    append_event(
        project_dir,
        "eval_hf_started",
        f"Submitted eval_hf for {repo}@{revision} (run {run_name}, job {job_id})",
        run_name=run_name,
        run_type="eval_hf",
        base_model=repo,
        remote="cloud",
    )

    # The consent prompt estimated the cap against the default GPU; the
    # backend planner may have upsized for a large model. Read back the
    # ACTUAL selected resource so the confirmation shows the real hard
    # cap, not the estimate. Best-effort: on any error fall back to the
    # requested timeout alone.
    compute_line = f"  Timeout: {timeout_minutes} min (hard compute-cost cap)\n"
    try:
        resource = (await backend.job_snapshot(job_id)).get("resource") or {}
        gpu = resource.get("gpu_type") or "?"
        actual_timeout = int(resource.get("timeout_minutes") or timeout_minutes)
        cap_micros = resource.get("worst_case_cost_billed_micros")
        cap_part = (
            f", hard cap ≈ ${float(cap_micros) / 1e6:.2f}"
            if isinstance(cap_micros, (int, float)) and cap_micros > 0 else ""
        )
        compute_line = (
            f"  Compute: {gpu} GPU, {actual_timeout} min timeout{cap_part}\n"
        )
    except Exception:  # noqa: BLE001 — display nicety, never fail the submit
        pass

    return ToolResult(
        content=(
            f"🧪 HF eval submitted\n"
            f"  Run:     {run_name}\n"
            f"  Repo:    {repo}@{revision}\n"
            f"  Method:  {training_method}"
            + (f" (base {base_model})" if training_method == 'lora' else "")
            + f"\n"
            f"  Judge:   judge:{judge_size}\n"
            + compute_line
            + f"  Job ID:  {job_id}\n\n"
            f"Use training_status to monitor; eval_result.json lands "
            f"under runs/{run_name}/ when done."
            + _submit_advisories(backend)
        ),
        workflow_launched=True,
    )


async def _start_local_eval_remote(
    project_dir: Path,
    model_path: str,
    dataset: str | list[Any],
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
    from lqh.remote.compute import ssh_remote_name
    from lqh.remote.config import get_remote
    from lqh.remote.ssh_direct import SSHDirectBackend

    # Normalise the remote arg: ``ssh:toka`` → ``toka``. Without this
    # the lookup keys on the literal "ssh:toka" string and fails.
    ssh_name = ssh_remote_name(remote_name) or remote_name
    remote_config = get_remote(project_dir, ssh_name)
    if remote_config is None:
        return ToolResult.fail("config", f"Error: remote '{ssh_name}' not found.")
    remote_name = ssh_name

    if remote_config.type == "ssh_slurm":
        return ToolResult.fail("config", "Error: SSH+Slurm backend is not yet implemented.")

    # Validate eval dataset source(s) — one path or a list of held-out
    # sources, scored separately and macro-averaged (same as the local path).
    eval_sources, eval_resolved, ds_err = _resolve_training_sources(
        project_dir, dataset, kind="dataset", allow_repeat=False
    )
    if ds_err:
        return ToolResult.fail("validation", ds_err)

    scorer_resolved = _validate_path(project_dir, scorer)
    if not scorer_resolved.exists():
        return ToolResult.fail("not_found", f"Error: scorer not found at {scorer}")

    try:
        system_prompt, schema_dict = _resolve_eval_extras(
            project_dir,
            system_prompt_path=system_prompt_path,
            response_format_path=response_format_path,
        )
    except FileNotFoundError as e:
        return ToolResult.fail("not_found", f"Error: {e}")

    run_name, claim_err = _claim_run_name(project_dir, run_name or None, "remote_eval")
    if claim_err:
        return ToolResult.fail("conflict", f"Error: {claim_err}")

    run_dir = project_dir / "runs" / run_name
    config: dict[str, Any] = {
        "type": "infer",
        "spec_sha256": _eval_spec_hash(project_dir),
        "base_model": model_path,
        "dataset": _sources_to_config(eval_sources),
        "scorer": scorer,
        "max_new_tokens": max_new_tokens,
        "num_samples": sum((_parquet_metadata(path)[0] or 0) for path in eval_resolved),
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
        return ToolResult.fail("upstream", f"Error launching remote inference: {e}")

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
        ),
        workflow_launched=True,
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
        return ToolResult.fail("validation", f"Error: {e}")

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
        return ToolResult.fail(
            "not_found",
            (
                f"Error: machine '{name}' not found globally. "
                f"Use remote_add to create it first."
            ),
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
            return ToolResult.fail(
                "upstream",
                f"Error resolving '{remote_root}' on {machine.hostname}: {e}",
            )
        if rc != 0:
            return ToolResult.fail(
                "upstream",
                (
                    f"Error resolving '{remote_root}' on {machine.hostname}: "
                    f"{stderr.strip() or 'ssh exited with code ' + str(rc)}"
                ),
            )
        resolved = stdout.strip()
        if not resolved or not resolved.startswith("/"):
            return ToolResult.fail(
                "validation",
                (
                    f"Error: could not resolve '{remote_root}' to an absolute path "
                    f"on {machine.hostname} (got: {resolved!r})"
                ),
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
        return ToolResult.fail("validation", f"Error: {e}")

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
        return ToolResult.fail("config", f"Error: remote '{name}' not bound to this project.")

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
        return ToolResult.fail("not_found", f"Error: machine '{name}' not found globally.")

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
        return ToolResult.fail("config", f"Error: remote '{name}' not found.")

    if remote_config.type == "ssh_slurm":
        return ToolResult.fail("config", "Error: SSH+Slurm backend is not yet implemented.")

    # Check SSH connectivity first
    reachable = await ssh_check(remote_config.hostname)
    if not reachable:
        return ToolResult.fail(
            "upstream",
            (
                f"Error: cannot reach {remote_config.hostname} via SSH. "
                f"Check that SSH public key auth is configured and the host "
                f"is reachable."
            ),
            retryable=True,
        )

    backend = SSHDirectBackend(remote_config, project_dir)
    try:
        log = await backend.setup()
    except Exception as e:
        return ToolResult.fail("upstream", f"Error during setup: {e}")

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
        return ToolResult.fail("not_found", f"Error: machine '{name}' not found globally.")

    hostname = machine.hostname

    # Check SSH connectivity first
    reachable = await ssh_check(hostname)
    if not reachable:
        return ToolResult.fail(
            "upstream",
            (
                f"❌ Cannot reach **{name}** ({hostname}) via SSH.\n"
                f"Check that SSH public key auth is configured and the host "
                f"is reachable."
            ),
            retryable=True,
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
    overwrite: bool = False,
    _overwrite_consent: bool = False,
    **kwargs: Any,
) -> ToolResult:
    """Score a user-brought dataset and emit a filtered subset."""
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_data_filter

    # output_dataset becomes a path component under datasets/ — require a
    # plain directory name so it can't escape the project layout.
    if (
        not output_dataset
        or output_dataset in (".", "..")
        or "/" in output_dataset
        or "\\" in output_dataset
    ):
        return ToolResult.fail("validation", (
            f"Error: output_dataset must be a plain name (no path "
            f"separators), got {output_dataset!r}"
        ))

    input_abs = _validate_path(project_dir, input_path)
    scorer_abs = _validate_path(project_dir, scorer_path)
    if not input_abs.exists():
        return ToolResult.fail("not_found", f"Error: input '{input_path}' does not exist")
    if not scorer_abs.exists():
        return ToolResult.fail("not_found", f"Error: scorer '{scorer_path}' does not exist")

    # Immutable-by-default outputs (PERSISTENCY_PLAN.md R5). Fast
    # read-only refusal here; the atomic claim happens inside the
    # release-protected block below so an auth/config failure can never
    # leak a held claim.
    from lqh.dataset_guard import claim_output, overwrite_refusal, release_output

    early_refusal = overwrite_refusal(
        project_dir, output_dataset, overwrite=overwrite
    )
    if early_refusal:
        return ToolResult.fail("conflict", f"Error: {early_refusal}")

    output_dir = project_dir / "datasets" / output_dataset
    if overwrite and (output_dir / "data.parquet").exists() and not _overwrite_consent:
        return ToolResult(
            content="OVERWRITE_CONFIRMATION_REQUIRED",
            requires_user_input=True,
            question=(
                f"The agent wants to OVERWRITE datasets/{output_dataset}/ "
                "with a filtered dataset — the existing data.parquet will be "
                "destroyed. Allow?"
            ),
            options=[
                "Yes, destroy and replace this dataset",
                "No, keep the existing data",
            ],
        )

    config = load_config()
    token = require_token()
    # max_retries=0 — see handle_run_scoring: the SDK's replay layer would
    # sit under scoring's own ladder and multiply every timeout.
    client = create_client(token, config.api_base_url, max_retries=0)

    from lqh.progress import ProgressReporter

    reporter = ProgressReporter(
        task_kind="data_filter",
        label="Data filtering",
        callback=kwargs.get("on_pipeline_progress"),
        legacy_callback=bool(kwargs.get("legacy_progress_callback", True)),
    )
    reporter.update(
        phase="setup", phase_label="preparing filter",
        overall_fraction=0, unit="samples", force=True,
    )

    def on_progress(completed: int, total: int) -> None:
        reporter.update(
            phase="filtering", phase_label="filtering",
            completed=completed, total=total, unit="samples",
            overall_fraction=completed / max(total, 1),
            concurrency=min(100, total), force=completed == total,
        )

    # Captured pre-run: the manifest must reflect the spec the filter ran
    # under, not one edited while it ran.
    from lqh.project_meta import compute_spec_sha256 as _spec_hash

    pre_run_spec_sha256 = _spec_hash(project_dir)

    succeeded = False
    claimed = False
    try:
        refusal = claim_output(project_dir, output_dataset, overwrite=overwrite)
        if refusal:
            return ToolResult.fail("conflict", f"Error: {refusal}")
        claimed = True
        result = await run_data_filter(
            input_path=input_abs,
            scorer_path=scorer_abs,
            output_dataset_dir=output_dir,
            client=client,
            threshold=threshold,
            model_size=model_size,
            on_progress=on_progress,
        )
        succeeded = True
    except Exception as exc:
        return ToolResult.fail("runtime", f"❌ run_data_filter failed: {type(exc).__name__}: {exc}")
    finally:
        if claimed:
            release_output(project_dir, output_dataset)
        if succeeded:
            # A run where nothing could be judged has filtered nothing, and
            # the tool returns a failure below — the progress surface must not
            # be showing a green "ready" at 100% next to that. Same shape as
            # handle_run_scoring's "no valid scores".
            vetted = result.scored > 0
            reporter.update(
                phase="completed",
                phase_label=(
                    "filtered dataset ready" if vetted else "nothing could be judged"
                ),
                completed=result.total, total=result.total, unit="samples",
                overall_fraction=1.0, result_ready=vetted, force=True,
            )
        on_done = kwargs.get("on_pipeline_done")
        if on_done:
            on_done()

    # Finalization manifest: a filtered output is a DERIVATIVE (subset) of
    # its input — recorded as derived_from, not as a supplement. Unknown
    # input provenance stays unknown (purpose defaults to "unspecified").
    from lqh.manifest import inherit_purpose, write_dataset_manifest

    manifest_written = write_dataset_manifest(
        project_dir,
        output_dir,
        purpose=inherit_purpose(input_abs.parent),
        rows=result.kept,
        spec_sha256=pre_run_spec_sha256,
        source_paths=[input_path],
        scorer_path=scorer_path,
        threshold=threshold,
        derived_from=input_path,
        # Provenance has to carry how much of this output was never actually
        # vetted — a manifest naming a scorer and a threshold otherwise reads
        # as "every row here cleared that bar".
        kept_unjudged=result.kept_unjudged,
    ) is not None
    manifest_warning = (
        "" if manifest_written else
        "\n  ⚠️ Provenance manifest could not be written (check disk/logs)."
    )

    distribution = _format_score_distribution(output_dir / "scores.parquet")
    from lqh.scoring import failure_warning

    # Fail open keeps rows the judge could not score — but a run where NOTHING
    # was scored has filtered nothing. Reporting that as a success hands back
    # a dataset byte-identical to the input, with a manifest naming a scorer
    # and a threshold that were never actually applied, which is exactly the
    # artifact that gets mistaken for vetted data later.
    if result.total > 0 and result.scored == 0:
        return ToolResult.fail(
            "runtime",
            f"❌ Filter did not vet anything: all {result.total} samples failed "
            f"to score (judge: {model_size}).\n"
            f"  datasets/{output_dataset}/ now holds an unfiltered copy of the "
            "input — do NOT treat it as filtered data. Its manifest names the "
            "scorer and threshold but records kept_unjudged="
            f"{result.kept_unjudged}, i.e. nothing was actually vetted.\n"
            "  Check the judge model, the scorer file, and API access, then "
            f"re-run with overwrite=true — datasets/{output_dataset}/ exists "
            "now, so a plain re-run will hit overwrite protection.",
        )

    warning = failure_warning(result.failed, result.total)
    return ToolResult(
        content=(
            f"✅ Filtered dataset written\n"
            f"  Input:     {input_path} ({result.total} rows)\n"
            f"  Threshold: {threshold} (judge: {model_size})\n"
            f"  Kept:      {result.kept} / {result.total} ({result.kept / max(result.total, 1):.0%})\n"
            f"  Dropped:   {result.dropped}\n"
            f"  Failed:    {result.failed}"
            + (
                " (kept unjudged — the judge could not score them, so they were "
                "NOT dropped; re-run the filter to vet them)"
                if result.kept_unjudged else ""
            )
            + f"\n  Mean score: {result.mean_score:.2f}\n"
            + (f"{warning}\n" if warning else "")
            + (f"\n{distribution}\n" if distribution else "")
            + f"  Output:    datasets/{output_dataset}/ (data.parquet, scores.parquet, summary.json)"
            + manifest_warning
        )
    )


async def handle_exit_auto_mode(
    *, status: str, reason: str,
    summary: str | None = None,
    artifacts: list[Any] | None = None,
    metrics: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Terminate auto mode. Only meaningful when the agent runs in auto mode.

    ``summary``/``artifacts``/``metrics`` are optional structured-exit
    fields for headless runs (CLI_PLAN §4.6). They are model CLAIMS — the
    run driver validates artifact paths and labels metrics "reported"
    unless corroborated; its own deterministic ledger is authoritative.
    """
    status_norm = (status or "").strip().lower()
    if status_norm not in ("success", "failure"):
        return ToolResult(
            content=(
                f"Error: status must be 'success' or 'failure', got {status!r}. "
                "Call exit_auto_mode again with a valid status."
            ),
        )
    details: dict[str, Any] = {}
    if summary:
        details["summary"] = str(summary)
    if isinstance(artifacts, list):
        details["artifacts"] = artifacts
    if isinstance(metrics, dict):
        details["metrics"] = metrics
    return ToolResult(
        content=f"Exiting auto mode: {status_norm} — {reason}",
        exit_auto_mode=True,
        auto_status=status_norm,
        auto_reason=reason,
        details=details or None,
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
    "pull": handle_pull,
    "push": handle_push,
    "gguf_convert": handle_gguf_convert,
    "artifacts": handle_artifacts,
    "push_to_production": handle_push_to_production,
    "list_deployments": handle_list_deployments,
    "get_deployment": handle_get_deployment,
    "stop_deployment": handle_stop_deployment,
    "restart_deployment": handle_restart_deployment,
    "create_inference_key": handle_create_inference_key,
    "list_inference_keys": handle_list_inference_keys,
    "revoke_inference_key": handle_revoke_inference_key,
    "start_training": handle_start_training,
    "training_status": handle_training_status,
    "stop_training": handle_stop_training,
    "start_local_eval": handle_start_local_eval,
    "eval_hf_model": handle_eval_hf_model,
    "remote_list": handle_remote_list,
    "remote_add": handle_remote_add,
    "remote_bind": handle_remote_bind,
    "remote_remove": handle_remote_remove,
    "remote_remove_machine": handle_remote_remove_machine,
    "remote_setup": handle_remote_setup,
    "remote_status": handle_remote_status,
    "compute_set": handle_compute_set,
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
        return ToolResult.fail("validation", f"Error: unknown tool '{tool_name}'")

    # Underscore-prefixed keys are loop-internal signals (e.g. the
    # consent flags a permission grant sets). They may only arrive via
    # extra_kwargs from the agent loop — never from model-controlled
    # arguments, where they could bypass a permission gate.
    arguments = {k: v for k, v in arguments.items() if not k.startswith("_")}

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
