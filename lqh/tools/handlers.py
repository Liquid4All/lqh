"""Tool execution dispatch and handlers for lqh agent."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from lqh.skills import list_available_skills, load_skill_content


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
    _script_consent: bool = False,
    _cloud_consent: bool = False,
    _overwrite_consent: bool = False,
    **kwargs: Any,
) -> ToolResult:
    """Execute a data generation pipeline. Requires user permission.

    ``execution="cloud"`` submits a background CPU cloud job instead of
    running in-process (CLOUD_OFFLOAD_PLAN.md §2), gated on a prior
    successful local run of the same pipeline version plus user consent.
    ``_script_consent`` / ``_cloud_consent`` are internal: the agent loop
    sets them when re-invoking after the user granted the corresponding
    prompt, so a one-time grant works without persisting anything.
    """
    from lqh.tools.permissions import check_permission

    if execution not in ("local", "cloud"):
        return ToolResult(
            content=f"Error: execution must be 'local' or 'cloud', got {execution!r}"
        )
    if num_samples <= 0:
        return ToolResult(
            content=f"Error: num_samples must be positive, got {num_samples}"
        )
    if samples_per_item <= 0:
        return ToolResult(
            content=f"Error: samples_per_item must be positive, got {samples_per_item}"
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
        return ToolResult(
            content=(
                f"Error: output_dataset must be a plain name (no path separators), "
                f"got {output_dataset!r}"
            )
        )

    # Fast read-only immutability refusal BEFORE the permission prompt so
    # the agent gets immediate feedback; the atomic claim below re-checks
    # under the cross-process lock right before work starts.
    from lqh.dataset_guard import overwrite_refusal

    early_refusal = overwrite_refusal(project_dir, output_dataset, overwrite=overwrite)
    if early_refusal:
        return ToolResult(content=f"Error: {early_refusal}")

    target = _validate_path(project_dir, script_path)
    if not target.exists():
        return ToolResult(content=f"Error: script '{script_path}' does not exist")
    # Pipelines must sit directly under data_gen/: the engine derives the
    # project root as script_path.parent.parent, so a script anywhere else
    # resolves source() against the wrong directory (and the cloud bundle
    # layout + import pre-scan assume the same location).
    rel_parts = target.relative_to(project_dir.resolve()).parts
    if len(rel_parts) != 2 or rel_parts[0] != "data_gen" or not rel_parts[1].endswith(".py"):
        return ToolResult(
            content=(
                f"Error: pipeline scripts must be .py files directly under data_gen/ "
                f"(got '{script_path}'). Move the script to data_gen/<name>.py and retry."
            )
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
                return ToolResult(
                    content=(
                        f"Error: Pipeline has incorrect import: `{display}`\n"
                        f"Fix: use `from lqh.pipeline import Pipeline, ChatMLMessage, Conversation`\n"
                        f"All pipeline imports must come from `lqh.pipeline`, not `data_gen` or `pipeline`."
                    )
                )
    except OSError:
        pass

    # Check if we already have permission (script-execution domain;
    # applies to both execution targets — cloud runs the same script).
    if not _script_consent and not check_permission(project_dir, script_path):
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

    # Expensive outputs are immutable by default (PERSISTENCY_PLAN.md R5).
    # claim_output is an atomic check-and-reserve under a cross-process
    # lock: it refuses when finalized data exists (a leftover
    # data.partial.jsonl does NOT bypass this — resume only applies while
    # no data.parquet exists) and when another live process is currently
    # generating into the same name.
    from lqh.dataset_guard import claim_output, release_output

    existing = project_dir / "datasets" / output_dataset / "data.parquet"
    if overwrite and existing.exists() and not _overwrite_consent:
        # overwrite=true from the model is a REQUEST, not consent —
        # destroying data needs an explicit human yes (the agent loop
        # relays this prompt and re-invokes with the consent flag).
        return ToolResult(
            content="OVERWRITE_CONFIRMATION_REQUIRED",
            requires_user_input=True,
            question=(
                f"The agent wants to OVERWRITE datasets/{output_dataset}/ — "
                "the existing data.parquet (and its scores/summaries) will "
                "be destroyed and regenerated. Data generation is expensive "
                "and this cannot be undone. Allow?"
            ),
            options=[
                "Yes, destroy and regenerate this dataset",
                "No, keep the existing data",
            ],
        )

    refusal = claim_output(project_dir, output_dataset, overwrite=overwrite)
    if refusal:
        return ToolResult(content=f"Error: {refusal}")

    if overwrite and existing.exists():
        # Confirmed overwrite: also drop co-located artifacts describing
        # the OLD contents, so summary can't report stale scores/filters.
        for stale in ("scores.parquet", "summary.json", ".lqh_source.json"):
            try:
                (existing.parent / stale).unlink(missing_ok=True)
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
                consent=_cloud_consent,
                on_bg_started=kwargs.get("on_background_task_started"),
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


async def _fetch_data_gen_rate_usd() -> float | None:
    """Billed data_gen $/hr from GET /v1/cloud/pricing; None if unreachable.

    Fetched live so the consent prompt can't drift from operator
    overrides of the rate or margin envvars; callers fall back to the
    defaults with an explicit "at default rates" caveat.
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
            micros = resp.json().get("data_gen_cpu_rate_billed_micros_per_hour")
            if isinstance(micros, (int, float)) and micros > 0:
                return float(micros) / 1e6
            return None
    except Exception:
        return None


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
    consent: bool,
    on_bg_started: Callable[[str, str, str, str | None], None] | None = None,
) -> ToolResult:
    """Submit the pipeline as a background cloud CPU job.

    Two gates, in order (both after the script-execution permission
    handled by the caller):

    1. Correctness gate — a successful LOCAL run of this exact pipeline
       version (content hash) must be on record. Handler-enforced, not
       prompt-trusted: an agent edit to the file re-arms it.
    2. Consent gate — the user approves the submit (sample count + cost
       estimate) unless the project carries a standing grant.
    """
    from lqh.data_gen_validation import check_validation
    from lqh.tools.permissions import check_cloud_data_gen_permission

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
            return ToolResult(content=f"Error: validation_instructions: {e}")
        if not val_resolved.exists():
            return ToolResult(
                content=f"Error: validation_instructions file "
                        f"'{validation_instructions}' does not exist"
            )
        validation_instructions = val_resolved.relative_to(
            project_dir.resolve()
        ).as_posix()

    record = check_validation(project_dir, target)
    if record is None:
        return ToolResult(
            content=(
                "VALIDATION_REQUIRED: cloud execution is locked until this exact "
                "pipeline version and its recorded local inputs have succeeded locally.\n"
                f"Run `run_data_gen_pipeline` with execution='local' first — a draft "
                f"(num_samples=3, purpose='smoke') to check correctness, then an "
                f"inspection batch (num_samples≈20, purpose='inspection') to review "
                f"quality — and retry execution='cloud' afterwards.\n"
                f"Note: any edit to {script_path} re-arms this gate."
            )
        )

    # The gate hashes code, not data — a recorded seed input deleted or
    # moved since validation would silently vanish from the bundle
    # (resolve_manifest skips missing paths) and only fail in the paid
    # sandbox. Catch it here instead.
    missing_sources = [s for s in record.source_paths if not (project_dir / s).exists()]
    if missing_sources:
        return ToolResult(
            content=(
                "VALIDATION_REQUIRED: recorded source inputs no longer exist: "
                + ", ".join(missing_sources[:5])
                + (" …" if len(missing_sources) > 5 else "")
                + "\nRestore them or re-run the pipeline locally (which re-records "
                "its inputs), then retry execution='cloud'."
            )
        )

    total_calls = num_samples * max(1, samples_per_item)
    if not consent and not check_cloud_data_gen_permission(project_dir):
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
            # job — this is a consent prompt, not a changelog.
            if os.environ.get("HF_TOKEN"):
                hf_line = (
                    "  HF:      pipeline streams a Hugging Face dataset — your "
                    "local HF_TOKEN is sent with this job (not persisted) and is "
                    "available to the trusted pipeline Python process\n"
                )
            else:
                hf_line = (
                    "  HF:      pipeline streams a Hugging Face dataset — no "
                    "local HF_TOKEN found; private datasets need one (or a "
                    "stored account token) or the job will fail. Any stored token "
                    "used is available to the trusted pipeline Python process\n"
                )
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
    if purpose not in {"smoke", "inspection", "validation", "training", "failures", "imported", "unspecified"}:
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
        # The validated local run observed lqh.sources.hf_dataset — donate
        # the local HF_TOKEN (if the env carries one) so a PRIVATE dataset
        # that worked locally also works in the sandbox. Without this the
        # sandbox only gets the account-stored token, and a user relying
        # on an env token validates locally, then fails after paying for
        # the launch. submit_run reads this flag for the donate path.
        hf_token_configured=record.needs_hf,
    )
    backend = CloudBackend(cfg, project_dir)
    try:
        job_id = await backend.submit_run(
            str(run_dir), config,
            module="lqh.remote.data_gen",
            telemetry_workflow_id=workflow_id,
        )
    except CloudError as e:
        if telemetry:
            await telemetry.run_deferred(telemetry.event, "data_generation_failed", {
                "workflow_kind": "data_generation", "purpose": purpose,
                "execution_target": "cloud", "outcome": "failed",
                "requested_count": num_samples,
            }, workflow_id)
        return ToolResult(content=f"Error submitting cloud data-gen job: {e}")
    except Exception as e:
        if telemetry:
            await telemetry.run_deferred(telemetry.event, "data_generation_failed", {
                "workflow_kind": "data_generation", "purpose": purpose,
                "execution_target": "cloud", "outcome": "failed",
                "requested_count": num_samples,
            }, workflow_id)
        return ToolResult(
            content=f"Error submitting cloud data-gen job: {type(e).__name__}: {e}"
        )

    # Durable finalization marker: the TUI watcher downloads the dataset
    # and notifies when it sees this file on a terminal job — including
    # after a TUI restart where the running→terminal transition was
    # never observed. Also carries the workflow id so the completion
    # telemetry closes the workflow opened above.
    # Provenance captured AT SUBMISSION: the job runs the submitted
    # pipeline against the submitted spec, regardless of local edits made
    # while it executes.
    from lqh.project_log import file_hash_prefix as _hash_prefix
    from lqh.project_meta import compute_spec_sha256 as _spec_hash

    marker = {
        "workflow_id": workflow_id,
        "output_dataset": output_dataset,
        "purpose": purpose,
        "script_path": script_path,
        "pipeline_hash": _hash_prefix(project_dir / script_path, n=12),
        "spec_sha256": _spec_hash(project_dir),
        "job_id": job_id,
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
    if purpose not in {"smoke", "inspection", "validation", "training", "failures", "imported", "unspecified"}:
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

        def on_progress(completed: int, total: int) -> None:
            reporter.update(
                phase="generation", phase_label="generating",
                completed=completed, total=total, unit="samples",
                overall_fraction=completed / max(total, 1),
                concurrency=concurrency,
            )

        import sys as _sys

        # Evict project-local modules cached by earlier runs in this
        # session: a cached module makes its import a no-op, so it would
        # be invisible to the newly-loaded detection below and a
        # non-self-contained pipeline could validate for cloud execution
        # (where that dependency won't exist). Eviction is safe — the
        # next import re-executes the module fresh.
        project_resolved = project_dir.resolve()
        for _name, _mod in list(_sys.modules.items()):
            _mod_file = getattr(_mod, "__file__", None)
            if not _mod_file:
                continue
            try:
                Path(_mod_file).resolve().relative_to(project_resolved)
            except (OSError, ValueError):
                continue
            del _sys.modules[_name]

        modules_before = set(_sys.modules)
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
        # Detect project-local imports the static pre-scan can't (e.g.
        # `import data_gen.helper` via a package path, or any other
        # module resolved from inside the project). Such a pipeline runs
        # locally but its dependency will not exist in the cloud bundle,
        # so it must not validate for cloud execution.
        project_imports: list[str] = []
        target_resolved = target.resolve()
        for name in set(_sys.modules) - modules_before:
            mod = _sys.modules.get(name)
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                continue
            try:
                mod_path = Path(mod_file).resolve()
            except OSError:
                continue
            if mod_path == target_resolved:
                continue  # the pipeline module itself
            try:
                mod_path.relative_to(project_resolved)
            except ValueError:
                continue
            project_imports.append(name)

        if result.succeeded <= 0:
            if telemetry_started and telemetry and telemetry.cached_consent_active(consent_epoch):
                await telemetry.run_deferred(telemetry.event, "data_generation_failed", {
                    "workflow_kind":"data_generation", "purpose":purpose,
                    "execution_target":"local", "outcome":"failed",
                    "wall_duration_ms":int((time.monotonic()-started_mono)*1000),
                    "active_duration_ms":int(max(telemetry.state_snapshot()[2]-started_active, 0)*1000),
                    "requested_count":num_samples, "succeeded_count":0,
                    "failed_count":result.failed, "sample_count":result.total,
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
                error="no successful samples",
            )
            reporter.update(
                phase="failed", phase_label="no samples generated",
                completed=result.total, total=result.total, unit="samples",
                overall_fraction=1.0, force=True,
            )
            return ToolResult(content=(
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
                "failed_count":result.failed,"sample_count":result.total,
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
        )
        reporter.update(
            phase="completed", phase_label="dataset ready",
            completed=result.total, total=result.total, unit="samples",
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
        # (the cloud bundle manifest) — UNLESS the run imported
        # project-local modules (they won't exist in the bundle) or
        # resumed from a partial file (its source recording covers only
        # this process, so the manifest would be incomplete).
        # Best-effort — never fail the run over gate bookkeeping.
        validation_note = ""
        if project_imports:
            validation_note = (
                "\n⚠️ Not validated for cloud execution: the pipeline imported "
                f"project-local modules ({', '.join(sorted(project_imports)[:5])}) — "
                "cloud pipelines must be self-contained single files."
            )
        elif result.resumed_samples > 0:
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
        return ToolResult(content=f"Error: scope must be 'global' or 'project', got {scope!r}")

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
        return ToolResult(content=(
            f"Error: value must be 'cloud', 'local', or 'ssh:<remote_name>', "
            f"got {value!r}."
        ))

    if scope == "global":
        save_global_default(value)
        return ToolResult(content=f"✅ Default compute set to '{value}' (global).")
    save_project_default(project_dir, value)
    return ToolResult(content=f"✅ Default compute set to '{value}' for this project.")


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
    export_path: str | None = None,
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

    export_note = ""
    if export_path:
        # Durable, untruncated export (the feedback/ workflow): full
        # messages plus the origin of each case — which eval run and
        # which model produced it. Confined to feedback/ and never
        # overwrites: an errant path must not be able to replace
        # SPEC.md, NOTES.md, or an artifact.
        normalized = export_path.replace("\\", "/")
        if not normalized.startswith("feedback/") or ".." in normalized.split("/"):
            return ToolResult(content=(
                f"Error: export_path must live under feedback/ "
                f"(got {export_path!r}) — e.g. 'feedback/eval_failures_v1.jsonl'."
            ))
        export_abs = _validate_path(project_dir, export_path)
        if export_abs.exists():
            return ToolResult(content=(
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
                        "threshold": threshold,
                        "exported_at": exported_at,
                        "scoring_error": False,
                    }, ensure_ascii=False) + "\n")
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
            export_note = (
                f"\n💾 Exported {len(failures)} failure(s)"
                + (f" + {len(scoring_errors)} scoring error(s)" if scoring_errors else "")
                + f" (full, untruncated) to {export_path}"
            )
        except OSError as exc:
            export_note = f"\n⚠️ Export to {export_path} failed: {exc}"

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
    lines.append("  orchestration                  — frontier agent model with tool calling")

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

            # Liquid checkpoints can no longer be evaluated through the API:
            # the router.liquid.ai inference API has been retired (see MODELS.md).
            # Redirect to the HuggingFace inference path. Pool/baseline names
            # (small/medium/large/orchestration) still run via the API here.
            from lqh.models import is_liquid_model_name

            if is_liquid_model_name(inference_model):
                return ToolResult(
                    content=(
                        f"Error: Liquid model '{inference_model}' cannot be evaluated via "
                        "run_scoring mode='model_eval' — the router.liquid.ai API has been "
                        "retired (see MODELS.md). To evaluate a Liquid checkpoint, use the "
                        "HuggingFace inference path instead:\n"
                        "  - eval_hf_model  — cloud eval of a HuggingFace repo id / revision\n"
                        "  - start_local_eval — local or SSH-remote GPU eval of a checkpoint dir\n"
                        "run_scoring mode='model_eval' remains available for non-Liquid "
                        "baselines (small / medium / large / orchestration)."
                    )
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
            from lqh.project_meta import compute_spec_sha256 as _spec_hash2

            config_data: dict[str, Any] = {
                "eval_dataset": dataset,
                "scorer": scorer,
                "mode": mode,
                "spec_sha256": _spec_hash2(project_dir),
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
            # and summary.json just written above).
            from lqh.manifest import write_run_manifest

            write_run_manifest(project_dir, output_dir, state="completed")
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
        return ToolResult(content=f"Error: {e}")

    if loc.scheme == "hf":
        return await handle_hf_pull(
            project_dir, repo_id=loc.value, local_path=dest, revision=loc.revision,
        )
    if loc.scheme == "lqh":
        return await _pull_lqh_artifact(project_dir, loc.value, dest)
    return ToolResult(
        content=(
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
        return ToolResult(content=f"Error: {e}")

    store = BackendArtifactStore()
    try:
        await store.download(artifact_id, target)
    except ArtifactError as e:
        return ToolResult(content=f"Error downloading lqh:{artifact_id}: {e}")
    except Exception as e:  # noqa: BLE001 - surface any client error to the agent
        return ToolResult(content=f"Error downloading lqh:{artifact_id}: {e}")

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
        return ToolResult(content=f"Error: {e}")

    if dst.scheme != "hf":
        return ToolResult(
            content=f"Error: push destination must be 'hf:owner/repo'; got {dest!r}"
        )

    if src.scheme == "local":
        return await handle_hf_push(
            project_dir, local_path=src.value, repo_id=dst.value, private=private,
        )
    if src.scheme == "lqh":
        return await _push_lqh_to_hf(project_dir, src.value, dst.value, private)
    return ToolResult(
        content=(
            f"Error: push source must be a local path or 'lqh:<artifact_id>'; "
            f"got {source!r}"
        ),
    )


async def _push_lqh_to_hf(
    project_dir: Path, artifact_id: str, target_repo: str, private: bool,
) -> ToolResult:
    """Submit a CPU-only transfer job that copies an R2 artifact to HF."""
    from lqh.remote.transfer import submit_transfer

    try:
        job_id = await submit_transfer(
            project_id=project_dir.name,
            source_artifact_id=artifact_id,
            target_hf_repo=target_repo,
            private=private,
        )
    except Exception as e:  # noqa: BLE001 - surface clearly to the agent
        return ToolResult(content=f"Error starting transfer of lqh:{artifact_id}: {e}")
    return ToolResult(
        content=(
            f"🚚 Transferring lqh:{artifact_id} → hf:{target_repo} via a CPU sandbox "
            f"(job {job_id}). The checkpoint is uploaded from R2 directly; check "
            "training_status or the artifact's hf_repo once it completes. Requires a "
            "stored HF token (run /hf_login) since the upload happens in the cloud."
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
        return ToolResult(content="Error: quant_types must list at least one type.")

    try:
        job_id = await submit_gguf(
            project_id=project_dir.name,
            source_artifact_id=artifact_id,
            quant_types=quant_types,
            target_hf_repo=target_hf_repo,
            private=private,
            include_f16=include_f16,
            base_model=base_model,
            artifact_format=artifact_format,
        )
    except Exception as e:  # noqa: BLE001 - surface clearly to the agent
        return ToolResult(content=f"Error starting gguf conversion of lqh:{artifact_id}: {e}")

    quants = ", ".join(quant_types)
    push = f" and pushing to hf:{target_hf_repo}" if target_hf_repo else ""
    return ToolResult(
        content=(
            f"🧱 Converting lqh:{artifact_id} → GGUF ({quants}){push} via a CPU sandbox "
            f"(job {job_id}). Each quant is converted from R2 directly and smoke-tested; "
            "the produced .gguf files register as new artifacts (kind 'gguf'). Check "
            "training_status for progress, then 'artifacts' (action=list) to download them."
            + (" HF push requires a stored token (run /hf_login)." if target_hf_repo else "")
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
                project_dir.name, kind=kind, limit=limit,
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
            return ToolResult(content=f"Error: action '{act}' requires artifact_id")

        if act == "pin":
            await store.pin(artifact_id)
            return ToolResult(content=f"📌 Pinned {artifact_id} — exempt from auto-expiry.")
        if act == "unpin":
            await store.unpin(artifact_id)
            return ToolResult(content=f"Unpinned {artifact_id} — per-kind expiry re-armed.")
        if act == "delete":
            await store.delete(artifact_id)
            return ToolResult(content=f"Deleted {artifact_id} (R2 bytes purged on the next retention tick).")
        return ToolResult(content=f"Error: unknown action '{act}' (use list/pin/unpin/delete)")
    except ArtifactError as e:
        return ToolResult(content=f"Error: {e}")
    except Exception as e:  # noqa: BLE001
        return ToolResult(content=f"Error: {e}")


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
    body: dict[str, Any] = {
        "name": name,
        "artifact_id": artifact_id,
        "tier": tier,
        "project_id": project_id or project_dir.name,
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
        return ToolResult(content=f"Error: {e}")

    if status == 402:
        return ToolResult(
            content=(
                "❌ Out of credits — the deployment was not created. The org has "
                "insufficient credits to run a GPU deployment; top up and retry."
            )
        )
    if status == 409:
        return ToolResult(
            content=(
                f"❌ Deployment name '{name}' is already taken. Pick a different "
                "name (list_deployments shows the existing ones) and retry."
            )
        )
    if status not in (200, 201):
        return ToolResult(
            content=f"Error creating deployment: {_api_error_message(status, data)}"
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
        return ToolResult(content=f"Error: {e}")
    if status != 200:
        return ToolResult(
            content=f"Error listing deployments: {_api_error_message(status, data)}"
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
        return ToolResult(content=f"Error: {e}")
    if status != 200:
        return ToolResult(
            content=f"Error fetching deployment: {_api_error_message(status, dep)}"
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
        return ToolResult(content=f"Error: {e}")
    if status != 200:
        return ToolResult(
            content=f"Error on {action}: {_api_error_message(status, dep)}"
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
        return ToolResult(content=f"Error: {e}")
    if status == 403:
        return ToolResult(
            content=(
                "❌ The org has reached its inference-key cap. Revoke an unused "
                "key (list_inference_keys / revoke_inference_key) and retry."
            )
        )
    if status not in (200, 201):
        return ToolResult(
            content=f"Error creating inference key: {_api_error_message(status, data)}"
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
        return ToolResult(content=f"Error: {e}")
    if status != 200:
        return ToolResult(
            content=f"Error listing inference keys: {_api_error_message(status, data)}"
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
        return ToolResult(content=f"Error: {e}")
    if status != 200:
        return ToolResult(
            content=f"Error revoking key: {_api_error_message(status, data)}"
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
    token = os.environ.get("HF_TOKEN")  # optional for public repos

    try:
        api = _get_hf_api()
    except ValueError as e:
        return ToolResult(content=f"Error: {e}")

    resolved_type, err = _resolve_hf_pull_repo_type(api, repo_id, repo_type)
    if err is not None:
        return ToolResult(content=f"Error: {err}")
    repo_type = resolved_type

    repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    if local_path is None:
        local_path = f"{'datasets' if repo_type == 'dataset' else 'models'}/{repo_name}"

    target = _validate_path(project_dir, local_path)
    # Dataset immutability applies to imports too: refuse to clobber an
    # existing local dataset's parquet files without explicit overwrite.
    if repo_type == "dataset" and not overwrite:
        existing_parquet = sorted(
            p.name for p in target.glob("*.parquet")
        ) if target.is_dir() else []
        if existing_parquet:
            return ToolResult(content=(
                f"Error: {local_path}/ already contains "
                f"{', '.join(existing_parquet[:3])} — refusing to overwrite "
                "an existing dataset. Pull into a different local_path "
                "(e.g. a versioned name), or pass overwrite=true only "
                "after the user confirmed replacing it."
            ))
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
                return ToolResult(
                    content="Error: split/subset are dataset-only options; omit them for model pulls."
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
        return ToolResult(content=f"Error downloading {repo_id}: {e}{hint}")


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
    **kwargs: Any,
) -> ToolResult:
    """Push a local dataset or model checkpoint to HF Hub. Requires permission."""
    # Check HF token first
    try:
        api = _get_hf_api()
    except ValueError as e:
        return ToolResult(content=f"Error: {e}")

    # Validate local path
    target = _validate_path(project_dir, local_path)
    if not target.exists():
        return ToolResult(content=f"Error: '{local_path}' does not exist")
    if not target.is_dir():
        return ToolResult(
            content=f"Error: '{local_path}' is not a directory. hf_push expects a folder containing either parquet files (dataset) or model files (config.json + weights)."
        )

    if repo_type is not None and repo_type not in ("dataset", "model"):
        return ToolResult(content=f"Error: invalid repo_type '{repo_type}' (must be 'dataset' or 'model')")

    detected, parquet_files, model_files = _detect_hf_repo_type(target)

    if repo_type is None:
        if detected is None:
            if parquet_files and model_files:
                return ToolResult(
                    content=(
                        f"Error: '{local_path}' contains both parquet files and model files — "
                        f"cannot auto-detect repo type. Pass repo_type='dataset' or repo_type='model' to disambiguate."
                    )
                )
            return ToolResult(
                content=(
                    f"Error: '{local_path}' is not recognizable as a dataset or model folder. "
                    f"Expected either .parquet files or HF-style model files "
                    f"(config.json, *.safetensors, *.bin, *.ckpt, *.pt, *.pth)."
                )
            )
        repo_type = detected
    else:
        # Validate explicit override against what we found.
        if repo_type == "dataset" and not parquet_files:
            return ToolResult(
                content=f"Error: repo_type='dataset' but no .parquet files found in '{local_path}'."
            )
        if repo_type == "model" and not model_files:
            return ToolResult(
                content=(
                    f"Error: repo_type='model' but '{local_path}' has no model files "
                    f"(config.json, *.safetensors, *.bin, *.ckpt, *.pt, *.pth)."
                )
            )

    # Auto-generate repo_id if not provided
    if repo_id is None:
        try:
            info = api.whoami()
            username = info.get("name", "unknown")
        except Exception as e:
            return ToolResult(content=f"Error getting HF username: {e}")

        project_name = project_dir.name
        repo_id = f"{username}/{project_name}-{target.name}"

    # Check permission
    from lqh.tools.permissions import check_hf_permission

    if not check_hf_permission(project_dir, repo_id):
        details = f"  Split: {split}\n" if repo_type == "dataset" else ""
        return ToolResult(
            content="PERMISSION_REQUIRED",
            requires_user_input=True,
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
        return ToolResult(content=f"Error pushing to {repo_id}: {e}{hint}")


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
        return ToolResult(content=f"Error preparing adapter for upload: {exc}")

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
    num_epochs: int = 3,
    learning_rate: float | None = None,
    num_iterations: int = 5,
    dpo_beta: float = 0.1,
    golden_source: str = "dataset",
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
    - DPO: fixed held-out judge score, with chosen-response CE retained as a
      catastrophic-collapse veto. DPO's own ``eval_loss`` is NOT used — it
      can improve by suppressing rejected likelihood while task quality falls.

    DPO sweeps are deliberately judge-backed and therefore more expensive than
    SFT sweeps. This avoids ranking configurations on incomparable on-policy
    preference subsets.

    Pass ``enable_sweep=false`` to fall back to single-config behaviour
    only when the user explicitly asks for it (e.g. "just train one
    config with lr=2e-5"). Specific ``learning_rate``/``num_epochs``/
    ``dpo_beta`` values supplied by the agent are honoured under
    ``enable_sweep=false``; under sweep they are overridden by the grid.

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
    from lqh.tools.permissions import check_training_permission

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

    # Validate dataset source(s). A single string or a list of sources to
    # combine; train sources may carry an integer `repeat` over-sampling
    # factor. Resolves to canonical {"path", "repeat", "source"} entries.
    dataset_sources, train_resolved, ds_err = _resolve_training_sources(
        project_dir, dataset, kind="dataset", allow_repeat=True
    )
    if ds_err:
        return ToolResult(content=ds_err)

    # eval_dataset is mandatory: the sweep needs a held-out signal to pick its
    # winner, and the judge eval-of-best needs rollouts to score. (The tool
    # schema marks it required; this guards non-schema callers.)
    if not eval_dataset:
        return ToolResult(
            content=(
                "Error: eval_dataset is required. Pass the project's held-out eval "
                "set (e.g. 'datasets/<name>_eval'). It is the signal used to select "
                "the sweep winner and the set the best checkpoint is judge-scored on."
            )
        )

    eval_sources, eval_resolved, eval_err = _resolve_training_sources(
        project_dir, eval_dataset, kind="eval_dataset", allow_repeat=False
    )
    if eval_err:
        return ToolResult(content=eval_err)

    # Reject duplicate eval sources — scoring the same set twice would
    # double-count it in the macro-average.
    eval_seen: set[str] = set()
    for p in eval_resolved:
        key = str(p)
        if key in eval_seen:
            return ToolResult(
                content=(
                    "Error: eval_dataset lists the same source twice "
                    f"({p.name}). Each eval source must be distinct so the "
                    "macro-average weights them once each."
                )
            )
        eval_seen.add(key)

    # dataset and eval_dataset must be DISTINCT. Evaluating on the training
    # prompts is exactly the leak the train/eval split exists to prevent —
    # reject any overlap between a train source and an eval source.
    overlap = {str(p) for p in train_resolved} & eval_seen
    if overlap:
        names = ", ".join(sorted(Path(p).as_posix() for p in overlap))
        return ToolResult(
            content=(
                "Error: eval_dataset must be different from dataset — these "
                f"source(s) appear in both: {names}. Evaluating on the training "
                "prompts leaks train into eval. Pass separate held-out eval "
                "set(s) (e.g. 'datasets/<name>_eval')."
            )
        )

    # On-policy DPO builds its preference pairs by judge-scoring generated
    # rollouts every iteration, so a scorer is mandatory — scoring cannot be
    # disabled the way it can for SFT (where it only gates the final eval).
    if type in ("on_policy_dpo", "dpo") and disable_scoring:
        return ToolResult(
            content=(
                "Error: scoring cannot be disabled for DPO. On-policy DPO assembles "
                "its preference pairs from scored rollouts each iteration, so a scorer "
                "is required — pass `scorer=<path>` (the project's default/best scorer)."
            )
        )

    # Scoring must be an explicit decision: pass a scorer, or opt out via
    # disable_scoring. Silently omitting the scorer would degrade eval-of-best
    # to proxy-only with no judge score — a common, quiet failure mode.
    if not scorer and not disable_scoring:
        return ToolResult(
            content=(
                "Error: no scorer provided. The best checkpoint needs a scorer to get "
                "a real judge score. Pass `scorer=<path>` set to the project's "
                "default/current scorer (the one under evals/scorers/ used for the "
                "baseline eval), or — only if the user explicitly asked not to score — "
                "set disable_scoring=true."
            )
        )

    scorer_path: str | None = None
    if scorer:
        scorer_resolved = _validate_path(project_dir, scorer)
        if not scorer_resolved.exists():
            return ToolResult(content=f"Error: scorer not found at {scorer}")
        scorer_path = scorer

    # Vision-language (LFM-VL) bases switch the run into the vision path:
    # AutoProcessor + image collation in the subprocess, the Liquid VLM
    # LoRA recipe, and conservative batch defaults (the text calibration
    # probe is skipped for vision). SFT-only for now.
    from lqh.models import is_vlm_model_name

    is_vision = is_vlm_model_name(base_model)
    if is_vision and type != "sft":
        return ToolResult(
            content=(
                f"Error: {type} is not supported for vision-language models yet — "
                f"only SFT is. Train {base_model} with type='sft'."
            )
        )

    # Generate run name
    if not run_name:
        prefix = "sft" if type == "sft" else "dpo"
        run_name = _next_run_name(project_dir, prefix)

    run_dir = project_dir / "runs" / run_name

    if run_dir.exists() and (run_dir / "config.json").exists():
        return ToolResult(content=f"Error: run '{run_name}' already exists")

    # Build config
    default_lr = 2e-5 if type == "sft" else 5e-6
    if is_vision:
        default_lr = 5e-4  # Liquid VLM LoRA recipe
    lr = learning_rate if learning_rate is not None else default_lr
    if is_vision:
        # No calibration probe for vision — start conservative and let the
        # OOM self-heal (report_oom_downgrade) shrink further if needed.
        default_micro_batch = 2
        default_effective_batch = 16
    elif lora and type in ("on_policy_dpo", "dpo"):
        # DPO preference batches are normally only a few hundred rows.  The
        # old LoRA-wide default of 256 reduced those batches to one or two
        # optimizer updates per on-policy iteration.  Keep enough updates to
        # learn from the pairs; the GPU calibrator may still reduce the micro
        # batch for memory safety without increasing this effective target.
        default_micro_batch = 16
        default_effective_batch = 16
    elif lora:
        default_micro_batch = 256
        default_effective_batch = 256
    else:
        default_micro_batch = 1
        default_effective_batch = 16 if type == "sft" else 2
    default_grad_accum = max(
        1,
        (default_effective_batch + default_micro_batch - 1) // default_micro_batch,
    )

    if is_vision:
        lora_defaults: dict[str, Any] = {
            "enabled": lora,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            # Liquid VLM recipe: attention + FFN + vision-tower MLPs (fc1/
            # fc2) + multimodal projector (linear). Intentionally different
            # from the text LFM module list.
            "target_modules": [
                "q_proj", "v_proj", "fc1", "fc2", "linear",
                "gate_proj", "up_proj", "down_proj",
            ],
        }
    else:
        lora_defaults = {
            "enabled": lora,
            "r": 32,
            "alpha": 64,
            "dropout": 0.02,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj", "out_proj", "w1", "w2", "w3",
            ],
        }

    from lqh.project_meta import compute_spec_sha256

    config: dict[str, Any] = {
        "type": type,
        "base_model": base_model,
        "dataset": _sources_to_config(dataset_sources),
        # Spec revision at submission time: checkpoints trained from this
        # config stay traceable to the spec they were built against even
        # after SPEC.md changes mid-run (PERSISTENCY_PLAN.md R6).
        "spec_sha256": compute_spec_sha256(project_dir),
        "training": {
            "learning_rate": lr,
            "max_seq_length": 2048,
            "per_device_batch_size": default_micro_batch,
            "gradient_accumulation_steps": default_grad_accum,
            "effective_batch_size": default_effective_batch,
            "auto_batch": True,
        },
        "lora": lora_defaults,
        "manifest": ["base_model", "dataset"],
    }
    if is_vision:
        config["modality"] = "vision"
        # Per-image token budget for the processor. Effective text budget
        # is roughly max_seq_length − n_images × max_image_tokens.
        config["training"]["max_image_tokens"] = 256

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

    if type == "sft":
        config["training"]["num_epochs"] = num_epochs
    elif type in ("on_policy_dpo", "dpo"):
        config["num_iterations"] = num_iterations
        config["dpo_beta"] = dpo_beta
        config["golden_source"] = golden_source
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
    if not check_training_permission(project_dir, run_name):
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
                f"  GPU:       {gpu_info}{size_warning}\n\n"
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
            job_id = await backend.submit_run(str(run_dir), config, module=module)
        except Exception as e:
            return ToolResult(content=f"Error launching cloud training: {e}")

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
        return ToolResult(
            content=(
                f"🚀 Cloud training submitted\n"
                f"  Run:     {run_name}\n"
                f"  Type:    {config.get('type', 'unknown')}\n"
                f"  Job ID:  {job_id}\n\n"
                f"Backend: LQH Cloud (api.lqh.ai). Use training_status to monitor progress."
            ),
            workflow_launched=True,
        )

    # --- SSH path (existing behavior) ---
    from lqh.remote.config import get_remote
    from lqh.remote.ssh_direct import SSHDirectBackend

    ssh_name = ssh_remote_name(remote_name) or remote_name
    remote_config = get_remote(project_dir, ssh_name)
    if remote_config is None:
        return ToolResult(
            content=f"Error: remote '{ssh_name}' not found. Use remote_list to see configured remotes."
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

    return ToolResult(
        content=(
            f"🚀 Remote training started on '{ssh_name}'\n"
            f"  Run:      {run_name}\n"
            f"  Type:     {config['type']}\n"
            f"  Job ID:   {job_id}\n"
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

    return ToolResult(
        content=(
            f"🚀 Training started\n"
            f"  Run:    {run_name}\n"
            f"  Type:   {config.get('type', 'unknown')}\n"
            f"  PID:    {pid}\n"
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
            return ToolResult(content=f"Error: run '{run_name}' not found")
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
        return ToolResult(content=f"Error: no remote job metadata for run '{run_name}'.")
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
            return ToolResult(content=f"Error: remote '{ssh_name}' not found.")
        backend = SSHDirectBackend(remote_config, project_dir)
        display_remote = ssh_name

    try:
        # Sync progress first
        await backend.sync_progress(remote_run_dir, str(run_dir))
        status = await backend.poll_status(job_id)
    except Exception as e:
        return ToolResult(content=_format_training_status_error(e))

    state_emoji = {
        "running": "🏃", "completed": "✅", "failed": "❌",
        "waiting_for_scoring": "⏳", "unknown": "❓",
    }
    emoji = state_emoji.get(status.state, "❓")
    lines = [f"{emoji} **{run_name}** — {status.state} (remote: {display_remote})"]
    if status.current_step is not None:
        lines.append(f"  Step: {status.current_step}")
    if status.error:
        lines.append(f"  Error: {status.error}")

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
                        # When multiple eval sources were scored, the headline
                        # mean is a macro-average — show the per-source breakdown.
                        per_source = result.get("per_source") or {}
                        if len(per_source) > 1:
                            for label in sorted(per_source):
                                src_mean = (
                                    per_source[label].get("scores", {}).get("mean")
                                )
                                if src_mean is not None:
                                    eval_results.append(
                                        f"      {label}: mean={src_mean:.2f}"
                                    )
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
    **kwargs: Any,
) -> ToolResult:
    """Stop a training subprocess.

    Whether the run is remote is derived from its persisted
    ``remote_job.json`` (written at launch), not from a caller argument.
    """
    from lqh.subprocess_manager import SubprocessManager

    run_dir = _validate_path(project_dir, f"runs/{run_name}")
    if not run_dir.exists():
        return ToolResult(content=f"Error: run '{run_name}' not found")

    meta = _read_remote_meta(run_dir)
    if meta is not None:
        return await _stop_training_remote(project_dir, run_name, meta["remote_name"])

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
    """Stop a remote training run.

    Branches on ``remote_name``: ``"cloud"`` routes through
    ``CloudBackend``; anything else is treated as an SSH remote.
    """
    from lqh.remote.compute import is_cloud

    run_dir = project_dir / "runs" / run_name
    meta_file = run_dir / "remote_job.json"
    if not meta_file.exists():
        return ToolResult(content=f"Error: no remote job metadata for run '{run_name}'.")

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
            return ToolResult(content=f"Error: remote '{ssh_name}' not found.")
        remote_name = ssh_name
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
        return ToolResult(content=f"❌ {err}")

    # Validate paths
    model_dir = _validate_path(project_dir, model_path)
    if not model_dir.exists():
        return ToolResult(content=f"Error: model not found at {model_path}")

    # Eval dataset(s) — one path or a list of held-out sources. Multiple
    # sources are scored separately and combined into a macro-average (each
    # source weighted equally), same as the training eval-of-best.
    eval_sources, eval_resolved, ds_err = _resolve_training_sources(
        project_dir, dataset, kind="dataset", allow_repeat=False
    )
    if ds_err:
        return ToolResult(content=ds_err)

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
    if eval_run_dir.exists():
        return ToolResult(content=(
            f"Error: run '{run_name}' already exists — run names must be "
            "unique (an existing run's config/logs would be overwritten). "
            "Pick a different run_name or omit it for an auto-generated one."
        ))

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
        return ToolResult(
            content=f"Error: training_method must be 'lora' or 'full', got {training_method!r}"
        )
    if training_method == "lora" and not base_model:
        return ToolResult(
            content="Error: base_model is required when training_method='lora'"
        )
    if judge_size not in ("small", "medium", "large"):
        return ToolResult(
            content=f"Error: judge_size must be small/medium/large, got {judge_size!r}"
        )

    # Eval dataset(s) — one path or a list of held-out sources, scored
    # separately and macro-averaged sandbox-side.
    eval_sources, eval_resolved, ds_err = _resolve_training_sources(
        project_dir, eval_dataset, kind="eval_dataset", allow_repeat=False
    )
    if ds_err:
        return ToolResult(content=ds_err)

    scorer_resolved = _validate_path(project_dir, scorer)
    if not scorer_resolved.exists():
        return ToolResult(content=f"Error: scorer not found at {scorer}")

    try:
        system_prompt, schema_dict = _resolve_eval_extras(
            project_dir,
            system_prompt_path=system_prompt_path,
            response_format_path=None,
        )
    except FileNotFoundError as e:
        return ToolResult(content=f"Error: {e}")

    if not run_name:
        run_name = _next_run_name(project_dir, "eval_hf")
    run_dir = project_dir / "runs" / run_name
    if run_dir.exists():
        return ToolResult(content=(
            f"Error: run '{run_name}' already exists — run names must be "
            "unique (an existing run's config/logs would be overwritten). "
            "Pick a different run_name or omit it for an auto-generated one."
        ))
    run_dir.mkdir(parents=True, exist_ok=True)

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
        "num_samples": sum((_parquet_metadata(path)[0] or 0) for path in eval_resolved),
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
    if system_prompt_path:
        # Also include the source file in the bundle so the
        # artifact_lineage row can pin it (the publisher records the
        # config alongside the eval artifacts).
        config["system_prompt_path"] = system_prompt_path
        config["manifest"].append("system_prompt_path")

    # --- Submit to LQH Cloud ---
    from lqh.remote.backend import RemoteConfig
    from lqh.remote.cloud import CloudBackend

    cfg = RemoteConfig(
        name="cloud",
        type="cloud",
        hostname="api.lqh.ai",
        remote_root="cloud:lqh",
    )
    backend = CloudBackend(cfg, project_dir)
    try:
        job_id = await backend.submit_run(
            str(run_dir), config, module="lqh.infer.eval_hf",
        )
    except Exception as e:  # noqa: BLE001
        return ToolResult(content=f"Error submitting eval_hf job: {e}")

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

    return ToolResult(
        content=(
            f"🧪 HF eval submitted\n"
            f"  Run:     {run_name}\n"
            f"  Repo:    {repo}@{revision}\n"
            f"  Method:  {training_method}"
            + (f" (base {base_model})" if training_method == 'lora' else "")
            + f"\n"
            f"  Judge:   judge:{judge_size}\n"
            f"  Job ID:  {job_id}\n\n"
            f"Use training_status to monitor; eval_result.json lands "
            f"under runs/{run_name}/ when done."
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
        return ToolResult(content=f"Error: remote '{ssh_name}' not found.")
    remote_name = ssh_name

    if remote_config.type == "ssh_slurm":
        return ToolResult(content="Error: SSH+Slurm backend is not yet implemented.")

    # Validate eval dataset source(s) — one path or a list of held-out
    # sources, scored separately and macro-averaged (same as the local path).
    eval_sources, eval_resolved, ds_err = _resolve_training_sources(
        project_dir, dataset, kind="dataset", allow_repeat=False
    )
    if ds_err:
        return ToolResult(content=ds_err)

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
    if run_dir.exists():
        return ToolResult(content=(
            f"Error: run '{run_name}' already exists — run names must be "
            "unique (an existing run's config/logs would be overwritten). "
            "Pick a different run_name or omit it for an auto-generated one."
        ))
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
        return ToolResult(content=(
            f"Error: output_dataset must be a plain name (no path "
            f"separators), got {output_dataset!r}"
        ))

    input_abs = _validate_path(project_dir, input_path)
    scorer_abs = _validate_path(project_dir, scorer_path)
    if not input_abs.exists():
        return ToolResult(content=f"Error: input '{input_path}' does not exist")
    if not scorer_abs.exists():
        return ToolResult(content=f"Error: scorer '{scorer_path}' does not exist")

    # Immutable-by-default outputs (PERSISTENCY_PLAN.md R5). Fast
    # read-only refusal here; the atomic claim happens inside the
    # release-protected block below so an auth/config failure can never
    # leak a held claim.
    from lqh.dataset_guard import claim_output, overwrite_refusal, release_output

    early_refusal = overwrite_refusal(
        project_dir, output_dataset, overwrite=overwrite
    )
    if early_refusal:
        return ToolResult(content=f"Error: {early_refusal}")

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
    client = create_client(token, config.api_base_url)

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
            return ToolResult(content=f"Error: {refusal}")
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
        return ToolResult(content=f"❌ run_data_filter failed: {type(exc).__name__}: {exc}")
    finally:
        if claimed:
            release_output(project_dir, output_dataset)
        if succeeded:
            reporter.update(
                phase="completed", phase_label="filtered dataset ready",
                completed=result.total, total=result.total, unit="samples",
                overall_fraction=1.0, result_ready=True, force=True,
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
    ) is not None
    manifest_warning = (
        "" if manifest_written else
        "\n  ⚠️ Provenance manifest could not be written (check disk/logs)."
    )

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
            + manifest_warning
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
        return ToolResult(content=f"Error: unknown tool '{tool_name}'")

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
