"""Artifact provenance manifests (Phase 4 of PERSISTENCY_PLAN.md).

A small, user-inspectable ``manifest.json`` is written next to an
artifact when it is finalized: which spec revision it was built against,
what produced it (pipeline/scorer/run/cloud job), what it was made from,
and what it is for. The readers already exist — the ``summary`` tool
renders provenance and the startup signals flag spec drift — this module
is the writer side.

Provenance correctness rules:

* Hashes describe the inputs **as they were when the work started** —
  callers capture ``spec_sha256``/``pipeline_hash`` before launching and
  pass them in; computing at manifest-write time would mis-attribute an
  artifact to files edited while a long job ran. The compute-now
  fallback exists only for callers that finalize immediately.
* Incomplete provenance is *marked*, never invented (``warnings``,
  ``provenance_note``, ``spec_sha256_source``).
* Writers are best-effort — bookkeeping must never fail the workflow
  that produced the artifact — but they return None on failure so
  callers can surface it to the user instead of implying traceability.

Manifests are durable facts for the *agent* to ``read_file``, not inputs
to a reconciler: keep them small, flat, and self-explanatory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lqh.fsio import atomic_write_json
from lqh.project_log import file_hash_prefix
from lqh.project_meta import compute_spec_sha256

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# What an artifact is for. Mirrors the data-gen tool's purpose values,
# plus the feedback-loop purposes from PERSISTENCY_PLAN.md.
PURPOSES = (
    "smoke",
    "inspection",
    "validation",
    "training",
    "failures",
    "imported",
    "unspecified",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _content_hash(path: Path) -> str | None:
    """Streamed sha256 of an artifact file; None when unreadable."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _existing_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _relativize(path_str: str, project_dir: Path) -> str:
    """Project-relative form of a path so manifests survive project moves."""
    try:
        p = Path(path_str)
        if p.is_absolute():
            return str(p.resolve().relative_to(project_dir.resolve()))
    except ValueError:
        pass
    return str(path_str)


def _hashed_sources(
    source_paths: list[str], project_dir: Path
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw in source_paths:
        rel = _relativize(str(raw), project_dir)
        entry: dict[str, Any] = {"path": rel}
        candidate = project_dir / rel
        if candidate.is_file():
            entry["hash"] = file_hash_prefix(candidate, n=12)
        entries.append(entry)
    return entries


def write_dataset_manifest(
    project_dir: Path,
    dataset_dir: Path,
    *,
    purpose: str = "unspecified",
    rows: int | None = None,
    pipeline_path: str | None = None,
    pipeline_hash: str | None = None,
    spec_sha256: str | None = None,
    source_paths: list[str] | None = None,
    scorer_path: str | None = None,
    threshold: float | None = None,
    parent_dataset: str | None = None,
    derived_from: str | None = None,
    provenance_note: str | None = None,
    run_name: str | None = None,
    job_id: str | None = None,
    cloud_artifact_id: str | None = None,
) -> Path | None:
    """Write ``<dataset_dir>/manifest.json`` after finalization.

    ``spec_sha256``/``pipeline_hash`` should be the values captured when
    the producing work STARTED (see module docstring). ``parent_dataset``
    marks a supplement (new data extending an existing set);
    ``derived_from`` marks a derivative (e.g. a filtered subset).

    Rewriting the same logical dataset keeps its stable ``dataset_id``
    and bumps ``version``. Returns the manifest path, or None when the
    write failed (logged, never raised) — callers should surface None to
    the user rather than implying the artifact is traceable.
    """
    try:
        manifest_path = dataset_dir / "manifest.json"
        previous = _existing_manifest(manifest_path)
        if purpose not in PURPOSES:
            purpose = "unspecified"

        warnings: list[str] = []
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": previous.get("dataset_id") or str(uuid.uuid4()),
            "version": int(previous.get("version", 0) or 0) + 1,
            "name": dataset_dir.name,
            "created_at": _now(),
            "purpose": purpose,
            "spec_sha256": spec_sha256 or compute_spec_sha256(project_dir),
        }
        if spec_sha256 is None:
            # Computed now rather than captured at work start — only
            # correct when finalization is immediate.
            manifest["spec_sha256_source"] = "finalization"
        if rows is not None:
            manifest["rows"] = rows
        data_file = dataset_dir / "data.parquet"
        if data_file.exists():
            content_hash = _content_hash(data_file)
            if content_hash:
                manifest["content_sha256"] = content_hash
            else:
                warnings.append("content hash could not be computed")
        else:
            # Multi-file datasets (e.g. imported HF splits): hash every
            # parquet so the content is still fingerprinted.
            split_hashes = {}
            for parquet in sorted(dataset_dir.glob("*.parquet")):
                if parquet.name == "scores.parquet":
                    continue
                split_hash = _content_hash(parquet)
                if split_hash:
                    split_hashes[parquet.name] = split_hash
                else:
                    warnings.append(f"content hash failed for {parquet.name}")
            if split_hashes:
                manifest["content_sha256_files"] = split_hashes
        if pipeline_path:
            manifest["pipeline_path"] = _relativize(pipeline_path, project_dir)
            if pipeline_hash:
                manifest["pipeline_hash"] = pipeline_hash
            else:
                pipeline_file = project_dir / manifest["pipeline_path"]
                if pipeline_file.exists():
                    manifest["pipeline_hash"] = file_hash_prefix(pipeline_file, n=12)
                    manifest["pipeline_hash_source"] = "finalization"
        if source_paths:
            manifest["sources"] = _hashed_sources(list(source_paths), project_dir)
            # Honest timing: source files are hashed now, not when the
            # work began (they are only discovered during the run).
            manifest["sources_hashed_at"] = "finalization"
        if scorer_path:
            manifest["scorer_path"] = _relativize(scorer_path, project_dir)
            scorer_file = project_dir / manifest["scorer_path"]
            if scorer_file.is_file():
                manifest["scorer_hash"] = file_hash_prefix(scorer_file, n=12)
        if threshold is not None:
            manifest["threshold"] = threshold
        if parent_dataset:
            manifest["parent_dataset"] = _relativize(parent_dataset, project_dir)
        if derived_from:
            manifest["derived_from"] = _relativize(derived_from, project_dir)
        if provenance_note:
            manifest["provenance_note"] = provenance_note
        if run_name:
            manifest["run_name"] = run_name
        if job_id:
            manifest["job_id"] = job_id
        if cloud_artifact_id:
            manifest["cloud_artifact_id"] = cloud_artifact_id
        if warnings:
            manifest["warnings"] = warnings

        atomic_write_json(manifest_path, manifest)
        return manifest_path
    except Exception:
        logger.warning(
            "could not write dataset manifest for %s", dataset_dir, exc_info=True
        )
        return None


def annotate_manifest(dataset_dir: Path, **fields: Any) -> None:
    """Merge extra fields into an existing dataset manifest (best-effort).

    Used for post-hoc facts like data-quality scoring. No-op when the
    dataset has no manifest — annotation must not invent provenance.
    """
    manifest_path = dataset_dir / "manifest.json"
    manifest = _existing_manifest(manifest_path)
    if not manifest:
        return
    try:
        manifest.update({k: v for k, v in fields.items() if v is not None})
        atomic_write_json(manifest_path, manifest)
    except Exception:
        logger.warning(
            "could not annotate manifest for %s", dataset_dir, exc_info=True
        )


def inherit_purpose(dataset_dir: Path, default: str = "unspecified") -> str:
    """Purpose to carry forward when deriving from ``dataset_dir``.

    Unknown provenance stays unknown — the default is "unspecified", not
    an invented value.
    """
    purpose = _existing_manifest(dataset_dir / "manifest.json").get("purpose")
    return purpose if purpose in PURPOSES else default


def _last_progress_metrics(run_dir: Path) -> dict[str, Any] | None:
    """Last metrics row (step/loss/lr/epoch) from progress.jsonl."""
    path = run_dir / "progress.jsonl"
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    keys = ("step", "loss", "lr", "epoch")
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if any(k in row for k in keys):
            return {k: row[k] for k in keys if k in row}
    return None


def write_run_manifest(
    project_dir: Path,
    run_dir: Path,
    *,
    state: str,
    error: str | None = None,
) -> Path | None:
    """Write ``<run_dir>/manifest.json`` when a run reaches a terminal state.

    Records what is needed to trace a checkpoint/eval back to its inputs
    and to identify its result: spec revision (as submitted when the
    config recorded one), config content hash, base model or evaluated
    HF repo/revision, dataset composition, training hyperparameters,
    final metrics, eval scores, and checkpoint list.
    """
    try:
        config: dict[str, Any] = {}
        config_raw: str | None = None
        try:
            config_raw = (run_dir / "config.json").read_text(encoding="utf-8")
            config = json.loads(config_raw)
        except (OSError, ValueError):
            pass

        manifest_path = run_dir / "manifest.json"
        previous = _existing_manifest(manifest_path)
        submitted_spec = config.get("spec_sha256")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": previous.get("run_id") or str(uuid.uuid4()),
            "name": run_dir.name,
            "created_at": _now(),
            "kind": config.get("type") or "",
            "state": state,
            "spec_sha256": submitted_spec or compute_spec_sha256(project_dir),
            # "submission" hashes are trustworthy even if SPEC.md changed
            # mid-run; "completion" means the config predates spec-hash
            # stamping and the value reflects the spec at finalization.
            "spec_sha256_source": "submission" if submitted_spec else "completion",
        }
        if config_raw is not None:
            manifest["config_sha256"] = hashlib.sha256(
                config_raw.encode("utf-8")
            ).hexdigest()[:12]
        if error:
            manifest["error"] = str(error)[:500]
        for key in (
            "base_model",
            "dataset",
            "eval_dataset",
            "scorer",
            "training",
            "hf_repo",
            "revision",
            "training_method",
            "num_samples",
            "inference_model",
            "judge_size",
            "system_prompt_path",
            "response_format_path",
        ):
            if config.get(key) is not None:
                manifest[key] = config[key]
        metrics = _last_progress_metrics(run_dir)
        if metrics:
            manifest["final_metrics"] = metrics
        for result_name in ("eval_result.json", "summary.json", "sweep_summary.json"):
            result_path = run_dir / result_name
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if isinstance(result, dict):
                        manifest["result_summary"] = {
                            k: result[k]
                            for k in (
                                "scores", "num_samples", "num_scored",
                                "num_failed", "best_run", "best_config",
                                "best_score", "num_configs",
                            )
                            if k in result
                        } or None
                except (OSError, ValueError):
                    pass
                break
        if manifest.get("result_summary") is None:
            manifest.pop("result_summary", None)
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.is_dir():
            checkpoints = sorted(
                p.name for p in checkpoints_dir.iterdir() if p.is_dir()
            )
            manifest["checkpoints"] = checkpoints
            if "final" in checkpoints:
                manifest["selected_checkpoint"] = "final"

        atomic_write_json(manifest_path, manifest)
        return manifest_path
    except Exception:
        logger.warning(
            "could not write run manifest for %s", run_dir, exc_info=True
        )
        return None
