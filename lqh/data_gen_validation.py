"""Local-validation records for data-gen pipelines.

Cloud data_gen submission (CLOUD_OFFLOAD_PLAN.md §2) is gated on a
successful *local* run of the exact same pipeline version: the handler
refuses ``execution="cloud"`` unless a record exists here whose content
hash matches the pipeline file on disk. Editing the pipeline changes
the hash and re-arms the gate, forcing a fresh local correctness check
before the next big cloud run.

The record also carries the ``lqh.sources`` paths the validated run
actually read (see ``lqh.sources.record_source_paths``) — that recorded
set is the trusted bundle manifest for shipping seed data (image
folders, prompt files, ...) to the sandbox.

Storage: ``<project>/.lqh/data_gen_validation.json``, one entry per
pipeline file keyed by its project-relative path. Written atomically
(tmp + ``os.replace``) like the sibling ``.lqh`` state files.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "ValidationRecord",
    "validation_file_path",
    "script_sha256",
    "pipeline_digest",
    "source_digest",
    "record_validation",
    "check_validation",
]

_VERSION = 2


@dataclass
class ValidationRecord:
    """A successful local run of one pipeline version."""

    sha256: str
    validated_at: str
    num_samples: int
    succeeded: int
    failed: int
    # Project-relative paths (folders or files) the run read via
    # lqh.sources helpers.
    source_paths: list[str]
    # sha256 fingerprints keyed by project-relative source path. Local files
    # are part of the validated execution input, not merely a bundle manifest.
    source_digests: dict[str, str]
    # Whether the run used lqh.sources.hf_dataset (observed, not
    # guessed) — gates HF-token injection into the cloud sandbox.
    needs_hf: bool = False


def validation_file_path(project_dir: Path) -> Path:
    """Path to the project-level data_gen_validation.json file."""
    return project_dir / ".lqh" / "data_gen_validation.json"


def script_sha256(script_path: Path) -> str:
    """Content hash of one file (bytes, not normalized)."""
    return hashlib.sha256(script_path.read_bytes()).hexdigest()


def pipeline_digest(script_path: Path) -> str:
    """Content hash of the pipeline's executable code surface.

    Pipelines are self-contained single files (sibling imports are
    unsupported and fail at local load time), so the entry file IS the
    code surface — hashing siblings would only re-arm the gate when an
    unrelated pipeline in the same directory changes. (Validation-
    instruction files are not hashed either: the engine currently
    doesn't use them, so they can't change execution.)
    """
    return hashlib.sha256(script_path.read_bytes()).hexdigest()


def source_digest(source_path: Path) -> str:
    """Stable content digest for one recorded source file or directory."""
    path = source_path.resolve()
    h = hashlib.sha256()
    if path.is_file():
        h.update(b"file\0")
        h.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        h.update(b"dir\0")
        for child in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
            rel = child.relative_to(path).as_posix().encode()
            h.update(len(rel).to_bytes(8, "big"))
            h.update(rel)
            h.update(child.stat().st_size.to_bytes(8, "big"))
            with child.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
        return h.hexdigest()
    raise FileNotFoundError(path)


def _script_key(project_dir: Path, script_path: Path) -> str:
    """Project-relative posix path used as the record key."""
    resolved = script_path.resolve()
    try:
        return resolved.relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        # Script outside the project (shouldn't happen via the tool
        # layer) — fall back to the absolute path so the gate still
        # round-trips.
        return resolved.as_posix()


def _load_all(project_dir: Path) -> dict:
    path = validation_file_path(project_dir)
    if not path.exists():
        return {"version": _VERSION, "pipelines": {}}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"version": _VERSION, "pipelines": {}}
    if (
        not isinstance(data, dict)
        or data.get("version") != _VERSION
        or not isinstance(data.get("pipelines"), dict)
    ):
        return {"version": _VERSION, "pipelines": {}}
    return data


def record_validation(
    project_dir: Path,
    script_path: Path,
    *,
    num_samples: int,
    succeeded: int,
    failed: int,
    source_paths: list[Path] | None = None,
    needs_hf: bool = False,
) -> None:
    """Persist a successful local run of *script_path*.

    A run that produced nothing validates nothing — zero-success calls
    are ignored (enforced here, not just at the caller).
    """
    if succeeded <= 0:
        return
    resolved_project = project_dir.resolve()
    rel_sources: list[str] = []
    source_digests: dict[str, str] = {}
    for p in source_paths or []:
        try:
            resolved_source = Path(p).resolve()
            rel = resolved_source.relative_to(resolved_project).as_posix()
            rel_sources.append(rel)
            if resolved_source.exists():
                source_digests[rel] = source_digest(resolved_source)
        except ValueError:
            # Outside the project — can't be bundled; skip rather than
            # store a path the bundle builder would reject.
            continue

    data = _load_all(project_dir)
    data["version"] = _VERSION
    data["pipelines"][_script_key(project_dir, script_path)] = {
        "sha256": pipeline_digest(script_path),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": num_samples,
        "succeeded": succeeded,
        "failed": failed,
        "source_paths": sorted(set(rel_sources)),
        "source_digests": source_digests,
        # Observed during the run (lqh.sources.hf_dataset) — drives
        # whether the cloud sandbox receives the stored HF token.
        "needs_hf": bool(needs_hf),
    }

    path = validation_file_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def check_validation(project_dir: Path, script_path: Path) -> ValidationRecord | None:
    """Return the validation record for the *current* pipeline content.

    None when the pipeline was never validated locally, when the file
    changed since validation (hash mismatch), or when the file is
    missing/unreadable.
    """
    try:
        current_hash = pipeline_digest(script_path)
    except OSError:
        return None
    entry = _load_all(project_dir)["pipelines"].get(_script_key(project_dir, script_path))
    if not isinstance(entry, dict):
        return None
    if entry.get("sha256") != current_hash:
        return None
    try:
        source_paths = [str(s) for s in entry.get("source_paths", [])]
        raw_digests = entry.get("source_digests", {})
        if not isinstance(raw_digests, dict):
            return None
        # The record file lives inside the project and is therefore
        # forgeable in principle — never honor path shapes record_validation
        # can't produce. Recorded paths are always project-relative with no
        # traversal; an absolute or '..' entry would ride the bundle
        # manifest (which tolerates absolute paths, shipping them under
        # extern/) and upload arbitrary readable local files to the cloud.
        # A tampered record simply reads as "not validated".
        for s in source_paths:
            p = Path(s)
            if p.is_absolute() or ".." in p.parts or not s.strip():
                return None
            expected = raw_digests.get(s)
            source = project_dir / s
            # New records bind every existing local input. Missing digests on
            # a legacy/manual record are tolerated only when the path itself
            # is absent; the submit handler reports that as missing input.
            if source.exists():
                if not isinstance(expected, str) or source_digest(source) != expected:
                    return None
        # needs_hf drives intentional HF-credential injection under the
        # trusted-pipeline contract. Keep the observational flag bound to a
        # pipeline that at least names hf_dataset so ordinary pipelines do
        # not receive unrelated credentials accidentally.
        needs_hf = bool(entry.get("needs_hf", False))
        if needs_hf and b"hf_dataset" not in script_path.read_bytes():
            needs_hf = False
        return ValidationRecord(
            sha256=entry["sha256"],
            validated_at=str(entry.get("validated_at", "")),
            num_samples=int(entry.get("num_samples", 0)),
            succeeded=int(entry.get("succeeded", 0)),
            failed=int(entry.get("failed", 0)),
            source_paths=source_paths,
            source_digests={str(k): str(v) for k, v in raw_digests.items()},
            needs_hf=needs_hf,
        )
    except (TypeError, ValueError):
        return None
