"""Stable project identity (Phase 3 of PERSISTENCY_PLAN.md).

``.lqh/project.json`` is the project's identity file (schema v3):

```json
{
  "schema_version": 3,
  "project_id": "stable-uuid",
  "display_name": "support-triage",
  "cloud_project_id": "stable-uuid-or-null",
  "legacy_cloud_name": "basename-at-first-sight-or-null",
  "forked_from": null,
  "last_seen_path": "/abs/path/of/project",
  "last_seen_hostname": "laptop",
  "copy_decision": null
}
```

Rules (R4):

* Identity creation is UNCONDITIONAL — never gated on telemetry consent
  or authentication (telemetry delegates here, not the other way round).
* The cloud key is ``cloud_project_id`` when set. For unmigrated legacy
  projects the fallback is ``legacy_cloud_name`` — the basename recorded
  the FIRST time the identity saw the project — never the *current*
  basename, so renaming an unmigrated folder cannot orphan its
  basename-keyed cloud history. The basename remains the DISPLAY name.
* Brand-new projects adopt their UUID as the cloud key immediately —
  a fresh project must never inherit a same-named stranger's cloud
  namespace.
* A corrupt identity file is an ERROR (``ProjectIdentityError``), never
  silently replaced — overwriting it would rotate the stable ID and
  permanently disconnect the directory from its cloud history. Cloud
  key resolution fails closed for the same reason: no basename fallback
  on failure.
* ``last_seen_path`` + ``last_seen_hostname`` detect folder copies: two
  directories sharing one ``project_id`` require an explicit
  continue-vs-fork decision (recorded in ``copy_decision``); a plain
  move/rename on the same machine continues automatically. When the
  recorded path is unreachable AND the hostname differs, we cannot rule
  out a copy — the caller must ask.
* Forking detaches inherited cloud state: the snapshot cache and
  seen-job records are deleted (safe derived caches) and per-run cloud
  markers are renamed ``*.pre-fork`` so the fork can never display,
  watch, or finalize jobs belonging to its parent.
"""

from __future__ import annotations

import json
import logging
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lqh.fsio import atomic_write_json, file_lock

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

# Per-run files that bind a run directory to a cloud/remote job owned by
# a project identity. A fork renames them to ``<name>.pre-fork`` so they
# stay inspectable but inert.
_RUN_CLOUD_MARKERS = (
    "remote_job.json",
    "cloud_state.json",
    "submit_intent.json",
    ".lqh_data_gen.json",
)


class ProjectIdentityError(Exception):
    """The identity file exists but cannot be trusted (corrupt/unreadable),
    or identity storage is unavailable. Cloud operations must fail closed
    on this — falling back to the directory basename would recreate the
    shared-namespace collisions Phase 3 exists to remove."""


def _identity_path(project_dir: Path) -> Path:
    return project_dir / ".lqh" / "project.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hostname() -> str:
    try:
        return platform.node() or "unknown"
    except Exception:  # pragma: no cover - platform quirk
        return "unknown"


def _meaningful_artifacts(project_dir: Path) -> bool:
    """Whether the directory already contains real project work."""
    if (project_dir / "SPEC.md").exists():
        return True
    for name in ("data_gen", "datasets", "runs", "other_specs", "evals", "prompts"):
        directory = project_dir / name
        if directory.is_dir() and any(directory.iterdir()):
            return True
    return False


def _read_identity(project_dir: Path) -> dict[str, Any] | None:
    """Read the identity file.

    Returns None only when the file does not exist. A file that exists
    but cannot be parsed/validated raises ``ProjectIdentityError`` —
    treating corruption as absence would let ``ensure_identity`` mint a
    fresh UUID and silently disconnect the directory from its history.
    """
    path = _identity_path(project_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ProjectIdentityError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("identity is not a JSON object")
        if not data.get("project_id"):
            raise ValueError("identity has no project_id")
        uuid.UUID(str(data["project_id"]))  # reject corrupt ids
        int(data.get("schema_version", 1))
    except (ValueError, TypeError) as exc:
        raise ProjectIdentityError(
            f"corrupt project identity file {path} ({exc}). Refusing to "
            "replace it — that would mint a new project ID and disconnect "
            "this directory from its cloud history. Fix or restore the "
            "file (it is small JSON), or delete it ONLY if this project "
            "has no cloud history worth keeping."
        ) from exc
    return data


def _write_identity(project_dir: Path, identity: dict[str, Any]) -> None:
    path = _identity_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, identity)


def _upgraded(existing: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    """Upgrade a v1/v2 identity dict to v3 IN PLACE semantics.

    Every unrecognized key is preserved verbatim (telemetry fields,
    pipeline_readiness, spec_capture, …) — migration must never drop
    state other writers persisted here.
    """
    upgraded = dict(existing)
    version = int(existing.get("schema_version", 1))
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded.setdefault("display_name", project_dir.name)
    upgraded.setdefault("cloud_project_id", None)
    upgraded.setdefault("forked_from", None)
    upgraded.setdefault("last_seen_path", str(project_dir.resolve()))
    upgraded.setdefault("last_seen_hostname", _hostname())
    upgraded.setdefault("copy_decision", None)
    if "legacy_cloud_name" not in upgraded:
        # Recorded once, at the moment the identity first covers cloud
        # keying. Pre-v3 projects have existed for a while → their
        # basename-keyed cloud history (if any) is keyed by the CURRENT
        # basename as of this upgrade; unmigrated projects keep using it
        # even if the folder is renamed later.
        upgraded["legacy_cloud_name"] = (
            None if upgraded.get("cloud_project_id") else project_dir.name
        )
    if version < 2:
        # v1 (telemetry-era) predates cloud keying entirely.
        upgraded.setdefault("legacy_cloud_name", project_dir.name)
    return upgraded


def ensure_identity(project_dir: Path) -> tuple[dict[str, Any], str]:
    """Read-or-create the identity file; migrate v1/v2 in place.

    Returns ``(identity, classification)`` with classification one of
    ``"new"`` (fresh empty directory), ``"pre_existing"`` (artifacts but
    no identity yet — a legacy project), or ``"reopened"``.

    Raises ``ProjectIdentityError`` when an existing identity file is
    corrupt (never overwrites it) or storage fails.
    """
    project_dir = Path(project_dir)
    try:
        (project_dir / ".lqh").mkdir(parents=True, exist_ok=True)
        with file_lock(project_dir / ".lqh" / "project.lock"):
            existing = _read_identity(project_dir)
            if existing is not None:
                if int(existing.get("schema_version", 1)) < SCHEMA_VERSION:
                    existing = _upgraded(existing, project_dir)
                    _write_identity(project_dir, existing)
                return existing, "reopened"

            classification = (
                "pre_existing" if _meaningful_artifacts(project_dir) else "new"
            )
            project_id = str(uuid.uuid4())
            identity: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "project_id": project_id,
                "display_name": project_dir.name,
                # Fresh projects adopt the UUID as their cloud key at
                # birth; pre-existing projects may have basename-keyed
                # cloud history — record that basename NOW so a later
                # folder rename cannot orphan it.
                "cloud_project_id": project_id if classification == "new" else None,
                "legacy_cloud_name": (
                    None if classification == "new" else project_dir.name
                ),
                "forked_from": None,
                "last_seen_path": str(project_dir.resolve()),
                "last_seen_hostname": _hostname(),
                "copy_decision": None,
            }
            _write_identity(project_dir, identity)
            return identity, classification
    except ProjectIdentityError:
        raise
    except OSError as exc:
        raise ProjectIdentityError(
            f"project identity storage unavailable under {project_dir}: {exc}"
        ) from exc


def cloud_project_key(project_dir: Path) -> str:
    """The key for every cloud project/artifact/deployment operation.

    Stable UUID when the project has one adopted/migrated; the recorded
    ``legacy_cloud_name`` as the fallback for unmigrated projects (NOT
    the current basename — a renamed-but-unmigrated folder must keep
    addressing its original cloud namespace).

    Fails closed: raises ``ProjectIdentityError`` when the identity is
    corrupt or unavailable. Callers doing cloud work surface the error;
    silently falling back to the basename would reintroduce namespace
    collisions.
    """
    project_dir = Path(project_dir)
    identity, _ = ensure_identity(project_dir)
    return (
        identity.get("cloud_project_id")
        or identity.get("legacy_cloud_name")
        or project_dir.name
    )


def project_uuid(project_dir: Path) -> str:
    """The stable identity UUID (NOT the cloud key — this one never
    changes across migration). Recorded into job/submit markers so a
    marker copied into another project's directory is recognizably
    foreign instead of silently acted on."""
    identity, _ = ensure_identity(Path(project_dir))
    return identity["project_id"]


def marker_is_foreign(project_dir: Path, marker: Any) -> bool:
    """Whether a job/submit marker belongs to a DIFFERENT project identity.

    Markers written since Phase 3 carry ``owner_project_id``; a marker
    without one is treated as local (pre-Phase-3 files). Foreign markers
    must be skipped by watchers/signals/finalizers — acting on them would
    observe or mutate another project's jobs. Fork already renames
    inherited markers to ``*.pre-fork``; this guards the remaining path
    (run dirs copied around by hand)."""
    if not isinstance(marker, dict):
        return False
    owner = marker.get("owner_project_id")
    if not owner:
        return False
    try:
        return owner != project_uuid(project_dir)
    except Exception:
        # Identity unavailable — don't invent foreignness.
        return False


def adopt_cloud_id(project_dir: Path, cloud_project_id: str) -> None:
    """Record the migrated/adopted cloud key (atomic, locked)."""
    project_dir = Path(project_dir)
    # NOTE: ensure_identity takes the same flock on its own fd — it must
    # run BEFORE we acquire the lock (nesting would self-deadlock).
    ensure_identity(project_dir)
    with file_lock(project_dir / ".lqh" / "project.lock"):
        identity = _read_identity(project_dir)
        if identity is None:
            return
        identity["cloud_project_id"] = cloud_project_id
        _write_identity(project_dir, identity)


def detect_copy(project_dir: Path) -> str:
    """Classify the directory relative to its recorded path.

    Returns:

    * ``"same"`` — path matches the record (or no record yet);
    * ``"moved"`` — same machine, recorded path gone: a rename/move,
      identity continues automatically (the record is updated);
    * ``"copied"`` — an explicit continue-vs-fork decision is needed:
      the recorded path still exists with the same project_id (two live
      directories share one identity), OR the recorded location cannot
      be verified (different hostname, unreadable original) so a copy
      cannot be ruled out.
    """
    project_dir = Path(project_dir)
    identity = _read_identity(project_dir)
    if identity is None:
        return "same"
    recorded = identity.get("last_seen_path")
    current = str(project_dir.resolve())
    if not recorded or recorded == current:
        if recorded != current:
            record_path(project_dir)
        return "same"
    old = Path(recorded)
    try:
        old_exists = old.exists()
    except OSError:
        old_exists = False
    if old_exists:
        try:
            other = json.loads(
                (old / ".lqh" / "project.json").read_text(encoding="utf-8")
            )
            if other.get("project_id") == identity.get("project_id"):
                return "copied"
            # The recorded path now hosts a DIFFERENT project — ours was
            # moved here and something else took the old spot.
            record_path(project_dir)
            return "moved"
        except (OSError, ValueError):
            # The old path exists but its identity is unreadable — a
            # live copy cannot be ruled out. Ask.
            return "copied"
    # Recorded path is gone. On the same machine that is a move/rename;
    # on a different machine we cannot see the original at all, so a
    # copy (rsync/scp to another box) cannot be ruled out.
    recorded_host = identity.get("last_seen_hostname")
    if recorded_host and recorded_host != _hostname():
        return "copied"
    record_path(project_dir)
    return "moved"


def record_path(project_dir: Path) -> None:
    """Update last_seen_path/hostname after a move."""
    project_dir = Path(project_dir)
    with file_lock(project_dir / ".lqh" / "project.lock"):
        identity = _read_identity(project_dir)
        if identity is None:
            return
        identity["last_seen_path"] = str(project_dir.resolve())
        identity["last_seen_hostname"] = _hostname()
        identity["display_name"] = project_dir.name
        _write_identity(project_dir, identity)


def record_continue_decision(project_dir: Path) -> None:
    """Record an explicit user 'continue' choice for a copied directory.

    Persists the decision (the plan requires the choice to be recorded,
    not merely acted on) and updates the location record so the copy
    becomes the identity's current home.
    """
    project_dir = Path(project_dir)
    with file_lock(project_dir / ".lqh" / "project.lock"):
        identity = _read_identity(project_dir)
        if identity is None:
            return
        identity["copy_decision"] = {
            "choice": "continue",
            "at": _now(),
            "previous_path": identity.get("last_seen_path"),
        }
        identity["last_seen_path"] = str(project_dir.resolve())
        identity["last_seen_hostname"] = _hostname()
        identity["display_name"] = project_dir.name
        _write_identity(project_dir, identity)


def _detach_inherited_cloud_state(project_dir: Path) -> None:
    """Make cloud state copied from a parent project inert.

    Runs BEFORE the fork identity is written: if any step fails, the
    fork is aborted and the directory stays in the copied state (the
    prompt reappears next open). Caches are deleted (both are derived
    and safe to delete by design); per-run job markers are renamed
    ``*.pre-fork`` so they remain inspectable but can never be picked
    up by watchers, signals, or finalization again.
    """
    for cache in (".lqh/snapshot.json", ".lqh/job_seen.json"):
        (project_dir / cache).unlink(missing_ok=True)
    runs_dir = project_dir / "runs"
    if runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for marker in _RUN_CLOUD_MARKERS:
                source = run_dir / marker
                if source.exists():
                    source.rename(run_dir / f"{marker}.pre-fork")


def fork_identity(project_dir: Path) -> dict[str, Any]:
    """Mint a NEW identity for a copied directory (explicit fork).

    The fork gets a fresh UUID and a fresh cloud namespace (its cloud
    key is the new UUID immediately — the original's cloud history stays
    with the original), with ``forked_from`` provenance. Inherited cloud
    state (snapshot cache, seen-job records, per-run job markers) is
    detached first so the fork can never observe or act on the parent's
    jobs; if detaching fails the fork is aborted.
    """
    project_dir = Path(project_dir)
    with file_lock(project_dir / ".lqh" / "project.lock"):
        try:
            old = _read_identity(project_dir)
        except ProjectIdentityError:
            old = None
        _detach_inherited_cloud_state(project_dir)
        new_id = str(uuid.uuid4())
        identity = {
            "schema_version": SCHEMA_VERSION,
            "project_id": new_id,
            "display_name": project_dir.name,
            "cloud_project_id": new_id,
            "legacy_cloud_name": None,
            "forked_from": (old or {}).get("project_id"),
            "last_seen_path": str(project_dir.resolve()),
            "last_seen_hostname": _hostname(),
            "copy_decision": {
                "choice": "fork",
                "at": _now(),
                "previous_path": (old or {}).get("last_seen_path"),
            },
        }
        _write_identity(project_dir, identity)
        return identity


async def migrate_cloud_identity(project_dir: Path) -> str | None:
    """One-time authenticated cutover from the legacy name to the UUID.

    Returns the resulting cloud key, or None when nothing could be
    decided (offline/unauthenticated/backend refused) — the caller keeps
    using the legacy fallback and retries on a later startup.

    ONE idempotent rename call against the RECORDED legacy name (never
    the current basename — the folder may have been renamed since):

    * 2xx → history (project row, or row-less artifacts/deployments)
      moved to the UUID → adopt. The backend records the rename, so a
      lost response or a racing continued copy sees the retry succeed
      instead of 404/409 — which is also why a transport error is
      retried once here rather than deferred a whole session (a
      committed-but-unacknowledged rename would otherwise leave THIS
      session writing under the basename and recreating the namespace).
    * 404 → truly nothing under the legacy name → adopt directly.
    * 409 (active jobs / conflicting target) → defer, retry next start.

    Probing GET /projects/{legacy} first would be wrong: artifacts and
    deployments can exist WITHOUT a projects row (the submit-time upsert
    is best-effort), and a 404 probe would orphan them under the
    basename forever.
    """
    import httpx

    from lqh.project_meta import rename_project

    project_dir = Path(project_dir)
    identity, _ = ensure_identity(project_dir)
    if identity.get("cloud_project_id"):
        return identity["cloud_project_id"]

    stable_id = identity["project_id"]
    legacy = identity.get("legacy_cloud_name") or project_dir.name
    try:
        try:
            for attempt in (0, 1):
                try:
                    await rename_project(legacy, stable_id, timeout=10.0)
                    break
                except httpx.TransportError:
                    if attempt:
                        raise
                    logger.warning(
                        "rename response lost; retrying once (idempotent)"
                    )
            has_history = True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            # Only OUR handler's 404 is authoritative ("nothing under the
            # legacy name"). A backend without the rename route (mid-
            # deploy, rollback) or an intermediary also answers 404 — as
            # a bare non-JSON body. Adopting on that would orphan real
            # basename history, so defer instead.
            try:
                if "error" not in exc.response.json():
                    raise ValueError("no error envelope")
            except ValueError:
                logger.warning(
                    "rename route answered a non-API 404 (old backend/"
                    "proxy?); deferring migration"
                )
                return None
            has_history = False

        adopt_cloud_id(project_dir, stable_id)
        logger.info(
            "cloud project identity migrated: %s -> %s (history=%s)",
            legacy, stable_id, has_history,
        )
        return stable_id
    except Exception:
        # Offline, unauthenticated, active jobs (409), target exists…
        # — stay on the legacy key and retry on a later startup.
        logger.warning("cloud identity migration deferred", exc_info=True)
        return None
