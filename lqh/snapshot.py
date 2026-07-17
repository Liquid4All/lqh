"""Cloud project snapshot cache.

Wraps the read APIs in ``lqh.project_meta`` with the persistence layer the
"reopen the laptop" path needs (see PERSISTENCY_PLAN.md):

* one short-timeout fetch at TUI startup;
* a sanitized copy cached at ``.lqh/snapshot.json`` (atomic write) so an
  offline reopen still has the last known cloud state, clearly labeled
  stale;
* the ``summary`` tool renders from the cache only — it never touches the
  network.

The cloud project key is the directory basename — the same key cloud job
submits use (stable backend IDs arrive in Phase 3 of the persistency
plan).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from lqh.fsio import atomic_write_json
from lqh.project_meta import (
    fetch_deployments,
    fetch_project_artifacts,
    fetch_snapshot,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Keys whose values must never be cached locally (signed URLs, secrets).
_SENSITIVE_KEY_MARKERS = ("url", "token", "signature", "secret", "key")


def _snapshot_path(project_dir: Path) -> Path:
    return project_dir / ".lqh" / "snapshot.json"


def sanitize(value: Any) -> Any:
    """Recursively drop dict keys that look like URLs/credentials."""
    if isinstance(value, dict):
        return {
            k: sanitize(v)
            for k, v in value.items()
            if not any(marker in k.lower() for marker in _SENSITIVE_KEY_MARKERS)
        }
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def read_cached_snapshot(project_dir: Path) -> dict | None:
    """Return the cached snapshot wrapper, or None.

    Tolerates the legacy unwrapped format (bare backend payload) by
    wrapping it with unknown freshness.
    """
    path = _snapshot_path(project_dir)
    if not path.exists():
        return None
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("cached snapshot unreadable; ignoring", exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    if "snapshot" in data and "schema_version" in data:
        return data
    # Legacy write_local_snapshot format: the payload itself.
    return {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": None,
        "project_key": project_dir.name,
        "snapshot": data,
    }


async def fetch_and_cache_snapshot(
    project_dir: Path, *, timeout: float = 3.0
) -> tuple[dict | None, bool]:
    """Fetch the cloud snapshot and cache it; fall back to the cache.

    Returns ``(wrapper, fresh)``:

    * success → freshly wrapped snapshot (jobs/spend/best plus the
      project artifact list and deployment state), ``fresh=True``;
    * 404 (no cloud activity for this project) → ``(None, True)`` — an
      authoritative answer; any stale cache is REMOVED so old cloud
      facts cannot resurface as current;
    * auth/network/timeout failure → last cached wrapper (or None),
      ``fresh=False``.
    """
    import asyncio

    # One wall-clock budget: the three requests run concurrently, each
    # bounded by the same timeout, so the whole refresh takes ~timeout —
    # not 3× it.
    payload, artifacts_result, deployments_result = await asyncio.gather(
        fetch_snapshot(project_dir.name, timeout=timeout),
        fetch_project_artifacts(project_dir.name, timeout=timeout),
        fetch_deployments(timeout=timeout),
        return_exceptions=True,
    )

    if isinstance(payload, BaseException):
        if (
            isinstance(payload, httpx.HTTPStatusError)
            and payload.response.status_code == 404
        ):
            # Authoritative: this project has no cloud state. A leftover
            # cache would feed obsolete jobs/spend into summary forever.
            try:
                _snapshot_path(project_dir).unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove stale snapshot cache", exc_info=True)
            return None, True
        # Offline, not logged in, timeout, DNS, 5xx… — every one of these
        # means "use the cache and label it stale".
        logger.warning("snapshot fetch failed; using cache: %r", payload)
        return read_cached_snapshot(project_dir), False

    # Enrichment is best-effort — but a failed section must carry the
    # previously cached facts forward (marked stale), never erase them.
    previous = read_cached_snapshot(project_dir) or {}
    stale_sections: list[str] = []
    if isinstance(artifacts_result, BaseException):
        logger.warning("artifact list fetch failed; keeping cached: %r", artifacts_result)
        artifacts = previous.get("artifacts")
        stale_sections.append("artifacts")
    else:
        artifacts = sanitize(artifacts_result)
    if isinstance(deployments_result, BaseException):
        logger.warning("deployments fetch failed; keeping cached: %r", deployments_result)
        deployments = previous.get("deployments")
        stale_sections.append("deployments")
    else:
        # /v1/deployments is account-wide; keep only rows attributed to
        # this project (unattributed rows are kept — they may be this
        # project's pre-stamping deployments, and hiding them risks a
        # duplicate redeploy).
        deployments = sanitize([
            d for d in deployments_result
            if not isinstance(d, dict)
            or not d.get("project_id")
            or d.get("project_id") == project_dir.name
        ])

    wrapper = {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_key": project_dir.name,
        "snapshot": sanitize(payload),
        "artifacts": artifacts,
        "deployments": deployments,
        "stale_sections": stale_sections,
    }
    try:
        atomic_write_json(_snapshot_path(project_dir), wrapper)
    except OSError:
        logger.warning("could not cache snapshot", exc_info=True)
    return wrapper, True
