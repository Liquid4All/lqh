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


def _sensitive_value(value: Any) -> bool:
    """Value-shape check: catches URLs/credentials under unexpected keys."""
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return (
        lowered.startswith(("http://", "https://"))
        or lowered.startswith("bearer ")
        or lowered.startswith("eyj")  # JWT-shaped
    )


def sanitize(value: Any) -> Any:
    """Recursively drop entries that look like URLs/credentials.

    Both key-name markers AND value shapes are checked — key heuristics
    alone would let a signed URL survive under an innocent key.
    """
    if isinstance(value, dict):
        return {
            k: sanitize(v)
            for k, v in value.items()
            if not any(marker in k.lower() for marker in _SENSITIVE_KEY_MARKERS)
            and not _sensitive_value(v)
        }
    if isinstance(value, list):
        return [sanitize(v) for v in value if not _sensitive_value(v)]
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
    # Legacy write_local_snapshot format: the payload itself, possibly
    # written before sanitization existed — scrub it on the way in.
    return {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": None,
        "project_key": project_dir.name,
        "snapshot": sanitize(data),
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
    # bounded by the same timeout — and the WHOLE operation is bounded by
    # an outer deadline too. Per-request httpx timeouts don't cover every
    # stall (slow trickling bodies, misbehaving mocks/proxies), and CLI
    # startup must never hang on this.
    try:
        payload, artifacts_result, deployments_result = await asyncio.wait_for(
            asyncio.gather(
                fetch_snapshot(project_dir.name, timeout=timeout),
                fetch_project_artifacts(project_dir.name, timeout=timeout),
                fetch_deployments(timeout=timeout),
                return_exceptions=True,
            ),
            timeout=timeout * 2,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning("snapshot refresh exceeded its deadline; using cache")
        return read_cached_snapshot(project_dir), False

    def _scoped_deployments(rows: Any) -> list | None:
        if not isinstance(rows, list):
            return None
        return sanitize([
            d for d in rows
            if not isinstance(d, dict)
            or not d.get("project_id")
            or d.get("project_id") == project_dir.name
        ])

    if isinstance(payload, BaseException):
        if (
            isinstance(payload, httpx.HTTPStatusError)
            and payload.response.status_code == 404
        ):
            # Authoritative: this project has no cloud jobs/spend. A
            # leftover cache would feed obsolete facts into summary
            # forever — but deployment state must not vanish with it:
            # freshly fetched rows are kept, and a FAILED deployment
            # refresh carries the previously cached rows forward marked
            # stale (hiding a possibly-live deployment risks a duplicate
            # redeploy).
            previous_cache = read_cached_snapshot(project_dir) or {}
            try:
                _snapshot_path(project_dir).unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove stale snapshot cache", exc_info=True)
            stale: list[str] = []
            deployments = _scoped_deployments(deployments_result)
            if deployments is None and previous_cache.get("deployments"):
                deployments = previous_cache.get("deployments")
                stale = ["deployments"]
            if deployments:
                wrapper = {
                    "schema_version": SCHEMA_VERSION,
                    "fetched_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "project_key": project_dir.name,
                    "snapshot": {},
                    "artifacts": [],
                    "deployments": deployments,
                    "stale_sections": stale,
                }
                try:
                    atomic_write_json(_snapshot_path(project_dir), wrapper)
                except OSError:
                    logger.warning("could not cache snapshot", exc_info=True)
                return wrapper, True
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
        deployments = _scoped_deployments(deployments_result)

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
