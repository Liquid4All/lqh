"""Cloud project snapshot cache.

Wraps the read APIs in ``lqh.project_meta`` with the persistence layer the
"reopen the laptop" path needs (see PERSISTENCY_PLAN.md):

* one short-timeout fetch at TUI startup (jobs and artifacts are paged,
  with explicit truncation flags when the client cap is hit);
* a sanitized copy cached at ``.lqh/snapshot.json`` (atomic write) so an
  offline reopen still has the last known cloud state, clearly labeled
  stale;
* the ``summary`` tool renders from the cache only — it never touches the
  network.

The cloud project key comes from ``lqh.project_identity.cloud_project_key``:
the stable project UUID once adopted/migrated, or the recorded legacy
basename for unmigrated pre-Phase-3 projects. A cached snapshot is only
honored when it was written for the SAME key — a forked or migrated
project must never read a cache inherited from another identity.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from lqh.fsio import atomic_write_json
from lqh.project_identity import cloud_project_key
from lqh.project_meta import (
    fetch_deployments,
    fetch_project_artifacts,
    fetch_snapshot,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Keys whose values must never be cached locally (signed URLs, secrets).
_SENSITIVE_KEY_MARKERS = ("url", "token", "signature", "secret", "key")

# Paging: sections are fetched in pages of _PAGE_SIZE up to _MAX_ITEMS
# per section; hitting the cap sets an explicit *_truncated flag so the
# summary can say what was omitted instead of silently capping.
_PAGE_SIZE = 100
_MAX_ITEMS = 500


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

    A wrapper recorded for a DIFFERENT project key is ignored: after a
    fork or an identity migration, facts cached under the old key must
    not resurface as this project's cloud state.

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
        cached_key = data.get("project_key")
        current_key = cloud_project_key(project_dir)
        if cached_key and cached_key != current_key:
            logger.warning(
                "cached snapshot belongs to project key %r (current %r); ignoring",
                cached_key, current_key,
            )
            return None
        return data
    # Legacy write_local_snapshot format: the payload itself, possibly
    # written before sanitization existed — scrub it on the way in.
    return {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": None,
        "project_key": cloud_project_key(project_dir),
        "snapshot": sanitize(data),
    }


async def _fetch_snapshot_paged(project_key: str, timeout: float) -> dict[str, Any]:
    """Snapshot with the jobs section paged in.

    The FIRST request's failure propagates unchanged (the caller's 404
    handling depends on it); a failure while paging keeps the jobs
    collected so far and marks the section truncated.
    """
    payload = await fetch_snapshot(
        project_key, timeout=timeout, jobs_limit=_PAGE_SIZE, jobs_offset=0
    )
    jobs = list(payload.get("jobs") or [])
    has_more = bool(payload.get("jobs_has_more"))
    truncated = False
    try:
        while has_more and len(jobs) < _MAX_ITEMS:
            page_payload = await fetch_snapshot(
                project_key,
                timeout=timeout,
                jobs_limit=_PAGE_SIZE,
                jobs_offset=len(jobs),
            )
            page = page_payload.get("jobs") or []
            if not page:
                break
            jobs.extend(page)
            has_more = bool(page_payload.get("jobs_has_more"))
    except Exception:
        logger.warning("job paging failed; keeping partial job list", exc_info=True)
        truncated = True
    payload["jobs"] = jobs[:_MAX_ITEMS]
    payload["jobs_truncated"] = truncated or (has_more and len(jobs) >= _MAX_ITEMS)
    return payload


async def _fetch_artifacts_paged(
    project_key: str, timeout: float
) -> dict[str, Any]:
    """Artifact list paged to _MAX_ITEMS. First-page failure propagates
    (the caller carries the cached section forward); later-page failure
    keeps the partial list, marked truncated.

    Exactly-at-the-cap is NOT assumed truncated: the loop probes one
    page past the cap, so the flag is set only when more items actually
    exist (or a paging failure left the tail unknown)."""
    items: list[Any] = []
    truncated = False
    try:
        while True:
            page = await fetch_project_artifacts(
                project_key, limit=_PAGE_SIZE, offset=len(items), timeout=timeout
            )
            items.extend(page)
            if len(page) < _PAGE_SIZE or len(items) > _MAX_ITEMS:
                break
    except Exception:
        if not items:
            raise
        logger.warning(
            "artifact paging failed; keeping partial list", exc_info=True
        )
        truncated = True
    truncated = truncated or len(items) > _MAX_ITEMS
    return {"items": items[:_MAX_ITEMS], "truncated": truncated}


def _split_deployments(rows: Any, project_key: str) -> tuple[list, list] | None:
    """Split account-wide deployment rows into (this project's,
    unattributed). Rows attributed to OTHER projects are dropped;
    unattributed rows are kept but in their own bucket — they may
    predate project stamping, and hiding a possibly-live deployment
    risks a duplicate redeploy, but they must never be presented as
    this project's."""
    if not isinstance(rows, list):
        return None
    scoped = [
        d for d in rows
        if not isinstance(d, dict) or d.get("project_id") == project_key
    ]
    unattributed = [
        d for d in rows
        if isinstance(d, dict) and not d.get("project_id")
    ]
    return sanitize(scoped), sanitize(unattributed)


async def fetch_and_cache_snapshot(
    project_dir: Path, *, timeout: float = 3.0
) -> tuple[dict | None, bool]:
    """Fetch the cloud snapshot and cache it; fall back to the cache.

    Returns ``(wrapper, fresh)``:

    * success → freshly wrapped snapshot (paged jobs/spend/best plus the
      paged project artifact list and deployment state), ``fresh=True``;
    * 404 (no cloud activity for this project) → ``(None, True)`` — an
      authoritative answer; any stale cache is REMOVED so old cloud
      facts cannot resurface as current;
    * auth/network/timeout failure → last cached wrapper (or None),
      ``fresh=False``.
    """
    import asyncio

    # One wall-clock budget for the whole refresh. The three sections
    # run concurrently; jobs and artifacts may take up to
    # _MAX_ITEMS/_PAGE_SIZE sequential page requests each, so the outer
    # deadline allows for a full paging pass — per-request httpx
    # timeouts don't cover every stall, and CLI startup must never hang
    # on this.
    project_key = cloud_project_key(project_dir)
    try:
        payload, artifacts_result, deployments_result = await asyncio.wait_for(
            asyncio.gather(
                _fetch_snapshot_paged(project_key, timeout),
                _fetch_artifacts_paged(project_key, timeout),
                fetch_deployments(timeout=timeout),
                return_exceptions=True,
            ),
            timeout=timeout * (2 + _MAX_ITEMS // _PAGE_SIZE),
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning("snapshot refresh exceeded its deadline; using cache")
        return read_cached_snapshot(project_dir), False

    if isinstance(payload, BaseException):
        if (
            isinstance(payload, httpx.HTTPStatusError)
            and payload.response.status_code == 404
        ):
            # Authoritative: this project has no projects row (no cloud
            # jobs/spend). A leftover cache would feed obsolete facts
            # into summary forever — but the OTHER sections must not
            # vanish with it: artifacts and deployments can exist
            # without a projects row (row-less legacy history, failed
            # submit-time upsert), so freshly fetched rows are kept, and
            # a FAILED section refresh carries the previously cached
            # rows forward marked stale (hiding a live deployment or an
            # expensive artifact risks duplicating it).
            previous_cache = read_cached_snapshot(project_dir) or {}
            try:
                _snapshot_path(project_dir).unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove stale snapshot cache", exc_info=True)
            stale: list[str] = []
            if isinstance(artifacts_result, BaseException):
                artifacts = previous_cache.get("artifacts") or []
                artifacts_truncated = bool(previous_cache.get("artifacts_truncated"))
                if artifacts:
                    stale.append("artifacts")
            else:
                artifacts = sanitize(artifacts_result["items"])
                artifacts_truncated = artifacts_result["truncated"]
            split = _split_deployments(deployments_result, project_key)
            if split is None:
                deployments = previous_cache.get("deployments")
                unattributed = previous_cache.get("unattributed_deployments")
                if deployments or unattributed:
                    stale.append("deployments")
            else:
                deployments, unattributed = split
            if artifacts or deployments or unattributed:
                wrapper = {
                    "schema_version": SCHEMA_VERSION,
                    "fetched_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "project_key": project_key,
                    "snapshot": {},
                    "artifacts": artifacts,
                    "artifacts_truncated": artifacts_truncated,
                    "deployments": deployments or [],
                    "unattributed_deployments": unattributed or [],
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
    artifacts_truncated = False
    if isinstance(artifacts_result, BaseException):
        logger.warning("artifact list fetch failed; keeping cached: %r", artifacts_result)
        artifacts = previous.get("artifacts")
        artifacts_truncated = bool(previous.get("artifacts_truncated"))
        stale_sections.append("artifacts")
    else:
        artifacts = sanitize(artifacts_result["items"])
        artifacts_truncated = artifacts_result["truncated"]
    split = _split_deployments(deployments_result, project_key)
    if split is None:
        logger.warning("deployments fetch failed; keeping cached: %r", deployments_result)
        deployments = previous.get("deployments")
        unattributed = previous.get("unattributed_deployments")
        stale_sections.append("deployments")
    else:
        deployments, unattributed = split

    jobs_truncated = bool(payload.pop("jobs_truncated", False))
    wrapper = {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_key": project_key,
        "snapshot": sanitize(payload),
        "artifacts": artifacts,
        "artifacts_truncated": artifacts_truncated,
        "jobs_truncated": jobs_truncated,
        "deployments": deployments or [],
        "unattributed_deployments": unattributed or [],
        "stale_sections": stale_sections,
    }
    try:
        atomic_write_json(_snapshot_path(project_dir), wrapper)
    except OSError:
        logger.warning("could not cache snapshot", exc_info=True)
    return wrapper, True
