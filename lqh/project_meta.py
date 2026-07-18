"""Client-side helpers for the backend `projects` and lineage APIs.

`projects` rows are owned by the backend — the client never POSTs to
create one; the row is upserted whenever a cloud job is submitted with
metadata. What this module *does* provide:

  * `compute_spec_sha256(project_dir)` — hash of SPEC.md, attached to
    cloud-submit meta so the backend can detect spec drift across
    submits.
  * `gather_project_meta(project_dir, config)` — best-effort extract
    of display_name / base_model / reward_model from the on-disk
    project so the TUI doesn't have to wire them everywhere.
  * `fetch_snapshot(project_id)` / `fetch_lineage(...)` — read APIs
    for the "reopen the laptop, rebuild the world" reconstruction
    path.

All HTTP helpers are async (httpx) and pick up bearer auth via
``lqh.auth.require_token`` so they work in the TUI process and inside
a cloud sandbox (where ``LQH_API_TOKEN`` is set by the launcher).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from lqh.auth import api_root, require_token

logger = logging.getLogger(__name__)


__all__ = [
    "ProjectMeta",
    "compute_spec_sha256",
    "gather_project_meta",
    "fetch_snapshot",
    "fetch_lineage",
]


@dataclass
class ProjectMeta:
    """Bundle of optional project metadata sent on cloud-submit. Each
    field is independent — the backend MERGEs non-empty values into
    the existing projects row (COALESCE), so missing values never
    overwrite previously-recorded ones."""

    display_name: str | None = None
    spec_sha256: str | None = None
    base_model: str | None = None
    reward_model: str | None = None

    def to_meta_dict(self) -> dict[str, str]:
        """Render for inclusion in the /v1/cloud/jobs `meta` JSON."""
        out: dict[str, str] = {}
        if self.display_name:
            out["display_name"] = self.display_name
        if self.spec_sha256:
            out["spec_sha256"] = self.spec_sha256
        if self.base_model:
            out["base_model"] = self.base_model
        if self.reward_model:
            out["reward_model"] = self.reward_model
        return out


def compute_spec_sha256(project_dir: Path | str) -> str | None:
    """Return sha256 of ``<project_dir>/SPEC.md``, or None if missing.

    Used so cloud-submit meta carries the spec hash forward into the
    projects row; the client can later compare against the local
    SPEC.md to warn the user if they edited it after submitting.
    """
    p = Path(project_dir) / "SPEC.md"
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gather_project_meta(
    project_dir: Path | str,
    config: dict[str, Any] | None = None,
) -> ProjectMeta:
    """Best-effort extraction of project metadata from the on-disk
    project + the cloud-submit config dict.

    Heuristics:
      * display_name = the project folder basename when nothing
        better is available
      * spec_sha256  = hash of SPEC.md if present
      * base_model   = config["base_model"] (SFT/DPO/eval all use
        the same key) or config["model"] as a fallback
      * reward_model = config["reward_model"] or
        config.get("scoring", {}).get("model")

    Missing fields stay None; the backend treats None as "don't
    overwrite the previously-recorded value".
    """
    project_dir = Path(project_dir)
    cfg = config or {}

    base = cfg.get("base_model") or cfg.get("model")
    reward = cfg.get("reward_model")
    if reward is None:
        scoring = cfg.get("scoring") or {}
        if isinstance(scoring, dict):
            reward = scoring.get("model") or scoring.get("judge")

    return ProjectMeta(
        display_name=project_dir.name or None,
        spec_sha256=compute_spec_sha256(project_dir),
        base_model=str(base) if base else None,
        reward_model=str(reward) if reward else None,
    )


# ---------------------------------------------------------------------
# Read API: project snapshot + lineage
# ---------------------------------------------------------------------


def _auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token or require_token()}"}


async def fetch_snapshot(
    project_id: str,
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """GET /v1/projects/{project_id} → snapshot dict.

    The snapshot includes project metadata, the most recent cloud
    jobs, lifetime spend, and (optionally) the best checkpoint. Raises
    httpx.HTTPStatusError on non-2xx; the caller decides how to handle
    a 404 (probably "no cloud activity yet for this project").
    """
    base = (api_base or api_root()).rstrip("/")
    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        r = await client.get(
            f"/v1/projects/{project_id}",
            headers=_auth_headers(token),
        )
        r.raise_for_status()
        return r.json()


async def fetch_project_artifacts(
    project_id: str,
    *,
    limit: int = 25,
    api_base: str | None = None,
    token: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """GET /v1/projects/{project_id}/artifacts → newest-first raw items."""
    base = (api_base or api_root()).rstrip("/")
    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        r = await client.get(
            f"/v1/projects/{project_id}/artifacts",
            params={"limit": str(limit)},
            headers=_auth_headers(token),
        )
        r.raise_for_status()
        items = r.json().get("artifacts", [])
        return items if isinstance(items, list) else []


async def fetch_deployments(
    *,
    api_base: str | None = None,
    token: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """GET /v1/deployments → raw deployment rows for the caller's account."""
    base = (api_base or api_root()).rstrip("/")
    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        r = await client.get("/v1/deployments", headers=_auth_headers(token))
        r.raise_for_status()
        items = r.json().get("deployments", [])
        return items if isinstance(items, list) else []


async def fetch_lineage(
    project_id: str,
    *,
    artifact_id: str | None = None,
    api_base: str | None = None,
    token: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """GET /v1/projects/{project_id}/lineage → dict.

    When ``artifact_id`` is None, returns ``{"lineage": [...]}``;
    otherwise returns a single lineage row. The caller is expected
    to disambiguate by checking the shape (the backend's response
    schema is oneOf).
    """
    base = (api_base or api_root()).rstrip("/")
    params: dict[str, str] = {}
    if artifact_id:
        params["artifact_id"] = artifact_id
    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        r = await client.get(
            f"/v1/projects/{project_id}/lineage",
            params=params,
            headers=_auth_headers(token),
        )
        r.raise_for_status()
        return r.json()


def write_local_snapshot(project_dir: Path | str, snapshot: dict[str, Any]) -> Path:
    """Persist a snapshot dict under ``<project_dir>/.lqh/snapshot.json``.

    Prefer ``lqh.snapshot.fetch_and_cache_snapshot`` — it wraps, scopes,
    and enriches. This low-level writer is kept for direct payloads; it
    sanitizes (no signed URLs/credentials on disk) and writes atomically.
    """
    from lqh.fsio import atomic_write_json
    from lqh.snapshot import sanitize

    project_dir = Path(project_dir)
    target = project_dir / ".lqh" / "snapshot.json"
    atomic_write_json(target, sanitize(snapshot))
    return target
