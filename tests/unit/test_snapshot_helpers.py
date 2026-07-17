"""Characterization tests for the cloud project snapshot helpers.

Phase 0 of the persistency work (see PERSISTENCY_PLAN.md). The helpers in
``lqh.project_meta`` (fetch_snapshot / fetch_lineage / write_local_snapshot)
are currently dead code — defined but never called by the TUI. These tests
pin their behavior before Phase 2 wires them into startup (via
``lqh/snapshot.py``, which will add caching, sanitization, and offline
fallback on top).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from lqh import project_meta
from lqh.project_meta import (
    compute_spec_sha256,
    fetch_snapshot,
    write_local_snapshot,
)

_SNAPSHOT = {
    "project_id": "myproj",
    "spec_sha256": "ab" * 32,
    "jobs": [{"job_id": "j1", "status": "completed"}],
    "lifetime_spend_micros": 1_500_000,
}

_ARTIFACTS = [
    {"artifact_id": "art-1", "kind": "checkpoint", "download_url": "https://signed"},
]

_DEPLOYMENTS = [
    {"id": "dep-1", "name": "triage-prod", "status": "running"},
    # Belongs to a different project: must be filtered out of the cache.
    {"id": "dep-2", "name": "other", "status": "running", "project_id": "someone-else"},
]


@pytest.fixture
def fake_projects_api(monkeypatch: pytest.MonkeyPatch):
    """Serve the project read APIs from an in-process transport.

    Mirrors the ``fake_cloud`` pattern in test_cloud_backend.py: wrap the
    real AsyncClient so a MockTransport is injected by default.
    """
    state = {"status": 200, "artifacts_status": 200, "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        path = request.url.path
        if path.endswith("/artifacts"):
            if state["artifacts_status"] != 200:
                return httpx.Response(state["artifacts_status"], json={"error": "nope"})
            return httpx.Response(200, json={"artifacts": _ARTIFACTS})
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": "nope"})
        if path == "/v1/deployments":
            return httpx.Response(200, json={"deployments": _DEPLOYMENTS})
        return httpx.Response(200, json=_SNAPSHOT)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(project_meta.httpx, "AsyncClient", _patched)
    return state


async def test_fetch_snapshot_returns_payload(fake_projects_api) -> None:
    snap = await fetch_snapshot("myproj", api_base="http://test", token="tok")

    assert snap == _SNAPSHOT
    request = fake_projects_api["requests"][0]
    assert request.url.path == "/v1/projects/myproj"
    assert request.headers["Authorization"] == "Bearer tok"


async def test_fetch_snapshot_raises_on_404(fake_projects_api) -> None:
    fake_projects_api["status"] = 404

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await fetch_snapshot("ghost", api_base="http://test", token="tok")

    assert exc_info.value.response.status_code == 404


async def test_fetch_snapshot_raises_on_auth_failure(fake_projects_api) -> None:
    fake_projects_api["status"] = 401

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await fetch_snapshot("myproj", api_base="http://test", token="bad")

    assert exc_info.value.response.status_code == 401


def test_write_local_snapshot_creates_cache_file(project_dir: Path) -> None:
    target = write_local_snapshot(project_dir, _SNAPSHOT)

    assert target == project_dir / ".lqh" / "snapshot.json"
    assert json.loads(target.read_text()) == _SNAPSHOT


# ---------------------------------------------------------------------------
# lqh.snapshot: cache, sanitization, offline fallback (Phase 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lqh.project_meta.require_token", lambda: "tok")


async def test_fetch_and_cache_writes_wrapper(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    from lqh.snapshot import fetch_and_cache_snapshot

    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert fresh is True
    assert wrapper["snapshot"] == _SNAPSHOT
    assert wrapper["project_key"] == project_dir.name
    assert wrapper["fetched_at"]
    # Enrichment: project artifacts and deployment state ride along,
    # sanitized (no signed URLs in the cache) and deployment rows scoped
    # to this project (foreign project_id rows dropped, unattributed kept).
    assert wrapper["artifacts"] == [{"artifact_id": "art-1", "kind": "checkpoint"}]
    assert wrapper["deployments"] == [_DEPLOYMENTS[0]]
    assert wrapper["stale_sections"] == []
    cached = json.loads((project_dir / ".lqh" / "snapshot.json").read_text())
    assert cached == wrapper


async def test_partial_refresh_keeps_cached_sections_and_labels_them(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    """A failed artifact/deployment request must carry the previously
    cached list forward and mark the section stale — not erase it while
    reporting the snapshot as fully fresh."""
    from lqh.snapshot import fetch_and_cache_snapshot

    first, _ = await fetch_and_cache_snapshot(project_dir)
    assert first["artifacts"]

    fake_projects_api["artifacts_status"] = 503
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert fresh is True  # the core snapshot IS fresh
    assert wrapper["artifacts"] == first["artifacts"]  # carried forward
    assert wrapper["stale_sections"] == ["artifacts"]
    assert wrapper["deployments"] == [_DEPLOYMENTS[0]]  # this one succeeded


async def test_fetch_404_means_no_cloud_activity(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    from lqh.snapshot import fetch_and_cache_snapshot

    fake_projects_api["status"] = 404
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert wrapper is None
    assert fresh is True
    assert not (project_dir / ".lqh" / "snapshot.json").exists()


async def test_authoritative_404_removes_stale_cache(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    """A 404 is an authoritative 'no cloud state' — an old cache must not
    keep feeding obsolete jobs/spend into summary as current."""
    from lqh.snapshot import fetch_and_cache_snapshot, read_cached_snapshot

    await fetch_and_cache_snapshot(project_dir)
    assert (project_dir / ".lqh" / "snapshot.json").exists()

    fake_projects_api["status"] = 404
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert (wrapper, fresh) == (None, True)
    assert not (project_dir / ".lqh" / "snapshot.json").exists()
    assert read_cached_snapshot(project_dir) is None


async def test_fetch_failure_falls_back_to_cache(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    from lqh.snapshot import fetch_and_cache_snapshot

    first, _ = await fetch_and_cache_snapshot(project_dir)
    fake_projects_api["status"] = 503
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert fresh is False
    assert wrapper == first


async def test_fetch_offline_without_cache_returns_none(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch, snapshot_auth
) -> None:
    from lqh.snapshot import fetch_and_cache_snapshot

    async def _boom(*args, **kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr("lqh.snapshot.fetch_snapshot", _boom)
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert wrapper is None
    assert fresh is False


def test_sanitize_drops_urls_and_credentials() -> None:
    from lqh.snapshot import sanitize

    dirty = {
        "jobs": [{"job_id": "j1", "download_url": "https://signed", "status": "ok"}],
        "api_key": "secret",
        "nested": {"upload_token": "x", "kept": 1},
    }
    clean = sanitize(dirty)

    assert clean == {"jobs": [{"job_id": "j1", "status": "ok"}], "nested": {"kept": 1}}


def test_read_cached_snapshot_wraps_legacy_format(project_dir: Path) -> None:
    from lqh.snapshot import read_cached_snapshot

    write_local_snapshot(project_dir, _SNAPSHOT)
    wrapper = read_cached_snapshot(project_dir)

    assert wrapper is not None
    assert wrapper["snapshot"] == _SNAPSHOT
    assert wrapper["fetched_at"] is None  # unknown freshness → treated stale


def test_compute_spec_sha256(project_dir: Path) -> None:
    assert compute_spec_sha256(project_dir) is None

    (project_dir / "SPEC.md").write_text("# spec\n")
    digest = compute_spec_sha256(project_dir)

    assert isinstance(digest, str) and len(digest) == 64
    # Deterministic: same content, same hash.
    assert digest == compute_spec_sha256(project_dir)
