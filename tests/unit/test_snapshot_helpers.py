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
    state = {
        "status": 200,
        "artifacts_status": 200,
        "deployments_status": None,  # None → follow "status"
        "requests": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        path = request.url.path
        if path.endswith("/artifacts"):
            if state["artifacts_status"] != 200:
                return httpx.Response(state["artifacts_status"], json={"error": "nope"})
            return httpx.Response(200, json={"artifacts": _ARTIFACTS})
        if path == "/v1/deployments":
            dep_status = state["deployments_status"]
            if dep_status is None:
                dep_status = state["status"]
            if dep_status != 200:
                return httpx.Response(dep_status, json={"error": "nope"})
            return httpx.Response(200, json={"deployments": _DEPLOYMENTS})
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": "nope"})
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
    # Phase 3: the wrapper records the RESOLVED cloud key (the stable
    # UUID for fresh projects; the basename only for unmigrated legacy
    # projects).
    from lqh.project_identity import cloud_project_key

    assert wrapper["project_key"] == cloud_project_key(project_dir)
    assert wrapper["fetched_at"]
    # Enrichment: project artifacts and deployment state ride along,
    # sanitized (no signed URLs in the cache) and deployment rows scoped
    # STRICTLY to this project — foreign project_id rows are dropped and
    # unattributed rows land in their own clearly-separated bucket
    # (never presented as this project's).
    assert wrapper["artifacts"] == [{"artifact_id": "art-1", "kind": "checkpoint"}]
    assert wrapper["deployments"] == []
    assert wrapper["unattributed_deployments"] == [_DEPLOYMENTS[0]]
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
    # The deployment refresh succeeded: dep-1 (unattributed) stays in its
    # own bucket, never in this project's list.
    assert wrapper["deployments"] == []
    assert wrapper["unattributed_deployments"] == [_DEPLOYMENTS[0]]


async def test_fetch_404_means_no_cloud_activity(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    """404 on the project AND nothing in the other sections → truly no
    cloud activity: (None, True), no cache left behind."""
    from lqh.snapshot import fetch_and_cache_snapshot

    fake_projects_api["status"] = 404  # deployments follow
    fake_projects_api["artifacts_status"] = 404
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert wrapper is None
    assert fresh is True
    assert not (project_dir / ".lqh" / "snapshot.json").exists()


async def test_project_404_keeps_fetched_artifacts(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    """Artifacts can exist WITHOUT a projects row (row-less legacy
    history, failed submit-time upsert) — a project 404 must not
    discard an artifact list that WAS successfully fetched."""
    from lqh.snapshot import fetch_and_cache_snapshot

    fake_projects_api["status"] = 404
    fake_projects_api["deployments_status"] = 404
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert fresh is True
    assert wrapper is not None
    assert wrapper["snapshot"] == {}  # jobs/spend authoritatively absent
    assert wrapper["artifacts"] == [{"artifact_id": "art-1", "kind": "checkpoint"}]


async def test_authoritative_404_clears_jobs_but_keeps_deployment_state(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    """A 404 is authoritative for jobs/spend — the old core snapshot must
    not resurface. But when the deployment refresh ALSO failed, the last
    known deployment state is carried forward marked stale rather than
    erased (a possibly-live deployment must never become invisible)."""
    from lqh.snapshot import fetch_and_cache_snapshot, read_cached_snapshot

    await fetch_and_cache_snapshot(project_dir)
    assert (project_dir / ".lqh" / "snapshot.json").exists()

    fake_projects_api["status"] = 404  # deployments follow → also fail
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert fresh is True
    assert wrapper["snapshot"] == {}  # obsolete jobs/spend are gone
    # Artifacts were still fetched fresh (they can exist row-less);
    # deployments carried forward from the previous cache, still in the
    # unattributed bucket, marked stale.
    assert wrapper["artifacts"] == [{"artifact_id": "art-1", "kind": "checkpoint"}]
    assert wrapper["deployments"] == []
    assert wrapper["unattributed_deployments"] == [_DEPLOYMENTS[0]]
    assert wrapper["stale_sections"] == ["deployments"]

    # With nothing in ANY section (and no cache), 404 clears everything.
    (project_dir / ".lqh" / "snapshot.json").unlink()
    fake_projects_api["artifacts_status"] = 404
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)
    assert (wrapper, fresh) == (None, True)
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

    # ALL THREE requests must be stubbed — patching only the core fetch
    # once left the enrichment requests hitting the real network and
    # hanging this test.
    monkeypatch.setattr("lqh.snapshot.fetch_snapshot", _boom)
    monkeypatch.setattr("lqh.snapshot.fetch_project_artifacts", _boom)
    monkeypatch.setattr("lqh.snapshot.fetch_deployments", _boom)
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert wrapper is None
    assert fresh is False


async def test_refresh_has_a_hard_deadline(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch, snapshot_auth
) -> None:
    """A stalled request that evades per-request timeouts must not hang
    CLI startup — the whole refresh is bounded by an outer deadline."""
    import asyncio
    import time

    from lqh.snapshot import fetch_and_cache_snapshot

    async def _hang(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr("lqh.snapshot.fetch_snapshot", _hang)
    monkeypatch.setattr("lqh.snapshot.fetch_project_artifacts", _hang)
    monkeypatch.setattr("lqh.snapshot.fetch_deployments", _hang)

    start = time.monotonic()
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir, timeout=0.2)

    assert time.monotonic() - start < 2.0
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


def test_sanitize_scrubs_by_value_shape() -> None:
    """URL/credential VALUES under innocent key names must not survive —
    key-name heuristics alone are not a privacy boundary."""
    from lqh.snapshot import sanitize

    dirty = {
        "note": "https://r2.example/signed?sig=abc",
        "auth": "Bearer abc123",
        "jwt_ish": "eyJhbGciOiJIUzI1NiJ9.x.y",
        "fine": "a plain string",
        "items": ["https://leak", "kept"],
    }

    assert sanitize(dirty) == {"fine": "a plain string", "items": ["kept"]}


async def test_core_404_keeps_fetched_deployments(
    project_dir: Path, fake_projects_api, snapshot_auth
) -> None:
    """No project row (404) is authoritative for jobs/spend — but live
    deployment state that WAS fetched must not be discarded with it
    (hiding a deployment risks a duplicate redeploy)."""
    from lqh.snapshot import fetch_and_cache_snapshot

    fake_projects_api["status"] = 404
    fake_projects_api["deployments_status"] = 200
    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert fresh is True
    assert wrapper is not None
    assert wrapper["snapshot"] == {}
    assert wrapper["deployments"] == []  # nothing attributed to this project
    assert wrapper["unattributed_deployments"] == [_DEPLOYMENTS[0]]


def test_read_cached_snapshot_wraps_legacy_format(project_dir: Path) -> None:
    from lqh.snapshot import read_cached_snapshot

    write_local_snapshot(project_dir, _SNAPSHOT)
    wrapper = read_cached_snapshot(project_dir)

    assert wrapper is not None
    assert wrapper["snapshot"] == _SNAPSHOT
    assert wrapper["fetched_at"] is None  # unknown freshness → treated stale


def test_cached_snapshot_for_another_key_is_ignored(project_dir: Path) -> None:
    """A cache written for a different project key (pre-fork, pre-
    migration, or copied in) must never resurface as this project's
    cloud state."""
    from lqh.snapshot import read_cached_snapshot

    (project_dir / ".lqh").mkdir(exist_ok=True)
    (project_dir / ".lqh" / "snapshot.json").write_text(json.dumps({
        "schema_version": 1,
        "fetched_at": "2026-07-17T00:00:00+00:00",
        "project_key": "someone-elses-project",
        "snapshot": {"jobs": [{"job_id": "foreign"}]},
    }))

    assert read_cached_snapshot(project_dir) is None


async def test_jobs_are_paged_until_exhausted(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """More jobs than one page → the client follows jobs_has_more and
    aggregates, recording jobs_truncated=False when it got everything."""
    from lqh.snapshot import fetch_and_cache_snapshot

    calls: list[int] = []

    async def fake_snapshot(pid, *, jobs_limit=None, jobs_offset=None, **kwargs):
        calls.append(jobs_offset)
        if jobs_offset == 0:
            return {
                "jobs": [{"job_id": f"j{i}"} for i in range(jobs_limit)],
                "jobs_has_more": True,
            }
        return {"jobs": [{"job_id": "j-last"}], "jobs_has_more": False}

    async def fake_artifacts(pid, **kwargs):
        return []

    async def fake_deployments(**kwargs):
        return []

    monkeypatch.setattr("lqh.snapshot.fetch_snapshot", fake_snapshot)
    monkeypatch.setattr("lqh.snapshot.fetch_project_artifacts", fake_artifacts)
    monkeypatch.setattr("lqh.snapshot.fetch_deployments", fake_deployments)

    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert fresh is True
    jobs = wrapper["snapshot"]["jobs"]
    assert len(jobs) == 101  # one full page + the final short page
    assert jobs[-1] == {"job_id": "j-last"}
    assert calls == [0, 100]
    assert wrapper["jobs_truncated"] is False


async def test_artifact_paging_caps_and_reports_truncation(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project with more artifacts than the client cap gets the cap,
    with an explicit truncation flag — never a silent 25-item slice."""
    from lqh.snapshot import fetch_and_cache_snapshot

    async def fake_snapshot(pid, **kwargs):
        return {"jobs": [], "jobs_has_more": False}

    async def fake_artifacts(pid, *, limit=100, offset=0, **kwargs):
        # Endless supply: always a full page.
        return [{"artifact_id": f"a{offset + i}"} for i in range(limit)]

    async def fake_deployments(**kwargs):
        return []

    monkeypatch.setattr("lqh.snapshot.fetch_snapshot", fake_snapshot)
    monkeypatch.setattr("lqh.snapshot.fetch_project_artifacts", fake_artifacts)
    monkeypatch.setattr("lqh.snapshot.fetch_deployments", fake_deployments)

    wrapper, fresh = await fetch_and_cache_snapshot(project_dir)

    assert fresh is True
    assert len(wrapper["artifacts"]) == 500  # the documented client cap
    assert wrapper["artifacts_truncated"] is True


async def test_exactly_at_cap_is_not_flagged_truncated(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly 500 artifacts (cap) with nothing beyond → the client
    probes one page further and does NOT claim more exist."""
    from lqh.snapshot import fetch_and_cache_snapshot

    async def fake_snapshot(pid, **kwargs):
        return {"jobs": [], "jobs_has_more": False}

    async def fake_artifacts(pid, *, limit=100, offset=0, **kwargs):
        remaining = max(0, 500 - offset)
        return [{"artifact_id": f"a{offset + i}"} for i in range(min(limit, remaining))]

    async def fake_deployments(**kwargs):
        return []

    monkeypatch.setattr("lqh.snapshot.fetch_snapshot", fake_snapshot)
    monkeypatch.setattr("lqh.snapshot.fetch_project_artifacts", fake_artifacts)
    monkeypatch.setattr("lqh.snapshot.fetch_deployments", fake_deployments)

    wrapper, _ = await fetch_and_cache_snapshot(project_dir)

    assert len(wrapper["artifacts"]) == 500
    assert wrapper["artifacts_truncated"] is False


def test_compute_spec_sha256(project_dir: Path) -> None:
    assert compute_spec_sha256(project_dir) is None

    (project_dir / "SPEC.md").write_text("# spec\n")
    digest = compute_spec_sha256(project_dir)

    assert isinstance(digest, str) and len(digest) == 64
    # Deterministic: same content, same hash.
    assert digest == compute_spec_sha256(project_dir)
