"""Tests for stable project identity (lqh/project_identity.py, Phase 3)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest

from lqh.project_identity import (
    ProjectIdentityError,
    adopt_cloud_id,
    cloud_project_key,
    detect_copy,
    ensure_identity,
    fork_identity,
    marker_is_foreign,
    migrate_cloud_identity,
    project_uuid,
    record_continue_decision,
    record_path,
)


def _identity(project_dir: Path) -> dict:
    return json.loads((project_dir / ".lqh" / "project.json").read_text())


# ---------------------------------------------------------------------------
# ensure_identity / classification / cloud key
# ---------------------------------------------------------------------------


def test_new_project_adopts_uuid_key_at_birth(project_dir: Path) -> None:
    identity, classification = ensure_identity(project_dir)

    assert classification == "new"
    uuid.UUID(identity["project_id"])
    # Fresh projects never inherit a same-named stranger's namespace.
    assert identity["cloud_project_id"] == identity["project_id"]
    assert identity["display_name"] == project_dir.name
    assert identity["forked_from"] is None
    assert cloud_project_key(project_dir) == identity["project_id"]


def test_pre_existing_project_stays_on_legacy_key(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# spec\n")

    identity, classification = ensure_identity(project_dir)

    assert classification == "pre_existing"
    # Basename-keyed cloud history may exist: stay on the legacy key
    # until the authenticated migration decides.
    assert identity["cloud_project_id"] is None
    assert cloud_project_key(project_dir) == project_dir.name


def test_identity_is_stable_across_opens(project_dir: Path) -> None:
    first, _ = ensure_identity(project_dir)
    second, classification = ensure_identity(project_dir)

    assert classification == "reopened"
    assert second["project_id"] == first["project_id"]


def test_v1_identity_migrates_in_place(project_dir: Path) -> None:
    """Telemetry-era v1 files keep their UUID and gain the v2 fields; the
    project is treated as legacy (cloud key undecided)."""
    old_id = str(uuid.uuid4())
    (project_dir / ".lqh" / "project.json").write_text(
        json.dumps({"schema_version": 1, "project_id": old_id}) + "\n"
    )

    identity, classification = ensure_identity(project_dir)

    assert classification == "reopened"
    assert identity["schema_version"] == 3
    assert identity["project_id"] == old_id
    assert identity["cloud_project_id"] is None
    # The basename is recorded ONCE as the legacy cloud name — a later
    # folder rename must keep addressing the original namespace.
    assert identity["legacy_cloud_name"] == project_dir.name
    assert cloud_project_key(project_dir) == project_dir.name


def test_telemetry_delegates_to_identity_module(project_dir: Path) -> None:
    from lqh.telemetry import ensure_project_identity

    project_id, classification = ensure_project_identity(project_dir)

    assert classification == "new"
    assert project_id == _identity(project_dir)["project_id"]


# ---------------------------------------------------------------------------
# Copy / move / fork
# ---------------------------------------------------------------------------


def test_move_continues_identity_automatically(tmp_path: Path) -> None:
    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    moved = tmp_path / "renamed"
    original.rename(moved)

    assert detect_copy(moved) == "moved"
    assert _identity(moved)["last_seen_path"] == str(moved.resolve())
    assert detect_copy(moved) == "same"


def test_copy_is_detected_and_fork_mints_new_identity(tmp_path: Path) -> None:
    import shutil

    original = tmp_path / "proj"
    original.mkdir()
    first, _ = ensure_identity(original)
    copy = tmp_path / "proj_copy"
    shutil.copytree(original, copy)

    assert detect_copy(copy) == "copied"
    # Explicit continue: the CHOICE is recorded, not just acted on.
    record_continue_decision(copy)
    assert detect_copy(copy) == "same"
    decision = _identity(copy)["copy_decision"]
    assert decision["choice"] == "continue"
    assert decision["previous_path"] == str(original.resolve())

    # Explicit fork instead: fresh identity + namespace, provenance kept.
    forked = fork_identity(copy)
    assert forked["project_id"] != first["project_id"]
    assert forked["forked_from"] == first["project_id"]
    assert forked["cloud_project_id"] == forked["project_id"]
    # The original is untouched.
    assert _identity(original)["project_id"] == first["project_id"]


# ---------------------------------------------------------------------------
# Cloud migration cutover
# ---------------------------------------------------------------------------


async def test_migrate_without_history_adopts_uuid(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename endpoint answering 404 means NOTHING exists under the
    legacy name (no project row, no row-less artifacts/deployments) —
    adopt the UUID directly."""
    (project_dir / "SPEC.md").write_text("# spec\n")  # legacy project

    called = {}

    async def fake_rename(old, new, **kwargs):
        called["rename"] = (old, new)
        raise httpx.HTTPStatusError(
            "404", request=httpx.Request("POST", "http://t"),
            # The API's error envelope — a bare route-404 would defer.
            response=httpx.Response(
                404, json={"error": {"message": "project not found"}}
            ),
        )

    monkeypatch.setattr("lqh.project_meta.rename_project", fake_rename)

    key = await migrate_cloud_identity(project_dir)

    identity = _identity(project_dir)
    assert key == identity["project_id"]
    assert identity["cloud_project_id"] == identity["project_id"]
    assert called["rename"] == (project_dir.name, identity["project_id"])
    assert cloud_project_key(project_dir) == key


async def test_migrate_with_history_renames_backend_project(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project_dir / "SPEC.md").write_text("# spec\n")

    async def fake_snapshot(pid, **kwargs):
        return {"project_id": pid}

    called = {}

    async def fake_rename(old, new, **kwargs):
        called["rename"] = (old, new)
        return {"project_id": new}

    monkeypatch.setattr("lqh.project_meta.fetch_snapshot", fake_snapshot)
    monkeypatch.setattr("lqh.project_meta.rename_project", fake_rename)

    key = await migrate_cloud_identity(project_dir)

    identity = _identity(project_dir)
    assert called["rename"] == (project_dir.name, identity["project_id"])
    assert key == identity["project_id"]
    assert cloud_project_key(project_dir) == key


async def test_migrate_refusal_stays_on_legacy_key(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """409 (active jobs) / network failure: keep the legacy key and retry
    on a later startup — never half-migrate."""
    (project_dir / "SPEC.md").write_text("# spec\n")

    async def fake_snapshot(pid, **kwargs):
        return {"project_id": pid}

    async def fake_rename(old, new, **kwargs):
        raise httpx.HTTPStatusError(
            "409", request=httpx.Request("POST", "http://t"),
            response=httpx.Response(409),
        )

    monkeypatch.setattr("lqh.project_meta.fetch_snapshot", fake_snapshot)
    monkeypatch.setattr("lqh.project_meta.rename_project", fake_rename)

    key = await migrate_cloud_identity(project_dir)

    assert key is None
    assert _identity(project_dir)["cloud_project_id"] is None
    assert cloud_project_key(project_dir) == project_dir.name


async def test_route_level_404_defers_instead_of_adopting(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare 404 (backend without the rename route mid-deploy, proxy)
    is NOT 'no history' — adopting on it would orphan real basename
    history. Only our JSON error envelope is authoritative."""
    (project_dir / "SPEC.md").write_text("# spec\n")

    async def fake_rename(old, new, **kwargs):
        raise httpx.HTTPStatusError(
            "404", request=httpx.Request("POST", "http://t"),
            response=httpx.Response(404, text="404 page not found"),
        )

    monkeypatch.setattr("lqh.project_meta.rename_project", fake_rename)

    assert await migrate_cloud_identity(project_dir) is None
    assert _identity(project_dir)["cloud_project_id"] is None
    assert cloud_project_key(project_dir) == project_dir.name


async def test_migrate_retries_once_on_lost_response(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transport error may mean the backend COMMITTED the rename and
    the response was lost — the client retries once (the backend is
    idempotent) instead of running a whole session on the legacy key."""
    (project_dir / "SPEC.md").write_text("# spec\n")
    attempts = []

    async def flaky_rename(old, new, **kwargs):
        attempts.append((old, new))
        if len(attempts) == 1:
            raise httpx.ConnectError("response lost")
        return {"project_id": new}

    monkeypatch.setattr("lqh.project_meta.rename_project", flaky_rename)

    key = await migrate_cloud_identity(project_dir)

    assert len(attempts) == 2
    assert key == _identity(project_dir)["project_id"]
    assert cloud_project_key(project_dir) == key


async def test_migrate_is_idempotent_once_adopted(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adopt_cloud_id(project_dir, "already-migrated")

    async def explode(*args, **kwargs):
        raise AssertionError("no network call expected")

    monkeypatch.setattr("lqh.project_meta.fetch_snapshot", explode)
    monkeypatch.setattr("lqh.project_meta.rename_project", explode)

    assert await migrate_cloud_identity(project_dir) == "already-migrated"


# ---------------------------------------------------------------------------
# The resolver feeds the cloud call sites
# ---------------------------------------------------------------------------


def test_cloud_submit_uses_stable_key(project_dir: Path) -> None:
    """A fresh project's submit meta must carry the UUID key, not the
    directory basename. (The end-to-end CloudBackend submit is asserted
    in test_cloud_backend.py::test_submit_carries_stable_key_and_owner.)"""
    from lqh.project_identity import cloud_project_key as resolver

    key = resolver(project_dir)
    assert key != project_dir.name
    uuid.UUID(key)


# ---------------------------------------------------------------------------
# Legacy name survives renames; corrupt identities fail closed
# ---------------------------------------------------------------------------


def test_rename_before_migration_keeps_legacy_namespace(tmp_path: Path) -> None:
    """Renaming an UNMIGRATED legacy folder must not orphan its
    basename-keyed cloud history: the key stays the recorded legacy
    name, not the current basename (acceptance criterion 5)."""
    original = tmp_path / "old-name"
    original.mkdir()
    (original / "SPEC.md").write_text("# spec\n")
    ensure_identity(original)
    assert cloud_project_key(original) == "old-name"

    moved = tmp_path / "new-name"
    original.rename(moved)
    assert detect_copy(moved) == "moved"

    # Still addresses the original cloud namespace.
    assert cloud_project_key(moved) == "old-name"
    assert _identity(moved)["display_name"] == "new-name"


async def test_migration_queries_recorded_legacy_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "old-name"
    original.mkdir()
    (original / "SPEC.md").write_text("# spec\n")
    ensure_identity(original)
    moved = tmp_path / "new-name"
    original.rename(moved)
    detect_copy(moved)  # records the move

    queried = {}

    async def fake_rename(old, new, **kwargs):
        queried["rename"] = (old, new)
        return {"project_id": new}

    monkeypatch.setattr("lqh.project_meta.rename_project", fake_rename)

    key = await migrate_cloud_identity(moved)

    assert queried["rename"] == ("old-name", _identity(moved)["project_id"])
    assert key == _identity(moved)["project_id"]


def test_corrupt_identity_raises_and_never_rotates(project_dir: Path) -> None:
    path = project_dir / ".lqh" / "project.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{{{ not json")

    with pytest.raises(ProjectIdentityError):
        ensure_identity(project_dir)
    with pytest.raises(ProjectIdentityError):
        cloud_project_key(project_dir)
    # The corrupt file is preserved byte-for-byte — never replaced with
    # a fresh UUID (that would disconnect the cloud history).
    assert path.read_text() == "{{{ not json"


def test_invalid_project_id_is_corrupt_not_absent(project_dir: Path) -> None:
    path = project_dir / ".lqh" / "project.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 2, "project_id": "not-a-uuid"}))

    with pytest.raises(ProjectIdentityError):
        ensure_identity(project_dir)
    assert json.loads(path.read_text())["project_id"] == "not-a-uuid"


def test_v1_migration_preserves_unknown_fields(project_dir: Path) -> None:
    """Fields other writers persisted here (pipeline_readiness,
    spec_capture, telemetry_*) must survive the v1→v3 upgrade."""
    old_id = str(uuid.uuid4())
    (project_dir / ".lqh").mkdir(parents=True, exist_ok=True)
    (project_dir / ".lqh" / "project.json").write_text(json.dumps({
        "schema_version": 1,
        "project_id": old_id,
        "telemetry_state": {"consent": True},
        "pipeline_readiness": {"data_gen/pipe.py": "abc123"},
        "spec_capture": {"completed": True},
    }))

    identity, _ = ensure_identity(project_dir)

    assert identity["project_id"] == old_id
    assert identity["telemetry_state"] == {"consent": True}
    assert identity["pipeline_readiness"] == {"data_gen/pipe.py": "abc123"}
    assert identity["spec_capture"] == {"completed": True}
    # And they survive on disk, not just in the returned dict.
    on_disk = _identity(project_dir)
    assert on_disk["pipeline_readiness"] == {"data_gen/pipe.py": "abc123"}


# ---------------------------------------------------------------------------
# Copy detection across hosts / unreadable originals
# ---------------------------------------------------------------------------


def _rewrite_identity(project_dir: Path, **overrides) -> None:
    path = project_dir / ".lqh" / "project.json"
    data = json.loads(path.read_text())
    data.update(overrides)
    path.write_text(json.dumps(data))


def test_missing_original_on_other_host_asks(project_dir: Path) -> None:
    """Recorded path gone AND hostname differs → a copy to another
    machine cannot be ruled out; the caller must ask."""
    ensure_identity(project_dir)
    _rewrite_identity(
        project_dir,
        last_seen_path="/somewhere/that/does/not/exist",
        last_seen_hostname="a-different-machine",
    )
    assert detect_copy(project_dir) == "copied"


def test_missing_original_on_same_host_is_a_move(project_dir: Path) -> None:
    import platform

    ensure_identity(project_dir)
    _rewrite_identity(
        project_dir,
        last_seen_path="/somewhere/that/does/not/exist",
        last_seen_hostname=platform.node() or "unknown",
    )
    assert detect_copy(project_dir) == "moved"


def test_unreadable_original_identity_asks(tmp_path: Path) -> None:
    """The recorded path exists but its identity cannot be read — a live
    copy cannot be ruled out, so ask instead of silently continuing."""
    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    copy = tmp_path / "copy"
    import shutil

    shutil.copytree(original, copy)
    # Corrupt the ORIGINAL's identity file: detect_copy(copy) can no
    # longer verify whether both share one project_id.
    (original / ".lqh" / "project.json").write_text("garbage")

    assert detect_copy(copy) == "copied"


# ---------------------------------------------------------------------------
# Fork detaches inherited cloud state
# ---------------------------------------------------------------------------


def test_fork_detaches_inherited_cloud_state(tmp_path: Path) -> None:
    import shutil

    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    run = original / "runs" / "sft_v1"
    run.mkdir(parents=True)
    for marker in (
        "remote_job.json", "cloud_state.json",
        "submit_intent.json", ".lqh_data_gen.json",
    ):
        (run / marker).write_text('{"job_id": "j-parent"}')
    (original / ".lqh" / "snapshot.json").write_text('{"snapshot": {}}')
    (original / ".lqh" / "job_seen.json").write_text('{"runs": {}}')

    copy = tmp_path / "proj_copy"
    shutil.copytree(original, copy)
    forked = fork_identity(copy)

    # Caches deleted; per-run markers renamed inert (but preserved).
    assert not (copy / ".lqh" / "snapshot.json").exists()
    assert not (copy / ".lqh" / "job_seen.json").exists()
    copied_run = copy / "runs" / "sft_v1"
    for marker in (
        "remote_job.json", "cloud_state.json",
        "submit_intent.json", ".lqh_data_gen.json",
    ):
        assert not (copied_run / marker).exists()
        assert (copied_run / f"{marker}.pre-fork").exists()
    # Decision recorded; the ORIGINAL keeps everything untouched.
    assert forked["copy_decision"]["choice"] == "fork"
    assert (run / "remote_job.json").exists()
    assert (original / ".lqh" / "snapshot.json").exists()


def test_marker_ownership(project_dir: Path) -> None:
    ensure_identity(project_dir)
    own = {"owner_project_id": project_uuid(project_dir)}
    foreign = {"owner_project_id": str(uuid.uuid4())}
    legacy = {"job_id": "j1"}  # pre-Phase-3 marker: treated as local

    assert marker_is_foreign(project_dir, own) is False
    assert marker_is_foreign(project_dir, foreign) is True
    assert marker_is_foreign(project_dir, legacy) is False


# ---------------------------------------------------------------------------
# Legacy copy + continue converges instead of splitting
# ---------------------------------------------------------------------------


async def test_legacy_copy_continue_converges_on_one_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unmigrated legacy project copied and CONTINUED shares both the
    project_id and the recorded legacy name — whichever directory
    migrates first renames the backend project; the other adopts the
    SAME UUID via the 404 path. One namespace, never a split."""
    import shutil

    backend_projects = {"proj"}

    async def fake_snapshot(pid, **kwargs):
        if pid not in backend_projects:
            raise httpx.HTTPStatusError(
                "404", request=httpx.Request("GET", "http://t"),
                response=httpx.Response(404),
            )
        return {"project_id": pid}

    async def fake_rename(old, new, **kwargs):
        backend_projects.discard(old)
        backend_projects.add(new)
        return {"project_id": new}

    monkeypatch.setattr("lqh.project_meta.fetch_snapshot", fake_snapshot)
    monkeypatch.setattr("lqh.project_meta.rename_project", fake_rename)

    original = tmp_path / "proj"
    original.mkdir()
    (original / "SPEC.md").write_text("# spec\n")
    ensure_identity(original)

    copy = tmp_path / "proj_backup"
    shutil.copytree(original, copy)
    assert detect_copy(copy) == "copied"
    record_continue_decision(copy)

    # The copy migrates first: renames proj → UUID on the backend.
    key_copy = await migrate_cloud_identity(copy)
    assert key_copy == _identity(copy)["project_id"]
    assert backend_projects == {key_copy}

    # The original migrates later: 404 on the legacy name → adopts the
    # SAME UUID (shared project_id), reaching the renamed history.
    key_original = await migrate_cloud_identity(original)
    assert key_original == key_copy
    assert cloud_project_key(original) == cloud_project_key(copy)


# ---------------------------------------------------------------------------
# Migration HTTP contract (real fetch/rename through a mock transport)
# ---------------------------------------------------------------------------


async def test_migration_http_contract(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No function-level mocking of fetch/rename: the real helpers run
    against a mock transport, pinning the URL shapes and the rename
    request body."""
    from lqh import project_meta

    (project_dir / "SPEC.md").write_text("# spec\n")
    identity, _ = ensure_identity(project_dir)
    legacy = project_dir.name
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if (
            request.method == "POST"
            and request.url.path == f"/v1/projects/{legacy}/rename"
        ):
            body = json.loads(request.content)
            assert body == {"new_project_id": identity["project_id"]}
            return httpx.Response(200, json={"project_id": body["new_project_id"]})
        return httpx.Response(404, json={"error": "unexpected route"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(project_meta.httpx, "AsyncClient", _patched)
    monkeypatch.setattr("lqh.project_meta.require_token", lambda: "tok")
    monkeypatch.setattr("lqh.project_meta.api_root", lambda: "http://test")

    key = await migrate_cloud_identity(project_dir)

    assert key == identity["project_id"]
    # ONE idempotent rename call — no GET probe (a 404 probe would
    # orphan row-less artifact/deployment history under the basename).
    assert [r.method for r in requests] == ["POST"]
    assert requests[0].headers["Authorization"] == "Bearer tok"
    assert cloud_project_key(project_dir) == identity["project_id"]