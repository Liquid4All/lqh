"""TUI finalization of cloud data-gen runs (download, retry, guards)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from lqh.tui.app import LqhApp


class _NullTelemetry:
    """Inert telemetry stand-in for the finalization tests."""

    def state_snapshot(self):
        return (False, 0, 0.0, None)

    def defer(self, *args, **kwargs):
        return None


@pytest.fixture
def app(tmp_path: Path) -> LqhApp:
    """A bare LqhApp carrying only what _finalize_data_gen_run touches.

    Deliberately bypasses __init__: full app construction drags in
    prompt_toolkit/session/telemetry side effects that destabilize the
    timing-sensitive interrupt tests running later in the suite.
    """
    instance = LqhApp.__new__(LqhApp)
    instance.project_dir = tmp_path
    instance._telemetry = _NullTelemetry()  # type: ignore[assignment]
    instance._data_gen_gave_up = set()
    return instance


def _make_run(
    project: Path,
    run_name: str = "data_gen_ds_x",
    *,
    output_dataset: str = "ds",
    marker_extra: dict | None = None,
) -> Path:
    run_dir = project / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "type": "data_gen", "output_dataset": output_dataset,
    }))
    marker = {
        "workflow_id": "wf-1",
        "output_dataset": output_dataset,
        "job_id": "job-1",
        "submitted_at": time.time(),
        **(marker_extra or {}),
    }
    (run_dir / ".lqh_data_gen.json").write_text(json.dumps(marker))
    (run_dir / "artifacts.json").write_text(json.dumps({
        "artifacts": [{"artifact_id": "art-1", "kind": "dataset"}],
    }))
    return run_dir


class _FakeStore:
    """Stands in for BackendArtifactStore; behavior set per test."""

    fail = False
    downloads: list[tuple[str, Path]] = []
    listed: list = []  # handles returned by list_for_project

    def __init__(self, **_kw) -> None: ...

    async def download(self, artifact_id, dest: Path) -> None:
        if _FakeStore.fail:
            raise RuntimeError("boom")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1")
        _FakeStore.downloads.append((str(artifact_id), dest))

    async def list_for_project(self, project_id, *, kind=None, job_id=None, limit=100):
        return [
            h for h in _FakeStore.listed
            if job_id is None or getattr(h, "job_id", None) == job_id
        ]


@pytest.fixture(autouse=True)
def fake_store(monkeypatch: pytest.MonkeyPatch) -> type[_FakeStore]:
    _FakeStore.fail = False
    _FakeStore.downloads = []
    _FakeStore.listed = []
    monkeypatch.setattr("lqh.artifacts.BackendArtifactStore", _FakeStore)
    return _FakeStore


async def test_finalize_success_downloads_and_consumes_marker(app: LqhApp) -> None:
    run_dir = _make_run(app.project_dir)
    text = await app._finalize_data_gen_run(run_dir.name, "completed", None)
    assert text is not None and "completed" in text
    assert (app.project_dir / "datasets" / "ds" / "data.parquet").exists()
    assert not (run_dir / ".lqh_data_gen.json").exists()
    assert not app._data_gen_pending(run_dir.name)


async def test_finalize_failed_job_consumes_marker(app: LqhApp) -> None:
    run_dir = _make_run(app.project_dir)
    text = await app._finalize_data_gen_run(run_dir.name, "failed", "sandbox oom")
    assert text is not None and "failed" in text and "sandbox oom" in text
    assert not (run_dir / ".lqh_data_gen.json").exists()


async def test_failed_job_with_published_dataset_is_recovered(
    app: LqhApp, fake_store: type[_FakeStore],
) -> None:
    """A backend restart can mislabel a finished job failed; if the
    dataset artifact exists, recover it instead of reporting failure."""
    from types import SimpleNamespace

    run_dir = _make_run(app.project_dir)
    fake_store.listed = [SimpleNamespace(id="art-9", kind="dataset", job_id="job-1")]

    text = await app._finalize_data_gen_run(run_dir.name, "failed", "orphaned")
    assert text is not None and "recovered" in text
    assert (app.project_dir / "datasets" / "ds" / "data.parquet").exists()
    assert not (run_dir / ".lqh_data_gen.json").exists()


async def test_finalize_cancelled_job_notifies_and_consumes_marker(app: LqhApp) -> None:
    run_dir = _make_run(app.project_dir)
    text = await app._finalize_data_gen_run(run_dir.name, "cancelled", None)
    assert text is not None and "was cancelled" in text
    assert not (run_dir / ".lqh_data_gen.json").exists()


def _clear_backoff(run_dir: Path) -> None:
    """Simulate the backoff window elapsing between watcher scans."""
    marker_path = run_dir / ".lqh_data_gen.json"
    marker = json.loads(marker_path.read_text())
    marker.pop("retry_after", None)
    marker_path.write_text(json.dumps(marker))


async def test_transient_download_failure_keeps_marker_and_retries(
    app: LqhApp, fake_store: type[_FakeStore],
) -> None:
    run_dir = _make_run(app.project_dir)
    marker_path = run_dir / ".lqh_data_gen.json"

    fake_store.fail = True
    # Attempts 1..7: silent (None), marker kept with a bumped counter
    # and a scheduled retry_after (backoff between watcher scans).
    for expected_attempts in range(1, 8):
        text = await app._finalize_data_gen_run(run_dir.name, "completed", None)
        assert text is None
        marker = json.loads(marker_path.read_text())
        assert marker["download_attempts"] == expected_attempts
        assert marker["retry_after"] > 0
        _clear_backoff(run_dir)

    # Attempt 8: give up FOR THIS SESSION with an actionable message.
    # The marker survives (attempts reset) so a TUI restart retries;
    # the session set suppresses further scans meanwhile.
    text = await app._finalize_data_gen_run(run_dir.name, "completed", None)
    assert text is not None and "artifacts tool" in text
    marker = json.loads(marker_path.read_text())
    assert marker["download_attempts"] == 0
    assert marker["workflow_closed"] is True
    assert run_dir.name in app._data_gen_gave_up


async def test_backoff_window_suppresses_immediate_retry(
    app: LqhApp, fake_store: type[_FakeStore],
) -> None:
    run_dir = _make_run(app.project_dir)
    fake_store.fail = True
    assert await app._finalize_data_gen_run(run_dir.name, "completed", None) is None
    # Within the backoff window nothing runs — even though the store
    # would now succeed, the scheduled retry hasn't arrived yet.
    fake_store.fail = False
    assert await app._finalize_data_gen_run(run_dir.name, "completed", None) is None
    assert not (app.project_dir / "datasets" / "ds" / "data.parquet").exists()


async def test_restart_retries_after_give_up(
    app: LqhApp, fake_store: type[_FakeStore],
) -> None:
    run_dir = _make_run(app.project_dir)
    fake_store.fail = True
    for _ in range(8):
        await app._finalize_data_gen_run(run_dir.name, "completed", None)
        if (run_dir / ".lqh_data_gen.json").exists():
            _clear_backoff(run_dir)
    assert app._data_gen_pending(run_dir.name)  # marker survived give-up

    # "Restart": a fresh session has an empty gave-up set; the download
    # now works and the dataset lands.
    app._data_gen_gave_up.clear()
    fake_store.fail = False
    text = await app._finalize_data_gen_run(run_dir.name, "completed", None)
    assert text is not None and "completed" in text
    assert (app.project_dir / "datasets" / "ds" / "data.parquet").exists()
    assert not app._data_gen_pending(run_dir.name)


async def test_download_recovers_after_transient_failure(
    app: LqhApp, fake_store: type[_FakeStore],
) -> None:
    run_dir = _make_run(app.project_dir)
    fake_store.fail = True
    assert await app._finalize_data_gen_run(run_dir.name, "completed", None) is None
    _clear_backoff(run_dir)
    fake_store.fail = False
    text = await app._finalize_data_gen_run(run_dir.name, "completed", None)
    assert text is not None and "completed" in text
    assert (app.project_dir / "datasets" / "ds" / "data.parquet").exists()


async def test_newer_submission_wins_dataset(
    app: LqhApp, fake_store: type[_FakeStore],
) -> None:
    """Two jobs targeting one dataset: the newest SUBMISSION wins,
    regardless of finish order."""
    t0 = time.time()
    run_a = _make_run(app.project_dir, "data_gen_ds_a",
                      marker_extra={"submitted_at": t0 - 100, "job_id": "job-a"})
    run_b = _make_run(app.project_dir, "data_gen_ds_b",
                      marker_extra={"submitted_at": t0 - 50, "job_id": "job-b"})

    # A (older submission) finishes first and lands the dataset...
    text_a = await app._finalize_data_gen_run(run_a.name, "completed", None)
    assert text_a is not None and "downloaded" in text_a
    sidecar = json.loads(
        (app.project_dir / "datasets" / "ds" / ".lqh_source.json").read_text()
    )
    assert sidecar["job_id"] == "job-a"

    # ...then B (newer submission) finishes and must replace it, even
    # though the file's mtime is after B's submission time.
    text_b = await app._finalize_data_gen_run(run_b.name, "completed", None)
    assert text_b is not None and "downloaded" in text_b
    sidecar = json.loads(
        (app.project_dir / "datasets" / "ds" / ".lqh_source.json").read_text()
    )
    assert sidecar["job_id"] == "job-b"


async def test_older_submission_does_not_clobber_newer_result(
    app: LqhApp, fake_store: type[_FakeStore],
) -> None:
    t0 = time.time()
    run_a = _make_run(app.project_dir, "data_gen_ds_a",
                      marker_extra={"submitted_at": t0 - 100, "job_id": "job-a"})
    run_b = _make_run(app.project_dir, "data_gen_ds_b",
                      marker_extra={"submitted_at": t0 - 50, "job_id": "job-b"})

    # B (newer submission) finishes first...
    await app._finalize_data_gen_run(run_b.name, "completed", None)
    # ...then A completes; its result must NOT replace B's.
    text_a = await app._finalize_data_gen_run(run_a.name, "completed", None)
    assert text_a is not None and "NEWER submission" in text_a
    sidecar = json.loads(
        (app.project_dir / "datasets" / "ds" / ".lqh_source.json").read_text()
    )
    assert sidecar["job_id"] == "job-b"


async def test_newer_local_dataset_is_not_clobbered(app: LqhApp) -> None:
    run_dir = _make_run(
        app.project_dir, marker_extra={"submitted_at": time.time() - 3600},
    )
    dest = app.project_dir / "datasets" / "ds" / "data.parquet"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"NEWER LOCAL")
    os.utime(dest, (time.time(), time.time()))  # mtime after submitted_at

    text = await app._finalize_data_gen_run(run_dir.name, "completed", None)
    assert text is not None and "kept" in text
    assert dest.read_bytes() == b"NEWER LOCAL"
    assert not (run_dir / ".lqh_data_gen.json").exists()


async def test_unsafe_output_dataset_from_config_falls_back(app: LqhApp) -> None:
    run_dir = _make_run(app.project_dir, output_dataset="../escape")
    text = await app._finalize_data_gen_run(run_dir.name, "completed", None)
    assert text is not None
    # Falls back to the run name, never a path outside datasets/.
    assert not (app.project_dir.parent / "escape").exists()
    assert (app.project_dir / "datasets" / run_dir.name / "data.parquet").exists()


async def test_locally_modified_after_download_is_kept(
    app: LqhApp, fake_store: type[_FakeStore],
) -> None:
    """A dataset edited/regenerated locally AFTER its cloud download must
    not be clobbered by a later (newer-submission) cloud completion —
    local work always wins over cloud jobs."""
    t0 = time.time()
    run_a = _make_run(app.project_dir, "data_gen_ds_a",
                      marker_extra={"submitted_at": t0 - 100, "job_id": "job-a"})
    run_b = _make_run(app.project_dir, "data_gen_ds_b",
                      marker_extra={"submitted_at": t0 - 50, "job_id": "job-b"})

    # A downloads its result (sidecar records downloaded_at)...
    await app._finalize_data_gen_run(run_a.name, "completed", None)
    dest = app.project_dir / "datasets" / "ds" / "data.parquet"

    # ...the user modifies the file locally afterwards...
    dest.write_bytes(b"LOCAL EDIT")
    future = time.time() + 30
    os.utime(dest, (future, future))

    # ...then B (the newer submission) completes: local edit is kept.
    text_b = await app._finalize_data_gen_run(run_b.name, "completed", None)
    assert text_b is not None and "kept" in text_b
    assert dest.read_bytes() == b"LOCAL EDIT"
    assert not (run_b.parent / run_b.name / ".lqh_data_gen.json").exists()
