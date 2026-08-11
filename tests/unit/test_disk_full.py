"""A full disk must not paint tracebacks over the TUI (feedback #50).

`_configure_logging` routes logging to a file so background tasks can't
corrupt the screen — but when that file's write fails, Python's logging
prints "--- Logging error ---" plus a traceback to stderr per record. The
safety net became the spam source, and the poll that mirrors a cloud run
died with it, leaving a healthy run looking hung forever.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from lqh import diskspace
from lqh.cli import _configure_logging
from lqh.jobs import JobSupervisor, SupervisorHooks
from lqh.remote.watcher import RemoteRunWatcher


@pytest.fixture(autouse=True)
def _clear_latch(monkeypatch: pytest.MonkeyPatch):
    """The ENOSPC latch is module state and never clears by design."""
    monkeypatch.setattr(diskspace, "_saw_enospc", False)


@pytest.fixture
def _restore_logging():
    root = logging.getLogger()
    saved = (root.handlers[:], root.level, logging.lastResort, logging.raiseExceptions)
    yield
    root.handlers, root.level, logging.lastResort, logging.raiseExceptions = saved


def test_a_failing_log_handler_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _restore_logging: None,
) -> None:
    """The reported symptom, through the real logging setup."""
    _configure_logging(tmp_path)

    class _FullDisk:
        def write(self, _data: str) -> int:
            raise OSError(28, "No space left on device")

        def flush(self) -> None:
            pass

    logging.getLogger().handlers = [logging.StreamHandler(_FullDisk())]
    logging.getLogger("lqh.test").warning("mirror write failed")

    assert capsys.readouterr() == ("", "")


def test_an_uncreatable_log_file_does_not_abort_startup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _restore_logging: None,
) -> None:
    (tmp_path / ".lqh").mkdir()
    (tmp_path / ".lqh" / "lqh.log").mkdir()  # IsADirectoryError on open

    _configure_logging(tmp_path)
    logging.getLogger("lqh.test").warning("nothing should reach the terminal")

    assert capsys.readouterr() == ("", "")


def test_enospc_is_found_through_wrappers_and_rsync_text() -> None:
    try:
        try:
            raise OSError(28, "No space left on device")
        except OSError as exc:
            raise RuntimeError("stream failed") from exc
    except RuntimeError as outer:
        assert diskspace.is_enospc(outer)
    # rsync exits non-zero with the message; no errno survives.
    assert diskspace.is_enospc(RuntimeError("rsync: ... No space left on device (28)"))
    assert not diskspace.is_enospc(OSError(13, "Permission denied"))


class _FullDiskBackend:
    inline_scoring = True

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or OSError(28, "No space left on device")
        self.polled = False

    async def sync_progress(self, remote_run_dir: str, local_run_dir: str) -> None:
        raise self.exc

    async def poll_status(self, job_id: str) -> Any:
        from lqh.remote.backend import JobStatus

        self.polled = True
        return JobStatus(state="completed")


def test_the_watcher_logs_once_and_keeps_polling(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    run_dir = tmp_path / "runs" / "run_1"
    run_dir.mkdir(parents=True)
    watcher = RemoteRunWatcher(
        run_dir=run_dir, config={"type": "sft"}, project_dir=tmp_path,
        api_key="k", backend=_FullDiskBackend(), remote_run_dir="/r", job_id="1",
    )

    with caplog.at_level(logging.WARNING, logger="lqh.remote.watcher"):
        for _ in range(3):
            asyncio.run(watcher._sync_from_remote())

    disk = [r for r in caplog.records if "disk space" in r.message]
    assert len(disk) == 1 and disk[0].exc_info is None


def _cloud_run(project: Path) -> tuple[Path, dict]:
    run_dir = project / "runs" / "run_1"
    run_dir.mkdir(parents=True)
    meta = {"remote_name": "cloud", "backend": "cloud", "job_id": "1",
            "remote_run_dir": "/remote/run_1"}
    (run_dir / "remote_job.json").write_text(json.dumps(meta) + "\n")
    return run_dir, meta


def test_a_full_mirror_does_not_hide_the_job_from_the_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stranding bug.

    sync_progress writes every streamed event, so on a full disk it
    raises before poll_status is reached; scan_jobs turns that into
    "unknown" and the watch loop skips unknown runs — so the run is
    never marked complete. poll_status needs no local files.
    """
    run_dir, meta = _cloud_run(tmp_path)
    sup = JobSupervisor(tmp_path)
    backend = _FullDiskBackend()
    monkeypatch.setattr(sup, "make_remote_backend", lambda _m: backend)

    state, _err = asyncio.run(sup.poll_remote(run_dir, meta))

    assert backend.polled is True
    assert state == "completed"


def test_other_mirror_failures_still_abort_the_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, meta = _cloud_run(tmp_path)
    sup = JobSupervisor(tmp_path)
    backend = _FullDiskBackend(OSError(13, "Permission denied"))
    monkeypatch.setattr(sup, "make_remote_backend", lambda _m: backend)

    with pytest.raises(OSError):
        asyncio.run(sup.poll_remote(run_dir, meta))
    assert backend.polled is False


def _supervisor(project: Path) -> tuple[JobSupervisor, list[tuple[str, str, str]]]:
    notices: list[tuple[str, str, str]] = []
    hooks = SupervisorHooks(on_notice=lambda r, t, s: notices.append((r, t, s)))
    return JobSupervisor(project, hooks=hooks), notices


def test_the_advisory_fires_once_and_never_becomes_a_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sup, notices = _supervisor(tmp_path)
    monkeypatch.setattr(diskspace, "free_mb", lambda _p: 4)

    sup.maybe_notify_disk_full("run_1")
    sup.maybe_notify_disk_full("run_1")

    assert len(notices) == 1
    assert "out of disk space (4 MB free)" in notices[0][1]
    # An advisory: a parked auto-mode agent must stay parked.
    assert sup.pending_completions == {}


def test_a_quota_still_produces_an_advisory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EDQUOT with room to spare: detected by errno, invisible to the probe."""
    sup, notices = _supervisor(tmp_path)
    monkeypatch.setattr(diskspace, "free_mb", lambda _p: 500_000)
    diskspace.note_enospc(OSError(28, "No space left on device"))

    sup.maybe_notify_disk_full("run_1")

    assert len(notices) == 1


def test_no_advisory_while_there_is_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sup, notices = _supervisor(tmp_path)
    monkeypatch.setattr(diskspace, "free_mb", lambda _p: 500_000)

    sup.maybe_notify_disk_full("run_1")

    assert notices == []


def test_a_full_disk_is_not_reported_as_a_stalled_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """progress.jsonl froze because it cannot be written.

    Calling that a stall would invite the agent to offer stop_training()
    on a run that is doing fine.
    """
    import os
    import time

    run_dir = tmp_path / "runs" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}\n")
    progress = run_dir / "progress.jsonl"
    progress.write_text(json.dumps({"phase": "training", "step": 50}) + "\n")
    old = time.time() - 600 * 60
    os.utime(progress, (old, old))

    sup, notices = _supervisor(tmp_path)
    monkeypatch.setattr(diskspace, "free_mb", lambda _p: 4)

    sup.maybe_notify_stall("run_1")

    assert notices == []
