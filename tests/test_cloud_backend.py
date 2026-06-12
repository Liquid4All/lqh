"""CloudBackend round-trip + disconnect-resume tests.

What disconnect-resilience means here:

  1. The submit POST persists ``cloud_state.json`` containing
     ``{job_id, last_seq=0, status='pending'}``. After this, the client
     can crash, sleep, or be SIGKILL'd; the job continues server-side.

  2. On reconnect ``sync_progress`` reads ``cloud_state.json``, opens
     the SSE stream with ``?last_seq=<saved>``, and the server replays
     any events the client missed (from the ``cloud_job_events`` row
     log). We assert that resuming twice — first capturing some events,
     then "crashing" and re-opening — produces the same final files as
     a single uninterrupted run.

The fake backend is in-process httpx.MockTransport, same pattern as
``test_artifacts.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from lqh.remote.backend import RemoteConfig
from lqh.remote.cloud import CloudBackend, CloudError, _CloudState


# ---------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------


class _FakeCloudBackend:
    """Minimal stand-in for api.lqh.ai cloud routes. Stores submitted
    jobs and an event log indexed by seq, replays from a given seq on
    GET /stream."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.events: dict[str, list[dict]] = {}  # job_id -> list of SSE events
        self.cancelled: set[str] = set()
        self._next_id = 1

    def add_events(self, job_id: str, events: list[dict]) -> None:
        """Seed events in the synthetic log. Each event must look like
        ``{kind, payload}`` — seq is assigned automatically."""
        log = self.events.setdefault(job_id, [])
        for ev in events:
            seq = len(log) + 1
            log.append({
                "kind": ev["kind"],
                "seq": seq,
                "payload": ev.get("payload", {}),
                "ts": "2026-05-21T12:00:00Z",
            })

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if path == "/v1/cloud/jobs" and method == "POST":
            jid = f"job-{self._next_id:04d}"
            self._next_id += 1
            self.jobs[jid] = {
                "id": jid,
                "status": "running",
                "project_id": "test",
                "kind": "infer",
                "submitted_at": "2026-05-21T12:00:00Z",
            }
            return httpx.Response(201, json={"job_id": jid, "status": "pending"})

        if path.startswith("/v1/cloud/jobs/") and path.endswith("/stream") and method == "GET":
            jid = path.split("/")[4]
            if jid not in self.jobs:
                return httpx.Response(404, json={"error": {"message": "no job"}})
            last_seq = int(request.url.params.get("last_seq", "0"))
            # Build an SSE body containing every event with seq > last_seq.
            chunks = []
            for ev in self.events.get(jid, []):
                if ev["seq"] <= last_seq:
                    continue
                data = json.dumps({"ts": ev["ts"], "seq": ev["seq"], "payload": ev["payload"]})
                chunks.append(f"event: {ev['kind']}\n")
                chunks.append(f"data: {data}\n\n")
            body = "".join(chunks)
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=body,
            )

        if path.startswith("/v1/cloud/jobs/") and method == "GET":
            jid = path.split("/")[4]
            if jid not in self.jobs:
                return httpx.Response(404, json={"error": {"message": "no job"}})
            return httpx.Response(200, json=self.jobs[jid])

        if path.startswith("/v1/cloud/jobs/") and method == "DELETE":
            jid = path.split("/")[4]
            self.cancelled.add(jid)
            if jid in self.jobs:
                self.jobs[jid]["status"] = "cancelled"
            return httpx.Response(204)

        return httpx.Response(404, json={"error": {"message": "not mocked"}})


@pytest.fixture
def fake_cloud(monkeypatch):
    be = _FakeCloudBackend()
    transport = httpx.MockTransport(be.handler)
    real_async_client = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("lqh.remote.cloud.httpx.AsyncClient", _patched)
    return be


def _make_backend(project_dir: Path) -> CloudBackend:
    cfg = RemoteConfig(
        name="cloud",
        type="cloud",
        hostname="api.mock.lqh.test",
        remote_root="cloud:test",
    )
    return CloudBackend(
        cfg,
        project_dir,
        api_base="https://mock.lqh.test",
        token="test-token",
    )


# ---------------------------------------------------------------------
# Submit + happy-path streaming
# ---------------------------------------------------------------------


def test_submit_writes_state_and_run_metadata(tmp_path, fake_cloud):
    project = tmp_path / "proj"
    project.mkdir()
    run_dir = project / ".lqh" / "runs" / "run_001"
    backend = _make_backend(project)

    job_id = asyncio.run(backend.submit_run(
        str(run_dir),
        {"manifest": [], "type": "infer"},
        module="lqh.infer",
    ))
    assert job_id.startswith("job-")

    meta = json.loads((run_dir / "remote_job.json").read_text())
    assert meta["job_id"] == job_id
    assert meta["backend"] == "cloud"

    state = _CloudState.load(run_dir / "cloud_state.json")
    assert state is not None
    assert state.job_id == job_id
    assert state.last_seq == 0


def test_sync_progress_writes_progress_and_status(tmp_path, fake_cloud):
    project = tmp_path / "proj"
    project.mkdir()
    run_dir = project / ".lqh" / "runs" / "run_001"
    backend = _make_backend(project)

    job_id = asyncio.run(backend.submit_run(
        str(run_dir),
        {"manifest": [], "type": "infer"},
        module="lqh.infer",
    ))

    fake_cloud.add_events(job_id, [
        {"kind": "status", "payload": {"status": "running"}},
        {"kind": "log", "payload": {"stream": "stdout", "line": "starting"}},
        {"kind": "progress", "payload": {"step": 1, "loss": 1.5}},
        {"kind": "progress", "payload": {"step": 2, "loss": 1.2}},
        # Runner's final terminal carries exit_code; trainer sentinels
        # don't (and aren't job-terminal by themselves).
        {"kind": "status", "payload": {"status": "completed", "exit_code": 0}},
    ])

    asyncio.run(backend.sync_progress(f"cloud:{job_id}", str(run_dir)))

    # status.json reflects the terminal state.
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"

    # progress.jsonl has one row per progress event + one terminal status row.
    lines = (run_dir / "progress.jsonl").read_text().strip().splitlines()
    parsed = [json.loads(l) for l in lines]
    steps = [r.get("step") for r in parsed if "step" in r]
    assert steps == [1, 2]
    assert any(r.get("status") == "completed" for r in parsed)

    # stdout.log contains the log line.
    assert "starting" in (run_dir / "stdout.log").read_text()

    # last_seq is at the final event.
    state = _CloudState.load(run_dir / "cloud_state.json")
    assert state.last_seq == 5
    assert state.status == "completed"


# ---------------------------------------------------------------------
# The headline scenario: laptop sleeps mid-stream, then resumes.
# ---------------------------------------------------------------------


def test_disconnect_then_resume_replays_missed_events(tmp_path, fake_cloud):
    """Simulate the laptop-closes-mid-finetune scenario.

    Sequence:
      1. Submit a job. State persisted to disk.
      2. The server has 3 events; sync_progress consumes them, last_seq=3.
      3. The "client process exits" — we forget the in-memory backend.
      4. Server emits 4 more events (server-side persistence is what
         cloud_job_events handles in prod; we just append to the fake).
      5. A fresh CloudBackend instance reads cloud_state.json, opens
         the stream with ?last_seq=3, and consumes events 4..7.
      6. Final state on disk matches a hypothetical uninterrupted run.
    """
    project = tmp_path / "proj"
    project.mkdir()
    run_dir = project / ".lqh" / "runs" / "run_001"

    backend = _make_backend(project)
    job_id = asyncio.run(backend.submit_run(
        str(run_dir),
        {"manifest": [], "type": "infer"},
        module="lqh.infer",
    ))

    # ---- Pre-disconnect: first three events ----
    fake_cloud.add_events(job_id, [
        {"kind": "status", "payload": {"status": "running"}},
        {"kind": "log", "payload": {"stream": "stdout", "line": "pre-1"}},
        {"kind": "progress", "payload": {"step": 1, "loss": 2.0}},
    ])
    asyncio.run(backend.sync_progress(f"cloud:{job_id}", str(run_dir)))

    state_after_first = _CloudState.load(run_dir / "cloud_state.json")
    assert state_after_first.last_seq == 3, "first sync should have consumed 3 events"

    # ---- The "client crashes" — drop the backend instance. ----
    del backend

    # ---- Server emits more events while client is gone. ----
    fake_cloud.add_events(job_id, [
        {"kind": "log", "payload": {"stream": "stdout", "line": "post-1"}},
        {"kind": "progress", "payload": {"step": 2, "loss": 1.5}},
        {"kind": "progress", "payload": {"step": 3, "loss": 1.0}},
        {"kind": "status", "payload": {"status": "completed", "exit_code": 0}},
    ])

    # ---- Fresh client process picks up where we left off. ----
    backend2 = _make_backend(project)
    asyncio.run(backend2.sync_progress(f"cloud:{job_id}", str(run_dir)))

    final_state = _CloudState.load(run_dir / "cloud_state.json")
    assert final_state.last_seq == 7
    assert final_state.status == "completed"

    # progress.jsonl has steps from both phases — no duplicates, in order.
    lines = (run_dir / "progress.jsonl").read_text().strip().splitlines()
    parsed = [json.loads(l) for l in lines]
    steps = [r.get("step") for r in parsed if "step" in r]
    assert steps == [1, 2, 3], "step rows must be contiguous across the gap"

    # status.json reflects the post-disconnect terminal state.
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"

    # stdout.log captures both pre-disconnect and post-disconnect lines.
    log = (run_dir / "stdout.log").read_text()
    assert "pre-1" in log and "post-1" in log


def test_sync_progress_preserves_sweep_progress_payload(tmp_path, fake_cloud):
    project = tmp_path / "proj"
    project.mkdir()
    run_dir = project / ".lqh" / "runs" / "run_001"

    backend = _make_backend(project)
    job_id = asyncio.run(backend.submit_run(
        str(run_dir),
        {"manifest": [], "type": "sweep"},
        module="lqh.train.sweep",
    ))
    fake_cloud.add_events(job_id, [
        {
            "kind": "progress",
            "payload": {
                "phase": "sweep_config_progress",
                "config_id": "cfg",
                "config_index": 1,
                "n_configs": 6,
                "step": 42,
                "child_step": 42,
                "child_loss": 0.9,
                "child_max_steps": 300,
            },
        },
    ])

    asyncio.run(backend.sync_progress(f"cloud:{job_id}", str(run_dir)))

    rows = [
        json.loads(line)
        for line in (run_dir / "progress.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["phase"] == "sweep_config_progress"
    assert rows[-1]["config_index"] == 1
    assert rows[-1]["n_configs"] == 6
    assert rows[-1]["child_step"] == 42
    assert rows[-1]["child_max_steps"] == 300


def test_resume_does_not_replay_already_seen_events(tmp_path, fake_cloud):
    """If sync_progress is invoked twice in a row with no new server-side
    activity, the second call must be a no-op (no duplicate rows)."""
    project = tmp_path / "proj"
    project.mkdir()
    run_dir = project / ".lqh" / "runs" / "run_001"

    backend = _make_backend(project)
    job_id = asyncio.run(backend.submit_run(
        str(run_dir),
        {"manifest": [], "type": "infer"},
        module="lqh.infer",
    ))
    fake_cloud.add_events(job_id, [
        {"kind": "progress", "payload": {"step": 1, "loss": 1.0}},
        {"kind": "status", "payload": {"status": "completed", "exit_code": 0}},
    ])

    asyncio.run(backend.sync_progress(f"cloud:{job_id}", str(run_dir)))
    first_rows = (run_dir / "progress.jsonl").read_text()

    # Re-invoke; CloudBackend should short-circuit on terminal status.
    asyncio.run(backend.sync_progress(f"cloud:{job_id}", str(run_dir)))
    second_rows = (run_dir / "progress.jsonl").read_text()

    assert first_rows == second_rows, "re-syncing a terminal job must not duplicate rows"


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------


def test_poll_status_maps_states(tmp_path, fake_cloud):
    project = tmp_path / "proj"
    project.mkdir()
    backend = _make_backend(project)
    job_id = asyncio.run(backend.submit_run(
        str(project / ".lqh" / "runs" / "x"),
        {"manifest": [], "type": "infer"},
        module="lqh.infer",
    ))

    status = asyncio.run(backend.poll_status(job_id))
    assert status.state == "running"

    fake_cloud.jobs[job_id]["status"] = "completed"
    status = asyncio.run(backend.poll_status(job_id))
    assert status.state == "completed"


def test_poll_status_propagates_429(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    backend = _make_backend(project)

    async def rate_limited(job_id: str):
        raise CloudError("429: Too Many Requests")

    monkeypatch.setattr(backend, "_get_snapshot", rate_limited)

    with pytest.raises(CloudError, match="429"):
        asyncio.run(backend.poll_status("job-0001"))


def test_teardown_hits_delete(tmp_path, fake_cloud):
    project = tmp_path / "proj"
    project.mkdir()
    backend = _make_backend(project)
    job_id = asyncio.run(backend.submit_run(
        str(project / ".lqh" / "runs" / "x"),
        {"manifest": [], "type": "infer"},
        module="lqh.infer",
    ))
    asyncio.run(backend.teardown(job_id))
    assert job_id in fake_cloud.cancelled


def test_is_job_alive_false_when_terminal(tmp_path, fake_cloud):
    project = tmp_path / "proj"
    project.mkdir()
    backend = _make_backend(project)
    job_id = asyncio.run(backend.submit_run(
        str(project / ".lqh" / "runs" / "x"),
        {"manifest": [], "type": "infer"},
        module="lqh.infer",
    ))
    assert asyncio.run(backend.is_job_alive(job_id)) is True
    fake_cloud.jobs[job_id]["status"] = "completed"
    assert asyncio.run(backend.is_job_alive(job_id)) is False
