"""LQH_EVENT_JSON sentinel emission from lqh.train.progress.

These tests guard the cloud-sandbox progress pipeline. The trainer
inside a cloud sandbox writes progress events to a file the host
can't read directly; the sentinel-on-stdout path is what lets the
cloud runner translate trainer progress into SSE events the client
consumes.

Three things must hold:

  1. With LQH_JOB_ID unset (local/SSH path), no sentinel is emitted.
     Local runs must not get noisy stdout.
  2. With LQH_JOB_ID set (cloud path), every write_progress and
     write_status call produces exactly one parseable sentinel.
  3. The wire format matches what the backend's cloud runner
     parseSentinel expects: ``LQH_EVENT_JSON: {"kind": "...", "payload": {...}}``.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from lqh.train import progress


SENTINEL_PREFIX = "LQH_EVENT_JSON:"


def _parse_sentinels(captured: str) -> list[dict]:
    """Pull every sentinel line out of captured stdout, parsed as JSON.
    Non-sentinel lines (regular log output) are ignored."""
    out: list[dict] = []
    for line in captured.splitlines():
        if line.startswith(SENTINEL_PREFIX):
            body = line[len(SENTINEL_PREFIX):].strip()
            out.append(json.loads(body))
    return out


def test_no_sentinel_outside_sandbox(tmp_path: Path, monkeypatch):
    """Local + SSH runs (no LQH_JOB_ID) emit nothing to stdout."""
    monkeypatch.delenv("LQH_JOB_ID", raising=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        progress.write_progress(tmp_path, step=1, loss=0.5)
        progress.write_status(tmp_path, "completed")
    # File side still works.
    assert (tmp_path / "progress.jsonl").exists()
    # But no sentinel chatter on stdout.
    assert SENTINEL_PREFIX not in buf.getvalue()


def test_progress_sentinel_in_sandbox(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LQH_JOB_ID", "test-job-123")
    buf = io.StringIO()
    with redirect_stdout(buf):
        progress.write_progress(tmp_path, step=42, loss=1.5, lr=2e-5, epoch=0.7)

    events = _parse_sentinels(buf.getvalue())
    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == "progress"
    assert ev["payload"]["step"] == 42
    assert ev["payload"]["loss"] == 1.5
    assert ev["payload"]["lr"] == 2e-5
    assert ev["payload"]["epoch"] == 0.7
    # Timestamp must be present (the SSE consumer relies on it).
    assert "timestamp" in ev["payload"]


def test_status_sentinel_in_sandbox(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LQH_JOB_ID", "test-job-123")
    buf = io.StringIO()
    with redirect_stdout(buf):
        progress.write_status(tmp_path, "completed")
        progress.write_status(tmp_path, "failed", error="OOM at step 1000")

    events = _parse_sentinels(buf.getvalue())
    assert len(events) == 2
    assert events[0]["kind"] == "status"
    assert events[0]["payload"]["status"] == "completed"
    assert events[1]["kind"] == "status"
    assert events[1]["payload"]["status"] == "failed"
    assert events[1]["payload"]["error"] == "OOM at step 1000"


def test_sentinel_wire_format_one_line(tmp_path: Path, monkeypatch):
    """The cloud runner parses line-by-line; payloads MUST be single-line
    JSON or the SSE consumer will see truncated events."""
    monkeypatch.setenv("LQH_JOB_ID", "test-job-123")
    buf = io.StringIO()
    with redirect_stdout(buf):
        # Include a nested dict in extra to be sure complex payloads don't
        # break the one-line invariant.
        progress.write_progress(
            tmp_path,
            step=1,
            loss=0.3,
            extra={"grad_norm": 2.5, "lr_schedule": {"warmup": 100, "decay": "linear"}},
        )

    sentinel_lines = [
        l for l in buf.getvalue().splitlines() if l.startswith(SENTINEL_PREFIX)
    ]
    assert len(sentinel_lines) == 1, f"expected one sentinel line, got {sentinel_lines!r}"
    # The line itself must parse cleanly after the prefix.
    body = sentinel_lines[0][len(SENTINEL_PREFIX):].strip()
    parsed = json.loads(body)
    # write_progress merges ``extra`` into the payload (not nested
    # under an "extra" key), so lr_schedule lands at the top of payload.
    assert parsed["payload"]["lr_schedule"]["warmup"] == 100
    assert parsed["payload"]["grad_norm"] == 2.5


def test_emit_sentinel_swallows_unserializable(tmp_path: Path, monkeypatch):
    """A weird value type in extra (e.g. a custom object) must NOT take
    down a training run. The fallback _json_default coerces to str."""
    monkeypatch.setenv("LQH_JOB_ID", "test-job-123")

    class Weird:
        def __repr__(self) -> str:
            return "Weird()"

    buf = io.StringIO()
    with redirect_stdout(buf):
        # Should not raise.
        progress.write_progress(
            tmp_path,
            step=1,
            extra={"strange": Weird()},
        )
    events = _parse_sentinels(buf.getvalue())
    assert len(events) == 1
    # The fallback stringifies — value should be "Weird()".
    assert "Weird()" in json.dumps(events[0]["payload"])


def test_file_still_written_when_sandbox(tmp_path: Path, monkeypatch):
    """Sentinel emission is ADDITIVE — progress.jsonl must still be
    written so the file-based watcher path (used by SSH+local) keeps
    working even when LQH_JOB_ID happens to be set."""
    monkeypatch.setenv("LQH_JOB_ID", "test-job-123")
    buf = io.StringIO()
    with redirect_stdout(buf):
        progress.write_progress(tmp_path, step=7, loss=0.1)

    jsonl = (tmp_path / "progress.jsonl").read_text().strip()
    assert jsonl, "progress.jsonl should be written even in sandbox mode"
    entry = json.loads(jsonl)
    assert entry["step"] == 7
    assert entry["loss"] == 0.1
