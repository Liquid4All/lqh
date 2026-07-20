"""Characterization tests for conversation session persistence.

Phase 0 of the persistency work (see PERSISTENCY_PLAN.md): these tests pin
down the CURRENT behavior of ``lqh.session.Session`` — including its known
defects — before the storage format changes. Tests marked with a
``CURRENT:`` comment document behavior that is expected to FLIP when the
append-only session format lands; the file/case names stay stable so the
flipped assertions become the regression suite.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from lqh.session import Session, sessions_dir


def _make_session(project_dir: Path, n_messages: int = 4) -> Session:
    session = Session.create(project_dir)
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        session.add_message({"role": role, "content": f"message {i}"})
    return session


# ---------------------------------------------------------------------------
# Round-trip basics
# ---------------------------------------------------------------------------


def test_round_trip_preserves_messages_and_order(project_dir: Path) -> None:
    session = _make_session(project_dir, n_messages=6)
    session.prompt_tokens = 123
    session.completion_tokens = 45
    session.save()

    loaded = Session.load(project_dir, session.id)

    assert loaded.id == session.id
    assert loaded.created_at == session.created_at
    assert loaded.prompt_tokens == 123
    assert loaded.completion_tokens == 45
    assert [m["content"] for m in loaded.messages] == [
        f"message {i}" for i in range(6)
    ]


def test_save_is_noop_without_user_message(project_dir: Path) -> None:
    session = Session.create(project_dir)
    session.add_message({"role": "system", "content": "system only"})

    assert list(sessions_dir(project_dir).glob("*")) == []


def test_add_message_persists_immediately(project_dir: Path) -> None:
    session = Session.create(project_dir)
    session.add_message({"role": "user", "content": "hello"})

    loaded = Session.load(project_dir, session.id)
    assert [m["content"] for m in loaded.messages] == ["hello"]

    session.add_message({"role": "assistant", "content": "hi"})
    loaded = Session.load(project_dir, session.id)
    assert [m["content"] for m in loaded.messages] == ["hello", "hi"]


def test_tool_call_message_shapes_survive_round_trip(project_dir: Path) -> None:
    session = Session.create(project_dir)
    session.add_message({"role": "user", "content": "run it"})
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "summary", "arguments": "{}"},
            }
        ],
    }
    tool = {"role": "tool", "tool_call_id": "call_1", "content": "result"}
    session.add_message(assistant)
    session.add_message(tool)

    loaded = Session.load(project_dir, session.id)
    assert loaded.messages[1] == assistant
    assert loaded.messages[2] == tool


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def test_list_sessions_sorted_newest_first(project_dir: Path) -> None:
    old = Session.create(project_dir)
    old.created_at = "2026-01-01T00:00:00+00:00"
    old.add_message({"role": "user", "content": "old task"})

    new = Session.create(project_dir)
    new.created_at = "2026-06-01T00:00:00+00:00"
    new.add_message({"role": "user", "content": "new task"})

    listed = Session.list_sessions(project_dir)
    assert [s["id"] for s in listed] == [new.id, old.id]
    assert listed[0]["preview"] == "new task"


def test_list_sessions_skips_malformed_header(project_dir: Path) -> None:
    good = Session.create(project_dir)
    good.add_message({"role": "user", "content": "fine"})

    sdir = sessions_dir(project_dir)
    (sdir / "not-json.jsonl").write_text("this is not json\n")
    (sdir / "no-marker.jsonl").write_text(json.dumps({"id": "x"}) + "\n")

    listed = Session.list_sessions(project_dir)
    assert [s["id"] for s in listed] == [good.id]


def test_metadata_has_lifecycle_state(project_dir: Path) -> None:
    """Flipped from Phase 0: meta.json now records state/updated_at/pid so
    a crash (stale active + dead pid) is distinguishable from a clean
    exit."""
    session = _make_session(project_dir)
    meta = json.loads(
        (sessions_dir(project_dir) / session.id / "meta.json").read_text()
    )

    assert meta["state"] == "active"
    assert meta["updated_at"]
    assert meta["pid"] > 0
    assert meta["last_seq"] == 4


def test_mark_state_and_repair(project_dir: Path) -> None:
    session = _make_session(project_dir)

    session.mark_state("completed")
    assert Session.list_sessions(project_dir)[0]["state"] == "completed"
    assert Session.repair_states(project_dir) == []

    # Simulate a crash: active state owned by a dead pid.
    meta_path = sessions_dir(project_dir) / session.id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["state"] = "active"
    meta["pid"] = 2**22 + 1  # extremely unlikely to be alive
    meta_path.write_text(json.dumps(meta))

    assert Session.repair_states(project_dir) == [session.id]
    assert Session.list_sessions(project_dir)[0]["state"] == "interrupted"


# ---------------------------------------------------------------------------
# Durability defects (documented, expected to flip)
# ---------------------------------------------------------------------------


class _ExplodingFile:
    """File wrapper that writes ``fail_after`` characters then raises."""

    def __init__(self, inner, fail_after: int) -> None:
        self._inner = inner
        self._budget = fail_after

    def write(self, data: str) -> int:
        if len(data) > self._budget:
            self._inner.write(data[: self._budget])
            self._inner.flush()
            self._budget = 0
            raise OSError("disk full (simulated kill mid-write)")
        self._budget -= len(data)
        return self._inner.write(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._inner.close()
        return False


def test_kill_during_write_keeps_acknowledged_messages(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipped from Phase 0: the log is append-only, so a write that dies
    partway can at worst tear its own line — every previously acknowledged
    message stays readable."""
    session = _make_session(project_dir, n_messages=6)
    path = sessions_dir(project_dir) / session.id / "messages.jsonl"
    assert len(path.read_text().splitlines()) == 6

    real_open = builtins.open

    def flaky_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if str(file) == str(path) and "a" in mode:
            return _ExplodingFile(handle, fail_after=30)
        return handle

    monkeypatch.setattr(builtins, "open", flaky_open)
    with pytest.raises(OSError):
        session.add_message({"role": "user", "content": "one more"})
    monkeypatch.undo()

    loaded = Session.load(project_dir, session.id)
    assert [m["content"] for m in loaded.messages] == [
        f"message {i}" for i in range(6)
    ]


def test_corrupt_final_line_is_quarantined(project_dir: Path) -> None:
    """Flipped from Phase 0: a garbage final line (torn append) is moved to
    quarantine.log and every intact message loads."""
    session = _make_session(project_dir, n_messages=4)
    path = sessions_dir(project_dir) / session.id / "messages.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"seq": 5, "msg": {"role": "assistant", "content": "torn wri')

    loaded = Session.load(project_dir, session.id)

    assert [m["content"] for m in loaded.messages] == [
        f"message {i}" for i in range(4)
    ]
    quarantine = sessions_dir(project_dir) / session.id / "quarantine.log"
    assert "torn wri" in quarantine.read_text()
    # The log itself is clean again: appending and reloading works.
    loaded.add_message({"role": "user", "content": "after repair"})
    reloaded = Session.load(project_dir, session.id)
    assert reloaded.messages[-1]["content"] == "after repair"


# ---------------------------------------------------------------------------
# Concurrency: two live handles on the same session
# ---------------------------------------------------------------------------


def test_concurrent_handles_never_duplicate_sequences(project_dir: Path) -> None:
    """Two Session objects (two CLIs / a stale handle) appending to the
    same conversation must allocate unique, monotonic sequence numbers."""
    a = Session.create(project_dir)
    a.add_message({"role": "user", "content": "from a 1"})
    b = Session.load(project_dir, a.id)

    b.add_message({"role": "user", "content": "from b 1"})
    a.add_message({"role": "user", "content": "from a 2"})
    b.add_message({"role": "user", "content": "from b 2"})

    entries = Session.load(project_dir, a.id).log_entries()
    seqs = [seq for seq, _ in entries]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs)) == 4
    contents = {m["content"] for _, m in entries}
    assert contents == {"from a 1", "from b 1", "from a 2", "from b 2"}


def test_stale_save_does_not_regress_last_seq(project_dir: Path) -> None:
    """A stale handle calling save() (e.g. the TUI's periodic meta flush)
    must not overwrite a newer writer's last_seq — that regression would
    make the next append allocate a duplicate sequence."""
    a = Session.create(project_dir)
    a.add_message({"role": "user", "content": "one"})

    b = Session.load(project_dir, a.id)
    b.add_message({"role": "user", "content": "two"})
    b.add_message({"role": "user", "content": "three"})

    a.save()  # stale: a.last_seq is 1, disk is 3

    meta = json.loads(
        (sessions_dir(project_dir) / a.id / "meta.json").read_text()
    )
    assert meta["last_seq"] == 3

    a.add_message({"role": "user", "content": "four"})
    entries = Session.load(project_dir, a.id).log_entries()
    seqs = [seq for seq, _ in entries]
    assert len(seqs) == len(set(seqs)) == 4


def test_seq_allocation_survives_meta_write_failure(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sequence allocation consults the log tail, not just meta.json — a
    failed meta update after a successful append must not let a stale
    handle allocate a duplicate seq."""
    import lqh.session as session_module

    a = Session.create(project_dir)
    a.add_message({"role": "user", "content": "one"})
    b = Session.load(project_dir, a.id)

    # b appends, but its meta update fails: the log has seq 2, meta says 1.
    real_write = session_module.atomic_write_json

    def failing_meta_write(path, obj, **kwargs):
        if path.name == "meta.json":
            raise OSError("disk full (simulated)")
        return real_write(path, obj, **kwargs)

    monkeypatch.setattr(session_module, "atomic_write_json", failing_meta_write)
    b.add_message({"role": "user", "content": "two"})
    monkeypatch.undo()
    meta = json.loads(
        (sessions_dir(project_dir) / a.id / "meta.json").read_text()
    )
    assert meta["last_seq"] == 1  # meta is genuinely stale

    # The stale handle appends next: must allocate seq 3, not reuse 2.
    a.add_message({"role": "user", "content": "three"})

    entries = Session.load(project_dir, a.id).log_entries()
    seqs = [seq for seq, _ in entries]
    assert seqs == [1, 2, 3]
    assert [m["content"] for _, m in entries] == ["one", "two", "three"]


def test_seq_allocation_survives_huge_tail_message(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log-tail scan must find the last envelope even when it is
    larger than the initial read window — otherwise a stale handle would
    reuse its sequence number."""
    import lqh.session as session_module

    a = Session.create(project_dir)
    a.add_message({"role": "user", "content": "one"})
    b = Session.load(project_dir, a.id)

    real_write = session_module.atomic_write_json

    def failing_meta_write(path, obj, **kwargs):
        if path.name == "meta.json":
            raise OSError("disk full (simulated)")
        return real_write(path, obj, **kwargs)

    monkeypatch.setattr(session_module, "atomic_write_json", failing_meta_write)
    b.add_message({"role": "user", "content": "x" * 200_000})  # > 64 KiB line
    monkeypatch.undo()

    a.add_message({"role": "user", "content": "three"})

    entries = Session.load(project_dir, a.id).log_entries()
    seqs = [seq for seq, _ in entries]
    assert seqs == [1, 2, 3]


def test_repair_preserves_recency(project_dir: Path) -> None:
    """Repair must not bump updated_at — an old abandoned session would
    otherwise hijack the 'newest interrupted' startup offer."""
    session = _make_session(project_dir)
    meta_path = sessions_dir(project_dir) / session.id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["state"] = "active"
    meta["pid"] = 2**22 + 7
    meta["updated_at"] = "2026-01-01T00:00:00+00:00"
    meta_path.write_text(json.dumps(meta))

    assert Session.repair_states(project_dir) == [session.id]

    repaired = json.loads(meta_path.read_text())
    assert repaired["state"] == "interrupted"
    assert repaired["updated_at"] == "2026-01-01T00:00:00+00:00"
    assert repaired["interrupted_at"]


def test_session_without_meta_is_still_discoverable(project_dir: Path) -> None:
    """If the meta write failed after a durable append, the raw log alone
    must keep the session listed and resumable."""
    session = _make_session(project_dir, n_messages=2)
    (sessions_dir(project_dir) / session.id / "meta.json").unlink()

    listed = Session.list_sessions(project_dir)
    assert [s["id"] for s in listed] == [session.id]
    assert listed[0]["preview"] == "message 0"

    loaded = Session.load(project_dir, session.id)
    assert [m["content"] for m in loaded.messages] == ["message 0", "message 1"]

    # Recovery is complete: created_at/updated_at/preview come from the
    # log envelopes, so the next save persists real values, not blanks.
    assert loaded.created_at
    assert loaded.updated_at
    assert loaded.save()
    meta = json.loads(
        (sessions_dir(project_dir) / session.id / "meta.json").read_text()
    )
    assert meta["preview"] == "message 0"
    assert meta["created_at"] == loaded.created_at


def test_torn_checkpoint_tail_is_quarantined(project_dir: Path) -> None:
    """A torn checkpoints.jsonl tail must be repaired on load — otherwise
    the next checkpoint append concatenates onto the partial line and
    compaction progress silently stops advancing."""
    session = _make_session(project_dir, n_messages=8)
    session.set_compacted_view("first summary", covers_to_seq=4)
    ckpt_path = sessions_dir(project_dir) / session.id / "checkpoints.jsonl"
    with open(ckpt_path, "a", encoding="utf-8") as f:
        f.write('{"schema_version": 1, "covers_to_seq": 6, "summary": "torn wri')

    loaded = Session.load(project_dir, session.id)

    assert loaded.latest_checkpoint()["covers_to_seq"] == 4
    quarantine = sessions_dir(project_dir) / session.id / "quarantine.log"
    assert "torn wri" in quarantine.read_text()

    # Appending a new checkpoint works and stays readable.
    loaded.set_compacted_view("second summary", covers_to_seq=6)
    reloaded = Session.load(project_dir, session.id)
    assert reloaded.latest_checkpoint()["covers_to_seq"] == 6


def test_malformed_checkpoint_never_blocks_loading(project_dir: Path) -> None:
    """Checkpoints are a DERIVED cache: a structurally malformed record
    (hand-edited, corrupted) is skipped — it must never make the intact
    raw transcript unresumable."""
    session = _make_session(project_dir, n_messages=6)
    ckpt_path = sessions_dir(project_dir) / session.id / "checkpoints.jsonl"
    ckpt_path.write_text(
        json.dumps({
            "schema_version": 1,
            "covers_to_seq": "not-an-int",
            "carried_seqs": [],
            "summary": "bogus",
        }) + "\n"
    )

    loaded = Session.load(project_dir, session.id)

    # The full raw transcript loads; the corrupt checkpoint is ignored.
    assert len(loaded.messages) == 6
    assert loaded.latest_checkpoint() is None

    # A malformed carried_seqs list is equally non-fatal.
    ckpt_path.write_text(
        json.dumps({
            "schema_version": 1,
            "covers_to_seq": 3,
            "carried_seqs": ["x", {}],
            "summary": "bogus",
        }) + "\n"
    )
    reloaded = Session.load(project_dir, session.id)
    assert len(reloaded.messages) == 6


# ---------------------------------------------------------------------------
# Legacy (v1 single-file) migration
# ---------------------------------------------------------------------------


def _write_legacy_session(
    project_dir: Path,
    session_id: str,
    messages: list[dict],
    *,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> Path:
    path = sessions_dir(project_dir) / f"{session_id}.jsonl"
    header = {
        "__metadata__": True,
        "id": session_id,
        "created_at": created_at,
        "preview": "legacy preview",
        "prompt_tokens": 11,
        "completion_tokens": 7,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for message in messages:
            f.write(json.dumps(message) + "\n")
    return path


def test_legacy_session_migrates_on_load(project_dir: Path) -> None:
    msgs = [
        {"role": "user", "content": "legacy question"},
        {"role": "assistant", "content": "legacy answer"},
    ]
    legacy_path = _write_legacy_session(project_dir, "legacy-1", msgs)

    loaded = Session.load(project_dir, "legacy-1")

    assert [m["content"] for m in loaded.messages] == [
        "legacy question",
        "legacy answer",
    ]
    assert loaded.created_at == "2026-01-01T00:00:00+00:00"
    assert loaded.prompt_tokens == 11
    assert loaded.completion_tokens == 7
    assert loaded.state == "completed"
    # Original preserved as a backup; directory format now authoritative.
    assert not legacy_path.exists()
    assert legacy_path.with_suffix(".jsonl.bak").exists()
    assert (sessions_dir(project_dir) / "legacy-1" / "meta.json").exists()

    # Migrated sessions appear exactly once in the listing.
    listed = Session.list_sessions(project_dir)
    assert [s["id"] for s in listed] == ["legacy-1"]
    assert listed[0]["state"] == "completed"

    # And they accept new messages in the new format.
    loaded.add_message({"role": "user", "content": "continued"})
    reloaded = Session.load(project_dir, "legacy-1")
    assert reloaded.messages[-1]["content"] == "continued"


def test_legacy_migration_quarantines_bad_lines(project_dir: Path) -> None:
    path = _write_legacy_session(
        project_dir, "legacy-2", [{"role": "user", "content": "ok"}]
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write("not json at all\n")
        f.write(json.dumps({"role": "assistant", "content": "fine"}) + "\n")

    loaded = Session.load(project_dir, "legacy-2")

    assert [m["content"] for m in loaded.messages] == ["ok", "fine"]
    quarantine = sessions_dir(project_dir) / "legacy-2" / "quarantine.log"
    assert "not json at all" in quarantine.read_text()
