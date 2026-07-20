"""Durable conversation storage.

Format v2 (see PERSISTENCY_PLAN.md): one directory per conversation under
``.lqh/conversations/<id>/``:

- ``messages.jsonl`` — append-only log of envelopes
  ``{"seq": N, "ts": iso, "msg": {...}}``. Never rewritten; fsync'd per
  append. This is the raw transcript and survives compaction.
- ``meta.json`` — atomically replaced metadata (state, timestamps, tokens,
  preview, last_seq, owning pid).
- ``checkpoints.jsonl`` — derived compaction summaries with coverage
  markers. Safe to delete; only the last line is used.
- ``lock`` — cross-process append lock.
- ``quarantine.log`` — corrupt tail bytes moved aside on load.

``Session.messages`` is the in-memory *working view* sent to the API: the
full log normally, or ``carried system messages + summary + tail`` after
compaction. Compaction never mutates the log (`set_compacted_view`).

Legacy v1 single-file sessions (``<id>.jsonl`` with a metadata header
line) are migrated lazily on load; the original is kept as ``.jsonl.bak``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lqh.fsio import append_line_durable, atomic_write_json, file_lock

logger = logging.getLogger("lqh.session")

SCHEMA_VERSION = 2

STATE_ACTIVE = "active"
STATE_INTERRUPTED = "interrupted"
STATE_COMPLETED = "completed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_start_time(pid: int) -> int | None:
    """Process start time in clock ticks since boot (Linux), else None.

    Field 22 of /proc/<pid>/stat; parsed after the last ')' because the
    comm field may contain spaces/parens. Used to defeat PID reuse: a
    recycled PID has a different start time than the one recorded when
    the session was claimed.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        # fields[0] is field 3 ("state"); field 22 is fields[19].
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _pid_alive(pid: int | None, pid_start: int | None = None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    if pid_start is not None:
        current_start = _pid_start_time(pid)
        if current_start is not None and current_start != pid_start:
            return False  # PID reused by an unrelated process
    return True


def sessions_dir(project_dir: Path) -> Path:
    """Return the conversations directory for a project, creating it if needed."""
    path = project_dir / ".lqh" / "conversations"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Session:
    id: str
    project_dir: Path
    messages: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    state: str = STATE_ACTIVE
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_seq: int = 0
    # Optional human/agent-assigned title (meta-only; preview is derived).
    title: str = ""
    # Preview of the first user message, frozen at first persist so it
    # survives compaction of the working view.
    _preview: str = ""
    # True once the on-disk directory exists (lazy creation: nothing is
    # written until the conversation has a user message).
    _persisted: bool = False

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def _dir(self) -> Path:
        return sessions_dir(self.project_dir) / self.id

    @property
    def _messages_path(self) -> Path:
        return self._dir / "messages.jsonl"

    @property
    def _meta_path(self) -> Path:
        return self._dir / "meta.json"

    @property
    def _checkpoints_path(self) -> Path:
        return self._dir / "checkpoints.jsonl"

    @property
    def _lock_path(self) -> Path:
        return self._dir / "lock"

    # ------------------------------------------------------------------
    # Creation / basic accessors
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, project_dir: Path) -> Session:
        """Create a new session with a fresh UUID and current timestamp.

        Nothing touches the disk until the first user message is appended.
        """
        now = _now()
        return cls(
            id=str(uuid.uuid4()),
            project_dir=project_dir,
            messages=[],
            created_at=now,
            updated_at=now,
        )

    def first_user_message(self) -> str | None:
        """Return the content of the first user message, or None."""
        for msg in self.messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _meta_payload(self) -> dict[str, Any]:
        pid = os.getpid()
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state,
            "pid": pid,
            "pid_start": _pid_start_time(pid),
            "title": self.title,
            "preview": self._preview,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "last_seq": self.last_seq,
        }

    def _write_meta_unlocked(self) -> bool:
        """Write meta.json; caller must hold the session lock.

        ``last_seq`` is merged with the on-disk value first so a stale
        Session object (e.g. an old handle calling save() concurrently
        with a fresher writer) can never regress the sequence counter —
        that regression would let the next append allocate a duplicate
        seq. Returns False when the write failed (callers like
        ``mark_state`` surface this instead of pretending success).
        """
        self.last_seq = max(self.last_seq, self._disk_last_seq())
        try:
            atomic_write_json(self._meta_path, self._meta_payload())
            return True
        except OSError:
            # The message log append already succeeded (or nothing durable
            # changed); stale metadata is recoverable from the log itself
            # (list_sessions falls back to scanning messages.jsonl).
            logger.warning("session meta write failed for %s", self.id, exc_info=True)
            return False

    def _write_meta(self) -> bool:
        with file_lock(self._lock_path):
            return self._write_meta_unlocked()

    def _disk_last_seq(self) -> int:
        """Best-effort read of last_seq from disk (for cross-process appends)."""
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            return int(meta.get("last_seq", 0))
        except (OSError, ValueError, TypeError):
            return 0

    def _log_tail_seq(self) -> int:
        """Highest seq in messages.jsonl itself — the authoritative source.

        meta.json can lag the log (its write is best-effort after the
        append), so sequence allocation must consult the log tail or a
        concurrent stale handle could allocate a duplicate seq. The read
        window grows until the final envelope parses — a single message
        larger than the initial window must not be skipped (its seq would
        be reused).
        """
        path = self._messages_path
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                window = 65536
                while True:
                    f.seek(max(0, size - window))
                    tail = f.read().decode("utf-8", errors="replace")
                    lines = tail.splitlines()
                    if size > window:
                        # The first line is (likely) a partial read —
                        # only trust lines after the first newline.
                        lines = lines[1:]
                    for line in reversed(lines):
                        if not line.strip():
                            continue
                        try:
                            return int(json.loads(line)["seq"])
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                            continue
                    if window >= size:
                        return 0
                    window *= 8
        except OSError:
            return 0

    def _append_to_log(self, message: dict) -> None:
        """Durably append one message envelope under the session lock."""
        with file_lock(self._lock_path):
            self.last_seq = max(
                self.last_seq, self._disk_last_seq(), self._log_tail_seq()
            ) + 1
            envelope = {"seq": self.last_seq, "ts": _now(), "msg": message}
            append_line_durable(self._messages_path, json.dumps(envelope))
            self.updated_at = _now()
            self._write_meta_unlocked()

    def add_message(self, message: dict) -> None:
        """Append a message to the conversation and persist it durably.

        Until the conversation has a user message nothing is written
        (matching the historical "empty sessions leave no residue"
        behavior); the buffered prefix is flushed together with the first
        user message.
        """
        self.messages.append(message)

        if not self._persisted:
            if self.first_user_message() is None:
                return
            self._preview = (
                (self.first_user_message() or "")[:80].replace("\n", " ").strip()
            )
            self._dir.mkdir(parents=True, exist_ok=True)
            self._persisted = True
            for buffered in self.messages:
                self._append_to_log(buffered)
            return

        self._append_to_log(message)

    def save(self) -> bool:
        """Flush metadata (tokens, timestamps, state) to disk.

        Messages are already durable at ``add_message`` time; this only
        refreshes ``meta.json``. No-op (True) for never-persisted
        sessions; returns False when the write failed so callers can
        react instead of assuming the state landed.
        """
        if not self._persisted:
            return True
        self.updated_at = _now()
        return self._write_meta()

    def mark_state(self, state: str) -> bool:
        """Atomically update the lifecycle state (active/interrupted/completed).

        Returns False (and logs) when nothing could be persisted — a
        "completed" that silently failed would otherwise resurface as an
        interrupted-session offer on the next start.
        """
        self.state = state
        return self.save()

    # ------------------------------------------------------------------
    # Log access and compaction
    # ------------------------------------------------------------------

    def _read_log_entries(self) -> list[tuple[int, dict]]:
        """Parse messages.jsonl into (seq, message) pairs, tolerating damage."""
        if not self._messages_path.exists():
            return []
        entries: list[tuple[int, dict]] = []
        try:
            raw_lines = self._messages_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.warning("session log unreadable for %s", self.id, exc_info=True)
            return []
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                envelope = json.loads(line)
                seq = int(envelope["seq"])
                msg = envelope["msg"]
                if not isinstance(msg, dict):
                    raise TypeError("msg is not a dict")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning(
                    "skipping malformed log line in session %s", self.id
                )
                continue
            entries.append((seq, msg))
        return entries

    def log_entries(self) -> list[tuple[int, dict]]:
        """Return (seq, message) pairs from the durable log."""
        return self._read_log_entries()

    def read_log(self, limit: int | None = None) -> list[dict]:
        """Return raw transcript messages (full history, ignoring compaction).

        For never-persisted sessions this is the in-memory message list.
        ``limit`` returns only the newest N messages.
        """
        if not self._persisted:
            msgs = list(self.messages)
        else:
            msgs = [msg for _, msg in self._read_log_entries()]
        if limit is not None:
            msgs = msgs[-limit:]
        return msgs

    def latest_checkpoint(self) -> dict | None:
        """Return the most recent valid checkpoint record, or None."""
        if not self._checkpoints_path.exists():
            return None
        latest: dict | None = None
        try:
            lines = self._checkpoints_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if self._valid_checkpoint(record):
                latest = record
        return latest

    @staticmethod
    def _valid_checkpoint(record: Any) -> bool:
        """Structural validation of a checkpoint record.

        Checkpoints are a DERIVED cache over the raw log — a malformed
        one (hand-edited, torn write) must be skipped, never allowed to
        make the intact transcript unloadable.
        """
        if not isinstance(record, dict):
            return False
        try:
            int(record["covers_to_seq"])
        except (KeyError, TypeError, ValueError):
            return False
        carried = record.get("carried_seqs", [])
        if not isinstance(carried, list):
            return False
        try:
            [int(seq) for seq in carried]
        except (TypeError, ValueError):
            return False
        return isinstance(record.get("summary", ""), str)

    @staticmethod
    def _summary_message(covers_to_seq: int, summary: str) -> dict:
        return {
            "role": "system",
            "content": (
                f"[Context compacted] Summary of the conversation up to "
                f"message {covers_to_seq}:\n\n{summary}"
            ),
        }

    def set_compacted_view(
        self, summary_text: str, *, covers_to_seq: int, model: str = ""
    ) -> None:
        """Record a compaction checkpoint and rebuild the working view.

        The raw log is never touched. The new view is::

            carried system messages (from the covered log range)
            + summary system message
            + every log message with seq > covers_to_seq

        The caller chooses ``covers_to_seq`` and MUST only cover messages
        that actually entered the summary — a coverage marker beyond the
        summarized range would silently drop conversation understanding
        (the raw log keeps the bytes, but no future context would ever
        see them). Raises ``ValueError`` on an empty/invalid coverage.
        """
        if not self._persisted:
            raise ValueError("cannot compact a session with no persisted messages")
        # Validate and append under one lock: checking the previous
        # checkpoint outside it would let two concurrent compactions both
        # pass validation and append in an order where the lower-coverage
        # checkpoint ends up last (and therefore authoritative).
        with file_lock(self._lock_path):
            entries = self._read_log_entries()
            if not entries:
                raise ValueError("nothing to compact")
            if covers_to_seq >= entries[-1][0]:
                raise ValueError("coverage must leave at least one uncovered message")
            prev = self.latest_checkpoint()
            if prev and covers_to_seq <= int(prev.get("covers_to_seq", 0)):
                raise ValueError("coverage must advance past the previous checkpoint")
            if not any(seq <= covers_to_seq for seq, _ in entries):
                raise ValueError("coverage includes no messages")

            tail = [(seq, msg) for seq, msg in entries if seq > covers_to_seq]
            # Carry only the NEWEST few covered system messages (active
            # skill instructions). Older ones — superseded skills, stale
            # injections — are represented by the summary instead;
            # carrying them all forever would grow the view unboundedly
            # and retain conflicting instructions.
            carried = [
                (seq, msg)
                for seq, msg in entries
                if seq <= covers_to_seq and msg.get("role") == "system"
            ][-4:]

            checkpoint = {
                "schema_version": 1,
                "created_at": _now(),
                "model": model,
                "covers_to_seq": covers_to_seq,
                "carried_seqs": [seq for seq, _ in carried],
                "summary": summary_text,
            }
            append_line_durable(self._checkpoints_path, json.dumps(checkpoint))

        self.messages = (
            [msg for _, msg in carried]
            + [self._summary_message(covers_to_seq, summary_text)]
            + [msg for _, msg in tail]
        )

    def _assemble_view(self, entries: list[tuple[int, dict]]) -> list[dict]:
        checkpoint = self.latest_checkpoint()
        if checkpoint is None:
            return [msg for _, msg in entries]
        try:
            covers_to_seq = int(checkpoint.get("covers_to_seq", 0))
            carried_seqs = {int(s) for s in checkpoint.get("carried_seqs", [])}
            summary = str(checkpoint.get("summary", ""))
        except (TypeError, ValueError):
            # Belt-and-braces (latest_checkpoint already validates): a
            # corrupt derived cache must never make the intact raw log
            # unresumable — fall back to the full transcript.
            logger.warning("malformed checkpoint ignored; using full log")
            return [msg for _, msg in entries]
        carried = [msg for seq, msg in entries if seq in carried_seqs]
        tail = [msg for seq, msg in entries if seq > covers_to_seq]
        return carried + [self._summary_message(covers_to_seq, summary)] + tail

    # ------------------------------------------------------------------
    # Loading and migration
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, project_dir: Path, session_id: str) -> Session:
        """Load a session, migrating legacy single-file sessions on demand."""
        sdir = sessions_dir(project_dir)
        v2_dir = sdir / session_id

        if not (v2_dir / "meta.json").exists():
            legacy = sdir / f"{session_id}.jsonl"
            if legacy.exists():
                cls._migrate_legacy(project_dir, session_id, legacy)
            elif not v2_dir.exists():
                raise FileNotFoundError(f"Session not found: {session_id}")

        session = cls(
            id=session_id,
            project_dir=project_dir,
            messages=[],
            _persisted=True,
        )
        session._quarantine_torn_tail()

        meta: dict[str, Any] = {}
        try:
            meta = json.loads((v2_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "session meta unreadable for %s; recovering from log", session_id
            )

        entries = session._read_log_entries()
        session.created_at = meta.get("created_at", "")
        session.updated_at = meta.get("updated_at", meta.get("created_at", ""))
        session.state = meta.get("state", STATE_COMPLETED)
        session.prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
        session.completion_tokens = int(meta.get("completion_tokens", 0) or 0)
        session.title = meta.get("title", "")
        session._preview = meta.get("preview", "")
        session.last_seq = max(
            int(meta.get("last_seq", 0) or 0),
            max((seq for seq, _ in entries), default=0),
        )
        session.messages = session._assemble_view(entries)
        # Meta missing/unreadable: recover what the log itself knows, so a
        # subsequent save() persists real values instead of blanks.
        if entries:
            if not session.created_at:
                session.created_at = session._log_ts(first=True)
            if not session.updated_at:
                session.updated_at = session._log_ts(first=False)
            if not session._preview:
                for _, msg in entries:
                    content = msg.get("content")
                    if msg.get("role") == "user" and isinstance(content, str):
                        session._preview = (
                            content[:80].replace("\n", " ").strip()
                        )
                        break
        return session

    def _log_ts(self, *, first: bool) -> str:
        """Timestamp of the first/last log envelope (raw file read)."""
        try:
            lines = self._messages_path.read_text(encoding="utf-8").splitlines()
            iterable = lines if first else reversed(lines)
            for line in iterable:
                if not line.strip():
                    continue
                try:
                    ts = json.loads(line).get("ts", "")
                    if ts:
                        return ts
                except (json.JSONDecodeError, AttributeError):
                    continue
        except OSError:
            pass
        return ""

    def _quarantine_torn_tail(self) -> None:
        """Repair torn final lines in the append-only session files.

        Covers messages.jsonl AND checkpoints.jsonl — a torn checkpoint
        line would otherwise corrupt every later checkpoint append,
        silently freezing compaction progress.
        """
        try:
            with file_lock(self._lock_path):
                self._repair_torn_file(self._messages_path)
                self._repair_torn_file(self._checkpoints_path)
        except OSError:
            logger.warning(
                "torn-tail check failed for session %s", self.id, exc_info=True
            )

    def _repair_torn_file(self, path: Path) -> None:
        """Move a torn final line of ``path`` into quarantine.log.

        Appends write ``line + "\\n"`` in a single call, so only the bytes
        after the last newline can be a partial (torn) append. Interior
        malformed lines are newline-delimited already and are simply
        skipped by the readers. If the tail parses as complete JSON, only
        the missing newline is repaired. Caller holds the session lock.
        """
        if not path.exists():
            return
        data = path.read_bytes()
        tail = data.split(b"\n")[-1]
        if not tail:
            return
        try:
            json.loads(tail.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            with open(self._dir / "quarantine.log", "ab") as q:
                q.write(
                    b"--- quarantined from " + path.name.encode() + b" "
                    + _now().encode() + b" ---\n" + tail + b"\n"
                )
            with open(path, "r+b") as f:
                f.truncate(len(data) - len(tail))
            logger.warning(
                "quarantined %d torn bytes of %s in session %s",
                len(tail),
                path.name,
                self.id,
            )
        else:
            with open(path, "ab") as f:
                f.write(b"\n")

    @classmethod
    def _migrate_legacy(
        cls, project_dir: Path, session_id: str, legacy_path: Path
    ) -> None:
        """Convert a v1 single-file session into the v2 directory format.

        The staging directory is renamed into place atomically; the legacy
        file is kept as ``<id>.jsonl.bak``. Malformed message lines are
        collected into quarantine.log rather than aborting the migration.
        """
        raw_lines = legacy_path.read_text(encoding="utf-8").splitlines()
        if not raw_lines:
            raise ValueError(f"Session file is empty: {legacy_path}")

        header = json.loads(raw_lines[0])
        messages: list[dict] = []
        bad_lines: list[str] = []
        for line in raw_lines[1:]:
            if not line.strip():
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines.append(line)

        created_at = header.get("created_at", "")
        try:
            mtime = datetime.fromtimestamp(
                legacy_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            mtime = created_at

        sdir = sessions_dir(project_dir)
        staging = sdir / f"{session_id}.migrating-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        with open(staging / "messages.jsonl", "w", encoding="utf-8") as f:
            for seq, msg in enumerate(messages, start=1):
                envelope = {"seq": seq, "ts": created_at, "msg": msg}
                f.write(json.dumps(envelope) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if bad_lines:
            with open(staging / "quarantine.log", "w", encoding="utf-8") as q:
                q.write(f"--- quarantined during migration {_now()} ---\n")
                q.write("\n".join(bad_lines) + "\n")

        atomic_write_json(
            staging / "meta.json",
            {
                "schema_version": SCHEMA_VERSION,
                "id": session_id,
                "created_at": created_at,
                "updated_at": mtime,
                "state": STATE_COMPLETED,
                "pid": None,
                "preview": header.get("preview", ""),
                "prompt_tokens": header.get("prompt_tokens", 0),
                "completion_tokens": header.get("completion_tokens", 0),
                "last_seq": len(messages),
                "migrated_from": legacy_path.name,
            },
        )

        target = sdir / session_id
        try:
            os.rename(staging, target)
        except OSError:
            # Another process won the migration race; discard our staging.
            shutil.rmtree(staging, ignore_errors=True)
            if not (target / "meta.json").exists():
                raise
        try:
            os.rename(legacy_path, legacy_path.with_suffix(".jsonl.bak"))
        except OSError:
            logger.warning(
                "could not archive legacy session file %s", legacy_path
            )

    # ------------------------------------------------------------------
    # Listing and repair
    # ------------------------------------------------------------------

    @classmethod
    def list_sessions(cls, project_dir: Path) -> list[dict]:
        """Return session summaries sorted newest-first.

        Covers both v2 directories and unmigrated legacy files. Each dict
        has ``id``, ``created_at``, ``updated_at``, ``preview``, ``state``,
        ``prompt_tokens``, and ``completion_tokens``.
        """
        sdir = sessions_dir(project_dir)
        sessions: dict[str, dict] = {}

        for meta_path in sdir.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            sid = meta.get("id", meta_path.parent.name)
            sessions[sid] = {
                "id": sid,
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", meta.get("created_at", "")),
                "preview": meta.get("preview", ""),
                "state": meta.get("state", STATE_COMPLETED),
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
            }

        # Fallback: v2 directories whose meta.json is missing/unreadable
        # (e.g. the meta write failed after a durable append). The raw log
        # is authoritative — acknowledged history must stay discoverable.
        for log_path in sdir.glob("*/messages.jsonl"):
            sid = log_path.parent.name
            if sid in sessions or sid.endswith(".migrating") or ".migrating-" in sid:
                continue
            try:
                preview = ""
                with open(log_path, "r", encoding="utf-8") as f:
                    for _ in range(50):
                        line = f.readline()
                        if not line:
                            break
                        try:
                            msg = json.loads(line).get("msg", {})
                        except (json.JSONDecodeError, AttributeError):
                            continue
                        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                            preview = msg["content"][:80].replace("\n", " ").strip()
                            break
                mtime = datetime.fromtimestamp(
                    log_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                sessions[sid] = {
                    "id": sid,
                    "created_at": mtime,
                    "updated_at": mtime,
                    "preview": preview,
                    "state": STATE_COMPLETED,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            except OSError:
                continue

        for jsonl_path in sdir.glob("*.jsonl"):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                if not first_line.strip():
                    continue
                metadata = json.loads(first_line)
                if not metadata.get("__metadata__"):
                    continue
                sid = metadata.get("id", jsonl_path.stem)
                if sid in sessions:
                    continue  # already migrated; the directory wins
                sessions[sid] = {
                    "id": sid,
                    "created_at": metadata.get("created_at", ""),
                    "updated_at": metadata.get("created_at", ""),
                    "preview": metadata.get("preview", ""),
                    "state": STATE_COMPLETED,
                    "prompt_tokens": metadata.get("prompt_tokens", 0),
                    "completion_tokens": metadata.get("completion_tokens", 0),
                }
            except (json.JSONDecodeError, OSError):
                continue

        listed = list(sessions.values())
        listed.sort(
            key=lambda s: (
                s.get("updated_at") or s.get("created_at", ""),
                s.get("created_at", ""),
            ),
            reverse=True,
        )
        return listed

    @classmethod
    def repair_states(cls, project_dir: Path) -> list[str]:
        """Mark active sessions owned by dead processes as interrupted.

        Returns the ids that were repaired. Called once at startup before
        offering to resume an interrupted session.
        """
        sdir = sessions_dir(project_dir)
        repaired: list[str] = []
        for meta_path in sdir.glob("*/meta.json"):
            try:
                # Per-session lock: repairing races a live owner's append
                # or a concurrent resume otherwise. Re-read under the lock
                # before deciding.
                with file_lock(meta_path.parent / "lock"):
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("state") != STATE_ACTIVE:
                        continue
                    # pid_start defeats PID reuse: a recycled PID would
                    # otherwise keep a truly interrupted session "active"
                    # forever.
                    if _pid_alive(meta.get("pid"), meta.get("pid_start")):
                        continue
                    meta["state"] = STATE_INTERRUPTED
                    # updated_at stays untouched: it reflects the session's
                    # real last activity, and bumping it here would let an
                    # old abandoned session hijack the "newest interrupted"
                    # startup offer from genuinely recent work.
                    meta["interrupted_at"] = _now()
                    atomic_write_json(meta_path, meta)
                    repaired.append(meta.get("id", meta_path.parent.name))
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "could not repair session state for %s",
                    meta_path.parent.name,
                    exc_info=True,
                )
        return repaired
