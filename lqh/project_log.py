"""Project activity log for tracking major events in .lqh/project.log (JSONL).

The log is a best-effort recovery hint, not a ledger: writes never raise
(workflow execution always wins over logging), but failures are logged
instead of silently swallowed, appends are serialized across processes,
and each entry is stamped with the conversation session that produced it
when the TUI has registered one via ``set_log_session``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Conversation session responsible for subsequent append_event calls. A
# plain module global (not a ContextVar): long-lived background tasks
# (the job watcher) are created once and must observe later /clear and
# /resume switches, which a ContextVar snapshot would hide from them.
# One process has exactly one active conversation, so a global is
# correct.
_session_id: str | None = None


def set_log_session(session_id: str | None) -> None:
    """Register the active conversation session for event attribution."""
    global _session_id
    _session_id = session_id


def file_hash_prefix(path: Path, n: int = 6) -> str:
    """Return first *n* hex chars of the SHA-256 of *path*'s contents."""
    try:
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:n]
    except Exception:
        return "?" * n


def is_spec_file(rel_path: str) -> bool:
    """True if *rel_path* is the main spec or lives under other_specs/."""
    return rel_path == "SPEC.md" or rel_path.startswith("other_specs/")


def append_event(project_dir: Path, event: str, desc: str, **kwargs: Any) -> None:
    """Append one JSONL line to .lqh/project.log.  Never raises."""
    try:
        from lqh.fsio import append_line_durable, file_lock

        log_dir = project_dir / ".lqh"
        log_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "desc": desc,
            **kwargs,
        }
        if _session_id:
            entry.setdefault("session_id", _session_id)
        with file_lock(log_dir / "project.log.lock"):
            append_line_durable(
                log_dir / "project.log",
                json.dumps(entry, ensure_ascii=False),
            )
    except Exception:
        # Best-effort by contract, but observable: a workflow must never
        # fail because its log line couldn't be written.
        logger.warning("project.log append failed", exc_info=True)


def read_recent(project_dir: Path, n: int = 50) -> list[dict[str, Any]]:
    """Return the last *n* entries from the project log."""
    log_path = project_dir / ".lqh" / "project.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    entries: list[dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def format_log_for_context(entries: list[dict[str, Any]]) -> str:
    """Format log entries as compact plain text for the agent's system context."""
    lines: list[str] = []
    for e in entries:
        ts_raw = e.get("ts", "")
        # Shorten ISO timestamp to YYYY-MM-DD HH:MMZ
        try:
            dt = datetime.fromisoformat(ts_raw)
            ts = dt.strftime("%Y-%m-%d %H:%MZ")
        except Exception:
            ts = ts_raw[:16] if ts_raw else "?"

        event = e.get("event", "?")
        desc = e.get("desc", "")

        # Append script hash for data_gen events
        suffix_parts: list[str] = []
        if "script_path" in e:
            s = e["script_path"]
            if "script_hash" in e:
                s += f"@{e['script_hash']}"
            suffix_parts.append(f"script={s}")
        if e.get("session_id"):
            suffix_parts.append(f"session={str(e['session_id'])[:8]}")

        suffix = f"  ({', '.join(suffix_parts)})" if suffix_parts else ""
        line = f"[{ts}] {event} — {desc}{suffix}"
        # Cap line length
        if len(line) > 160:
            line = line[:157] + "..."
        lines.append(line)
    return "\n".join(lines)
