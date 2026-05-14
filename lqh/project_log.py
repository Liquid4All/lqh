"""Project activity log for tracking major events in .lqh/project.log (JSONL)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        log_dir = project_dir / ".lqh"
        log_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "desc": desc,
            **kwargs,
        }
        with (log_dir / "project.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


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

        suffix = f"  ({', '.join(suffix_parts)})" if suffix_parts else ""
        line = f"[{ts}] {event} — {desc}{suffix}"
        # Cap line length
        if len(line) > 160:
            line = line[:157] + "..."
        lines.append(line)
    return "\n".join(lines)
