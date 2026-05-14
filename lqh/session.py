from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


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
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def create(cls, project_dir: Path) -> Session:
        """Create a new session with a fresh UUID and current timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            project_dir=project_dir,
            messages=[],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def first_user_message(self) -> str | None:
        """Return the content of the first user message, or None."""
        for msg in self.messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        return None

    def save(self) -> None:
        """Write messages as JSONL to .lqh/conversations/{self.id}.jsonl.

        The first line is a metadata JSON header (id, created_at, preview).
        Does nothing if there are no user messages.
        """
        if self.first_user_message() is None:
            return

        preview = (self.first_user_message() or "")[:80].replace("\n", " ").strip()
        metadata = {
            "__metadata__": True,
            "id": self.id,
            "created_at": self.created_at,
            "preview": preview,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

        path = sessions_dir(self.project_dir) / f"{self.id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")
            for message in self.messages:
                f.write(json.dumps(message) + "\n")

    @classmethod
    def load(cls, project_dir: Path, session_id: str) -> Session:
        """Load a session from its JSONL file. First line is metadata, rest are messages."""
        path = sessions_dir(project_dir) / f"{session_id}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f"Session file is empty: {path}")

        metadata = json.loads(lines[0])
        messages: list[dict] = [json.loads(line) for line in lines[1:] if line.strip()]

        return cls(
            id=metadata.get("id", session_id),
            project_dir=project_dir,
            messages=messages,
            created_at=metadata.get("created_at", ""),
            prompt_tokens=metadata.get("prompt_tokens", 0),
            completion_tokens=metadata.get("completion_tokens", 0),
        )

    @classmethod
    def list_sessions(cls, project_dir: Path) -> list[dict]:
        """Return a list of session summaries sorted by created_at descending.

        Each dict contains id, created_at, and preview (first user message trimmed to 80 chars).
        """
        sdir = sessions_dir(project_dir)
        sessions: list[dict] = []

        for jsonl_path in sdir.glob("*.jsonl"):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                if not first_line.strip():
                    continue
                metadata = json.loads(first_line)
                if not metadata.get("__metadata__"):
                    continue
                sessions.append({
                    "id": metadata.get("id", jsonl_path.stem),
                    "created_at": metadata.get("created_at", ""),
                    "preview": metadata.get("preview", ""),
                })
            except (json.JSONDecodeError, OSError):
                continue

        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions

    def add_message(self, message: dict) -> None:
        """Append a message to the conversation and persist to disk.

        Saving on every append means a kill mid-loop (e.g. while the agent is
        in a guided ask_user dialogue) doesn't lose progress. JSONL is rewritten
        in full each time, but at typical session sizes the I/O is negligible.
        """
        self.messages.append(message)
        self.save()
