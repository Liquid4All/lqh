"""Durable filesystem primitives shared by persistency code.

Extracted from the battle-tested patterns in ``lqh.config`` (atomic
replace + fsync, cross-process file lock) so session storage, snapshot
caching, and the project log can share one implementation. ``lqh.config``
itself intentionally keeps its private copies — do not re-wire it.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows path
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX path
    msvcrt = None  # type: ignore[assignment]

# One in-process lock per lock-file path: fcntl locks are per-process, so
# threads in the same process must serialize among themselves first.
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.RLock:
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    """Serialize read-modify-write operations across CLI processes.

    Takes an in-process thread lock first, then an exclusive advisory lock
    on ``lock_path`` (created if missing). POSIX uses ``fcntl.flock``;
    Windows falls back to ``msvcrt.locking``.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock_for(lock_path):
        with lock_path.open("a+b") as handle:
            try:
                lock_path.chmod(0o600)
            except OSError:
                pass
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows path
                handle.seek(0)
                if handle.read(1) == b"":
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows path
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def atomic_write_json(path: Path, obj: Any, *, mode: int | None = 0o600) -> None:
    """Write ``obj`` as JSON to ``path`` via temp file + atomic replace.

    The temp file is fsync'd before the rename and the parent directory is
    fsync'd after, so a crash leaves either the old or the new complete
    document — never a torn write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            json.dump(obj, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            try:
                tmp.chmod(mode)
            except OSError:
                # Non-POSIX/sandboxed filesystems may not support chmod;
                # the write itself must still succeed.
                pass
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def append_line_durable(path: Path, line: str) -> None:
    """Append one line to ``path`` and fsync before returning.

    ``line`` must not contain a newline; one is added. On return the line
    is durable — a subsequent crash cannot lose it (a crash *during* the
    write can at worst leave a torn final line, which readers must
    tolerate/quarantine).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
