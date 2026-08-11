"""Out-of-space detection for the client's own file writes (feedback #50).

A full disk breaks LQH's error reporting instead of being reported by it:
the mirror, the status files and the log file fail at once.
"""

from __future__ import annotations

import errno
import shutil
from pathlib import Path

#: Writes start failing before the disk is literally at zero —
#: ``fsio.atomic_write_json`` needs room for a second copy of the file.
LOW_DISK_MB = 64

# Texts as well as errnos: rsync reports a full disk by exiting non-zero.
_ERRNOS = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}
_TEXTS = ("no space left on device", "disk quota exceeded")

# A quota refuses writes while the filesystem reports gigabytes free, so
# the probe alone would detect the failure and then say nothing about it.
# Never cleared, and read ONLY by the one-shot advisory — suppressing any
# other diagnosis with it would need "is this latch stale" semantics.
_saw_enospc = False


def free_mb(path: Path | str) -> int | None:
    """Free megabytes on ``path``'s filesystem, or None if unknown."""
    try:
        return shutil.disk_usage(path).free // (1024 * 1024)
    except (OSError, ValueError):
        return None


def note_enospc(exc: BaseException | None) -> bool:
    """Latch an out-of-space failure. True if ``exc`` was one."""
    global _saw_enospc
    if not is_enospc(exc):
        return False
    _saw_enospc = True
    return True


def failing(path: Path | str) -> bool:
    """Has a write failed for want of space, or is one about to?"""
    return _saw_enospc or is_low(path)


def is_low(path: Path | str) -> bool:
    """Is the filesystem nearly full? Unknown reads as "no"."""
    free = free_mb(path)
    return free is not None and free < LOW_DISK_MB


def is_enospc(exc: BaseException | None) -> bool:
    """True if ``exc`` — or anything it was raised from — is out-of-space.

    Callers catch a wrapper, so the whole cause/context graph is walked.
    """
    seen: set[int] = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, OSError) and cur.errno in _ERRNOS:
            return True
        if any(t in str(cur).lower() for t in _TEXTS):
            return True
        stack += [cur.__cause__, cur.__context__]
    return False
