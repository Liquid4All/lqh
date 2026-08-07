"""Continuation helpers for preemptible cloud training jobs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def continuation_index() -> int:
    """Return the backend-provided worker continuation index."""
    raw = os.environ.get("LQH_CONTINUATION", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def is_continuation() -> bool:
    return continuation_index() > 0


def checkpoint_candidates(checkpoint_root: Path) -> list[Path]:
    """Return HF Trainer checkpoint dirs newest-first."""
    if not checkpoint_root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        m = _CKPT_RE.match(child.name)
        if not m:
            continue
        found.append((int(m.group(1)), child))
    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in found]


def train_with_checkpoint_fallback(
    trainer: Any,
    checkpoint_root: Path,
    *,
    label: str,
    continuation: bool | None = None,
) -> Any:
    """Resume from the newest valid checkpoint on continuation.

    If a checkpoint fails to load/train, try the next older checkpoint.
    If all candidates fail, restart the current training unit from scratch.
    Normal non-continuation runs keep the existing behavior.
    """
    if continuation is None:
        continuation = is_continuation()
    if not continuation:
        return trainer.train()

    candidates = checkpoint_candidates(checkpoint_root)
    for ckpt in candidates:
        try:
            # ckpt.name, not the path: job stdout is user-visible and the
            # sandbox mount layout adds nothing. trainer gets the real path.
            print(f"{label}: resuming from {ckpt.name}", flush=True)
            return trainer.train(resume_from_checkpoint=str(ckpt))
        except Exception as exc:  # noqa: BLE001
            print(
                f"{label}: resume from {ckpt.name} failed ({exc}); "
                f"trying older checkpoint",
                flush=True,
            )

    if candidates:
        print(
            f"{label}: all checkpoint resume attempts failed; restarting from scratch",
            flush=True,
        )
    else:
        print(f"{label}: no checkpoints found; starting from scratch", flush=True)
    return trainer.train()
