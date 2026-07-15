"""Lightweight readers for DPO iteration quality artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def read_held_out_mean(iter_dir: Path) -> float | None:
    """Read a held-out judge mean from any supported iteration artifact."""
    for path in (
        iter_dir / "held_out_eval" / "summary.json",
        iter_dir / "held_out_eval.json",
        iter_dir / "eval_result.json",
    ):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        candidates = [payload.get("mean")]
        scores = payload.get("scores")
        if isinstance(scores, dict):
            candidates.append(scores.get("mean"))
        summary = payload.get("summary")
        if isinstance(summary, dict) and isinstance(summary.get("scores"), dict):
            candidates.append(summary["scores"].get("mean"))
        for value in candidates:
            if isinstance(value, (int, float)):
                return float(value)
    return None


def find_best_held_out_iter(
    iterations_dir: Path,
) -> tuple[int | None, float | None]:
    """Return the earliest iteration with the highest held-out judge mean."""
    if not iterations_dir.exists():
        return None, None
    best_iter: int | None = None
    best_mean: float | None = None
    for directory in sorted(iterations_dir.iterdir()):
        if not directory.is_dir() or not directory.name.startswith("iter_"):
            continue
        try:
            iteration = int(directory.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        mean = read_held_out_mean(directory)
        if mean is not None and (best_mean is None or mean > best_mean):
            best_iter = iteration
            best_mean = mean
    return best_iter, best_mean
