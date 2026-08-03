"""Lightweight readers for DPO iteration quality artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path


def derive_effective_batch(
    *,
    pair_count: int,
    epochs: int,
    target_optimizer_steps: int,
    minimum: int = 8,
    maximum: int = 128,
) -> int:
    """Choose a power-of-two batch that preserves a minimum update budget."""
    pair_count = max(1, int(pair_count))
    epochs = max(1, int(epochs))
    target_optimizer_steps = max(1, int(target_optimizer_steps))
    minimum = max(1, int(minimum))
    maximum = max(minimum, int(maximum))
    upper = max(
        minimum,
        min(maximum, (pair_count * epochs) // target_optimizer_steps),
    )
    power = 1 << max(0, int(math.floor(math.log2(upper))))
    return max(minimum, min(maximum, power))


def has_no_train_signal(
    *,
    optimizer_steps: int,
    train_loss: float | None,
    chosen_ce_delta_ref: float | None,
) -> bool:
    """Return whether a DPO iteration is effectively an optimization no-op."""
    if optimizer_steps < 10:
        return True
    return bool(
        train_loss is not None
        and train_loss > 0.68
        and chosen_ce_delta_ref is not None
        and abs(chosen_ce_delta_ref) < 0.005
    )


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


def held_out_stop_reason(
    iterations_dir: Path,
    *,
    regression_delta: float = 0.5,
    plateau_patience: int = 2,
    min_improvement: float = 0.05,
) -> dict[str, float | int | str] | None:
    """Return a quality-stop payload for the latest held-out checkpoint.

    A large regression stops immediately. Otherwise, ``plateau_patience``
    consecutive checkpoints without a meaningful improvement stop the
    on-policy loop, whose best-checkpoint restore then keeps the peak model.
    """
    history: list[tuple[int, float]] = []
    if not iterations_dir.exists():
        return None
    for directory in sorted(iterations_dir.iterdir()):
        if not directory.is_dir() or not directory.name.startswith("iter_"):
            continue
        try:
            iteration = int(directory.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        mean = read_held_out_mean(directory)
        if mean is not None:
            history.append((iteration, mean))
    if len(history) < 2:
        return None

    current_iteration, current_mean = history[-1]
    best_prior_iteration, best_prior_mean = max(
        history[:-1], key=lambda item: item[1]
    )
    if (
        regression_delta > 0
        and current_mean < best_prior_mean - regression_delta
    ):
        return {
            "source": "held_out_regression",
            "iteration": current_iteration,
            "held_out_mean": current_mean,
            "best_iteration": best_prior_iteration,
            "best_held_out_mean": best_prior_mean,
            "regression": current_mean - best_prior_mean,
            "threshold": -regression_delta,
            "reason": (
                f"held-out judge mean regressed {current_mean - best_prior_mean:+.3f} "
                f"from the best checkpoint (threshold {-regression_delta:+.3f})"
            ),
        }

    if plateau_patience <= 0:
        return None
    significant_best_iteration, significant_best_mean = history[0]
    for iteration, mean in history[1:]:
        if mean >= significant_best_mean + min_improvement:
            significant_best_iteration = iteration
            significant_best_mean = mean
    if current_iteration - significant_best_iteration >= plateau_patience:
        return {
            "source": "held_out_plateau",
            "iteration": current_iteration,
            "held_out_mean": current_mean,
            "best_iteration": significant_best_iteration,
            "best_held_out_mean": significant_best_mean,
            "patience": plateau_patience,
            "min_improvement": min_improvement,
            "reason": (
                f"held-out judge mean did not improve by {min_improvement:.3f} "
                f"for {plateau_patience} checkpoints"
            ),
        }
    return None
