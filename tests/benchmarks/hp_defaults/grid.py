"""The hyperparameter grid under study, and how it gets split across jobs.

Axes are learning rate × epochs — the two the product's sweep already varies,
so the study's answer is directly installable in ``lqh/train/defaults.py``.

The learning-rate range is deliberately **wider than the product grid**
(`{1e-4, 3e-4, 1e-3}`), and brackets the shipped LoRA default of 2e-4 on both
sides: 2e-5 confirms the full-fine-tuning-style rate the product used to ship
(and which a customer's flat runs argued against) is genuinely worse, and 1e-3
tests whether the correction went far enough. A study that only searched inside
the product grid could not discover that the whole grid sits in the wrong place
— which is exactly what happened when the default was 2e-5.
"""

from __future__ import annotations

from dataclasses import dataclass

LEARNING_RATES: tuple[float, ...] = (2e-5, 1e-4, 2e-4, 5e-4, 1e-3)
EPOCHS: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class GridPoint:
    """One hyperparameter configuration, as the sweep's grid_override wants it."""

    learning_rate: float
    num_epochs: int
    seed: int | None = None

    @property
    def id(self) -> str:
        base = f"lr{self.learning_rate:g}_e{self.num_epochs}"
        return f"{base}_s{self.seed}" if self.seed is not None else base

    def to_override(self) -> dict:
        training: dict = {
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
        }
        if self.seed is not None:
            # sft.py reads training.seed and training.data_seed; setting both
            # via the seed key keeps shuffling and init varying together, which
            # is the run-to-run noise the replicates are meant to measure.
            training["seed"] = self.seed
            training["data_seed"] = self.seed
        return {"id": self.id, "overrides": {"training": training}}


def study_grid(
    *,
    learning_rates: tuple[float, ...] = LEARNING_RATES,
    epochs: tuple[int, ...] = EPOCHS,
) -> list[GridPoint]:
    return [
        GridPoint(learning_rate=lr, num_epochs=e)
        for lr in learning_rates
        for e in epochs
    ]


def replicate_grid(points: list[GridPoint], seeds: tuple[int, ...]) -> list[GridPoint]:
    """Same configs, different training seeds — the noise-floor measurement.

    Without this the study cannot tell a real per-dimension difference from
    run-to-run variance, and would happily recommend splitting the defaults on
    a gap that is pure noise.
    """
    return [
        GridPoint(learning_rate=p.learning_rate, num_epochs=p.num_epochs, seed=s)
        for p in points
        for s in seeds
    ]


def parse_points(spec: str) -> list[GridPoint]:
    """Parse ``lr=2e-5:e2,lr=1e-4:e1`` into grid points (for smoke runs)."""
    points: list[GridPoint] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lr = epochs = None
        for part in chunk.split(":"):
            part = part.strip()
            if part.startswith("lr="):
                lr = float(part[3:])
            elif part.startswith("e"):
                epochs = int(part[1:])
        if lr is None or epochs is None:
            raise SystemExit(
                f"bad grid point {chunk!r}; expected the form 'lr=2e-5:e2'"
            )
        points.append(GridPoint(learning_rate=lr, num_epochs=epochs))
    return points


def chunk_points(points: list[GridPoint], size: int) -> list[list[GridPoint]]:
    """Split a cell's grid across several jobs.

    A sweep trains its configs sequentially in one job, and the cloud runner
    caps ``train_sft_sweep`` at 720 minutes. A 15-config cell on the largest
    dataset can exceed that, and a timeout loses every config in the job — not
    just the one that ran over. Chunking bounds the blast radius and lets cells
    fan out wider. Rows from all chunks are merged at analysis time.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [points[i:i + size] for i in range(0, len(points), size)]
