"""Statistics and report helpers for the DPO value benchmark."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedDelta:
    n: int
    mean: float
    ci_low: float
    ci_high: float


def paired_bootstrap(
    treatment: dict[int, float],
    control: dict[int, float],
    *,
    samples: int = 10_000,
    seed: int = 20260715,
) -> PairedDelta:
    """Bootstrap the mean treatment-control delta on shared sample IDs."""
    shared = sorted(treatment.keys() & control.keys())
    if not shared:
        raise ValueError("paired comparison has no shared scored samples")
    deltas = [treatment[index] - control[index] for index in shared]
    mean = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return PairedDelta(1, mean, mean, mean)

    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        boot.append(
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    boot.sort()
    low_index = max(0, int(samples * 0.025))
    high_index = min(samples - 1, int(samples * 0.975))
    return PairedDelta(len(deltas), mean, boot[low_index], boot[high_index])
