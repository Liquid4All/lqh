"""Turning a grid of measurements into a defaults recommendation.

Every function here is pure — the study's conclusion is exactly the kind of
thing that can be quietly wrong, so the logic that produces it is separated
from the code that runs GPUs and unit-tested against synthetic tables.

The central quantity is **regret**: how much judge score a config gives up
against the best config in the same cell.

    regret(config, cell) = oracle(cell) - judge_mean(config, cell)

A default is good when its *average* regret across contexts is small. That is a
different question from "which config won most often" — a config that wins six
cells by 0.05 and loses two by 2.0 is a bad default, and mean regret says so
while a win count does not. Both are reported; regret decides.

Two guardrails run throughout:

- **Balanced panels.** Configs are only ever compared on cells where *all* of
  them were measured, and each cell's oracle is computed within that same set.
  Otherwise a config that only ran on the easy cells looks better than it is.
- **A measured noise floor.** Seed replicates say how much a judge score moves
  for no reason at all. A per-dimension difference smaller than that is not a
  finding, and `conditional_defaults` refuses to call it one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Dimensions the defaults could plausibly need to vary along, in the order the
# report presents them.
DIMENSIONS: tuple[str, ...] = ("model_kind", "param_count", "train_size", "task")


@dataclass(frozen=True)
class Observation:
    """One (cell, config) measurement."""

    cell_id: str
    config_id: str
    dimensions: dict[str, Any]
    judge_mean: float | None = None
    eval_loss: float | None = None
    num_epochs: int | None = None
    best_epoch: float | None = None
    elapsed_s: float | None = None
    judge_num_scored: int = 0
    judge_std: float | None = None
    seed: int | None = None
    note: str | None = None

    @property
    def scored(self) -> bool:
        return isinstance(self.judge_mean, (int, float))


@dataclass
class ConfigStats:
    config_id: str
    mean_regret: float
    max_regret: float
    mean_rank: float
    wins: int
    n_cells: int
    mean_elapsed_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "mean_regret": self.mean_regret,
            "max_regret": self.max_regret,
            "mean_rank": self.mean_rank,
            "wins": self.wins,
            "n_cells": self.n_cells,
            "mean_elapsed_s": self.mean_elapsed_s,
        }


@dataclass
class Recommendation:
    config_id: str | None
    mean_regret: float | None
    ci_low: float | None
    ci_high: float | None
    panel_cells: list[str]
    compared_configs: list[str]
    ranking: list[ConfigStats] = field(default_factory=list)
    excluded_configs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "mean_regret": self.mean_regret,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_panel_cells": len(self.panel_cells),
            "panel_cells": self.panel_cells,
            "compared_configs": self.compared_configs,
            "excluded_configs": self.excluded_configs,
            "ranking": [c.as_dict() for c in self.ranking],
        }


@dataclass
class LevelFinding:
    dimension: str
    level: Any
    best_config: str
    n_cells: int
    global_regret_here: float
    best_regret_here: float
    gain: float
    ci_low: float
    ci_high: float
    beats_noise: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "level": self.level,
            "best_config": self.best_config,
            "n_cells": self.n_cells,
            "global_regret_here": self.global_regret_here,
            "best_regret_here": self.best_regret_here,
            "gain": self.gain,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "beats_noise": self.beats_noise,
        }


@dataclass
class DimensionFinding:
    dimension: str
    split_recommended: bool
    levels: list[LevelFinding] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "split_recommended": self.split_recommended,
            "levels": [level.as_dict() for level in self.levels],
        }


@dataclass
class NoiseFloor:
    """How much a judge score moves when nothing about the config changed."""

    seed_range: float | None
    seed_std: float | None
    n_groups: int
    median_judge_sem: float | None

    @property
    def basis(self) -> str:
        """What the floor is actually built from — not all bases are equal.

        - ``seed_replicates``: the real thing. Re-running an identical config
          under a different seed varies both the training run and the judging,
          which is exactly the noise a claimed improvement has to beat.
        - ``judge_sem``: a **lower bound only**. It captures the judge's
          sampling error but not training variance, which is usually the larger
          term — so a gain that clears this floor may still be noise.
        - ``none``: nothing to go on.
        """
        if self.seed_range is not None:
            return "seed_replicates"
        if self.median_judge_sem is not None:
            return "judge_sem"
        return "none"

    @property
    def value(self) -> float:
        """The threshold a claimed improvement must clear to count."""
        if self.seed_range is not None:
            return self.seed_range
        if self.median_judge_sem is not None:
            return 2 * self.median_judge_sem
        return 0.0

    @property
    def measured(self) -> bool:
        """Whether the floor rests on replicates — the only basis that
        justifies acting on a per-dimension split."""
        return self.basis == "seed_replicates"

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_range": self.seed_range,
            "seed_std": self.seed_std,
            "n_replicate_groups": self.n_groups,
            "median_judge_sem": self.median_judge_sem,
            "value": self.value,
            "basis": self.basis,
            "measured": self.measured,
        }


@dataclass
class ProxyReport:
    """Does `eval_loss` still rank configs the way the judge does?

    The shipped sweep selects winners on `eval_loss` alone. This study happens
    to judge every config, so it can check that assumption for free — and say
    whether it holds at every model size and dataset scale, or only some.
    """

    n_cells: int
    mean_spearman: float | None
    median_spearman: float | None
    top1_hit_rate: float | None
    per_cell: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "mean_spearman": self.mean_spearman,
            "median_spearman": self.median_spearman,
            "top1_hit_rate": self.top1_hit_rate,
            "per_cell": self.per_cell,
        }


@dataclass
class EpochReport:
    """Is the `num_epochs` axis doing work checkpoint selection already did?

    ``sft.py`` trains with ``load_best_model_at_end`` on ``eval_loss``, so a
    3-epoch run can save an epoch-1 model. Where that happens, epochs is a
    ceiling rather than a tuned quantity — and the product grid could drop the
    axis entirely.
    """

    n_runs: int
    n_stopped_early: int
    early_stop_rate: float | None
    by_num_epochs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "n_stopped_early": self.n_stopped_early,
            "early_stop_rate": self.early_stop_rate,
            "by_num_epochs": self.by_num_epochs,
        }


# ---------------------------------------------------------------------------
# Panels and regret
# ---------------------------------------------------------------------------


def scored(observations: Iterable[Observation]) -> list[Observation]:
    """Drop measurements with no judge score.

    A config that crashed or failed to score has no result — not a zero. Left
    in, it would win every argmin of regret it appeared in.
    """
    return [o for o in observations if o.scored]


def config_ids(observations: Iterable[Observation]) -> list[str]:
    return sorted({o.config_id for o in observations})


def balanced_panel(
    observations: Sequence[Observation], configs: Sequence[str],
) -> list[str]:
    """Cells where every config in *configs* has a judge score."""
    if not configs:
        return []
    by_cell: dict[str, set[str]] = {}
    for o in scored(observations):
        by_cell.setdefault(o.cell_id, set()).add(o.config_id)
    wanted = set(configs)
    return sorted(cell for cell, seen in by_cell.items() if wanted <= seen)


def widest_panel(
    observations: Sequence[Observation], *, min_cells: int = 1,
) -> tuple[list[str], list[str]]:
    """Pick the config set worth comparing, and the cells to compare it on.

    A two-stage study measures many configs on a few cells and a few finalists
    on many cells, so there is a genuine trade-off between comparing more
    configs and comparing them over more contexts. Rather than pick for the
    caller, this maximises coverage — configs × cells — which keeps the full
    grid when it was run everywhere and narrows to the finalists when it was
    not.

    Returns ``(configs, cells)``; configs are sorted by descending cell count so
    dropping the sparsest first is what generates the candidate sets.
    """
    obs = scored(observations)
    if not obs:
        return [], []

    coverage: dict[str, set[str]] = {}
    for o in obs:
        coverage.setdefault(o.config_id, set()).add(o.cell_id)
    ordered = sorted(coverage, key=lambda c: (-len(coverage[c]), c))

    best: tuple[int, list[str], list[str]] = (0, [], [])
    for cutoff in range(1, len(ordered) + 1):
        candidate = sorted(ordered[:cutoff])
        panel = balanced_panel(obs, candidate)
        if len(panel) < min_cells:
            continue
        score = len(candidate) * len(panel)
        if score > best[0]:
            best = (score, candidate, panel)
    return best[1], best[2]


def cell_oracles(
    observations: Sequence[Observation],
    *,
    configs: Sequence[str] | None = None,
    cells: Sequence[str] | None = None,
) -> dict[str, float]:
    """Best judge score achieved in each cell, within the compared set.

    Scoped to *configs* on purpose: an oracle set by a config that only ran in
    the screening stage would inflate every finalist's regret and make the
    recommendation look worse than it is.
    """
    config_set = set(configs) if configs is not None else None
    cell_set = set(cells) if cells is not None else None
    oracles: dict[str, float] = {}
    for o in scored(observations):
        if config_set is not None and o.config_id not in config_set:
            continue
        if cell_set is not None and o.cell_id not in cell_set:
            continue
        current = oracles.get(o.cell_id)
        if current is None or o.judge_mean > current:
            oracles[o.cell_id] = float(o.judge_mean)
    return oracles


def regrets(
    observations: Sequence[Observation],
    *,
    configs: Sequence[str],
    cells: Sequence[str],
) -> dict[str, dict[str, float]]:
    """``{config_id: {cell_id: regret}}`` over the given panel."""
    oracles = cell_oracles(observations, configs=configs, cells=cells)
    config_set, cell_set = set(configs), set(cells)
    out: dict[str, dict[str, float]] = {c: {} for c in configs}
    for o in scored(observations):
        if o.config_id not in config_set or o.cell_id not in cell_set:
            continue
        oracle = oracles.get(o.cell_id)
        if oracle is None:
            continue
        out[o.config_id][o.cell_id] = oracle - float(o.judge_mean)
    return out


def rank_configs(
    observations: Sequence[Observation],
    *,
    configs: Sequence[str],
    cells: Sequence[str],
) -> list[ConfigStats]:
    """Per-config summary over the panel, best (lowest mean regret) first."""
    table = regrets(observations, configs=configs, cells=cells)
    elapsed: dict[str, list[float]] = {}
    for o in scored(observations):
        if o.config_id in table and o.cell_id in set(cells):
            if isinstance(o.elapsed_s, (int, float)):
                elapsed.setdefault(o.config_id, []).append(float(o.elapsed_s))

    # Within-cell rank, 1 = best. Ties share the mean of the tied positions so
    # a cell where everything scored identically cannot favour one config.
    ranks: dict[str, list[float]] = {c: [] for c in configs}
    wins: dict[str, int] = {c: 0 for c in configs}
    for cell in cells:
        scores = [
            (c, table[c][cell]) for c in configs if cell in table.get(c, {})
        ]
        if not scores:
            continue
        for config, rank in _average_ranks(scores).items():
            ranks[config].append(rank)
        best_regret = min(v for _, v in scores)
        for config, regret in scores:
            if math.isclose(regret, best_regret, abs_tol=1e-12):
                wins[config] += 1

    stats: list[ConfigStats] = []
    for config in configs:
        values = list(table.get(config, {}).values())
        if not values:
            continue
        times = elapsed.get(config, [])
        stats.append(ConfigStats(
            config_id=config,
            mean_regret=sum(values) / len(values),
            max_regret=max(values),
            mean_rank=(sum(ranks[config]) / len(ranks[config])) if ranks[config] else float("nan"),
            wins=wins[config],
            n_cells=len(values),
            mean_elapsed_s=(sum(times) / len(times)) if times else None,
        ))
    # Ties on regret go to the faster config — a default that is as good and
    # cheaper is strictly better.
    stats.sort(key=lambda s: (round(s.mean_regret, 6), s.mean_elapsed_s or 0.0, s.config_id))
    return stats


def _average_ranks(scored_pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(scored_pairs, key=lambda kv: kv[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and math.isclose(
            ordered[j + 1][1], ordered[i][1], abs_tol=1e-12
        ):
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = shared
        i = j + 1
    return ranks


def recommend(
    observations: Sequence[Observation],
    *,
    configs: Sequence[str] | None = None,
    cells: Sequence[str] | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 20260805,
) -> Recommendation:
    """The single global default: lowest mean regret across the panel."""
    obs = scored(observations)
    if configs is None or cells is None:
        auto_configs, auto_cells = widest_panel(obs)
        configs = configs if configs is not None else auto_configs
        cells = cells if cells is not None else auto_cells
    cells = list(cells)
    configs = list(configs)

    excluded = sorted(set(config_ids(obs)) - set(configs))
    if not configs or not cells:
        return Recommendation(
            config_id=None, mean_regret=None, ci_low=None, ci_high=None,
            panel_cells=cells, compared_configs=configs, excluded_configs=excluded,
        )

    ranking = rank_configs(obs, configs=configs, cells=cells)
    if not ranking:
        return Recommendation(
            config_id=None, mean_regret=None, ci_low=None, ci_high=None,
            panel_cells=cells, compared_configs=configs, excluded_configs=excluded,
        )

    winner = ranking[0]
    table = regrets(obs, configs=configs, cells=cells)
    values = [table[winner.config_id][c] for c in cells if c in table[winner.config_id]]
    low, high = bootstrap_ci(values, samples=bootstrap_samples, seed=seed)
    return Recommendation(
        config_id=winner.config_id,
        mean_regret=winner.mean_regret,
        ci_low=low,
        ci_high=high,
        panel_cells=cells,
        compared_configs=configs,
        ranking=ranking,
        excluded_configs=excluded,
    )


# ---------------------------------------------------------------------------
# Does any dimension deserve its own default?
# ---------------------------------------------------------------------------


def conditional_defaults(
    observations: Sequence[Observation],
    *,
    global_config: str,
    configs: Sequence[str],
    cells: Sequence[str],
    noise: NoiseFloor,
    dimensions: Sequence[str] = DIMENSIONS,
    min_cells_per_level: int = 2,
    bootstrap_samples: int = 2000,
    seed: int = 20260805,
) -> list[DimensionFinding]:
    """For each dimension, what a per-level default would buy over the global one.

    The gain is only called real when it clears BOTH the measured noise floor
    and a bootstrap CI that excludes zero. Splitting the defaults on a
    difference smaller than run-to-run variance would be worse than useless:
    it adds a rule nobody can reproduce.
    """
    obs = scored(observations)
    cell_dims = {o.cell_id: o.dimensions for o in obs}
    table = regrets(obs, configs=configs, cells=cells)

    findings: list[DimensionFinding] = []
    for dim in dimensions:
        levels: list[LevelFinding] = []
        by_level: dict[Any, list[str]] = {}
        for cell in cells:
            level = cell_dims.get(cell, {}).get(dim)
            if level is not None:
                by_level.setdefault(level, []).append(cell)

        for level, level_cells in sorted(by_level.items(), key=lambda kv: str(kv[0])):
            if len(level_cells) < min_cells_per_level:
                continue
            local = rank_configs(obs, configs=configs, cells=level_cells)
            if not local:
                continue
            best = local[0]
            global_here = [
                table[global_config][c]
                for c in level_cells if c in table.get(global_config, {})
            ]
            best_here = [
                table[best.config_id][c]
                for c in level_cells if c in table.get(best.config_id, {})
            ]
            if not global_here or not best_here:
                continue
            paired = [
                table[global_config][c] - table[best.config_id][c]
                for c in level_cells
                if c in table.get(global_config, {})
                and c in table.get(best.config_id, {})
            ]
            gain = sum(paired) / len(paired)
            low, high = bootstrap_ci(paired, samples=bootstrap_samples, seed=seed)
            levels.append(LevelFinding(
                dimension=dim,
                level=level,
                best_config=best.config_id,
                n_cells=len(level_cells),
                global_regret_here=sum(global_here) / len(global_here),
                best_regret_here=sum(best_here) / len(best_here),
                gain=gain,
                ci_low=low,
                ci_high=high,
                beats_noise=(gain > noise.value and low > 0.0),
            ))

        findings.append(DimensionFinding(
            dimension=dim,
            split_recommended=any(level.beats_noise for level in levels),
            levels=levels,
        ))
    return findings


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------


def noise_floor(observations: Sequence[Observation]) -> NoiseFloor:
    """Measure run-to-run variance from seed replicates and judge SEM."""
    obs = scored(observations)

    groups: dict[tuple[str, str], list[float]] = {}
    for o in obs:
        if o.seed is None:
            continue
        groups.setdefault((o.cell_id, o.config_id), []).append(float(o.judge_mean))
    replicated = [v for v in groups.values() if len(v) >= 2]

    seed_range = seed_std = None
    if replicated:
        seed_range = sum(max(v) - min(v) for v in replicated) / len(replicated)
        seed_std = _pooled_std(replicated)

    sems = [
        o.judge_std / math.sqrt(o.judge_num_scored)
        for o in obs
        if isinstance(o.judge_std, (int, float)) and o.judge_num_scored > 1
    ]
    return NoiseFloor(
        seed_range=seed_range,
        seed_std=seed_std,
        n_groups=len(replicated),
        median_judge_sem=_median(sems) if sems else None,
    )


def _pooled_std(groups: Sequence[Sequence[float]]) -> float | None:
    numerator = 0.0
    dof = 0
    for values in groups:
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        numerator += sum((v - mean) ** 2 for v in values)
        dof += len(values) - 1
    if dof == 0:
        return None
    return math.sqrt(numerator / dof)


# ---------------------------------------------------------------------------
# Free by-products: proxy validity and the epochs question
# ---------------------------------------------------------------------------


def proxy_validation(
    observations: Sequence[Observation],
    *,
    cells: Sequence[str] | None = None,
) -> ProxyReport:
    """Correlate the sweep's selection proxy against the judge, per cell."""
    obs = [
        o for o in scored(observations)
        if isinstance(o.eval_loss, (int, float))
        and (cells is None or o.cell_id in set(cells))
    ]
    by_cell: dict[str, list[Observation]] = {}
    for o in obs:
        by_cell.setdefault(o.cell_id, []).append(o)

    per_cell: list[dict[str, Any]] = []
    rhos: list[float] = []
    hits = 0
    counted = 0
    for cell, rows in sorted(by_cell.items()):
        if len(rows) < 3:
            continue
        rho = spearman(
            [float(r.eval_loss) for r in rows],
            [float(r.judge_mean) for r in rows],
        )
        proxy_pick = min(rows, key=lambda r: r.eval_loss).config_id
        judge_pick = max(rows, key=lambda r: r.judge_mean).config_id
        hit = proxy_pick == judge_pick
        counted += 1
        hits += int(hit)
        if rho is not None:
            rhos.append(rho)
        per_cell.append({
            "cell_id": cell,
            "n_configs": len(rows),
            "spearman": rho,
            "proxy_pick": proxy_pick,
            "judge_pick": judge_pick,
            "top1_hit": hit,
        })

    return ProxyReport(
        n_cells=len(per_cell),
        mean_spearman=(sum(rhos) / len(rhos)) if rhos else None,
        median_spearman=_median(rhos) if rhos else None,
        top1_hit_rate=(hits / counted) if counted else None,
        per_cell=per_cell,
    )


def epoch_report(observations: Sequence[Observation]) -> EpochReport:
    """How often the saved checkpoint came from before the last epoch."""
    rows = [
        o for o in observations
        if isinstance(o.best_epoch, (int, float))
        and isinstance(o.num_epochs, int)
    ]
    if not rows:
        return EpochReport(n_runs=0, n_stopped_early=0, early_stop_rate=None)

    def stopped_early(o: Observation) -> bool:
        # Half an epoch of slack: eval fires on a step schedule, so the final
        # eval can land marginally before the epoch boundary without that
        # meaning anything.
        return float(o.best_epoch) < float(o.num_epochs) - 0.5

    by_epochs: dict[int, list[Observation]] = {}
    for o in rows:
        by_epochs.setdefault(int(o.num_epochs), []).append(o)

    summary = []
    for n_epochs, group in sorted(by_epochs.items()):
        early = [o for o in group if stopped_early(o)]
        summary.append({
            "num_epochs": n_epochs,
            "n_runs": len(group),
            "n_stopped_early": len(early),
            "mean_best_epoch": sum(float(o.best_epoch) for o in group) / len(group),
        })

    early_total = sum(1 for o in rows if stopped_early(o))
    return EpochReport(
        n_runs=len(rows),
        n_stopped_early=early_total,
        early_stop_rate=early_total / len(rows),
        by_num_epochs=summary,
    )


# ---------------------------------------------------------------------------
# Small statistics helpers (kept dependency-free and testable)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260805,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for the mean. Deterministic for a given seed."""
    data = [float(v) for v in values]
    if not data:
        return None, None
    if len(data) == 1:
        return data[0], data[0]
    rng = random.Random(seed)
    n = len(data)
    means = []
    for _ in range(samples):
        means.append(sum(data[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return (
        means[int(tail * len(means))],
        means[min(len(means) - 1, int((1.0 - tail) * len(means)))],
    )


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation. None when either side is constant (rho undefined)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    return pearson(rx, ry)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _ranks(values: Sequence[float]) -> list[float]:
    """Ascending ranks with ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and math.isclose(
            values[order[j + 1]], values[order[i]], abs_tol=1e-12
        ):
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
