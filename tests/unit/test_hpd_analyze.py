"""Tests for the hp_defaults analysis.

This module decides which hyperparameters ship as the product's defaults, and
it does so from data no reviewer will re-derive by hand. A bug here does not
crash anything — it just recommends the wrong learning rate, forever. So every
claim the report can make is pinned to a synthetic table with a known answer.

The cases worth reading first are the ones about *refusing* to conclude:
`test_conditional_split_rejected_when_gain_is_inside_the_noise_floor` and
`test_regret_ignores_configs_that_failed_to_score`.
"""

from __future__ import annotations

import math

import pytest

from tests.benchmarks.hp_defaults.analyze import (
    NoiseFloor,
    Observation,
    balanced_panel,
    bootstrap_ci,
    cell_oracles,
    conditional_defaults,
    epoch_report,
    noise_floor,
    proxy_validation,
    rank_configs,
    recommend,
    spearman,
    widest_panel,
)


def obs(
    cell: str, config: str, judge: float | None = None, **kw
) -> Observation:
    dims = kw.pop("dimensions", None) or {
        "model_kind": "instruct", "param_count": 1.2,
        "train_size": 2000, "task": "translation",
    }
    return Observation(
        cell_id=cell, config_id=config, judge_mean=judge, dimensions=dims, **kw
    )


# ---------------------------------------------------------------------------
# Regret and oracles
# ---------------------------------------------------------------------------


def test_oracle_is_the_best_score_in_each_cell():
    data = [
        obs("c1", "a", 7.0), obs("c1", "b", 8.0),
        obs("c2", "a", 5.0), obs("c2", "b", 4.0),
    ]
    assert cell_oracles(data) == {"c1": 8.0, "c2": 5.0}


def test_oracle_is_scoped_to_the_compared_configs():
    """A config from an earlier stage must not set an oracle the finalists
    were never given a chance to reach — that would inflate their regret."""
    data = [
        obs("c1", "a", 7.0), obs("c1", "b", 8.0), obs("c1", "screened_only", 9.5),
    ]
    assert cell_oracles(data, configs=["a", "b"]) == {"c1": 8.0}


def test_mean_regret_picks_the_config_that_loses_least_not_the_one_that_wins_most():
    """The headline reason regret is the decision metric.

    'b' wins three of four cells by a hair and then falls off a cliff in the
    fourth. 'a' never wins. As a *default* — one setting applied everywhere —
    'a' is plainly the right choice, and a win count would say otherwise.
    """
    data = [
        obs("c1", "a", 7.9), obs("c1", "b", 8.0),
        obs("c2", "a", 7.9), obs("c2", "b", 8.0),
        obs("c3", "a", 7.9), obs("c3", "b", 8.0),
        obs("c4", "a", 7.9), obs("c4", "b", 3.0),
    ]
    ranking = rank_configs(data, configs=["a", "b"], cells=["c1", "c2", "c3", "c4"])
    assert ranking[0].config_id == "a"
    assert ranking[0].wins == 1
    by_id = {r.config_id: r for r in ranking}
    assert by_id["b"].wins == 3
    assert by_id["a"].mean_regret == pytest.approx(0.075)
    assert by_id["b"].mean_regret == pytest.approx(1.225)
    # Worst case matters for a default too, and is reported alongside.
    assert by_id["b"].max_regret == pytest.approx(4.9)


def test_ties_are_broken_towards_the_faster_config():
    """Equal quality for less wall-clock is strictly better as a default."""
    data = [
        obs("c1", "slow", 8.0, elapsed_s=600.0),
        obs("c1", "fast", 8.0, elapsed_s=100.0),
    ]
    ranking = rank_configs(data, configs=["slow", "fast"], cells=["c1"])
    assert ranking[0].config_id == "fast"


def test_regret_ignores_configs_that_failed_to_score():
    """A missing judge score is not a score of zero.

    Left in as 0.0 a crashed config would set the worst regret in the cell and,
    worse, could win an argmin elsewhere. It must simply not be counted.
    """
    data = [
        obs("c1", "a", 8.0), obs("c1", "b", None, note="oom"),
        obs("c2", "a", 6.0), obs("c2", "b", 9.0),
    ]
    ranking = rank_configs(data, configs=["a", "b"], cells=["c1", "c2"])
    by_id = {r.config_id: r for r in ranking}
    assert by_id["a"].n_cells == 2
    assert by_id["b"].n_cells == 1        # only c2 counted
    assert cell_oracles(data)["c1"] == 8.0


def test_average_rank_shares_tied_positions():
    data = [obs("c1", "a", 8.0), obs("c1", "b", 8.0), obs("c1", "c", 5.0)]
    ranking = rank_configs(data, configs=["a", "b", "c"], cells=["c1"])
    by_id = {r.config_id: r for r in ranking}
    assert by_id["a"].mean_rank == pytest.approx(1.5)
    assert by_id["b"].mean_rank == pytest.approx(1.5)
    assert by_id["c"].mean_rank == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Panels — comparing like with like across a two-stage study
# ---------------------------------------------------------------------------


def test_balanced_panel_is_the_cells_where_every_config_was_measured():
    data = [
        obs("c1", "a", 7.0), obs("c1", "b", 8.0),
        obs("c2", "a", 7.0),                        # b missing here
        obs("c3", "a", 7.0), obs("c3", "b", 6.0),
    ]
    assert balanced_panel(data, ["a", "b"]) == ["c1", "c3"]
    assert balanced_panel(data, ["a"]) == ["c1", "c2", "c3"]


def test_widest_panel_keeps_the_full_grid_when_it_ran_everywhere():
    data = [obs(f"c{i}", cfg, 7.0) for i in range(3) for cfg in ("a", "b", "c")]
    configs, cells = widest_panel(data)
    assert configs == ["a", "b", "c"]
    assert len(cells) == 3


def test_widest_panel_narrows_to_finalists_when_that_buys_more_cells():
    """The two-stage shape: 3 configs screened on 2 cells, 1 finalist confirmed
    on 10. Comparing 2 finalists over 10 cells beats 3 over 2."""
    data = []
    for i in range(2):                                   # screening cells
        for cfg in ("a", "b", "screened_only"):
            data.append(obs(f"screen{i}", cfg, 7.0))
    for i in range(10):                                  # confirmation cells
        for cfg in ("a", "b"):
            data.append(obs(f"confirm{i}", cfg, 7.0))

    configs, cells = widest_panel(data)
    assert configs == ["a", "b"]
    assert len(cells) == 12
    assert "screened_only" not in configs


def test_recommendation_reports_what_it_excluded():
    data = [
        obs("c1", "a", 7.0), obs("c1", "b", 8.0), obs("c1", "rare", 9.0),
        obs("c2", "a", 7.0), obs("c2", "b", 8.0),
        obs("c3", "a", 7.0), obs("c3", "b", 8.0),
    ]
    rec = recommend(data, bootstrap_samples=200)
    assert rec.config_id == "b"
    assert rec.excluded_configs == ["rare"]
    assert len(rec.panel_cells) == 3


def test_recommendation_is_empty_rather_than_wrong_when_nothing_scored():
    rec = recommend([obs("c1", "a", None)], bootstrap_samples=100)
    assert rec.config_id is None
    assert rec.mean_regret is None


# ---------------------------------------------------------------------------
# Conditional defaults — the "do we need more than one default?" question
# ---------------------------------------------------------------------------


def _split_dataset(base_gap: float):
    """Cells where 'a' suits instruct models and 'b' suits base models."""
    data = []
    for i in range(6):
        kind = "instruct" if i < 3 else "base"
        dims = {
            "model_kind": kind, "param_count": 1.2,
            "train_size": 2000, "task": "translation",
        }
        if kind == "instruct":
            a_score, b_score = 8.0, 8.0 - base_gap
        else:
            a_score, b_score = 8.0 - base_gap, 8.0
        data.append(obs(f"c{i}", "a", a_score, dimensions=dims))
        data.append(obs(f"c{i}", "b", b_score, dimensions=dims))
    return data


def test_conditional_split_reported_when_the_gain_is_real():
    data = _split_dataset(base_gap=2.0)
    cells = [f"c{i}" for i in range(6)]
    findings = conditional_defaults(
        data, global_config="a", configs=["a", "b"], cells=cells,
        noise=NoiseFloor(seed_range=0.1, seed_std=0.05, n_groups=4,
                         median_judge_sem=0.05),
        dimensions=["model_kind"], bootstrap_samples=400,
    )
    finding = findings[0]
    assert finding.split_recommended is True
    base_level = next(lv for lv in finding.levels if lv.level == "base")
    assert base_level.best_config == "b"
    assert base_level.gain == pytest.approx(2.0)
    assert base_level.beats_noise is True
    # The global default is already optimal for instruct, so nothing to gain.
    instruct_level = next(lv for lv in finding.levels if lv.level == "instruct")
    assert instruct_level.gain == pytest.approx(0.0)
    assert instruct_level.beats_noise is False


def test_conditional_split_rejected_when_gain_is_inside_the_noise_floor():
    """The guardrail that stops the study inventing rules out of variance.

    Same shape as above, but the per-level advantage is 0.05 judge points while
    identical configs move by 0.4 across seeds. That is not a finding.
    """
    data = _split_dataset(base_gap=0.05)
    cells = [f"c{i}" for i in range(6)]
    findings = conditional_defaults(
        data, global_config="a", configs=["a", "b"], cells=cells,
        noise=NoiseFloor(seed_range=0.4, seed_std=0.2, n_groups=4,
                         median_judge_sem=0.2),
        dimensions=["model_kind"], bootstrap_samples=400,
    )
    finding = findings[0]
    assert finding.split_recommended is False
    base_level = next(lv for lv in finding.levels if lv.level == "base")
    assert base_level.gain == pytest.approx(0.05)
    assert base_level.beats_noise is False


def test_conditional_split_rejected_when_the_gain_is_not_consistent():
    """A large average gain driven by one outlier cell must not pass.

    The bootstrap CI over cells straddles zero, so even though the mean gain
    (0.5) clears the noise floor, the split is not recommended.
    """
    data = []
    swings = [3.0, -1.0, -1.0, -1.0, 3.0, -1.0]
    for i, swing in enumerate(swings):
        dims = {"model_kind": "base", "param_count": 1.2,
                "train_size": 2000, "task": "translation"}
        data.append(obs(f"c{i}", "a", 8.0, dimensions=dims))
        data.append(obs(f"c{i}", "b", 8.0 + swing, dimensions=dims))

    findings = conditional_defaults(
        data, global_config="a", configs=["a", "b"],
        cells=[f"c{i}" for i in range(6)],
        noise=NoiseFloor(seed_range=0.1, seed_std=0.05, n_groups=4,
                         median_judge_sem=0.05),
        dimensions=["model_kind"], bootstrap_samples=1000,
    )
    level = findings[0].levels[0]
    assert level.ci_low < 0 < level.ci_high
    assert level.beats_noise is False


def test_levels_with_too_few_cells_are_skipped():
    """One cell cannot support a claim about a whole dimension."""
    data = [
        obs("c1", "a", 8.0, dimensions={"model_kind": "base", "param_count": 1.2,
                                        "train_size": 500, "task": "t"}),
        obs("c1", "b", 9.0, dimensions={"model_kind": "base", "param_count": 1.2,
                                        "train_size": 500, "task": "t"}),
        obs("c2", "a", 8.0, dimensions={"model_kind": "instruct", "param_count": 1.2,
                                        "train_size": 500, "task": "t"}),
        obs("c2", "b", 7.0, dimensions={"model_kind": "instruct", "param_count": 1.2,
                                        "train_size": 500, "task": "t"}),
    ]
    findings = conditional_defaults(
        data, global_config="a", configs=["a", "b"], cells=["c1", "c2"],
        noise=NoiseFloor(0.1, 0.05, 4, 0.05),
        dimensions=["model_kind"], min_cells_per_level=2, bootstrap_samples=200,
    )
    assert findings[0].levels == []
    assert findings[0].split_recommended is False


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------


def test_noise_floor_from_seed_replicates():
    data = [
        obs("c1", "a", 8.0, seed=1), obs("c1", "a", 8.4, seed=2),
        obs("c2", "a", 7.0, seed=1), obs("c2", "a", 7.2, seed=2),
    ]
    floor = noise_floor(data)
    assert floor.n_groups == 2
    assert floor.seed_range == pytest.approx(0.3)   # mean of 0.4 and 0.2
    assert floor.basis == "seed_replicates"
    assert floor.measured is True
    assert floor.value == pytest.approx(0.3)


def test_noise_floor_falls_back_to_judge_standard_error():
    """No replicates run — use the judge's own sampling error instead.

    But it is only a LOWER BOUND: it captures judging variance and misses
    seed-to-seed training variance, so `measured` stays False and the report
    has to say the split is unconfirmed.
    """
    data = [
        obs("c1", "a", 8.0, judge_std=2.0, judge_num_scored=100),
        obs("c2", "a", 7.0, judge_std=2.0, judge_num_scored=100),
    ]
    floor = noise_floor(data)
    assert floor.seed_range is None
    assert floor.median_judge_sem == pytest.approx(0.2)
    assert floor.value == pytest.approx(0.4)   # 2 SEM
    assert floor.basis == "judge_sem"
    assert floor.measured is False


def test_noise_floor_reports_itself_unmeasured_rather_than_zero():
    """An unmeasured floor must be visible, not silently permissive."""
    floor = noise_floor([obs("c1", "a", 8.0)])
    assert floor.basis == "none"
    assert floor.measured is False
    assert floor.value == 0.0


def test_single_seed_groups_do_not_count_as_replicates():
    data = [obs("c1", "a", 8.0, seed=1), obs("c2", "a", 7.0, seed=1)]
    assert noise_floor(data).n_groups == 0


# ---------------------------------------------------------------------------
# Proxy validation
# ---------------------------------------------------------------------------


def test_proxy_validation_detects_a_working_proxy():
    """Lower eval_loss should mean higher judge score: rho near -1."""
    data = [
        obs("c1", "a", 6.0, eval_loss=1.5),
        obs("c1", "b", 7.0, eval_loss=1.0),
        obs("c1", "c", 8.0, eval_loss=0.5),
    ]
    report = proxy_validation(data)
    assert report.n_cells == 1
    assert report.mean_spearman == pytest.approx(-1.0)
    assert report.top1_hit_rate == 1.0
    assert report.per_cell[0]["proxy_pick"] == "c"


def test_proxy_validation_detects_a_broken_proxy():
    """If the proxy ranks backwards, the study must say so — this is the
    assumption the shipped sweep's winner selection rests on."""
    data = [
        obs("c1", "a", 8.0, eval_loss=1.5),
        obs("c1", "b", 7.0, eval_loss=1.0),
        obs("c1", "c", 6.0, eval_loss=0.5),
    ]
    report = proxy_validation(data)
    assert report.mean_spearman == pytest.approx(1.0)
    assert report.top1_hit_rate == 0.0


def test_proxy_validation_skips_cells_with_too_few_configs():
    data = [obs("c1", "a", 8.0, eval_loss=1.0), obs("c1", "b", 7.0, eval_loss=1.5)]
    assert proxy_validation(data).n_cells == 0


# ---------------------------------------------------------------------------
# The epochs question
# ---------------------------------------------------------------------------


def test_epoch_report_counts_runs_that_stopped_before_the_last_epoch():
    """If 3-epoch runs keep saving epoch-1 checkpoints, the epochs axis is
    mostly re-doing what load_best_model_at_end already does."""
    data = [
        obs("c1", "e3", 8.0, num_epochs=3, best_epoch=1.0),
        obs("c2", "e3", 8.0, num_epochs=3, best_epoch=3.0),
        obs("c3", "e1", 7.0, num_epochs=1, best_epoch=1.0),
    ]
    report = epoch_report(data)
    assert report.n_runs == 3
    assert report.n_stopped_early == 1
    assert report.early_stop_rate == pytest.approx(1 / 3)
    by_epochs = {row["num_epochs"]: row for row in report.by_num_epochs}
    assert by_epochs[3]["n_stopped_early"] == 1
    assert by_epochs[1]["n_stopped_early"] == 0


def test_epoch_report_tolerates_sub_epoch_eval_cadence():
    """Eval fires on a step schedule, so best_epoch is rarely a round number.
    2.8 out of 3 epochs is the last epoch, not an early stop."""
    data = [obs("c1", "e3", 8.0, num_epochs=3, best_epoch=2.8)]
    assert epoch_report(data).n_stopped_early == 0


def test_epoch_report_is_empty_without_epoch_data():
    assert epoch_report([obs("c1", "a", 8.0)]).n_runs == 0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def test_spearman_handles_ties_and_constants():
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert spearman([1.0], [2.0]) is None


def test_spearman_is_rank_based_not_value_based():
    """A monotone but wildly non-linear relation is still rho = 1."""
    assert spearman([1.0, 2.0, 3.0], [1.0, 10.0, 1000.0]) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [1.0, 1.1, 0.9, 1.05, 0.95] * 4
    low, high = bootstrap_ci(values, samples=500, seed=7)
    assert low < 1.0 < high
    assert (low, high) == bootstrap_ci(values, samples=500, seed=7)


def test_bootstrap_ci_of_a_constant_has_zero_width():
    low, high = bootstrap_ci([2.0, 2.0, 2.0], samples=200)
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(2.0)


def test_bootstrap_ci_of_nothing_is_none():
    assert bootstrap_ci([]) == (None, None)


# ---------------------------------------------------------------------------
# End-to-end on a table with a known answer
# ---------------------------------------------------------------------------


def test_full_analysis_recovers_a_planted_global_default():
    """Build cells where lr=5e-5 is uniformly best by a clear margin, add
    seed noise smaller than that margin, and check the study finds it and
    declines to split any dimension."""
    data = []
    for task in ("translation", "extraction"):
        for size in (500, 2000):
            for kind in ("base", "instruct"):
                cell = f"{task}_{size}_{kind}"
                dims = {"task": task, "train_size": size,
                        "model_kind": kind, "param_count": 1.2}
                for config, quality in (
                    ("lr1e-5_e2", 6.5), ("lr5e-5_e2", 8.0), ("lr2e-4_e2", 5.0),
                ):
                    data.append(obs(
                        cell, config, quality,
                        dimensions=dims,
                        eval_loss=10.0 - quality,   # a well-behaved proxy
                        num_epochs=2, best_epoch=2.0,
                        judge_std=2.0, judge_num_scored=400,
                    ))
    # Replicates: identical configs differ by 0.1 across seeds.
    for seed, delta in ((1, 0.0), (2, 0.1)):
        data.append(obs("translation_500_base", "lr5e-5_e2", 8.0 + delta,
                        seed=seed, dimensions={
                            "task": "translation", "train_size": 500,
                            "model_kind": "base", "param_count": 1.2}))

    configs, cells = widest_panel([o for o in data if o.seed is None])
    rec = recommend(data, configs=configs, cells=cells, bootstrap_samples=500)
    assert rec.config_id == "lr5e-5_e2"
    assert rec.mean_regret == pytest.approx(0.0)

    floor = noise_floor(data)
    assert floor.seed_range == pytest.approx(0.1)

    findings = conditional_defaults(
        data, global_config=rec.config_id, configs=configs, cells=cells,
        noise=floor, bootstrap_samples=500,
    )
    assert all(not f.split_recommended for f in findings), (
        "one default is optimal everywhere here; no dimension should split"
    )

    proxy = proxy_validation(data, cells=cells)
    assert proxy.top1_hit_rate == 1.0
    assert proxy.mean_spearman is not None and proxy.mean_spearman < -0.99


def test_full_analysis_recovers_a_planted_per_dimension_default():
    """Same shape, but small models genuinely want a higher learning rate.
    The study must split on param_count and only on param_count."""
    data = []
    for task in ("translation", "extraction"):
        for size in (500, 2000):
            for params in (0.35, 1.2):
                cell = f"{task}_{size}_{params}"
                dims = {"task": task, "train_size": size,
                        "model_kind": "instruct", "param_count": params}
                if params == 0.35:
                    scores = {"lr5e-5_e2": 6.0, "lr2e-4_e2": 8.0}
                else:
                    scores = {"lr5e-5_e2": 8.0, "lr2e-4_e2": 6.0}
                for config, quality in scores.items():
                    data.append(obs(cell, config, quality, dimensions=dims))
    for seed, delta in ((1, 0.0), (2, 0.1)):
        data.append(obs("translation_500_0.35", "lr2e-4_e2", 8.0 + delta,
                        seed=seed, dimensions={
                            "task": "translation", "train_size": 500,
                            "model_kind": "instruct", "param_count": 0.35}))

    configs, cells = widest_panel([o for o in data if o.seed is None])
    rec = recommend(data, configs=configs, cells=cells, bootstrap_samples=500)
    findings = conditional_defaults(
        data, global_config=rec.config_id, configs=configs, cells=cells,
        noise=noise_floor(data), bootstrap_samples=500,
    )
    by_dim = {f.dimension: f for f in findings}

    assert by_dim["param_count"].split_recommended is True
    small = next(lv for lv in by_dim["param_count"].levels if lv.level == 0.35)
    large = next(lv for lv in by_dim["param_count"].levels if lv.level == 1.2)
    assert small.best_config == "lr2e-4_e2"
    assert large.best_config == "lr5e-5_e2"
    # Task and train_size carry no signal here and must not be split on.
    assert by_dim["task"].split_recommended is False
    assert by_dim["train_size"].split_recommended is False


def test_analysis_of_an_empty_table_does_not_crash():
    assert recommend([]).config_id is None
    assert noise_floor([]).measured is False
    assert proxy_validation([]).n_cells == 0
    assert epoch_report([]).n_runs == 0
    assert widest_panel([]) == ([], [])
    assert math.isclose(NoiseFloor(None, None, 0, None).value, 0.0)
