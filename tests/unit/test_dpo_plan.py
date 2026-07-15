"""Regression tests for the DPO remediation plan."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


async def test_gap_selector_aligns_sample_ids_and_excludes_bad_pairs(
    tmp_path: Path,
) -> None:
    from lqh.golden import generate_golden

    output = tmp_path / "iter_000"
    output.mkdir()
    conversations = {
        0: [
            {"role": "user", "content": "p0"},
            {"role": "assistant", "content": "rejected 0"},
        ],
        1: [
            {"role": "user", "content": "p1"},
            {"role": "assistant", "content": "rejected 1"},
        ],
        2: [
            {"role": "user", "content": "p2"},
            {"role": "assistant", "content": "rejected 2"},
        ],
    }
    pq.write_table(
        pa.table({
            "sample_index": [2, 0, 1],
            "messages": [json.dumps(conversations[i]) for i in (2, 0, 1)],
        }),
        output / "predictions.parquet",
    )
    # Deliberately use a different order. Chosen scores are [2, 5, 8], so
    # samples 0 and 1 are inverted/tied while only sample 2 has a valid gap.
    pq.write_table(
        pa.table({
            "sample_index": [1, 2, 0],
            "score": [5.0, 4.0, 3.0],
            "reasoning": ["tie", "valid", "inverted"],
        }),
        output / "results.parquet",
    )
    dataset = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({
            "messages": [
                json.dumps([
                    {"role": "user", "content": f"p{i}"},
                    {"role": "assistant", "content": f"chosen {i}"},
                ])
                for i in range(3)
            ],
        }),
        dataset,
    )

    await generate_golden(
        predictions_path=output / "predictions.parquet",
        scores_path=output / "results.parquet",
        dataset_path=str(dataset),
        config={
            "golden_source": "dataset",
            "selection": {
                "top_quantile": 1.0,
                "min_gap": 1.0,
                "min_pairs_per_iter": 1,
            },
        },
        client=MagicMock(),
        output_dir=output,
        chosen_scores=[2.0, 5.0, 8.0],
    )

    preferences = pq.read_table(output / "preferences.parquet")
    assert preferences.num_rows == 1
    assert preferences["chosen"][0].as_py() == "chosen 2"
    assert preferences["rejected"][0].as_py() == "rejected 2"
    stats = json.loads((output / "preference_stats.json").read_text())
    assert stats["inverted_pairs"] == 1
    assert stats["tied_pairs"] == 1
    assert stats["pairs_after_min_gap"] == 1


def _write_iteration(
    root: Path,
    iteration: int,
    *,
    judge_mean: float,
    ce_delta: float,
    artifact: str = "summary",
) -> None:
    directory = root / "iterations" / f"iter_{iteration:03d}"
    directory.mkdir(parents=True)
    (directory / "chosen_ce_summary.json").write_text(json.dumps({
        "eval_ce_chosen_mean": 0.8,
        "eval_ce_chosen_delta_ref": ce_delta,
    }))
    if artifact == "summary":
        held_out = directory / "held_out_eval"
        held_out.mkdir()
        (held_out / "summary.json").write_text(json.dumps({
            "scores": {"mean": judge_mean},
        }))
    else:
        (directory / "eval_result.json").write_text(json.dumps({
            "summary": {"scores": {"mean": judge_mean}},
        }))


def test_dpo_sweep_uses_fixed_heldout_judge_and_best_iteration(
    tmp_path: Path,
) -> None:
    from lqh.train.sweep import _read_dpo_proxy

    _write_iteration(tmp_path, 0, judge_mean=5.0, ce_delta=0.01)
    _write_iteration(
        tmp_path, 1, judge_mean=6.25, ce_delta=0.02, artifact="eval_result",
    )
    proxy = _read_dpo_proxy(tmp_path)
    assert proxy["primary"] == pytest.approx(-6.25)
    assert proxy["held_out_judge_mean"] == pytest.approx(6.25)
    assert proxy["best_iteration"] == 1
    assert proxy["selection_source"] == "held_out_judge"


def test_late_ce_collapse_makes_dpo_run_ineligible(tmp_path: Path) -> None:
    from lqh.train.sweep import _is_collapsed, _read_dpo_proxy

    _write_iteration(tmp_path, 0, judge_mean=5.0, ce_delta=0.01)
    _write_iteration(tmp_path, 1, judge_mean=7.0, ce_delta=0.75)
    proxy = _read_dpo_proxy(tmp_path)
    assert proxy["max_eval_ce_chosen_delta_ref"] == pytest.approx(0.75)
    assert _is_collapsed(proxy, "on_policy_dpo") is True


def test_best_iteration_reader_supports_inline_scoring_artifacts(
    tmp_path: Path,
) -> None:
    from lqh.train.dpo_metrics import find_best_held_out_iter

    _write_iteration(tmp_path, 0, judge_mean=5.0, ce_delta=0.0)
    _write_iteration(
        tmp_path, 1, judge_mean=6.0, ce_delta=0.0, artifact="eval_result",
    )
    assert find_best_held_out_iter(tmp_path / "iterations") == (1, 6.0)


def test_paired_bootstrap_detects_consistent_gain() -> None:
    from tests.benchmarks.dpo_value.stats import paired_bootstrap

    control = {i: float(i % 5) for i in range(100)}
    treatment = {i: score + 0.5 for i, score in control.items()}
    result = paired_bootstrap(treatment, control, samples=1_000, seed=7)
    assert result.n == 100
    assert result.mean == pytest.approx(0.5)
    assert result.ci_low == pytest.approx(0.5)
    assert result.ci_high == pytest.approx(0.5)


def test_voice_metrics_surface_frustration_misses(tmp_path: Path) -> None:
    from tests.benchmarks.dpo_value.voice_metrics import voice_metrics

    def payload(score: int, failures: list[str]) -> str:
        return json.dumps({
            "reasoning": "specific reason",
            "score": score,
            "failure_tags": failures,
            "success_tags": [] if failures else ["success"],
            "failed_turns": [1] if failures else [],
            "successful_turns": [] if failures else [1],
        })

    reference = tmp_path / "reference.parquet"
    prediction = tmp_path / "prediction.parquet"
    pq.write_table(pa.table({"messages": [json.dumps([
        {"role": "user", "content": "transcript"},
        {"role": "assistant", "content": payload(2, ["wrong_action"])},
    ])]}), reference)
    pq.write_table(pa.table({
        "sample_index": [0],
        "messages": [json.dumps([
            {"role": "user", "content": "transcript"},
            {"role": "assistant", "content": payload(5, [])},
        ])],
    }), prediction)

    metrics = voice_metrics(prediction, reference)
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["score_direction_accuracy"] == 0.0
    assert metrics["frustration_miss_rate"] == 1.0
    assert metrics["failure_tags_exact_rate"] == 0.0


def test_watcher_discovers_dpo_sweep_child_iterations(tmp_path: Path) -> None:
    from lqh.watcher import RunWatcher

    run = tmp_path / "runs" / "dpo_sweep"
    direct = run / "iterations" / "iter_000"
    child = run / "sweep_lr1e-6" / "iterations" / "iter_001"
    direct.mkdir(parents=True)
    child.mkdir(parents=True)
    watcher = RunWatcher(
        run_dir=run,
        config={
            "type": "sweep",
            "base_config": {"type": "on_policy_dpo"},
        },
        project_dir=tmp_path,
        api_key="test",
    )
    assert watcher._iteration_dirs() == [direct, child]


def test_base_benchmark_dpo_config_uses_fixed_heldout_and_small_batch() -> None:
    from tests.benchmarks.base_vs_instruct.run import _base_config

    config = _base_config(
        run_type="on_policy_dpo",
        base_model="model",
        dataset_rel="datasets/fresh_dpo/data.parquet",
        eval_rel="datasets/validation/data.parquet",
        scorer_rel="scorers/task.md",
        train_size=2_000,
    )
    assert config["held_out_eval_dataset"] == "datasets/validation/data.parquet"
    assert config["training"]["effective_batch_size"] == 16
    assert config["selection"]["min_gap"] == 1.0
