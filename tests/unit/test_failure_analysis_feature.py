"""Tests for the failure-analysis feature set (IMPROVE.md).

Covers the shared score-distribution stats (single source of truth between
the run_scoring rendering and the GPU-eval ``eval_result.json``), the
``get_eval_failures`` browse mode, the dataset-row-count run context, the
run-root final-eval rendering, and the ``failure_analysis`` skill registry
entries.

All pure-Python — the scoring test patches ``run_scoring`` so nothing hits
a judge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# score_distribution_stats / format_score_distribution_text
# ---------------------------------------------------------------------------


class TestScoreDistributionStats:
    def test_empty_returns_none(self) -> None:
        from lqh.scoring import score_distribution_stats

        assert score_distribution_stats([]) is None

    def test_percentiles_nearest_rank_and_histogram(self) -> None:
        from lqh.scoring import score_distribution_stats

        dist = score_distribution_stats([float(s) for s in range(1, 11)])
        assert dist is not None
        assert dist["n"] == 10
        # Nearest-INDEX estimator (numpy method="nearest"): idx =
        # round(p * 9) with banker's rounding, so p50 → idx round(4.5) == 4
        # → value 5.0. Deliberately identical to the legacy run_scoring
        # rendering; NOT classical nearest-rank (which would give p10=1).
        assert dist["percentiles"] == {
            "p10": 2.0, "p25": 3.0, "p50": 5.0, "p75": 8.0, "p90": 9.0,
        }
        # Histogram keys are strings so the dict is JSON-stable.
        assert dist["histogram"]["1"] == 1
        assert dist["histogram"]["10"] == 1
        assert set(dist["histogram"]) == {str(b) for b in range(0, 11)}
        # Round-trips through JSON unchanged.
        assert json.loads(json.dumps(dist)) == dist

    def test_buckets_floor_and_clamp(self) -> None:
        from lqh.scoring import score_distribution_stats

        dist = score_distribution_stats([0.0, 0.5, 6.7, 6.2, 11.0])
        assert dist is not None
        assert dist["histogram"]["0"] == 2   # 0 is the worst GRADE, not an error
        assert dist["histogram"]["6"] == 2   # floor(6.7) == floor(6.2) == 6
        assert dist["histogram"]["10"] == 1  # 11.0 clamped down
        assert set(dist["histogram"]) == {str(b) for b in range(0, 11)}

    def test_valid_zero_grades_kept_errors_dropped(self, tmp_path: Path) -> None:
        """A judge-issued 0/10 counts in the distribution; only samples with
        scoring-error reasoning are excluded. The wrapper matches the shared
        renderer byte-for-byte."""
        from lqh.scoring import (
            format_score_distribution_text,
            score_distribution_stats,
        )
        from lqh.tools.handlers import _format_score_distribution

        scores = [0.0, 0.0, 3.0, 5.0, 8.0, 10.0]
        reasons = [
            "empty response — worst grade",     # real 0/10
            "[Scoring error] judge 429",        # error placeholder → dropped
            "ok", "ok", "ok", "ok",
        ]
        parquet = tmp_path / "results.parquet"
        pq.write_table(pa.table({"score": scores, "reasoning": reasons}), parquet)

        expected = format_score_distribution_text(
            score_distribution_stats([0.0, 3.0, 5.0, 8.0, 10.0])
        )
        assert _format_score_distribution(parquet) == expected
        assert "Score distribution (n=5):" in expected
        assert "\n     0 | " in expected  # bucket 0 rendered

    def test_wrapper_without_reasoning_column_uses_legacy_filter(
        self, tmp_path: Path
    ) -> None:
        """No reasoning column → no way to tell a real 0 from an error
        placeholder; the legacy s>0 proxy applies."""
        from lqh.scoring import (
            format_score_distribution_text,
            score_distribution_stats,
        )
        from lqh.tools.handlers import _format_score_distribution

        parquet = tmp_path / "results.parquet"
        pq.write_table(pa.table({"score": [0.0, 3.0, 8.0]}), parquet)
        expected = format_score_distribution_text(
            score_distribution_stats([3.0, 8.0])
        )
        assert _format_score_distribution(parquet) == expected


class TestEvalResultDistribution:
    async def test_payload_contains_distribution_excluding_errors(
        self, tmp_path: Path
    ) -> None:
        from lqh.scoring import ScoringResult, score_predictions_by_source

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        preds = tmp_path / "predictions.parquet"
        pq.write_table(
            pa.table({"sample_index": [0, 1, 2], "messages": ["a", "b", "c"]}),
            preds,
        )

        async def fake_run_scoring(*, dataset_path, scorer_path, output_dir, client, **kw):
            pq.write_table(
                pa.table({
                    "sample_index": [0, 1, 2],
                    "messages": ["a", "b", "c"],
                    "score": [8.0, 4.0, 0.0],
                    "reasoning": ["ok", "ok", "[Scoring error] judge down"],
                }),
                output_dir / "results.parquet",
            )
            return ScoringResult(
                total=3, scored=2, failed=1,
                mean_score=6.0, median_score=6.0, output_dir=output_dir,
            )

        with patch("lqh.scoring.run_scoring", side_effect=fake_run_scoring):
            payload = await score_predictions_by_source(
                predictions_path=preds,
                scorer_path=tmp_path / "scorer.md",
                output_dir=out_dir,
                client=object(),
            )

        dist = payload["score_distribution"]
        assert dist["n"] == 2  # the scoring-error sample is excluded
        assert dist["histogram"]["8"] == 1
        assert dist["histogram"]["4"] == 1
        on_disk = json.loads((out_dir / "eval_result.json").read_text())
        assert on_disk["score_distribution"] == dist


# ---------------------------------------------------------------------------
# browse_results + get_eval_failures browse mode
# ---------------------------------------------------------------------------


def _write_results(path: Path, n: int = 40, error_at: int | None = 5) -> None:
    pq.write_table(
        pa.table({
            "sample_index": list(range(n)),
            "messages": [
                json.dumps([
                    {"role": "user", "content": f"prompt {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ])
                for i in range(n)
            ],
            "score": [float(1 + (i % 10)) for i in range(n)],
            "reasoning": [
                "[Scoring error] judge timeout" if i == error_at else f"judge {i}"
                for i in range(n)
            ],
        }),
        path,
    )


class TestBrowseResults:
    def test_range_filter_and_total(self, tmp_path: Path) -> None:
        from lqh.scoring import browse_results

        results = tmp_path / "results.parquet"
        _write_results(results)
        rows, errors, total = browse_results(
            results, score_min=4, score_max=6, limit=25
        )
        # Scores 4/5/6 → 12 samples, minus the scoring-error one (score 6).
        assert total == 11
        assert len(rows) == 11
        assert all(4 <= r["score"] <= 6 for r in rows)
        assert [e["sample_index"] for e in errors] == [5]

    def test_sort_desc_and_paging(self, tmp_path: Path) -> None:
        from lqh.scoring import browse_results

        results = tmp_path / "results.parquet"
        _write_results(results, error_at=None)
        page1, _, total = browse_results(results, sort="desc", limit=4)
        page2, _, _ = browse_results(results, sort="desc", limit=4, offset=4)
        assert total == 40
        assert [r["score"] for r in page1] == [10.0, 10.0, 10.0, 10.0]
        assert page1[0]["score"] >= page2[0]["score"]
        assert {r["sample_index"] for r in page1}.isdisjoint(
            {r["sample_index"] for r in page2}
        )

    def test_random_is_seed_stable_across_pages(self, tmp_path: Path) -> None:
        from lqh.scoring import browse_results

        results = tmp_path / "results.parquet"
        _write_results(results, error_at=None)
        whole, _, _ = browse_results(results, sort="random", seed=7, limit=10)
        p1, _, _ = browse_results(results, sort="random", seed=7, limit=5)
        p2, _, _ = browse_results(results, sort="random", seed=7, limit=5, offset=5)
        assert [r["sample_index"] for r in p1 + p2] == [
            r["sample_index"] for r in whole
        ]
        other, _, _ = browse_results(results, sort="random", seed=8, limit=10)
        assert [r["sample_index"] for r in other] != [
            r["sample_index"] for r in whole
        ]

    def test_sample_indices_exact_order(self, tmp_path: Path) -> None:
        from lqh.scoring import browse_results

        results = tmp_path / "results.parquet"
        _write_results(results)
        rows, _, total = browse_results(results, sample_indices=[3, 1, 999])
        assert [r["sample_index"] for r in rows] == [3, 1]
        assert total == 2


class TestGetEvalFailuresBrowseMode:
    @pytest.fixture()
    def project(self, tmp_path: Path) -> Path:
        run = tmp_path / "runs" / "probe"
        run.mkdir(parents=True)
        _write_results(run / "results.parquet")
        return tmp_path

    async def test_browse_header_and_paging_hint(self, project: Path) -> None:
        from lqh.tools.handlers import handle_get_eval_failures

        result = await handle_get_eval_failures(
            project, eval_run="runs/probe", score_min=4, score_max=6, limit=5
        )
        assert "## Samples 1–5 of 11 matching (score in [4, 6], sort=asc)" in result.content
        assert "use offset=5 to continue" in result.content
        assert "judge/scoring errors" in result.content

    async def test_legacy_mode_unchanged(self, project: Path) -> None:
        from lqh.tools.handlers import handle_get_eval_failures

        result = await handle_get_eval_failures(project, eval_run="runs/probe")
        assert result.content.startswith(
            "## Failure Cases (15 of 40 samples, threshold < 6.0)"
        )
        assert "## Scoring Errors (1 samples" in result.content

    async def test_no_match_message(self, project: Path) -> None:
        from lqh.tools.handlers import handle_get_eval_failures

        result = await handle_get_eval_failures(
            project, eval_run="runs/probe", score_min=9.5, score_max=9.9
        )
        assert result.content.startswith("No samples match the given filters.")

    async def test_max_chars_per_message(self, project: Path) -> None:
        from lqh.tools.handlers import handle_get_eval_failures

        run = project / "runs" / "long"
        run.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "sample_index": [0],
                "messages": [json.dumps([
                    {"role": "user", "content": "x" * 1000},
                ])],
                "score": [5.0],
                "reasoning": ["judge"],
            }),
            run / "results.parquet",
        )
        short = await handle_get_eval_failures(
            project, eval_run="runs/long", score_max=6
        )
        assert "x" * 500 + "..." in short.content
        assert "x" * 501 not in short.content
        long = await handle_get_eval_failures(
            project, eval_run="runs/long", score_max=6, max_chars_per_message=900
        )
        assert "x" * 900 + "..." in long.content

    async def test_multimodal_content_is_capped(self, project: Path) -> None:
        """VLM samples carry list content with base64 image data-URLs —
        those must render as placeholders, never raw, and the char cap
        applies to the joined text."""
        from lqh.tools.handlers import handle_get_eval_failures

        run = project / "runs" / "vlm"
        run.mkdir(parents=True)
        data_url = "data:image/png;base64," + "A" * 100_000
        pq.write_table(
            pa.table({
                "sample_index": [0],
                "messages": [json.dumps([
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "describe the image"},
                    ]},
                    {"role": "assistant", "content": "a cat"},
                ])],
                "score": [3.0],
                "reasoning": ["judge"],
            }),
            run / "results.parquet",
        )
        result = await handle_get_eval_failures(
            project, eval_run="runs/vlm", score_max=6
        )
        assert "base64,A" not in result.content
        assert "[image: 100,022 chars]" in result.content
        assert "describe the image" in result.content
        assert len(result.content) < 5000

    async def test_browse_export_carries_filter_not_errors(
        self, project: Path
    ) -> None:
        from lqh.tools.handlers import handle_get_eval_failures

        result = await handle_get_eval_failures(
            project,
            eval_run="runs/probe",
            score_max=3.0,
            limit=25,
            export_path="feedback/browse_v1.jsonl",
        )
        assert "💾 Exported" in result.content
        lines = [
            json.loads(l)
            for l in (project / "feedback" / "browse_v1.jsonl").read_text().splitlines()
        ]
        assert all(l["score"] <= 3.0 for l in lines)
        assert all(l["scoring_error"] is False for l in lines)
        assert lines[0]["filter"] == {"score_max": 3.0, "limit": 25}
        assert "threshold" not in lines[0]


# ---------------------------------------------------------------------------
# Run context: dataset rows + final eval rendering
# ---------------------------------------------------------------------------


class TestRunContext:
    def test_training_data_line(self) -> None:
        from lqh.tools.handlers import _training_data_line

        assert _training_data_line({
            "dataset_rows": {"train": 2000, "train_effective": 6000, "eval": 300},
        }) == "train 2,000 rows (eff. 6,000) · eval 300 rows"
        # Sweep-wrapped configs are unwrapped; equal effective count is elided.
        assert _training_data_line({
            "type": "sweep",
            "base_config": {
                "dataset_rows": {"train": 2000, "train_effective": 2000, "eval": 300},
            },
        }) == "train 2,000 rows · eval 300 rows"
        # Pre-feature configs render nothing.
        assert _training_data_line({}) == ""

    def test_sweep_manifest_carries_lineage(self, tmp_path: Path) -> None:
        """Default (sweep-wrapped) run configs nest the real config under
        base_config; the manifest must still record the lineage fields."""
        from lqh.manifest import write_run_manifest

        run_dir = tmp_path / "runs" / "sft_v1"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "type": "sweep",
            "base_config": {
                "type": "sft",
                "base_model": "lfm2.5-350m",
                "dataset": "datasets/train/data.parquet",
                "dataset_rows": {"train": 2000, "train_effective": 2000, "eval": 300},
                "num_samples": 2000,
                "training": {"learning_rate": 5e-5},
                "spec_sha256": "abc123",
            },
        }))
        path = write_run_manifest(tmp_path, run_dir, state="completed")
        manifest = json.loads(path.read_text())
        assert manifest["base_model"] == "lfm2.5-350m"
        assert manifest["dataset_rows"]["train"] == 2000
        assert manifest["num_samples"] == 2000
        assert manifest["dataset"] == "datasets/train/data.parquet"
        assert manifest["spec_sha256"] == "abc123"
        assert manifest["spec_sha256_source"] == "submission"

    def test_format_final_eval_block(self, tmp_path: Path) -> None:
        from lqh.scoring import score_distribution_stats
        from lqh.tools.handlers import _format_final_eval_block

        assert _format_final_eval_block(tmp_path) == []

        (tmp_path / "eval_result.json").write_text(json.dumps({
            "num_scored": 4, "num_failed": 1,
            "scores": {"mean": 6.5, "median": 7.0},
            "scores_weighted_mean": 6.4,
            "per_source": {
                "a": {"num_scored": 2, "scores": {"mean": 6.0}},
                "b": {"num_scored": 2, "scores": {"mean": 7.0}},
            },
            "score_distribution": score_distribution_stats([5.0, 6.0, 7.0, 8.0]),
        }))
        lines = _format_final_eval_block(tmp_path)
        assert lines[0] == "  Final eval: mean=6.50 (macro), weighted=6.40, scored=4 (1 failed)"
        assert "    a: mean=6.00" in lines
        assert any("Score distribution (n=4):" in ln for ln in lines)

        # Old-schema file (no distribution, single source) still renders.
        (tmp_path / "eval_result.json").write_text(json.dumps({
            "num_scored": 10,
            "scores": {"mean": 5.6, "median": 6.0},
        }))
        lines = _format_final_eval_block(tmp_path)
        assert lines == ["  Final eval: mean=5.60, scored=10"]

    def test_final_eval_block_falls_back_to_checkpoints_final(
        self, tmp_path: Path
    ) -> None:
        """Non-sweep SFT writes its final eval under checkpoints/final/ —
        the block must render from there when the run root has none."""
        from lqh.tools.handlers import _format_final_eval_block

        final = tmp_path / "checkpoints" / "final"
        final.mkdir(parents=True)
        (final / "eval_result.json").write_text(json.dumps({
            "num_scored": 50,
            "scores": {"mean": 6.8, "median": 7.0},
        }))
        lines = _format_final_eval_block(tmp_path)
        assert lines == ["  Final eval: mean=6.80, scored=50"]


class TestTrainingHealthBlock:
    """The mechanical training signals (steps, loss trajectory, token accuracy,
    LR) that separate "the run didn't learn" from "the data is bad"."""

    def _history(self, tmp_path: Path, rows: list[dict[str, Any]]) -> None:
        (tmp_path / "eval_history.json").write_text(json.dumps(rows))

    def _config(self, tmp_path: Path, training: dict[str, Any]) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"type": "sft", "training": training})
        )

    def test_absent_or_malformed_history_renders_nothing(
        self, tmp_path: Path
    ) -> None:
        from lqh.tools.handlers import _format_training_health_block

        assert _format_training_health_block(tmp_path) == []
        (tmp_path / "eval_history.json").write_text("{not json")
        assert _format_training_health_block(tmp_path) == []
        # Well-formed JSON of the wrong shape, and an empty history.
        (tmp_path / "eval_history.json").write_text(json.dumps({"loss": 1.0}))
        assert _format_training_health_block(tmp_path) == []
        (tmp_path / "eval_history.json").write_text(json.dumps([]))
        assert _format_training_health_block(tmp_path) == []

    def test_renders_the_full_line(self, tmp_path: Path) -> None:
        from lqh.tools.handlers import _format_training_health_block

        self._config(tmp_path, {"learning_rate": 2e-4})
        self._history(tmp_path, [
            {"step": 50, "loss": 3.74, "mean_token_accuracy": 0.38},
            {"step": 100, "loss": 2.40, "mean_token_accuracy": 0.54},
            {"step": 100, "eval_loss": 2.51},
            {"step": 100, "train_runtime": 900.0, "train_loss": 3.0},
        ])
        lines = _format_training_health_block(tmp_path)
        assert lines == [
            "  Training health: 100 steps · loss 3.74 → 2.40 · "
            "eval_loss 2.51 · token_acc 54% · lr 2.0e-04"
        ]

    def test_warns_when_the_run_took_too_few_steps(self, tmp_path: Path) -> None:
        """The 21-step run that read as 'the dataset is bad'."""
        from lqh.tools.handlers import _format_training_health_block

        self._history(tmp_path, [{"step": 21, "loss": 3.7}])
        lines = _format_training_health_block(tmp_path)
        assert "21 steps" in lines[0]
        assert len(lines) == 2
        assert "Only 21 optimizer updates" in lines[1]
        # The advice must name levers start_training actually accepts — there
        # is no batch-size parameter, so suggesting one would be a no-op call.
        assert "num_epochs" in lines[1]
        assert "batch" not in lines[1]

    def test_no_warning_on_a_healthy_step_count(self, tmp_path: Path) -> None:
        from lqh.tools.handlers import _format_training_health_block

        self._history(tmp_path, [{"step": 120, "loss": 3.7}])
        assert len(_format_training_health_block(tmp_path)) == 1

    def test_partial_history_renders_what_it_has(self, tmp_path: Path) -> None:
        """A run whose loss logged once (the old logging_steps=50 against 54
        total steps) still has to render — that run is the one being diagnosed.
        """
        from lqh.tools.handlers import _format_training_health_block

        self._history(tmp_path, [{"step": 54, "loss": 3.7}])
        assert _format_training_health_block(tmp_path) == [
            "  Training health: 54 steps · loss 3.70"
        ]

    def test_falls_back_to_the_end_of_training_summary_loss(
        self, tmp_path: Path
    ) -> None:
        from lqh.tools.handlers import _format_training_health_block

        self._history(tmp_path, [{"step": 60, "train_loss": 2.25}])
        assert _format_training_health_block(tmp_path) == [
            "  Training health: 60 steps · loss 2.25"
        ]

    def test_prefers_eval_token_accuracy(self, tmp_path: Path) -> None:
        from lqh.tools.handlers import _format_training_health_block

        self._config(tmp_path, {"learning_rate": 3e-4})
        self._history(tmp_path, [
            {"step": 100, "mean_token_accuracy": 0.5},
            {"step": 100, "eval_mean_token_accuracy": 0.61},
        ])
        line = _format_training_health_block(tmp_path)[0]
        assert "token_acc 61%" in line
        assert "lr 3.0e-04" in line

    def test_sweep_reads_the_winning_child_run(self, tmp_path: Path) -> None:
        """A sweep writes no eval_history.json at the run root — each grid
        point trains in sweep_<config_id>/. Reporting the parent's base config
        would also report the wrong LR: the base deliberately omits the swept
        hyperparameters."""
        from lqh.tools.handlers import _format_training_health_block

        (tmp_path / "config.json").write_text(json.dumps({
            "type": "sweep",
            "base_config": {"training": {"max_seq_length": 2048}},
        }))
        (tmp_path / "sweep_summary.json").write_text(json.dumps({
            "mode": "sft",
            "rows": [
                {"config_id": "sft_lr0.0003_e3", "primary": 1.9},
                {"config_id": "sft_lr0.001_e3", "primary": 2.4},
            ],
            "winner": {"config_id": "sft_lr0.0003_e3", "primary": 1.9},
        }))
        winner = tmp_path / "sweep_sft_lr0.0003_e3"
        winner.mkdir()
        (winner / "config.json").write_text(
            json.dumps({"type": "sft", "training": {"learning_rate": 3e-4}})
        )
        self._history(winner, [
            {"step": 55, "loss": 3.6},
            {"step": 110, "loss": 2.1},
            {"step": 110, "eval_loss": 1.9},
        ])
        assert _format_training_health_block(tmp_path) == [
            "  Training health: 110 steps · loss 3.60 → 2.10 · "
            "eval_loss 1.90 · lr 3.0e-04"
        ]

    def test_sweep_without_a_winner_renders_nothing(self, tmp_path: Path) -> None:
        """Every config collapsed / failed: no winner, so there is no run to
        report — better empty than the wrong child's numbers."""
        from lqh.tools.handlers import _format_training_health_block

        (tmp_path / "sweep_summary.json").write_text(
            json.dumps({"mode": "sft", "rows": [], "winner": None})
        )
        (tmp_path / "sweep_sft_lr0.001_e3").mkdir()
        self._history(tmp_path / "sweep_sft_lr0.001_e3", [{"step": 9, "loss": 4.0}])
        assert _format_training_health_block(tmp_path) == []

    def test_hydrate_pulls_the_sweep_winners_history(self, tmp_path: Path) -> None:
        """Two-pass hydration: the leaderboard has to land before we know which
        child dir to fetch."""
        import asyncio

        import lqh.tools.handlers as handlers

        (tmp_path / "sweep_summary.json").write_text(
            json.dumps({"winner": {"config_id": "sft_lr0.0003_e3"}})
        )
        fetched: list[str] = []

        async def fake_fetch(target: Path) -> bool:
            fetched.append(target.name if target.parent == tmp_path
                           else f"{target.parent.name}/{target.name}")
            return False

        with patch.object(handlers, "_fetch_run_artifact", fake_fetch):
            asyncio.run(handlers._hydrate_run_eval_artifacts(tmp_path))
        assert "sweep_sft_lr0.0003_e3/eval_history.json" in fetched
        assert "sweep_sft_lr0.0003_e3/config.json" in fetched

    def test_ignores_non_numeric_and_boolean_fields(self, tmp_path: Path) -> None:
        """log_history carries strings and bools; a bool is not a loss."""
        from lqh.tools.handlers import _format_training_health_block

        self._config(tmp_path, {"learning_rate": "fast"})
        self._history(tmp_path, [
            {"step": 100, "loss": True, "eval_loss": "n/a"},
            {"step": 100, "loss": 1.5},
        ])
        assert _format_training_health_block(tmp_path) == [
            "  Training health: 100 steps · loss 1.50"
        ]

    def test_hydrate_pulls_the_history_for_cloud_runs(self, tmp_path: Path) -> None:
        """Published as an artifact but never downloaded → the block was empty
        for every cloud run, i.e. for every real run."""
        import lqh.tools.handlers as handlers

        fetched: list[str] = []

        async def fake_fetch(target: Path) -> bool:
            fetched.append(target.name)
            return False

        with patch.object(handlers, "_fetch_run_artifact", fake_fetch):
            import asyncio

            asyncio.run(handlers._hydrate_run_eval_artifacts(tmp_path))
        assert "eval_history.json" in fetched


class TestCompletionNotice:
    async def test_notice_includes_percentile_snippet(self, tmp_path: Path) -> None:
        from lqh.jobs import JobSupervisor
        from lqh.scoring import score_distribution_stats

        run_dir = tmp_path / "runs" / "myeval"
        run_dir.mkdir(parents=True)
        (run_dir / "eval_result.json").write_text(json.dumps({
            "num_scored": 4,
            "scores": {"mean": 5.6, "median": 6.0},
            "score_distribution": score_distribution_stats([3.0, 5.0, 6.0, 8.0]),
        }))

        sup = JobSupervisor(tmp_path)

        async def fake_resolve(run_name):
            return {"artifact_id": "a1"}, True

        sup.resolve_eval_hf_result_artifact = fake_resolve  # type: ignore[method-assign]
        notice = await sup.finalize_eval_hf_run("myeval", "completed", None, "cloud")
        assert "Judge mean 5.600 over 4 scored samples." in notice
        assert "p10/p50/p90 = 3.0/6.0/8.0." in notice

    async def test_notice_tolerates_old_schema(self, tmp_path: Path) -> None:
        from lqh.jobs import JobSupervisor

        run_dir = tmp_path / "runs" / "oldeval"
        run_dir.mkdir(parents=True)
        (run_dir / "eval_result.json").write_text(json.dumps({
            "num_scored": 4,
            "scores": {"mean": 5.6, "median": 6.0},
        }))

        sup = JobSupervisor(tmp_path)

        async def fake_resolve(run_name):
            return {"artifact_id": "a1"}, True

        sup.resolve_eval_hf_result_artifact = fake_resolve  # type: ignore[method-assign]
        notice = await sup.finalize_eval_hf_run("oldeval", "completed", None, None)
        assert "Judge mean 5.600" in notice
        assert "p10/p50/p90" not in notice


class TestCloudEvalArtifacts:
    def test_publish_candidates_cover_results_parquet(self, tmp_path: Path) -> None:
        """results.parquet is published (root AND checkpoints/final — the
        non-sweep SFT layout) under the predictions retention kind, never
        the non-expiring eval_result kind (it carries raw per-sample
        content)."""
        from lqh.remote.publish import _resolve_candidates

        run = tmp_path / "runs" / "r"
        (run / "checkpoints" / "final").mkdir(parents=True)
        (run / "results.parquet").write_bytes(b"x")
        (run / "eval_result.json").write_text("{}")
        (run / "checkpoints" / "final" / "results.parquet").write_bytes(b"x")
        (run / "checkpoints" / "final" / "eval_result.json").write_text("{}")

        kinds = {c.relpath: c.kind for c in _resolve_candidates(run)}
        assert kinds["results.parquet"] == "predictions"
        assert kinds["checkpoints/final/results.parquet"] == "predictions"
        assert kinds["eval_result.json"] == "eval_result"
        assert kinds["checkpoints/final/eval_result.json"] == "eval_result"

    async def test_fetch_run_artifact_resolves_nested_relpath(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The artifacts.json manifest lives at the run root; a nested
        target (checkpoints/final/results.parquet) must match its manifest
        entry by run-relative path."""
        import lqh.artifacts as artifacts_mod
        from lqh.tools.handlers import _fetch_run_artifact

        run = tmp_path / "runs" / "r"
        (run / "checkpoints" / "final").mkdir(parents=True)
        (run / "artifacts.json").write_text(json.dumps({"artifacts": [
            {"artifact_id": "a9", "kind": "predictions",
             "relpath": "checkpoints/final/results.parquet"},
        ]}))

        class FakeStore:
            async def download(self, artifact_id: str, target: Path) -> None:
                assert artifact_id == "a9"
                Path(target).write_bytes(b"pq")

        monkeypatch.setattr(artifacts_mod, "BackendArtifactStore", FakeStore)
        target = run / "checkpoints" / "final" / "results.parquet"
        assert await _fetch_run_artifact(target) is True
        assert target.read_bytes() == b"pq"
        # Unknown relpath → False, no crash.
        assert await _fetch_run_artifact(run / "nope.parquet") is False


# ---------------------------------------------------------------------------
# Inference budget
# ---------------------------------------------------------------------------


class TestInferenceBudget:
    def test_param_count_parsing(self) -> None:
        from lqh.models import model_param_count

        assert model_param_count("lfm2.5-350m") == 350e6
        assert model_param_count("LiquidAI/LFM2.5-1.2B-Instruct") == 1.2e9
        assert model_param_count("LiquidAI/LFM2.5-8B-A1B") == 8e9
        assert model_param_count("lfm2-24b-a2b") == 24e9
        assert model_param_count("LiquidAI/LFM2.5-VL-450M") == 450e6
        assert model_param_count("runs/sft_v1/model-lora") is None

    def test_parse_budget_line(self) -> None:
        from lqh.models import parse_inference_budget

        assert parse_inference_budget(None) == ("auto", None)
        assert parse_inference_budget("# Spec\nno budget here") == ("auto", None)
        spec = "## Inference Budget\n- **Budget**: auto\n"
        assert parse_inference_budget(spec) == ("auto", None)
        spec = "## Inference Budget\n- **Budget**: pinned:lfm2.5-1.2b-instruct\n"
        assert parse_inference_budget(spec) == ("pinned", "lfm2.5-1.2b-instruct")
        spec = "## Inference Budget\n- **Budget**: max:1.2B   # on-device\n"
        assert parse_inference_budget(spec) == ("max", "1.2B")

    def test_check_budget(self) -> None:
        from lqh.models import check_budget_for_model

        cap = "## Inference Budget\n- **Budget**: max:1.2B\n"
        assert check_budget_for_model(cap, "lfm2.5-350m") is None
        assert check_budget_for_model(cap, "LiquidAI/LFM2.5-1.2B-Instruct") is None
        assert check_budget_for_model(cap, "lfm2.5-2.6b") is not None

        pin = "## Inference Budget\n- **Budget**: pinned:lfm2.5-1.2b-instruct\n"
        # Short id and HF id both count as the pinned model.
        assert check_budget_for_model(pin, "LiquidAI/LFM2.5-1.2B-Instruct") is None
        assert check_budget_for_model(pin, "lfm2.5-1.2b-instruct") is None
        assert check_budget_for_model(pin, "lfm2.5-350m") is not None

        auto = "## Inference Budget\n- **Budget**: auto\n"
        assert check_budget_for_model(auto, "lfm2-24b-a2b") is None

    def test_parser_scoped_to_inference_budget_section(self) -> None:
        from lqh.models import parse_inference_budget

        # A Budget line OUTSIDE the section (e.g. a cost budget) is ignored.
        spec = "## Project Budget\n- **Budget**: max:1.2B\n\n## Overview\n"
        assert parse_inference_budget(spec) == ("auto", None)
        # Only the section's line counts, wherever the section sits.
        spec = (
            "## Project Budget\n- **Budget**: $500\n\n"
            "## Inference Budget\n- **Budget**: max:1.2B\n"
        )
        assert parse_inference_budget(spec) == ("max", "1.2B")

    def test_malformed_budget_fails_closed(self) -> None:
        from lqh.models import check_budget_for_model, parse_inference_budget

        spec = "## Inference Budget\n- **Budget**: about 1B or so\n"
        assert parse_inference_budget(spec) == ("invalid", "about 1B or so")
        violation = check_budget_for_model(spec, "lfm2.5-350m")
        assert violation is not None and "unparsable" in violation

        # A cap with no parsable size is unenforceable → also a violation.
        cap_spec = "## Inference Budget\n- **Budget**: max:small\n"
        violation = check_budget_for_model(cap_spec, "lfm2.5-350m")
        assert violation is not None and "max:small" in violation

    def _write_checkpoint(self, project_dir: Path, base_model: str) -> str:
        """Create runs/sft_001/model-lora with lineage pointing at *base_model*;
        returns the project-relative checkpoint path."""
        ckpt = project_dir / "runs" / "sft_001" / "model-lora"
        ckpt.mkdir(parents=True)
        (ckpt / "lineage.json").write_text(json.dumps({
            "artifact_kind": "checkpoint", "base_model": base_model,
        }))
        return "runs/sft_001/model-lora"

    def test_pinned_budget_allows_checkpoint_continuation(
        self, tmp_path: Path
    ) -> None:
        """DPO continuation from a checkpoint OF the pinned model must pass —
        the guard resolves the lineage, not the path string."""
        from lqh.models import check_budget_for_model
        from lqh.tools.handlers import _budget_base_model

        ckpt_path = self._write_checkpoint(tmp_path, "lfm2.5-1.2b-instruct")
        resolved = _budget_base_model(tmp_path, ckpt_path)
        assert resolved == "lfm2.5-1.2b-instruct"
        pin = "## Inference Budget\n- **Budget**: pinned:lfm2.5-1.2b-instruct\n"
        assert check_budget_for_model(pin, resolved) is None

    def test_max_budget_sees_through_checkpoint(self, tmp_path: Path) -> None:
        """A size cap must resolve a checkpoint path to its real base size —
        a path with no size token must not slip past max:."""
        from lqh.models import check_budget_for_model
        from lqh.tools.handlers import _budget_base_model

        ckpt_path = self._write_checkpoint(tmp_path, "lfm2.5-2.6b")
        resolved = _budget_base_model(tmp_path, ckpt_path)
        assert resolved == "lfm2.5-2.6b"
        cap = "## Inference Budget\n- **Budget**: max:1.2B\n"
        assert check_budget_for_model(cap, resolved) is not None

    def test_lineage_falls_back_to_run_config(self, tmp_path: Path) -> None:
        """No lineage.json/adapter_config.json → the owning run's (sweep-
        wrapped) config.json supplies the base model."""
        from lqh.tools.handlers import _budget_base_model

        ckpt = tmp_path / "runs" / "sft_002" / "model"
        ckpt.mkdir(parents=True)
        (tmp_path / "runs" / "sft_002" / "config.json").write_text(json.dumps({
            "type": "sweep",
            "base_config": {"type": "sft", "base_model": "lfm2.5-350m"},
        }))
        assert _budget_base_model(tmp_path, "runs/sft_002/model") == "lfm2.5-350m"
        # Non-paths and unresolvable paths pass through unchanged.
        assert _budget_base_model(tmp_path, "lfm2.5-350m") == "lfm2.5-350m"
        assert _budget_base_model(tmp_path, "runs/nope/model") == "runs/nope/model"

    async def test_start_training_rejects_over_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lqh.tools import handlers
        from lqh.tools.handlers import handle_start_training

        # Pin the compute path so the guard is reached regardless of this
        # machine's GPU/remote configuration.
        monkeypatch.setattr(handlers, "_compute_pick_options", lambda p: None)
        monkeypatch.setattr(handlers, "_resolve_compute_target", lambda p: "cloud")

        (tmp_path / "SPEC.md").write_text(
            "# Spec\n\n## Inference Budget\n\n- **Budget**: max:1.2B\n",
            encoding="utf-8",
        )
        result = await handle_start_training(
            tmp_path,
            type="sft",
            base_model="lfm2.5-2.6b",
            dataset="datasets/train",
            eval_dataset="datasets/eval",
            scorer="evals/scorers/s.md",
        )
        assert result.error_kind == "permission"
        assert "override_budget" in result.content

        # An in-budget model passes the guard (fails later on the missing
        # dataset instead, proving the guard was the only blocker).
        result2 = await handle_start_training(
            tmp_path,
            type="sft",
            base_model="lfm2.5-350m",
            dataset="datasets/train",
            eval_dataset="datasets/eval",
            scorer="evals/scorers/s.md",
        )
        assert result2.error_kind == "validation"


# ---------------------------------------------------------------------------
# Skill registry
# ---------------------------------------------------------------------------


class TestFailureAnalysisSkill:
    def test_registered_and_aliases_resolve(self) -> None:
        from lqh.skills import list_available_skills, load_skill_content

        entry = next(
            s for s in list_available_skills() if s["name"] == "failure_analysis"
        )
        assert entry["command"] == "/improve"
        content = load_skill_content("failure_analysis")
        assert content.startswith("# Skill: Failure Analysis")
        for alias in ("improve", "/improve", "failures", "failure-analysis"):
            assert load_skill_content(alias) == content

    def test_tui_command_wired(self) -> None:
        from lqh.tui.commands import COMMANDS

        assert any(c.name == "/improve" for c in COMMANDS)

    def test_probe_purpose_accepted(self) -> None:
        from lqh.manifest import PURPOSES

        assert "probe" in PURPOSES
