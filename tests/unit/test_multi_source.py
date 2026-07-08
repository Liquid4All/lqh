"""Tests for multi-source dataset support (SFT/DPO training, sweep, eval).

Covers the shared normalizer + plural loaders, the remote-shipping resolvers
that must expand list-valued dataset keys, the per-source judge scoring with a
macro-average headline, and the handler validation (distinctness across lists,
duplicate eval rejection, canonical config build).

All tests here are pure-Python (no GPU, no live API) — the one scoring test
patches ``run_scoring`` so it never hits the judge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# normalize_sources
# ---------------------------------------------------------------------------


class TestNormalizeSources:
    def test_bare_string_is_one_source(self) -> None:
        from lqh.train.data_utils import normalize_sources

        out = normalize_sources("datasets/a/data.parquet", allow_repeat=True)
        assert out == [{"path": "datasets/a/data.parquet", "repeat": 1, "source": "a"}]

    def test_list_of_strings(self) -> None:
        from lqh.train.data_utils import normalize_sources

        out = normalize_sources(
            ["datasets/a/data.parquet", "datasets/b/data.parquet"], allow_repeat=True
        )
        assert [e["source"] for e in out] == ["a", "b"]
        assert all(e["repeat"] == 1 for e in out)

    def test_object_with_repeat(self) -> None:
        from lqh.train.data_utils import normalize_sources

        out = normalize_sources(
            [{"path": "datasets/a/data.parquet", "repeat": 3}], allow_repeat=True
        )
        assert out[0]["repeat"] == 3

    def test_eval_rejects_explicit_repeat(self) -> None:
        """Eval is unweighted — an explicit repeat on an eval source is a
        loud error, not silently dropped."""
        from lqh.train.data_utils import normalize_sources

        with pytest.raises(ValueError, match="not allowed on eval"):
            normalize_sources(
                [{"path": "datasets/a/data.parquet", "repeat": 5}], allow_repeat=False
            )

    def test_eval_string_sources_have_repeat_one(self) -> None:
        from lqh.train.data_utils import normalize_sources

        out = normalize_sources(
            ["datasets/a/data.parquet"], allow_repeat=False
        )
        assert out[0]["repeat"] == 1

    def test_label_collision_disambiguated(self) -> None:
        from lqh.train.data_utils import normalize_sources

        # Two different parent dirs that share a name would collide; here we
        # force a collision via identical dir names under different roots.
        out = normalize_sources(
            ["x/eval/data.parquet", "y/eval/data.parquet"], allow_repeat=False
        )
        assert [e["source"] for e in out] == ["eval", "eval_2"]

    def test_invalid_repeat_rejected(self) -> None:
        from lqh.train.data_utils import normalize_sources

        with pytest.raises(ValueError):
            normalize_sources(
                [{"path": "datasets/a/data.parquet", "repeat": 0}], allow_repeat=True
            )

    def test_missing_path_rejected(self) -> None:
        from lqh.train.data_utils import normalize_sources

        with pytest.raises(ValueError):
            normalize_sources([{"repeat": 2}], allow_repeat=True)


# ---------------------------------------------------------------------------
# load_chatml_datasets / load_eval_sources
# ---------------------------------------------------------------------------


class TestPluralLoaders:
    def test_concat_count_is_sum(
        self,
        tmp_path: Path,
        write_chatml_parquet: Callable[..., Path],
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.data_utils import load_chatml_datasets

        a = write_chatml_parquet(tmp_path / "a" / "data.parquet", sample_conversations(4))
        b = write_chatml_parquet(tmp_path / "b" / "data.parquet", sample_conversations(6))
        out = load_chatml_datasets([str(a), str(b)])
        assert len(out) == 10

    def test_repeat_multiplies(
        self,
        tmp_path: Path,
        write_chatml_parquet: Callable[..., Path],
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.data_utils import load_chatml_datasets

        a = write_chatml_parquet(tmp_path / "a" / "data.parquet", sample_conversations(3))
        out = load_chatml_datasets([{"path": str(a), "repeat": 4}])
        assert len(out) == 12

    def test_single_string_matches_legacy(
        self,
        tmp_path: Path,
        write_chatml_parquet: Callable[..., Path],
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.data_utils import load_chatml_dataset, load_chatml_datasets

        a = write_chatml_parquet(tmp_path / "a" / "data.parquet", sample_conversations(5))
        assert load_chatml_datasets(str(a)) == load_chatml_dataset(str(a))

    def test_eval_sources_kept_distinct(
        self,
        tmp_path: Path,
        write_chatml_parquet: Callable[..., Path],
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.data_utils import load_eval_sources

        a = write_chatml_parquet(
            tmp_path / "type_a" / "data.parquet", sample_conversations(2)
        )
        b = write_chatml_parquet(
            tmp_path / "type_b" / "data.parquet", sample_conversations(3)
        )
        out = load_eval_sources([str(a), str(b)])
        assert [label for label, _ in out] == ["type_a", "type_b"]
        assert [len(c) for _, c in out] == [2, 3]

    def test_eval_sources_with_tools_flattens_with_labels(
        self,
        tmp_path: Path,
        write_chatml_parquet: Callable[..., Path],
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.train.data_utils import load_eval_sources_with_tools

        a = write_chatml_parquet(
            tmp_path / "type_a" / "data.parquet", sample_conversations(2)
        )
        b = write_chatml_parquet(
            tmp_path / "type_b" / "data.parquet", sample_conversations(3)
        )
        convos, tools, sources = load_eval_sources_with_tools([str(a), str(b)])
        assert len(convos) == 5
        assert sources == ["type_a", "type_a", "type_b", "type_b", "type_b"]
        assert len(tools) == 5


# ---------------------------------------------------------------------------
# Remote shipping: iter_path_values, resolve_manifest, ssh rewrite
# ---------------------------------------------------------------------------


class TestRemoteResolvers:
    def test_iter_path_values_forms(self) -> None:
        from lqh.sync import iter_path_values

        assert iter_path_values("a/b.parquet") == ["a/b.parquet"]
        assert iter_path_values(["a", "b"]) == ["a", "b"]
        assert iter_path_values(
            [{"path": "a", "repeat": 2}, {"path": "b"}]
        ) == ["a", "b"]
        assert iter_path_values(None) == []

    def test_resolve_manifest_expands_list_and_dedups(
        self,
        tmp_path: Path,
        write_chatml_parquet: Callable[..., Path],
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.sync import resolve_manifest

        write_chatml_parquet(tmp_path / "a" / "data.parquet", sample_conversations(1))
        write_chatml_parquet(tmp_path / "b" / "data.parquet", sample_conversations(1))
        config = {
            "dataset": [
                {"path": "a/data.parquet", "repeat": 2},
                {"path": "b/data.parquet"},
                # duplicate of the first source — must be deduped
                "a/data.parquet",
            ],
            "manifest": ["dataset"],
        }
        paths = resolve_manifest(config, tmp_path)
        names = sorted(p.parent.name for p in paths)
        assert names == ["a", "b"]
        assert len(paths) == 2

    def test_resolve_manifest_legacy_string(
        self,
        tmp_path: Path,
        write_chatml_parquet: Callable[..., Path],
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.sync import resolve_manifest

        write_chatml_parquet(tmp_path / "a" / "data.parquet", sample_conversations(1))
        config = {"dataset": "a/data.parquet", "manifest": ["dataset"]}
        paths = resolve_manifest(config, tmp_path)
        assert [p.name for p in paths] == ["data.parquet"]

    def test_ssh_resolve_local_manifest_expands_sources(
        self,
        tmp_path: Path,
        write_chatml_parquet: Callable[..., Path],
        sample_conversations: Callable[[int], list],
    ) -> None:
        from lqh.remote.ssh_direct import _resolve_local_manifest

        write_chatml_parquet(tmp_path / "a" / "data.parquet", sample_conversations(1))
        write_chatml_parquet(tmp_path / "b" / "data.parquet", sample_conversations(1))
        config = {
            "dataset": ["a/data.parquet", {"path": "b/data.parquet", "repeat": 2}],
            "eval_dataset": "a/data.parquet",  # overlaps train → deduped
        }
        paths = _resolve_local_manifest(config, tmp_path)
        # Each .parquet maps to its parent dir; a appears once after dedup.
        names = sorted(p.name for p in paths)
        assert names == ["a", "b"]

    def test_ssh_rewrite_list_dataset_paths(self) -> None:
        from lqh.remote.ssh_direct import _rewrite_config_paths

        project = Path("/home/u/proj")
        config = {
            "base_model": "/home/u/proj/runs/x/model",
            "dataset": [
                "/home/u/proj/datasets/a/data.parquet",
                {"path": "/home/u/proj/datasets/b/data.parquet", "repeat": 3},
            ],
            "eval_dataset": "/home/u/proj/datasets/e/data.parquet",
        }
        out = _rewrite_config_paths(config, project, "/remote/root")
        assert out["base_model"] == "/remote/root/runs/x/model"
        assert out["dataset"][0] == "/remote/root/datasets/a/data.parquet"
        assert out["dataset"][1] == {
            "path": "/remote/root/datasets/b/data.parquet",
            "repeat": 3,
        }
        assert out["eval_dataset"] == "/remote/root/datasets/e/data.parquet"


# ---------------------------------------------------------------------------
# Per-source scoring → macro-average headline
# ---------------------------------------------------------------------------


class TestPerSourceScoring:
    def test_score_stats(self) -> None:
        from lqh.scoring import _score_stats

        s = _score_stats([8.0, 8.0, 8.0])
        assert s["mean"] == 8.0 and s["median"] == 8.0 and s["std"] == 0.0
        assert _score_stats([])["mean"] == 0.0

    def test_source_map(
        self,
        tmp_path: Path,
    ) -> None:
        from lqh.scoring import _source_map

        preds = tmp_path / "predictions.parquet"
        pq.write_table(
            pa.table(
                {
                    "sample_index": [0, 1, 2],
                    "messages": ["a", "b", "c"],
                    "source": ["x", "x", "y"],
                }
            ),
            preds,
        )
        assert _source_map(preds) == {0: "x", 1: "x", 2: "y"}

    def test_source_map_absent_column(self, tmp_path: Path) -> None:
        from lqh.scoring import _source_map

        preds = tmp_path / "predictions.parquet"
        pq.write_table(pa.table({"sample_index": [0], "messages": ["a"]}), preds)
        assert _source_map(preds) == {}

    async def test_macro_average_headline(self, tmp_path: Path) -> None:
        """Macro headline == mean of per-source means, regardless of size."""
        from lqh.scoring import ScoringResult, score_predictions_by_source

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Predictions: source x has 3 rows, source y has 1 row (imbalanced).
        preds = tmp_path / "predictions.parquet"
        pq.write_table(
            pa.table(
                {
                    "sample_index": [0, 1, 2, 3],
                    "messages": ["a", "b", "c", "d"],
                    "source": ["x", "x", "x", "y"],
                }
            ),
            preds,
        )

        # Patched run_scoring writes results.parquet (x avg 8, y avg 4) and
        # returns the by-count ScoringResult so the helper can compute both
        # the weighted and macro means.
        async def fake_run_scoring(*, dataset_path, scorer_path, output_dir, client, **kw):
            pq.write_table(
                pa.table(
                    {
                        "sample_index": [0, 1, 2, 3],
                        "messages": ["a", "b", "c", "d"],
                        "score": [8.0, 8.0, 8.0, 4.0],
                        "reasoning": ["ok", "ok", "ok", "ok"],
                    }
                ),
                output_dir / "results.parquet",
            )
            return ScoringResult(
                total=4, scored=4, failed=0,
                mean_score=7.0, median_score=8.0, output_dir=output_dir,
            )

        with patch("lqh.scoring.run_scoring", side_effect=fake_run_scoring):
            payload = await score_predictions_by_source(
                predictions_path=preds,
                scorer_path=tmp_path / "scorer.md",
                output_dir=out_dir,
                client=object(),
            )

        # Macro: mean(8, 4) = 6.0; weighted (by count): 7.0.
        assert payload["scores"]["mean"] == 6.0
        assert payload["scores_weighted_mean"] == 7.0
        assert payload["per_source"]["x"]["scores"]["mean"] == 8.0
        assert payload["per_source"]["y"]["scores"]["mean"] == 4.0
        assert payload["per_source"]["x"]["num_scored"] == 3
        # eval_result.json was written with the same payload.
        on_disk = json.loads((out_dir / "eval_result.json").read_text())
        assert on_disk["scores"]["mean"] == 6.0

    async def test_failed_source_stays_visible(self, tmp_path: Path) -> None:
        """A source whose samples all fail to score is still reported (with
        num_scored=0) and excluded from the macro average."""
        from lqh.scoring import (
            ScoringResult,
            is_scoring_error,
            score_predictions_by_source,
        )

        # Sanity: the marker we use below is recognised as a scoring error.
        err = "[Scoring error] judge unreachable"
        assert is_scoring_error(err)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        preds = tmp_path / "predictions.parquet"
        pq.write_table(
            pa.table(
                {
                    "sample_index": [0, 1, 2],
                    "messages": ["a", "b", "c"],
                    "source": ["good", "good", "bad"],
                }
            ),
            preds,
        )

        async def fake_run_scoring(*, dataset_path, scorer_path, output_dir, client, **kw):
            pq.write_table(
                pa.table(
                    {
                        "sample_index": [0, 1, 2],
                        "messages": ["a", "b", "c"],
                        "score": [8.0, 8.0, 0.0],
                        "reasoning": ["ok", "ok", err],
                    }
                ),
                output_dir / "results.parquet",
            )
            return ScoringResult(
                total=3, scored=2, failed=1,
                mean_score=8.0, median_score=8.0, output_dir=output_dir,
            )

        with patch("lqh.scoring.run_scoring", side_effect=fake_run_scoring):
            payload = await score_predictions_by_source(
                predictions_path=preds,
                scorer_path=tmp_path / "scorer.md",
                output_dir=out_dir,
                client=object(),
            )

        # 'bad' is visible with zero scored samples, and the macro headline
        # is the mean over 'good' only (8.0), not dragged down to 0.
        assert payload["per_source"]["bad"]["num_scored"] == 0
        assert payload["scores"]["mean"] == 8.0

    async def test_single_source_equals_legacy(self, tmp_path: Path) -> None:
        """With one source the macro headline equals that source's mean."""
        from lqh.scoring import ScoringResult, score_predictions_by_source

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        preds = tmp_path / "predictions.parquet"
        pq.write_table(
            pa.table(
                {
                    "sample_index": [0, 1],
                    "messages": ["a", "b"],
                    "source": ["only", "only"],
                }
            ),
            preds,
        )

        async def fake_run_scoring(*, dataset_path, scorer_path, output_dir, client, **kw):
            pq.write_table(
                pa.table(
                    {
                        "sample_index": [0, 1],
                        "messages": ["a", "b"],
                        "score": [7.0, 9.0],
                        "reasoning": ["ok", "ok"],
                    }
                ),
                output_dir / "results.parquet",
            )
            return ScoringResult(
                total=2, scored=2, failed=0,
                mean_score=8.0, median_score=9.0, output_dir=output_dir,
            )

        with patch("lqh.scoring.run_scoring", side_effect=fake_run_scoring):
            payload = await score_predictions_by_source(
                predictions_path=preds,
                scorer_path=tmp_path / "scorer.md",
                output_dir=out_dir,
                client=object(),
            )
        assert payload["scores"]["mean"] == 8.0
        assert payload["scores_weighted_mean"] == 8.0
        assert list(payload["per_source"]) == ["only"]


# ---------------------------------------------------------------------------
# Handler: multi-source validation + canonical config build
# ---------------------------------------------------------------------------


@pytest.fixture
def ms_workspace(
    tmp_path: Path,
    write_chatml_parquet: Callable[..., Path],
    sample_conversations: Callable[[int], list],
) -> Path:
    """Project with type_a/type_b train + eval datasets, scorer, allow-all."""
    for name, n in [("type_a", 5), ("type_b", 6)]:
        write_chatml_parquet(
            tmp_path / "datasets" / name / "data.parquet", sample_conversations(n)
        )
    for name, n in [("type_a_eval", 3), ("type_b_eval", 4)]:
        write_chatml_parquet(
            tmp_path / "datasets" / name / "data.parquet", sample_conversations(n)
        )
    scorer_dir = tmp_path / "evals" / "scorers"
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "test.md").write_text("Score 1-10.")
    lqh_dir = tmp_path / ".lqh"
    lqh_dir.mkdir(parents=True)
    (lqh_dir / "permissions.json").write_text(
        json.dumps({"project_allow_all": True, "training_allow_all": True})
    )
    return tmp_path


@pytest.fixture
def _no_local_gpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("lqh.tools.handlers._local_gpu_available", lambda: False)


@pytest.fixture
def stub_torch():
    with patch("lqh.tools.handlers._check_torch_available", return_value=None):
        yield


class TestHandlerMultiSource:
    async def test_multi_source_config_built(
        self, ms_workspace: Path, stub_torch, _no_local_gpu,
    ) -> None:
        from lqh.tools import handlers

        captured: dict[str, Any] = {}

        async def fake_exec(project_dir, run_dir, config, run_name, *a, **kw):
            captured["config"] = config
            from lqh.tools.handlers import ToolResult

            return ToolResult(content="started")

        with (
            # Route through the remote path (skips the local CUDA probe) and
            # intercept the launch so we can inspect the config that was built.
            patch.object(handlers, "_compute_pick_options", return_value=None),
            patch.object(handlers, "_resolve_compute_target", return_value="cloud"),
            patch.object(handlers, "_execute_start_training_remote", side_effect=fake_exec),
        ):
            await handlers.handle_start_training(
                ms_workspace,
                type="sft",
                base_model="m",
                dataset=["datasets/type_a", {"path": "datasets/type_b", "repeat": 3}],
                eval_dataset=["datasets/type_a_eval", "datasets/type_b_eval"],
                scorer="evals/scorers/test.md",
                enable_sweep=False,
            )

        cfg = captured["config"]
        assert cfg["dataset"] == [
            {"path": "datasets/type_a/data.parquet", "repeat": 1, "source": "type_a"},
            {"path": "datasets/type_b/data.parquet", "repeat": 3, "source": "type_b"},
        ]
        assert cfg["eval_dataset"] == [
            {"path": "datasets/type_a_eval/data.parquet", "source": "type_a_eval"},
            {"path": "datasets/type_b_eval/data.parquet", "source": "type_b_eval"},
        ]
        assert "eval_dataset" in cfg["manifest"]

    async def test_single_source_config_stays_legacy_string(
        self, ms_workspace: Path, stub_torch, _no_local_gpu,
    ) -> None:
        """A single source with no repeat must serialize as the legacy bare
        path string (not a list) so existing configs/bundles/tests are
        unaffected."""
        from lqh.tools import handlers

        captured: dict[str, Any] = {}

        async def fake_exec(project_dir, run_dir, config, run_name, *a, **kw):
            captured["config"] = config
            from lqh.tools.handlers import ToolResult

            return ToolResult(content="started")

        with (
            patch.object(handlers, "_compute_pick_options", return_value=None),
            patch.object(handlers, "_resolve_compute_target", return_value="cloud"),
            patch.object(handlers, "_execute_start_training_remote", side_effect=fake_exec),
        ):
            await handlers.handle_start_training(
                ms_workspace,
                type="sft",
                base_model="m",
                dataset="datasets/type_a",
                eval_dataset="datasets/type_a_eval",
                scorer="evals/scorers/test.md",
                enable_sweep=False,
            )

        cfg = captured["config"]
        assert cfg["dataset"] == "datasets/type_a/data.parquet"
        assert cfg["eval_dataset"] == "datasets/type_a_eval/data.parquet"

    async def test_rejects_train_eval_overlap_in_lists(
        self, ms_workspace: Path, stub_torch, _no_local_gpu,
    ) -> None:
        from lqh.tools.handlers import handle_start_training

        result = await handle_start_training(
            ms_workspace,
            type="sft",
            base_model="m",
            dataset=["datasets/type_a", "datasets/type_b"],
            # type_a is also a training source → leak
            eval_dataset=["datasets/type_a", "datasets/type_a_eval"],
            scorer="evals/scorers/test.md",
        )
        assert "must be different from dataset" in result.content

    async def test_rejects_duplicate_eval_source(
        self, ms_workspace: Path, stub_torch, _no_local_gpu,
    ) -> None:
        from lqh.tools.handlers import handle_start_training

        result = await handle_start_training(
            ms_workspace,
            type="sft",
            base_model="m",
            dataset="datasets/type_a",
            eval_dataset=["datasets/type_a_eval", "datasets/type_a_eval"],
            scorer="evals/scorers/test.md",
        )
        assert "same source twice" in result.content

    async def test_per_source_not_found_message(
        self, ms_workspace: Path, stub_torch, _no_local_gpu,
    ) -> None:
        from lqh.tools.handlers import handle_start_training

        result = await handle_start_training(
            ms_workspace,
            type="sft",
            base_model="m",
            dataset=["datasets/type_a", "datasets/missing"],
            eval_dataset="datasets/type_a_eval",
            scorer="evals/scorers/test.md",
        )
        assert "dataset source 2 not found" in result.content


class TestLocalEvalMultiSource:
    async def test_local_eval_accepts_list(
        self, ms_workspace: Path, stub_torch,
    ) -> None:
        """start_local_eval builds an infer config whose dataset is the
        multi-source list, scored per-source downstream."""
        from lqh.tools import handlers

        captured: dict[str, Any] = {}

        def fake_start(self, run_dir, config, *, module, project_dir):
            captured["config"] = config
            return 4321

        # A model dir to satisfy the existence check.
        (ms_workspace / "runs" / "m" / "model").mkdir(parents=True)

        with patch(
            "lqh.subprocess_manager.SubprocessManager.start", new=fake_start
        ):
            result = await handlers.handle_start_local_eval(
                ms_workspace,
                model_path="runs/m/model",
                dataset=["datasets/type_a_eval", "datasets/type_b_eval"],
                scorer="evals/scorers/test.md",
            )

        assert "started" in result.content.lower()
        cfg = captured["config"]
        assert cfg["dataset"] == [
            {"path": "datasets/type_a_eval/data.parquet", "source": "type_a_eval"},
            {"path": "datasets/type_b_eval/data.parquet", "source": "type_b_eval"},
        ]

    async def test_local_eval_remote_accepts_list(self, ms_workspace: Path) -> None:
        """The SSH-backed branch builds the same multi-source list config as
        the local branch (and ships every source via the manifest)."""
        from types import SimpleNamespace

        from lqh.tools import handlers

        captured: dict[str, Any] = {}

        class FakeBackend:
            def __init__(self, remote_config, project_dir):
                pass

            async def submit_run(self, run_dir, config, *, module):
                captured["config"] = config
                return "job-123"

        fake_remote = SimpleNamespace(type="ssh", hostname="gpu-box")

        with (
            patch("lqh.remote.config.get_remote", return_value=fake_remote),
            patch("lqh.remote.ssh_direct.SSHDirectBackend", FakeBackend),
        ):
            result = await handlers._start_local_eval_remote(
                ms_workspace,
                "runs/m/model",
                ["datasets/type_a_eval", "datasets/type_b_eval"],
                "evals/scorers/test.md",
                None,
                "toka",
            )

        assert "Remote eval started" in result.content
        assert captured["config"]["dataset"] == [
            {"path": "datasets/type_a_eval/data.parquet", "source": "type_a_eval"},
            {"path": "datasets/type_b_eval/data.parquet", "source": "type_b_eval"},
        ]

    async def test_local_eval_single_source_string(
        self, ms_workspace: Path, stub_torch,
    ) -> None:
        from lqh.tools import handlers

        captured: dict[str, Any] = {}

        def fake_start(self, run_dir, config, *, module, project_dir):
            captured["config"] = config
            return 4321

        (ms_workspace / "runs" / "m" / "model").mkdir(parents=True)
        with patch(
            "lqh.subprocess_manager.SubprocessManager.start", new=fake_start
        ):
            await handlers.handle_start_local_eval(
                ms_workspace,
                model_path="runs/m/model",
                dataset="datasets/type_a_eval",
                scorer="evals/scorers/test.md",
            )
        assert captured["config"]["dataset"] == "datasets/type_a_eval/data.parquet"
