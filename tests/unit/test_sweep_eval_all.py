"""Tests for the ``eval_all`` sweep mode.

``eval_all`` judge-scores EVERY config in the grid instead of only the winner.
The hp_defaults calibration study needs it: to know how much a default costs
you, it must compare each config against the best score anything achieved in
that cell, which means every config needs a real metric.

Four guarantees pinned here:

  1. **Containment.** A per-config eval writes only inside
     ``sweep_<id>/eval_of_config/``. A ``predictions.parquet`` at the run root
     is what the publisher classifies as THE run's eval result — one config's
     eval must never claim that slot.
  2. **The row is the transport.** Judge scores land on the config's row, which
     ``sweep_summary.json`` and ``runs.jsonl`` already carry back from a cloud
     sandbox. Nothing new has to be published.
  3. **Off by default.** The flag multiplies judge cost by the grid size, so
     an ordinary sweep must be untouched.
  4. **Honest absence.** A config that failed to train or failed to score
     carries no ``judge_mean`` at all, rather than a zero that would look like
     a real (terrible) score and win an argmin.

No real inference or judging runs — the infer subprocess and the inline scorer
are both faked so the control flow is exercised in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lqh.train import sweep


def _fake_infer(rc: int = 0):
    """Emulate ``python -m lqh.infer`` writing its outputs next to its config."""
    def runner(argv, *, stdout, stderr, check):
        if rc == 0:
            eval_dir = Path(argv[-1]).parent
            (eval_dir / "predictions.parquet").write_bytes(b"PARQUET")
            (eval_dir / "eval_request.json").write_text('{"status": "ready"}')

        class _Result:
            returncode = rc
        return _Result()
    return runner


def _fake_scorer(mean: float | None, *, num_scored: int = 40):
    """Emulate the inline judge, writing eval_result.json where it's told."""
    def scorer(run_dir: Path, config: dict):
        if mean is None:
            return None
        payload = {
            "num_scored": num_scored,
            "num_failed": 0,
            "scores": {"mean": mean, "median": mean, "std": 1.25},
        }
        (run_dir / "eval_result.json").write_text(json.dumps(payload))
        return payload
    return scorer


def _install_fakes(monkeypatch, *, mean: float | None = 7.5, rc: int = 0):
    monkeypatch.setattr(sweep.subprocess, "run", _fake_infer(rc))
    monkeypatch.setattr(
        "lqh.train.cloud_score.score_run_eval_inline", _fake_scorer(mean),
    )


def _make_config_dir(run_dir: Path, config_id: str) -> Path:
    """A finished sweep child: an adapter plus some checkpoint clutter."""
    sub = run_dir / f"sweep_{config_id}"
    (sub / "model-lora").mkdir(parents=True)
    (sub / "checkpoints" / "step_50").mkdir(parents=True)
    (sub / "checkpoints" / "step_50" / "weights.bin").write_bytes(b"x" * 100)
    return sub


# ---------------------------------------------------------------------
# Containment — the point that keeps eval_all from corrupting the run
# ---------------------------------------------------------------------


def test_per_config_eval_stays_inside_its_own_subdir(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    sub = _make_config_dir(run_dir, "lr2e-5_e2")
    _install_fakes(monkeypatch)

    sweep._run_eval_for_config(sub, {"eval_dataset": "datasets/heldout"}, "lr2e-5_e2")

    assert (sub / "eval_of_config" / "predictions.parquet").exists()
    assert (sub / "eval_of_config" / "eval_result.json").exists()
    # Neither the sweep root nor the config root may gain the artifacts that
    # the publisher reads as the run's own eval result.
    for name in ("predictions.parquet", "eval_request.json", "eval_result.json"):
        assert not (run_dir / name).exists()
        assert not (sub / name).exists()


def test_per_config_eval_does_not_report_into_parent_progress(
    tmp_path: Path, monkeypatch,
):
    """N configs cannot each own the parent's final-inference progress slice."""
    run_dir = tmp_path / "run"
    sub = _make_config_dir(run_dir, "c1")
    _install_fakes(monkeypatch)

    sweep._run_eval_for_config(sub, {"eval_dataset": "datasets/heldout"}, "c1")

    cfg = json.loads((sub / "eval_of_config" / "config.json").read_text())
    assert "progress_run_dir" not in cfg
    assert "progress_start" not in cfg
    assert not (run_dir / "progress.jsonl").exists()


# ---------------------------------------------------------------------
# The row is the transport
# ---------------------------------------------------------------------


def test_judge_fields_land_on_the_row(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    sub = _make_config_dir(run_dir, "c1")
    _install_fakes(monkeypatch, mean=8.25)

    fields = sweep._run_eval_for_config(sub, {"eval_dataset": "datasets/e"}, "c1")

    assert fields["judge_mean"] == 8.25
    assert fields["judge_median"] == 8.25
    assert fields["judge_std"] == 1.25
    assert fields["judge_num_scored"] == 40


def test_failed_scoring_leaves_no_judge_mean(tmp_path: Path, monkeypatch):
    """A zero here would look like a real score and win an argmin."""
    run_dir = tmp_path / "run"
    sub = _make_config_dir(run_dir, "c1")
    _install_fakes(monkeypatch, mean=None)

    fields = sweep._run_eval_for_config(sub, {"eval_dataset": "datasets/e"}, "c1")

    assert "judge_mean" not in fields
    assert "judge_skipped" in fields


def test_failed_inference_leaves_no_judge_mean(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    sub = _make_config_dir(run_dir, "c1")
    _install_fakes(monkeypatch, rc=9)

    fields = sweep._run_eval_for_config(sub, {"eval_dataset": "datasets/e"}, "c1")

    assert "judge_mean" not in fields
    assert "9" in fields["judge_skipped"]


def test_config_without_a_model_skips_cleanly(tmp_path: Path, monkeypatch):
    """A crashed child leaves no model dir; eval must not invent one."""
    sub = tmp_path / "run" / "sweep_c1"
    sub.mkdir(parents=True)
    monkeypatch.setattr(
        sweep.subprocess, "run",
        lambda *a, **k: pytest.fail("infer must not run without a model"),
    )

    fields = sweep._run_eval_for_config(sub, {"eval_dataset": "datasets/e"}, "c1")

    assert "judge_mean" not in fields
    assert "no model" in fields["judge_skipped"]


def test_judge_mean_from_summary_ignores_malformed_payloads():
    assert sweep._judge_mean_from_summary({}) == {}
    assert sweep._judge_mean_from_summary({"skipped": "nope"}) == {}
    assert sweep._judge_mean_from_summary({"score_summary": None}) == {}
    assert sweep._judge_mean_from_summary({"score_summary": {"scores": None}}) == {}
    assert sweep._judge_mean_from_summary(
        {"score_summary": {"scores": {"mean": "8.0"}}}
    ) == {}


# ---------------------------------------------------------------------
# Disk hygiene
# ---------------------------------------------------------------------


def test_checkpoints_are_dropped_but_the_model_is_kept(tmp_path: Path):
    """15 configs of HF checkpoints will fill a sandbox before the grid ends."""
    sub = _make_config_dir(tmp_path / "run", "c1")

    sweep._discard_child_checkpoints(sub)

    assert not (sub / "checkpoints").exists()
    assert (sub / "model-lora").exists()


def test_discarding_checkpoints_is_idempotent(tmp_path: Path):
    sub = tmp_path / "run" / "sweep_c1"
    sub.mkdir(parents=True)
    sweep._discard_child_checkpoints(sub)  # must not raise


# ---------------------------------------------------------------------
# best_epoch — is the epochs axis doing work checkpoint selection already did?
# ---------------------------------------------------------------------


def test_best_epoch_reads_the_argmin_of_eval_loss(tmp_path: Path):
    """sft.py trains with load_best_model_at_end, so the saved model may not
    be the last epoch. The study needs to know which epoch it actually was."""
    sub = tmp_path / "sweep_c1"
    sub.mkdir()
    (sub / "eval_history.json").write_text(json.dumps([
        {"loss": 1.9, "epoch": 0.5, "step": 10},          # train row, no eval
        {"eval_loss": 1.20, "epoch": 1.0, "step": 20},
        {"eval_loss": 0.95, "epoch": 2.0, "step": 40},    # best
        {"eval_loss": 1.05, "epoch": 3.0, "step": 60},
    ]))

    assert sweep._read_best_epoch(sub) == {"best_epoch": 2.0, "best_step": 40}


def test_best_epoch_absent_when_eval_never_fired(tmp_path: Path):
    sub = tmp_path / "sweep_c1"
    sub.mkdir()
    (sub / "eval_history.json").write_text(json.dumps([{"loss": 1.9, "step": 10}]))
    assert sweep._read_best_epoch(sub) == {}


def test_best_epoch_tolerates_missing_or_corrupt_history(tmp_path: Path):
    sub = tmp_path / "sweep_c1"
    sub.mkdir()
    assert sweep._read_best_epoch(sub) == {}
    (sub / "eval_history.json").write_text("{not json")
    assert sweep._read_best_epoch(sub) == {}


# ---------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------


def test_eval_all_defaults_off(tmp_path: Path, monkeypatch):
    """An ordinary sweep must not start paying judge cost per config."""
    calls: list[str] = []
    monkeypatch.setattr(
        sweep, "_run_eval_for_config",
        lambda sub, base, cid: calls.append(cid) or {},
    )
    monkeypatch.setattr(sweep, "_run_child", lambda sub_run_dir, cfg: 0)
    monkeypatch.setattr(sweep, "_read_sft_proxy", lambda sub: {"primary": 0.5})
    monkeypatch.setattr(sweep, "_materialize_best_model", lambda winner, run: None)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sweep.sweep_loop(run_dir, {
        "base_config": {
            "type": "sft",
            "base_model": "m",
            "dataset": "datasets/train",
            "eval_dataset": "datasets/eval",
        },
        "grid_override": [
            {"id": "a", "overrides": {"training": {"learning_rate": 1e-5}}},
            {"id": "b", "overrides": {"training": {"learning_rate": 1e-4}}},
        ],
        "eval_best": False,
    })

    assert calls == []


def test_eval_all_scores_every_config_and_stamps_the_summary(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setattr(
        sweep, "_run_eval_for_config",
        lambda sub, base, cid: {"judge_mean": 9.0 if cid == "b" else 6.0},
    )
    monkeypatch.setattr(sweep, "_run_child", lambda sub_run_dir, cfg: 0)
    # 'a' has the better proxy but the worse judge score — the study exists to
    # measure exactly this kind of disagreement, so both must be recorded.
    monkeypatch.setattr(
        sweep, "_read_sft_proxy",
        lambda sub: {"primary": 0.1 if sub.name.endswith("a") else 0.9},
    )
    monkeypatch.setattr(sweep, "_materialize_best_model", lambda winner, run: None)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sweep.sweep_loop(run_dir, {
        "base_config": {
            "type": "sft",
            "base_model": "m",
            "dataset": "datasets/train",
            "eval_dataset": "datasets/eval",
        },
        "grid_override": [
            {"id": "a", "overrides": {"training": {"learning_rate": 1e-5}}},
            {"id": "b", "overrides": {"training": {"learning_rate": 1e-4}}},
        ],
        "eval_all": True,
        "eval_best": False,
    })

    summary = json.loads((run_dir / "sweep_summary.json").read_text())
    judged = {r["config_id"]: r["judge_mean"] for r in summary["rows"]}
    assert judged == {"a": 6.0, "b": 9.0}
    # Winner selection still runs on the proxy; eval_all only observes.
    assert summary["winner"]["config_id"] == "a"

    # runs.jsonl is the resume ledger — a resumed sweep must not re-run an
    # already-judged config, so the score has to be in there too.
    ledger = [json.loads(line) for line in
              (run_dir / "runs.jsonl").read_text().splitlines() if line.strip()]
    assert {r["config_id"]: r["judge_mean"] for r in ledger} == {"a": 6.0, "b": 9.0}


def test_eval_all_needs_an_eval_dataset(tmp_path: Path, monkeypatch):
    """Without a held-out set there is nothing to judge against."""
    calls: list[str] = []
    monkeypatch.setattr(
        sweep, "_run_eval_for_config",
        lambda sub, base, cid: calls.append(cid) or {},
    )
    monkeypatch.setattr(sweep, "_run_child", lambda sub_run_dir, cfg: 0)
    monkeypatch.setattr(sweep, "_read_sft_proxy", lambda sub: {"primary": 0.5})
    monkeypatch.setattr(sweep, "_materialize_best_model", lambda winner, run: None)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sweep.sweep_loop(run_dir, {
        "base_config": {"type": "sft", "base_model": "m", "dataset": "datasets/t"},
        "grid_override": [{"id": "a", "overrides": {}}],
        "eval_all": True,
        "eval_best": False,
    })

    assert calls == []


# ---------------------------------------------------------------------------
# Judge spread — the input to the study's fallback noise floor
# ---------------------------------------------------------------------------


def test_judge_std_is_pooled_from_per_source_stats():
    """`score_predictions_by_source` puts only mean/median under `scores`; the
    standard deviation lives per-source. Reading `scores.std` alone silently
    yields nothing, which left the study reporting no noise floor at all when a
    lower bound was available.
    """
    summary = {
        "score_summary": {
            "num_scored": 40,
            "scores": {"mean": 5.8, "median": 6.0},
            "per_source": {
                "a": {"num_scored": 20, "scores": {"mean": 6.0, "std": 2.0}},
                "b": {"num_scored": 20, "scores": {"mean": 5.6, "std": 2.0}},
            },
        },
    }
    fields = sweep._judge_mean_from_summary(summary)
    assert fields["judge_mean"] == 5.8
    assert fields["judge_std"] == pytest.approx(2.0)


def test_judge_std_pooling_weights_by_source_size():
    summary = {"score_summary": {"scores": {"mean": 5.0}, "per_source": {
        "big": {"num_scored": 101, "scores": {"std": 1.0}},
        "small": {"num_scored": 2, "scores": {"std": 9.0}},
    }}}
    std = sweep._judge_mean_from_summary(summary)["judge_std"]
    # Dominated by the 100-dof source, not dragged to 9.0 by the 1-dof one.
    assert 1.0 < std < 2.0


def test_judge_std_prefers_a_directly_reported_value():
    summary = {"score_summary": {
        "scores": {"mean": 5.0, "std": 1.5},
        "per_source": {"a": {"num_scored": 10, "scores": {"std": 9.0}}},
    }}
    assert sweep._judge_mean_from_summary(summary)["judge_std"] == pytest.approx(1.5)


def test_judge_std_absent_when_no_source_has_enough_samples():
    """One sample has no spread — reporting 0.0 would look like a perfectly
    consistent judge and set the noise floor to nothing."""
    summary = {"score_summary": {"scores": {"mean": 5.0}, "per_source": {
        "a": {"num_scored": 1, "scores": {"std": 0.0}},
    }}}
    assert "judge_std" not in sweep._judge_mean_from_summary(summary)


# ---------------------------------------------------------------------------
# Per-child batch re-derivation
# ---------------------------------------------------------------------------


class TestRederiveSftBatch:
    """A grid point that overrides num_epochs must get the batch a standalone
    run at those hyperparameters would get.

    Submission derives the batch once, from the dataset and the DEFAULT epoch
    count. Inheriting it means the epochs axis silently becomes an
    optimizer-update axis (for 2,000 rows the 3-epoch batch of 60 gives a
    1-epoch child ~34 updates against ~102), and the winning config is not
    reproducible outside the sweep.
    """

    def _base(self, rows: int = 2_000, epochs: int = 3) -> dict:
        from lqh.train import defaults

        batch = defaults.sft_effective_batch(rows, epochs)
        return {
            "type": "sft",
            "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "dataset_rows": {"train": rows, "train_effective": rows},
            "lora": {"enabled": True, "r": 32},
            "training": {
                "num_epochs": epochs,
                "per_device_batch_size": batch,
                "effective_batch_size": batch,
                "gradient_accumulation_steps": 1,
            },
        }

    def test_child_batch_tracks_its_own_epoch_count(self) -> None:
        from lqh.train import defaults
        from lqh.train.sweep import _rederive_sft_batch

        for epochs in (1, 2, 3, 8):
            cfg = self._base()
            cfg["training"]["num_epochs"] = epochs
            _rederive_sft_batch(cfg)
            expected = defaults.sft_effective_batch(2_000, epochs)
            assert cfg["training"]["effective_batch_size"] == expected
            steps = defaults.optimizer_steps(
                train_rows=2_000,
                num_epochs=epochs,
                effective_batch_size=expected,
            )
            assert steps >= defaults.SFT_MIN_HEALTHY_OPTIMIZER_STEPS, (epochs, steps)

    def test_unchanged_epochs_is_a_noop(self) -> None:
        from lqh.train.sweep import _rederive_sft_batch

        cfg = self._base()
        before = dict(cfg["training"])
        _rederive_sft_batch(cfg)
        assert cfg["training"] == before

    def test_a_pinned_small_micro_batch_is_respected(self) -> None:
        """A caller who lowered the micro-batch for memory keeps it; only the
        target and the accumulation move."""
        from lqh.train.sweep import _rederive_sft_batch

        cfg = self._base()
        cfg["training"]["num_epochs"] = 1
        cfg["training"]["per_device_batch_size"] = 4
        _rederive_sft_batch(cfg)
        training = cfg["training"]
        assert training["per_device_batch_size"] == 4
        assert training["effective_batch_size"] == 20
        assert training["gradient_accumulation_steps"] == 5

    def test_noop_without_rows_or_for_dpo_vision_and_full_finetune(self) -> None:
        from lqh.train.sweep import _rederive_sft_batch

        # No row count: nothing to derive from (older configs, the study's
        # pre-fix launch payloads).
        cfg = self._base()
        cfg.pop("dataset_rows")
        cfg["training"]["num_epochs"] = 1
        before = dict(cfg["training"])
        _rederive_sft_batch(cfg)
        assert cfg["training"] == before

        for mutate in (
            lambda c: c.update(type="on_policy_dpo"),
            lambda c: c.update(modality="vision"),
            lambda c: c.update(lora={"enabled": False}),
        ):
            cfg = self._base()
            cfg["training"]["num_epochs"] = 1
            mutate(cfg)
            before = dict(cfg["training"])
            _rederive_sft_batch(cfg)
            assert cfg["training"] == before
