"""Tests for the eval-of-best step in lqh.train.sweep.

Three guarantees these tests pin down:

  1. The infer config we hand to ``lqh.infer`` carries every field
     the trainer needs (base_model, dataset, scorer/system_prompt
     when present), and skips cleanly when there's no eval_dataset
     in the base config.

  2. The post-eval handoff (symlink/copy of ``predictions.parquet``
     and ``eval_request.json`` up to the run_dir level) produces
     files the host-side watcher's existing path detection picks up.

  3. The orchestration is robust: a failing infer subprocess returns
     a "skipped" summary without crashing the sweep's terminal write.

We don't run real inference — that'd need a GPU, a model, and a
dataset. Instead the test fakes the subprocess via monkeypatch so the
control flow can be exercised end-to-end in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lqh.train import sweep


# ---------------------------------------------------------------------
# Pure config builder
# ---------------------------------------------------------------------


def test_build_eval_config_carries_required_fields(tmp_path: Path):
    # Materialize a fake winner model so the builder doesn't bail.
    (tmp_path / "model").mkdir()

    base = {
        "type": "sft",
        "base_model": "LiquidAI/LFM2-1.2B",
        "dataset": "datasets/train",
        "eval_dataset": "datasets/heldout",
        "scorer": "evals/scorers/translate.md",
        "max_new_tokens": 2048,
    }
    cfg = sweep._build_eval_of_best_config(base, tmp_path)
    assert cfg is not None
    assert cfg["type"] == "infer"
    assert cfg["dataset"] == "datasets/heldout"
    assert cfg["scorer"] == "evals/scorers/translate.md"
    assert cfg["max_new_tokens"] == 2048
    # base_model must point at the winner model, NOT the original
    # pre-fine-tune model — eval-of-best evaluates the WINNER.
    assert cfg["base_model"].endswith("/model")
    # Manifest lists every path the bundle builder would need to sync.
    assert "base_model" in cfg["manifest"]
    assert "dataset" in cfg["manifest"]
    assert "scorer" in cfg["manifest"]


def test_build_eval_config_accepts_lora_winner_dir(tmp_path: Path):
    (tmp_path / "model-lora").mkdir()
    base = {"eval_dataset": "datasets/heldout"}

    cfg = sweep._build_eval_of_best_config(base, tmp_path)

    assert cfg is not None
    assert cfg["base_model"].endswith("/model-lora")


def test_build_eval_config_skips_when_no_eval_dataset(tmp_path: Path):
    (tmp_path / "model").mkdir()
    base = {"type": "sft", "base_model": "x", "dataset": "datasets/train"}
    assert sweep._build_eval_of_best_config(base, tmp_path) is None


def test_build_eval_config_skips_when_no_winner_model(tmp_path: Path):
    # No tmp_path/model directory exists — sweep failed to materialize
    # a winner, so eval-of-best must skip.
    base = {"type": "sft", "eval_dataset": "datasets/heldout"}
    assert sweep._build_eval_of_best_config(base, tmp_path) is None


def test_build_eval_config_includes_optional_system_prompt_and_format(tmp_path: Path):
    (tmp_path / "model").mkdir()
    base = {
        "eval_dataset": "datasets/heldout",
        "system_prompt": "You are a translator.",
        "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
    }
    cfg = sweep._build_eval_of_best_config(base, tmp_path)
    assert cfg["system_prompt"] == "You are a translator."
    assert cfg["response_format"]["type"] == "json_schema"


# ---------------------------------------------------------------------
# Orchestration / file handoff
# ---------------------------------------------------------------------


def _fake_infer_success(predictions_bytes: bytes):
    """Return a fake subprocess.run that emulates a successful
    ``python -m lqh.infer`` invocation by writing predictions.parquet
    and eval_request.json into eval_of_best/ as the real binary would."""
    def runner(argv, *, stdout, stderr, check):
        # argv = [python, "-m", "lqh.infer", "<config-path>"]
        cfg_path = Path(argv[-1])
        eval_dir = cfg_path.parent
        (eval_dir / "predictions.parquet").write_bytes(predictions_bytes)
        (eval_dir / "eval_request.json").write_text(
            json.dumps({"status": "ready", "predictions": "predictions.parquet"}) + "\n"
        )

        # Emulate subprocess.CompletedProcess just enough for sweep's
        # check on .returncode.
        class _Result:
            returncode = 0
        return _Result()
    return runner


def _fake_infer_failure(rc: int = 7):
    def runner(argv, *, stdout, stderr, check):
        class _Result:
            returncode = rc
        return _Result()
    return runner


def test_eval_of_best_happy_path_surfaces_predictions(tmp_path: Path, monkeypatch):
    # Materialize a fake winner (sweep would have symlinked the
    # winner's model/ here).
    (tmp_path / "model").mkdir()
    base = {
        "type": "sft",
        "eval_dataset": "datasets/heldout",
        "scorer": "evals/scorers/x.md",
    }
    monkeypatch.setattr(sweep.subprocess, "run", _fake_infer_success(b"PARQUETBYTES"))

    summary = sweep._run_eval_of_best(tmp_path, base)
    assert summary == {"ok": True, "dataset": "datasets/heldout", "model": str((tmp_path / "model").resolve())}

    # Predictions surfaced at the run-dir level so the host watcher
    # finds them via its existing path detection.
    assert (tmp_path / "predictions.parquet").read_bytes() == b"PARQUETBYTES"
    eval_req = json.loads((tmp_path / "eval_request.json").read_text())
    assert eval_req["status"] == "ready"

    # The eval subdir got an infer config and a log file, so an
    # operator can debug a failed run by reading eval_of_best/.
    assert (tmp_path / "eval_of_best" / "config.json").exists()
    assert (tmp_path / "eval_of_best" / "infer.log").exists()


def test_eval_of_best_skips_when_no_eval_dataset(tmp_path: Path, monkeypatch):
    (tmp_path / "model").mkdir()
    base = {"type": "sft"}  # no eval_dataset
    # The subprocess should NEVER be called if config-build returns None.
    monkeypatch.setattr(sweep.subprocess, "run",
                        lambda *a, **k: pytest.fail("subprocess.run should not be called"))

    summary = sweep._run_eval_of_best(tmp_path, base)
    assert "skipped" in summary
    assert not (tmp_path / "predictions.parquet").exists()


def test_eval_of_best_skips_when_infer_fails(tmp_path: Path, monkeypatch):
    (tmp_path / "model").mkdir()
    base = {"eval_dataset": "datasets/heldout"}
    monkeypatch.setattr(sweep.subprocess, "run", _fake_infer_failure(rc=42))

    summary = sweep._run_eval_of_best(tmp_path, base)
    assert "skipped" in summary
    assert "42" in summary["skipped"]
    # No phantom predictions land at the run-dir.
    assert not (tmp_path / "predictions.parquet").exists()


def test_eval_of_best_overwrites_existing_predictions(tmp_path: Path, monkeypatch):
    """If a prior eval left predictions.parquet at the run-dir level,
    a fresh eval-of-best must replace it rather than fail the symlink."""
    (tmp_path / "model").mkdir()
    (tmp_path / "predictions.parquet").write_bytes(b"OLD")
    base = {"eval_dataset": "datasets/heldout"}
    monkeypatch.setattr(sweep.subprocess, "run", _fake_infer_success(b"NEW"))

    summary = sweep._run_eval_of_best(tmp_path, base)
    assert summary.get("ok")
    # Updated to the fresh bytes — either via symlink-replace or
    # copy-with-overwrite; the test asserts the observable result.
    assert (tmp_path / "predictions.parquet").read_bytes() == b"NEW"


def test_materialize_best_model_preserves_lora_dir_name(tmp_path: Path):
    winner = tmp_path / "winner"
    run_dir = tmp_path / "run"
    lora_dir = winner / "model-lora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_config.json").write_text("{}")
    (run_dir / "model").mkdir(parents=True)

    sweep._materialize_best_model(winner, run_dir)

    assert (run_dir / "model-lora").exists()
    assert not (run_dir / "model").exists()
