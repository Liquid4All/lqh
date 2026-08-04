"""Unit tests for remote publish candidate classification."""

from __future__ import annotations

import json
from pathlib import Path

from lqh.remote.publish import _resolve_candidates


def test_publish_infers_lora_lineage_for_adapter_named_model(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "LiquidAI/base"}) + "\n"
    )

    candidates = _resolve_candidates(tmp_path)
    checkpoint = next(c for c in candidates if c.relpath == "model")

    assert checkpoint.kind == "checkpoint"
    assert checkpoint.lineage is not None
    assert checkpoint.lineage["training_method"] == "lora"
    assert checkpoint.lineage["base_model"] == "LiquidAI/base"


def test_publish_sidecar_lineage_wins_over_adapter_inference(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "LiquidAI/base"}) + "\n"
    )
    (model_dir / "lineage.json").write_text(
        json.dumps(
            {
                "artifact_kind": "checkpoint",
                "training_method": "full",
                "base_model": "explicit/base",
            }
        )
        + "\n"
    )

    candidates = _resolve_candidates(tmp_path)
    checkpoint = next(c for c in candidates if c.relpath == "model")

    assert checkpoint.lineage is not None
    assert checkpoint.lineage["training_method"] == "full"
    assert checkpoint.lineage["base_model"] == "explicit/base"


def _model_dir(run_dir: Path) -> None:
    model = run_dir / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}\n")


def test_a_failed_run_still_publishes_a_protected_model(tmp_path: Path):
    """checkpoint_role="final" marks the run's headline model, which the
    retention engine refuses to expire. It is NOT a success oracle:
    a sweep materializes its running winner after every winning config,
    and SFT saves the model before final evaluation can fail. Those
    weights are real and must survive, so the role stands regardless of
    how the run ended — and nothing downstream may read it as
    "this job completed".
    """
    _model_dir(tmp_path)
    checkpoint = next(c for c in _resolve_candidates(tmp_path) if c.relpath == "model")
    assert checkpoint.checkpoint_role == "final"
