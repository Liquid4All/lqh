"""Parity tests for lqh/train/defaults.py.

These pin the recommended hyperparameters to the exact literals that
``handle_start_training`` inlined before they were centralised, so the
extraction is provably behaviour-preserving. When the hp_defaults calibration
study lands new values, THIS FILE is what changes alongside defaults.py — and a
diff here is the signal that a shipped default moved.
"""

from __future__ import annotations

import pytest

from lqh.train import defaults


TEXT_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "w1", "w2", "w3",
]
VISION_MODULES = [
    "q_proj", "v_proj", "fc1", "fc2", "linear",
    "gate_proj", "up_proj", "down_proj",
]


@pytest.mark.parametrize(
    "run_type,lora,modality,expected_lr",
    [
        ("sft", True, "text", 2e-5),
        ("sft", False, "text", 2e-5),
        ("sft", True, "vision", 5e-4),
        ("on_policy_dpo", True, "text", 1e-6),
        ("on_policy_dpo", False, "text", 1e-6),
        ("dpo", True, "text", 1e-6),
    ],
)
def test_learning_rate_matches_pre_extraction_literals(
    run_type, lora, modality, expected_lr
):
    hp = defaults.recommended(run_type=run_type, lora=lora, modality=modality)
    assert hp.learning_rate == expected_lr


@pytest.mark.parametrize(
    "run_type,lora,modality,micro,effective",
    [
        # Vision ignores the LoRA/full split: no calibration probe runs, so it
        # starts conservative either way.
        ("sft", True, "vision", 2, 16),
        ("sft", False, "vision", 2, 16),
        # LoRA SFT is throughput-oriented.
        ("sft", True, "text", 256, 256),
        # LoRA DPO deliberately does NOT inherit 256 — a few hundred
        # preference rows would collapse to one or two optimizer updates.
        ("on_policy_dpo", True, "text", 16, 16),
        ("dpo", True, "text", 16, 16),
        # Full fine-tuning is memory-bound.
        ("sft", False, "text", 1, 16),
        ("on_policy_dpo", False, "text", 1, 2),
    ],
)
def test_batch_sizes_match_pre_extraction_literals(
    run_type, lora, modality, micro, effective
):
    hp = defaults.recommended(run_type=run_type, lora=lora, modality=modality)
    assert hp.per_device_batch_size == micro
    assert hp.effective_batch_size == effective


@pytest.mark.parametrize(
    "run_type,lora,modality,micro,effective,expected_accum",
    [
        ("sft", True, "vision", 2, 16, 8),
        ("sft", True, "text", 256, 256, 1),
        ("on_policy_dpo", True, "text", 16, 16, 1),
        ("sft", False, "text", 1, 16, 16),
        ("on_policy_dpo", False, "text", 1, 2, 2),
    ],
)
def test_gradient_accumulation_is_ceil_of_batch_ratio(
    run_type, lora, modality, micro, effective, expected_accum
):
    """The old handler computed ceil(effective / micro) inline."""
    hp = defaults.recommended(run_type=run_type, lora=lora, modality=modality)
    assert hp.gradient_accumulation_steps == expected_accum
    assert hp.gradient_accumulation_steps * micro >= effective


def test_text_lora_config_matches_pre_extraction_literals():
    hp = defaults.recommended(run_type="sft", lora=True, modality="text")
    assert hp.lora == {
        "enabled": True,
        "r": 32,
        "alpha": 64,
        "dropout": 0.02,
        "target_modules": TEXT_MODULES,
    }


def test_vision_lora_config_matches_liquid_vlm_recipe():
    hp = defaults.recommended(run_type="sft", lora=True, modality="vision")
    assert hp.lora == {
        "enabled": True,
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": VISION_MODULES,
    }


def test_lora_disabled_still_carries_the_shape():
    """The old handler always built the full dict and only flipped `enabled`.

    Downstream code reads `lora.r` unconditionally, so dropping the other keys
    when LoRA is off would be a behaviour change, not a cleanup.
    """
    hp = defaults.recommended(run_type="sft", lora=False, modality="text")
    assert hp.lora["enabled"] is False
    assert hp.lora["r"] == 32
    assert hp.lora["target_modules"] == TEXT_MODULES


def test_lora_dict_is_not_shared_between_calls():
    """Callers mutate config["lora"]; a shared dict would leak across runs."""
    first = defaults.recommended(run_type="sft")
    second = defaults.recommended(run_type="sft")
    first.lora["r"] = 999
    first.lora["target_modules"].append("bogus")
    assert second.lora["r"] == 32
    assert "bogus" not in second.lora["target_modules"]


def test_sft_gets_epochs_and_dpo_does_not():
    """DPO is bounded by num_iterations; num_epochs is meaningless there."""
    assert defaults.recommended(run_type="sft").num_epochs == 3
    assert defaults.recommended(run_type="on_policy_dpo").num_epochs is None
    assert defaults.recommended(run_type="dpo").num_epochs is None


def test_training_config_shape():
    training = defaults.recommended(run_type="sft", lora=True).training_config()
    assert training == {
        "learning_rate": 2e-5,
        "max_seq_length": 2048,
        "per_device_batch_size": 256,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 256,
        "auto_batch": True,
        "num_epochs": 3,
    }


def test_training_config_omits_num_epochs_for_dpo():
    training = defaults.recommended(run_type="on_policy_dpo").training_config()
    assert "num_epochs" not in training
    assert training["learning_rate"] == 1e-6


def test_sweeps_by_default_is_off_for_sft_on_for_dpo():
    """The policy flip: SFT trains once at these defaults, DPO still searches."""
    assert defaults.sweeps_by_default("sft") is False
    assert defaults.sweeps_by_default("dpo") is True
    assert defaults.sweeps_by_default("on_policy_dpo") is True


def test_provenance_is_recorded():
    """A default whose origin is unknown is a default nobody can revisit."""
    assert defaults.PROVENANCE.strip()
