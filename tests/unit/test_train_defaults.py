"""Parity tests for lqh/train/defaults.py.

These pin the recommended hyperparameters to the values the product ships.
Originally that was the exact literals ``handle_start_training`` inlined before
they were centralised; the text LoRA learning rate and the SFT batch derivation
have since moved (see ``defaults.PROVENANCE``). When the hp_defaults calibration
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
        # LoRA moves only the adapter and wants an order of magnitude more than
        # a full fine-tune; the two must not share one literal.
        ("sft", True, "text", 2e-4),
        ("sft", False, "text", 2e-5),
        ("sft", True, "vision", 5e-4),
        ("on_policy_dpo", True, "text", 1e-6),
        ("on_policy_dpo", False, "text", 1e-6),
        ("dpo", True, "text", 1e-6),
    ],
)
def test_learning_rate_matches_shipped_literals(
    run_type, lora, modality, expected_lr
):
    hp = defaults.recommended(run_type=run_type, lora=lora, modality=modality)
    assert hp.learning_rate == expected_lr


def test_text_lora_learning_rate_is_in_the_lora_range():
    """The bug this guards: a full-fine-tuning rate on a LoRA adapter produced
    three flat customer runs. LoRA literature wants 1e-4–5e-4."""
    lora_lr = defaults.recommended(run_type="sft", lora=True).learning_rate
    full_lr = defaults.recommended(run_type="sft", lora=False).learning_rate
    assert 1e-4 <= lora_lr <= 5e-4
    assert lora_lr > full_lr


@pytest.mark.parametrize(
    "run_type,lora,modality,micro,effective",
    [
        # Vision ignores the LoRA/full split: no calibration probe runs, so it
        # starts conservative either way.
        ("sft", True, "vision", 2, 16),
        ("sft", False, "vision", 2, 16),
        # LoRA SFT with no row count known: the throughput-oriented ceiling,
        # exactly as before the batch became dataset-derived.
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
def test_batch_sizes_match_shipped_literals(
    run_type, lora, modality, micro, effective
):
    hp = defaults.recommended(run_type=run_type, lora=lora, modality=modality)
    assert hp.per_device_batch_size == micro
    assert hp.effective_batch_size == effective


# ---------------------------------------------------------------------------
# Optimizer-step floor (the fixed batch of 256 gave a 1,790-row 3-epoch run 21
# updates in total, which reads as "the dataset is bad")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows,epochs",
    [
        (1_790, 3),   # the customer's first run
        (4_433, 3),   # their second, after scaling the data
        (500, 3),     # a pilot
        (2_000, 1),   # one epoch, mid-size
        (200_000, 3),  # far past the ceiling
    ],
)
def test_derived_batch_clears_the_step_floor(rows, epochs):
    hp = defaults.recommended(
        run_type="sft", lora=True, train_rows=rows, num_epochs=epochs
    )
    steps = defaults.optimizer_steps(
        train_rows=rows,
        num_epochs=epochs,
        effective_batch_size=hp.effective_batch_size,
    )
    at_floor = hp.effective_batch_size == defaults.SFT_MIN_EFFECTIVE_BATCH
    assert steps >= defaults.SFT_TARGET_OPTIMIZER_STEPS or at_floor
    assert steps >= defaults.SFT_MIN_HEALTHY_OPTIMIZER_STEPS


def test_derived_batch_stays_within_bounds():
    for rows in (1, 50, 500, 5_000, 50_000, 5_000_000):
        batch = defaults.sft_effective_batch(rows, 3)
        assert defaults.SFT_MIN_EFFECTIVE_BATCH <= batch
        assert batch <= defaults.SFT_MAX_EFFECTIVE_BATCH


def test_large_dataset_keeps_the_old_throughput_batch():
    """The derivation must not slow down runs that were already fine: the
    ceiling is the value every LoRA run used before."""
    hp = defaults.recommended(
        run_type="sft", lora=True, train_rows=100_000, num_epochs=3
    )
    assert hp.effective_batch_size == 256
    assert hp.per_device_batch_size == 256


def test_micro_batch_never_exceeds_the_effective_target():
    """A micro-batch above the effective target silently raises the true
    optimizer batch (accumulation cannot go below 1)."""
    hp = defaults.recommended(
        run_type="sft", lora=True, train_rows=1_790, num_epochs=3
    )
    assert hp.per_device_batch_size <= hp.effective_batch_size
    assert hp.gradient_accumulation_steps == 1


def test_row_count_does_not_touch_full_finetune_or_dpo_or_vision():
    """Only the text LoRA SFT branch derives its batch."""
    full = defaults.recommended(
        run_type="sft", lora=False, train_rows=500, num_epochs=3
    )
    assert (full.per_device_batch_size, full.effective_batch_size) == (1, 16)
    dpo = defaults.recommended(
        run_type="on_policy_dpo", lora=True, train_rows=500
    )
    assert (dpo.per_device_batch_size, dpo.effective_batch_size) == (16, 16)
    vision = defaults.recommended(
        run_type="sft", lora=True, modality="vision", train_rows=500, num_epochs=3
    )
    assert (vision.per_device_batch_size, vision.effective_batch_size) == (2, 16)


def test_epoch_override_is_honoured_and_defaults_to_three():
    assert defaults.recommended(run_type="sft", num_epochs=8).num_epochs == 8
    assert defaults.recommended(run_type="sft").num_epochs == 3
    # DPO is bounded by num_iterations — an epoch override must not leak in.
    assert defaults.recommended(run_type="dpo", num_epochs=8).num_epochs is None


@pytest.mark.parametrize(
    "rows,epochs,batch,expected",
    [
        (1_790, 3, 256, 21),   # the run that bought nothing
        (4_433, 3, 256, 54),
        (100, 1, 16, 7),
        (0, 3, 16, 0),         # unknown row count → unknowable
        (100, 3, 0, 0),        # nonsense batch → no answer, no ZeroDivisionError
    ],
)
def test_optimizer_steps_matches_hf_arithmetic(rows, epochs, batch, expected):
    """ceil(rows / batch) * epochs — the last partial batch of each epoch is
    still a step, since dataloader_drop_last is off."""
    assert defaults.optimizer_steps(
        train_rows=rows, num_epochs=epochs, effective_batch_size=batch
    ) == expected


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
        "learning_rate": 2e-4,
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
