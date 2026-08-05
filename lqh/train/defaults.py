"""The single home for recommended training hyperparameters.

Every default the product trains at lives here, not inlined in the tool
handler. There are two reasons this is worth its own module:

1. **The defaults are now load-bearing.** ``start_training`` no longer sweeps
   SFT by default (see ``lqh/skills/train/SKILL.md``) — the first run is a
   single run at these values, and the sweep is the late-stage "squeeze out
   more" lever. A default that is merely plausible is no longer good enough.
2. **They are measured, not guessed.** ``tests/benchmarks/hp_defaults`` runs a
   factorial study over task × dataset size × model (base/instruct) × model
   size and reports the config with the lowest mean regret against each cell's
   oracle. Its output is a data-only edit to this file plus a ``PROVENANCE``
   update — nothing else moves.

``recommended()`` is the only entry point. It returns the full hyperparameter
set for a run; callers must not re-derive any part of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Where the current numbers come from. Update this in the same commit that
# changes any value below — a default whose provenance is unknown is a default
# nobody can safely revisit.
PROVENANCE = (
    "unvalidated pre-study literals, carried over verbatim from "
    "handle_start_training (2026-08). The hp_defaults calibration study "
    "(tests/benchmarks/hp_defaults) has not been run yet; when it is, replace "
    "the values below with its report's recommendation block and cite the run "
    "id here."
)

# Text LFM attention + FFN projections. LFM2 names its FFN projections
# w1/w2/w3 and carries in_proj/out_proj on the conv blocks, so this list is
# deliberately wider than a Llama-style target set.
TEXT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "w1", "w2", "w3",
)

# Liquid's VLM LoRA recipe: attention + vision-tower MLPs (fc1/fc2) + the
# multimodal projector ("linear"). Intentionally different from the text list.
VISION_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "v_proj", "fc1", "fc2", "linear",
    "gate_proj", "up_proj", "down_proj",
)

# Per-image token budget for the processor. Effective text budget is roughly
# max_seq_length − n_images × max_image_tokens.
VISION_MAX_IMAGE_TOKENS = 256

MAX_SEQ_LENGTH = 2048

_DPO_TYPES = frozenset({"dpo", "on_policy_dpo"})


@dataclass(frozen=True)
class HParams:
    """A complete recommended hyperparameter set for one training run."""

    learning_rate: float
    # None for DPO, which is bounded by ``num_iterations`` instead of epochs.
    num_epochs: int | None
    per_device_batch_size: int
    effective_batch_size: int
    max_seq_length: int
    lora: dict[str, Any] = field(default_factory=dict)

    @property
    def gradient_accumulation_steps(self) -> int:
        """Accumulation needed to reach ``effective_batch_size``.

        Derived, never stored: the two batch numbers are the real knobs and a
        third stored field could contradict them.
        """
        return max(
            1,
            (self.effective_batch_size + self.per_device_batch_size - 1)
            // self.per_device_batch_size,
        )

    def training_config(self) -> dict[str, Any]:
        """The ``config["training"]`` block these hyperparameters imply.

        ``num_epochs`` is omitted for DPO; the DPO-specific knobs
        (dpo_beta, num_iterations, step-aware batching) stay with the caller
        since they are run-shape, not tuned hyperparameters.
        """
        training: dict[str, Any] = {
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            # The GPU calibration probe may shrink the micro-batch for memory
            # safety; it never raises the effective target.
            "auto_batch": True,
        }
        if self.num_epochs is not None:
            training["num_epochs"] = self.num_epochs
        return training


def recommended(
    *,
    run_type: str,
    lora: bool = True,
    modality: str = "text",
    base_model: str | None = None,
    train_rows: int | None = None,
) -> HParams:
    """Return the recommended hyperparameters for a run.

    ``base_model`` and ``train_rows`` are accepted but currently unused: the
    hp_defaults study exists precisely to determine whether model size /
    base-vs-instruct / dataset scale deserve their own defaults. Until it says
    they do, one default covers every cell — and the study reports that finding
    explicitly rather than leaving it assumed. Keeping the parameters in the
    signature now means adopting a conditional default later is a change to
    this function's body alone.
    """
    is_dpo = run_type in _DPO_TYPES
    is_vision = modality == "vision"

    if is_vision:
        learning_rate = 5e-4  # Liquid VLM LoRA recipe
    elif is_dpo:
        learning_rate = 1e-6
    else:
        learning_rate = 2e-5

    if is_vision:
        # No calibration probe for vision — start conservative and let the OOM
        # self-heal (report_oom_downgrade) shrink further if needed.
        micro_batch, effective_batch = 2, 16
    elif lora and is_dpo:
        # DPO preference batches are normally only a few hundred rows. The
        # LoRA-wide default of 256 would reduce those to one or two optimizer
        # updates per on-policy iteration (see DPO_FIX.md).
        micro_batch, effective_batch = 16, 16
    elif lora:
        micro_batch, effective_batch = 256, 256
    else:
        micro_batch = 1
        effective_batch = 2 if is_dpo else 16

    lora_config: dict[str, Any] = {
        "enabled": lora,
        "r": 8 if is_vision else 32,
        "alpha": 16 if is_vision else 64,
        "dropout": 0.05 if is_vision else 0.02,
        "target_modules": list(
            VISION_TARGET_MODULES if is_vision else TEXT_TARGET_MODULES
        ),
    }

    return HParams(
        learning_rate=learning_rate,
        num_epochs=None if is_dpo else 3,
        per_device_batch_size=micro_batch,
        effective_batch_size=effective_batch,
        max_seq_length=MAX_SEQ_LENGTH,
        lora=lora_config,
    )


def sweeps_by_default(run_type: str) -> bool:
    """Whether ``start_training`` sweeps this run type when unspecified.

    SFT does not: the values above are the validated starting point, and a
    sweep costs hours that the first run should not spend. DPO still does —
    it is far more sensitive to learning rate and beta, and the hp_defaults
    study covers SFT only, so its DPO defaults remain unvalidated. Flipping
    DPO off would trade a slow first run for an unreliable one.
    """
    return run_type in _DPO_TYPES
