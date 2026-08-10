"""The single home for recommended training hyperparameters.

Every default the product trains at lives here, not inlined in the tool
handler. There are two reasons this is worth its own module:

1. **The defaults are now load-bearing.** ``start_training`` no longer sweeps
   SFT by default (see ``lqh/skills/train/SKILL.md``) — the first run is a
   single run at these values, and the sweep is the late-stage "squeeze out
   more" lever. A default that is merely plausible is no longer good enough.
2. **They are meant to be measured, not guessed.**
   ``tests/benchmarks/hp_defaults`` runs a factorial study over task × dataset
   size × model (base/instruct) × model size and reports the config with the
   lowest mean regret against each cell's oracle. Its output is a data-only
   edit to this file plus a ``PROVENANCE`` update — nothing else moves. It has
   not been run yet, so ``PROVENANCE`` currently reads "literals" — check it
   before treating any number here as validated.

``recommended()`` is the only entry point. It returns the full hyperparameter
set for a run; callers must not re-derive any part of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Where the current numbers come from. Update this in the same commit that
# changes any value below — a default whose provenance is unknown is a default
# nobody can safely revisit.
PROVENANCE = (
    "unvalidated pre-study literals, carried over verbatim from "
    "handle_start_training (2026-08), with two field-driven corrections in "
    "2026-08 (feedback item 47): (a) the text LoRA learning rate was split "
    "off from the full-fine-tuning rate and raised 2e-5 -> 2e-4, after a "
    "customer's trilingual-translation task moved +1.30 judge points from a "
    "hand-set 5e-4 while three runs at 2e-5 produced nothing; (b) the LoRA "
    "effective batch size is now derived from the dataset so a run always "
    "takes enough optimizer steps to learn (the fixed 256 gave a 1,790-row "
    "3-epoch run 21 updates in total). Both are still literals, not "
    "measurements: the hp_defaults calibration study "
    "(tests/benchmarks/hp_defaults) has not been run yet. When it is, replace "
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

DEFAULT_SFT_EPOCHS = 3

# How many optimizer updates a text SFT run should get. A LoRA adapter at a
# fixed effective batch of 256 gets ~20 updates out of a 2k-row 3-epoch run,
# which is not enough for the adapter to move regardless of the learning rate —
# and the symptom (flat judge score) looks exactly like "the dataset is bad".
# The batch is therefore derived from the dataset, the same reasoning the DPO
# branch in ``recommended()`` already applies to preference sets.
SFT_TARGET_OPTIMIZER_STEPS = 100

# Bounds on that derivation. The floor keeps gradients from getting noisy on
# tiny datasets; the ceiling is the throughput-oriented value LoRA runs used
# unconditionally before, so a large dataset trains exactly as it did.
SFT_MIN_EFFECTIVE_BATCH = 16
SFT_MAX_EFFECTIVE_BATCH = 256

# Below this many total updates, a run is worth warning about at train time
# (see lqh/train/sft.py) even after the derivation above — e.g. a caller who
# passed an explicit tiny batch, or a dataset small enough to hit the floor.
SFT_MIN_HEALTHY_OPTIMIZER_STEPS = 50

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


def optimizer_steps(
    *,
    train_rows: int | None,
    num_epochs: int | None,
    effective_batch_size: int,
) -> int:
    """Total optimizer updates a run will take, or 0 when unknowable.

    Mirrors HF Trainer's arithmetic: the last partial batch of each epoch is
    still a step (``dataloader_drop_last`` is off), so it is
    ``ceil(rows / batch) * epochs``. Slightly optimistic when the run splits an
    internal eval slice off the training set (``eval_split_ratio``, default
    10%) — the reader wants the order of magnitude, not the exact count.
    """
    if not train_rows or train_rows <= 0 or effective_batch_size <= 0:
        return 0
    per_epoch = max(1, math.ceil(train_rows / effective_batch_size))
    return per_epoch * max(1, num_epochs or 1)


def sft_effective_batch(
    train_rows: int | None = None,
    num_epochs: int | None = None,
) -> int:
    """Effective batch for a text LoRA SFT run, floored on optimizer steps.

    Returns the largest batch (capped at ``SFT_MAX_EFFECTIVE_BATCH``) that
    still buys ``SFT_TARGET_OPTIMIZER_STEPS`` updates over the whole run, never
    going below ``SFT_MIN_EFFECTIVE_BATCH``. With no row count available the
    answer is the ceiling — the pre-derivation behaviour.
    """
    if not train_rows or train_rows <= 0:
        return SFT_MAX_EFFECTIVE_BATCH
    epochs = max(1, num_epochs or DEFAULT_SFT_EPOCHS)
    batch = int(train_rows * epochs) // SFT_TARGET_OPTIMIZER_STEPS
    return max(SFT_MIN_EFFECTIVE_BATCH, min(SFT_MAX_EFFECTIVE_BATCH, batch))


def recommended(
    *,
    run_type: str,
    lora: bool = True,
    modality: str = "text",
    base_model: str | None = None,
    train_rows: int | None = None,
    num_epochs: int | None = None,
) -> HParams:
    """Return the recommended hyperparameters for a run.

    ``train_rows`` (the number of training rows the run will actually see,
    including ``repeat`` weighting) and ``num_epochs`` (the caller's explicit
    epoch override, if any) size the text LoRA batch so the run takes enough
    optimizer steps — see :func:`sft_effective_batch`. Pass them whenever they
    are known; omitting them keeps the old fixed-256 batch.

    ``base_model`` is accepted but currently unused: the hp_defaults study
    exists precisely to determine whether model size / base-vs-instruct deserve
    their own defaults. Until it says they do, one default covers every cell —
    and the study reports that finding explicitly rather than leaving it
    assumed. Keeping the parameter in the signature now means adopting a
    conditional default later is a change to this function's body alone.
    """
    is_dpo = run_type in _DPO_TYPES
    is_vision = modality == "vision"

    if is_vision:
        learning_rate = 5e-4  # Liquid VLM LoRA recipe
    elif is_dpo:
        learning_rate = 1e-6
    elif lora:
        # LoRA wants 1e-4–5e-4, not a full-fine-tuning rate: only the adapter
        # moves, so the same 2e-5 that suits every weight in the model barely
        # moves a rank-32 adapter (see PROVENANCE — three customer runs at
        # 2e-5 were flat, the same data at 5e-4 gained +1.30).
        learning_rate = 2e-4
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
        # Same failure mode as DPO's, one branch up: derive the batch from the
        # dataset instead of pinning 256 and hoping the dataset is big enough.
        effective_batch = sft_effective_batch(train_rows, num_epochs)
        micro_batch = effective_batch
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
        num_epochs=None if is_dpo else (num_epochs or DEFAULT_SFT_EPOCHS),
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
