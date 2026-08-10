"""The single home for recommended training hyperparameters.

Every default the product trains at lives here, not inlined in the tool
handler. There are two reasons this is worth its own module:

1. **The defaults are now load-bearing.** ``start_training`` no longer sweeps
   SFT by default (see ``lqh/skills/train/SKILL.md``) — the first run is a
   single run at these values, and the sweep is the late-stage "squeeze out
   more" lever. A default that is merely plausible is no longer good enough.
2. **The learning rate is measured; everything else is not.**
   ``tests/benchmarks/hp_defaults`` runs a factorial study over task × dataset
   size × model (base/instruct) × model size and reports the config with the
   lowest mean regret against each cell's oracle. Its output is a data-only
   edit to this file plus a ``PROVENANCE`` update — nothing else moves. Its
   stage-A screen has run and set the text LoRA learning rate; ``PROVENANCE``
   records the run, the regret, and four limits on how far that generalises —
   including that its report is not archived here. Everything else in this file
   (epoch count, batch sizes, LoRA shape, DPO knobs) is an unmeasured literal,
   and the per-value comments say which is which.

``recommended()`` is the only entry point for a *new* run's hyperparameters:
callers must not invent their own values or copy a literal out of this file. The
one legitimate re-derivation is by this module's own helpers — a sweep child
whose grid point changes ``num_epochs`` calls :func:`sft_effective_batch` again
(``lqh.train.sweep._rederive_sft_batch``) so its batch matches what a standalone
run at those hyperparameters would get.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Where the current numbers come from. Update this in the same commit that
# changes any value below — a default whose provenance is unknown is a default
# nobody can safely revisit.
PROVENANCE = (
    "learning_rate (text LoRA SFT, 1e-4): reported by hp_defaults study run "
    "hpd-stageA (2026-08) — lr1e-4_e3 won on mean regret 0.015 judge points "
    "(95% CI 0.002-0.032) over 6 cells x 15 configs, as an interior optimum "
    "(5e-5 below and 2e-4 above both worse). No dimension earned its own "
    "default. FOUR LIMITS. (1) NOT INDEPENDENTLY AUDITABLE: the report was "
    "read from the run's output, but neither report/results.json nor "
    "report/report.md is archived in this repo and the hpd-stageA workdir on "
    "the dev machine holds no run records — every number here is a transcript, "
    "not a reproducible artifact. Commit the report before treating this "
    "provenance as complete. (2) Stage A only: the anchor screen, 6 cells, not "
    "the 48-cell confirm, with 6 chunks lost to orphaned cloud jobs. (3) No "
    "seed replicates, so the 0.127-point noise floor is a LOWER bound (judge "
    "sampling error only, not training variance) and the top five configs "
    "(regret 0.015-0.097) are statistically tied — 1e-4 vs 2e-4 is a coin "
    "flip. What the study establishes firmly is the size of the OLD default's "
    "error, not the winner's precise value: lr2e-5_e3, the value actually "
    "shipped before item 47, measured 0.523 mean / 1.17 worst-case regret (the "
    "2e-5 row reached 0.960 mean / 2.61 worst at 1 epoch, which was never the "
    "shipped config). A customer's 1.2B task separately gained +1.30 from a "
    "hand-set 5e-4, a value the study never tested. (4) 5 of the 6 contributing "
    "cells were 350M models, so this is close to a 350M-only result. Run stage "
    "B with --replicate-seeds and working 1.2B cells before treating 1e-4 as "
    "settled above 350M. "
    "num_epochs (3) and effective_batch_size: NOT from the study — see their "
    "own comments below. The batch is derived from the dataset (item 47) so a "
    "run aims at enough optimizer steps to learn — datasets too small to reach "
    "the target at the minimum batch still fall short, by design; the fixed 256 "
    "gave a 1,790-row 3-epoch run 21 updates in total."
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

# NOT measured — carried over unchanged, and the study cannot settle it.
# hpd-stageA's 3-epoch configs did win, but that comparison is confounded: its
# sweep derived one batch from the 3-epoch default and then overrode epochs
# without resizing it, so the 1- and 2-epoch configs ran at roughly a third and
# two thirds of the optimizer updates they would get as an installed default.
# "More updates won" is not "more epochs won". (lqh/train/sweep.py now resizes a
# child's batch for its own epoch count, so a re-run measures the axis properly.)
# The downside of 3 is bounded either way: load_best_model_at_end keeps the best
# checkpoint by eval_loss, and only ~10% of study runs kept one from before the
# final epoch.
DEFAULT_SFT_EPOCHS = 3

# How many optimizer updates a text SFT run should aim for. A LoRA adapter at a
# fixed effective batch of 256 gets ~20 updates out of a 2k-row 3-epoch run,
# which is not enough for the adapter to move regardless of the learning rate —
# and the symptom (flat judge score) looks exactly like "the dataset is bad".
# The batch is therefore derived from the dataset, the same reasoning the DPO
# branch in ``recommended()`` already applies to preference sets.
#
# 100 is a JUDGEMENT CALL, not a measurement: it is round, comfortably above the
# ~20 that demonstrably failed for a customer (feedback item 47), and below the
# point where a small dataset's batch would collapse to noise. No study has
# swept it across tasks and model sizes — hp_defaults did not vary it. Treat it
# as "enough to rule out update starvation as a cause", not as an optimum.
SFT_TARGET_OPTIMIZER_STEPS = 100

# Bounds on that derivation. The floor keeps gradients from getting noisy on
# tiny datasets; the ceiling is the throughput-oriented value LoRA runs used
# unconditionally before, so a large dataset trains exactly as it did. Note the
# floor wins on small datasets: below ~270 effective rows at 3 epochs the target
# above is unreachable, so those runs take fewer updates by design.
SFT_MIN_EFFECTIVE_BATCH = 16
SFT_MAX_EFFECTIVE_BATCH = 256

# Below this many total updates, a run is worth warning about at train time (see
# lqh/train/sft.py) even after the derivation above — e.g. a caller who passed an
# explicit tiny batch, or a dataset small enough to hit the floor. Also a
# judgement call (half the target), and the same caveat applies: it flags a
# plausible cause, it does not establish one.
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
        # Measured (hpd-stageA): lowest mean regret AND lowest worst-case of
        # the 15 configs, with 5e-5 below and 2e-4 above both worse. LoRA needs
        # an order of magnitude more than a full fine-tune because only the
        # adapter moves — 2e-5 here measured at 0.52-0.96 regret. See
        # PROVENANCE for what this number does and does not cover.
        learning_rate = 1e-4
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
