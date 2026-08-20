"""GRPO branches in lqh/train/defaults.py + kind inference.

Same contract as test_train_defaults.py: these pin the shipped GRPO
defaults, so a diff here is the signal a default moved (and PROVENANCE /
the grpo_value benchmark should say why).
"""

from __future__ import annotations

from lqh.train import defaults


def test_grpo_recommended_lora() -> None:
    hp = defaults.recommended(run_type="grpo", lora=True)
    # Measured default (grpo_value from-base trial 2026-08-16): 2e-6
    # provably moved nothing; 1e-5 extracts the reward's signal.
    assert hp.learning_rate == 1e-5
    assert hp.num_epochs is None  # bounded by grpo.max_steps, not epochs
    assert hp.per_device_batch_size == 8
    assert hp.effective_batch_size == 64
    # 64 completions at G=8 → 8 whole prompt groups per optimizer step.
    assert hp.effective_batch_size % defaults.GRPO_NUM_GENERATIONS == 0


def test_grpo_recommended_full_ft() -> None:
    hp = defaults.recommended(run_type="grpo", lora=False)
    assert hp.learning_rate == 2e-6
    assert hp.num_epochs is None


def test_grpo_does_not_sweep_by_default() -> None:
    # MASTER_PORT collision + org judge RPM make parallel GRPO hostile;
    # v1 has no sweep support at all.
    assert defaults.sweeps_by_default("grpo") is False
    assert defaults.sweeps_by_default("on_policy_grpo") is False


def test_grpo_knob_defaults_pinned() -> None:
    assert defaults.GRPO_NUM_GENERATIONS == 8
    assert defaults.GRPO_MAX_STEPS == 300
    # Two measured sampling profiles (exploration study, RESULTS.md
    # 2026-08-18): continuation keeps the conservative LFM profile;
    # from-base gets full exploration (3/3 seeds, ~2.4x the gain).
    assert defaults.GRPO_TEMPERATURE == 0.3  # continuation (do-no-harm)
    assert defaults.GRPO_TOP_P == 1.0
    assert defaults.GRPO_MIN_P == 0.05
    assert defaults.GRPO_REPETITION_PENALTY == 1.05
    assert defaults.GRPO_TEMPERATURE_FROM_BASE == 1.0
    assert defaults.GRPO_MIN_P_FROM_BASE == 0.0
    assert defaults.GRPO_REPETITION_PENALTY_FROM_BASE == 1.0
    assert defaults.GRPO_BETA == 0.001      # measured with lr 1e-5 (RESULTS.md)
    # Continuation: KL off — TRL's PEFT reference (adapter disabled) is
    # the raw base, the wrong anchor for a continued SFT adapter.
    assert defaults.GRPO_BETA_CONTINUATION == 0.0
    assert defaults.GRPO_LOSS_TYPE == "dapo"
    assert defaults.GRPO_SCALE_REWARDS == "group"


def test_fill_missing_hyperparameters_grpo() -> None:
    cfg: dict = {}
    filled = defaults.fill_missing_hyperparameters(cfg, run_type="grpo", lora=True)
    assert filled["learning_rate"] == 1e-5
    assert "num_epochs" not in filled  # GRPO has no epoch knob


def test_chatml_to_grpo_rows() -> None:
    from lqh.train.data_utils import chatml_to_grpo_rows

    convos = [
        # normal: user turn + assistant answer → prompt-only + reference
        [{"role": "user", "content": "hi"},
         {"role": "assistant", "content": "hello"}],
        # multi trailing assistant turns all strip into the reference
        [{"role": "user", "content": "go"},
         {"role": "assistant", "content": "a"},
         {"role": "assistant", "content": "b"}],
        # assistant-only conversation → dropped (nothing to prompt with)
        [{"role": "assistant", "content": "orphan"}],
        # no trailing assistant → reference is None
        [{"role": "user", "content": "open-ended"}],
    ]
    tools = [None, [{"type": "function"}], None, None]
    rows = chatml_to_grpo_rows(convos, tools)
    assert len(rows) == 3
    assert rows[0]["prompt"] == [{"role": "user", "content": "hi"}]
    assert rows[0]["reference"] == "hello"
    assert rows[0]["has_tools"] is False
    assert rows[1]["reference"] == "a\nb"
    assert rows[1]["has_tools"] is True
    assert rows[2]["reference"] is None
    # sample_id is a stable content hash: same prompt → same id.
    again = chatml_to_grpo_rows(convos, tools)
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in again]
    assert len({r["sample_id"] for r in rows}) == 3


def test_infer_kind_grpo() -> None:
    from lqh.remote.cloud import _infer_kind

    assert _infer_kind({"type": "grpo"}, "lqh.train") == "train_grpo"
    assert _infer_kind({"type": "on_policy_grpo"}, "lqh.train") == "train_grpo"
    # Explicit kind still wins, and the SFT default is unchanged.
    assert _infer_kind({"kind": "train_grpo"}, "lqh.train") == "train_grpo"
    assert _infer_kind({"type": "sft"}, "lqh.train") == "train_sft"
