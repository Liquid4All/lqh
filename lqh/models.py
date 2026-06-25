"""Curated catalog of Liquid AI models.

Source of truth for the Liquid model IDs the agent can evaluate. Mirrors the
repo-root ``MODELS.md``; keep the two in sync when models are added or removed.

These models are evaluated exclusively through the **HuggingFace inference
path** (``eval_hf_model`` for cloud, ``start_local_eval`` for local/SSH). The
old ``router.liquid.ai`` inference API has been retired, so there is no
API-based path for running a Liquid checkpoint anymore (see ``MODELS.md``).

Sampling note (from ``MODELS.md``): the small Liquid models want a very low
temperature — greedy (``temperature=0.0``) is recommended, or 0.1–0.3 max.
Inference in lqh is already greedy everywhere (``do_sample=False`` /
``temperature=0.0``); we keep that default. If temperature is ever enabled
(>0), the recommended companions are ``repetition_penalty=1.05`` and
``min_p=0.05``. Do not add a temperature param where one is absent — greedy is
the intended default.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LiquidModel",
    "LIQUID_MODELS",
    "SIZE_RECOMMENDATION",
    "is_liquid_model_name",
    "format_catalog",
]

# Short, non-extreme starting-size guidance surfaced to the orchestration agent
# via `list_models`. The full reasoning (zero-shot as a complexity gauge,
# stepping up when fine-tuning struggles, base-vs-instruct) lives in the agent
# system prompt and the auto skill; keep this terse and consistent with those.
SIZE_RECOMMENDATION = (
    "Recommended starting size: 1.2B for most tasks; 2.6B or the 8B-A1B MoE for "
    "more complex tasks; 350M for very simple ones (avoid the 230M / 24B extremes "
    "unless the task clearly calls for it). Ask the user which size to start with "
    "before the first eval or fine-tune run; use the zero-shot baseline as a rough "
    "read on task complexity, and step up a size if fine-tuning keeps struggling "
    "despite good data and a sane scorer. For the SFT base, instruct/no-suffix is "
    "the safe default; '-Base' has a slight edge at large dataset sizes."
)


@dataclass(frozen=True)
class LiquidModel:
    """A Liquid AI model available for HuggingFace-based evaluation.

    ``kind`` follows MODELS.md naming conventions:
      - ``base``     — pre-trained checkpoint, only lightly instruction-tuned.
                       Not recommended for zero-shot eval; usually the strongest
                       base for fine-tuning on a specific task.
      - ``instruct`` — SFT + preference + RL, non-thinking. Good zero-shot and a
                       balanced base for fine-tuning.
      - ``thinking`` — emits a ``<think>…</think>`` reasoning trace. Strong
                       zero-shot, but a poor base for fine-tuning on
                       non-thinking data.
    """

    id: str          # short handle, e.g. "lfm2.5-1.2b-instruct"
    hf_id: str       # HuggingFace repo id, e.g. "LiquidAI/LFM2.5-1.2B-Instruct"
    kind: str        # "base" | "instruct" | "thinking"

    @property
    def good_finetune_base(self) -> bool:
        """Whether this model is a good starting point for fine-tuning.

        Base models are strongest after fine-tuning; instruct models are a
        balanced choice; thinking models are a poor base for non-thinking data.
        """
        return self.kind in ("base", "instruct")


# NOTE: Vision-Language Models (LiquidAI/LFM2.5-VL-450M, LiquidAI/LFM2.5-VL-1.6B)
# are intentionally deferred — MODELS.md adds them "at a later point".
LIQUID_MODELS: list[LiquidModel] = [
    LiquidModel("lfm2.5-230m-base", "LiquidAI/LFM2.5-230M-Base", "base"),
    LiquidModel("lfm2.5-230m", "LiquidAI/LFM2.5-230M", "instruct"),
    LiquidModel("lfm2.5-350m", "LiquidAI/LFM2.5-350M", "instruct"),
    LiquidModel("lfm2.5-350m-base", "LiquidAI/LFM2.5-350M-Base", "base"),
    LiquidModel("lfm2.5-1.2b-instruct", "LiquidAI/LFM2.5-1.2B-Instruct", "instruct"),
    LiquidModel("lfm2.5-1.2b-thinking", "LiquidAI/LFM2.5-1.2B-Thinking", "thinking"),
    LiquidModel("lfm2.5-1.2b-base", "LiquidAI/LFM2.5-1.2B-Base", "base"),
    LiquidModel("lfm2.5-8b-a1b", "LiquidAI/LFM2.5-8B-A1B", "thinking"),
    LiquidModel("lfm2.5-8b-a1b-base", "LiquidAI/LFM2.5-8B-A1B-Base", "base"),
    LiquidModel("lfm2-2.6b-exp", "LiquidAI/LFM2-2.6B-Exp", "instruct"),
    LiquidModel("lfm2-24b-a2b", "LiquidAI/LFM2-24B-A2B", "instruct"),
]


def is_liquid_model_name(name: str | None) -> bool:
    """Return True if *name* refers to a Liquid model.

    Used to steer evaluation of Liquid checkpoints toward the HuggingFace
    inference path (the router.liquid.ai API is retired). Matches, case-
    insensitively:
      - any catalog short ``id`` or HuggingFace ``hf_id``;
      - the ``LiquidAI/`` HF-org prefix (covers VLMs / future models too);
      - the legacy ``lfm`` short-name prefix (e.g. ``lfm2.5-1.2b-instruct``).

    Pool/utility names (``small``, ``medium``, ``large``, ``judge:*``,
    ``orchestration``, ``random:*``) are NOT Liquid models — they are
    OpenRouter-backed baselines/judges served by api.lqh.ai.
    """
    if not name:
        return False
    n = name.strip().lower()
    for m in LIQUID_MODELS:
        if n == m.id.lower() or n == m.hf_id.lower():
            return True
    return n.startswith("liquidai/") or n.startswith("lfm")


def format_catalog() -> str:
    """Render the catalog as an aligned table for the ``list_models`` tool."""
    lines = ["Liquid AI model catalog (evaluate via eval_hf_model / start_local_eval):\n"]
    lines.append(f"{'Model ID':<24} {'Kind':<9} {'Finetune base':<13} {'HuggingFace ID'}")
    lines.append("-" * 92)
    for m in LIQUID_MODELS:
        ft = "yes" if m.good_finetune_base else "no"
        lines.append(f"{m.id:<24} {m.kind:<9} {ft:<13} {m.hf_id}")
    lines.append("")
    lines.append(SIZE_RECOMMENDATION)
    return "\n".join(lines)
