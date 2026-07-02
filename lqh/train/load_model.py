"""Unified model loading for HF Hub IDs, merged dirs, and PEFT adapter dirs.

A "model path" in lqh configs (``base_model`` in sft / dpo / infer) can now
be any of three things, and downstream consumers should not care which:

- ``"hub"``     — a HF Hub id like ``"LiquidAI/LFM2-1.2B"`` (no local dir).
- ``"merged"``  — a local dir containing ``config.json`` + weights.
- ``"adapter"`` — a local dir containing ``adapter_config.json`` and
                  adapter weights (typically ``adapter_model.safetensors``).

This module classifies a path and dispatches to the right ``from_pretrained``
chain. Adapter dirs are produced by :mod:`lqh.train.sft` when
``lora.merge=False`` — the artifact is ~tens of MB instead of multi-GB
merged model, which sidesteps publish-time tar OOMs on resource-bounded
sandboxes.

Backwards compatibility: anything that worked before (hub id, merged dir)
keeps working with no config change. Detection is automatic.

Reference-model gotcha (for DPO): when starting from an adapter dir, both
the policy and the reference model must be loaded through
:func:`load_for_training` so they share the same effective starting point
(base + pre-existing adapter merged in). Loading the reference from a
bare base id while the policy starts from an adapter dir produces wrong
KL — the older DPO code path silently did this.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

ModelKind = Literal["hub", "merged", "adapter"]
Modality = Literal["text", "vision"]

__all__ = [
    "ModelKind",
    "Modality",
    "detect_kind",
    "detect_modality",
    "resolve_base_model",
    "load_for_inference",
    "load_for_training",
]


def detect_kind(path_or_id: str) -> ModelKind:
    """Classify a model reference as hub / merged / adapter.

    Anything that isn't an existing directory is treated as a hub id
    (the caller will get a clean HF download error if the id is bogus).
    """
    p = Path(path_or_id)
    if not p.exists() or not p.is_dir():
        return "hub"
    if (p / "adapter_config.json").is_file():
        return "adapter"
    if (p / "config.json").is_file():
        return "merged"
    # A dir without either — most likely an empty / corrupted save.
    # Fall back to "merged" so the AutoModel path raises a clear
    # FileNotFoundError on its own terms instead of us masking it.
    return "merged"


def resolve_base_model(adapter_dir: str, override: str | None = None) -> str:
    """Find the base model for an adapter dir.

    ``override`` wins when set (lets callers pin a base even when the
    adapter was trained against a hub id that's since moved). Otherwise
    reads ``adapter_config.json["base_model_name_or_path"]``.

    Raises ``ValueError`` with a clear message if neither is available.
    """
    if override:
        return override
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        raise ValueError(
            f"{adapter_dir} is not an adapter dir (no adapter_config.json); "
            f"cannot resolve base model. Pass base_override= explicitly."
        )
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{cfg_path}: invalid JSON: {exc}") from exc
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"{cfg_path} has no 'base_model_name_or_path'; "
            f"pass base_override= explicitly."
        )
    return str(base)


def detect_modality(path_or_id: str, *, base_override: str | None = None) -> Modality:
    """Classify a model reference as text or vision (image-text-to-text).

    Reads the HF ``AutoConfig`` for hub ids and merged dirs; adapter dirs
    resolve their base first (the adapter_config carries no architecture
    info). Vision iff the config declares a ``vision_config`` (the LFM-VL /
    generic VLM convention) or an ``lfm2*vl``-family ``model_type``.
    """
    ref = path_or_id
    if detect_kind(path_or_id) == "adapter":
        ref = resolve_base_model(path_or_id, base_override)

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(ref)
    model_type = str(getattr(cfg, "model_type", "") or "").lower().replace("_", "-")
    if "vl" in model_type.split("-") or getattr(cfg, "vision_config", None) is not None:
        return "vision"
    return "text"


def _model_cls(modality: Modality):
    """The AutoModel class for a modality."""
    if modality == "vision":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _load_processor(
    primary: str, fallback: str, *, max_image_tokens: int | None = None
) -> Any:
    """AutoProcessor twin of :func:`_load_tokenizer` (vision models).

    The returned processor goes in the tokenizer slot of the loader return
    values: ``ProcessorMixin`` exposes ``apply_chat_template`` and
    ``decode``/``batch_decode``, and the raw tokenizer is available as
    ``.tokenizer`` for callers that need pad tokens or logits processors.
    """
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {}
    if max_image_tokens is not None:
        kwargs["max_image_tokens"] = max_image_tokens
    try:
        return AutoProcessor.from_pretrained(primary, **kwargs)
    except (OSError, ValueError) as exc:
        if primary == fallback:
            raise
        logger.debug("processor load from %s failed (%s); falling back to %s",
                     primary, exc, fallback)
        return AutoProcessor.from_pretrained(fallback, **kwargs)


def _load_tokenizer(primary: str, fallback: str) -> "PreTrainedTokenizerBase":
    """Try the primary location first; fall back to the secondary on
    failure. Adapter dirs from SFT include the tokenizer, but some PEFT
    adapter dirs in the wild don't — the base ships its own."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(primary)
    except (OSError, ValueError) as exc:
        if primary == fallback:
            raise
        logger.debug("tokenizer load from %s failed (%s); falling back to %s",
                     primary, exc, fallback)
        return AutoTokenizer.from_pretrained(fallback)


def load_for_inference(
    path_or_id: str,
    *,
    dtype: "torch.dtype | None" = None,
    device_map: "str | dict | None" = "auto",
    base_override: str | None = None,
    modality: Modality | None = None,
    max_image_tokens: int | None = None,
) -> "tuple[PreTrainedModel, PreTrainedTokenizerBase]":
    """Return a ready-to-infer model + tokenizer.

    For hub / merged: a single ``from_pretrained`` call. For adapter: load
    the base, wrap with ``PeftModel``, then ``merge_and_unload`` — the merge
    is transient (no disk write) since the model is already in memory.

    ``modality=None`` auto-detects. Vision models load via
    ``AutoModelForImageTextToText`` and return the ``AutoProcessor`` in the
    tokenizer slot (raw tokenizer at ``.tokenizer``).
    """
    import torch  # local: keep import-time light

    if dtype is None:
        dtype = torch.bfloat16
    if modality is None:
        modality = detect_modality(path_or_id, base_override=base_override)
    model_cls = _model_cls(modality)

    def _tok(primary: str, fallback: str) -> Any:
        if modality == "vision":
            return _load_processor(primary, fallback, max_image_tokens=max_image_tokens)
        return _load_tokenizer(primary, fallback)

    kind = detect_kind(path_or_id)
    if kind in ("hub", "merged"):
        model = model_cls.from_pretrained(
            path_or_id, dtype=dtype, device_map=device_map,
        )
        tokenizer = _tok(path_or_id, path_or_id)
        return model, tokenizer

    # adapter
    from peft import PeftModel

    base = resolve_base_model(path_or_id, base_override)
    logger.info("load_for_inference: adapter %s on base %s (transient merge)",
                path_or_id, base)
    base_model = model_cls.from_pretrained(
        base, dtype=dtype, device_map=device_map,
    )
    wrapped = PeftModel.from_pretrained(base_model, path_or_id)
    merged = wrapped.merge_and_unload()
    tokenizer = _tok(path_or_id, base)
    return merged, tokenizer


def load_for_training(
    path_or_id: str,
    *,
    dtype: "torch.dtype | None" = None,
    device_map: "str | dict | None" = "auto",
    base_override: str | None = None,
    merge_before_attach: bool = True,
    modality: Modality | None = None,
    max_image_tokens: int | None = None,
) -> "tuple[PreTrainedModel, PreTrainedTokenizerBase, str]":
    """Like ``load_for_inference`` but returns the resolved base id too.

    Returns ``(model, tokenizer, effective_base_id)``.

    The third return is what callers should use to load a reference copy
    (DPO needs this — see module docstring). For hub/merged paths,
    ``effective_base_id == path_or_id``. For adapter paths, it's the
    underlying base. When ``merge_before_attach=True`` (the default and
    only sensible choice for current callers), the returned model is the
    fully-merged result and the caller can attach a fresh ``LoraConfig``
    on top without PEFT-on-PEFT awkwardness.
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    if modality is None:
        modality = detect_modality(path_or_id, base_override=base_override)
    model_cls = _model_cls(modality)

    def _tok(primary: str, fallback: str) -> Any:
        if modality == "vision":
            return _load_processor(primary, fallback, max_image_tokens=max_image_tokens)
        return _load_tokenizer(primary, fallback)

    kind = detect_kind(path_or_id)
    if kind in ("hub", "merged"):
        model = model_cls.from_pretrained(
            path_or_id, dtype=dtype, device_map=device_map,
        )
        tokenizer = _tok(path_or_id, path_or_id)
        return model, tokenizer, path_or_id

    # adapter
    from peft import PeftModel

    base = resolve_base_model(path_or_id, base_override)
    base_model = model_cls.from_pretrained(
        base, dtype=dtype, device_map=device_map,
    )
    wrapped = PeftModel.from_pretrained(base_model, path_or_id)
    if merge_before_attach:
        model = wrapped.merge_and_unload()
    else:
        # Caller wants the live PeftModel — they accept responsibility
        # for any PEFT-on-PEFT gymnastics that follow.
        model = wrapped
    tokenizer = _tok(path_or_id, base)
    return model, tokenizer, base
