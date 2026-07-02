"""Vision-language data bridging for VLM (LFM2.5-VL) training and eval.

lqh datasets store images inline in the ChatML ``messages`` as OpenAI
``image_url`` parts with base64 data-URLs. The LFM2.5-VL processor's chat
template instead wants ``{"type": "image", "image": <PIL.Image>}`` parts.
This module bridges the two:

- :func:`split_image_parts` normalizes one conversation at dataset-build
  time — image parts become bare ``{"type": "image"}`` placeholders and the
  *compressed* image bytes move to a parallel list. Arrow then stores
  JPEG/PNG bytes (~100s of KB) instead of decoded pixel tensors (~MBs),
  keeping ``Dataset.from_list`` memory flat.
- :class:`VLMCollator` re-inserts the images as PIL objects lazily, per
  batch, and tokenizes via ``processor.apply_chat_template`` (the Liquid
  TRL recipe's collate_fn).
- :func:`vlm_generate` is the eval-side helper: processor-based generation
  for a single conversation whose messages still carry data-URL parts.

Only imported inside the training subprocess / infer paths — keeps torch,
PIL, and transformers out of the main process import graph.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

__all__ = [
    "conversation_has_images",
    "split_image_parts",
    "chatml_to_vlm_dataset",
    "decode_image",
    "VLMCollator",
    "vlm_generate",
]


def conversation_has_images(conv: list[dict[str, Any]]) -> bool:
    """True if any message carries an ``image_url`` content part."""
    for msg in conv:
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            return True
    return False


def _decode_data_url(url: str) -> bytes:
    """Decode a ``data:image/...;base64,...`` URL into raw image bytes."""
    if not url.startswith("data:"):
        raise ValueError(
            f"expected a data: URL for an image part, got {url[:64]!r}; "
            "remote image URLs are not supported in training datasets"
        )
    try:
        _header, b64 = url.split(",", 1)
        # binascii.Error subclasses ValueError, so one except covers both.
        return base64.b64decode(b64, validate=True)
    except ValueError as exc:
        raise ValueError(f"malformed image data-URL ({exc})") from exc


def split_image_parts(
    conv: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Normalize one ChatML conversation for the VLM chat template.

    Returns ``(normalized_conv, images)`` where every ``image_url`` part is
    replaced (in place, in document order) with ``{"type": "image"}`` and
    *images* holds the corresponding decoded-but-still-compressed bytes.
    The input conversation is not mutated.
    """
    normalized: list[dict[str, Any]] = []
    images: list[bytes] = []
    for msg in conv:
        content = msg.get("content")
        if not isinstance(content, list):
            normalized.append(dict(msg))
            continue
        new_parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                images.append(_decode_data_url(url))
                new_parts.append({"type": "image"})
            else:
                new_parts.append(part)
        new_msg = dict(msg)
        new_msg["content"] = new_parts
        normalized.append(new_msg)
    return normalized, images


def chatml_to_vlm_dataset(
    conversations: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Convert ChatML conversations to VLM trainer rows.

    Rows are ``{"messages": <json str>, "images": [bytes, ...]}``. The
    messages are JSON-encoded (not nested dicts) because Arrow struct
    inference chokes on heterogeneous content values (str vs list of
    parts); the collator decodes them per batch.
    """
    rows: list[dict[str, Any]] = []
    for conv in conversations:
        normalized, images = split_image_parts(conv)
        rows.append({"messages": json.dumps(normalized), "images": images})
    return rows


def decode_image(data: bytes):
    """Decode compressed image bytes into an RGB PIL image."""
    from PIL import Image

    img = Image.open(BytesIO(data))
    img.load()
    return img.convert("RGB")


def _reinsert_images(
    conv: list[dict[str, Any]], images: list[Any]
) -> list[dict[str, Any]]:
    """Replace bare ``{"type": "image"}`` parts with PIL-loaded image parts."""
    it = iter(images)
    out: list[dict[str, Any]] = []
    for msg in conv:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image" and "image" not in part:
                try:
                    img = next(it)
                except StopIteration:
                    raise ValueError(
                        "conversation has more image placeholders than images"
                    ) from None
                parts.append({"type": "image", "image": img})
            else:
                parts.append(part)
        new_msg = dict(msg)
        new_msg["content"] = parts
        out.append(new_msg)
    return out


class VLMCollator:
    """Multimodal collate_fn for ``SFTTrainer`` (Liquid TRL recipe shape).

    Decodes each row's compressed image bytes to PIL lazily (per batch),
    re-inserts them into the message structure, applies the processor's
    chat template with tokenization + padding, and builds ``labels`` from
    ``input_ids`` with pad positions masked to ``-100``.

    Samples whose rendered length exceeds *max_length* are NOT truncated —
    truncation could sever an image-token span, which corrupts training.
    They are dropped with a warning instead (and if a whole batch renders
    over-long, an error is raised so the run fails loudly rather than
    training on nothing).
    """

    def __init__(self, processor: Any, max_length: int | None = None) -> None:
        self.processor = processor
        self.max_length = max_length
        self._dropped = 0

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        convs = []
        for sample in samples:
            msgs = sample["messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            images = [decode_image(b) for b in sample.get("images", [])]
            convs.append(_reinsert_images(msgs, images))

        batch = self.processor.apply_chat_template(
            convs,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        if self.max_length is not None and batch["input_ids"].shape[1] > self.max_length:
            # Re-render sample-by-sample and keep only the ones that fit.
            kept = []
            for conv in convs:
                single = self.processor.apply_chat_template(
                    [conv],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                )
                if single["input_ids"].shape[1] <= self.max_length:
                    kept.append(conv)
                else:
                    self._dropped += 1
            if not kept:
                raise ValueError(
                    f"every sample in the batch renders longer than "
                    f"max_length={self.max_length}; raise training.max_seq_length "
                    f"or lower training.max_image_tokens"
                )
            if len(kept) < len(convs):
                import logging

                logging.getLogger(__name__).warning(
                    "VLMCollator: dropped %d over-long sample(s) this batch "
                    "(%d total this run) — truncating through image tokens "
                    "would corrupt training",
                    len(convs) - len(kept),
                    self._dropped,
                )
            batch = self.processor.apply_chat_template(
                kept,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def vlm_generate(
    model: Any,
    processor: Any,
    prompt_msgs: list[dict[str, Any]],
    *,
    max_new_tokens: int = 1024,
    **generate_kwargs: Any,
) -> str:
    """Greedy processor-based generation for one vision conversation.

    *prompt_msgs* is a ChatML prompt (no trailing assistant turn) whose
    messages may carry ``image_url`` data-URL parts. Extra keyword args
    (e.g. ``prefix_allowed_tokens_fn`` for constrained decoding) are
    forwarded to ``model.generate``. Returns the decoded completion text.
    """
    import torch

    normalized, image_bytes = split_image_parts(prompt_msgs)
    images = [decode_image(b) for b in image_bytes]
    conv = _reinsert_images(normalized, images)

    inputs = processor.apply_chat_template(
        conv,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # The inputs dict carries pixel_values alongside input_ids — the whole
    # thing must move to the model device, not just input_ids.
    inputs = {
        k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **generate_kwargs,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True
    )
