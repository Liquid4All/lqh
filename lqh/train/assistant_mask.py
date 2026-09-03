"""Assistant-turn-only loss: text SFT always trains on assistant tokens alone.

There is no switch. System, user and tool turns are context the model reads,
not text it should learn to produce, so trl's ``assistant_only_loss`` is on
for every text SFT run. The one thing that can go wrong is the base model's
chat template: transformers derives the assistant mask from a
``{% generation %}`` block around the assistant reply, and a template without
one yields an all-zero mask. That is caught at two points, both before any
training step ("fail quickly": a user reports the model and the template is
fixed upstream; the trainer never silently falls back to full-sequence loss):

- :func:`read_chat_template` + :func:`template_lacks_generation_block` —
  client side (``handle_start_training``), before a cloud job is provisioned.
  The CLI has no ``transformers``, so it reads the template *text* (local
  checkpoint dir or the Hub) and looks for the tag. Undeterminable (offline,
  gated, adapter dir without tokenizer files) means "let the trainer decide",
  never "assume fine".
- :func:`require_assistant_mask_support` — the trainer, right after the
  tokenizer loads: renders a two-turn probe through the real template with
  ``return_assistant_tokens_mask=True`` and raises when no token is marked.

Vision SFT is excluded: trl 1.0 refuses ``assistant_only_loss`` for VLMs, so
those runs keep full-sequence loss (``lqh/train/sft.py`` does not pass the
flag there).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ``{% generation %}`` with or without whitespace control (``{%- generation -%}``).
_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")

# Two turns are enough: an assistant turn is present, so a template that marks
# assistant tokens must return at least one 1 in the mask.
_PROBE_MESSAGES = [
    {"role": "user", "content": "ping"},
    {"role": "assistant", "content": "pong"},
]

NO_GENERATION_BLOCK = "its chat template has no {% generation %} block"


def unsupported_message(base_model: str, reason: str) -> str:
    """The one error text both the launch check and the trainer raise."""
    return (
        f"{base_model} cannot be fine-tuned: lqh computes the SFT loss on "
        f"assistant turns only, and {reason}. The model's chat template needs "
        "a {% generation %} ... {% endgeneration %} block around the assistant "
        "reply (every LFM2.5 instruct/thinking template has one; the -Base "
        "checkpoints do not). Pick a base model whose template marks its "
        "assistant turns, and report this model so its template can be fixed "
        "upstream."
    )


# ---------------------------------------------------------------------------
# Client side: template text, no transformers
# ---------------------------------------------------------------------------


def template_lacks_generation_block(template: str) -> bool:
    return _GENERATION_TAG.search(template) is None


def _template_from_tokenizer_config(text: str) -> str | None:
    """The ``chat_template`` of a ``tokenizer_config.json`` body.

    Older configs store a list of ``{"name", "template"}`` entries; all of
    them are joined so a generation block in any variant counts.
    """
    try:
        cfg = json.loads(text)
    except ValueError:
        return None
    tpl = cfg.get("chat_template") if isinstance(cfg, dict) else None
    if isinstance(tpl, str):
        return tpl
    if isinstance(tpl, list):
        parts = [
            t.get("template") for t in tpl
            if isinstance(t, dict) and isinstance(t.get("template"), str)
        ]
        return "\n".join(parts) if parts else None
    return None


def _template_from_dir(directory: Path) -> str | None:
    jinja = directory / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    cfg = directory / "tokenizer_config.json"
    if cfg.is_file():
        return _template_from_tokenizer_config(cfg.read_text(encoding="utf-8"))
    return None


def _hub_file(repo_id: str, filename: str, project_dir: Path | None) -> str | None:
    """Path to a Hub file for *repo_id* (disk-cached); raises when absent.

    Separate so the unit suite can stub the one network call.
    """
    from huggingface_hub import hf_hub_download

    from lqh.hf_token import local_hf_token

    return hf_hub_download(
        repo_id=repo_id, filename=filename, token=local_hf_token(project_dir)
    )


def read_chat_template(base_model: str, project_dir: Path | None) -> str | None:
    """The chat template text of *base_model*, or None if it cannot be read here.

    A local checkpoint directory (absolute, or relative to *project_dir*) is
    read directly; anything else is a Hub id. Every failure — offline, gated
    repo, a directory without tokenizer files — returns None: the launch check
    then defers to the trainer instead of guessing.
    """
    try:
        local = Path(base_model)
        if not local.is_absolute() and project_dir is not None:
            local = project_dir / base_model
        if local.is_dir():
            return _template_from_dir(local)
    except (OSError, ValueError):
        return None
    for filename in ("chat_template.jinja", "tokenizer_config.json"):
        text = _hub_text(base_model, filename, project_dir)
        if text is None:
            continue
        if filename.endswith(".jinja"):
            return text
        return _template_from_tokenizer_config(text)
    return None


def _hub_text(repo_id: str, filename: str, project_dir: Path | None) -> str | None:
    try:
        path = _hub_file(repo_id, filename, project_dir)
        return Path(path).read_text(encoding="utf-8") if path else None
    except Exception:  # noqa: BLE001 — missing file, offline, gated: unreadable
        return None


# ---------------------------------------------------------------------------
# Trainer side: the real tokenizer
# ---------------------------------------------------------------------------


def render_row(tokenizer: Any, row: dict[str, Any], **kwargs: Any) -> Any:
    """``apply_chat_template`` for one :func:`chatml_to_sft_dataset` row.

    Decodes the JSON-encoded ``tools`` column the way trl does. Older
    tokenizers without a ``tools`` kwarg are retried without it, but only
    when the row has no tools to lose; any other TypeError is a real template
    failure trl would hit too.
    """
    tools = row.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools) if tools else None
    try:
        return tokenizer.apply_chat_template(
            row["messages"], tools=tools, tokenize=True, **kwargs
        )
    except TypeError as exc:
        if "tools" not in str(exc) or tools is not None:
            raise
        return tokenizer.apply_chat_template(row["messages"], tokenize=True, **kwargs)


def _assistant_mask(encoded: Any) -> list[int] | None:
    try:
        mask = encoded["assistant_masks"]
    except (KeyError, TypeError, IndexError):
        return None
    return list(mask) if mask is not None else None


def assistant_mask_unsupported(tokenizer: Any) -> str | None:
    """Why *tokenizer* cannot mark assistant tokens, or None if it can."""
    try:
        encoded = tokenizer.apply_chat_template(
            _PROBE_MESSAGES,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception as exc:  # noqa: BLE001 — any template failure is the reason
        detail = str(exc).strip() or exc.__class__.__name__
        return f"rendering its chat template failed: {detail}"
    mask = _assistant_mask(encoded)
    if not mask or 1 not in mask:
        return NO_GENERATION_BLOCK
    return None


def require_assistant_mask_support(tokenizer: Any, base_model: str) -> None:
    """Raise ``ValueError`` unless *tokenizer*'s template marks assistant turns."""
    reason = assistant_mask_unsupported(tokenizer)
    if reason:
        raise ValueError(unsupported_message(base_model, reason))


def drop_rows_without_assistant_labels(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Drop rows whose assistant tokens all sit past *max_length*.

    trl truncates in the collator, after the mask is built, so a row with a
    prompt longer than the limit reaches the loss with every label masked
    out: a zero-or-NaN batch instead of a rejected row. The data-derived
    sequence length (``lqh.train.seq_length``) already drops over-long rows,
    which makes this a no-op there; it matters for a pinned
    ``max_seq_length``. One extra tokenization pass (~0.7 ms/row on the
    LFM2.5 fast tokenizer).
    """
    if not max_length:
        return rows, 0
    kept = []
    for row in rows:
        mask = _assistant_mask(
            render_row(
                tokenizer, row, return_dict=True, return_assistant_tokens_mask=True
            )
        )
        if mask and 1 in mask[:max_length]:
            kept.append(row)
    return kept, len(rows) - len(kept)
