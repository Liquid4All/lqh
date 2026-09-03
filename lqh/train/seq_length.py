"""Submit-time estimate of a text-SFT dataset's longest row, in tokens.

The client sizes ``training.max_seq_length`` from the data before a run is
submitted, so the backend planner can pick a GPU that fits it (the planner
reads ``training.max_seq_length``; see backend/internal/cloud/planner.go). The
trainer then re-measures the exact value with the real chat template
(``lqh.train.data_utils.tokenized_row_lengths``) before its calibration probe
runs, so this estimate only has to be close — a little high is fine, and the
character fallback is deliberately so.

Constraints that shape this module:

- The client does not have ``transformers`` (a ``train`` extra), only the
  Rust ``tokenizers`` wheel, so it cannot render the chat template. Message
  contents are tokenized on their own and a fixed per-message / per-
  conversation overhead stands in for the template's role and separator
  tokens (LFM2.5's template spends ~4–5 tokens per message; 8 is used).
- Datasets can be large. The parquet is streamed in batches and never
  materialised, matching the "metadata only" contract the submit path keeps
  elsewhere.
- It must never fail a submit: any problem obtaining the tokenizer (offline,
  gated repo, local checkpoint without ``tokenizer.json``) falls back to a
  characters-per-token estimate, and any problem reading a file yields a
  zero-row estimate for that file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lqh.train.defaults import MAX_SEQ_LENGTH_CEILING

# Template tokens the tokenizer-only pass cannot see: role header, end-of-turn
# marker and newlines per message; BOS plus a default system prompt per
# conversation (LFM2.5's template injects one when none is given). Generous
# on purpose: the estimate must err HIGH, because the GPU is chosen from it
# and the trainer's exact value may only safely come in at or below it.
PER_MESSAGE_OVERHEAD = 8
PER_CONVERSATION_OVERHEAD = 32

# Fallback when no tokenizer is available. LFM2.5's tokenizer averages ~4
# characters per token on English prose; 3 over-estimates prose by ~25%,
# which costs at most a larger GPU tier at the margin and never an OOM.
# Non-ASCII text (CJK, emoji, accented scripts) tokenizes far denser — often
# one token per character — so those characters are counted one-to-one
# instead of being divided, otherwise a CJK dataset would be under-estimated
# threefold and land on too small a card.
CHARS_PER_TOKEN = 3

_BATCH_ROWS = 256


def _chars_estimate(texts: list[str]) -> int:
    ascii_chars = 0
    other_chars = 0
    for t in texts:
        n_ascii = sum(1 for ch in t if ord(ch) < 128)
        ascii_chars += n_ascii
        other_chars += len(t) - n_ascii
    return ascii_chars // CHARS_PER_TOKEN + other_chars


@dataclass(frozen=True)
class SeqEstimate:
    """Longest-row estimate over one or more parquet datasets."""

    longest_tokens: int
    rows: int
    rows_over_ceiling: int
    # "tokenizer" when a real tokenizer counted, "chars" for the fallback.
    source: str


def _texts_of_row(messages: Any, tools: Any) -> list[str]:
    """Every string the chat template would render for one row.

    Message content may be a string or a list of parts (vision-style
    ``{"type": "text", "text": ...}`` entries; image parts contribute no
    text). Tool calls and tool definitions are rendered as JSON, which is
    what the template does with them too.
    """
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except ValueError:
            return [messages]
    texts: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        calls = msg.get("tool_calls")
        if calls:
            texts.append(json.dumps(calls) if not isinstance(calls, str) else calls)
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str):
            texts.append(reasoning)
    if tools:
        texts.append(tools if isinstance(tools, str) else json.dumps(tools))
    return texts


def _row_token_count(
    messages: Any, tools: Any, tokenizer: Any | None,
) -> int:
    texts = _texts_of_row(messages, tools)
    n_messages = 0
    if isinstance(messages, str):
        try:
            n_messages = len(json.loads(messages))
        except ValueError:
            n_messages = 1
    elif isinstance(messages, list):
        n_messages = len(messages)
    overhead = PER_CONVERSATION_OVERHEAD + PER_MESSAGE_OVERHEAD * max(1, n_messages)
    if tokenizer is not None:
        try:
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            return overhead + sum(len(e.ids) for e in encodings)
        except Exception:  # noqa: BLE001 — fall through to the char estimate
            pass
    return overhead + _chars_estimate(texts)


def _iter_rows(path: Path) -> Iterable[tuple[Any, Any]]:
    """Yield ``(messages, tools)`` per row, streaming the parquet."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    columns = [c for c in ("messages", "tools") if c in pf.schema_arrow.names]
    if "messages" not in columns:
        return
    for batch in pf.iter_batches(batch_size=_BATCH_ROWS, columns=columns):
        cols = batch.to_pydict()
        msgs = cols.get("messages") or []
        tools = cols.get("tools") or [None] * len(msgs)
        for m, t in zip(msgs, tools):
            yield m, t


def _load_fast_tokenizer(base_model: str, project_dir: Path | None) -> Any | None:
    """A ``tokenizers.Tokenizer`` for *base_model*, or None.

    Local checkpoint directories are checked for ``tokenizer.json`` first
    (the trainers save one next to the weights); otherwise the file is
    fetched from the Hub through ``huggingface_hub`` (disk-cached, so only
    the first submit per base model touches the network).
    """
    try:
        from tokenizers import Tokenizer
    except Exception:  # noqa: BLE001 — optional at import time, never fatal
        return None
    name = (base_model or "").strip()
    if not name:
        return None
    candidates: list[Path] = []
    if project_dir is not None:
        candidates.append(Path(project_dir) / name)
    candidates.append(Path(name))
    for cand in candidates:
        try:
            tok_file = cand / "tokenizer.json"
            if tok_file.is_file():
                return Tokenizer.from_file(str(tok_file))
        except (OSError, ValueError):
            continue
    if "/" not in name or any(name.startswith(p) for p in ("./", "../", "/")):
        return None
    try:
        path = _hub_tokenizer_file(name, project_dir)
        return Tokenizer.from_file(path) if path else None
    except Exception:  # noqa: BLE001 — offline, gated, missing: use the fallback
        return None


def _hub_tokenizer_file(repo_id: str, project_dir: Path | None) -> str | None:
    """Path to the Hub's ``tokenizer.json`` for *repo_id* (disk-cached).

    Separate so the unit suite can stub the one network call.
    """
    from huggingface_hub import hf_hub_download

    from lqh.hf_token import local_hf_token

    return hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        token=local_hf_token(project_dir),
    )


def estimate_longest_row_tokens(
    paths: Iterable[str | Path],
    *,
    base_model: str,
    project_dir: Path | None,
    ceiling: int = MAX_SEQ_LENGTH_CEILING,
) -> SeqEstimate:
    """Longest row (tokens) across *paths* and how many rows exceed *ceiling*.

    Never raises: unreadable files count as empty, and a missing tokenizer
    degrades to the character estimate (``source="chars"``).
    """
    tokenizer = _load_fast_tokenizer(base_model, project_dir)
    longest = 0
    rows = 0
    over = 0
    for p in paths:
        try:
            for messages, tools in _iter_rows(Path(p)):
                n = _row_token_count(messages, tools, tokenizer)
                rows += 1
                if n > longest:
                    longest = n
                if n > ceiling:
                    over += 1
        except Exception:  # noqa: BLE001 — a bad file must not fail the submit
            continue
    return SeqEstimate(
        longest_tokens=longest,
        rows=rows,
        rows_over_ceiling=over,
        source="tokenizer" if tokenizer is not None else "chars",
    )


# ---------------------------------------------------------------------------
# Trainer side: the exact pass (kept torch-free so it is unit-testable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeqResolution:
    """Outcome of :func:`resolve_text_seq_length`."""

    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    # False when auto mode was off or the rows could not be measured; the
    # config was then left exactly as submitted.
    changed: bool
    max_seq_length: int
    longest_tokens: int = 0
    dropped_train: int = 0
    dropped_eval: int = 0


def resolve_text_seq_length(
    training_cfg: dict[str, Any],
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    model_max_positions: int | None = None,
    log: Any = print,
) -> SeqResolution:
    """Replace the submit-time sequence-length estimate with the exact value.

    Text SFT derives ``training.max_seq_length`` from its data (see
    :func:`lqh.train.defaults.derived_seq_length`). The client only estimates
    it — it has no chat template — so the trainer re-measures every row through
    the real template and overwrites the value before the calibration probe
    runs, which is what makes the probe an exact fit test for this run. Rows
    longer than the limit are dropped, not truncated: truncation removes the
    assistant turn that carries the training signal.

    The limit is additionally capped at *model_max_positions* (the model's
    ``max_position_embeddings``) so an arbitrary Hub model with a short context
    window never sees positions it was not trained for.

    Only runs when ``training.auto_seq_length`` is set. A pinned value (the
    LQH_MAX_SEQ_LENGTH expert override, DPO/GRPO/vision, or a hand-written
    config) keeps trl's truncation exactly as before. Mutates *training_cfg*
    in place (``max_seq_length``, ``longest_row_tokens``).
    """
    from lqh.train.data_utils import tokenized_row_lengths
    from lqh.train.defaults import derived_seq_length

    submitted = int(training_cfg.get("max_seq_length") or 0)
    if not bool(training_cfg.get("auto_seq_length", False)):
        return SeqResolution(train_rows, eval_rows, False, submitted)

    try:
        train_lens = tokenized_row_lengths(train_rows, tokenizer)
        eval_lens = tokenized_row_lengths(eval_rows, tokenizer)
    except Exception as exc:  # noqa: BLE001 — keep the estimate, never die here
        log(
            f"  WARNING: could not measure row lengths ({exc}); keeping the "
            f"submitted max_seq_length={submitted}. Rows longer than that will "
            "be truncated by the trainer."
        )
        return SeqResolution(train_rows, eval_rows, False, submitted)

    longest = max(train_lens + eval_lens, default=0)
    exact = derived_seq_length(longest)
    cap_note = ""
    if model_max_positions and 0 < int(model_max_positions) < exact:
        exact = int(model_max_positions)
        cap_note = f"; capped at the model's {exact}-token context window"
    training_cfg["max_seq_length"] = exact
    # For the no-fit message and lineage readers; not consumed by trl.
    training_cfg["longest_row_tokens"] = longest
    log(
        f"  sequence length: {exact} (longest row {longest} tokens; "
        f"submitted estimate {submitted}{cap_note})"
    )

    kept_train = [r for r, n in zip(train_rows, train_lens) if n <= exact]
    kept_eval = [r for r, n in zip(eval_rows, eval_lens) if n <= exact]
    dropped_train = len(train_rows) - len(kept_train)
    dropped_eval = len(eval_rows) - len(kept_eval)
    if dropped_train or dropped_eval:
        log(
            f"  skipped {dropped_train} train / {dropped_eval} eval conversations "
            f"longer than the training limit ({exact} tokens)"
        )
    if eval_rows and not kept_eval:
        log(
            "  WARNING: every eval conversation was longer than the training "
            "limit — this run has NO in-training eval (no eval_loss, no "
            "best-checkpoint selection)."
        )
    if not kept_train:
        raise ValueError(
            "Every training conversation is longer than the training limit "
            f"({exact} tokens; the shortest is {min(train_lens):,}). Shorten or "
            "split the conversations before training."
        )
    return SeqResolution(
        kept_train, kept_eval, True, exact,
        longest_tokens=longest,
        dropped_train=dropped_train, dropped_eval=dropped_eval,
    )
