"""Data loading utilities for the training subprocess.

Converts lqh's parquet ChatML format into the structures expected by
trl's SFTTrainer and DPOTrainer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, TypeVar

import pyarrow.parquet as pq

T = TypeVar("T")


def split_train_eval(
    items: list[T],
    ratio: float,
    *,
    seed: int = 0,
    min_eval: int = 10,
) -> tuple[list[T], list[T]]:
    """Deterministic train/eval split.

    Returns ``(train, eval)``. When ``ratio * len(items) < min_eval``,
    returns ``(items, [])`` — the eval split would be too small to be
    statistically meaningful, so we'd rather train on everything.

    Uses a fixed-seed shuffle so the split is reproducible across runs
    on the same dataset.
    """
    n = len(items)
    eval_size = int(round(n * ratio))
    if eval_size < min_eval:
        return items, []
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[eval_size:], shuffled[:eval_size]


def load_chatml_dataset(
    parquet_path: str | Path,
) -> list[list[dict[str, str]]]:
    """Load a parquet dataset and return a list of ChatML conversations.

    Each conversation is a list of ``{"role": ..., "content": ...}`` dicts.
    The parquet file is expected to have a ``messages`` column containing
    JSON-encoded ChatML conversations (the standard lqh format).
    """
    table = pq.read_table(str(parquet_path))
    messages_col = table.column("messages")

    conversations: list[list[dict[str, str]]] = []
    for i in range(len(table)):
        raw = messages_col[i].as_py()
        msgs = json.loads(raw) if isinstance(raw, str) else raw
        conversations.append(msgs)

    return conversations


def load_chatml_dataset_with_tools(
    parquet_path: str | Path,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]] | None]]:
    """Load a parquet dataset returning conversations and per-sample tools.

    Returns ``(conversations, tools_per_sample)``.  Works with parquet
    files that lack a ``tools`` column (returns ``None`` for each sample).
    """
    table = pq.read_table(str(parquet_path))
    messages_col = table.column("messages")
    has_tools = "tools" in table.column_names
    tools_col = table.column("tools") if has_tools else None

    conversations: list[list[dict[str, Any]]] = []
    tools_list: list[list[dict[str, Any]] | None] = []

    for i in range(len(table)):
        raw = messages_col[i].as_py()
        conversations.append(json.loads(raw) if isinstance(raw, str) else raw)

        if tools_col is not None:
            raw_tools = tools_col[i].as_py()
            tools_list.append(
                json.loads(raw_tools) if isinstance(raw_tools, str) and raw_tools else None
            )
        else:
            tools_list.append(None)

    return conversations, tools_list


def chatml_to_sft_dataset(
    conversations: list[list[dict[str, str]]],
    tools_per_sample: list[list[dict[str, Any]] | None] | None = None,
) -> list[dict[str, Any]]:
    """Convert ChatML conversations to trl SFTTrainer format.

    SFTTrainer with ``packing=False`` expects a list of dicts with a
    ``"messages"`` key containing the ChatML list directly (not JSON-encoded).

    When *tools_per_sample* is provided, entries with tool definitions
    include a ``"tools"`` key alongside ``"messages"`` so the tokenizer's
    ``apply_chat_template(tools=...)`` can use them.

    Returns a list suitable for ``datasets.Dataset.from_list()``.
    """
    result: list[dict[str, Any]] = []
    for i, conv in enumerate(conversations):
        entry: dict[str, Any] = {"messages": conv}
        if tools_per_sample is not None and i < len(tools_per_sample):
            tools = tools_per_sample[i]
            if tools is not None:
                entry["tools"] = tools
        result.append(entry)
    return result


def chatml_to_dpo_dataset(
    preferences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert preference pairs to trl DPOTrainer format.

    Each entry in *preferences* is expected to have:
    - ``"prompt"`` — the ChatML messages up to (but not including) the
      final assistant turn
    - ``"chosen"`` — the preferred assistant response (string)
    - ``"rejected"`` — the dispreferred assistant response (string)

    Returns a list suitable for ``datasets.Dataset.from_list()``.
    """
    return [
        {
            "prompt": pref["prompt"],
            "chosen": pref["chosen"],
            "rejected": pref["rejected"],
        }
        for pref in preferences
    ]


def load_preferences_parquet(
    parquet_path: str | Path,
) -> list[dict[str, Any]]:
    """Load a preferences.parquet file written by the main process.

    Expected columns: ``prompt`` (JSON-encoded ChatML list), ``chosen``
    (string), ``rejected`` (string).
    """
    table = pq.read_table(str(parquet_path))
    result: list[dict[str, Any]] = []
    for i in range(len(table)):
        prompt_raw = table.column("prompt")[i].as_py()
        prompt = json.loads(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
        result.append(
            {
                "prompt": prompt,
                "chosen": table.column("chosen")[i].as_py(),
                "rejected": table.column("rejected")[i].as_py(),
            }
        )
    return result
