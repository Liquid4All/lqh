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


def normalize_sources(
    spec: str | list[Any],
    *,
    allow_repeat: bool,
) -> list[dict[str, Any]]:
    """Normalize a ``dataset``/``eval_dataset`` config value to a list of
    source entries.

    Accepts three forms (so legacy single-string configs keep working):

    - a bare string ``"datasets/x/data.parquet"`` → one source;
    - a list of strings → one source each;
    - a list of ``{"path": ..., "repeat": N}`` objects → ``repeat``-weighted
      sources.

    A string that is itself a JSON array (``'["datasets/a", "datasets/b"]'``)
    is decoded first. The tool schema declares ``oneOf [string, array]``, but
    smaller pool models routinely emit the array JSON-encoded inside the string
    field; treating that as a path produced a baffling "not found at
    ["datasets/a", "datasets/b"]/data.parquet" and cost one customer a third of
    their training data (feedback item 47). No real path starts with ``[``, so
    the repair cannot swallow a legitimate one.

    Returns ``[{"path": str, "repeat": int, "source": str}, ...]``. ``repeat``
    defaults to 1 and is forced to 1 when *allow_repeat* is False (eval
    sources, where over-sampling would only distort the score). ``source`` is
    a stable label derived from the parent directory name, disambiguated on
    collision (``name``, ``name_2``, ...) so per-source eval summaries never
    overwrite each other.
    """
    if isinstance(spec, str):
        raw_items: list[Any] = [spec]
        if spec.lstrip().startswith("["):
            try:
                decoded = json.loads(spec)
            except json.JSONDecodeError:
                pass  # Malformed — fall through and fail as a path, as before.
            else:
                if isinstance(decoded, list):
                    raw_items = decoded
    elif isinstance(spec, list):
        raw_items = spec
    else:
        raise ValueError(
            f"dataset source spec must be a string or list, got {type(spec).__name__}"
        )

    if not raw_items:
        # An empty list is not "no sources requested", it is a caller that lost
        # its paths — including the `"[]"` shape the decode above produces.
        # Returning [] would train (or eval) on nothing, silently.
        raise ValueError("dataset source list is empty")

    entries: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            path, repeat = item, 1
        elif isinstance(item, dict):
            path = item.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"dataset source object missing 'path': {item!r}")
            # Eval sources are unweighted by design — surface a `repeat` on an
            # eval source as an error rather than silently dropping it, so a
            # caller who thought it mattered finds out.
            if not allow_repeat and "repeat" in item:
                raise ValueError(
                    "'repeat' is not allowed on eval sources — eval is unweighted "
                    f"(each source contributes equally): {item!r}"
                )
            repeat = item.get("repeat", 1)
        else:
            raise ValueError(
                f"dataset source must be a string or object, got {type(item).__name__}"
            )

        if not allow_repeat:
            repeat = 1
        else:
            if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
                raise ValueError(
                    f"dataset source 'repeat' must be an integer >= 1, got {repeat!r}"
                )

        entries.append({"path": path, "repeat": repeat})

    # Derive stable, collision-free source labels from the parent dir name.
    seen: dict[str, int] = {}
    for entry in entries:
        base = Path(entry["path"]).parent.name or Path(entry["path"]).stem
        if base in seen:
            seen[base] += 1
            label = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
            label = base
        entry["source"] = label

    return entries


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


def load_chatml_datasets(
    sources: str | list[Any],
) -> list[list[dict[str, str]]]:
    """Load and concatenate ChatML conversations from one or more sources.

    Normalizes *sources* via :func:`normalize_sources` (``allow_repeat=True``),
    loads each via :func:`load_chatml_dataset`, repeats each source ``repeat``
    times, and returns the concatenation. A single-string argument reproduces
    :func:`load_chatml_dataset` exactly (one source, repeat 1).
    """
    out: list[list[dict[str, str]]] = []
    for entry in normalize_sources(sources, allow_repeat=True):
        convos = load_chatml_dataset(entry["path"])
        for _ in range(entry["repeat"]):
            out.extend(convos)
    return out


def load_chatml_datasets_with_tools(
    sources: str | list[Any],
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]] | None]]:
    """Plural variant of :func:`load_chatml_dataset_with_tools`.

    Concatenates conversations and per-sample tools across one or more
    sources, applying each source's integer ``repeat`` factor to both. A
    single-string argument reproduces :func:`load_chatml_dataset_with_tools`.
    """
    all_convos: list[list[dict[str, Any]]] = []
    all_tools: list[list[dict[str, Any]] | None] = []
    for entry in normalize_sources(sources, allow_repeat=True):
        convos, tools = load_chatml_dataset_with_tools(entry["path"])
        for _ in range(entry["repeat"]):
            all_convos.extend(convos)
            all_tools.extend(tools)
    return all_convos, all_tools


def load_eval_sources(
    sources: str | list[Any],
) -> list[tuple[str, list[list[dict[str, str]]]]]:
    """Load eval sources, kept DISTINCT, as ``[(source_label, conversations)]``.

    Used by the eval path to tag predictions with their source so they can be
    judge-scored separately. ``repeat`` is forced to 1 — over-sampling eval
    data would only distort the score.
    """
    result: list[tuple[str, list[list[dict[str, str]]]]] = []
    for entry in normalize_sources(sources, allow_repeat=False):
        result.append((entry["source"], load_chatml_dataset(entry["path"])))
    return result


def load_eval_sources_with_tools(
    sources: str | list[Any],
) -> tuple[
    list[list[dict[str, Any]]],
    list[list[dict[str, Any]] | None],
    list[str],
]:
    """Load one or more (eval) sources flattened, returning conversations,
    per-sample tools, and a parallel per-sample ``source`` label.

    ``repeat`` is forced to 1 (generation/eval doesn't benefit from
    over-sampling). A single-string argument yields one source whose label is
    its parent-dir name. Used by the infer prediction loop so eval-of-best
    predictions carry the source tag the per-source judge scoring needs.
    """
    convos: list[list[dict[str, Any]]] = []
    tools: list[list[dict[str, Any]] | None] = []
    sources_per_sample: list[str] = []
    for entry in normalize_sources(sources, allow_repeat=False):
        c, t = load_chatml_dataset_with_tools(entry["path"])
        convos.extend(c)
        tools.extend(t)
        sources_per_sample.extend([entry["source"]] * len(c))
    return convos, tools, sources_per_sample


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
