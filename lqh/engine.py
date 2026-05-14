"""Pipeline execution engine.

Dynamically loads pipeline scripts, runs them with concurrency and retries,
and writes results as parquet datasets.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI

from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    GenerationError,
    Pipeline,
)

__all__ = [
    "load_pipeline",
    "load_dataset_with_tools",
    "EngineResult",
    "run_pipeline",
]

logger = logging.getLogger(__name__)


def load_pipeline(script_path: Path) -> type[Pipeline]:
    """Dynamically load a pipeline script and return its Pipeline subclass.

    The script must contain exactly one concrete ``Pipeline`` subclass.
    Raises ``ValueError`` if zero or more than one are found.
    """
    spec = importlib.util.spec_from_file_location(
        f"lqh_pipeline_{script_path.stem}", script_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except (ImportError, ModuleNotFoundError) as exc:
        hint = ""
        err_str = str(exc)
        if "data_gen" in err_str or "pipeline" in err_str.lower():
            hint = (
                "\n\nHint: Pipeline files must import from lqh.pipeline, not data_gen:\n"
                "  from lqh.pipeline import Pipeline, ChatMLMessage, Conversation"
            )
        raise ValueError(f"Failed to load {script_path}: {exc}{hint}") from exc

    # Find all concrete Pipeline subclasses defined in the module.
    pipeline_classes: list[type[Pipeline]] = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, Pipeline)
            and obj is not Pipeline
            and obj.__module__ == module.__name__
        ):
            pipeline_classes.append(obj)

    if len(pipeline_classes) == 0:
        raise ValueError(
            f"No Pipeline subclass found in {script_path}. "
            "The file must define exactly one class that inherits from Pipeline."
        )
    if len(pipeline_classes) > 1:
        names = ", ".join(cls.__name__ for cls in pipeline_classes)
        raise ValueError(
            f"Multiple Pipeline subclasses found in {script_path}: {names}. "
            "The file must define exactly one."
        )

    return pipeline_classes[0]


@dataclass
class EngineResult:
    """Summary of a pipeline run."""

    total: int
    succeeded: int
    failed: int
    output_path: Path


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_message(msg: ChatMLMessage) -> dict[str, Any]:
    """Convert a single ChatMLMessage to a JSON-friendly dict."""
    d: dict[str, Any] = {"role": msg.role}

    if msg.content is not None:
        d["content"] = msg.content

    if msg.tools is not None:
        d["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in msg.tools
        ]

    if msg.tool_calls is not None:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id

    if msg.name is not None:
        d["name"] = msg.name

    return d


def _extract_tools(conv: Conversation) -> list[dict[str, Any]] | None:
    """Extract tool definitions from a conversation's messages.

    Scans all messages for ``tools`` fields and collects unique tool
    definitions.  Returns ``None`` if no tools are found.
    """
    tools: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for msg in conv:
        if msg.tools is not None:
            for t in msg.tools:
                if t.name not in seen_names:
                    seen_names.add(t.name)
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    })
    return tools if tools else None


def _serialize_conversation(conv: Conversation) -> dict[str, Any]:
    """Serialize a conversation to a dict suitable for a parquet row.

    Returns a dict with:
    - ``messages``: list of message dicts (role, content, tool_calls, etc.)
    - ``audio``: dict mapping message index (str) to base64-encoded WAV bytes,
      or ``None`` if no messages carry audio.
    - ``tools``: list of tool definitions in OpenAI format, or ``None``.
    """
    messages: list[dict[str, Any]] = []
    audio: dict[str, str] = {}

    for idx, msg in enumerate(conv):
        messages.append(_serialize_message(msg))
        if msg.audio is not None:
            audio[str(idx)] = base64.b64encode(msg.audio).decode("ascii")

    return {
        "messages": messages,
        "audio": audio if audio else None,
        "tools": _extract_tools(conv),
    }


# ---------------------------------------------------------------------------
# Incremental save helpers
# ---------------------------------------------------------------------------


def _append_partial(path: Path, index: int, row: dict[str, str | None]) -> None:
    """Append one completed sample to the partial JSONL file."""
    line = json.dumps({"index": index, **row}, ensure_ascii=False)
    with open(path, "a") as f:
        f.write(line + "\n")


def _load_partial(
    path: Path, total: int
) -> tuple[set[int], list[dict[str, Any] | None], int]:
    """Read partial JSONL, return (done_indices, results, succeeded_count).

    If the meta header's total doesn't match, returns empty state (start fresh).
    Handles truncated last lines and duplicate indices gracefully.
    """
    results: list[dict[str, Any] | None] = [None] * total
    seen: dict[int, dict[str, Any]] = {}

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated last line
        if "_meta" in entry:
            if entry.get("total") != total:
                logger.warning(
                    "Partial file total (%s) doesn't match current (%d), starting fresh",
                    entry.get("total"), total,
                )
                return set(), [None] * total, 0
            continue
        idx = entry.pop("index", None)
        if idx is not None and 0 <= idx < total:
            seen[idx] = entry

    done = set(seen.keys())
    for idx, entry in seen.items():
        # Reconstruct the internal result format (messages as parsed list)
        messages = entry.get("messages")
        audio = entry.get("audio")
        tools_raw = entry.get("tools")
        results[idx] = {
            "messages": json.loads(messages) if isinstance(messages, str) else messages,
            "audio": json.loads(audio) if isinstance(audio, str) and audio else audio,
            "tools": json.loads(tools_raw) if isinstance(tools_raw, str) and tools_raw else tools_raw,
        }

    return done, results, len(done)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

async def run_pipeline(
    script_path: Path,
    num_samples: int,
    output_dir: Path,
    client: AsyncOpenAI,
    *,
    max_retries: int = 3,
    concurrency: int = 5,
    samples_per_item: int = 1,
    validation_instructions: str | None = None,
    on_progress: Callable[[int, int], Any] | None = None,
) -> EngineResult:
    """Execute a pipeline and write results as parquet.

    Parameters
    ----------
    script_path:
        Path to the ``.py`` pipeline file.
    num_samples:
        Maximum number of source items to consume (pure-generation mode) or
        cap on source items (bring-your-data mode).
    output_dir:
        Directory where ``data.parquet`` will be written.
    client:
        Pre-configured ``AsyncOpenAI`` instance pointed at api.lqh.ai.
    max_retries:
        How many times the engine retries a failed sample (fresh instance each
        time) before marking it as permanently failed.
    concurrency:
        Maximum number of samples generated in parallel.
    samples_per_item:
        How many times ``generate()`` is called per source item (only relevant
        in bring-your-data mode).
    validation_instructions:
        Optional text with LLM validation criteria (reserved for future use).
    on_progress:
        Optional callback invoked as ``on_progress(completed, total)`` after
        each sample finishes (success or permanent failure).
    """
    pipeline_cls = load_pipeline(script_path)

    # Determine the work items: list of (input_item | None) to process.
    project_dir = script_path.parent.parent  # data_gen/ -> project root
    source_items = pipeline_cls.source(project_dir)

    work: list[Any]
    if source_items is not None:
        # Bring-your-data mode: consume up to num_samples items, each
        # repeated samples_per_item times.
        raw_items: list[Any] = []
        for item in source_items:
            raw_items.append(item)
            if len(raw_items) >= num_samples:
                break
        work = []
        for item in raw_items:
            for _ in range(samples_per_item):
                work.append(item)
    else:
        # Pure generation mode: num_samples tasks, each with input=None.
        work = [None] * num_samples

    total = len(work)
    results: list[dict[str, Any] | None] = [None] * total
    succeeded = 0
    failed = 0
    completed = 0
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    # Incremental saves: write each completed sample to a JSONL file so
    # progress survives process kills.  On restart, already-done samples
    # are skipped automatically.
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "data.partial.jsonl"
    done_indices: set[int] = set()

    if partial_path.exists():
        done_indices, results, succeeded = _load_partial(partial_path, total)
        completed = len(done_indices)
        if done_indices:
            logger.info("Resuming: %d/%d samples already completed", len(done_indices), total)
    else:
        with open(partial_path, "w") as f:
            f.write(json.dumps({"_meta": True, "total": total}) + "\n")

    # Set when a deterministic code bug is detected — signals all tasks to abort.
    abort_error: Exception | None = None

    async def _run_one(index: int, input_item: Any) -> None:
        nonlocal succeeded, failed, completed, abort_error
        async with sem:
            if abort_error is not None:
                return  # another sample already hit a fatal bug

            result: dict[str, Any] | None = None
            for attempt in range(max_retries + 1):
                instance = pipeline_cls()
                try:
                    # Pass input positionally to tolerate pipelines that
                    # omit the ``input`` parameter from their signature.
                    if input_item is not None:
                        conv = await instance.generate(client, input_item)
                    else:
                        conv = await instance.generate(client)
                    result = _serialize_conversation(conv)
                    break
                except GenerationError as exc:
                    if attempt < max_retries:
                        logger.warning(
                            "Sample %d failed (attempt %d/%d): %s",
                            index, attempt + 1, max_retries + 1, exc,
                        )
                        continue
                    logger.error(
                        "Sample %d permanently failed after %d attempts: %s",
                        index, max_retries + 1, exc,
                    )
                except Exception as exc:
                    # Deterministic code bugs — abort the entire run
                    # immediately so the agent gets the error fast.
                    # Exclude JSONDecodeError (transient: LLM returned bad JSON).
                    import json as _json
                    if isinstance(exc, _json.JSONDecodeError):
                        logger.debug(
                            "Sample %d JSON parse error (attempt %d/%d): %s",
                            index, attempt + 1, max_retries + 1, exc,
                        )
                        if attempt >= max_retries:
                            break
                        continue
                    logger.error(
                        "Sample %d error: %s: %s",
                        index, type(exc).__name__, exc,
                    )
                    if isinstance(exc, (TypeError, AttributeError, NameError,
                                        SyntaxError, ValueError, ImportError)):
                        abort_error = exc
                        return
                    if attempt >= max_retries:
                        break

            async with lock:
                if result is not None:
                    results[index] = result
                    row = {
                        "messages": json.dumps(result["messages"], ensure_ascii=False),
                        "audio": json.dumps(result["audio"], ensure_ascii=False) if result["audio"] is not None else None,
                        "tools": json.dumps(result["tools"], ensure_ascii=False) if result["tools"] is not None else None,
                    }
                    _append_partial(partial_path, index, row)
                    succeeded += 1
                else:
                    failed += 1
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total)

    tasks = [
        asyncio.create_task(_run_one(i, item))
        for i, item in enumerate(work)
        if i not in done_indices
    ]
    await asyncio.gather(*tasks)

    # If a deterministic bug aborted the run, raise it so the caller
    # (tool handler) gets a clear error message immediately.
    if abort_error is not None:
        raise abort_error

    # Build parquet table from successful results.
    rows: list[dict[str, str | None]] = []
    for r in results:
        if r is not None:
            rows.append({
                "messages": json.dumps(r["messages"], ensure_ascii=False),
                "audio": json.dumps(r["audio"], ensure_ascii=False) if r["audio"] is not None else None,
                "tools": json.dumps(r["tools"], ensure_ascii=False) if r.get("tools") is not None else None,
            })

    schema = pa.schema([
        pa.field("messages", pa.string()),
        pa.field("audio", pa.string()),
        pa.field("tools", pa.string()),
    ])

    if rows:
        table = pa.table(
            {
                "messages": [row["messages"] for row in rows],
                "audio": [row["audio"] for row in rows],
                "tools": [row["tools"] for row in rows],
            },
            schema=schema,
        )
    else:
        table = pa.table(
            {"messages": [], "audio": [], "tools": []},
            schema=schema,
        )

    output_path = output_dir / "data.parquet"
    pq.write_table(table, output_path)
    partial_path.unlink(missing_ok=True)

    return EngineResult(
        total=total,
        succeeded=succeeded,
        failed=failed,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset_with_tools(
    parquet_path: Path,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]] | None]]:
    """Load a parquet dataset returning conversations and per-sample tools.

    Returns ``(conversations, tools)`` where each element in *tools* is
    either a list of OpenAI-format tool definitions or ``None``.

    Works with both old parquet files (no ``tools`` column) and new ones.
    """
    table = pq.read_table(str(parquet_path))
    messages_col = table.column("messages")
    has_tools = "tools" in table.column_names
    tools_col = table.column("tools") if has_tools else None

    conversations: list[list[dict[str, Any]]] = []
    tools: list[list[dict[str, Any]] | None] = []

    for i in range(len(table)):
        raw_msgs = messages_col[i].as_py()
        conversations.append(json.loads(raw_msgs) if isinstance(raw_msgs, str) else raw_msgs)

        if tools_col is not None:
            raw_tools = tools_col[i].as_py()
            tools.append(json.loads(raw_tools) if isinstance(raw_tools, str) and raw_tools else None)
        else:
            tools.append(None)

    return conversations, tools
