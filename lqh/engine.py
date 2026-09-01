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
import math
from dataclasses import dataclass, field
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
from lqh.sources import hf_dataset_was_used, record_source_paths

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

    # Sample accounting always adds up: `total` is what the caller asked
    # for, `succeeded` is the rows written, and `failed` is the shortfall
    # (`total - succeeded`) — samples that were requested and not
    # delivered. Failures an over-commit spare made up for are therefore
    # NOT counted here; `sample_failures` below has the raw count.
    total: int
    succeeded: int
    failed: int
    output_path: Path
    # Project files the pipeline read via lqh.sources helpers during
    # this run. Recorded so a validated local run doubles as the
    # bundle manifest for a cloud data_gen submit.
    source_paths: list[Path] = field(default_factory=list)
    # Whether the run consumed lqh.sources.hf_dataset — observed usage
    # gates HF-token injection into the cloud sandbox.
    used_hf: bool = False
    # Samples carried over from an interrupted run's partial file. A
    # resumed run's source_paths/used_hf only reflect THIS process, so
    # callers must not treat it as a complete observation (the cloud
    # validation gate skips recording when this is non-zero).
    resumed_samples: int = 0
    # Every work item that failed permanently, over-commit spares
    # included — the pipeline-reliability signal, which `failed` hides
    # once a spare has covered the loss. Always >= `failed`.
    sample_failures: int = 0


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


def _append_failure(path: Path, index: int) -> None:
    """Record a permanently failed sample.

    Only successes are resumable, so this line exists purely to carry the
    failure count across an interruption — without it, a continuation
    reports a clean run when the first attempt failed samples. Written
    with no ``index`` key so older readers skip it instead of mistaking
    it for a completed sample.
    """
    with open(path, "a") as f:
        f.write(json.dumps({"_failed": index}) + "\n")


def _load_partial(
    path: Path,
    total: int,
    digest: str | None = None,
    *,
    header_total: int | None = None,
) -> tuple[set[int], list[dict[str, Any] | None], int, int]:
    """Read partial JSONL: (done_indices, results, succeeded, failures).

    *total* sizes the result list (the full work list, spares included);
    *header_total* is what the ``_meta`` header must say — the requested
    sample count, which is stable across engine versions and independent
    of the over-commit margin. They differ only for over-committed runs.

    If the meta header's total doesn't match — or the header's pipeline
    digest differs from *digest* — returns empty state (start fresh).
    The digest binding matters for the cloud-validation gate: without
    it, an edited pipeline could "complete" a local run by absorbing an
    older version's leftover partial samples without executing them.
    Handles truncated last lines and duplicate indices gracefully.
    """
    expected_total = total if header_total is None else header_total
    results: list[dict[str, Any] | None] = [None] * total
    seen: dict[int, dict[str, Any]] = {}
    failures = 0

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated last line
        if "_meta" in entry:
            if entry.get("total") != expected_total:
                logger.warning(
                    "Partial file total (%s) doesn't match current (%d), starting fresh",
                    entry.get("total"), expected_total,
                )
                return set(), [None] * total, 0, 0
            if digest is not None and entry.get("digest") != digest:
                # Strict: a digest-less legacy header also restarts —
                # resumed samples must be attributable to THIS version.
                logger.warning(
                    "Partial file was written by a different pipeline version, starting fresh",
                )
                return set(), [None] * total, 0, 0
            continue
        if "_failed" in entry:
            failures += 1
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

    return done, results, len(done), failures


def _partial_has_samples(path: Path) -> bool:
    """True if the partial JSONL holds at least one completed sample line."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_meta" not in entry and entry.get("index") is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Straggler handling
# ---------------------------------------------------------------------------

# A run finishes no faster than its slowest sample. A couple of items stuck
# in the retry ladder (each attempt can burn the client's 300 s timeout)
# leaves a 200-sample run sitting at "198/200" for many minutes with an
# otherwise idle worker pool. So queue a few spare samples and stop as soon
# as the requested count is in hand, cancelling whatever is still crawling.
_OVERCOMMIT_MIN_TOTAL = 50   # below this the tail *is* the run — no margin
_OVERCOMMIT_FRACTION = 0.03  # ~3% spares
_OVERCOMMIT_MAX = 10         # cap the wasted spend on large runs

# Watchdog on one generate() attempt. The client bounds each LLM call at
# 300 s and a sample may legitimately chain several, so this sits far
# above any healthy run and exists only to break a wedged pipeline.
_SAMPLE_TIMEOUT_S = 1800.0

# How long to wait for cancelled samples to actually stop before giving
# up on them. Generous — a healthy pipeline unwinds in milliseconds.
_CANCEL_SETTLE_S = 60.0


async def _run_until_enough(
    tasks: list[asyncio.Task[None]], enough: asyncio.Event,
) -> None:
    """Await *tasks*, cutting the run short once *enough* is set.

    Cancelling the stragglers is the point: past the requested count the
    remaining items are spares nobody is waiting for, and a sample deep
    in its retry ladder would otherwise hold the run open for minutes.
    """
    if not tasks:
        return

    gathered = asyncio.gather(*tasks)
    # Read the gather's outcome however it ends — an unread CancelledError
    # makes asyncio log "_GatheringFuture exception was never retrieved"
    # with a traceback, and this future is abandoned on every early finish.
    # A done callback covers the paths where nothing awaits it, including
    # a cancellation landing while the samples are still unwinding.
    gathered.add_done_callback(lambda f: None if f.cancelled() else f.exception())
    waiter = asyncio.create_task(enough.wait())
    try:
        await asyncio.wait({gathered, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        # asyncio.wait() leaves what it waits on running, so without this
        # an interrupted run (Ctrl-C, tool cancellation) would keep every
        # sample task alive — still spending, still appending to the
        # partial file — after the caller has unwound. Awaiting a gather()
        # directly used to do this for us.
        await _stop_all(tasks)
        raise
    finally:
        waiter.cancel()

    still_running = [t for t in tasks if not t.done()]
    if gathered.done():
        exc = gathered.exception()
        if exc is None:
            return
        # gather() reports the first exception without touching its
        # siblings; stop them so nothing writes after we unwind — and so
        # the error surfaces now rather than behind the slowest straggler.
        await _stop_all(tasks)
        raise exc

    # Samples already recorded under the lock stand; the cancelled ones
    # simply never happened.
    await _stop_all(tasks)
    if still_running:
        logger.info(
            "Requested sample count reached — cancelled %d unfinished sample(s)",
            len(still_running),
        )


async def _stop_all(tasks: list[asyncio.Task[None]]) -> None:
    """Cancel every task and wait for the cancellations to settle.

    Cancels the tasks themselves rather than the gather wrapping them: a
    gather that has already finished (first sample exception) ignores
    ``cancel()``, which would leave the stragglers running.

    Cancellation is cooperative, so the deadline here returns *this*
    coroutine — it cannot stop a task that swallows ``CancelledError``,
    and nothing at this layer can preempt one that blocks the event loop.
    An abandoned task keeps running until the loop shuts down (where
    ``asyncio.run`` waits for it) and may still spend, but it can no
    longer affect the dataset: the run's bookkeeping is closed to it (see
    ``closed`` in ``_run_one``). Handing control back beats blocking the
    caller on a pipeline that refuses to stop.
    """
    for task in tasks:
        task.cancel()
    settle = asyncio.gather(*tasks, return_exceptions=True)
    try:
        await asyncio.wait_for(asyncio.shield(settle), _CANCEL_SETTLE_S)
    except TimeoutError:
        settle.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        logger.warning(
            "%d sample(s) ignored cancellation for %.0fs; continuing without "
            "them. They are excluded from the dataset but may still be "
            "running (and spending) until the event loop shuts down.",
            sum(1 for t in tasks if not t.done()), _CANCEL_SETTLE_S,
        )


def _overcommit_margin(total: int) -> int:
    """Extra samples to queue beyond *total* so stragglers can be dropped.

    Zero for small runs: a draft/validation run of 1-3 samples must
    surface every failure, and at that size the spares would be a large
    fraction of the cost.
    """
    if total < _OVERCOMMIT_MIN_TOTAL:
        return 0
    return min(_OVERCOMMIT_MAX, max(2, math.ceil(total * _OVERCOMMIT_FRACTION)))


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
    concurrency: int = 100,
    samples_per_item: int = 1,
    sample_timeout: float | None = _SAMPLE_TIMEOUT_S,
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
        cap on source items (bring-your-data mode). Pure-generation runs
        may *attempt* a few more than this — see ``_overcommit_margin`` —
        but never write more rows, and never read past the cap in
        bring-your-data mode.
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
    sample_timeout:
        Watchdog on a single ``generate()`` attempt, in seconds (``None``
        disables it). Deliberately far above any healthy sample — the HTTP
        client already bounds each LLM call — so it only fires on a wedged
        pipeline. A timed-out attempt is retried like any other failure.
    validation_instructions:
        Optional text with LLM validation criteria (reserved for future use).
    on_progress:
        Optional callback invoked as ``on_progress(done, total)`` after each
        sample finishes (success or permanent failure), where *done* is the
        samples in hand so far and *total* is ``num_samples``. *done* only
        advances on success, so it repeats after a failure and a run that
        can't deliver the full count ends below *total* — the honest signal
        that the caller got fewer samples than it asked for.
    """
    # Record every lqh.sources path the pipeline touches — across both
    # source() and generate() (helpers like seed_data are commonly
    # called per-sample). The recorded set becomes the cloud-submit
    # bundle manifest via the validation record.
    with record_source_paths() as recorded_paths:
        result = await _run_pipeline_inner(
            script_path,
            num_samples,
            output_dir,
            client,
            max_retries=max_retries,
            concurrency=concurrency,
            samples_per_item=samples_per_item,
            sample_timeout=sample_timeout,
            validation_instructions=validation_instructions,
            on_progress=on_progress,
        )
        # Read before the context exits — the flag resets with it.
        result.used_hf = hf_dataset_was_used()
    result.source_paths = sorted(recorded_paths)
    return result


async def _run_pipeline_inner(
    script_path: Path,
    num_samples: int,
    output_dir: Path,
    client: AsyncOpenAI,
    *,
    max_retries: int,
    concurrency: int,
    samples_per_item: int,
    sample_timeout: float | None,
    validation_instructions: str | None,
    on_progress: Callable[[int, int], Any] | None,
) -> EngineResult:
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

    # What the caller asked for. Anything past this is a spare, queued
    # only so the run can stop at *target* instead of waiting out a
    # straggler; surplus successes are trimmed from the output below.
    #
    # Spares exist only in pure-generation mode, where samples are
    # interchangeable. Bring-your-data runs treat num_samples as a hard
    # cap on source items consumed (see the docstring and the data-
    # generation skill), and their items aren't interchangeable: reading
    # past the cap would silently swap a requested record for a later
    # one, and could abort a run on a deterministic bug in an item the
    # caller never asked to process.
    target = len(work)
    margin = 0 if source_items is not None else _overcommit_margin(target)
    if margin > 0:
        work.extend([None] * margin)

    total = len(work)
    results: list[dict[str, Any] | None] = [None] * total
    succeeded = 0
    failed = 0
    completed = 0
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    # Set once *target* samples are in hand — the signal to stop.
    enough = asyncio.Event()
    # Set once the run is within *margin* of done — the signal to release
    # the spares. Nothing to wait for when there are none.
    tail = asyncio.Event()
    if margin == 0:
        tail.set()

    # Incremental saves: write each completed sample to a JSONL file so
    # progress survives process kills.  On restart, already-done samples
    # are skipped automatically — but only when the partial was written
    # by the SAME pipeline version (digest binding): stale samples from
    # an edited pipeline must not count toward this run (they'd also
    # satisfy the cloud-validation gate without being executed).
    # NOTE: resume is index-based — it assumes source() yields the same
    # items in the same order on every run (the skill mandates
    # deterministic sources). A nondeterministic iterator would resume
    # against different records.
    from lqh.data_gen_validation import pipeline_digest

    digest = pipeline_digest(script_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "data.partial.jsonl"
    done_indices: set[int] = set()

    if partial_path.exists():
        done_indices, results, succeeded, failed = _load_partial(
            partial_path, total, digest, header_total=target,
        )
        completed = len(done_indices)
        if done_indices:
            logger.info("Resuming: %d/%d samples already completed", len(done_indices), target)
        else:
            # Invalidated (or empty) partial — rewrite the header so the
            # digest/total on disk match this run. If the old file holds
            # samples we're discarding (pre-digest legacy header, edited
            # pipeline, or changed total), preserve them under a stale
            # name rather than destroying paid-for work; they stay
            # excluded from this run and from the validation gate.
            if _partial_has_samples(partial_path):
                stale_path = output_dir / "data.partial.stale.jsonl"
                partial_path.replace(stale_path)
                logger.warning(
                    "Existing partial doesn't match this run (pipeline version "
                    "or sample count); its samples were preserved at %s "
                    "(not counted toward this run)",
                    stale_path,
                )
            with open(partial_path, "w") as f:
                f.write(json.dumps({"_meta": True, "total": target, "digest": digest}) + "\n")
    else:
        with open(partial_path, "w") as f:
            f.write(json.dumps({"_meta": True, "total": target, "digest": digest}) + "\n")

    # Set when a deterministic code bug is detected — signals all tasks to abort.
    abort_error: Exception | None = None
    # Set once the run stops accepting results, so a sample that outlives
    # its cancellation can't write into a finished run.
    closed = False

    async def _run_one(index: int, input_item: Any) -> None:
        nonlocal succeeded, failed, completed, abort_error
        if index >= target:
            # A spare. Hold it outside the semaphore until the run reaches
            # its tail: a healthy run should not pay for spares it will
            # only cancel, and a merely-slow sample deserves the whole
            # tail to finish before a spare can overtake it.
            await tail.wait()
        async with sem:
            if abort_error is not None:
                return  # another sample already hit a fatal bug
            if enough.is_set():
                return  # target already reached — don't start new work

            result: dict[str, Any] | None = None
            for attempt in range(max_retries + 1):
                if attempt and (enough.is_set() or abort_error is not None):
                    # Abandoned mid-ladder — not a failure, just work the
                    # run no longer needs.
                    return
                instance = pipeline_cls()
                try:
                    # Pass input positionally to tolerate pipelines that
                    # omit the ``input`` parameter from their signature.
                    async with asyncio.timeout(sample_timeout) as watchdog:
                        if input_item is not None:
                            conv = await instance.generate(client, input_item)
                        else:
                            conv = await instance.generate(client)
                    result = _serialize_conversation(conv)
                    break
                except TimeoutError:
                    # The watchdog is not a tuning knob: the HTTP client
                    # already bounds each call at 300 s, so it only fires on
                    # a pipeline wedged on something else. Without it a
                    # single hung sample pins its slot forever. A
                    # TimeoutError raised by the pipeline itself is a
                    # different animal — don't report it as the watchdog.
                    if watchdog.expired():
                        logger.warning(
                            "Sample %d timed out after %ss (attempt %d/%d)",
                            index, sample_timeout, attempt + 1, max_retries + 1,
                        )
                    else:
                        logger.warning(
                            "Sample %d raised TimeoutError (attempt %d/%d)",
                            index, attempt + 1, max_retries + 1,
                        )
                    if attempt >= max_retries:
                        break
                    continue
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
                        # Signal the stop as well as recording the error.
                        # Returning alone only stops *new* work from
                        # starting — the over-commit spares are parked on
                        # `tail`, which only a completion sets, and after
                        # an abort nothing completes. The gather would
                        # then never finish and the run would hang until
                        # something outside killed it (a cloud job burns
                        # its whole wall-clock cap in silence). `enough`
                        # is what `_run_until_enough` waits on to cancel
                        # the stragglers and hand the error back.
                        enough.set()
                        return
                    if attempt >= max_retries:
                        break

            async with lock:
                if closed:
                    # A sample that outlived its cancellation. The run has
                    # moved on: touching the counters or the partial file
                    # now would corrupt a dataset nobody is waiting for.
                    return
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
                    # Persisted so an interrupted run's failures still show
                    # up in the continuation's reliability count.
                    _append_failure(partial_path, index)
                    failed += 1
                completed += 1
                if completed >= target - margin:
                    tail.set()
                if succeeded >= target:
                    enough.set()
                if on_progress is not None:
                    # Report against what was requested, not the
                    # over-committed work list: spares are an internal
                    # detail, and "samples in hand" is the number the
                    # user is waiting on.
                    on_progress(min(succeeded, target), target)

    # A resume can already be in the tail — re-check before starting, or
    # the spares wait on a gate that only a *new* completion would open.
    # (Resume 58 of 60 with two hung originals left: nothing ever
    # completes, so nothing would release the spares that exist to
    # replace them.)
    if completed >= target - margin:
        tail.set()

    tasks: list[asyncio.Task[None]] = []
    if succeeded < target:
        tasks = [
            asyncio.create_task(_run_one(i, item))
            for i, item in enumerate(work)
            if i not in done_indices
        ]
    try:
        await _run_until_enough(tasks, enough)
    finally:
        closed = True

    # If a deterministic bug aborted the run, raise it so the caller
    # (tool handler) gets a clear error message immediately.
    if abort_error is not None:
        raise abort_error

    # Build parquet columns directly from results, releasing each parsed
    # conversation as it's serialized. The previous rows-of-dicts
    # intermediate held a second full copy of every sample during
    # finalization — for VLM datasets (base64 images inside messages)
    # that doubled peak memory exactly at the step where a constrained
    # sandbox OOMs.
    messages_col: list[str] = []
    audio_col: list[str | None] = []
    tools_col: list[str | None] = []
    spares_used = 0
    for i, r in enumerate(results):
        if r is None:
            continue
        if len(messages_col) >= target:
            # Over-commit surplus: a couple of spares can land before the
            # cancellation reaches them. The caller asked for *target*
            # samples, so that's what the dataset holds.
            results[i] = None
            continue
        if i >= target:
            spares_used += 1  # a spare that filled in for a lost sample
        messages_col.append(json.dumps(r["messages"], ensure_ascii=False))
        audio_col.append(
            json.dumps(r["audio"], ensure_ascii=False) if r["audio"] is not None else None
        )
        tools_col.append(
            json.dumps(r["tools"], ensure_ascii=False) if r.get("tools") is not None else None
        )
        results[i] = None  # free the parsed dict promptly

    schema = pa.schema([
        pa.field("messages", pa.string()),
        pa.field("audio", pa.string()),
        pa.field("tools", pa.string()),
    ])
    table = pa.table(
        {"messages": messages_col, "audio": audio_col, "tools": tools_col},
        schema=schema,
    )

    output_path = output_dir / "data.parquet"
    pq.write_table(table, output_path)
    partial_path.unlink(missing_ok=True)

    # Report against the requested count, not the over-committed work
    # list: `total` is what callers show as the denominator, `succeeded`
    # must match the rows actually written, and `failed` is the gap
    # between them so the three never contradict each other.
    written = len(messages_col)
    if failed:
        logger.info(
            "%d sample(s) failed permanently; %d spare(s) filled in",
            failed, spares_used,
        )
    return EngineResult(
        total=target,
        succeeded=written,
        failed=target - written,
        output_path=output_path,
        resumed_samples=min(len(done_indices), target),
        sample_failures=failed,
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
