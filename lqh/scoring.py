"""Scoring engine for evaluating datasets using LLM-as-judge.

Reads labelled ChatML samples, optionally strips assistant turns for model
inference, then scores each sample against spec-derived criteria using a
scoring LLM.  Results are written as parquet + JSON summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI

from lqh.config import default_api_base_url
from lqh.runner import APIModelRunner, ModelRunner

__all__ = [
    "DEFAULT_JUDGE_MODEL_SIZE",
    "JUDGE_MODELS",
    "FilterResult",
    "ScoringResult",
    "extract_failures",
    "run_data_filter",
    "run_scoring",
    "run_data_scoring",
]

logger = logging.getLogger(__name__)

# Dedicated judge models on api.lqh.ai for LLM-as-judge scoring.
#   judge:small  — fast, cheap, good for iteration and testing
#   judge:medium — balanced quality/cost for production scoring
#   judge:large  — highest quality, use for final evaluations
JUDGE_MODELS: dict[str, str] = {
    "small": "judge:small",
    "medium": "judge:medium",
    "large": "judge:large",
}
DEFAULT_JUDGE_MODEL_SIZE = "small"

# JSON schema for structured output (reasoning before score for chain-of-thought)
SCORE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "scoring_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief evaluation (2-3 sentences) explaining the score.",
                },
                "score": {
                    "type": "integer",
                    "description": "Score from 0 to 10 (0 is the worst grade, reserved for empty/refusal/unrelated output).",
                },
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class ScoringResult:
    """Summary of a scoring run."""

    total: int
    scored: int
    failed: int
    mean_score: float
    median_score: float
    output_dir: Path


@dataclass
class FilterResult:
    """Summary of a score-and-filter run over a user-brought dataset."""

    total: int
    scored: int
    kept: int
    dropped: int
    failed: int
    threshold: float
    mean_score: float
    output_dataset_dir: Path
    scores_path: Path
    summary_path: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_samples(
    parquet_path: Path,
) -> tuple[list[list[dict]], list[list[dict] | None]]:
    """Load ChatML conversations and per-sample tools from a parquet file.

    Returns ``(samples, tools_per_sample)``.  Works with parquet files
    that lack a ``tools`` column (returns ``None`` for each sample).
    """
    table = pq.read_table(parquet_path)
    messages_col = table.column("messages")
    has_tools = "tools" in table.column_names
    tools_col = table.column("tools") if has_tools else None

    samples: list[list[dict]] = []
    tools_list: list[list[dict] | None] = []
    for i in range(len(table)):
        raw = messages_col[i].as_py()
        samples.append(json.loads(raw) if raw else [])
        if tools_col is not None:
            raw_tools = tools_col[i].as_py()
            tools_list.append(json.loads(raw_tools) if isinstance(raw_tools, str) and raw_tools else None)
        else:
            tools_list.append(None)
    return samples, tools_list


def _strip_trailing_assistant(messages: list[dict]) -> list[dict]:
    """Remove trailing assistant messages to create an unlabelled sample.

    Walks backwards from the end and removes consecutive assistant messages
    until a non-assistant message is found.
    """
    trimmed = list(messages)
    while trimmed and trimmed[-1].get("role") == "assistant":
        trimmed.pop()
    return trimmed


def _has_tool_calls(messages: list[dict]) -> bool:
    """Check if any message in the conversation contains tool calls."""
    return any(msg.get("tool_calls") for msg in messages)


def _format_tool_calls(tool_calls: list[dict]) -> str:
    """Format tool calls as readable text for the judge."""
    parts: list[str] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        args = func.get("arguments", "{}")
        parts.append(f"  -> {name}({args})")
    return "\n".join(parts)


def _render_content(content: Any, images: list[str] | None = None) -> str:
    """Render a ChatML content value as judge-readable text.

    String content passes through unchanged. Multi-part (list) content — the
    OpenAI vision format — renders its text parts in order and replaces each
    image part with an ``[image N]`` placeholder. When *images* is given, the
    image URLs are appended to it (deduplicated, in order of first
    appearance) so the caller can re-attach them to the judge request as real
    image parts; N is the 1-based index into that list.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    rendered: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            rendered.append(str(part))
            continue
        ptype = part.get("type")
        if ptype == "text":
            rendered.append(part.get("text", ""))
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if images is None or not url:
                rendered.append("[image]")
            else:
                if url in images:
                    idx = images.index(url) + 1
                else:
                    images.append(url)
                    idx = len(images)
                rendered.append(f"[image {idx}]")
        else:
            rendered.append(f"[{ptype or 'unknown'} part]")
    return "\n".join(r for r in rendered if r != "")


def _format_conversation(
    messages: list[dict],
    tools: list[dict] | None = None,
    images: list[str] | None = None,
) -> str:
    """Format a ChatML conversation as readable text for the judge.

    Presents each turn clearly labelled, so the judge doesn't confuse
    the role/content wrapper with the actual model output. Image parts in
    multi-part content are shown as ``[image N]`` placeholders; see
    ``_render_content`` for how *images* collects the URLs.
    """
    parts: list[str] = []

    if tools:
        tool_names = [t.get("function", {}).get("name", "?") for t in tools]
        parts.append(f"[Available Tools: {', '.join(tool_names)}]")

    for msg in messages:
        role = msg.get("role", "unknown")
        content = _render_content(msg.get("content", ""), images)
        if role == "system":
            parts.append(f"[System Prompt]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "assistant":
            tc = msg.get("tool_calls")
            if tc:
                formatted_calls = _format_tool_calls(tc)
                if content:
                    parts.append(f"[Assistant]\n{content}\n[Tool Calls]\n{formatted_calls}")
                else:
                    parts.append(f"[Assistant]\n[Tool Calls]\n{formatted_calls}")
            else:
                parts.append(f"[Assistant]\n{content}")
        elif role == "tool":
            name = msg.get("name", "tool")
            parts.append(f"[Tool Result: {name}]\n{content}")
    return "\n\n".join(parts)


_TOOL_CALL_JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of AI tool-calling behavior. "
    "You will receive a conversation where the assistant uses function-calling tools. "
    "Evaluate based on the scoring criteria provided. Pay special attention to:\n"
    "1. Whether the correct tool was called for the user's request\n"
    "2. Whether the tool arguments are correct (allow equivalent values, "
    "e.g. 'SF' and 'San Francisco' are both acceptable for a location)\n"
    "3. Whether the assistant properly used the tool result in its response\n"
    "4. Only evaluate actual tool calls (marked as [Tool Calls]), "
    "NOT text mentions of tools in the assistant's prose\n"
    "First write your reasoning (2-3 concise sentences), then give "
    "a score from 0 to 10. Output JSON with keys: reasoning, score."
)

_DEFAULT_JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of AI-generated responses. "
    "You will receive a conversation sample and scoring criteria. "
    "First write your reasoning (2-3 concise sentences), then give "
    "a score from 0 to 10. Output JSON with keys: reasoning, score."
)


def _build_scoring_prompt(
    scorer_text: str,
    messages: list[dict],
    *,
    reference_messages: list[dict] | None = None,
    tools: list[dict] | None = None,
) -> list[dict]:
    """Build the prompt for the scoring LLM.

    Parameters
    ----------
    scorer_text:
        The scoring criteria markdown.
    messages:
        The conversation to score (must end with assistant turn).
    reference_messages:
        Optional ground-truth conversation for comparison.
    tools:
        Optional tool definitions for tool-calling conversations.
    """
    is_tool_calling = _has_tool_calls(messages) or tools is not None
    # Collect image URLs across the sample and the reference (deduplicated)
    # so vision conversations are judged against the actual images rather
    # than a stringified content list.
    images: list[str] = []
    formatted = _format_conversation(messages, tools=tools, images=images)

    user_content = (
        "Score the following conversation according to the criteria below.\n\n"
        "## Scoring Criteria\n\n"
        f"{scorer_text}\n\n"
        "## Conversation to Score\n\n"
        f"{formatted}\n\n"
    )

    if reference_messages:
        ref_formatted = _format_conversation(reference_messages, tools=tools, images=images)
        user_content += (
            "## Reference (ground truth)\n\n"
            f"{ref_formatted}\n\n"
        )

    if is_tool_calling:
        user_content += (
            "## Instructions\n\n"
            "Evaluate the assistant's tool-calling behavior against the scoring criteria. "
            "Focus on whether the correct tools were called with correct arguments, "
            "and whether the assistant properly interpreted tool results. "
            "Only consider actual tool invocations (shown as [Tool Calls]), "
            "not casual mentions of tools in text. "
            "Think step by step, then assign a score from 1 to 10."
        )
    else:
        user_content += (
            "## Instructions\n\n"
            "Evaluate the assistant's final response against the scoring criteria. "
            "Focus on the content of what the assistant said, not the conversation format. "
            "Think step by step about what the response does well and where it "
            "falls short, then assign a score from 1 to 10."
        )

    system_content = _TOOL_CALL_JUDGE_SYSTEM if is_tool_calling else _DEFAULT_JUDGE_SYSTEM

    # Vision conversations: attach the actual images after the text so the
    # judge sees them. The [image N] placeholders in the transcript refer to
    # the attached images in order. The backend routes judge:* requests with
    # images to a vision-capable judge model.
    if images:
        user_content += (
            f"\n(The conversation contains {len(images)} attached image(s); "
            "[image N] marks where each appears.)"
        )
        user_message_content: Any = [{"type": "text", "text": user_content}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in images
        ]
    else:
        user_message_content = user_content

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message_content},
    ]


def is_scoring_error(reasoning: str | None) -> bool:
    """Return True iff a sample's reasoning indicates the judge failed to score it.

    Failures fall into two buckets we want to treat as "could not be scored"
    rather than "model performed badly": API errors (e.g. upstream 429s) write
    ``[Scoring error] ...`` and JSON-parse failures write ``[Parse error] ...``.
    Real low scores have a normal LLM reasoning string. Callers use this
    distinction so analysis tools don't conflate scoring infrastructure
    failures with model-quality regressions.
    """
    if not reasoning:
        return False
    head = reasoning.lstrip()
    return head.startswith("[Scoring error]") or head.startswith("[Parse error")


def _parse_score_response(text: str) -> tuple[float, str]:
    """Parse the scoring LLM's JSON response into (score, reasoning).

    With structured output mode the response is guaranteed to be valid JSON,
    but we keep a fallback path for robustness.
    """
    text = text.strip()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return 0.0, f"[Parse error: expected JSON object, got {type(data).__name__}] {text[:500]}"
        score = float(data.get("score", 0))
        reasoning = str(data.get("reasoning", ""))
        return score, reasoning
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        logger.warning("Could not parse scoring response as JSON: %s", text[:200])
        return 0.0, f"[Parse error] {text[:500]}"


# ---------------------------------------------------------------------------
# Scoring runners
# ---------------------------------------------------------------------------

async def run_scoring(
    dataset_path: Path,
    scorer_path: Path,
    output_dir: Path,
    client: AsyncOpenAI,
    *,
    model_size: str = "small",
    concurrency: int = 100,
    max_retries: int = 2,
    run_inference: bool = False,
    inference_model: str | None = None,
    inference_system_prompt: str | None = None,
    inference_runner: ModelRunner | None = None,
    inference_response_format: dict[str, Any] | None = None,
    on_progress: Callable[[int, int], Any] | None = None,
    debug: bool = False,
) -> ScoringResult:
    """Score a dataset against criteria from a scorer file.

    Parameters
    ----------
    dataset_path:
        Path to the parquet file containing labelled ChatML conversations.
    scorer_path:
        Path to the scorer ``.md`` file with judging criteria.
    output_dir:
        Directory to write results (results.parquet, summary.json, config.json).
    client:
        Pre-configured AsyncOpenAI client.
    model_size:
        Judge model size, maps to ``judge:<size>`` on api.lqh.ai:
        "small" (default, fast/cheap, good for iteration),
        "medium" (balanced quality/cost), or
        "large" (highest quality, for final evaluations).
    concurrency:
        Max parallel scoring calls.
    max_retries:
        Retries per sample on scoring failure.
    run_inference:
        If True, strip trailing assistant turns and run model inference
        before scoring (for model evaluation).
    inference_model:
        Model to run inference with (required if run_inference=True).
    inference_system_prompt:
        Optional system prompt override for inference.
    inference_runner:
        Optional ``ModelRunner`` to use for inference.  If ``None``
        (default), an ``APIModelRunner`` wrapping *client* is used.
    on_progress:
        Callback ``on_progress(completed, total)`` after each sample.
    """
    scoring_model = JUDGE_MODELS.get(model_size, JUDGE_MODELS[DEFAULT_JUDGE_MODEL_SIZE])
    scorer_text = scorer_path.read_text(encoding="utf-8")
    samples, tools_per_sample = _load_samples(dataset_path)
    total = len(samples)

    if total == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        return ScoringResult(
            total=0, scored=0, failed=0,
            mean_score=0.0, median_score=0.0,
            output_dir=output_dir,
        )

    # Results storage
    results: list[dict[str, Any] | None] = [None] * total
    debug_log: list[dict[str, Any]] = []  # low-scoring samples for debugging
    scored = 0
    failed_count = 0
    completed = 0
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def _score_one(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal scored, failed_count, completed

        async with sem:
            original_messages = messages
            scored_messages = messages
            inf_messages: list[dict] | None = None
            assistant_content: str | None = None

            # Optionally run inference
            if run_inference:
                unlabelled = _strip_trailing_assistant(messages)
                if not unlabelled:
                    async with lock:
                        failed_count += 1
                        completed += 1
                        if on_progress:
                            on_progress(completed, total)
                    return

                # Run model inference via runner
                # Always strip existing system messages — the system prompt
                # is managed separately (via inference_system_prompt / prompts/).
                inf_messages = [m for m in unlabelled if m.get("role") != "system"]
                if inference_system_prompt:
                    inf_messages.insert(0, {"role": "system", "content": inference_system_prompt})

                runner = inference_runner or APIModelRunner(client)
                try:
                    inf_response = await runner.complete(
                        inf_messages,
                        model=inference_model or "orchestration",
                        temperature=0.0,
                        response_format=inference_response_format,
                        tools=sample_tools,
                    )
                    assistant_content = inf_response.content
                    scored_messages = unlabelled + [{"role": "assistant", "content": assistant_content}]
                except Exception as exc:
                    logger.error("Inference failed for sample %d: %s", index, exc)
                    async with lock:
                        failed_count += 1
                        completed += 1
                        if on_progress:
                            on_progress(completed, total)
                    return

            # Score the sample
            scoring_prompt = _build_scoring_prompt(
                scorer_text,
                scored_messages,
                reference_messages=original_messages if run_inference else None,
                tools=sample_tools,
            )

            score = 0.0
            reasoning = ""
            success = False

            for attempt in range(max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model=scoring_model,
                        messages=scoring_prompt,
                        temperature=0.0,
                        response_format=SCORE_RESPONSE_SCHEMA,
                    )
                    if not response.choices:
                        raise ValueError("Empty choices in scoring response")
                    raw = response.choices[0].message.content or ""
                    score, reasoning = _parse_score_response(raw)
                    # A parseable judge response is a real score — including 0,
                    # which is the rubric's worst-quality grade (empty/refusal/
                    # unrelated), NOT an infrastructure failure. Only parse/API
                    # errors (flagged by is_scoring_error) are retried/dropped.
                    if not is_scoring_error(reasoning):
                        success = True
                        break
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                except Exception as exc:
                    logger.warning(
                        "Scoring failed for sample %d (attempt %d/%d): %s",
                        index, attempt + 1, max_retries + 1, exc,
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        reasoning = f"[Scoring error] {exc}"

            async with lock:
                final_score = score if success else 0.0
                results[index] = {
                    "sample_index": index,
                    "messages": json.dumps(scored_messages, ensure_ascii=False),
                    "score": final_score,
                    "reasoning": reasoning,
                }
                if success:
                    scored += 1
                else:
                    failed_count += 1
                completed += 1
                if on_progress:
                    on_progress(completed, total)

                # Debug log for low-scoring samples (< 6/10)
                if final_score < 6.0 and run_inference:
                    debug_entry = {
                        "sample_index": index,
                        "score": final_score,
                        "reasoning": reasoning,
                        "inference_model": inference_model or "orchestration",
                        "inference_messages_sent": inf_messages if run_inference else None,
                        "model_response": assistant_content if run_inference else None,
                        "reference_messages": original_messages,
                    }
                    debug_log.append(debug_entry)

    tasks = [
        asyncio.create_task(_score_one(i, sample, tools_per_sample[i]))
        for i, sample in enumerate(samples)
    ]
    await asyncio.gather(*tasks)

    # Build output. Aggregate over every successfully judged sample — a score
    # of 0 is a valid grade, not a failure. Only parse/API errors (which carry
    # an is_scoring_error reasoning marker) are excluded from the stats.
    rows = [r for r in results if r is not None]
    scores = [r["score"] for r in rows if not is_scoring_error(r["reasoning"])]

    mean_score = sum(scores) / len(scores) if scores else 0.0
    sorted_scores = sorted(scores)
    median_score = (
        sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0.0
    )

    # Write results.parquet
    output_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        table = pa.table(
            {
                "sample_index": [r["sample_index"] for r in rows],
                "messages": [r["messages"] for r in rows],
                "score": [r["score"] for r in rows],
                "reasoning": [r["reasoning"] for r in rows],
            },
            schema=pa.schema([
                pa.field("sample_index", pa.int64()),
                pa.field("messages", pa.string()),
                pa.field("score", pa.float64()),
                pa.field("reasoning", pa.string()),
            ]),
        )
    else:
        table = pa.table(
            {"sample_index": [], "messages": [], "score": [], "reasoning": []},
            schema=pa.schema([
                pa.field("sample_index", pa.int64()),
                pa.field("messages", pa.string()),
                pa.field("score", pa.float64()),
                pa.field("reasoning", pa.string()),
            ]),
        )

    pq.write_table(table, output_dir / "results.parquet")

    # Write summary.json
    std_score = 0.0
    if len(scores) > 1:
        mean = mean_score
        std_score = (sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)) ** 0.5

    summary = {
        "dataset": str(dataset_path),
        "scorer": str(scorer_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_samples": total,
        "num_scored": scored,
        "num_failed": failed_count,
        "scores": {
            "mean": round(mean_score, 2),
            "median": round(median_score, 2),
            "std": round(std_score, 2),
            "min": round(min(scores), 2) if scores else 0.0,
            "max": round(max(scores), 2) if scores else 0.0,
        },
    }

    summary["scoring_model"] = scoring_model
    summary["scoring_model_size"] = model_size
    if run_inference:
        summary["inference_model"] = inference_model or "orchestration"
        if inference_system_prompt:
            summary["inference_system_prompt"] = inference_system_prompt

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Write debug log for low-scoring samples (when debug mode is enabled)
    if debug and debug_log:
        debug_log.sort(key=lambda d: d["score"])
        debug_path = output_dir / "debug_low_scores.jsonl"
        with open(debug_path, "w", encoding="utf-8") as f:
            for entry in debug_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Generate curl replay scripts (uses env var, never hardcodes token)
        curl_dir = output_dir / "curl_debug"
        curl_dir.mkdir(exist_ok=True)
        for entry in debug_log:
            idx = entry["sample_index"]
            sc = entry["score"]
            payload = {
                "model": entry["inference_model"],
                "temperature": 0.0,
                "messages": entry["inference_messages_sent"],
            }
            judge_preview = entry["reasoning"][:100].replace('"', '\\"')
            script = (
                f"#!/bin/bash\n"
                f"# Sample {idx} | Score: {sc}/10\n"
                f'# Judge: {judge_preview}...\n'
                f"#\n"
                f'# Requires: export LQH_DEBUG_API_KEY="your_key"\n'
                f"\n"
                f"curl -s {default_api_base_url()}/chat/completions \\\n"
                f'  -H "Authorization: Bearer $LQH_DEBUG_API_KEY" \\\n'
                f'  -H "Content-Type: application/json" \\\n'
                f"  -d '{json.dumps(payload, ensure_ascii=False)}'"
                f" | python3 -m json.tool\n"
            )
            script_path = curl_dir / f"sample_{idx:03d}_score_{sc:.0f}.sh"
            script_path.write_text(script, encoding="utf-8")
            script_path.chmod(0o755)

        logger.info(
            "Debug: wrote %d entries + curl scripts to %s",
            len(debug_log), output_dir,
        )

    return ScoringResult(
        total=total,
        scored=scored,
        failed=failed_count,
        mean_score=mean_score,
        median_score=median_score,
        output_dir=output_dir,
    )


def _source_map(predictions_path: Path) -> dict[int, str]:
    """Read a predictions parquet and return ``{sample_index: source}``.

    Returns an empty map when the file has no ``source`` column (legacy
    single-source predictions) — callers then treat everything as one group.
    """
    try:
        table = pq.read_table(str(predictions_path))
    except Exception:
        return {}
    if "source" not in table.column_names:
        return {}
    idx_col = table.column("sample_index")
    src_col = table.column("source")
    return {idx_col[i].as_py(): src_col[i].as_py() for i in range(len(table))}


def _score_stats(values: list[float]) -> dict[str, float]:
    """mean/median/std/min/max over a list of scores (empty → zeros)."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    mean = sum(values) / len(values)
    median = ordered[len(ordered) // 2]
    std = (
        (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        if len(values) > 1
        else 0.0
    )
    return {
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


async def score_predictions_by_source(
    predictions_path: Path,
    scorer_path: Path,
    output_dir: Path,
    client: AsyncOpenAI,
    *,
    model_size: str = "small",
    **scoring_kwargs: Any,
) -> dict[str, Any]:
    """Judge-score predictions, partition scores by their ``source`` column,
    and write a multi-source ``eval_result.json``.

    Runs :func:`run_scoring` once (one efficient pass over all predictions),
    then groups the per-sample scores by the source each prediction came from.
    The headline ``scores.mean`` is the **macro-average** — the mean of the
    per-source means, weighting every source equally regardless of size;
    ``scores_weighted_mean`` is the by-count mean for reference. When the
    predictions carry no ``source`` column the whole set is one group labelled
    ``"all"`` — byte-for-byte the legacy single-source behaviour.

    Writes ``output_dir/eval_result.json`` and returns that payload (so cloud
    callers can surface it without re-reading the file).
    """
    result = await run_scoring(
        dataset_path=predictions_path,
        scorer_path=scorer_path,
        output_dir=output_dir,
        client=client,
        model_size=model_size,
        run_inference=False,
        **scoring_kwargs,
    )

    src_by_idx = _source_map(predictions_path)

    # Collect valid (non-error) scores grouped by source from results.parquet.
    grouped: dict[str, list[float]] = {}
    results_path = output_dir / "results.parquet"
    if results_path.exists():
        rtable = pq.read_table(str(results_path))
        cols = rtable.column_names
        idx_c = rtable.column("sample_index")
        score_c = rtable.column("score")
        reason_c = rtable.column("reasoning") if "reasoning" in cols else None
        for i in range(len(rtable)):
            reasoning = reason_c[i].as_py() if reason_c is not None else None
            if is_scoring_error(reasoning):
                continue
            sample_idx = idx_c[i].as_py()
            label = src_by_idx.get(sample_idx, "all")
            grouped.setdefault(label, []).append(float(score_c[i].as_py()))

    # Every source that appears in the predictions must show up in per_source,
    # even one whose samples ALL failed to score — otherwise an all-failures
    # source would silently vanish and the macro headline would be averaged
    # over the survivors only. Such a source is reported with num_scored=0 but
    # excluded from the macro means (it has no mean to contribute).
    all_labels = set(grouped) | set(src_by_idx.values())
    if not all_labels:
        all_labels = {"all"}

    per_source: dict[str, Any] = {}
    per_source_means: list[float] = []
    per_source_medians: list[float] = []
    for label in sorted(all_labels):
        scores_list = grouped.get(label, [])
        stats = _score_stats(scores_list)
        per_source[label] = {
            "num_scored": len(scores_list),
            "scores": stats,
        }
        if scores_list:
            per_source_means.append(stats["mean"])
            per_source_medians.append(stats["median"])

    # Macro-average headline: each source weighted equally.
    macro_mean = (
        round(sum(per_source_means) / len(per_source_means), 2)
        if per_source_means
        else round(result.mean_score, 2)
    )
    macro_median = (
        round(sum(per_source_medians) / len(per_source_medians), 2)
        if per_source_medians
        else round(result.median_score, 2)
    )

    payload: dict[str, Any] = {
        "num_scored": result.scored,
        "num_failed": result.failed,
        # Headline = macro-average (mean of per-source means). Kept under the
        # existing scores.mean/median keys so all current readers keep working.
        "scores": {"mean": macro_mean, "median": macro_median},
        # By-count mean over every scored sample, for reference / comparison.
        "scores_weighted_mean": round(result.mean_score, 2),
        "per_source": per_source,
    }
    (output_dir / "eval_result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


async def run_data_scoring(
    dataset_dir: Path,
    scorer_path: Path,
    client: AsyncOpenAI,
    *,
    model_size: str = "small",
    concurrency: int = 100,
    on_progress: Callable[[int, int], Any] | None = None,
) -> ScoringResult:
    """Score data quality of a dataset (no inference, co-located output).

    Writes ``scores.parquet`` alongside the dataset's ``data.parquet``.

    Parameters
    ----------
    dataset_dir:
        Directory containing ``data.parquet`` with ChatML conversations.
    scorer_path:
        Path to the scorer ``.md`` file with judging criteria.
    client:
        Pre-configured AsyncOpenAI client.
    model_size:
        Judge model size, maps to ``judge:<size>`` on api.lqh.ai:
        "small" (default, fast/cheap, good for iteration),
        "medium" (balanced quality/cost), or
        "large" (highest quality, for final evaluations).
    concurrency:
        Max parallel scoring calls.
    on_progress:
        Callback ``on_progress(completed, total)`` after each sample.
    """
    scoring_model = JUDGE_MODELS.get(model_size, JUDGE_MODELS[DEFAULT_JUDGE_MODEL_SIZE])
    data_path = dataset_dir / "data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"No data.parquet in {dataset_dir}")

    scorer_text = scorer_path.read_text(encoding="utf-8")
    samples, tools_per_sample = _load_samples(data_path)
    total = len(samples)

    if total == 0:
        return ScoringResult(
            total=0, scored=0, failed=0,
            mean_score=0.0, median_score=0.0,
            output_dir=dataset_dir,
        )

    # Score each sample
    results: list[dict[str, Any]] = []
    scored = 0
    failed_count = 0
    completed = 0
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def _score_one(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal scored, failed_count, completed

        async with sem:
            scoring_prompt = _build_scoring_prompt(scorer_text, messages, tools=sample_tools)
            score = 0.0
            reasoning = ""
            success = False

            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model=scoring_model,
                        messages=scoring_prompt,
                        temperature=0.0,
                        response_format=SCORE_RESPONSE_SCHEMA,
                    )
                    if not response.choices:
                        raise ValueError("Empty choices in scoring response")
                    raw = response.choices[0].message.content or ""
                    score, reasoning = _parse_score_response(raw)
                    # 0 is a valid worst-quality grade, not a failure; only
                    # parse/API errors are retried and excluded from the stats.
                    if not is_scoring_error(reasoning):
                        success = True
                        break
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                except Exception as exc:
                    logger.warning("Scoring sample %d attempt %d: %s", index, attempt + 1, exc)
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        reasoning = f"[Scoring error] {exc}"

            async with lock:
                results.append({
                    "sample_index": index,
                    "score": score if success else 0.0,
                    "reasoning": reasoning,
                    "scorer": scorer_path.name,
                })
                if success:
                    scored += 1
                else:
                    failed_count += 1
                completed += 1
                if on_progress:
                    on_progress(completed, total)

    tasks = [
        asyncio.create_task(_score_one(i, sample, tools_per_sample[i]))
        for i, sample in enumerate(samples)
    ]
    await asyncio.gather(*tasks)

    # Sort by sample_index
    results.sort(key=lambda r: r["sample_index"])

    # Write scores.parquet co-located with data.parquet
    scores_table = pa.table(
        {
            "sample_index": [r["sample_index"] for r in results],
            "score": [r["score"] for r in results],
            "reasoning": [r["reasoning"] for r in results],
            "scorer": [r["scorer"] for r in results],
        },
        schema=pa.schema([
            pa.field("sample_index", pa.int64()),
            pa.field("score", pa.float64()),
            pa.field("reasoning", pa.string()),
            pa.field("scorer", pa.string()),
        ]),
    )
    pq.write_table(scores_table, dataset_dir / "scores.parquet")

    # Aggregate over judged samples (0 included); exclude only parse/API errors.
    scores = [r["score"] for r in results if not is_scoring_error(r["reasoning"])]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    sorted_scores = sorted(scores)
    median_score = sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0.0

    return ScoringResult(
        total=total,
        scored=scored,
        failed=failed_count,
        mean_score=mean_score,
        median_score=median_score,
        output_dir=dataset_dir,
    )


# ---------------------------------------------------------------------------
# Bring-your-data: score + filter
# ---------------------------------------------------------------------------


async def run_data_filter(
    input_path: Path,
    scorer_path: Path,
    output_dataset_dir: Path,
    client: AsyncOpenAI,
    *,
    threshold: float = 6.0,
    model_size: str = "small",
    concurrency: int = 100,
    max_retries: int = 2,
    on_progress: Callable[[int, int], Any] | None = None,
) -> FilterResult:
    """Score a user-brought dataset and emit a filtered subset.

    The input parquet must follow the same ChatML schema lqh uses for its
    own datasets (``messages`` column, optional ``tools`` / ``audio``).
    Each sample is judged against *scorer_path*; samples scoring strictly
    below *threshold* are dropped.  Failures to score (judge errors) are
    dropped as well so nothing unvetted slips through.

    Outputs under *output_dataset_dir*:

    * ``data.parquet`` — kept rows, full schema preserved (so the output
      is a drop-in dataset for training).
    * ``scores.parquet`` — per-sample score + reasoning + ``kept`` bool.
    * ``summary.json`` — counts, threshold, mean score, keep-rate.
    """
    import pyarrow.parquet as pq

    if not input_path.exists():
        raise FileNotFoundError(f"run_data_filter: input {input_path} does not exist")

    scoring_model = JUDGE_MODELS.get(model_size, JUDGE_MODELS[DEFAULT_JUDGE_MODEL_SIZE])
    scorer_text = scorer_path.read_text(encoding="utf-8")

    input_table = pq.read_table(input_path)
    samples, tools_per_sample = _load_samples(input_path)
    total = len(samples)

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dataset_dir / "scores.parquet"
    summary_path = output_dataset_dir / "summary.json"
    data_path = output_dataset_dir / "data.parquet"

    if total == 0:
        # Empty input — emit empty artifacts and short-circuit.
        pq.write_table(input_table, data_path)
        empty = pa.table(
            {"sample_index": [], "score": [], "reasoning": [], "kept": []},
            schema=pa.schema([
                pa.field("sample_index", pa.int64()),
                pa.field("score", pa.float64()),
                pa.field("reasoning", pa.string()),
                pa.field("kept", pa.bool_()),
            ]),
        )
        pq.write_table(empty, scores_path)
        summary_path.write_text(
            json.dumps(
                {
                    "input": str(input_path),
                    "scorer": str(scorer_path),
                    "threshold": threshold,
                    "total": 0,
                    "scored": 0,
                    "kept": 0,
                    "dropped": 0,
                    "failed": 0,
                    "keep_rate": 0.0,
                    "mean_score": 0.0,
                    "scoring_model": scoring_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return FilterResult(
            total=0, scored=0, kept=0, dropped=0, failed=0,
            threshold=threshold, mean_score=0.0,
            output_dataset_dir=output_dataset_dir,
            scores_path=scores_path, summary_path=summary_path,
        )

    results: list[dict[str, Any] | None] = [None] * total
    scored = 0
    failed_count = 0
    completed = 0
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def _score_one(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal scored, failed_count, completed
        async with sem:
            scoring_prompt = _build_scoring_prompt(scorer_text, messages, tools=sample_tools)
            score = 0.0
            reasoning = ""
            success = False
            for attempt in range(max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model=scoring_model,
                        messages=scoring_prompt,
                        temperature=0.0,
                        response_format=SCORE_RESPONSE_SCHEMA,
                    )
                    if not response.choices:
                        raise ValueError("Empty choices in scoring response")
                    raw = response.choices[0].message.content or ""
                    score, reasoning = _parse_score_response(raw)
                    # 0 is a valid worst-quality grade (it just won't clear the
                    # keep threshold); only parse/API errors are retried/dropped.
                    if not is_scoring_error(reasoning):
                        success = True
                        break
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                except Exception as exc:
                    logger.warning(
                        "run_data_filter: sample %d attempt %d failed: %s",
                        index, attempt + 1, exc,
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        reasoning = f"[Scoring error] {exc}"

            async with lock:
                results[index] = {
                    "sample_index": index,
                    "score": score if success else 0.0,
                    "reasoning": reasoning,
                    "kept": success and score >= threshold,
                }
                if success:
                    scored += 1
                else:
                    failed_count += 1
                completed += 1
                if on_progress:
                    on_progress(completed, total)

    tasks = [
        asyncio.create_task(_score_one(i, sample, tools_per_sample[i]))
        for i, sample in enumerate(samples)
    ]
    await asyncio.gather(*tasks)

    # Assume nothing went missing: results is fully populated.
    rows = [r for r in results if r is not None]
    rows.sort(key=lambda r: r["sample_index"])

    scores_table = pa.table(
        {
            "sample_index": [r["sample_index"] for r in rows],
            "score": [r["score"] for r in rows],
            "reasoning": [r["reasoning"] for r in rows],
            "kept": [r["kept"] for r in rows],
        },
        schema=pa.schema([
            pa.field("sample_index", pa.int64()),
            pa.field("score", pa.float64()),
            pa.field("reasoning", pa.string()),
            pa.field("kept", pa.bool_()),
        ]),
    )
    pq.write_table(scores_table, scores_path)

    # Emit filtered data.parquet preserving the input schema.
    kept_indices = [r["sample_index"] for r in rows if r["kept"]]
    if kept_indices:
        kept_table = input_table.take(pa.array(kept_indices, type=pa.int64()))
    else:
        kept_table = input_table.slice(0, 0)
    pq.write_table(kept_table, data_path)

    kept_count = len(kept_indices)
    dropped = total - kept_count - failed_count
    # Mean over judged samples (0 included); exclude only parse/API errors.
    valid_scores = [r["score"] for r in rows if not is_scoring_error(r["reasoning"])]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    summary = {
        "input": str(input_path),
        "scorer": str(scorer_path),
        "threshold": threshold,
        "total": total,
        "scored": scored,
        "kept": kept_count,
        "dropped": dropped,
        "failed": failed_count,
        "keep_rate": round(kept_count / total, 4) if total else 0.0,
        "mean_score": round(mean_score, 2),
        "scoring_model": scoring_model,
        "scoring_model_size": model_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return FilterResult(
        total=total,
        scored=scored,
        kept=kept_count,
        dropped=dropped,
        failed=failed_count,
        threshold=threshold,
        mean_score=mean_score,
        output_dataset_dir=output_dataset_dir,
        scores_path=scores_path,
        summary_path=summary_path,
    )


# ---------------------------------------------------------------------------
# Failure extraction
# ---------------------------------------------------------------------------

def extract_failures(
    results_path: Path,
    *,
    threshold: float = 6.0,
    min_failures: int = 5,
    max_failures: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract failure cases from a results.parquet file.

    Returns ``(model_failures, scoring_errors)``. Scoring errors (judge API
    failures, parse errors) are partitioned out so callers can render them
    separately — they're not evidence of model regression and shouldn't be
    analyzed alongside genuinely low-scoring samples.

    Hybrid model-failure selection: takes all samples scoring below
    *threshold*, then pads with the lowest-scoring samples until at least
    *min_failures* are collected. Caps at *max_failures*. Sorted ascending.

    Parameters
    ----------
    results_path:
        Path to ``results.parquet`` (must have columns: sample_index,
        messages, score, reasoning).
    threshold:
        Samples scoring strictly below this are considered failures.
    min_failures:
        Minimum number of model-failure samples to return.
    max_failures:
        Cap for both the model-failure and the scoring-error list.
    """
    table = pq.read_table(results_path)
    all_rows: list[dict[str, Any]] = []
    for i in range(len(table)):
        score_val = table.column("score")[i].as_py()
        reasoning = table.column("reasoning")[i].as_py() or ""
        all_rows.append({
            "sample_index": table.column("sample_index")[i].as_py(),
            "messages": json.loads(table.column("messages")[i].as_py()),
            "score": float(score_val) if score_val is not None else 0.0,
            "reasoning": reasoning,
        })

    # Partition: scoring infrastructure errors vs. genuinely scored samples
    scoring_errors = [r for r in all_rows if is_scoring_error(r["reasoning"])]
    valid_rows = [r for r in all_rows if not is_scoring_error(r["reasoning"])]

    # Sort valid rows by score ascending (worst first)
    valid_rows.sort(key=lambda r: r["score"])

    # Collect: all below threshold
    below = [r for r in valid_rows if r["score"] < threshold]

    # Pad with bottom-N if fewer than min_failures
    if len(below) < min_failures:
        seen = {r["sample_index"] for r in below}
        for r in valid_rows:
            if r["sample_index"] not in seen:
                below.append(r)
                seen.add(r["sample_index"])
            if len(below) >= min_failures:
                break

    # Cap and re-sort
    below = below[:max_failures]
    below.sort(key=lambda r: r["score"])

    scoring_errors = scoring_errors[:max_failures]
    return below, scoring_errors
