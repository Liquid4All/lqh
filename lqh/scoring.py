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
from openai import AsyncOpenAI, RateLimitError

from lqh.client import _parse_retry_after
from lqh.config import default_api_base_url
from lqh.runner import APIModelRunner, ModelRunner

__all__ = [
    "DEFAULT_JUDGE_MODEL_SIZE",
    "DEFAULT_MAX_RETRIES",
    "FAILURE_WARN_FRACTION",
    "JUDGE_MODELS",
    "JUDGE_TIMEOUT_S",
    "MODEL_EVAL_TIMEOUT_S",
    "FilterResult",
    "ScoringResult",
    "extract_failures",
    "failure_warning",
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


# Per-sample deadline, in seconds. The bound covers a whole sample — every
# retry inside it — rather than each attempt, so a wedged sample cannot
# multiply the retry ladder by the upstream budget (270 s per call). A judge
# call is short structured JSON at temperature 0, so healthy is single-digit
# seconds and this is still ~20x p99. Expiry takes the ordinary judge-failure
# path: score 0.0 with an is_scoring_error reasoning, excluded from
# mean/median and counted in ``failed``.
JUDGE_TIMEOUT_S = 120.0
# The backend bounds a single upstream call at LQH_UPSTREAM_BUDGET (270 s) and
# surfaces the overrun as a 504. model_eval makes two such calls back to back —
# the generation, then the judge — so its deadline is their sum. Anything less
# would turn a slow-but-correct generation into a scored failure: the sample
# answers at 275 s, the judge never gets to run, and the run records a 0.
_UPSTREAM_BUDGET_S = 270.0
MODEL_EVAL_TIMEOUT_S = _UPSTREAM_BUDGET_S + JUDGE_TIMEOUT_S

# Attempts per sample = DEFAULT_MAX_RETRIES + 1. One retry absorbs a
# disconnect or a single upstream glitch; needing more than that means
# something is actually wrong, and every extra attempt is another
# upstream-budget's worth of waiting for the user. The clients these runners
# are given must set the OpenAI SDK's own ``max_retries=0`` — that layer is
# invisible and silently multiplies this ladder (see lqh.client.create_client).
DEFAULT_MAX_RETRIES = 1

# Share of a run that may fail to score before the result is suspect rather
# than merely unlucky. Above it, callers surface a warning: at that rate the
# cause is usually the judge, the scorer, or API access, not bad luck.
FAILURE_WARN_FRACTION = 0.10

# Smallest source for which the per-source thinning check means anything. On
# 8 predictions one transient judge error is 12.5% and would raise a run-level
# warning that propagates to every sweep row.
_THINNING_MIN_SOURCE = 20


def failure_warning(failed: int, total: int) -> str | None:
    """One-line warning when more than :data:`FAILURE_WARN_FRACTION` of a run
    could not be scored, else ``None``.

    A handful of judge errors is noise — they're already reported inline and
    excluded from the statistics. A large share is a broken run wearing the
    costume of a finished one, so it gets said out loud.
    """
    if total <= 0 or failed <= 0:
        return None
    share = failed / total
    if share <= FAILURE_WARN_FRACTION:
        return None
    return (
        f"  ⚠️  {failed}/{total} samples ({share:.0%}) could not be scored — "
        f"above {FAILURE_WARN_FRACTION:.0%}, which usually means the judge, "
        "the scorer, or API access is at fault rather than transient errors. "
        "Treat the scores below as provisional."
    )


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
    # Rows kept without a verdict because the judge failed on them. Counted in
    # both ``kept`` and ``failed`` — see ``run_data_filter`` for why the filter
    # fails open.
    kept_unjudged: int = 0


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


# A 429 is the server pacing us, not a failed attempt, so it does not consume
# the retry budget. When a deadline wraps the sample, that deadline IS the
# bound and the wait count is unlimited — capping it there would fail a sample
# the server was going to serve, with time still on the clock. These two only
# govern the unbounded case (``sample_timeout=0``), where something has to stop
# a sample from waiting forever.
_MAX_RATE_LIMIT_WAITS = 5
_RATE_LIMIT_WAIT_CAP_S = 30.0


def _rate_limit_wait(
    exc: BaseException,
    waits_so_far: int,
    *,
    bounded: bool,
    remaining: float | None = None,
) -> float:
    """How long to wait after a 429: the server's ``Retry-After`` if it sent
    one, else exponential backoff.

    The server's number is honoured rather than clamped to some smaller
    constant — retrying earlier than it asked just earns another 429 and
    spends an attempt to learn nothing. But it is never longer than the
    sample has left (*remaining*): sleeping past your own deadline converts
    time you could have spent on an attempt into a guaranteed timeout. With
    no deadline at all, a fixed cap is the only thing standing between a
    misreported ``Retry-After`` and an unbounded sleep.
    """
    try:
        retry_after = _parse_retry_after(exc)  # type: ignore[arg-type]
    except Exception:
        # Header parsing must never be the thing that kills a run: this runs
        # inside an ``except`` clause, so anything raised here escapes the
        # per-sample handling entirely and aborts every other sample with it.
        retry_after = None
    wait = float(retry_after if retry_after is not None else 2 ** (waits_so_far - 1))
    if not bounded:
        return max(1.0, min(wait, _RATE_LIMIT_WAIT_CAP_S))
    if remaining is not None:
        wait = min(wait, max(0.0, remaining))
    return max(0.0, wait)


async def _judge_sample(
    client: AsyncOpenAI,
    scoring_model: str,
    scoring_prompt: list[dict],
    *,
    max_retries: int,
    index: int,
    label: str,
    bounded: bool,
    deadline: float | None = None,
) -> tuple[float, str, bool]:
    """Run the judge on one prepared prompt. Returns ``(score, reasoning, ok)``.

    Shared by all three runners so the retry ladder cannot drift between them.
    A parseable judge response is a real score — including 0, which is the
    rubric's worst grade (empty/refusal/unrelated), NOT an infrastructure
    failure. Only parse/API errors (flagged by :func:`is_scoring_error`) are
    retried, and when the last attempt still fails the caller gets
    ``ok=False`` with the failure recorded in *reasoning*.

    Rate limits are handled separately from failures: scoring runs with the
    OpenAI SDK's own retry layer switched off (it would multiply the deadline
    below), so this is the only thing left that honours ``Retry-After``.
    Without it a 429 burst — 100-wide concurrency against a per-minute limit —
    would burn both attempts in a second and fail a large share of the run.

    No deadline of its own: callers bound the whole sample (see
    :data:`JUDGE_TIMEOUT_S`), which is what keeps the ladder from multiplying.
    """
    score = 0.0
    reasoning = ""
    attempt = 0
    rate_limit_waits = 0
    started = asyncio.get_running_loop().time()
    while True:
        last_attempt = attempt >= max_retries
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
            if not is_scoring_error(reasoning):
                return score, reasoning, True
        except RateLimitError as exc:
            # Only the unbounded case counts waits — under a deadline, that
            # deadline decides when to give up.
            if not bounded and rate_limit_waits >= _MAX_RATE_LIMIT_WAITS:
                reasoning = f"[Scoring error] {exc}"
                break
            rate_limit_waits += 1
            remaining = (
                deadline - (asyncio.get_running_loop().time() - started)
                if deadline is not None
                else None
            )
            wait = _rate_limit_wait(
                exc, rate_limit_waits, bounded=bounded, remaining=remaining,
            )
            logger.warning(
                "%s: sample %d rate-limited (wait %d), sleeping %.1fs",
                label, index, rate_limit_waits, wait,
            )
            await asyncio.sleep(wait)
            continue  # not a failed attempt — don't spend the retry budget
        except Exception as exc:
            logger.warning(
                "%s: sample %d attempt %d/%d failed: %s",
                label, index, attempt + 1, max_retries + 1, exc,
            )
            if last_attempt:
                reasoning = f"[Scoring error] {exc}"
        if last_attempt:
            break
        await asyncio.sleep(2 ** attempt)
        attempt += 1
    return 0.0, reasoning, False


async def _run_inference(
    runner: ModelRunner,
    inf_messages: list[dict],
    *,
    model: str,
    response_format: dict[str, Any] | None,
    tools: list[dict] | None,
    max_retries: int,
    index: int,
    bounded: bool,
    deadline: float | None = None,
) -> Any:
    """Generate one model_eval response, with the same retry ladder the judge
    gets — rate-limit handling included.

    The judge owns its retries in :func:`_judge_sample`; without a matching
    ladder here the generation half of a model_eval sample would be the one
    unprotected call in the pipeline — scoring clients run with the OpenAI
    SDK's retry layer switched off (it multiplies the deadline), so a single
    dropped connection would fail the sample outright with most of the
    deadline unspent. Bounded by that same deadline, not multiplied by it.

    429s take the same branch they take on the judge side: honour
    ``Retry-After``, don't spend an attempt. Treating them as ordinary
    failures here would undo the judge-side fix for any model_eval run, since
    a rate-limit burst hits generation first.
    """
    last_exc: Exception | None = None
    started = asyncio.get_running_loop().time()
    rate_limit_waits = 0
    attempt = 0
    # max(0, ...) so a negative max_retries still makes one attempt, matching
    # _judge_sample rather than falling out of the loop with nothing to raise.
    max_attempts = max(0, max_retries) + 1
    while attempt < max_attempts:
        try:
            return await runner.complete(
                inf_messages,
                model=model,
                temperature=0.0,
                response_format=response_format,
                tools=tools,
            )
        except RateLimitError as exc:
            last_exc = exc
            if not bounded and rate_limit_waits >= _MAX_RATE_LIMIT_WAITS:
                break
            rate_limit_waits += 1
            remaining = (
                deadline - (asyncio.get_running_loop().time() - started)
                if deadline is not None
                else None
            )
            wait = _rate_limit_wait(
                exc, rate_limit_waits, bounded=bounded, remaining=remaining,
            )
            logger.warning(
                "Inference for sample %d rate-limited (wait %d), sleeping %.1fs",
                index, rate_limit_waits, wait,
            )
            await asyncio.sleep(wait)
            continue  # not a failed attempt — don't spend the retry budget
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Inference for sample %d failed (attempt %d/%d): %s",
                index, attempt + 1, max_attempts, exc,
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)
        attempt += 1
    if last_exc is None:
        # Only reachable if max_attempts was 0, which max(0, ...) prevents.
        raise RuntimeError("inference made no attempts")
    raise last_exc


def _timeout_reasoning(seconds: float) -> str:
    """Reasoning string for a sample that blew its deadline.

    Carries the ``[Scoring error]`` marker so it flows through the existing
    failure path — excluded from mean/median, counted in ``failed``.
    """
    return f"[Scoring error] sample timed out after {seconds:.0f}s"


# Interrupted runs write here instead of over the real artifact names. Callers
# gate on the existence of results.parquet / summary.json — the DPO watcher
# skips held-out scoring for any iteration that already has a summary.json, and
# that mean feeds best-checkpoint selection — so a partial file under the real
# name would pin an iteration to a score computed over a handful of samples,
# forever. Under these names an interrupted run correctly looks unscored, and
# the work already paid for is still on disk to salvage.
PARTIAL_SUFFIX = ".partial"


def _results_name(partial: bool) -> str:
    return f"results{PARTIAL_SUFFIX}.parquet" if partial else "results.parquet"


def _summary_name(partial: bool) -> str:
    return f"summary{PARTIAL_SUFFIX}.json" if partial else "summary.json"


def _valid_concurrency(concurrency: int) -> int:
    """At least one slot.

    ``asyncio.Semaphore(0)`` can never be acquired, and the per-sample
    deadline only starts once a sample holds a slot — so a zero here would
    hang the whole run forever with no timeout to rescue it. Negative values
    are clamped the same way rather than raising: a bad concurrency is never
    worth failing a run over.
    """
    return max(1, int(concurrency))


def _resolve_sample_timeout(
    sample_timeout: float | None, *, run_inference: bool = False
) -> float | None:
    """Resolve the per-sample deadline. ``None`` picks the default for the
    mode; a non-positive value disables the bound entirely."""
    if sample_timeout is None:
        return MODEL_EVAL_TIMEOUT_S if run_inference else JUDGE_TIMEOUT_S
    return sample_timeout if sample_timeout > 0 else None


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
    max_retries: int = DEFAULT_MAX_RETRIES,
    sample_timeout: float | None = None,
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
        Retries per sample on scoring failure (attempts = this + 1).
    sample_timeout:
        Per-sample deadline in seconds, covering the sample as a whole
        including its retries. ``None`` (default) picks
        :data:`JUDGE_TIMEOUT_S`, or :data:`MODEL_EVAL_TIMEOUT_S` when
        *run_inference* is set; a non-positive value disables the bound.
        A sample that blows it is recorded as a judge failure, so the run
        finishes in bounded time instead of waiting out one straggler.
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
    # Which samples have already been counted. Not the same as ``results[i] is
    # not None``: the inference-failure paths below count a sample without
    # writing a row for it, and the deadline can expire in the instant between
    # the body finishing and the watchdog being checked. Without this the
    # sample would be counted twice.
    recorded: list[bool] = [False] * total
    # What each sample is actually being judged on, published before the judge
    # call. In model_eval that's the model's own output, and it is exactly the
    # evidence you want kept when the judge is the thing that timed out — both
    # for the results row and for the debug replay script.
    judged: list[dict[str, Any] | None] = [None] * total
    # Set once the run stops accepting results, so a sample that outlives its
    # cancellation cannot write into a run that has already been serialized.
    # Same guard engine.py uses. Today the write path below contains no await,
    # so no task can interleave with it anyway — this keeps that safe if one
    # is ever added.
    closed = False
    debug_log: list[dict[str, Any]] = []  # low-scoring samples for debugging
    scored = 0
    failed_count = 0
    completed = 0
    sem = asyncio.Semaphore(_valid_concurrency(concurrency))
    lock = asyncio.Lock()
    deadline = _resolve_sample_timeout(sample_timeout, run_inference=run_inference)

    async def _score_one(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal failed_count, completed

        # The deadline starts once the sample actually has a slot — queueing
        # behind the semaphore is not the sample's fault and must not count
        # against it.
        async with sem:
            try:
                async with asyncio.timeout(deadline):
                    await _score_one_body(index, messages, sample_tools)
            except TimeoutError:
                logger.warning(
                    "Scoring timed out for sample %d after %ss", index, deadline,
                )
                async with lock:
                    if closed or recorded[index]:
                        return  # run already serialized, or body already counted
                    recorded[index] = True
                    ctx = judged[index]
                    reasoning = _timeout_reasoning(deadline or 0.0)
                    results[index] = {
                        "sample_index": index,
                        "messages": json.dumps(
                            ctx["scored_messages"] if ctx else messages,
                            ensure_ascii=False,
                        ),
                        "score": 0.0,
                        "reasoning": reasoning,
                    }
                    failed_count += 1
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

                    # A sample that hung is precisely the one a user chasing a
                    # slow run wants to replay by hand, so it gets a debug
                    # entry like any other — including when inference itself
                    # is what hung, where model_response is simply null and
                    # the replay script reproduces the call that never
                    # answered.
                    if run_inference and ctx is not None:
                        debug_log.append({
                            "sample_index": index,
                            "score": 0.0,
                            "reasoning": reasoning,
                            "inference_model": inference_model or "orchestration",
                            "inference_messages_sent": ctx["inf_messages"],
                            "model_response": ctx["assistant_content"],
                            "reference_messages": ctx["original_messages"],
                        })

    async def _score_one_body(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal scored, failed_count, completed

        original_messages = messages
        scored_messages = messages
        inf_messages: list[dict] | None = None
        assistant_content: str | None = None

        # Optionally run inference
        if run_inference:
            unlabelled = _strip_trailing_assistant(messages)
            if not unlabelled:
                async with lock:
                    if closed:
                        return
                    recorded[index] = True
                    # A row like every other failure mode, so results.parquet
                    # always has one row per sample.
                    results[index] = {
                        "sample_index": index,
                        "messages": json.dumps(messages, ensure_ascii=False),
                        "score": 0.0,
                        "reasoning": (
                            "[Scoring error] nothing to infer from: the sample "
                            "has no non-assistant turns"
                        ),
                    }
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

            # Published BEFORE the call, not after: if inference is the thing
            # that hangs, the timeout handler must not fall back to the
            # original conversation, which still carries the gold assistant
            # turn and would be recorded as though the model had produced it.
            # Until inference answers, what is on the table is the unlabelled
            # prompt and nothing else.
            judged[index] = {
                "scored_messages": unlabelled,
                "inf_messages": inf_messages,
                "assistant_content": None,
                "original_messages": original_messages,
            }

            runner = inference_runner or APIModelRunner(client)
            try:
                inf_response = await _run_inference(
                    runner,
                    inf_messages,
                    model=inference_model or "orchestration",
                    response_format=inference_response_format,
                    tools=sample_tools,
                    max_retries=max_retries,
                    index=index,
                    bounded=deadline is not None,
                    deadline=deadline,
                )
                assistant_content = inf_response.content
                scored_messages = unlabelled + [{"role": "assistant", "content": assistant_content}]
            except Exception as exc:
                logger.error("Inference failed for sample %d: %s", index, exc)
                async with lock:
                    if closed:
                        return
                    recorded[index] = True
                    # A row, like every other failure mode. Without one this
                    # sample is counted in `failed` but absent from
                    # results.parquet, so the file silently has fewer rows
                    # than the run had samples and the failure is invisible to
                    # anything reading the artifact.
                    reasoning = f"[Scoring error] inference failed: {exc}"
                    results[index] = {
                        "sample_index": index,
                        "messages": json.dumps(unlabelled, ensure_ascii=False),
                        "score": 0.0,
                        "reasoning": reasoning,
                    }
                    failed_count += 1
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)
                    if debug:
                        debug_log.append({
                            "sample_index": index,
                            "score": 0.0,
                            "reasoning": reasoning,
                            "inference_model": inference_model or "orchestration",
                            "inference_messages_sent": inf_messages,
                            "model_response": None,
                            "reference_messages": original_messages,
                        })
                return

        # Score the sample
        scoring_prompt = _build_scoring_prompt(
            scorer_text,
            scored_messages,
            reference_messages=original_messages if run_inference else None,
            tools=sample_tools,
        )
        judged[index] = {
            "scored_messages": scored_messages,
            "inf_messages": inf_messages,
            "assistant_content": assistant_content,
            "original_messages": original_messages,
        }

        score, reasoning, success = await _judge_sample(
            client, scoring_model, scoring_prompt,
            max_retries=max_retries, index=index, label="run_scoring",
            bounded=deadline is not None, deadline=deadline,
        )

        async with lock:
            if closed:
                return
            recorded[index] = True
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
    cancellation: BaseException | None = None
    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt) as exc:
        # Ctrl-C at 199/200 used to throw away all 199 finished scores and the
        # money spent on them, because nothing is written until every task has
        # returned. The scores are already in hand, so they go to disk on the
        # way out — but under PARTIAL_SUFFIX names, never the real ones.
        #
        # That naming is load-bearing, not cosmetic. Callers gate on the
        # existence of results.parquet / summary.json: the DPO watcher skips
        # held-out scoring for any iteration whose summary.json exists, and
        # the resulting mean feeds best-checkpoint selection. A partial file
        # under the real name would permanently pin that iteration to a score
        # computed over a handful of samples — worse than the data loss this
        # is meant to fix. Under a partial name the run simply looks unscored,
        # which it is, and the scores are still there for a human to salvage.
        #
        # Covers cancellation, not process death; durable incremental writes
        # with a resume path are separate work.
        #
        # run_data_scoring and run_data_filter deliberately do NOT do this.
        # Their output is a dataset, not a report: a partial filtered
        # data.parquet is a poisoned training input with no marker inside the
        # file, and handlers read their summary.json for kept/total
        # provenance. Losing an interrupted filter run is the safe outcome
        # there; losing an interrupted eval is not.
        cancellation = exc
        closed = True
        for task in tasks:
            task.cancel()
        logger.warning(
            "run_scoring interrupted: writing the %d/%d samples already scored "
            "to %s (partial artifacts)", completed, total, output_dir,
        )

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

    pq.write_table(table, output_dir / _results_name(cancellation is not None))

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
    # Optional key — absent on healthy runs and on files written before this
    # existed, so readers must treat it as such.
    run_warning = failure_warning(failed_count, total)
    if run_warning:
        summary["failure_warning"] = run_warning.strip()
    if cancellation is not None:
        # Belt and braces on top of the partial filename: num_samples is the
        # run that was asked for, and only these say how much of it ran.
        summary["interrupted"] = True
        summary["num_completed"] = completed
    if run_inference:
        summary["inference_model"] = inference_model or "orchestration"
        if inference_system_prompt:
            summary["inference_system_prompt"] = inference_system_prompt

    (output_dir / _summary_name(cancellation is not None)).write_text(
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

    # Logged as well as returned: most callers of run_scoring are automated
    # (watcher checkpoint evals, DPO iterations, cloud scoring) and render
    # only the mean, so without this a run that lost a third of its samples
    # looks exactly like a healthy one.
    warning = failure_warning(failed_count, total)
    if warning:
        logger.warning("run_scoring: %s", warning.strip())

    if cancellation is not None:
        # The artifacts are on disk now; the cancellation still has to happen.
        # Swallowing it would turn Ctrl-C into "returned a partial result as
        # though it were the whole run". The ORIGINAL exception is re-raised,
        # not a fresh CancelledError: a KeyboardInterrupt that arrives here
        # must stay a KeyboardInterrupt, or the CLI's `except KeyboardInterrupt`
        # handlers stop matching and Ctrl-C prints a traceback instead of
        # exiting cleanly.
        raise cancellation

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


def score_distribution_stats(scores: list[float]) -> dict[str, Any] | None:
    """Percentiles + per-integer-bucket histogram over judge scores.

    Single source of truth for distribution stats — the ``run_scoring`` tool
    rendering and the GPU-eval ``eval_result.json`` both go through here so
    the two paths can never disagree. Returns ``None`` for empty input.

    Callers must pass only genuinely scored samples (filter scoring errors
    via :func:`is_scoring_error` first) — 0 is the rubric's worst-quality
    grade, not an error marker, so it gets its own bucket.

    Percentile estimator: nearest-index on the sorted values,
    ``idx = round(p * (n - 1))`` with Python banker's rounding — i.e.
    numpy's ``method="nearest"``, NOT the classical nearest-rank
    ``ceil(p * n)`` definition (for scores 1..10 this reports p10=2 where
    nearest-rank would report 1). Chosen deliberately: it is what the
    ``run_scoring`` rendering has always used, and changing it would make
    historical and new distributions incomparable.

    Histogram keys are strings ("0".."10") so the dict round-trips through
    JSON unchanged.
    """
    if not scores:
        return None
    ordered = sorted(scores)
    n = len(ordered)

    def _q(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(ordered[idx])

    buckets = {b: 0 for b in range(0, 11)}
    for s in ordered:
        b = max(0, min(10, int(s)))
        buckets[b] += 1

    return {
        "n": n,
        "percentiles": {
            "p10": _q(0.10),
            "p25": _q(0.25),
            "p50": _q(0.50),
            "p75": _q(0.75),
            "p90": _q(0.90),
        },
        "histogram": {str(b): c for b, c in buckets.items()},
    }


def format_score_distribution_text(dist: dict[str, Any]) -> str:
    """Render :func:`score_distribution_stats` output as the 4-6 line block
    (quantile line + horizontal mini-histogram) shown in tool results."""
    n = dist["n"]
    pct = dist["percentiles"]
    buckets = {int(b): c for b, c in dist["histogram"].items()}

    max_count = max(buckets.values()) if buckets else 1
    bar_width = 24
    lines: list[str] = []
    lines.append("  Score distribution (n={}):".format(n))
    lines.append(
        "    p10={:.1f}  p25={:.1f}  p50={:.1f}  p75={:.1f}  p90={:.1f}".format(
            pct["p10"], pct["p25"], pct["p50"], pct["p75"], pct["p90"]
        )
    )
    # Bucket 0 (worst grade) included; absent in pre-fix dicts → .get.
    for b in range(10, -1, -1):
        c = buckets.get(b, 0)
        if c == 0:
            continue
        bar = "█" * max(1, int(round(c / max_count * bar_width)))
        share = 100.0 * c / n
        lines.append(f"    {b:>2} | {bar:<{bar_width}}  {c:>5}  ({share:4.1f}%)")
    return "\n".join(lines)


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

    # How many predictions each source actually contributed, so a source that
    # was scored on 3 of its 100 rows is distinguishable from one scored on
    # all 100. Without this the macro-average weights them identically and a
    # badly thinned source silently carries the same vote as a complete one.
    attempted_by_label: dict[str, int] = {}
    for sample_idx in src_by_idx:
        attempted_by_label[src_by_idx[sample_idx]] = (
            attempted_by_label.get(src_by_idx[sample_idx], 0) + 1
        )
    if not src_by_idx:
        attempted_by_label = {"all": result.total}

    per_source: dict[str, Any] = {}
    per_source_means: list[float] = []
    per_source_medians: list[float] = []
    thinned: list[str] = []
    wiped: list[str] = []
    for label in sorted(all_labels):
        scores_list = grouped.get(label, [])
        stats = _score_stats(scores_list)
        attempted = attempted_by_label.get(label, len(scores_list))
        per_source[label] = {
            "num_scored": len(scores_list),
            "num_attempted": attempted,
            "num_failed": max(0, attempted - len(scores_list)),
            "scores": stats,
        }
        # A global 10% threshold can't see one small source losing most of its
        # rows, so each source is checked on its own too — but only once it is
        # big enough for a share to mean anything. Below the floor a single
        # transient error clears 10% on its own, and this string rides all the
        # way out to every sweep row.
        #
        # A source that lost EVERYTHING bypasses the floor: it contributes no
        # mean at all, so it silently drops out of the macro-average and the
        # headline is computed over the survivors. Size has nothing to do with
        # whether that is worth saying.
        if attempted > 0 and not scores_list:
            wiped.append(f"{label} (0/{attempted})")
        elif attempted >= _THINNING_MIN_SOURCE and failure_warning(
            attempted - len(scores_list), attempted
        ):
            thinned.append(f"{label} ({len(scores_list)}/{attempted})")
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
    # Distribution over all valid scores (no per-source split — token bloat
    # for little routing value). Omitted entirely when nothing scored;
    # readers must treat the key as optional (old files / version skew).
    all_valid_scores = [s for scores_list in grouped.values() for s in scores_list]
    dist = score_distribution_stats(all_valid_scores)
    if dist is not None:
        payload["score_distribution"] = dist
    # Optional key — readers must tolerate its absence (old files, healthy
    # runs). Present so a sweep comparing configs can see that one of the
    # means was taken over a badly thinned sample set.
    warning = failure_warning(result.failed, result.total)
    if (thinned or wiped) and not warning:
        # The run as a whole looks fine, but at least one source doesn't. A
        # wiped-only run must not get the thinning sentence: that source was
        # scored on NONE of its predictions and carries NO weight in the
        # macro-average, which is the opposite of what thinning prose says.
        warning = (
            "  ⚠️  Some sources produced no usable scores at all."
            if wiped and not thinned
            else (
                "  ⚠️  Some sources were scored on only part of their "
                "predictions; their per-source means carry the same weight in "
                "the macro-average as fully scored ones."
            )
        )
    if warning:
        parts = [warning.strip()]
        if wiped:
            # Stronger than thinning: these contribute no mean, so they drop
            # out of the macro-average entirely and the headline is computed
            # over the survivors alone.
            parts.append(
                "Sources with NO usable scores, excluded from the headline: "
                f"{', '.join(wiped)}."
            )
        if thinned:
            parts.append(f"Thinned sources: {', '.join(thinned)}.")
        warning = " ".join(parts)
        payload["failure_warning"] = warning
        logger.warning("score_predictions_by_source: %s", warning)
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
    max_retries: int = DEFAULT_MAX_RETRIES,
    sample_timeout: float | None = None,
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
    max_retries:
        Retries per sample on scoring failure (attempts = this + 1).
    sample_timeout:
        Per-sample deadline in seconds covering the sample and its retries;
        ``None`` picks :data:`JUDGE_TIMEOUT_S`, non-positive disables it.
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

    # Score each sample. Slots are indexed by sample so the timeout path can
    # tell "not recorded yet" from "already recorded".
    slots: list[dict[str, Any] | None] = [None] * total
    scored = 0
    failed_count = 0
    completed = 0
    sem = asyncio.Semaphore(_valid_concurrency(concurrency))
    lock = asyncio.Lock()
    deadline = _resolve_sample_timeout(sample_timeout)

    async def _score_one(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal failed_count, completed

        # Deadline starts after the semaphore: waiting for a slot is not the
        # sample's fault.
        async with sem:
            try:
                async with asyncio.timeout(deadline):
                    await _score_one_body(index, messages, sample_tools)
            except TimeoutError:
                logger.warning(
                    "Scoring timed out for sample %d after %ss", index, deadline,
                )
                async with lock:
                    if slots[index] is not None:
                        return  # the body recorded before the deadline landed
                    slots[index] = {
                        "sample_index": index,
                        "score": 0.0,
                        "reasoning": _timeout_reasoning(deadline or 0.0),
                        "scorer": scorer_path.name,
                    }
                    failed_count += 1
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

    async def _score_one_body(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal scored, failed_count, completed

        scoring_prompt = _build_scoring_prompt(scorer_text, messages, tools=sample_tools)
        score, reasoning, success = await _judge_sample(
            client, scoring_model, scoring_prompt,
            max_retries=max_retries, index=index, label="run_data_scoring",
            bounded=deadline is not None, deadline=deadline,
        )

        async with lock:
            slots[index] = {
                "sample_index": index,
                "score": score if success else 0.0,
                "reasoning": reasoning,
                "scorer": scorer_path.name,
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

    # Already in sample order by construction.
    results = [r for r in slots if r is not None]

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

    warning = failure_warning(failed_count, total)
    if warning:
        logger.warning("run_data_scoring: %s", warning.strip())

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
    max_retries: int = DEFAULT_MAX_RETRIES,
    sample_timeout: float | None = None,
    on_progress: Callable[[int, int], Any] | None = None,
) -> FilterResult:
    """Score a user-brought dataset and emit a filtered subset.

    The input parquet must follow the same ChatML schema lqh uses for its
    own datasets (``messages`` column, optional ``tools`` / ``audio``).
    Each sample is judged against *scorer_path*; samples scoring strictly
    below *threshold* are dropped.

    **Judge failures fail open**: a sample the judge could not score (API
    error, unparseable verdict, blown deadline) is KEPT and reported as
    ``kept_unjudged``. This is user-brought data — deleting a row the user
    deliberately supplied because our judge had a bad minute is a much worse
    outcome than letting a few unvetted rows through, and the count is
    surfaced so a re-run is an informed choice.

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
                    "kept_unjudged": 0,
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
    sem = asyncio.Semaphore(_valid_concurrency(concurrency))
    lock = asyncio.Lock()
    deadline = _resolve_sample_timeout(sample_timeout)

    async def _score_one(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal failed_count, completed
        # Deadline starts after the semaphore: waiting for a slot is not the
        # sample's fault.
        async with sem:
            try:
                async with asyncio.timeout(deadline):
                    await _score_one_body(index, messages, sample_tools)
            except TimeoutError:
                logger.warning(
                    "run_data_filter: sample %d timed out after %ss", index, deadline,
                )
                async with lock:
                    if results[index] is not None:
                        return  # the body recorded before the deadline landed
                    results[index] = {
                        "sample_index": index,
                        "score": 0.0,
                        "reasoning": _timeout_reasoning(deadline or 0.0),
                        # Fail open — see the docstring.
                        "kept": True,
                    }
                    failed_count += 1
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

    async def _score_one_body(index: int, messages: list[dict], sample_tools: list[dict] | None) -> None:
        nonlocal scored, failed_count, completed

        scoring_prompt = _build_scoring_prompt(scorer_text, messages, tools=sample_tools)
        score, reasoning, success = await _judge_sample(
            client, scoring_model, scoring_prompt,
            max_retries=max_retries, index=index, label="run_data_filter",
            bounded=deadline is not None, deadline=deadline,
        )

        async with lock:
            results[index] = {
                "sample_index": index,
                "score": score if success else 0.0,
                "reasoning": reasoning,
                # A judged sample lives or dies by the threshold; an unjudged
                # one is kept (fail open — see the docstring).
                "kept": score >= threshold if success else True,
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
    # Unjudged rows are kept, so "dropped" is exactly the judged rows that
    # missed the threshold — everything not kept.
    dropped = total - kept_count
    kept_unjudged = sum(1 for r in rows if r["kept"] and is_scoring_error(r["reasoning"]))
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
        "kept_unjudged": kept_unjudged,
        "dropped": dropped,
        "failed": failed_count,
        "keep_rate": round(kept_count / total, 4) if total else 0.0,
        "mean_score": round(mean_score, 2),
        "scoring_model": scoring_model,
        "scoring_model_size": model_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    warning = failure_warning(failed_count, total)
    if warning:
        logger.warning("run_data_filter: %s", warning.strip())

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
        kept_unjudged=kept_unjudged,
    )


# ---------------------------------------------------------------------------
# Failure extraction
# ---------------------------------------------------------------------------

def _load_result_rows(
    results_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a ``results.parquet`` into row dicts and partition scoring
    infrastructure errors (judge API failures, parse errors) from genuinely
    scored samples. Returns ``(valid_rows, scoring_errors)``."""
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

    scoring_errors = [r for r in all_rows if is_scoring_error(r["reasoning"])]
    valid_rows = [r for r in all_rows if not is_scoring_error(r["reasoning"])]
    return valid_rows, scoring_errors


def browse_results(
    results_path: Path,
    *,
    score_min: float | None = None,
    score_max: float | None = None,
    sort: str = "asc",
    limit: int = 15,
    offset: int = 0,
    seed: int = 0,
    sample_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Flexible browse over an eval run's scored samples.

    Complements :func:`extract_failures` (which answers "show me the worst"):
    this answers "show me a slice" — a score band, a page, a random draw, or
    exact samples. The failure-analysis skill uses it to tell extreme-outlier
    failures apart from uniformly mediocre ones.

    Returns ``(rows, scoring_errors, total_matching)`` where *rows* is the
    ``[offset:offset+limit]`` window of the filtered+sorted selection and
    *total_matching* the filtered count before paging. ``sort="random"``
    shuffles with ``random.Random(seed)`` so pages are stable across calls.
    ``sample_indices`` short-circuits every filter and returns exactly those
    samples in the given order (missing indices are silently skipped).
    """
    import random as _random

    valid_rows, scoring_errors = _load_result_rows(results_path)

    if sample_indices is not None:
        by_idx = {r["sample_index"]: r for r in valid_rows}
        rows = [by_idx[i] for i in sample_indices if i in by_idx]
        return rows, scoring_errors, len(rows)

    matching = [
        r for r in valid_rows
        if (score_min is None or r["score"] >= score_min)
        and (score_max is None or r["score"] <= score_max)
    ]
    if sort == "desc":
        matching.sort(key=lambda r: r["score"], reverse=True)
    elif sort == "random":
        _random.Random(seed).shuffle(matching)
    else:
        matching.sort(key=lambda r: r["score"])

    total = len(matching)
    return matching[offset:offset + limit], scoring_errors, total


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
    valid_rows, scoring_errors = _load_result_rows(results_path)

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
