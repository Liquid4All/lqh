"""LLM judge verification for E2E test artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from tests.harness.scenarios import Scenario

logger = logging.getLogger(__name__)

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief evaluation (2-3 sentences).",
                },
                "score": {
                    "type": "integer",
                    "description": "Score from 1 to 10.",
                },
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class JudgeResult:
    artifact: str  # e.g., "SPEC.md"
    score: int
    reasoning: str


def _coerce_judge_payload(data: object) -> tuple[int, str]:
    """Extract (score, reasoning) from whatever the judge model returned.

    Even with JUDGE_SCHEMA enforced, the model has been observed returning
    bare JSON values (a single number, a string, or a list) instead of a
    dict. Handle those shapes gracefully so one bad judge response doesn't
    zero-out a benchmark score via an ``AttributeError: 'float' object has
    no attribute 'get'`` down the line.
    """
    # The judge scores on a 0-10 scale. Clamp every parsed value into that
    # range: a model occasionally emits a wildly out-of-range bare number
    # (e.g. "1" followed by hundreds of zeros), which would otherwise poison
    # artifact_judge_score -> composite_score and every downstream average.
    def _clamp(raw: object) -> int:
        try:
            return max(0, min(10, int(float(raw))))
        except (TypeError, ValueError, OverflowError):
            return 0

    # Normal case: JSON object with the expected keys.
    if isinstance(data, dict):
        score = _clamp(data.get("score", 0))
        reasoning = str(data.get("reasoning", ""))
        return score, reasoning

    # Model returned a bare number ("7", 7.5, etc.) — treat it as the score.
    if isinstance(data, (int, float)):
        return _clamp(data), "(judge returned bare number instead of JSON object)"

    # Model returned a string — try to coerce, otherwise fall through to 0.
    if isinstance(data, str):
        try:
            float(data.strip())  # validate it parses
        except (TypeError, ValueError):
            return 0, f"(judge returned unparseable string: {data[:200]!r})"
        return _clamp(data.strip()), f"(judge returned bare string: {data[:120]!r})"

    # Model returned a list or something else — give up but record the shape.
    return 0, f"(judge returned unexpected type {type(data).__name__}: {str(data)[:200]!r})"


async def judge_artifacts(
    client: AsyncOpenAI,
    scenario: Scenario,
    artifacts: dict[str, str],
    exclude_paths: set[str] | None = None,
) -> list[JudgeResult]:
    """Run LLM judge on each key artifact.

    Judges artifacts against the scenario's judge_criteria.
    Only judges text artifacts that are likely meaningful
    (SPEC.md, scorers, prompts).

    ``exclude_paths`` is a set of project-relative paths that the judge
    should skip because they were pre-seeded by the scenario (so they are
    not the agent's work and shouldn't be scored against the agent's task
    criteria). The typical caller is ``score_result``, which passes in
    ``result.seeded_files``.
    """
    results: list[JudgeResult] = []
    exclude_paths = exclude_paths or set()

    # Select artifacts worth judging
    judgeable = {}
    for path, content in artifacts.items():
        if path in exclude_paths:
            continue
        if content.startswith("<binary"):
            continue
        if any(path.endswith(ext) for ext in (".md", ".txt")):
            judgeable[path] = content
        elif path == "SPEC.md":
            judgeable[path] = content

    if not judgeable:
        return results

    for path, content in judgeable.items():
        try:
            response = await client.chat.completions.create(
                model="judge:large",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an evaluator judging whether an AI-generated artifact "
                            "correctly captures the requirements of a given task. "
                            "First write your reasoning (2-3 sentences), then give a score 1-10. "
                            "Output JSON with keys: reasoning, score."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"## Task Description\n\n{scenario.judge_criteria}\n\n"
                            f"## Artifact: {path}\n\n```\n{content[:4000]}\n```\n\n"
                            f"Score this artifact for how well it captures the task requirements."
                        ),
                    },
                ],
                temperature=0.0,
                response_format=JUDGE_SCHEMA,
            )

            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            score, reasoning = _coerce_judge_payload(data)
            results.append(JudgeResult(
                artifact=path,
                score=score,
                reasoning=reasoning,
            ))
        except Exception as e:
            logger.error("Judge failed for %s: %s", path, e)
            results.append(JudgeResult(
                artifact=path,
                score=0,
                reasoning=f"Judge error: {e}",
            ))

    return results
