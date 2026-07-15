"""Deterministic diagnostics for the voice-satisfaction task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_EXPECTED_KEYS = [
    "reasoning",
    "score",
    "failure_tags",
    "success_tags",
    "failed_turns",
    "successful_turns",
]


def _assistant_payload(messages_value: Any) -> dict[str, Any] | None:
    try:
        messages = (
            json.loads(messages_value)
            if isinstance(messages_value, str)
            else messages_value
        )
        content = messages[-1]["content"]
        payload = json.loads(content)
        return payload if isinstance(payload, dict) else None
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return None


def voice_metrics(predictions_path: Path, reference_path: Path) -> dict[str, float | int]:
    """Compare structured model outputs with generated reference assessments."""
    predictions = pq.read_table(predictions_path)
    references = pq.read_table(reference_path)
    gold = {
        index: _assistant_payload(value)
        for index, value in enumerate(references["messages"].to_pylist())
    }
    counters = {
        "valid_json": 0,
        "valid_schema": 0,
        "score_direction_correct": 0,
        "failure_tags_exact": 0,
        "failed_turns_exact": 0,
        "frustration_cases": 0,
        "frustration_misses": 0,
    }
    score_abs_error = 0.0
    score_count = 0
    total = 0
    for index, value in zip(
        predictions["sample_index"].to_pylist(),
        predictions["messages"].to_pylist(),
        strict=True,
    ):
        reference = gold.get(int(index))
        if reference is None:
            continue
        total += 1
        predicted = _assistant_payload(value)
        if predicted is None:
            if reference.get("failure_tags"):
                counters["frustration_cases"] += 1
                counters["frustration_misses"] += 1
            continue
        counters["valid_json"] += 1
        if list(predicted) == _EXPECTED_KEYS:
            counters["valid_schema"] += 1

        reference_score = reference.get("score")
        predicted_score = predicted.get("score")
        if isinstance(reference_score, int) and isinstance(predicted_score, int):
            score_count += 1
            score_abs_error += abs(predicted_score - reference_score)
            if (reference_score <= 3) == (predicted_score <= 3):
                counters["score_direction_correct"] += 1

        reference_failures = reference.get("failure_tags") or []
        predicted_failures = predicted.get("failure_tags") or []
        if set(predicted_failures) == set(reference_failures):
            counters["failure_tags_exact"] += 1
        if (predicted.get("failed_turns") or []) == (reference.get("failed_turns") or []):
            counters["failed_turns_exact"] += 1
        if reference_failures:
            counters["frustration_cases"] += 1
            if not predicted_failures or not isinstance(predicted_score, int) or predicted_score >= 4:
                counters["frustration_misses"] += 1

    def rate(value: int, denominator: int = total) -> float:
        return value / denominator if denominator else 0.0

    return {
        "n": total,
        "json_valid_rate": rate(counters["valid_json"]),
        "schema_valid_rate": rate(counters["valid_schema"]),
        "score_mae": score_abs_error / score_count if score_count else 0.0,
        "score_direction_accuracy": rate(counters["score_direction_correct"]),
        "failure_tags_exact_rate": rate(counters["failure_tags_exact"]),
        "failed_turns_exact_rate": rate(counters["failed_turns_exact"]),
        "frustration_miss_rate": rate(
            counters["frustration_misses"], counters["frustration_cases"],
        ),
        "frustration_cases": counters["frustration_cases"],
    }
