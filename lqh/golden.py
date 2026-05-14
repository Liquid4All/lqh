"""Golden trajectory generation and preference pair assembly for DPO.

Runs in the main lqh process (no torch imports).  Called by the watcher
when a DPO iteration's predictions have been scored.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

__all__ = ["generate_golden"]


async def generate_golden(
    predictions_path: Path,
    scores_path: Path,
    dataset_path: str,
    config: dict[str, Any],
    client: AsyncOpenAI,
    output_dir: Path,
    *,
    chosen_scores: list[float | None] | None = None,
) -> None:
    """Generate golden trajectories and assemble preference pairs.

    Two selection modes:

    **Gap-quantile** (preferred, when ``chosen_scores`` is provided):
    For each sample, compute ``gap = chosen_score - rejected_score``.
    Sort pairs by gap descending. Take top ``selection.top_quantile``
    (default 0.2 = top 20% by gap), then drop pairs with
    ``gap < selection.min_gap`` (default 0.5). If fewer than
    ``selection.min_pairs_per_iter`` pairs remain (default 50), emit
    an empty preferences.parquet so the subprocess detects "no
    informative pairs this iter" and continues. This is the
    informativeness-driven selector — the chosen is *verified* better
    than rejected by the same judge, not just assumed-good.

    **Threshold** (legacy, when ``chosen_scores`` is None): keep all
    pairs where ``rejected_score < rejection_threshold`` (default 6.0).
    No verification of chosen, no per-pair gap filter.

    Always writes ``preference_stats.json`` to ``output_dir`` with the
    funnel counts and gap percentiles for diagnostics.

    Parameters
    ----------
    predictions_path : Path
        The subprocess-generated predictions parquet.
    scores_path : Path
        The scored results parquet (from ``run_scoring``).
    dataset_path : str
        Path to the original training dataset (for ``golden_source="dataset"``).
    config : dict
        The run config with ``golden_source``, ``golden_model``,
        ``rejection_threshold``, ``selection``, etc.
    client : AsyncOpenAI
        API client for generating golden responses.
    output_dir : Path
        Where to write ``golden.parquet``, ``preferences.parquet``,
        ``preference_stats.json``.
    chosen_scores : list[float | None] | None
        Per-sample-index judge score for the *chosen* responses (the
        original training-set assistant turns). Same length as the
        training set; index ``i`` is the chosen score at training
        index ``i``. When provided, enables gap-quantile selection.
    """
    selection_cfg = config.get("selection", {})
    top_quantile = float(selection_cfg.get("top_quantile", 0.2))
    min_gap = float(selection_cfg.get("min_gap", 0.5))
    min_pairs_per_iter = int(selection_cfg.get("min_pairs_per_iter", 50))
    rejection_threshold = config.get("rejection_threshold", 6.0)
    golden_source = config.get("golden_source", "dataset")
    golden_model = config.get("golden_model", "large")

    # Load predictions
    pred_table = pq.read_table(str(predictions_path))
    pred_messages = [
        json.loads(pred_table.column("messages")[i].as_py())
        for i in range(len(pred_table))
    ]

    # Load scores
    score_table = pq.read_table(str(scores_path))
    scores = [
        score_table.column("score")[i].as_py()
        for i in range(len(score_table))
    ]

    # ---- Selection ----
    # Output diagnostic stats so we can see exactly how the funnel
    # narrowed at every iter.
    stats: dict[str, Any] = {
        "rejected_scored": sum(1 for s in scores if s is not None),
        "total_predictions": len(scores),
        "selector": "gap_quantile" if chosen_scores is not None else "threshold",
    }

    if chosen_scores is not None:
        # Gap-quantile selector — floor FIRST, then take top quantile of
        # qualifying. Rationale: a pair where chosen-rejected gap is
        # below judge-noise (~0.5) is not preference signal, it's noise;
        # we never want to train on it. Quantile is the *refiner* that
        # picks the strongest signal among qualifying pairs when there
        # are many. When qualifying pairs are scarce (weak-signal
        # dataset), we keep what we have rather than artificially
        # narrowing further.
        pairs: list[tuple[int, float]] = []  # (idx, gap)
        for i, rejected_score in enumerate(scores):
            if rejected_score is None:
                continue
            if i >= len(chosen_scores):
                continue
            cs = chosen_scores[i]
            if cs is None:
                continue
            gap = cs - rejected_score
            pairs.append((i, gap))

        stats["pairs_with_both_scored"] = len(pairs)
        # Distribution over ALL pairs (informative for diagnosing whether
        # the chosen pool actually beats the rejected pool).
        if pairs:
            gaps_all = sorted([g for _, g in pairs])
            n = len(gaps_all)
            stats["gap_p10"] = gaps_all[max(0, n // 10 - 1)]
            stats["gap_p50"] = gaps_all[n // 2]
            stats["gap_p90"] = gaps_all[min(n - 1, (n * 9) // 10)]
            stats["gap_max"] = gaps_all[-1]
            stats["gap_min"] = gaps_all[0]

        # Step 1: hard floor on gap. Anything below min_gap is judge
        # noise, drop it.
        qualifying = [(i, g) for (i, g) in pairs if g >= min_gap]
        stats["min_gap"] = min_gap
        stats["pairs_after_min_gap"] = len(qualifying)

        # Distribution over QUALIFYING pairs (the actual training pool
        # before quantile narrowing — the most useful diagnostic).
        if qualifying:
            gaps_q = sorted([g for _, g in qualifying])
            n = len(gaps_q)
            stats["qualifying_gap_p10"] = gaps_q[max(0, n // 10 - 1)]
            stats["qualifying_gap_p50"] = gaps_q[n // 2]
            stats["qualifying_gap_p90"] = gaps_q[min(n - 1, (n * 9) // 10)]

        # Step 2: among qualifying, take the top by gap. Target size is
        # max(min_pairs_per_iter, top_quantile * len(qualifying)) capped
        # by len(qualifying). This means:
        #   - lots of qualifying → take top X% (focus on strongest signal)
        #   - few qualifying     → take all (use what we have, don't narrow further)
        # The min_pairs_per_iter floor in the *target* (not the result)
        # ensures we don't artificially pick a tiny set when more are
        # available.
        qualifying.sort(key=lambda x: x[1], reverse=True)
        target = max(
            min_pairs_per_iter,
            int(round(len(qualifying) * top_quantile)),
        )
        target = min(target, len(qualifying))
        kept_pairs = qualifying[:target]
        stats["top_quantile"] = top_quantile
        stats["pairs_after_quantile"] = len(kept_pairs)

        # Final min-pair check: if even the qualifying pool is too small,
        # skip the iter rather than overfit on outliers.
        stats["min_pairs_per_iter"] = min_pairs_per_iter
        if len(kept_pairs) < min_pairs_per_iter:
            logger.warning(
                "Only %d preference pairs above min_gap %.2f (need %d). "
                "Skipping this DPO iter.",
                len(kept_pairs), min_gap, min_pairs_per_iter,
            )
            stats["kept"] = 0
            stats["skipped_reason"] = "below_min_pairs_per_iter"
            (output_dir / "preference_stats.json").write_text(
                json.dumps(stats, indent=2) + "\n"
            )
            _write_empty_preferences(output_dir)
            return

        low_indices = [i for i, _ in kept_pairs]
        stats["kept"] = len(low_indices)
        logger.info(
            "Gap-quantile selection: %d pairs scored, %d above min_gap %.2f, "
            "%d kept after top %.0f%% quantile",
            stats["pairs_with_both_scored"], len(qualifying),
            min_gap, len(low_indices), top_quantile * 100,
        )
    else:
        # Legacy threshold selector.
        low_indices = [
            i for i, s in enumerate(scores)
            if s is not None and s < rejection_threshold
        ]
        stats["rejection_threshold"] = rejection_threshold
        stats["kept"] = len(low_indices)
        logger.info(
            "%d of %d samples below threshold %.1f",
            len(low_indices), len(scores), rejection_threshold,
        )

        if not low_indices:
            (output_dir / "preference_stats.json").write_text(
                json.dumps(stats, indent=2) + "\n"
            )
            _write_empty_preferences(output_dir)
            return

    # Generate golden (chosen) responses
    golden_responses: dict[int, str] = {}

    if golden_source == "dataset":
        golden_responses = _golden_from_dataset(
            low_indices, pred_messages, dataset_path
        )
    elif golden_source == "api":
        golden_responses = await _golden_from_api(
            low_indices, pred_messages, client, golden_model
        )
    else:
        logger.warning("Unknown golden_source: %s, falling back to dataset", golden_source)
        golden_responses = _golden_from_dataset(
            low_indices, pred_messages, dataset_path
        )

    # Write golden.parquet
    golden_entries = []
    for idx, response in golden_responses.items():
        golden_entries.append(
            {
                "sample_index": idx,
                "messages": json.dumps(
                    _get_prompt(pred_messages[idx])
                    + [{"role": "assistant", "content": response}]
                ),
            }
        )

    if golden_entries:
        golden_table = pa.table(
            {
                "sample_index": [e["sample_index"] for e in golden_entries],
                "messages": [e["messages"] for e in golden_entries],
            }
        )
        pq.write_table(golden_table, output_dir / "golden.parquet")

    # Assemble preference pairs
    pref_entries = []
    for idx in low_indices:
        if idx not in golden_responses:
            continue

        prompt = _get_prompt(pred_messages[idx])
        rejected = _get_last_assistant(pred_messages[idx])
        chosen = golden_responses[idx]

        if rejected and chosen:
            pref_entries.append(
                {
                    "prompt": json.dumps(prompt),
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    stats["pairs_written"] = len(pref_entries)
    (output_dir / "preference_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )

    if pref_entries:
        pref_table = pa.table(
            {
                "prompt": [e["prompt"] for e in pref_entries],
                "chosen": [e["chosen"] for e in pref_entries],
                "rejected": [e["rejected"] for e in pref_entries],
            }
        )
        pq.write_table(pref_table, output_dir / "preferences.parquet")
        logger.info("Assembled %d preference pairs", len(pref_entries))
    else:
        _write_empty_preferences(output_dir)
        logger.warning("No valid preference pairs assembled")


# ---------------------------------------------------------------------------
# Golden sources
# ---------------------------------------------------------------------------


def _golden_from_dataset(
    low_indices: list[int],
    pred_messages: list[list[dict[str, str]]],
    dataset_path: str,
) -> dict[int, str]:
    """Pull golden responses from the original training dataset."""
    try:
        table = pq.read_table(dataset_path)
    except Exception:
        logger.warning("Could not load dataset at %s for golden responses", dataset_path)
        return {}

    dataset_messages = [
        json.loads(table.column("messages")[i].as_py())
        for i in range(len(table))
    ]

    result: dict[int, str] = {}
    for idx in low_indices:
        if idx < len(dataset_messages):
            assistant_content = _get_last_assistant(dataset_messages[idx])
            if assistant_content:
                result[idx] = assistant_content

    return result


async def _golden_from_api(
    low_indices: list[int],
    pred_messages: list[list[dict[str, str]]],
    client: AsyncOpenAI,
    model: str,
) -> dict[int, str]:
    """Generate golden responses by calling the API with a strong model."""
    import asyncio

    semaphore = asyncio.Semaphore(5)
    result: dict[int, str] = {}

    async def _generate_one(idx: int) -> None:
        prompt = _get_prompt(pred_messages[idx])
        if not prompt:
            return

        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if content:
                    result[idx] = content
            except Exception:
                logger.warning("Failed to generate golden for sample %d", idx)

    await asyncio.gather(*[_generate_one(idx) for idx in low_indices])
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Get all messages except the last assistant turn."""
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def _get_last_assistant(messages: list[dict[str, str]]) -> str | None:
    """Get the content of the last assistant message."""
    if messages and messages[-1].get("role") == "assistant":
        return messages[-1].get("content")
    return None


def _write_empty_preferences(output_dir: Path) -> None:
    """Write an empty preferences.parquet so the subprocess can detect completion."""
    table = pa.table(
        {
            "prompt": pa.array([], type=pa.string()),
            "chosen": pa.array([], type=pa.string()),
            "rejected": pa.array([], type=pa.string()),
        }
    )
    pq.write_table(table, output_dir / "preferences.parquet")
