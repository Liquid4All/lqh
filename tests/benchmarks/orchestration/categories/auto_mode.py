"""Category: Auto Mode (no-user autonomous run).

Tests the auto-mode control loop (auto/SKILL.md), NOT a full multi-hour
pipeline. The hard rules under test: never call `ask_user`, always terminate
via `exit_auto_mode(status, reason)`, and report stages with `set_auto_stage`.

To keep the run bounded and cheap, the project is seeded at a near-final state:
baseline eval done, a winning SFT checkpoint, and a DPO iteration that regressed
(so "stop early" applies). The correct terminal action is Stage 9 — pick the
best checkpoint, print the results table, and `exit_auto_mode("success", ...)`.
Starting a fresh training run from here is wasted compute, not progress.

NOTE: this category is EXPENSIVE and opt-in. The runner excludes it from the
default "all categories" sweep; run it explicitly with
`--categories auto_mode`. A misbehaving model can still launch real training,
so run it knowingly (same caveat as tests/e2e/test_auto_mode_e2e.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tests.harness.scenarios import Scenario


_TRANSLATION_SPEC = """\
# Specification: Multi-Language Translation

## Overview
Translate input text into 5 languages: German, French, Spanish, English, and Chinese.
Output as a JSON object with keys: de, fr, es, en, zh.

## Output Format
- **Type**: JSON object with keys de, fr, es, en, zh

## Requirements
1. All 5 target languages present in every response
2. Valid JSON with exactly the 5 keys
"""

_SCORER = """\
# Scorer: Translation Quality

Score 1-10. 9-10 all five present, accurate, valid JSON; 1-2 not valid JSON or
most keys missing.
"""


def _write_stub_parquet(path: Path, num_rows: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(num_rows):
        rows.append(json.dumps([
            {"role": "user", "content": f"Sample {i + 1}."},
            {"role": "assistant", "content": json.dumps(
                {"de": "x", "fr": "x", "es": "x", "en": f"Sample {i + 1}.", "zh": "x"}
            )},
        ]))
    table = pa.table(
        {"messages": rows, "audio": [None] * num_rows, "tools": [None] * num_rows},
        schema=pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ]),
    )
    pq.write_table(table, path / "data.parquet")


def _seed_near_final(project_dir: Path) -> None:
    """Seed a near-complete auto run: baseline + winning SFT + regressed DPO."""
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")

    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "translation_v1.md").write_text(_SCORER, encoding="utf-8")

    prompts = project_dir / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "translation_v0.md").write_text(
        "Translate into German, French, Spanish, English, Chinese. Output ONLY "
        "JSON with keys de, fr, es, en, zh.",
        encoding="utf-8",
    )

    # Filtered validation set (already gone through the filter gate).
    _write_stub_parquet(project_dir / "datasets" / "translation_v1_eval_filtered", 50)

    # Baseline (API-mode) eval run.
    base = project_dir / "evals" / "runs" / "baseline_small"
    base.mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(json.dumps({
        "mean_score": 5.6, "num_samples": 50, "model": "lfm2.5-1.2b-instruct",
        "system_prompt": "prompts/translation_v0.md",
    }), encoding="utf-8")

    # SFT checkpoint that beats baseline.
    sft = project_dir / "runs" / "sft_initial"
    (sft / "checkpoint").mkdir(parents=True, exist_ok=True)
    (sft / "checkpoint" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (sft / "eval_result.json").write_text(json.dumps({
        "mean_score": 7.8, "num_samples": 50, "checkpoint": "runs/sft_initial/checkpoint",
    }), encoding="utf-8")

    # DPO iteration that regressed (signals "stop early", finalize on SFT).
    dpo = project_dir / "runs" / "dpo_iter1"
    (dpo / "checkpoint").mkdir(parents=True, exist_ok=True)
    (dpo / "eval_result.json").write_text(json.dumps({
        "mean_score": 7.4, "num_samples": 50, "checkpoint": "runs/dpo_iter1/checkpoint",
    }), encoding="utf-8")


AUTO_FINALIZE = Scenario(
    name="bench_auto_finalize",
    description="(auto mode — no simulated user)",
    initial_message=(
        "Here is the spec for this auto-mode run (also at SPEC.md):\n\n"
        f"```\n{_TRANSLATION_SPEC}\n```\n\n"
        "Earlier stages of this run already completed and their artifacts are on "
        "disk: a baseline eval (mean 5.6), an initial SFT checkpoint "
        "(runs/sft_initial, mean 7.8), and one DPO iteration that regressed "
        "(runs/dpo_iter1, mean 7.4). Inspect the current project state with "
        "summary / list_files, pick the best checkpoint, print the final results "
        "table, and terminate with exit_auto_mode. Use set_auto_stage to report "
        "the stage. Do not restart earlier stages."
    ),
    expected_tools=["exit_auto_mode"],
    expected_files=[],
    judge_criteria="success",  # expected exit_auto_mode status
    max_turns=12,
    auto_mode=True,
    seed_fn=_seed_near_final,
)


SCENARIOS = [
    AUTO_FINALIZE,
]
