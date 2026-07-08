"""Category: Scorer Validation loop (v0.3.1).

Tests Phase 2.4/2.5 of the data_generation skill: after a draft dataset is
approved, the agent must author a scorer from the spec AND *validate it on the
draft samples* via `run_scoring(mode="data_quality")`, then inspect
scores.parquet — confirming the scorer ranks good samples high and weak ones
low before it is ever used to filter. "Creating a scorer is not the same as
having a working scorer."

The draft is seeded with a deliberate mix of strong and broken samples so a
working scorer must discriminate between them.
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

## Input Format
- **Type**: Plain text, 1-5 sentences
- **Language**: Any language (auto-detected)

## Output Format
- **Type**: JSON object
- **Keys**: de, fr, es, en, zh

## Requirements
1. All 5 target languages must be present in every response
2. Translations must be accurate and natural
3. Output must be valid JSON with exactly the 5 keys
"""

_PIPELINE_CODE = '''\
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step
import json
import liquidrandom

class TranslationPipeline(Pipeline):
    """Generate translation training samples."""

    async def generate(self, client, input=None) -> Conversation:
        persona = liquidrandom.persona()
        resp = await client.chat.completions.create(
            model="random:small",
            messages=[{"role": "user", "content": f"Write a short sentence a {persona.brief()} would write. Output ONLY the text."}],
        )
        source = resp.choices[0].message.content.strip()
        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[
                {"role": "system", "content": "Translate into German, French, Spanish, English, Chinese. Return ONLY JSON with keys de, fr, es, en, zh."},
                {"role": "user", "content": source},
            ],
            response_format={"type": "json_object"},
        )
        return [ChatMLMessage("user", source), ChatMLMessage("assistant", resp.choices[0].message.content.strip())]
'''


def _write_draft_with_mixed_quality(path: Path) -> None:
    """Approved draft dataset: mostly good translations, a few broken ones."""
    path.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []

    good_inputs = [
        "The weather is nice today.",
        "Please send me the report by Friday.",
        "I really enjoyed the concert last night.",
        "Can you help me find the train station?",
        "Our meeting is scheduled for next Tuesday.",
    ]
    for src in good_inputs:
        rows.append(json.dumps([
            {"role": "user", "content": src},
            {"role": "assistant", "content": json.dumps({
                "de": "...", "fr": "...", "es": "...", "en": src, "zh": "……",
            }, ensure_ascii=False)},
        ]))

    # Broken samples a working scorer must catch: missing keys, not-JSON.
    rows.append(json.dumps([
        {"role": "user", "content": "Good morning."},
        {"role": "assistant", "content": json.dumps({"de": "Guten Morgen", "en": "Good morning"})},  # missing fr/es/zh
    ]))
    rows.append(json.dumps([
        {"role": "user", "content": "Thank you very much."},
        {"role": "assistant", "content": "Danke schön / Merci beaucoup"},  # not JSON
    ]))

    n = len(rows)
    table = pa.table(
        {"messages": rows, "audio": [None] * n, "tools": [None] * n},
        schema=pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ]),
    )
    pq.write_table(table, path / "data.parquet")


def _seed_approved_draft(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "translation_v1.py").write_text(_PIPELINE_CODE, encoding="utf-8")
    _write_draft_with_mixed_quality(project_dir / "datasets" / "translation_v1_draft")


_SCORER_VALIDATION_USER = (
    "You are a user with a translation project. The spec is at SPEC.md, the "
    "pipeline is at data_gen/translation_v1.py, and an approved ~20-sample draft "
    "dataset is at datasets/translation_v1_draft. You want a scorer created and, "
    "importantly, VERIFIED to actually work on the draft before you trust it for "
    "filtering and evaluation.\n\n"
    "Behavior rules:\n"
    "- When the agent proposes creating a scorer, say 'go ahead'\n"
    "- When the agent proposes testing the scorer on the draft samples, say "
    "'yes, run it on the drafts and show me the scores'\n"
    "- When shown the per-sample scores, confirm they look right (the broken "
    "samples should score low) and say 'looks good'\n"
    "- Do NOT suggest specific tools or modes yourself\n"
    "- After the scorer has been created AND validated on the draft, say "
    "'I'm done for now'"
)


SCORER_VALIDATION_TRANSLATION = Scenario(
    name="bench_scorer_validation_translation",
    description=_SCORER_VALIDATION_USER,
    initial_message=(
        "I have an approved draft dataset at datasets/translation_v1_draft. "
        "Create a scorer for this task and verify it actually works on the draft "
        "before we use it."
    ),
    expected_tools=["create_file", "run_scoring"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the scorer file (evals/scorers/*.md) the agent created for the "
        "translation task.\n"
        "Check for:\n"
        "- A clear 1-10 scoring scale\n"
        "- Format-compliance check (valid JSON with exactly de/fr/es/en/zh)\n"
        "- A critical-failure condition for missing keys or invalid JSON\n"
        "- Accuracy / completeness dimensions\n\n"
        "10 = thorough, discriminating rubric, 5 = basic, 1 = unusable"
    ),
    max_turns=25,
    stage_limits={"data_generation": 20},
    seed_fn=_seed_approved_draft,
)


SCENARIOS = [
    SCORER_VALIDATION_TRANSLATION,
]
