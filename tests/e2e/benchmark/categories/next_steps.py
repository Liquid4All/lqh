"""Category 5: Next Steps benchmark scenarios.

Tests whether the agent chooses the correct next action given different
project states.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tests.e2e.scenarios import Scenario


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
3. Handle informal text, slang, and idioms gracefully
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
            messages=[{
                "role": "user",
                "content": f"Write a short sentence that a {persona.brief()} would write. Output ONLY the text.",
            }],
        )
        source_text = resp.choices[0].message.content.strip()

        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[
                {"role": "system", "content": "Translate into German, French, Spanish, English, and Chinese. Return ONLY JSON with keys: de, fr, es, en, zh."},
                {"role": "user", "content": source_text},
            ],
            response_format={"type": "json_object"},
        )
        translations = resp.choices[0].message.content.strip()

        return [
            ChatMLMessage("user", source_text),
            ChatMLMessage("assistant", translations),
        ]
'''

_SCORER = """\
# Scorer: Translation Quality

## Task
Score the translation quality (de, fr, es, en, zh as JSON).

## Scoring Scale
- **9-10**: All 5 translations present, accurate, valid JSON
- **7-8**: All present with minor issues
- **5-6**: Valid JSON but some inaccurate
- **3-4**: Missing keys or multiple wrong
- **1-2**: Not valid JSON or mostly missing
"""

_PROMPT_V0 = (
    "Translate the following text into German, French, Spanish, English, "
    "and Chinese. Output ONLY a JSON object with keys: de, fr, es, en, zh."
)

_PROMPT_V1 = (
    "You are an expert multilingual translator. Translate the following text "
    "into German, French, Spanish, English, and Chinese. Preserve the original "
    "tone and formality level. Output ONLY a JSON object with keys: de, fr, es, en, zh. "
    "Ensure all translations are natural and idiomatic."
)


def _write_stub_parquet(path: Path, num_rows: int) -> None:
    """Write a minimal valid parquet file with stub translation data."""
    path.mkdir(parents=True, exist_ok=True)
    messages_list = []
    for i in range(num_rows):
        msgs = json.dumps([
            {"role": "user", "content": f"Sample text number {i + 1}."},
            {"role": "assistant", "content": json.dumps(
                {"de": f"Text {i+1}", "fr": f"Texte {i+1}", "es": f"Texto {i+1}",
                 "en": f"Sample text number {i+1}.", "zh": f"示例文本{i+1}"}
            )},
        ])
        messages_list.append(msgs)

    table = pa.table(
        {"messages": messages_list, "audio": [None] * num_rows, "tools": [None] * num_rows},
        schema=pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ]),
    )
    pq.write_table(table, path / "data.parquet")


def _seed_after_spec(project_dir: Path) -> None:
    """State: only SPEC.md exists."""
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")


def _seed_after_draft(project_dir: Path) -> None:
    """State: SPEC + pipeline + draft dataset (no scorer yet)."""
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "translation_v1.py").write_text(_PIPELINE_CODE, encoding="utf-8")
    _write_stub_parquet(project_dir / "datasets" / "translation_v1_draft", 20)


def _seed_after_eval(project_dir: Path) -> None:
    """State: SPEC + pipeline + eval dataset + scorer."""
    _seed_after_draft(project_dir)
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "translation_v1.md").write_text(_SCORER, encoding="utf-8")
    _write_stub_parquet(project_dir / "datasets" / "translation_v1_eval", 50)


def _seed_after_baseline(project_dir: Path) -> None:
    """State: SPEC + eval + baseline eval run + prompt."""
    _seed_after_eval(project_dir)
    prompts = project_dir / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "translation_v0.md").write_text(_PROMPT_V0, encoding="utf-8")

    # Baseline eval run
    run_dir = project_dir / "evals" / "runs" / "baseline_small"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "mean_score": 5.8, "median_score": 6.0, "num_samples": 50,
        "model": "lfm2.5-1.2b-instruct", "system_prompt": "prompts/translation_v0.md",
    }), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": "datasets/translation_v1_eval",
        "scorer": "evals/scorers/translation_v1.md",
        "model": "lfm2.5-1.2b-instruct",
    }), encoding="utf-8")


def _seed_after_prompt_opt(project_dir: Path) -> None:
    """State: SPEC + optimized prompt + improved eval."""
    _seed_after_baseline(project_dir)
    prompts = project_dir / "prompts"
    (prompts / "translation_v1.md").write_text(_PROMPT_V1, encoding="utf-8")

    # Improved eval run
    run_dir = project_dir / "evals" / "runs" / "prompt_v1_small"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "mean_score": 7.4, "median_score": 8.0, "num_samples": 50,
        "model": "lfm2.5-1.2b-instruct", "system_prompt": "prompts/translation_v1.md",
    }), encoding="utf-8")

    # Training data at scale
    _write_stub_parquet(project_dir / "datasets" / "translation_v1", 200)


_PASSIVE_USER = (
    "You are a passive user who follows the agent's suggestions. "
    "You want to continue making progress on the project.\n\n"
    "Behavior rules:\n"
    "- When the agent suggests a next step, agree and say 'sounds good, go ahead'\n"
    "- When asked to choose, pick whatever the agent recommends\n"
    "- After the agent takes the next action, say 'I'm done for now'\n"
    "- Do NOT suggest specific actions yourself"
)


NEXT_AFTER_SPEC = Scenario(
    name="bench_next_after_spec",
    description=_PASSIVE_USER,
    initial_message="What should we do next with this project?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    judge_criteria="data_generation",  # Expected next step (used by scorer)
    max_turns=20,
    stage_limits={"data_generation": 15},
    seed_fn=_seed_after_spec,
)

NEXT_AFTER_DRAFT = Scenario(
    name="bench_next_after_draft",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    judge_criteria="data_generation",  # Should create scorer + full eval set
    max_turns=20,
    stage_limits={"data_generation": 15},
    seed_fn=_seed_after_draft,
)

NEXT_AFTER_EVAL = Scenario(
    name="bench_next_after_eval",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    judge_criteria="evaluation",  # Should run model eval
    max_turns=20,
    stage_limits={"evaluation": 15},
    seed_fn=_seed_after_eval,
)

NEXT_AFTER_BASELINE = Scenario(
    name="bench_next_after_baseline",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    judge_criteria="prompt_optimization",  # Should optimize prompt
    max_turns=20,
    stage_limits={"prompt_optimization": 15},
    seed_fn=_seed_after_baseline,
)

NEXT_AFTER_PROMPT_OPT = Scenario(
    name="bench_next_after_prompt_opt",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    judge_criteria="train",  # Should suggest training
    max_turns=20,
    stage_limits={"train": 15},
    seed_fn=_seed_after_prompt_opt,
)


SCENARIOS = [
    NEXT_AFTER_SPEC,
    NEXT_AFTER_DRAFT,
    NEXT_AFTER_EVAL,
    NEXT_AFTER_BASELINE,
    NEXT_AFTER_PROMPT_OPT,
]
