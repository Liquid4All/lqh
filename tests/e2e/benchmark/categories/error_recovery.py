"""Category 4: Error Recovery benchmark scenarios.

Tests whether the LLM can diagnose and fix a broken data generation pipeline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tests.e2e.scenarios import Scenario

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

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
- **Example**: {"de": "...", "fr": "...", "es": "...", "en": "...", "zh": "..."}

## Requirements
1. All 5 target languages must be present in every response
2. Translations must be accurate and natural
3. Handle informal text, slang, and idioms gracefully
"""


def _make_seed_fn(fixture_name: str):
    """Create a seed function that places SPEC.md + broken pipeline."""
    def seed(project_dir: Path) -> None:
        # Write SPEC.md
        (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")

        # Copy broken pipeline fixture
        dg = project_dir / "data_gen"
        dg.mkdir(parents=True, exist_ok=True)
        src = FIXTURES_DIR / fixture_name
        shutil.copy(src, dg / "translation_v1.py")

    return seed


_RECOVERY_DESCRIPTION = (
    "You are a user whose data generation pipeline is broken. You don't know "
    "what's wrong and need the agent to diagnose and fix it.\n\n"
    "Behavior rules:\n"
    "- When the agent explains the error, say 'I see, please fix it'\n"
    "- When the agent asks for permission to edit, say 'go ahead'\n"
    "- Do NOT give any hints about what the bug is\n"
    "- When the pipeline runs successfully, say 'looks good, I'm done for now'\n"
    "- If the agent asks about anything else, say 'just fix the pipeline'"
)

_INITIAL_MESSAGE = (
    "The data generation pipeline at data_gen/translation_v1.py is failing. "
    "Please diagnose the issue and fix it so it runs successfully."
)


RECOVERY_IMPORT = Scenario(
    name="bench_recovery_import",
    description=_RECOVERY_DESCRIPTION,
    initial_message=_INITIAL_MESSAGE,
    expected_tools=["read_file", "edit_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria="",  # Not used; error recovery uses transcript scoring
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_make_seed_fn("broken_pipeline_import.py"),
)

RECOVERY_RETURN_TYPE = Scenario(
    name="bench_recovery_return_type",
    description=_RECOVERY_DESCRIPTION,
    initial_message=_INITIAL_MESSAGE,
    expected_tools=["read_file", "edit_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria="",
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_make_seed_fn("broken_pipeline_return_type.py"),
)

RECOVERY_JSON_KEYS = Scenario(
    name="bench_recovery_json_keys",
    description=_RECOVERY_DESCRIPTION,
    initial_message=_INITIAL_MESSAGE,
    expected_tools=["read_file", "edit_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria="",
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_make_seed_fn("broken_pipeline_json_keys.py"),
)

RECOVERY_API_HALLUC = Scenario(
    name="bench_recovery_api_halluc",
    description=_RECOVERY_DESCRIPTION,
    initial_message=_INITIAL_MESSAGE,
    expected_tools=["read_file", "edit_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria="",
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_make_seed_fn("broken_pipeline_api_halluc.py"),
)

RECOVERY_SYSTEM_MSG = Scenario(
    name="bench_recovery_system_msg",
    description=_RECOVERY_DESCRIPTION,
    initial_message=_INITIAL_MESSAGE,
    expected_tools=["read_file", "edit_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria="",
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_make_seed_fn("broken_pipeline_system_msg.py"),
)


SCENARIOS = [
    RECOVERY_IMPORT,
    RECOVERY_RETURN_TYPE,
    RECOVERY_JSON_KEYS,
    RECOVERY_API_HALLUC,
    RECOVERY_SYSTEM_MSG,
]
