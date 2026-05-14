"""Category 7: Context Management benchmark scenarios.

Tests whether the agent stays coherent over long conversations without
blowing up the context window.
"""

from __future__ import annotations

from pathlib import Path

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
1. All 5 target languages must be present
2. Translations must be accurate and natural
3. Handle informal text, slang, and idioms gracefully
"""


CONTEXT_LONG_SPEC_CAPTURE = Scenario(
    name="bench_context_long_spec_capture",
    description=(
        "You are a user with a very complex task that requires extensive discussion. "
        "You are building a multi-modal content moderation system that handles text, "
        "images (via description), and video transcripts. There are many requirements.\n\n"
        "Behavior rules:\n"
        "- Give detailed answers to every question (3-5 sentences each)\n"
        "- When asked about content types, explain: text posts, image captions, "
        "video transcripts, comment threads, and user bios\n"
        "- When asked about categories, list: hate speech, violence, sexual content, "
        "spam, harassment, self-harm, misinformation, and 'safe'\n"
        "- When asked about severity, explain 3 levels: low (flag for review), "
        "medium (auto-hide, notify moderator), high (auto-remove, ban user)\n"
        "- When asked about edge cases, describe: sarcasm, cultural context, "
        "news reporting vs promotion, artistic expression, educational content\n"
        "- When asked about languages, say English, Spanish, Portuguese, Hindi, Arabic\n"
        "- When asked about output format, say JSON with: category, severity, "
        "confidence, explanation, recommended_action\n"
        "- When asked about false positives, say they should be minimized especially "
        "for news and educational content\n"
        "- When asked about regulatory requirements, mention GDPR, DSA, and "
        "US section 230 implications\n"
        "- When asked about anything else, give a detailed thoughtful answer\n"
        "- Do NOT say 'I'm done' until the agent creates SPEC.md\n"
        "- After SPEC.md is created and shown, say 'looks good, I'm done for now'"
    ),
    initial_message=(
        "I need to build a content moderation system. It needs to handle multiple "
        "content types and multiple languages with different severity levels."
    ),
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for a content moderation system.\n"
        "Check completeness: content types, moderation categories, severity levels, "
        "output format, languages, edge cases.\n"
        "10 = comprehensive, 5 = missing major sections, 1 = incomplete"
    ),
    max_turns=35,
    stage_limits={"spec_capture": 30},
)

CONTEXT_MULTISTAGE = Scenario(
    name="bench_context_multistage",
    description=(
        "You are a user with an existing translation spec who wants to go through "
        "data generation AND evaluation in a single session.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'let's generate training data'\n"
        "- When asked about data generation, say 'go ahead with your approach'\n"
        "- When shown draft samples, say 'looks good'\n"
        "- When asked about sample count, say '10 samples for the eval set'\n"
        "- After data is generated, say 'now let's create a scorer and evaluate "
        "the lfm2.5-1.2b-instruct model'\n"
        "- When asked about scoring criteria, say 'go ahead with your approach'\n"
        "- After eval results, say 'I'm done for now'\n"
        "- Stay engaged throughout - do not disengage early"
    ),
    initial_message=(
        "Let's generate some training data and then evaluate a baseline model, "
        "all in this session."
    ),
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score whether the agent maintained coherence across both stages.\n"
        "Check: data gen worked, eval was attempted, agent stayed on track.\n"
        "10 = both stages completed, 5 = only one stage, 1 = confused/stuck"
    ),
    max_turns=45,
    stage_limits={"data_generation": 20, "evaluation": 15},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _TRANSLATION_SPEC, encoding="utf-8"
    ),
)


SCENARIOS = [
    CONTEXT_LONG_SPEC_CAPTURE,
    CONTEXT_MULTISTAGE,
]
