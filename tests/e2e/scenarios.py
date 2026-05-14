"""E2E test scenario definitions."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """A reusable E2E test scenario."""

    name: str
    description: str  # Full description for the simulated human
    initial_message: str  # First user message to kick off the conversation
    expected_tools: list[str] = field(default_factory=list)  # Tools that must be called
    expected_files: list[str] = field(default_factory=list)  # Files that must exist
    judge_criteria: str = ""  # Criteria for LLM judge on artifacts
    max_turns: int = 20  # Safety limit
    stage_limits: dict[str, int] = field(default_factory=dict)  # per-skill turn limits
    seed_fn: Callable | None = None  # async or sync fn(project_dir) to pre-populate before agent starts


TRANSLATION_5LANG = Scenario(
    name="translation_5lang",
    description=(
        "You are a user who wants to customize the LFM2.5-1.2B-Instruct model "
        "to translate input text into 5 languages: German, French, Spanish, "
        "English, and Chinese. The output should be a JSON object with keys: "
        "de, fr, es, en, zh. Typical inputs are 1-5 sentences in any language. "
        "You want the model to handle informal text, slang, and short phrases "
        "gracefully. You care about accuracy over style. You prefer the spec "
        "to be clear and concise.\n\n"
        "Behavior rules:\n"
        "- When the agent asks for examples, do NOT say you'll provide one. "
        "Instead, let the agent create examples and give feedback like 'looks good' "
        "or 'the Chinese translation should be simplified Chinese not traditional'.\n"
        "- When offered next steps after spec creation, choose 'Start generating data'.\n"
        "- When shown draft data samples, review them and say they look good "
        "or suggest small fixes (e.g., 'the JSON keys look correct').\n"
        "- When asked about judge/eval criteria, confirm the proposed dimensions.\n"
        "- When asked about validation set size, accept the agent's suggestion.\n"
        "- After the validation set is generated and scored, say you are done for now."
    ),
    initial_message=(
        "I want to build a translation model. It should take any text and "
        "translate it into German, French, Spanish, English, and Chinese, "
        "returning the results as JSON."
    ),
    expected_tools=[
        "summary",
        "ask_user",
        "create_file",
        "read_file",
        "run_data_gen_pipeline",
    ],
    expected_files=[
        "SPEC.md",
    ],
    judge_criteria=(
        "Score the artifact for how well it captures a multi-language translation task.\n"
        "The task is: translate input text into 5 languages (de, fr, es, en, zh) as JSON.\n"
        "Inputs are 1-5 sentences in any language.\n\n"
        "Check for:\n"
        "- All 5 target languages are mentioned (German/de, French/fr, Spanish/es, English/en, Chinese/zh)\n"
        "- JSON output format is specified with the correct keys\n"
        "- Input format is described (any language, 1-5 sentences)\n"
        "- The artifact is clear and actionable\n\n"
        "10 = all requirements captured precisely\n"
        "5 = some requirements missing or vague\n"
        "1 = does not describe the translation task"
    ),
    max_turns=20,
    stage_limits={"spec_capture": 10, "data_generation": 12},
)


# ---------------------------------------------------------------------------
# Pre-seeded scenario: Model Eval + Prompt Optimization
# ---------------------------------------------------------------------------

_TRANSLATION_SPEC = """\
# Specification: Multi-Language Translation

## Overview
Translate input text into 5 languages: German, French, Spanish, English, and Chinese.
Output as a JSON object with keys: de, fr, es, en, zh.

## Input Format
- **Type**: Plain text, 1-5 sentences
- **Language**: Any language (auto-detected)
- **Typical length**: 10-100 words

## Output Format
- **Type**: JSON object
- **Keys**: de, fr, es, en, zh
- **Example**: {"de": "...", "fr": "...", "es": "...", "en": "...", "zh": "..."}

## Requirements
1. All 5 target languages must be present in every response
2. Translations must be accurate and natural
3. Preserve proper nouns, brand names, numbers
4. Match the formality level of the source text
5. Handle informal text, slang, and idioms gracefully

## Examples

### Example 1
**Input:** The weather is nice today.
**Output:** {"de": "Das Wetter ist heute schön.", "fr": "Le temps est beau aujourd'hui.", "es": "El tiempo está bonito hoy.", "en": "The weather is nice today.", "zh": "今天天气很好。"}

## Edge Cases
| Scenario | Expected Behavior |
|----------|-------------------|
| Input already in target language | Include as-is for that language |
| Mixed language input | Translate each part appropriately |
| Very short input (1-2 words) | Still provide all 5 translations |
"""

_TRANSLATION_PIPELINE = '''\
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step
import json
import random
import liquidrandom

class TranslationEvalPipeline(Pipeline):
    """Generate diverse translation eval samples."""

    SAMPLE_TYPES = [
        "casual message", "formal email", "technical sentence",
        "idiomatic expression", "short phrase", "multi-sentence paragraph",
    ]

    async def generate(self, client, input=None) -> Conversation:
        self.persona = liquidrandom.persona()
        self.sample_type = random.choice(self.SAMPLE_TYPES)
        self.seed = f"{self.persona.name}-{self.sample_type}"

        await self._generate_source(client)
        await self._generate_translations(client)

        # No system message — system prompts are managed separately via prompts/
        return [
            ChatMLMessage("user", self.source_text),
            ChatMLMessage("assistant", self.translations_json),
        ]

    @step(retries=3)
    async def _generate_source(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Write a short {self.sample_type} (1-3 sentences) that "
                    f"a {self.persona.brief()} would write. "
                    f"Output ONLY the text, nothing else."
                ),
            }],
        )
        self.source_text = resp.choices[0].message.content.strip()
        if len(self.source_text) < 5:
            raise GenerationError("Source text too short")

    @step(retries=3)
    async def _generate_translations(self, client):
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[
                {"role": "system", "content": "Translate the text into all 5 languages. Return ONLY a JSON object with keys: de, fr, es, en, zh."},
                {"role": "user", "content": self.source_text},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise GenerationError(f"Expected JSON object, got {type(data).__name__}")
        required = {"de", "fr", "es", "en", "zh"}
        if not required.issubset(data.keys()):
            raise GenerationError(f"Missing keys: {required - set(data.keys())}")
        self.translations_json = json.dumps(data, ensure_ascii=False)
'''

_TRANSLATION_SCORER = """\
# Scorer: Multi-Language Translation Quality

Based on: SPEC.md

## Task Description
The model translates input text into 5 languages (de, fr, es, en, zh) as JSON.

## Conversation Format
You will receive the conversation in this format:

```
[User]
<the input text to translate>

[Assistant]
<the model's translation response — this is what you score>
```

Focus your evaluation on the text after `[Assistant]`.

## Scoring Scale
- **9-10**: All 5 translations present, accurate, natural, valid JSON
- **7-8**: All translations present with minor quality issues (small mistranslations, slightly unnatural phrasing)
- **5-6**: Valid JSON but some translations significantly inaccurate or unnatural
- **3-4**: Missing language keys, or multiple translations are wrong
- **1-2**: Output is not valid JSON, or most translations are missing/wrong

## Scoring Dimensions

### 1. Format Compliance
- The assistant's response must be a valid JSON object (not wrapped in markdown code blocks)
- Must have exactly 5 keys: "de", "fr", "es", "en", "zh"
- No extra keys, no duplicate keys

### 2. Translation Accuracy
- Meaning preserved across all languages
- No hallucinated content added
- No significant parts of the input omitted

### 3. Natural Language Quality
- Translations read naturally in each target language
- Appropriate formality level matching the input

### 4. Completeness
- All 5 languages present in the JSON
- Full input text translated (not truncated mid-sentence)

## Critical Failures (automatic score <= 3)
- Output is not valid JSON
- More than 1 language key missing
- Translation is completely wrong (different meaning)
"""


async def seed_translation_eval_async(project_dir: Path) -> None:
    """Pre-populate a project with spec, pipeline, 50 eval samples, and scorer."""

    # Write SPEC.md
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")

    # Write pipeline
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "translation_v1.py").write_text(_TRANSLATION_PIPELINE, encoding="utf-8")

    # Write scorer
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "translation_v1.md").write_text(_TRANSLATION_SCORER, encoding="utf-8")

    # Write v0 baseline prompt (derived from spec, used for zero-shot eval)
    prompts = project_dir / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "translation_v0.md").write_text(
        "Translate the following text into German, French, Spanish, English, "
        "and Chinese. Format your response as a JSON object with the keys "
        '"de", "fr", "es", "en", and "zh". Output ONLY the JSON object, '
        "nothing else.",
        encoding="utf-8",
    )

    # Write response format schema (auto-discovered by eval runner)
    (prompts / "translation.schema.json").write_text(
        json.dumps({
            "type": "json_schema",
            "json_schema": {
                "name": "translation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "de": {"type": "string"},
                        "fr": {"type": "string"},
                        "es": {"type": "string"},
                        "en": {"type": "string"},
                        "zh": {"type": "string"},
                    },
                    "required": ["de", "fr", "es", "en", "zh"],
                    "additionalProperties": False,
                },
            },
        }, indent=2),
        encoding="utf-8",
    )

    # Generate 50 eval samples by running the pipeline
    logger.info("Seeding: generating 50 eval samples via pipeline...")
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    output_dir = project_dir / "datasets" / "translation_v1_eval"
    result = await run_pipeline(
        script_path=dg / "translation_v1.py",
        num_samples=50,
        output_dir=output_dir,
        client=client,
        concurrency=5,
    )
    logger.info("Seeding: %d/%d samples generated", result.succeeded, result.total)


seed_translation_eval = seed_translation_eval_async


TRANSLATION_EVAL_PROMPTOPT = Scenario(
    name="translation_eval_promptopt",
    description=(
        "You are a user with an existing translation project. The spec, a 50-sample "
        "eval dataset, a scorer, and a baseline system prompt (prompts/translation_v0.md) "
        "are already set up. You want to:\n"
        "1. Evaluate the lfm2.5-1.2b-instruct model using the baseline prompt\n"
        "2. Then optimize the system prompt for that model\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, ask to run model evaluation "
        "using the baseline prompt prompts/translation_v0.md\n"
        "- When asked which model, say 'lfm2.5-1.2b-instruct'\n"
        "- When shown eval results, say 'let's optimize the prompt'\n"
        "- When asked about prompt optimization preferences, say 'go ahead'\n"
        "- After prompt optimization results are shown, say 'I'm done for now'"
    ),
    initial_message=(
        "I have my translation project set up with a spec, eval dataset, scorer, "
        "and a baseline prompt at prompts/translation_v0.md. "
        "Let's evaluate the lfm2.5-1.2b-instruct model with that prompt, "
        "then optimize it."
    ),
    expected_tools=["run_scoring", "list_models", "load_skill"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the artifact for how well it serves as a system prompt for a "
        "multi-language translation model (de, fr, es, en, zh as JSON).\n"
        "Check for: clear role definition, output format specification, "
        "language keys listed, quality instructions.\n"
        "10 = excellent prompt, 5 = mediocre, 1 = useless"
    ),
    seed_fn=seed_translation_eval,
    max_turns=15,
    stage_limits={"evaluation": 6, "prompt_optimization": 10},
)


# ---------------------------------------------------------------------------
# Scenario 1: Spec capture only (email summarization)
# ---------------------------------------------------------------------------

SPEC_ONLY_EMAIL = Scenario(
    name="spec_only_email",
    description=(
        "You are a user who wants to build a model that summarizes emails into "
        "concise 1-2 sentence subject lines. The emails come from a corporate "
        "environment (meeting requests, status updates, customer complaints). "
        "You want the summaries to capture the key action item or topic.\n\n"
        "Behavior rules:\n"
        "- Answer the agent's questions specifically and helpfully\n"
        "- When asked about input format, say emails are plain text, typically 3-20 sentences\n"
        "- When asked about output format, say 1-2 sentence subject lines\n"
        "- When asked about examples, let the agent generate them and give feedback\n"
        "- When asked about edge cases, mention forwarded emails with FW: prefixes\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message=(
        "I want to build a model that summarizes emails into 1-2 sentence subject lines"
    ),
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for how well it captures an email summarization task.\n"
        "Check for:\n"
        "- Input described as emails / plain text\n"
        "- Output described as short subject-line summaries (1-2 sentences)\n"
        "- Some mention of quality criteria or edge cases\n"
        "- Clear and actionable specification\n\n"
        "10 = comprehensive and clear, 5 = vague or missing details, 1 = wrong task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)


# ---------------------------------------------------------------------------
# Scenario 2: Spec + data generation (sentiment classification)
# ---------------------------------------------------------------------------

SPEC_AND_DATAGEN_SENTIMENT = Scenario(
    name="spec_and_datagen_sentiment",
    description=(
        "You are a user who needs a sentiment classifier that labels customer "
        "reviews as positive, negative, or neutral. Reviews come from an "
        "e-commerce platform and are typically 1-5 sentences.\n\n"
        "Behavior rules:\n"
        "- Answer spec questions helpfully: reviews are 1-5 sentences, labels "
        "are positive/negative/neutral, output should be just the label\n"
        "- When asked for examples, let the agent generate them\n"
        "- When offered next steps after spec creation, choose 'Start generating data'\n"
        "- When shown draft data samples, say they look good\n"
        "- When asked about judge criteria, confirm the proposed dimensions\n"
        "- When asked about validation set size, say '50 samples is fine'\n"
        "- After the validation set is generated, say you are done for now"
    ),
    initial_message=(
        "I need a sentiment classifier that labels customer reviews as positive, "
        "negative, or neutral"
    ),
    expected_tools=["ask_user", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score for how well it captures a sentiment classification task.\n"
        "Check for: 3 labels (positive/negative/neutral), input as customer reviews, "
        "clear output format, quality criteria.\n"
        "10 = precise, 5 = vague, 1 = wrong task"
    ),
    max_turns=40,
    stage_limits={"spec_capture": 15, "data_generation": 20},
)


# ---------------------------------------------------------------------------
# Scenario 3: Seeded spec → datagen + eval (translation)
# ---------------------------------------------------------------------------

DATAGEN_AND_EVAL_TRANSLATION = Scenario(
    name="datagen_and_eval_translation",
    description=(
        "You are a user with an existing translation SPEC.md. You want to "
        "generate training data and then evaluate a baseline API model.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'let's generate training data'\n"
        "- When asked about data generation preferences, say 'go ahead'\n"
        "- When shown draft samples, say they look good\n"
        "- When asked about eval criteria, confirm the proposed dimensions\n"
        "- When asked about validation set size, say '30 samples'\n"
        "- After data is generated, say 'now let's evaluate a baseline model'\n"
        "- When asked which model, say 'use lfm2.5-1.2b-instruct'\n"
        "- If the agent suggests lfm2-8b or LFM2-8B, tell it that model is "
        "currently offline and to use lfm2.5-1.2b-instruct instead\n"
        "- After eval results are shown, say 'I'm done for now'"
    ),
    initial_message=(
        "Let's generate training data and evaluate a baseline model"
    ),
    expected_tools=["ask_user", "create_file", "run_data_gen_pipeline", "run_scoring"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the scorer/eval criteria for how well it evaluates translation quality.\n"
        "Check for: scoring scale, language completeness check, format compliance, "
        "accuracy criteria.\n"
        "10 = thorough scoring rubric, 5 = basic criteria, 1 = unusable"
    ),
    max_turns=45,
    stage_limits={"data_generation": 20, "evaluation": 10},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _TRANSLATION_SPEC, encoding="utf-8"
    ),
)


# ---------------------------------------------------------------------------
# Scenario 4: Seeded spec+data → train on remote
# ---------------------------------------------------------------------------

async def seed_full_pipeline_async(project_dir: Path) -> None:
    """Pre-populate project with spec, datasets, scorer for training."""
    # Write spec
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")

    # Write pipeline
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "translation_v1.py").write_text(_TRANSLATION_PIPELINE, encoding="utf-8")

    # Write scorer
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "translation_v1.md").write_text(_TRANSLATION_SCORER, encoding="utf-8")

    # Write prompt
    prompts = project_dir / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "translation_v0.md").write_text(
        "Translate the following text into German, French, Spanish, English, "
        "and Chinese. Output ONLY a JSON object with keys: de, fr, es, en, zh.",
        encoding="utf-8",
    )

    # Generate training + eval datasets
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    for name, n in [("translation_v1", 80), ("translation_v1_eval", 20)]:
        output_dir = project_dir / "datasets" / name
        result = await run_pipeline(
            script_path=dg / "translation_v1.py",
            num_samples=n,
            output_dir=output_dir,
            client=client,
            concurrency=5,
        )
        logger.info("Seeded %s: %d/%d", name, result.succeeded, result.total)

    # Seed remote: add BOTH global machine and project binding.
    # Only seeding the project binding leaves an orphan that the agent
    # cannot use — the global machine is required for get_remote() to resolve.
    _seed_remote(project_dir, prefix="pipeline")


def _get_remote_host() -> str | None:
    """Get remote host from environment (set by test conftest/CLI)."""
    import os
    return os.environ.get("LQH_E2E_REMOTE_HOST")


def _seed_remote(project_dir: Path, *, prefix: str) -> None:
    """Register the E2E remote host globally and bind it to this project.

    Reads ``LQH_E2E_REMOTE_HOST`` for the hostname.  The same value is used
    as the remote name (so the agent can refer to it by that name).  Uses
    ``lqh.remote.config.add_remote()`` which writes both layers atomically.
    """
    remote_host = _get_remote_host()
    if not remote_host:
        return

    from lqh.remote.backend import RemoteConfig
    from lqh.remote.config import add_remote

    remote_root = f"/tmp/lqh-e2e-{prefix}-{uuid4().hex[:8]}"
    config = RemoteConfig(
        name=remote_host,
        type="ssh_direct",
        hostname=remote_host,
        remote_root=remote_root,
    )
    add_remote(project_dir, config)
    logger.info("Seeded remote %s -> %s", remote_host, remote_root)


FULL_PIPELINE_TRANSLATION = Scenario(
    name="full_pipeline_translation",
    description=(
        "You are a user with an existing translation project that has spec, "
        "training data, eval data, and a scorer ready. A remote GPU host is "
        "ALREADY configured in the project — the agent should call remote_list "
        "to discover it, NOT try to add a new one.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'let's fine-tune the model "
        "on the configured remote GPU — check remote_list to find it'\n"
        "- When asked about training config, accept the defaults\n"
        "- When asked which base model, say 'LFM2.5-1.2B-Instruct'\n"
        "- If the agent suggests lfm2-8b or LFM2-8B, say that model is "
        "currently offline and to use LFM2.5-1.2B-Instruct instead\n"
        "- When asked about LoRA, say 'LoRA is fine'\n"
        "- If the agent tries to add a new remote machine, stop it and say "
        "'the remote is already configured, just use remote_list to find it'\n"
        "- Monitor training progress when the agent shows status updates\n"
        "- After training completes or evaluation results are shown, say 'I'm done for now'"
    ),
    initial_message=(
        "I'd like to fine-tune a model on this data. Use the remote GPU."
    ),
    expected_tools=["summary", "read_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score whether the project is in a state where training was attempted or completed.\n"
        "Check for: training run directory exists, config.json present, "
        "or agent communicated training status.\n"
        "10 = training completed, 5 = training started, 1 = no training attempted"
    ),
    max_turns=30,
    stage_limits={"train": 25},
    seed_fn=seed_full_pipeline_async,
)


# ---------------------------------------------------------------------------
# Scenario 5: Spec capture for tool-calling task (customer support)
# ---------------------------------------------------------------------------

SPEC_ONLY_TOOLS_SUPPORT = Scenario(
    name="spec_only_tools_support",
    description=(
        "You are a user who wants to build a customer support agent that uses "
        "API tools to look up orders, process refunds, and escalate issues. "
        "The agent should call the right tool based on the customer's request.\n\n"
        "Behavior rules:\n"
        "- When asked about tools, describe 3: lookup_order (takes order_id), "
        "process_refund (takes order_id and reason), escalate_issue (takes issue_id "
        "and priority level)\n"
        "- When asked about input format, say customers write in natural language\n"
        "- When asked about output, the agent should call the right tool and "
        "then explain the result to the customer\n"
        "- When asked about edge cases, mention cases where customers provide "
        "wrong order IDs or ask for things the tools can't do\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message=(
        "I want to build a customer support agent that can look up orders, "
        "process refunds, and escalate issues using API tools"
    ),
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for how well it captures a tool-calling customer support task.\n"
        "Check for:\n"
        "- Tool definitions mentioned (lookup_order, process_refund, escalate_issue or similar)\n"
        "- Tool parameters described\n"
        "- The agent should call tools, not just respond with text\n"
        "- Edge cases for tool usage\n\n"
        "10 = comprehensive tool-calling spec, 5 = mentions tools vaguely, 1 = no tool mention"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)


# ---------------------------------------------------------------------------
# Scenario 6: Spec + datagen for tool-calling task (scheduling assistant)
# ---------------------------------------------------------------------------

SPEC_AND_DATAGEN_TOOLS_CALENDAR = Scenario(
    name="spec_and_datagen_tools_calendar",
    description=(
        "You are a user who needs a scheduling assistant that can check calendar "
        "availability, book meetings, and send invites via tool calls.\n\n"
        "Behavior rules:\n"
        "- When asked about tools, describe 3: check_availability (takes date and "
        "participants list), book_meeting (takes title, date, time, duration, "
        "participants), send_invite (takes meeting_id and message)\n"
        "- Input: natural language requests like 'schedule a meeting with Alice'\n"
        "- Output: the assistant should call the right tool with correct arguments, "
        "then confirm the result\n"
        "- When offered next steps after spec, choose 'Start generating data'\n"
        "- When shown draft data, check that tool calls are present and say 'looks good'\n"
        "- When asked about validation set size, say '30 samples'\n"
        "- After validation set is generated, say 'I'm done for now'"
    ),
    initial_message=(
        "I need a scheduling assistant that can check calendar availability, "
        "book meetings, and send invites via tool calls"
    ),
    expected_tools=["ask_user", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score for how well it captures a tool-calling scheduling task.\n"
        "Check for: tool definitions with parameters, tool call format described, "
        "scheduling domain covered.\n"
        "10 = precise tool-calling spec, 5 = vague, 1 = no tools mentioned"
    ),
    max_turns=45,
    stage_limits={"spec_capture": 15, "data_generation": 25},
)


# ---------------------------------------------------------------------------
# Scenario 7: Seeded tool-calling spec → datagen + eval (IT helpdesk)
# ---------------------------------------------------------------------------

_HELPDESK_SPEC = """\
# Specification: IT Helpdesk Tool-Calling Agent

## Overview
Build an IT helpdesk assistant that uses tools to check system status, create
support tickets, and restart services. The assistant receives natural language
requests from employees and calls the appropriate tool.

## Available Tools

### check_system_status
- **Description**: Check the current status of an IT system or service
- **Parameters**:
  - `system_name` (string, required): Name of the system (e.g., "email", "vpn", "database")
- **Returns**: JSON with status (up/down/degraded), uptime, last_incident

### create_ticket
- **Description**: Create a support ticket for an issue
- **Parameters**:
  - `title` (string, required): Brief title of the issue
  - `description` (string, required): Detailed description of the problem
  - `priority` (string, required): "low", "medium", "high", or "critical"
  - `category` (string, required): "network", "software", "hardware", "access"
- **Returns**: JSON with ticket_id, status, estimated_resolution_time

### restart_service
- **Description**: Restart a specific IT service
- **Parameters**:
  - `service_name` (string, required): Name of the service to restart
  - `reason` (string, required): Reason for the restart
- **Returns**: JSON with success (boolean), new_status, restart_time

## Input Format
- Natural language from employees describing IT issues
- May be vague ("email is broken") or specific ("can't connect to VPN from home office")

## Output Format
- The assistant calls the appropriate tool with correct arguments
- After receiving tool results, provides a helpful response to the employee

## Requirements
1. Correctly identify which tool to use based on the employee's request
2. Extract correct parameters from natural language
3. Handle ambiguous requests by asking for clarification or making reasonable assumptions
4. Provide clear, non-technical explanations of tool results

## Examples

### Example 1
**User**: "The VPN is acting weird, I can't connect from home"
**Assistant**: [calls check_system_status(system_name="vpn")]
**Tool result**: {"status": "degraded", "uptime": "45 mins", "last_incident": "VPN gateway overloaded"}
**Assistant**: "The VPN is currently experiencing some issues — it's in a degraded state..."

### Example 2
**User**: "My email hasn't been working for 2 hours, this is urgent"
**Assistant**: [calls create_ticket(title="Email not working", description="Employee reports email down for 2 hours", priority="high", category="software")]
"""

DATAGEN_AND_EVAL_TOOLS_HELPDESK = Scenario(
    name="datagen_and_eval_tools_helpdesk",
    description=(
        "You are a user with an existing IT helpdesk tool-calling SPEC.md. "
        "You want to generate training data with tool calls and evaluate a baseline.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'let's generate training data'\n"
        "- When asked about data generation preferences, say 'go ahead'\n"
        "- When shown draft samples, check tool_calls are present and say 'looks good'\n"
        "- When asked about eval criteria, confirm the proposed dimensions\n"
        "- When asked about validation set size, say '30 samples'\n"
        "- After data is generated, say 'now let's evaluate a baseline model'\n"
        "- When asked which model, say 'use lfm2.5-1.2b-instruct'\n"
        "- If the agent suggests lfm2-8b or LFM2-8B, tell it that model is "
        "currently offline and to use lfm2.5-1.2b-instruct instead\n"
        "- After eval results are shown, say 'I'm done for now'"
    ),
    initial_message=(
        "Generate training data with tool calls and evaluate the baseline"
    ),
    expected_tools=["ask_user", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the scorer/eval criteria for how well it evaluates tool-calling quality.\n"
        "Check for: correct tool selection criteria, argument accuracy criteria, "
        "response quality after tool results.\n"
        "10 = thorough tool-calling rubric, 5 = basic, 1 = no tool evaluation"
    ),
    max_turns=45,
    stage_limits={"data_generation": 20, "evaluation": 10},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _HELPDESK_SPEC, encoding="utf-8"
    ),
)


# ---------------------------------------------------------------------------
# Scenario 8: Seeded tool-calling data → train on remote
# ---------------------------------------------------------------------------

async def seed_full_pipeline_tools_async(project_dir: Path) -> None:
    """Pre-populate project with tool-calling spec, datasets, scorer for training."""
    # Write spec
    (project_dir / "SPEC.md").write_text(_HELPDESK_SPEC, encoding="utf-8")

    # Copy the tool-calling pipeline
    pipeline_src = Path(__file__).parent.parent.parent / "data_gen" / "tool_calling.py"
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    shutil.copy(pipeline_src, dg / "tool_calling_v1.py")

    # Write scorer
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "tool_calling_v1.md").write_text(
        "# Scorer: Tool Calling Quality\n\n"
        "## Task Description\n"
        "The assistant calls tools to handle IT helpdesk requests.\n\n"
        "## Scoring Scale\n"
        "- **9-10**: Correct tool, accurate arguments, helpful response\n"
        "- **7-8**: Correct tool, minor argument issues\n"
        "- **5-6**: Correct tool, significant argument problems\n"
        "- **3-4**: Wrong tool or missing tool call\n"
        "- **1-2**: No tool call or completely wrong\n\n"
        "## Critical Failures (score <= 3)\n"
        "- Wrong tool selected\n"
        "- No tool call when one was needed\n",
        encoding="utf-8",
    )

    # Generate training + eval datasets
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline

    config = load_config()
    token = require_token()
    client = create_client(token, config.api_base_url)

    for name, n in [("tool_calling_v1", 80), ("tool_calling_v1_eval", 20)]:
        output_dir = project_dir / "datasets" / name
        result = await run_pipeline(
            script_path=dg / "tool_calling_v1.py",
            num_samples=n,
            output_dir=output_dir,
            client=client,
            concurrency=5,
        )
        logger.info("Seeded %s: %d/%d", name, result.succeeded, result.total)

    # Seed remote (both global machine + project binding)
    _seed_remote(project_dir, prefix="tools")


FULL_PIPELINE_TOOLS = Scenario(
    name="full_pipeline_tools",
    description=(
        "You are a user with an existing tool-calling project (IT helpdesk) that "
        "has spec, tool-calling training data, eval data, and a scorer. A remote "
        "GPU host is ALREADY configured in the project — the agent should call "
        "remote_list to discover it, NOT try to add a new one.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'let's fine-tune the model "
        "on the configured remote GPU — check remote_list to find it'\n"
        "- When asked about training config, accept the defaults\n"
        "- When asked which base model, say 'LFM2.5-1.2B-Instruct'\n"
        "- If the agent suggests lfm2-8b or LFM2-8B, say that model is "
        "currently offline and to use LFM2.5-1.2B-Instruct instead\n"
        "- When asked about LoRA, say 'LoRA is fine'\n"
        "- If the agent tries to add a new remote machine, stop it and say "
        "'the remote is already configured, just use remote_list to find it'\n"
        "- After training completes or results are shown, say 'I'm done for now'"
    ),
    initial_message=(
        "Fine-tune a model on this tool-calling data using the remote GPU"
    ),
    expected_tools=["summary", "read_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score whether training on tool-calling data was attempted.\n"
        "Check for: training run directory, config, status updates.\n"
        "10 = training completed, 5 = started, 1 = not attempted"
    ),
    max_turns=30,
    stage_limits={"train": 25},
    seed_fn=seed_full_pipeline_tools_async,
)
