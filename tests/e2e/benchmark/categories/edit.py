"""Category 6: Edit Spec and Pipeline benchmark scenarios.

Tests whether the agent can update requirements without breaking
existing functionality.
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
- **Example**: {"de": "...", "fr": "...", "es": "...", "en": "...", "zh": "..."}

## Requirements
1. All 5 target languages must be present in every response
2. Translations must be accurate and natural
3. Preserve proper nouns, brand names, numbers
4. Match the formality level of the source text
5. Handle informal text, slang, and idioms gracefully
"""

_TRANSLATION_PIPELINE = '''\
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step
import json
import random
import liquidrandom


class TranslationPipeline(Pipeline):
    """Generate translation training samples."""

    SAMPLE_TYPES = [
        "casual message", "formal email", "technical sentence",
        "idiomatic expression", "short phrase",
    ]

    async def generate(self, client, input=None) -> Conversation:
        self.persona = liquidrandom.persona()
        self.sample_type = random.choice(self.SAMPLE_TYPES)
        self.seed = f"{self.persona.name}-{self.sample_type}"

        await self._generate_source(client)
        await self._generate_translations(client)

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
        required = {"de", "fr", "es", "en", "zh"}
        if not required.issubset(data.keys()):
            raise GenerationError(f"Missing keys: {required - set(data.keys())}")
        self.translations_json = json.dumps(data, ensure_ascii=False)
'''

_SENTIMENT_SPEC = """\
# Specification: Sentiment Classification

## Overview
Classify customer reviews into sentiment categories.

## Input Format
- **Type**: Plain text customer review, 1-5 sentences

## Output Format
- **Type**: JSON object
- **Fields**: {"label": "positive|negative|neutral"}

## Requirements
1. Three sentiment classes: positive, negative, neutral
2. Handle short reviews (1-3 words)
3. Classify by overall tone for mixed sentiment
"""

_SENTIMENT_PIPELINE = '''\
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step
import json
import random
import liquidrandom


class SentimentPipeline(Pipeline):
    """Generate sentiment classification training samples."""

    LABELS = ["positive", "negative", "neutral"]

    async def generate(self, client, input=None) -> Conversation:
        self.persona = liquidrandom.persona()
        self.label = random.choice(self.LABELS)
        self.seed = f"{self.persona.name}-{self.label}"

        await self._generate_review(client)

        return [
            ChatMLMessage("user", self.review),
            ChatMLMessage("assistant", json.dumps({"label": self.label})),
        ]

    @step(retries=3)
    async def _generate_review(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Write a {self.label} product review (1-3 sentences) from "
                    f"a {self.persona.brief()}. Output ONLY the review text."
                ),
            }],
        )
        self.review = resp.choices[0].message.content.strip()
        if len(self.review) < 5:
            raise GenerationError("Review too short")
'''


def _write_stub_parquet(path: Path, num_rows: int, template: str = "translation") -> None:
    """Write a minimal valid parquet file."""
    path.mkdir(parents=True, exist_ok=True)
    messages_list = []
    for i in range(num_rows):
        if template == "translation":
            msgs = json.dumps([
                {"role": "user", "content": f"Sample text {i + 1}."},
                {"role": "assistant", "content": json.dumps(
                    {"de": f"Text {i+1}", "fr": f"Texte {i+1}", "es": f"Texto {i+1}",
                     "en": f"Sample text {i+1}.", "zh": f"示例{i+1}"}
                )},
            ])
        else:
            msgs = json.dumps([
                {"role": "user", "content": f"Review text {i + 1}."},
                {"role": "assistant", "content": json.dumps({"label": "positive"})},
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


def _seed_translation_project(project_dir: Path) -> None:
    """Seed a complete translation project."""
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "translation_v1.py").write_text(_TRANSLATION_PIPELINE, encoding="utf-8")
    _write_stub_parquet(project_dir / "datasets" / "translation_v1", 50)


def _seed_sentiment_project(project_dir: Path) -> None:
    """Seed a complete sentiment project."""
    (project_dir / "SPEC.md").write_text(_SENTIMENT_SPEC, encoding="utf-8")
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "sentiment_v1.py").write_text(_SENTIMENT_PIPELINE, encoding="utf-8")
    _write_stub_parquet(project_dir / "datasets" / "sentiment_v1", 50, template="sentiment")


EDIT_SPEC_ADD_LANGUAGE = Scenario(
    name="bench_edit_spec_add_language",
    description=(
        "You are a user with an existing 5-language translation project. You want "
        "to add Portuguese as a 6th language.\n\n"
        "Behavior rules:\n"
        "- Tell the agent 'I need to add Portuguese (pt) as a 6th language'\n"
        "- When asked about details, say 'same quality requirements as the other languages'\n"
        "- When the agent updates the spec, verify Portuguese is added and say 'looks good'\n"
        "- When the agent suggests updating the pipeline too, say 'yes, update it'\n"
        "- After changes are made and pipeline runs, say 'I'm done for now'"
    ),
    initial_message=(
        "I need to add Portuguese as a 6th target language. Please update the "
        "SPEC.md and the data generation pipeline to include 'pt' as a key."
    ),
    expected_tools=["read_file", "edit_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score whether the SPEC.md and pipeline were correctly updated to include Portuguese.\n"
        "Check for:\n"
        "- SPEC.md mentions Portuguese / pt\n"
        "- Output format includes 'pt' key\n"
        "- Pipeline code updated to include pt\n"
        "- Other 5 languages preserved\n\n"
        "10 = both updated correctly, 5 = partial, 1 = broken"
    ),
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_seed_translation_project,
)

EDIT_SPEC_CHANGE_FORMAT = Scenario(
    name="bench_edit_spec_change_format",
    description=(
        "You are a user with a sentiment classifier. You want to change the output "
        "to include a confidence score in addition to the label.\n\n"
        "Behavior rules:\n"
        "- Tell the agent about the change\n"
        "- When asked about confidence range, say '0.0 to 1.0'\n"
        "- When shown updated spec, verify it includes confidence and say 'looks good'\n"
        "- When the agent updates the pipeline, say 'yes, go ahead'\n"
        "- After changes, say 'I'm done for now'"
    ),
    initial_message=(
        "I want to change the output format. Instead of just {\"label\": \"...\"}, "
        "I need {\"label\": \"...\", \"confidence\": 0.85}. Please update the spec "
        "and pipeline."
    ),
    expected_tools=["read_file", "edit_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score whether the spec and pipeline were updated for confidence score.\n"
        "Check for:\n"
        "- SPEC.md output format includes 'confidence' field\n"
        "- Confidence range 0.0-1.0 mentioned\n"
        "- Original label field preserved\n\n"
        "10 = both updated, 5 = only spec, 1 = broken"
    ),
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_seed_sentiment_project,
)

EDIT_PIPELINE_FORMALITY = Scenario(
    name="bench_edit_pipeline_formality",
    description=(
        "You are a user whose translation pipeline generates translations that "
        "ignore the formality level of the source text. You want the pipeline "
        "to better preserve formality.\n\n"
        "Behavior rules:\n"
        "- When the agent reads the pipeline, explain the problem\n"
        "- Say 'the translations always come out formal even when the input is casual'\n"
        "- When the agent proposes a fix, say 'sounds good, go ahead'\n"
        "- After the pipeline runs successfully, say 'I'm done for now'"
    ),
    initial_message=(
        "The translation pipeline is generating translations that always sound formal, "
        "even when the input is casual or informal. Can you fix the pipeline to better "
        "preserve the formality level?"
    ),
    expected_tools=["read_file", "edit_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score whether the pipeline was updated to preserve formality.\n"
        "Check for:\n"
        "- Pipeline prompt or logic mentions formality/tone\n"
        "- Pipeline still runs without errors\n"
        "- No unnecessary file recreation\n\n"
        "10 = good fix, runs, 5 = attempted but unclear, 1 = broken"
    ),
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_seed_translation_project,
)

EDIT_PIPELINE_DIVERSITY = Scenario(
    name="bench_edit_pipeline_diversity",
    description=(
        "You are a user whose sentiment pipeline generates too many 'positive' "
        "samples. You want a more balanced distribution across all three labels.\n\n"
        "Behavior rules:\n"
        "- Say the problem: 'about 70%% of samples are positive, I need roughly equal distribution'\n"
        "- When the agent proposes a fix, say 'sounds good'\n"
        "- After the pipeline runs, say 'I'm done for now'"
    ),
    initial_message=(
        "The sentiment pipeline is generating unbalanced data - about 70%% of samples "
        "come out as 'positive'. I need roughly equal distribution across positive, "
        "negative, and neutral. Can you fix the pipeline?"
    ),
    expected_tools=["read_file", "edit_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score whether the pipeline was updated for balanced label distribution.\n"
        "Check for:\n"
        "- Pipeline has explicit label balancing logic\n"
        "- All three labels (positive/negative/neutral) are generated\n"
        "- Pipeline still runs\n\n"
        "10 = balanced + runs, 5 = attempted, 1 = broken"
    ),
    max_turns=30,
    stage_limits={"data_generation": 25},
    seed_fn=_seed_sentiment_project,
)


SCENARIOS = [
    EDIT_SPEC_ADD_LANGUAGE,
    EDIT_SPEC_CHANGE_FORMAT,
    EDIT_PIPELINE_FORMALITY,
    EDIT_PIPELINE_DIVERSITY,
]
