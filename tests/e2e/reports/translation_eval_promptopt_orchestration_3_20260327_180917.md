# E2E Test Report: translation_eval_promptopt

## Summary

| Metric | Value |
|--------|-------|
| Orchestration model | `orchestration:3` |
| Duration | 186.6s |
| User turns | 0 |
| Tool calls | 0 |
| Skills loaded | none |
| Errors | 1 |
| Artifacts created | 5 |
| SPEC.md | ✅ |
| Scorer | ✅ |

## Scenario
> You are a user with an existing translation project. The spec, a 50-sample eval dataset, a scorer, and a baseline system prompt (prompts/translation_v0.md) are already set up. You want to:
1. Evaluate the lfm2.5-1.2b-instruct model using the baseline prompt
2. Then optimize the system prompt for that model

Behavior rules:
- When the agent shows project state, ask to run model evaluation using the baseline prompt prompts/translation_v0.md
- When asked which model, say 'lfm2.5-1.2b-instruct'
- When shown eval results, say 'let's optimize the prompt'
- When asked about prompt optimization preferences, say 'go ahead'
- After prompt optimization results are shown, say 'I'm done for now'

## Errors
- Crashed: AttributeError: 'list' object has no attribute 'keys'

## Tool Usage


## Conversation Transcript

## Artifacts Created

### SPEC.md
```md
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

```

### data_gen/__pycache__/translation_v1.cpython-314.pyc
*<binary, 4506 bytes>*

### data_gen/translation_v1.py
```py
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
                {"role": "user", "content"

*[truncated]*
```

### evals/scorers/translation_v1.md
```md
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

```

### prompts/translation_v0.md
```md
Translate the following text into German, French, Spanish, English, and Chinese. Format your response as a JSON object with the keys "de", "fr", "es", "en", and "zh". Output ONLY the JSON object, nothing else.
```
