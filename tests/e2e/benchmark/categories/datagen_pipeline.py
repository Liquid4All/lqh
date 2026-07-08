"""Category 3: Data Generation Pipeline benchmark scenarios.

Tests whether the LLM can implement a working pipeline following the
lqh.pipeline interface given a SPEC.md.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.scenarios import Scenario


_SENTIMENT_SPEC = """\
# Specification: Sentiment Classification

## Overview
Classify customer reviews into sentiment categories.

## Input Format
- **Type**: Plain text customer review, 1-5 sentences
- **Source**: E-commerce product reviews
- **Language**: English only

## Output Format
- **Type**: JSON object
- **Fields**: {"label": "positive|negative|neutral"}
- **Example**: {"label": "negative"}

## Requirements
1. Three sentiment classes: positive, negative, neutral
2. Reviews mentioning specific defects or issues should be "negative"
3. Reviews with mixed sentiment should be classified by the overall tone
4. Very short reviews (1-3 words like "Great product!") should be handled
5. Reviews with sarcasm should be classified by intended meaning

## Examples

### Example 1
**Input**: "This laptop is amazing! The battery lasts all day and the screen is gorgeous."
**Output**: {"label": "positive"}

### Example 2
**Input**: "Arrived broken. Returning immediately."
**Output**: {"label": "negative"}

### Example 3
**Input**: "It works as described. Nothing special but does the job."
**Output**: {"label": "neutral"}
"""

_MATH_QA_SPEC = """\
# Specification: Math Word Problem Q&A

## Overview
Solve math word problems and provide structured answers.

## Input Format
- **Type**: Natural language math word problem
- **Level**: Elementary to high school level
- **Topics**: Arithmetic, percentages, ratios, basic algebra, geometry

## Output Format
- **Type**: JSON object
- **Fields**:
  - "reasoning": Step-by-step solution explanation (string)
  - "answer": The numerical answer (number)
  - "unit": The unit of the answer if applicable (string or null)
- **Example**: {"reasoning": "If 3 apples cost $6, one apple costs $6/3 = $2", "answer": 2, "unit": "dollars"}

## Requirements
1. Show step-by-step reasoning before the final answer
2. Handle problems with multiple steps
3. Round decimal answers to 2 decimal places
4. Include units when the problem specifies them
5. If a problem is ambiguous or unsolvable, set answer to null and explain in reasoning

## Examples

### Example 1
**Input**: "A store offers 20% off all items. If a jacket costs $80, how much do you pay after the discount?"
**Output**: {"reasoning": "Discount = 20% of $80 = $16. Final price = $80 - $16 = $64.", "answer": 64, "unit": "dollars"}
"""

_TOOL_CALLING_SPEC = """\
# Specification: Restaurant Booking Assistant

## Overview
Build an assistant that helps users find and book restaurants using tool calls.

## Available Tools

### search_restaurants
- **Description**: Search for restaurants by criteria
- **Parameters**:
  - `cuisine` (string, optional): Type of cuisine (e.g., "Italian", "Japanese")
  - `location` (string, required): Area or neighborhood
  - `party_size` (integer, required): Number of guests
  - `date` (string, required): Date in YYYY-MM-DD format
- **Returns**: List of available restaurants with name, rating, price_range

### make_reservation
- **Description**: Book a table at a restaurant
- **Parameters**:
  - `restaurant_id` (string, required): ID from search results
  - `date` (string, required): Date in YYYY-MM-DD format
  - `time` (string, required): Time in HH:MM format
  - `party_size` (integer, required): Number of guests
  - `name` (string, required): Reservation name
- **Returns**: Confirmation with reservation_id and details

### cancel_reservation
- **Description**: Cancel an existing reservation
- **Parameters**:
  - `reservation_id` (string, required): ID from booking confirmation
- **Returns**: Cancellation confirmation

## Input Format
- Natural language requests from users
- May mention preferences, dietary restrictions, occasions

## Output Format
- The assistant calls the appropriate tool
- After tool results, provides a friendly summary
- Multi-turn: may need search -> select -> book flow

## Requirements
1. Correctly map user requests to tool calls with proper arguments
2. Handle multi-turn conversations (search, then book)
3. Ask for missing required information before calling tools
4. Provide helpful summaries after tool calls
"""


DATAGEN_PIPELINE_SENTIMENT = Scenario(
    name="bench_datagen_sentiment",
    description=(
        "You are a user with a sentiment classification SPEC.md. You want the agent "
        "to create and test a data generation pipeline.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'create a data generation pipeline'\n"
        "- When asked about preferences, say 'go ahead with your approach'\n"
        "- When shown draft data samples, say 'looks good'\n"
        "- When asked about sample count, say '3 samples for testing'\n"
        "- After the pipeline runs, let the agent create a scorer for the data "
        "and test it on the samples — say 'go ahead' if it asks. Only once it "
        "has created AND validated a scorer, say 'I'm done for now'"
    ),
    initial_message="Create a data generation pipeline for my sentiment classification task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the data generation pipeline code and any generated samples.\n"
        "Check for:\n"
        "- Imports from lqh.pipeline (Pipeline, ChatMLMessage, etc.)\n"
        "- Pipeline subclass with generate() method\n"
        "- Generated samples have user/assistant messages\n"
        "- Sentiment labels match spec (positive/negative/neutral)\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor quality, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _SENTIMENT_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_MATH_QA = Scenario(
    name="bench_datagen_math_qa",
    description=(
        "You are a user with a math Q&A SPEC.md. You want the agent to create and "
        "test a data generation pipeline with structured JSON output.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'create a data generation pipeline'\n"
        "- When asked about preferences, say 'go ahead'\n"
        "- When shown draft samples, check that the JSON has reasoning/answer/unit fields\n"
        "- When asked about sample count, say '3 for testing'\n"
        "- After the pipeline runs, let the agent create a scorer for the data "
        "and test it on the samples — say 'go ahead' if it asks. Only once it "
        "has created AND validated a scorer, say 'I'm done for now'"
    ),
    initial_message="Create a data generation pipeline for the math Q&A task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and Pipeline subclass\n"
        "- Samples contain math word problems as user messages\n"
        "- Assistant responses are JSON with reasoning, answer, unit\n\n"
        "10 = correct pipeline + good JSON samples, 5 = runs but bad format, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _MATH_QA_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_TOOL_CALLING = Scenario(
    name="bench_datagen_tool_calling",
    description=(
        "You are a user with a restaurant booking assistant SPEC.md that uses tool "
        "calling. You want a data generation pipeline that produces training samples "
        "with tool_calls.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'create a data generation pipeline'\n"
        "- When asked about preferences, say 'go ahead'\n"
        "- When shown draft samples, verify tool calls are present and say 'looks good'\n"
        "- When asked about sample count, say '3 for testing'\n"
        "- After the pipeline runs, let the agent create a scorer for the data "
        "and test it on the samples — say 'go ahead' if it asks. Only once it "
        "has created AND validated a scorer, say 'I'm done for now'"
    ),
    initial_message="Create a data generation pipeline for the restaurant booking assistant with tool calls",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports including tool call support\n"
        "- Pipeline generates multi-turn conversations with tool_calls\n"
        "- Tool definitions match the spec (search_restaurants, make_reservation, cancel_reservation)\n\n"
        "10 = correct tool-calling pipeline, 5 = runs but missing tool calls, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _TOOL_CALLING_SPEC, encoding="utf-8"
    ),
)


# ---------------------------------------------------------------------------
# Additional tasks. Each seeds a SPEC.md and asks the agent to build, run, and
# validate a data-generation pipeline. The user behaviour is identical across
# tasks (create pipeline -> approve draft -> create + validate scorer -> done),
# so it is factored into a small builder; only the draft-inspection hint and the
# task phrasing differ.
# ---------------------------------------------------------------------------


def _datagen_user(task_phrase: str, draft_check: str) -> str:
    return (
        f"You are a user with a {task_phrase} SPEC.md. You want the agent to "
        "create and test a data generation pipeline.\n\n"
        "Behavior rules:\n"
        "- When the agent shows project state, say 'create a data generation pipeline'\n"
        "- When asked about preferences, say 'go ahead with your approach'\n"
        f"- When shown draft data samples, {draft_check}\n"
        "- When asked about sample count, say '3 samples for testing'\n"
        "- After the pipeline runs, let the agent create a scorer for the data "
        "and test it on the samples — say 'go ahead' if it asks. Only once it "
        "has created AND validated a scorer, say 'I'm done for now'"
    )


_SUMMARIZATION_SPEC = """\
# Specification: Article Summarization

## Overview
Produce a short abstractive summary of a news/research article.

## Input Format
- **Type**: Plain text article, 300-3000 words
- **Language**: English

## Output Format
- **Type**: JSON object
- **Fields**:
  - "summary": 2-4 sentence abstractive summary (string)
  - "key_points": list of exactly 3 short bullet strings
- **Example**: {"summary": "...", "key_points": ["...", "...", "..."]}

## Requirements
1. Summary must be abstractive (rewritten), not copied sentences
2. Summary is 2-4 sentences, under 120 words
3. Exactly 3 key_points, each a short phrase
4. No information that is not present in the article
5. Neutral, factual tone
"""

_NER_SPEC = """\
# Specification: Named Entity Recognition

## Overview
Extract named entities from English text.

## Input Format
- **Type**: Plain text sentence or paragraph, up to 200 words
- **Language**: English

## Output Format
- **Type**: JSON list of objects
- **Fields per object**: "text" (exact span), "type" (PERSON|ORG|LOCATION|DATE|MONEY)
- **Example**: [{"text": "Tim Cook", "type": "PERSON"}, {"text": "Berlin", "type": "LOCATION"}]

## Requirements
1. Only the 5 entity types: PERSON, ORG, LOCATION, DATE, MONEY
2. Spans must be exact substrings of the input
3. Return an empty list [] when there are no entities
4. Nationalities/adjectives (e.g. "French") are not LOCATION
5. No overlapping spans
"""

_TEXT_TO_SQL_SPEC = """\
# Specification: Text-to-SQL

## Overview
Translate a natural-language question into a PostgreSQL SELECT query, given a
schema in the prompt.

## Input Format
- **Type**: A table schema (CREATE TABLE statements) followed by a question
- **Language**: English questions

## Output Format
- **Type**: Raw SQL string (no markdown fences)
- **Example**: SELECT name FROM customers WHERE country = 'DE';

## Requirements
1. PostgreSQL dialect
2. SELECT queries only — never INSERT/UPDATE/DELETE
3. Use only tables/columns present in the provided schema
4. Return the string UNANSWERABLE if the question cannot be answered from the schema
5. Output only the query, no explanation or fences
"""

_INTENT_SLOT_SPEC = """\
# Specification: Intent + Slot Filling

## Overview
Classify a smart-home utterance into an intent and extract its slots.

## Input Format
- **Type**: A short English command, 1-2 sentences
- **Language**: English

## Output Format
- **Type**: JSON object
- **Fields**: "intent" (set_temperature|turn_on_device|turn_off_device|play_music|out_of_scope), "slots" (object)
- **Example**: {"intent": "set_temperature", "slots": {"room": "bedroom", "temperature": 21}}

## Requirements
1. Exactly the 5 intents listed (including out_of_scope)
2. Only include slots relevant to the predicted intent
3. out_of_scope utterances return empty slots {}
4. temperature slot is numeric; room/device/song are strings
5. Never invent slots not implied by the utterance
"""

_EMAIL_REPLY_SPEC = """\
# Specification: Customer Email Reply Drafting

## Overview
Draft a professional reply to an inbound customer support email.

## Input Format
- **Type**: A single customer email (subject + body), English
- **Language**: English

## Output Format
- **Type**: Plain text email reply (free text)

## Requirements
1. Acknowledge the customer's specific issue in the first sentence
2. Provide a concrete next step or resolution
3. Professional, warm, concise tone (under 150 words)
4. Include a greeting and a sign-off
5. Never promise refunds or commitments not implied by the email
"""

_GROUNDED_QA_SPEC = """\
# Specification: Grounded Question Answering

## Overview
Answer a question using ONLY a provided context passage.

## Input Format
- **Type**: A context passage followed by a question
- **Language**: English

## Output Format
- **Type**: JSON object
- **Fields**: "answer" (string), "supported" (boolean)
- **Example**: {"answer": "1969", "supported": true}

## Requirements
1. Answer strictly from the context — no outside knowledge
2. If the context does not contain the answer, set answer to "" and supported to false
3. supported is true only when the answer is directly stated in the context
4. Keep answers short (a phrase, not a paragraph)
5. Never fabricate citations or facts
"""

_DOCSTRING_SPEC = """\
# Specification: Python Docstring Generation

## Overview
Generate a Google-style docstring for a given Python function.

## Input Format
- **Type**: A Python function definition (source code)
- **Language**: Python 3

## Output Format
- **Type**: JSON object
- **Fields**: "docstring" (the docstring body text, without triple quotes)
- **Example**: {"docstring": "Add two numbers.\\n\\nArgs:\\n    a: ...\\n    b: ...\\n\\nReturns:\\n    The sum."}

## Requirements
1. One-line summary first, imperative mood
2. Args section documenting each parameter by name
3. Returns section describing the return value
4. Raises section only if the function raises
5. Do not invent parameters that are not in the signature
"""

_PARAPHRASE_SPEC = """\
# Specification: Formal Paraphrasing

## Overview
Rewrite an informal English sentence into a formal register while preserving meaning.

## Input Format
- **Type**: A single informal English sentence
- **Language**: English

## Output Format
- **Type**: JSON object
- **Fields**: "formal" (the rewritten sentence)
- **Example**: {"formal": "I would like to request a refund for my order."}

## Requirements
1. Preserve the original meaning exactly
2. Remove slang, contractions, and filler words
3. Output a single sentence
4. Do not add information not present in the input
5. Keep it natural, not stilted
"""


DATAGEN_PIPELINE_SUMMARIZATION = Scenario(
    name="bench_datagen_summarization",
    description=_datagen_user(
        "article summarization",
        "check that the JSON has a summary and exactly 3 key_points, then say 'looks good'",
    ),
    initial_message="Create a data generation pipeline for my article summarization task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and a Pipeline subclass\n"
        "- User messages are articles; assistant messages are JSON with summary + 3 key_points\n"
        "- Summaries look abstractive and grounded in the article\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor format, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _SUMMARIZATION_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_NER = Scenario(
    name="bench_datagen_ner",
    description=_datagen_user(
        "named-entity-recognition",
        "check that assistant outputs are JSON lists of {text, type} and say 'looks good'",
    ),
    initial_message="Create a data generation pipeline for my NER task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and a Pipeline subclass\n"
        "- User messages are sentences; assistant messages are JSON lists of {text, type}\n"
        "- Only the 5 allowed entity types appear; spans are substrings of the input\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor format, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _NER_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_TEXT_TO_SQL = Scenario(
    name="bench_datagen_text_to_sql",
    description=_datagen_user(
        "text-to-SQL",
        "check that each sample includes a schema + question and a SELECT query, then say 'looks good'",
    ),
    initial_message="Create a data generation pipeline for my text-to-SQL task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and a Pipeline subclass\n"
        "- User messages contain a schema and a question; assistant messages are SELECT queries\n"
        "- Queries reference only schema tables/columns; no write statements\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor format, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _TEXT_TO_SQL_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_INTENT_SLOT = Scenario(
    name="bench_datagen_intent_slot",
    description=_datagen_user(
        "intent + slot-filling",
        "check that assistant outputs are JSON with intent + slots, then say 'looks good'",
    ),
    initial_message="Create a data generation pipeline for my intent + slot task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and a Pipeline subclass\n"
        "- User messages are commands; assistant messages are JSON with intent + slots\n"
        "- Only the 5 allowed intents appear; slots match the intent\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor format, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _INTENT_SLOT_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_EMAIL_REPLY = Scenario(
    name="bench_datagen_email_reply",
    description=_datagen_user(
        "customer email reply drafting",
        "check that assistant messages are professional free-text replies and say 'looks good'",
    ),
    initial_message="Create a data generation pipeline for my email-reply drafting task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and a Pipeline subclass\n"
        "- User messages are inbound customer emails; assistant messages are reply drafts\n"
        "- Replies acknowledge the issue, give a next step, and are concise/professional\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor quality, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _EMAIL_REPLY_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_GROUNDED_QA = Scenario(
    name="bench_datagen_grounded_qa",
    description=_datagen_user(
        "grounded question-answering",
        "check that samples include a context + question and JSON {answer, supported}, then say 'looks good'",
    ),
    initial_message="Create a data generation pipeline for my grounded QA task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and a Pipeline subclass\n"
        "- User messages contain a context passage + question; assistant messages are JSON {answer, supported}\n"
        "- Answers are grounded in the context; unsupported questions yield supported=false\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor format, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _GROUNDED_QA_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_DOCSTRING = Scenario(
    name="bench_datagen_docstring",
    description=_datagen_user(
        "Python docstring generation",
        "check that user messages are functions and assistant messages are JSON {docstring}, then say 'looks good'",
    ),
    initial_message="Create a data generation pipeline for my docstring-generation task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and a Pipeline subclass\n"
        "- User messages are Python functions; assistant messages are JSON {docstring}\n"
        "- Docstrings are Google-style with Args/Returns matching the signature\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor format, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _DOCSTRING_SPEC, encoding="utf-8"
    ),
)

DATAGEN_PIPELINE_PARAPHRASE = Scenario(
    name="bench_datagen_paraphrase",
    description=_datagen_user(
        "formal-paraphrasing",
        "check that assistant messages are JSON {formal} preserving meaning, then say 'looks good'",
    ),
    initial_message="Create a data generation pipeline for my formal-paraphrasing task",
    expected_tools=["read_file", "create_file", "run_data_gen_pipeline"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the pipeline code and generated samples.\n"
        "Check for:\n"
        "- Correct lqh.pipeline imports and a Pipeline subclass\n"
        "- User messages are informal sentences; assistant messages are JSON {formal}\n"
        "- Paraphrases preserve meaning, drop slang/contractions, stay one sentence\n\n"
        "10 = correct pipeline + good samples, 5 = runs but poor quality, 1 = broken"
    ),
    max_turns=50,
    stage_limits={"data_generation": 45},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _PARAPHRASE_SPEC, encoding="utf-8"
    ),
)


SCENARIOS = [
    DATAGEN_PIPELINE_SENTIMENT,
    DATAGEN_PIPELINE_MATH_QA,
    DATAGEN_PIPELINE_TOOL_CALLING,
    DATAGEN_PIPELINE_SUMMARIZATION,
    DATAGEN_PIPELINE_NER,
    DATAGEN_PIPELINE_TEXT_TO_SQL,
    DATAGEN_PIPELINE_INTENT_SLOT,
    DATAGEN_PIPELINE_EMAIL_REPLY,
    DATAGEN_PIPELINE_GROUNDED_QA,
    DATAGEN_PIPELINE_DOCSTRING,
    DATAGEN_PIPELINE_PARAPHRASE,
]
