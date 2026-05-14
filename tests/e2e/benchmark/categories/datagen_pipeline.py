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
        "- After pipeline runs successfully, say 'I'm done for now'"
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
    max_turns=40,
    stage_limits={"data_generation": 35},
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
        "- After pipeline runs successfully, say 'I'm done for now'"
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
    max_turns=40,
    stage_limits={"data_generation": 35},
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
        "- After pipeline runs successfully, say 'I'm done for now'"
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
    max_turns=40,
    stage_limits={"data_generation": 35},
    seed_fn=lambda project_dir: Path(project_dir / "SPEC.md").write_text(
        _TOOL_CALLING_SPEC, encoding="utf-8"
    ),
)


SCENARIOS = [
    DATAGEN_PIPELINE_SENTIMENT,
    DATAGEN_PIPELINE_MATH_QA,
    DATAGEN_PIPELINE_TOOL_CALLING,
]
