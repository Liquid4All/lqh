# Skill: Data Generation + Eval Criteria

You are now in **data generation** mode. This skill covers three phases:
1. **Draft iteration** — Build a pipeline, generate ~20 draft samples, iterate with the user
2. **Judge/scorer creation** — While user feedback is fresh, create evaluation criteria
3. **Validation set** — Generate a larger eval dataset and auto-score it

## Overview

The human-in-the-loop draft iteration is the most important part. During draft review, \
the user reveals their real requirements — edge cases, formatting preferences, quality \
expectations. That context is exactly what you need to write a good judge prompt. \
This is why data generation and eval criteria creation are in the same skill.

## Rules

1. **Always read the spec first.** Use `read_file` on `SPEC.md` (and any files in `other_specs/`) before writing any code.
2. **Create ONE pipeline file in `data_gen/`.** Name it descriptively (e.g., `data_gen/translation_v1.py`). Use `create_file` for the first version.
3. **MANDATORY: Generate draft samples first.** Always start with ~20 draft samples. Never skip draft review with the user. Use `show_file` + `ask_user` to get feedback.
4. **Fix the EXISTING file on failure.** If the pipeline errors, use `edit_file` to fix the SAME file and re-run. Do NOT create new files like `debug_pipeline.py`, `test_basic.py`, etc. Fix in place. Maximum 5 fix attempts before asking the user for help.
5. **Do NOT ask the user how many samples to generate or what content to cover.** You have the spec — design the pipeline yourself, generate drafts, and let the user react to concrete samples. Showing is better than asking.
6. **Update SPEC.md based on feedback.** If draft review reveals new requirements or edge cases, update the spec files before proceeding.
7. **Transient API errors are NOT code bugs.** If you see 405 or 502 errors, the pipeline code is fine — just re-run it. Do not rewrite the pipeline for API errors.
8. **NO system messages in generated data.** Pipelines must only generate user + assistant turns. System prompts are managed separately (via `prompts/` files) and injected at eval time. This allows prompt optimization to test different system prompts on the same data.
9. **Break out of failure loops.** If three consecutive `run_data_gen_pipeline` calls fail with the **same error message**, STOP retrying. Use `ask_user` to describe what you've tried and what the error says — the user may spot something you missed. Do not keep editing until you have a new hypothesis.

## Pipeline Interface

⚠️ **CRITICAL: Import correctly!** Every pipeline MUST start with `from lqh.pipeline import ...`.
Do NOT use `from data_gen.base import ...`, `from data_gen import ...`, `from pipeline import ...`, or any relative imports. The ONLY valid import path is `lqh.pipeline`.

**Quick reference — common mistakes to avoid:**

| ❌ Wrong | ✅ Right |
|---|---|
| `from lqh import Pipeline` | `from lqh.pipeline import Pipeline` |
| `from lqh.pipeline import Sample` | `from lqh.pipeline import ChatMLMessage` (there is no `Sample`) |
| `await client(messages=...)` | `await client.chat.completions.create(model=..., messages=...)` |
| `client = AsyncOpenAI(...)` | `client` is the argument to `generate()` — never construct your own |
| `Conversation(messages=[...])` | `[ChatMLMessage(...), ...]` (return a plain list) |
| `ChatMLMessage("system", ...)` in the returned list | Only `user` and `assistant` turns; system prompts live in `prompts/` |

⚠️ **CRITICAL: Conversation is a plain list!** `Conversation` is a type alias for `list[ChatMLMessage]`. Do NOT call `Conversation(messages=[...])` — that will fail. Return a plain list:
```python
return [ChatMLMessage("user", "..."), ChatMLMessage("assistant", "...")]
```

Every pipeline file must follow this structure:

```python
from lqh.pipeline import (
    Pipeline, ChatMLMessage, Conversation, GenerationError, step,
    ToolCall, FunctionCall, ToolDef,  # only if generating tool-call data
)
import json
import random
import liquidrandom

class MyPipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        # ... generate one training sample ...
        # NOTE: Do NOT include system messages. Only user + assistant turns.
        # System prompts are managed separately via prompts/ files.
        return [
            ChatMLMessage("user", "..."),
            ChatMLMessage("assistant", "..."),
        ]
```

### Key Constraints

- **One `Pipeline` subclass per file.** The engine discovers it automatically.
- **No top-level execution.** No `if __name__ == "__main__"`, no `asyncio.run()`. The engine calls `generate()`.
- **`self` is safe.** Each sample gets a fresh instance. Store intermediate state freely on `self`.
- **The `client` is pre-configured.** Do not construct your own `AsyncOpenAI`. Auth, base URL, timeouts, and rate limiting are handled by the engine.
- **Available imports:** `lqh.pipeline` (all pipeline types), `liquidrandom`, `json`, `random`, `base64`, `pathlib.Path`. Standard library modules are fine. Do not import external packages beyond these.

## Model Selection Guide

The `client` is an `AsyncOpenAI` instance pointed at `api.lqh.ai`. Use these model strings:

| Model | When to Use | Cost/Speed |
|---|---|---|
| `random:small` | **Default for most steps.** Single-turn generation, simple Q&A, persona creation, reformulations, extracting/rephrasing text. Use for ~80% of pipeline steps. | Cheapest, fastest |
| `random:medium` | Multi-turn generation in a single call, complex reasoning, structured JSON with nested objects, tool-call generation. ~15% of pipeline steps. | Moderate |
| `random:large` | Use sparingly. Only when the user explicitly requests it or for tasks requiring frontier-level quality. ~5% at most. | Expensive, slow |
| `small` / `medium` / `large` | Non-random default model from each pool. Use when you need **consistency** (e.g., a validation/grading step where the same model every time matters). | Same cost |
| `random:<size>:<seed>` | Deterministic random: same seed = same model. Use for consistency **within** a sample (e.g., all turns in one conversation use the same model) while diversity **across** samples. | Same cost |

**Rule of thumb:** Start with `random:small`. Upgrade to `medium` only if output quality is insufficient for that step. Use `large` only when the user explicitly asks for it.

### Output Length

The harness installs a default `max_tokens=16384` on the client, which fits the vast majority of single-message and short-thread generations. Pass `max_tokens=...` explicitly only when a step legitimately needs a different ceiling (e.g. very long-form generation that needs more, or a step where you want to cap costs by setting it lower). Don't bother passing it just to be defensive — the default already covers normal cases.

### JSON Mode

For structured output, use `response_format`:

```python
resp = await client.chat.completions.create(
    model="random:small",
    messages=[{"role": "user", "content": "Return JSON with name and age"}],
    response_format={"type": "json_object"},
)
data = json.loads(resp.choices[0].message.content)
```

## Using liquidrandom for Diversity

`liquidrandom` provides pre-generated seed data to inject variety into your prompts. **This is essential** -- without it, LLM-generated data will be repetitive and lack diversity.

### Available Categories

| Function | Returns | Use For |
|---|---|---|
| `liquidrandom.persona()` | Name, age, gender, occupation, nationality, personality, background | Varying the "voice" or perspective of generated content |
| `liquidrandom.job()` | Job category, sector, title, skills | Work-related scenarios |
| `liquidrandom.coding_task()` | Title, language, difficulty, description, constraints | Code generation data |
| `liquidrandom.math_category()` | Topic, field, description, example problems | Math/reasoning data |
| `liquidrandom.writing_style()` | Category, tone, characteristics | Varying output style |
| `liquidrandom.scenario()` | Title, theme, setting, context, stakes | Situational prompts |
| `liquidrandom.domain()` | Category, area, name, key concepts | Domain-specific content |
| `liquidrandom.science_topic()` | Topic, field, subfield, description | Science/research content |
| `liquidrandom.language()` | Category, register, region | Linguistic variation |
| `liquidrandom.reasoning_pattern()` | Name, category, description | Reasoning task variation |
| `liquidrandom.emotional_state()` | Category, intensity, valence, description | Emotional tone variation |
| `liquidrandom.instruction_complexity()` | Name, level, ambiguity, description | Instruction style variation |
| `liquidrandom.tool_group()` | Domain, tools with variations | Tool-calling data |

### Detail Levels

- `str(x)` or `x.detailed()` -- full details (all fields)
- `x.brief()` -- summary only (broad fields)

**Important:** When combining multiple seed data types in a single prompt, **always use `.brief()` for each.** Brief outputs are shorter, more compatible, and less likely to overwhelm the model. Reserve `.detailed()` for cases where a single seed type is the primary focus.

### Field Access

```python
persona = liquidrandom.persona()
persona.name          # "Carla"
persona.age           # 45
persona.occupation    # "Lead Midwife"
```

## The @step Decorator

Use `@step(retries=N)` on individual pipeline methods for granular retry control. When a method raises `GenerationError`, it retries up to N times before the error propagates. All state from prior steps is preserved.

```python
@step(retries=3)
async def _generate_question(self, client):
    resp = await client.chat.completions.create(
        model="random:small",
        messages=[{"role": "user", "content": f"Ask a question about: {self.topic}"}],
    )
    self.question = resp.choices[0].message.content.strip()
    if len(self.question) < 10:
        raise GenerationError("Question too short, retrying")
```

If a `GenerationError` escapes `generate()` (all step retries exhausted), the engine discards the instance and retries the entire sample from scratch (up to 3 times by default).

**Only raise `GenerationError` for retriable failures** (bad LLM output, missing fields, quality check failures). Other exceptions (import errors, logic bugs) are not retried.

## Pipeline Design Patterns

### Pattern 1: Simple Single-Turn Q&A

```python
class SimpleQA(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        persona = liquidrandom.persona()
        domain = liquidrandom.domain()

        # Generate question
        resp = await client.chat.completions.create(
            model="random:small",
            messages=[{
                "role": "user",
                "content": (
                    f"You are {persona.brief()}. "
                    f"Ask a specific question about {domain.brief()}. "
                    f"Write only the question."
                ),
            }],
        )
        question = resp.choices[0].message.content.strip()

        # Generate answer
        resp2 = await client.chat.completions.create(
            model="random:small",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant. Give a clear, accurate answer."},
                {"role": "user", "content": question},
            ],
        )
        answer = resp2.choices[0].message.content.strip()

        # No system message — only user + assistant turns
        return [
            ChatMLMessage("user", question),
            ChatMLMessage("assistant", answer),
        ]
```

### Pattern 2: Multi-Turn with Consistent Seed

```python
class MultiTurnConversation(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.persona = liquidrandom.persona()
        self.topic = liquidrandom.science_topic()
        self.seed = f"{self.persona.name}-{self.topic.name}"

        await self._turn_1(client)
        await self._turn_2(client)

        return [
            ChatMLMessage("user", self.q1),
            ChatMLMessage("assistant", self.a1),
            ChatMLMessage("user", self.q2),
            ChatMLMessage("assistant", self.a2),
        ]

    @step(retries=3)
    async def _turn_1(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": f"You are {self.persona.brief()}. Ask about {self.topic.brief()}.",
            }],
        )
        self.q1 = resp.choices[0].message.content.strip()

        resp2 = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[
                {"role": "system", "content": "You are a helpful tutor. Explain clearly."},
                {"role": "user", "content": self.q1},
            ],
        )
        self.a1 = resp2.choices[0].message.content.strip()

    @step(retries=3)
    async def _turn_2(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[
                {"role": "user", "content": self.q1},
                {"role": "assistant", "content": self.a1},
                {"role": "user", "content": "Ask a follow-up question. Only the question."},
            ],
        )
        self.q2 = resp.choices[0].message.content.strip()

        resp2 = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[
                {"role": "system", "content": "You are a helpful tutor."},
                {"role": "user", "content": self.q1},
                {"role": "assistant", "content": self.a1},
                {"role": "user", "content": self.q2},
            ],
        )
        self.a2 = resp2.choices[0].message.content.strip()
```

### Pattern 3: JSON Extraction with Validation

```python
class StructuredExtraction(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.scenario = liquidrandom.scenario()
        self.style = liquidrandom.writing_style()

        await self._generate_document(client)
        await self._generate_extraction(client)

        return [
            ChatMLMessage("user", f"Extract key information from this text:\n\n{self.document}"),
            ChatMLMessage("assistant", self.extraction_json),
        ]

    @step(retries=3)
    async def _generate_document(self, client):
        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[{
                "role": "user",
                "content": (
                    f"Write a short document (150-300 words) about this scenario: {self.scenario.brief()}. "
                    f"Style: {self.style.brief()}. "
                    f"Include specific names, dates, numbers, and locations."
                ),
            }],
        )
        self.document = resp.choices[0].message.content.strip()
        if len(self.document) < 100:
            raise GenerationError("Document too short")

    @step(retries=5)
    async def _generate_extraction(self, client):
        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[
                {"role": "system", "content": "Extract all key entities and facts from the text. Return a JSON object with keys: people, places, dates, organizations, key_facts."},
                {"role": "user", "content": self.document},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        required_keys = {"people", "places", "dates", "organizations", "key_facts"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            raise GenerationError(f"Missing keys in extraction: {missing}")
        self.extraction_json = raw
```

### Pattern 4: Tool-Calling Conversations

```python
class ToolCallingPipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.group = liquidrandom.tool_group()
        self.persona = liquidrandom.persona()
        self.seed = f"{self.persona.name}-tools"

        # Pick a consistent variation index for all tools
        self.variation_idx = random.randint(0, 7)

        await self._generate_scenario(client)
        await self._generate_tool_interaction(client)

        tool_defs = []
        for tool in self.group.tools:
            if tool.variations:
                var = tool.variations[self.variation_idx % len(tool.variations)]
                tool_defs.append(ToolDef(
                    name=var.name,
                    description=var.description if hasattr(var, 'description') else tool.canonical_name,
                    parameters=var.parameters,
                ))

        return [
            ChatMLMessage("system", self.system_prompt, tools=tool_defs),
            ChatMLMessage("user", self.user_request),
            ChatMLMessage("assistant", None, tool_calls=[self.tool_call]),
            ChatMLMessage("tool", self.tool_result, tool_call_id=self.tool_call.id, name=self.tool_call.function.name),
            ChatMLMessage("assistant", self.final_response),
        ]

    @step(retries=3)
    async def _generate_scenario(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"You are {self.persona.brief()}. "
                    f"You need help with something related to: {self.group.domain}. "
                    f"Available tools: {self.group.description}. "
                    f"Write a natural user request. Only the request, nothing else."
                ),
            }],
        )
        self.user_request = resp.choices[0].message.content.strip()
        self.system_prompt = f"You are a helpful assistant with access to tools for {self.group.domain}."

    @step(retries=5)
    async def _generate_tool_interaction(self, client):
        # Pick a tool to call
        available_tools = [t for t in self.group.tools if t.variations]
        if not available_tools:
            raise GenerationError("No tools with variations available")
        tool = random.choice(available_tools)
        var = tool.variations[self.variation_idx % len(tool.variations)]

        # Generate arguments
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Given this user request: \"{self.user_request}\"\n"
                    f"And this tool: {var.name} with parameters: {json.dumps(var.parameters)}\n"
                    f"Generate realistic arguments for calling this tool. Return only a JSON object with the arguments."
                ),
            }],
            response_format={"type": "json_object"},
        )
        args = json.loads(resp.choices[0].message.content)
        self.tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name=var.name, arguments=json.dumps(args)),
        )

        # Generate tool result
        resp2 = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Generate a realistic JSON response for the tool '{var.name}' "
                    f"called with arguments: {json.dumps(args)}. Return only JSON."
                ),
            }],
            response_format={"type": "json_object"},
        )
        self.tool_result = resp2.choices[0].message.content.strip()

        # Generate final response
        resp3 = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_request},
                {"role": "assistant", "content": f"[Tool result: {self.tool_result}]"},
                {"role": "user", "content": "Now write a natural response using the tool result. Only the response."},
            ],
        )
        self.final_response = resp3.choices[0].message.content.strip()
```

#### Quick Reference: Tool-Calling Pipeline Invariants

Before writing a tool-calling pipeline, check for an existing reference in the project:
- If `data_gen/tool_calling.py` exists, `read_file` it first — it is a known-working scaffold you can copy the structure from.

Invariants every tool-calling pipeline MUST satisfy:

1. **Imports** from `lqh.pipeline` include the tool types:
   ```python
   from lqh.pipeline import (
       Pipeline, ChatMLMessage, Conversation,
       FunctionCall, ToolCall, ToolDef,
       GenerationError, step,
   )
   ```
2. **Tools live in the system message**, not as a separate argument:
   ```python
   ChatMLMessage("system", "You are a helpful assistant.", tools=[tool_def])
   ```
   The system message is the only place tools go. Do NOT put tools in the user message or assign them to the assistant message.
3. **Assistant messages with tool calls** use `tool_calls=[...]` and `content=None` (tool-call-only turns have no text):
   ```python
   ChatMLMessage("assistant", None, tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="...", arguments=json.dumps({...})))])
   ```
4. **Tool result turns** use `role="tool"` with matching `tool_call_id`:
   ```python
   ChatMLMessage("tool", json.dumps({"result": ...}), tool_call_id="call_1")
   ```
5. **Final assistant turn** after tool result is a plain text response:
   ```python
   ChatMLMessage("assistant", "Here's what I found...")
   ```
6. Return a **flat list** (not a `Conversation(...)` constructor call) containing these turns in order: system → user → assistant(tool_call) → tool → assistant(text).

If any invariant is unclear, open `data_gen/tool_calling.py` or `tests/e2e/benchmark/fixtures/broken_pipeline_system_msg.py` (for a counter-example of what NOT to do).

### Pattern 5: Bring-Your-Own-Data

When the user has their own data (images, prompts, seed data, or a full
dataset), **use the helpers in `lqh.sources`** instead of writing your own
file I/O. They are tested, path-traversal-safe, and fail with clear errors
on missing files — which prevents the infinite fix-loops we see when the
agent hand-rolls glob/parquet/HF code.

**Rule:** for image folders, prompt files, parquet/jsonl datasets, HF
datasets, and seed files, import the matching helper from `lqh.sources`.
Only write custom loading if no helper fits. Mixing helpers with manual
`os.listdir`/`glob`/`pyarrow` for the same folder is forbidden.

Before writing the pipeline, call `list_user_data` — it tells you what
the user has placed in the project and which helper to use.

#### 5a. Image folder → label/caption

```python
import lqh.sources as sources

class LabelSatellite(Pipeline):
    @classmethod
    def source(cls, project_dir):
        # Subfolders carry coarse labels; set include_subfolder_label=True.
        return sources.image_folder(
            project_dir / "images", include_subfolder_label=True,
        )

    async def generate(self, client, input: sources.ImageItem) -> Conversation:
        label_hint = f" (hint: subfolder is '{input.subfolder}')" if input.subfolder else ""
        return [
            ChatMLMessage("user", [
                {"type": "text", "text": f"What type of land is in this image?{label_hint}"},
                {"type": "image_url", "image_url": {"url": input.as_data_url()}},
            ]),
            ChatMLMessage("assistant", await self._caption(client, input)),
        ]

    @step(retries=2)
    async def _caption(self, client, item):
        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the land use in this image."},
                    {"type": "image_url", "image_url": {"url": item.as_data_url()}},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()
```

#### 5b. Bring-your-prompts → complete them

```python
import lqh.sources as sources

class CompletePrompts(Pipeline):
    @classmethod
    def source(cls, project_dir):
        return sources.prompts(project_dir / "prompts.jsonl")

    async def generate(self, client, input: sources.PromptItem) -> Conversation:
        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[{"role": "user", "content": input.prompt}],
        )
        return [
            ChatMLMessage("user", input.prompt),
            ChatMLMessage("assistant", resp.choices[0].message.content.strip()),
        ]
```

#### 5c. Bring-your-seed-data → combine with liquidrandom

```python
import random
import liquidrandom
import lqh.sources as sources

class FloristWriting(Pipeline):
    @classmethod
    def source(cls, project_dir):
        # seed_data/flowers.{jsonl,csv,txt} dropped by the user.
        # Returning None would mean pure synthesis; returning the seed list
        # means each entry is iterated once (set samples_per_item>1 on the
        # tool call to iterate many times per seed).
        return sources.seed_data("flowers")

    async def generate(self, client, input) -> Conversation:
        flower = input if isinstance(input, str) else input["name"]
        style = liquidrandom.writing_style().brief()
        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[
                {"role": "system", "content": f"Respond in this style: {style}"},
                {"role": "user", "content": f"Write a short paragraph about {flower}s."},
            ],
        )
        return [
            ChatMLMessage("user", f"Write a short paragraph about {flower}s."),
            ChatMLMessage("assistant", resp.choices[0].message.content.strip()),
        ]
```

#### 5d. HF dataset → annotate

```python
import lqh.sources as sources

class AnnotateFromHF(Pipeline):
    @classmethod
    def source(cls, project_dir):
        # Streaming: doesn't download the full dataset; engine caps at num_samples.
        return sources.hf_dataset("squad", split="train", streaming=True)

    async def generate(self, client, input: dict) -> Conversation:
        ...  # same shape as bring-your-prompts
```

#### Map vs iterate

The engine takes `num_samples` and `samples_per_item`:

- **map 1x**: `num_samples=len(source)`, `samples_per_item=1` (default) — one output per input item.
- **iterate N×**: `samples_per_item=N` — generate N outputs per input, using fresh `self` each time (good for small seed lists + `liquidrandom` diversity).
- **partial**: `num_samples=N` smaller than the source size — iterate through the source until N items have been enqueued, then stop.

## Workflow

### Phase 1: Draft Iteration (Human-in-the-Loop)

This phase is **mandatory**. Never skip it.

> **Note on quality checks**: Phase 3.5 will filter generated data with the
> LLM judge before training. So you do not need to over-engineer pipeline-side
> validation here — the judge is the catch-all sanity check. Pipeline-side
> assertions are still useful for *structural* invariants (required fields
> present, correct types, dataset shape) but don't try to encode subjective
> quality rules in `@step` decorators; that's the scorer's job.

#### Step 1.1: Read the Spec

Use `read_file` on `SPEC.md`. Also check `other_specs/` and read any files there.
Identify: task type, input/output format, requirements, examples, edge cases.

#### Step 1.2: Design and Create the Pipeline

Think about:
- What seed data types from `liquidrandom` map to the spec's requirements?
- How many turns does the conversation need?
- What quality checks can you add (length, required fields, format)?

Create the pipeline file with `create_file` in `data_gen/`.

#### Step 1.3: Test with 1-3 Samples

Run `run_data_gen_pipeline` with `num_samples=1`. If it fails, fix the pipeline and retry. Once it runs without errors, generate 2-3 more to check diversity.

#### Step 1.4: Generate Draft Set (~20 samples)

Run with `num_samples=20` and output to `datasets/{name}_draft/`.

#### Step 1.5: Show Drafts to User

Use `show_file` on the draft parquet to let the user browse samples, then `ask_user`:

```
ask_user(
    question="I've generated 20 draft samples. How do they look?",
    options=[
        "Samples look good, proceed to evaluation criteria",
        "Some issues - let me explain what to fix",
        "Major problems - needs significant changes",
        "Update the spec first, then regenerate"
    ]
)
```

#### Step 1.6: Iterate Based on Feedback

- If the user reports issues: fix the pipeline, regenerate drafts, show again
- If the user wants spec changes: update `SPEC.md` with `edit_file`, then regenerate
- **Keep iterating until the user approves the draft quality**
- Each iteration: fix → regenerate → show → ask

### Phase 2: Judge/Scorer Creation

This phase happens **immediately after** the user approves the drafts. The user's feedback from Phase 1 is still in context — use it to write better evaluation criteria.

#### Step 2.1: Propose Judge Dimensions

Based on the spec AND the user's draft feedback, propose specific evaluation dimensions. Use `ask_user` with multi-select:

```
ask_user(
    question="I'll create evaluation criteria for scoring. Which dimensions matter most?",
    options=[
        "Format compliance (JSON structure, required fields)",
        "Content accuracy (factual correctness)",
        "Completeness (all required elements present)",
        "Natural language quality (fluency, tone)",
        "Edge case handling"
    ],
    multi_select=true
)
```

Add dimensions specific to what the user cared about during draft review. If they flagged issues with tone, add a tone dimension. If they cared about specific formatting, add that.

#### Step 2.2: Create the Scorer

Create a scorer `.md` file in `evals/scorers/` with:
- Task description (from spec)
- Scoring scale (1-10 with descriptions)
- Specific dimensions the user confirmed
- Critical failure conditions (from spec + user feedback)
- Examples of good/bad scores

**Important: conversation format.** The scorer will receive conversations formatted as:

```
[User]
<the user's input text>

[Assistant]
<the model's response to score>
```

For multi-turn conversations, each turn is labelled with `[User]` or `[Assistant]`.
The scorer should refer to "the assistant's response" when describing what to evaluate,
not "the JSON object" or "the output array". The assistant's response IS the text
after `[Assistant]`.

Use `show_file` + `ask_user` to confirm the scorer with the user before proceeding.

#### Step 2.3: Create Response Format Schema (if applicable)

If the task requires structured output (JSON), create a `prompts/{task}.schema.json` file with the JSON schema. This constrains model output during evaluation, preventing format errors.

Example for a translation task:
```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "translation",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "de": {"type": "string"},
        "fr": {"type": "string"},
        "es": {"type": "string"},
        "en": {"type": "string"},
        "zh": {"type": "string"}
      },
      "required": ["de", "fr", "es", "en", "zh"],
      "additionalProperties": false
    }
  }
}
```

The schema is auto-discovered at eval time: when `system_prompt_path="prompts/translation_v0.md"` is used, the system looks for `prompts/translation.schema.json` in the same directory. Only create this for tasks with structured output — free-form text tasks don't need it.

### Phase 3: Validation Set Generation

#### Step 3.1: Suggest Validation Set Size

Suggest a validation set size based on task complexity:
- Simple tasks (classification, short Q&A): 100-200 samples
- Medium tasks (summarization, translation): 200-300 samples
- Complex tasks (multi-turn, tool use): 300-500 samples

Use `ask_user` to confirm:
```
ask_user(
    question="I recommend generating 200 validation samples. This gives enough coverage for reliable scoring. Sound good?",
    options=["Yes, generate 200", "More (300-500)", "Fewer (50-100)"]
)
```

#### Step 3.2: Generate Validation Set

Run the approved pipeline with the agreed sample count:
```
run_data_gen_pipeline(
    script_path="data_gen/{task}_v1.py",
    num_samples=200,
    output_dataset="{task}_eval"
)
```

#### Step 3.3: Auto-Score the Validation Set

Run data quality scoring on the validation set:
```
run_scoring(
    dataset="datasets/{task}_eval",
    scorer="evals/scorers/{task}_v1.md",
    mode="data_quality"
)
```

Show the scoring results to the user with `show_file` on the scores.parquet.

#### Step 3.4: Report and Next Steps

Present a summary: total samples, mean/median score, any issues found. Then offer next steps.

### Phase 3.5: Filter Before Training (Default-On)

**Rule: always filter the training set with `run_data_filter` before SFT or
DPO unless the user explicitly opts out.** Do not hand a raw generated dataset
to training.

This is not optional polish — it is a load-bearing sanity check. Here's why:

- **Mixed-model generation**: lqh pipelines call multiple generator models
  (often `random:small` / `random:medium`), so per-sample quality varies
  meaningfully. The mean is fine; the tail is not. Training on the tail teaches
  the model the wrong thing.
- **Catches scorer/spec mismatch early**: if filtering keeps <50% of samples,
  either the data is bad *or* the scorer is off-axis with the spec. Both
  conditions block training, both surface here. Easier to fix at this stage
  than after a failed SFT run.
- **Cheap**: an LLM judge call per sample is ~10× cheaper than a wasted
  training run.

#### Step 3.5.1: Run the filter

```
run_data_filter(
    input_path="datasets/{task}_train_raw/data.parquet",
    scorer_path="evals/scorers/{task}_v1.md",
    output_dataset="{task}_train_filtered",
    threshold=7.0,            # 7.0 is the default starting point; adjust per task
)
```

The `threshold` parameter (1-10) is the minimum judge score to keep a sample.
**Start at 7.0**. Tighten if the kept set is too noisy after a quick eyeball;
loosen if you're losing too much volume. The `summary.json` reports kept/dropped
counts and mean score per bucket, so you can see at a glance what was cut.

Repeat for the validation set if you generated it via the pipeline (skip this
if the eval set is human-curated):

```
run_data_filter(
    input_path="datasets/{task}_eval_raw/data.parquet",
    scorer_path="evals/scorers/{task}_v1.md",
    output_dataset="{task}_eval_filtered",
    threshold=7.0,
)
```

#### Step 3.5.2: Sanity-check the kept ratio with the user

```
ask_user(
    question=(
        "Filtered N → M samples (kept X%). Does this look reasonable?"
    ),
    options=[
        "Looks good, proceed to training",
        "Tighten threshold (drop more)",
        "Loosen threshold (keep more)",
        "Kept ratio is too low — improve the pipeline first",
    ],
)
```

Heuristics for interpreting the ratio:
- **kept ≥ 75%**: pipeline quality is good; threshold could possibly be tightened.
- **50–75%**: typical and healthy; proceed.
- **<50%**: yellow flag. Either bad pipeline or wrong scorer. Investigate
  with `extract_failures` on the dropped samples (high-scoring drops are
  scorer/spec disagreement; low-scoring drops are real quality issues).

#### Step 3.5.3: Read the score distribution, not just the mean

`run_data_filter` and `run_scoring` now print a per-bucket histogram and
five quantiles (p10/p25/p50/p75/p90) under the headline numbers. The
*shape* matters more than the mean for deciding what to do next:

- **Tight unimodal at high score** (e.g. p10=8, p50=9, p90=10): great.
  Filter is barely doing anything but the data was already good.
  Consider raising the threshold to 8 to keep only the cleanest set,
  or just proceed with this dataset as-is.
- **Bimodal** (e.g. p25=8, p50=9, p10=4 with a separate cluster at
  4-5): the dataset is a *mix* of good and broken samples. The mean
  hides this. Run `extract_failures` on the low cluster — usually
  reveals one or two failure modes the pipeline can fix at the source.
  Re-running datagen after the fix is more efficient than tightening
  the threshold (you'll just regenerate the same mix).
- **Uniformly mediocre** (e.g. p10=5, p50=6, p90=7): the pipeline is
  consistently producing OK-but-not-great samples. Threshold won't
  help — even the top decile is mediocre. The fix is usually a
  better generator model (`random:medium` → `random:large` for
  hard fields) or a sharper system prompt in the pipeline.
- **Long high tail, low floor** (p10=2, p50=8, p90=9): occasional bad
  samples among mostly-good. Use threshold filtering (default 7)
  and proceed; the rare failures are probably API-side issues, not
  systematic.

#### Step 3.5.4: When SFT or DPO show no improvement, **scale data**

If you've validated that SFT (and possibly DPO) gives a real but
small improvement over baseline (e.g. +0.2-0.4 on the eval mean),
scaling data is usually the highest-leverage next move:

- **Generate 2-5× more samples** with the same pipeline. Big returns
  for free if the pipeline is already producing high-quality data.
  Re-filter and re-train. Diminishing returns set in around 10k
  samples for most tasks.
- **Diversify the seed dimensions** in the pipeline (more
  `liquidrandom` categories, more scenario types, more edge cases).
  Often more useful than raw volume — fixes long-tail eval failures
  before they become a pattern.
- **Mix in a bring-your-own-data pass** (see `data_filtering` skill)
  if there's a real-world dataset you can score and filter into the
  training set.

If the post-SFT eval is roughly *equal* to the API baseline (no gap
left to close), more data won't move the needle. The bottleneck is
either the model size or the eval scorer being too lenient — neither
fixed by data scaling.

#### Step 3.5.5: Hand the *filtered* dataset to training

When you launch SFT/DPO, point at `datasets/{task}_train_filtered/data.parquet`,
**never** at the unfiltered raw dataset.

#### Cross-reference

For advanced filtering scenarios — multi-pass filtering, custom thresholds per
sample, bring-your-own-data — see the `data_filtering` skill.

### Phase 3.6: DPO data viability check (when DPO is being considered)

DPO needs a *preference signal* — pairs where the chosen response
clearly beats the rejected one according to the same judge that
will eval the model. If that signal isn't there in the data, DPO
overfits to whatever weak preferences it can find and the model
gets worse, not better. Before kicking off DPO, run this check:

1. Look at the chosen-pool score distribution from
   `run_data_filter` (or re-score the filtered set with
   `run_scoring`). The mean should be **noticeably higher** than the
   model's baseline on the same eval. If the chosen pool's mean is
   8 and the base model's eval is also 8, there's no gap to learn
   from — DPO will pick noise.
2. If the chosen pool is too close to baseline:
   - Use `golden_source: "api"` with `golden_model: "large"` in the
     DPO config. The chosen comes from a stronger API model,
     producing a real quality differential.
   - Or skip DPO; for tasks where the base is already strong,
     diminishing returns from DPO are real.
3. After DPO starts, call `training_status(run_name=...)` to see the
   per-iter selection funnel + held-out eval inline. Each iter shows
   one line, e.g.:

   ```
     DPO iterations:
       iter_000: 254/1268 pairs  gap p50=2.5, p90=5.5  → held-out mean=6.08 (Δ -0.37)
       iter_001: 254/1268 pairs  gap p50=3.0, p90=6.0  → held-out mean=6.22 (Δ -0.23)
       iter_002: 0/1268 pairs  gap p50=0.5, p90=1.0 ⚠️ skipped: below_min_pairs_per_iter
   ```

   Read iter 0 before letting the run continue all iterations:
   - `gap p50 ≥ 1.0`: real preference signal, DPO should help.
   - `gap p50 ≈ 0`: weak signal. The chosen and rejected are
     scored similarly by the judge. Not worth running 3 iters —
     stop early or fall back to stronger chosen as above.
   - `kept < 100` of a 2000-sample dataset: same diagnosis — not
     enough informative pairs above the gap floor.
   - **Held-out trajectory** is the most direct signal: if iter 0's
     held-out delta is meaningfully negative (<−0.3), DPO is
     hurting; stop_training and reconsider rather than waiting
     for the harness's auto-abort threshold.

### Phase 3.7: When DPO regresses — don't continue a doomed path

If DPO comes back as failed or aborted (the harness writes
`early_abort.json`; `training_status` surfaces the reason inline),
**do not just declare failure to the user**. Run the diagnostic
ladder below and try the *next* config rather than waiting for
human input. Stop only after two failed retries.

#### Step 1: read the chosen-pool ceiling

`training_status` shows a line like:

```
  Chosen-pool ceiling: 7.55 — model can't exceed this on the same judge.
```

Compare it to the post-SFT (or baseline, for DPO-only) eval mean.

- **post-SFT within 0.5 of ceiling** (e.g. ceiling 7.55, post-SFT
  7.20): the model is at the data ceiling. Hyperparameter tuning
  cannot help. → Go to Phase 3.5.4 (scale data) instead of retrying
  DPO with different settings.
- **headroom ≥ 1.0** (e.g. ceiling 7.55, post-SFT 6.72): real room
  to grow exists. Continue to step 2.

#### Step 2: read the iter-0 held-out delta

The first DPO iter's held-out vs baseline (visible in
`training_status` as "Δ vs baseline" or in
`iter_000/held_out_eval.json`) is the cleanest signal of whether
DPO is even the right tool:

- **iter 0 ≥ +0.1**: DPO worked at iter 0; the regression is from
  cumulative drift over later iters. → Step 3 (lower learning rate
  or fewer iters).
- **iter 0 ≈ 0**: marginal signal; could go either way. → Step 3
  with caution.
- **iter 0 ≤ −0.3**: DPO is hurting from the very first step. The
  preference data isn't useful for this model state. → Skip DPO,
  return to SFT-only or scale data.

#### Step 3: retry with adjusted hyperparams

Pick ONE adjustment, halve the relevant knob:

- **Trend-abort fired** (held-out trajectory was heading down
  cumulatively): halve the DPO learning rate (5e-6 → 2.5e-6) OR
  drop `num_iterations` to 1. The iter-0 model is the best-so-far
  and was saved as the abort checkpoint, so restart from there.
- **Absolute-abort fired** at iter 1+ (delta vs baseline): same
  fix as above — fewer / smaller steps. Likely the LoRA is
  drifting too fast.

Edit `e2e_config.json` (or the active config), launch a fresh DPO
run, monitor `training_status` again.

#### Step 4: stop after two failed retries

If two adjusted configs both regress, **the data is the bottleneck,
not the hyperparams.** Stop iterating on DPO. Surface the situation
to the user with:
- The chosen-pool ceiling
- The post-SFT eval mean (current best result)
- The two failed DPO config attempts and their held-out trajectories
- A recommendation: either accept post-SFT, or scale data per Phase 3.5.4.

Do not endlessly tune learning rates. The marginal gain is small;
the cost is large; an honest "DPO doesn't help here" is more useful
than five hours of wishful experiments.

## Pipeline Troubleshooting

### Root-cause discipline for data quality issues

When the user reports a problem with generated samples (class imbalance, lack of
diversity, wrong tone, formality drift, repetitive outputs), do **not** just
tweak a prompt string and re-run. That usually does not fix the root cause.

Instead:
1. `read_file` on the pipeline.
2. Identify the **mechanism** producing the issue. Ask: which variable in the code
   controls the dimension the user is unhappy about? Is there no such variable?
3. `edit_file` to introduce or fix that mechanism.

Common recipes:

| Symptom | Root cause | Mechanism to add |
|---------|-----------|------------------|
| Class imbalance (e.g., 70% positive reviews) | No explicit label selection | `self.label = random.choice(LABELS)` then thread `self.label` into the generation prompt |
| Tone / formality drift (always formal, always casual) | Style not controlled per sample | `self.style = liquidrandom.writing_style()` or pick `random.choice(FORMALITY_LEVELS)` and pass into the prompt |
| Lack of input diversity (same topic every time) | No persona / topic seed | `self.persona = liquidrandom.persona()` plus `random.choice(SAMPLE_TYPES)` |
| Repetitive outputs with the same model | Model determinism | Use seeded model IDs like `random:small:{self.seed}` where `self.seed` varies per sample |

### Validate edits with a small re-run

After any fix, re-run `run_data_gen_pipeline` with `num_samples=5` and inspect
the actual output distribution via `read_file` on the generated parquet.
Report the observed distribution to the user before declaring the issue fixed.
Do not rely on "it should work now" reasoning without evidence.

### Diagnosing runtime errors

- `GenerationError("Missing keys: {'en', 'de', ...}")` — the required-key set in
  your validation logic does not match what the LLM returns. **Cross-reference
  SPEC.md's output format** — the spec's key names are authoritative. Fix the
  code's `required_keys` set, not the prompt.
- `ModuleNotFoundError` on pipeline import — you used the wrong import path.
  The ONLY valid path is `from lqh.pipeline import ...`.
- `AttributeError: 'AsyncOpenAI' object has no attribute '...'` — you called
  a method that does not exist on the client (e.g., `client.models.list()`).
  The client supports `client.chat.completions.create(...)` and
  `client.audio.speech.create(...)`. Use model IDs directly without discovery.
- `TypeError: Conversation() ...` — `Conversation` is a type alias for
  `list[ChatMLMessage]`, not a class. Return a plain list, not `Conversation(...)`.
- Recurring `GenerationError` with the same message across 3+ retries — see
  Rule 9: stop, call `ask_user`, describe what you tried.

## Next Steps

After the validation set is generated and scored, the recommended path is **validate → scale → polish**:

1. **"Run a pilot training run"** (recommended) — Load the `train` skill and run a small SFT pilot (200-500 samples) to validate that the data improves model performance. This is fast (under a minute) and confirms the pipeline produces useful training signal before investing in larger runs.
2. **"Evaluate models on this data"** — Load `/eval` to run zero-shot baselines on different LFMs and compare which model performs best on this task.
3. **"Refine the data or scorer"** — If quality scores are low, iterate on the pipeline or scorer.
4. **"Start prompt optimization"** — Load `/prompt` to optimize a system prompt if a model is already chosen.
5. **"I'm done for now"** — End the session.

When the user chooses to train, generate a **training dataset** (separate from the validation set) using the same pipeline. Start with 200-500 samples for the pilot, then scale to thousands once improvement is confirmed. Load the `train` skill for the full training workflow.

## Common Mistakes to Avoid

1. **Forgetting `liquidrandom`**: Every pipeline should use at least one `liquidrandom` category. Without it, the data will be monotonous.
2. **Using `random:large` by default**: Start with `random:small`. It is cheaper, faster, and sufficient for most steps.
3. **No validation in steps**: Always check the LLM output (length, required fields, format) and raise `GenerationError` if it does not meet requirements.
4. **Putting everything in one step**: Break the pipeline into multiple `@step`-decorated methods. This gives granular retry control and makes debugging easier.
5. **Using `.detailed()` for everything**: Use `.brief()` when combining multiple seed types in one prompt.
6. **Constructing your own client**: The `client` argument is already configured. Do not create a new `AsyncOpenAI` instance.
7. **Adding `if __name__` blocks**: The engine loads and runs the pipeline. No top-level execution code.
8. **Not testing first**: Always run with num_samples=1-3 before scaling up. Fix issues before burning through API credits.

## Tips for High-Quality Data

- **Vary the system prompt**: Use `liquidrandom.writing_style()` or `liquidrandom.persona()` to create diverse system prompts across samples.
- **Use seeds for within-sample consistency**: `random:small:myseed` ensures the same model for all steps in one sample, while different samples get different models.
- **Add quality gates**: Check output length, required keywords, format compliance. Reject and retry bad samples early.
- **Think about edge cases**: The spec's edge cases section tells you what tricky inputs to generate. Make sure some fraction of your data covers these.
- **Match the spec's examples**: Your pipeline output should look like the examples in SPEC.md. If it does not, adjust your prompts.
