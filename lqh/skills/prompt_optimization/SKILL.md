# Skill: Prompt Optimization

You are now in **prompt optimization** mode. Your goal is to create and iteratively refine a system prompt that maximizes a model's performance on the user's task.

## Overview

1. Read the spec → create an initial system prompt → save to `prompts/`
2. Run model eval with that prompt on an eval dataset
3. Extract failure cases using `get_eval_failures`
4. Analyze failure patterns and refine the prompt
5. Repeat steps 2-4 autonomously (2-3 iterations)
6. Present a comparison table and the final prompt to the user

**This loop runs autonomously.** Do not call `ask_user` between iterations. Run the full optimization loop, then report results.

## Prerequisites

Before starting, verify these exist (use `summary`):
- **SPEC.md** — the project specification (required)
- **Eval dataset** — a labelled dataset in `datasets/` with `_eval` suffix (required)
- **Scorer** — a scoring criteria file in `evals/scorers/` (required)

If any are missing, tell the user what's needed and offer to create them (load the appropriate skill).

## Prompt file conventions

Store prompts as `.md` files in `prompts/`:

```
prompts/
  {task}_v1.md    # initial prompt from spec
  {task}_v2.md    # refined after iteration 1
  {task}_v3.md    # refined after iteration 2
```

The prompt file contains **only the raw system prompt text** — no frontmatter, no metadata. The agent should be able to read it and pass it directly as a system message.

## Eval run naming

Each optimization iteration produces an eval run:

```
evals/runs/{task}_prompt_v{N}_iter{M}/
```

Examples:
- `evals/runs/summarization_prompt_v1_iter1/` — first prompt, first eval
- `evals/runs/summarization_prompt_v2_iter2/` — refined prompt, second eval

## The optimization loop

### Step 0: Gather context

1. Use `summary` to understand the project state
2. Use `read_file` on `SPEC.md` (and any `other_specs/` files)
3. Check if an eval dataset and scorer already exist
4. Check if previous prompt optimization runs exist (resume from last iteration)
5. Use `list_models` to see available models

### Step 1: Create the initial prompt

Read the spec carefully. Create a system prompt in `prompts/{task}_v1.md` that:

- **Defines the role clearly** — "You are a {specific role} that {specific task}"
- **States the output format explicitly** — what structure, what fields, what length
- **Lists constraints from the spec** — must-haves, must-nots, edge cases
- **Includes 1-2 examples** if the spec provides them

**Do NOT** just dump the spec into the prompt. Translate spec requirements into clear, actionable instructions for the model.

Use `create_file` to save it to `prompts/{task}_v1.md`.

### Step 2: Run eval

```
run_scoring(
    dataset="datasets/{task}_eval",
    scorer="evals/scorers/{task}_v1.md",
    mode="model_eval",
    run_name="{task}_prompt_v1_iter1",
    inference_model="small",  # or user-specified model
    system_prompt_path="prompts/{task}_v1.md"
)
```

After the eval completes, read the summary:
```
read_file("evals/runs/{task}_prompt_v1_iter1/summary.json")
```

Record the mean score for comparison.

### Step 3: Extract and analyze failures

```
get_eval_failures(
    eval_run="evals/runs/{task}_prompt_v1_iter1",
    threshold=6.0,
    min_failures=5
)
```

Analyze the failures. Look for patterns:
- **Format violations** — model ignoring output format instructions → add explicit format examples
- **Missing information** — model omitting required content → add checklist of required elements
- **Hallucination** — model inventing facts → add "only use information from the input" constraint
- **Wrong scope** — model too verbose or too brief → add length/detail guidance
- **Edge case failures** — specific input patterns causing issues → add handling instructions for those patterns

### Step 4: Refine the prompt

Based on failure analysis, create an improved prompt:

1. **Targeted fixes** — address the specific failure patterns, don't rewrite everything
2. **Add examples** — if the model fails on a pattern, add an example of correct behavior
3. **Strengthen weak constraints** — if "be concise" isn't working, try "respond in 1-2 sentences maximum"
4. **Don't remove working instructions** — only add or modify, don't strip things that work

Save as `prompts/{task}_v{N+1}.md` using `create_file`.

### Step 5: Repeat or stop

**Continue** if:
- Iteration count < 3
- Score improved by >= 0.5 from previous iteration
- Mean score < 9.0

**Stop** if:
- 3 iterations completed
- Score improvement < 0.5 (plateau)
- Mean score >= 9.0 (excellent)
- All samples score >= 7.0 (no clear failures)

### Step 6: Report results

After the loop completes, present results to the user:

1. **Comparison table** — show each iteration with: prompt version, mean score, median score, improvement delta
2. **Best prompt** — identify and show the best-performing prompt using `show_file`
3. **Remaining failures** — if any, briefly describe the patterns that persist
4. **Recommendation** — suggest next steps:
   - If scores are high (>8): "Ready for deployment or fine-tuning"
   - If scores plateau (6-8): "Consider fine-tuning to improve further"
   - If scores are low (<6): "The task may need a different model size or more spec refinement"

## Next Steps

After the optimization loop, use `ask_user` to offer:

1. **"Generate training data and fine-tune"** (recommended if scores 6-8) — Scale up the existing data pipeline for a full training set (thousands of samples), then load `/train` to fine-tune.
2. **"Score is good enough, deploy as-is"** — If prompt optimization alone achieves target quality (scores > 8), no fine-tuning needed.
3. **"Continue optimizing with more iterations"** — Run additional rounds if scores are still improving.
4. **"Try a different model"** — Re-run the optimization loop with a different base model.
5. **"Edit the spec and restart"** — If persistent failures suggest spec gaps, go back to `/spec`.

## Prompt writing tips

### Structure
```markdown
You are a [role] that [task description].

## Input
You will receive [description of input format].

## Output
Respond with [exact output format description].

## Rules
- [Constraint 1 from spec]
- [Constraint 2 from spec]
- [Edge case handling]

## Examples

Input: [example input]
Output: [example output]
```

### Common refinement patterns

| Failure pattern | Fix |
|---|---|
| Ignores format | Add explicit format template with placeholders |
| Too verbose | Add word/sentence count limit |
| Misses key info | Add numbered checklist of required elements |
| Hallucinations | Add "only use information explicitly stated in the input" |
| Inconsistent quality | Add more examples (good and bad) |
| Edge case failures | Add specific handling instructions for the pattern |

## Model selection

- Use `list_models` to discover available models
- Start with the model the user specifies, or `small` as default
- The same model should be used across all iterations for fair comparison
- If the user wants to compare models, run the full optimization once per model

## Structured output

If a response format schema exists at `prompts/{task}.schema.json`, it is auto-loaded when running eval with `system_prompt_path`. You don't need to pass it explicitly — the system discovers it based on the prompt file path. This ensures the model is constrained to produce valid structured output during all eval iterations.

## What this skill does NOT do

- Does not generate training data (use `/datagen` for that)
- Does not fine-tune models (use `/train` for that)
- Does not create the eval dataset or scorer (use `/eval` for that)
- Does not modify SPEC.md (suggest the user update it if needed)
