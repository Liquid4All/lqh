# Skill: Model Evaluation

You are now in **model evaluation** mode. Your goal is to benchmark different Liquid Foundation Models (LFMs) on the project's evaluation dataset and help the user choose the best model.

**Prerequisites**: This skill expects that a validation/eval dataset and a scorer already exist (created during data generation via `/datagen`). If they don't exist, tell the user to run `/datagen` first.

## Overview

You will:
1. Discover the Liquid model catalog with `list_models`
2. Benchmark Liquid checkpoints with `eval_hf_model` (their HuggingFace ids) and, optionally, run non-Liquid frontier/pool baselines with `run_scoring` (mode=model_eval), **always passing a baseline system prompt** (see "System prompts" below)
3. Compare results and present a recommendation
4. Suggest next steps: prompt optimization or fine-tuning

**Important:** Liquid models are evaluated via the **HuggingFace inference path** (`eval_hf_model` for cloud, `start_local_eval` for a local/SSH checkpoint dir). The old `router.liquid.ai` API has been retired, so `run_scoring` mode=`model_eval` no longer accepts a Liquid model as `inference_model` — it is reserved for the pool/frontier baselines (`small`/`medium`/`large`/`orchestration`).

## System prompts for baseline eval

**Always pass a system prompt when evaluating a non-fine-tuned base model.** Small base models (1B–3B) score near zero without task instructions — that's a confused model, not a meaningful baseline.

- If `prompts/{task}_v0.md` exists, pass `system_prompt_path="prompts/{task}_v0.md"`.
- Otherwise, derive a short prompt from `SPEC.md` and pass it as `inference_system_prompt="..."`.
- A true no-prompt run is only useful as a lower-bound sanity check, never as the headline baseline.

## Scoring concepts

### Model evaluation (`mode='model_eval'`)
Strips the final assistant turn(s) from labelled eval samples, runs model inference to produce new outputs, then scores those outputs using the judge. Results go to `evals/runs/<run_name>/`.

## Rules

1. **Check prerequisites first.** Use `summary` to verify an eval dataset (with `_eval` suffix) and a scorer exist. If not, suggest `/datagen`.
   - **Evaluate on the FILTERED eval set.** A pipeline-generated eval set must be passed through `run_data_filter` (with the scorer) before you benchmark on it — otherwise baselines are measured against noise the pipeline let through. Prefer the `*_eval_filtered` dataset. If only a raw generated `*_eval` set exists, filter it first (`run_data_filter(input_path="datasets/{task}_eval/data.parquet", scorer_path="evals/scorers/{task}_v1.md", output_dataset="{task}_eval_filtered", threshold=7.0, model_size=...)`), then evaluate on the result. Skip filtering only when the eval set is human-curated.
2. **Test multiple models.** Use `list_models` to see the Liquid catalog, then benchmark at least 2-3 different checkpoints (via `eval_hf_model`) for comparison, optionally alongside a pool baseline.
3. **Use descriptive run names.** E.g., `baseline_lfm2.5_1.2b`, `baseline_small`, `baseline_medium`.
4. **After scoring, show results.** Use `read_file` on each `evals/runs/*/summary.json` and present a comparison table.

## Workflow

### Step 1: Check Prerequisites

Use `summary` to verify:
- A **filtered** eval dataset exists (e.g., `datasets/{task}_eval_filtered/data.parquet`). If only a raw `datasets/{task}_eval` exists, filter it first per Rule 1 (unless it is human-curated).
- A scorer exists (e.g., `evals/scorers/{task}_v1.md`)

If either is missing, tell the user and suggest running `/datagen` first.

### Step 2: Discover Available Models

Use `list_models` to see the Liquid model catalog (HuggingFace ids + kind) and the pool baselines. Present the options to the user.

### Step 3: Run Baselines

Benchmark 2-4 Liquid checkpoints. Start with a small model and work up. Liquid checkpoints go through `eval_hf_model` (their HuggingFace id):

```
eval_hf_model(
    repo="LiquidAI/LFM2.5-1.2B-Instruct",      # a Liquid HF id from list_models
    training_method="full",
    eval_dataset="datasets/{task}_eval_filtered",  # filtered eval set, not the raw generated one
    scorer="evals/scorers/{task}_v1.md",
    run_name="baseline_lfm2.5_1.2b",
    system_prompt_path="prompts/{task}_v0.md",
)
```

For a checkpoint that lives on a local/SSH filesystem (e.g. a fresh fine-tune), use `start_local_eval(model_path=..., dataset=..., scorer=...)` instead.

Optionally include a **non-Liquid frontier/pool baseline** via `run_scoring` (mode=`model_eval`) for context:

```
run_scoring(
    dataset="datasets/{task}_eval_filtered",
    scorer="evals/scorers/{task}_v1.md",
    mode="model_eval",
    run_name="baseline_small",
    inference_model="small",                   # pool/frontier only — NOT a Liquid id
    system_prompt_path="prompts/{task}_v0.md",  # or inference_system_prompt="..."
)
```

Run at least: 2-3 Liquid checkpoints of interest, optionally one pool baseline for context.

### Step 4: Compare and Recommend

Read each `evals/runs/*/summary.json` and present a comparison table:

| Model | Mean Score | Median | Samples Scored |
|-------|-----------|--------|----------------|
| small | 6.2 | 6.0 | 200 |
| medium | 7.8 | 8.0 | 200 |
| lfm2.5-1.2b-instruct | 5.5 | 5.0 | 200 |

Recommend the best-performing model and suggest next steps.

## Tips

- **Use the same eval set across ALL runs.** Consistency is critical for fair comparison.
- **Include pool models AND specific LFMs.** Pool models (`small`, `medium`) give a baseline; specific LFMs show which foundation model to customize.
- **Always provide a system prompt for baseline evaluation.** Without one, small base models (1B–3B) score near zero — that's a confused model, not a meaningful baseline, and the user will (rightly) be alarmed by the numbers. Use `prompts/{task}_v0.md` if it exists, otherwise derive a minimal prompt from `SPEC.md`. Reserve true no-prompt runs for lower-bound sanity checks only.

## Next Steps

After comparing model baselines, use `ask_user`:

1. **"Optimize the system prompt"** (recommended) — Load `/prompt` to iteratively refine a system prompt for the best model.
2. **"Generate training data and fine-tune"** — Scale up data generation for training, then load `/train`.
3. **"Try more models"** — Run additional baselines with different models or configurations.
4. **"I'm done for now"** — End the session.
