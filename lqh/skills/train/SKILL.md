# Skill: Training

## Overview

The training skill fine-tunes Liquid AI foundation models (LFMs) on your datasets using **supervised fine-tuning (SFT)** or **on-policy DPO** (direct preference optimization). Training runs as a background subprocess so you can continue chatting while it trains.

## Requirements

Install the optional training dependencies:

```bash
pip install lqh[train]
```

This installs `torch`, `transformers`, `trl`, and `peft`. If unavailable, training tools will show a clear error. All other lqh features work without them.

## Training Strategy: Validate → Scale → Polish

Unless the user explicitly requests a different approach, follow this three-phase strategy:

### Phase 1: Validate (Pilot SFT)
Run a small SFT training run with 200-500 samples to confirm the data produces measurable improvement. This is fast (under a minute on a single GPU) and catches data quality issues early. If the pilot shows no improvement, fix the data or pipeline before investing more compute.

### Phase 2: Scale (Larger SFT)
Once the pilot confirms improvement, scale up the training dataset to thousands of samples and run SFT again. More data generally means better results — if scaling continues to improve scores, keep generating more data. Run multiple iterations if needed: generate more data → train → evaluate → repeat.

### Phase 3: Polish (On-Policy DPO)
DPO is best suited for **fixing specific failure modes** — when the model scores well on average but has a few consistent failure cases that need correction. DPO is ~100x slower than SFT and gains are smaller, so only use it after SFT has plateaued. Use a small dataset (200-500 samples) and few iterations (2-3). Watch for overfitting: if iteration scores improve during training but the final post-training eval drops, reduce the number of iterations.

**Important:** If the user explicitly requests DPO from the start, or wants to skip SFT, follow their instructions. The above is the default recommendation, not a hard rule.

## Soft thresholds for "did training work?" (defaults — adjust to the task)

These are starting points to judge a checkpoint. They are not hard rules
— look at the baseline first, then pick the right comparison.

- **Absolute target:** ~7/10 is a good headline number for most tasks.
- **Baseline-relative judgement:**
  - baseline ≈ 1–3/10 → 6/10 is already a solid result; don't fail a
    run just because absolute is below 7.
  - baseline ≈ 4–6/10 → aim for ≥7/10.
  - baseline already ≥7/10 → aim for at least +1.0 absolute improvement.
- **Failure signal:** improvement < ~0.5 over baseline with no clear
  trajectory across runs is a failure. Stop spending compute and report
  it (in auto mode, call `exit_auto_mode("failure", ...)`).
- **DPO iterations:** default to 3–5. **Stop early on regression** —
  if iteration N+1 scores below iteration N, the previous checkpoint is
  the keeper and further iterations will likely hurt.

## Workflow

### 1. Pre-requisites

Before training, you should have:
- A **SPEC.md** defining the task
- A **training dataset** in `datasets/<name>/data.parquet` (ChatML format)
- An **eval dataset** in `datasets/<name>_eval/data.parquet`
- A **scorer** in `evals/scorers/<name>.md`
- A **baseline eval** run to compare against (via `run_scoring` with `mode=model_eval`)

### 2. Start SFT Training

Use the `start_training` tool:

```
start_training(
    type="sft",
    base_model="LiquidAI/LFM2.5-1.2B-Instruct",
    dataset="datasets/summarization_v1",
    eval_dataset="datasets/summarization_v1_eval",
    scorer="evals/scorers/summarization_v1.md",
)
```

This:
1. Writes `config.json` to `runs/<run_name>/`
2. Spawns `python -m lqh.train` as a background subprocess
3. The subprocess writes training progress to `progress.jsonl`
4. At checkpoints, the subprocess generates eval predictions
5. The main process automatically scores checkpoint predictions via the API judge
6. Scores are written to `eval_result.json` in each checkpoint directory

### 3. Monitor Progress

Use `training_status` to check on the run. It shows:
- Current step, loss, learning rate, epoch
- Whether the subprocess is alive
- Checkpoint eval scores (if eval is configured)

### 4. Evaluate the Result

After training completes, the final model is saved to `runs/<run_name>/model/`. Use `start_local_eval` to run inference with the fine-tuned model and score the results:

```
start_local_eval(
    model_path="runs/sft_001/model",
    dataset="datasets/summarization_v1_eval",
    scorer="evals/scorers/summarization_v1.md",
)
```

Compare the scores with your baseline eval to measure improvement.

### 5. On-Policy DPO (Advanced)

If SFT alone isn't enough, run on-policy DPO to further improve the model:

```
start_training(
    type="on_policy_dpo",
    base_model="runs/sft_001/model",
    dataset="datasets/summarization_v1",
    eval_dataset="datasets/summarization_v1_eval",
    scorer="evals/scorers/summarization_v1.md",
    golden_source="api",
)
```

DPO iteratively:
1. Generates model responses on the eval set
2. Scores them with the API judge
3. Gets "golden" (better) responses for low-scoring samples
4. Runs a DPO optimization step using (golden, low-scoring) pairs
5. Repeats for `num_iterations` rounds

**`golden_source`** controls where the preferred responses come from:
- `"dataset"` — uses the original assistant turn from training data (free, no API call)
- `"api"` — calls the API with a strong model to generate better responses

## Remote Training

If the local machine has no GPU (or the user wants to train on a separate box), training can run on a remote machine over SSH. The local lqh process orchestrates; the remote runs the actual subprocess.

### One-time machine setup

1. `remote_add(name=..., type="ssh_direct", hostname=...)` — register the machine globally. The hostname must be SSH-reachable (typically configured in `~/.ssh/config`).
2. `remote_bind(name=..., remote_root="~/lqh/<project basename>")` — bind the machine to the current project. Use the `~/lqh/<basename>` default without asking the user; only request a different path if they've indicated a non-default location. The handler resolves `~` to an absolute path on the remote.
3. `remote_setup(name=...)` — provisions a venv with `lqh[train]`, syncs the lqh source, and detects GPUs. Must complete before training.

### Launching a remote run

Pass `remote=<name>` to `start_training`. Everything else stays the same:

```
start_training(
    type="sft",
    base_model="LiquidAI/LFM2.5-1.2B-Instruct",
    dataset="datasets/summarization_v1",
    eval_dataset="datasets/summarization_v1_eval",
    scorer="evals/scorers/summarization_v1.md",
    remote="toka",
)
```

The launcher syncs the dataset, scorer, and config to the remote, starts the subprocess there, and returns a job ID. Use `training_status(remote=...)` to monitor — progress and checkpoint scores are pulled back to the local mirror.

The local machine does **not** need `lqh[train]` installed when training remotely.

## Training Configuration

### Hyperparameter sweeping is the default

When you call `start_training` (SFT or DPO), the harness **automatically sweeps a small grid of hyperparameters** and picks the best config using a cheap in-training proxy. You do not need to (and should not) pick `learning_rate` / `num_epochs` / `dpo_beta` values yourself, and you should NOT ask the user to confirm that you can sweep — sweeping is the default for a reason and the user expects it.

**Default grids** (6 configs each):
- SFT: `lr ∈ {2e-5, 5e-5, 1e-4} × epochs ∈ {2, 3}`
- DPO: `lr ∈ {3e-7, 1e-6, 3e-6} × β ∈ {0.05, 0.10}`

**Cost**: roughly `2–3×` a single-config training, so plan for ~2-3h on a single GPU. The cost is worth it: in the validation experiment on `ar_to_de` (2026-05-11), the swept winner beat the zero-shot default hyperparameters by +0.44 mean judge score for SFT.

### Why sweep? Why a proxy?

The fine-tuning cost structure is asymmetric:
- **Data generation** (rollout + judge) and **judge-eval-on-held-out** are expensive — hours.
- **Training itself** on a fixed dataset is cheap — minutes.

So we sweep training cheaply, pick a winner using an in-training proxy that costs nothing extra, and only then pay for one judge eval on the winner.

### The proxy

- **SFT** uses HF's `eval_loss` on a held-out 10% split. This is reliable (Pearson r = −0.90 with judge_mean, top-1 picked correctly).

- **DPO** uses `eval_ce_chosen_mean` — absolute cross-entropy of the policy on the *chosen* response in the held-out preferences. Validated with Spearman ρ = −1.0, top-1 picked correctly. The companion metric `eval_ce_chosen_delta_ref` (delta vs the frozen reference model) is monotone equivalent.

  **Why not DPO's own `eval_loss`?** Because DPO loss is a *ratio* `−log σ(β · (log P(chosen) − log P(rejected)))`. The policy can drive that ratio (and the related `eval_rewards/margins`) to look great by making *rejected* drastically less likely — even while it simultaneously makes *chosen* less likely. Generation collapses, judge score craters, but DPO eval_loss says everything is fine. This is "DPO reward hacking" (cf. Pal et al. *Smaug / DPO-Positive*). We confirmed it directly: in the validation experiment DPO eval_loss correlated with judge_mean in the **wrong direction** (Pearson r = +0.92).

  Chosen-CE is hack-resistant because the *reference* model is frozen, so the only way to make `delta_ref` look good is to actually raise `P(chosen)` — which is what we care about.

### When to opt out

Pass `enable_sweep=false` only if the user explicitly asks: "don't tune, just run with these hyperparameters", "skip the sweep", "I want a single run", etc. Under `enable_sweep=false`, the agent's `learning_rate` / `num_epochs` / `dpo_beta` arguments are honoured directly.

### Optional knobs (single-config path only)

These are read when `enable_sweep=false`:
- **`lora`** (default: true) — use LoRA for parameter-efficient fine-tuning.
- **`num_epochs`** (default: 3) — SFT only.
- **`learning_rate`** (default: 2e-5 for SFT, 5e-6 for DPO).
- **`num_iterations`** (default: 5) — DPO only.
- **`dpo_beta`** (default: 0.1) — DPO KL anchor strength.

## Directory Structure

```
runs/<run_name>/
  config.json                # sweep config (wraps base + grid spec)
  pid                        # subprocess PID
  progress.jsonl             # step-by-step metrics (sweep + per-config)
  stdout.log / stderr.log    # parent sweep subprocess
  model/                     # winner's model (symlink → sweep_<winner>/model)
  sweep_summary.json         # per-config table + winner pointer
  runs.jsonl                 # append-only per-config results
  sweep_<config_id>/         # one dir per grid point
    config.json              # single-config payload for python -m lqh.train
    progress.jsonl
    model/                   # this config's trained model
    eval_history.json        # SFT: full HF Trainer log_history (incl. eval_loss)
    iterations/iter_000/     # DPO only
      preferences.parquet
      eval_history.json
      chosen_ce_summary.json # winner-selection signal for DPO
      dpo_result.json
```

Single-config runs (`enable_sweep=false`) skip the sweep wrapper and use the same layout as before (no `sweep_*` subdirs, model directly under `runs/<run_name>/model/`).

## Agent Guidelines

When helping the user with training:

1. **Always run a baseline eval first** — before training, run `run_scoring` with `mode=model_eval` on the base model to establish a score baseline.

2. **Do NOT ask the user whether to hyperparameter-tune.** Sweeping is the default and the user expects it. Just kick off the run. If sweeping will surprise the user (e.g. they expected a fast single-config run), inform them in one sentence after starting: *"I'm running a 6-config sweep — this will pick the best hyperparameters automatically. Use `enable_sweep=false` next time if you'd prefer a single config."* Do **not** gate the run on confirmation.

3. **Only pass `enable_sweep=false` when the user explicitly opts out.** Phrases that count as opt-out: "don't tune", "skip the sweep", "just one run", "use these hyperparameters", or any concrete `learning_rate=…` value attached to "just this once". When `enable_sweep=false`, you may pass specific `learning_rate` / `num_epochs` / `dpo_beta` values.

4. **Follow the validate → scale → polish strategy** — unless the user explicitly requests otherwise:
   - Start with a pilot SFT run (200-500 samples) to confirm improvement.
   - Scale up the dataset and run SFT again if the pilot succeeds.
   - Only suggest DPO after SFT has plateaued, and frame it as polishing specific failure cases.

5. **Check dataset quality** — before training, verify the training data quality with `run_scoring` in `data_quality` mode. Low-quality training data = low-quality fine-tuned model.

6. **Use `training_status` proactively** — after starting a run, periodically check status. The sweep table in `training_status` shows per-config results with the validated proxy (`eval_loss` for SFT, `eval_ce_chosen_mean` for DPO). It is intentional that DPO `eval_loss` and `eval_rewards/margins` are NOT shown — those metrics would mislead you (they can look great when the model has actually collapsed). Trust the sweep's chosen winner.

7. **Suggest next steps** — after a sweep completes:
   - Run local eval to compare the winner with baseline.
   - If scores improved and more data is available, suggest scaling up (more samples → retrain).
   - If scores plateaued with sufficient data, suggest DPO to polish specific failure cases.
   - If every DPO config in the sweep collapsed (`sweep_summary.json` winner is null), the preference set may have no useful signal for the current model — suggest either better preference filtering, smaller preference quantile, or skipping DPO.
   - If the model is ready, suggest pushing to HF Hub.

8. **Handle errors gracefully** — if training fails (CUDA OOM, etc.), read `stderr.log` (or `sweep_<config>/stderr.log` for a specific config) and suggest fixes (lower batch size, enable gradient checkpointing, etc.).

9. **Respect user preferences** — if the user wants to start with DPO, skip the pilot, or use a different strategy, follow their instructions. The validate → scale → polish strategy is a default recommendation, not a requirement.
