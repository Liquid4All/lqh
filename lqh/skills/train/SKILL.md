# Skill: Training

## Overview

The training skill fine-tunes Liquid AI foundation models (LFMs) on your datasets using **supervised fine-tuning (SFT)** or **on-policy DPO** (direct preference optimization). Training runs as a background subprocess so you can continue chatting while it trains.

## The default process (follow it; warn on skip)

Unless the user directs otherwise, follow this order by default:

1. **Confirm data + scorer quality first** — you have read the demo data and the
   scorer's outputs yourself and both are good (else iterate in `/datagen`).
2. **Filter** the training and eval data with the scorer (never train on raw data).
3. **Build + filter a held-out eval (validation) set** (a few hundred samples) that
   is used only for evaluation, never for training.
4. **Zero-shot baseline eval** of the base model on the eval set — **with a
   well-structured system prompt** (see below). This is your comparison point.
5. **Pilot SFT** on a small training set (a few hundred–low thousands) at **default
   hyperparameters, no sweep**, just to confirm the direction is right.
6. **Evaluate the pilot** vs baseline, then decide: improvement → scale; degradation
   → sweep, else improve data/scorer; unchanged → scale with caution.
7. **Scaled SFT** (tens of thousands of samples) **with the hyperparameter sweep**.
8. **Evaluate + score** the best SFT checkpoint; scale further or fix data as the
   trajectory dictates.
9. **DPO** only once SFT is stuck/plateaued (or high average with outlier failures):
   base = best SFT checkpoint, **sweep directly**. Then compare best DPO vs best SFT
   vs baseline.

If the user explicitly asks to skip a step or jump ahead, do so — but **warn in one
sentence** that it deviates from the ideal process and may give suboptimal results.

## Requirements

Install the optional training dependencies:

```bash
pip install lqh[train]
```

This installs `torch`, `transformers`, `trl`, and `peft`. If unavailable, training tools will show a clear error. All other lqh features work without them.

## Training Strategy: Validate → Scale → Polish

Unless the user explicitly requests a different approach, follow this three-phase strategy:

### Phase 1: Validate (Pilot SFT)
Run a small SFT training run with 200-500 samples to confirm the data produces measurable improvement. **Run the pilot at default hyperparameters with `enable_sweep=false` — no sweep at this stage.** Its only job is to check the direction is right, and a sweep would pay 2-3x cost for a signal you don't need yet. This is fast (under a minute on a single GPU) and catches data quality issues early. If the pilot shows no improvement, fix the data or pipeline before investing more compute. (Don't conclude too much from a few-hundred-sample pilot either — see the decision tree below.)

### Phase 2: Scale (Larger SFT)
Once the pilot confirms improvement, scale up the training dataset to **thousands → tens of thousands** of samples and run SFT again — **this time with the hyperparameter sweep on (the default)**. More data generally means better results — if scaling continues to improve scores, keep generating more data. Run multiple iterations if needed: generate more data → train → evaluate → repeat. Expect to need at least a few thousand high-quality samples before a meaningful gain appears.

### Phase 3: Polish (On-Policy DPO)
DPO is best suited for **fixing specific failure modes** — when the model scores well on average but has a few consistent failure cases that need correction. DPO is ~100x slower than SFT and gains are smaller, so only use it after SFT has plateaued or is stuck. The **base model for DPO is your best SFT checkpoint**, and because DPO is very hyperparameter-sensitive (and you already have a large SFT dataset by this point) it **sweeps directly — no default-hyperparameter first run, unlike the SFT pilot**. Use a small dataset (200-500 samples) and few iterations (2-3). Watch for overfitting: if iteration scores improve during training but the final post-training eval drops, reduce the number of iterations. Finally, compare best DPO vs best SFT vs baseline.

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

## After evaluating a checkpoint — decide the next move

Every checkpoint score is read **relative to the baseline** (and to the previous
run). Based on the comparison:

- **Improvement / right direction** → **scale the dataset** (pilot's low thousands
  → tens of thousands) and train again. After the scaled run, if it keeps
  improving, scale further; this is the main lever.
- **Degradation** → first **run the hyperparameter sweep** (if this was the
  default-config pilot, the scaled run already sweeps; for a swept run, the sweep
  has already tried the grid). If tuning still doesn't recover it, the bottleneck
  is **data + scorer quality** — go back to `/datagen`, improve the pipeline and
  scorer, re-filter, and retrain. Don't keep retraining the same bad data.
- **Unchanged (flat vs baseline)** → you *may* scale, but **with caution**: a flat
  result can signal under-fitting (too little data/too few epochs — scaling helps)
  or over-fitting / a data ceiling (scaling won't help). Look at train vs eval loss
  and the score distribution before committing more compute.

**Don't over- or under-react to the pilot.** A few-hundred-sample pilot gives a
*direction*, not a final verdict — you typically need at least a few thousand
high-quality samples before a real gain appears. Equally, don't jump straight from
a tiny pilot to a 20k run without reading the pilot signal first.

## Workflow

### 1. Pre-requisites

Before training, you should have:
- **Confirmed data + scorer quality yourself.** Before *any* training run, you must
  have read the demo/example samples AND the scorer's outputs (the `scores.parquet`
  reasoning + scores) with your own eyes and judged both to be good. Training on
  data or a scorer you have not inspected is the #1 way to waste compute. If either
  is poor, go back to the `data_generation` skill and iterate (its Phase 2 Steps
  2.4–2.5 cover testing and fixing the scorer) — do not train to "see what happens".
- A **SPEC.md** defining the task
- A **scorer** in `evals/scorers/<name>.md`
- A **filtered training dataset** in `datasets/<name>_train_filtered/data.parquet`
  (ChatML format). **Pipeline-generated data must be passed through
  `run_data_filter` with the scorer before training — never train on the raw
  generated set.** Filtering is skipped only when the data is human-curated.
- A **filtered eval dataset** in `datasets/<name>_eval_filtered/data.parquet`
  (same rule; skip filtering only for a human-curated eval set)
- A **baseline eval** run to compare against (via `run_scoring` with `mode=model_eval`)

If only raw generated datasets exist, filter them first (see the
`data_generation` skill, Phase 3.5):
```
run_data_filter(
    input_path="datasets/<name>_train/data.parquet",
    scorer_path="evals/scorers/<name>.md",
    output_dataset="<name>_train_filtered",
    threshold=7.0,
    model_size="small",   # the judge size validated during data gen
)
```

### 2. Start SFT Training

Use the `start_training` tool. The **pilot** runs at default hyperparameters with
`enable_sweep=false`:

```
start_training(
    type="sft",
    base_model="LiquidAI/LFM2.5-1.2B-Instruct",
    dataset="datasets/summarization_v1_train_filtered",
    eval_dataset="datasets/summarization_v1_eval_filtered",
    scorer="evals/scorers/summarization_v1.md",
    enable_sweep=false,   # pilot: validate direction at default hyperparameters
)
```

For the **scaled** SFT run (Phase 2), drop `enable_sweep=false` so the sweep runs.

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
    dataset="datasets/summarization_v1_eval_filtered",
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
    dataset="datasets/summarization_v1_train_filtered",
    eval_dataset="datasets/summarization_v1_eval_filtered",
    scorer="evals/scorers/summarization_v1.md",
    golden_source="api",
)
```

DPO iteratively:
1. Generates model responses on the **training** prompts (`dataset`)
2. Scores them with the API judge
3. Gets "golden" (better) responses for low-scoring samples
4. Runs a DPO optimization step using (golden, low-scoring) pairs
5. Repeats for `num_iterations` rounds

`dataset` vs `eval_dataset` are strictly separated, same as SFT: DPO builds its
preference pairs from rollouts on `dataset` (training prompts), and the best
checkpoint is judge-scored on the held-out `eval_dataset` (unseen prompts).
`eval_dataset` never feeds the DPO loop. (Note: the DPO sweep selects its winner
on a held-out split of the *preferences* — `eval_ce_chosen_mean` — not on
`eval_dataset` directly; `eval_dataset` is the final judge eval-of-best set.)

**`golden_source`** controls where the preferred responses come from:
- `"dataset"` — uses the original assistant turn from training data (free, no API call)
- `"api"` — calls the API with a strong model to generate better responses

## Where training runs (compute target)

**The compute target is fixed per project — you never pass it.** Just call `start_training` / `start_local_eval` with no compute or remote argument; the harness routes automatically:

- Cloud-only project (no bring-your-own-compute remote, no local GPU) → runs on LQH Cloud silently.
- A project that has a real choice (a configured BYOC remote and/or a local GPU) and hasn't pinned a target yet → the first `start_training`/`start_local_eval` triggers a one-time system picker (LQH Cloud / Local (this machine) / each remote). The user's pick is persisted to the project and reused automatically. Do not ask the user where to run; the picker handles it.

Never pass a `remote=` (or similar) argument — those tools no longer accept one, and a wrong value is exactly the kind of mistake the per-project pin exists to prevent.

### Setting up a bring-your-own-compute (SSH) machine

To make a user's own GPU box available as a target, walk them through the one-time setup, then let the picker route to it:

1. `remote_add(name=..., type="ssh_direct", hostname=...)` — register the machine globally. The hostname must be SSH-reachable (typically configured in `~/.ssh/config`).
2. `remote_bind(name=..., remote_root="~/lqh/<project basename>")` — bind the machine to the current project. Use the `~/lqh/<basename>` default without asking the user; only request a different path if they've indicated a non-default location. The handler resolves `~` to an absolute path on the remote.
3. `remote_setup(name=...)` — provisions a venv with `lqh[train]`, syncs the lqh source, and detects GPUs. Must complete before training.

After setup, the next `start_training` offers the new remote in the picker. The launcher then syncs the dataset, scorer, and config to it, starts the subprocess there, and returns a job ID. Use `training_status(run_name=...)` to monitor — progress and checkpoint scores are pulled back to the local mirror. The local machine does **not** need `lqh[train]` installed when training on a remote. To change a project's pinned target later, use `compute_set`.

## Training Configuration

### Hyperparameter sweeping (default for scaled SFT and DPO; OFF for the pilot)

When you call `start_training`, the harness **automatically sweeps a small grid of hyperparameters** and picks the best config using a cheap in-training proxy. This is the default for the **scaled SFT run (Phase 2)** and for **all DPO runs (Phase 3)** — you do not pick `learning_rate` / `num_epochs` / `dpo_beta` yourself, and you should NOT ask the user to confirm sweeping.

**The one exception is the pilot SFT (Phase 1): run it with `enable_sweep=false` at default hyperparameters.** The pilot only needs to confirm direction, so paying the 2-3x sweep cost there is wasteful. Sweep once you scale.

In all cases, **do not ask the user about hyperparameters** — follow this ideal setup (pilot off, scale/DPO on) automatically unless they explicitly instruct otherwise.

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

### When `enable_sweep=false` applies

Two cases: (1) **the pilot SFT**, where it is the default (validate direction at
default hyperparameters), and (2) when the user explicitly opts out of the sweep
for a scaled/DPO run: "don't tune, just run with these hyperparameters", "skip the
sweep", "I want a single run", etc. Under `enable_sweep=false`, the agent's
`learning_rate` / `num_epochs` / `dpo_beta` arguments are honoured directly.

### Optional knobs (single-config path only)

These are read when `enable_sweep=false`:
- **`lora`** (default: true) — use LoRA for parameter-efficient fine-tuning.
- **`num_epochs`** (default: 3) — SFT only.
- **`learning_rate`** (default: 2e-5 for SFT, 5e-6 for DPO).
- **`num_iterations`** (default: 5) — DPO only.
- **`dpo_beta`** (default: 0.1) — DPO KL anchor strength.

### Combining multiple data sources

Both `dataset` and `eval_dataset` accept **either a single path or a list** of dataset dirs, so you can train on several files at once instead of picking just one. **Every source you list must still be a filtered/scored dataset** — the same rule as a single-source run: pipeline-generated data goes through `run_data_filter` first, and you only train on the `*_filtered` outputs you have inspected. Never list a raw, unfiltered generated file (human-curated data is the only exception).

There are two distinct reasons to use a list — and they call for different `repeat` settings:

**1. Use ALL the good data you have — `repeat: 1` for every source.** When you accumulate more data over time (scale-up: a first 2k-sample file, then a 10k-sample file, then 20k), do **not** throw the smaller earlier files away and train only on the newest one. Train on **everything** — more good data is better. These are the *same kind* of data, just collected in batches, so each source gets the default `repeat: 1` (plain concatenation, no over-sampling):

```python
start_training(
    type="sft",
    base_model="LiquidAI/LFM2.5-1.2B-Instruct",
    # Scale-up: same task, collected in batches → train on all of it, no repeats.
    dataset=[
        "datasets/summarization_v1_train_filtered",       # earlier 2k batch
        "datasets/summarization_v2_train_filtered",        # later 10k batch
    ],
    eval_dataset="datasets/summarization_eval_filtered",
    scorer="evals/scorers/summarization.md",
)
```

(A bare string and a list of strings both default to `repeat: 1` — `dataset=["a", "b"]` is plain concatenation.)

**2. Balance imbalanced data *types* — use `repeat` to over-sample the smaller type.** When the list mixes **different kinds** of data (e.g. type-A requests and type-B requests from separate pipelines) and they're imbalanced — say 2k type-A vs 10k type-B — left as-is the model sees ~5× more type-B per batch and under-learns type-A. Set `repeat` on the smaller type to bring the mix closer to balanced (here `repeat: 5` makes type-A ≈ type-B in the blend):

```python
start_training(
    type="sft",
    base_model="LiquidAI/LFM2.5-1.2B-Instruct",
    # Different types, imbalanced → repeat the smaller type to balance the mix.
    dataset=[
        {"path": "datasets/type_a_train_filtered", "repeat": 5},   # ~2k samples
        "datasets/type_b_train_filtered",                           # ~10k samples (repeat 1)
    ],
    eval_dataset=["datasets/type_a_eval_filtered", "datasets/type_b_eval_filtered"],
    scorer="evals/scorers/default.md",
)
```

Other notes:

- **Mixing in HuggingFace datasets** — `pull`/`hf_pull` the hub dataset first (it lands at `datasets/<repo>/`), filter/inspect it like any other source, then list it alongside your own dirs.
- **`repeat` is integer over-sampling** of the *same* rows (it changes the per-batch source ratio; it is orthogonal to `num_epochs`, which controls passes over the whole blend). Only reach for it to correct a type imbalance — for "use all my data," leave it at 1. DPO concatenates its prompt sources the same way.
- **Eval (`eval_dataset`) sources are kept SEPARATE** (never repeated): the best checkpoint is judge-scored on each source independently, and `training_status` / `eval_result.json` show a **per-source breakdown** plus a **macro-average** headline (each source weighted equally, regardless of size). The sweep still selects its winner on the in-training `eval_loss` over the concatenation of all eval sources. List eval sources when you want to see how the model does on each sub-task; `repeat` is rejected for eval.
- Every eval source must be **distinct** from every training source (the call is rejected on overlap).

### `eval_dataset` is required; scoring is on by default

**`eval_dataset` is mandatory.** `start_training` rejects the call without it. It is the held-out set the sweep selects the winner on (for SFT this is the in-training `eval_loss`; for DPO the proxy is a preference split) **and** the set the best checkpoint is judge-scored on. The proxy only *selects* the winner — it is not the result you report to the user.

**Pass `scorer` by default — set it to the project's default or currently-best scorer** (typically the one under `evals/scorers/` you used for the baseline eval). This is what makes a run produce a **real judge score** on the best checkpoint.

- **Scoring must be an explicit decision.** `start_training` rejects the call unless you either pass `scorer` or set `disable_scoring=true`. There is no silent "no scorer" path anymore — a missing judge score is always deliberate.
- **Only set `disable_scoring=true` if the user explicitly says not to score** — "don't score it", "skip the eval", "just train, no scoring". This is the exception, and it is **SFT-only**.
- **DPO always requires a scorer — `disable_scoring` is rejected for DPO.** On-policy DPO builds its preference pairs by judge-scoring generated rollouts every iteration, so without a scorer the method cannot run at all (it's not just the final eval, as with SFT). For DPO you must always pass `scorer`.
- **Without a scorer (SFT), eval-of-best degrades to proxy-only** — you get no judge number, only the val_loss proxy. That's why scoring is opt-out, not opt-in.
- On LQH Cloud the judge scoring runs **inside the sandbox** with a scoped token, so it completes even if the user closes their laptop. The score is uploaded as an artifact and is available on reconnect — the sandbox does not need to still be alive to read it.

### Fetch the judge score after the run

A finished run does **not** push the judge score into your context automatically — you must fetch it. Once the run reaches a terminal state (including when reconnecting after the laptop was closed during a long sweep + eval):

1. Call `training_status(run_name=...)` — the sweep table surfaces the per-config proxy and the winner.
2. Read the eval-of-best **judge score** from the run artifacts (`eval_result.json`, or `sweep_summary.json`'s `eval_of_best`). These are pulled from the artifact store, so this works on reconnect.
3. Report the **judge score vs. baseline** — not the val_loss proxy — when telling the user whether training worked.

If you passed `scorer` + `eval_dataset`, the eval-of-best already ran as part of the run, so you generally do **not** need a separate `start_local_eval` to get the winner's score. Only run one to evaluate a *different* model or held-out set.

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

1. **Always run a baseline eval first — with a well-structured system prompt.**
   Before training, run `run_scoring` with `mode=model_eval` on the base model to
   establish a score baseline. **VERY IMPORTANT: always pass a system prompt for
   this baseline.** A small base model with no system prompt does not know what the
   task is, produces confused output, and scores near zero — a meaningless baseline
   that then makes any trained model look falsely amazing. Forgetting the system
   prompt here is one of the most common and most damaging mistakes. The system
   prompt should be **well-structured**: clear task instructions, the expected
   output format, and ideally one or two short examples. Pass
   `system_prompt_path="prompts/{task}_v0.md"` if it exists; otherwise derive a
   minimal-but-complete prompt from `SPEC.md`, write it to `prompts/{task}_v0.md`,
   and pass it. A true no-prompt run is only a lower-bound sanity check, never the
   headline baseline. (See the `evaluation` skill's "System prompts for baseline
   eval" for the same rule.)

2. **Always pass `eval_dataset` and a `scorer`** — `eval_dataset` is required (the run is rejected without it). Set `scorer` to the project's default/best scorer (the one under `evals/scorers/` used for the baseline eval) so the run produces a real judge score, not just the internal proxy. The run is rejected unless you pass `scorer` or set `disable_scoring=true` — **set `disable_scoring=true` only when the user explicitly asks not to score** ("don't score it", "skip the eval"). See *`eval_dataset` is required; scoring is on by default*.

3. **Fetch the judge score after the run** — a finished run does not push the score into your context. Once the run is terminal (including on reconnect after a closed laptop), call `training_status(run_name=...)` and read the eval-of-best judge score from the run artifacts (`eval_result.json` / `sweep_summary.json` `eval_of_best`), then report **judge score vs. baseline**. See *Fetch the judge score after the run*.

4. **Do NOT ask the user whether to hyperparameter-tune.** Follow the default
   automatically: **pilot SFT off (`enable_sweep=false`), scaled SFT and DPO on.**
   Just kick off the run. When the scaled/DPO sweep might surprise the user, inform
   them in one sentence after starting: *"I'm running a 6-config sweep — this will
   pick the best hyperparameters automatically."* Do **not** gate the run on
   confirmation.

5. **Sweep on/off follows the stage by default — the user can override either way.**
   The pilot uses `enable_sweep=false`; the scaled SFT run and DPO sweep. Honor an
   explicit override: "don't tune" / "skip the sweep" / "just one run" / a concrete
   `learning_rate=…` "just this once" → `enable_sweep=false` (and you may then pass
   specific `learning_rate` / `num_epochs` / `dpo_beta`); "tune the pilot too" →
   sweep the pilot.

6. **Follow the validate → scale → polish strategy** — unless the user explicitly requests otherwise:
   - Start with a pilot SFT run (200-500 samples, `enable_sweep=false`, default hyperparameters) to confirm improvement.
   - If the pilot succeeds, scale up the dataset (toward tens of thousands) and run SFT again **with the sweep on**.
   - Only suggest DPO after SFT has plateaued/is stuck; base it on the best SFT checkpoint, sweep it directly, and frame it as polishing specific failure cases.
   - Use the "After evaluating a checkpoint" decision tree to choose scale vs. tune vs. fix-data at each step.

7. **Filter the data before training** — pipeline-generated training (and eval) data must be passed through `run_data_filter` with the scorer before it is used, and training must point at the `*_filtered` dataset. Filtering both checks quality and removes the low-scoring tail; low-quality training data = low-quality fine-tuned model. Skip only for human-curated data or when the user explicitly opts out.

8. **Use `training_status` proactively** — after starting a run, periodically check status. The sweep table in `training_status` shows per-config results with the validated proxy (`eval_loss` for SFT, `eval_ce_chosen_mean` for DPO). It is intentional that DPO `eval_loss` and `eval_rewards/margins` are NOT shown — those metrics would mislead you (they can look great when the model has actually collapsed). Trust the sweep's chosen winner.

9. **Suggest next steps** — after a sweep completes:
   - Run local eval to compare the winner with baseline.
   - If scores improved and more data is available, suggest scaling up (more samples → retrain).
   - If scores plateaued with sufficient data, suggest DPO to polish specific failure cases.
   - If every DPO config in the sweep collapsed (`sweep_summary.json` winner is null), the preference set may have no useful signal for the current model — suggest either better preference filtering, smaller preference quantile, or skipping DPO.
   - If the model is ready, suggest pushing to HF Hub.

10. **Handle errors gracefully** — if training fails (CUDA OOM, etc.), read `stderr.log` (or `sweep_<config>/stderr.log` for a specific config) and suggest fixes (lower batch size, enable gradient checkpointing, etc.).

11. **Respect user preferences** — if the user wants to start with DPO, skip the pilot, or use a different strategy, follow their instructions (with a one-sentence warning that it deviates from the ideal process). The validate → scale → polish strategy is a default recommendation, not a requirement.

## Common Failure Modes and Issues

Watch for these — they are the recurring ways a training effort goes wrong:

- **Training before inspecting data + scorer quality.** Kicking off SFT without
  having read the demo samples and the scorer's outputs yourself leads to poor
  models and wasted compute. Confirm both are good first (see Pre-requisites).
- **Forgetting the system prompt in the zero-shot baseline.** A base model with no
  system prompt is confused and scores near zero — a meaningless baseline that
  makes later runs look falsely great and hides real improvement. Always pass a
  well-structured system prompt (instructions + output format + ideally examples).
- **Staying at a tiny dataset, or scaling blindly.** Both extremes fail: a
  few-hundred-sample run is only a direction check (expect a real gain only past a
  few thousand high-quality samples), but jumping straight to 20k without reading
  the pilot signal wastes compute on an unvalidated direction.
- **Training on raw, unfiltered data.** Always filter the training and eval sets
  with the scorer first; the low-scoring tail teaches the model the wrong thing.
- **Skipping hyperparameter tuning at scale.** The pilot is fine at defaults, but
  the scaled SFT and DPO runs should sweep — leaving them unswept commonly leaves
  a real improvement on the table.
- **Jumping straight to DPO.** DPO is a polish step. Reaching for it before a
  thorough SFT (filtered data → scale → sweep) usually disappoints; DPO's gains are
  small and it is hyperparameter-sensitive.
- **Base model too small for the task.** A 350M model cannot learn a task that
  genuinely needs a model orders of magnitude larger, no matter how good the data.
  If a model scores poorly even after training on a large, high-quality, filtered
  dataset, try a **larger base model** before assuming the data is the problem.
