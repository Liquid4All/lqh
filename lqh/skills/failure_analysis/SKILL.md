# Skill: Failure Analysis & Post-Eval Improvement

You are now in **failure analysis** mode. Your goal is to decide what to do
next after a post-training evaluation — and, when the quantitative levers are
exhausted, to run a qualitative analysis of the failure cases and turn the
findings into targeted training data.

**When to load this skill:** after ANY post-training eval completes (SFT or
DPO), or whenever you are unsure what the right next step is given an eval
score. A score alone ("5.6/10") means nothing without context — this skill
tells you how to read it and what to do about it.

## Step 0 — Assemble the decision context

Never route on the score alone. Collect all of these first:

| Fact | Where to find it |
|---|---|
| Latest score + distribution | `training_status(run_name=...)` — the "Final eval" block (mean, percentiles, histogram); raw JSON in `runs/<name>/eval_result.json` |
| How the run trained | `training_status` — the "Training health" line: optimizer steps, train-loss start → end, eval loss, token accuracy, learning rate. A flat score with a flat loss is a *training* problem; a flat score with a falling loss is a data or capacity problem. |
| Zero-shot baseline | The pre-training eval run of the same base model on the same eval set (`evals/runs/*/summary.json` or `runs/<baseline>/eval_result.json`). Did SFT actually improve the model? |
| Model size | `runs/<name>/config.json` → `base_model`. Text ladder: 230M / 350M / 1.2B / 2.6B / 8B-A1B; VLM ladder: 450M / 1.6B (see `list_models`). |
| Training-set size | `training_status` "Data:" line, or `runs/<name>/config.json` → `dataset_rows` (`train`, `train_effective`, `eval`). Older runs: use `summary` to read dataset row counts. |
| Inference budget | `SPEC.md` → `## Inference Budget` (`**Budget**:` line). If the section is missing, treat as `auto` and confirm with the user before stepping far up the ladder. |

If no zero-shot baseline exists for this eval set, run one now (`eval_hf_model`
on the base model) — every decision below is relative to it.

**Δ is only meaningful under identical eval conditions.** Before comparing,
check the baseline and post-training runs' `config.json`/`summary.json` agree
on ALL of: eval dataset, scorer file (same version — `spec_sha256`/scorer
path), judge size, system prompt, response-format schema, and model revision.
Decoding is greedy everywhere in lqh, so it matches by default. If any of
these differ, the comparison is invalid — re-run the baseline under the
post-training run's exact conditions before routing on Δ.

**VLM tasks (LFM2.5-VL) differ from the text playbook.** The ladder is
450M → 1.6B (450M is the "small model" in the routing below, not 350M);
**DPO is not supported for VLMs** — skip that lever and go from data/model
scaling straight to the qualitative loop; **GGUF export is not supported for
VLMs** — at ≥8 offer deploy-as-API only.

## Outcome: read Δ first, absolute score second

Two numbers decide the route, in this order:

**1. Δ = post-training mean − zero-shot baseline. Δ picks the routing branch.**

- **Δ ≥ ~0.5 — training works.** Go to "Routing: training works" below,
  *even if the absolute score is still low*. A 1.0 → 3.8 run is not a broken
  model — it is a working trend that needs scaling, not a rebuild.
- **Δ < ~0.5, or a regression (Δ < 0)** — training isn't moving the model.
  Go to "Routing: training isn't working". On a regression the previous
  checkpoint (or the baseline model) is the current keeper — never discard it.

**2. The absolute score decides what "done" and "urgent" mean:**

- **≥ 8/10 AND no regression vs baseline** — the model is good. Use
  `ask_user` to offer:
  1. **Deploy as API** — `push_to_production` + `create_inference_key`.
  2. **Export for on-device** — `gguf_convert` (quantized GGUF, optional HF
     push). Text models only — GGUF is not supported for VLMs.
  3. **Keep improving** — continue with this skill (scale, DPO, qualitative loop).
  A high absolute score does NOT excuse a regression: if the baseline was
  already higher (e.g. 8.6 zero-shot → 8.0 post-SFT), the checkpoint is
  strictly worse than not training — route as "training isn't working" and
  do not offer deploying it.
- **7–8/10** — near target. Usually one more lever closes it (a modest data
  scale-up, or DPO polish for consistent failure modes). Whether 7.x already
  clears the bar is task-relative — use the train skill's baseline-relative
  thresholds; when it does, treat it like the ≥8 case and offer deployment.
- **< 4/10** — urgent regardless of branch: before spending compute, also
  sanity-check the eval itself (scorer quality, system prompt present, right
  checkpoint evaluated). Then follow the branch Δ picked — a low-but-improving
  run still routes to "training works" (keep scaling), not to a rebuild.

## Routing: training isn't working (Δ < ~0.5 or regression)

**This branch is ordered by cost, cheapest first.** A rerun at a different
learning rate is ~40 min on the same GPU; scaling the data adds a data-gen
pipeline run plus a longer training run; a model step-up moves to a bigger,
more expensive GPU. Spending the expensive levers first is what turns a
two-attempt fix into a six-attempt one — check these in order:

1. **Did the model actually train?** Read the "Training health" line from
   `training_status` before anything else — it is free. Two distinct failures
   live here and they have *different* fixes:
   - **Too few optimizer updates** (the line warns below ~50): the run barely
     stepped, so nothing about the data or the learning rate is testable yet.
     The effective batch is derived from `rows × epochs`, so the fix is more
     of either: `start_training(..., num_epochs=<2× the run's epochs>)` on the
     same data, or more rows. **A higher learning rate does not create
     updates** — do not reach for it here.
   - **Enough updates but the train loss barely moved** (e.g. 3.7 → 3.6):
     the adapter isn't learning at the configured rate. Retrain the *same data
     on the same model* at **5× the learning rate**
     (`start_training(..., learning_rate=<5 × the run's lr>)` — read the run's
     lr from the same line; if the line shows no `lr`, do **not** take it from
     the run-root `config.json` on a swept run, that is the pre-grid value —
     read `runs/<name>/sweep_<winner_config_id>/config.json`).
   - Both cases are the one place where you override a hyperparameter yourself
     without asking: a run that didn't learn is a broken run, not a tuning
     opportunity.
   - **Loss fell steadily but the score didn't move**: the run trained fine
     and the problem is downstream — continue to 2.
2. **Training-set size < 2k rows?** Fix this next. 2–5k good samples should
   already produce a measurable improvement; below ~2k, no other conclusion
   is safe. Scale the data-gen pipeline to ≥ 2k (reuse it with a higher
   `num_samples`), retrain, re-eval.
3. **≥ 2k rows but the model is at the small end of its ladder** (≤ 350M
   text, 450M VLM)? Very small models lack capacity for complex tasks.
   Step up one size (within budget — see below): zero-shot the larger
   model first to gauge headroom, then train it on the same dataset.
4. **Large model still flat?** The dataset itself is the suspect: not
   diverse enough, or not representative of the task. Go back to
   `/datagen` — review sample diversity, scorer quality, and the dataset's
   own score distribution (the data-generation skill's distribution-shape
   guidance applies to the *dataset*; this skill applies the same reading
   to the *model's* eval scores).

**After two consecutive Δ ≈ 0 runs, stop advancing one lever per round.**
Single-variable isolation is the right default for the *first* retry — it is
what makes the result interpretable — but each round costs the user 40–70
minutes of wall clock and a decision. Once two rounds have bought nothing,
`ask_user` with a **combined jump** as the recommended option (e.g. 5× learning
rate *and* 3× the data *and* the next model size up), and single-variable
isolation as the alternative for a user who wants to know which lever mattered.
Say what each option costs in time and money.

**Grad norm is not a diagnostic here.** Small `grad_norm` values (0.02, 0.008)
appear in both healthy and flat runs — a successful +1.30 run logged a *lower*
grad norm than the two flat runs it replaced. Read the train-loss trajectory and
`mean_token_accuracy` against the optimizer-step count instead.

## Routing: training works, but the score isn't there yet (Δ ≥ ~0.5)

Pull these levers in order — re-eval after each and stop when the score
clears the deployment bar above:

1. **Scale the training data** while scaling keeps paying. If 2k samples
   moved the score, go to 10k–20k with the same pipeline and compare. Stop
   scaling when a 2–5× data increase moves the mean by < ~0.3.
2. **Step up the model size** (within budget). Same dataset, next size up.
3. **Sweep the SFT hyperparameters** — `start_training(..., enable_sweep=true)`
   on the settled dataset and model size. **This is the first lever that is
   purely about tuning, and it belongs here, not earlier.** SFT runs at the
   shipped defaults (`lqh/train/defaults.py`), so a sweep is chasing what
   those defaults left on the table: expect a fraction of a point, for
   several times the wall-clock of one run. It only makes sense once levers
   1 and 2 have stopped paying — a sweep cannot rescue a run that is short
   on data or on model capacity. (A run that isn't learning at all is the
   other branch's step 1, and a single rerun there, not a sweep.)
4. **On-policy DPO** on the best SFT checkpoint (text models only — VLMs
   skip this lever). DPO polishes consistent failure modes; it is
   task-dependent and much slower than SFT — follow the train skill's DPO
   guidance (small pair sets, held-out trajectory; DPO sweeps by default).
5. **All of the above exhausted?** Run the qualitative failure-analysis
   loop below — quantitative scaling can only get you so far.

## Inference budget compliance

`SPEC.md` → `## Inference Budget` constrains every model-size decision:

- **`auto`** (default): explore freely. Start non-extreme (per `list_models`
  guidance) and recommend the smallest size that reaches the target.
- **`pinned:<model>`**: never train or recommend a different model without
  asking the user first.
- **`max:<size>`** (e.g. `max:1.2B`): stay at or below the cap.

Only when ALL in-budget levers (data scale, DPO, targeted data from the loop
below) are exhausted and the model is still not good enough: use `ask_user`
to suggest exceeding the budget — present the expected gain (e.g. the
zero-shot score of the next size up) and let the user decide. Never silently
train past a pin or cap — `start_training` enforces this and rejects an
over-budget base model unless you pass `override_budget=true`, which you may
do only after the user's explicit yes.

## Qualitative failure-analysis loop

When quantitative levers are exhausted (or the same failure keeps surviving
retraining), analyze *what* the model gets wrong, not just *how much*.

### 1. Generate a probe dataset

Run the existing data-gen pipeline into a NEW dataset with
`purpose="probe"`, e.g. `datasets/{task}_probe_v1`, 100–300 samples,
filtered as usual. A separate set is non-negotiable:

- The **training set** is what the model was trained on — its failures are
  not representative (not i.i.d. for this model).
- The **validation set** must stay untouched as the selection signal —
  mining it for training data is leakage.

### 2. Run the model on the probe set and read the distribution

Evaluate the current best checkpoint on the probe set (`eval_hf_model` /
`start_local_eval`), then read the score distribution from `training_status`
(or `eval_result.json` → `score_distribution`). The shape decides the lens:

- **Extreme low outliers among mostly-good scores** (e.g. p50=8 but p10=2):
  specific inputs or instructions trigger hard failure. Hunt the outliers:
  ```
  get_eval_failures(eval_run="runs/{probe_run}", score_max=3, sort="asc",
                    limit=20, max_chars_per_message=2000)
  ```
  Look for what the failing *inputs* have in common: a topic, a format, a
  length, an instruction phrasing, a language.
- **Uniformly mediocre** (narrow histogram around 5–6, no real tail): the
  model struggles with an *aspect* of the task everywhere. Browse a
  representative mid-band slice:
  ```
  get_eval_failures(eval_run="runs/{probe_run}", score_min=4, score_max=6,
                    sort="random", seed=0, limit=20, max_chars_per_message=2000)
  ```
  Look for what the *responses* consistently lack: phrasing/naturalness,
  coherence, format compliance, a specific sub-skill the judge keeps
  docking points for.

Read ~20 samples either way — sample + model response + judge reasoning +
score. Page with `offset` or fetch specific ones with `sample_indices` if a
pattern needs more evidence. Reading a few high scorers (`sort="desc"`) for
contrast often sharpens the pattern.

### 3. Write the findings report

Write `reports/failure_analysis_v<N>.md` in the workspace:

- **Why** the analysis was run (score, band, what was already tried).
- **Setup**: probe dataset, checkpoint, judge; the score distribution.
- **Patterns found**, each with 2–3 referenced sample indices and the
  judge's recurring criticism.
- **Planned remediation**: what data to synthesize, SFT or DPO, expected
  effect.

Optionally export the raw cases durably:
`get_eval_failures(..., export_path="feedback/probe_failures_v<N>.jsonl")`.

### 4. Synthesize targeted training data

Build a data-gen pipeline that specifically covers the identified failure
modes (new instructions/topics for outlier triggers; harder or more varied
examples of the weak aspect for uniform mediocrity). Generate into a NEW
supplemental dataset with `purpose="failures"` (same convention as the
production feedback loop), filtered with the scorer.

### 5. Retrain and compare

Train on original + supplemental together (multi-source `start_training`,
e.g. `dataset=[{"path": "datasets/{task}_train", "repeat": 1}, {"path":
"datasets/{task}_failures_v1", "repeat": 2}]`). Use SFT when the failures
are coverage gaps; use DPO when the model is close but consistently wrong
in a specific way. Re-eval on the SAME validation set and compare against
the previous checkpoint.

### 6. Iterate

This is a loop, not a one-shot. Each round starts from the previous report
(read it — don't rediscover known patterns) and probes with a fresh
`_probe_v<N+1>` set when the old probe influenced training data. Stop when:

- the score clears the deployment bar (**≥ 8** without regression, or 7.x
  that meets the task's baseline-relative target) → offer deployment (see
  "Outcome: read Δ first");
- improvement is < ~0.5 across two consecutive rounds → plateau. Report to
  the user what was tried and, if the budget caps the model size, present
  the case for a larger model (see budget compliance);
- the failure pattern points at the spec itself (ambiguous or conflicting
  requirements) → update `SPEC.md` with the user, then regenerate.

## Maintain NOTES.md

Before finishing a work phase here (and whenever you make a significant decision
or launch a long-running job), update the project-root `NOTES.md`: what was
decided and why, which approach is active, what is blocked, and the explicit
next steps. A future session resumes from that file — write for a reader with
none of this conversation's context. NOTES.md is advisory prose; job status and
artifacts are always verified with tools, never from notes.
