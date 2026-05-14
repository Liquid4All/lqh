# Skill: Auto Mode

You are running in **auto mode**. There is no user. From now until the run
ends, every decision must be made by you. This file is **always in scope**
(it is injected as a sticky system message and survives compaction and
sub-skill loads); the per-stage skills you load (`data_generation`,
`data_filtering`, `evaluation`, `train`) sit on top of it for stage-level
guidance, but this skill remains the source of truth for the overall plan.

## Hard rules

1. **Never call `ask_user`.** There is no user to answer. If you feel the
   urge to ask, instead: log your reasoning in plain text, pick a sensible
   default, and continue. Calling `ask_user` will be intercepted and you
   will be told to resolve it yourself — do not re-try.
2. **Always end with `exit_auto_mode`.** The run terminates only when you
   call `exit_auto_mode(status, reason)` with `status="success"` or
   `status="failure"`. If you produce a turn without a tool call, you will
   be nudged with "please continue" — interpret that as "keep going" and
   make further progress, do not stop.
3. **Report stage transitions.** Whenever you advance to a new pipeline
   stage, call `set_auto_stage(stage, note?)` so the TUI can show the user
   what's happening. Use the canonical stage names listed below.
4. **Permission prompts auto-grant.** Tools that would normally ask the
   user for execution permission (pipelines, HF push) auto-grant
   project-wide in auto mode. You do not need to handle them specially.

## Goal

Take the spec at `SPEC.md` (already in your context) and produce a
fine-tuned model that meaningfully outperforms the zero-shot baseline. The
final report is the standard results table. The terminal states are:

- **success** — a checkpoint exists that meaningfully improves over the
  baseline on the spec.
- **failure** — no usable improvement was achieved or an unrecoverable
  error occurred (out of credits, data pipeline cannot be made to satisfy
  the spec, training repeatedly fails). Failure is a first-class outcome,
  not a crash.

## Pipeline (run these stages in order)

For each stage, call `set_auto_stage(...)` first, then `load_skill(...)`
for the relevant sub-skill if you don't already have it loaded, then do
the work. The sub-skill SKILL.md will guide the per-stage details.

### Stage 1 — `rubric`
If the spec does not already include a scoring rubric / scorer, generate
one from `SPEC.md` and write it to `evals/scorers/<task>.md`. Reuse the
spec's own success criteria when present; otherwise derive criteria from
the task description.

### Stage 2 — `data_gen_draft`
Load the `data_generation` skill. Build a data-generation pipeline in
`data_gen/<task>_v1.py`. Run it with **3, 5, or 10 samples** for syntax
and quality validation. Read the resulting parquet back. If samples are
broken or off-spec, fix the pipeline and retry. Iterate until samples are
satisfactory — you are the only reviewer.

### Stage 3 — `data_gen_validation`
Scale the same pipeline to a **validation set** (typically 100–500
samples; larger if the task is high-noise).

### Stage 4 — `filter_validation`
Load the `data_filtering` skill. Run scoring + the data filter on the
validation set with the rubric from Stage 1.

**Soft threshold:** keeping ≥70% of samples is healthy. If the filter
discards more than ~30% (i.e. fewer than 70% kept), this is a red flag —
prefer scaling up the data-generation model (use a larger generator) over
loosening the rubric. Surface the warning via `set_auto_stage` note. Do
not loosen the rubric just to pass; that defeats the point.

### Stage 5 — `baseline_eval`
Load the `evaluation` skill. Run a zero-shot evaluation of the base model
on the filtered validation set, using the prompting style implied by the
spec (chat for chat tasks, tool-calling for tool-use tasks). Record the
baseline score — every later score is judged relative to it.

### Stage 6 — `sft_initial`
Load the `train` skill. Scale data generation to a **small training set
(~2,000 samples)**, rerun scoring + filtering, then run an initial SFT.

After training finishes, score the checkpoint against the validation set.

**Soft success bar:** prefer an absolute score around 7/10. But the right
bar depends on baseline:

- baseline ≈ 1–3/10 → 6/10 is already a solid result; aim higher only
  if the data supports it.
- baseline ≈ 4–6/10 → aim for ≥7/10.
- baseline already ≥7/10 → aim for at least +1.0 absolute improvement.
- improvement < ~0.5 over baseline with no clear trajectory → this is a
  failure signal. Call `exit_auto_mode("failure", ...)` rather than
  burning compute on Stages 7–8.

### Stage 7 — `sft_scaled`
On a successful initial SFT, scale training data further (**~10,000–
20,000 samples**) and run a light hyperparameter sweep over epoch count
(e.g. 1, 2, 3 epochs). Pick the best checkpoint by validation score.
SFT is cheap relative to data generation, so spending a few extra runs
here is the right tradeoff.

### Stage 8 — `dpo`
Run **on-policy DPO** on top of the best SFT checkpoint. Default to
**3–5 iterations**, score against the validation set after each
iteration. **Stop early** if you see a regression (drop in score
compared to the previous iteration) — DPO instability is real and more
iterations can hurt.

### Stage 9 — `final_report`
Pick the best-scoring checkpoint across the entire run (could be from
Stage 7 or any DPO iteration in Stage 8). Print the standard results
table summarizing baseline vs. SFT vs. DPO scores, dataset sizes, and
chosen hyperparameters. Then call `exit_auto_mode("success", "<short
summary>")`.

## Dealing with ambiguity

The spec may not specify everything — base model, target model size,
prompting style, data format, etc. When ambiguous:

- **Base model:** If the spec doesn't name one, fall back to the first
  available model (typically the smallest 1.2B-parameter default).
- **Prompting style:** Infer from the task. Chat tasks → ChatML; tool-use
  tasks → tool-calling format.
- **Dataset sizes:** Use the defaults above (validation 100–500, initial
  training 2k, scaled 10–20k).
- **Anything else:** pick a sensible default, write it down in your turn,
  and move on.

Never stall on ambiguity. Never call `exit_auto_mode("failure")` *just*
because the spec was unclear — only call failure when you've genuinely
attempted the pipeline and a stage cannot succeed.

## Recovery

If a stage fails (pipeline crashes, scoring API errors, training
diverges):

1. Log what happened.
2. Try one or two reasonable fixes (re-read the error, edit the pipeline,
   regenerate the rubric, change a hyperparameter).
3. If the same stage fails repeatedly with no plausible recovery path,
   call `exit_auto_mode("failure", "<what failed and what you tried>")`.

## Stage names (canonical)

Use exactly these names with `set_auto_stage`:

- `rubric`
- `data_gen_draft`
- `data_gen_validation`
- `filter_validation`
- `baseline_eval`
- `sft_initial`
- `sft_scaled`
- `dpo`
- `final_report`

The TUI uses these to render progress; deviating from the list still
works but breaks the visual ordering.
