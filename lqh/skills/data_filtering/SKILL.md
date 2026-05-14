# Skill: Data Filtering (Bring-Your-Data-For-Scoring)

You are now in **data filtering** mode. The user has brought their own
dataset and wants to keep only the high-quality samples. Your job is to
turn their intent into a scorer and run `run_data_filter` to produce a
filtered dataset ready for training.

## When to use this skill

- User has a parquet file with ChatML samples (`messages` column) and says
  things like "I have 10k samples, filter out the bad ones" or "score
  this and keep only the good ones".
- User has uploaded a HF dataset that needs pruning before training.

If the user wants to *generate* new data, use `data_generation` instead.
If they want to *score* for evaluation (not to filter), use `evaluation`.

## Workflow

1. **Locate the input.** Call `list_user_data` or `list_files` to find
   the user-brought parquet. If they brought prompts or a non-ChatML
   format, they need `data_generation` first to build conversations.

2. **Understand the quality criteria.** Ask 2-4 focused questions about
   what makes a sample "good": correctness, relevance, tone, format,
   language, length, safety. Use `ask_user` with concrete options where
   possible.

3. **Write a scorer file.** Create `evals/filter_<task>.md` with the
   criteria. The judge reads this as markdown, so use headings and
   bullet points. Example structure:

   ```markdown
   # Scorer: <task> filter

   Score samples from 1-10.

   ## Criteria
   - **Correctness (0-4):** ...
   - **Format (0-3):** ...
   - **Tone (0-3):** ...

   ## Keep if
   - score >= 7 AND no hallucinated facts

   ## Drop if
   - assistant refused or deflected
   - output is in the wrong language
   - tool calls are malformed
   ```

4. **Pick a threshold.** Default is 6.0. If the user has strong quality
   bar, use 7 or 8. Remind them that higher threshold = smaller kept set.

5. **Dry-run small.** Call `run_data_filter` on a small slice first (ask
   the user to point at a subset, or use `run_data_filter` with
   `model_size="small"` for speed) and inspect `datasets/<output>/scores.parquet`
   via `read_file`. Look at the kept/dropped examples together with the
   user.

6. **Iterate the scorer.** If the judge is keeping garbage or dropping
   gold, edit `evals/filter_<task>.md` and re-run. Do not change the
   threshold to paper over a bad scorer.

7. **Full run.** Once the user is happy with the scorer on a sample,
   call `run_data_filter` on the full input. Report kept/dropped counts
   and the summary path.

## Tool

- `run_data_filter(input_path, scorer_path, output_dataset, threshold?, model_size?)`

Outputs under `datasets/<output_dataset>/`:
- `data.parquet` — kept rows, same schema as input (drop-in for training).
- `scores.parquet` — one row per input sample with score, reasoning, `kept`.
- `summary.json` — counts, threshold, mean score, keep-rate, judge model.

## Soft thresholds (defaults — adjust to the task)

These are starting points, not hard rules. Pick what fits the task and
explain your choice.

- **Quality threshold:** 6/10 or 7/10 is a healthy default for the
  judge cutoff. Tasks with a sharp correct/incorrect signal (math,
  classification) can run higher (8/10). Tasks with subjective quality
  (style, summarization) often work at 6/10.
- **Keep rate:** keeping ≥70% of samples is a sign of a well-aligned
  generator. **Dropping more than ~30% of samples is a red flag** —
  the right fix is usually to scale up the data-generation model (use
  a larger generator) or tighten the pipeline prompt, not to loosen
  the rubric. In auto mode, surface this warning via `set_auto_stage`.
- The threshold encodes quality intent, not a quota. If kept-rate is
  too low, fix the upstream generator first; only then revisit the
  scorer.

## Reading the output

`run_data_filter` returns a per-bucket histogram and five quantiles
(p10/p25/p50/p75/p90) under the headline numbers. The *shape* is
more useful than the mean for deciding what to do next:
- **bimodal** (two peaks separated by a valley) → fix the bad cluster
  upstream rather than tightening the threshold.
- **uniformly mediocre** → the generator is the bottleneck, not the
  rubric.
- **long low tail** → keep the threshold, the rare bad samples are
  noise.

The same shape-based heuristics that the `data_generation` skill uses
for synthetic data apply to user-brought data too — see Phase 3.5.3
("Read the score distribution") in that skill for the long-form
guidance.

## Rules

- **Never change the threshold to hit a target kept-count.** Edit the
  scorer instead. The threshold expresses quality intent, not quota.
- **Never filter without a scorer file the user has seen.** Show the
  scorer via `show_file` before the full run.
- **Never filter synthetic data we just generated in the same session**
  — that's what `run_scoring` is for. This skill is specifically for
  user-brought data.
