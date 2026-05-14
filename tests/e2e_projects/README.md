# E2E task projects

Each subdirectory here is a self-contained project that the
`tests/remote/experiment_e2e_pipeline.py` harness can drive end-to-end:
**datagen → LLM-judge filter → SFT → eval (→ DPO → eval, optionally)**.

Each project mirrors the structure of `example_project/`:

```
<task>/
├── e2e_config.json        # harness-readable; see schema below
├── SPEC.md                # human-readable task spec
├── data_gen/
│   └── pipeline.py        # one Pipeline subclass, the engine loads it
├── prompts/
│   ├── system.md          # system prompt prepended at infer time
│   └── schema.json        # OpenAI-style envelope for JSON tasks
├── evals/
│   └── scorers/
│       └── scorer.md      # judge criteria (markdown)
└── runs/                  # created by the harness; gitignored
```

For tool-calling tasks (Phase 4), add `tools/<task>_tools.json`. For
open-ended tasks (Phase 3), drop `prompts/schema.json`.

## `e2e_config.json` schema

```jsonc
{
  "task_kind": "json" | "tools" | "open",     // drives harness behavior
  "base_model": "LiquidAI/LFM2.5-1.2B-Instruct",

  "train_samples": 2000,                       // *post-filter* target
  "eval_samples": 200,                         // *post-filter* target
  "filter_threshold": 7.0,                     // judge score 1-10

  "datagen_pipeline": "data_gen/pipeline.py",  // relative to project dir
  "system_prompt": "prompts/system.md",        // optional
  "schema": "prompts/schema.json",             // json tasks only
  "tools": "tools/<task>_tools.json",          // tools tasks only (Phase 4+)
  "scorer": "evals/scorers/scorer.md",

  "sft": {
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "lora_r": 32,
    "lora_alpha": 64,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048
  },

  "dpo": null,                                 // or { num_iterations, beta } (Phase 3+)

  "max_new_tokens": 8192,                      // for inference
  "regression_threshold": -0.5                 // FAIL if Δ(SFT - baseline) < this
}
```

## Running a task

```bash
# Full run (toka, GPU required, hours).
python -m tests.remote.experiment_e2e_pipeline --task translation

# Quick shake-out — small sample sizes, all stages:
python -m tests.remote.experiment_e2e_pipeline --task translation \
    --train-samples 8 --eval-samples 4

# Iterating on the SFT / eval stage with an existing dataset:
python -m tests.remote.experiment_e2e_pipeline --task translation \
    --skip-generate-train --skip-generate-eval \
    --skip-filter-train --skip-filter-eval

# Run an alternate config (DPO-only on feedback_bot):
python -m tests.remote.experiment_e2e_pipeline --task feedback_bot \
    --config-name dpo_only
```

## Config variants

A task can have multiple configs side-by-side without duplicating the
pipeline / scorer / prompts. Drop additional `e2e_config_<NAME>.json`
files in the project dir and select with `--config-name <NAME>`.
Example: `feedback_bot` ships with both:

- `e2e_config.json` — SFT only (the default)
- `e2e_config_dpo_only.json` — DPO from baseline, no SFT

To skip SFT entirely, set `"sft": null` and provide a `dpo` block.
The harness then runs `baseline → DPO → post-DPO eval` and DPO starts
from `base_model` directly.

The harness writes `runs/<exp_name>/report.md` with a verdict
(`✅ PASS / ⚠️ WARN / ❌ FAIL`) on whether SFT improved over zero-shot baseline.

## Adding a new task

1. `mkdir tests/e2e_projects/<task>/{data_gen,prompts,evals/scorers}`.
2. Write `SPEC.md` describing input/output and any constraints.
3. Write `data_gen/pipeline.py` — one `Pipeline` subclass with a
   `generate(self, client, input=None) -> Conversation` method.
   See `tests/e2e_projects/translation/data_gen/pipeline.py` for a
   reference.
4. Write `prompts/system.md` and (for JSON tasks) `prompts/schema.json`.
5. Write `evals/scorers/scorer.md` — concise judging criteria; keep it
   under 1 page so the judge prompt stays cheap.
6. Write `e2e_config.json` per the schema above.
7. Smoke-test with small numbers:
   ```bash
   python -m tests.remote.experiment_e2e_pipeline --task <new> \
       --train-samples 8 --eval-samples 4
   ```
8. If the smoke run is sane, do the full run on toka.

## Current tasks

| task               | kind | DPO  | status                          |
|--------------------|------|------|---------------------------------|
| `translation`      | json | no   | ✅ Phase 2                      |
| `summarization`    | open | yes  | ✅ Phase 3                      |
| `feedback_bot`     | tools| no   | ✅ Phase 4                      |
| `email_extraction` | json | no   | ✅ Phase 5                      |
| `email_triage`     | json | yes  | ✅ Phase 5                      |
| `spam_filter`      | json | opt  | ✅ binary classifier (yes/no); 3 configs: default/with_dpo/dpo_only |
| `ar_to_de`         | open | yes  | ✅ AR→DE translation, web + conversational; SFT+DPO default + dpo_only variant |

Tracking plan in `/home/mathias/.claude/plans/question-the-agent-has-lazy-kitten.md`.
