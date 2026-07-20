# Driving lqh from an agent harness

lqh version @@VERSION@@ · envelope schema_version 1

## What LQH is

LQH (Liquid Harness) customizes Liquid AI foundation models (LFMs) into
task-specific models, starting from a written spec. A *project* is just a
directory: `SPEC.md` (the task specification), `data_gen/` (generation
pipelines), `datasets/`, `runs/` (training/eval runs), `evals/`,
`prompts/`, `feedback/` (production failure cases), and `.lqh/` (state:
sessions, permissions, project identity). lqh ships an interactive TUI
agent (`lqh` with no arguments) — but every pipeline step is also
callable headlessly through the commands documented here, so any agent
harness (Claude Code, Codex, …) can orchestrate the same workflow with
its own judgment.

## The fine-tuning workflow

1. Derive `SPEC.md` — the task specification.
2. Write scoring criteria (rubric / scorer).
3. Build a data-generation pipeline under `data_gen/`.
4. Smoke-test it (n=3), then inspect quality (n≈20, read the samples).
5. Generate validation + training datasets.
6. Score both datasets and filter low-quality samples.
7. Zero-shot eval on the validation set → baseline.
8. Fine-tune (SFT) on the training dataset.
9. On-policy preference optimization (DPO) to refine.
10. Deploy: API endpoint (`push_to_production`) or GGUF export for edge
    (`gguf_convert`).
11. Real-world evaluation of the deployed model.
12. Feedback (failure cases under `feedback/`, or spec changes) →
    re-enter at whichever earlier step the feedback implicates.

Iteration is the norm, not the exception: poor data quality sends you
back to 3–6, a model that doesn't learn sends you back to 5–8.

## Integration modes

- **`lqh tool …`**: call individual pipeline steps and get a JSON
  envelope per call. You keep the orchestration loop. Fine-grained.
- **`lqh run "<task>"`**: delegate a whole task to lqh's own agent
  headlessly — it plans, executes tools, waits on runs, and returns one
  structured JSON result (plus a resumable session). Use it when the
  step is coarse ("generate and score a 500-sample training set") and
  you don't want to micro-manage.

## Consent model

Direct `lqh tool call` invocations are **pre-consented**: your harness's
own permission system is the gate, so lqh's interactive permission
prompts never fire on this surface. The one exception is data
destruction: overwriting an existing dataset/run still requires the
explicit `"overwrite": true` argument in your call — without it the call
fails with `error.kind: "conflict"` (allocate a versioned name like
`my_dataset_v2` instead, unless the user explicitly asked to replace).

`lqh run` auto-grants task-implied work (scripts, cloud data-gen,
training) for the run, but **publishing** (`hf_push`,
`push_to_production`, `create_inference_key`) is gated: without
`--allow-publish` the run terminates with `status: "needs_permission"`
and the exact re-invocation.

### `lqh run` result (stdout, exactly one JSON document)

```json
{
  "schema_version": 1, "run_id": "…",
  "status": "success",
  "reason": "…", "summary": "…markdown…",
  "artifacts": [ {"kind": "run", "path": "runs/sft_v1", "source": "ledger"} ],
  "metrics": { "post_sft": {"value": 0.78, "provenance": "reported"} },
  "session_id": "…",
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "turns": 0 },
  "duration_s": 0
}
```

`status` ∈ `success | failure | needs_permission | needs_configuration |
auth_required | limit_exceeded | interrupted | timed_out`. Artifacts with
`source: "ledger"` were recorded deterministically from successful tool
calls; `"reported"` ones are validated model claims (metric provenance is
likewise `"reported"` — corroborate against eval artifacts yourself).
Progress events stream on stderr as NDJSON
(`{"schema_version","run_id","seq","event",…}` with events `start`,
`agent_message`, `tool_call`, `tool_result`, `progress`, `job_running`,
`stage`, `end`) — but stderr is a MIXED stream: log lines, warnings, and
redirected library output appear between events, so parse only lines
that start with `{` and JSON-decode. `--resume <session_id>` continues a
prior run contextually — e.g. after granting `--allow-publish`.

## Contracts

### `lqh tool call <name> --args '<json>'` envelope (stdout, always exactly one JSON document)

```json
{
  "schema_version": 1,
  "ok": true,
  "tool": "start_training",
  "result": { "text": "…", "secret": null, "details": {} },
  "error": null,
  "meta": { "duration_s": 3.2, "lqh_version": "@@VERSION@@" }
}
```

On failure `result` is `null` and `error` is
`{ "kind", "message", "retryable", "details" }` with `kind` one of:
`auth`, `permission`, `config`, `validation`, `not_found`, `conflict`,
`upstream`, `runtime`. Legacy results not yet classified carry
`meta.classified: false`. Progress/diagnostic output goes to stderr.

### Exit codes (all subcommands)

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | failure (runtime / upstream / not_found / conflict) |
| 2 | usage or validation error |
| 3 | permission denied |
| 4 | authentication required — run `lqh login` |
| 5 | configuration required (e.g. compute target unset, unresolved project copy) |
| 6 | interrupted |

### Secrets

A tool that mints a one-time secret (e.g. `create_inference_key`)
returns it in `result.secret` — it lands in your transcript by design.
Add `--save-secret` to also persist it into the project's `.env`.

### Auth

`lqh login` runs a device-flow login: verification URL + code on stderr,
one JSON result on stdout (`status`: `already_logged_in` | `logged_in` |
`expired` | `error`). Tools tagged `auth` below need it; everything else
works offline/locally.

## Commands

```
lqh hello                       # this guide (alias: lqh docs agents)
lqh docs skills                 # list built-in skills
lqh docs skill <name>           # print a skill's SKILL.md (workflow playbooks)
lqh login [--no-browser]        # device-flow auth
lqh run "<task>" [--project DIR] [--allow-publish] [--resume ID]
        [--max-turns N] [--max-tool-calls N] [--timeout SECONDS]
        [--prompt-file f|-] [--quiet] [--save-secret]
lqh tool list [--json]          # the tools below
lqh tool schema <name>          # JSON schema for a tool's arguments
lqh tool call <name> --args '<json>' [--args-file f|-] [--pretty] [--save-secret]
lqh tool call training_status --args '{"run_name": "…"}' --wait
                                # park until the run is terminal (results incl.)
lqh status [--json]             # run states + attention signals at a glance
lqh project continue|fork       # resolve a copied project directory (see below)
```

## Tools

@@TOOL_TABLE@@

`lqh tool schema <name>` gives the full argument schema; `--args` takes
the same JSON object lqh's own agent would emit.

## Worked examples

Discover project state (read-only, no auth):

```
lqh tool call summary
```

Delegate a whole step to lqh's agent:

```
lqh run "Generate a 200-sample draft training set for the spec, score it, and report the quality distribution."
```

Run a data-generation pipeline (smoke test, 3 samples):

```
lqh tool call run_data_gen_pipeline --args '{
  "script_path": "data_gen/my_task.py",
  "num_samples": 3,
  "output_dataset": "smoke_v1",
  "purpose": "smoke"
}'
```

Start a training run, then poll it:

```
lqh tool call start_training --args '{
  "type": "sft",
  "base_model": "lfm2-1.2b",
  "dataset": "datasets/train_v1",
  "eval_dataset": "datasets/val_v1",
  "scorer": "data_gen/scorer.md"
}'
lqh tool call training_status --args '{"run_name": "<run>"}' --wait
```

`--wait` blocks (LLM-free) until the run is terminal — including scoring
results and cloud data-gen dataset downloads — then returns the final
status. Prefer it over polling.

## Project conventions you must follow

Your harness plays the role lqh's built-in agent normally plays, so the
same conventions apply:

- **Read and maintain `NOTES.md`.** It is the advisory prose handoff
  between sessions (decisions, gotchas, current state). Read it before
  acting; update it after finishing a work phase. Verify its claims with
  tools — it is advisory, not authoritative.
- **Treat datasets, runs, and evals as immutable.** Allocate versioned
  names (`train_v2`) instead of overwriting; pass `"overwrite": true`
  only on explicit user intent.
- **Read `manifest.json`** co-located with datasets/runs for provenance
  (spec hash, source inputs, producing run) instead of guessing from
  filenames.
- **Drop production failure cases under `feedback/`** so the iteration
  loop can pick them up.
- **Heed warnings** from `summary` and startup signals about spec drift
  or orphaned cloud jobs before spending compute.
- **Copied project directories:** if a project folder was copied, cloud
  operations are blocked (exit 5) until you resolve the identity —
  `lqh project continue` (this copy keeps the original identity) or
  `lqh project fork` (fresh identity + cloud namespace). Ask the user
  which they intend if unclear.
