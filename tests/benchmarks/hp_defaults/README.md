# Default hyperparameters — calibration study

Answers: **what should LQH's default SFT hyperparameters be, and does one
setting work across tasks, dataset sizes, and models?**

## Why this exists

`start_training` used to sweep a hyperparameter grid on every call. A sweep
trains its configs one after another inside a single job, so the first run
after a dataset was ready blocked for hours — users reported it as "the sweep
takes forever", and they were right. It is inherent to hyperparameter search,
not a bug.

So the product changed: **SFT now runs once at the defaults in
`lqh/train/defaults.py`**, and sweeping became a late-stage lever that
`/improve` reaches for after data volume and model size are exhausted. That
only works if the defaults are good — and they were unvalidated literals
inlined in a tool handler, chosen once and never measured across models or
dataset scales.

This study measures them.

## Design

A **cell** is one `(task × train_size × model)` combination. The full grid of
hyperparameters is trained *and judge-scored* inside every cell, so the study
can ask the question that actually matters for a default:

> Averaged across contexts, how much judge score does this config give up
> against the best config **in that same context**?

That quantity is **regret**, and it is the decision metric. A config that wins
six cells by 0.05 and loses two by 2.0 is a bad default; a win count says
otherwise, regret does not.

| axis | levels |
|---|---|
| task | `translation`, `extraction`, `classification`, `voice_satisfaction` |
| train size | 500, 2000, 8000 |
| model | 350M / 1.2B × base / instruct |
| **hyperparameters** | lr ∈ {2e-5, 1e-4, 2e-4, 5e-4, 1e-3} × epochs ∈ {1, 2, 3} |

48 cells × 15 configs. The learning-rate range is deliberately **wider than the
product's sweep grid** (`{5e-5, 1e-4, 5e-4}`) and brackets the shipped LoRA
default of 1e-4 on both sides. A search confined to the product grid could never
discover that the whole grid sits in the wrong place — which is exactly what
happened while the default was 2e-5.

**Stage A has run** (`hpd-stageA`, 2026-08) on the earlier axis
`{1e-5, 2e-5, 5e-5, 1e-4, 2e-4}`: `lr1e-4_e3` won at 0.015 mean regret, no
dimension earned its own default, and the shipped defaults were updated from it.
What it did **not** settle, and what a follow-up should target:

- **Model size.** 5 of the 6 contributing cells were 350M — the 1.2B cells were
  mostly lost to orphaned cloud jobs (see "Jobs that did not complete" in its
  report). The `param_count` verdict is therefore near-vacuous.
- **Training variance.** No `--replicate-seeds`, so the 0.127-point noise floor
  is a lower bound and the top five configs are statistically tied.
- **The range above 2e-4.** Untested, though a customer's 1.2B task gained +1.30
  from 5e-4 — hence the widened axis.

Tasks come from `base_vs_instruct/pipelines/`. `voice_satisfaction` is the
non-saturating one — the others reach >8/10 after SFT, which compresses the
differences between configs and would make everything look equally good.

**Datasets are nested.** Each task's largest split is generated once and the
smaller sizes are *prefixes of it*, so a difference between the 500-row and
8000-row cells is about volume rather than about the 500 rows happening to be
easier. The eval set is shared across every cell of a task, so judge scores are
comparable within it.

## Two guardrails

The study is built to be able to say "we found nothing", and both guardrails
exist to make that outcome reachable:

- **Balanced panels.** Configs are only ever compared on cells where *all* of
  them were measured, and each cell's oracle is computed within that same set.
  A config that only ran on the easy cells cannot look good by accident.
- **A measured noise floor.** `--replicate-seeds` re-runs identical configs
  under different training seeds. A per-dimension difference smaller than that
  spread is not a finding, and the analysis refuses to call it one — it also
  requires a bootstrap CI over cells that excludes zero, so one outlier cell
  cannot carry a split.

Without replicates the report falls back to the judge's standard error and
**says so explicitly**: that is a lower bound, since it captures judging
variance but not training variance.

## Protocol

Staged, because 48 × 15 judged runs is a lot of GPU for a shortlist:

| stage | what | cells | configs |
|---|---|---|---|
| **A — screen** | full grid on a balanced anchor subset | ~12 | 15 |
| **B — confirm** | stage-A finalists everywhere | 48 | ~4 |
| **replicates** | top configs re-run under 3 seeds | ~4 | 2 |

`anchor_cells` picks the stage-A subset so that **no level of any dimension is
missing** — a shortlist chosen on instruct-only cells would be a bad shortlist
for base models.

```bash
# Smoke first — local GPU, no cloud spend, proves the whole path.
uv run python -m tests.benchmarks.hp_defaults.run --compute local \
    --tasks translation --models 350M-Instruct --sizes 100 \
    --grid-points lr=1e-4:e1,lr=5e-4:e1 --eval-size 20 --yes

# Stage A — screen.
uv run python -m tests.benchmarks.hp_defaults.run --anchors-only --yes

# Stage B — confirm the finalists everywhere, with a stronger judge.
uv run python -m tests.benchmarks.hp_defaults.run \
    --grid-points lr=1e-4:e2,lr=1e-4:e3,lr=5e-4:e2,lr=5e-4:e3 \
    --judge-size large --yes

# Noise floor.
uv run python -m tests.benchmarks.hp_defaults.run \
    --tasks translation,extraction --sizes 2000 \
    --grid-points lr=1e-4:e3,lr=5e-4:e3 --replicate-seeds 1,2,3 --yes
```

`run.py` prints the cell/run count and a cost estimate and refuses to launch
without `--yes`. Order of magnitude for the full protocol: ~350 training runs,
roughly **$400–900** of billed A100 time plus data generation.

## Before the first cloud run: deploy `lqh`

Cloud sandboxes run the `lqh` baked into the Modal training image, **not** your
working tree. The study depends on two things that image must already have:

- `eval_all` in `lqh.train.sweep` — without it the sweep silently ignores the
  flag, trains every config, and judges none of them.
- `sweep_summary.json` / `runs.jsonl` in the publisher's allowlist
  (`lqh/remote/publish.py`) — without it the per-config results never leave the
  sandbox volume.

So publish `lqh` and rebuild the image before running anything on cloud (see
`CLAUDE.md`, "push to production"): bump the version, commit and push, run
`build_and_deploy.sh`, then from `backend/`:

```bash
.venv/bin/python3 scripts/modal_build_image.py \
    --purpose all --promote --register --env-file ../envvars.prod
```

`runner.py` fails loudly with this instruction if a job completes without
publishing a leaderboard, rather than reporting an empty study.

## How it runs

Each cell chunk is one `train_sft_sweep` cloud job carrying a `grid_override`
and `eval_all: true` — the sweep trains every config and judge-scores it *in
the same sandbox*, and each result comes back on that config's row in
`sweep_summary.json`. Cells fan out concurrently (`--max-concurrent-jobs`).

Grids are chunked (`--chunk-size`, default 6) because the cloud runner caps
`train_sft_sweep` at 720 minutes and a timeout loses **every** config in the
job, not just the one that ran over.

Everything resumes: a chunk whose rows are already complete is skipped, and a
chunk that dies is noted in the report rather than sinking the study.

## Output

`report/results.json` (every observation, so the analysis can be re-run and
argued with without touching a GPU) and `report/report.md`, which leads with:

1. **The recommended default** and its mean regret — the number that says
   whether "one run at defaults, sweep later" is defensible.
2. **Whether any dimension needs its own default**, with gains, CIs, and an
   explicit verdict.
3. **A proxy check**, free: the shipped sweep selects winners on `eval_loss`
   alone, and because this study judged every config it can report the
   Spearman correlation and top-1 agreement per cell.
4. **The epochs question**: training already keeps the best checkpoint by eval
   loss, so a 3-epoch run can save an epoch-1 model. If that is the norm,
   `num_epochs` is a ceiling rather than a tuned quantity and the product's
   grid can drop the axis.
5. **A paste-ready `defaults.py` snippet.**

## Installing the result

Paste the snippet into `lqh/train/defaults.py`, update `PROVENANCE` in the same
commit, and re-run `tests/unit/test_train_defaults.py`. **The parity test will
fail** until it is updated to the new values — that is deliberate. A shipped
default should never move silently.

## Files

| file | role |
|---|---|
| `cells.py` | the design matrix and the anchor-subset chooser |
| `grid.py` | the HP grid, replicates, and job chunking |
| `runner.py` | launch payload; cloud and local execution |
| `run.py` | orchestrator: datagen → fan-out → report |
| `analyze.py` | regret, panels, conditional defaults, noise floor (pure) |
| `report.py` | `results.json` + `report.md` |

Tests live with the rest of the unit suite (`testpaths = ["tests/unit"]`), so
they run in CI: `tests/unit/test_hpd_analyze.py` covers the analysis —
including two planted-answer end-to-end cases, one where a single default is
correct and one where a per-dimension split is — and
`tests/unit/test_hpd_harness.py` covers the design matrix, grid and launch
payload.
