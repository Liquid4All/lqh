# Base vs Instruct — fine-tuning benchmark

Answers: **which LFM2.5 variant is the better base for fine-tuning, and does
the ranking hold across tasks?** For each `(task, model)` it runs the full
pipeline and reports three judge scores — baseline, best-SFT, best-DPO — on a
held-out eval set.

Models (from `BASE_VS_INSTRUCT.md`):

| key              | HuggingFace id                   | generation |
|------------------|----------------------------------|------------|
| `350M-Instruct`  | `LiquidAI/LFM2.5-350M`           | LFM2.5     |
| `350M-Base`      | `LiquidAI/LFM2.5-350M-Base`      | LFM2.5     |
| `1.2B-Instruct`  | `LiquidAI/LFM2.5-1.2B-Instruct`  | LFM2.5     |
| `1.2B-Base`      | `LiquidAI/LFM2.5-1.2B-Base`      | LFM2.5     |
| `LFM2-1.2B`      | `LiquidAI/LFM2-1.2B`             | LFM2 (old) |
| `LFM2-350M`      | `LiquidAI/LFM2-350M`            | LFM2 (old) |

The `LFM2-*` rows are the *previous* generation (instruct only — LFM2 has no
separately published `-Base`), included so the benchmark also answers **"how
fine-tuneable is the older model vs the newer LFM2.5?"**. The two generations
were saved with different transformers versions (the v4→v5 transition): LFM2 and
LFM2.5-1.2B ship `transformers_version` 4.x metadata, LFM2.5-350M ships 5.x.
They all load fine on transformers `>=5,<6`, and a **preflight** (below) proves
that per model before any GPU time is spent.

Tasks: `translation` (EN→DE, format discipline), `extraction` (invite text →
JSON), `classification` (3-class sentiment), `messy_extraction` (noisy support
thread → latest-value JSON), `style_rewrite` (support reply rewritten to a
tone/style brief), and `voice_satisfaction` (the observability task — read a
human↔voice-assistant interaction and emit a structured satisfaction
assessment). Each is a self-contained datagen pipeline under `pipelines/` plus a
judge rubric in `tasks.py`.

`voice_satisfaction` is the **non-saturating** task: translation/extraction/
classification reach >8/10 after SFT, leaving little headroom for DPO. Scoring a
voice interaction well needs genuine judgement (frustration sensitivity, correct
failure/success-tag attribution, faithful per-turn reasoning), so the judge
rarely hands out a 10 — which keeps room for DPO to show movement. Its gold is
an LLM-generated JSON assessment validated against the SPEC's hard rules
(`validate_output`) before entering train/eval; it generates both transcript and
event-log (rich + minimal metadata) input formats, which the model must treat
equivalently.

## How it works

- **Compute is local GPU.** Training (`lqh.train.sweep`) and inference
  (`lqh.infer`) run as local subprocesses on this machine's CUDA GPU.
- **Eval is local, not API.** Every reported number comes from one primitive:
  local inference of the model weights → judge scoring via `run_scoring`. The
  judge (`judge:small|medium|large`) is the only API call; the model never
  runs through the LFM router. This is what lets us score the `-Base` variants
  and trained checkpoints, none of which the API serves.
- **The system prompt is baked into the ChatML** at datagen time, so training,
  eval, and scoring all share it.
- **Winner selection** uses the sweep's validated in-training proxy
  (`eval_loss` for SFT, `eval_ce_chosen_mean` for DPO) — no judge needed. The
  script then re-evaluates the winner's `model/` dir with the local-eval
  primitive for the reported score.
- **Self-scoring contract.** A standalone script has no TUI watcher, so
  `run.py` exports `LQH_API_TOKEN` + `LQH_BASE_URL`; the training subprocess
  then self-scores inline (`lqh.train.cloud_score.is_cloud_mode`). This is
  **mandatory for on-policy DPO**, which builds preference pairs from
  judge-scored rollouts every iteration.

## Run it

Authenticate first (`lqh` → `/login`, or set `LQH_API_TOKEN`). Then:

```bash
# Smoke (cheap, validates the whole pipeline end-to-end)
uv run python -m tests.benchmarks.base_vs_instruct.run \
    --tasks translation --models 350M-Instruct,350M-Base \
    --train-size 100 --eval-size 20 --grid-size tiny

# Full run (the spec's 20k/400, all tasks, all models)
uv run python -m tests.benchmarks.base_vs_instruct.run \
    --train-size 20000 --eval-size 400 --grid-size small
```

Outputs land in the workdir (default `~/.lqh-bvi/<run-name>/`):
`datasets/`, `scorers/`, `runs/<task>__<model>__{baseline,sft,sft_eval,dpo,dpo_eval}/`,
and `report/{results.json,report.md}`. The report is rewritten after every
`(task, model)` so a long run is inspectable mid-flight.

### Key flags

- `--train-size` / `--eval-size` — dataset sizes (default 200 / 40 smoke).
- `--grid-size {tiny,small}` — sweep grid (3 vs 6 configs).
- `--skip-dpo` — SFT only.
- `--dpo-train-size` — prompt count for the DPO stage (default 1000, capped at
  `--train-size`). DPO regenerates rollouts on **all** its prompts every
  iteration, so it must stay bounded and decoupled from the (possibly 20k) SFT
  train set — SFT uses the full set; DPO uses a slice. Set to `0` for an
  SFT-only comparison. Positive values must be ≥ 400 or DPO auto-skips.
- `--judge-size {small,medium,large}` — scoring judge (use `large` for the
  final reported run).
- `--filter-threshold` — scorer-based quality gate for generated datasets
  (default `7.0`). Each generated sample's gold is scored against the task
  scorer; samples below the threshold are dropped and regenerated until the
  target count is reached. This keeps weak/templated golds out of train **and**
  eval (the failure mode that tanked the old `style_rewrite` set). Applies to
  both splits.
- `--overgen-factor` — how much to over-generate before filtering (default
  `1.6`), sized to absorb the expected drop rate; the run then tops up using the
  observed keep-rate, so the final dataset always has exactly the target count.
- `--no-filter` — disable the scorer filter and use raw pipeline output as-is
  (the old behaviour; cheaper, but no quality gate).
- `--no-resume` — recompute everything (default resumes: completed datasets,
  sweeps, and scored evals are reused).
- `--skip-preflight` — skip the transformers v4/v5 model-compatibility check
  (on by default; see below).
- `--workdir`, `--run-name`, `--sweep-timeout`, `--eval-timeout`.

> **Cost note:** filtering adds one judge call per *generated* sample (not just
> the kept ones) plus the over-generation. At `--train-size 10000` with a ~60%
> keep-rate that is roughly `10000/0.6 ≈ 16.7k` extra generations and judge
> calls per task. Use `--no-filter` for cheap smokes; keep it on for reported
> runs.

## Cost & time

Full scale is large: 20k datagen × 5 tasks (≈100k generation calls), then
4 models × 5 tasks × (SFT sweep + DPO sweep + 3 evals). Validate with the smoke
command, then scale up one task/model at a moderate `--train-size` (e.g. 1000)
to gauge wall-time before committing to 20k.

## Known caveats

- **Transformers v4/v5 preflight.** Before any datagen/GPU work, `run.py`
  probes every selected model's *config + tokenizer* (no weights — a couple of
  small downloads) under the installed transformers: it confirms the
  `model_type` maps to a registered causal-LM class and the chat template
  renders, and it logs the `transformers_version` each repo was saved with so a
  v4/v5 mix is obvious. An unsupported architecture or missing template aborts
  the run *here*, with an actionable message ("upgrade transformers", "drop the
  model", or `--skip-preflight`), instead of crashing a sweep child hours in.
  The decision logic (`preflight.verdict`) is a pure, unit-tested function.
  Verified state: all six models load on transformers `>=5,<6` despite the
  v4-saved / v5-saved mix.
- **Base-model chat template — verified OK.** SFT and infer call
  `tokenizer.apply_chat_template`. The `-Base` repos ship a
  `chat_template.jinja` (separate-file format), which transformers 5.x loads
  automatically — `LiquidAI/LFM2.5-350M-Base` applies the same `<|im_start|>`
  ChatML template as the instruct variant. No fallback needed on this stack.
  If you pin an older transformers (<4.43) that ignores `chat_template.jinja`,
  inject a template before training, or upgrade.
- **Sweep grid breadth.** The shared `sft_grid_small` (lr 2e-5–1e-4) may be too
  narrow for base variants, which often want different LRs — the whole premise
  of the comparison. The sweep supports `grid_override`; widen per-variant if
  the base models look under-trained.
- **Small-dataset cadence (handled).** Two thresholds in the training stack
  assume larger datasets, and the orchestrator now scales around both:
  - SFT evals/saves on a *step* schedule (default 50). `run.py` sizes
    `eval_steps`/`save_steps` to the dataset so the sweep's `eval_loss` proxy
    actually gets logged (otherwise every config is marked "failed").
  - The DPO sweep proxy (`eval_ce_chosen_mean`) is read from **iter_000**'s
    held-out split of the per-iteration preference pairs, and `split_train_eval`
    has a hard floor of 10 eval examples. On-policy preference yield is only
    ~0.13–0.37 pairs per train prompt per iter — and the **stronger the SFT
    model, the fewer pairs** it produces (a near-perfect greedy rollout rarely
    disagrees with the gold; the 1.2B-Instruct SFT yielded only ~13 pairs from
    100 prompts). So a small train set cannot produce 10 held-out pairs at any
    split ratio. `run.py` therefore **auto-skips DPO when `--train-size < 400`**
    (`_DPO_MIN_TRAIN_SIZE`), logging a warning and noting it in the report, and
    sizes `eval_split_ratio` off a conservative 0.13 yield when it does run.
    **For a DPO smoke use `--train-size 400`+**; the fastest SFT-only smoke is
    `--train-size 100 --dpo-train-size 0` (or `--skip-dpo`, or just let it
    auto-skip). DPO's real signal needs `--train-size` in the thousands, where
    pairs are plentiful.
