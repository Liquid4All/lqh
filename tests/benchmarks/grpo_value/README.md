# GRPO incremental-value benchmark

Answers one narrow question: after SFT, does GRPO on fresh prompts improve
the task more than simply continuing SFT on that same fresh data?

Fork of `dpo_value` (GRPO plan Phase 4). Four independently generated,
scorer-filtered, disjoint splits: SFT training data; a fresh RL prompt
pool (shared by GRPO and the continued-SFT control); fixed validation
data; a final test set. Arms per seed:

1. **SFT** — sweep on the SFT split (the baseline checkpoint);
2. **Continued SFT** — lr×epochs grid on the fresh pool from the SFT
   checkpoint (the extra-data control that decides the question);
3. **GRPO arms** — same checkpoint, same pool: `grpo_rank` (group-rank
   judge + pointwise anchor + guards), optional ablations
   `grpo_pointwise` and `guards_only` via `--arms`.

Final-test evals run under the training judge AND a second robustness
judge (`--robustness-judge-size`, default medium): a gain that vanishes
under a different judge is reward hacking (plan gate G3). A seed
demonstrates GRPO value only when GRPO−SFT ≥ +0.3 with the paired 95%
interval excluding zero, GRPO ≥ continued-SFT, and the robustness delta
is positive. Check across all seeds.

The GRPO trainer cannot run in the dev environment (needs vLLM + trl
1.10 — see `GRPO_IMPLEMENTATION.md`); it runs as a subprocess of
`--grpo-python`, a dedicated venv mirroring the production grpo image:

```bash
uv venv ~/grpo-venv -p 3.12
uv pip install --python ~/grpo-venv/bin/python vllm==0.27.1 trl==1.12.0 \
    "peft>=0.20" accelerate datasets pyarrow "openai>=2.0,<3" "httpx>=0.27" \
    "prompt_toolkit>=3.0" "rich>=13.0" "pillow>=10.0" "packaging>=24.0" \
    "huggingface_hub>=0.20" "lm-format-enforcer>=0.11" hf_transfer
uv pip install --python ~/grpo-venv/bin/python --no-deps -e <lqh_py checkout>
```

Full run (defaults: voice_satisfaction, LFM2.5-1.2B-Instruct, 3 seeds):

```bash
uv run python -m tests.benchmarks.grpo_value.run
```

Pilot / plumbing smoke:

```bash
uv run python -m tests.benchmarks.grpo_value.run \
  --sft-train-size 2000 --rl-train-size 400 \
  --validation-size 60 --test-size 100 --seeds 17 \
  --grid-size tiny --grpo-max-steps 60
```

The task default is `voice_satisfaction` deliberately: the Phase-0.2
spike showed that on a saturated task (translation) an SFT-adjacent
policy's same-prompt samples are quality-equivalent, so the group-rank
signal starts as noise. Watch `frac_reward_zero_std` and the per-reward
stds in each GRPO run's `grpo_diagnostics.jsonl` — flat group variance
throughout a run means the task has no headroom for GRPO regardless of
hyperparameters. Use `--workdir` and the default resume behavior to
continue an interrupted run.
