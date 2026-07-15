# DPO incremental-value benchmark

This benchmark answers one narrow question: after SFT, does DPO on fresh data
improve `voice_satisfaction` more than simply continuing SFT on that same fresh
data?

It independently generates four scorer-filtered splits:

1. SFT training data;
2. fresh DPO/continued-SFT training data;
3. fixed validation data used for checkpoint and hyperparameter selection;
4. a final test set used only after winners are selected.

The DPO baseline is greedy on-policy generation. Candidate sampling at high
temperature is intentionally not part of this benchmark. Preferences require a
same-judge chosen-minus-rejected gap of at least 1.0, and DPO uses effective
batch 16 so a useful preference set yields tens of updates instead of two.

The report preserves per-example judge scores and gives paired bootstrap 95%
confidence intervals for `DPO - SFT`, `continued SFT - SFT`, and
`DPO - continued SFT`. A DPO gain is called demonstrated only when its mean is
at least +0.3 and the paired interval excludes zero. Results should be checked
across all three default training seeds. For `voice_satisfaction`, the report
also computes deterministic JSON validity, score-direction accuracy,
frustration-miss rate, failure-tag exact match, and failed-turn exact match.

Full run:

```bash
uv run python -m tests.benchmarks.dpo_value.run
```

Small plumbing smoke test:

```bash
uv run python -m tests.benchmarks.dpo_value.run \
  --sft-train-size 200 --dpo-train-size 400 \
  --validation-size 40 --test-size 40 --seeds 17 --grid-size tiny
```

The benchmark defaults to `LiquidAI/LFM2.5-1.2B-Instruct` and
`voice_satisfaction`. Use `--workdir` and the default resume behavior to
continue an interrupted run.
