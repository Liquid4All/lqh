# Skill: RL (GRPO)

## Overview

GRPO (group-relative policy optimization) improves a model by generating
G candidate responses per prompt, judge-ranking them against the
project's scorer, and reinforcing the winners. No labels are needed —
the reward comes from the judge, so GRPO can improve on prompts that
have no gold answers, and it can squeeze out gains after SFT has
plateaued. It runs on LQH Cloud only (the `grpo` image carries the
vLLM + TRL runtime that local/SSH environments lack).

**Evidence status (2026-08-20, `tests/benchmarks/grpo_value/RESULTS.md`):**
the shipped recipe below is the first configuration to beat SFT across
3/3 seeds (+0.52 to +0.68 judge points over the SFT winner, above the
continued-SFT control, robust under a second judge). Every earlier
"GRPO doesn't help" result traced to a wrong knob, not the algorithm —
which is why the defaults matter so much here.

## When to use GRPO (and when not)

Reach for GRPO when ONE of these holds:

1. **SFT is plateaued** and fresh gold data is exhausted or expensive —
   GRPO extracts more from the same prompts without new labels.
2. **No labels exist** for the behavior you want — only a scorer that
   can rank responses.
3. **The objective is not expressible as cross-entropy** on a gold
   answer (e.g. "shorter while keeping the score", multi-criteria
   trade-offs the scorer encodes).

Do NOT reach for GRPO when:

- **Fresh gold data is still cheap.** Continued SFT on new data beats
  GRPO per GPU-hour until SFT saturates. Exhaust SFT first — the
  train skill's process is still the default path.
- **The scorer is weak or unvalidated.** GRPO amplifies whatever the
  judge rewards. A scorer you wouldn't trust for eval filtering will be
  reward-hacked, not learned from.
- **The base is a vision model** — GRPO v1 is text-only.

## The recipe is the defaults — do not override

The defaults in `lqh/train/defaults.py` + `lqh/train/grpo.py` encode a
measured, seed-replicated recipe. They RESOLVE DIFFERENTLY depending on
whether the run continues an SFT adapter or starts from a raw base, and
each ingredient was individually shown necessary:

| Knob | Continuing an SFT adapter | From a raw base | Why (measured) |
|---|---|---|---|
| Rollout sampling | T=0.3, min_p 0.05, rep 1.05 unless set | T=1.0, top_p=1.0, no min_p/rep-penalty | Exploration is the from-base lever (+0.83 vs +0.34); off a converged policy hot sampling with KL measured negative |
| KL β | **0 (no KL)** | 0.001 | With LoRA, TRL's KL reference is the adapter-DISABLED model = the RAW base — for continuation that anchor is wrong and pulls the policy away from the SFT solution (−0.19 vs +0.55) |
| Learning rate (LoRA) | 1e-5 | 1e-5 | 2e-6 provably learns nothing; 4e-5 destroys a converged policy (−1.61) |
| Training judge | **judge:medium** | judge:small | The judge that filtered the SFT data has no discrimination left post-SFT (3-seed null); one tier up passes the gate 3/3. From base the cheap judge suffices (medium is a 3-seed null there) |
| Steps / batch | 300 steps × 64 completions (8 groups × G=8) | same | At constant budget, MORE updates beat bigger batches (32-group × 75-step arm: +0.27 vs +0.83) |

Consequences for you:

- **Pass only `type`, `base_model`, `dataset`, `eval_dataset`,
  `scorer`.** Do not pass learning_rate, temperature, or beta unless
  the user prescribes values — the conditional defaults are the recipe.
- **The training-judge rule generalizes**: one tier ABOVE the judge
  that filtered/evaluated the data. If the project filtered with
  judge:medium, a continuation run should set
  `reward.judge_size: "large"` explicitly.
- Never "stabilize" a continuation run by adding KL back or lowering
  the temperature — both were measured to erase the gain.

## Default process

1. **Prerequisites** — a trained SFT checkpoint (the run's
   `model-lora`), a validated scorer, and a fresh prompt pool: a
   dataset of prompts (assistant turns are stripped; they need not be
   gold). ~2k prompts supports a full 300-step run. The prompt pool
   must be DISJOINT from the eval set.
2. **Baseline** — the SFT checkpoint's eval score on the held-out set
   (you usually have this from the training run; if not, run
   `start_local_eval` first). GRPO's gain is measured against this.
3. **Launch** — one run at defaults:

   ```
   start_training(
       type="grpo",
       base_model="runs/sft_001/model-lora",   # the SFT winner
       dataset="datasets/task_rl_prompts",      # fresh prompt pool
       eval_dataset="datasets/task_eval",
       scorer="evals/scorers/task_v1.md",
   )
   ```

   The scorer is MANDATORY — it is the reward, not just the final
   grade. There is no sweep for GRPO (one run at defaults).
4. **Evaluate** — after completion, `training_status` for the judge
   score, then compare GRPO vs the SFT baseline. The honest bar (the
   one the benchmark used): the gain should also survive scoring under
   a judge one size larger than the eval judge — a gain that exists
   only under the training judge is reward hacking. If fresh gold data
   was available, a continued-SFT run on the same pool is the fair
   control.

## Cost and duration — tell the user before launching

A default 300-step run makes ~19,200 rollouts and roughly **22k judge
calls** (~2.4k group-rank + ~19.2k pointwise anchor). For a
continuation run these are `judge:medium` calls — this is the dominant
cost after GPU time, and it is why GRPO is not the first tool to reach
for. Wall clock is judge-latency-bound: expect **4–8 h** on the cloud
A100 for 300 steps at a 1.2B base. Completions are capped at 512
tokens; raise `grpo.max_completion_length` only for long-output tasks
(watch the clipped-ratio warning below).

## Reading a run (diagnostics)

The run dir carries `grpo_diagnostics.jsonl` (one line per log step)
and a `rewards/` ledger of every judge call. What matters:

- **`frac_reward_zero_std`** — fraction of dead groups (all G
  completions rewarded identically → zero gradient). Healthy: ~0.
  Persistently high → the judge cannot distinguish the completions:
  wrong judge tier (too small) or the task is saturated.
- **`completions_clipped_ratio`** — completions hitting the length
  cap. The trainer masks truncated completions, so a fully-clipped
  step trains on NOTHING (a loud warning prints once). Persistent
  clipping → raise `grpo.max_completion_length`.
- **`kl`** — for continuation runs (β=0) this is informational drift;
  from-base (β=0.001) it should stay small (~0.01). An exploding KL
  with a falling reward is a broken run, not progress.
- **`judge_rank_failures` / `judge_rank_parse_retries`** — the reward
  engine retries malformed judge responses (temp-escalating). A
  failure fraction over 50% of recent completions aborts the run as
  `RewardChannelDown` (rate limits, auth, judge outage) — that is an
  infrastructure failure, not a training signal; see the job_recovery
  skill, fix the channel, resubmit.
- **Training reward trend** — the headline `reward` should drift up
  over the run. Flat reward with healthy diagnostics after ~100+ steps
  usually means the judge has no signal left at this policy strength:
  step up the judge tier before touching any other knob.

## Failure modes → correct responses

| Symptom | Wrong response | Right response |
|---|---|---|
| Final score ≈ baseline | raise learning rate | step the training judge one tier up; check dead-group fraction |
| Score up on training judge, flat/down one tier up | ship it | it's reward hacking — treat as a null result |
| Run aborted `RewardChannelDown` | retry blindly | job_recovery skill; the reward channel (judge API) was down |
| Score DROPPED vs baseline (continuation) | add KL back | verify β resolved to 0 and sampling to the continuation profile (run config shows resolved values); a hand-set T=1.0 with β>0 is the measured way to lose points |
| All-clipped warning in logs | ignore | raise `grpo.max_completion_length` above the task's output length |

## Relationship to the other skills

- **train** — produces the SFT checkpoint GRPO starts from; its
  process (data quality → filter → baseline → SFT) is still the
  prerequisite path. GRPO is Phase 4, after SFT (and instead of DPO
  when no preference-worthy gold data exists).
- **failure_analysis / improve** — decides WHETHER another training
  round is worth it; GRPO is one of its options once SFT is plateaued.
- **evaluation** — the eval suite and scorer it maintains are both the
  measuring stick and (one tier up) the reward.
