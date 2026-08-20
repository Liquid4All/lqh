# GRPO value benchmark — results (2026-08-16)

Committed verbatim from the run artifacts (`~/.lqh-grpo-value/full1` on
toka) so the numbers behind any default or product decision are
auditable in-repo — the hp_defaults provenance lesson.

Setup: `voice_satisfaction`, `LiquidAI/LFM2.5-1.2B-Instruct`, seeds
17/29/41, splits sft_train 10k / rl_train 2k / validation 200 / test 400
(399 after cross-split dedupe), scorer-filtered ≥7.0, sweep grid
`small`, GRPO 300 steps × 64 completions/step at G=8, T=0.3,
`judge:small` training judge, `judge:medium` robustness judge. GRPO arm
trained under the grpo-venv (vllm 0.26.0, trl 1.10.0, transformers
5.15) on an RTX 6000 Ada; ~12.5 h per seed end-to-end.

## Main result (3 seeds, shipped GRPO defaults: lr 2e-6, β 0.005)

| Seed | SFT | Continued SFT | grpo_rank | grpo_rank−SFT [95% CI] | grpo_rank−continued [95% CI] |
|---:|---:|---:|---:|---:|---:|
| 17 | 7.59 | 7.73 | 7.62 | +0.03 [−0.18, +0.24] | −0.12 [−0.33, +0.10] |
| 29 | 7.52 | 7.61 | 7.51 | −0.01 [−0.22, +0.21] | −0.09 [−0.31, +0.14] |
| 41 | 7.48 | 7.60 | 7.62 | +0.13 [−0.07, +0.34] | +0.03 [−0.19, +0.24] |

Mean grpo_rank−SFT: **+0.05**. Gate (≥ +0.3, CI excluding zero, not
behind continued-SFT, positive under the robustness judge): **0/3 seeds
pass.** Continued-SFT on the same fresh pool beat SFT by ~+0.11 mean and
edged GRPO in 2/3 seeds. No format regressions in any arm (JSON validity
100% everywhere — the guard penalties did no harm). Voice-satisfaction
sub-metrics flat across arms within noise.

Training health across all GRPO runs: zero dead groups
(`frac_reward_zero_std` = 0 throughout), within-group pointwise judge
std ~0.25, ~0 judge-call failures over ~100k reward calls. The reward
CHANNEL worked; the question was whether its signal helps.

## Update-budget probe (seed 17, lr 1e-5, β 0.001 — 5× the movement)

| Arm | Test score | Final training KL |
|---|---:|---:|
| grpo_rank (shipped defaults) | 7.62 | ≈0.005 |
| grpo_probe (hot budget) | **7.48** | ≈0.130 |

The probe resolves the null's ambiguity: with 26× the policy movement,
the score went DOWN. The 3-seed null is not under-training — the
rank-judge reward has no exploitable signal on this task at this
maturity, and the conservative KL leash was protecting the checkpoint.

## From-base signal trial (seed 17, 2026-08-16)

`--from-base` mode: GRPO trains directly on the raw
`LFM2.5-1.2B-Instruct` (no SFT), so the reward has ~3 judge points of
headroom to show any signal. Same 2k RL pool, 300 steps, both
update-budget points:

| | Raw model | GRPO lr 2e-6 / β 0.005 | GRPO lr 1e-5 / β 0.001 | SFT reference |
|---|---:|---:|---:|---:|
| judge:small | 4.48 | 4.55 (+0.06 [−0.10, +0.23]) | **4.82 (+0.34 [+0.15, +0.52])** | 7.59 |
| judge:medium Δ vs raw | — | +0.01 | **+0.28** | — |

**The reward has real, learnable signal**: the hot arm's gain excludes
zero and replicates under the second judge (not hacking). But (a) the
then-default lr 2e-6 provably learns nothing in ANY regime tested —
defaults.py was updated to lr 1e-5 / β 0.001 on this evidence — and
(b) the effect is ~10% of what SFT extracts from the same model
(+0.34 vs +3.11): the prompted-judge signal is weak (consistent with the
G0.2 rank self-agreement of tau ≈ 0.1–0.2 — a small consistent
component under heavy noise), and it is swamped entirely once the policy
is post-SFT strong (main table above).

## Best-of-G exploration ceiling (raw 1.2B, 60 test prompts, 2026-08-17)

G=8 samples per prompt under the training sampling params, every sample
judged pointwise:

| | T=0.3 (shipped) | T=0.8 |
|---|---:|---:|
| mean of 8 | 4.24 | 4.09 |
| best of 8 (per-round ceiling) | 5.98 | 6.12 |
| groups containing a ≥7 sample | 42% | 38% |

The per-round selection ceiling is ~6.0: a PERFECT judge reinforcing
best-of-8 had +1.75 of span available per round; GRPO's +0.34 captured
~20% of it — judge fidelity (tau ≈ 0.15), not exploration, is the
binding constraint. Raising temperature buys no ceiling (5.98→6.12,
noise) while degrading the mean — the LFM low-temperature rule costs
GRPO nothing. 42% of groups already contain an SFT-quality completion;
the system's failure mode is recognizing winners, not generating them.

## Exploration study — production-RL sampling profile (seed 17, 2026-08-18)

Motivated by the production LFM-release GRPO settings (Slack, 2026-08-14:
rollouts at T=1.0/top_p=1.0, no min_p/rep-penalty, KL in {0.001, 0.01, 0},
"below 0.7 messes up the exploration"). All arms use the pure sampling
profile (top_p=1.0, min_p=0, repetition_penalty=1.0 — knobs added to
grpo.py for this study); 300 steps, G=8, same splits/judges/controls as
above, so numbers are directly comparable. Run via
`exploration_study.py` split across two boxes.

From raw model (control: raw 4.48; SFT +3.12 [+2.84, +3.41]):

| Arm | T | lr | β | Test | Δ vs raw [95% CI] | Δ judge:medium |
|---|---:|---:|---:|---:|---:|---:|
| T=0.3 shipped (above) | 0.3 | 1e-5 | 0.001 | 4.82 | +0.34 [+0.15, +0.52] | +0.28 |
| base_t10 | 1.0 | 1e-5 | 0.001 | **5.32** | **+0.83 [+0.61, +1.06]** | +0.73 |
| base_t07 | 0.7 | 1e-5 | 0.001 | 4.51 | +0.02 [−0.18, +0.22] | +0.54 |
| base_t10_hot | 1.0 | 4e-5 | 0.001 | 4.94 | +0.47 [+0.18, +0.76] | **+1.21** |
| base_t10_nokl | 1.0 | 1e-5 | 0 | 4.65 | +0.17 [−0.04, +0.38] | +0.63 |

From the SFT winner (control: SFT 7.59; continued-SFT 7.73;
T=0.3 SFT+GRPO 7.62):

| Arm | T | lr | β | Test | Δ vs SFT [95% CI] | Δ judge:medium |
|---|---:|---:|---:|---:|---:|---:|
| sft_t10 | 1.0 | 1e-5 | 0.001 | 7.40 | −0.19 [−0.47, +0.08] | −0.77 |
| sft_t07 | 0.7 | 1e-5 | 0.001 | 7.91 | +0.32 [+0.05, +0.59] | −0.00 |
| sft_t10_hot | 1.0 | 4e-5 | 0.001 | 5.98 | −1.61 [−1.90, −1.33] | −2.10 |
| sft_t10_nokl | 1.0 | 1e-5 | 0 | **8.14** | **+0.55 [+0.30, +0.79]** | +0.18 |

Findings (single seed — screening evidence, not yet default-moving):

1. **Exploration temperature was the binding sampling constraint.** From
   base, T=1.0 pure sampling extracts 2.4× the shipped T=0.3 gain
   (+0.83 vs +0.34) at identical lr/KL/steps, replicating under the
   second judge. The LFM low-temperature discipline (MODELS.md) is right
   for *serving* but was costing GRPO most of its training signal.
2. **`sft_t10_nokl` is the first SFT+GRPO arm to clear the value gate**
   (+0.55, CI excluding zero, ABOVE the continued-SFT control 7.73,
   positive under judge:medium). The 3-seed main-table null was partly a
   sampling/KL artifact, not purely judge-fidelity ceiling.
3. **KL direction flips with the starting point**: from base, β=0.001
   beats β=0 (+0.83 vs +0.17); from SFT, β=0 beats β=0.001 (+0.55 vs
   −0.19). With the reference = the SFT policy, the KL term pins the
   model to exactly the behavior GRPO must move away from.
4. **lr 4e-5 (LoRA) is destructive from a converged SFT policy** (−1.61)
   and trades training-judge score for robustness-judge score from base.
   lr 1e-5 stays the right LoRA setting; the production full-FT grid
   (2e-8..2e-6) does not transfer upward beyond that.
5. T=0.7 is NOT a safe middle from base (+0.02 null) — the "≥0.7" rule
   from production does not transfer; it's T≈1.0 or stay home.

**Confirmation of sft_t10_nokl (seeds 29/41, 2026-08-18): FAILS the
gate — 1/3 seeds.**

| Seed | SFT | sft_t10_nokl | Δ vs SFT [95% CI] | Δ judge:medium |
|---:|---:|---:|---:|---:|
| 17 | 7.59 | 8.14 | +0.55 [+0.30, +0.79] | +0.18 |
| 29 | 7.52 | 7.64 | +0.12 [−0.13, +0.37] | −0.33 |
| 41 | 7.48 | 7.42 | −0.05 [−0.33, +0.22] | −1.11 |

Mean +0.21; the seed-17 pass was a favorable draw. Worse, the
robustness-judge deltas on 29/41 are negative — at T=1.0/β=0 the arm can
actively degrade quality as seen by a second judge while holding its
training-judge score. Post-SFT GRPO remains not reliably valuable on
this task even with exploration-corrected hyperparameters; the
judge-fidelity ceiling (Interpretation below) reasserts itself once the
policy is strong. No default changes from metric 2.

**Replication of base_t10 (seeds 29/41, 2026-08-18): CONFIRMS — 3/3
seeds.**

| Seed | Raw | base_t10 | Δ vs raw [95% CI] | Δ judge:medium |
|---:|---:|---:|---:|---:|
| 17 | 4.48 | 5.32 | +0.83 [+0.61, +1.06] | +0.73 |
| 29 | 4.48 | 5.22 | +0.73 [+0.52, +0.94] | +1.15 |
| 41 | 4.48 | 5.58 | +1.10 [+0.87, +1.35] | +1.17 |

Mean +0.89 (~29% of SFT's ~+3.06 from the same model, up from ~11% at
T=0.3), every CI far from zero, and the judge:medium deltas are as large
or larger — learning, not hacking. **Defaults changed on this evidence**
(defaults.py / grpo.py): from-base GRPO runs now default to the
exploration profile (T=1.0, top_p=1.0, min_p=0, repetition_penalty=1.0);
adapter-continuation runs keep the conservative LFM profile (T=0.3,
min_p=0.05, rep 1.05), where T=1.0 measured negative and T=0.3 is
do-no-harm. lr 1e-5 / β=0.001 unchanged for both.

Not varied here: num_iterations. Groups/step and judge fidelity were
probed next (below).

## Follow-up probes (seed 17, from base, T=1.0 profile, 2026-08-19)

Both single-seed, compared against base_t10 (5.32, +0.83):

**Batch scale (base_t10_bs32): more groups/step LOSES at constant
budget.** 32 groups × 75 steps — the SAME 19.2k-completion rollout/judge
spend as 8 × 300 — scored 4.75 (+0.27 [+0.07, +0.47], robust +0.17).
The production advice "small batches update too frequently" does not
transfer at our budget: update COUNT dominates update quality (the DPO
lesson again). 75 steps is below the 100-step health floor, but that
confound IS the tradeoff being measured. Production batch sizes
presumably work because their budgets afford big batches AND hundreds
of steps.

**Judge fidelity (base_t10_jm): training on judge:medium is the best
from-base result measured.** Same config as base_t10 but judge:medium
as the training reward: 5.58 (+1.10 [+0.91, +1.29] under judge:small —
the INDEPENDENT judge for this arm — and +1.35 under medium). Direct
paired comparison vs base_t10: **+0.26 [+0.04, +0.48]**, significant
despite the eval judge favoring base_t10 (its own training judge).
Against the measured best-of-8 selection ceiling (~5.98 at T=0.3):
small-judge training captured ~56% of the raw→ceiling span, medium
~73%. Caveats: judge:medium broke the ranking schema on ~8% of rank
calls (answered in the scorer's pointwise format; those groups train
nothing — failure rate well under the 50% abort budget), p50 rank
latency 7.2s vs 4.7s, and higher per-call cost. Judge fidelity remains
the ceiling, and buying more of it measurably works.

KL default note (2026-08-19): TRL 1.10 computes the PEFT reference via
`disable_adapter()` — the KL anchor is the base UNDER the adapter, not
the training start. Correct from-base (keep β 0.001, measured +0.83 vs
+0.17 at β=0); wrong for continuation (anchors to the raw base, away
from the SFT solution — measured −0.19 at β 0.001 vs +0.55 at β=0,
T=1.0 seed 17). `GRPO_BETA_CONTINUATION = 0.0` shipped on this
mechanism + measurement.

## Judge-fidelity study (2026-08-19/20) — the post-SFT null OVERTURNED

Training-judge probes at the confirmed exploration profile (T=1.0 pure
sampling, lr 1e-5). Ops note: judge:medium's upstream (qwen3.5-27b)
degraded to 35% broken-JSON on 08-19 (digit-loop ints, wrapped arrays);
the reward engine gained a parse-retry (temp 0 → 0.3 → 0.5, 3 attempts
— a greedy retry would deterministically repeat the failure) which held
net failures under ~10% and is now permanent.

**From base: fidelity buys nothing (3-seed null).** base_t10_jm
(judge:medium reward, β 0.001) vs same-seed base_t10 under judge:small
— the independent judge for jm: +0.26 / +0.10 / −0.32, mean ≈ 0.
base_t10_jl (judge:large, seed 17) = 5.66, matching not beating jm's
5.58. The T=1.0 exploration gain is the from-base lever; a costlier
training judge is not.

**Post-SFT: judge:medium passes the value gate 3/3 seeds — the first
replicated SFT+GRPO win.** sft_t10_nokl_jm = the best-known
continuation config (T=1.0 pure sampling, β=0, lr 1e-5) with
judge:medium as the training reward. Test scores under judge:small —
the judge it was NOT trained on:

| Seed | SFT | Cont.-SFT | sft_t10_nokl (small judge) | sft_t10_nokl_jm | Δ vs SFT [95% CI] | Δ robust |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 7.59 | 7.73 | 8.14 | **8.28** | +0.68 [+0.43, +0.94] | +0.95 |
| 29 | 7.52 | 7.61 | 7.64 | **8.15** | +0.62 [+0.39, +0.86] | +0.86 |
| 41 | 7.48 | 7.60 | 7.42 | **8.00** | +0.52 [+0.28, +0.76] | +0.46 |

Every CI excludes zero, every robustness delta is positive, every seed
beats the continued-SFT control (mean +0.55 over continued-SFT). The
same config with the small training judge was +0.55/+0.12/−0.05 — a
1/3 fluke.

**Mechanism.** judge:small filtered the SFT data (≥7.0 threshold) and
is the eval judge — post-SFT, the policy sits exactly where that
judge's discrimination is exhausted, so its rank signal is noise (the
original 3-seed null, tau ≈ 0.15). judge:medium still resolves
differences there, and the gain transfers to the independent judge.
From base the policy is far below EITHER judge's ceiling, so the cheap
judge suffices. Rule of thumb shipped as a default: the GRPO training
judge should sit one tier ABOVE the judge that filtered/evaluated the
data (grpo.py resolves reward.judge_size to medium for
adapter-continuation runs, small from-base).

**The full replicated continuation recipe** (all defaults now):
T=1.0 / top_p=1.0 / no min_p / no repetition penalty · β=0 (TRL's PEFT
KL anchor is the raw base — wrong for continuation) · lr 1e-5 LoRA ·
judge one tier up. Each ingredient measured necessary on this task:
T=0.3 → null; β=0.001 → negative; lr 4e-5 → destructive; small judge →
1/3 fluke.

## Interpretation

1. **Context that moved under the plan:** the task was chosen for
   headroom (SFT ≈ 5.67 in the planning docs); at 10k scorer-filtered
   rows SFT reaches ~7.5, so every post-SFT method is fighting for
   scraps — continued-SFT's own gain is only ~+0.11.
2. **The DPO conclusion replicates for GRPO** (the plan's G2 fallback,
   verbatim): for these task/data shapes the bottleneck is supervised
   data quality, not the alignment objective. Fresh data spent on
   continued SFT is worth more than the same data spent on GRPO.
3. **Consequence:** Phase 5 (agent-facing `start_training(type="grpo")`)
   stays gated. GRPO remains available headless (config `type: "grpo"`,
   cloud kind `train_grpo`) for settings where its regime actually
   differs — programmatic/verifiable rewards, tasks where SFT is far
   from ceiling, or reward functions a supervised loss can't express.
4. **Not measured here:** GRPO with programmatic rewards, weaker-SFT
   starting points, other tasks, judge:medium/large as the training
   reward, or the `grpo_pointwise`/`guards_only` ablations (moot with a
   null primary arm).
