# GRPO implementation plan (replacing DPO in the post-SFT stage)

Status: design sketch, nothing implemented. Written 2026-08-06 against
`lqh_py` @ v0.9.2 (trl 1.0.0 locked, transformers >=5.0,<5.4, peft >=0.15)
and the `lqh` backend at `d049127`. Since 2026-09 (lqh 0.17) the shared
stack is trl 1.12 / transformers 5.16 / torch 2.13 and the grpo image is
vLLM 0.27.1 + trl 1.12; version-specific remarks below are historical.

Companion docs: `DPO_PLAN.md` (why DPO failed and what the measurement
protocol must look like), `../TRAINING_INFRA.md`, `../INFERENCE.md`,
`../ISSUE_4_PLAN.md` (the `gpu_eval` image precedent, which is the single
most relevant piece of prior art here).

---

## 0. Executive summary

**GRPO is buildable on this stack, and the algorithm itself is the easy
part.** TRL 1.0.0 — the version already in `uv.lock` — ships `GRPOTrainer`
on its stable API surface, with native PEFT support, native async reward
functions, and vLLM integration. The trainer is a normal HF `Trainer`
subclass, so `lqh/train/resume.py::train_with_checkpoint_fallback` and the
whole preemption/continuation story work unchanged. That is a much smaller
integration than DPO was: GRPO is one `trainer.train()` call, not a
generate → ship-to-laptop → score → write-parquet → resume ping-pong.

**Three things are genuinely hard, in this order:**

1. **The rollout engine.** GRPO needs `num_generations × prompts_per_step`
   samples *every optimizer step*. The current DPO rollout path
   (`lqh/train/dpo.py::_generate_predictions`) is a **batch-size-1, greedy
   `model.generate()` loop**. At GRPO volumes that is not slow, it is
   arithmetically impossible (§3). We need vLLM (or sglang) in the training
   sandbox, and that collides head-on with the pin discipline that made
   `gpu_eval` a separate image in the first place. **This is the critical
   path and the main reason to start with a spike, not with code.**

2. **The reward.** A prompted API judge is the right call and the plumbing
   already exists (`lqh/train/cloud_score.py` runs scoring *inside* the
   sandbox against `judge:*` with a scoped token). But our judge emits a
   **coarse 1–10 absolute score**, and GRPO's advantage is the
   *within-group* deviation. Coarse absolute scores collapse group variance
   → zero advantage → no gradient. The fix is to stop scoring pointwise and
   start judging the group *relatively* (§4). This is also 8× cheaper.

3. **Signal, not infrastructure.** DPO didn't fail because TRL was broken;
   it failed on update budget, pair construction, and selection protocol.
   GRPO inherits every one of those failure modes in a new costume. The
   measurement protocol from `DPO_PLAN.md` §"Validation matrix" is
   non-negotiable and should be ported before the first real run.

**Recommendation:** three-week-ish shape — one week de-risking the image +
a learnability spike, one week landing `lqh/train/grpo.py` + the reward
module, one week on cloud wiring and the value benchmark. Do **not** wire
the agent-facing `start_training(type="grpo")` surface until the benchmark
clears the gates in §9.

---

## 1. What already exists and transfers

Considerably more than for DPO. Inventory:

| Need | Already have | Where |
|---|---|---|
| Cloud job submit / lease / preemption / artifacts | yes | `lqh/remote/cloud.py`, `backend/internal/cloud/runner.go` |
| Resume from HF Trainer checkpoint after preemption | yes | `lqh/train/resume.py` |
| Progress reporting with phase weights | yes | `lqh/progress.py`, `lqh/train/progress.py` |
| In-sandbox LLM judge with scoped token | **yes** | `lqh/train/cloud_score.py`, `lqh/scoring.py::run_scoring` |
| Backend gate so job tokens can only call judges | yes | `handler/chat.go` — `chat.score` scope → `IsJudgeModel` only |
| Judge model tiers | yes | `judge:small\|medium\|large` → qwen3.5-9b / 27b / 397b-a17b (`backend/internal/model/pool.go`) |
| Adapter-aware model loading, merge, save | yes | `lqh/train/load_model.py`, `lqh/remote/merge_lora.py` |
| LoRA target modules for LFM2 (incl. `in_proj`/`out_proj`/`w1..w3`) | yes | `lqh/train/defaults.py::TEXT_TARGET_MODULES` |
| Ephemeral in-sandbox inference server pattern | **yes** | `lqh/infer/engine_sglang.py::_SglangServer` |
| Sweep harness + held-out eval-of-best | yes | `lqh/train/sweep.py`, `lqh/train/dpo_sweep.py` |
| Value-benchmark scaffold w/ paired bootstrap | yes | `tests/benchmarks/dpo_value/` |
| OOM self-heal + batch calibration | yes | `lqh/train/calibrate.py` |

The pieces that do **not** exist: a fast batched rollout path, a reward
function abstraction, a `grpo` image, and the `train_grpo` job kind.

---

## 2. TRL 1.0.0 — verified capabilities

Checked directly against the locked wheel
(`~/.cache/uv/archive-v0/OTKfzaDrmscoLCoD/trl`, `Version: 1.0.0`), not
just docs:

- `GRPOTrainer(..., rollout_func=...)` **exists** (`grpo_trainer.py:277`),
  guarded by an experimental warning at line 421. A rollout func returns
  `prompt_ids` / `completion_ids` / `logprobs` and can also supply
  `env_mask` and `extra_fields`. This is the escape hatch that lets us
  point generation at *our own* server instead of TRL's vLLM.
- **Async reward functions are first-class** (`grpo_trainer.py:1189`,
  `:1203-1218`). Coroutine reward funcs are detected via
  `inspect.iscoroutinefunction`, batched into `asyncio.gather`, and run on
  a persistent daemon loop. Our judge calls are `AsyncOpenAI` calls
  already — they drop straight in.
- **`None` rewards become `NaN`, not `0.0`** (`:1198`, `:1210`). Critical:
  a judge API failure must return `None`, never `0.0`, or we teach the
  policy that a 429 is a bad answer. `lqh/scoring.py::is_scoring_error`
  already distinguishes `[Scoring error]`/`[Parse error]` from genuine low
  scores — wire it to `None`.
- `GRPOConfig` defaults worth knowing: `num_generations=8`, `beta=0.0`
  (**reference model not loaded at all** unless beta>0),
  `loss_type="dapo"`, `scale_rewards="group"`,
  `importance_sampling_level="token"`, `epsilon=0.2`,
  `max_completion_length=256`, `use_vllm=False`, `vllm_mode="colocate"`,
  `vllm_gpu_memory_utilization=0.3`, `num_iterations=1` (μ).
- PEFT: pass `peft_config` to the trainer, same as our SFT path.
- Not in 1.0.0: `entropy_coef` / adaptive entropy (those are on `main`),
  and there is no async/disaggregated trainer — GRPO in TRL is fully
  synchronous, generation blocks training.

The `<1.1` pin is fine. The `>=1.0` floor is what matters, and we're on it.

---

## 3. The rollout engine — the critical path

### 3.1 Why the current generation path cannot be reused

`lqh/train/dpo.py::_generate_predictions` loops one conversation at a time
with `do_sample=False`, `model.generate(input_ids, ...)` — micro-batch 1.
DPO tolerates this because it generates ~350–1000 completions *per
iteration*, five times per run.

GRPO's arithmetic is different. A modest run — 8 prompts × G=8 = 64
completions per optimizer step, 300 steps — is **19,200 completions**, and
they must be interleaved with training, not batched once. At a generous
40 tok/s for unbatched HF generate on a 1.2B model and 300 completion
tokens, that is ~40 hours of pure generation for a run whose gradient work
is maybe 20 minutes. Batching HF generate to 32 buys maybe 10–15×, which
still leaves generation as ~90% of wall clock and burns H100-hours to do it.

**A real inference engine is not an optimization here. It is the
feasibility condition.** Everything else in this document is downstream of
solving it.

### 3.2 The pin conflict (read `ISSUE_4_PLAN.md` first)

We already know this problem. `backend/scripts/modal_build_image.py` builds
`gpu_eval` **from the pinned sglang base** and has a shouted comment: torch
/ transformers / numpy come from that base and **must not be reinstalled**,
because our training pins (`transformers>=5.0,<5.4`) would downgrade
sglang's stack. The `gpu_eval` image exists precisely because merging
"training pins" and "serving pins" into one environment was judged too
risky, and `modal_serve_harness.py` exists because "Modal + sglang + LFM is
brittle".

GRPO needs *both* stacks in one sandbox. Three ways out:

**Option A — new `grpo` image built FROM the vLLM base (recommended).**
Mirror exactly what `gpu_eval` did with sglang: take vLLM's published
image as the base, install `trl==1.0.0`, `peft`, `accelerate`, and `lqh_py`
on top, and do **not** touch torch/transformers. Accept vLLM's transformers
pin instead of ours.

- vLLM supports LFM2 natively as `Lfm2ForCausalLM` since v0.23.0, with
  published recipes for `LiquidAI/LFM2.5-1.2B-Base` and `LFM2.5-230M` — no
  `--trust-remote-code` needed (which matters, because TRL issue #4129 says
  you *can't* pass `trust_remote_code` to vLLM in colocate mode).
- Risk (historical): the `<5.4` transformers pin of the time existed because
  5.12 broke `lfm2_vl` generation, a **VLM** constraint. The pin is gone
  (lqh 0.17 runs transformers 5.16); taking vLLM's transformers still needs a
  smoke test rather than an assumption (`lqh/train/sft.py` and
  `load_model.py` run in this same image).
- Cost: one more image purpose to build/promote/register on every dep
  change (`--purpose grpo`, refusing to combine, same as `gpu_eval`).

**Option B — two virtualenvs in one image (fallback).** `/opt/vllm-venv`
serves, `/opt/train-venv` trains, sandbox runs both. This is what
`_SglangServer` already implies structurally — it `Popen`s
`sys.executable -m sglang.launch_server` as a *separate process*, so
pointing it at a different interpreter is a small change. Fully decouples
the pins. Cost: weight sync becomes cross-process (see below), and image
size roughly doubles.

**Option C — external inference (scale-out, later).** Use TRL's
`rollout_func` to call a *deployed* sglang pod — we already run those
(`backend/scripts/modal_inference_pod.py`), already do dynamic LoRA
loading, and already have a regression gate for the combination. Weight
sync = save adapter → push → `load_lora_adapter`. Zero pin conflict, real
per-step latency (adapter save + transfer + load, order 10–30s/step → 1–2.5h
over 300 steps), and it needs a second GPU lease. Worth keeping in the back
pocket for 8B+ models; overkill for 350M–1.2B.

**Recommendation: A for v1, B as the pre-planned fallback, C documented but
not built.** Land A behind a `modal_grpo_harness.py` regression gate in the
same spirit as `modal_serve_harness.py`: new pin → run the harness → promote.

### 3.3 Known TRL + vLLM colocate landmines

Verify each of these in the spike; all are documented open issues:

- **PEFT + colocate + multi-GPU hangs** (trl#3671): trainer completes one
  or a few iterations then hangs silently. Reported as multi-GPU only.
  → **Stay single-GPU for v1.** Our models are 350M–1.2B; one H100 is
  plenty. This is a real constraint, not a preference.
- **Hardcoded `MASTER_PORT=12345`** (trl#3979): two colocate runs on one
  host collide. → Matters for *sweeps*. If a GRPO sweep ever runs N configs
  on one machine, they must be separate sandboxes or get distinct ports.
- **vLLM logprobs ignore temperature** (trl#4159) → severe instability at
  temperature ≠ 1. Needs `logprobs_mode="processed_logprobs"` on
  vLLM ≥0.10.2. TRL 1.0 mitigates the general train/infer mismatch with
  `vllm_importance_sampling_correction=True` by default — confirm it's
  actually on and that our temperature is handled.
- **`max_prompt_length` doesn't truncate conversational datasets under
  vLLM** (trl#4358, since 0.25.0). Our datasets are ChatML. → Truncate
  ourselves before handing prompts to the trainer.
- Memory: `vllm_gpu_memory_utilization=0.3` is conservative but the policy,
  the optional reference model (only if `beta>0`), optimizer state, *and*
  the KV cache share the card. `lqh/train/calibrate.py` calibrates for
  SFT's memory profile and will be wrong here. Set GRPO batch sizes
  explicitly at first; add a GRPO mode to `calibrate.py` later, or accept
  the existing OOM self-heal (`report_oom_downgrade`) as the safety net.
- `vllm_enable_sleep_mode=True` offloads vLLM weights/cache during the
  optimizer step — likely worth enabling on a shared card.

---

## 4. The reward model — the interesting problem

### 4.1 What we have

`lqh/train/cloud_score.py` already solves the authorization and topology
question: a training subprocess in a Modal sandbox detects cloud mode
(`LQH_API_TOKEN` + `LQH_BASE_URL`), builds an `AsyncOpenAI` client against
`api.lqh.ai/v1`, and calls `judge:*`. The backend restricts job tokens with
`chat.score` scope to judge models only. **The reward channel exists and is
already secured.** A GRPO reward function is a ~100-line module that reuses
`_build_scoring_prompt` and `_parse_score_response` from `lqh/scoring.py`.

### 4.2 Why pointwise 1–10 scoring will underperform

`_build_scoring_prompt` ends with "assign a score from 1 to 10" and
`_parse_score_response` reads a scalar out of JSON. GRPO computes
advantage as the within-group deviation of reward (`scale_rewards="group"`
normalizes by group mean/std). So:

- If all G completions in a group get the same integer, **advantage is
  exactly zero and the group contributes no gradient** — the rollout and
  the judge call are pure waste.
- For a saturated task (our `classification` / `extraction` benchmarks sit
  near ceiling) that's most groups. It is the exact same "no preference
  pairs available" failure that `DPO_PLAN.md` documented, arriving through
  a different door.
- Even where scores differ, a 10-point scale from a 9B judge is noisy at
  ±1. Group std of ~0.5 on a scale where judge noise is ~1 means we are
  normalizing noise up to unit variance and calling it advantage. With
  `scale_rewards="group"`, small-std groups get *amplified* — this is the
  documented "question-level difficulty bias" that `scale_rewards=False`
  exists to defuse.

### 4.3 What to do instead: judge the group, not the sample

The literature converged on this in 2026 (Tournament-GRPO, Rubric-Grounded
RL): absolute scores are hard to calibrate, weakly discriminative among
same-query rollouts, and saturate during training; *relative* comparison
within the group fixes all three. Concretely, the reward function receives
all G completions for a prompt at once, so we can ask for one ranking:

```
reward_group_rank(prompt, [c_1..c_G]) -> one judge call
  → judge returns an ordering / K winners
  → min-max normalize rank position to [0,1] → reward
```

Properties, all of which we want:

- **Guaranteed variance.** A ranking always separates the group. No dead
  groups.
- **8× fewer judge calls** than pointwise at G=8 (O(1) per group vs O(G)),
  and far below the O(G²) of exhaustive pairwise.
- **No calibration burden** on the judge — "which of these is better" is a
  much easier question for a 9B model than "is this a 6 or a 7".
- Cheaper per step in latency too: 8 concurrent calls/step instead of 64.

Trade-off: rank rewards are ordinal, so they discard "this whole group is
bad". Fix with a **composite reward**, which TRL supports natively by
passing a list of reward functions:

1. `judge_rank` — group-relative, the main signal (weight ~1.0).
2. `judge_absolute` — a *single* pointwise `judge:small` score per
   completion, normalized to [0,1], at low weight (~0.2). Anchors the group
   so a uniformly-bad group still gets pushed down, and gives us a
   dashboard number comparable to existing eval scores.
3. Deterministic guards, free, zero latency, and the main defense against
   reward hacking: JSON-validity, schema conformance, required-tag
   presence, tool-call well-formedness (reuse `lqh/train/tool_format.py`),
   length-in-band. Start these as small negative penalties, not positive
   rewards — a positive format reward is the fastest way to get a policy
   that emits perfect empty JSON.

Keep each as a separate entry in `reward_funcs` rather than pre-summing:
TRL logs per-function reward means, which is exactly the diagnostic we
lacked in DPO.

### 4.4 Cost and rate-limit budget

Rough, stated assumptions: 300 optimizer steps, 8 prompts/step, G=8 →
2,400 groups / 19,200 completions.

| Scheme | Judge calls | Calls/step | Notes |
|---|---:|---:|---|
| Pointwise only | 19,200 | 64 | ~44M judge tokens; wall-clock ~5–10s/step at concurrency 64 |
| Group-rank only | 2,400 | 8 | longer prompts (G completions in context), ~8× fewer calls |
| Rank + cheap pointwise | 21,600 | 72 | recommended; pointwise at `judge:small` is the cheap half |

At `judge:small` = qwen3.5-9b these are low-tens-of-dollars of raw
inference for a full run — **judge cost is not the constraint; judge
*latency* is.** Rate limits are the thing to check: `LQH_USER_CHAT_RPM`
defaults to 600 and `LQH_ORG_CHAT_RPM` to 6000
(`backend/internal/config/config.go:302`). 72 calls/step at 600 RPM caps us
at ~8 steps/min from the limiter alone — fine for a single run, but a 6-way
GRPO sweep would hit the *org* ceiling. Either raise the job-token limits
or serialize sweep children.

Note `ratelimit/redis.go` **fails open** on Redis errors, so the limiter
won't hard-stop a run; but a 429 storm turning into `None` rewards will
quietly starve training. The reward function must log its `None` rate as a
first-class metric (`log_metric` is passed to reward funcs) and the run
should abort if it exceeds a threshold.

### 4.5 Reward hacking — assume it, instrument for it

Small policies + a prompted judge is the regime where reward hacking is
best documented. Non-negotiable mitigations:

- `max_completion_length` hard cap plus `mask_truncated_completions=True`,
  so a truncated ramble neither trains nor scores.
- Log completion-length distribution per step. Length inflation is the
  canonical first symptom.
- `beta > 0` (e.g. `0.001–0.01`) for v1 despite TRL's default of `0.0`.
  Yes it costs a reference-model copy; on a 1.2B model that is affordable,
  and a KL leash is worth more to us than the memory. Revisit once the
  method works.
- **Evaluate with a different judge than you train with.** Train on
  `judge:small`, gate on `judge:medium`/`large`. If the gain evaporates
  when the judge changes, it was hacking, not learning.
- Keep the deterministic guards from §4.3 — they're the only rewards the
  policy cannot talk its way around.

### 4.6 Prompt-source and leakage discipline

Straight from `DPO_PLAN.md` §Phase 1, and it applies verbatim: the GRPO
prompt pool must be **fresh prompts SFT never saw**, the validation set
separate, and the final test set untouched. GRPO doesn't need reference
answers at all — only prompts — which actually makes this *easier* than it
was for DPO. Also mirror the matched **continued-SFT control**: if more
supervised data on the same prompts matches GRPO's gain, we've learned
something important and cheap.

---

## 5. Implementation plan

### Phase 0 — spike (do this before writing product code)

Two independent questions, both answerable in a couple of days:

1. **Can vLLM + TRL 1.0 + PEFT train an LFM2.5-1.2B on one GPU in a Modal
   sandbox?** Success = 50 optimizer steps, no hang, no OOM, reward mean
   moves on a *trivially learnable synthetic reward* (e.g. "reward = 1 if
   output is valid JSON"). This isolates infrastructure from signal. If it
   hangs, fall back to Option B (two venvs) immediately rather than
   debugging TRL.
2. **Does the group-rank judge discriminate?** Offline, no training: take
   the existing SFT checkpoint for `voice_satisfaction`, sample G=8 at
   temperature 1.0 for 200 prompts, and measure — (a) fraction of groups
   where pointwise scores are all identical (the dead-group rate we're
   trying to kill), (b) rank-judge self-agreement across two calls, (c)
   correlation between rank order and the existing pointwise score. If the
   judge can't rank consistently, GRPO has no reward and the project stops
   here for a cheaper price than a training run.

Gate: both must pass before Phase 1.

### Phase 1 — `lqh/train/grpo.py`

- New `grpo_loop(run_dir, config)`, dispatched from
  `lqh/train/__main__.py` on `type in ("grpo", "on_policy_grpo")` alongside
  the existing `sft`/`dpo` branches. Keep the same OOM detection,
  `write_status`, and `begin_run_attempt` contract — that's what makes the
  backend's lease classification work.
- Reuse `load_for_training` (adapter-aware, handles continuing an SFT
  adapter in place — the same reasoning as `dpo_loop`'s
  `continuing_adapter` path applies and gives us a deployable adapter-only
  artifact).
- Structurally *much* simpler than `dpo.py` (1720 lines): no iteration
  loop, no `predictions.parquet` handshake, no `wait_for_file`, no watcher
  ping-pong. One `GRPOTrainer`, one `train_with_checkpoint_fallback(...)`
  from `lqh/train/resume.py`.
- Progress: GRPO has no natural iteration boundaries, so add a
  `task_kind="grpo"` phase map to `lqh/progress.py` — setup → training
  (step/max_steps, the dominant phase) → final inference → final scoring.
  A `TrainerCallback` mirroring `_DPOProgressCallback` emits it.
- Diagnostics artifact per N steps, in the spirit of `DPO_PLAN.md` Phase 0
  (that diagnosis was only possible because the artifacts existed): reward
  mean/std **per reward function**, fraction of zero-variance groups,
  completion length percentiles, truncation rate, judge-`None` rate, KL,
  clip fraction, entropy, cumulative optimizer steps.
- **Port the min-update guard.** DPO's headline bug was 10 optimizer steps
  for a whole run. Warn/fail if projected steps < 100.

### Phase 2 — `lqh/train/reward.py`

- `build_reward_funcs(config) -> list[Callable]`, returning async callables
  with TRL's signature (`prompts`, `completions`, `completion_ids`,
  `**kwargs`; extra dataset columns arrive through `kwargs`).
- Group reconstruction: TRL passes a flat list of B×G completions. The
  ranking reward must regroup by prompt — the trainer emits them grouped,
  but assert it rather than assume it. (`DPO_PLAN.md` §"Preference assembly
  aligns rows by `sample_index`" is the scar tissue here: an off-by-group
  misalignment is silent and fatal.)
- Reuses `lqh/scoring.py` for prompt construction, and the scorer `.md`
  file resolution from `cloud_score.py::_resolve_scorer_path` so the same
  project scorer drives GRPO, DPO and eval.
- Errors → `None` (never `0.0`), counted and logged via `log_metric`.
- Local/non-cloud mode: `is_cloud_mode()` is False on SSH backends. Either
  run the judge from the trainer with the user's own token, or refuse GRPO
  on backends without a scoped token. Decide explicitly — don't let it
  silently no-op the way gap-selection did for DPO.

### Phase 3 — cloud wiring

Backend, mirroring the `data_gen` (migration 0043) and `gguf_convert`
(0038) precedents:

- Migration `00NN_grpo_jobs.up/down.sql`: extend the `cloud_jobs.kind`
  CHECK with `train_grpo` (and `train_grpo_sweep` if sweeps land).
- `backend/internal/cloud/runner.go`: `KindTrainGRPO JobKind = "train_grpo"`.
- `backend/internal/handler/cloud_jobs.go`: `moduleForKind` →
  `lqh.train` (same entrypoint as SFT/DPO, dispatched on config `type`);
  `imagePurposeForKind` → `"grpo"`.
- `backend/internal/handler/modal_images.go`: add `grpo` to
  `validImagePurposes`.
- `backend/scripts/modal_build_image.py`: `--purpose grpo`, built FROM the
  vLLM base, refusing to combine with other purposes (same rule as
  `gpu_eval` / `cpu_data_gen`). Update the build block in `CLAUDE.md` §
  "push to production".
- `lqh/remote/cloud.py::_infer_kind`: map `cfg_type == "grpo"` →
  `train_grpo`; add the workflow_kind/subtype mapping next to the existing
  `train_*` entries.
- `lqh/remote/failure.py`: GRPO-specific remediation hints (the existing
  `"dpo" in kind` branch suggests fewer iterations/rollout samples; the
  GRPO analogues are fewer generations, shorter completions, lower
  `vllm_gpu_memory_utilization`).

### Phase 4 — defaults

Add a GRPO branch to `lqh/train/defaults.py` next to `_DPO_TYPES`. Starting
point, to be replaced by measured values (and `PROVENANCE` updated in the
same commit, per that file's own rule):

| Knob | Start | Why |
|---|---|---|
| `num_generations` | 8 | TRL default; below 4 the group statistic is junk |
| prompts per optimizer step | 8 | → 64 completions/step |
| `per_device_train_batch_size` | 16 completions | ×4 grad-accum = 64 |
| target optimizer steps | 300–500 | DPO's lesson: count updates first |
| `learning_rate` | 1e-6 → 5e-6 (LoRA) | GRPO is far more LR-sensitive than SFT |
| `beta` (KL) | 0.005 | non-default on purpose (§4.5) |
| `loss_type` | `dapo` (default) | `dr_grpo` if length bias shows up |
| `scale_rewards` | `"group"`, test `False` | `False` removes difficulty bias |
| `max_completion_length` | task-dependent, hard cap | with `mask_truncated_completions=True` |
| temperature | 1.0 | ≠1 triggers the vLLM logprob issue (trl#4159) |
| `num_iterations` (μ) | 1 | >1 adds off-policy correction we don't need yet |
| `vllm_gpu_memory_utilization` | 0.3 | raise only after OOM headroom is measured |
| `vllm_enable_sleep_mode` | True | shared card |

### Phase 5 — measurement (the part that decides the project)

Fork `tests/benchmarks/dpo_value/` into `tests/benchmarks/grpo_value/`. It
already does the right things: disjoint SFT-train / RL-prompt / validation
/ untouched-test splits, three seeds, paired bootstrap CIs, deterministic
`voice_satisfaction` sub-metrics. Arms:

| Arm | Purpose |
|---|---|
| SFT checkpoint | baseline |
| Continued SFT on the RL prompt pool | **the control that matters** — matched tokens/updates |
| GRPO, group-rank reward | the proposal |
| GRPO, pointwise reward | isolates the reward-design claim from §4.2 |
| GRPO, deterministic guards only | how much is just format compliance? |

Primary task `voice_satisfaction` (real headroom, SFT ≈ 5.67 at
1.2B-Instruct); `translation` at 350M as secondary; `style_rewrite` as a
regression canary. Skip the saturated tasks — they can't show a gain.

### Phase 6 — product surface (only after Phase 5 passes)

`lqh/tools/definitions.py:1530` currently enumerates
`["sft", "on_policy_dpo"]` for `start_training`; `handlers.py` branches on
`type in ("on_policy_dpo", "dpo")` in several places (~5073, 5126, 5210,
5220). Add `grpo`, update `lqh/skills/train/SKILL.md` routing, and decide
DPO's fate — leave it working but undocumented, or remove it. If GRPO
lands, the `sft → dpo` post-eval routing in the `/improve` skill
(`definitions.py:2258` lists `'dpo'` as a route) should point at GRPO.

---

## 6. What will be difficult, ranked

1. **The image.** Highest risk, highest blast radius, and it blocks
   everything. The repo's own history (`ISSUE_4_PLAN.md`, the `gpu_eval`
   comments, `modal_serve_harness.py`'s existence) says merging a serving
   stack with the training stack is where this codebase gets hurt. Budget
   real time; build the regression gate before promoting.
2. **Signal on small models.** 350M–1.2B policies have limited capacity to
   exploit a noisy preference signal. This is the same wall DPO hit. The
   §4.3 reward redesign is the main mitigation, and the Phase 0.2 spike is
   the cheap early read.
3. **Reward hacking under a prompted judge.** Expected, not hypothetical.
   Instrument for it (§4.5) rather than hoping.
4. **TRL colocate stability with PEFT.** trl#3671 is open. Single-GPU
   should dodge it; verify in the spike, and know that Option B exists.
5. **Wall clock and cost per experiment.** GRPO in TRL is synchronous —
   generation and training do not overlap, by design. Expect a GRPO run to
   cost several × an SFT run of the same data. A full sweep is a serious
   spend; prefer a small grid over the sweep machinery for v1.
6. **Preemption.** Cloud leases get preempted. `train_with_checkpoint_fallback`
   handles trainer state, but a resumed run rebuilds the vLLM engine and
   re-pays cold start. Save checkpoints more often than SFT does.
7. **Sweeps.** `MASTER_PORT` collision (trl#3979) plus judge-side org rate
   limits make N-parallel GRPO configs on one host hostile. Design sweeps
   as separate sandboxes from the start, or skip sweeps for v1.

---

## 7. What is *not* hard (worth saying explicitly)

- The judge/reward channel: already built and already authorized.
- Preemption/resume: GRPOTrainer is an HF Trainer, so `resume.py` works.
- LoRA save/merge/deploy: unchanged from SFT, and the adapter stays
  deployable if we continue the SFT adapter in place.
- The DPO-style laptop↔sandbox handshake: **gone entirely.** No
  `predictions.parquet`, no `wait_for_file`, no watcher ping-pong, no
  `cloud_score.py` inline-vs-watcher dual path. That machinery
  (`lqh/remote/watcher.py`, `lqh/watcher.py:506`, `cloud_score.py`) was a
  large share of DPO's complexity and its bug surface. GRPO deletes it.

---

## 8. Decision gates

Stop and reassess at each:

- **G0 (spike):** synthetic reward moves + rank judge discriminates.
  Fail → don't build.
- **G1 (learnability):** on a task with an unambiguous programmatic reward,
  GRPO measurably improves that metric. This is `DPO_PLAN.md`'s "tiny
  learnability test", and DPO shipped without one. Fail → the loop is
  wrong, not the task.
- **G2 (value):** on `voice_satisfaction`, ≥ +0.3 over the SFT checkpoint
  with a paired 95% CI excluding zero, reproducing in ≥2 of 3 seeds, no
  regression in JSON validity / tag accuracy / turn-index accuracy, **and
  beating the matched continued-SFT control.** Same bar DPO was held to.
- **G3 (robustness):** the gain survives switching the eval judge from
  `judge:small` to `judge:medium`/`large`. Fail → reward hacking.

If G2 fails the same way DPO's did — GRPO ≈ continued SFT — the honest
conclusion is that for these task/data shapes the bottleneck is supervised
data quality, not the alignment objective, and the right move is to invest
in data generation rather than a third alignment algorithm.

---

## 9. Open questions

- Which judge tier for training? `judge:small` (9B) is cheap and fast but
  is it a *good enough ranker*? Phase 0.2 answers this. If not,
  `judge:medium` roughly triples cost — still affordable, but latency per
  step rises.
- Should the group-rank judge see the reference/gold answer when one
  exists? It would sharpen ranking, but it also drags GRPO back toward
  imitation and toward DPO's "relearn what SFT already saw" failure.
  Probably: no for the rank reward, optional for the absolute anchor.
- Do we need `grpo_sweep`, or is a fixed measured default enough?
  `defaults.py` says the product no longer sweeps SFT by default; GRPO
  should probably follow suit given the cost.
- Multi-turn / tool-calling rollouts. TRL 1.0 has environments and tools,
  and `voice_satisfaction` is conversational. v1 should be single-turn
  completions; multi-turn is a separate project.
- Does `train_grpo` need a distinct GPU-hour billing category, or does it
  ride the existing `train_*` path in `internal/cloud`?

---

## Sources

- [GRPO Trainer — TRL docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [TRL v1.0: Post-Training Library Built to Move with the Field](https://huggingface.co/blog/trl-v1)
- [trl/trainer/grpo_trainer.py (main)](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) — plus the locally cached v1.0.0 wheel, which is what the pin actually resolves to
- [TRL vLLM Integration](https://huggingface.co/docs/trl/vllm_integration)
- [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) — TRL is synchronous-only; when to reach for verl/SkyRL instead
- [No GPU left behind: Co-located vLLM in TRL](https://huggingface.co/blog/vllm-colocate)
- Open TRL issues: [#3671 colocate+PEFT hang](https://github.com/huggingface/trl/issues/3671), [#3979 MASTER_PORT](https://github.com/huggingface/trl/issues/3979), [#4129 trust_remote_code](https://github.com/huggingface/trl/issues/4129), [#4159 vLLM logprobs/temperature](https://github.com/huggingface/trl/issues/4159), [#4358 max_prompt_length](https://github.com/huggingface/trl/issues/4358)
- [vLLM recipe: LiquidAI/LFM2.5-1.2B-Base](https://recipes.vllm.ai/LiquidAI/LFM2.5-1.2B-Base) — native `Lfm2ForCausalLM`, vLLM ≥0.23.0
- [Tournament-GRPO: Group-Wise Tournament Rewards](https://arxiv.org/html/2605.26958v1) — relative > absolute judging; O(MK) vs O(K²)
- [Rubric-Grounded RL](https://arxiv.org/abs/2605.08061) — dropping zero-variance rubrics
- [Rubrics as Rewards](https://arxiv.org/pdf/2507.17746)
- [Mitigating Reward Hacking in RLHF via Advantage Sign Robustness](https://arxiv.org/pdf/2604.02986) — small-model reward hacking; GRPO's relative resistance
