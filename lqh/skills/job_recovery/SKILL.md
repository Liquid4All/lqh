# Skill: Job Recovery

You are now in **job recovery** mode. A cloud job was interrupted. Your goal is to
say truthfully what happened, what survived, what it cost, and what the next
attempt should look like — and to avoid the two ways this goes wrong: calling an
infrastructure failure a "transient issue", and retrying the same job shape until
the user's money is gone.

**Load this when:** a training or eval run failed with an infrastructure class
(preempted, orphaned, timeout), a run has failed more than once, or the user asks
why their jobs keep dying.

## Step 0 — Get the facts before saying anything

| Fact | Where |
|---|---|
| Failure class + attempt history | `training_status(run_name=...)` — the `Diagnosis:`, `Attempts:` and `Billed:` lines |
| How far it got | the `Progress:` / `Sweep:` lines, and `runs/<name>/progress.jsonl` |
| What the trainer said | `runs/<name>/stderr.log`, or `runs/<name>/sweep_<config>/stderr.log` |
| What was published | the `artifacts` tool — a checkpoint that reached storage survives the sandbox |
| Prior attempts on this project | `summary`, and `NOTES.md` |

Never diagnose from the run name or from memory. Read the status card first.

## What a kill does and does not destroy

- **Published artifacts survive.** Anything the `artifacts` tool lists is in
  storage and is unaffected by the sandbox dying. Check this before telling the
  user their work is gone — a timed-out run often published its checkpoint and
  only lost the evaluation.
- **Unpublished checkpoints do not come back to you.** They live on the cloud
  volume under the *original job id*. The backend's own relaunch can resume from
  them; a resubmit cannot — it is a new job with a new volume directory, and the
  old one is garbage-collected after 7 days.
- **Run names cannot be reused.** `start_training` rejects an existing name.
  There is no "same run name to resume" trick and no client-side resume at all.
  Say "start a fresh run", never "resume".
- **Local files are untouched** — datasets, scorers, prompts, and every
  previously completed run are exactly as they were.

## The recovery decision

| Class | Cause | Do this |
|---|---|---|
| **preempted** | the provider reclaimed the GPU | Infrastructure. One fresh attempt is reasonable — smaller if the run is long. On a second preemption, shrink the exposure instead of retrying. |
| **orphaned** | the sandbox stopped appearing in the provider's live list, with no terminal event | An observation, not a cause. Usually a preemption — but a workload that died while the backend was restarting looks identical from here. Check `artifacts` and `stderr.log` FIRST; if nothing points at the run, treat it as preempted. |
| **timeout** | hit the wall-clock cap | Sizing, NOT infrastructure — the cap was consented to at submit time and the job outgrew it. An identical resubmit fails identically. Shrink first, always. |
| **oom** | memory exceeded | Config. Lower `per_device_train_batch_size` (raise `gradient_accumulation_steps` to keep the effective batch size), shorten or split the longest conversations in the dataset, enable gradient checkpointing, or use a smaller model. |
| **crashed** | the trainer raised | Read `stderr.log`. Fix the exception. Do not resubmit unchanged. |
| **config** | bad input | Fix the input. Do not blame infrastructure. |

## Reducing exposure (the menu, cheapest first)

A shorter job is preempted less often — exposure is roughly proportional to wall
clock. In order of what to give up first:

1. **Drop the sweep** — `enable_sweep=false` with the best-known
   hyperparameters. Usually the largest single cut, since a sweep trains its
   configs sequentially in one job. Only applies to a run that was sweeping:
   SFT does not by default, so check before reaching for this.
2. **Fewer epochs** — one epoch on a large set beats three on a small one.
3. **Smaller training set** — validate the direction, then scale.
4. **Smaller base model** — one step down the ladder, with the user's consent and
   only if `SPEC.md`'s `## Inference Budget` allows it.
5. **Split into stages** — train, publish, then continue from the published
   checkpoint as a new run's `base_model`. This is the only real way to carry
   work across an interruption, and it is worth proposing for any job the user
   wants that runs longer than a few hours.

## Talking about cost — say the true thing

The user is billed for GPU time on attempts that produced nothing. Do not hide it
and do not oversell it:

- **Say the number, after checking what survived.** "Those three attempts billed
  about $6.40; the checkpoint from the second one is published, the rest is
  gone." Look at `artifacts` before claiming nothing came out of it.
- **Own it.** Preemption is our infrastructure's behaviour, not a mistake the
  user made. Never say "that's outside our control" and stop there — it is true
  of the provider and irrelevant to the user.
- **Do not promise a refund or credit.** You cannot issue one. Point at
  `/feedback` if the user wants the charge reviewed by the LQH team.
- **Price the next attempt before starting it.** The `training_status` card and
  the consent prompt both show the hard cap. Give the user the number and let
  them choose.

## When to stop

Stop proposing retries and escalate to the user — or, with no user attached,
`exit_auto_mode(status="failure", ...)` — when any of these is true:

- two infrastructure failures on the same job shape (in auto/subagent mode, where
  you cannot ask, the single retry must already be the smaller job),
- three failures of any kind on the same run configuration,
- the same error text three times in a row,
- the cost so far exceeds what the user agreed to.

Report what was tried, what it cost, and exactly which exposure reduction you
recommend. A blocked run reported honestly is a better outcome than a fourth
attempt.

## Maintain NOTES.md

Record the interruption in the project-root `NOTES.md`: which run, which class,
what was billed, what you changed for the next attempt. A repeat of the same
class on the same shape is the signal to stop — and a future session can only
see that if you wrote it down.
