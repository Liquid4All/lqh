"""Classification of cloud job interruptions, and what to do about them.

LQH Cloud runs every GPU job in a preemptible sandbox — there is no paid
opt-out for GPU sandboxes — so preemption, orphaning, and wall-clock
timeouts are NORMAL states of the system, not anomalies. The agent needs
to react correctly to them at any point in a session, including after a
context compaction that dropped whatever guidance it once read.

So the taxonomy is *computed*, not recalled: this module turns a backend
job snapshot into (a) a failure class and (b) the one recovery step to
take, and both the `[System: ...]` notification (lqh/jobs.py) and the
`training_status` card (lqh/tools/handlers.py) render from it.

Two facts every string here is written around:

* **A resubmit is a fresh run from step 0.** The cloud volume run dir is
  keyed on the backend job id, `start_training` rejects a reused run
  name, and run dirs are garbage-collected after 7 days. Only the
  backend's own relaunch can continue from a checkpoint. Calling a
  resubmit a "resume" is a lie, and it is the specific lie that cost us
  a user (feedback #37).
* **Infrastructure failures still bill GPU time.** Say the number, don't
  promise a refund we cannot issue, point at /feedback.

Structured fields from the backend's `recovery` object are preferred;
the raw ``error`` string is a fallback so this keeps working against a
backend that predates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "JobFailure",
    "classify_job_failure",
    "describe_failure",
    "attempt_lines",
    "completion_note",
    "INFRA_CLASSES",
]

# Classes whose cause is our infrastructure rather than the user's
# configuration. The distinction drives the whole message: an infra
# failure means "nothing is wrong with your setup", an oom/crashed one
# means "here is the lever to pull".
#
# `timeout` is deliberately NOT here. A wall-clock cap is a budget the
# user consented to at submit time; the job outgrew it. Calling that an
# infrastructure failure contradicted this module's own timeout
# diagnosis ("this is a SIZING problem") and made the startup signal
# apologise for a working system.
#
# `orphaned` IS here, but it is the weakest member: all we observed is
# that the sandbox stopped appearing in the provider's list. See
# describe_failure, which says so rather than asserting a preemption.
INFRA_CLASSES = frozenset({"preempted", "orphaned"})

# Structured terminal reasons the backend can store
# (migrations 0022 + 0054) mapped to our class.
#
# 'failed' is deliberately ABSENT. It is the backend's default for every
# terminal it could not attribute — a launcher error, a failed bundle
# download, expired credentials, a publish failure — and mapping it
# straight to "crashed" made the agent tell users "the trainer itself
# raised" about problems the trainer never saw. When the reason is
# 'failed' we read the error text instead, and only say "the trainer
# raised" when the text looks like a traceback.
_TERMINAL_REASON_TO_CLASS = {
    "preempted": "preempted",
    "orphaned": "orphaned",
    "timeout": "timeout",
    "oom": "oom",
    "cancelled": "cancelled",
    "stalled": "stalled",
}


@dataclass(frozen=True)
class JobFailure:
    """What happened to a cloud job, and whether we caused it."""

    cls: str = "unknown"
    infra: bool = False
    leases: int = 1
    # Leases with a CONFIRMED provider preemption signal.
    preemptions: int = 0
    # Leases lost for any reason we did not attribute to the user's
    # config: confirmed preemptions, unattributed SIGKILLs, orphans.
    interruptions: int = 0
    # The backend relaunched this job on a fresh sandbox at least once.
    # Named `continued`, not `resumed`: a relaunch lands on the same
    # volume, but whether it picked up a checkpoint depends on whether
    # the previous lease lived long enough to write one.
    continued: bool = False
    budget_exhausted: bool = False
    runtime_minutes: int | None = None
    timeout_minutes: int | None = None
    gpu_type: str | None = None
    last_stage: str | None = None
    billed_micros: int | None = None
    # Cloud job kind (train_sft, train_dpo, data_gen, eval_hf, ...). The
    # recovery menu is kind-specific: telling a data-generation job to
    # "use fewer sweep configs or a smaller base model" is noise.
    kind: str | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def known(self) -> bool:
        return self.cls != "unknown"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _classify_from_error(error: str) -> str:
    """Fallback classifier over the backend's error strings.

    Ordered: the first match wins. These are the literal strings the Go
    side emits (internal/cloud/reconciler.go, terminal_gate.go, and the
    sandbox runner), so keep them in sync when those change.
    """
    text = (error or "").lower()
    if not text:
        return "unknown"
    if "no live sandbox" in text or "orphaned" in text:
        return "orphaned"
    # "wall-clock timeout" ONLY. A bare "exit code 124" must not become a
    # timeout here: the backend deliberately withholds that label unless
    # the sandbox actually burned its budget, because 124 is also an exit
    # code a workload can produce on its own. Re-deriving it from the raw
    # string would undo that guard on the client and tell a user whose
    # script exited 124 after five minutes that their job was too big.
    if "wall-clock timeout" in text:
        return "timeout"
    # Precise markers only: the standard exit-137 message literally
    # reads "preemption, OOM, or termination", so a bare "oom" substring
    # would classify every preemption as a memory problem.
    if any(m in text for m in ("out of memory", "outofmemory", "oom-kill", "oom_kill")):
        return "oom"
    # Ambiguity markers FIRST. The runner's standard message reads
    # "exit code 137 (SIGKILL: worker interruption, preemption, OOM, or
    # termination)" — it contains the word "preemption" precisely
    # BECAUSE it cannot tell. Matching "preempt" before checking for
    # 137/SIGKILL turned that disclaimer into a confirmed diagnosis,
    # which is how an un-signalled OOM got "INFRASTRUCTURE, NOT YOUR
    # CONFIG" and an unchanged retry.
    if "sigkill" in text or "exit code 137" in text:
        return "interrupted"
    if "preempt" in text:
        # No 137/SIGKILL anywhere in the text: something asserted a
        # preemption on its own terms.
        return "preempted"
    if "cancelled" in text or "canceled" in text:
        return "cancelled"
    # Narrow phrases only. Bare "invalid" / "not found" matched half the
    # error strings a cloud job can produce — including provider and
    # network errors that have nothing to do with the user's config.
    if "submission stalled" in text or "invalid multipart" in text:
        return "config"
    # A trainer traceback is a trainer bug; everything else could equally
    # be the launcher, the bundle download, the publish step, or auth.
    # Only claim "the trainer raised" when the text really looks like a
    # Python traceback — "error:" alone appears in almost every error.
    if any(m in text for m in ("traceback", "exception:", "assertionerror")):
        return "crashed"
    return "failed"


def classify_job_failure(
    snapshot: dict[str, Any] | None,
    error: str | None = None,
) -> JobFailure:
    """Build a JobFailure from a `GET /v1/cloud/jobs/{id}` snapshot.

    Both arguments are optional and every field is defensive: an older
    backend, a snapshot that failed to fetch, or a job that predates the
    recovery bookkeeping must degrade to ``cls="unknown"``, which callers
    render as today's plain message rather than an invented diagnosis.
    """
    snap = snapshot or {}
    recovery = snap.get("recovery") or {}
    resource = snap.get("resource") or {}
    err = error if error is not None else snap.get("error")

    attempts = [a for a in (recovery.get("attempts") or []) if isinstance(a, dict)]
    reasons = tuple(
        str(a.get("terminal_reason")) for a in attempts if a.get("terminal_reason")
    )
    leases = _as_int(recovery.get("lease_no"))
    leases = (leases + 1) if leases is not None else max(len(attempts), 1)

    status = str(snap.get("status") or "")
    if status == "cancelled":
        cls = "cancelled"
    else:
        cls = _TERMINAL_REASON_TO_CLASS.get(
            str(recovery.get("failure_class") or (reasons[-1] if reasons else "")),
            "",
        )
        if not cls:
            cls = _classify_from_error(str(err or ""))

    # budget_exhausted / max_leases are only meaningful once the backend
    # runs a resume policy; until then they are simply absent and every
    # dependent phrase stays unsaid rather than guessed.
    max_leases = _as_int(recovery.get("max_leases"))
    budget_exhausted = bool(recovery.get("budget_exhausted"))
    if not budget_exhausted and max_leases:
        budget_exhausted = leases >= max_leases

    return JobFailure(
        cls=cls,
        infra=cls in INFRA_CLASSES,
        leases=leases,
        # Confirmed preemptions only. Counting orphans and unattributed
        # SIGKILLs here made the directive say "the sandbox was preempted
        # 3×" about leases we never diagnosed.
        preemptions=sum(1 for r in reasons if r == "preempted"),
        interruptions=sum(
            1 for r in reasons if r in ("preempted", "orphaned", "interrupted")
        ),
        continued=bool(recovery.get("continued_count"))
        or any(a.get("continued") for a in attempts),
        budget_exhausted=budget_exhausted,
        runtime_minutes=_runtime_minutes(snap, attempts),
        timeout_minutes=_as_int(resource.get("timeout_minutes")),
        gpu_type=(resource.get("gpu_type") or None),
        last_stage=(recovery.get("last_progress_stage") or None),
        billed_micros=_as_int(recovery.get("billed_cost_micros")),
        kind=(snap.get("kind") or None),
        reasons=reasons,
    )


def _runtime_minutes(snap: dict[str, Any], attempts: list[dict[str, Any]]) -> int | None:
    """Minutes of GPU time the job actually consumed.

    Prefers the sum of lease runtimes (what we ran) over start→end (which
    includes any gap between a sandbox vanishing and us noticing).
    """
    total = sum(
        s for s in (_as_int(a.get("runtime_seconds")) for a in attempts) if s
    )
    if total:
        return max(1, total // 60)
    started, ended = snap.get("started_at"), snap.get("ended_at")
    if isinstance(started, str) and isinstance(ended, str) and started and ended:
        try:
            from datetime import datetime

            t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(ended.replace("Z", "+00:00"))
            return max(0, int((t1 - t0).total_seconds() // 60))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _dollars(micros: int | None) -> str | None:
    if not micros or micros <= 0:
        return None
    return f"${micros / 1e6:.2f}"


def _cost_tail(f: JobFailure) -> str:
    """What to say about money on a failure we caused."""
    attempts = "attempt" if f.leases == 1 else f"{f.leases} attempts"
    amount = f"billed ≈{billed}" if (billed := _dollars(f.billed_micros)) else "billed GPU time"
    return (
        f" Those {attempts} {amount}. Check `artifacts` before telling the user "
        "the work is gone — an interrupted run often published a checkpoint "
        "first, and only the evaluation was lost. Then say what was billed in "
        "one sentence, do not promise a refund (you cannot issue one), and "
        "point them at /feedback if they want the LQH team to review the charge."
    )


def _shrink_menu(f: JobFailure) -> str:
    """How to make THIS kind of job smaller.

    Classification is applied to every cloud terminal snapshot, so the
    generic sweep/epochs/base-model advice was being handed to data
    generation and eval jobs that have none of those knobs.
    """
    kind = (f.kind or "").lower()
    if "data_gen" in kind:
        return (
            "fewer samples, a shorter prompt set, or splitting the "
            "generation into batches"
        )
    if "eval" in kind or "infer" in kind:
        return (
            "fewer eval samples, a shorter max_seq_length, or splitting the "
            "eval set into batches"
        )
    if "grpo" in kind:
        return (
            "fewer optimizer steps (grpo.max_steps), fewer generations per "
            "prompt (grpo.num_generations), shorter completions "
            "(grpo.max_completion_length), or a lower "
            "grpo.vllm_gpu_memory_utilization"
        )
    if "dpo" in kind:
        return (
            "fewer iterations, fewer rollout samples per iteration, or a "
            "smaller base model"
        )
    if "train" in kind or "sft" in kind:
        return (
            "fewer sweep configs, fewer epochs, a smaller training set, or a "
            "smaller base model"
        )
    # Unknown kind: name the axis without inventing knobs it may not have.
    return "less work per job — split it into stages, or use a smaller model"


def _artifact_clause(f: JobFailure) -> str:
    kind = (f.kind or "").lower()
    if "data_gen" in kind:
        return (
            "Check `artifacts` first: a partial dataset may already have been "
            "published, and generation resumes from it."
        )
    if "eval" in kind or "infer" in kind:
        return "Check `artifacts` first: partial predictions may already be published."
    return (
        "Check `artifacts` first: if a checkpoint was published before the "
        "cap, the training itself is salvageable and only the evaluation "
        "needs rerunning."
    )


def _stage_clause(f: JobFailure) -> str:
    bits = []
    if f.last_stage:
        bits.append(f"reaching {f.last_stage}")
    if f.runtime_minutes:
        bits.append(f"after {f.runtime_minutes} min")
    return (" " + " ".join(bits) + ".") if bits else ""


def describe_failure(f: JobFailure | None, run_name: str) -> str:
    """The recovery directive for one failure, addressed to the agent.

    Returns "" for an unclassifiable failure so the caller's existing
    message stands unchanged.
    """
    if f is None or not f.known:
        return ""

    if f.cls == "orphaned":
        gpu = f" on {f.gpu_type}" if f.gpu_type else ""
        # Say what was observed, not what it means. A sandbox that stops
        # appearing in the provider's list without a terminal event is
        # USUALLY a preemption — but it also covers a workload that died
        # while the backend was down, a publish that failed, and a clean
        # exit nobody saw. Asserting "NOT YOUR CONFIG" here would send
        # the agent past a real bug in the user's run.
        return (
            f"ORPHANED{gpu}: the sandbox stopped appearing in the provider's "
            f"live list and never reported a terminal event.{_stage_clause(f)} "
            "That is an OBSERVATION, not a diagnosis — most often it is a "
            "preemption (LQH Cloud GPU sandboxes are preemptible by "
            "construction and no setting prevents it), but it also covers a "
            "workload that died while the backend was restarting. RECOVERY: "
            "check `artifacts` FIRST — an interrupted run often published a "
            "checkpoint and lost only the evaluation — and read "
            f"runs/{run_name}/stderr.log if it was captured, because that is "
            "what tells a preemption apart from a crash. If nothing points at "
            "the run itself, treat it as infrastructure and resubmit under a "
            "NEW run name. A resubmit always starts from step 0 — the previous "
            "run's cloud checkpoints belong to the finished job and cannot be "
            "reattached — so tell the user plainly that the retry starts over. "
            "If a second interruption follows, STOP retrying and reduce the "
            f"job's exposure instead: {_shrink_menu(f)}." + _cost_tail(f)
        )

    if f.cls == "preempted":
        budget = (
            " and the automatic relaunch budget is exhausted"
            if f.budget_exhausted
            else ""
        )
        cause = f"the GPU sandbox was preempted {f.preemptions or 1}×{budget}"
        gpu = f" on {f.gpu_type}" if f.gpu_type else ""
        resumed = (
            " The backend already relaunched it once on the same volume and it "
            "was interrupted again."
            if f.continued
            else ""
        )
        repeated = (
            f" It died the same way on all {f.leases} attempts — exit 137 is "
            "preemption AND out-of-memory AND an operator terminate, and only "
            "the trainer's own OOM signal tells them apart, so if a smaller "
            "batch size or sequence length is plausible here, treat memory "
            "pressure as the likelier cause."
            if f.leases > 2
            else ""
        )
        return (
            f"INFRASTRUCTURE, NOT YOUR CONFIG: {cause}{gpu}.{_stage_clause(f)}"
            f"{resumed}{repeated} LQH Cloud GPU sandboxes are preemptible by construction — "
            "no setting prevents this and nothing about the training config, the "
            "data, or the code caused it. RECOVERY: resubmit under a NEW run "
            "name. A resubmit always starts from step 0 — the previous run's "
            "cloud checkpoints belong to the finished job and cannot be "
            "reattached — so tell the user plainly that the retry starts over. "
            "If a second interruption follows, STOP retrying and reduce the "
            f"job's exposure instead: {_shrink_menu(f)}." + _cost_tail(f)
        )

    if f.cls == "interrupted":
        gpu = f" on {f.gpu_type}" if f.gpu_type else ""
        repeated = (
            f" It died the same way on all {f.leases} attempts, which shifts the "
            "odds toward memory rather than preemption — a preemption twice in a "
            "row on the same job is uncommon."
            if f.leases > 2
            else ""
        )
        return (
            f"KILLED{gpu} (SIGKILL / exit 137).{_stage_clause(f)}{repeated} This exit "
            "code means preemption OR out-of-memory OR an operator terminate, and "
            "only the trainer's own OOM signal tells them apart — it did not fire "
            "here, so the cause is genuinely unknown. Do NOT tell the user this was "
            "definitely infrastructure. RECOVERY: check `artifacts` for anything "
            f"published, read runs/{run_name}/stderr.log for an allocator or CUDA "
            "message, and check whether the config is near the memory limit for this "
            "GPU. If memory is plausible, lower per_device_train_batch_size (raise "
            "gradient_accumulation_steps to keep the effective batch size) or "
            "max_seq_length before resubmitting. If it is not, treat it as an "
            "interruption: resubmit under a NEW run name, which starts from step 0."
            + _cost_tail(f)
        )

    if f.cls == "timeout":
        cap = (
            f"its {f.timeout_minutes}-minute cap"
            if f.timeout_minutes
            else "its wall-clock cap"
        )
        return (
            f"WALL-CLOCK TIMEOUT: the job hit {cap} before finishing."
            f"{_stage_clause(f)} This is a SIZING problem, not a transient one — "
            "an identical resubmit will time out again at the same point. "
            f"RECOVERY: shrink the job so it fits — {_shrink_menu(f)}. "
            f"{_artifact_clause(f)} If the user wants the full job, explain that "
            "it needs more wall clock than the current plan allows and offer to "
            "split it into stages." + _cost_tail(f)
        )

    if f.cls == "oom":
        gpu = f" on {f.gpu_type}" if f.gpu_type else ""
        repeated = (
            f" It died the same way on every one of {f.leases} attempts, which is "
            "the tell that separates OOM from preemption."
            if f.leases > 1
            else ""
        )
        return (
            f"OUT OF MEMORY{gpu}: the trainer exceeded device or host memory."
            f"{repeated} This IS a config lever, not infrastructure. RECOVERY: "
            "lower per_device_train_batch_size (raise gradient_accumulation_steps "
            "to keep the effective batch size), lower max_seq_length, enable "
            "gradient checkpointing, or reduce the model size. Do NOT resubmit "
            "the identical config — it will OOM again."
        )

    if f.cls == "stalled":
        return (
            "STALLED: the sandbox stopped reporting progress and was recovered."
            f"{_stage_clause(f)} Check `artifacts` for anything published before "
            "the stall, then resubmit a smaller job rather than the same one."
        )

    if f.cls == "config":
        return (
            "CONFIGURATION ERROR — the job never reached real training (bad "
            "dataset path or columns, a missing eval set, or credentials). Fix "
            "the input and resubmit; do not retry unchanged and do not describe "
            "this as an infrastructure problem."
        )

    if f.cls == "cancelled":
        return (
            "The job was cancelled (by the user or by an operator). No recovery "
            "action unless the user asks for one."
        )

    if f.cls == "crashed":
        return (
            "The trainer itself raised — this is a real error, not a preemption. "
            f"Read runs/{run_name}/stderr.log (or "
            f"runs/{run_name}/sweep_<config>/stderr.log for one sweep config) "
            "BEFORE proposing anything, and fix the actual exception rather "
            "than resubmitting."
        )
    return (
        "The job failed for a reason the harness could not classify. It may be "
        "the trainer, but it may equally be the launcher, the input bundle, "
        "credentials, or the publish step — do NOT assume a training bug. Read "
        f"runs/{run_name}/stderr.log and the error text above, say what you "
        "actually found, and only then propose a fix. If the error is about "
        "the environment rather than the code, say so plainly instead of "
        "editing the training config."
    )


def completion_note(f: JobFailure | None) -> str:
    """Appended to a SUCCESS notification for a run that survived
    interruptions, so the elapsed time and bill aren't a surprise."""
    if f is None or f.leases <= 1:
        return ""
    times = "once" if f.leases == 2 else f"{f.leases - 1} times"
    return (
        f" The GPU sandbox was interrupted {times} mid-run and automatically "
        "relaunched on the same volume, so the elapsed wall-clock and the "
        "billed GPU time are higher than a clean run — the result itself is "
        "valid. Mention the restart in one clause so the user isn't surprised "
        "by the elapsed time."
    )


def _lease_label(attempt: dict[str, Any]) -> str:
    reason = str(attempt.get("terminal_reason") or "running")
    # "continued", never "resumed from <checkpoint>": the backend knows
    # this lease followed another on the same volume and nothing more.
    if attempt.get("continued"):
        return f"{reason} (continued)"
    return reason


def attempt_lines(snapshot: dict[str, Any] | None) -> list[str]:
    """`Attempts:` / `Billed:` lines for the training_status card.

    Returns [] for a job that ran to completion on one uninterrupted
    lease, so a clean run's card looks exactly as it always has.
    """
    snap = snapshot or {}
    recovery = snap.get("recovery") or {}
    attempts = [a for a in (recovery.get("attempts") or []) if isinstance(a, dict)]
    lease_no = _as_int(recovery.get("lease_no")) or 0
    leases = max(len(attempts), lease_no + 1 if attempts or lease_no else 0)
    lines: list[str] = []

    state = str(snap.get("status") or "")
    # Show the history for every restart, and for a FAILED single lease:
    # a one-lease orphan or timeout is the common case, and it is exactly
    # the case where the user wants to know what it cost. A clean
    # single-lease run still renders nothing.
    if leases > 1 or (leases == 1 and state == "failed"):
        labels = [_lease_label(a) for a in attempts]
        if state not in ("completed", "failed", "cancelled") and len(labels) < leases:
            # The live lease has no attempt row yet (rows are written as
            # leases die), so `continued_count` — which counts rows —
            # cannot see it. But a live lease that follows any dead one
            # IS a continuation, and saying so is the whole point of
            # showing the history while the job is still running.
            labels.append("running (continued)" if attempts else "running")
        noun = "lease" if leases == 1 else "leases"
        parts = (
            [f"{leases} {noun} — {', '.join(labels)}"] if labels
            else [f"{leases} {noun}"]
        )
        # budget_exhausted / state are written by the resume supervisor,
        # which has not shipped yet. Guarded rather than assumed: today
        # the backend never sends them and neither clause is printed.
        if recovery.get("budget_exhausted"):
            parts.append("relaunch budget exhausted")
        elif recovery.get("state"):
            parts.append(str(recovery["state"]))
        lines.append(f"  Attempts: {' · '.join(parts)}")

        billed = _dollars(_as_int(recovery.get("billed_cost_micros")))
        if billed:
            across = "" if leases == 1 else f" across {leases} leases"
            lines.append(f"  Billed: ≈{billed}{across}")
    return lines


def diagnosis_line(snapshot: dict[str, Any] | None, error: str | None = None) -> list[str]:
    """One `Diagnosis:` line naming the class, for a failed job's card."""
    snap = snapshot or {}
    if str(snap.get("status") or "") != "failed":
        return []
    f = classify_job_failure(snap, error)
    # `failed` is the absence of a diagnosis, not one: it is what we
    # report when neither the backend nor the error text identified a
    # cause. Printing "Diagnosis: failed" told the reader nothing while
    # looking like it had told them something.
    if not f.known or f.cls == "failed":
        return []
    hint = {
        "preempted": "cloud infrastructure — a resubmit starts from step 0",
        "orphaned": (
            "the sandbox vanished with no terminal event — usually ours, "
            "but check artifacts and stderr.log first"
        ),
        "interrupted": (
            "SIGKILL — preemption or OOM, indistinguishable; check memory"
        ),
        "timeout": "wall-clock cap — shrink the job, don't repeat it",
        "oom": "memory — lower batch size or sequence length",
        "config": "input error — fix the config, not the infrastructure",
        "crashed": "the trainer raised — read stderr.log",
        "interrupted": "SIGKILL — preemption or OOM, indistinguishable; check memory",
        "stalled": "the sandbox stopped reporting progress",
        "cancelled": "cancelled",
    }.get(f.cls, "")
    return [f"  Diagnosis: {f.cls}{f' ({hint})' if hint else ''}"]


def failure_to_dict(f: JobFailure | None) -> dict[str, Any] | None:
    """Serializable form, for JobStatus.failure and cloud_failure.json."""
    if f is None:
        return None
    return {
        "cls": f.cls,
        "infra": f.infra,
        "leases": f.leases,
        "preemptions": f.preemptions,
        "continued": f.continued,
        "budget_exhausted": f.budget_exhausted,
        "runtime_minutes": f.runtime_minutes,
        "timeout_minutes": f.timeout_minutes,
        "gpu_type": f.gpu_type,
        "last_stage": f.last_stage,
        "billed_micros": f.billed_micros,
        # Without this the supervisor's from_dict round-trip dropped the
        # kind before the completion notice was formatted, so every
        # notification fell back to the training menu — including for
        # data_gen and eval_hf runs that have none of those knobs.
        "kind": f.kind,
        "interruptions": f.interruptions,
        "reasons": list(f.reasons),
    }


def failure_from_dict(data: dict[str, Any] | None) -> JobFailure | None:
    if not data:
        return None
    known: Iterable[str] = JobFailure.__dataclass_fields__  # type: ignore[assignment]
    kwargs = {k: v for k, v in data.items() if k in set(known)}
    reasons = kwargs.get("reasons")
    if isinstance(reasons, list):
        kwargs["reasons"] = tuple(reasons)
    return JobFailure(**kwargs)
