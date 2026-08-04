"""The computed failure taxonomy (lqh/remote/failure.py).

Two things are being pinned here:

1. Classification works both from the backend's structured `recovery`
   object and, for older backends, from the literal error strings the Go
   side emits.
2. The directives never tell the agent the thing that cost us a user:
   that a resubmit resumes anything.
"""

from __future__ import annotations

from lqh.remote.failure import (
    attempt_lines,
    classify_job_failure,
    completion_note,
    describe_failure,
    diagnosis_line,
    failure_from_dict,
    failure_to_dict,
)


def snap(**over):
    base = {
        "status": "failed",
        "started_at": "2026-08-02T10:00:00Z",
        "ended_at": "2026-08-02T13:30:00Z",
        "resource": {"gpu_type": "A100-80GB", "timeout_minutes": 720},
    }
    base.update(over)
    return base


# --- classification from structured fields ----------------------------------


def test_classifies_preemption_with_exhausted_budget():
    f = classify_job_failure(
        snap(
            recovery={
                "billed_cost_micros": 6_400_000,
                "lease_no": 2,
                "failure_class": "preempted",
                "budget_exhausted": True,
                "continued_count": 2,
                "last_progress_stage": "config 3/6",
                "attempts": [
                    {"terminal_reason": "preempted", "runtime_seconds": 7180},
                    {"terminal_reason": "preempted", "runtime_seconds": 5000,
                     "continued": True},
                    {"terminal_reason": "preempted", "runtime_seconds": 900},
                ],
            }
        )
    )
    assert f.cls == "preempted"
    assert f.infra is True
    assert f.leases == 3
    assert f.preemptions == 3
    assert f.continued is True
    assert f.budget_exhausted is True
    assert f.gpu_type == "A100-80GB"
    assert f.last_stage == "config 3/6"
    # Runtime comes from the lease sum, not start→end wall clock, so the
    # gap between a sandbox vanishing and us noticing isn't counted.
    assert f.runtime_minutes == (7180 + 5000 + 900) // 60


def test_budget_exhaustion_inferred_from_max_leases():
    f = classify_job_failure(
        snap(recovery={"lease_no": 5, "failure_class": "preempted", "max_leases": 6})
    )
    assert f.leases == 6
    assert f.budget_exhausted is True


def test_cancelled_status_wins_over_error_text():
    f = classify_job_failure(snap(status="cancelled"), "exit code 137")
    assert f.cls == "cancelled"
    assert f.infra is False


# --- fallback classification from today's production strings ----------------


def test_fallback_orphan_string():
    # internal/cloud/reconciler.go
    f = classify_job_failure(
        snap(), "orphaned: provider has no live sandbox for this job"
    )
    assert f.cls == "orphaned"
    assert f.infra is True


def test_fallback_wall_clock_string():
    # internal/cloud/terminal_gate.go
    f = classify_job_failure(
        snap(),
        "job hit its 720-minute wall-clock timeout before finishing; the job "
        "needs to be smaller — fewer sweep configs, fewer epochs, a smaller "
        "dataset, or a smaller base model",
    )
    assert f.cls == "timeout"
    # A consented wall-clock cap that the job outgrew is not an
    # infrastructure failure, and calling it one contradicted this
    # module's own advice to shrink the job.
    assert f.infra is False


def test_a_bare_exit_124_is_not_called_a_timeout():
    """The backend deliberately withholds the timeout label unless the
    sandbox actually burned its budget — 124 is also an exit code a
    workload produces on its own. Re-deriving it from the raw string
    would undo that guard on the client."""
    assert classify_job_failure(snap(), "exit code 124").cls == "failed"


def test_the_runners_own_disclaimer_is_not_read_as_a_diagnosis():
    """The standard exit-137 message contains the word "preemption"
    precisely BECAUSE the runner cannot tell what happened. Matching
    "preempt" before checking for 137/SIGKILL turned that disclaimer
    into a confirmed diagnosis."""
    f = classify_job_failure(
        snap(),
        "exit code 137 (SIGKILL: worker interruption, preemption, OOM, or termination)",
    )
    assert f.cls == "interrupted"
    assert f.infra is False
    assert "NOT YOUR CONFIG" not in describe_failure(f, "sft_003")


def test_fallback_never_invents_oom_from_a_repeated_sigkill():
    """Exit 137 is preemption AND OOM AND an operator terminate.

    Only the trainer's cooperative OOM sentinel tells them apart
    (PREEMPTION_RELAUNCH.md §2.2), and the backend classifies on that.
    Guessing OOM from the string would send the agent to shrink a batch
    size that was never the problem.
    """
    f = classify_job_failure(snap(), "exit code 137 after 2 worker continuations")
    # Not "preempted" either — that claim is just as unfounded, and it is
    # the one that produced "INFRASTRUCTURE, NOT YOUR CONFIG" for what may
    # have been an un-signalled OOM.
    assert f.cls == "interrupted"
    assert f.infra is False
    text = describe_failure(f, "sft_003")
    assert "NOT YOUR CONFIG" not in text
    assert "preemption OR out-of-memory" in text
    assert "per_device_train_batch_size" in text
    # ...but a run the BACKEND classified as preempted (it ruled OOM out
    # via the trainer's sentinel) should still hear the alternative when
    # it died the same way repeatedly.
    repeated = classify_job_failure(
        snap(recovery={"failure_class": "preempted", "lease_no": 3})
    )
    assert "memory pressure" in describe_failure(repeated, "sft_003")


def test_unattributed_failures_are_not_blamed_on_the_trainer():
    f = classify_job_failure(snap(), "launcher: bundle download failed")
    assert f.cls == "failed"
    text = describe_failure(f, "sft_003")
    assert "do NOT assume a training bug" in text
    assert "launcher" in text

    # A real traceback still gets the trainer directive.
    crashed = classify_job_failure(snap(), "Traceback (most recent call last): ...")
    assert crashed.cls == "crashed"
    assert "The trainer itself raised" in describe_failure(crashed, "sft_003")


def test_fallback_cuda_oom():
    f = classify_job_failure(snap(), "CUDA out of memory. Tried to allocate 2.00 GiB")
    assert f.cls == "oom"


def test_unknown_when_there_is_nothing_to_go_on():
    for empty in (None, {}):
        f = classify_job_failure(empty, None)
        assert f.cls == "unknown"
        assert f.known is False
        # The caller must fall back to today's message byte-for-byte.
        assert describe_failure(f, "run_1") == ""


# --- the invariants that matter ---------------------------------------------


def test_infra_directives_say_step_zero_and_never_say_resume():
    for cls in ("preempted", "orphaned"):
        f = classify_job_failure(snap(recovery={"failure_class": cls, "lease_no": 1}))
        text = describe_failure(f, "sft_003")
        assert "step 0" in text, cls
        # "resumed" describing what the BACKEND did is fine; telling the
        # agent to resume is not. Nothing may suggest the retry continues.
        lowered = text.lower()
        assert "resubmit under a new run name" in lowered, cls
        for lie in ("resume the run", "resume it", "resume training", "continues from"):
            assert lie not in lowered, cls


def test_oom_is_never_blamed_on_infrastructure():
    f = classify_job_failure(snap(recovery={"failure_class": "oom", "lease_no": 0}))
    text = describe_failure(f, "sft_003")
    assert f.infra is False
    assert "NOT YOUR CONFIG" not in text
    assert "not infrastructure" in text
    assert "config lever" in text
    assert "batch_size" in text


def test_timeout_directive_refuses_an_identical_retry():
    f = classify_job_failure(snap(recovery={"failure_class": "timeout", "lease_no": 0}))
    text = describe_failure(f, "sft_003")
    assert "SIZING problem" in text
    assert "720-minute cap" in text
    assert "time out again" in text


def test_infra_directives_name_the_bill_but_promise_nothing():
    f = classify_job_failure(snap(recovery={
        "failure_class": "orphaned", "lease_no": 2,
        "billed_cost_micros": 6_400_000,
    }))
    text = describe_failure(f, "sft_003")
    assert "$6.40" in text
    assert "/feedback" in text
    assert "do not promise a refund" in text
    # Never assert the work is gone without looking: an interrupted run
    # often published a checkpoint and lost only the evaluation.
    assert "Check `artifacts` before telling the user the work is gone" in text


def test_billed_line_uses_the_margin_applied_field():
    """actual_cost_micros is RAW on the wire; the margin is applied at
    read sites (CLAUDE.md §10). Labelling raw cost "Billed" would
    understate what the user actually pays by the margin factor."""
    f = classify_job_failure(snap(
        actual_cost_micros=6_400_000,
        recovery={"failure_class": "orphaned", "lease_no": 1},
    ))
    assert f.billed_micros is None

    with_billed = classify_job_failure(snap(recovery={
        "failure_class": "orphaned", "lease_no": 1,
        "billed_cost_micros": 12_800_000,
    }))
    assert with_billed.billed_micros == 12_800_000


def test_crashed_directive_sends_the_agent_to_stderr():
    f = classify_job_failure(
        snap(), "Traceback (most recent call last):\n  ValueError: expected 2 columns"
    )
    assert f.cls == "crashed"
    text = describe_failure(f, "sft_003")
    assert "runs/sft_003/stderr.log" in text
    assert "not a preemption" in text


# --- rendering helpers ------------------------------------------------------


def test_completion_note_only_fires_for_interrupted_runs():
    clean = classify_job_failure(snap(status="completed", recovery={"lease_no": 0}))
    assert completion_note(clean) == ""

    survived = classify_job_failure(snap(status="completed", recovery={"lease_no": 2}))
    note = completion_note(survived)
    assert "interrupted 2 times" in note
    assert "the result itself is valid" in note


def test_attempt_lines_are_silent_for_a_clean_single_lease_run():
    assert attempt_lines(snap(status="completed")) == []
    assert attempt_lines({}) == []
    assert attempt_lines(None) == []


def test_attempt_lines_render_the_lease_history():
    lines = attempt_lines(
        snap(
            recovery={
                "lease_no": 2,
                "budget_exhausted": True,
                "billed_cost_micros": 6_400_000,
                "attempts": [
                    {"terminal_reason": "preempted"},
                    {"terminal_reason": "preempted", "continued": True},
                    {"terminal_reason": "orphaned"},
                ],
            }
        )
    )
    assert lines[0] == (
        "  Attempts: 3 leases — preempted, preempted (continued), "
        "orphaned · relaunch budget exhausted"
    )
    assert lines[1] == "  Billed: ≈$6.40 across 3 leases"


def test_diagnosis_line_only_for_failed_jobs():
    assert diagnosis_line(snap(status="running")) == []
    assert diagnosis_line(snap(status="completed")) == []
    line = diagnosis_line(snap(), "orphaned: provider has no live sandbox for this job")
    assert line == [
        "  Diagnosis: orphaned (the sandbox vanished with no terminal event — "
        "usually ours, but check artifacts and stderr.log first)"
    ]
    # "failed" is the ABSENCE of a diagnosis. Printing it looked like an
    # answer while saying nothing.
    assert diagnosis_line(snap(), "upstream returned 503") == []


def test_failure_dict_round_trip():
    f = classify_job_failure(snap(recovery={"failure_class": "preempted", "lease_no": 1}))
    again = failure_from_dict(failure_to_dict(f))
    assert again == f
    assert failure_from_dict(None) is None
    # Unknown keys from a newer client must not explode an older one.
    assert failure_from_dict({"cls": "timeout", "not_a_field": 1}).cls == "timeout"


def test_renders_only_what_the_backend_actually_sends():
    """No phrase may depend on a field production does not emit.

    The recovery object today carries lease history, the failure class,
    the furthest stage and the billed cost. It does NOT carry a resume
    budget — that ships with the recovery supervisor — so nothing may
    claim the relaunch budget was exhausted until it does.
    """
    backend_shape = {
        "status": "failed",
        "resource": {"gpu_type": "A100-80GB", "timeout_minutes": 720},
        "recovery": {
            "lease_no": 1,
            "continued_count": 1,
            "failure_class": "preempted",
            "last_progress_stage": "config 3/6",
            "billed_cost_micros": 6_400_000,
            "attempts": [
                {"continuation_no": 0, "terminal_reason": "preempted",
                 "runtime_seconds": 7180, "gpu_type": "A100-80GB"},
                {"continuation_no": 1, "terminal_reason": "preempted",
                 "runtime_seconds": 5000, "gpu_type": "A100-80GB",
                 "continued": True},
            ],
        },
    }
    f = classify_job_failure(backend_shape)
    assert f.budget_exhausted is False
    text = describe_failure(f, "sft_003")
    assert "relaunch budget" not in text
    assert "preempted 2×" in text

    lines = attempt_lines(backend_shape)
    assert lines[0] == "  Attempts: 2 leases — preempted, preempted (continued)"
    assert lines[1] == "  Billed: ≈$6.40 across 2 leases"


def test_backend_failed_reason_is_not_blamed_on_the_trainer():
    """`failed` is the backend's catch-all for every terminal it could
    not attribute — a launcher error, a bundle download, expired
    credentials, a publish failure. Mapping it straight to "crashed"
    made the agent assert "the trainer itself raised" about problems the
    trainer never saw, and sent it editing training configs in response.
    """
    launcher = classify_job_failure(
        snap(
            recovery={"failure_class": "failed", "lease_no": 0},
        ),
        "launcher: bundle download failed (403)",
    )
    assert launcher.cls == "failed"
    text = describe_failure(launcher, "sft_003")
    assert "trainer itself raised" not in text
    assert "launcher" in text

    # A traceback in the text is the one thing that DOES justify it.
    crashed = classify_job_failure(
        snap(recovery={"failure_class": "failed"}),
        "Traceback (most recent call last): ValueError: bad column",
    )
    assert crashed.cls == "crashed"
    assert "trainer itself raised" in describe_failure(crashed, "sft_003")


def test_single_failed_lease_still_reports_what_it_cost():
    """The common case is ONE lease that got orphaned or timed out.
    Hiding the history there hid the billed amount in exactly the
    situation where the user most wants it.
    """
    lines = attempt_lines(
        snap(
            status="failed",
            recovery={
                "lease_no": 0,
                "billed_cost_micros": 1_200_000,
                "attempts": [
                    {"continuation_no": 0, "terminal_reason": "orphaned",
                     "runtime_seconds": 12600, "gpu_type": "A100-80GB"},
                ],
            },
        )
    )
    assert lines == ["  Attempts: 1 lease — orphaned", "  Billed: ≈$1.20"]


def test_a_clean_single_lease_run_still_renders_nothing():
    assert attempt_lines(
        snap(
            status="completed",
            recovery={
                "lease_no": 0,
                "billed_cost_micros": 1_200_000,
                "attempts": [{"continuation_no": 0, "terminal_reason": "completed"}],
            },
        )
    ) == []


def test_a_running_job_shows_its_live_lease():
    """While lease 2 runs, only leases 0 and 1 have rows. The backend
    reports lease_no=2 for the live one; the card must count it, or a
    restart is invisible for exactly as long as it is worth showing."""
    lines = attempt_lines(
        snap(
            status="running",
            recovery={
                "lease_no": 2,
                "continued_count": 1,
                "attempts": [
                    {"continuation_no": 0, "terminal_reason": "preempted"},
                    {"continuation_no": 1, "terminal_reason": "preempted",
                     "continued": True},
                ],
            },
        )
    )
    # The live lease has no attempt row yet, so continued_count cannot
    # see it — but a running lease that follows two dead ones IS a
    # continuation, and that is the state the live view exists to show.
    assert lines[0] == (
        "  Attempts: 3 leases — preempted, preempted (continued), running (continued)"
    )


def test_no_lease_claims_a_checkpoint_resume():
    """A continuation lands on the same volume; whether it picked up a
    checkpoint depends on whether its predecessor lived long enough to
    write one, and nothing in the system observes that."""
    text = describe_failure(
        classify_job_failure(
            snap(
                recovery={
                    "failure_class": "preempted",
                    "lease_no": 1,
                    "continued_count": 1,
                    "attempts": [
                        {"terminal_reason": "preempted"},
                        {"terminal_reason": "preempted", "continued": True},
                    ],
                }
            )
        ),
        "sft_003",
    )
    assert "from its on-volume checkpoint" not in text
    assert "relaunched it once on the same volume" in text


def test_orphaned_states_an_observation_not_a_diagnosis():
    """A sandbox that stops appearing in the provider's list without a
    terminal event is USUALLY a preemption — but it also covers a
    workload that died while the backend was restarting, a failed
    publish, and a clean exit nobody saw. Asserting "NOT YOUR CONFIG"
    would send the agent straight past a real bug in the user's run.
    """
    f = classify_job_failure(snap(recovery={"failure_class": "orphaned", "lease_no": 0}))
    text = describe_failure(f, "sft_003")

    assert "NOT YOUR CONFIG" not in text
    assert "OBSERVATION, not a diagnosis" in text
    assert "stderr.log" in text          # the thing that tells them apart
    assert "`artifacts`" in text
    # It still reaches the right conclusion when nothing points at the run.
    assert "resubmit under a NEW run name" in text
    assert "step 0" in text


def test_a_confirmed_preemption_still_speaks_plainly():
    f = classify_job_failure(snap(recovery={"failure_class": "preempted", "lease_no": 1}))
    text = describe_failure(f, "sft_003")
    assert "INFRASTRUCTURE, NOT YOUR CONFIG" in text


def test_narrow_matchers_do_not_swallow_provider_errors():
    """Bare "invalid" / "not found" / "error:" matched half the strings a
    cloud job can produce, including provider and network errors that
    have nothing to do with the user's configuration."""
    for text in (
        "upstream returned 503: model endpoint not found",
        "s3: invalid response while downloading bundle",
        "error: connection reset by peer",
    ):
        cls = classify_job_failure(snap(), text).cls
        assert cls == "failed", (text, cls)


def test_recovery_advice_matches_the_job_kind():
    """Classification runs on every cloud terminal snapshot, so the
    generic sweep/epochs/base-model menu was handed to data generation
    and eval jobs that have none of those knobs."""
    gen = classify_job_failure(
        snap(kind="data_gen", recovery={"failure_class": "timeout", "lease_no": 0})
    )
    text = describe_failure(gen, "gen_001")
    assert "sweep configs" not in text
    assert "fewer samples" in text
    assert "partial dataset" in text

    ev = classify_job_failure(
        snap(kind="eval_hf", recovery={"failure_class": "timeout", "lease_no": 0})
    )
    assert "fewer eval samples" in describe_failure(ev, "eval_001")

    sft = classify_job_failure(
        snap(kind="train_sft", recovery={"failure_class": "timeout", "lease_no": 0})
    )
    assert "fewer sweep configs" in describe_failure(sft, "sft_003")


def test_only_confirmed_preemptions_are_counted_as_preemptions():
    """"the sandbox was preempted 3×" must mean three confirmed
    preemptions — not two orphan observations and an unattributed
    SIGKILL that happened to end with one."""
    f = classify_job_failure(
        snap(
            recovery={
                "failure_class": "preempted",
                "lease_no": 3,
                "attempts": [
                    {"terminal_reason": "orphaned"},
                    {"terminal_reason": "interrupted"},
                    {"terminal_reason": "orphaned"},
                    {"terminal_reason": "preempted"},
                ],
            }
        )
    )
    assert f.preemptions == 1
    assert f.interruptions == 4
    assert "preempted 1×" in describe_failure(f, "sft_003")


def test_kind_survives_the_disk_round_trip():
    """The supervisor restores the taxonomy through failure_from_dict
    before formatting the completion notice, so a field dropped by
    failure_to_dict is a field production never sees — however many
    direct-call tests prove it works."""
    for kind, expected in (
        ("data_gen", "fewer samples"),
        ("eval_hf", "fewer eval samples"),
        ("train_sft", "fewer sweep configs"),
    ):
        f = classify_job_failure(
            snap(kind=kind, recovery={"failure_class": "timeout", "lease_no": 0})
        )
        restored = failure_from_dict(failure_to_dict(f))
        assert restored == f
        assert expected in describe_failure(restored, "run_1"), kind


def test_the_unclassified_directive_is_not_silence():
    """Honest accounting of what the fallback actually does.

    An unmatched non-empty error becomes class `failed`, which is
    "known", so production DOES add a directive where it previously
    printed nothing. Only a failure with no error text at all leaves the
    old output untouched. The claim to check is the narrow one.
    """
    nothing = classify_job_failure(snap(), "")
    assert nothing.cls == "unknown"
    assert describe_failure(nothing, "sft_003") == ""
    assert diagnosis_line(snap(), "") == []

    unmatched = classify_job_failure(snap(), "upstream returned 503 from the gateway")
    assert unmatched.cls == "failed"
    text = describe_failure(unmatched, "sft_003")
    assert text != "", "an unmatched error DOES get a directive — say so"
    # ...but it must not pretend to a diagnosis it does not have.
    assert "could not classify" in text
    assert "do NOT assume a training bug" in text
    assert diagnosis_line(snap(), "upstream returned 503 from the gateway") == []
