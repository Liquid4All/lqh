from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from lqh.progress import (
    ProgressEvent,
    ProgressReporter,
    dpo_overall_fraction,
    estimate_eta_seconds,
    format_event_oneline,
    nonnegative_int,
    percent_for,
    read_progress_events,
    training_end_for,
)


def test_dpo_fraction_respects_final_scoring_reservation() -> None:
    assert dpo_overall_fraction(4, 5, 1.0, 0.9) == pytest.approx(0.9)
    assert dpo_overall_fraction(7, 5, 1.0, 0.9) == pytest.approx(0.9)


def test_training_reserves_inference_without_a_scorer() -> None:
    assert training_end_for({
        "eval_on_checkpoints": True,
        "eval_dataset": "eval.parquet",
    }) == pytest.approx(0.95)


def test_explicit_zero_iterations_is_preserved() -> None:
    assert nonnegative_int(0, 5) == 0
    assert nonnegative_int(None, 5) == 5


def test_incomplete_progress_never_rounds_to_100() -> None:
    event = ProgressEvent(
        task_kind="evaluation", label="eval", phase="scoring",
        phase_label="judging", overall_fraction=0.9999,
    )
    assert percent_for(event) == 99
    assert percent_for(ProgressEvent(
        task_kind="evaluation", label="eval", phase="completed",
        phase_label="complete", overall_fraction=1, result_ready=True,
    )) == 100


def test_reporter_is_monotonic_and_writes_common_protocol(tmp_path) -> None:
    seen: list[ProgressEvent] = []
    reporter = ProgressReporter(
        task_kind="data_gen", label="Data generation", callback=seen.append,
        run_dir=tmp_path, min_interval=0,
    )
    reporter.update(
        phase="generation", phase_label="generating", completed=5, total=10,
        unit="samples", overall_fraction=0.5,
    )
    reporter.update(
        phase="generation", phase_label="generating", completed=4, total=10,
        unit="samples", overall_fraction=0.4,
    )
    rows = read_progress_events(tmp_path)
    assert [row["overall_fraction"] for row in rows] == [0.5, 0.5]
    assert seen[-1].schema_version == 1


def test_direct_event_write_inherits_run_attempt(tmp_path, monkeypatch) -> None:
    from lqh.progress import write_progress_event

    monkeypatch.setenv("LQH_RUN_ATTEMPT_ID", "attempt-2")
    write_progress_event(tmp_path, ProgressEvent(
        task_kind="sft", label="run", phase="completed",
        phase_label="complete", overall_fraction=1, result_ready=True,
    ))

    assert read_progress_events(tmp_path)[-1]["attempt_id"] == "attempt-2"


def test_reporter_adapts_legacy_three_argument_callback() -> None:
    calls: list[tuple[int, int, int]] = []

    def legacy(completed: int, total: int, concurrency: int) -> None:
        calls.append((completed, total, concurrency))

    reporter = ProgressReporter(
        task_kind="data_gen", label="gen", callback=legacy, min_interval=0,
        legacy_callback=True,
    )
    reporter.update(
        phase="generation", phase_label="generating", completed=2, total=4,
        concurrency=3, overall_fraction=0.5,
    )
    assert calls == [(2, 4, 3)]


def test_legacy_callback_skips_setup_throttle_and_duplicate_terminal() -> None:
    calls: list[tuple[int, int, int]] = []
    reporter = ProgressReporter(
        task_kind="evaluation",
        label="eval",
        callback=lambda completed, total, concurrency: calls.append(
            (completed, total, concurrency)
        ),
        legacy_callback=True,
        min_interval=60,
    )
    reporter.update(
        phase="setup", phase_label="setup", completed=0, total=None,
        overall_fraction=0, force=True,
    )
    for completed in (1, 2, 3):
        reporter.update(
            phase="evaluation", phase_label="evaluating",
            completed=completed, total=3, concurrency=3,
            overall_fraction=completed / 3,
        )
    reporter.update(
        phase="completed", phase_label="ready", completed=3, total=3,
        overall_fraction=1, result_ready=True, force=True,
    )

    assert calls == [(1, 3, 3), (2, 3, 3), (3, 3, 3)]


def test_eta_requires_stable_recent_progress() -> None:
    start = datetime.now(timezone.utc) - timedelta(seconds=40)
    rows = [
        ProgressEvent(
            task_kind="data_gen", label="gen", phase="generation",
            phase_label="generating", completed=i, total=10, unit="samples",
            overall_fraction=i / 10,
            timestamp=(start + timedelta(seconds=i * 5)).isoformat(),
        )
        for i in range(1, 7)
    ]
    eta = estimate_eta_seconds(rows, now=(start + timedelta(seconds=30)).timestamp())
    assert eta is not None
    assert 19 <= eta <= 21
    line, pct = format_event_oneline(rows[-1], history=rows)
    assert pct == 60
    assert "ETA 20s" in line


def test_eta_hidden_after_phase_change() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        ProgressEvent(
            task_kind="evaluation", label="eval", phase="inference",
            phase_label="inference", overall_fraction=i / 20,
            timestamp=(now + timedelta(seconds=i * 5)).isoformat(),
        )
        for i in range(5)
    ]
    rows.append(ProgressEvent(
        task_kind="evaluation", label="eval", phase="scoring",
        phase_label="scoring", overall_fraction=0.5,
        timestamp=(now + timedelta(seconds=30)).isoformat(),
    ))
    assert estimate_eta_seconds(rows, now=(now + timedelta(seconds=30)).timestamp()) is None
