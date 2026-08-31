from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lqh.progress import (
    DPO_HELD_OUT_SHARE,
    EtaEstimate,
    ProgressEvent,
    ProgressReporter,
    TRAINING_END,
    checkpoint_eval_band,
    dpo_overall_fraction,
    estimate_eta,
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


def _gen_rows(
    gaps: list[float], *, total: int = 200, start: datetime | None = None,
) -> list[ProgressEvent]:
    """Sample-completion events spaced by ``gaps`` seconds."""
    origin = start or (datetime.now(timezone.utc) - timedelta(seconds=sum(gaps)))
    rows: list[ProgressEvent] = []
    elapsed = 0.0
    for index, gap in enumerate(gaps, start=1):
        elapsed += gap
        rows.append(ProgressEvent(
            task_kind="data_gen", label="gen", phase="generation",
            phase_label="generating", completed=index, total=total,
            unit="samples", overall_fraction=index / total,
            timestamp=(origin + timedelta(seconds=elapsed)).isoformat(),
        ))
    return rows


def test_eta_survives_a_bursty_concurrent_producer() -> None:
    # Concurrent generation lands samples in bursts, so per-interval rates
    # swing wildly even though throughput is steady. The ETA must still
    # settle instead of flickering out of the status bar.
    gaps = [0.3, 2.4, 0.4, 3.1, 0.3, 0.3, 2.9, 0.4, 3.3, 0.3,
            0.4, 2.6, 0.3, 3.4, 0.3, 0.4, 2.7, 0.3, 3.2, 0.4]
    rows = _gen_rows(gaps * 3, total=200)
    now = datetime.fromisoformat(rows[-1].timestamp).timestamp()
    eta = estimate_eta_seconds(rows, now=now)
    assert eta is not None
    # 60 of 200 samples in 83s; 140 left at that rate is ~3m14s. The old
    # median-of-intervals estimator refused this trace outright.
    assert eta == pytest.approx(194, rel=0.08)
    line, _ = format_event_oneline(rows[-1], history=rows, observed_at=None)
    assert "at current rate" in line


def test_eta_promises_an_estimate_while_warming_up() -> None:
    rows = _gen_rows([0.5] * 6, total=200)
    now = datetime.fromisoformat(rows[-1].timestamp).timestamp()
    assert estimate_eta(rows, now=now) == EtaEstimate(warming_up=True)
    line, _ = format_event_oneline(rows[-1], history=rows, observed_at=None)
    assert "ETA soon" in line
    assert "at current rate" not in line


def test_setup_phase_does_not_promise_an_eta() -> None:
    # `setup` reports once and then blocks on a model load or a checkpoint
    # download. Nothing has moved, so "ETA soon" would be a promise the
    # phase cannot keep.
    row = ProgressEvent(
        task_kind="evaluation", label="eval", phase="setup",
        phase_label="downloading checkpoint", overall_fraction=0,
    )
    assert estimate_eta([row]) == EtaEstimate()
    line, _ = format_event_oneline(row, history=[row])
    assert "ETA" not in line


def test_warm_up_needs_both_a_long_enough_window_and_enough_advances() -> None:
    brief = _gen_rows([2.0] * 6, total=200)
    assert estimate_eta(brief).warming_up is True   # 10s of movement
    sparse = _gen_rows([12.0] * 3, total=200)
    assert estimate_eta(sparse).warming_up is True  # only two real advances


def test_eta_survives_the_callers_truncated_history() -> None:
    # Every caller feeds this a 256-row tail (read_progress_events last_n, and
    # the TUI's foreground history cap). An overnight cloud run emits far more
    # than that, so a gate that compares the retained window against the whole
    # job would hide the ETA for hours. 60k samples, one row per second.
    rows = _gen_rows([1.0] * 4000, total=60_000)
    now = datetime.fromisoformat(rows[-1].timestamp).timestamp()
    truncated = rows[-256:]
    eta = estimate_eta_seconds(truncated, now=now)
    assert eta is not None
    # 56,000 samples left at 1/s — a shade under 16 hours.
    assert eta == pytest.approx(56_000, rel=0.05)
    assert estimate_eta_seconds(rows, now=now) == pytest.approx(eta, rel=0.05)


def _slice_rows(
    base: float, span: float, *, count: int, gap: float,
) -> list[ProgressEvent]:
    """A phase covering ``span`` of the whole job in ``count`` steps."""
    start = datetime.now(timezone.utc) - timedelta(seconds=count * gap)
    return [
        ProgressEvent(
            task_kind="dpo", label="run", phase="held_out_scoring",
            phase_label="held-out eval", completed=i, total=count,
            unit="samples", overall_fraction=base + span * i / count,
            timestamp=(start + timedelta(seconds=i * gap)).isoformat(),
        )
        for i in range(count + 1)
    ]


def test_a_narrow_dpo_phase_still_produces_an_eta() -> None:
    # DPO reserves under 2% of the run for held-out scoring. A gate written in
    # whole-job terms could never be satisfied inside a slice that thin, so
    # the phase would say "ETA soon" for its entire 15 minutes.
    span = dpo_overall_fraction(4, 5, 1.0, 0.9) - dpo_overall_fraction(
        4, 5, 1.0 - DPO_HELD_OUT_SHARE, 0.9,
    )
    assert span < 0.02
    rows = _slice_rows(
        dpo_overall_fraction(4, 5, 1.0 - DPO_HELD_OUT_SHARE, 0.9),
        span, count=300, gap=3.0,
    )
    assert estimate_eta_seconds(rows) is not None


def test_a_thin_phase_resolves_rather_than_promising_forever() -> None:
    # Even a phase covering 0.2% of the run must land on a number: "ETA soon"
    # that never resolves is the failure mode the label exists to avoid.
    rows = _slice_rows(0.10, 0.002, count=240, gap=5.0)
    early = datetime.fromisoformat(rows[2].timestamp).timestamp()
    assert estimate_eta(rows[:3], now=early).warming_up is True
    assert estimate_eta_seconds(rows) is not None


def test_setup_stall_before_the_first_sample_does_not_skew_the_rate() -> None:
    # The phase is announced at fraction 0, then the client and pipeline take
    # a minute to load. Counting that minute against 200 samples' throughput
    # used to inflate the ETA several-fold.
    rows = _gen_rows([1.0] * 60, total=200)
    stalled = [
        ProgressEvent(
            task_kind="data_gen", label="gen", phase="generation",
            phase_label="generating", completed=0, total=200, unit="samples",
            overall_fraction=0,
            timestamp=(
                datetime.fromisoformat(rows[0].timestamp) - timedelta(seconds=60)
            ).isoformat(),
        ),
        *rows,
    ]
    now = datetime.fromisoformat(rows[-1].timestamp).timestamp()
    # 140 samples left at 1/s, regardless of the minute spent starting up.
    assert estimate_eta_seconds(stalled, now=now) == pytest.approx(140, abs=6)


def test_warm_up_resolves_into_a_number_on_one_continuous_stream() -> None:
    rows = _gen_rows([1.0] * 60, total=200)
    seen = []
    for count in range(2, len(rows) + 1):
        window = rows[:count]
        now = datetime.fromisoformat(window[-1].timestamp).timestamp()
        estimate = estimate_eta(window, now=now)
        seen.append("eta" if estimate.seconds is not None else (
            "soon" if estimate.warming_up else "blank"
        ))
    # Promise first, number second, and never a gap back to nothing.
    assert "blank" not in seen
    assert seen[0] == "soon" and seen[-1] == "eta"
    assert seen == sorted(seen, key=["soon", "eta"].index)


def test_remote_warm_up_survives_clock_skew() -> None:
    remote_start = datetime.now(timezone.utc) - timedelta(hours=2)
    rows = _gen_rows([0.5] * 6, total=200, start=remote_start)
    line, _ = format_event_oneline(
        rows[-1], history=rows,
        observed_at=datetime.now(timezone.utc).timestamp(),
    )
    assert "ETA soon" in line


def test_a_repeated_stale_marker_cannot_fake_an_advance() -> None:
    # lqh.remote.cloud re-appends the last row with a fresh timestamp when the
    # backend goes quiet. It must not reset staleness or invent throughput.
    rows = _gen_rows([5.0] * 8, total=200)
    last = datetime.fromisoformat(rows[-1].timestamp).timestamp()
    marker = ProgressEvent(
        task_kind="data_gen", label="gen", phase="generation",
        phase_label="generating", completed=8, total=200, unit="samples",
        overall_fraction=rows[-1].overall_fraction,
        detail="no update from the backend in 10m",
        timestamp=datetime.fromtimestamp(last + 600, timezone.utc).isoformat(),
    )
    assert estimate_eta([*rows, marker], now=last + 601) == EtaEstimate()


def test_stalled_phase_promises_nothing() -> None:
    # Last movement ten minutes ago: neither an ETA nor a promise of one —
    # the status bar's "↑10m" age is the honest signal there.
    rows = _gen_rows(
        [5.0] * 20, total=200,
        start=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    last = datetime.fromisoformat(rows[-1].timestamp).timestamp()
    assert estimate_eta_seconds(rows, now=last + 1) is not None
    assert estimate_eta(rows) == EtaEstimate()
    line, _ = format_event_oneline(rows[-1], history=rows, observed_at=None)
    assert "ETA" not in line


def test_checkpoint_eval_band_spans_one_training_step() -> None:
    """A mid-run band must stay inside the training band, not borrow 0.90."""
    band = checkpoint_eval_band(TRAINING_END, 50, 81)
    assert band is not None
    start, end = band
    assert start == pytest.approx(TRAINING_END * 50 / 81)
    assert end == pytest.approx(TRAINING_END * 51 / 81)
    # The next logged training step must still be able to move forward.
    assert end < TRAINING_END * 55 / 81
    # Last step: the band collapses instead of running past the band end.
    assert checkpoint_eval_band(TRAINING_END, 81, 81) == (
        pytest.approx(TRAINING_END), pytest.approx(TRAINING_END),
    )
    # Unusable step counts give nothing to anchor to.
    assert checkpoint_eval_band(TRAINING_END, 50, 0) is None
    assert checkpoint_eval_band(TRAINING_END, 0, 81) is None


# ---------------------------------------------------------------------------
# The wiring itself: ProgressCallback.on_save -> _run_checkpoint_eval.
#
# lqh/train/sft.py imports the torch stack at module scope, so the fixture
# below stubs whichever of those packages is missing (the unit suite runs
# without `pip install lqh[train]`) and imports the module fresh.
# ---------------------------------------------------------------------------


class _FakeIds:
    """Minimum tensor surface `_run_checkpoint_eval` touches."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def to(self, device: object) -> "_FakeIds":
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, i: int) -> list[int]:
        return self.rows[i]


class _FakeTokenizer:
    def apply_chat_template(self, msgs, **kwargs):  # noqa: ANN001, ANN201
        return {"input_ids": _FakeIds([[1, 2, 3]])}

    def decode(self, ids, **kwargs) -> str:  # noqa: ANN001
        return "generated"


class _FakeModel:
    device = "cpu"

    def eval(self) -> None:
        return None

    def generate(self, input_ids, **kwargs):  # noqa: ANN001, ANN201
        return _FakeIds([[1, 2, 3, 4]])


@pytest.fixture
def sft_module(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    import importlib
    import importlib.machinery
    import importlib.util
    import sys
    import types

    def stub(name: str, **attrs: object) -> None:
        if importlib.util.find_spec(name) is not None:
            return  # real package installed — nothing to fake
        mod = types.ModuleType(name)
        # `datasets` probes find_spec("torch"), which raises on a spec-less
        # stub module.
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    import contextlib

    stub("torch", no_grad=contextlib.nullcontext, bfloat16="bf16")
    stub("datasets", Dataset=type("Dataset", (), {}))
    stub("peft", LoraConfig=type("LoraConfig", (), {}))
    stub(
        "transformers",
        TrainerCallback=type("TrainerCallback", (), {}),
        TrainerControl=type("TrainerControl", (), {}),
        TrainerState=type("TrainerState", (), {}),
        TrainingArguments=type("TrainingArguments", (), {}),
    )
    stub(
        "trl",
        SFTConfig=type("SFTConfig", (), {}),
        SFTTrainer=type("SFTTrainer", (), {}),
    )

    monkeypatch.delitem(sys.modules, "lqh.train.sft", raising=False)
    module = importlib.import_module("lqh.train.sft")
    yield module
    # Never leave a stub-built module cached for the rest of the session.
    sys.modules.pop("lqh.train.sft", None)


def test_on_save_reports_the_mid_run_eval_without_rewinding(
    tmp_path: Path,
    sft_module,  # noqa: ANN001
    sample_conversations,  # noqa: ANN001
    write_chatml_parquet,  # noqa: ANN001
) -> None:
    """The interleave a stalled-looking run actually produces.

    Training reaches step 50 of 81, `on_save` runs the checkpoint eval over
    the whole eval set, then training resumes at step 55. Every generated
    sample must append a row — the stall watchdog stats this file — and the
    resumed step must still be able to move the fraction forward.

    Driven through `ProgressCallback`, so dropping the reporter/band wiring
    in `sft.py` fails here.
    """
    run_dir = tmp_path / "sft_001"
    run_dir.mkdir()
    eval_path = write_chatml_parquet(
        tmp_path / "eval" / "eval.parquet", sample_conversations(3),
    )
    config = {
        "type": "sft",
        "eval_on_checkpoints": True,
        "eval_dataset": str(eval_path),
        "scorer": "judge:small",
    }

    callback = sft_module.ProgressCallback(
        run_dir=run_dir, config=config, tokenizer=_FakeTokenizer(),
    )
    # Real generation takes ~30s a sample; without this the reporter's time
    # throttle would collapse the whole loop into one row.
    callback.reporter.min_interval = 0.0

    callback.on_save(
        None,
        SimpleNamespace(global_step=50, max_steps=81, epoch=1.0),
        None,
        model=_FakeModel(),
    )
    callback.on_log(
        None,
        SimpleNamespace(global_step=55, max_steps=81, epoch=1.1),
        None,
        logs={"loss": 0.5},
    )

    assert (run_dir / "checkpoints" / "step_50" / "predictions.parquet").exists()

    rows = [
        json.loads(line)
        for line in (run_dir / "progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    fractions = [
        row["overall_fraction"] for row in rows if "overall_fraction" in row
    ]
    assert fractions == sorted(fractions), fractions

    eval_rows = [row for row in rows if row.get("phase") == "checkpoint_eval"]
    # One opening row plus one per generated sample.
    assert len(eval_rows) == 4, "each generated sample must refresh the file"
    assert eval_rows[-1]["phase_label"] == "evaluating checkpoint step 50"
    assert eval_rows[-1]["completed"] == 3

    # The band is the sliver of the training band this checkpoint sits on —
    # borrowing the 0.90 final-inference band would pin the rest of training.
    assert eval_rows[0]["overall_fraction"] == pytest.approx(
        TRAINING_END * 50 / 81
    )
    assert eval_rows[-1]["overall_fraction"] == pytest.approx(
        TRAINING_END * 51 / 81
    )
    resumed = [row for row in rows if row.get("phase") == "training"][-1]
    assert resumed["overall_fraction"] == pytest.approx(TRAINING_END * 55 / 81)


def test_checkpoint_eval_stops_inside_the_publish_reserve(
    tmp_path: Path,
    sft_module,  # noqa: ANN001
    sample_conversations,  # noqa: ANN001
    write_chatml_parquet,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eval pass must hand the rest of the cap back to the publish.

    The reported failure: training finished, the unbounded generation pass
    ran past the sandbox's wall-clock cap, and the run was killed before it
    published anything — six GPU-hours of training thrown away and retrained
    from scratch only to get an eval score.
    """
    run_dir = tmp_path / "sft_001"
    run_dir.mkdir()
    eval_path = write_chatml_parquet(
        tmp_path / "eval" / "eval.parquet", sample_conversations(3),
    )
    config = {
        "type": "sft",
        "eval_on_checkpoints": True,
        "eval_dataset": str(eval_path),
        "scorer": "judge:small",
    }

    # Time runs out after the first sample, which is where a per-sample
    # check earns its keep: checking only before the pass would let this
    # one run to the end.
    calls = {"n": 0}

    def fake_past_deadline(*args: object, **kwargs: object) -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(sft_module, "past_deadline", fake_past_deadline)

    callback = sft_module.ProgressCallback(
        run_dir=run_dir, config=config, tokenizer=_FakeTokenizer(),
    )
    callback.reporter.min_interval = 0.0
    callback.on_save(
        None,
        SimpleNamespace(global_step=50, max_steps=81, epoch=1.0),
        None,
        model=_FakeModel(),
    )

    # No predictions and no scoring request: a partial pass would be scored
    # as if it covered the eval set, and the judge stage is unbounded too.
    checkpoint = run_dir / "checkpoints" / "step_50"
    assert not (checkpoint / "predictions.parquet").exists()
    assert not (checkpoint / "eval_request.json").exists()

    rows = [
        json.loads(line)
        for line in (run_dir / "progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    generated = [
        row for row in rows
        if row.get("phase") == "checkpoint_eval" and row.get("completed")
    ]
    assert [row["completed"] for row in generated] == [1]


def test_checkpoint_eval_promises_no_whole_job_eta() -> None:
    """An interlude's rate must not be projected over the rest of the job.

    The mid-run pass crawls through the thin band `checkpoint_eval_band`
    gives it (0.5556 -> 0.5667 over 72 minutes for the reported run). Read as
    a whole-job rate that is ~46 hours remaining, quoted while training has
    half an hour left — the same class of wrong number that talked an agent
    into cancelling three healthy runs.
    """
    base = datetime(2026, 8, 22, 0, 42, tzinfo=timezone.utc)
    start, end = checkpoint_eval_band(TRAINING_END, 50, 81)
    rows = [
        {
            "phase": "checkpoint_eval",
            "phase_label": "evaluating checkpoint step 50",
            "overall_fraction": start + (end - start) * i / 149,
            "completed": i,
            "total": 149,
            "unit": "samples",
            "timestamp": (base + timedelta(seconds=29 * i)).isoformat(),
        }
        for i in range(1, 40)
    ]
    last = (base + timedelta(seconds=29 * 39)).timestamp()
    assert estimate_eta(rows, now=last + 1) == EtaEstimate()
    line, _ = format_event_oneline(rows[-1], history=rows, observed_at=None)
    assert "ETA" not in line
    # The phase still says what it is doing — this suppresses the number,
    # not the row.
    assert "39/149 samples" in line


def test_cap_eval_sources_samples_every_source(sft_module) -> None:  # noqa: ANN001
    """Capping keeps each source represented, evenly spaced and stable."""
    cap = sft_module.cap_eval_sources
    big = [[{"role": "user", "content": str(i)}] for i in range(100)]
    small = [[{"role": "user", "content": f"s{i}"}] for i in range(3)]

    capped = cap([("big", big), ("small", small)], 24)
    kept = {label: convos for label, convos in capped}
    assert sum(len(c) for c in kept.values()) == pytest.approx(24, abs=2)
    # A source 3% the size of the other still contributes to the per-source
    # macro-average instead of vanishing from it.
    assert len(kept["small"]) >= 1
    assert all(conv in big for conv in kept["big"])
    # Evenly spaced, not the first N: a head slice would score only the
    # beginning of the eval set.
    assert kept["big"][0] == big[0]
    assert kept["big"][-1] != big[len(kept["big"]) - 1]
    # Same rows at every checkpoint, so the curve compares like with like.
    assert cap([("big", big), ("small", small)], 24) == capped

    # At or under the limit, and "no cap", are both pass-throughs.
    srcs = [("small", small)]
    assert cap(srcs, 24) is srcs
    assert cap([("big", big)], 0) == [("big", big)]


def test_mid_run_checkpoint_eval_is_capped_and_the_final_one_is_not(
    tmp_path: Path,
    sft_module,  # noqa: ANN001
    sample_conversations,  # noqa: ANN001
    write_chatml_parquet,  # noqa: ANN001
) -> None:
    """The pass that nothing reads must not outrun the training it interrupts.

    149 rows at ~29s each was 72 minutes an L4 spent on a mid-run curve, for
    output only `checkpoints/final/` ever surfaces (feedback #80). The final
    eval — the one whose score is reported — still generates over every row.
    """
    import pyarrow.parquet as pq

    run_dir = tmp_path / "sft_001"
    run_dir.mkdir()
    eval_path = write_chatml_parquet(
        tmp_path / "eval" / "eval.parquet", sample_conversations(60),
    )
    config = {
        "type": "sft",
        "eval_on_checkpoints": True,
        "eval_dataset": str(eval_path),
    }

    callback = sft_module.ProgressCallback(
        run_dir=run_dir, config=config, tokenizer=_FakeTokenizer(),
    )
    callback.reporter.min_interval = 0.0
    callback.on_save(
        None,
        SimpleNamespace(global_step=50, max_steps=81, epoch=1.0),
        None,
        model=_FakeModel(),
    )
    step_dir = run_dir / "checkpoints" / "step_50"
    mid = pq.read_table(step_dir / "predictions.parquet")
    assert mid.num_rows == sft_module.MID_RUN_CHECKPOINT_EVAL_SAMPLES

    # The thinned score gets labelled where it is read side by side with the
    # final one, so the sidecar has to round-trip through that renderer.
    from lqh.tools.handlers import _checkpoint_eval_sampling

    assert _checkpoint_eval_sampling(step_dir) == (mid.num_rows, 60)

    final_dir = run_dir / "checkpoints" / "final"
    final_dir.mkdir(parents=True)
    sft_module._run_checkpoint_eval(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        config=config,
        checkpoint_dir=final_dir,
    )
    assert pq.read_table(final_dir / "predictions.parquet").num_rows == 60
    # Nothing was thinned, so nothing claims it was.
    assert _checkpoint_eval_sampling(final_dir) is None

    # Opt-out restores the old whole-eval-set behaviour.
    uncapped_dir = run_dir / "checkpoints" / "step_60"
    uncapped_dir.mkdir(parents=True)
    sft_module._run_checkpoint_eval(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        config={**config, "checkpoint_eval_samples": 0},
        checkpoint_dir=uncapped_dir,
    )
    assert pq.read_table(uncapped_dir / "predictions.parquet").num_rows == 60
    assert _checkpoint_eval_sampling(uncapped_dir) is None


def test_a_relaunch_that_stops_sampling_clears_the_stale_label(
    tmp_path: Path,
    sft_module,  # noqa: ANN001
    sample_conversations,  # noqa: ANN001
    write_chatml_parquet,  # noqa: ANN001
) -> None:
    """A resume re-runs the same step_N into the same directory.

    So the sidecar has to be cleared, not just written: inheriting the
    previous attempt's file would label a full-set score as sampled — the
    exact confusion the label exists to prevent.
    """
    from lqh.tools.handlers import _checkpoint_eval_sampling

    run_dir = tmp_path / "sft_001"
    step_dir = run_dir / "checkpoints" / "step_50"
    step_dir.mkdir(parents=True)
    eval_path = write_chatml_parquet(
        tmp_path / "eval" / "eval.parquet", sample_conversations(60),
    )
    config = {"eval_on_checkpoints": True, "eval_dataset": str(eval_path)}

    sft_module._run_checkpoint_eval(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        config=config, checkpoint_dir=step_dir,
    )
    assert _checkpoint_eval_sampling(step_dir) is not None

    # Relaunched with the cap turned off: same directory, full eval set.
    sft_module._run_checkpoint_eval(
        model=_FakeModel(), tokenizer=_FakeTokenizer(),
        config={**config, "checkpoint_eval_samples": 0},
        checkpoint_dir=step_dir,
    )
    assert _checkpoint_eval_sampling(step_dir) is None


def test_a_junk_sample_limit_falls_back_to_the_default(sft_module) -> None:  # noqa: ANN001
    """A hand-written config must not turn the cap off by being unparseable."""
    read = sft_module.mid_run_checkpoint_eval_samples
    default = sft_module.MID_RUN_CHECKPOINT_EVAL_SAMPLES
    assert read({}) == default
    assert read({"checkpoint_eval_samples": None}) == default
    assert read({"checkpoint_eval_samples": "lots"}) == default
    assert read({"checkpoint_eval_samples": "8"}) == 8
    assert read({"checkpoint_eval_samples": 0}) == 0
