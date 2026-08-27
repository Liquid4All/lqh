"""The deadline callback is the only reason a capped run yields anything.

If it stops too late the sandbox dies before publish (today's failure);
if it stops when time remains it throws away paid GPU time.
"""

from __future__ import annotations

import json
import time

from lqh.train.deadline import DeadlineStopCallback, deadline_epoch


class _Control:
    should_training_stop = False
    should_save = False


class _State:
    global_step = 42
    max_steps = 100
    epoch = 1.5


def test_no_deadline_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("LQH_DEADLINE_EPOCH", raising=False)
    cb = DeadlineStopCallback(tmp_path, label="sft")
    control = _Control()
    cb.on_step_end(None, _State(), control)
    assert not control.should_training_stop
    assert not cb.triggered
    assert not (tmp_path / "stopped_early.json").exists()


def test_keeps_training_while_the_reserve_is_intact(tmp_path):
    cb = DeadlineStopCallback(
        tmp_path, label="sft", reserve_seconds=60, deadline=time.time() + 3600,
    )
    control = _Control()
    cb.on_step_end(None, _State(), control)
    assert not control.should_training_stop
    assert not cb.triggered


def test_stops_saves_and_records_when_the_reserve_is_gone(tmp_path):
    cb = DeadlineStopCallback(
        tmp_path, label="sft", reserve_seconds=600, deadline=time.time() + 30,
    )
    control = _Control()
    cb.on_step_end(None, _State(), control)
    assert control.should_training_stop
    assert control.should_save
    assert cb.triggered and cb.stopped_at_step == 42

    marker = json.loads((tmp_path / "stopped_early.json").read_text())
    assert marker["reason"] == "wall_clock_deadline"
    assert marker["stopped_at_step"] == 42
    assert marker["max_steps"] == 100

    # Fires once: a second step must not rewrite the marker or re-print.
    control2 = _Control()
    cb.on_step_end(None, _State(), control2)
    assert not control2.should_training_stop


def test_deadline_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LQH_DEADLINE_EPOCH", "1800000000")
    assert deadline_epoch() == 1800000000.0
    for bad in ("", "   ", "later", "0", "-5"):
        monkeypatch.setenv("LQH_DEADLINE_EPOCH", bad)
        assert deadline_epoch() is None
