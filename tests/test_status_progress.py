"""Tests for live background-progress in the status bar.

Covers the compact progress formatter, the status-bar summary (progress +
freshness), and that the session-id slot is hidden by default.
"""

from __future__ import annotations

import time

from lqh.train.progress import format_progress_oneline
from lqh.tui import status_bar as status_bar_mod
from lqh.tui.background_tasks import BackgroundTask
from lqh.tui.status_bar import StatusBar


class TestFormatProgressOneline:
    def test_sweep_row_with_percent(self) -> None:
        line, pct = format_progress_oneline({
            "phase": "sweep_config_progress",
            "config_index": 5,
            "n_configs": 6,
            "child_step": 1640,
            "child_max_steps": 2000,
        })
        assert line == "6/6 · step 1640/2000 (82%)"
        assert pct == 82

    def test_sweep_row_without_max(self) -> None:
        line, pct = format_progress_oneline({
            "phase": "sweep_config_progress",
            "config_index": 0,
            "n_configs": 6,
            "child_step": 10,
        })
        assert line == "1/6 · step 10"
        assert pct is None

    def test_plain_run_with_percent(self) -> None:
        line, pct = format_progress_oneline({"step": 500, "max_steps": 1000})
        assert line == "step 500/1000 (50%)"
        assert pct == 50

    def test_plain_run_step_only(self) -> None:
        line, pct = format_progress_oneline({"step": 500})
        assert line == "step 500"
        assert pct is None

    def test_plain_run_step_and_epoch(self) -> None:
        line, pct = format_progress_oneline({"step": 500, "epoch": 1.5})
        assert line == "step 500 · epoch 1.50"
        assert pct is None

    def test_empty_inputs(self) -> None:
        assert format_progress_oneline(None) == ("", None)
        assert format_progress_oneline({}) == ("", None)

    def test_percent_clamped_and_rounded(self) -> None:
        # step beyond max never exceeds 100%.
        _, pct = format_progress_oneline({"step": 1100, "max_steps": 1000})
        assert pct == 100


class TestBgSummary:
    def _bar(self) -> StatusBar:
        return StatusBar()

    def test_progress_and_freshness(self) -> None:
        bar = self._bar()
        bar.bg_tasks = [BackgroundTask(
            "dpo_v1", "train", "dpo_v1", "running", "toka",
            progress="6/6 · step 1640/2000 (82%)",
            updated_at=time.time() - 8,
        )]
        summary = bar._format_bg_summary()
        assert summary.startswith("watching train:dpo_v1@toka · 6/6 · step 1640/2000 (82%)")
        assert "↑" in summary  # freshness shown

    def test_no_progress_keeps_plain_label(self) -> None:
        bar = self._bar()
        bar.bg_tasks = [BackgroundTask("dpo_v1", "train", "dpo_v1", "running", "toka")]
        assert bar._format_bg_summary() == "watching train:dpo_v1@toka"

    def test_progress_without_timestamp_has_no_freshness(self) -> None:
        bar = self._bar()
        bar.bg_tasks = [BackgroundTask(
            "r", "train", "r", "running", None, progress="step 5", updated_at=None,
        )]
        summary = bar._format_bg_summary()
        assert summary == "watching train:r · step 5"

    def test_multi_task_summary_unchanged(self) -> None:
        bar = self._bar()
        bar.bg_tasks = [
            BackgroundTask("a", "train", "a", "running"),
            BackgroundTask("b", "eval", "b", "running"),
        ]
        assert bar._format_bg_summary() == "watching 2 tasks (1 eval, 1 train)"

    def test_format_age(self) -> None:
        assert StatusBar._format_age(8) == "8s"
        assert StatusBar._format_age(190) == "3m"
        assert StatusBar._format_age(4400) == "1h13m"


class TestSessionIdHidden:
    def test_session_id_absent_by_default(self) -> None:
        assert status_bar_mod.SHOW_SESSION_ID is False
        bar = StatusBar()
        bar.session_id = "1a2b3c4d5e6f"
        text = "".join(t for _, t in bar.get_formatted_text())
        assert "📋" not in text
        assert "1a2b3c4d" not in text
