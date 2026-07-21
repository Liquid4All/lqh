"""Tests for the full-screen viewer application layer.

Covers the rich↔prompt_toolkit width-mismatch clipping and drives the real
Application through prompt_toolkit's pipe input (no TTY needed).
"""

from __future__ import annotations

import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

from lqh.tui.dataset_viewer import DatasetViewer
from lqh.tui.dataset_viewer_app import build_viewer_app, clip_line_to_width

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

EMOJI_TEXT = (
    "Family: 👨‍👩‍👧‍👦 flags: 🇦🇹🇨🇭 skin tone: 👍🏽 "
    "VS16: ❤️ ☂️ plain: 🎉🚀 " * 3
)


def strip_ansi(line: str) -> str:
    return _ANSI_RE.sub("", line)


# ---------------------------------------------------------------------------
# clip_line_to_width
# ---------------------------------------------------------------------------

def test_clip_noop_for_fitting_ascii():
    line = "\x1b[1mhello\x1b[0m world"
    assert clip_line_to_width(line, 40) == line


def test_clip_bounds_pt_width_for_zwj_emoji():
    # rich counts a ZWJ family as 2 cells; prompt_toolkit counts 8. A line
    # rich considered fitting can massively over-measure in pt — the clip
    # must bring the pt measurement back under the window width.
    line = "👨‍👩‍👧‍👦" * 10  # rich: 20 cells, pt: 80 cells
    clipped = clip_line_to_width(line, 20)
    assert get_cwidth(strip_ansi(clipped)) <= 20


def test_clip_never_splits_grapheme_clusters():
    family = "👨‍👩‍👧‍👦"  # pt: 8 cells
    line = f"A{family}B"
    # Family doesn't fit after "A" (1+8 > 5): drop it whole, never emit a
    # broken "👨‍" fragment.
    clipped = strip_ansi(clip_line_to_width(line, 5))
    assert clipped == "A"
    assert "‍" not in clipped
    # With room for the whole cluster it is kept intact.
    clipped9 = strip_ansi(clip_line_to_width(line, 9))
    assert clipped9 == f"A{family}"


def test_clip_keeps_flag_pairs_and_skin_tones_whole():
    flags = "🇦🇹🇨🇭"  # each pair: pt 4 cells
    assert strip_ansi(clip_line_to_width(flags, 4)) == "🇦🇹"
    assert strip_ansi(clip_line_to_width(flags, 3)) == ""
    thumbs = "👍🏽"  # base + skin tone modifier: pt 4 cells
    assert strip_ansi(clip_line_to_width(thumbs, 4)) == thumbs
    assert strip_ansi(clip_line_to_width(thumbs, 3)) == ""


def test_clip_preserves_escape_sequences():
    line = "\x1b[31m" + "x" * 30 + "\x1b[0m"
    clipped = clip_line_to_width(line, 10)
    assert clipped.startswith("\x1b[31m")
    assert clipped.endswith("\x1b[0m")  # style reset appended at the cut
    assert strip_ansi(clipped) == "x" * 10


@pytest.mark.parametrize("width", [20, 40, 80])
def test_clip_emoji_fixture_lines(tmp_path, width):
    rows = [{"messages": [{"role": "user", "content": EMOJI_TEXT}]}]
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(rows[0]), encoding="utf-8")
    v = DatasetViewer(p)
    for line in v.body_lines(width):
        assert get_cwidth(strip_ansi(clip_line_to_width(line, width))) <= width


# ---------------------------------------------------------------------------
# Driving the full-screen Application
# ---------------------------------------------------------------------------

class _SizedOutput(DummyOutput):
    """DummyOutput with a fixed size and a real (devnull) fileno."""

    def __init__(self, rows: int = 24, columns: int = 80) -> None:
        self._size = Size(rows=rows, columns=columns)
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def get_size(self) -> Size:
        return self._size

    def fileno(self) -> int:
        return self._devnull


def _write_dataset(tmp_path, n=5):
    rows = [
        {"messages": [
            {"role": "user", "content": f"question {n} 🎉"},
            {"role": "assistant", "content": "\n\n".join(f"para {j}" for j in range(40))},
        ]}
        for n in range(n)
    ]
    pq.write_table(pa.table({
        "messages": [json.dumps(r["messages"]) for r in rows],
        "audio": [None] * len(rows),
    }), tmp_path / "data.parquet")
    return tmp_path / "data.parquet"


async def _drive(viewer: DatasetViewer, keys: list[str], rows=24, columns=80) -> None:
    app = build_viewer_app(viewer)
    with create_pipe_input() as pipe:
        app.input = pipe
        app.output = _SizedOutput(rows=rows, columns=columns)
        for key in keys:
            pipe.send_text(key)
        await app.run_async()


@pytest.mark.asyncio
async def test_app_navigation_and_exit(tmp_path):
    viewer = DatasetViewer(_write_dataset(tmp_path))
    await _drive(viewer, ["j", "j", " ", "n", "n", "p", "g", "G", "x", "5", "q"])
    # "x" and "5" are swallowed; navigation lands on sample 2 (0-indexed 1).
    assert viewer.current_index == 1
    assert viewer.viewed_indices == {0, 1, 2}


@pytest.mark.asyncio
async def test_app_escape_exits(tmp_path):
    viewer = DatasetViewer(_write_dataset(tmp_path))
    await _drive(viewer, ["\x1b"])  # Esc
    assert viewer.current_index == 0


@pytest.mark.asyncio
async def test_app_scrolls_long_sample(tmp_path):
    viewer = DatasetViewer(_write_dataset(tmp_path))
    await _drive(viewer, ["j", "j", "j", "q"])
    assert viewer.scroll_offset == 3


@pytest.mark.asyncio
async def test_app_small_terminal(tmp_path):
    viewer = DatasetViewer(_write_dataset(tmp_path))
    await _drive(viewer, ["j", "n", "q"], rows=10, columns=30)
    assert viewer.current_index == 1
