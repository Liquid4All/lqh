"""Unit tests for the dataset viewer model (lqh/tui/dataset_viewer.py).

The model is pure (rich → ANSI strings, no prompt_toolkit/TTY), so mode
detection, scroll math, and width discipline are all testable directly.
"""

from __future__ import annotations

import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lqh.tui.dataset_viewer import DatasetViewer, ViewMode

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(line: str) -> str:
    return _ANSI_RE.sub("", line)


def cell_width(line: str) -> int:
    from rich.cells import cell_len

    return cell_len(strip_ansi(line))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMOJI_TEXT = (
    "Family: 👨‍👩‍👧‍👦 flags: 🇦🇹🇨🇭 skin tone: 👍🏽 "
    "VS16: ❤️ ☂️ plain: 🎉🚀 " * 3
)


def make_chat_rows() -> list[dict]:
    long_content = "\n".join(f"line {i}: some assistant prose." for i in range(120))
    return [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello **world** [not markup]"},
                {"role": "assistant", "content": "Hi! *markdown* here.\n\n- a\n- b"},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": EMOJI_TEXT},
                {"role": "assistant", "content": EMOJI_TEXT},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "long one"},
                {"role": "assistant", "content": long_content},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search", "arguments": json.dumps({"q": "x" * 100})}}
                    ],
                },
                {"role": "tool", "name": "search", "content": "result"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at this"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                },
            ],
        },
    ]


def write_chat_parquet(path, rows=None, audio=None):
    rows = rows if rows is not None else make_chat_rows()
    table = pa.table({
        "messages": [json.dumps(r["messages"]) for r in rows],
        "audio": [json.dumps(audio) if audio else None for _ in rows],
    })
    pq.write_table(table, path)
    return path


def write_scores_parquet(path, n=3, variant="scorer"):
    cols = {
        "sample_index": list(range(n)),
        "score": [7.5 - i for i in range(n)],
        "reasoning": [f"reasoning for sample {i}" for i in range(n)],
    }
    if variant == "scorer":
        cols["scorer"] = ["judge-v2"] * n
    else:
        cols["kept"] = [i % 2 == 0 for i in range(n)]
    pq.write_table(pa.table(cols), path)
    return path


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def test_chat_mode_from_parquet(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)
    assert v.mode is ViewMode.CHAT
    assert v.total_rows == 3
    assert not v.empty


def test_scored_mode_with_sibling(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    write_scores_parquet(tmp_path / "scores.parquet")
    v = DatasetViewer(p)
    assert v.mode is ViewMode.SCORED_CHAT
    assert v.scores is not None and 0 in v.scores


def test_scored_mode_kept_variant(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    write_scores_parquet(tmp_path / "scores.parquet", variant="kept")
    v = DatasetViewer(p)
    assert v.mode is ViewMode.SCORED_CHAT
    body = "\n".join(v.body_lines(100))
    assert "kept ✔" in strip_ansi(body)


def test_opening_scores_directly_joins_data(tmp_path):
    write_chat_parquet(tmp_path / "data.parquet")
    scores = write_scores_parquet(tmp_path / "scores.parquet")
    v = DatasetViewer(scores)
    assert v.mode is ViewMode.SCORED_CHAT
    assert v.total_rows == 3  # data rows, not score rows


def test_scores_without_data_sibling_is_records(tmp_path):
    scores = write_scores_parquet(tmp_path / "scores.parquet")
    v = DatasetViewer(scores)
    assert v.mode is ViewMode.RECORDS


def test_malformed_scores_falls_back_to_chat(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    # Valid parquet but missing sample_index → malformed as a scores file.
    pq.write_table(pa.table({"unrelated": [1, 2]}), tmp_path / "scores.parquet")
    v = DatasetViewer(p)
    assert v.mode is ViewMode.CHAT


def test_records_mode_generic_parquet(tmp_path):
    p = tmp_path / "generic.parquet"
    pq.write_table(
        pa.table({"name": ["a", "b"], "value": [1, 2], "nested": ['{"k": 1}', None]}), p
    )
    v = DatasetViewer(p)
    assert v.mode is ViewMode.RECORDS
    body = strip_ansi("\n".join(v.body_lines(80)))
    assert "name" in body and "nested" in body


def test_jsonl_chat(tmp_path):
    p = tmp_path / "data.jsonl"
    rows = make_chat_rows()
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    v = DatasetViewer(p)
    assert v.mode is ViewMode.CHAT
    assert v.total_rows == 3


def test_json_list_records(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    v = DatasetViewer(p)
    assert v.mode is ViewMode.RECORDS
    assert v.total_rows == 2


def test_empty_dataset(tmp_path):
    p = tmp_path / "data.parquet"
    pq.write_table(pa.table({"messages": pa.array([], type=pa.string())}), p)
    v = DatasetViewer(p)
    assert v.empty
    assert "empty" in v.get_summary()
    assert v.body_lines(80)  # renders the empty notice, doesn't crash
    assert v.position_summary() == "0 samples"


# ---------------------------------------------------------------------------
# Width discipline (the emoji fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [40, 80, 120, 220])
def test_body_lines_respect_width(tmp_path, width):
    p = write_chat_parquet(tmp_path / "data.parquet")
    write_scores_parquet(tmp_path / "scores.parquet")
    v = DatasetViewer(p)
    for idx in range(v.total_rows):
        v._goto(idx)
        for line in v.body_lines(width):
            assert cell_width(line) <= width, f"overflow at sample {idx}: {line!r}"


@pytest.mark.parametrize("width", [40, 80, 220])
def test_header_and_legend_respect_width(tmp_path, width):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p, agent_message="Review these samples for tone 🎯")
    for text in (v.header_text(width), v.legend_text(width)):
        for line in text.splitlines():
            assert cell_width(line) <= width


def test_no_panel_borders_in_output(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)
    body = strip_ansi("\n".join(v.body_lines(100)))
    assert "╭" not in body and "│" not in body  # no rich Panel boxes


# ---------------------------------------------------------------------------
# Scrolling
# ---------------------------------------------------------------------------

def make_long_viewer(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)
    v._goto(2)  # the 120-line sample
    return v


def test_scroll_clamps(tmp_path):
    v = make_long_viewer(tmp_path)
    lines = v.body_lines(80)
    assert len(lines) > 50

    visible = v.visible_lines(80, 20)
    assert len(visible) == 20
    assert visible == lines[:20]

    v.scroll(-5)
    assert v.scroll_offset == 0
    v.scroll(10)
    assert v.scroll_offset == 10
    v.scroll(10_000)
    assert v.scroll_offset == len(lines) - 20  # clamped to bottom
    assert v.visible_lines(80, 20) == lines[-20:]


def test_scroll_page_and_jumps(tmp_path):
    v = make_long_viewer(tmp_path)
    v.visible_lines(80, 30)
    v.scroll_page(1)
    assert v.scroll_offset == 29  # viewport - 1
    v.scroll_bottom()
    assert v.scroll_offset == len(v.body_lines(80)) - 30
    v.scroll_top()
    assert v.scroll_offset == 0


def test_scroll_resets_on_navigation(tmp_path):
    v = make_long_viewer(tmp_path)
    v.visible_lines(80, 20)
    v.scroll(30)
    assert v.scroll_offset == 30
    v.go_prev()
    assert v.scroll_offset == 0
    assert v.current_index == 1


def test_scroll_reclamps_on_width_change(tmp_path):
    v = make_long_viewer(tmp_path)
    v.visible_lines(60, 20)
    v.scroll_bottom()
    old = v.scroll_offset
    # Wider render → fewer wrapped lines → offset must stay in range.
    visible = v.visible_lines(220, 20)
    assert v.scroll_offset <= max(0, len(v.body_lines(220)) - 20)
    assert visible
    assert old >= v.scroll_offset


def test_visible_lines_short_sample(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)  # sample 0 is short
    visible = v.visible_lines(80, 500)
    assert len(visible) == len(v.body_lines(80))
    v.scroll(5)
    assert v.scroll_offset == 0  # nothing to scroll


# ---------------------------------------------------------------------------
# Header / legend / status
# ---------------------------------------------------------------------------

def test_header_contents(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p, agent_message="Check tool calls")
    header = strip_ansi(v.header_text(100))
    assert "💬 Check tool calls" in header
    assert "Sample 1" in header and "of 3" in header and "data.parquet" in header


def test_legend_degrades_when_narrow(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)
    wide = strip_ansi(v.legend_text(120))
    narrow = strip_ansi(v.legend_text(50))
    assert "q/Esc" in wide and "random" in wide
    assert "q/Esc done" in narrow and "random" not in narrow


@pytest.mark.parametrize("width", [20, 30, 50, 80, 120, 220])
def test_essential_keys_always_in_legend(tmp_path, width):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)
    legend = strip_ansi(v.legend_text(width))
    assert "…" not in legend  # never truncated mid-legend
    # Scroll, sample navigation, and exit stay visible at every width.
    assert "jk" in legend
    assert "pn" in legend or "np" in legend
    assert "q" in legend and "done" in legend


def test_position_summary(tmp_path):
    v = make_long_viewer(tmp_path)
    v.visible_lines(80, 20)
    v.scroll(10)
    s = v.position_summary()
    assert s.startswith("Sample 3/3")
    assert "lines 11–30/" in s


# ---------------------------------------------------------------------------
# Summary for the agent
# ---------------------------------------------------------------------------

def test_summary_counts_viewed(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)
    v.go_next()
    v.go_next()
    s = v.get_summary()
    assert "3 sample(s)" in s
    assert "chat mode" in s


def test_summary_includes_score_stats(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    write_scores_parquet(tmp_path / "scores.parquet")
    v = DatasetViewer(p)
    s = v.get_summary()
    assert "3 samples scored" in s and "mean" in s


def test_summary_kept_stats(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    write_scores_parquet(tmp_path / "scores.parquet", variant="kept")
    v = DatasetViewer(p)
    assert "Kept 2/3" in v.get_summary()


# ---------------------------------------------------------------------------
# Audio indicator
# ---------------------------------------------------------------------------

def test_audio_indicator(tmp_path):
    rows = [{"messages": [{"role": "assistant", "content": "hi"}]}]
    p = write_chat_parquet(tmp_path / "data.parquet", rows=rows, audio={"0": "base64=="})
    v = DatasetViewer(p)
    assert "🔊 audio attached" in strip_ansi("\n".join(v.body_lines(80)))


# ---------------------------------------------------------------------------
# show_file handler threading
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_show_file_message(tmp_path):
    from lqh.tools.handlers import handle_show_file

    write_chat_parquet(tmp_path / "data.parquet")
    result = await handle_show_file(tmp_path, path="data.parquet", message="Review these")
    assert result.show_file_path == "data.parquet"
    assert result.show_file_message == "Review these"

    result = await handle_show_file(tmp_path, path="data.parquet")
    assert result.show_file_message is None  # backward compatible


@pytest.mark.asyncio
async def test_handle_show_file_routes_jsonl(tmp_path):
    from lqh.tools.handlers import handle_show_file

    (tmp_path / "data.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    result = await handle_show_file(tmp_path, path="data.jsonl")
    assert result.show_file_path == "data.jsonl"


# ---------------------------------------------------------------------------
# Score alignment (filtered datasets)
# ---------------------------------------------------------------------------

def test_filtered_dataset_scores_align_by_kept(tmp_path):
    # run_data_filter writes a COMPACTED data.parquet (kept rows only) while
    # scores.parquet keeps pre-filter sample_index values. Displayed row j
    # must get the j-th kept score row, not the score at index j.
    rows = make_chat_rows()  # 3 rows displayed
    write_chat_parquet(tmp_path / "data.parquet", rows=rows)
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2, 3, 4],
        "score": [1.0, 8.0, 2.0, 9.0, 7.0],
        "reasoning": ["r0", "r1", "r2", "r3", "r4"],
        "kept": [False, True, False, True, True],
    }), tmp_path / "scores.parquet")

    v = DatasetViewer(tmp_path / "data.parquet")
    assert v.mode is ViewMode.SCORED_CHAT
    assert v.scores_warning is None
    assert v.scores[0]["score"] == 8.0  # original index 1
    assert v.scores[1]["score"] == 9.0  # original index 3
    assert v.scores[2]["score"] == 7.0  # original index 4


def test_kept_scores_on_unfiltered_data_align_by_index(tmp_path):
    # Viewing the ORIGINAL (pre-filter) dataset: indices are all in range,
    # so the direct sample_index mapping applies.
    rows = make_chat_rows()  # 3 rows
    write_chat_parquet(tmp_path / "data.parquet", rows=rows)
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2],
        "score": [1.0, 8.0, 2.0],
        "reasoning": ["r0", "r1", "r2"],
        "kept": [False, True, False],
    }), tmp_path / "scores.parquet")

    v = DatasetViewer(tmp_path / "data.parquet")
    assert v.scores[0]["score"] == 1.0
    assert v.scores[2]["score"] == 2.0


def test_misaligned_scores_ignored_with_warning(tmp_path):
    # Sibling from a different/bigger dataset: indices out of range and no
    # kept-count match — must NOT be paired with the wrong conversations.
    write_chat_parquet(tmp_path / "data.parquet")  # 3 rows
    pq.write_table(pa.table({
        "sample_index": [5, 6, 7, 8],
        "score": [1.0, 2.0, 3.0, 4.0],
        "reasoning": ["a", "b", "c", "d"],
        "scorer": ["j"] * 4,
    }), tmp_path / "scores.parquet")

    v = DatasetViewer(tmp_path / "data.parquet")
    assert v.mode is ViewMode.CHAT
    assert v.scores is None
    assert v.scores_warning and "align" in v.scores_warning
    assert "⚠" in strip_ansi(v.header_text(100))
    assert "Note:" in v.get_summary()


def test_unreadable_scores_sibling_warns(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    pq.write_table(pa.table({"unrelated": [1, 2]}), tmp_path / "scores.parquet")
    v = DatasetViewer(p)
    assert v.mode is ViewMode.CHAT
    assert v.scores_warning and "unreadable" in v.scores_warning


# ---------------------------------------------------------------------------
# Inline scored ChatML (results.parquet shape)
# ---------------------------------------------------------------------------

def test_inline_scored_results_parquet(tmp_path):
    # A scoring run's results.parquet: sample_index, messages, score,
    # reasoning — scores live on the rows themselves, no sibling file.
    rows = make_chat_rows()
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2],
        "messages": [json.dumps(r["messages"]) for r in rows],
        "score": [7.0, 3.5, 9.0],
        "reasoning": ["good", "meh", "great"],
    }), tmp_path / "results.parquet")

    v = DatasetViewer(tmp_path / "results.parquet")
    assert v.mode is ViewMode.SCORED_CHAT
    body = strip_ansi("\n".join(v.body_lines(100)))
    assert "★ 7.00" in body and "good" in body
    assert "3 samples scored" in v.get_summary()


def test_inline_scores_preferred_over_sibling(tmp_path):
    rows = make_chat_rows()
    pq.write_table(pa.table({
        "messages": [json.dumps(r["messages"]) for r in rows],
        "score": [7.0, 3.5, 9.0],
        "reasoning": ["inline"] * 3,
    }), tmp_path / "results.parquet")
    write_scores_parquet(tmp_path / "scores.parquet")  # sibling would differ

    v = DatasetViewer(tmp_path / "results.parquet")
    assert v.scores[0]["reasoning"] == "inline"


def test_extra_metadata_columns_rendered(tmp_path):
    rows = make_chat_rows()[:1]
    pq.write_table(pa.table({
        "messages": [json.dumps(rows[0]["messages"])],
        "source_id": ["dataset-v2/00042"],
        "meta": [json.dumps({"lang": "de"})],
    }), tmp_path / "data.parquet")

    v = DatasetViewer(tmp_path / "data.parquet")
    body = strip_ansi("\n".join(v.body_lines(120)))
    assert "source_id: dataset-v2/00042" in body
    assert "lang" in body


def test_jsonl_gets_sibling_scores(tmp_path):
    rows = make_chat_rows()
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    write_scores_parquet(tmp_path / "scores.parquet")
    v = DatasetViewer(p)
    assert v.mode is ViewMode.SCORED_CHAT


# ---------------------------------------------------------------------------
# Tool-call arguments
# ---------------------------------------------------------------------------

def test_tool_call_arguments_as_object(tmp_path):
    # JSON data can carry function.arguments as a parsed object, not a
    # string — must render, not crash on dict slicing.
    rows = [{"messages": [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "run", "arguments": {"cmd": "ls", "n": 3}}}],
    }]}]
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(rows[0]), encoding="utf-8")
    v = DatasetViewer(p)
    body = strip_ansi("\n".join(v.body_lines(100)))
    assert '-> run({"cmd": "ls", "n": 3})' in body


def test_tool_call_arguments_not_truncated(tmp_path):
    long_args = json.dumps({"query": "x" * 200})
    rows = [{"messages": [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search", "arguments": long_args}}],
    }]}]
    write_chat_parquet(tmp_path / "data.parquet", rows=rows)
    v = DatasetViewer(tmp_path / "data.parquet")
    body = strip_ansi("\n".join(v.body_lines(80))).replace("\n", "").replace(" ", "")
    assert "x" * 200 in body  # full arguments survive (wrapped, not cut)


# ---------------------------------------------------------------------------
# Header banner wrapping (viewport math contract)
# ---------------------------------------------------------------------------

def test_long_banner_wraps_to_multiple_header_lines(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    msg = "Please review these samples carefully for tool-call correctness and tone"
    v = DatasetViewer(p, agent_message=msg)
    header = v.header_text(30)
    assert len(header.splitlines()) > 2  # banner wraps; app must measure this
    for line in header.splitlines():
        assert cell_width(line) <= 30


def test_status_text_never_wraps(tmp_path):
    v = make_long_viewer(tmp_path)
    v.visible_lines(80, 20)
    for width in (20, 40, 200):
        status = v.status_text(width)
        assert "\n" not in status
        assert cell_width(status) <= width


# ---------------------------------------------------------------------------
# on_show_file callback compatibility
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_show_file_new_signature():
    from lqh.agent import _call_show_file

    calls = []

    async def new_style(path: str, message: str | None = None) -> str | None:
        calls.append((path, message))
        return "ok"

    assert await _call_show_file(new_style, "a.parquet", "hi") == "ok"
    assert calls == [("a.parquet", "hi")]


@pytest.mark.asyncio
async def test_call_show_file_legacy_single_arg():
    from lqh.agent import _call_show_file

    calls = []

    async def legacy(path: str) -> str | None:
        calls.append(path)
        return None

    assert await _call_show_file(legacy, "a.parquet", "ignored") is None
    assert calls == ["a.parquet"]  # invoked exactly once, no TypeError


@pytest.mark.asyncio
async def test_call_show_file_var_positional():
    from lqh.agent import _call_show_file

    calls = []

    async def splat(*args) -> str | None:
        calls.append(args)
        return "s"

    assert await _call_show_file(splat, "p", None) == "s"
    assert calls == [("p", None)]


@pytest.mark.asyncio
async def test_call_show_file_keyword_only_message():
    from lqh.agent import _call_show_file

    calls = []

    async def kwonly(path: str, *, message: str | None = None) -> str | None:
        calls.append((path, message))
        return "k"

    assert await _call_show_file(kwonly, "p", "hi") == "k"
    assert calls == [("p", "hi")]


@pytest.mark.asyncio
async def test_call_show_file_var_keyword():
    from lqh.agent import _call_show_file

    calls = []

    async def kwargs_style(path: str, **kwargs) -> str | None:
        calls.append((path, kwargs.get("message")))
        return "v"

    assert await _call_show_file(kwargs_style, "p", "m") == "v"
    assert calls == [("p", "m")]


# ---------------------------------------------------------------------------
# Full scoring-result summaries (filtered data, judge failures)
# ---------------------------------------------------------------------------

def test_filtered_summary_reports_full_scoring_result(tmp_path):
    # 5 samples scored, 3 kept: the summary must say 3/5 and aggregate all
    # five scores, not just the kept subset shown on screen.
    write_chat_parquet(tmp_path / "data.parquet")  # 3 displayed rows
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2, 3, 4],
        "score": [1.0, 8.0, 2.0, 9.0, 7.0],
        "reasoning": ["r"] * 5,
        "kept": [False, True, False, True, True],
    }), tmp_path / "scores.parquet")

    v = DatasetViewer(tmp_path / "data.parquet")
    s = v.get_summary()
    assert "Kept 3/5" in s
    assert "5 of 5 samples scored" in s
    assert "mean 5.40" in s  # (1+8+2+9+7)/5


def test_scoring_errors_excluded_from_stats(tmp_path):
    write_chat_parquet(tmp_path / "data.parquet")  # 3 rows
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2],
        "score": [8.0, 0.0, 6.0],
        "reasoning": ["fine", "[Scoring error] upstream 429", "ok"],
        "scorer": ["j"] * 3,
    }), tmp_path / "scores.parquet")

    v = DatasetViewer(tmp_path / "data.parquet")
    s = v.get_summary()
    assert "2 of 3 samples scored" in s
    assert "mean 7.00" in s  # the error's 0.0 is excluded
    assert "min 6.00" in s
    assert "1 sample(s) failed scoring" in s

    v._goto(1)
    body = strip_ansi("\n".join(v.body_lines(100)))
    assert "scoring failed" in body
    assert "★ 0.00" not in body


# ---------------------------------------------------------------------------
# Loosely-typed JSON score values
# ---------------------------------------------------------------------------

def test_string_score_and_kept_normalized(tmp_path):
    rows = [{
        "messages": [{"role": "user", "content": "hi"}],
        "score": "7.5",
        "reasoning": "ok",
        "kept": "false",
    }]
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(rows[0]), encoding="utf-8")

    v = DatasetViewer(p)
    assert v.mode is ViewMode.SCORED_CHAT
    body = strip_ansi("\n".join(v.body_lines(100)))
    assert "★ 7.50" in body       # "7.5" did not crash float formatting
    assert "dropped ✘" in body    # "false" is not truthy
    assert "mean 7.50" in v.get_summary()


def test_unparseable_score_rendered_raw(tmp_path):
    rows = [{
        "messages": [{"role": "user", "content": "hi"}],
        "score": "excellent",
        "reasoning": "ok",
    }]
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(rows[0]), encoding="utf-8")
    v = DatasetViewer(p)
    body = strip_ansi("\n".join(v.body_lines(100)))
    assert "★ excellent" in body  # shown as-is, no ValueError


# ---------------------------------------------------------------------------
# Default instruction banner + score provenance
# ---------------------------------------------------------------------------

def test_default_banner_without_agent_message(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)
    header = strip_ansi(v.header_text(100))
    assert "press q when done" in header


def test_header_shows_score_provenance(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    write_scores_parquet(tmp_path / "scores.parquet")
    v = DatasetViewer(p)
    assert "scores: scores.parquet" in strip_ansi(v.header_text(120))

    rows = make_chat_rows()
    pq.write_table(pa.table({
        "messages": [json.dumps(r["messages"]) for r in rows],
        "score": [1.0, 2.0, 3.0],
        "reasoning": ["r"] * 3,
    }), tmp_path / "results.parquet")
    v2 = DatasetViewer(tmp_path / "results.parquet")
    assert "scores: inline" in strip_ansi(v2.header_text(120))


# ---------------------------------------------------------------------------
# Stable sample identity (source indices)
# ---------------------------------------------------------------------------

def test_source_index_surfaced_for_filtered_data(tmp_path):
    write_chat_parquet(tmp_path / "data.parquet")  # 3 displayed rows
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2, 3, 4],
        "score": [1.0, 8.0, 2.0, 9.0, 7.0],
        "reasoning": ["r"] * 5,
        "kept": [False, True, False, True, True],
    }), tmp_path / "scores.parquet")

    v = DatasetViewer(tmp_path / "data.parquet")
    assert v.source_index(0) == 1
    assert v.source_index(1) == 3
    assert v.source_index(2) == 4
    v._goto(1)
    assert "(source #3)" in strip_ansi(v.header_text(120))


def test_source_index_hidden_when_identical(tmp_path):
    rows = make_chat_rows()
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2],
        "messages": [json.dumps(r["messages"]) for r in rows],
        "score": [1.0, 2.0, 3.0],
        "reasoning": ["r"] * 3,
    }), tmp_path / "results.parquet")
    v = DatasetViewer(tmp_path / "results.parquet")
    assert v.source_index(0) == 0
    assert "source #" not in strip_ansi(v.header_text(120))


# ---------------------------------------------------------------------------
# Tiny terminals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [8, 10, 14])
def test_exit_key_survives_tiny_terminals(tmp_path, width):
    from prompt_toolkit.utils import get_cwidth

    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)
    legend = strip_ansi(v.legend_text(width))
    assert "q" in legend
    # Both width models agree the legend fits, so nothing can clip "q".
    assert cell_width(legend) <= width
    assert get_cwidth(legend) <= width


def test_tiny_terminal_rewraps_instead_of_clipping(tmp_path):
    # At width 8 the body must be REWRAPPED at 8 columns, not rendered at a
    # wider floor and then clipped (which would silently drop characters).
    content = "abcdefghijklmnopqrstuvwxyz0123456789"
    rows = [{"messages": [{"role": "user", "content": content}]}]
    write_chat_parquet(tmp_path / "data.parquet", rows=rows)
    v = DatasetViewer(tmp_path / "data.parquet")
    lines = v.body_lines(8)
    for line in lines:
        assert cell_width(line) <= 8
    flat = "".join(strip_ansi(line).strip() for line in lines)
    assert content in flat  # every character survived the narrow wrap


@pytest.mark.parametrize("width", [400, 3840])
def test_wide_terminals_capped_for_readability(tmp_path, width):
    from lqh.tui.dataset_viewer import MAX_CONTENT_WIDTH

    rows = [{"messages": [{"role": "user", "content": "word " * 300}]}]
    write_chat_parquet(tmp_path / "data.parquet", rows=rows)
    v = DatasetViewer(tmp_path / "data.parquet", agent_message="hi")
    for line in v.body_lines(width) + v.header_text(width).splitlines():
        assert cell_width(line) <= MAX_CONTENT_WIDTH


# ---------------------------------------------------------------------------
# Malformed / loosely shaped chat rows must not crash
# ---------------------------------------------------------------------------

def test_unhashable_role_and_bad_tool_call_shapes(tmp_path):
    record = {"messages": [
        {"role": ["not", "hashable"], "content": "odd role"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": "not-a-dict"}]},
        {"role": "assistant", "content": "", "tool_calls": "not-a-list"},
        "just a string message",
    ]}
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(record), encoding="utf-8")
    v = DatasetViewer(p)
    body = strip_ansi("\n".join(v.body_lines(80)))  # must not raise
    assert "odd role" in body


def test_render_falls_back_to_raw_record_on_error(tmp_path, monkeypatch):
    p = write_chat_parquet(tmp_path / "data.parquet")
    v = DatasetViewer(p)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(v, "_render_chat", boom)
    body = strip_ansi("\n".join(v.body_lines(80)))
    assert "could not render this sample" in body
    assert "messages" in body  # raw record shown instead


def test_giant_sample_truncated_with_notice(tmp_path):
    content = "\n".join(f"l{i}" for i in range(6000))
    rows = [{"messages": [{"role": "user", "content": content}]}]
    write_chat_parquet(tmp_path / "data.parquet", rows=rows)
    v = DatasetViewer(tmp_path / "data.parquet")
    lines = v.body_lines(80)
    assert len(lines) <= 5001
    assert "truncated" in strip_ansi(lines[-1])


# ---------------------------------------------------------------------------
# Partial inline scores, banner cap, random navigation, scores.parquet intent
# ---------------------------------------------------------------------------

def test_partial_inline_scores_use_dataset_denominator(tmp_path):
    rows = make_chat_rows()
    pq.write_table(pa.table({
        "messages": [json.dumps(r["messages"]) for r in rows],
        "score": [7.0, None, None],
        "reasoning": ["good", None, None],
    }), tmp_path / "results.parquet")
    v = DatasetViewer(tmp_path / "results.parquet")
    assert "1 of 3 samples scored" in v.get_summary()


def test_banner_capped_at_two_lines(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")
    msg = "very long instruction " * 20
    v = DatasetViewer(p, agent_message=msg)
    header = v.header_text(40)
    # 2 banner lines + rule (no warning here) — never squeezes the viewport.
    assert len(header.splitlines()) <= 3
    for line in header.splitlines():
        assert cell_width(line) <= 40


def test_go_random_never_repeats_current(tmp_path):
    p = write_chat_parquet(tmp_path / "data.parquet")  # 3 rows
    v = DatasetViewer(p)
    for _ in range(30):
        before = v.current_index
        v.go_random()
        assert v.current_index != before


def test_scores_with_dropped_rows_opened_directly_shows_all_rows(tmp_path):
    # Filtered run: data.parquet keeps 1 of 3 samples. Opening scores.parquet
    # directly must show ALL score rows (records mode), not silently redirect
    # to the compacted data file and hide the dropped samples' reasoning.
    write_chat_parquet(tmp_path / "data.parquet", rows=make_chat_rows()[:1])
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2],
        "score": [8.0, 1.0, 2.0],
        "reasoning": ["keep", "drop-a", "drop-b"],
        "kept": [True, False, False],
    }), tmp_path / "scores.parquet")

    v = DatasetViewer(tmp_path / "scores.parquet")
    assert v.mode is ViewMode.RECORDS
    assert v.total_rows == 3  # every score row inspectable, incl. dropped
    v._goto(1)
    assert "drop-a" in strip_ansi("\n".join(v.body_lines(80)))


def test_summary_uses_one_based_positions_and_source_ids(tmp_path):
    write_chat_parquet(tmp_path / "data.parquet")  # 3 displayed rows
    pq.write_table(pa.table({
        "sample_index": [0, 1, 2, 3, 4],
        "score": [1.0, 8.0, 2.0, 9.0, 7.0],
        "reasoning": ["r"] * 5,
        "kept": [False, True, False, True, True],
    }), tmp_path / "scores.parquet")
    v = DatasetViewer(tmp_path / "data.parquet")
    v.go_next()
    s = v.get_summary()
    assert "positions 1, 2, 1-based" in s or "(positions 1, 2, 1-based)" in s
    assert "Source sample_index of viewed: 1, 3" in s
