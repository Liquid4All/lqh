"""Dataset viewer model: loads parquet/json/jsonl data and renders samples.

Pure (no prompt_toolkit dependency) so it is unit-testable without a TTY.
The full-screen presentation lives in ``lqh/tui/dataset_viewer_app.py``.

Three view modes, auto-detected:

- CHAT         — rows with a ``messages`` column (ChatML conversations).
- SCORED_CHAT  — CHAT plus scores, from either inline columns on the rows
                 (``score``/``reasoning``/``kept``/``scorer``, e.g. a scoring
                 run's ``results.parquet``) or a sibling ``scores.parquet``
                 joined by ``sample_index``.
- RECORDS      — anything else: generic per-row key/value display.

Sibling-score alignment: the filter pipeline (``run_data_filter``) writes a
*compacted* ``data.parquet`` next to a ``scores.parquet`` whose
``sample_index`` still refers to pre-filter positions. Alignment therefore
tries, in order: (1) kept-rows-in-order mapping when the kept count matches
the data row count, (2) direct ``sample_index`` mapping when every index is in
range. If neither fits, the sibling is ignored and a warning is surfaced
instead of pairing conversations with the wrong scores.

Rendering contract: all output is produced by a single rich console at the
exact display width, so rich's width math is the only wrapping applied (the
presentation window must use ``wrap_lines=False``). No ``Panel`` borders —
right-edge boxes are what made emoji width disagreements visibly break.

Future (not implemented): per-sample comments/scores. Navigation funnels
through ``_goto`` and ``self.annotations`` is reserved as the storage hook;
keys ``0``-``9`` and ``c`` are left unbound in the presentation app.
"""

from __future__ import annotations

import json
import random
import re
from enum import Enum
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text

SCORES_FILENAME = "scores.parquet"

# Readability cap: on very wide terminals (4K full-screen) prose must not
# span thousands of columns. Content renders at min(terminal, this) cells.
MAX_CONTENT_WIDTH = 120

# Safety cap on rendered lines per sample so one pathological sample cannot
# stall the paint loop; the viewer notes the truncation explicitly.
_MAX_BODY_LINES = 5000

# Agent banner is capped to this many rendered lines so a long message can
# never squeeze the body viewport out of existence on small terminals.
_MAX_BANNER_LINES = 2


def _effective_width(width: int) -> int:
    return max(4, min(int(width), MAX_CONTENT_WIDTH))


_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_sgr(text: str) -> str:
    return _SGR_RE.sub("", text)

# Row fields with dedicated rendering; everything else on a chat row is
# shown as generic metadata.
_KNOWN_CHAT_FIELDS = frozenset(
    {"messages", "audio", "score", "reasoning", "kept", "scorer", "sample_index"}
)
_SCORE_FIELDS = ("score", "reasoning", "kept", "scorer")

# Body indent for message content, in cells.
_CONTENT_INDENT = 4


class ViewMode(Enum):
    CHAT = "chat"
    SCORED_CHAT = "scored chat"
    RECORDS = "records"


def _make_console(width: int) -> tuple[StringIO, Console]:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=width, color_system="truecolor"
    )
    return buf, console


# Role styling
_ROLE_STYLE = {
    "system": ("dim", "🔧 System"),
    "user": ("bold cyan", "👤 User"),
    "assistant": ("bold magenta", "🧪 Assistant"),
    "tool": ("yellow", "🔧 Tool"),
}


def _maybe_json(value: object) -> object:
    """Parse a JSON string, returning the original value on failure."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _load_records(path: Path) -> list[dict]:
    """Load rows from a parquet/jsonl/json file as a list of dicts."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if suffix == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return [r if isinstance(r, dict) else {"value": r} for r in records]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [r if isinstance(r, dict) else {"value": r} for r in data]
        return [data if isinstance(data, dict) else {"value": data}]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_score_rows(scores_path: Path) -> list[dict] | None:
    """Load a scores.parquet sibling; None when missing or unreadable."""
    if not scores_path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(scores_path).to_pylist()
        if not rows or any("sample_index" not in r for r in rows):
            return None
        return rows
    except Exception:
        return None


def _is_scoring_error(reasoning: object) -> bool:
    """True when a score row records a judge failure, not a real grade."""
    try:
        from lqh.scoring import is_scoring_error

        return is_scoring_error(None if reasoning is None else str(reasoning))
    except Exception:
        return str(reasoning or "").startswith(("[Scoring error]", "[Parse error]"))


def _normalize_score_row(row: dict) -> dict:
    """Coerce loosely-typed score fields (arbitrary JSON) in place.

    ``"score": "7.5"`` must not crash float formatting, and ``"kept":
    "false"`` must not be truthy just because it is a non-empty string.
    """
    score = row.get("score")
    if score is not None and not isinstance(score, (int, float)):
        try:
            row["score"] = float(score)
        except (TypeError, ValueError):
            pass  # left as-is; rendered without numeric formatting
    kept = row.get("kept")
    if isinstance(kept, str):
        row["kept"] = kept.strip().lower() not in ("false", "0", "no", "n", "")
    return row


def _align_scores(score_rows: list[dict], total_rows: int) -> dict[int, dict] | None:
    """Map score rows onto displayed row positions; None when nothing fits."""
    try:
        rows = sorted(score_rows, key=lambda r: int(r["sample_index"]))
        # Filter output: data.parquet holds only the kept rows, in original
        # order, while sample_index still counts pre-filter positions.
        kept = [r for r in rows if r.get("kept")]
        if kept and len(kept) == total_rows:
            return dict(enumerate(kept))
        if all(0 <= int(r["sample_index"]) < total_rows for r in rows):
            return {int(r["sample_index"]): r for r in rows}
        return None
    except (TypeError, ValueError):
        return None


class DatasetViewer:
    """Loads a dataset and renders one sample at a time with scrolling."""

    def __init__(self, path: Path, *, agent_message: str | None = None) -> None:
        self._path = path
        self.agent_message = agent_message
        self.scores_warning: str | None = None

        data_path = path
        sibling_scores_path: Path | None = None
        if path.name == SCORES_FILENAME:
            # Pointed directly at scores: view the sibling data (pipeline
            # convention names it data.parquet) with the scores overlaid —
            # but ONLY when that view is lossless. Filtered datasets keep
            # just the kept rows in data.parquet, so redirecting there would
            # hide the dropped samples' scores and reasoning; in that case
            # the requested file itself is shown (records mode, all rows).
            sibling = path.parent / "data.parquet"
            if sibling.exists() and self._lossless_overlay(path, sibling):
                data_path = sibling
                sibling_scores_path = path
        else:
            candidate = path.parent / SCORES_FILENAME
            if candidate.exists() and candidate != path:
                sibling_scores_path = candidate

        records = _load_records(data_path)
        self._data_path = data_path
        self.total_rows = len(records)

        is_chat = bool(records) and all(
            isinstance(r, dict) and "messages" in r for r in records
        )
        scores: dict[int, dict] | None = None
        # All score rows (including dropped/unaligned ones) — the summary
        # must report the full scoring result, not just the displayed subset.
        self._score_rows_all: list[dict] = []
        self._score_total = 0  # denominator for "X of Y samples scored"
        self.scores_source: str | None = None
        # Original sample_index per displayed row, when it differs from the
        # display position (filtered/subset data). Feeds the header and the
        # future annotation feature so feedback names stable identities.
        self.source_indices: list[int | None] = [None] * len(records)
        if is_chat:
            self._rows = []
            inline_scores: dict[int, dict] = {}
            for i, r in enumerate(records):
                inline = {k: r[k] for k in _SCORE_FIELDS if r.get(k) is not None}
                if inline:
                    inline_scores[i] = _normalize_score_row(inline)
                if isinstance(r.get("sample_index"), int):
                    self.source_indices[i] = r["sample_index"]
                self._rows.append({
                    "messages": _maybe_json(r.get("messages")) or [],
                    "audio": _maybe_json(r.get("audio")),
                    "extra": {
                        k: v for k, v in r.items() if k not in _KNOWN_CHAT_FIELDS
                    },
                })
            if inline_scores:
                # Rows carry their own scores (e.g. results.parquet) —
                # aligned by construction, preferred over any sibling.
                scores = inline_scores
                self._score_rows_all = list(inline_scores.values())
                # Rows without score fields are still part of the dataset:
                # "1 of 3 samples scored", never "1 of 1".
                self._score_total = len(records)
                self.scores_source = "inline"
            elif sibling_scores_path is not None:
                score_rows = _load_score_rows(sibling_scores_path)
                if score_rows is None:
                    self.scores_warning = (
                        f"{sibling_scores_path.name} present but unreadable — ignored"
                    )
                else:
                    score_rows = [_normalize_score_row(r) for r in score_rows]
                    scores = _align_scores(score_rows, self.total_rows)
                    if scores is None:
                        self.scores_warning = (
                            f"{sibling_scores_path.name} does not align with this "
                            "dataset (row/index mismatch) — ignored"
                        )
                    else:
                        self._score_rows_all = score_rows
                        self._score_total = len(score_rows)
                        self.scores_source = sibling_scores_path.name
                        for i, row in scores.items():
                            if isinstance(row.get("sample_index"), int):
                                self.source_indices[i] = row["sample_index"]
        else:
            self._rows = records

        self.scores = scores
        if not is_chat:
            self.mode = ViewMode.RECORDS
        elif scores:
            self.mode = ViewMode.SCORED_CHAT
        else:
            self.mode = ViewMode.CHAT

        self.current_index = 0
        self.scroll_offset = 0
        self.viewed_indices: set[int] = {0} if self.total_rows else set()
        # Reserved for the future comment/score feature. Key annotations by
        # source_index(display_index) when it is not None so feedback names
        # stable sample identities, not transient display positions.
        self.annotations: dict[int, dict] = {}

        # Render cache: (index, width) -> body lines.
        self._cache_key: tuple[int, int] | None = None
        self._cache_lines: list[str] = []
        self._last_viewport_height = 20

    @staticmethod
    def _lossless_overlay(scores_path: Path, data_path: Path) -> bool:
        """True when every score row stays visible after redirecting to data."""
        score_rows = _load_score_rows(scores_path)
        if score_rows is None:
            return False
        try:
            records = _load_records(data_path)
        except Exception:
            return False
        if not records or not all(
            isinstance(r, dict) and "messages" in r for r in records
        ):
            return False
        aligned = _align_scores(
            [_normalize_score_row(dict(r)) for r in score_rows], len(records)
        )
        return aligned is not None and len(aligned) == len(score_rows)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @property
    def empty(self) -> bool:
        return self.total_rows == 0

    def _goto(self, index: int) -> None:
        """Single funnel for sample changes (future annotation hook)."""
        self.current_index = index
        self.scroll_offset = 0
        self.viewed_indices.add(index)

    def go_next(self) -> None:
        if self.current_index < self.total_rows - 1:
            self._goto(self.current_index + 1)

    def go_prev(self) -> None:
        if self.current_index > 0:
            self._goto(self.current_index - 1)

    def go_random(self) -> None:
        if self.total_rows > 1:
            # Never a visible no-op: pick among the *other* samples.
            pick = random.randrange(self.total_rows - 1)
            if pick >= self.current_index:
                pick += 1
            self._goto(pick)

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    def _max_offset(self) -> int:
        return max(0, len(self._cache_lines) - self._last_viewport_height)

    def scroll(self, delta: int) -> None:
        self.scroll_offset = max(0, min(self.scroll_offset + delta, self._max_offset()))

    def scroll_page(self, direction: int) -> None:
        page = max(1, self._last_viewport_height - 1)
        self.scroll(direction * page)

    def scroll_top(self) -> None:
        self.scroll_offset = 0

    def scroll_bottom(self) -> None:
        self.scroll_offset = self._max_offset()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def body_lines(self, width: int) -> list[str]:
        """Render the current sample as ANSI lines pre-wrapped at `width`."""
        width = _effective_width(width)
        key = (self.current_index, width)
        if key == self._cache_key:
            return self._cache_lines

        buf, console = _make_console(width)
        try:
            if self.empty:
                console.print(Text("Dataset is empty (0 rows)", style="dim italic"))
            elif self.mode is ViewMode.RECORDS:
                self._render_record(console, self._rows[self.current_index])
            else:
                if self.mode is ViewMode.SCORED_CHAT:
                    self._render_score_block(console)
                self._render_chat(console, self._rows[self.current_index], width)
        except Exception as e:
            # Arbitrary JSON can defeat any shape assumption — degrade to the
            # raw record rather than crashing the viewer.
            console.print(Text(
                f"⚠ could not render this sample ({type(e).__name__}: {e}) — raw record:",
                style="bold red",
            ))
            console.print(Text(json.dumps(
                self._rows[self.current_index], indent=2, ensure_ascii=False, default=str,
            )))

        lines = buf.getvalue().splitlines() or [""]
        if len(lines) > _MAX_BODY_LINES:
            dropped = len(lines) - _MAX_BODY_LINES
            lines = lines[:_MAX_BODY_LINES]
            lines.append(f"\x1b[1;33m⚠ sample truncated — {dropped} more lines not shown\x1b[0m")
        self._cache_key = key
        self._cache_lines = lines
        # A width change can shrink the line count; keep the offset valid.
        self.scroll_offset = min(self.scroll_offset, self._max_offset())
        return self._cache_lines

    def visible_lines(self, width: int, height: int) -> list[str]:
        """The viewport slice of body_lines for the given terminal size."""
        height = max(1, height)
        self._last_viewport_height = height
        lines = self.body_lines(width)
        self.scroll_offset = max(0, min(self.scroll_offset, self._max_offset()))
        return lines[self.scroll_offset : self.scroll_offset + height]

    def _render_score_block(self, console: Console) -> None:
        assert self.scores is not None
        row = self.scores.get(self.current_index)
        if row is None:
            console.print(Text("○ no score for this sample", style="dim"))
            console.print()
            return

        badge = Text()
        score = row.get("score")
        if _is_scoring_error(row.get("reasoning")):
            badge.append("⚠ scoring failed", style="bold red")
        elif score is not None:
            if isinstance(score, (int, float)):
                badge.append(f"★ {score:.2f}", style="bold yellow")
            else:
                badge.append(f"★ {score}", style="bold yellow")
        if row.get("kept") is not None:
            kept = bool(row["kept"])
            badge.append("  ")
            badge.append(
                "kept ✔" if kept else "dropped ✘",
                style="bold green" if kept else "bold red",
            )
        scorer = row.get("scorer")
        if scorer:
            badge.append(f"  scorer: {scorer}", style="dim")
        console.print(badge)

        reasoning = row.get("reasoning")
        if reasoning:
            console.print(Padding(Text(str(reasoning), style="dim italic"), (0, 0, 0, 2)))
        console.print()

    def _render_chat(self, console: Console, row: dict, width: int) -> None:
        extra = row.get("extra")
        if extra:
            for key, value in extra.items():
                meta = Text()
                meta.append(f"{key}: ", style="dim bold")
                parsed = _maybe_json(value)
                if isinstance(parsed, (dict, list)):
                    meta.append(
                        json.dumps(parsed, ensure_ascii=False, default=str),
                        style="dim",
                    )
                else:
                    meta.append(str(parsed), style="dim")
                console.print(meta)
            console.print()

        messages = row.get("messages") or []
        audio = row.get("audio")  # dict mapping message index -> base64 wav, or None
        if not isinstance(messages, list):
            messages = [{"role": "unknown", "content": str(messages)}]

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                console.print(Text(str(msg)))
                continue
            role = msg.get("role", "unknown")
            if not isinstance(role, str):
                role = str(role)  # unhashable/odd roles must not crash the lookup
            content = msg.get("content", "")
            style, label = _ROLE_STYLE.get(role, ("", f"  {role}"))

            # Tool messages: show tool name if available
            if role == "tool":
                name = msg.get("name", "")
                tool_call_id = msg.get("tool_call_id", "")
                if name:
                    label = f"🔧 Tool ({name})"
                elif tool_call_id:
                    label = f"🔧 Tool [{tool_call_id[:8]}]"

            console.print(Text(f"  {label}", style=style))

            indent = (0, 0, 0, _CONTENT_INDENT)
            if content:
                if role == "assistant" and isinstance(content, str):
                    console.print(Padding(Markdown(content), indent))
                elif isinstance(content, list):
                    # Multi-part content (e.g., vision messages)
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                console.print(Padding(Text(part.get("text", "")), indent))
                            else:
                                console.print(Padding(
                                    Text(f"[{part.get('type', '?')}]", style="dim"),
                                    indent,
                                ))
                        else:
                            console.print(Padding(Text(str(part)), indent))
                else:
                    console.print(Padding(Text(str(content)), indent))

            # Show tool_calls if present. Arguments may arrive as a JSON
            # string or an already-parsed object; show them in full — the
            # viewer scrolls, and reviewing tool calls is a primary use.
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                tool_calls = None
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if not isinstance(fn, dict):
                        fn = {}
                    fn_name = fn.get("name", "?")
                    fn_args = fn.get("arguments", "{}")
                    if not isinstance(fn_args, str):
                        fn_args = json.dumps(fn_args, ensure_ascii=False, default=str)
                    console.print(Padding(
                        Text(f"-> {fn_name}({fn_args})", style="dim yellow"), indent,
                    ))

            # Audio indicator
            if isinstance(audio, dict) and str(i) in audio:
                console.print(Padding(Text("🔊 audio attached", style="dim magenta"), indent))

            console.print()  # spacing between messages

    def _render_record(self, console: Console, row: dict) -> None:
        if not isinstance(row, dict):
            console.print(Text(str(row)))
            return
        for key, value in row.items():
            console.print(Text(str(key), style="bold cyan"))
            indent = (0, 0, 0, _CONTENT_INDENT)
            parsed = _maybe_json(value)
            if isinstance(parsed, (dict, list)):
                console.print(Padding(
                    Text(json.dumps(parsed, indent=2, ensure_ascii=False, default=str)),
                    indent,
                ))
            elif parsed is None:
                console.print(Padding(Text("null", style="dim"), indent))
            else:
                console.print(Padding(Text(str(parsed)), indent))
            console.print()

    def header_text(self, width: int) -> str:
        """Banner (agent message) + sample position line, as ANSI.

        May span multiple lines (long banner, narrow terminal, warning) —
        the presentation layer must size the header window to the actual
        line count, not assume a fixed height.
        """
        width = _effective_width(width)
        buf, console = _make_console(width)
        if self.agent_message:
            # Cap the banner so a long message can never squeeze the body
            # viewport away — the full text is in the agent conversation.
            inner_buf, inner_console = _make_console(width)
            inner_console.print(
                Text(f"💬 {self.agent_message}", style="bold bright_yellow")
            )
            banner_lines = inner_buf.getvalue().splitlines()
            if len(banner_lines) > _MAX_BANNER_LINES:
                from rich.cells import cell_len

                banner_lines = banner_lines[:_MAX_BANNER_LINES]
                if cell_len(_strip_sgr(banner_lines[-1])) < width:
                    banner_lines[-1] += "\x1b[2m…\x1b[0m"
            buf.write("\n".join(banner_lines) + "\n")
        else:
            # Always give first-time users an orientation line (the "what am
            # I looking at and how do I leave" feedback from DATAVIEWER.md).
            console.print(Text(
                "💬 Review the samples below — press q when done",
                style="dim",
            ))
        if self.scores_warning:
            console.print(Text(f"⚠ {self.scores_warning}", style="dim yellow"))

        title = Text()
        title.append(f"Sample {min(self.current_index + 1, self.total_rows)}", style="bold bright_cyan")
        title.append(f" of {self.total_rows}", style="dim")
        source = self.source_index(self.current_index)
        if source is not None and source != self.current_index:
            title.append(f" (source #{source})", style="dim")
        title.append(f" · {self._data_path.name}", style="dim italic")
        if self.mode is not ViewMode.CHAT:
            title.append(f" · {self.mode.value}", style="dim")
        if self.scores_source:
            title.append(f" · scores: {self.scores_source}", style="dim")
        console.print(Rule(title=title, style="bright_cyan", characters="─"))
        return buf.getvalue().rstrip("\n")

    def source_index(self, display_index: int) -> int | None:
        """Stable pre-filter sample_index for a displayed row, if known."""
        if 0 <= display_index < len(self.source_indices):
            return self.source_indices[display_index]
        return None

    def legend_text(self, width: int) -> str:
        """Always-visible keybinding help.

        Picks the widest legend variant that fits, so scroll, sample
        navigation, and the exit key stay visible even on tiny terminals.
        """
        from rich.cells import cell_len

        width = _effective_width(width)
        variants: list[list[tuple[str, str]]] = [
            [
                ("↑↓/jk", "scroll"), ("Space/b", "page"), ("g/G", "top/bottom"),
                ("←→/pn", "sample"), ("r", "random"), ("q/Esc", "done"),
            ],
            [
                ("↑↓/jk", "scroll"), ("Space/b", "page"),
                ("←→/pn", "sample"), ("r", "random"), ("q/Esc", "done"),
            ],
            [("↑↓/jk", "scroll"), ("←→/pn", "sample"), ("r", "random"), ("q/Esc", "done")],
            [("↑↓/jk", "scroll"), ("←→/pn", "sample"), ("q/Esc", "done")],
            [("jk", "scroll"), ("np", "sample"), ("q", "done")],
            [("jk", ""), ("np", ""), ("q", "done")],
            # Pathologically narrow: the exit key wins over everything else.
            [("q", "done")],
            [("q", "")],
        ]

        def plain(items: list[tuple[str, str]]) -> str:
            return " · ".join(f"{k} {d}".strip() for k, d in items)

        entries = variants[-1]
        for candidate in variants:
            if cell_len(plain(candidate)) <= width:
                entries = candidate
                break

        buf, console = _make_console(width)
        legend = Text()
        for i, (keys, desc) in enumerate(entries):
            if i:
                legend.append(" · ", style="dim")
            legend.append(keys, style="bold bright_cyan")
            if desc:
                legend.append(f" {desc}", style="dim")
        console.print(legend, no_wrap=True, overflow="ellipsis")
        return buf.getvalue().rstrip("\n")

    def position_summary(self) -> str:
        """Status line: sample position and position within the sample."""
        if self.empty:
            return "0 samples"
        total_lines = len(self._cache_lines) or 1
        top = self.scroll_offset + 1
        bottom = min(self.scroll_offset + self._last_viewport_height, total_lines)
        pct = int(bottom * 100 / total_lines)
        return (
            f"Sample {self.current_index + 1}/{self.total_rows}"
            f" · lines {top}–{bottom}/{total_lines} ({pct}%)"
            f" · viewed {len(self.viewed_indices)}"
        )

    def status_text(self, width: int) -> str:
        """Width-aware ANSI rendering of position_summary (never wraps)."""
        width = _effective_width(width)
        buf, console = _make_console(width)
        console.print(
            Text(self.position_summary(), style="dim"),
            no_wrap=True,
            overflow="ellipsis",
        )
        return buf.getvalue().rstrip("\n")

    # ------------------------------------------------------------------
    # Agent summary
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Return a summary string for the agent."""
        if self.empty:
            return f"Dataset {self._path.name} is empty (0 rows)."

        viewed = sorted(self.viewed_indices)
        # 1-based positions, matching the "Sample N of M" header the user saw.
        positions = [i + 1 for i in viewed]
        if len(positions) <= 10:
            positions_str = ", ".join(str(p) for p in positions)
        else:
            positions_str = ", ".join(str(p) for p in positions[:10]) + f" ... ({len(positions)} total)"

        summary = (
            f"User viewed {len(viewed)} sample(s) (positions {positions_str}, 1-based) "
            f"of {self.total_rows} total rows in {self._data_path.name}"
            f" [{self.mode.value} mode]."
        )
        # Stable identities for filtered/subset data: display position N may
        # really be source sample_index 47.
        sources = [self.source_index(i) for i in viewed]
        if any(s is not None and s != i for i, s in zip(viewed, sources)):
            src_str = ", ".join("?" if s is None else str(s) for s in sources[:10])
            if len(sources) > 10:
                src_str += " ..."
            summary += f" Source sample_index of viewed: {src_str}."
        # Stats run over ALL score rows (including rows dropped by a filter,
        # which are absent from the display mapping) and, like the scoring
        # pipeline, exclude judge failures from the aggregates.
        score_rows = self._score_rows_all
        if score_rows:
            errors = sum(1 for r in score_rows if _is_scoring_error(r.get("reasoning")))
            values = [
                r["score"] for r in score_rows
                if isinstance(r.get("score"), (int, float))
                and not _is_scoring_error(r.get("reasoning"))
            ]
            if values:
                summary += (
                    f" Scores: {len(values)} of {self._score_total} samples scored,"
                    f" mean {sum(values) / len(values):.2f},"
                    f" min {min(values):.2f}, max {max(values):.2f}."
                )
            if errors:
                summary += f" {errors} sample(s) failed scoring."
            kept_flags = [r["kept"] for r in score_rows if r.get("kept") is not None]
            if kept_flags:
                summary += f" Kept {sum(bool(k) for k in kept_flags)}/{len(kept_flags)}."
        if self.scores_warning:
            summary += f" Note: {self.scores_warning}."
        return summary
