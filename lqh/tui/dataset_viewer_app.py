"""Full-screen presentation for the dataset viewer.

A standalone prompt_toolkit ``Application`` (``full_screen=True`` → alternate
screen, like ``less``): the main bottom-docked TUI suspends itself via
``in_terminal()`` and awaits :func:`run_dataset_viewer`; on exit the terminal
is restored and the chat UI repaints.

All layout math (wrapping, scrolling, cache) lives in the
:class:`~lqh.tui.dataset_viewer.DatasetViewer` model — this module only wires
keys to model methods and paints the model's ANSI output. The body window uses
``wrap_lines=False`` so the model's rich-rendered width is the only wrapping
applied (this is what keeps emoji-heavy content from being scrambled by a
second wrap pass).
"""

from __future__ import annotations

import re

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.utils import get_cwidth

from lqh.tui.dataset_viewer import DatasetViewer

# Footer rows reserved outside the body viewport (status + legend). The
# header's share is measured from its actual rendered line count — a long
# agent-message banner can wrap across several rows on narrow terminals.
_FOOTER_ROWS = 2

_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")

_ZWJ = "\u200d"
_VARIATION_SELECTORS = {"\ufe0e", "\ufe0f"}


def _is_regional_indicator(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def _is_cluster_extender(ch: str) -> bool:
    import unicodedata

    return (
        ch in _VARIATION_SELECTORS
        or 0x1F3FB <= ord(ch) <= 0x1F3FF  # skin tone modifiers
        or unicodedata.combining(ch) != 0
    )


def _iter_clusters(text: str):
    """Yield approximate grapheme clusters (ZWJ sequences, flag pairs,
    modifiers/combining marks). Not full UAX #29, but covers the emoji
    families that must never be split by a width cut."""
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        if _is_regional_indicator(text[i]) and j < n and _is_regional_indicator(text[j]):
            j += 1  # flag = regional-indicator pair
        else:
            while j < n:
                if text[j] == _ZWJ and j + 1 < n:
                    j += 2  # ZWJ joins the next char into this cluster
                elif _is_cluster_extender(text[j]):
                    j += 1
                else:
                    break
        yield text[i:j]
        i = j


def clip_line_to_width(line: str, width: int) -> str:
    """Trim an ANSI line so prompt_toolkit measures it as <= width cells.

    Rich (which wraps the content) and prompt_toolkit (which places it on
    screen) use different Unicode width tables — e.g. a ZWJ family emoji is
    2 cells to rich but 8 to prompt_toolkit. Without this pass, a line that
    rich wrapped to fit can exceed the window in prompt_toolkit's math and
    be clipped mid-escape-sequence, or overflow padding calculations. Here
    the cut is explicit, escape-safe, lands only on grapheme-cluster
    boundaries (an emoji family is kept whole or dropped whole, never split
    into broken fragments), and is terminated with a style reset. Only the
    ragged tail of an emoji-dense line is ever lost.
    """
    used = 0
    out: list[str] = []
    i = 0
    while i < len(line):
        m = _SGR_RE.match(line, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        # Take the next grapheme cluster from the plain-text run.
        cluster = next(_iter_clusters(line[i:]))
        # An SGR escape can interrupt a run; never swallow one into a cluster.
        esc = cluster.find("\x1b")
        if esc > 0:
            cluster = cluster[:esc]
        w = max(0, get_cwidth(cluster))
        if used + w > width:
            out.append("\x1b[0m")
            break
        out.append(cluster)
        used += w
        i += len(cluster)
    return "".join(out)


def build_viewer_app(viewer: DatasetViewer) -> Application:
    """Assemble the full-screen viewer application around a model."""

    def viewport_size() -> tuple[int, int]:
        from prompt_toolkit.application import get_app

        size = get_app().output.get_size()
        header_rows = viewer.header_text(size.columns).count("\n") + 1
        chrome = header_rows + _FOOTER_ROWS
        return size.columns, max(1, size.rows - chrome)

    def clipped(text: str, width: int) -> str:
        return "\n".join(
            clip_line_to_width(line, width) for line in text.splitlines()
        )

    def header_fragments():
        width, _ = viewport_size()
        return ANSI(clipped(viewer.header_text(width), width))

    def body_fragments():
        width, height = viewport_size()
        return ANSI(clipped("\n".join(viewer.visible_lines(width, height)), width))

    def footer_fragments():
        width, _ = viewport_size()
        return ANSI(clipped(
            viewer.status_text(width) + "\n" + viewer.legend_text(width), width,
        ))

    header = Window(
        content=FormattedTextControl(header_fragments),
        dont_extend_height=True,
        wrap_lines=False,
    )
    body = Window(
        content=FormattedTextControl(body_fragments),
        wrap_lines=False,
    )
    footer = Window(
        content=FormattedTextControl(footer_fragments),
        height=2,
        wrap_lines=False,
    )

    kb = KeyBindings()

    def _refresh(event) -> None:
        event.app.invalidate()

    @kb.add("j")
    @kb.add("down")
    def _scroll_down(event):
        viewer.scroll(1)
        _refresh(event)

    @kb.add("k")
    @kb.add("up")
    def _scroll_up(event):
        viewer.scroll(-1)
        _refresh(event)

    @kb.add(" ")
    @kb.add("pagedown")
    @kb.add("c-d")
    def _page_down(event):
        viewer.scroll_page(1)
        _refresh(event)

    @kb.add("b")
    @kb.add("pageup")
    @kb.add("c-u")
    def _page_up(event):
        viewer.scroll_page(-1)
        _refresh(event)

    @kb.add("g")
    @kb.add("home")
    def _top(event):
        viewer.scroll_top()
        _refresh(event)

    @kb.add("G")
    @kb.add("end")
    def _bottom(event):
        viewer.scroll_bottom()
        _refresh(event)

    @kb.add("n")
    @kb.add("right")
    def _next(event):
        viewer.go_next()
        _refresh(event)

    @kb.add("p")
    @kb.add("left")
    def _prev(event):
        viewer.go_prev()
        _refresh(event)

    @kb.add("r")
    def _random(event):
        viewer.go_random()
        _refresh(event)

    @kb.add("q")
    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _close(event):
        event.app.exit()

    # Swallow everything else so stray typing does nothing. Exact-key
    # bindings above still win over <any>. Keys 0-9 and "c" are deliberately
    # unbound (reserved for the future per-sample scoring/comment feature).
    @kb.add(Keys.Any)
    def _sink(event):
        pass

    return Application(
        layout=Layout(HSplit([header, body, footer])),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
    )


async def run_dataset_viewer(viewer: DatasetViewer) -> str:
    """Run the full-screen viewer until the user closes it; return the summary."""
    app = build_viewer_app(viewer)
    await app.run_async()
    return viewer.get_summary()
