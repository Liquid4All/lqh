"""Scrollable conversation view for the lqh TUI."""

from __future__ import annotations

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, AnyFormattedText
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

MOUSE_SCROLL_LINES = 3


class _ConversationControl(FormattedTextControl):
    """Formatted text control with explicit mouse-wheel scrolling."""

    def __init__(self, view: "ConversationView") -> None:
        super().__init__(
            view.get_formatted_text,
            focusable=False,
            show_cursor=False,
            get_cursor_position=view.get_cursor_position,
        )
        self._view = view

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._view.scroll_up(lines=MOUSE_SCROLL_LINES)
            return None

        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._view.scroll_down(lines=MOUSE_SCROLL_LINES)
            return None

        return super().mouse_handler(mouse_event)


class ConversationView:
    """Manages the scrollable conversation display area.

    The window scroll position is owned here instead of being inferred from the
    control cursor. That keeps manual scrolling stable while still allowing the
    view to follow the bottom when desired.
    """

    def __init__(self) -> None:
        self._fragments: list[str] = []
        self._text: str = ""
        self._line_count: int = 1
        self._window_height: int = 0
        self._window: Window | None = None
        self._preferred_scroll: int = 0
        self._follow_bottom: bool = True

    def append(self, ansi_text: str) -> None:
        """Append rendered ANSI text to the conversation."""
        self._fragments.append(ansi_text)
        self._rebuild_text()
        if self._follow_bottom:
            self._pin_to_bottom()
        else:
            self._clamp_scroll()

    def pop(self) -> str | None:
        """Remove and return the last fragment, or None if empty."""
        if not self._fragments:
            return None
        removed = self._fragments.pop()
        self._rebuild_text()
        self._clamp_scroll()
        return removed

    def clear(self) -> None:
        """Clear the conversation view."""
        self._fragments.clear()
        self._text = ""
        self._line_count = 1
        self._preferred_scroll = 0
        self._follow_bottom = True
        if self._window:
            self._window.vertical_scroll = 0

    def scroll_up(self, lines: int = 3) -> None:
        """Scroll up by N lines."""
        self._follow_bottom = False
        self._set_scroll(self._preferred_scroll - lines)

    def scroll_down(self, lines: int = 3) -> None:
        """Scroll down by N lines."""
        target = self._preferred_scroll + lines
        self._set_scroll(target)
        self._follow_bottom = self._preferred_scroll >= self._max_scroll()

    def scroll_to_bottom(self) -> None:
        """Jump back to the bottom."""
        self._follow_bottom = True
        self._pin_to_bottom()

    def sync_scroll(self) -> bool:
        """Clamp scroll after render and re-apply pinned-to-bottom behavior.

        This runs in `Application.after_render`, once prompt_toolkit has
        measured the window height and content height for the current frame.
        """
        if self._window and self._window.render_info:
            self._line_count = self._window.render_info.content_height
            self._window_height = self._window.render_info.window_height

        target = self._max_scroll() if self._follow_bottom else self._clamped_scroll()
        changed = target != self._preferred_scroll
        self._preferred_scroll = target

        if self._window and self._window.vertical_scroll != target:
            self._window.vertical_scroll = target
            changed = True

        return changed

    def _rebuild_text(self) -> None:
        """Refresh cached text and its logical line count."""
        self._text = "".join(self._fragments)
        self._line_count = self._compute_line_count()

    def _set_scroll(self, value: int) -> None:
        """Update the preferred viewport and mirror it onto the live window."""
        self._preferred_scroll = max(0, min(value, self._max_scroll()))
        if self._window:
            self._window.vertical_scroll = self._preferred_scroll

    def _clamped_scroll(self) -> int:
        """Return the current preferred scroll, clamped to the valid range."""
        return max(0, min(self._preferred_scroll, self._max_scroll()))

    def _clamp_scroll(self) -> None:
        """Clamp the preferred scroll in place after content changes."""
        self._set_scroll(self._preferred_scroll)

    def _pin_to_bottom(self) -> None:
        """Move the viewport to the bottom-most valid position."""
        self._set_scroll(self._max_scroll())

    def _max_scroll(self) -> int:
        """Return the maximum valid vertical scroll for the current layout."""
        return max(0, self._line_count - self._window_height)

    def _compute_line_count(self) -> int:
        """Return the number of logical lines in the conversation."""
        return self._text.count("\n") + 1

    def get_formatted_text(self) -> AnyFormattedText:
        """Return the full conversation as ANSI formatted text."""
        return ANSI(self._text)

    def get_cursor_position(self) -> Point:
        """Expose a cursor position that keeps the intended viewport valid."""
        return Point(x=0, y=min(self._preferred_scroll, max(0, self._line_count - 1)))

    def get_vertical_scroll(self, _window: Window) -> int:
        """Return the preferred scroll position for the next render."""
        return self._preferred_scroll

    def make_window(self) -> Window:
        """Create a Window with scroll support."""
        control = _ConversationControl(self)
        self._window = Window(
            content=control,
            wrap_lines=True,
            allow_scroll_beyond_bottom=False,
            get_vertical_scroll=self.get_vertical_scroll,
        )
        return self._window
