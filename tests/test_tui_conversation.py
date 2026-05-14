"""Regression tests for TUI conversation scrolling."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from lqh.tui.conversation import ConversationView, MOUSE_SCROLL_LINES


class ConversationViewTests(unittest.TestCase):
    """Exercise the scroll state machine without a full terminal session."""

    def setUp(self) -> None:
        self.view = ConversationView()
        self.window = self.view.make_window()

    def _set_layout(self, *, content_height: int, window_height: int) -> None:
        """Inject the window metrics that prompt_toolkit would provide post-render."""
        self.window.render_info = SimpleNamespace(
            content_height=content_height,
            window_height=window_height,
        )
        self.view.sync_scroll()

    def test_new_content_stays_pinned_to_bottom(self) -> None:
        for i in range(50):
            self.view.append(f"line {i}\n")

        self._set_layout(content_height=51, window_height=10)
        self.assertEqual(self.window.vertical_scroll, 41)

        self.view.append("tail\n")
        self._set_layout(content_height=52, window_height=10)
        self.assertEqual(self.window.vertical_scroll, 42)

    def test_manual_scroll_is_preserved_when_new_content_arrives(self) -> None:
        for i in range(50):
            self.view.append(f"line {i}\n")

        self._set_layout(content_height=51, window_height=10)
        self.view.scroll_up(5)
        self.assertEqual(self.window.vertical_scroll, 36)

        self.view.append("tail\n")
        self._set_layout(content_height=52, window_height=10)
        self.assertEqual(self.window.vertical_scroll, 36)

    def test_scroll_down_repins_when_bottom_is_reached(self) -> None:
        for i in range(50):
            self.view.append(f"line {i}\n")

        self._set_layout(content_height=51, window_height=10)
        self.view.scroll_up(5)
        self.view.scroll_down(5)

        self.assertEqual(self.window.vertical_scroll, 41)

        self.view.append("tail\n")
        self._set_layout(content_height=52, window_height=10)
        self.assertEqual(self.window.vertical_scroll, 42)

    def test_mouse_wheel_updates_scroll_state(self) -> None:
        for i in range(50):
            self.view.append(f"line {i}\n")

        self._set_layout(content_height=51, window_height=10)
        self.view.scroll_up(10)

        self.window.content.mouse_handler(
            MouseEvent(
                position=Point(x=0, y=0),
                event_type=MouseEventType.SCROLL_DOWN,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            )
        )

        self.assertEqual(self.window.vertical_scroll, 41 - 10 + MOUSE_SCROLL_LINES)


if __name__ == "__main__":
    unittest.main()
