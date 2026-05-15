"""Regression tests for TUI conversation scrolling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from lqh.tui.conversation import ConversationView, MOUSE_SCROLL_LINES


@pytest.fixture
def view() -> ConversationView:
    return ConversationView()


@pytest.fixture
def populated_view(view: ConversationView) -> ConversationView:
    """A view pre-loaded with 50 lines of content."""
    for i in range(50):
        view.append(f"line {i}\n")
    return view


@pytest.fixture
def set_layout(view: ConversationView) -> Callable[..., None]:
    """Inject window metrics that ``prompt_toolkit`` would set after render."""
    window = view.make_window()
    # The fixture-built window is needed by callers via view's attached state.
    view._test_window = window  # type: ignore[attr-defined]

    def _apply(*, content_height: int, window_height: int) -> None:
        window.render_info = SimpleNamespace(
            content_height=content_height,
            window_height=window_height,
        )
        view.sync_scroll()

    return _apply


class TestConversationViewScroll:
    """Exercise the scroll state machine without a full terminal session."""

    def test_new_content_stays_pinned_to_bottom(
        self, populated_view: ConversationView, set_layout,
    ) -> None:
        window = populated_view._test_window
        set_layout(content_height=51, window_height=10)
        assert window.vertical_scroll == 41

        populated_view.append("tail\n")
        set_layout(content_height=52, window_height=10)
        assert window.vertical_scroll == 42

    def test_manual_scroll_is_preserved_when_new_content_arrives(
        self, populated_view: ConversationView, set_layout,
    ) -> None:
        window = populated_view._test_window
        set_layout(content_height=51, window_height=10)
        populated_view.scroll_up(5)
        assert window.vertical_scroll == 36

        populated_view.append("tail\n")
        set_layout(content_height=52, window_height=10)
        assert window.vertical_scroll == 36

    def test_scroll_down_repins_when_bottom_is_reached(
        self, populated_view: ConversationView, set_layout,
    ) -> None:
        window = populated_view._test_window
        set_layout(content_height=51, window_height=10)
        populated_view.scroll_up(5)
        populated_view.scroll_down(5)
        assert window.vertical_scroll == 41

        populated_view.append("tail\n")
        set_layout(content_height=52, window_height=10)
        assert window.vertical_scroll == 42

    def test_mouse_wheel_updates_scroll_state(
        self, populated_view: ConversationView, set_layout,
    ) -> None:
        window = populated_view._test_window
        set_layout(content_height=51, window_height=10)
        populated_view.scroll_up(10)

        window.content.mouse_handler(
            MouseEvent(
                position=Point(x=0, y=0),
                event_type=MouseEventType.SCROLL_DOWN,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            )
        )

        assert window.vertical_scroll == 41 - 10 + MOUSE_SCROLL_LINES
