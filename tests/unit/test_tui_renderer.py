"""Regression tests for scrollback message rendering."""

from __future__ import annotations

from lqh.tui.renderer import (
    render_agent_message,
    render_system_message,
    render_user_message,
)


class TestRenderer:
    """Top-level messages render as separated indented blocks."""

    def test_agent_messages_start_new_indented_block(self) -> None:
        rendered = render_agent_message("Hello")
        lines = rendered.splitlines()

        assert rendered.startswith("\n")
        assert not lines[1].startswith("  ")
        assert lines[2].startswith("  ")
        assert "Liquid" in rendered

    def test_user_messages_indent_content(self) -> None:
        rendered = render_user_message("hello there")

        assert not rendered.splitlines()[1].startswith("  ")
        assert "You" in rendered
        assert "\n  hello there" in rendered

    def test_inline_system_messages_skip_block_spacing(self) -> None:
        rendered = render_system_message("Type your response:", separated=False)

        assert not rendered.startswith("\n")
        assert not rendered.startswith("  ")
        assert "Type your response:" in rendered
