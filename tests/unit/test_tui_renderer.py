"""Regression tests for scrollback message rendering."""

from __future__ import annotations

from lqh.tui.renderer import (
    LIQUID_AI_LOGO,
    WELCOME_LOGO,
    render_agent_message,
    render_system_message,
    render_user_message,
    render_welcome,
)
from lqh import __version__


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

    def test_welcome_includes_version(self) -> None:
        assert f"v{__version__}" in render_welcome()

    def test_welcome_tells_the_user_how_to_exit(self) -> None:
        rendered = render_welcome()

        assert "/exit" in rendered
        assert "Ctrl+C twice" in rendered

    def test_welcome_uses_compact_lqh_logo(self) -> None:
        assert len(WELCOME_LOGO) == 6
        assert max(map(len, WELCOME_LOGO)) < 40

    def test_welcome_includes_liquid_ai_ascii_logo(self) -> None:
        assert len(LIQUID_AI_LOGO) == 15
        assert set("".join(LIQUID_AI_LOGO)) == {" ", "L", "Q", "H"}
        assert chr(64) not in "".join(LIQUID_AI_LOGO)
