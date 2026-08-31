"""Regression tests for scrollback message rendering."""

from __future__ import annotations

import os
import re

from lqh.tui import renderer
from lqh.tui.renderer import (
    LIQUID_AI_LOGO,
    WELCOME_LOGO,
    render_agent_message,
    render_system_message,
    render_user_message,
    render_welcome,
)
from lqh import __version__


def _plain(line: str) -> str:
    """Strip ANSI styling so a line's printed width can be measured."""
    return re.sub(r"\x1b\[[0-9;]*m", "", line)


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

    def test_blocks_wrap_to_a_narrow_terminal(self, monkeypatch) -> None:
        """A long message must fit the terminal, not a fixed 100 columns.

        Rendered blocks go to stdout verbatim, so anything wider is
        re-wrapped by the terminal itself into ragged fragments.
        """
        monkeypatch.setattr(
            renderer.shutil,
            "get_terminal_size",
            lambda fallback=(80, 24): os.terminal_size((60, 24)),
        )
        rendered = render_system_message("word " * 80)

        widths = [len(_plain(line)) for line in rendered.splitlines()]
        assert max(widths) <= 60
        assert max(widths) > 40  # still uses the width it has

    def test_blocks_keep_their_width_without_a_terminal(self, monkeypatch) -> None:
        """No tty (piped output): keep the readable 100-column default."""
        def _no_terminal(fallback=(80, 24)):
            return os.terminal_size(fallback)

        monkeypatch.setattr(renderer.shutil, "get_terminal_size", _no_terminal)
        rendered = render_system_message("word " * 80)

        widths = [len(_plain(line)) for line in rendered.splitlines()]
        assert 80 < max(widths) <= 100
