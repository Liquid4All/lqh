"""Regression tests for scrollback message rendering."""

from __future__ import annotations

import unittest

from lqh.tui.renderer import render_agent_message, render_system_message, render_user_message


class RendererTests(unittest.TestCase):
    """Verify top-level messages render as separated indented blocks."""

    def test_agent_messages_start_new_indented_block(self) -> None:
        rendered = render_agent_message("Hello")

        self.assertTrue(rendered.startswith("\n"))
        self.assertFalse(rendered.splitlines()[1].startswith("  "))
        self.assertTrue(rendered.splitlines()[2].startswith("  "))
        self.assertIn("Liquid", rendered)

    def test_user_messages_indent_content(self) -> None:
        rendered = render_user_message("hello there")

        self.assertFalse(rendered.splitlines()[1].startswith("  "))
        self.assertIn("You", rendered)
        self.assertIn("\n  hello there", rendered)

    def test_inline_system_messages_skip_block_spacing(self) -> None:
        rendered = render_system_message("Type your response:", separated=False)

        self.assertFalse(rendered.startswith("\n"))
        self.assertFalse(rendered.startswith("  "))
        self.assertIn("Type your response:", rendered)


if __name__ == "__main__":
    unittest.main()
