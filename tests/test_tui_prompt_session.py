"""Regression tests for the bottom-docked TUI application."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from lqh.tui.app import LqhApp
from lqh.tui.renderer import render_options, render_system_message


class PromptSessionTests(unittest.TestCase):
    """Verify the application preserves native terminal scroll/select behavior."""

    def test_application_disables_mouse_capture_and_alt_screen(self) -> None:
        app = LqhApp(Path("."))
        application = app._create_application()

        self.assertFalse(application.mouse_support())
        self.assertFalse(application.full_screen)
        self.assertTrue(application.erase_when_done)

    def test_ask_user_selection_updates_managed_area(self) -> None:
        app = LqhApp(Path("."))
        app._ask_user_options = ["one", "two"]
        app._ask_user_selected = 1
        app._set_managed_text(render_options(app._ask_user_options, app._ask_user_selected))

        self.assertIn("two", app._managed_ansi)

    def test_layout_keeps_status_below_input(self) -> None:
        app = LqhApp(Path("."))
        application = app._create_application()

        children = application.layout.container.children
        self.assertEqual(len(children), 5)

    def test_plain_ask_user_prompt_stays_in_managed_area(self) -> None:
        async def run_test() -> None:
            app = LqhApp(Path("."))
            task = asyncio.create_task(
                app._wait_for_user_response(
                    managed_text=render_system_message("Type your response:")
                )
            )

            await asyncio.sleep(0)
            self.assertIn("Type your response:", app._managed_ansi)

            future = app._ask_user_future
            self.assertIsNotNone(future)
            app._ask_user_future = None
            future.set_result("typed response")

            result = await task
            self.assertEqual(result, "typed response")
            self.assertEqual(app._managed_ansi, "")

        asyncio.run(run_test())

    def test_wait_for_app_task_ignores_cancellation(self) -> None:
        async def run_test() -> None:
            async def never_finishes() -> None:
                await asyncio.sleep(60)

            task = asyncio.create_task(never_finishes())
            task.cancel()
            await LqhApp._wait_for_app_task(task)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
