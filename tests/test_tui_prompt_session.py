"""Regression tests for the bottom-docked TUI application."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lqh.tui.app import LqhApp
from lqh.tui.renderer import render_options, render_system_message


@pytest.fixture
def app() -> LqhApp:
    return LqhApp(Path("."))


class TestPromptSession:
    """Verify the application preserves native terminal scroll/select behavior."""

    def test_application_disables_mouse_capture_and_alt_screen(
        self, app: LqhApp,
    ) -> None:
        application = app._create_application()
        assert not application.mouse_support()
        assert not application.full_screen
        assert application.erase_when_done

    def test_ask_user_selection_updates_managed_area(self, app: LqhApp) -> None:
        app._ask_user_options = ["one", "two"]
        app._ask_user_selected = 1
        app._set_managed_text(
            render_options(app._ask_user_options, app._ask_user_selected)
        )
        assert "two" in app._managed_ansi

    def test_layout_keeps_status_below_input(self, app: LqhApp) -> None:
        application = app._create_application()
        assert len(application.layout.container.children) == 5

    async def test_plain_ask_user_prompt_stays_in_managed_area(self, app: LqhApp) -> None:
        task = asyncio.create_task(
            app._wait_for_user_response(
                managed_text=render_system_message("Type your response:")
            )
        )

        await asyncio.sleep(0)
        assert "Type your response:" in app._managed_ansi

        future = app._ask_user_future
        assert future is not None
        app._ask_user_future = None
        future.set_result("typed response")

        result = await task
        assert result == "typed response"
        assert app._managed_ansi == ""

    async def test_wait_for_app_task_ignores_cancellation(self) -> None:
        async def never_finishes() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(never_finishes())
        task.cancel()
        await LqhApp._wait_for_app_task(task)
