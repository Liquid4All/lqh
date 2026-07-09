"""Regression tests for the bottom-docked TUI application."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lqh.tui.app import LqhApp, OTHER_OPTION, _is_other_option
from lqh.tui.renderer import render_options, render_system_message


@pytest.fixture
def app() -> LqhApp:
    return LqhApp(Path("."))


async def _drive_ask_user(
    app: LqhApp,
    options: list[str],
    keystrokes: list[tuple[str, object]],
    *,
    multi_select: bool = True,
    allow_other: bool = False,
) -> str:
    """Drive an ask_user prompt through the real prompt_toolkit key processor.

    ``keystrokes`` is a list of ``(text, callback)`` pairs: each ``text`` is fed
    to the input pipe, then ``callback(app)`` runs (use it to assert intermediate
    state). Returns the resolved ask_user response.
    """
    application = app._create_application()
    app._app = application
    result: dict[str, str] = {}

    with create_pipe_input() as pipe:
        application.input = pipe
        application.output = DummyOutput()

        async def drive() -> None:
            await asyncio.sleep(0.1)
            fut = asyncio.create_task(
                app._wait_for_user_response(
                    options=options,
                    multi_select=multi_select,
                    allow_other=allow_other,
                )
            )
            await asyncio.sleep(0.05)
            for text, callback in keystrokes:
                pipe.send_text(text)
                await asyncio.sleep(0.05)
                if callback is not None:
                    callback(app)
            result["res"] = await asyncio.wait_for(fut, timeout=2)
            application.exit()

        task = asyncio.create_task(drive())
        await application.run_async()
        await task

    return result["res"]


# Key sequences sent through the pipe input.
ENTER = "\r"
SPACE = " "
DOWN = "\x1b[B"


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

    async def test_multi_select_empty_enter_guards_before_none(self, app: LqhApp) -> None:
        """First Enter with nothing toggled warns instead of answering none."""
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        app._ask_user_future = future
        app._ask_user_options = ["alpha", "beta"]
        app._ask_user_multi_select = True
        app._ask_user_checked = set()

        # First Enter: guard fires, future stays pending, warning is shown.
        app._resolve_ask_user("")
        assert not future.done()
        assert app._ask_user_confirm_none is True
        assert "Nothing selected" in app._managed_ansi

        # Second Enter: user really meant none.
        app._resolve_ask_user("")
        assert future.result() == "(none selected)"

    async def test_keyboard_space_toggle_then_confirm(self, app: LqhApp) -> None:
        """Space actually toggles a checkbox through the real key processor."""
        res = await _drive_ask_user(
            app,
            ["alpha", "beta", "gamma"],
            [(SPACE, None), (DOWN, None), (SPACE, None), (ENTER, None)],
        )
        assert res == "alpha, beta"

    async def test_keyboard_enter_guard_then_space_recovers(self, app: LqhApp) -> None:
        """Enter with nothing toggled guards; then Space recovers and confirms."""

        def _assert_guarded(a: LqhApp) -> None:
            assert a._ask_user_confirm_none is True
            assert "Nothing selected" in a._managed_ansi

        res = await _drive_ask_user(
            app,
            ["alpha", "beta"],
            [(ENTER, _assert_guarded), (SPACE, None), (ENTER, None)],
        )
        assert res == "alpha"

    async def test_keyboard_double_enter_confirms_none(self, app: LqhApp) -> None:
        """Two Enters with nothing toggled answer with none selected."""
        res = await _drive_ask_user(
            app,
            ["alpha", "beta"],
            [(ENTER, None), (ENTER, None)],
        )
        assert res == "(none selected)"

    async def test_keyboard_navigation_clears_guard_warning(self, app: LqhApp) -> None:
        """Navigating off the guarded row cancels the pending confirm-none."""

        def _assert_guarded(a: LqhApp) -> None:
            assert a._ask_user_confirm_none is True

        def _assert_cleared(a: LqhApp) -> None:
            assert a._ask_user_confirm_none is False
            assert "Nothing selected" not in a._managed_ansi

        # Enter guards; Down clears the warning; Space toggles beta; Enter answers.
        res = await _drive_ask_user(
            app,
            ["alpha", "beta"],
            [
                (ENTER, _assert_guarded),
                (DOWN, _assert_cleared),
                (SPACE, None),
                (ENTER, None),
            ],
        )
        assert res == "beta"

    async def test_keyboard_warning_never_shows_on_other_row(self, app: LqhApp) -> None:
        """With the Other option present, Enter on a real row guards on that row,
        and Enter on the Other row opens free-text instead of the none-guard."""
        # Highlight is on the first real row when the guard fires, so the banner's
        # "toggle the highlighted option" always refers to a toggleable row.
        res = await _drive_ask_user(
            app,
            ["alpha", "beta", OTHER_OPTION],
            [(ENTER, None), (SPACE, None), (ENTER, None)],
            allow_other=True,
        )
        assert res == "alpha"

    @pytest.mark.parametrize(
        "model_option",
        [
            "Other",
            "Other (please specify)",
            "Other (please enter)",
            "  other  ",
            OTHER_OPTION,
        ],
    )
    async def test_model_produced_other_is_not_duplicated(
        self, app: LqhApp, model_option: str,
    ) -> None:
        """A model that includes its own 'Other' row must not yield two of them.

        The TUI appends exactly one OTHER_OPTION; the model's variant is filtered.
        """
        captured: dict[str, list[str]] = {}

        async def fake_wait(*, options, allow_other, multi_select, relock_after):
            captured["options"] = options
            return options[0]

        app._wait_for_user_response = fake_wait  # type: ignore[assignment]
        await app._on_ask_user("Pick one", ["alpha", model_option], allow_other=True)

        rendered = captured["options"]
        assert rendered.count(OTHER_OPTION) == 1
        assert sum(_is_other_option(o) for o in rendered) == 1
        assert rendered == ["alpha", OTHER_OPTION]

    def test_single_select_render_shows_navigation_hint(self, app: LqhApp) -> None:
        """Single-select mode must spell out how to answer (it had no hint before)."""
        out = render_options(["alpha", "beta"], 0, allow_other=True)
        assert "Enter: select" in out
        assert "type" in out.lower()  # invites a custom free-text answer

    def test_single_select_hint_omits_custom_answer_without_other(
        self, app: LqhApp,
    ) -> None:
        out = render_options(["alpha", "beta"], 0, allow_other=False)
        assert "Enter: select" in out
        assert "Other" not in out

    async def test_wait_for_app_task_ignores_cancellation(self) -> None:
        async def never_finishes() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(never_finishes())
        task.cancel()
        await LqhApp._wait_for_app_task(task)
