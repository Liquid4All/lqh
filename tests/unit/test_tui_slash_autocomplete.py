"""Tests for the slash-command autocomplete menu in the bottom-docked TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lqh.tui.app import LqhApp
from lqh.tui.commands import COMMANDS, SlashCommandCompleter


@pytest.fixture
def app() -> LqhApp:
    return LqhApp(Path("."))


def _complete(text: str, *, enabled=None) -> list[str]:
    """Run the completer over *text* (cursor at end) and return command names."""
    completer = SlashCommandCompleter(enabled=enabled)
    doc = Document(text=text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, None)]


class TestSlashCommandCompleter:
    def test_bare_slash_offers_every_command(self) -> None:
        assert _complete("/") == [cmd.name for cmd in COMMANDS]

    def test_prefix_filters_commands(self) -> None:
        assert _complete("/he") == ["/help"]

    def test_non_slash_text_yields_nothing(self) -> None:
        assert _complete("hello") == []

    def test_space_after_command_closes_menu(self) -> None:
        """Once the user is typing arguments the menu must stay closed."""
        assert _complete("/spec ") == []
        assert _complete("/spec my task") == []

    def test_multiline_message_yields_nothing(self) -> None:
        assert _complete("/he\nsecond line") == []

    def test_no_match_yields_nothing(self) -> None:
        assert _complete("/zzz") == []

    def test_disabled_predicate_suppresses_completions(self) -> None:
        assert _complete("/he", enabled=lambda: False) == []
        assert _complete("/he", enabled=lambda: True) == ["/help"]


async def _drive_input(
    app: LqhApp,
    keystrokes: list[tuple[str, object]],
    *,
    expect_submit: bool = True,
) -> str | None:
    """Feed keystrokes through the real key processor.

    Each ``(text, callback)`` pair sends *text* to the input pipe, waits for
    the async completer to settle, then runs ``callback(app)`` for
    intermediate assertions. Returns the submitted message (or None when
    ``expect_submit`` is False).
    """
    application = app._create_application()
    app._app = application
    result: dict[str, str | None] = {"res": None}

    with create_pipe_input() as pipe:
        application.input = pipe
        application.output = DummyOutput()

        async def drive() -> None:
            try:
                await asyncio.sleep(0.1)
                for text, callback in keystrokes:
                    pipe.send_text(text)
                    # Escape needs the key-processor flush timeout to
                    # disambiguate from escape sequences.
                    await asyncio.sleep(0.8 if text == ESC else 0.1)
                    if callback is not None:
                        callback(app)
                if expect_submit:
                    result["res"] = await asyncio.wait_for(
                        app._input_queue.get(), timeout=2
                    )
            finally:
                # Exit unconditionally so a failed assertion above surfaces
                # as a test failure instead of hanging run_async() forever.
                application.exit()

        task = asyncio.create_task(drive())
        await application.run_async()
        await task

    return result["res"]


ENTER = "\r"
TAB = "\t"
UP = "\x1b[A"
DOWN = "\x1b[B"
ESC = "\x1b"


def _menu_rows(app: LqhApp) -> list[str]:
    state = app._input_buffer.complete_state
    if state is None:
        return []
    return [c.text for c in state.completions]


class TestSlashAutocompleteKeyboard:
    async def test_typing_slash_opens_menu(self, app: LqhApp) -> None:
        def _assert_open(a: LqhApp) -> None:
            assert a._completion_menu_active()
            # COMMANDS order is preserved: /hf_login is declared before /help.
            assert _menu_rows(a) == ["/hf_login", "/help"]

        await _drive_input(app, [("/h", _assert_open)], expect_submit=False)

    async def test_down_enter_runs_highlighted_command(self, app: LqhApp) -> None:
        res = await _drive_input(app, [("/he", None), (DOWN, None), (ENTER, None)])
        assert res == "/help"

    async def test_up_wraps_to_last_row(self, app: LqhApp) -> None:
        """Up from no selection highlights the bottom row (wrap-around)."""
        res = await _drive_input(app, [("/", None), (UP, None), (ENTER, None)])
        assert res == COMMANDS[-1].name

    async def test_tab_completes_without_submitting(self, app: LqhApp) -> None:
        def _assert_filled(a: LqhApp) -> None:
            assert a._input_buffer.text == "/help"

        res = await _drive_input(
            app,
            [("/he", None), (TAB, _assert_filled), (ENTER, None)],
        )
        assert res == "/help"

    async def test_enter_without_navigation_submits_typed_text(self, app: LqhApp) -> None:
        """Enter with nothing highlighted must not silently pick a command."""
        res = await _drive_input(app, [("/he", None), (ENTER, None)])
        assert res == "/he"

    async def test_typing_past_command_closes_menu(self, app: LqhApp) -> None:
        def _assert_closed(a: LqhApp) -> None:
            assert not a._completion_menu_active()

        res = await _drive_input(app, [("/spec x", _assert_closed), (ENTER, None)])
        assert res == "/spec x"

    async def test_escape_dismisses_menu(self, app: LqhApp) -> None:
        def _assert_open(a: LqhApp) -> None:
            assert a._completion_menu_active()

        def _assert_closed(a: LqhApp) -> None:
            assert not a._completion_menu_active()

        await _drive_input(
            app,
            [("/he", _assert_open), (ESC, _assert_closed)],
            expect_submit=False,
        )

    async def test_menu_render_marks_selected_row(self, app: LqhApp) -> None:
        def _assert_rendered(a: LqhApp) -> None:
            flat = [(style, text) for style, text in a._get_completion_menu_text()]
            selected = [t for s, t in flat if "selected" in s and "/help" in t]
            assert selected, "highlighted row should render with the selected style"
            assert any("Show available commands" in t for _, t in flat)

        await _drive_input(
            app,
            [("/he", None), (DOWN, _assert_rendered)],
            expect_submit=False,
        )
