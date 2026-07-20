"""Bottom-docked terminal UI for lqh."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from prompt_toolkit import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style

from lqh.agent import Agent, AgentCallbacks
from lqh.auth import LoginExpired, get_token, login_device_code
from lqh.project_identity import cloud_project_key as _ckey
from lqh.session import Session
from lqh.tui.commands import COMMANDS, SlashCommandCompleter, is_command, parse_command
from lqh.tui.dataset_viewer import DatasetViewer
from lqh.tui.renderer import (
    render_agent_message,
    render_error,
    render_file_view,
    render_options,
    render_secret,
    render_system_message,
    render_tool_call,
    render_tool_result,
    render_user_message,
    render_welcome,
)
from lqh.tui.background_tasks import BackgroundTask, BackgroundTaskRegistry
from lqh.tui.status_bar import StatusBar
from lqh.update_check import check_for_update
from lqh.telemetry import TelemetryClient, notice_needed, set_active_telemetry

if TYPE_CHECKING:
    from lqh.subprocess_manager import SubprocessManager

logger = logging.getLogger(__name__)


TUI_STYLE = Style.from_dict({
    "status": "bg:#1a1a2e #e0e0e0",
    "status.spinner": "bg:#1a1a2e #00ff88 bold",
    "status.separator": "bg:#1a1a2e #555555",
    "status.warning": "bg:#1a1a2e #ff4444 bold",
    "status.caution": "bg:#1a1a2e #ffaa00",
    "status.dim": "bg:#1a1a2e #666666",
    "input-border": "#444444",
    "input-prompt": "bold #888888",
    "input-area": "bg:#16202a #f5f7fa",
    # NB: deliberately NOT named "completion-menu" — that class exists in
    # prompt_toolkit's default UI style (bg:#bbbbbb) and would bleed a light
    # background through any partial override.
    "slash-menu": "#c0c8d0",
    "slash-menu.selected": "bg:#16202a #00ff88 bold",
    "slash-menu.meta": "#777777",
    "slash-menu.meta.selected": "bg:#16202a #aaaaaa",
    "slash-menu.hint": "#555555 italic",
})

OTHER_OPTION = "Other (type your own answer)"


def _is_other_option(option: str) -> bool:
    """True if *option* looks like a catch-all "Other" choice.

    The TUI always appends its own ``OTHER_OPTION``, so any model-produced
    variant ("Other", "Other (please specify)", "Other (please enter)", …)
    must be stripped first — otherwise the list shows two "Other" rows.
    Matching on the ``other`` prefix is deliberately liberal.
    """
    return option.strip().lower().startswith("other")

# Maximum rows the input area grows to before it scrolls internally.
INPUT_MAX_LINES = 8

# A second Ctrl+C within this window exits the app. Outside it, Ctrl+C is a
# fresh press: interrupt the running agent turn, or clear the typed input.
CTRL_C_EXIT_WINDOW_SEC = 2.0

# Sentinel pushed into the input queue on shutdown so any consumer parked on
# the queue (the auto-mode pause prompt, the main input loop) wakes up and
# unwinds instead of waiting for input that will never come.
_SHUTDOWN_SENTINEL = "\x00__lqh_shutdown__"


class AgentInterrupted(Exception):
    """The user cancelled the in-flight agent turn (Esc / Ctrl+C)."""

# Job-supervision cadence/grace constants live with the extracted
# supervisor (lqh/jobs.py); re-exported here for existing importers.
from lqh.jobs import (  # noqa: E402
    JOB_POLL_INTERVAL_SEC,
    SCORING_GRACE_SEC,
    SLEEP_GAP_FACTOR,
    JobSupervisor,
    SupervisorHooks,
)

TELEMETRY_FLUSH_INTERVAL_SEC = 60.0
TELEMETRY_HEARTBEAT_INTERVAL_SEC = 300.0
RECONNECT_BACKOFF_SEC = (3.0, 20.0, 60.0)


class LqhApp:
    """Persistent bottom-bar application that prints output into scrollback."""

    def __init__(
        self,
        project_dir: Path,
        *,
        auto_mode: bool = False,
        extra_spec: str | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.auto_mode = auto_mode
        self.extra_spec = extra_spec
        self._status_bar = StatusBar(project_dir=project_dir)
        self._status_bar.auto_mode = auto_mode
        self._foreground_progress_history: list[Any] = []
        # Auto-mode progress state, rendered into the managed area.
        self._auto_stage: str | None = None
        self._auto_stage_note: str | None = None
        self._auto_history: list[str] = []
        self._auto_done: bool = False
        self._processing = False
        self._ask_user_future: asyncio.Future[str] | None = None
        self._ask_user_options: list[str] | None = None
        self._ask_user_allow_other = False
        self._ask_user_selected = 0
        self._ask_user_multi_select = False
        self._ask_user_checked: set[int] = set()
        self._ask_user_confirm_none = False
        self._dataset_viewer: DatasetViewer | None = None
        self._dataset_viewer_future: asyncio.Future[str] | None = None
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._session: Session | None = None
        self._agent: Agent | None = None
        self._spinner_task: asyncio.Task | None = None
        self._app: Application | None = None
        # Background-job supervision is owned by the shared headless
        # JobSupervisor (lqh/jobs.py); the hooks route UI + telemetry side
        # effects back here. The aliases below keep the rest of the TUI
        # (status bar, progress refresh, shutdown) reading/writing the
        # SAME underlying state, not copies.
        self._job_watcher_task: asyncio.Task | None = None
        self._progress_refresh_task: asyncio.Task | None = None
        self._update_check_task: asyncio.Task[None] | None = None
        self._supervisor = JobSupervisor(
            project_dir,
            hooks=SupervisorHooks(
                on_registry_change=self._invalidate,
                on_gap=self._on_watch_gap,
                on_notice=self._on_job_notice,
                on_running=self._on_job_running,
                on_terminal=self._on_job_terminal,
                has_job_record=self._has_telemetry_job_record,
                on_record_completion=self._record_telemetry_completion,
                on_data_gen_terminal=self._on_data_gen_terminal,
            ),
        )
        self._job_last_state = self._supervisor.job_last_state
        self._pending_completions = self._supervisor.pending_completions
        self._data_gen_gave_up = self._supervisor.data_gen_gave_up
        self._run_watchers = self._supervisor.run_watchers
        self._tasks = self._supervisor.tasks
        # Live progress tracking for the status bar: the last step seen per run
        # and the wall time it last advanced (drives the "↑8s ago" freshness).
        self._job_last_step: dict[str, tuple[str, float]] = {}
        self._job_progress_at: dict[str, float] = {}
        self._ctrl_c_pressed = False
        self._ctrl_c_at = 0.0
        # The in-flight agent turn, wrapped in a task so Esc / Ctrl+C can
        # cancel it without waiting for the current LLM call to finish.
        self._agent_task: asyncio.Task | None = None
        self._interrupt_requested = False
        # Auto mode: True while the run is paused waiting for a user
        # instruction (single Ctrl+C). Unhides the input row.
        self._auto_paused = False
        self._input_buffer: Buffer | None = None
        self._managed_ansi = ""
        self._shutdown_requested = False
        self._telemetry = TelemetryClient(project_dir, auto_mode=auto_mode)
        self._telemetry_heartbeat_task: asyncio.Task[None] | None = None
        self._telemetry_flush_tasks: set[asyncio.Task[None]] = set()
        # workflow id, monotonic start, wall start, prior active seconds,
        # current-session active baseline, kind, target
        self._telemetry_jobs: dict[str, tuple[str, float | None, float, float, float, str, str, int]] = {}
        self._pending_reconnect: Callable[[], Awaitable[None]] | None = None
        self._pending_reconnect_error: str | None = None
        self._reconnect_backoffs = RECONNECT_BACKOFF_SEC

    def _create_layout(self) -> Layout:
        """Create a small bottom-docked layout."""
        # Interactive tools render here; regular chat output is printed into scrollback.
        # In auto mode, the managed window is always visible (it shows the progress panel).
        managed_window = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(self._get_managed_text),
                wrap_lines=True,
                dont_extend_height=True,
            ),
            filter=Condition(lambda: self.auto_mode or bool(self._managed_ansi)),
        )

        input_pad_top = Window(
            height=1,
            content=FormattedTextControl(
                lambda: FormattedText([("", "")])
            ),
        )

        self._input_buffer = Buffer(
            name="input",
            completer=SlashCommandCompleter(
                enabled=lambda: (
                    self._ask_user_future is None and self._dataset_viewer is None
                ),
            ),
            complete_while_typing=True,
            multiline=False,
            accept_handler=self._on_accept,
        )
        input_window = Window(
            content=BufferControl(
                buffer=self._input_buffer,
                focusable=True,
            ),
            # No explicit `preferred`: the window then sizes itself to the
            # buffer's line count (wrap-aware), clamped to INPUT_MAX_LINES.
            height=Dimension(min=1, max=INPUT_MAX_LINES),
            wrap_lines=True,
            style="class:input-area",
        )

        input_label = Window(
            content=FormattedTextControl(self._get_prompt_label),
            width=5,
            height=1,
        )

        input_row = VSplit([input_label, input_window])

        input_pad_bottom = Window(
            height=1,
            content=FormattedTextControl(
                lambda: FormattedText([("", "")])
            ),
        )

        completion_menu = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(self._get_completion_menu_text),
                dont_extend_height=True,
            ),
            filter=Condition(self._completion_menu_active),
        )

        # In auto mode the input field is hidden — the agent runs without
        # user input and the managed window is the only visible surface
        # besides the status bar. Exception: while the run is paused by a
        # user interrupt, the input row appears to collect the instruction.
        not_auto = Condition(lambda: not self.auto_mode or self._auto_paused)
        hidden_input_pad_top = ConditionalContainer(content=input_pad_top, filter=not_auto)
        hidden_input_row = ConditionalContainer(content=input_row, filter=not_auto)
        hidden_input_pad_bottom = ConditionalContainer(content=input_pad_bottom, filter=not_auto)

        status_window = Window(
            height=1,
            content=FormattedTextControl(self._get_status_text),
        )

        return Layout(
            HSplit([
                managed_window,
                hidden_input_pad_top,
                hidden_input_row,
                completion_menu,
                hidden_input_pad_bottom,
                status_window,
            ]),
            focused_element=input_window,
        )

    def _create_keybindings(self) -> KeyBindings:
        """Create key bindings for the persistent application."""
        kb = KeyBindings()

        is_ask_mode = Condition(
            lambda: self._ask_user_future is not None and self._ask_user_options is not None
        )
        is_dataset_mode = Condition(lambda: self._dataset_viewer is not None)

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _newline(event):
            """Insert newline on Alt+Enter (or Ctrl+J where Alt+Enter is swallowed)."""
            event.app.current_buffer.newline()

        @kb.add("c-c", eager=True)
        def _interrupt(event):
            """Ctrl+C once interrupts the agent (or warns); twice in a row exits."""
            now = time.monotonic()
            if self._ctrl_c_pressed and now - self._ctrl_c_at <= CTRL_C_EXIT_WINDOW_SEC:
                self._save_session()
                self._request_shutdown()
                return

            self._ctrl_c_pressed = True
            self._ctrl_c_at = now

            # Agent busy: a single Ctrl+C cancels the in-flight turn and
            # returns control to the user (feedback is emitted by the
            # interrupt path). The typed buffer is preserved.
            if self._request_agent_interrupt():
                return

            event.app.current_buffer.reset()
            asyncio.get_event_loop().create_task(
                self._emit(render_system_message("Press Ctrl+C again to exit, or continue typing."))
            )

        @kb.add("c-d", eager=True)
        def _exit(event):
            """Ctrl+D exits."""
            self._save_session()
            self._request_shutdown()

        @kb.add("up", filter=is_ask_mode, eager=True)
        def _ask_up(event):
            if self._ask_user_options:
                self._ask_user_selected = max(0, self._ask_user_selected - 1)
                # Moving off the guarded row cancels the pending "confirm none".
                self._ask_user_confirm_none = False
                self._render_ask_user_options()
                event.app.invalidate()

        @kb.add("down", filter=is_ask_mode, eager=True)
        def _ask_down(event):
            if self._ask_user_options:
                self._ask_user_selected = min(
                    len(self._ask_user_options) - 1,
                    self._ask_user_selected + 1,
                )
                self._ask_user_confirm_none = False
                self._render_ask_user_options()
                event.app.invalidate()

        # Slash-command autocomplete. The completer only fires on a
        # single-line "/word" prefix (and never in ask/dataset mode), so
        # these bindings cannot collide with the ask-mode arrows above.
        has_completion_menu = Condition(self._completion_menu_active)
        completion_selected = Condition(
            lambda: bool(
                self._input_buffer
                and self._input_buffer.complete_state
                and self._input_buffer.complete_state.current_completion
            )
        )

        @kb.add("up", filter=has_completion_menu, eager=True)
        def _completion_up(event):
            event.app.current_buffer.complete_previous()

        @kb.add("down", filter=has_completion_menu, eager=True)
        def _completion_down(event):
            event.app.current_buffer.complete_next()

        @kb.add("c-i", filter=has_completion_menu, eager=True)  # Tab
        def _completion_tab(event):
            """Tab fills the buffer with the highlighted (or first) command."""
            buff = event.app.current_buffer
            state = buff.complete_state
            completion = state.current_completion or state.completions[0]
            buff.apply_completion(completion)

        @kb.add("enter", filter=completion_selected, eager=True)
        def _completion_enter(event):
            """Enter on a highlighted row runs that command immediately."""
            buff = event.app.current_buffer
            buff.apply_completion(buff.complete_state.current_completion)
            buff.validate_and_handle()

        @kb.add("escape", filter=has_completion_menu, eager=True)
        def _completion_escape(event):
            event.app.current_buffer.cancel_completion()

        # Esc interrupts the in-flight agent turn, like a single Ctrl+C.
        # Excluded while another overlay owns Esc (completion menu, dataset
        # viewer). `eager` means Esc no longer waits to disambiguate from
        # Alt+Enter while the agent is busy — acceptable, since composing a
        # multiline message mid-turn is far rarer than wanting to interrupt.
        is_agent_busy = Condition(self._agent_busy)

        @kb.add(
            "escape",
            filter=is_agent_busy & ~has_completion_menu & ~is_dataset_mode,
            eager=True,
        )
        def _esc_interrupt(event):
            self._request_agent_interrupt()

        is_multi_select = Condition(
            lambda: self._ask_user_multi_select and self._ask_user_options is not None
        )

        @kb.add(" ", filter=is_multi_select, eager=True)
        def _ask_toggle(event):
            """Space toggles the current option in multi-select mode."""
            if not self._ask_user_options:
                return
            idx = self._ask_user_selected
            # "Other" option cannot be toggled via checkbox
            if self._ask_user_allow_other and self._ask_user_options[idx] == OTHER_OPTION:
                return
            if idx in self._ask_user_checked:
                self._ask_user_checked.discard(idx)
            else:
                self._ask_user_checked.add(idx)
            # Any toggle clears a pending "confirm none" warning.
            self._ask_user_confirm_none = False
            self._render_ask_user_options()
            event.app.invalidate()

        @kb.add("n", filter=is_dataset_mode, eager=True)
        def _dataset_next(event):
            if self._dataset_viewer:
                self._dataset_viewer.go_next()
                asyncio.get_event_loop().create_task(self._render_dataset_viewer())

        @kb.add("p", filter=is_dataset_mode, eager=True)
        def _dataset_prev(event):
            if self._dataset_viewer:
                self._dataset_viewer.go_prev()
                asyncio.get_event_loop().create_task(self._render_dataset_viewer())

        @kb.add("r", filter=is_dataset_mode, eager=True)
        def _dataset_random(event):
            if self._dataset_viewer:
                self._dataset_viewer.go_random()
                asyncio.get_event_loop().create_task(self._render_dataset_viewer())

        @kb.add("q", filter=is_dataset_mode, eager=True)
        def _dataset_close(event):
            asyncio.get_event_loop().create_task(self._close_dataset_viewer())

        @kb.add("escape", filter=is_dataset_mode, eager=True)
        def _dataset_escape(event):
            asyncio.get_event_loop().create_task(self._close_dataset_viewer())

        return kb

    def _create_application(self) -> Application:
        """Create the persistent bottom application."""
        return Application(
            layout=self._create_layout(),
            key_bindings=self._create_keybindings(),
            style=TUI_STYLE,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    def _get_prompt_label(self) -> FormattedText:
        """Return the current prompt label."""
        if self._dataset_viewer is not None:
            label = " ds> "
        elif self._ask_user_future is not None:
            label = " ? "
        else:
            label = " > "
        return FormattedText([("class:input-prompt", label)])

    def _completion_menu_active(self) -> bool:
        """True while the slash-command menu has rows to show."""
        buf = self._input_buffer
        return bool(buf and buf.complete_state and buf.complete_state.completions)

    def _get_completion_menu_text(self) -> FormattedText:
        """Render the slash-command menu below the input row."""
        buf = self._input_buffer
        state = buf.complete_state if buf else None
        if not state or not state.completions:
            return FormattedText([])

        width = max(len(c.text) for c in state.completions)
        fragments: list[tuple[str, str]] = []
        for i, completion in enumerate(state.completions):
            selected = i == state.complete_index
            name_style = "class:slash-menu.selected" if selected else "class:slash-menu"
            meta_style = (
                "class:slash-menu.meta.selected" if selected
                else "class:slash-menu.meta"
            )
            marker = "❯ " if selected else "  "
            fragments.append((name_style, f"   {marker}{completion.text:<{width}}"))
            fragments.append((meta_style, f"  {completion.display_meta_text}"))
            fragments.append(("", "\n"))
        fragments.append(
            ("class:slash-menu.hint",
             "   ↑/↓ choose · Enter run · Tab complete · Esc dismiss")
        )
        return FormattedText(fragments)

    def _get_status_text(self) -> FormattedText:
        """Render the status bar, with mode hints when applicable."""
        self._status_bar.bg_tasks = self._tasks.snapshot()
        parts = list(self._status_bar.get_formatted_text())

        composing_multiline = (
            self._ask_user_options is None
            and self._dataset_viewer is None
            and self._input_buffer is not None
            and "\n" in self._input_buffer.text
        )
        if composing_multiline:
            parts.extend([
                ("class:status.separator", " │ "),
                ("class:status", "Enter send"),
                ("class:status.separator", " │ "),
                ("class:status", "Alt+Enter newline"),
            ])

        if self._ask_user_options:
            if self._ask_user_multi_select:
                parts.extend([
                    ("class:status.separator", " │ "),
                    ("class:status.spinner", "↑/↓ navigate"),
                    ("class:status.separator", " │ "),
                    ("class:status", "Space toggle"),
                    ("class:status.separator", " │ "),
                    ("class:status", "Enter confirm"),
                ])
            else:
                parts.extend([
                    ("class:status.separator", " │ "),
                    ("class:status.spinner", "↑/↓ navigate"),
                    ("class:status.separator", " │ "),
                    ("class:status", "Enter select"),
                ])

        if self._dataset_viewer is not None:
            parts.extend([
                ("class:status.separator", " │ "),
                ("class:status.spinner", "n/p/r/q dataset"),
            ])

        if self._auto_paused:
            parts.extend([
                ("class:status.separator", " │ "),
                ("class:status.spinner", "⏸ paused"),
                ("class:status.separator", " │ "),
                ("class:status", "Enter continue"),
            ])
        elif (
            self._agent_busy()
            and self._ask_user_future is None
            and self._dataset_viewer is None
        ):
            hint = "Ctrl+C interrupt" if self.auto_mode else "Esc interrupt"
            parts.extend([
                ("class:status.separator", " │ "),
                ("class:status", hint),
            ])

        return FormattedText(parts)

    def _get_managed_text(self) -> ANSI:
        """Return the currently managed interactive area."""
        if self.auto_mode and not self._managed_ansi:
            return ANSI(self._render_auto_progress())
        return ANSI(self._managed_ansi)

    def _render_auto_progress(self) -> str:
        """Render the auto-mode progress panel."""
        from lqh.tui.renderer import render_auto_progress

        return render_auto_progress(
            stage=self._auto_stage,
            note=self._auto_stage_note,
            history=self._auto_history,
            done=self._auto_done,
        )

    def _set_managed_text(self, ansi_text: str = "") -> None:
        """Update the managed interactive area."""
        self._managed_ansi = ansi_text
        self._invalidate()

    def _render_ask_user_options(self) -> None:
        """Refresh the selectable ask-user list inside the managed area."""
        if self._ask_user_options is None:
            self._set_managed_text("")
            return

        self._set_managed_text(
            render_options(
                self._ask_user_options,
                self._ask_user_selected,
                checked=self._ask_user_checked if self._ask_user_multi_select else None,
                warn_empty=self._ask_user_confirm_none,
                allow_other=self._ask_user_allow_other,
            )
        )

    def _dataset_view_text(self, prefix: str = "") -> str:
        """Build the managed dataset viewer content."""
        if self._dataset_viewer is None:
            return prefix

        return (
            prefix
            + self._dataset_viewer.render_sample()
            + self._dataset_viewer.render_nav_bar()
        )

    def _lock_input(self) -> None:
        """Prevent regular input submission while the agent is busy."""
        self._processing = True
        self._invalidate()

    def _unlock_input(self) -> None:
        """Allow user input again."""
        self._processing = False
        self._invalidate()

    def _invalidate(self) -> None:
        """Refresh the live bottom application."""
        if self._app:
            self._app.invalidate()

    def _exit_application(self) -> None:
        """Exit the prompt_toolkit app only if it is still actively running."""
        # Drop the advisory agent-loop marker (owned-by-us only, so an
        # accidental double call or a foreign owner is never affected).
        from lqh.headless import release_loop

        release_loop(self.project_dir)
        if self._app and self._app.is_running:
            self._app.exit()

    def _request_shutdown(self) -> None:
        """Mark the session for shutdown and stop the live application."""
        self._shutdown_requested = True
        # Cancel any in-flight agent turn so callers awaiting it unwind
        # instead of blocking shutdown until the turn completes.
        self._request_agent_interrupt()
        # Wake any consumer parked on the input queue (auto-mode pause,
        # main loop) so it sees the shutdown instead of waiting forever.
        self._input_queue.put_nowait(_SHUTDOWN_SENTINEL)
        self._exit_application()

    def _agent_busy(self) -> bool:
        """True while an agent turn is in flight (and thus interruptible)."""
        return self._agent_task is not None and not self._agent_task.done()

    def _request_agent_interrupt(self) -> bool:
        """Cancel the in-flight agent turn. Returns True if one was running."""
        task = self._agent_task
        if task is None or task.done():
            return False
        self._interrupt_requested = True
        task.cancel()
        return True

    async def _run_interruptible(self, action: Callable[[], Awaitable[Any]]) -> Any:
        """Run an agent action as a task so Esc / Ctrl+C can cancel it mid-flight.

        Cancellation does NOT wait for the current LLM call or tool to finish.
        On a user interrupt the session is repaired (unanswered tool calls get
        synthetic results so the API accepts the history), transient status UI
        is cleared, and ``AgentInterrupted`` is raised for the caller to
        resume its own flow.
        """
        task = asyncio.create_task(action())
        self._agent_task = task
        try:
            return await task
        except asyncio.CancelledError:
            if not self._interrupt_requested:
                # We were cancelled from outside — propagate, but don't leave
                # the agent turn running detached.
                task.cancel()
                raise
            self._interrupt_requested = False
            if self._agent:
                self._agent.abort_turn()
            # The cancel can land while the spinner / pipeline status is live.
            self._on_spinner_stop()
            self._on_pipeline_done()
            self._save_session()
            if not self._shutdown_requested:
                await self._emit(render_system_message("⏹ Interrupted."))
            raise AgentInterrupted() from None
        finally:
            self._agent_task = None

    def _start_application_task(self) -> asyncio.Task:
        """Create and start a fresh bottom-docked application instance."""
        self._app = self._create_application()
        return asyncio.create_task(self._app.run_async())

    @staticmethod
    async def _wait_for_app_task(app_task: asyncio.Task) -> None:
        """Treat prompt_toolkit shutdown cancellation as a normal exit path."""
        try:
            await app_task
        except asyncio.CancelledError:
            return

    def _on_accept(self, buff: Buffer) -> bool:
        """Handle Enter in the bottom input area."""
        self._ctrl_c_pressed = False
        text = buff.text.strip()

        if self._dataset_viewer is not None:
            buff.reset()
            asyncio.get_event_loop().create_task(self._handle_dataset_input(text))
            return False

        if self._processing and self._ask_user_future is None:
            asyncio.get_event_loop().create_task(
                self._emit(render_system_message(
                    "⏳ Please wait for the current operation to finish "
                    "(or press Esc / Ctrl+C to interrupt it)."
                ))
            )
            return False

        buff.reset()

        if self._ask_user_future is not None:
            # Slash commands are dispatched only on the main input path
            # (_handle_input). While an ask_user prompt is active, a leading
            # "/" would otherwise be swallowed as the literal answer, so
            # intercept it here and tell the user to answer the question first.
            if is_command(text):
                asyncio.get_event_loop().create_task(
                    self._emit(render_system_message(
                        "Commands aren't available while a question is pending. "
                        "Please answer the question above first."
                    ))
                )
                return False
            self._resolve_ask_user(text)
            return False

        if text:
            self._input_queue.put_nowait(text)

        return False

    def _resolve_ask_user(self, text: str) -> None:
        """Resolve an active ask-user request from buffer input."""
        if self._ask_user_future is None:
            return

        if self._ask_user_options and not text:
            if self._ask_user_multi_select:
                # Multi-select: check if "Other" is the focused item and selected
                selected_opt = self._ask_user_options[self._ask_user_selected]
                if self._ask_user_allow_other and selected_opt == OTHER_OPTION:
                    # Switch to free-text mode for additional items
                    self._ask_user_options = None
                    self._ask_user_allow_other = False
                    self._ask_user_multi_select = False
                    # Preserve checked items so far as prefix
                    checked_names = [
                        self._ask_user_options_snapshot[i]
                        for i in sorted(self._ask_user_checked)
                        if i < len(self._ask_user_options_snapshot)
                    ] if hasattr(self, "_ask_user_options_snapshot") else []
                    self._ask_user_checked_prefix = checked_names
                    self._set_managed_text(
                        render_system_message(
                            "✎ Type additional items (comma-separated), then press Enter:",
                            separated=False,
                        )
                    )
                    self._invalidate()
                    return

                # Collect all checked options
                checked_names = [
                    self._ask_user_options[i]
                    for i in sorted(self._ask_user_checked)
                    if i < len(self._ask_user_options) and self._ask_user_options[i] != OTHER_OPTION
                ]
                if not checked_names and not self._ask_user_confirm_none:
                    # First Enter with nothing toggled: users routinely expect the
                    # highlighted row to count. Guard once with a prominent hint
                    # instead of silently answering "(none selected)".
                    self._ask_user_confirm_none = True
                    self._render_ask_user_options()
                    self._invalidate()
                    return
                response = ", ".join(checked_names) if checked_names else "(none selected)"
            else:
                # Single-select
                selected = self._ask_user_options[self._ask_user_selected]
                if self._ask_user_allow_other and selected == OTHER_OPTION:
                    self._ask_user_options = None
                    self._ask_user_allow_other = False
                    self._set_managed_text(
                        render_system_message(
                            "✎ Type your own answer, then press Enter:",
                            separated=False,
                        )
                    )
                    self._invalidate()
                    return
                response = selected
        else:
            # Free-text response — in multi-select "Other" mode, prepend checked items
            prefix_items = getattr(self, "_ask_user_checked_prefix", [])
            if prefix_items and text:
                all_items = prefix_items + [t.strip() for t in text.split(",") if t.strip()]
                response = ", ".join(all_items)
            elif prefix_items:
                response = ", ".join(prefix_items)
            else:
                response = text
            self._ask_user_checked_prefix = []

        future = self._ask_user_future
        self._ask_user_future = None
        self._ask_user_options = None
        self._ask_user_allow_other = False
        self._ask_user_selected = 0
        self._ask_user_multi_select = False
        self._ask_user_checked = set()
        self._ask_user_confirm_none = False
        self._set_managed_text("")
        self._invalidate()

        asyncio.get_event_loop().create_task(self._emit(render_user_message(response)))

        if not future.done():
            future.set_result(response)

    async def _wait_for_user_response(
        self,
        *,
        options: list[str] | None = None,
        allow_other: bool = False,
        multi_select: bool = False,
        managed_text: str | None = None,
        relock_after: bool = False,
    ) -> str:
        """Wait for input through the persistent bottom prompt."""
        # Interactive tools borrow the managed pane without taking over terminal scrollback.
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._ask_user_future = future
        self._ask_user_options = options
        self._ask_user_allow_other = allow_other
        self._ask_user_multi_select = multi_select
        self._ask_user_checked = set()
        self._ask_user_confirm_none = False
        self._ask_user_selected = 0
        # Snapshot options for "Other" flow in multi-select
        if multi_select and options:
            self._ask_user_options_snapshot = list(options)
        if options:
            self._render_ask_user_options()
        elif managed_text is not None:
            self._set_managed_text(managed_text)
        else:
            self._set_managed_text("")
        self._invalidate()

        if relock_after:
            self._unlock_input()

        try:
            return await future
        finally:
            if relock_after:
                self._lock_input()
            if self._ask_user_future is future:
                # The wait was cancelled (user interrupt) before the prompt
                # was answered — clear the ask state so the input row returns
                # to normal instead of swallowing the next message as an answer.
                self._ask_user_future = None
                self._ask_user_options = None
                self._ask_user_allow_other = False
                self._ask_user_multi_select = False
                self._ask_user_checked = set()
                self._ask_user_confirm_none = False
                self._ask_user_selected = 0
            if self._ask_user_future is None:
                self._set_managed_text("")

    async def _emit(self, ansi_text: str) -> None:
        """Print ANSI output above the live application."""
        if self._app and not self._app.is_done:
            # prompt_toolkit temporarily removes the bottom app so this lands in real scrollback.
            await run_in_terminal(lambda: self._write_output(ansi_text))
        else:
            self._write_output(ansi_text)

    @staticmethod
    def _write_output(ansi_text: str) -> None:
        """Write ANSI text directly to stdout."""
        sys.stdout.write(ansi_text)
        sys.stdout.flush()

    async def _handle_input(self, text: str) -> bool:
        """Handle one line of user input. Returns False to exit."""
        if is_command(text):
            return await self._handle_command(text)

        await self._handle_message(text)
        return True

    async def _handle_command(self, text: str) -> bool:
        """Handle a slash command."""
        command, _args = parse_command(text)

        tracked_command = command.removeprefix("/")
        if tracked_command in {"spec", "datagen", "validate", "train", "eval", "prompt", "clear", "resume", "feedback"}:
            # File locks may contend with another CLI in the same project;
            # keep that synchronous disk work off the UI event loop.
            await self._telemetry.run_deferred(self._telemetry.record_workflow_command, tracked_command)

        if command == "/telemetry":
            from lqh.config import load_config, telemetry_enabled, update_config
            action = _args.strip().lower() or "status"
            if action not in {"on", "off", "status"}:
                await self._emit(render_error("Usage: /telemetry on|off|status"))
                return True
            if action != "status":
                def update_telemetry_consent(config):
                    config.telemetry_enabled = action == "on"
                    config.telemetry_consent_epoch += 1

                config = update_config(update_telemetry_consent)
                # Opt-out is a hard boundary and may wait for an in-flight
                # sender's cross-process lock. Keep the UI responsive while
                # that privacy barrier completes.
                await self._telemetry.run_deferred(
                    self._telemetry.set_enabled,
                    telemetry_enabled(config),
                    timeout=None if action == "off" else 0.25,
                )
                if action == "off":
                    self._clear_persisted_telemetry_jobs()
            else:
                config = load_config()
            env_override = os.environ.get("LQH_TELEMETRY")
            suffix = " (overridden by LQH_TELEMETRY)" if env_override is not None else ""
            enabled = await self._telemetry.run_deferred(self._telemetry.is_enabled)
            if enabled is None:
                enabled = self._telemetry.state_snapshot()[0]
            state = "on" if enabled else "off"
            await self._emit(render_system_message(f"Telemetry is {state}{suffix}."))
            return True

        if command == "/quit":
            self._save_session()
            self._request_shutdown()
            return False

        if command == "/help":
            lines = ["**Available Commands:**\n"]
            for cmd in COMMANDS:
                lines.append(f"  `{cmd.name}` - {cmd.description}")
            lines.append(
                "\nTip: Alt+Enter (or Ctrl+J) inserts a newline; Enter sends."
            )
            lines.append(
                "Tip: Esc or Ctrl+C interrupts the agent while it's working; "
                "Ctrl+C twice exits."
            )
            await self._emit(render_system_message("\n".join(lines)))
            return True

        if command == "/reconnect":
            await self._do_reconnect()
            return True

        if command == "/feedback":
            await self._do_feedback(_args)
            return True

        if command == "/login":
            await self._do_login()
            return True

        if command == "/hf_login":
            await self._do_hf_login(_args)
            return True

        if command == "/clear":
            from lqh.project_log import set_log_session

            if self._session is not None:
                self._session.mark_state("completed")
            self._session = Session.create(self.project_dir)
            set_log_session(self._session.id)
            self._status_bar.session_id = self._session.id
            self._status_bar.prompt_tokens = 0
            self._status_bar.completion_tokens = 0
            self._status_bar.active_skill = ""
            self._agent = self._create_agent()
            await self._emit(render_system_message("Started new session."))
            # A fresh conversation still needs the current project context.
            await self._prepare_agent_context()
            return True

        if command == "/resume":
            await self._do_resume()
            return True

        skill_map = {
            "/spec": "spec_capture",
            "/datagen": "data_generation",
            "/validate": "data_validation",
            "/train": "train",
            "/eval": "evaluation",
            "/prompt": "prompt_optimization",
        }
        skill_name = skill_map.get(command)
        if skill_name:
            try:
                from lqh.skills import load_skill_content

                content = load_skill_content(skill_name)
            except FileNotFoundError as e:
                await self._emit(render_error(str(e)))
                return True

            if self._session:
                self._session.add_message({"role": "system", "content": content})
            self._status_bar.active_skill = skill_name
            self._invalidate()
            await self._emit(render_system_message(f"Loaded skill: {skill_name}"))

            if self._agent:
                self._lock_input()
                try:
                    await self._run_interruptible(
                        lambda: self._run_agent_with_reconnect(
                            lambda: self._agent.process_user_input(
                                f"[System: The {skill_name} skill is now active. "
                                f"Begin the workflow described in the skill instructions.]"
                            ),
                            lambda: self._agent.continue_after_interruption(),
                        )
                    )
                except AgentInterrupted:
                    pass
                except Exception as e:
                    await self._emit(render_error(f"{type(e).__name__}: {e}"))
                finally:
                    self._unlock_input()
            return True

        await self._emit(render_error(f"Unknown command: {command}"))
        return True

    async def _handle_message(self, text: str) -> None:
        """Handle a regular user message."""
        if not get_token():
            await self._emit(render_error("Not logged in. Please run /login first."))
            return

        await self._telemetry.run_deferred(self._telemetry.record_user_turn, "message")

        self._lock_input()
        await self._emit(render_user_message(text))

        try:
            if self._agent:
                await self._run_interruptible(
                    lambda: self._run_agent_with_reconnect(
                        lambda: self._agent.process_user_input(text),
                        lambda: self._agent.continue_after_interruption(),
                    )
                )
        except AgentInterrupted:
            pass
        except Exception as e:
            await self._emit(render_error(f"{type(e).__name__}: {e}"))
        finally:
            self._unlock_input()
            self._save_session()

    async def _run_agent_with_reconnect(
        self,
        start: Callable[[], Awaitable[None]],
        retry: Callable[[], Awaitable[None]],
    ) -> bool:
        """Run an agent action with bounded reconnect retries.

        ``start`` may append a new user/system message to the session. After a
        transient failure, ``retry`` must resume the same turn without adding
        another message.
        """
        action = start
        last_error: Exception | None = None

        for attempt in range(len(self._reconnect_backoffs) + 1):
            if attempt > 0:
                delay = self._reconnect_backoffs[attempt - 1]
                await self._emit(render_system_message(
                    f"Connection interrupted. Retrying in {delay:.0f}s..."
                ))
                await asyncio.sleep(delay)

            try:
                await action()
            except Exception as e:
                if not self._is_reconnectable_error(e):
                    raise
                last_error = e
                action = retry
                continue

            self._pending_reconnect = None
            self._pending_reconnect_error = None
            return True

        self._pending_reconnect = retry
        self._pending_reconnect_error = (
            f"{type(last_error).__name__}: {last_error}" if last_error else "Unknown error"
        )
        await self._emit(render_error(
            "Connection interrupted and automatic reconnect attempts failed. "
            "Run /reconnect to try again.\n"
            f"{self._pending_reconnect_error}"
        ))
        return False

    async def _do_reconnect(self) -> None:
        """Retry the last interrupted agent operation, if any."""
        if self._pending_reconnect is None:
            await self._emit(render_system_message("No reconnect is pending."))
            return

        action = self._pending_reconnect
        previous_error = self._pending_reconnect_error
        self._pending_reconnect = None
        self._pending_reconnect_error = None

        if previous_error:
            await self._emit(render_system_message(
                f"Retrying interrupted operation after: {previous_error}"
            ))
        else:
            await self._emit(render_system_message("Retrying interrupted operation."))

        was_processing = self._processing
        if not was_processing:
            self._lock_input()
        try:
            if self._agent:
                await self._run_interruptible(
                    lambda: self._run_agent_with_reconnect(action, action)
                )
        except AgentInterrupted:
            pass
        finally:
            if not was_processing:
                self._unlock_input()
            self._save_session()

    @staticmethod
    def _is_reconnectable_error(exc: Exception) -> bool:
        """Return True for transient network/API failures."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True

        try:
            import httpx
            if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
                return True
        except Exception:
            pass

        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                RateLimitError,
            )
            if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
                return True
            if isinstance(exc, APIStatusError):
                if exc.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                    return True
                # A 400 "request rejected by upstream model" is a transient
                # pool-side rejection, not a malformed request — re-sending the
                # same turn can land on a healthy pool member.
                from lqh.client import is_transient_upstream_error
                if is_transient_upstream_error(exc):
                    return True
        except Exception:
            pass

        if isinstance(exc, OSError):
            import errno
            transient_errno = {
                errno.ECONNABORTED,
                errno.ECONNRESET,
                errno.EHOSTDOWN,
                errno.EHOSTUNREACH,
                errno.ENETDOWN,
                errno.ENETRESET,
                errno.ENETUNREACH,
                errno.ETIMEDOUT,
            }
            return getattr(exc, "errno", None) in transient_errno

        return False

    async def _do_login(self) -> None:
        """Handle the /login command."""
        if os.environ.get("LQH_DEBUG_API_KEY"):
            await self._emit(render_system_message(
                "Using LQH_DEBUG_API_KEY from env; skipping device-code flow."
            ))
            self._status_bar.logged_in = True
            self._invalidate()
            return

        async def on_user_code(uri: str, code: str) -> None:
            await self._emit(render_system_message(
                f"Open {uri} and enter:\n\n   {code}\n\nWaiting for approval…"
            ))

        try:
            user = await login_device_code(on_user_code=on_user_code)
            self._status_bar.logged_in = True
            self._invalidate()
            email = user.get("email", "?") if isinstance(user, dict) else "?"
            await self._emit(render_system_message(f"✅ Logged in as {email}"))
            # The app may have started before a bearer existed. Record the
            # original CLI open now, preserving its new/pre-existing project
            # classification, and begin draining the account-bound queue.
            await self._telemetry.run_deferred(self._telemetry.refresh_account_binding)
            await self._telemetry.run_deferred(self._telemetry.start_session)
            self._start_telemetry_flush()
            # Late login: startup may have run logged out, skipping the
            # one-time cloud identity migration and the snapshot fetch.
            # Run both now so this session stops using the legacy
            # basename key and sees current cloud facts.
            try:
                from lqh.project_identity import migrate_cloud_identity

                await migrate_cloud_identity(self.project_dir)
            except Exception:
                logger.warning(
                    "post-login identity migration failed", exc_info=True
                )
            await self._refresh_cloud_snapshot()
        except LoginExpired:
            await self._emit(render_error("Device code expired. Run /login again."))
        except Exception as e:
            await self._emit(render_error(f"Login failed: {type(e).__name__}: {e}"))

    async def _do_hf_login(self, args: str) -> None:
        """Handle /hf_login: store a Hugging Face token on the backend so
        cloud jobs can use it for private repos and pushes."""
        import getpass

        from lqh.auth import set_hf_token

        token = (args or "").strip()
        if not token:
            try:
                token = (await run_in_terminal(
                    lambda: getpass.getpass("Paste your Hugging Face token (hidden): ")
                )).strip()
            except Exception:
                token = ""
        if not token:
            await self._emit(render_system_message("HF login cancelled (no token entered)."))
            return
        try:
            await set_hf_token(token)
        except Exception as e:
            await self._emit(render_error(f"Failed to store HF token: {type(e).__name__}: {e}"))
            return
        await self._refresh_hf_status()
        await self._emit(render_system_message(
            "✅ Hugging Face token stored (encrypted on the backend). Cloud jobs will "
            "use it for private models/datasets and pushes — the laptop env is not needed. "
            "For cloud data generation, the token is available directly to the trusted "
            "pipeline Python process; use a fine-grained, read-only token when possible."
        ))
        self._invalidate()

    async def _do_feedback(self, args: str) -> None:
        """Handle /feedback: collect free-text feedback and send it (along with
        the current conversation context) to the backend for super-admin
        review. The conversation is attached so the team can understand the
        feedback in context (FEEDBACK.md)."""
        import httpx

        from lqh.auth import send_feedback

        if not get_token():
            await self._emit(render_error("Not logged in. Please run /login first."))
            return

        await self._emit(render_system_message(
            "Your feedback, this session's full conversation — including "
            "system and tool messages, not just your chat — and a snapshot of "
            "your environment (OS, CPU/RAM/GPU, Python and package versions) "
            "will be sent to the lqh team for review. By submitting, you agree "
            "to our privacy policy: https://lqh.ai/privacy/"
        ))

        message = (args or "").strip()
        if not message:
            message = (await self._wait_for_user_response(
                managed_text=render_system_message(
                    "Type your feedback and press Enter (empty to cancel):",
                    separated=False,
                )
            )).strip()
        if not message:
            await self._emit(render_system_message("Feedback cancelled (nothing entered)."))
            return

        # Full raw transcript, not the (possibly compacted) working view.
        context = self._session.read_log() if self._session else []
        session_id = self._session.id if self._session else None

        # Lock input and show an in-flight indicator so the user knows the
        # request is outstanding and stray keystrokes aren't taken as a new
        # message while it's in flight.
        self._lock_input()
        await self._emit(render_system_message("⏳ Sending your feedback…"))
        try:
            await send_feedback(message, context, session_id)
        except httpx.TransportError:
            await self._emit(render_error(
                "Couldn't reach the lqh server (network/timeout). Your feedback "
                "was not sent — check your connection and run /feedback again."
            ))
            return
        except Exception as e:
            detail = str(e).strip() or type(e).__name__
            await self._emit(render_error(f"Failed to send feedback: {detail}"))
            return
        finally:
            self._unlock_input()

        await self._emit(render_system_message(
            "✅ Thanks — your feedback was sent to the lqh team."
        ))
        self._invalidate()

    async def _refresh_hf_status(self) -> None:
        """Resolve the project's compute target and, for cloud projects,
        query the backend for the stored-HF-token status so the 🤗
        indicator reflects what the sandbox will actually have.
        Best-effort — never raises into the caller."""
        try:
            from lqh.remote.compute import resolve_compute
            target = resolve_compute(self.project_dir)
        except Exception:
            target = "cloud"
        is_cloud = target == "cloud"
        self._status_bar.compute_is_cloud = is_cloud
        if is_cloud and get_token():
            try:
                from lqh.auth import hf_token_status
                status = await hf_token_status()
                self._status_bar.hf_cloud_configured = bool(status.get("configured"))
            except Exception:
                self._status_bar.hf_cloud_configured = None
        self._invalidate()

    async def _refresh_startup_state(self) -> None:
        """One-shot startup refresh, BEFORE any context/signals are built.

        1. Sync remote job state to disk (cloud_state.json / progress files
           are only as fresh as the last sync — without this, a job that
           finished while LQH was closed would be signaled as still
           running, or not at all).
        2. Fetch and cache the cloud snapshot (offline → cached copy,
           marked not fresh; logged out without a cache → not fresh so the
           unavailability is signaled).

        The results land on the app (picked up by every subsequently
        created agent) and on the current agent. The finished-while-away
        diff is computed exactly once here — it consumes the
        job_seen.json baseline, so /clear and /resume must reuse this
        list rather than recompute an already-consumed diff.
        """
        self._jobs_refreshed = True
        try:
            from lqh.subprocess_manager import SubprocessManager

            await asyncio.wait_for(
                self._supervisor.scan_jobs(SubprocessManager()), timeout=20.0
            )
        except Exception:
            # Stale run-state files will be read below; the signals must
            # say so instead of presenting them as trustworthy.
            self._jobs_refreshed = False

        try:
            from lqh.signals import (
                finished_while_away_signals,
                observe_run_states,
                record_seen_states,
            )

            run_states = observe_run_states(self.project_dir)
            self._startup_diff_signals = finished_while_away_signals(
                self.project_dir, run_states
            )
            if self._jobs_refreshed:
                # Only advance the baseline when the states are real; a
                # failed refresh must not consume terminal transitions it
                # never actually observed.
                record_seen_states(self.project_dir, run_states)
        except Exception:
            self._startup_diff_signals = []

        from lqh.snapshot import fetch_and_cache_snapshot, read_cached_snapshot

        self._cloud_snapshot = None
        self._cloud_snapshot_fresh = True
        try:
            if get_token():
                self._cloud_snapshot, self._cloud_snapshot_fresh = (
                    await fetch_and_cache_snapshot(self.project_dir)
                )
            else:
                self._cloud_snapshot = read_cached_snapshot(self.project_dir)
                self._cloud_snapshot_fresh = False
        except Exception:
            self._cloud_snapshot_fresh = False
        if self._agent:
            self._apply_startup_facts(self._agent)

    async def _refresh_cloud_snapshot(self) -> None:
        """Re-fetch the cloud snapshot outside the one-shot startup path
        (e.g. after a late /login). Does NOT recompute the seen-jobs
        diff — that baseline is consumed exactly once at startup."""
        from lqh.snapshot import fetch_and_cache_snapshot

        try:
            self._cloud_snapshot, self._cloud_snapshot_fresh = (
                await fetch_and_cache_snapshot(self.project_dir)
            )
        except Exception:
            logger.warning("cloud snapshot refresh failed", exc_info=True)
            self._cloud_snapshot_fresh = False
        if self._agent:
            self._apply_startup_facts(self._agent)

    def _apply_startup_facts(self, agent: Agent) -> None:
        agent.set_startup_facts(
            snapshot=getattr(self, "_cloud_snapshot", None),
            snapshot_fresh=getattr(self, "_cloud_snapshot_fresh", True),
            jobs_refreshed=getattr(self, "_jobs_refreshed", True),
            diff_signals=getattr(self, "_startup_diff_signals", None),
        )

    async def _prepare_agent_context(self) -> None:
        """Run the agent's ephemeral context preparation and announce it.

        Called at startup and after /clear and /resume so every
        conversation (re)start sees the current SPEC.md, NOTES.md,
        one-line inventory, and attention signals (R3: everything
        deeper — summaries, logs — is pull-side via the agent's tools).
        """
        if not self._agent:
            return
        mode = await self._agent.prepare_context()
        await self._announce_context_mode(mode)

    async def _announce_context_mode(self, mode: str) -> None:
        if mode == "new_project":
            self._status_bar.active_skill = "spec_capture"
            self._invalidate()
            await self._emit(
                render_agent_message(
                    "👋 **Welcome to Liquid Harness!**\n\n"
                    "I don't see a `SPEC.md` in this directory yet, so let's start "
                    "by figuring out what problem you want to solve.\n\n"
                    "**What kind of model do you want to build?** Describe the task — "
                    "for example: *\"I need a model that summarizes academic papers into "
                    "bullet points for students\"* or *\"I want a model that classifies "
                    "customer support tickets by urgency.\"*\n\n"
                    "The more context you give me, the better I can help. I'll ask "
                    "follow-up questions to nail down the details before we create "
                    "your specification."
                )
            )
        else:
            await self._emit(
                render_system_message(
                    "📂 Loaded SPEC.md, notes, and project signals into "
                    "context (the agent pulls full summaries with its "
                    "tools). Ready to go."
                )
            )

    async def _render_session_history(self, limit: int = 200) -> None:
        """Re-render the resumed conversation from the raw transcript."""
        if self._session is None:
            return
        for message in self._session.read_log(limit=limit):
            role = message.get("role")
            content = message.get("content", "")
            if role == "user" and isinstance(content, str) and not content.startswith("[System:"):
                await self._emit(render_user_message(content))
            elif role == "assistant" and content:
                await self._emit(render_agent_message(str(content)))

    def _adopt_session(self, session: Session) -> None:
        """Point the TUI and a fresh agent at ``session``."""
        from lqh.project_log import set_log_session

        self._session = session
        set_log_session(session.id)
        self._status_bar.session_id = session.id
        self._status_bar.prompt_tokens = session.prompt_tokens
        self._status_bar.completion_tokens = session.completion_tokens
        self._agent = self._create_agent()
        self._invalidate()

    async def _offer_interrupted_resume(self) -> None:
        """Offer to pick up the newest interrupted session at startup.

        Sessions left ``active`` by a dead process were just repaired to
        ``interrupted`` (``Session.repair_states`` in ``run()``). Only the
        newest one is offered, resume preselected; anything older stays
        reachable through /resume.
        """
        try:
            sessions = Session.list_sessions(self.project_dir)
        except Exception:
            return
        if not sessions or sessions[0].get("state") != "interrupted":
            return
        newest = sessions[0]
        preview = (newest.get("preview") or "(no preview)")[:60]
        choice = await self._wait_for_user_response(options=[
            f"Resume interrupted session: {preview} ({newest.get('updated_at', '?')})",
            "Start a new session",
        ])
        if not choice.startswith("Resume"):
            return
        try:
            session = Session.load(self.project_dir, newest["id"])
        except Exception:
            await self._emit(render_error(
                "Could not load the interrupted session — starting fresh."
            ))
            return
        # Atomic ownership claim (CLI_PLAN §7): a headless `lqh run` may
        # own this session right now — never interleave two loops in one
        # conversation.
        if not session.claim_active():
            await self._emit(render_error(
                "That session is active in another process (e.g. `lqh run`) "
                "— starting fresh instead."
            ))
            return
        self._adopt_session(session)
        await self._emit(render_system_message(f"Resumed session {newest['id'][:8]}"))
        await self._render_session_history()

    async def _do_resume(self) -> None:
        """Handle the /resume command."""
        sessions = Session.list_sessions(self.project_dir)
        if not sessions:
            await self._emit(render_system_message("No previous sessions found."))
            return

        options = []
        for i, session_info in enumerate(sessions[:10]):
            preview = session_info.get("preview", "(empty)")[:60]
            timestamp = session_info.get("created_at", "?")
            options.append(f"{i + 1}. {timestamp} - {preview}")

        await self._emit(render_system_message("Select a session to resume:"))
        selected = await self._wait_for_user_response(options=options)

        try:
            index = options.index(selected)
        except ValueError:
            await self._emit(render_system_message(
                "Selection did not match a session — nothing resumed."
            ))
            return

        session_info = sessions[index]
        loaded = Session.load(self.project_dir, session_info["id"])
        # Atomic ownership claim (CLI_PLAN §7): refuse to interleave with a
        # session a headless `lqh run` currently owns.
        if not loaded.claim_active():
            await self._emit(render_error(
                "That session is active in another process (e.g. `lqh run`) "
                "— not resumed."
            ))
            return
        if self._session is not None:
            self._session.mark_state("completed")
        self._adopt_session(loaded)

        await self._emit(render_system_message(f"Resumed session {session_info['id'][:8]}"))
        await self._render_session_history()
        # Stored messages are restored verbatim; the current project state
        # arrives as the agent's ephemeral context prefix.
        await self._prepare_agent_context()

    def _create_agent(self) -> Agent:
        """Create an agent with TUI callbacks."""
        if self._session is None:
            raise RuntimeError("Session is not initialized.")

        callbacks = AgentCallbacks(
            on_agent_message=self._on_agent_message,
            on_tool_call=self._on_tool_call,
            on_tool_result=self._on_tool_result,
            on_ask_user=self._on_ask_user,
            on_show_file=self._on_show_file,
            on_show_secret=self._on_show_secret,
            on_spinner_start=self._on_spinner_start,
            on_spinner_stop=self._on_spinner_stop,
            on_token_update=self._on_token_update,
            on_skill_loaded=self._on_skill_loaded,
            on_pipeline_progress=self._on_pipeline_progress,
            legacy_pipeline_progress_callback=False,
            on_pipeline_done=self._on_pipeline_done,
            on_background_task_started=self._on_background_task_started,
            on_await_background=self._await_background,
            on_auto_stage=self._on_auto_stage,
        )
        agent = Agent(
            self.project_dir,
            self._session,
            callbacks,
            auto_mode=self.auto_mode,
            extra_spec=self.extra_spec,
        )
        self._apply_startup_facts(agent)
        return agent

    def _on_auto_stage(self, stage: str, note: str | None) -> None:
        """Auto-mode: agent reported a stage transition."""
        self._auto_stage = stage
        self._auto_stage_note = note
        line = f"• {stage}" + (f" — {note}" if note else "")
        # Only append if it's a new stage (avoid duplicates from chatty agents)
        if not self._auto_history or self._auto_history[-1] != line:
            self._auto_history.append(line)
        self._invalidate()

    def _update_task_progress(self, run_name: str) -> None:
        """Push the run's latest step/percent into the status-bar registry.

        Reads ``progress.jsonl`` (already rsynced locally for remote runs by
        ``_poll_remote``) and updates the task's ``progress`` string. The
        ``updated_at`` timestamp only advances when the step itself advances,
        so a stalled run shows a growing "↑Xm ago" age in the status bar.
        """
        import math

        from lqh.progress import (
            format_event_oneline,
            read_progress_events,
            select_display_event,
        )
        from lqh.train.progress import format_progress_oneline, read_current_attempt_id

        run_dir = self.project_dir / "runs" / run_name
        try:
            history = read_progress_events(run_dir, last_n=256)
        except Exception:
            return
        current_attempt = read_current_attempt_id(run_dir)
        if isinstance(current_attempt, str) and current_attempt:
            # A known new attempt with no v1 rows must show setup/legacy state,
            # never the previous attempt's high-water mark.
            history = [
                row for row in history
                if row.get("attempt_id") == current_attempt
            ]
            if not history:
                self._job_last_step.pop(run_name, None)
                self._job_progress_at.pop(run_name, None)
                self._tasks.update(run_name, progress=None, updated_at=None)
                return
        v1_rows = [
            row for row in history
            if isinstance(row.get("overall_fraction"), (int, float))
            and math.isfinite(float(row["overall_fraction"]))
        ]
        # Fractions are the cross-machine ordering key. Raw ISO timestamps may
        # come from skewed producer and observer clocks.
        v1_rows.sort(key=lambda row: float(row["overall_fraction"]))
        if v1_rows:
            # Whole-job progress is authoritative once a v1 producer appears.
            # Pick the furthest monotonic event so another process cannot make
            # the display regress by appending an older/legacy observation.
            latest = select_display_event(v1_rows)
            if latest is None:
                return
            step = latest.get("overall_fraction")
        else:
            latest = next(
                (row for row in reversed(history) if "step" in row), None,
            )
            step = (
                latest.get("child_step", latest.get("step"))
                if latest else None
            )
        if latest is None:
            if isinstance(current_attempt, str) and current_attempt:
                self._job_last_step.pop(run_name, None)
                self._job_progress_at.pop(run_name, None)
                self._tasks.update(run_name, progress=None, updated_at=None)
            return
        progress_key = (
            ("fraction" if v1_rows else "step"), float(step),
        ) if isinstance(step, (int, float)) else None
        if progress_key is not None and self._job_last_step.get(run_name) != progress_key:
            self._job_last_step[run_name] = progress_key
            self._job_progress_at[run_name] = time.time()
        if v1_rows:
            latest_phase = latest.get("phase")
            display_history = [
                row for row in v1_rows if row.get("phase") == latest_phase
            ]
            line, _pct = format_event_oneline(
                latest,
                history=display_history,
                observed_at=self._job_progress_at.get(run_name),
            )
        else:
            line, _pct = format_progress_oneline(latest, history=history)
        if not line:
            return
        self._tasks.update(
            run_name,
            progress=line,
            updated_at=self._job_progress_at.get(run_name),
        )

    def _on_background_task_started(
        self, task_id: str, kind: str, label: str, remote: str | None,
    ) -> None:
        """Handler hook: a tool just submitted a job that will notify later."""
        self._supervisor.register_started(task_id, kind, label, remote)
        self._ensure_progress_refresh_task()
        enabled, consent_epoch, active_baseline, _account_key = self._telemetry.state_snapshot()
        if remote != "cloud" and enabled:
            workflow_id = str(uuid.uuid4())
            workflow_kind = "zero_shot_evaluation" if kind == "eval" else "fine_tuning"
            target = "ssh" if remote else "local"
            metadata = {"workflow_kind": workflow_kind, "execution_target": target}
            if kind == "eval":
                metadata["subtype"] = "zero_shot"
            event_name = "zero_shot_evaluation_started" if kind == "eval" else "fine_tuning_started"
            self._telemetry.defer(self._telemetry.event, event_name, metadata, workflow_id)
            started_mono, started_wall = time.monotonic(), time.time()
            self._telemetry_jobs[task_id] = (
                workflow_id, started_mono, started_wall, 0.0, active_baseline,
                workflow_kind, target, consent_epoch,
            )
            self._persist_telemetry_job(
                task_id, workflow_id, started_mono, started_wall, 0.0,
                workflow_kind, target, consent_epoch,
            )

    def _telemetry_job_path(self, run_name: str) -> Path:
        return self.project_dir / "runs" / run_name / ".telemetry_workflow.json"

    def _persist_telemetry_job(
        self, run_name: str, workflow_id: str, started_mono: float | None,
        started_wall: float, active_seconds: float, workflow_kind: str,
        target: str, consent_epoch: int,
    ) -> None:
        path = self._telemetry_job_path(run_name)
        try:
            account_key = self._telemetry.state_snapshot()[3]
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "workflow_id": workflow_id, "started_mono": started_mono, "started_wall": started_wall,
                "workflow_kind": workflow_kind, "target": target,
                "account_key": account_key,
                "consent_epoch": consent_epoch,
                "active_seconds": active_seconds,
            }, separators=(",", ":")) + "\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except OSError:
            return

    def _load_telemetry_job(self, run_name: str) -> tuple[str, float | None, float, float, float, str, str, int] | None:
        try:
            path = self._telemetry_job_path(run_name)
            os.chmod(path, 0o600)
            value = json.loads(path.read_text())
            _enabled, consent_epoch, active_seconds, account_key = self._telemetry.state_snapshot()
            if (value.get("account_key") != account_key
                    or value.get("consent_epoch", 0) != consent_epoch):
                return None
            started_mono = float(value.get("started_mono", 0))
            if started_mono <= 0 or started_mono > time.monotonic():
                started_mono = None
            return (
                str(uuid.UUID(value["workflow_id"])), started_mono,
                float(value["started_wall"]), float(value.get("active_seconds", 0)),
                active_seconds, str(value["workflow_kind"]),
                str(value["target"]), int(value.get("consent_epoch", 0)),
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _clear_persisted_telemetry_jobs(self) -> None:
        self._telemetry_jobs.clear()
        try:
            for path in (self.project_dir / "runs").glob("*/.telemetry_workflow.json"):
                path.unlink(missing_ok=True)
        except OSError:
            return

    def _checkpoint_telemetry_jobs(self) -> None:
        """Persist interaction-active time for local/SSH jobs across restarts."""
        active_seconds = self._telemetry.state_snapshot()[2]
        for run_name, job in list(self._telemetry_jobs.items()):
            workflow_id, started_mono, started_wall, prior, baseline, kind, target, consent_epoch = job
            active = prior + max(active_seconds - baseline, 0)
            self._telemetry_jobs[run_name] = (
                workflow_id, started_mono, started_wall, active,
                active_seconds, kind, target, consent_epoch,
            )
            self._persist_telemetry_job(
                run_name, workflow_id, started_mono, started_wall, active,
                kind, target, consent_epoch,
            )

    def _ensure_progress_refresh_task(self) -> None:
        """Refresh progress from local run mirrors without polling job state."""
        if self._progress_refresh_task and not self._progress_refresh_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Some embedders/unit tests register display state before mounting
            # the async application. The normal run startup will retry.
            return

        async def refresh() -> None:
            try:
                while len(self._tasks):
                    for task in self._tasks.snapshot():
                        try:
                            self._update_task_progress(task.task_id)
                        except Exception:
                            # One malformed/deleted run must not kill refresh
                            # for every other active task.
                            continue
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return
            except Exception:
                # The lifecycle poll will recreate the refresh task; consume
                # unexpected failures so asyncio does not report an orphaned
                # task exception.
                return

        self._progress_refresh_task = loop.create_task(refresh())

    async def _on_agent_message(self, text: str) -> None:
        await self._emit(render_agent_message(text))

    async def _on_tool_call(self, name: str, args: dict) -> None:
        await self._emit(render_tool_call(name, args))

    async def _on_tool_result(self, name: str, content: str) -> None:
        await self._emit(render_tool_result(name, content))

    async def _on_ask_user(
        self,
        question: str,
        options: list[str] | None,
        multi_select: bool = False,
        allow_other: bool = True,
    ) -> str:
        """Handle ask_user tool requests.

        ``allow_other=False`` suppresses the auto-injected "Other (please
        specify)" free-text option — used for fixed-choice confirms (e.g. the
        secret-delivery prompt) where free text makes no sense.
        """
        await self._emit(render_agent_message(f"❓ {question}"))

        if options:
            filtered = [option for option in options if not _is_other_option(option)]
            all_options = filtered + [OTHER_OPTION] if allow_other else filtered
            response = await self._wait_for_user_response(
                options=all_options,
                allow_other=allow_other,
                multi_select=multi_select,
                relock_after=True,
            )
            await self._telemetry.run_deferred(self._telemetry.record_user_turn, "ask_user_answer")
            return response

        response = await self._wait_for_user_response(
            managed_text=render_system_message("Type your response:", separated=False),
            relock_after=True,
        )
        await self._telemetry.run_deferred(self._telemetry.record_user_turn, "ask_user_answer")
        return response

    async def _on_show_secret(self, text: str) -> None:
        """Display a one-time secret in a distinct panel (out-of-band).

        Never enters the conversation — the agent loop returns a redacted
        message in its place.
        """
        await self._emit(render_secret(text))

    async def _on_show_file(self, path: str) -> str | None:
        """Display a file to the user. Returns viewer summary for parquet files."""
        full_path = self.project_dir / path
        try:
            if full_path.suffix == ".parquet":
                return await self._open_dataset_viewer(full_path)

            content = full_path.read_text(encoding="utf-8")
            await self._emit(render_file_view(path, content))
            return None
        except Exception as e:
            await self._emit(render_error(f"Cannot display {path}: {e}"))
            return None

    async def _open_dataset_viewer(self, parquet_path: Path) -> str:
        """Open the interactive dataset viewer."""
        viewer = DatasetViewer(parquet_path)

        if viewer.empty:
            await self._emit(render_system_message(f"Dataset {parquet_path.name} is empty (0 rows)."))
            return viewer.get_summary()

        self._dataset_viewer = viewer
        self._dataset_viewer_future = asyncio.get_event_loop().create_future()
        self._invalidate()
        self._unlock_input()

        await self._render_dataset_viewer()

        try:
            return await self._dataset_viewer_future
        finally:
            self._dataset_viewer = None
            self._dataset_viewer_future = None
            self._set_managed_text("")
            self._lock_input()
            self._invalidate()

    async def _render_dataset_viewer(self) -> None:
        """Render the current dataset sample and nav help in the managed area."""
        if self._dataset_viewer is None:
            return

        self._set_managed_text(self._dataset_view_text())

    async def _handle_dataset_input(self, text: str) -> None:
        """Handle typed dataset viewer commands."""
        command = text.lower() if text else "n"
        if command in {"n", "next"} and self._dataset_viewer:
            self._dataset_viewer.go_next()
            await self._render_dataset_viewer()
        elif command in {"p", "prev", "previous"} and self._dataset_viewer:
            self._dataset_viewer.go_prev()
            await self._render_dataset_viewer()
        elif command in {"r", "random"} and self._dataset_viewer:
            self._dataset_viewer.go_random()
            await self._render_dataset_viewer()
        elif command in {"q", "quit", "exit"}:
            await self._close_dataset_viewer()
        else:
            self._set_managed_text(
                self._dataset_view_text(
                    render_system_message(
                        "Dataset viewer commands: n, p, r, q.",
                        separated=False,
                    )
                )
            )

    async def _close_dataset_viewer(self) -> None:
        """Close the dataset viewer and resolve the pending future."""
        if self._dataset_viewer is None or self._dataset_viewer_future is None:
            return

        await self._emit(
            render_system_message(
                f"Closed dataset viewer (viewed {len(self._dataset_viewer.viewed_indices)} sample(s))"
            )
        )
        if not self._dataset_viewer_future.done():
            self._dataset_viewer_future.set_result(self._dataset_viewer.get_summary())

    def _ensure_spinner_task(self) -> None:
        """Run one shared spinner loop for both thinking and pipeline updates."""
        if self._spinner_task and not self._spinner_task.done():
            return

        async def spin() -> None:
            try:
                while (
                    self._status_bar.spinning
                    or self._status_bar.pipeline_status
                    or (
                        self._status_bar.recent_completion is not None
                        and self._status_bar.recent_completion[1] > time.time()
                    )
                ):
                    animated = bool(
                        self._status_bar.spinning
                        or self._status_bar.pipeline_status
                    )
                    if animated:
                        self._status_bar.advance_spinner()
                    self._invalidate()
                    await asyncio.sleep(0.08 if animated else 1.0)
                self._invalidate()
            except asyncio.CancelledError:
                return

        self._spinner_task = asyncio.get_event_loop().create_task(spin())

    def _cancel_spinner_task(self) -> None:
        """Stop the background spinner loop if it is running."""
        if self._spinner_task:
            self._spinner_task.cancel()
            self._spinner_task = None

    def _on_spinner_start(self) -> None:
        """Start the spinner animation."""
        self._status_bar.start_spinning()
        self._ensure_spinner_task()
        self._invalidate()

    def _on_spinner_stop(self) -> None:
        """Stop the spinner animation."""
        self._status_bar.stop_spinning()
        if not self._status_bar.pipeline_status:
            self._cancel_spinner_task()
        self._invalidate()

    def _on_token_update(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token counts in the status bar."""
        self._status_bar.prompt_tokens = prompt_tokens
        self._status_bar.completion_tokens = completion_tokens
        self._invalidate()

    async def _on_skill_loaded(self, skill_name: str) -> None:
        """Update the active skill indicator."""
        self._status_bar.active_skill = skill_name
        self._invalidate()

    def _on_pipeline_progress(self, event: Any) -> None:
        """Update foreground progress using the shared event renderer."""
        from lqh.progress import ProgressEvent, format_event_oneline

        if not isinstance(event, ProgressEvent):
            return
        self._foreground_progress_history.append(event)
        self._foreground_progress_history = self._foreground_progress_history[-256:]
        line, _ = format_event_oneline(
            event, history=self._foreground_progress_history,
        )
        label = event.label or event.task_kind.replace("_", " ").title()
        self._status_bar.pipeline_status = f"{label} · {line}"
        if not self._status_bar.spinning:
            self._status_bar.start_spinning()
        self._ensure_spinner_task()
        self._invalidate()

    def _on_pipeline_done(self) -> None:
        """Clear pipeline progress from the status bar."""
        self._status_bar.pipeline_status = ""
        self._foreground_progress_history.clear()
        self._status_bar.stop_spinning()
        self._cancel_spinner_task()
        self._invalidate()

    def _save_session(self) -> None:
        """Save the current session if it has user messages."""
        if self._session:
            self._session.save()

    async def _run_auto_mode(self) -> None:
        """Drive a non-interactive auto-mode run from SPEC.md to terminal state."""
        if self._agent is None:
            return

        spec_path = self.project_dir / "SPEC.md"
        if not spec_path.is_file():
            await self._emit(render_error(
                f"Auto mode requires SPEC.md in {self.project_dir}, but none was found."
            ))
            return

        try:
            spec_text = spec_path.read_text(encoding="utf-8")
        except OSError as e:
            await self._emit(render_error(f"Cannot read SPEC.md: {e}"))
            return

        await self._emit(render_system_message(
            f"🤖 Auto mode starting in {self.project_dir}. "
            "The agent will run the full pipeline; no input needed."
        ))

        # Background job watcher is needed because auto mode also kicks off
        # training subprocesses whose completion notifications drive the
        # agent forward.
        self._job_watcher_task = asyncio.create_task(self._watch_jobs())

        kickoff = (
            "Here is the spec for this auto-mode run (also at SPEC.md):\n\n"
            "```\n"
            f"{spec_text}\n"
            "```\n\n"
            "Begin the auto-mode pipeline. Use set_auto_stage to report each "
            "stage. When you reach a terminal state, call "
            "exit_auto_mode(status, reason)."
        )

        try:
            # The pipeline is one long agent turn; a single Ctrl+C pauses it
            # (AgentInterrupted), collects a user instruction, and resumes the
            # run with that instruction injected. Auto mode itself is never
            # left — only interrupted.
            next_input = kickoff
            while True:
                try:
                    ok = await self._run_interruptible(
                        lambda msg=next_input: self._run_agent_with_reconnect(
                            lambda: self._agent.process_user_input(msg),
                            lambda: self._agent.continue_after_interruption(),
                        )
                    )
                except AgentInterrupted:
                    instruction = await self._pause_auto_mode()
                    if instruction is None:
                        # Shutdown requested while paused.
                        self._auto_done = True
                        self._invalidate()
                        return
                    await self._emit(render_user_message(instruction))
                    next_input = (
                        "[User interruption] The user paused the auto run to say:\n\n"
                        f"{instruction}\n\n"
                        "Acknowledge and incorporate this, then continue the "
                        "auto-mode pipeline. Only call exit_auto_mode if the "
                        "user asked you to stop."
                    )
                    continue
                if not ok:
                    self._auto_done = True
                    self._invalidate()
                    return
                break
        except Exception as e:
            await self._emit(render_error(
                f"Auto mode crashed: {type(e).__name__}: {e}"
            ))
            self._auto_done = True
            self._invalidate()
            return
        finally:
            if self._progress_refresh_task is not None:
                self._progress_refresh_task.cancel()
                self._progress_refresh_task = None
            if self._job_watcher_task is not None:
                self._job_watcher_task.cancel()
                try:
                    await self._job_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._job_watcher_task = None
            for watcher in list(self._run_watchers.values()):
                try:
                    await watcher.stop()
                except Exception:
                    pass
            self._run_watchers.clear()

        # Pipeline finished. Render the terminal state.
        self._auto_done = True
        self._invalidate()
        exit_info = self._agent._auto_exit
        if exit_info is None:
            await self._emit(render_error(
                "Auto mode ended without calling exit_auto_mode. "
                "Treating as failure."
            ))
            return
        status, reason = exit_info
        icon = "✅" if status == "success" else "❌"
        await self._emit(render_system_message(
            f"{icon} Auto mode {status}: {reason}"
        ))
        # Surface the final summary that the agent was instructed to print.
        await self._emit(render_system_message(
            "See the conversation log above for the full results table; "
            "checkpoints are under runs/, datasets under datasets/."
        ))

    async def _pause_auto_mode(self) -> str | None:
        """Collect a user instruction while an auto run is paused.

        Entered after a user interrupt (single Ctrl+C) cancelled the auto-mode
        agent turn. Unhides the input row, waits for a typed instruction, and
        returns it so the caller resumes the pipeline with it injected.
        Returns ``None`` when the app is shutting down instead.
        """
        self._auto_paused = True
        self._unlock_input()
        self._invalidate()
        await self._emit(render_system_message(
            "⏸ Auto mode paused. Type an instruction and press Enter to "
            "continue the run; /quit (or Ctrl+C twice) exits."
        ))
        try:
            while True:
                text = await self._input_queue.get()
                if text == _SHUTDOWN_SENTINEL or self._shutdown_requested:
                    return None
                if text.startswith("[System:"):
                    # Background completion notice, not user input. Dropping
                    # it is safe: the notice stays in _pending_completions and
                    # is re-delivered when the agent next parks on that run.
                    continue
                if is_command(text):
                    command, _ = parse_command(text)
                    if command == "/quit":
                        self._save_session()
                        self._request_shutdown()
                        return None
                    await self._emit(render_system_message(
                        "Slash commands aren't available during an auto run. "
                        "Type an instruction to continue, or /quit to exit."
                    ))
                    continue
                return text
        finally:
            self._auto_paused = False
            self._lock_input()
            self._invalidate()

    async def _await_background(
        self, run_names: list[str] | None, timeout: float,
    ) -> str | None:
        """Auto-mode: park the agent until a watched run reaches a terminal
        state. Delegates to the shared JobSupervisor (lqh/jobs.py); while
        parked the agent's inner loop is suspended — no LLM calls happen
        until the run is terminal. The user watches live progress in the
        status bar meanwhile."""
        return await self._supervisor.wait_for_runs(
            run_names, recheck_interval=timeout,
        )

    async def _watch_jobs(self) -> None:
        """Run the shared supervisor scan loop (see lqh/jobs.py)."""
        await self._supervisor.watch_loop()

    # ------------------------------------------------------------------
    # JobSupervisor hooks: UI + telemetry side effects of supervision.
    # ------------------------------------------------------------------

    async def _on_watch_gap(self) -> None:
        await self._emit(render_system_message(
            "Resuming background job monitoring after a connection/sleep gap."
        ))

    def _on_job_notice(self, run_name: str, text: str, state: str) -> None:
        """A completion notice was recorded — wake/notify the main loop."""
        self._input_queue.put_nowait(text)
        if state == "completed":
            self._status_bar.recent_completion = (run_name, time.time() + 10.0)
            self._ensure_spinner_task()

    def _on_job_running(self, run_name: str, remote: str | None) -> None:
        """A run was observed running — rehydrate telemetry + progress UI."""
        if run_name not in self._telemetry_jobs:
            persisted_job = self._load_telemetry_job(run_name)
            if persisted_job is not None:
                self._telemetry_jobs[run_name] = persisted_job
        self._ensure_progress_refresh_task()
        self._update_task_progress(run_name)

    def _on_job_terminal(self, run_name: str) -> None:
        self._job_last_step.pop(run_name, None)
        self._job_progress_at.pop(run_name, None)

    def _has_telemetry_job_record(
        self, run_name: str, first_observation: bool,
    ) -> bool:
        """Notification gate: a terminal run with a (persisted) telemetry
        workflow completed while this TUI was closed and must be
        reconciled exactly once."""
        if run_name in self._telemetry_jobs:
            return True
        return first_observation and self._load_telemetry_job(run_name) is not None

    def _on_data_gen_terminal(
        self, outcome: str, workflow_id: str, marker: dict,
    ) -> None:
        """Telemetry mirror of a cloud data-gen terminal outcome."""
        # A give-up already closed the workflow; a post-restart retry
        # must not close it a second time.
        if marker.get("workflow_closed"):
            return
        enabled, _epoch, _baseline, _key = self._telemetry.state_snapshot()
        if not enabled:
            return
        event_name = (
            "data_generation_completed" if outcome == "succeeded"
            else "data_generation_failed"
        )
        self._telemetry.defer(self._telemetry.event, event_name, {
            "workflow_kind": "data_generation",
            "execution_target": "cloud",
            "outcome": outcome,
        }, workflow_id)

    def _record_telemetry_completion(
        self, run_name: str, state: str, error: str | None, remote: str | None,
    ) -> None:
        """Telemetry mirror of a run completion (the supervisor already
        wrote the run manifest and the project-log event)."""
        telemetry_job = self._telemetry_jobs.pop(run_name, None) or self._load_telemetry_job(run_name)
        if telemetry_job is None:
            return
        workflow_id, started_mono, started_wall, prior_active, active_baseline, workflow_kind, target, consent_epoch = telemetry_job
        elapsed_seconds = time.monotonic() - started_mono if started_mono is not None else time.time() - started_wall
        elapsed_ms = int(max(elapsed_seconds, 0) * 1000)
        active_seconds = self._telemetry.state_snapshot()[2]
        active_ms = int((prior_active + max(active_seconds-active_baseline, 0)) * 1000)
        outcome = "succeeded" if state == "completed" else "failed"
        metadata: dict[str, Any] = {
            "workflow_kind": workflow_kind, "execution_target": target,
            "outcome": outcome, "wall_duration_ms": elapsed_ms,
            "active_duration_ms": active_ms,
        }
        config_path = self.project_dir / "runs" / run_name / "config.json"
        try:
            config = __import__("json").loads(config_path.read_text())
            config_type = str(config.get("type", ""))
            base_type = str((config.get("base_config") or {}).get("type", "")) if isinstance(config.get("base_config"), dict) else ""
            if workflow_kind == "zero_shot_evaluation":
                metadata["subtype"] = "zero_shot"
                metadata["sample_count"] = int(config.get("num_samples", 0) or 0)
            elif config_type == "sweep":
                metadata["subtype"] = "dpo_sweep" if base_type in {"dpo", "on_policy_dpo"} else "sft_sweep"
            else:
                metadata["subtype"] = "direct_dpo" if config_type in {"dpo", "on_policy_dpo"} else "direct_sft"
        except (OSError, ValueError, TypeError):
            if workflow_kind == "zero_shot_evaluation": metadata["subtype"] = "zero_shot"
        event_name = ("zero_shot_evaluation_" if workflow_kind == "zero_shot_evaluation" else "fine_tuning_") + ("completed" if state == "completed" else "failed")
        self._telemetry.defer(
            self._finalize_telemetry_job,
            event_name, metadata, workflow_id, consent_epoch,
            self._telemetry_job_path(run_name),
        )

    def _finalize_telemetry_job(
        self, event_name: str, metadata: dict[str, Any], workflow_id: str,
        consent_epoch: int, checkpoint_path: Path,
    ) -> None:
        """Emit and clear a job checkpoint on the ordered telemetry worker."""
        if self._telemetry.consent_active(consent_epoch):
            self._telemetry.event(event_name, metadata, workflow_id)
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError:
            pass

    async def run(self) -> None:
        """Run the persistent bottom application."""
        self._shutdown_requested = False
        # Stable project identity is the FIRST unconditional startup step
        # (R4) — before sessions, telemetry, welcome, or the login prompt,
        # so quitting early can never leave a project without an identity.
        # A corrupt identity file is surfaced (after the UI is up), never
        # silently replaced: cloud key resolution fails closed on it.
        from lqh.headless import claim_loop, headless_boot

        # Shared with the headless CLI (lqh/headless.py): identity, copy
        # detection, and repairing sessions left "active" by a dead
        # process so startup can offer to resume them.
        boot = headless_boot(self.project_dir)
        identity_error = boot.identity_error
        copy_status = boot.copy_status
        # Advisory loop marker (CLI_PLAN §7): lets a concurrent headless
        # `lqh run` / mutating `lqh tool call` warn that an agent loop is
        # already working here. Best-effort; released on shutdown.
        claim_loop(self.project_dir)
        self._session = Session.create(self.project_dir)
        from lqh.project_log import set_log_session
        set_log_session(self._session.id)
        # Telemetry sessions are intentionally independent from conversation
        # sessions (/clear creates another conversation, not another CLI open).
        set_active_telemetry(self._telemetry)
        await self._telemetry.run_deferred(self._telemetry.start_session)
        self._telemetry_heartbeat_task = asyncio.create_task(self._telemetry_heartbeat())
        self._status_bar.session_id = self._session.id
        self._agent = self._create_agent()
        app_task = self._start_application_task()
        await asyncio.sleep(0)

        await self._emit(render_welcome())
        if notice_needed():
            await self._emit(render_system_message(
                "LQH collects limited usage and workflow timing telemetry to improve the product. "
                "No prompts, responses, file names, paths, or file contents are collected. "
                "Use /telemetry off to opt out."
            ))
        self._start_telemetry_flush()
        self._update_check_task = asyncio.create_task(self._show_update_notice())

        token = get_token()
        self._status_bar.logged_in = bool(token)
        self._invalidate()
        if token:
            # Best-effort: reflect the stored HF-token status for cloud
            # projects in the 🤗 indicator.
            await self._refresh_hf_status()
        if token:
            await self._emit(render_system_message("✅ Logged in to lqh.ai"))
        else:
            await self._emit(render_system_message("⚠️ Not logged in. Run /login to authenticate."))

        if not self.auto_mode and not get_token():
            choice = await self._wait_for_user_response(
                options=["Yes, log in now", "No, continue without login"],
            )
            if choice.startswith("Yes"):
                await self._do_login()
                self._status_bar.logged_in = bool(get_token())
                self._invalidate()
            else:
                await self._emit(render_system_message(
                    "Continuing without login. Run /login when you're ready."
                ))

        # Stable project identity (Phase 3), interactive half: the file
        # itself was ensured at the very top of run(); here we surface a
        # corrupt identity, resolve folder copies explicitly, and run the
        # one-time basename→UUID cloud migration when authenticated.
        if identity_error:
            await self._emit(render_error(
                f"⚠️ Project identity problem: {identity_error}\n"
                "Cloud operations (jobs, artifacts, deployments) will fail "
                "until .lqh/project.json is fixed — it is NOT auto-replaced "
                "because that would disconnect this directory from its "
                "cloud history."
            ))
        elif copy_status == "copied":
            from lqh.project_identity import (
                fork_identity,
                record_continue_decision,
            )

            if self.auto_mode:
                # A copy needs a human continue-vs-fork decision; silently
                # continuing would share (and possibly bill against) the
                # original project's cloud namespace.
                await self._emit(render_error(
                    "📁 This directory is a COPY of another project (or its "
                    "original location cannot be verified), and --auto "
                    "cannot decide continue-vs-fork on its own. Run lqh "
                    "interactively once to choose, then re-run --auto."
                ))
                await self._stop_update_check()
                self._exit_application()
                await self._wait_for_app_task(app_task)
                self._save_session()
                if self._session is not None:
                    self._session.mark_state("completed")
                await self._finish_telemetry()
                return
            await self._emit(render_system_message(
                "📁 This directory looks like a COPY of another project "
                "(both share one identity), or its recorded location "
                "cannot be verified. Continue as the same cloud project, "
                "or fork into a new one?"
            ))
            choice = await self._wait_for_user_response(options=[
                "Continue as the same project (shares cloud history/jobs)",
                "Fork into a new project (fresh identity and cloud namespace)",
            ])
            try:
                if choice.startswith("Fork"):
                    fork_identity(self.project_dir)
                    await self._emit(render_system_message(
                        "🔀 Forked: new project identity minted "
                        "(forked_from recorded; inherited cloud job "
                        "markers were detached as *.pre-fork; the "
                        "original keeps its cloud history)."
                    ))
                else:
                    record_continue_decision(self.project_dir)
            except Exception as exc:
                # An unrecorded decision must STOP the session — running
                # on anyway would observe (and possibly bill against)
                # another project's cloud state, exactly what the prompt
                # exists to prevent. A partially detached fork is safe
                # to retry: the prompt reappears next start and already-
                # renamed markers are skipped.
                await self._emit(render_error(
                    f"Could not record the copy decision: {type(exc).__name__}: "
                    f"{exc}\nStopping. Fix the .lqh/ directory (permissions/"
                    "disk) and restart — you will be asked again."
                ))
                await self._stop_update_check()
                self._exit_application()
                await self._wait_for_app_task(app_task)
                self._save_session()
                if self._session is not None:
                    self._session.mark_state("completed")
                await self._finish_telemetry()
                return
        if not identity_error and get_token():
            try:
                from lqh.project_identity import migrate_cloud_identity

                await migrate_cloud_identity(self.project_dir)
            except Exception as exc:
                # migrate_cloud_identity defers ordinary failures itself;
                # anything reaching here is identity-level and worth seeing.
                await self._emit(render_error(
                    f"Cloud identity migration failed: {type(exc).__name__}: {exc}"
                ))

        await self._refresh_startup_state()

        # Auto mode: skip the interactive welcome / resume flow and run the
        # pipeline non-interactively. The agent's auto skill (sticky system
        # message) drives the pipeline; we only inject SPEC.md as the kickoff
        # user message and exit when exit_auto_mode is called.
        if self.auto_mode:
            try:
                # Auto mode gets the same ephemeral project context
                # (NOTES.md, summary, signals) — just without the
                # interactive announcements or resume offer.
                if self._agent:
                    try:
                        await self._agent.prepare_context()
                    except Exception:
                        pass
                await self._run_auto_mode()
            finally:
                await self._stop_update_check()
                self._exit_application()
                await self._wait_for_app_task(app_task)
                self._save_session()
                if self._session is not None:
                    self._session.mark_state("completed")
                await self._finish_telemetry()
            return

        await self._offer_interrupted_resume()
        await self._prepare_agent_context()

        self._job_watcher_task = asyncio.create_task(self._watch_jobs())

        try:
            while True:
                # The app stays mounted at the bottom; submitted input comes back via this queue.
                input_task = asyncio.create_task(self._input_queue.get())
                done, pending = await asyncio.wait(
                    {app_task, input_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if app_task in done:
                    await self._wait_for_app_task(app_task)
                    if self._shutdown_requested:
                        input_task.cancel()
                        break

                    # Some prompt_toolkit paths can end the non-fullscreen app early.
                    app_task = self._start_application_task()
                    await asyncio.sleep(0)

                    if input_task not in done:
                        input_task.cancel()
                        continue

                text = input_task.result()
                if text == _SHUTDOWN_SENTINEL:
                    break
                should_continue = await self._handle_input(text)
                if not should_continue:
                    break

                for task in pending:
                    task.cancel()
        finally:
            await self._stop_update_check()
            if self._progress_refresh_task is not None:
                self._progress_refresh_task.cancel()
                self._progress_refresh_task = None
            if self._job_watcher_task is not None:
                self._job_watcher_task.cancel()
                try:
                    await self._job_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._job_watcher_task = None
            for watcher in list(self._run_watchers.values()):
                try:
                    await watcher.stop()
                except Exception:
                    pass
            self._run_watchers.clear()
            self._exit_application()
            await self._wait_for_app_task(app_task)
            self._save_session()
            if self._session is not None:
                self._session.mark_state("completed")
            await self._finish_telemetry()

    async def _telemetry_heartbeat(self) -> None:
        next_heartbeat = time.monotonic() + TELEMETRY_HEARTBEAT_INTERVAL_SEC
        try:
            while True:
                await asyncio.sleep(TELEMETRY_FLUSH_INTERVAL_SEC)
                # Heartbeats carry the cumulative capped interaction time. The
                # backend converts cumulative snapshots into exactly-once
                # deltas, preserving activity if the process disappears before
                # a final session_ended event.
                now = time.monotonic()
                if now >= next_heartbeat:
                    await self._telemetry.run_deferred(self._telemetry.heartbeat)
                    # A timed-out waiter leaves the heartbeat queued. Persist
                    # the latest completed state either way; a later heartbeat
                    # or shutdown checkpoint captures any still-pending delta.
                    self._checkpoint_telemetry_jobs()
                    next_heartbeat = now + TELEMETRY_HEARTBEAT_INTERVAL_SEC
                # Drain queued telemetry even during an idle open session; a
                # heartbeat is intentionally omitted when activity is unchanged.
                await self._telemetry.flush()
        except asyncio.CancelledError:
            return

    def _start_telemetry_flush(self) -> None:
        """Retain fire-and-forget flushes until they finish."""
        task = asyncio.create_task(self._telemetry.flush())
        self._telemetry_flush_tasks.add(task)
        task.add_done_callback(self._telemetry_flush_tasks.discard)

    async def _finish_telemetry(self) -> None:
        task = self._telemetry_heartbeat_task
        if task is None:
            return
        self._telemetry_heartbeat_task = None
        task.cancel()
        pending = list(self._telemetry_flush_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._checkpoint_telemetry_jobs()
        await self._telemetry.run_deferred(
            self._telemetry.end_session,
            "cancelled" if self._interrupt_requested else "succeeded",
        )
        await self._telemetry.flush()
        set_active_telemetry(None)

    async def _show_update_notice(self) -> None:
        """Emit a non-blocking update hint when PyPI has a newer release."""
        update = await check_for_update()
        if update is None:
            return
        await self._emit(
            render_system_message(
                f"⬆️ lqh {update.latest} is available "
                f"(installed: {update.current}). Upgrade with: pip install -U lqh"
            )
        )

    async def _stop_update_check(self) -> None:
        """Finish or cancel the optional startup update task cleanly."""
        task = self._update_check_task
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        self._update_check_task = None
