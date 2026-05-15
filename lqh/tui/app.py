"""Bottom-docked terminal UI for lqh."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from lqh.session import Session
from lqh.tui.commands import COMMANDS, SlashCommandCompleter, is_command, parse_command
from lqh.tui.dataset_viewer import DatasetViewer
from lqh.tui.renderer import (
    render_agent_message,
    render_error,
    render_file_view,
    render_options,
    render_system_message,
    render_tool_call,
    render_tool_result,
    render_user_message,
    render_welcome,
)
from lqh.tui.background_tasks import BackgroundTask, BackgroundTaskRegistry
from lqh.tui.status_bar import StatusBar

if TYPE_CHECKING:
    from lqh.subprocess_manager import SubprocessManager


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
})

OTHER_OPTION = "Other (please specify)"

# Interval between background-job completion scans, in seconds.
# Trades freshness against SSH/filesystem load when remote runs are active.
JOB_POLL_INTERVAL_SEC = 60.0


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
        self._dataset_viewer: DatasetViewer | None = None
        self._dataset_viewer_future: asyncio.Future[str] | None = None
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._session: Session | None = None
        self._agent: Agent | None = None
        self._spinner_task: asyncio.Task | None = None
        self._app: Application | None = None
        # Background job watcher state: maps run_name → last observed state.
        # First-time observations are recorded silently; only running → terminal
        # transitions enqueue a notification.
        self._job_watcher_task: asyncio.Task | None = None
        self._job_last_state: dict[str, str] = {}
        # Per-run scoring/sync watchers (RunWatcher / RemoteRunWatcher),
        # keyed by run_name. Spawned lazily by _watch_jobs when a run is
        # observed in the running state and culled when they finish.
        self._run_watchers: dict[str, Any] = {}
        # Display-state registry for the status bar — covers any background
        # work that will later notify the agent (currently fed by _watch_jobs,
        # but exposed via self._tasks for future producers).
        self._tasks = BackgroundTaskRegistry(on_change=self._invalidate)
        self._ctrl_c_pressed = False
        self._input_buffer: Buffer | None = None
        self._managed_ansi = ""
        self._shutdown_requested = False

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
            completer=SlashCommandCompleter(),
            multiline=False,
            accept_handler=self._on_accept,
        )
        input_window = Window(
            content=BufferControl(
                buffer=self._input_buffer,
                focusable=True,
            ),
            height=Dimension(min=1, max=6, preferred=1),
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

        # In auto mode the input field is hidden — the agent runs without
        # user input and the managed window is the only visible surface
        # besides the status bar.
        not_auto = Condition(lambda: not self.auto_mode)
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
        def _newline(event):
            """Insert newline on Alt+Enter."""
            event.app.current_buffer.insert_text("\n")

        @kb.add("c-c", eager=True)
        def _interrupt(event):
            """Ctrl+C once warns, twice exits."""
            if self._ctrl_c_pressed:
                self._save_session()
                self._request_shutdown()
                return

            self._ctrl_c_pressed = True
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
                self._render_ask_user_options()
                event.app.invalidate()

        @kb.add("down", filter=is_ask_mode, eager=True)
        def _ask_down(event):
            if self._ask_user_options:
                self._ask_user_selected = min(
                    len(self._ask_user_options) - 1,
                    self._ask_user_selected + 1,
                )
                self._render_ask_user_options()
                event.app.invalidate()

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

    def _get_status_text(self) -> FormattedText:
        """Render the status bar, with mode hints when applicable."""
        self._status_bar.bg_tasks = self._tasks.snapshot()
        parts = list(self._status_bar.get_formatted_text())

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
        if self._app and self._app.is_running:
            self._app.exit()

    def _request_shutdown(self) -> None:
        """Mark the session for shutdown and stop the live application."""
        self._shutdown_requested = True
        self._exit_application()

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
                self._emit(render_system_message("⏳ Please wait for the current operation to finish."))
            )
            return False

        buff.reset()

        if self._ask_user_future is not None:
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
                        render_system_message("Type additional items (comma-separated):", separated=False)
                    )
                    self._invalidate()
                    return

                # Collect all checked options
                checked_names = [
                    self._ask_user_options[i]
                    for i in sorted(self._ask_user_checked)
                    if i < len(self._ask_user_options) and self._ask_user_options[i] != OTHER_OPTION
                ]
                response = ", ".join(checked_names) if checked_names else "(none selected)"
            else:
                # Single-select
                selected = self._ask_user_options[self._ask_user_selected]
                if self._ask_user_allow_other and selected == OTHER_OPTION:
                    self._ask_user_options = None
                    self._ask_user_allow_other = False
                    self._set_managed_text(
                        render_system_message("Type your response:", separated=False)
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

        if command == "/quit":
            self._save_session()
            self._request_shutdown()
            return False

        if command == "/help":
            lines = ["**Available Commands:**\n"]
            for cmd in COMMANDS:
                lines.append(f"  `{cmd.name}` - {cmd.description}")
            await self._emit(render_system_message("\n".join(lines)))
            return True

        if command == "/login":
            await self._do_login()
            return True

        if command == "/clear":
            self._save_session()
            self._session = Session.create(self.project_dir)
            self._status_bar.session_id = self._session.id
            self._status_bar.prompt_tokens = 0
            self._status_bar.completion_tokens = 0
            self._status_bar.active_skill = ""
            self._agent = self._create_agent()
            await self._emit(render_system_message("Started new session."))
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
                    await self._agent.process_user_input(
                        f"[System: The {skill_name} skill is now active. "
                        f"Begin the workflow described in the skill instructions.]"
                    )
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

        self._lock_input()
        await self._emit(render_user_message(text))

        try:
            if self._agent:
                await self._agent.process_user_input(text)
        except Exception as e:
            await self._emit(render_error(f"{type(e).__name__}: {e}"))
        finally:
            self._unlock_input()
            self._save_session()

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
        except LoginExpired:
            await self._emit(render_error("Device code expired. Run /login again."))
        except Exception as e:
            await self._emit(render_error(f"Login failed: {type(e).__name__}: {e}"))

    async def _do_resume(self) -> None:
        """Handle the /resume command."""
        sessions = Session.list_sessions(self.project_dir)
        if not sessions:
            await self._emit(render_system_message("No previous sessions found."))
            return

        options = []
        for session_info in sessions[:10]:
            preview = session_info.get("preview", "(empty)")[:60]
            timestamp = session_info.get("created_at", "?")
            options.append(f"{timestamp} - {preview}")

        await self._emit(render_system_message("Select a session to resume:"))
        selected = await self._wait_for_user_response(options=options)

        index = 0
        for i, option in enumerate(options):
            if option == selected or selected in option:
                index = i
                break

        session_info = sessions[index]
        self._save_session()
        self._session = Session.load(self.project_dir, session_info["id"])
        self._status_bar.session_id = self._session.id
        self._status_bar.prompt_tokens = self._session.prompt_tokens
        self._status_bar.completion_tokens = self._session.completion_tokens
        self._agent = self._create_agent()
        self._invalidate()

        await self._emit(render_system_message(f"Resumed session {session_info['id'][:8]}"))
        for message in self._session.messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user" and isinstance(content, str) and not content.startswith("[System:"):
                await self._emit(render_user_message(content))
            elif role == "assistant" and content:
                await self._emit(render_agent_message(str(content)))

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
            on_spinner_start=self._on_spinner_start,
            on_spinner_stop=self._on_spinner_stop,
            on_token_update=self._on_token_update,
            on_skill_loaded=self._on_skill_loaded,
            on_pipeline_progress=self._on_pipeline_progress,
            on_pipeline_done=self._on_pipeline_done,
            on_background_task_started=self._on_background_task_started,
            on_auto_stage=self._on_auto_stage,
        )
        return Agent(
            self.project_dir,
            self._session,
            callbacks,
            auto_mode=self.auto_mode,
            extra_spec=self.extra_spec,
        )

    def _on_auto_stage(self, stage: str, note: str | None) -> None:
        """Auto-mode: agent reported a stage transition."""
        self._auto_stage = stage
        self._auto_stage_note = note
        line = f"• {stage}" + (f" — {note}" if note else "")
        # Only append if it's a new stage (avoid duplicates from chatty agents)
        if not self._auto_history or self._auto_history[-1] != line:
            self._auto_history.append(line)
        self._invalidate()

    def _ensure_task_registered(self, run_name: str, remote: str | None) -> None:
        """Register a task for a live run if not already in the registry.

        Used by ``_watch_jobs`` to recover the indicator after a TUI
        restart, since handler-driven eager registration only fires on
        the original submission.
        """
        if any(t.task_id == run_name for t in self._tasks.snapshot()):
            return
        config_path = self.project_dir / "runs" / run_name / "config.json"
        kind = "train"
        if config_path.exists():
            try:
                import json
                run_type = json.loads(config_path.read_text()).get("type", "")
                kind = "eval" if run_type == "infer" else "train"
            except Exception:
                pass
        self._tasks.register(BackgroundTask(
            task_id=run_name,
            kind=kind,
            label=run_name,
            state="running",
            remote=remote,
        ))

    def _on_background_task_started(
        self, task_id: str, kind: str, label: str, remote: str | None,
    ) -> None:
        """Handler hook: a tool just submitted a job that will notify later."""
        self._tasks.register(BackgroundTask(
            task_id=task_id,
            kind=kind,
            label=label,
            state="running",
            remote=remote,
        ))

    async def _on_agent_message(self, text: str) -> None:
        await self._emit(render_agent_message(text))

    async def _on_tool_call(self, name: str, args: dict) -> None:
        await self._emit(render_tool_call(name, args))

    async def _on_tool_result(self, name: str, content: str) -> None:
        await self._emit(render_tool_result(name, content))

    async def _on_ask_user(self, question: str, options: list[str] | None, multi_select: bool = False) -> str:
        """Handle ask_user tool requests."""
        await self._emit(render_agent_message(f"❓ {question}"))

        if options:
            filtered = [
                option
                for option in options
                if "other" not in option.lower() or "please specify" not in option.lower()
            ]
            all_options = filtered + [OTHER_OPTION]
            response = await self._wait_for_user_response(
                options=all_options,
                allow_other=True,
                multi_select=multi_select,
                relock_after=True,
            )
            return response

        return await self._wait_for_user_response(
            managed_text=render_system_message("Type your response:", separated=False),
            relock_after=True,
        )

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
                while self._status_bar.spinning or self._status_bar.pipeline_status:
                    self._status_bar.advance_spinner()
                    self._invalidate()
                    await asyncio.sleep(0.08)
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

    def _on_pipeline_progress(self, completed: int, total: int, concurrency: int) -> None:
        """Update pipeline progress in the status bar."""
        self._status_bar.pipeline_status = (
            f"🚀 {completed}/{total} samples (concurrency {concurrency})"
        )
        if not self._status_bar.spinning:
            self._status_bar.start_spinning()
        self._ensure_spinner_task()
        self._invalidate()

    def _on_pipeline_done(self) -> None:
        """Clear pipeline progress from the status bar."""
        self._status_bar.pipeline_status = ""
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
            await self._agent.process_user_input(kickoff)
        except Exception as e:
            await self._emit(render_error(
                f"Auto mode crashed: {type(e).__name__}: {e}"
            ))
            self._auto_done = True
            self._invalidate()
            return
        finally:
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

    async def _watch_jobs(self) -> None:
        """Periodically scan background runs and inject completion notifications.

        Detects ``running → completed/failed`` transitions per run and pushes
        a ``[System: ...]`` message into the input queue, where it is picked
        up by the main loop just like typed input. The first observation of
        any run is silent — already-terminal runs at startup don't fire.
        """
        from lqh.subprocess_manager import SubprocessManager

        manager = SubprocessManager()
        terminal_states = {"completed", "failed"}

        while True:
            try:
                await asyncio.sleep(JOB_POLL_INTERVAL_SEC)
            except asyncio.CancelledError:
                return

            try:
                snapshots = await self._scan_jobs(manager)
            except Exception:
                # Never let scan errors kill the watcher.
                continue

            # Cull watchers whose runs have finished.
            for name in [n for n, w in self._run_watchers.items() if not w.is_running]:
                self._run_watchers.pop(name, None)
                self._tasks.unregister(name)

            for run_name, state, error, remote in snapshots:
                if state == "unknown":
                    # Transient SSH/FS hiccup — don't update last_state, retry next tick.
                    continue
                prev = self._job_last_state.get(run_name)
                if prev == "running" and state in terminal_states:
                    text = self._format_completion_message(
                        run_name, state, error, remote,
                    )
                    self._input_queue.put_nowait(text)
                    self._record_completion_event(run_name, state, error, remote)
                    self._tasks.unregister(run_name)
                # Keep the registry in sync with live state. The handler
                # eagerly registers on submission; this branch is the
                # fallback for jobs discovered after a TUI restart.
                if state == "running":
                    self._ensure_task_registered(run_name, remote)
                elif state in terminal_states:
                    self._tasks.unregister(run_name)
                self._job_last_state[run_name] = state

                # Ensure a scoring/sync watcher is attached to runs that may
                # still need work: live runs (rsync + score during run) AND
                # finished runs that don't yet have eval_result.json (handles
                # fast remote inferences that finish before our scan tick,
                # and completed-but-unscored runs after a TUI restart).
                if run_name in self._run_watchers:
                    continue
                if state == "failed":
                    continue
                run_dir = self.project_dir / "runs" / run_name
                if state == "completed" and (run_dir / "eval_result.json").exists():
                    continue
                try:
                    await self._spawn_run_watcher(run_name, remote)
                except Exception:
                    pass

    async def _scan_jobs(
        self, manager: "SubprocessManager",
    ) -> list[tuple[str, str, str | None, str | None]]:
        """Return ``(run_name, state, error, remote_name)`` for every run dir.

        Local runs are queried via ``SubprocessManager.get_status`` (PID +
        progress.jsonl). Remote runs (``remote_job.json`` present) are
        probed over SSH via the backend's ``poll_status``.
        """
        import json

        runs_dir = self.project_dir / "runs"
        if not runs_dir.is_dir():
            return []

        results: list[tuple[str, str, str | None, str | None]] = []
        for entry in sorted(runs_dir.iterdir()):
            if not entry.is_dir() or not (entry / "config.json").exists():
                continue

            remote_meta = entry / "remote_job.json"
            if remote_meta.exists():
                try:
                    meta = json.loads(remote_meta.read_text())
                    state, error = await self._poll_remote(
                        meta["remote_name"], str(meta["job_id"]),
                    )
                    results.append((entry.name, state, error, meta["remote_name"]))
                except Exception:
                    results.append((entry.name, "unknown", None, None))
                continue

            status = manager.get_status(entry)
            results.append((entry.name, status.state, status.error, None))

        return results

    async def _poll_remote(
        self, remote_name: str, job_id: str,
    ) -> tuple[str, str | None]:
        """SSH-poll a remote job. Returns (state, error)."""
        from lqh.remote.config import get_remote
        from lqh.remote.ssh_direct import SSHDirectBackend

        remote_config = get_remote(self.project_dir, remote_name)
        if remote_config is None or remote_config.type != "ssh_direct":
            return ("unknown", None)
        backend = SSHDirectBackend(remote_config, self.project_dir)
        status = await backend.poll_status(job_id)
        return (status.state, status.error)

    async def _spawn_run_watcher(self, run_name: str, remote: str | None) -> None:
        """Start a RunWatcher (or RemoteRunWatcher) for an active run.

        The watcher rsyncs predictions back from the remote (when applicable),
        invokes the API judge over predictions.parquet, writes eval_result.json,
        and self-stops when the run reaches a terminal state.
        """
        import json

        from lqh.auth import get_token
        from lqh.config import load_config

        run_dir = self.project_dir / "runs" / run_name
        config_path = run_dir / "config.json"
        if not config_path.exists():
            return
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            return

        api_key = get_token() or ""
        api_base_url = load_config().api_base_url

        if remote:
            from lqh.remote.config import get_remote
            from lqh.remote.ssh_direct import SSHDirectBackend
            from lqh.remote.watcher import RemoteRunWatcher

            remote_config = get_remote(self.project_dir, remote)
            if remote_config is None or remote_config.type != "ssh_direct":
                return

            meta_file = run_dir / "remote_job.json"
            if not meta_file.exists():
                return
            meta = json.loads(meta_file.read_text())

            backend = SSHDirectBackend(remote_config, self.project_dir)
            watcher = RemoteRunWatcher(
                run_dir=run_dir,
                config=config,
                project_dir=self.project_dir,
                api_key=api_key,
                api_base_url=api_base_url,
                backend=backend,
                remote_run_dir=meta["remote_run_dir"],
                job_id=str(meta["job_id"]),
            )
        else:
            from lqh.watcher import RunWatcher

            watcher = RunWatcher(
                run_dir=run_dir,
                config=config,
                project_dir=self.project_dir,
                api_key=api_key,
                api_base_url=api_base_url,
            )

        await watcher.start()
        self._run_watchers[run_name] = watcher

    def _format_completion_message(
        self, run_name: str, state: str, error: str | None, remote: str | None,
    ) -> str:
        location = f" on remote '{remote}'" if remote else ""
        if state == "completed":
            return f"[System: training run {run_name} completed successfully{location}.]"
        err_part = f": {error}" if error else "."
        return f"[System: training run {run_name} failed{location}{err_part}]"

    def _record_completion_event(
        self, run_name: str, state: str, error: str | None, remote: str | None,
    ) -> None:
        """Mirror the transition in the project log so summary/restart see it."""
        from lqh.project_log import append_event

        event = "training_completed" if state == "completed" else "training_failed"
        if state == "completed":
            desc = f"Run {run_name} completed"
            if remote:
                desc += f" on remote '{remote}'"
        else:
            desc = f"Run {run_name} failed"
            if remote:
                desc += f" on remote '{remote}'"
            if error:
                desc += f": {error}"
        kwargs: dict = {"run_name": run_name}
        if remote:
            kwargs["remote"] = remote
        if error:
            kwargs["error"] = error
        append_event(self.project_dir, event, desc, **kwargs)

    async def run(self) -> None:
        """Run the persistent bottom application."""
        self._shutdown_requested = False
        self._session = Session.create(self.project_dir)
        self._status_bar.session_id = self._session.id
        self._agent = self._create_agent()
        app_task = self._start_application_task()
        await asyncio.sleep(0)

        await self._emit(render_welcome())

        token = get_token()
        self._status_bar.logged_in = bool(token)
        self._invalidate()
        if token:
            await self._emit(render_system_message("✅ Logged in to lqh.ai"))
        else:
            await self._emit(render_system_message("⚠️ Not logged in. Run /login to authenticate."))

        # Auto mode: skip the interactive welcome / resume flow and run the
        # pipeline non-interactively. The agent's auto skill (sticky system
        # message) drives the pipeline; we only inject SPEC.md as the kickoff
        # user message and exit when exit_auto_mode is called.
        if self.auto_mode:
            try:
                await self._run_auto_mode()
            finally:
                self._exit_application()
                await self._wait_for_app_task(app_task)
                self._save_session()
            return

        if self._agent:
            mode = await self._agent.prepare_context()
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
                    render_system_message("📂 Loaded SPEC.md and project summary into context. Ready to go.")
                )

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
                should_continue = await self._handle_input(text)
                if not should_continue:
                    break

                for task in pending:
                    task.cancel()
        finally:
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
