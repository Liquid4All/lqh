"""Tests for user interruption of the agent (Esc / Ctrl+C) and auto-mode pause."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lqh.agent import Agent
from lqh.session import Session
from lqh.tui.app import (
    _SHUTDOWN_SENTINEL,
    AgentInterrupted,
    LqhApp,
)

CTRL_C = "\x03"
ESC = "\x1b"


@pytest.fixture
def app() -> LqhApp:
    return LqhApp(Path("."))


def _make_agent(tmp_path: Path) -> Agent:
    return Agent(tmp_path, Session.create(tmp_path))


class TestAbortTurn:
    """Session repair after a cancel lands mid-turn."""

    def test_fills_unanswered_tool_calls(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.session.add_message({"role": "user", "content": "go"})
        agent.session.add_message({
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "b", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}},
            ],
        })
        agent.session.add_message({"role": "tool", "tool_call_id": "a", "content": "ok"})

        agent.abort_turn()

        tail = agent.session.messages[-1]
        assert tail["role"] == "tool"
        assert tail["tool_call_id"] == "b"
        assert "Interrupted" in tail["content"]
        # Exactly one synthetic result — "a" was already answered.
        assert len(agent.session.messages) == 4

    def test_noop_when_no_dangling_calls(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.session.add_message({"role": "user", "content": "go"})
        agent.session.add_message({"role": "assistant", "content": "done"})

        agent.abort_turn()
        assert len(agent.session.messages) == 2

    def test_noop_when_last_message_is_user(self, tmp_path: Path) -> None:
        """Cancel landed during the LLM call — nothing to repair."""
        agent = _make_agent(tmp_path)
        agent.session.add_message({"role": "user", "content": "go"})

        agent.abort_turn()
        assert len(agent.session.messages) == 1

    def test_repairs_through_mid_turn_system_injection(self, tmp_path: Path) -> None:
        """A skill load appends a system message after its tool result; a cancel
        during a later tool call of the same turn must still repair the rest."""
        agent = _make_agent(tmp_path)
        agent.session.add_message({
            "role": "assistant",
            "tool_calls": [
                {"id": "skill", "type": "function",
                 "function": {"name": "load_skill", "arguments": "{}"}},
                {"id": "pending", "type": "function",
                 "function": {"name": "run_scoring", "arguments": "{}"}},
            ],
        })
        agent.session.add_message({"role": "tool", "tool_call_id": "skill", "content": "loaded"})
        agent.session.add_message({"role": "system", "content": "skill instructions"})

        agent.abort_turn()

        tail = agent.session.messages[-1]
        assert tail["role"] == "tool"
        assert tail["tool_call_id"] == "pending"

    def test_does_not_touch_earlier_answered_turns(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.session.add_message({
            "role": "assistant",
            "tool_calls": [{"id": "old", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        })
        agent.session.add_message({"role": "tool", "tool_call_id": "old", "content": "ok"})
        agent.session.add_message({"role": "user", "content": "next"})

        agent.abort_turn()
        assert len(agent.session.messages) == 3


class TestProtectedSubmissions:
    """Interrupts must not sever an in-flight job submission / key mint."""

    async def test_interrupt_defers_until_submission_finishes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from lqh.tools.handlers import ToolResult

        agent = _make_agent(tmp_path)
        started = asyncio.Event()
        finished = asyncio.Event()

        async def fake_execute(tool_name, arguments, project_dir, **extra):
            started.set()
            await asyncio.sleep(0.2)
            finished.set()
            return ToolResult(content="job submitted: run_abc")

        monkeypatch.setattr("lqh.agent.execute_tool", fake_execute)

        task = asyncio.create_task(
            agent._execute_shielded("start_training", {}, {})
        )
        await started.wait()
        task.cancel()

        result = await task  # completes despite the cancel
        assert finished.is_set()
        assert result.content == "job submitted: run_abc"
        assert agent._deferred_interrupt is True

    async def test_second_interrupt_force_cancels_submission(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from lqh.tools.handlers import ToolResult

        agent = _make_agent(tmp_path)
        finished = asyncio.Event()

        async def fake_execute(tool_name, arguments, project_dir, **extra):
            await asyncio.sleep(60)
            finished.set()  # pragma: no cover
            return ToolResult(content="never")

        monkeypatch.setattr("lqh.agent.execute_tool", fake_execute)

        task = asyncio.create_task(
            agent._execute_shielded("start_training", {}, {})
        )
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()

    async def test_loop_records_submission_before_interrupt_unwinds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end through the agent loop: the deferred interrupt is
        re-delivered only after the submission's tool result is in the session,
        and no further LLM call is made."""
        from types import SimpleNamespace

        from lqh.tools.handlers import ToolResult

        agent = _make_agent(tmp_path)
        agent._client = object()  # bypass login
        llm_calls: list[int] = []

        async def fake_chat(client, **kwargs):
            llm_calls.append(1)
            message = SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="t1",
                    function=SimpleNamespace(
                        name="start_training", arguments="{}",
                    ),
                )],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        async def fake_execute(tool_name, arguments, project_dir, **extra):
            await asyncio.sleep(0.2)
            return ToolResult(content="job submitted: run_abc")

        monkeypatch.setattr("lqh.agent.chat_with_retry", fake_chat)
        monkeypatch.setattr("lqh.agent.execute_tool", fake_execute)

        task = asyncio.create_task(agent.process_user_input("train it"))
        await asyncio.sleep(0.1)  # inside the submission sleep
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        tail = agent.session.messages[-1]
        assert tail == {
            "role": "tool", "tool_call_id": "t1",
            "content": "job submitted: run_abc",
        }
        assert llm_calls == [1]


class TestRunInterruptible:
    """The cancellable wrapper around agent turns."""

    async def test_interrupt_raises_and_repairs(self, app: LqhApp, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.session.add_message({
            "role": "assistant",
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        })
        app._agent = agent

        async def hang() -> None:
            await asyncio.sleep(60)

        run = asyncio.create_task(app._run_interruptible(hang))
        await asyncio.sleep(0.05)

        assert app._agent_busy()
        assert app._request_agent_interrupt() is True

        with pytest.raises(AgentInterrupted):
            await run

        assert not app._agent_busy()
        assert app._agent_task is None
        # Session was repaired.
        assert agent.session.messages[-1]["role"] == "tool"
        assert agent.session.messages[-1]["tool_call_id"] == "t1"

    async def test_external_cancel_propagates(self, app: LqhApp) -> None:
        """A cancel that is NOT a user interrupt must stay a CancelledError."""

        async def hang() -> None:
            await asyncio.sleep(60)

        run = asyncio.create_task(app._run_interruptible(hang))
        await asyncio.sleep(0.05)
        run.cancel()

        with pytest.raises(asyncio.CancelledError):
            await run
        assert app._agent_task is None

    async def test_result_passes_through(self, app: LqhApp) -> None:
        async def finish() -> str:
            return "done"

        assert await app._run_interruptible(finish) == "done"
        assert app._agent_task is None

    def test_interrupt_request_without_agent_is_noop(self, app: LqhApp) -> None:
        assert app._request_agent_interrupt() is False


async def _drive_keys(app: LqhApp, steps: list[tuple[str, float]]) -> None:
    """Feed key sequences through the real prompt_toolkit key processor.

    ``steps`` is a list of ``(text, wait_seconds)`` pairs; the wait after each
    chunk lets the input parser flush (a lone ESC needs the escape timeout).
    """
    application = app._create_application()
    app._app = application

    with create_pipe_input() as pipe:
        application.input = pipe
        application.output = DummyOutput()

        async def drive() -> None:
            await asyncio.sleep(0.1)
            for text, wait in steps:
                pipe.send_text(text)
                await asyncio.sleep(wait)
            if application.is_running:
                application.exit()

        task = asyncio.create_task(drive())
        await application.run_async()
        await task


class TestCtrlCAndEsc:
    async def test_single_ctrl_c_interrupts_busy_agent(self, app: LqhApp) -> None:
        interrupted = asyncio.Event()

        async def hang() -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                interrupted.set()
                raise

        run = asyncio.create_task(app._run_interruptible(hang))
        await asyncio.sleep(0.05)

        await _drive_keys(app, [(CTRL_C, 0.1)])

        # Cancellation delivery is asynchronous — wait for it rather than
        # asserting on a fixed sleep (flaky under full-suite load).
        await asyncio.wait_for(interrupted.wait(), timeout=2)
        assert not app._shutdown_requested
        with pytest.raises(AgentInterrupted):
            await asyncio.wait_for(run, timeout=2)

    async def test_double_ctrl_c_exits(self, app: LqhApp) -> None:
        await _drive_keys(app, [(CTRL_C, 0.1), (CTRL_C, 0.1)])
        assert app._shutdown_requested

    async def test_single_ctrl_c_idle_only_warns(self, app: LqhApp) -> None:
        await _drive_keys(app, [(CTRL_C, 0.1)])
        assert not app._shutdown_requested
        assert app._ctrl_c_pressed

    async def test_stale_first_ctrl_c_does_not_exit(self, app: LqhApp) -> None:
        """A first press outside the window must not make the second one exit."""
        import lqh.tui.app as app_mod

        original = app_mod.CTRL_C_EXIT_WINDOW_SEC
        app_mod.CTRL_C_EXIT_WINDOW_SEC = 0.05
        try:
            await _drive_keys(app, [(CTRL_C, 0.3), (CTRL_C, 0.1)])
            assert not app._shutdown_requested
        finally:
            app_mod.CTRL_C_EXIT_WINDOW_SEC = original

    async def test_esc_interrupts_busy_agent(self, app: LqhApp) -> None:
        async def hang() -> None:
            await asyncio.sleep(60)

        run = asyncio.create_task(app._run_interruptible(hang))
        await asyncio.sleep(0.05)

        # A lone ESC is only flushed after prompt_toolkit's escape timeout.
        await _drive_keys(app, [(ESC, 0.9)])

        with pytest.raises(AgentInterrupted):
            await asyncio.wait_for(run, timeout=2)
        assert not app._shutdown_requested

    async def test_esc_idle_does_nothing(self, app: LqhApp) -> None:
        await _drive_keys(app, [(ESC, 0.7)])
        assert not app._shutdown_requested
        assert not app._interrupt_requested


class TestAskUserCleanupOnCancel:
    async def test_cancel_clears_ask_state(self, app: LqhApp) -> None:
        task = asyncio.create_task(
            app._wait_for_user_response(options=["alpha", "beta"], multi_select=True)
        )
        await asyncio.sleep(0.05)
        assert app._ask_user_future is not None

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert app._ask_user_future is None
        assert app._ask_user_options is None
        assert app._ask_user_multi_select is False
        assert app._ask_user_checked == set()
        assert app._managed_ansi == ""


class TestAutoModePause:
    async def test_pause_returns_user_instruction(self, tmp_path: Path) -> None:
        app = LqhApp(tmp_path, auto_mode=True)

        pause = asyncio.create_task(app._pause_auto_mode())
        await asyncio.sleep(0.05)
        assert app._auto_paused
        assert not app._processing  # input unlocked while paused

        app._input_queue.put_nowait("focus on the eval set")
        assert await pause == "focus on the eval set"
        assert not app._auto_paused
        assert app._processing  # relocked for the resumed run

    async def test_pause_skips_system_notices(self, tmp_path: Path) -> None:
        app = LqhApp(tmp_path, auto_mode=True)

        pause = asyncio.create_task(app._pause_auto_mode())
        await asyncio.sleep(0.05)
        app._input_queue.put_nowait("[System: training run x completed.]")
        app._input_queue.put_nowait("real instruction")
        assert await pause == "real instruction"

    async def test_pause_shutdown_sentinel_returns_none(self, tmp_path: Path) -> None:
        app = LqhApp(tmp_path, auto_mode=True)

        pause = asyncio.create_task(app._pause_auto_mode())
        await asyncio.sleep(0.05)
        app._input_queue.put_nowait(_SHUTDOWN_SENTINEL)
        assert await pause is None
        assert not app._auto_paused

    async def test_pause_quit_command_shuts_down(self, tmp_path: Path) -> None:
        app = LqhApp(tmp_path, auto_mode=True)

        pause = asyncio.create_task(app._pause_auto_mode())
        await asyncio.sleep(0.05)
        app._input_queue.put_nowait("/quit")
        assert await pause is None
        assert app._shutdown_requested

    async def test_pause_unhides_input_row(self, tmp_path: Path) -> None:
        """The auto-mode input row is hidden while running, visible while paused."""
        app = LqhApp(tmp_path, auto_mode=True)
        application = app._create_application()
        hidden_input_row = application.layout.container.children[2]

        assert not hidden_input_row.filter()
        app._auto_paused = True
        assert hidden_input_row.filter()

    async def test_auto_mode_resumes_with_instruction(self, tmp_path: Path) -> None:
        """Interrupt an auto run, inject an instruction, and verify it resumes."""
        (tmp_path / "SPEC.md").write_text("# spec\n")
        app = LqhApp(tmp_path, auto_mode=True)

        calls: list[str] = []
        resumed = asyncio.Event()

        class FakeAgent:
            _auto_exit = ("success", "done")

            async def process_user_input(self, text: str) -> None:
                calls.append(text)
                if len(calls) == 1:
                    await asyncio.sleep(60)  # first turn hangs until interrupted
                resumed.set()

            async def continue_after_interruption(self) -> None:  # pragma: no cover
                pass

            def abort_turn(self) -> None:
                pass

        app._agent = FakeAgent()  # type: ignore[assignment]

        run = asyncio.create_task(app._run_auto_mode())
        await asyncio.sleep(0.1)

        assert app._request_agent_interrupt() is True
        await asyncio.sleep(0.05)
        assert app._auto_paused

        app._input_queue.put_nowait("switch to the 2.6B model")
        await asyncio.wait_for(resumed.wait(), timeout=2)
        await asyncio.wait_for(run, timeout=2)

        assert len(calls) == 2
        assert "switch to the 2.6B model" in calls[1]
        assert "[User interruption]" in calls[1]
        assert not app._auto_paused
        assert app._auto_done
