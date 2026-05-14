"""Unit + integration tests for auto mode (no API access required).

Covers the auto-mode plumbing: sticky system messages survive compaction,
ask_user is intercepted in auto mode, the no-tool-call assistant turn is
nudged with "please continue", exit_auto_mode terminates the loop, and
the auto-only tools are gated on the auto_mode flag.

The full end-to-end pipeline test (real spec → trained checkpoint) lives
in tests/e2e/test_auto_mode_e2e.py and is gated on API access.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from lqh.agent import Agent, AgentCallbacks
from lqh.session import Session
from lqh.tools.definitions import get_all_tools
from lqh.tools.handlers import (
    ToolResult,
    execute_tool,
    handle_exit_auto_mode,
    handle_set_auto_stage,
)


def _project_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="lqh_auto_test_"))


# ---------------------------------------------------------------------------
# Tool definitions: auto-mode tools are gated on the flag
# ---------------------------------------------------------------------------


class TestAutoModeToolGating(unittest.TestCase):
    def test_default_excludes_auto_tools(self) -> None:
        names = {t["function"]["name"] for t in get_all_tools()}
        self.assertNotIn("exit_auto_mode", names)
        self.assertNotIn("set_auto_stage", names)

    def test_auto_mode_includes_auto_tools(self) -> None:
        names = {t["function"]["name"] for t in get_all_tools(auto_mode=True)}
        self.assertIn("exit_auto_mode", names)
        self.assertIn("set_auto_stage", names)


# ---------------------------------------------------------------------------
# exit_auto_mode + set_auto_stage handlers
# ---------------------------------------------------------------------------


class TestAutoModeHandlers(unittest.TestCase):
    def test_exit_auto_mode_success(self) -> None:
        result = asyncio.run(handle_exit_auto_mode(
            status="success", reason="baseline 4.0 → final 7.5",
        ))
        self.assertTrue(result.exit_auto_mode)
        self.assertEqual(result.auto_status, "success")
        self.assertIn("baseline 4.0", result.auto_reason)

    def test_exit_auto_mode_failure(self) -> None:
        result = asyncio.run(handle_exit_auto_mode(
            status="failure", reason="data pipeline cannot satisfy spec",
        ))
        self.assertTrue(result.exit_auto_mode)
        self.assertEqual(result.auto_status, "failure")

    def test_exit_auto_mode_invalid_status(self) -> None:
        result = asyncio.run(handle_exit_auto_mode(status="kaput", reason="x"))
        self.assertFalse(result.exit_auto_mode)
        self.assertIn("Error", result.content)

    def test_set_auto_stage(self) -> None:
        result = asyncio.run(handle_set_auto_stage(
            stage="sft_initial", note="2k samples, score 6.8/10",
        ))
        self.assertEqual(result.auto_stage, "sft_initial")
        self.assertEqual(result.auto_stage_note, "2k samples, score 6.8/10")

    def test_set_auto_stage_empty(self) -> None:
        result = asyncio.run(handle_set_auto_stage(stage="", note=None))
        self.assertIsNone(result.auto_stage)
        self.assertIn("Error", result.content)

    def test_execute_tool_dispatch_no_project_dir(self) -> None:
        # exit_auto_mode and set_auto_stage are routed without project_dir
        result = asyncio.run(execute_tool(
            "exit_auto_mode",
            {"status": "success", "reason": "done"},
            Path("/nonexistent/should/not/be/read"),
        ))
        self.assertTrue(result.exit_auto_mode)


# ---------------------------------------------------------------------------
# Agent: sticky system messages
# ---------------------------------------------------------------------------


class TestStickySystemMessages(unittest.TestCase):
    def test_auto_mode_injects_skill_as_sticky(self) -> None:
        session = Session.create(_project_dir())
        agent = Agent(session.project_dir, session, auto_mode=True)
        self.assertEqual(len(agent.sticky_system_messages), 1)
        self.assertIn("auto mode", agent.sticky_system_messages[0].lower())

    def test_extra_spec_injected_alongside_auto(self) -> None:
        session = Session.create(_project_dir())
        agent = Agent(
            session.project_dir, session,
            auto_mode=True,
            extra_spec="use the smallest base model",
        )
        self.assertEqual(len(agent.sticky_system_messages), 2)
        self.assertIn("smallest base model", agent.sticky_system_messages[1])

    def test_extra_spec_in_interactive_mode(self) -> None:
        session = Session.create(_project_dir())
        agent = Agent(
            session.project_dir, session,
            auto_mode=False,
            extra_spec="hint Y",
        )
        self.assertFalse(agent.auto_mode)
        self.assertEqual(len(agent.sticky_system_messages), 1)
        self.assertIn("hint Y", agent.sticky_system_messages[0])

    def test_no_sticky_when_neither_set(self) -> None:
        session = Session.create(_project_dir())
        agent = Agent(session.project_dir, session)
        self.assertEqual(agent.sticky_system_messages, [])

    def test_build_messages_prepends_sticky(self) -> None:
        session = Session.create(_project_dir())
        agent = Agent(
            session.project_dir, session,
            auto_mode=True, extra_spec="hint Z",
        )
        session.messages = [{"role": "user", "content": "hello"}]
        msgs = agent._build_messages()
        # Expect: SYSTEM_PROMPT, auto-skill sticky, extra_spec sticky, user msg
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "system")
        self.assertIn("auto mode", msgs[1]["content"].lower())
        self.assertEqual(msgs[2]["role"], "system")
        self.assertIn("hint Z", msgs[2]["content"])
        self.assertEqual(msgs[3]["role"], "user")

    def test_sticky_survives_simulated_compaction(self) -> None:
        """_compact_context overwrites session.messages — sticky lives outside it."""
        session = Session.create(_project_dir())
        agent = Agent(
            session.project_dir, session,
            auto_mode=True, extra_spec="hint",
        )
        # Simulate the post-compaction state: session.messages has only the
        # compaction summary + recent turns (sticky entries are not in here).
        session.messages = [
            {"role": "system", "content": "[Context compacted] short summary"},
            {"role": "user", "content": "recent user msg"},
        ]
        msgs = agent._build_messages()
        sticky_count = sum(
            1 for m in msgs[:3]
            if m["role"] == "system"
            and ("auto mode" in m["content"].lower() or "hint" in m["content"])
        )
        self.assertEqual(sticky_count, 2)


# ---------------------------------------------------------------------------
# Agent loop: ask_user interception, no-tool-call nudge, exit termination
# ---------------------------------------------------------------------------


class _StubChatCompletion:
    """Minimal stand-in for an OpenAI ChatCompletion response."""

    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[tuple[str, str, dict]] | None = None,
        finish_reason: str = "stop",
    ) -> None:
        message = MagicMock()
        message.content = content
        if tool_calls:
            tcs = []
            for tc_id, name, args in tool_calls:
                tc = MagicMock()
                tc.id = tc_id
                tc.function = MagicMock()
                tc.function.name = name
                tc.function.arguments = json.dumps(args)
                tcs.append(tc)
            message.tool_calls = tcs
        else:
            message.tool_calls = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = finish_reason
        self.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        self.usage = usage


def _patched_agent(*, auto_mode: bool, responses: list[_StubChatCompletion]) -> Agent:
    """Build an Agent whose API calls are pre-scripted."""
    session = Session.create(_project_dir())
    agent = Agent(session.project_dir, session, auto_mode=auto_mode)
    agent._client = MagicMock()  # bypass _get_client / auth

    iterator = iter(responses)

    async def fake_chat(*args: Any, **kwargs: Any) -> _StubChatCompletion:
        try:
            return next(iterator)
        except StopIteration:
            # Default: no-op turn to terminate cleanly
            return _StubChatCompletion(content="done")

    agent._chat_with_retry = fake_chat  # type: ignore[attr-defined]
    return agent


class TestAutoModeAgentBehavior(unittest.TestCase):
    """Drive the agent loop with a stubbed chat API and verify auto-mode behaviors."""

    def _run(
        self,
        *,
        auto_mode: bool,
        responses: list[_StubChatCompletion],
        callbacks: AgentCallbacks | None = None,
    ) -> Agent:
        agent = _patched_agent(auto_mode=auto_mode, responses=responses)
        if callbacks is not None:
            agent.callbacks = callbacks
        with patch("lqh.agent.chat_with_retry", side_effect=responses + [_StubChatCompletion(content="done")] * 5):
            asyncio.run(agent.process_user_input("kickoff"))
        return agent

    def test_exit_auto_mode_breaks_loop(self) -> None:
        responses = [
            _StubChatCompletion(tool_calls=[
                ("call_1", "exit_auto_mode", {"status": "success", "reason": "ok"}),
            ]),
            # The loop should NOT consume this — if it does, the test will see
            # the wrong _auto_exit value.
            _StubChatCompletion(content="should not be reached"),
        ]
        agent = self._run(auto_mode=True, responses=responses)
        self.assertIsNotNone(agent._auto_exit)
        self.assertEqual(agent._auto_exit[0], "success")

    def test_ask_user_intercepted_in_auto_mode(self) -> None:
        # ask_user followed by exit_auto_mode. The agent loop should never
        # invoke the on_ask_user callback.
        callback_called = False

        async def on_ask_user(*args: Any, **kwargs: Any) -> str:
            nonlocal callback_called
            callback_called = True
            return "should-not-be-called"

        callbacks = AgentCallbacks(on_ask_user=on_ask_user)
        responses = [
            _StubChatCompletion(tool_calls=[
                ("call_1", "ask_user", {"question": "what?"}),
            ]),
            _StubChatCompletion(tool_calls=[
                ("call_2", "exit_auto_mode", {"status": "failure", "reason": "tried to ask"}),
            ]),
        ]
        agent = self._run(auto_mode=True, responses=responses, callbacks=callbacks)
        self.assertFalse(callback_called, "on_ask_user must not be called in auto mode")
        # The synthetic nudge must be in the conversation as a tool result.
        tool_results = [
            m for m in agent.session.messages
            if m.get("role") == "tool"
        ]
        self.assertTrue(any("auto mode" in r["content"].lower() for r in tool_results))

    def test_no_tool_call_nudges_in_auto_mode(self) -> None:
        # First response: assistant turn with no tool calls (would normally
        # exit the loop). In auto mode this should inject a "please continue"
        # message and re-loop. Second response: exit_auto_mode.
        responses = [
            _StubChatCompletion(content="I'll think about it"),
            _StubChatCompletion(tool_calls=[
                ("call_1", "exit_auto_mode", {"status": "success", "reason": "thought hard"}),
            ]),
        ]
        agent = self._run(auto_mode=True, responses=responses)
        self.assertIsNotNone(agent._auto_exit)
        # The nudge user-message must be in session.messages
        nudge_msgs = [
            m for m in agent.session.messages
            if m.get("role") == "user"
            and "auto mode" in (m.get("content") or "").lower()
            and "without a tool call" in (m.get("content") or "").lower()
        ]
        self.assertEqual(len(nudge_msgs), 1)

    def test_no_tool_call_returns_in_interactive_mode(self) -> None:
        # In interactive mode, a turn without tool calls returns to the user.
        responses = [
            _StubChatCompletion(content="ok, your turn"),
        ]
        agent = self._run(auto_mode=False, responses=responses)
        self.assertIsNone(agent._auto_exit)
        # No nudge injected
        nudge_msgs = [
            m for m in agent.session.messages
            if m.get("role") == "user"
            and "without a tool call" in (m.get("content") or "").lower()
        ]
        self.assertEqual(len(nudge_msgs), 0)

    def test_set_auto_stage_fires_callback(self) -> None:
        stages: list[tuple[str, str | None]] = []

        def on_auto_stage(stage: str, note: str | None) -> None:
            stages.append((stage, note))

        callbacks = AgentCallbacks(on_auto_stage=on_auto_stage)
        responses = [
            _StubChatCompletion(tool_calls=[
                ("call_1", "set_auto_stage", {"stage": "rubric", "note": "writing scorer"}),
            ]),
            _StubChatCompletion(tool_calls=[
                ("call_2", "exit_auto_mode", {"status": "success", "reason": "ok"}),
            ]),
        ]
        self._run(auto_mode=True, responses=responses, callbacks=callbacks)
        self.assertIn(("rubric", "writing scorer"), stages)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestCliParser(unittest.TestCase):
    def test_no_args_default(self) -> None:
        from lqh.cli import _build_parser
        args = _build_parser().parse_args([])
        self.assertIsNone(args.auto)
        self.assertIsNone(args.spec)

    def test_auto_flag(self) -> None:
        from lqh.cli import _build_parser
        args = _build_parser().parse_args(["--auto", "/tmp/myproj"])
        self.assertEqual(str(args.auto), "/tmp/myproj")

    def test_spec_flag(self) -> None:
        from lqh.cli import _build_parser
        args = _build_parser().parse_args(["--spec", "use small model"])
        self.assertEqual(args.spec, "use small model")

    def test_auto_and_spec(self) -> None:
        from lqh.cli import _build_parser
        args = _build_parser().parse_args([
            "--auto", "/tmp/p", "--spec", "X",
        ])
        self.assertEqual(str(args.auto), "/tmp/p")
        self.assertEqual(args.spec, "X")


if __name__ == "__main__":
    unittest.main()
