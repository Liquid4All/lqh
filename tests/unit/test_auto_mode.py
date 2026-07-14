"""Unit + integration tests for auto mode (no API access required).

Covers the auto-mode plumbing: sticky system messages survive compaction,
``ask_user`` is intercepted in auto mode, the no-tool-call assistant turn
is nudged with "please continue", ``exit_auto_mode`` terminates the loop,
and the auto-only tools are gated on the ``auto_mode`` flag.

The full end-to-end pipeline test (real spec → trained checkpoint) lives
in ``tests/e2e/test_auto_mode_e2e.py`` and is gated on API access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lqh.agent import Agent, AgentCallbacks
from lqh.session import Session
from lqh.tools.definitions import get_all_tools
from lqh.tools.handlers import (
    execute_tool,
    handle_exit_auto_mode,
    handle_set_auto_stage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auto_session(project_dir: Path) -> Session:
    """A fresh session rooted in an isolated project directory."""
    return Session.create(project_dir)


@pytest.fixture
def make_agent(auto_session: Session) -> Callable[..., Agent]:
    """Factory that builds an Agent against a shared throw-away session."""

    def _factory(**kwargs: Any) -> Agent:
        return Agent(auto_session.project_dir, auto_session, **kwargs)

    return _factory


def _make_tool_call_stub(tc_id: str, name: str, args: dict[str, Any]) -> MagicMock:
    tc = MagicMock()
    tc.id = tc_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _make_completion(
    *,
    content: str | None = None,
    tool_calls: list[tuple[str, str, dict[str, Any]]] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Minimal stand-in for an OpenAI ``ChatCompletion`` response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = (
        [_make_tool_call_stub(*tc) for tc in tool_calls] if tool_calls else None
    )
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


class ScriptedAgent:
    """An Agent paired with a queued ``chat_with_retry`` response list.

    Use ``async with scripted_agent.run() as agent:`` to install the patch
    over the module-level ``chat_with_retry`` while the agent runs.
    """

    def __init__(self, agent: Agent, responses: list[MagicMock]) -> None:
        self.agent = agent
        # Top off the queue so a runaway loop terminates without raising.
        self._responses = list(responses) + [_make_completion(content="done")] * 5

    async def run(self, user_input: str = "kickoff") -> Agent:
        with patch("lqh.agent.chat_with_retry", side_effect=self._responses):
            await self.agent.process_user_input(user_input)
        return self.agent


@pytest.fixture
def scripted_agent(make_agent: Callable[..., Agent]) -> Callable[..., ScriptedAgent]:
    """Drive an Agent's chat loop with a pre-scripted response queue."""

    def _factory(
        *,
        auto_mode: bool,
        responses: list[MagicMock],
        callbacks: AgentCallbacks | None = None,
    ) -> ScriptedAgent:
        agent = make_agent(auto_mode=auto_mode)
        if callbacks is not None:
            agent.callbacks = callbacks
        agent._client = MagicMock()  # bypass _get_client / auth
        return ScriptedAgent(agent, responses)

    return _factory


# ---------------------------------------------------------------------------
# Tool definitions: auto-mode tools are gated on the flag
# ---------------------------------------------------------------------------


class TestAutoModeToolGating:
    def test_default_excludes_auto_tools(self) -> None:
        names = {t["function"]["name"] for t in get_all_tools()}
        assert "exit_auto_mode" not in names
        assert "set_auto_stage" not in names

    def test_auto_mode_includes_auto_tools(self) -> None:
        names = {t["function"]["name"] for t in get_all_tools(auto_mode=True)}
        assert {"exit_auto_mode", "set_auto_stage"} <= names

    def test_training_status_description_discourages_polling(self) -> None:
        training_status = next(
            t for t in get_all_tools() if t["function"]["name"] == "training_status"
        )

        description = training_status["function"]["description"]
        assert "watched in the background" in description
        assert "do not repeatedly poll" in description
        assert "wake automatically" in description


# ---------------------------------------------------------------------------
# exit_auto_mode + set_auto_stage handlers
# ---------------------------------------------------------------------------


class TestAutoModeHandlers:
    @pytest.mark.parametrize(
        "status,reason",
        [
            ("success", "baseline 4.0 → final 7.5"),
            ("failure", "data pipeline cannot satisfy spec"),
        ],
    )
    async def test_exit_auto_mode_terminal_statuses(self, status: str, reason: str) -> None:
        result = await handle_exit_auto_mode(status=status, reason=reason)
        assert result.exit_auto_mode is True
        assert result.auto_status == status
        assert reason.split()[0] in result.auto_reason

    async def test_exit_auto_mode_invalid_status(self) -> None:
        result = await handle_exit_auto_mode(status="kaput", reason="x")
        assert result.exit_auto_mode is False
        assert "Error" in result.content

    async def test_set_auto_stage(self) -> None:
        result = await handle_set_auto_stage(
            stage="sft_initial", note="2k samples, score 6.8/10",
        )
        assert result.auto_stage == "sft_initial"
        assert result.auto_stage_note == "2k samples, score 6.8/10"

    async def test_set_auto_stage_empty(self) -> None:
        result = await handle_set_auto_stage(stage="", note=None)
        assert result.auto_stage is None
        assert "Error" in result.content

    async def test_execute_tool_dispatch_no_project_dir(self) -> None:
        # exit_auto_mode and set_auto_stage are routed without project_dir.
        result = await execute_tool(
            "exit_auto_mode",
            {"status": "success", "reason": "done"},
            Path("/nonexistent/should/not/be/read"),
        )
        assert result.exit_auto_mode is True


# ---------------------------------------------------------------------------
# Agent: sticky system messages
# ---------------------------------------------------------------------------


class TestStickySystemMessages:
    def test_default_orchestration_model_is_known_good(
        self, make_agent: Callable[..., Agent],
    ) -> None:
        agent = make_agent()

        assert agent.orchestration_model == "orchestration:12"

    def test_auto_mode_injects_skill_as_sticky(self, make_agent: Callable[..., Agent]) -> None:
        agent = make_agent(auto_mode=True)
        assert len(agent.sticky_system_messages) == 1
        assert "auto mode" in agent.sticky_system_messages[0].lower()

    def test_extra_spec_injected_alongside_auto(self, make_agent: Callable[..., Agent]) -> None:
        agent = make_agent(auto_mode=True, extra_spec="use the smallest base model")
        assert len(agent.sticky_system_messages) == 2
        assert "smallest base model" in agent.sticky_system_messages[1]

    def test_extra_spec_in_interactive_mode(self, make_agent: Callable[..., Agent]) -> None:
        agent = make_agent(auto_mode=False, extra_spec="hint Y")
        assert agent.auto_mode is False
        assert len(agent.sticky_system_messages) == 1
        assert "hint Y" in agent.sticky_system_messages[0]

    def test_no_sticky_when_neither_set(self, make_agent: Callable[..., Agent]) -> None:
        agent = make_agent()
        assert agent.sticky_system_messages == []

    def test_build_messages_prepends_sticky(
        self, make_agent: Callable[..., Agent], auto_session: Session,
    ) -> None:
        agent = make_agent(auto_mode=True, extra_spec="hint Z")
        auto_session.messages = [{"role": "user", "content": "hello"}]

        msgs = agent._build_messages()

        # SYSTEM_PROMPT, auto-skill sticky, extra_spec sticky, user msg.
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "system"
        assert "auto mode" in msgs[1]["content"].lower()
        assert msgs[2]["role"] == "system"
        assert "hint Z" in msgs[2]["content"]
        assert msgs[3]["role"] == "user"

    def test_sticky_survives_simulated_compaction(
        self, make_agent: Callable[..., Agent], auto_session: Session,
    ) -> None:
        """``_compact_context`` overwrites ``session.messages`` — sticky lives outside it."""
        agent = make_agent(auto_mode=True, extra_spec="hint")
        # Simulate post-compaction state: session.messages has only the
        # summary + recent turns. Sticky entries are not in there.
        auto_session.messages = [
            {"role": "system", "content": "[Context compacted] short summary"},
            {"role": "user", "content": "recent user msg"},
        ]

        msgs = agent._build_messages()
        sticky_count = sum(
            1 for m in msgs[:3]
            if m["role"] == "system"
            and ("auto mode" in m["content"].lower() or "hint" in m["content"])
        )
        assert sticky_count == 2


# ---------------------------------------------------------------------------
# Agent loop: ask_user interception, no-tool-call nudge, exit termination
# ---------------------------------------------------------------------------


class TestAutoModeAgentBehavior:
    """Drive the agent loop with a stubbed chat API and verify auto-mode behaviors."""

    async def test_exit_auto_mode_breaks_loop(
        self, scripted_agent: Callable[..., ScriptedAgent],
    ) -> None:
        scripted = scripted_agent(
            auto_mode=True,
            responses=[
                _make_completion(tool_calls=[
                    ("call_1", "exit_auto_mode", {"status": "success", "reason": "ok"}),
                ]),
                # The loop should NOT consume this.
                _make_completion(content="should not be reached"),
            ],
        )
        agent = await scripted.run()

        assert agent._auto_exit is not None
        assert agent._auto_exit[0] == "success"

    async def test_ask_user_intercepted_in_auto_mode(
        self, scripted_agent: Callable[..., ScriptedAgent],
    ) -> None:
        callback_called = False

        async def on_ask_user(*_: Any, **__: Any) -> str:
            nonlocal callback_called
            callback_called = True
            return "should-not-be-called"

        scripted = scripted_agent(
            auto_mode=True,
            responses=[
                _make_completion(tool_calls=[("call_1", "ask_user", {"question": "what?"})]),
                _make_completion(tool_calls=[
                    ("call_2", "exit_auto_mode", {"status": "failure", "reason": "tried to ask"}),
                ]),
            ],
            callbacks=AgentCallbacks(on_ask_user=on_ask_user),
        )
        agent = await scripted.run()

        assert not callback_called, "on_ask_user must not be called in auto mode"
        # The synthetic nudge must be in the conversation as a tool result.
        tool_results = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert any("auto mode" in r["content"].lower() for r in tool_results)

    async def test_no_tool_call_nudges_in_auto_mode(
        self, scripted_agent: Callable[..., ScriptedAgent],
    ) -> None:
        # First response: assistant turn with no tool calls (would normally
        # exit the loop). In auto mode this should inject a "please continue"
        # message and re-loop. Second response: exit_auto_mode.
        scripted = scripted_agent(
            auto_mode=True,
            responses=[
                _make_completion(content="I'll think about it"),
                _make_completion(tool_calls=[
                    ("call_1", "exit_auto_mode", {"status": "success", "reason": "thought hard"}),
                ]),
            ],
        )
        agent = await scripted.run()

        assert agent._auto_exit is not None
        nudges = [
            m for m in agent.session.messages
            if m.get("role") == "user"
            and "auto mode" in (m.get("content") or "").lower()
            and "without a tool call" in (m.get("content") or "").lower()
        ]
        assert len(nudges) == 1

    async def test_no_tool_call_returns_in_interactive_mode(
        self, scripted_agent: Callable[..., ScriptedAgent],
    ) -> None:
        scripted = scripted_agent(
            auto_mode=False,
            responses=[_make_completion(content="ok, your turn")],
        )
        agent = await scripted.run()

        assert agent._auto_exit is None
        nudges = [
            m for m in agent.session.messages
            if m.get("role") == "user"
            and "without a tool call" in (m.get("content") or "").lower()
        ]
        assert nudges == []

    async def test_empty_tool_call_finish_retries_in_interactive_mode(
        self, scripted_agent: Callable[..., ScriptedAgent],
    ) -> None:
        messages: list[str] = []

        async def on_agent_message(content: str) -> None:
            messages.append(content)

        scripted = scripted_agent(
            auto_mode=False,
            responses=[
                _make_completion(
                    content="Let me check that right now.",
                    finish_reason="tool_calls",
                ),
                _make_completion(content="Done after retry."),
            ],
            callbacks=AgentCallbacks(on_agent_message=on_agent_message),
        )
        agent = await scripted.run()

        recovery_messages = [
            m for m in agent.session.messages
            if m.get("role") == "user"
            and "finish_reason='tool_calls'" in (m.get("content") or "")
            and "no tool_calls payload" in (m.get("content") or "")
        ]
        assert len(recovery_messages) == 1
        assert messages == ["Done after retry."]

    async def test_empty_tool_call_finish_escalates_after_retries(
        self, scripted_agent: Callable[..., ScriptedAgent],
    ) -> None:
        messages: list[str] = []

        async def on_agent_message(content: str) -> None:
            messages.append(content)

        scripted = scripted_agent(
            auto_mode=False,
            responses=[
                _make_completion(
                    content="Let me check that right now.",
                    finish_reason="tool_calls",
                ),
                _make_completion(
                    content="Trying the tool again.",
                    finish_reason="tool_calls",
                ),
            ],
            callbacks=AgentCallbacks(on_agent_message=on_agent_message),
        )
        scripted.agent.max_empty_tool_call_retries = 1

        agent = await scripted.run()

        recovery_messages = [
            m for m in agent.session.messages
            if m.get("role") == "user"
            and "finish_reason='tool_calls'" in (m.get("content") or "")
            and "no tool_calls payload" in (m.get("content") or "")
        ]
        assert len(recovery_messages) == 1
        assert len([
            m for m in agent.session.messages
            if m.get("role") == "tool"
        ]) == 0
        assert messages == [
            "❌ Tool-call backend error: the orchestration model repeatedly "
            "returned finish_reason='tool_calls' but the response contained "
            "no tool_calls payload, so no tool could be executed. Please "
            "retry the request or switch orchestration models if this repeats."
        ]

    async def test_set_auto_stage_fires_callback(
        self, scripted_agent: Callable[..., ScriptedAgent],
    ) -> None:
        stages: list[tuple[str, str | None]] = []

        def on_auto_stage(stage: str, note: str | None) -> None:
            stages.append((stage, note))

        scripted = scripted_agent(
            auto_mode=True,
            responses=[
                _make_completion(tool_calls=[
                    ("call_1", "set_auto_stage", {"stage": "rubric", "note": "writing scorer"}),
                ]),
                _make_completion(tool_calls=[
                    ("call_2", "exit_auto_mode", {"status": "success", "reason": "ok"}),
                ]),
            ],
            callbacks=AgentCallbacks(on_auto_stage=on_auto_stage),
        )
        await scripted.run()

        assert ("rubric", "writing scorer") in stages


class TestAutoModeParking:
    """Auto mode parks on training_status instead of busy-polling runs."""

    async def test_training_status_parks_in_auto_mode(
        self, make_agent: Callable[..., Agent],
    ) -> None:
        from lqh.tools.handlers import ToolResult

        seen: list[tuple[list[str] | None, float]] = []

        async def on_await_background(run_names, timeout):
            seen.append((run_names, timeout))
            return "[System: training run sft_v1 completed successfully.]"

        agent = make_agent(
            auto_mode=True,
            callbacks=AgentCallbacks(on_await_background=on_await_background),
        )

        with patch(
            "lqh.agent.execute_tool",
            return_value=ToolResult(content="✅ **sft_v1** — completed"),
        ):
            result = await agent._handle_tool_call(
                "training_status", {"run_name": "sft_v1"}
            )

        # The callback was consulted with the queried run name, and its
        # completion notice is prepended to the freshly-read status.
        assert seen == [(["sft_v1"], pytest.approx(600.0))]
        assert "completed successfully" in result.content
        assert "sft_v1" in result.content

    async def test_training_status_no_park_when_nothing_running(
        self, make_agent: Callable[..., Agent],
    ) -> None:
        from lqh.tools.handlers import ToolResult

        async def on_await_background(run_names, timeout):
            return None  # nothing running

        agent = make_agent(
            auto_mode=True,
            callbacks=AgentCallbacks(on_await_background=on_await_background),
        )
        with patch(
            "lqh.agent.execute_tool",
            return_value=ToolResult(content="✅ **sft_v1** — completed"),
        ):
            result = await agent._handle_tool_call(
                "training_status", {"run_name": "sft_v1"}
            )

        # No completion notice prepended — just the plain status.
        assert result.content == "✅ **sft_v1** — completed"

    async def test_interactive_mode_does_not_park(
        self, make_agent: Callable[..., Agent],
    ) -> None:
        from lqh.tools.handlers import ToolResult

        called = False

        async def on_await_background(run_names, timeout):
            nonlocal called
            called = True
            return None

        agent = make_agent(
            auto_mode=False,
            callbacks=AgentCallbacks(on_await_background=on_await_background),
        )
        with patch(
            "lqh.agent.execute_tool",
            return_value=ToolResult(content="🏃 **sft_v1** — running"),
        ):
            await agent._handle_tool_call("training_status", {"run_name": "sft_v1"})

        assert called is False  # interactive mode keeps the old behavior

    @pytest.mark.parametrize(
        "tool_name", ["start_training", "start_local_eval", "eval_hf_model"],
    )
    async def test_background_callback_wired_for_run_starters(
        self, make_agent: Callable[..., Agent], tool_name: str,
    ) -> None:
        """Every run-starting tool gets on_background_task_started so the run
        is eagerly registered (otherwise auto-mode parking can't see it)."""
        from lqh.tools.handlers import ToolResult

        sentinel = MagicMock(name="on_background_task_started")
        captured: dict[str, Any] = {}

        async def fake_execute_tool(name, args, project_dir, **extra):
            captured.update(extra)
            return ToolResult(content="started")

        agent = make_agent(
            auto_mode=True,
            callbacks=AgentCallbacks(on_background_task_started=sentinel),
        )
        with patch("lqh.agent.execute_tool", side_effect=fake_execute_tool):
            await agent._handle_tool_call(tool_name, {})

        assert captured.get("on_background_task_started") is sentinel

    @pytest.mark.parametrize("sentinel", ["PERMISSION_REQUIRED", "COMPUTE_PICK_REQUIRED"])
    async def test_readiness_does_not_complete_for_submission_prompt(
        self, make_agent: Callable[..., Agent], sentinel: str,
    ) -> None:
        from lqh.tools.handlers import ToolResult

        telemetry = MagicMock()
        async def run_deferred(callback, *args):
            return callback(*args)
        telemetry.run_deferred = AsyncMock(side_effect=run_deferred)
        agent = make_agent(auto_mode=False)
        deferred = ToolResult(content=sentinel, requires_user_input=True)
        with (
            patch("lqh.agent.execute_tool", return_value=deferred),
            patch("lqh.telemetry.active_telemetry", return_value=telemetry),
        ):
            await agent._handle_tool_call("start_training", {})

        telemetry.complete_readiness.assert_not_called()

    async def test_readiness_completes_only_after_accepted_launch(
        self, make_agent: Callable[..., Agent],
    ) -> None:
        from lqh.tools.handlers import ToolResult

        telemetry = MagicMock()
        async def run_deferred(callback, *args):
            return callback(*args)
        telemetry.run_deferred = AsyncMock(side_effect=run_deferred)
        agent = make_agent(auto_mode=False)
        arguments = {"dataset": "datasets/generated"}
        launched = ToolResult(content="training started", workflow_launched=True)
        with (
            patch("lqh.agent.execute_tool", return_value=launched),
            patch("lqh.telemetry.active_telemetry", return_value=telemetry),
        ):
            await agent._handle_tool_call("start_training", arguments)

        telemetry.complete_readiness.assert_called_once_with(arguments)


class TestTrainingPermissionScope:
    """Approving a training run grants the training domain only — never the
    shared pipeline-execution flag — and auto mode stays autonomous."""

    async def test_interactive_approval_grants_only_this_run(
        self, make_agent: Callable[..., Agent],
    ) -> None:
        from lqh.tools.permissions import (
            check_permission,
            check_training_permission,
        )

        agent = make_agent(auto_mode=False)
        with patch.object(agent, "_reinvoke_tool", return_value=None) as reinvoke:
            await agent._handle_permission_response(
                "Start training", "start_training", {},
                permission_key="training:sft_1",
            )

        reinvoke.assert_awaited_once()
        # Exactly the approved run is granted...
        assert check_training_permission(agent.project_dir, "sft_1") is True
        assert check_training_permission(agent.project_dir, "sft_2") is False
        # ...and pipeline/script execution is untouched.
        assert check_permission(agent.project_dir, "data_gen/x.py") is False

    async def test_interactive_decline_does_not_grant(
        self, make_agent: Callable[..., Agent],
    ) -> None:
        from lqh.tools.permissions import check_training_permission

        agent = make_agent(auto_mode=False)
        result = await agent._handle_permission_response(
            "Do not start training", "start_training", {},
            permission_key="training:sft_1",
        )
        assert "declined" in result.content
        assert check_training_permission(agent.project_dir, "sft_1") is False

    async def test_auto_mode_grants_training_project_wide_not_pipeline(
        self, make_agent: Callable[..., Agent],
    ) -> None:
        from lqh.tools.permissions import (
            check_permission,
            check_training_permission,
        )

        agent = make_agent(auto_mode=True)
        with patch.object(agent, "_reinvoke_tool", return_value=None):
            await agent._handle_permission_response(
                "Execute and don't ask again for this project",
                "start_training", {},
                permission_key="training:sft_1",
            )

        # Autonomous: all future training is allowed without re-prompting...
        assert check_training_permission(agent.project_dir, "sft_1") is True
        assert check_training_permission(agent.project_dir, "anything") is True
        # ...but the pipeline-execution flag is still NOT set by training.
        assert check_permission(agent.project_dir, "data_gen/x.py") is False


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestCliParser:
    @pytest.fixture
    def parser(self):
        from lqh.cli import _build_parser

        return _build_parser()

    def test_no_args_default(self, parser) -> None:
        args = parser.parse_args([])
        assert args.auto is None
        assert args.spec is None

    def test_version_flag(self, parser, capsys) -> None:
        from lqh import __version__

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])

        assert exc_info.value.code == 0
        assert capsys.readouterr().out.strip() == f"lqh {__version__}"

    def test_auto_flag(self, parser) -> None:
        args = parser.parse_args(["--auto", "/tmp/myproj"])
        assert str(args.auto) == "/tmp/myproj"

    def test_spec_flag(self, parser) -> None:
        args = parser.parse_args(["--spec", "use small model"])
        assert args.spec == "use small model"

    def test_auto_and_spec(self, parser) -> None:
        args = parser.parse_args(["--auto", "/tmp/p", "--spec", "X"])
        assert str(args.auto) == "/tmp/p"
        assert args.spec == "X"
