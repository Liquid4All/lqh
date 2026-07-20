"""AgentPolicy presets and the sub-agent policy gates (CLI_PLAN §4.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lqh.agent import Agent
from lqh.agent_policy import (
    PUBLISH_TOOLS,
    TUI_AUTO,
    TUI_INTERACTIVE,
    subagent_policy,
)
from lqh.session import Session


def _agent(tmp_path: Path, **kwargs) -> Agent:
    return Agent(tmp_path, Session.create(tmp_path), **kwargs)


def test_auto_mode_maps_to_tui_auto(tmp_path: Path) -> None:
    agent = _agent(tmp_path, auto_mode=True)
    assert agent.policy == TUI_AUTO
    assert agent.auto_mode is True
    # The auto skill is injected sticky.
    assert any("auto mode" in m.lower() for m in agent.sticky_system_messages)


def test_default_maps_to_interactive(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    assert agent.policy == TUI_INTERACTIVE
    assert agent.auto_mode is False
    assert agent.sticky_system_messages == []


def test_subagent_policy_shape() -> None:
    policy = subagent_policy()
    assert policy.no_user is True
    assert policy.sticky_skill == "subagent"
    assert policy.auto_grant_permissions is False
    assert policy.granted_domains == {"script", "cloud_data_gen", "training"}
    assert policy.allow_publish is False
    assert policy.compute_default is None
    assert policy.secret_delivery == "result"

    publishing = subagent_policy(allow_publish=True)
    assert "hf_push" in publishing.granted_domains
    assert subagent_policy(save_secrets=True).secret_delivery == "env"


def test_subagent_gets_subagent_skill(tmp_path: Path) -> None:
    agent = _agent(tmp_path, policy=subagent_policy())
    assert agent.auto_mode is True  # no_user implies auto-style loop handling
    assert any("sub-agent" in m.lower() for m in agent.sticky_system_messages)


@pytest.mark.parametrize("tool", sorted(PUBLISH_TOOLS))
async def test_publish_gate_terminates_run(tmp_path: Path, tool: str) -> None:
    agent = _agent(tmp_path, policy=subagent_policy())
    result = await agent._handle_tool_call(tool, {})
    assert agent._policy_halt is not None
    status, hint = agent._policy_halt
    assert status == "needs_permission"
    assert "--allow-publish" in hint
    assert "terminate" in result.content


async def test_publish_allowed_with_flag(tmp_path: Path) -> None:
    """With --allow-publish the gate must NOT fire (the tool itself will
    fail later on missing auth — that's fine, the halt must stay unset)."""
    agent = _agent(tmp_path, policy=subagent_policy(allow_publish=True))
    await agent._handle_tool_call("hf_push", {"local_path": "nope"})
    assert agent._policy_halt is None


async def test_ask_user_intercepted_for_subagent(tmp_path: Path) -> None:
    agent = _agent(tmp_path, policy=subagent_policy())
    result = await agent._handle_tool_call("ask_user", {"question": "hm?"})
    assert "no user" in result.content.lower()
    assert agent._policy_halt is None


async def test_compute_pick_without_default_halts(tmp_path: Path, monkeypatch) -> None:
    from lqh.tools.handlers import COMPUTE_PICK_REQUIRED, ToolResult

    agent = _agent(tmp_path, policy=subagent_policy())

    async def fake_execute(tool, args, project_dir, **kw):
        return ToolResult(
            content=COMPUTE_PICK_REQUIRED,
            requires_user_input=True,
            question="pick",
            options=["cloud", "local"],
        )

    monkeypatch.setattr("lqh.agent.execute_tool", fake_execute)
    result = await agent._handle_tool_call("start_training", {"type": "sft"})
    assert agent._policy_halt is not None
    assert agent._policy_halt[0] == "needs_configuration"
    assert "compute_set" in result.content


async def test_granted_domains_flow_to_handlers(tmp_path: Path, monkeypatch) -> None:
    from lqh.tools.handlers import ToolResult

    seen: dict = {}

    async def fake_execute(tool, args, project_dir, **kw):
        seen.update(kw)
        return ToolResult(content="ok", ok=True)

    monkeypatch.setattr("lqh.agent.execute_tool", fake_execute)
    agent = _agent(tmp_path, policy=subagent_policy())
    await agent._handle_tool_call("summary", {})
    perms = seen.get("_permissions")
    assert perms is not None
    assert perms.allows_script(tmp_path, "data_gen/x.py") is True
    assert perms.allows_training(tmp_path, "r1") is True
    assert perms.allows_hf_push(tmp_path, "o/r") is False


async def test_secret_delivery_result_mode(tmp_path: Path, monkeypatch) -> None:
    from lqh.tools.handlers import (
        SECRET_DELIVERY_REQUIRED,
        SecretDelivery,
        ToolResult,
    )

    async def fake_execute(tool, args, project_dir, **kw):
        return ToolResult(
            content=SECRET_DELIVERY_REQUIRED,
            requires_user_input=True,
            secret=SecretDelivery(
                payload="sk-99", display="d", redacted="key created",
                env_var="LQH_KEY",
            ),
        )

    monkeypatch.setattr("lqh.agent.execute_tool", fake_execute)
    agent = _agent(tmp_path, policy=subagent_policy(allow_publish=True))
    result = await agent._handle_tool_call("create_inference_key", {})
    # Secret rides the payload channel, never .env, never the content.
    assert "sk-99" not in result.content
    assert not (tmp_path / ".env").exists()
    assert [s.payload for s in agent.delivered_secrets] == ["sk-99"]
