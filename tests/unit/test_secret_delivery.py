"""Tests for one-time secret delivery (inference keys).

The plaintext inference key must never enter the conversation (and therefore
never the local JSONL log or the backend payload capture). It is delivered to
the user out-of-band — shown in the TUI and/or appended to ``.env`` — while a
REDACTED message takes its place in the message history.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import lqh.agent as agentmod
from lqh.agent import Agent, AgentCallbacks
from lqh.env_secrets import append_env_secret
from lqh.session import Session
from lqh.tools.handlers import (
    SECRET_DELIVERY_REQUIRED,
    SecretDelivery,
    ToolResult,
    handle_create_inference_key,
)

PLAINTEXT = "sk_inf_SUPERSECRETPLAINTEXT99"


# --------------------------------------------------------------------------- #
# append_env_secret
# --------------------------------------------------------------------------- #

def test_append_env_secret_fresh(tmp_path: Path) -> None:
    note = append_env_secret(tmp_path, "LQH_INFERENCE_KEY", PLAINTEXT, "# key foo")
    env = (tmp_path / ".env").read_text()
    assert f"LQH_INFERENCE_KEY={PLAINTEXT}" in env
    assert "# key foo" in env
    # .env is gitignored.
    assert ".env" in (tmp_path / ".gitignore").read_text().splitlines()
    assert "Appended to .env" in note


def test_append_env_secret_never_deletes_on_duplicate(tmp_path: Path) -> None:
    append_env_secret(tmp_path, "LQH_INFERENCE_KEY", "first_value", "# first")
    note = append_env_secret(tmp_path, "LQH_INFERENCE_KEY", "second_value", "# second")
    env = (tmp_path / ".env").read_text()
    # Both entries survive — append-only, never rewrite.
    assert "first_value" in env and "second_value" in env
    assert "already existed" in note


def test_append_env_secret_warns_when_git_tracked(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / ".env").write_text("EXISTING=1\n")
    subprocess.run(["git", "add", ".env"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "add env"], cwd=tmp_path, check=True)

    note = append_env_secret(tmp_path, "LQH_INFERENCE_KEY", PLAINTEXT, None)
    assert "tracked by git" in note


# --------------------------------------------------------------------------- #
# handle_create_inference_key — channel split
# --------------------------------------------------------------------------- #

async def test_create_inference_key_returns_sentinel_with_split(monkeypatch) -> None:
    async def fake_backend(method, path, json_body=None):
        return 201, {"id": "key_123", "name": json_body["name"], "key": PLAINTEXT}

    monkeypatch.setattr("lqh.tools.handlers._backend_json", fake_backend)

    result = await handle_create_inference_key(Path("/tmp"), name="prod-key")
    assert result.content == SECRET_DELIVERY_REQUIRED
    assert result.secret is not None
    # Plaintext lives only in the out-of-band display, never in the redacted
    # message that lands in the conversation.
    assert PLAINTEXT in result.secret.display
    assert PLAINTEXT not in result.secret.redacted
    assert result.secret.env_var == "LQH_INFERENCE_KEY"
    assert result.options == ["Continue (hide key)", "Continue & append to .env"]


# --------------------------------------------------------------------------- #
# Agent interception — the plaintext never reaches the returned content
# --------------------------------------------------------------------------- #

def _sentinel_result() -> ToolResult:
    return ToolResult(
        content=SECRET_DELIVERY_REQUIRED,
        requires_user_input=True,
        secret=SecretDelivery(
            payload=PLAINTEXT,
            display=f"KEY: {PLAINTEXT}",
            redacted="REDACTED prefix sk_inf_SUPER…",
            env_var="LQH_INFERENCE_KEY",
            env_comment="# test",
        ),
        options=["Continue (hide key)", "Continue & append to .env"],
    )


def _agent(tmp_path: Path, callbacks: AgentCallbacks, *, auto: bool, monkeypatch) -> Agent:
    async def fake_execute(*a, **k):
        return _sentinel_result()

    monkeypatch.setattr(agentmod, "execute_tool", fake_execute)
    session = Session(id="t", project_dir=tmp_path)
    return Agent(tmp_path, session, callbacks, auto_mode=auto)


async def test_interactive_append_writes_env_and_redacts(tmp_path, monkeypatch) -> None:
    captured = {}

    async def on_show_secret(text):
        captured["shown"] = text

    async def on_ask_user(q, opts, ms=False, allow_other=True):
        captured["opts"] = opts
        captured["allow_other"] = allow_other
        return "Continue & append to .env"

    cbs = AgentCallbacks(on_show_secret=on_show_secret, on_ask_user=on_ask_user)
    agent = _agent(tmp_path, cbs, auto=False, monkeypatch=monkeypatch)

    result = await agent._handle_tool_call("create_inference_key", {})

    # The conversation gets the redacted text, never the key.
    assert PLAINTEXT not in result.content
    assert "REDACTED" in result.content
    # The key was shown out-of-band and saved to .env.
    assert PLAINTEXT in captured["shown"]
    assert PLAINTEXT in (tmp_path / ".env").read_text()
    # No free-text "Other" option on a fixed-choice confirm.
    assert captured["allow_other"] is False
    assert all("Other" not in o for o in captured["opts"])


async def test_interactive_hide_does_not_write_env(tmp_path, monkeypatch) -> None:
    async def on_show_secret(text):
        pass

    async def on_ask_user(q, opts, ms=False, allow_other=True):
        return "Continue (hide key)"

    cbs = AgentCallbacks(on_show_secret=on_show_secret, on_ask_user=on_ask_user)
    agent = _agent(tmp_path, cbs, auto=False, monkeypatch=monkeypatch)

    result = await agent._handle_tool_call("create_inference_key", {})
    assert PLAINTEXT not in result.content
    assert not (tmp_path / ".env").exists()


async def test_auto_mode_appends_env_without_prompting(tmp_path, monkeypatch) -> None:
    agent = _agent(tmp_path, AgentCallbacks(), auto=True, monkeypatch=monkeypatch)
    result = await agent._handle_tool_call("create_inference_key", {})
    assert PLAINTEXT not in result.content
    assert PLAINTEXT in (tmp_path / ".env").read_text()


async def test_headless_fallback_appends_env(tmp_path, monkeypatch) -> None:
    # No callbacks available (e.g. non-interactive run) → persist to .env so the
    # one-time key is never lost.
    agent = _agent(tmp_path, AgentCallbacks(), auto=False, monkeypatch=monkeypatch)
    result = await agent._handle_tool_call("create_inference_key", {})
    assert PLAINTEXT not in result.content
    assert PLAINTEXT in (tmp_path / ".env").read_text()
