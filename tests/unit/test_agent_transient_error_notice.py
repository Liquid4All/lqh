"""The agent must say something when an API call fails and is retried.

Feedback #49: "i keep getting 502 every so often. And recovering from that
seems non trivial. There is no indication of how to recover and how much of
work that the model spent behind the scenes churning is recoverable vs lost."
The retry ladder used to run silently behind a spinner, so a 502 on a long
reasoning turn looked exactly like a hang.
"""

from __future__ import annotations

from types import SimpleNamespace

from lqh.agent import (
    ORCHESTRATION_API_RETRIES,
    ORCHESTRATION_API_RETRIES_NESTED,
    Agent,
    AgentCallbacks,
)
from lqh.session import Session


def _completion(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


async def test_retry_notice_reaches_the_surface(project_dir, monkeypatch) -> None:
    notices: list[str] = []

    async def on_transient_error(text: str) -> None:
        notices.append(text)

    async def fake_chat_with_retry(_client, **kwargs):
        # Stand in for the real ladder: report one failed attempt, then answer.
        await kwargs["on_retry"]("HTTP 502: upstream model did not respond", 1, 2, 2.0)
        return _completion("done")

    monkeypatch.setattr("lqh.agent.chat_with_retry", fake_chat_with_retry)
    session = Session.create(project_dir)
    agent = Agent(
        project_dir,
        session,
        callbacks=AgentCallbacks(on_transient_error=on_transient_error),
    )
    agent._client = object()

    await agent.process_user_input("hello")

    assert len(notices) == 1
    notice = notices[0]
    assert "502" in notice
    assert "attempt 1 of 2" in notice
    # The whole point: the user learns what survived without having to ask.
    assert "Nothing is lost" in notice


async def test_no_callback_is_not_an_error(project_dir, monkeypatch) -> None:
    """Headless runs and the SDK construct callbacks without this field."""

    async def fake_chat_with_retry(_client, **kwargs):
        await kwargs["on_retry"]("HTTP 502", 1, 2, 2.0)
        return _completion("done")

    monkeypatch.setattr("lqh.agent.chat_with_retry", fake_chat_with_retry)
    session = Session.create(project_dir)
    agent = Agent(project_dir, session, callbacks=AgentCallbacks())
    agent._client = object()

    await agent.process_user_input("hello")

    assert session.messages[-1]["content"] == "done"


async def test_retry_depth_is_per_surface(project_dir, monkeypatch) -> None:
    """A standalone driver keeps the deep ladder; a nesting surface lowers it.

    `lqh run` has nothing above the agent — an escaped 502 ends the run — so it
    must keep the full count. The TUI retries whole turns itself, and there the
    two ladders would multiply into tens of minutes of silent retrying.
    """
    seen: list[int] = []

    async def fake_chat_with_retry(_client, **kwargs):
        seen.append(kwargs["max_retries"])
        return _completion("done")

    monkeypatch.setattr("lqh.agent.chat_with_retry", fake_chat_with_retry)
    session = Session.create(project_dir)
    agent = Agent(project_dir, session, callbacks=AgentCallbacks())
    agent._client = object()

    await agent.process_user_input("hello")
    assert seen == [ORCHESTRATION_API_RETRIES]

    agent.api_retries = ORCHESTRATION_API_RETRIES_NESTED
    await agent.process_user_input("again")
    assert seen[-1] == ORCHESTRATION_API_RETRIES_NESTED
    assert ORCHESTRATION_API_RETRIES_NESTED < ORCHESTRATION_API_RETRIES
