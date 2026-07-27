"""Tests for interactive ``ask_user`` handling in the agent loop."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from lqh.agent import Agent, AgentCallbacks
from lqh.session import Session


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


def _ask_user_call(call_id: str, question: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="ask_user",
            arguments=json.dumps({"question": question}),
        ),
    )


async def test_batched_ask_user_calls_are_resolved_sequentially(
    project_dir, monkeypatch,
) -> None:
    questions: list[str] = []
    prompt_active = False

    async def on_ask_user(
        question: str,
        options: list[str] | None,
        multi_select: bool = False,
    ) -> str:
        nonlocal prompt_active
        assert not prompt_active
        prompt_active = True
        questions.append(question)
        await asyncio.sleep(0)
        prompt_active = False
        return {"First question?": "first answer", "Second question?": "second answer"}[
            question
        ]

    batched = _completion(
        tool_calls=[
            _ask_user_call("call_first", "First question?"),
            _ask_user_call("call_second", "Second question?"),
        ],
        finish_reason="tool_calls",
    )
    final = _completion(content="Thanks, I have both answers.")
    llm_calls = 0
    second_call_messages: list[dict] = []

    async def fake_chat_with_retry(_client, **kwargs):
        nonlocal llm_calls, second_call_messages
        llm_calls += 1
        if llm_calls == 1:
            return batched
        assert questions == ["First question?", "Second question?"]
        second_call_messages = kwargs["messages"]
        return final

    monkeypatch.setattr("lqh.agent.chat_with_retry", fake_chat_with_retry)
    session = Session.create(project_dir)
    agent = Agent(
        project_dir,
        session,
        callbacks=AgentCallbacks(on_ask_user=on_ask_user),
    )
    agent._client = object()

    await agent.process_user_input("Ask both questions.")

    assert llm_calls == 2
    assert questions == ["First question?", "Second question?"]
    tool_results = [m for m in session.messages if m.get("role") == "tool"]
    assert tool_results == [
        {
            "role": "tool",
            "tool_call_id": "call_first",
            "content": "first answer",
        },
        {
            "role": "tool",
            "tool_call_id": "call_second",
            "content": "second answer",
        },
    ]
    assert second_call_messages[-2:] == tool_results
