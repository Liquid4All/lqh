"""The agent runs orchestration turns in async mode and can resume them."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from lqh.agent import ORCHESTRATION_MAX_TOKENS, Agent, AgentCallbacks, _has_unparseable_tool_call
from lqh.client import CompletionPollExhausted, CompletionLostError
from lqh.session import Session

import httpx


def _completion(content: str = "done", tool_calls=None, finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


def _agent(project_dir, monkeypatch, fake, callbacks=None) -> Agent:
    monkeypatch.setattr("lqh.agent.chat_with_retry", fake)
    session = Session.create(project_dir)
    agent = Agent(project_dir, session, callbacks=callbacks or AgentCallbacks())
    agent._client = object()
    return agent


async def test_main_loop_uses_async_mode_with_the_full_budget(project_dir, monkeypatch):
    seen: list[dict] = []

    async def fake(_client, **kwargs):
        seen.append(kwargs)
        return _completion()

    agent = _agent(project_dir, monkeypatch, fake)
    await agent.process_user_input("hi")

    kw = seen[0]
    assert kw["async_mode"] is True
    assert kw["resume_id"] is None
    assert kw["max_tokens"] == ORCHESTRATION_MAX_TOKENS == 131_072
    hooks = kw["async_hooks"]
    assert hooks.on_started is not None and hooks.on_lost is not None and hooks.on_poll_retry is not None


async def test_on_started_persists_record_and_notice_shows_once(project_dir, monkeypatch):
    notices: list[str] = []
    records: list[dict | None] = []

    async def on_transient_error(text: str) -> None:
        notices.append(text)

    async def fake(_client, **kwargs):
        hooks = kwargs["async_hooks"]
        await hooks.on_started("lqhc_" + "a" * 32)
        records.append(dict(agent.session.pending_completion or {}))
        hooks.on_progress(1234, 61.5)
        return _completion()

    progress: list[tuple[int, float]] = []
    agent = _agent(project_dir, monkeypatch, fake, AgentCallbacks(
        on_transient_error=on_transient_error,
        on_completion_progress=lambda t, e: progress.append((t, e)),
    ))
    await agent.process_user_input("hi")
    await agent.process_user_input("again")

    rec = records[0]
    assert rec["id"] == "lqhc_" + "a" * 32 and rec["kind"] == "turn"
    assert rec["model"] == agent.orchestration_model
    assert rec["turn_seq"] == 1  # last_seq when the first user message was the only entry
    assert agent.session.pending_completion is None  # cleared on the final response
    assert progress == [(1234, 61.5), (1234, 61.5)]
    assert sum("Long reasoning turn" in n for n in notices) == 1


async def test_terminal_error_clears_record_but_resumable_error_keeps_it(project_dir, monkeypatch):
    calls = {"n": 0}

    async def fake(_client, **kwargs):
        calls["n"] += 1
        await kwargs["async_hooks"].on_started("lqhc_" + "b" * 32)
        if calls["n"] == 1:
            raise CompletionPollExhausted("lqhc_" + "b" * 32, RuntimeError("x"), httpx.Request("GET", "https://x"))
        raise ValueError("bad request")

    agent = _agent(project_dir, monkeypatch, fake)
    try:
        await agent.process_user_input("hi")
    except CompletionPollExhausted:
        pass
    assert agent.session.pending_completion is not None
    assert agent._resumable_pending() is not None

    try:
        await agent.continue_after_interruption()
    except ValueError:
        pass
    assert agent.session.pending_completion is None


async def test_next_iteration_resumes_matching_record_and_drops_stale_one(project_dir, monkeypatch):
    seen: list[dict] = []

    async def fake(_client, **kwargs):
        seen.append(kwargs)
        return _completion()

    cancelled: list[str] = []

    async def fake_cancel(_client, cid):
        cancelled.append(cid)
        return True

    monkeypatch.setattr("lqh.agent.cancel_completion", fake_cancel)
    agent = _agent(project_dir, monkeypatch, fake)
    agent.session.add_message({"role": "user", "content": "hi"})
    agent.session.set_pending_completion({
        "id": "lqhc_" + "c" * 32, "model": agent.orchestration_model, "kind": "turn",
        "started_at": "now", "turn_seq": agent.session.last_seq,
    })
    await agent.continue_after_interruption()
    assert seen[0]["resume_id"] == "lqhc_" + "c" * 32
    assert cancelled == []

    # History moved on: the record no longer fits, so it is dropped and the
    # server-side generation cancelled instead of re-attached.
    agent.session.set_pending_completion({
        "id": "lqhc_" + "d" * 32, "model": agent.orchestration_model, "kind": "turn",
        "started_at": "now", "turn_seq": agent.session.last_seq - 1,
    })
    await agent.process_user_input("more")
    import asyncio
    await asyncio.sleep(0)
    assert seen[1]["resume_id"] is None
    assert cancelled == ["lqhc_" + "d" * 32]
    assert agent.session.pending_completion is None


async def test_cancel_pending_completion_deletes_and_clears_even_on_failure(project_dir, monkeypatch):
    async def fake(_client, **kwargs):
        return _completion()

    async def failing_cancel(_client, cid):
        raise RuntimeError("network")

    monkeypatch.setattr("lqh.agent.cancel_completion", failing_cancel)
    agent = _agent(project_dir, monkeypatch, fake)
    agent.session.add_message({"role": "user", "content": "hi"})
    agent.session.set_pending_completion({
        "id": "lqhc_" + "e" * 32, "model": agent.orchestration_model, "kind": "turn",
        "started_at": "now", "turn_seq": agent.session.last_seq,
    })
    await agent.cancel_pending_completion()
    assert agent.session.pending_completion is None
    await agent.cancel_pending_completion()  # nothing pending: no-op


async def test_lost_hook_clears_only_the_matching_record(project_dir, monkeypatch):
    async def fake(_client, **kwargs):
        hooks = kwargs["async_hooks"]
        await hooks.on_started("lqhc_" + "f" * 32)
        await hooks.on_lost("lqhc_" + "0" * 32)
        assert agent.session.pending_completion is not None
        await hooks.on_lost("lqhc_" + "f" * 32)
        assert agent.session.pending_completion is None
        return _completion()

    agent = _agent(project_dir, monkeypatch, fake)
    await agent.process_user_input("hi")


async def test_compaction_stays_synchronous_and_never_persists(project_dir, monkeypatch):
    seen: list[dict] = []

    async def fake(_client, **kwargs):
        seen.append(kwargs)
        return _completion("summary")

    agent = _agent(project_dir, monkeypatch, fake)
    for i in range(6):
        agent.session.add_message({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
    await agent._compact_context(force=True)
    kw = seen[-1]
    assert not kw.get("async_mode")
    assert agent.session.pending_completion is None


def test_truncation_signal_is_unparseable_tool_arguments():
    good = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments='{"path": "a"}'))])
    cut = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments='{"path": "a", "content": "unterm'))])
    empty = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments=""))])
    assert not _has_unparseable_tool_call(good)
    assert _has_unparseable_tool_call(cut)
    assert not _has_unparseable_tool_call(empty)
    assert not _has_unparseable_tool_call(SimpleNamespace(tool_calls=None))


async def test_cut_off_tool_call_triggers_the_recovery_notice(project_dir, monkeypatch):
    calls = {"n": 0}

    async def fake(_client, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tc = SimpleNamespace(id="t1", function=SimpleNamespace(name="read_file", arguments='{"path": "x'))
            return _completion(content=None, tool_calls=[tc], finish_reason="tool_calls")
        return _completion("ok")

    agent = _agent(project_dir, monkeypatch, fake)
    agent.max_tool_calls_per_turn = 5
    await agent.process_user_input("hi")
    notices = [m for m in agent.session.messages if m.get("role") == "user" and "cut off" in str(m.get("content"))]
    assert notices, "a cut-off tool call must inject the truncation recovery notice"
