"""Characterization tests for context compaction.

Phase 0 of the persistency work (see PERSISTENCY_PLAN.md). The central
defect documented here: ``Agent._compact_context()`` REPLACES
``session.messages`` with a model-generated summary, and the next save
persists that truncated list — the raw transcript is destroyed. Tests
marked ``CURRENT:`` flip when non-destructive (checkpoint-based)
compaction lands; names stay stable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lqh.agent import Agent
from lqh.session import Session, sessions_dir


def _agent_with_history(
    project_dir: Path, client, n_messages: int = 30
) -> tuple[Agent, Session]:
    session = Session.create(project_dir)
    agent = Agent(project_dir, session)
    agent._client = client  # bypass _get_client / auth
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        session.add_message({"role": role, "content": f"turn {i}: " + "x" * 40})
    return agent, session


async def test_compaction_below_threshold_is_noop(
    project_dir: Path, mock_openai_client
) -> None:
    agent, session = _agent_with_history(
        project_dir, mock_openai_client(), n_messages=6
    )
    before = list(session.messages)

    await agent._compact_context()

    assert session.messages == before
    agent._client.chat.completions.create.assert_not_awaited()


async def test_compaction_replaces_view_with_summary_and_tail(
    project_dir: Path, mock_openai_client
) -> None:
    client = mock_openai_client(content="Key decisions: trained on foo.")
    agent, session = _agent_with_history(project_dir, client, n_messages=30)
    tail_before = session.messages[-4:]

    await agent._compact_context()

    # Compacted view: one summary system message + the last 4 turns.
    assert len(session.messages) == 5
    summary_msg = session.messages[0]
    assert summary_msg["role"] == "system"
    assert "[Context compacted]" in summary_msg["content"]
    assert "Key decisions: trained on foo." in summary_msg["content"]
    assert session.messages[-4:] == tail_before


async def test_compaction_preserves_raw_transcript_on_disk(
    project_dir: Path, mock_openai_client
) -> None:
    """Flipped from Phase 0: compaction only rebuilds the working view —
    the append-only log keeps every raw message, so /resume and feedback
    context retain the full history."""
    client = mock_openai_client(content="summary text")
    agent, session = _agent_with_history(project_dir, client, n_messages=30)

    await agent._compact_context()
    session.save()

    loaded = Session.load(project_dir, session.id)
    raw = loaded.read_log()
    assert len(raw) == 30
    assert any("turn 0:" in m.get("content", "") for m in raw)
    # And the reloaded working view is the checkpoint-assembled compact one.
    assert len(loaded.messages) == 5
    assert "[Context compacted]" in loaded.messages[0]["content"]


async def test_compaction_writes_coverage_aware_checkpoint(
    project_dir: Path, mock_openai_client
) -> None:
    client = mock_openai_client(content="summary text")
    agent, session = _agent_with_history(project_dir, client, n_messages=30)

    await agent._compact_context()

    checkpoint = session.latest_checkpoint()
    assert checkpoint is not None
    assert checkpoint["covers_to_seq"] == 26  # everything but the 4-msg tail
    assert checkpoint["summary"] == "summary text"
    assert checkpoint["model"] == agent.orchestration_model


async def test_second_compaction_folds_in_previous_summary(
    project_dir: Path, mock_openai_client
) -> None:
    client = mock_openai_client(contents=["first summary", "second summary"])
    agent, session = _agent_with_history(project_dir, client, n_messages=30)

    await agent._compact_context()
    for i in range(30, 42):
        role = "user" if i % 2 == 0 else "assistant"
        session.add_message({"role": role, "content": f"turn {i}: " + "x" * 40})
    await agent._compact_context()

    # Second summarizer call received the first summary and only the
    # messages the first checkpoint did not cover.
    second_call = client.chat.completions.create.await_args_list[1].kwargs
    sent_contents = [m.get("content", "") for m in second_call["messages"]]
    assert any("first summary" in c for c in sent_contents)
    assert not any("turn 5:" in c for c in sent_contents)  # already covered
    assert any("turn 27:" in c for c in sent_contents)  # newly covered

    checkpoint = session.latest_checkpoint()
    assert checkpoint["covers_to_seq"] == 38
    assert checkpoint["summary"] == "second summary"
    # Raw log still complete after two compactions.
    assert len(session.read_log()) == 42


async def test_compaction_summarizer_covers_all_uncovered_messages(
    project_dir: Path, mock_openai_client
) -> None:
    """Flipped from Phase 0: the summarizer input is everything since the
    last checkpoint (here: all 26 non-tail messages), not just the last
    20 — nothing is dropped without being summarized."""
    client = mock_openai_client(content="summary text")
    agent, _ = _agent_with_history(project_dir, client, n_messages=30)

    await agent._compact_context()

    call_kwargs = client.chat.completions.create.await_args.kwargs
    sent = call_kwargs["messages"]
    # 1 instruction system message + all 26 messages before the kept tail.
    assert len(sent) == 27
    sent_contents = [m.get("content", "") for m in sent]
    assert any("turn 0:" in c for c in sent_contents)
    assert any("turn 25:" in c for c in sent_contents)
    assert not any("turn 26:" in c for c in sent_contents)  # in the tail


async def test_byte_capped_compaction_never_claims_unsummarized_coverage(
    project_dir: Path, mock_openai_client, monkeypatch
) -> None:
    """When the uncovered window exceeds the byte cap, coverage advances
    only over the chunk that actually entered the summary; later passes
    pick up the rest. Every covered message must have been sent to some
    summarizer call."""
    client = mock_openai_client(contents=[f"summary {i}" for i in range(10)])
    agent, session = _agent_with_history(project_dir, client, n_messages=30)
    # ~55 bytes per message; cap forces several messages per pass.
    monkeypatch.setattr(Agent, "_COMPACTION_INPUT_MAX_BYTES", 300)

    await agent._compact_context()
    first = session.latest_checkpoint()
    assert first is not None
    assert first["covers_to_seq"] < 26  # cap prevented full coverage

    # The uncovered remainder is still in the working view (not lost).
    view_contents = [m.get("content", "") for m in session.messages]
    next_uncovered = f"turn {first['covers_to_seq']}:"  # seq N+1 == turn N
    assert any(next_uncovered in c for c in view_contents)

    # Later passes keep advancing coverage (until the small-view guard
    # makes further compaction unnecessary).
    for _ in range(20):
        before = session.latest_checkpoint()["covers_to_seq"]
        await agent._compact_context()
        if session.latest_checkpoint()["covers_to_seq"] == before:
            break
    final_covered = session.latest_checkpoint()["covers_to_seq"]
    assert final_covered > first["covers_to_seq"]

    # The invariant: every covered message entered some summarizer call.
    sent = "\n".join(
        m.get("content", "")
        for call in client.chat.completions.create.await_args_list
        for m in call.kwargs["messages"]
    )
    for i in range(final_covered):  # seq N == turn N-1
        assert f"turn {i}:" in sent


async def test_single_oversized_message_gets_a_whole_solo_pass(
    project_dir: Path, mock_openai_client, monkeypatch
) -> None:
    """A message larger than the byte cap is summarized WHOLE in its own
    solo pass — coverage must never advance past content no summary
    actually saw (it fit the model's context when originally sent, so it
    fits the summarizer too)."""
    client = mock_openai_client(contents=["s1", "s2"])
    monkeypatch.setattr(Agent, "_COMPACTION_INPUT_MAX_BYTES", 500)
    session2 = Session.create(project_dir)
    agent2 = Agent(project_dir, session2)
    agent2._client = client
    for msg in (
        [{"role": "user", "content": "y" * 5_000}]
        + [
            {"role": "assistant" if i % 2 == 0 else "user", "content": f"turn {i}"}
            for i in range(11)
        ]
    ):
        session2.add_message(msg)

    await agent2._compact_context()

    # Solo pass: only the oversized message (whole) entered the summary,
    # and coverage claims exactly that one message.
    first_call = client.chat.completions.create.await_args_list[0].kwargs
    sent = [m.get("content") or "" for m in first_call["messages"]]
    assert any(c == "y" * 5_000 for c in sent)
    assert not any("turn 0" in c for c in sent)
    assert session2.latest_checkpoint()["covers_to_seq"] == 1

    # The next pass proceeds past it normally.
    await agent2._compact_context()
    assert session2.latest_checkpoint()["covers_to_seq"] > 1


async def test_compaction_boundary_never_splits_tool_call_groups(
    project_dir: Path, mock_openai_client
) -> None:
    """A coverage boundary that would land inside an assistant-tool_calls /
    tool-results group is walked back so the retained history stays
    API-valid."""
    client = mock_openai_client(content="summary text")
    session = Session.create(project_dir)
    agent = Agent(project_dir, session)
    agent._client = client

    tool_calls = [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}}
        for i in range(4)
    ]
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call_0", "content": "r0"},
        {"role": "tool", "tool_call_id": "call_1", "content": "r1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "r2"},
        {"role": "tool", "tool_call_id": "call_3", "content": "r3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]
    for m in msgs:
        session.add_message(m)

    await agent._compact_context()

    # Natural boundary (seq 8, inside the tool results) must walk back to
    # seq 5 — the last safe point before the tool-call group.
    checkpoint = session.latest_checkpoint()
    assert checkpoint["covers_to_seq"] == 5

    # The view keeps the whole group: assistant tool_calls immediately
    # followed by its four results.
    roles = [m.get("role") for m in session.messages]
    assert roles == ["system", "assistant", "tool", "tool", "tool", "tool", "user", "assistant"]
    assert session.messages[1].get("tool_calls")

    # And no orphaned tool message entered the summarizer input.
    sent = client.chat.completions.create.await_args.kwargs["messages"]
    assert all(m.get("role") != "tool" for m in sent)


async def test_compaction_failure_leaves_messages_intact(
    project_dir: Path, mock_openai_client
) -> None:
    client = mock_openai_client()
    client.chat.completions.create = AsyncMock(side_effect=ValueError("boom"))
    agent, session = _agent_with_history(project_dir, client, n_messages=30)
    before = list(session.messages)

    await agent._compact_context()  # must not raise

    assert session.messages == before


async def test_compaction_empty_summary_is_noop(
    project_dir: Path, mock_openai_client
) -> None:
    client = mock_openai_client(content="")
    agent, session = _agent_with_history(project_dir, client, n_messages=30)
    before = list(session.messages)

    await agent._compact_context()

    assert session.messages == before
