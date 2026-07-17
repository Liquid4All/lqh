"""Tests for Agent.prepare_context() (ephemeral project context).

Flipped from the Phase 0 characterization suite (see PERSISTENCY_PLAN.md):
context injections (SPEC.md, NOTES.md, summary, project log) now live in
``agent.context_messages`` — rebuilt from disk on every open, never
persisted into the conversation, and therefore impossible to duplicate on
resume.
"""

from __future__ import annotations

from pathlib import Path

from lqh.agent import Agent
from lqh.project_log import append_event
from lqh.session import Session


def _make_agent(project_dir: Path) -> Agent:
    return Agent(project_dir, Session.create(project_dir))


async def test_new_project_loads_spec_capture_skill(project_dir: Path) -> None:
    agent = _make_agent(project_dir)

    mode = await agent.prepare_context()

    assert mode == "new_project"
    assert len(agent.context_messages) >= 1
    assert agent.session.messages == []


async def test_existing_project_injects_spec_summary_and_log(
    project_dir: Path,
) -> None:
    (project_dir / "SPEC.md").write_text("# My spec\nTrain a triage model.\n")
    append_event(project_dir, "data_gen_completed", "generated foo dataset")

    agent = _make_agent(project_dir)
    mode = await agent.prepare_context()

    assert mode == "existing_project"
    contents = [m["content"] for m in agent.context_messages]
    assert any("Train a triage model." in c for c in contents)
    assert any("Current project state:" in c for c in contents)
    assert any("Project activity log" in c for c in contents)
    # The first context message labels the injection with its freshness.
    assert contents[0].startswith("Project context as of ")


async def test_missing_spec_still_injects_notes_and_summary(
    project_dir: Path,
) -> None:
    """A project with artifacts but no SPEC.md (deleted, or never written)
    must still brief the agent on notes and existing work."""
    (project_dir / "NOTES.md").write_text("spec was removed; rebuilding it\n")
    run = project_dir / "runs" / "sft_v1"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")

    agent = _make_agent(project_dir)
    mode = await agent.prepare_context()

    assert mode == "new_project"
    contents = [m["content"] for m in agent.context_messages]
    assert any("spec was removed" in c for c in contents)
    assert any("Current project state:" in c for c in contents)


async def test_summary_failure_does_not_abort_context_preparation(
    project_dir: Path, monkeypatch,
) -> None:
    """A bug or malformed artifact inside handle_summary must degrade the
    injected summary, never abort startup//clear//resume preparation."""
    (project_dir / "SPEC.md").write_text("# spec\n")

    async def broken_summary(_project_dir, **_kwargs):
        raise AttributeError("'NoneType' object has no attribute 'get'")

    monkeypatch.setattr("lqh.tools.handlers.handle_summary", broken_summary)

    agent = _make_agent(project_dir)
    mode = await agent.prepare_context()

    assert mode == "existing_project"
    contents = [m["content"] for m in agent.context_messages]
    assert any("summary unavailable" in c for c in contents)
    assert any("# spec" in c for c in contents)  # the rest still injected


async def test_notes_md_is_injected_when_present(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# spec\n")
    (project_dir / "NOTES.md").write_text(
        "Decided to use pipeline v2; next step is scoring.\n"
    )

    agent = _make_agent(project_dir)
    await agent.prepare_context()

    contents = [m["content"] for m in agent.context_messages]
    notes = [c for c in contents if "Agent notes (NOTES.md" in c]
    assert len(notes) == 1
    assert "pipeline v2" in notes[0]
    assert "advisory" in notes[0]


async def test_oversized_notes_are_truncated(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# spec\n")
    (project_dir / "NOTES.md").write_text("x" * 50_000)

    agent = _make_agent(project_dir)
    await agent.prepare_context()

    notes = [
        m["content"]
        for m in agent.context_messages
        if "Agent notes (NOTES.md" in m["content"]
    ][0]
    assert len(notes) < 25_000
    assert "[truncated — read NOTES.md for the rest]" in notes


async def test_injections_are_ephemeral(project_dir: Path) -> None:
    """Flipped from Phase 0: nothing is persisted — the conversation stays
    empty and context lives only on the agent."""
    (project_dir / "SPEC.md").write_text("# spec\n")

    agent = _make_agent(project_dir)
    await agent.prepare_context()

    assert agent.session.messages == []
    assert agent.session.read_log() == []
    assert len(agent.context_messages) >= 2


async def test_double_prepare_rebuilds_instead_of_duplicating(
    project_dir: Path,
) -> None:
    """Flipped from Phase 0: prepare_context() replaces the previous
    context wholesale, so repeated opens can never accumulate stale
    copies."""
    (project_dir / "SPEC.md").write_text("# spec\n")

    agent = _make_agent(project_dir)
    await agent.prepare_context()
    count_after_first = len(agent.context_messages)
    await agent.prepare_context()

    assert len(agent.context_messages) == count_after_first
    assert agent.session.messages == []


async def test_prepare_context_reflects_current_disk_state(
    project_dir: Path,
) -> None:
    """A spec edited between opens is what the next open sees — the
    ephemeral prefix always reflects the present filesystem."""
    (project_dir / "SPEC.md").write_text("# spec\noriginal requirement\n")
    agent = _make_agent(project_dir)
    await agent.prepare_context()

    (project_dir / "SPEC.md").write_text("# spec\nedited requirement\n")
    await agent.prepare_context()

    contents = [m["content"] for m in agent.context_messages]
    assert any("edited requirement" in c for c in contents)
    assert not any("original requirement" in c for c in contents)


async def test_build_messages_orders_system_sticky_context_history(
    project_dir: Path,
) -> None:
    (project_dir / "SPEC.md").write_text("# spec\n")
    agent = _make_agent(project_dir)
    agent.sticky_system_messages.append("sticky context")
    await agent.prepare_context()
    agent.session.messages.append({"role": "user", "content": "hi"})

    built = agent._build_messages()

    assert built[0]["role"] == "system"  # SYSTEM_PROMPT
    assert built[1] == {"role": "system", "content": "sticky context"}
    assert built[2]["content"].startswith("Project context as of ")
    assert built[-1] == {"role": "user", "content": "hi"}
