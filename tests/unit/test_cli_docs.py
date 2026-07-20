"""`lqh hello` / `lqh docs` — self-contained, offline, no project state."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lqh import __version__
from lqh.cli import main
from lqh.cli_cmds.docs_cmd import render_agents_guide


def _run(argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def test_hello_equals_docs_agents(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # zero project state
    assert _run(["lqh", "hello"], monkeypatch) == 0
    hello_out = capsys.readouterr().out
    assert _run(["lqh", "docs", "agents"], monkeypatch) == 0
    agents_out = capsys.readouterr().out
    assert hello_out == agents_out


def test_guide_contains_version_and_all_exposed_tools() -> None:
    from tests.unit.test_tool_registry import EXPOSED_TOOLS

    guide = render_agents_guide()
    assert __version__ in guide
    for name in EXPOSED_TOOLS:
        assert f"`{name}`" in guide
    assert "@@" not in guide  # all placeholders substituted
    # Core contract sections present.
    for heading in (
        "act, don't summarize",
        "### Read the stage skill BEFORE doing the stage's work",
        "lqh docs data",
        "## What LQH is",
        "## The fine-tuning workflow",
        "## Consent model",
        "## Contracts",
        "## Worked examples",
        "## Project conventions you must follow",
    ):
        assert heading in guide, heading


def test_docs_skills_lists_skills(monkeypatch, capsys) -> None:
    assert _run(["lqh", "docs", "skills"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert "auto" in out
    assert "spec_capture" in out


def test_docs_skill_raw_prints_verbatim(monkeypatch, capsys) -> None:
    from lqh.skills import load_skill_content

    assert _run(["lqh", "docs", "skill", "auto", "--raw"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert out == load_skill_content("auto")  # verbatim, no extra newline


def test_docs_skill_default_is_harness_rendered(monkeypatch, capsys) -> None:
    assert _run(["lqh", "docs", "skill", "data_generation"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert "Rendered for external harnesses" in out
    # Internal tool names generalized in prose...
    assert "[your file-edit tool]" in out
    assert "[ask your user]" in out
    # ...but the executable pipeline tool names stay literal.
    assert "run_data_gen_pipeline" in out


def test_render_for_harness_skips_code_fences() -> None:
    from lqh.cli_cmds.docs_cmd import render_for_harness

    content = (
        "Use `create_file` to make it.\n"
        "```python\n"
        "# create_file stays literal inside code\n"
        "```\n"
        "Then `ask_user` for feedback.\n"
    )
    rendered = render_for_harness(content)
    assert "[your file-create tool]" in rendered
    assert "[ask your user]" in rendered
    assert "# create_file stays literal inside code" in rendered


def test_docs_data_reference(monkeypatch, capsys) -> None:
    assert _run(["lqh", "docs", "data"], monkeypatch) == 0
    out = capsys.readouterr().out
    # The single-sourced technical blocks are present...
    assert "## Pipeline Interface" in out
    assert "from lqh.pipeline import" in out
    assert "## Common Mistakes to Avoid" in out
    # ...the internal-agent workflow phases are not...
    assert "Phase 1: Draft Iteration" not in out
    # ...markers don't leak, and execution goes through lqh.
    assert "lqh-docs-data" not in out
    assert "lqh tool call run_data_gen_pipeline" in out


def test_docs_unknown_skill_exit_2(monkeypatch, capsys) -> None:
    assert _run(["lqh", "docs", "skill", "nope"], monkeypatch) == 2
    err = capsys.readouterr().err
    assert "Unknown skill 'nope'" in err
    assert "auto" in err  # lists available names
