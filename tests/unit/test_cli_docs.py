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


def test_docs_skill_prints_verbatim(monkeypatch, capsys) -> None:
    from lqh.skills import load_skill_content

    assert _run(["lqh", "docs", "skill", "auto"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert out == load_skill_content("auto")  # verbatim, no extra newline


def test_docs_unknown_skill_exit_2(monkeypatch, capsys) -> None:
    assert _run(["lqh", "docs", "skill", "nope"], monkeypatch) == 2
    err = capsys.readouterr().err
    assert "Unknown skill 'nope'" in err
    assert "auto" in err  # lists available names
