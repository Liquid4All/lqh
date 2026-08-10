"""The agent must not repurpose a project directory for a second task.

Feedback #38: a user with an established project ("15 datasets, 11 runs")
asked to "start a new project"; the agent's answer wandered because nothing
in its instructions states the one-project-per-directory model or how to
open a second one. These assertions pin the guidance to the surfaces the
agent actually reads.
"""

from __future__ import annotations


class TestSystemPromptProjectScope:
    def test_prompt_states_one_task_per_directory(self) -> None:
        from lqh.agent import SYSTEM_PROMPT

        assert "## One project = one directory" in SYSTEM_PROMPT
        assert "exactly ONE task" in SYSTEM_PROMPT
        # The failure mode to prevent: clobbering an existing spec.
        assert "do NOT overwrite SPEC.md" in SYSTEM_PROMPT

    def test_prompt_gives_both_routes_to_a_new_project(self) -> None:
        from lqh.agent import SYSTEM_PROMPT

        # Shell route.
        assert "/quit" in SYSTEM_PROMPT
        assert "cd ~/my-new-task" in SYSTEM_PROMPT
        # Non-technical route: bare `lqh` in Home offers a project name and
        # creates the folder (lqh/cli.py:_choose_home_project).
        assert "~/lqh-projects/<name>" in SYSTEM_PROMPT


class TestSpecCaptureProjectScope:
    def test_skill_guards_an_existing_spec(self) -> None:
        from lqh.skills import load_skill_content

        content = load_skill_content("spec_capture")
        # /spec in an established project enters this flow directly, so the
        # skill needs its own "is this even the same task?" branch.
        assert "If SPEC.md already exists" in content
        assert "do NOT overwrite the spec" in content
