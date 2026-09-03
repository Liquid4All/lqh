"""Regression guard for the ``list_skills`` tool (feedback #103).

The ``train`` entry was missing its ``command`` key, so every call to the
tool died with ``Internal error executing list_skills: KeyError: 'command'``
— the agent could not see the skill catalog at all.
"""

from __future__ import annotations

import asyncio


class TestListSkills:
    def test_every_skill_entry_has_a_command(self) -> None:
        from lqh.skills import list_available_skills

        for entry in list_available_skills():
            assert "command" in entry, f"{entry['name']} is missing 'command'"

    def test_handler_lists_every_skill(self) -> None:
        from lqh.skills import list_available_skills
        from lqh.tools.handlers import handle_list_skills

        content = asyncio.run(handle_list_skills()).content
        for entry in list_available_skills():
            assert entry["description"] in content
