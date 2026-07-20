"""`lqh hello` / `lqh docs agents|skills|skill <name>`.

`hello` and `docs agents` are one implementation with two names — the
memorable front door for harnesses. Must work with zero project state,
no auth, and no network, in tens of milliseconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "docs" / "agents_guide.md"


def render_agents_guide() -> str:
    from lqh import __version__
    from lqh.tools.definitions import METADATA_KEY, get_all_tools

    rows: list[str] = []
    for tool in get_all_tools(auto_mode=False, include_meta=True):
        meta = tool[METADATA_KEY]
        if not meta["cli"]:
            continue
        func = tool["function"]
        params = func.get("parameters") or {}
        required = params.get("required") or []
        args = ", ".join(required) if required else "—"
        tags = "mutating" if meta["mutating"] else "read-only"
        if meta["needs_auth"]:
            tags += ", auth"
        description = " ".join(func["description"].split())
        stop = description.find(". ")
        if stop > 0:
            description = description[: stop + 1]
        rows.append(f"| `{func['name']}` | {tags} | {args} | {description} |")

    table = (
        "| Tool | Class | Required args | What it does |\n"
        "|---|---|---|---|\n" + "\n".join(rows)
    )
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("@@VERSION@@", __version__).replace(
        "@@TOOL_TABLE@@", table
    )


def cmd_docs(args: argparse.Namespace) -> int:
    command = getattr(args, "docs_command", None)
    if args.command == "hello" or command == "agents":
        print(render_agents_guide())
        return 0
    if command == "skills":
        from lqh.skills import list_available_skills

        for skill in list_available_skills():
            print(f"{skill['name']:<20} {skill['description']}")
        return 0
    if command == "skill":
        from lqh.skills import load_skill_content

        try:
            # Verbatim: SKILL.md already ends with a newline; print() would
            # append a second one.
            sys.stdout.write(load_skill_content(args.name))
        except FileNotFoundError:
            from lqh.skills import list_available_skills

            names = ", ".join(s["name"] for s in list_available_skills())
            print(
                f"Unknown skill '{args.name}'. Available: {names}",
                file=sys.stderr,
            )
            return 2
        return 0
    raise AssertionError(f"unhandled docs command {command!r}")
