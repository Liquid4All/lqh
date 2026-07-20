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

# lqh-internal tool names → generic phrases for external harnesses.
# Applied ONLY to `lqh docs` output (never to the skill content injected
# into lqh's own agent), outside fenced code blocks, on the backtick-
# quoted or bare word-bounded form. Pipeline tools (run_data_gen_pipeline,
# run_scoring, …) are deliberately NOT mapped — a harness calls those for
# real via `lqh tool call`.
_HARNESS_TOOL_MAP = {
    "create_file": "[your file-create tool]",
    "write_file": "[your file-write tool]",
    "edit_file": "[your file-edit tool]",
    "read_file": "[your file-read tool]",
    "list_files": "[your file-list tool]",
    "show_file": "[show the file to your user]",
    "ask_user": "[ask your user]",
    "load_skill": "[`lqh docs skill <name>`]",
}

_HARNESS_NOTE = (
    "> Rendered for external harnesses: bracketed [...] phrases replace "
    "lqh-internal tool names — use your own equivalents. Pipeline tools "
    "(`run_data_gen_pipeline`, `run_scoring`, `start_training`, ...) are "
    "real and invoked via `lqh tool call <name>`. Pass --raw for the "
    "verbatim skill text.\n"
)


def render_for_harness(content: str) -> str:
    """Generalize lqh-internal tool names for a third-party reader.

    Fence-aware: fenced code blocks are left byte-identical (a pipeline
    example must stay runnable and error-message quotes must stay
    matchable).
    """
    import re

    pattern = re.compile(
        r"`?\b(" + "|".join(map(re.escape, _HARNESS_TOOL_MAP)) + r")\b`?"
    )

    def _sub(match: "re.Match[str]") -> str:
        return _HARNESS_TOOL_MAP[match.group(1)]

    out: list[str] = []
    in_fence = False
    for line in content.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        out.append(line if in_fence else pattern.sub(_sub, line))
    return "".join(out)


def render_data_reference() -> str:
    """Assemble `lqh docs data`: everything a third-party agent needs to
    author a good data-gen pipeline.

    Single-sourced: the technical blocks are extracted from the
    data_generation skill via `<!-- lqh-docs-data:start/end -->` markers
    (the interleaved human-in-the-loop workflow phases stay out), then
    rendered for a harness reader.
    """
    skill_path = (
        Path(__file__).resolve().parent.parent
        / "skills" / "data_generation" / "SKILL.md"
    )
    content = skill_path.read_text(encoding="utf-8")
    blocks: list[str] = []
    rest = content
    while "<!-- lqh-docs-data:start -->" in rest:
        _, rest = rest.split("<!-- lqh-docs-data:start -->", 1)
        block, rest = rest.split("<!-- lqh-docs-data:end -->", 1)
        blocks.append(block.strip("\n"))

    header = (
        "# Authoring a data-generation pipeline (for external agents)\n\n"
        "This is the `lqh.pipeline` technical reference, extracted from "
        "lqh's `data_generation` skill. Write `data_gen/<name>.py` with "
        "your own file tools following the contract below, then execute "
        "it through lqh (never run it directly — lqh provides the client, "
        "engine, validation, and guards):\n\n"
        "```\n"
        "lqh tool call run_data_gen_pipeline --args '{\n"
        '  "script_path": "data_gen/<name>.py", "num_samples": 3,\n'
        '  "output_dataset": "<task>_v1_draft", "purpose": "smoke"\n'
        "}'\n"
        "```\n\n"
        "Smoke-test with num_samples=3 first, inspect ~20 samples with "
        "your user before any large run, and read `lqh docs skill "
        "data_generation` for the full workflow (draft iteration, scorer "
        "creation, filtering).\n"
    )
    return render_for_harness(header + "\n" + "\n\n".join(blocks) + "\n")


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
    if command == "data":
        sys.stdout.write(render_data_reference())
        return 0
    if command == "skill":
        from lqh.skills import load_skill_content

        try:
            content = load_skill_content(args.name)
            if getattr(args, "raw", False):
                # SKILL.md already ends with a newline; write() adds none.
                sys.stdout.write(content)
            else:
                sys.stdout.write(_HARNESS_NOTE + "\n")
                sys.stdout.write(render_for_harness(content))
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
