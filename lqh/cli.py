"""Liquid Harness CLI entry point.

Bare ``lqh`` (and ``lqh --auto``) launch the TUI, unchanged. The
subcommands (``hello``, ``docs``, ``tool``, ``login``, ``project``) are
the headless surface for third-party agent harnesses (CLI_PLAN.md):
they never load the TUI, never start telemetry, and keep imports lazy so
``lqh hello`` / ``lqh tool list`` answer in tens of milliseconds.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from lqh import __version__

_HARNESS_EPILOG = """\
for AI agents and harnesses (Claude Code, Codex, ...):
  If you are an AI agent driving lqh programmatically, first run:
      lqh hello
  It explains what LQH is, the full fine-tuning workflow, the headless
  commands (`lqh run`, `lqh tool ...`), their JSON contracts, and the
  project conventions (NOTES.md, manifests, immutable outputs).
"""


def _configure_logging(project_dir: Path) -> None:
    """Route library log output to ``.lqh/lqh.log``.

    The TUI owns the terminal, so any stderr write from a background
    asyncio task (RunWatcher exceptions, scoring retries, etc.) corrupts
    the dataset viewer / status bar. Replace the root handlers with a
    file handler and disable the stderr lastResort to keep the screen
    clean — errors are still inspectable via ``tail -f .lqh/lqh.log``.
    """
    log_dir = project_dir / ".lqh"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return  # read-only filesystem? leave default logging alone

    handler = logging.FileHandler(log_dir / "lqh.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # Without this Python falls back to a stderr StreamHandler when no
    # handler matches a message — defeats the file routing.
    logging.lastResort = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lqh",
        description=(
            "Liquid Harness — agent for customizing Liquid AI foundation "
            "models into task-specific models. Run `lqh` with no arguments "
            "in your project directory to start the interactive TUI."
        ),
        epilog=_HARNESS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the installed lqh version and exit.",
    )
    parser.add_argument(
        "--auto",
        metavar="SPEC_DIR",
        type=Path,
        default=None,
        help=(
            "Run in fully-autonomous auto mode against the spec at "
            "SPEC_DIR/SPEC.md. The agent runs the full pipeline (rubric → "
            "data gen → filter → baseline → SFT → DPO → report) without "
            "user prompts and terminates with success or failure."
        ),
    )
    parser.add_argument(
        "--spec",
        metavar="STRING",
        default=None,
        help=(
            "Extra system context appended to every turn (sticky, survives "
            "compaction). Useful for run-time directives such as 'use the "
            "smallest base model' that don't belong in SPEC.md. Works in "
            "both interactive and auto mode."
        ),
    )

    sub = parser.add_subparsers(
        dest="command", title="commands", metavar="[command]"
    )

    sub.add_parser(
        "hello",
        help="Print the guide for AI agents driving lqh. Start here.",
        description=(
            "Print the harness-facing guide (identical to `lqh docs agents`): "
            "what LQH is, the workflow, the headless commands and their JSON "
            "contracts."
        ),
    )

    login = sub.add_parser(
        "login",
        help="Authenticate via device flow (works without the TUI).",
        description=(
            "Device-flow login. Prints the verification URL and code on "
            "stderr and one machine-readable JSON result on stdout."
        ),
    )
    login.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not try to open the verification URL in a browser.",
    )

    docs = sub.add_parser(
        "docs",
        help="Print docs: skills, and the agent-harness guide.",
    )
    docs_sub = docs.add_subparsers(dest="docs_command", metavar="<what>")
    docs_sub.required = True
    docs_sub.add_parser("agents", help="Full harness-facing guide (markdown).")
    docs_sub.add_parser("skills", help="List available skills.")
    docs_skill = docs_sub.add_parser("skill", help="Print a skill's SKILL.md verbatim.")
    docs_skill.add_argument("name", help="Skill name (see `lqh docs skills`).")

    tool = sub.add_parser(
        "tool",
        help="List / inspect / call individual pipeline tools (JSON).",
    )
    tool_sub = tool.add_subparsers(dest="tool_command", metavar="<action>")
    tool_sub.required = True
    tool_list = tool_sub.add_parser("list", help="List CLI-exposed tools.")
    tool_list.add_argument(
        "--json", action="store_true", dest="json_out",
        help="Machine-readable JSON output.",
    )
    tool_schema = tool_sub.add_parser(
        "schema", help="Print a tool's JSON schema."
    )
    tool_schema.add_argument("name", help="Tool name (see `lqh tool list`).")
    tool_call = tool_sub.add_parser(
        "call", help="Call a tool; JSON envelope on stdout."
    )
    tool_call.add_argument("name", help="Tool name (see `lqh tool list`).")
    args_group = tool_call.add_mutually_exclusive_group()
    args_group.add_argument(
        "--args",
        metavar="JSON",
        help="Tool arguments as a JSON object (same shape the agent emits).",
    )
    args_group.add_argument(
        "--args-file",
        metavar="FILE",
        help="Read the JSON arguments from FILE ('-' for stdin).",
    )
    tool_call.add_argument(
        "--pretty", action="store_true", help="Pretty-print the envelope."
    )
    tool_call.add_argument(
        "--save-secret",
        action="store_true",
        help="Persist a delivered secret into the project's .env file.",
    )
    tool_call.add_argument(
        "--wait",
        action="store_true",
        help=(
            "training_status only: park until the run reaches a terminal "
            "state (scoring results included) before returning."
        ),
    )

    run = sub.add_parser(
        "run",
        help="Run one delegated task headlessly; JSON result on stdout.",
        description=(
            "Headless sub-agent mode: lqh's agent performs one delegated "
            "task with no user interaction, then exits with a structured "
            "JSON result on stdout (NDJSON progress events on stderr). "
            "Publishing tools (hf_push, push_to_production, "
            "create_inference_key) are gated behind --allow-publish."
        ),
    )
    run.add_argument(
        "task", nargs="?", default=None,
        help="The task prompt ('-' reads it from stdin).",
    )
    run.add_argument(
        "--prompt-file", metavar="FILE", default=None,
        help="Read the task prompt from FILE ('-' for stdin).",
    )
    run.add_argument(
        "--resume", metavar="SESSION_ID", default=None,
        help=(
            "Contextual resume of a prior run's session: the task (or a "
            "default continue instruction) is injected as a new message."
        ),
    )
    run.add_argument(
        "--allow-publish", action="store_true",
        help="Permit outward-facing publishing tools in this run.",
    )
    run.add_argument(
        "--max-turns", type=int, default=None, metavar="N",
        help="Abort with limit_exceeded after N LLM calls.",
    )
    run.add_argument(
        "--max-tool-calls", type=int, default=None, metavar="N",
        help="Abort with limit_exceeded after N tool calls (total).",
    )
    run.add_argument(
        "--save-secret", action="store_true",
        help="Persist delivered secrets to .env instead of (only) the result payload.",
    )
    run.add_argument(
        "--quiet", action="store_true",
        help="Suppress NDJSON progress events on stderr.",
    )
    run.add_argument(
        "--spec", metavar="STRING", default=None,
        help="Extra sticky context appended to every agent turn.",
    )

    status = sub.add_parser(
        "status",
        help="Show run states and attention signals for this project.",
    )
    status.add_argument(
        "--json", action="store_true", dest="json_out",
        help="Machine-readable JSON output.",
    )

    project = sub.add_parser(
        "project",
        help="Resolve a copied project: continue or fork identity.",
    )
    project_sub = project.add_subparsers(dest="project_command", metavar="<action>")
    project_sub.required = True
    project_sub.add_parser(
        "continue",
        help="Keep this copy attached to the original project identity.",
    )
    project_sub.add_parser(
        "fork",
        help="Give this copy a fresh identity and cloud namespace.",
    )

    return parser


def _launch_tui(project_dir: Path, auto_mode: bool, extra_spec: str | None) -> None:
    """Start the interactive TUI (the pre-subcommand `lqh` behavior)."""
    # Keep lightweight commands such as --help and --version from importing
    # the full TUI dependency graph.
    from lqh.tui.app import LqhApp

    _configure_logging(project_dir)
    app = LqhApp(project_dir, auto_mode=auto_mode, extra_spec=extra_spec)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ lqh error: {e}", file=sys.stderr)
        sys.exit(1)


def _dispatch(args: argparse.Namespace) -> int:
    """Route a headless subcommand. Imports are per-branch and lazy —
    never the TUI, never telemetry (lqh.telemetry.notice_needed has a
    marker-writing side effect and must not run on these paths)."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    if args.command in ("hello", "docs"):
        from lqh.cli_cmds.docs_cmd import cmd_docs

        return cmd_docs(args)
    if args.command == "login":
        from lqh.cli_cmds.login_cmd import cmd_login

        return cmd_login(args)
    if args.command == "tool":
        from lqh.cli_cmds.tool_cmd import cmd_tool

        return cmd_tool(args)
    if args.command == "run":
        from lqh.cli_cmds.run_cmd import cmd_run

        return cmd_run(args)
    if args.command == "status":
        from lqh.cli_cmds.status_cmd import cmd_status

        return cmd_status(args)
    if args.command == "project":
        from lqh.cli_cmds.project_cmd import cmd_project

        return cmd_project(args)
    raise AssertionError(f"unhandled command {args.command!r}")


def main() -> None:
    """Main entry point for the lqh CLI."""
    args = _build_parser().parse_args()

    if getattr(args, "command", None) is not None:
        sys.exit(_dispatch(args))

    if args.auto is not None:
        spec_dir = args.auto.resolve()
        if not spec_dir.is_dir():
            print(
                f"❌ --auto requires a directory, got: {args.auto}",
                file=sys.stderr,
            )
            sys.exit(2)
        if not (spec_dir / "SPEC.md").is_file():
            print(
                f"❌ --auto: SPEC.md not found in {spec_dir}. Auto mode "
                "requires a spec; run lqh interactively to create one first.",
                file=sys.stderr,
            )
            sys.exit(2)
        project_dir = spec_dir
        auto_mode = True
    else:
        project_dir = Path.cwd()
        auto_mode = False

    _launch_tui(project_dir, auto_mode, args.spec)


if __name__ == "__main__":
    main()
