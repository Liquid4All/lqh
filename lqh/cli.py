"""Liquid Harness CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


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
            "Liquid Harness — TUI agent for customizing Liquid AI foundation "
            "models into task-specific models."
        ),
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
    return parser


def main() -> None:
    """Main entry point for the lqh CLI."""
    from lqh.tui.app import LqhApp

    args = _build_parser().parse_args()

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

    _configure_logging(project_dir)
    app = LqhApp(project_dir, auto_mode=auto_mode, extra_spec=args.spec)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ lqh error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
