"""Headless CLI subcommands (CLI_PLAN.md).

Everything in this package follows the same rules: no TUI imports, no
telemetry, lazy imports per command so startup stays in the tens of
milliseconds.
"""
