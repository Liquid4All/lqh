"""`lqh tool list|schema|call` — direct tool access for harnesses."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lqh.cli_cmds import registry
from lqh.cli_cmds.envelope import (
    EXIT_INTERRUPTED,
    EXIT_USAGE,
    emit,
    error_envelope,
    interpret_result,
    stdout_to_stderr,
)


def cmd_tool(args: argparse.Namespace) -> int:
    if args.tool_command == "list":
        return _cmd_list(json_out=args.json_out)
    if args.tool_command == "schema":
        return _cmd_schema(args.name)
    if args.tool_command == "call":
        return _cmd_call(args)
    raise AssertionError(f"unhandled tool command {args.tool_command!r}")


def _first_sentence(description: str) -> str:
    text = " ".join(description.split())
    for stop in (". ", ".\n"):
        idx = text.find(stop)
        if idx > 0:
            return text[: idx + 1]
    return text[:120] + ("…" if len(text) > 120 else "")


def _cmd_list(*, json_out: bool) -> int:
    from lqh.tools.definitions import METADATA_KEY

    tools = registry.cli_tools()
    if json_out:
        payload = {
            "schema_version": 1,
            "tools": [
                {
                    "name": t["function"]["name"],
                    "description": _first_sentence(t["function"]["description"]),
                    "mutating": t[METADATA_KEY]["mutating"],
                    "needs_auth": t[METADATA_KEY]["needs_auth"],
                }
                for t in tools
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    width = max(len(t["function"]["name"]) for t in tools)
    for t in tools:
        meta = t[METADATA_KEY]
        tags = []
        tags.append("mutating" if meta["mutating"] else "read-only")
        if meta["needs_auth"]:
            tags.append("auth")
        name = t["function"]["name"]
        summary = _first_sentence(t["function"]["description"])
        print(f"{name:<{width}}  [{','.join(tags)}]  {summary}")
    print(
        f"\n{len(tools)} tools. `lqh tool schema <name>` for arguments, "
        "`lqh tool call <name> --args '<json>'` to invoke.",
    )
    return 0


def _cmd_schema(name: str) -> int:
    definition = registry.tool_definition(name)
    if definition is None:
        print(
            f"Unknown or unexposed tool '{name}'. See `lqh tool list`.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    print(json.dumps(definition, indent=2))
    return 0


def _load_args(ns: argparse.Namespace) -> tuple[dict | None, str | None]:
    """Parse --args / --args-file. Returns (args, error_message)."""
    raw: str | None = None
    if ns.args is not None:
        raw = ns.args
    elif ns.args_file is not None:
        if ns.args_file == "-":
            raw = sys.stdin.read()
        else:
            try:
                raw = Path(ns.args_file).read_text(encoding="utf-8")
            except OSError as e:
                return None, f"cannot read --args-file: {e}"
    if raw is None or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"--args is not valid JSON: {e}"
    if not isinstance(parsed, dict):
        return None, f"--args must be a JSON object, got {type(parsed).__name__}"
    return parsed, None


def _boot_gate(tool_name: str, meta: dict, project_dir: Path) -> tuple[dict, int] | None:
    """Identity/copy contract before a mutating or cloud call (CLI_PLAN §4.8).

    Read-only local tools skip all writes: only a passive copy check when
    an identity exists, warning on stderr, never blocking. Mutating/cloud
    calls run the full headless boot and fail closed (exit 5).
    """
    read_only_local = not meta["mutating"] and not meta["needs_auth"]

    if read_only_local:
        from lqh.project_identity import detect_copy

        if (project_dir / ".lqh" / "project.json").exists():
            try:
                if detect_copy(project_dir) == "copied":
                    print(
                        "warning: this project looks like an unresolved copy "
                        "(see `lqh project continue|fork`); read-only call "
                        "proceeding.",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"warning: project identity unreadable: {e}", file=sys.stderr)
        return None

    from lqh.headless import headless_boot

    boot = headless_boot(project_dir)
    if boot.identity_error:
        return error_envelope(
            tool_name,
            "config",
            "Project identity file is corrupt and will NOT be auto-replaced: "
            f"{boot.identity_error}\nFix or remove .lqh/project.json "
            "(this affects cloud-side project attribution), then retry.",
        )
    if boot.copy_status == "copied":
        return error_envelope(
            tool_name,
            "config",
            "This project directory is an unresolved COPY of another lqh "
            "project — running cloud/mutating operations now could bill work "
            "into the original project's cloud namespace.\nResolve it first:\n"
            "  lqh project continue   # keep the original identity here\n"
            "  lqh project fork       # give this copy a fresh identity\n"
            "then retry.",
        )
    return None


async def _execute_call(
    tool_name: str,
    call_args: dict,
    project_dir: Path,
    consent: dict,
    *,
    wait: bool,
):
    """Run the tool; with ``wait`` (training_status only), park on the
    shared JobSupervisor until the run is terminal — scoring watchers and
    cloud data-gen finalization run while parked, exactly as in the TUI."""
    import asyncio
    from contextlib import suppress

    from lqh.tools.handlers import execute_tool

    if not wait:
        return await execute_tool(tool_name, call_args, project_dir, **consent)

    from lqh.jobs import JobSupervisor

    supervisor = JobSupervisor(project_dir, poll_interval=10.0)
    loop_task = asyncio.create_task(supervisor.watch_loop())
    try:
        await supervisor.wait_primed()
        run_name = call_args.get("run_name")
        targets = [run_name] if run_name else None
        if targets:
            print(f"waiting for run {run_name} …", file=sys.stderr)
        else:
            print("waiting for running jobs …", file=sys.stderr)
        notice = await supervisor.wait_for_runs(targets)
        result = await execute_tool(tool_name, call_args, project_dir, **consent)
        if notice:
            result.content = f"{notice}\n\n{result.content}"
        return result
    finally:
        loop_task.cancel()
        with suppress(BaseException):
            await loop_task
        await supervisor.stop_watchers()


def _cmd_call(ns: argparse.Namespace) -> int:
    tool_name = ns.name
    project_dir = Path.cwd()

    call_args, parse_error = _load_args(ns)
    if parse_error is not None:
        envelope, code = error_envelope(tool_name, "validation", parse_error)
        emit(envelope, pretty=ns.pretty)
        return code

    meta = registry.tool_meta(tool_name)
    if meta is None:
        envelope, code = error_envelope(
            tool_name,
            "validation",
            f"Unknown or unexposed tool '{tool_name}'. See `lqh tool list`.",
        )
        emit(envelope, pretty=ns.pretty)
        return code

    wait = getattr(ns, "wait", False)
    if wait and tool_name != "training_status":
        envelope, code = error_envelope(
            tool_name,
            "validation",
            "--wait is only supported for training_status.",
        )
        emit(envelope, pretty=ns.pretty)
        return code

    errors = registry.validate(tool_name, call_args)
    if errors:
        envelope, code = error_envelope(
            tool_name,
            "validation",
            "Invalid arguments:\n" + "\n".join(f"  - {e}" for e in errors),
            details={"errors": errors},
        )
        emit(envelope, pretty=ns.pretty)
        return code

    if meta["needs_auth"]:
        from lqh.auth import get_token

        if not get_token():
            envelope, code = error_envelope(
                tool_name,
                "auth",
                "Not logged in. Run `lqh login` first.",
            )
            emit(envelope, pretty=ns.pretty)
            return code

    gate = _boot_gate(tool_name, meta, project_dir)
    if gate is not None:
        envelope, code = gate
        emit(envelope, pretty=ns.pretty)
        return code

    consent = registry.full_consent_kwargs(call_args)

    import asyncio

    start = time.monotonic()
    with stdout_to_stderr() as real_stdout:
        try:
            result = asyncio.run(
                _execute_call(
                    tool_name, call_args, project_dir, consent, wait=wait
                )
            )
        except KeyboardInterrupt:
            envelope, _ = error_envelope(
                tool_name, "runtime", "interrupted",
                duration_s=time.monotonic() - start,
            )
            emit(envelope, pretty=ns.pretty, fd=real_stdout)
            return EXIT_INTERRUPTED
        except Exception as e:  # noqa: BLE001 - envelope is the contract
            import traceback

            traceback.print_exc(file=sys.stderr)
            envelope, code = error_envelope(
                tool_name,
                "runtime",
                f"{type(e).__name__}: {e}",
                duration_s=time.monotonic() - start,
            )
            emit(envelope, pretty=ns.pretty, fd=real_stdout)
            return code

        envelope, code = interpret_result(
            tool_name,
            result,
            project_dir=project_dir,
            save_secret=ns.save_secret,
            duration_s=time.monotonic() - start,
        )
        emit(envelope, pretty=ns.pretty, fd=real_stdout)
    return code
