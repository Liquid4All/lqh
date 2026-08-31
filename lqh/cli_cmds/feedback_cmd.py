"""`lqh feedback` — submit feedback without the TUI.

Headless counterpart of the TUI's `/feedback` (FEEDBACK.md): unattended
and agent-driven runs need to report a defect from the surface they
actually exercise. Human-readable notes on stderr; exactly one
machine-readable JSON result on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PRIVACY_NOTICE = (
    "Your feedback, the attached conversation — including system and tool "
    "messages — and a snapshot of your environment (OS, CPU/RAM/GPU, Python "
    "and package versions) will be sent to the lqh team for review. By "
    "submitting, you agree to our privacy policy: https://lqh.ai/privacy/"
)


def _result(
    status: str,
    *,
    ok: bool,
    message: str | None = None,
    session_id: str | None = None,
    context_messages: int | None = None,
) -> str:
    payload: dict = {"schema_version": 1, "ok": ok, "status": status}
    if session_id is not None:
        payload["session_id"] = session_id
    if context_messages is not None:
        payload["context_messages"] = context_messages
    if message is not None:
        payload["message"] = message
    return json.dumps(payload)


class _UsageError(Exception):
    """Caller passed a contradictory combination of flags."""


def _read_message(ns: argparse.Namespace) -> str:
    if ns.message_file is not None:
        if ns.message:
            # The text is the whole payload — never silently drop half of it.
            raise _UsageError(
                "pass either a message string or --message-file, not both"
            )
        if ns.message_file == "-":
            return sys.stdin.read()
        return Path(ns.message_file).read_text(encoding="utf-8")
    if ns.message == "-":
        return sys.stdin.read()
    return ns.message or ""


def _collect_context(ns: argparse.Namespace, project_dir: Path):
    """Resolve (session_id, transcript) for the submission.

    `--no-context` sends none; `--session` picks one (full id or unique
    prefix, as everywhere else); otherwise the project's most recent
    session is attached — that is the run the caller just made.
    """
    from lqh.session import Session

    if ns.no_context:
        if ns.session:
            raise _UsageError("--session and --no-context are contradictory")
        return None, []

    if ns.session:
        session_id = Session.resolve_id(project_dir, ns.session)
    else:
        sessions = Session.list_sessions(project_dir)
        if not sessions:
            return None, []
        session_id = sessions[0]["id"]

    session = Session.load(project_dir, session_id)
    return session_id, session.read_log()


def cmd_feedback(ns: argparse.Namespace) -> int:
    import httpx

    from lqh.auth import LoginError, get_token, send_feedback

    project_dir = Path(ns.project).resolve() if ns.project else Path.cwd()
    if not project_dir.is_dir():
        print(_result("error", ok=False, message=f"--project is not a directory: {ns.project}"))
        return 2

    try:
        message = _read_message(ns).strip()
    except _UsageError as e:
        print(_result("error", ok=False, message=str(e)))
        return 2
    except OSError as e:
        print(_result("error", ok=False, message=f"cannot read the message: {e}"))
        return 2
    if not message:
        print(_result("error", ok=False, message="feedback message is required"))
        return 2

    if not get_token():
        print(_result("error", ok=False, message="not logged in — run `lqh login` first"))
        return 4

    try:
        session_id, context = _collect_context(ns, project_dir)
    except _UsageError as e:
        print(_result("error", ok=False, message=str(e)))
        return 2
    except Exception as e:  # noqa: BLE001 - a bad/unreadable session
        if ns.session:
            print(_result("error", ok=False, message=f"cannot read session {ns.session}: {e}"))
            return 2
        # Nothing the caller asked for: send the feedback without a
        # transcript rather than losing the report.
        print(f"warning: could not attach a conversation ({e})", file=sys.stderr)
        session_id, context = None, []

    print(PRIVACY_NOTICE, file=sys.stderr)
    if session_id:
        print(
            f"Attaching conversation {session_id[:8]} ({len(context)} messages).",
            file=sys.stderr,
        )

    try:
        asyncio.run(send_feedback(message, context, session_id))
    except KeyboardInterrupt:
        print(_result("error", ok=False, message="interrupted"))
        return 6
    except httpx.TransportError as e:
        print(_result("error", ok=False, message=f"could not reach the lqh server: {e}"))
        return 1
    except LoginError as e:
        # send_feedback raises LoginError for every non-2xx; only a
        # credential rejection is an auth (exit 4) outcome.
        detail = str(e)
        rejected = "(401)" in detail or "(403)" in detail
        print(_result("error", ok=False, message=detail))
        return 4 if rejected else 1
    except Exception as e:  # noqa: BLE001 - network and friends
        print(_result("error", ok=False, message=f"{type(e).__name__}: {e}"))
        return 1

    print(_result(
        "submitted", ok=True, session_id=session_id, context_messages=len(context)
    ))
    return 0
