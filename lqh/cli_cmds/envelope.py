"""JSON envelope, exit codes, and sentinel interpretation for `lqh tool`.

The envelope is the CLI's public contract (CLI_PLAN §5.4); `lqh run`
(phase 5) reuses `interpret_result` so both surfaces classify tool
outcomes identically. Bump ENVELOPE_SCHEMA_VERSION only on breaking
change.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from lqh.tools.handlers import ToolResult

ENVELOPE_SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_PERMISSION = 3
EXIT_AUTH = 4
EXIT_CONFIG = 5
EXIT_INTERRUPTED = 6

# error_kind -> exit code; kinds not listed exit 1.
_EXIT_BY_KIND = {
    "validation": EXIT_USAGE,
    "permission": EXIT_PERMISSION,
    "auth": EXIT_AUTH,
    "config": EXIT_CONFIG,
}


def exit_code_for_kind(kind: str | None) -> int:
    return _EXIT_BY_KIND.get(kind or "", EXIT_FAILURE)


def build_envelope(
    *,
    tool: str,
    ok: bool,
    text: str = "",
    secret: str | None = None,
    details: dict | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
    retryable: bool = False,
    error_details: dict | None = None,
    duration_s: float = 0.0,
    classified: bool = True,
) -> dict:
    from lqh import __version__

    meta: dict[str, Any] = {
        "duration_s": round(duration_s, 3),
        "lqh_version": __version__,
    }
    if not classified:
        meta["classified"] = False
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "ok": ok,
        "tool": tool,
        "result": (
            {"text": text, "secret": secret, "details": details or {}}
            if ok
            else None
        ),
        "error": (
            None
            if ok
            else {
                "kind": error_kind,
                "message": error_message if error_message is not None else text,
                "retryable": retryable,
                "details": error_details or {},
            }
        ),
        "meta": meta,
    }


def error_envelope(
    tool: str,
    kind: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict | None = None,
    duration_s: float = 0.0,
) -> tuple[dict, int]:
    envelope = build_envelope(
        tool=tool,
        ok=False,
        error_kind=kind,
        error_message=message,
        retryable=retryable,
        error_details=details,
        duration_s=duration_s,
    )
    return envelope, exit_code_for_kind(kind)


def emit(envelope: dict, *, pretty: bool = False, fd: int | None = None) -> None:
    """Write the envelope as exactly one JSON document.

    ``fd`` is the saved real-stdout file descriptor from
    ``stdout_to_stderr`` — used when sys.stdout has been redirected.
    """
    text = json.dumps(envelope, indent=2 if pretty else None, default=str)
    if fd is not None:
        os.write(fd, (text + "\n").encode())
    else:
        print(text)


@contextmanager
def stdout_to_stderr() -> Iterator[int]:
    """Redirect fd 1 (and sys.stdout) to stderr for the duration.

    Handlers and the subprocesses they spawn inherit fd 1, so a
    Python-level ``sys.stdout`` swap is not enough to keep the
    one-JSON-object stdout contract. Yields the saved real-stdout fd for
    ``emit(..., fd=saved)``.
    """
    saved = os.dup(1)
    try:
        sys.stdout.flush()
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        yield saved
    finally:
        sys.stdout = sys.__stdout__
        os.dup2(saved, 1)
        os.close(saved)


def interpret_result(
    tool_name: str,
    result: "ToolResult",
    *,
    project_dir: Path,
    save_secret: bool = False,
    duration_s: float = 0.0,
) -> tuple[dict, int]:
    """Map a ToolResult (incl. interactive sentinels) to (envelope, exit code)."""
    from lqh.tools.handlers import (
        COMPUTE_PICK_REQUIRED,
        SECRET_DELIVERY_REQUIRED,
    )

    content = result.content

    # One-time secret (e.g. a freshly minted inference key): plaintext
    # goes into result.secret — the caller's transcript is the delivery
    # channel on this surface (documented, intentional).
    if content == SECRET_DELIVERY_REQUIRED and result.secret is not None:
        delivery = result.secret
        text = delivery.redacted
        if save_secret:
            from lqh.env_secrets import append_env_secret

            note = append_env_secret(
                project_dir,
                delivery.env_var,
                delivery.payload,
                delivery.env_comment,
            )
            text = f"{text}\n{note}"
        envelope = build_envelope(
            tool=tool_name,
            ok=True,
            text=text,
            secret=delivery.payload,
            details={"env_var": delivery.env_var},
            duration_s=duration_s,
        )
        return envelope, EXIT_OK

    if content == COMPUTE_PICK_REQUIRED:
        options = result.options or []
        hint = (
            "No compute target is pinned for this project. Pick one with:\n"
            "  lqh tool call compute_set --args "
            "'{\"value\": \"<cloud|local|ssh:name>\", \"scope\": \"project\"}'\n"
            + (f"Options: {', '.join(options)}" if options else "")
        )
        message = f"{result.question or 'Compute target required.'}\n{hint}"
        return error_envelope(
            tool_name, "config", message, duration_s=duration_s
        )

    if content == "PERMISSION_REQUIRED":
        # Defensive: unreachable under full consent (CLI_PLAN §3.2).
        return error_envelope(
            tool_name,
            "permission",
            result.question or "Permission required.",
            details={"permission_key": result.permission_key},
            duration_s=duration_s,
        )

    if result.requires_user_input:
        # Includes OVERWRITE_CONFIRMATION_REQUIRED (only reachable when the
        # caller passed overwrite=true but consent threading failed) and any
        # future interactive sentinel: this surface has no user to ask.
        return error_envelope(
            tool_name,
            "validation",
            "tool requested interactive input; not available on this surface"
            + (f": {result.question}" if result.question else ""),
            duration_s=duration_s,
        )

    if result.ok is True:
        envelope = build_envelope(
            tool=tool_name,
            ok=True,
            text=content,
            details=result.details,
            duration_s=duration_s,
        )
        return envelope, EXIT_OK

    if result.ok is False:
        kind = result.error_kind or "runtime"
        envelope = build_envelope(
            tool=tool_name,
            ok=False,
            error_kind=kind,
            error_message=content,
            retryable=result.retryable,
            error_details=result.details,
            duration_s=duration_s,
        )
        return envelope, exit_code_for_kind(kind)

    # Legacy/unclassified result: best-effort prefix sniff (CLI_PLAN §5.3).
    is_error = content.startswith("Error:") or content.startswith("❌")
    if is_error:
        kind = "conflict" if "refusing to overwrite" in content else "runtime"
        envelope = build_envelope(
            tool=tool_name,
            ok=False,
            error_kind=kind,
            error_message=content,
            duration_s=duration_s,
            classified=False,
        )
        return envelope, EXIT_FAILURE
    envelope = build_envelope(
        tool=tool_name,
        ok=True,
        text=content,
        duration_s=duration_s,
        classified=False,
    )
    return envelope, EXIT_OK
