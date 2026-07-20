"""Thin shim over the tool definitions/permissions for the CLI surface.

All CLI-side knowledge of the phase-1 foundations (metadata layout,
consent kwargs, validation) is concentrated here so a change in those
interfaces is a one-file fix.
"""

from __future__ import annotations

from typing import Any


def cli_tools() -> list[dict]:
    """CLI-exposed tool definitions (with metadata), stable order."""
    from lqh.tools.definitions import METADATA_KEY, get_all_tools

    return [
        t
        for t in get_all_tools(auto_mode=False, include_meta=True)
        if t[METADATA_KEY]["cli"]
    ]


def tool_meta(name: str) -> dict | None:
    """The x-lqh metadata dict for a CLI-exposed tool, or None."""
    from lqh.tools.definitions import METADATA_KEY

    for tool in cli_tools():
        if tool["function"]["name"] == name:
            return tool[METADATA_KEY]
    return None


def tool_definition(name: str) -> dict | None:
    """The full `function` object (name/description/parameters) for a
    CLI-exposed tool, or None."""
    for tool in cli_tools():
        if tool["function"]["name"] == name:
            return tool["function"]
    return None


def validate(name: str, args: dict[str, Any]) -> list[str]:
    """Validate args against the tool's schema; [] means valid."""
    from lqh.tools.validation import validate_args

    definition = tool_definition(name)
    schema = definition.get("parameters") if definition else None
    return validate_args(args, schema)


def full_consent_kwargs(args: dict[str, Any]) -> dict[str, Any]:
    """Consent extra-kwargs for a direct `lqh tool call` (CLI_PLAN §3.2).

    Invocation is consent for the PERMISSION store — but NOT for the
    overwrite guard: ``_overwrite_consent`` is granted only when the
    caller's own args carry ``overwrite: true``. Without that, the guard's
    refusal (error_kind "conflict") must still fire, or a harness could
    destroy an expensive dataset it never mentioned.
    """
    from lqh.tools.permissions import PermissionContext

    consent: dict[str, Any] = {
        "_permissions": PermissionContext(full_consent=True),
    }
    if args.get("overwrite") is True:
        consent["_overwrite_consent"] = True
    return consent
