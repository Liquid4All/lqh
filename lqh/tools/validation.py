"""Pre-dispatch validation of tool-call arguments against their schema.

Used by the headless CLI surface (`lqh tool call`) so malformed input is
a clean usage error instead of a handler traceback. Supports the subset
of JSON Schema that `lqh/tools/definitions.py` actually uses: ``type``,
``required``, ``enum``, ``items``, nested ``properties``, ``oneOf`` and
``additionalProperties: False``. Unsupported keywords are ignored
(permissive by design — handlers still validate for real); there is no
coercion and no default injection.

One deliberate strictness: unknown TOP-LEVEL keys are errors. Handlers
swallow extras via ``**kwargs``, so a typo'd argument name would
otherwise be silently ignored.
"""

from __future__ import annotations

from typing import Any

# JSON Schema type -> accepted Python types. bool is excluded from
# integer/number because bool subclasses int in Python.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _type_ok(value: Any, expected: str) -> bool:
    accepted = _TYPE_MAP.get(expected)
    if accepted is None:  # unknown type keyword — permissive
        return True
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, accepted)


def _type_name(value: Any) -> str:
    return type(value).__name__


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        for branch in one_of:
            if isinstance(branch, dict) and not _validate_value(value, branch, path):
                return []
        return [f"{path}: matches none of the allowed forms"]

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if not _type_ok(value, expected_type):
            return [f"{path}: expected {expected_type}, got {_type_name(value)}"]
    elif isinstance(expected_type, list):
        if not any(_type_ok(value, t) for t in expected_type if isinstance(t, str)):
            return [
                f"{path}: expected one of {'/'.join(expected_type)}, "
                f"got {_type_name(value)}"
            ]

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        allowed = ", ".join(repr(v) for v in enum)
        errors.append(f"{path}: {value!r} is not one of [{allowed}]")

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                errors.extend(_validate_value(item, items, f"{path}[{i}]"))

    if isinstance(value, dict):
        errors.extend(_validate_object(value, schema, path, closed=None))

    return errors


def _validate_object(
    obj: dict[str, Any],
    schema: dict[str, Any],
    path: str,
    *,
    closed: bool | None,
) -> list[str]:
    """Validate an object's required/properties/extra keys.

    ``closed`` forces the unknown-key policy; ``None`` defers to the
    schema's own ``additionalProperties``.
    """
    errors: list[str] = []
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}

    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if key not in obj:
                errors.append(f"{path}: missing required argument '{key}'")

    if closed is None:
        closed = schema.get("additionalProperties") is False
    if closed:
        for key in obj:
            if key not in properties:
                errors.append(f"{path}: unknown argument '{key}'")

    for key, value in obj.items():
        subschema = properties.get(key)
        if isinstance(subschema, dict):
            errors.extend(_validate_value(value, subschema, f"{path}.{key}"))

    return errors


def validate_args(args: dict[str, Any], schema: dict[str, Any] | None) -> list[str]:
    """Validate a tool-call argument object against a parameter schema.

    Returns human-readable error strings; an empty list means valid.
    """
    if not isinstance(args, dict):
        return [f"args: expected object, got {_type_name(args)}"]
    if not isinstance(schema, dict):
        return []
    return _validate_object(args, schema, "args", closed=True)
