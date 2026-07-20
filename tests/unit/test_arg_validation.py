"""Schema validation of tool-call arguments (lqh/tools/validation.py)."""

from __future__ import annotations

from lqh.tools.definitions import get_all_tools
from lqh.tools.validation import validate_args


def _schema_for(name: str) -> dict:
    for tool in get_all_tools(auto_mode=True, include_meta=True):
        if tool["function"]["name"] == name:
            return tool["function"]["parameters"]
    raise AssertionError(f"no such tool: {name}")


SIMPLE = {
    "type": "object",
    "properties": {
        "script_path": {"type": "string"},
        "num_samples": {"type": "integer"},
        "ratio": {"type": "number"},
        "overwrite": {"type": "boolean"},
        "mode": {"type": "string", "enum": ["local", "cloud"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["script_path", "num_samples"],
}


def test_valid_args_pass() -> None:
    assert validate_args(
        {"script_path": "data_gen/x.py", "num_samples": 3}, SIMPLE
    ) == []


def test_missing_required() -> None:
    errors = validate_args({"script_path": "x"}, SIMPLE)
    assert any("num_samples" in e for e in errors)


def test_type_mismatches() -> None:
    errors = validate_args(
        {"script_path": 5, "num_samples": "three"}, SIMPLE
    )
    assert len(errors) == 2
    # bool is NOT a valid integer...
    assert validate_args({"script_path": "x", "num_samples": True}, SIMPLE)
    # ...but an int IS a valid number.
    assert validate_args(
        {"script_path": "x", "num_samples": 3, "ratio": 2}, SIMPLE
    ) == []


def test_enum_rejects_unknown_value() -> None:
    errors = validate_args(
        {"script_path": "x", "num_samples": 3, "mode": "warp"}, SIMPLE
    )
    assert any("mode" in e and "warp" in e for e in errors)


def test_unknown_top_level_key_rejected() -> None:
    errors = validate_args(
        {"script_path": "x", "num_samples": 3, "num_sample": 5}, SIMPLE
    )
    assert any("unknown argument 'num_sample'" in e for e in errors)


def test_nested_items_validated() -> None:
    errors = validate_args(
        {"script_path": "x", "num_samples": 3, "tags": ["a", 7]}, SIMPLE
    )
    assert any("tags[1]" in e for e in errors)


def test_non_object_args_rejected() -> None:
    assert validate_args(["not", "a", "dict"], SIMPLE)  # type: ignore[arg-type]


def test_oneof_accepts_any_branch() -> None:
    schema = {
        "type": "object",
        "properties": {
            "dataset": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            }
        },
        "required": ["dataset"],
    }
    assert validate_args({"dataset": "d1"}, schema) == []
    assert validate_args({"dataset": ["d1", "d2"]}, schema) == []
    errors = validate_args({"dataset": 42}, schema)
    assert any("allowed forms" in e for e in errors)


def test_real_start_training_schema() -> None:
    schema = _schema_for("start_training")
    errors = validate_args({"type": "sft"}, schema)
    # Missing base_model/dataset at minimum.
    assert errors
    assert any("base_model" in e for e in errors)


def test_validator_runs_over_all_real_schemas() -> None:
    for tool in get_all_tools(auto_mode=True, include_meta=True):
        # Must never raise, whatever the schema shape.
        validate_args({}, tool["function"]["parameters"])
