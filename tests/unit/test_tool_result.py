"""Structured ToolResult outcome fields (CLI_PLAN §5.3)."""

from __future__ import annotations

import pytest

from lqh.tools.handlers import ERROR_KINDS, ToolResult


def test_default_toolresult_is_unclassified() -> None:
    result = ToolResult(content="hi")
    assert result.ok is None
    assert result.error_kind is None
    assert result.retryable is False
    assert result.details is None


def test_fail_sets_structured_fields() -> None:
    result = ToolResult.fail("validation", "Error: num_samples must be positive")
    assert result.ok is False
    assert result.error_kind == "validation"
    assert result.retryable is False
    assert result.content == "Error: num_samples must be positive"


def test_fail_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        ToolResult.fail("catastrophe", "Error: boom")


def test_fail_passes_through_extra_fields() -> None:
    result = ToolResult.fail(
        "upstream",
        "Error: timeout",
        retryable=True,
        details={"status": 504},
        permission_key="k",
    )
    assert result.retryable is True
    assert result.details == {"status": 504}
    assert result.permission_key == "k"


def test_error_kinds_frozen_taxonomy() -> None:
    assert ERROR_KINDS == {
        "auth", "permission", "config", "validation",
        "not_found", "conflict", "upstream", "runtime",
    }
