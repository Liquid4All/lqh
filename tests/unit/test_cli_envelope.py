"""Envelope shape, exit-code mapping, and sentinel interpretation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from lqh import __version__
from lqh.cli_cmds.envelope import (
    ENVELOPE_SCHEMA_VERSION,
    build_envelope,
    error_envelope,
    exit_code_for_kind,
    interpret_result,
    stdout_to_stderr,
)
from lqh.tools.handlers import (
    COMPUTE_PICK_REQUIRED,
    SECRET_DELIVERY_REQUIRED,
    SecretDelivery,
    ToolResult,
)


def test_ok_envelope_shape() -> None:
    env = build_envelope(tool="summary", ok=True, text="hi", duration_s=1.234)
    assert env == {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "ok": True,
        "tool": "summary",
        "result": {"text": "hi", "secret": None, "details": {}},
        "error": None,
        "meta": {"duration_s": 1.234, "lqh_version": __version__},
    }


def test_error_envelope_shape_and_codes() -> None:
    env, code = error_envelope("t", "auth", "Not logged in.")
    assert env["ok"] is False
    assert env["result"] is None
    assert env["error"] == {
        "kind": "auth",
        "message": "Not logged in.",
        "retryable": False,
        "details": {},
    }
    assert code == 4


def test_exit_code_table() -> None:
    assert exit_code_for_kind("validation") == 2
    assert exit_code_for_kind("permission") == 3
    assert exit_code_for_kind("auth") == 4
    assert exit_code_for_kind("config") == 5
    for kind in ("not_found", "conflict", "upstream", "runtime", None):
        assert exit_code_for_kind(kind) == 1


def _interpret(result: ToolResult, tmp_path: Path, **kwargs):
    return interpret_result("t", result, project_dir=tmp_path, **kwargs)


def test_classified_failure(tmp_path: Path) -> None:
    result = ToolResult.fail("not_found", "Error: no data.parquet in 'd'")
    env, code = _interpret(result, tmp_path)
    assert env["ok"] is False
    assert env["error"]["kind"] == "not_found"
    assert code == 1
    assert "classified" not in env["meta"]


def test_classified_success(tmp_path: Path) -> None:
    result = ToolResult(content="done", ok=True)
    env, code = _interpret(result, tmp_path)
    assert env["ok"] is True and code == 0


def test_legacy_error_prefix_sniff(tmp_path: Path) -> None:
    env, code = _interpret(ToolResult(content="Error: boom"), tmp_path)
    assert env["ok"] is False and code == 1
    assert env["meta"]["classified"] is False

    env, code = _interpret(ToolResult(content="❌ nope"), tmp_path)
    assert env["ok"] is False and code == 1

    env, code = _interpret(ToolResult(content="all good"), tmp_path)
    assert env["ok"] is True and code == 0
    assert env["meta"]["classified"] is False


def test_secret_delivery(tmp_path: Path) -> None:
    result = ToolResult(
        content=SECRET_DELIVERY_REQUIRED,
        requires_user_input=True,
        secret=SecretDelivery(
            payload="sk-123",
            display="here is sk-123",
            redacted="key created (redacted)",
            env_var="LQH_INFERENCE_KEY",
        ),
    )
    env, code = _interpret(result, tmp_path)
    assert code == 0
    assert env["result"]["secret"] == "sk-123"
    assert "sk-123" not in env["result"]["text"]


def test_secret_delivery_save_secret(tmp_path: Path) -> None:
    result = ToolResult(
        content=SECRET_DELIVERY_REQUIRED,
        requires_user_input=True,
        secret=SecretDelivery(
            payload="sk-456",
            display="d",
            redacted="r",
            env_var="LQH_KEY",
        ),
    )
    env, code = _interpret(result, tmp_path, save_secret=True)
    assert code == 0
    assert "LQH_KEY=sk-456" in (tmp_path / ".env").read_text()


def test_compute_pick_maps_to_config(tmp_path: Path) -> None:
    result = ToolResult(
        content=COMPUTE_PICK_REQUIRED,
        requires_user_input=True,
        question="Where should training run?",
        options=["cloud", "local"],
    )
    env, code = _interpret(result, tmp_path)
    assert code == 5
    assert env["error"]["kind"] == "config"
    assert "compute_set" in env["error"]["message"]
    assert "cloud" in env["error"]["message"]


def test_permission_required_defensive(tmp_path: Path) -> None:
    result = ToolResult(
        content="PERMISSION_REQUIRED",
        requires_user_input=True,
        permission_key="script:data_gen/x.py",
    )
    env, code = _interpret(result, tmp_path)
    assert code == 3
    assert env["error"]["kind"] == "permission"


def test_other_interactive_defensive(tmp_path: Path) -> None:
    result = ToolResult(
        content="OVERWRITE_CONFIRMATION_REQUIRED",
        requires_user_input=True,
        question="destroy?",
    )
    env, code = _interpret(result, tmp_path)
    assert code == 2
    assert env["error"]["kind"] == "validation"


def test_stdout_guard_yields_one_json_doc(capfd) -> None:
    """Handler prints (Python-level AND fd-level) must not pollute stdout."""
    from lqh.cli_cmds.envelope import emit

    with stdout_to_stderr() as real_stdout:
        print("python-level noise")
        os.write(1, b"fd-level noise\n")
        emit({"ok": True}, fd=real_stdout)
    out, err = capfd.readouterr()
    assert json.loads(out) == {"ok": True}
    assert "python-level noise" in err
    assert "fd-level noise" in err
