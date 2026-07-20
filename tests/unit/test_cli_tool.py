"""`lqh tool list|schema|call` — exposure, validation, envelope, boot gate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lqh.cli_cmds.registry import full_consent_kwargs
from lqh.cli_cmds.tool_cmd import cmd_tool
from lqh.project_identity import ensure_identity


def _list_ns(json_out: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        command="tool", tool_command="list", json_out=json_out
    )


def _schema_ns(name: str) -> argparse.Namespace:
    return argparse.Namespace(command="tool", tool_command="schema", name=name)


def _call_ns(name: str, args: str | None = None, **kw) -> argparse.Namespace:
    return argparse.Namespace(
        command="tool",
        tool_command="call",
        name=name,
        args=args,
        args_file=kw.get("args_file"),
        pretty=kw.get("pretty", False),
        save_secret=kw.get("save_secret", False),
    )


# ---------------------------------------------------------------------------
# list / schema
# ---------------------------------------------------------------------------


def test_list_excludes_unexposed(capsys) -> None:
    assert cmd_tool(_list_ns()) == 0
    out = capsys.readouterr().out
    assert "summary" in out
    for hidden in ("create_file", "ask_user", "load_skill", "set_auto_stage"):
        assert hidden not in out


def test_list_json_shape(capsys) -> None:
    assert cmd_tool(_list_ns(json_out=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    by_name = {t["name"]: t for t in payload["tools"]}
    assert by_name["summary"]["mutating"] is False
    assert by_name["start_training"]["mutating"] is True
    assert set(by_name["summary"]) == {
        "name", "description", "mutating", "needs_auth",
    }


def test_schema_prints_definition(capsys) -> None:
    assert cmd_tool(_schema_ns("summary")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "summary"
    assert "parameters" in payload


def test_schema_unknown_tool(capsys) -> None:
    assert cmd_tool(_schema_ns("nope")) == 2
    assert "Unknown or unexposed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# call — validation and auth gates (no handler executed)
# ---------------------------------------------------------------------------


def _read_envelope(capfd) -> tuple[dict, str]:
    out, err = capfd.readouterr()
    return json.loads(out), err


def test_call_bad_json_args(tmp_path: Path, monkeypatch, capfd) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_tool(_call_ns("summary", args="{not json")) == 2
    envelope, _ = _read_envelope(capfd)
    assert envelope["error"]["kind"] == "validation"


def test_call_unexposed_tool(tmp_path: Path, monkeypatch, capfd) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_tool(_call_ns("create_file", args="{}")) == 2
    envelope, _ = _read_envelope(capfd)
    assert envelope["error"]["kind"] == "validation"
    assert "lqh tool list" in envelope["error"]["message"]


def test_call_schema_validation_failure(tmp_path: Path, monkeypatch, capfd) -> None:
    monkeypatch.chdir(tmp_path)
    ns = _call_ns("run_data_gen_pipeline", args='{"script_path": "x"}')
    assert cmd_tool(ns) == 2
    envelope, _ = _read_envelope(capfd)
    assert envelope["error"]["kind"] == "validation"
    assert any(
        "num_samples" in e for e in envelope["error"]["details"]["errors"]
    )


def test_call_needs_auth_without_token(tmp_path: Path, monkeypatch, capfd) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lqh.auth.get_token", lambda: None)
    assert cmd_tool(_call_ns("artifacts", args="{}")) == 4
    envelope, _ = _read_envelope(capfd)
    assert envelope["error"]["kind"] == "auth"
    assert "lqh login" in envelope["error"]["message"]


# ---------------------------------------------------------------------------
# call — real read-only execution
# ---------------------------------------------------------------------------


def test_call_summary_single_json_doc(tmp_path: Path, monkeypatch, capfd) -> None:
    monkeypatch.chdir(tmp_path)
    code = cmd_tool(_call_ns("summary"))
    out, _ = capfd.readouterr()
    envelope = json.loads(out)  # exactly one JSON document on stdout
    assert code == 0
    assert envelope["ok"] is True
    assert envelope["tool"] == "summary"
    assert envelope["result"]["text"]


# ---------------------------------------------------------------------------
# call — identity/copy boot gate
# ---------------------------------------------------------------------------


def _copied_project(tmp_path: Path) -> Path:
    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    copy = tmp_path / "proj_copy"
    shutil.copytree(original, copy)
    return copy


def test_mutating_call_blocked_on_copy(tmp_path: Path, monkeypatch, capfd) -> None:
    copy = _copied_project(tmp_path)
    monkeypatch.chdir(copy)
    ns = _call_ns(
        "run_data_gen_pipeline",
        args='{"script_path": "data_gen/x.py", "num_samples": 3, '
        '"output_dataset": "d"}',
    )
    assert cmd_tool(ns) == 5
    envelope, _ = _read_envelope(capfd)
    assert envelope["error"]["kind"] == "config"
    assert "lqh project continue" in envelope["error"]["message"]
    assert "lqh project fork" in envelope["error"]["message"]


def test_read_only_call_warns_on_copy_but_proceeds(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    copy = _copied_project(tmp_path)
    monkeypatch.chdir(copy)
    assert cmd_tool(_call_ns("summary")) == 0
    envelope, err = _read_envelope(capfd)
    assert envelope["ok"] is True
    assert "unresolved copy" in err


def test_mutating_call_blocked_on_corrupt_identity(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    ensure_identity(tmp_path)
    (tmp_path / ".lqh" / "project.json").write_text("garbage")
    monkeypatch.chdir(tmp_path)
    ns = _call_ns(
        "run_data_gen_pipeline",
        args='{"script_path": "data_gen/x.py", "num_samples": 3, '
        '"output_dataset": "d"}',
    )
    assert cmd_tool(ns) == 5
    envelope, _ = _read_envelope(capfd)
    assert envelope["error"]["kind"] == "config"
    assert "NOT be auto-replaced" in envelope["error"]["message"]


# ---------------------------------------------------------------------------
# call — overwrite guard stays armed under invocation-is-consent
# ---------------------------------------------------------------------------


def test_overwrite_guard_refusal_maps_to_conflict(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "datasets" / "d"
    dataset.mkdir(parents=True)
    (dataset / "data.parquet").write_bytes(b"x")
    ns = _call_ns(
        "run_data_gen_pipeline",
        args='{"script_path": "data_gen/x.py", "num_samples": 3, '
        '"output_dataset": "d"}',
    )
    code = cmd_tool(ns)
    envelope, _ = _read_envelope(capfd)
    assert code == 1
    assert envelope["error"]["kind"] == "conflict"
    assert "overwrite" in envelope["error"]["message"]


def test_explicit_overwrite_bypasses_guard(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    """With overwrite=true in the caller's own args the guard is consented
    away — the call then proceeds far enough to fail on the missing script
    (NOT on the guard)."""
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "datasets" / "d"
    dataset.mkdir(parents=True)
    (dataset / "data.parquet").write_bytes(b"x")
    ns = _call_ns(
        "run_data_gen_pipeline",
        args='{"script_path": "data_gen/x.py", "num_samples": 3, '
        '"output_dataset": "d", "overwrite": true}',
    )
    code = cmd_tool(ns)
    envelope, _ = _read_envelope(capfd)
    assert code != 5
    assert envelope["error"]["kind"] != "conflict"
    assert "does not exist" in envelope["error"]["message"]


def test_full_consent_kwargs_overwrite_gating() -> None:
    plain = full_consent_kwargs({"num_samples": 3})
    assert "_overwrite_consent" not in plain
    assert plain["_permissions"].full_consent is True

    with_overwrite = full_consent_kwargs({"overwrite": True})
    assert with_overwrite["_overwrite_consent"] is True

    # overwrite must be the literal True — not a truthy string.
    assert "_overwrite_consent" not in full_consent_kwargs({"overwrite": "yes"})
