"""`lqh run` driver: gating, result JSON, exit codes, NDJSON events."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from lqh.cli_cmds.run_cmd import cmd_run
from lqh.project_identity import ensure_identity


def _ns(task: str | None = None, **kw) -> argparse.Namespace:
    return argparse.Namespace(
        command="run",
        task=task,
        prompt_file=kw.get("prompt_file"),
        resume=kw.get("resume"),
        allow_publish=kw.get("allow_publish", False),
        max_turns=kw.get("max_turns"),
        max_tool_calls=kw.get("max_tool_calls"),
        save_secret=kw.get("save_secret", False),
        quiet=kw.get("quiet", False),
        spec=kw.get("spec"),
        project=kw.get("project"),
        timeout=kw.get("timeout"),
        compute=kw.get("compute"),
    )


def _read_result(capfd) -> tuple[dict, str]:
    out, err = capfd.readouterr()
    return json.loads(out), err


class _FakeAgent:
    """Stands in for lqh.agent.Agent inside the run driver."""

    # Class-level knobs set per test.
    auto_exit: tuple[str, str] | None = ("success", "did the thing")
    auto_exit_details: dict = {}
    policy_halt: tuple[str, str] | None = None
    script: list | None = None  # optional list of (tool, args, result_content)
    delay: float = 0.0  # simulated work duration

    def __init__(self, project_dir, session, callbacks=None, *, policy=None,
                 extra_spec=None, **_kw) -> None:
        self.project_dir = project_dir
        self.session = session
        self.callbacks = callbacks
        self.policy = policy
        self._auto_exit = None
        self._auto_exit_details = {}
        self._policy_halt = None
        self.delivered_secrets: list = []
        self._total_prompt_tokens = 11
        self._total_completion_tokens = 7
        self._run_prompt_tokens = 11
        self._run_completion_tokens = 7
        self._llm_calls_made = 1
        self.max_llm_calls = None
        self.max_total_tool_calls = None
        self._handle_tool_call = self._handle  # driver wraps this attribute

    async def _handle(self, tool_name, arguments, **kw):
        raise AssertionError("not used by fake")

    def set_startup_facts(self, **kw) -> None: ...

    async def prepare_context(self) -> str:
        return "existing_project"

    def abort_turn(self) -> None: ...

    async def process_user_input(self, text: str) -> None:
        import asyncio

        # Mirror the real agent: the user message lands in the session
        # (which materializes the lazy session dir on disk).
        self.session.add_message({"role": "user", "content": text})
        cls = type(self)
        if cls.delay:
            await asyncio.sleep(cls.delay)
        if self.callbacks and self.callbacks.on_tool_call:
            for i, (tool, args, content) in enumerate(cls.script or []):
                # Mirror the real agent's deterministic pre-dispatch cap.
                if (
                    self.max_total_tool_calls is not None
                    and i >= self.max_total_tool_calls
                ):
                    self._policy_halt = (
                        "limit_exceeded",
                        f"tool-call limit ({self.max_total_tool_calls}) reached",
                    )
                    return
                await self.callbacks.on_tool_call(tool, args)
                if self.callbacks.on_tool_result:
                    await self.callbacks.on_tool_result(tool, content)
        self._auto_exit = cls.auto_exit
        self._auto_exit_details = dict(cls.auto_exit_details)
        self._policy_halt = cls.policy_halt


@pytest.fixture()
def run_project(tmp_path, monkeypatch):
    ensure_identity(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lqh.auth.get_token", lambda: "tok")
    monkeypatch.setattr("lqh.agent.Agent", _FakeAgent)
    # No network snapshot fetches.
    async def _no_fetch(project_dir, **kw):
        raise RuntimeError("offline")

    monkeypatch.setattr("lqh.snapshot.fetch_and_cache_snapshot", _no_fetch)
    monkeypatch.setattr("lqh.cli_cmds.run_cmd._maybe_start_telemetry", lambda p: None)
    _FakeAgent.auto_exit = ("success", "did the thing")
    _FakeAgent.auto_exit_details = {}
    _FakeAgent.policy_halt = None
    _FakeAgent.script = None
    _FakeAgent.delay = 0.0
    return tmp_path


def test_no_task_is_usage_error(tmp_path, monkeypatch, capfd) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_run(_ns()) == 2
    out, err = capfd.readouterr()
    # Even usage errors keep the one-JSON-document contract (audit #2).
    payload = json.loads(out)
    assert payload["status"] == "failure"
    assert "task is required" in payload["reason"]
    assert "task is required" in err


def test_task_and_prompt_file_conflict(tmp_path, monkeypatch, capfd) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_run(_ns("x", prompt_file="f.md")) == 2


def test_unresolved_copy_blocks_run(tmp_path, monkeypatch, capfd) -> None:
    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    copy = tmp_path / "proj_copy"
    shutil.copytree(original, copy)
    monkeypatch.chdir(copy)
    assert cmd_run(_ns("do something")) == 5
    result, _ = _read_result(capfd)
    assert result["status"] == "needs_configuration"
    assert "lqh project continue" in result["reason"]


def test_compute_flag_persists_project_default(run_project, capfd) -> None:
    from lqh.remote.compute import load_project_default

    assert load_project_default(run_project) is None
    assert cmd_run(_ns("x", compute="cloud")) == 0
    assert load_project_default(run_project) == "cloud"


def test_compute_flag_rejects_bad_value(tmp_path, monkeypatch, capfd) -> None:
    ensure_identity(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cmd_run(_ns("x", compute="gpu-box")) == 2
    result, _ = _read_result(capfd)
    assert "--compute must be" in result["reason"]


def test_compute_flag_not_written_into_unresolved_copy(
    tmp_path, monkeypatch, capfd
) -> None:
    from lqh.remote.compute import compute_file_path

    original = tmp_path / "proj"
    original.mkdir()
    ensure_identity(original)
    copy = tmp_path / "proj_copy"
    shutil.copytree(original, copy)
    monkeypatch.chdir(copy)
    assert cmd_run(_ns("do something", compute="cloud")) == 5
    assert not compute_file_path(copy).exists()


def test_auth_required(tmp_path, monkeypatch, capfd) -> None:
    ensure_identity(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lqh.auth.get_token", lambda: None)
    assert cmd_run(_ns("do something")) == 4
    result, _ = _read_result(capfd)
    assert result["status"] == "auth_required"
    assert "lqh login" in result["reason"]


def test_successful_run_result_shape(run_project, capfd) -> None:
    _FakeAgent.auto_exit_details = {
        "summary": "made datasets/train_v1",
        "metrics": {"post_sft": 0.78},
        "artifacts": [{"kind": "dataset", "path": "datasets/train_v1"}],
    }
    (run_project / "datasets" / "train_v1").mkdir(parents=True)
    code = cmd_run(_ns("make a dataset"))
    result, err = _read_result(capfd)
    assert code == 0
    assert result["status"] == "success"
    assert result["summary"] == "made datasets/train_v1"
    assert result["metrics"] == {
        "post_sft": {"value": 0.78, "provenance": "reported"}
    }
    # Validated model claim merged with source label.
    assert {
        "kind": "dataset", "path": "datasets/train_v1", "source": "reported",
    } in result["artifacts"]
    assert result["session_id"]
    assert result["usage"]["prompt_tokens"] == 11
    # NDJSON events on stderr.
    events = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    kinds = [e["event"] for e in events]
    assert "start" in kinds and "end" in kinds
    assert all(e["run_id"] == result["run_id"] for e in events)


def test_claimed_artifact_outside_project_rejected(run_project, capfd) -> None:
    _FakeAgent.auto_exit_details = {
        "artifacts": [
            {"kind": "file", "path": "../evil"},
            {"kind": "file", "path": "/etc/passwd"},
            {"kind": "file", "path": "missing/thing"},
        ],
    }
    cmd_run(_ns("x"))
    result, _ = _read_result(capfd)
    assert result["artifacts"] == []


def test_policy_halt_maps_to_needs_permission(run_project, capfd) -> None:
    _FakeAgent.auto_exit = None
    _FakeAgent.policy_halt = ("needs_permission", "re-run with --allow-publish")
    assert cmd_run(_ns("push it")) == 3
    result, _ = _read_result(capfd)
    assert result["status"] == "needs_permission"
    assert "--allow-publish" in result["reason"]


def test_failure_status_exit_1(run_project, capfd) -> None:
    _FakeAgent.auto_exit = ("failure", "could not satisfy the spec")
    assert cmd_run(_ns("x")) == 1
    result, _ = _read_result(capfd)
    assert result["status"] == "failure"


def test_no_exit_call_is_failure(run_project, capfd) -> None:
    _FakeAgent.auto_exit = None
    assert cmd_run(_ns("x")) == 1
    result, _ = _read_result(capfd)
    assert result["status"] == "failure"
    assert "exit_auto_mode" in result["reason"]


def test_session_marked_completed(run_project, capfd) -> None:
    from lqh.session import Session

    cmd_run(_ns("x"))
    result, _ = _read_result(capfd)
    session = Session.load(run_project, result["session_id"])
    assert session.state == "completed"


def test_quiet_suppresses_events(run_project, capfd) -> None:
    cmd_run(_ns("x", quiet=True))
    out, err = capfd.readouterr()
    assert json.loads(out)["status"] == "success"
    assert not any(line.startswith("{") for line in err.splitlines())


def test_resume_unknown_session(run_project, capfd) -> None:
    assert cmd_run(_ns(None, resume="nope")) == 2


def test_max_tool_calls_limit(run_project, capfd) -> None:
    _FakeAgent.script = [
        ("summary", {}, "ok"),
        ("summary", {}, "ok"),
        ("summary", {}, "ok"),
    ]
    code = cmd_run(_ns("x", max_tool_calls=1))
    result, _ = _read_result(capfd)
    assert result["status"] == "limit_exceeded"
    assert "tool-call limit (1)" in result["reason"]
    assert code == 1


def test_resume_completed_session(run_project, capfd) -> None:
    from lqh.session import Session

    cmd_run(_ns("first task"))
    first, _ = _read_result(capfd)
    session_id = first["session_id"]

    code = cmd_run(_ns(None, resume=session_id))
    result, _ = _read_result(capfd)
    assert code == 0
    assert result["session_id"] == session_id
    # The resumed conversation got the continue instruction appended.
    session = Session.load(run_project, session_id)
    assert any("Resumed" in str(m.get("content")) for m in session.messages)


def test_resume_active_session_refused(run_project, capfd, monkeypatch) -> None:
    from lqh.session import sessions_dir

    cmd_run(_ns("first task"))
    first, _ = _read_result(capfd)
    session_id = first["session_id"]
    # Simulate a live FOREIGN owner: state active + pid 1 (alive, not us,
    # no pid_start so the reuse check is skipped).
    meta_path = sessions_dir(run_project) / session_id / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(state="active", pid=1, pid_start=None)
    meta_path.write_text(json.dumps(meta))
    assert cmd_run(_ns(None, resume=session_id)) == 2
    out, err = capfd.readouterr()
    assert "active in another process" in err
    # Usage errors still emit the JSON result document.
    assert json.loads(out)["status"] == "failure"


def test_timeout_maps_to_timed_out(run_project, capfd) -> None:
    _FakeAgent.delay = 10.0
    assert cmd_run(_ns("x", timeout=1)) == 6
    result, _ = _read_result(capfd)
    assert result["status"] == "timed_out"


def test_project_flag_selects_directory(run_project, tmp_path, monkeypatch, capfd) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    code = cmd_run(_ns("x", project=str(run_project)))
    result, _ = _read_result(capfd)
    assert code == 0
    # The session landed in the --project directory, not the cwd.
    assert (run_project / ".lqh" / "conversations").exists()
    assert not (elsewhere / ".lqh" / "conversations").exists()


def test_artifact_ledger_rules(tmp_path: Path) -> None:
    from lqh.cli_cmds.run_cmd import _ArtifactLedger
    from lqh.tools.handlers import ToolResult

    ledger = _ArtifactLedger(tmp_path)
    ok = ToolResult(content="done", ok=True)
    ledger.observe("run_data_gen_pipeline", {"output_dataset": "d1"}, ok)
    ledger.observe(
        "start_training", {},
        ToolResult(content="🚀 Started SFT sweep in runs/sft_v1 …", ok=True),
    )
    ledger.observe("hf_push", {"repo_id": "org/model-x"}, ok)
    # Failures never enter the ledger.
    ledger.observe(
        "run_data_gen_pipeline", {"output_dataset": "bad"},
        ToolResult.fail("runtime", "Error: boom"),
    )
    assert {"kind": "dataset", "path": "datasets/d1", "source": "ledger"} in ledger.entries
    assert {"kind": "run", "path": "runs/sft_v1", "source": "ledger"} in ledger.entries
    assert {"kind": "hf_repo", "id": "org/model-x", "source": "ledger"} in ledger.entries
    assert not any("bad" in str(e) for e in ledger.entries)


def test_headless_shows_notice_and_starts_telemetry(tmp_path: Path, monkeypatch, capfd) -> None:
    """Headless-first users must not stay invisible: `lqh run` shows the
    consent notice itself instead of waiting for a TUI session."""
    from lqh.cli_cmds.run_cmd import _maybe_start_telemetry
    from lqh.telemetry import notice_shown, set_active_telemetry

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LQH_TELEMETRY", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    ensure_identity(project)

    telemetry = _maybe_start_telemetry(project)
    _, err = capfd.readouterr()
    assert telemetry is not None
    assert "usage and workflow timing telemetry" in err
    assert "LQH_TELEMETRY=0" in err

    # Unattended: the disclosure is repeated, never marked as seen — nobody
    # watched the log it was written to.
    assert notice_shown() is False
    telemetry = _maybe_start_telemetry(project)
    _, err = capfd.readouterr()
    assert telemetry is not None
    assert "usage and workflow timing telemetry" in err

    # At a terminal it is shown once per machine and recorded.
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    assert _maybe_start_telemetry(project) is not None
    _, err = capfd.readouterr()
    assert "usage and workflow timing telemetry" in err
    assert notice_shown() is True
    assert _maybe_start_telemetry(project) is not None
    _, err = capfd.readouterr()
    assert err == ""
    set_active_telemetry(None)


@pytest.mark.parametrize("opt_out", ["env", "config"])
def test_headless_honors_opt_out(tmp_path: Path, monkeypatch, capfd, opt_out: str) -> None:
    """A user who said no is never tracked, by either route."""
    from lqh.cli_cmds.run_cmd import _maybe_start_telemetry
    from lqh.config import LqhConfig, save_config
    from lqh.telemetry import notice_shown

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LQH_TELEMETRY", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    ensure_identity(project)
    if opt_out == "env":
        monkeypatch.setenv("LQH_TELEMETRY", "0")
    else:
        save_config(LqhConfig(telemetry_enabled=False))

    assert _maybe_start_telemetry(project) is None
    _, err = capfd.readouterr()
    assert "telemetry: disabled for this run" in err
    assert "usage and workflow timing telemetry" not in err
    assert notice_shown() is False


async def test_end_telemetry_flushes_the_queue() -> None:
    """end_session only appends to the local queue; a headless-only machine
    has nothing else that would ever upload it."""
    from unittest.mock import AsyncMock, MagicMock

    from lqh.cli_cmds.run_cmd import _end_telemetry

    telemetry = MagicMock()
    telemetry.run_deferred = AsyncMock()
    telemetry.flush = AsyncMock()
    await _end_telemetry(telemetry, "success")
    telemetry.run_deferred.assert_awaited_once_with(telemetry.end_session, "succeeded")
    telemetry.flush.assert_awaited_once()
