"""CLI parser routing: legacy invocations byte-identical, subcommands wired.

The three regression-locked invocations are `lqh`, `lqh --auto DIR`, and
`lqh --version` (CLI_PLAN §8 phase 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lqh import __version__
from lqh.cli import _project_name_error, main


def _set_test_home(monkeypatch, home: Path, *, interactive: bool = True) -> None:
    """Platform-independent Home/TTY setup for startup-routing tests."""
    monkeypatch.setattr("lqh.cli._home_dir", lambda: home)
    monkeypatch.setattr("lqh.cli._stdin_is_interactive", lambda: interactive)


def test_bare_invocation_routes_to_tui(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    calls: list = []
    monkeypatch.setattr(
        "lqh.cli._launch_tui", lambda *a: calls.append(a)
    )
    main()
    assert calls == [(tmp_path, False, None, None)]


def test_bare_invocation_from_fresh_home_creates_named_project(
    monkeypatch, tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "Support Assistant")
    calls: list = []
    monkeypatch.setattr(
        "lqh.cli._launch_tui",
        lambda *args: calls.append((args, Path.cwd())),
    )

    main()

    project = (home / "lqh-projects" / "Support Assistant").resolve()
    assert project.is_dir()
    assert calls == [((project, False, None, None), project)]


def test_bare_invocation_below_home_does_not_prompt(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = home / "some_project_name"
    project.mkdir(parents=True)
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("a Home subdirectory must not prompt"),
    )
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *args: calls.append(args))

    main()

    assert calls == [(project, False, None, None)]


def test_global_home_config_does_not_suppress_prompt(
    monkeypatch, tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    global_config = home / ".lqh"
    global_config.mkdir(parents=True)
    (global_config / "config.json").write_text("{}\n")
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "new-project")
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *args: calls.append(args))

    main()

    project = (home / "lqh-projects" / "new-project").resolve()
    assert calls == [(project, False, None, None)]


def test_home_prompt_can_explicitly_use_home(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "SPEC.md").write_text("# Existing Home project\n")
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *args: calls.append(args))

    main()

    assert calls == [(home.resolve(), False, None, None)]


def test_resume_from_home_skips_project_prompt(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh", "--resume", "conversation-id"])
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("resume must stay in its project"),
    )
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *args: calls.append(args))

    main()

    assert calls == [(home, False, None, "conversation-id")]


def test_spec_from_home_uses_selected_project(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh", "--spec", "support replies"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "support-project")
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *args: calls.append(args))

    main()

    project = (home / "lqh-projects" / "support-project").resolve()
    assert calls == [(project, False, "support replies", None)]


def test_home_project_prompt_retries_invalid_names_and_reuses_directory(
    monkeypatch, tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    projects = home / "lqh-projects"
    existing = projects / "existing"
    existing.mkdir(parents=True)
    (projects / "taken").write_text("not a directory\n")
    answers = iter([".hidden", "../escape", "taken", "existing"])
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *args: calls.append(args))

    main()

    assert calls == [((existing.resolve()), False, None, None)]


def test_home_project_prompt_rejects_escaping_symlink(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    projects = home / "lqh-projects"
    outside = tmp_path / "outside"
    projects.mkdir(parents=True)
    outside.mkdir()
    try:
        (projects / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    answers = iter(["escape", "safe"])
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *args: calls.append(args))

    main()

    safe = (projects / "safe").resolve()
    assert calls == [(safe, False, None, None)]


def test_home_project_mkdir_failure_reprompts(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    answers = iter(["blocked", "safe"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    original_mkdir = Path.mkdir

    def flaky_mkdir(path: Path, *args, **kwargs) -> None:
        if path.name == "blocked":
            raise OSError("name rejected by filesystem")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *args: calls.append(args))

    main()

    project = (home / "lqh-projects" / "safe").resolve()
    assert calls == [(project, False, None, None)]


def test_home_project_prompt_eof_cancels_without_launch(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])

    def end_input(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    monkeypatch.setattr(
        "lqh.cli._launch_tui", lambda *_args: pytest.fail("must not launch")
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 130


def test_home_project_noninteractive_stdin_has_clear_error(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _set_test_home(monkeypatch, home, interactive=False)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: pytest.fail("must not read piped stdin")
    )
    monkeypatch.setattr(
        "lqh.cli._launch_tui", lambda *_args: pytest.fail("must not launch")
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    assert "interactive terminal" in capsys.readouterr().err


def test_home_project_open_failure_exits_without_launch(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _set_test_home(monkeypatch, home)
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "project")

    def fail_chdir(_path: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr("lqh.cli.os.chdir", fail_chdir)
    monkeypatch.setattr(
        "lqh.cli._launch_tui", lambda *_args: pytest.fail("must not launch")
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1


@pytest.mark.parametrize(
    "name",
    [
        ".hidden",
        "~scratch",
        "trailing.",
        "trailing ",
        "bad:name",
        "CON",
        "con.txt",
        "control\x01name",
        "x" * 81,
        "../escape",
    ],
)
def test_home_project_names_are_portable(name: str) -> None:
    assert _project_name_error(name)


def test_home_project_name_accepts_spaces_and_unicode() -> None:
    assert _project_name_error("Résumé Assistant 研究") is None


def test_spec_passthrough(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lqh", "--spec", "use the small model"])
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *a: calls.append(a))
    main()
    assert calls == [(tmp_path, False, "use the small model", None)]


def test_version_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["lqh", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"lqh {__version__}"


def test_auto_requires_directory(monkeypatch, tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nope"
    monkeypatch.setattr(sys, "argv", ["lqh", "--auto", str(missing)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"❌ --auto requires a directory, got: {missing}" in err


def test_auto_requires_spec_md(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["lqh", "--auto", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "SPEC.md not found" in err


def test_auto_with_spec_md_launches_tui(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "SPEC.md").write_text("# spec")
    monkeypatch.setattr(sys, "argv", ["lqh", "--auto", str(tmp_path)])
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *a: calls.append(a))
    main()
    assert calls == [(tmp_path.resolve(), True, None, None)]


def test_unknown_command_is_usage_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["lqh", "bogus"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_subcommand_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lqh", "hello"])
    seen: list = []

    def fake_dispatch(args):
        seen.append(args.command)
        return 0

    monkeypatch.setattr("lqh.cli._dispatch", fake_dispatch)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert seen == ["hello"]


def test_help_mentions_harness_bootstrap(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lqh", "--help"])
    with pytest.raises(SystemExit):
        main()
    out = capsys.readouterr().out
    assert "lqh hello" in out
    for command in ("hello", "login", "docs", "tool", "project"):
        assert command in out


def test_negative_limits_rejected(monkeypatch, capsys) -> None:
    for flag in ("--max-turns", "--max-tool-calls", "--timeout"):
        monkeypatch.setattr(sys, "argv", ["lqh", "run", "x", flag, "-3"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        capsys.readouterr()


def test_run_parse_error_emits_result_json(monkeypatch, capsys) -> None:
    """Parser-level errors keep the one-JSON-document stdout contract."""
    import json

    monkeypatch.setattr(sys, "argv", ["lqh", "run", "x", "--max-turns", "0"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["status"] == "failure"
    assert "usage" in payload["reason"]
    assert "error" in err


def test_tool_parse_error_emits_envelope_json(monkeypatch, capsys) -> None:
    import json

    monkeypatch.setattr(sys, "argv", ["lqh", "tool", "call", "summary", "--wat"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    out, _ = capsys.readouterr()
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["kind"] == "validation"


def test_root_spec_survives_run_subcommand(monkeypatch) -> None:
    """`lqh --spec X run task` must not silently drop X."""
    seen: dict = {}

    def fake_dispatch(args):
        seen["spec"] = args.spec
        return 0

    monkeypatch.setattr("lqh.cli._dispatch", fake_dispatch)
    monkeypatch.setattr(sys, "argv", ["lqh", "--spec", "use small model", "run", "t"])
    with pytest.raises(SystemExit):
        main()
    assert seen["spec"] == "use small model"

    # And the subcommand-level flag still wins when given.
    monkeypatch.setattr(
        sys, "argv", ["lqh", "run", "t", "--spec", "override"]
    )
    with pytest.raises(SystemExit):
        main()
    assert seen["spec"] == "override"


def test_resume_passthrough(monkeypatch, tmp_path: Path) -> None:
    """`lqh --resume ID` reaches the TUI launcher."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lqh", "--resume", "4f2a9c1e"])
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *a: calls.append(a))
    main()
    assert calls == [(tmp_path, False, None, "4f2a9c1e")]


def test_resume_with_subcommand_refused(monkeypatch, capsys) -> None:
    """The root --resume drives the TUI; `run` has its own."""
    monkeypatch.setattr(sys, "argv", ["lqh", "--resume", "abc", "status"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--resume cannot be combined" in capsys.readouterr().err


def test_run_resume_unaffected_by_root_flag(monkeypatch) -> None:
    """`lqh run --resume ID` keeps its own dest and still dispatches."""
    seen: dict = {}

    def fake_dispatch(args):
        seen["resume"] = args.resume
        seen["root"] = args.resume_session
        return 0

    monkeypatch.setattr("lqh.cli._dispatch", fake_dispatch)
    monkeypatch.setattr(sys, "argv", ["lqh", "run", "t", "--resume", "sid"])
    with pytest.raises(SystemExit):
        main()
    assert seen == {"resume": "sid", "root": None}


def test_auto_with_subcommand_refused(monkeypatch, tmp_path: Path, capsys) -> None:
    """`lqh --auto DIR run …` must error, not silently ignore --auto."""
    (tmp_path / "SPEC.md").write_text("# spec")
    monkeypatch.setattr(sys, "argv", ["lqh", "--auto", str(tmp_path), "run", "t"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--auto cannot be combined" in capsys.readouterr().err


def test_parser_import_hygiene() -> None:
    """Building the parser must not pull the TUI, handlers, or telemetry.

    Run in a subprocess: this pytest process has long since imported them.
    """
    import subprocess

    code = (
        "import sys\n"
        "from lqh.cli import _build_parser\n"
        "_build_parser()\n"
        "banned = ('lqh.tui', 'lqh.tui.app', 'lqh.telemetry', 'lqh.tools.handlers')\n"
        "loaded = [m for m in banned if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
