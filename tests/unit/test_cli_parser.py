"""CLI parser routing: legacy invocations byte-identical, subcommands wired.

The three regression-locked invocations are `lqh`, `lqh --auto DIR`, and
`lqh --version` (CLI_PLAN §8 phase 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lqh import __version__
from lqh.cli import main


def test_bare_invocation_routes_to_tui(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lqh"])
    calls: list = []
    monkeypatch.setattr(
        "lqh.cli._launch_tui", lambda *a: calls.append(a)
    )
    main()
    assert calls == [(tmp_path, False, None)]


def test_spec_passthrough(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lqh", "--spec", "use the small model"])
    calls: list = []
    monkeypatch.setattr("lqh.cli._launch_tui", lambda *a: calls.append(a))
    main()
    assert calls == [(tmp_path, False, "use the small model")]


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
    assert calls == [(tmp_path.resolve(), True, None)]


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
