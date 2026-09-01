"""Terminal background detection and the light/dark palettes."""

import os
import pathlib
import pty
import select
import subprocess
import sys
import time

import pytest

from lqh.tui import theme
from lqh.tui.renderer import render_resume_hint, render_welcome


@pytest.fixture(autouse=True)
def _clear_detection_cache():
    theme.active_palette.cache_clear()
    yield
    theme.active_palette.cache_clear()


class TestOSC11:
    def test_white_background_is_light(self):
        assert theme.classify_osc11(b"\x1b]11;rgb:ffff/ffff/ffff\x07") == theme.LIGHT

    def test_dark_background_is_dark(self):
        assert theme.classify_osc11(b"\x1b]11;rgb:1e1e/1e1e/1e1e\x07") == theme.DARK

    def test_short_components_are_scaled(self):
        # Terminals may answer with 1-4 hex digits per channel.
        assert theme.classify_osc11(b"\x1b]11;rgb:ff/ff/ff\x1b\\") == theme.LIGHT
        assert theme.classify_osc11(b"\x1b]11;rgb:0/0/0\x1b\\") == theme.DARK

    def test_non_response_is_undecided(self):
        assert theme.classify_osc11(b"") is None
        assert theme.classify_osc11(b"not a color") is None


class TestColorFGBG:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("0;15", theme.LIGHT),
            ("15;0", theme.DARK),
            ("0;default;15", theme.LIGHT),
            ("15;default;0", theme.DARK),
            ("0;7", theme.LIGHT),
        ],
    )
    def test_background_field(self, monkeypatch, value, expected):
        monkeypatch.setenv("COLORFGBG", value)
        assert theme._from_colorfgbg() == expected

    def test_unset_or_unparsable(self, monkeypatch):
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert theme._from_colorfgbg() is None
        monkeypatch.setenv("COLORFGBG", "default;default")
        assert theme._from_colorfgbg() is None


class TestDetection:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LQH_THEME", "light")
        monkeypatch.setenv("COLORFGBG", "15;0")
        assert theme.detect_background() == theme.LIGHT
        assert theme.active_palette() is theme.LIGHT_PALETTE

    def test_unknown_env_value_is_ignored(self, monkeypatch):
        monkeypatch.setenv("LQH_THEME", "solarized")
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert theme.detect_background() == theme.DARK

    def test_defaults_to_dark(self, monkeypatch):
        monkeypatch.delenv("LQH_THEME", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert theme.detect_background() == theme.DARK
        assert theme.active_palette() is theme.DARK_PALETTE

    def test_probe_failure_falls_through(self, monkeypatch):
        monkeypatch.delenv("LQH_THEME", raising=False)
        monkeypatch.setattr(theme, "_from_osc11", lambda: 1 / 0)
        monkeypatch.setenv("COLORFGBG", "0;15")
        assert theme.detect_background() == theme.LIGHT

    def test_osc11_is_skipped_without_a_tty(self, monkeypatch):
        # pytest captures stdio, so this is the normal test-run path: no query
        # is written and no terminal state is touched.
        assert theme._from_osc11() is None

    def test_palettes_define_the_same_style_classes(self):
        assert set(theme.LIGHT_PALETTE.tui_style) == set(theme.DARK_PALETTE.tui_style)


class TestRendererFollowsPalette:
    def test_logo_is_dark_ink_on_a_light_terminal(self, monkeypatch):
        monkeypatch.setenv("LQH_THEME", "light")
        assert "38;2;15;23;42" in render_welcome(80)

    def test_logo_is_light_ink_on_a_dark_terminal(self, monkeypatch):
        monkeypatch.setenv("LQH_THEME", "dark")
        assert "38;2;248;250;252" in render_welcome(80)

    def test_resume_hint_follows_the_palette(self, monkeypatch):
        monkeypatch.setenv("LQH_THEME", "light")
        assert "38;2;29;78;216" in render_resume_hint("abc")


_PROBE_CHILD = """
import os, select, sys, termios
sys.path.insert(0, {root!r})
from lqh.tui import theme
# PENDIN is a kernel-maintained status bit in lflag (set when input is
# pending across a mode switch), not one of ours to preserve; every other
# flag must come back exactly as it was.
PENDIN = getattr(termios, "PENDIN", 0)
def snapshot():
    attrs = termios.tcgetattr(sys.stdin.fileno())
    attrs[3] &= ~PENDIN
    return attrs
before = snapshot()
result = theme._from_osc11()
restored = snapshot() == before
readable = select.select([sys.stdin], [], [], 0.05)[0]
leftover = os.read(sys.stdin.fileno(), 256) if readable else b""
sys.stdout.write(
    "RESULT=%s LEFTOVER=%r RESTORED=%s\\n" % (result, leftover, restored)
)
sys.stdout.flush()
"""


def _probe_against_fake_terminal(reply: bytes | None) -> str:
    """Run ``_from_osc11`` in a child whose "terminal" answers *reply*.

    Runs in a real subprocess on a real pty — the probe needs genuine tty
    stdio, which pytest's captured streams are not. Guards the two properties
    that matter: the probe consumes the whole reply (nothing is left in the
    input buffer to surface later as junk keystrokes in the TUI), and it hands
    the terminal back in the mode it found it in (leaving it in raw mode would
    give the user a broken shell after lqh exits).
    """
    root = str(pathlib.Path(__file__).resolve().parents[2])
    env = {k: v for k, v in os.environ.items() if k not in ("LQH_THEME", "COLORFGBG")}
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, "-c", _PROBE_CHILD.format(root=root)],
        stdin=slave,
        stdout=slave,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=root,
    )
    os.close(slave)
    seen = b""
    answered = False
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline and b"RESULT=" not in seen:
            if not select.select([master], [], [], deadline - time.monotonic())[0]:
                break
            try:
                data = os.read(master, 1024)
            except OSError:
                break
            if not data:
                break
            seen += data
            if not answered and b"\x1b]11;?" in seen:
                answered = True
                if reply:
                    os.write(master, reply)
    finally:
        proc.wait(timeout=30)
        os.close(master)
    return seen.split(b"RESULT=")[-1].decode(errors="replace").strip()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires a POSIX pty")
class TestProbeAgainstARealTerminal:
    def test_light_terminal(self):
        out = _probe_against_fake_terminal(b"\x1b]11;rgb:ffff/ffff/ffff\x07\x1b[?1;2c")
        assert out == "light LEFTOVER=b'' RESTORED=True"

    def test_dark_terminal(self):
        out = _probe_against_fake_terminal(b"\x1b]11;rgb:1e1e/1e1e/1e1e\x07\x1b[?62;c")
        assert out == "dark LEFTOVER=b'' RESTORED=True"

    def test_terminal_that_answers_only_da1(self):
        out = _probe_against_fake_terminal(b"\x1b[?1;2c")
        assert out == "None LEFTOVER=b'' RESTORED=True"

    def test_terminal_that_answers_nothing(self):
        out = _probe_against_fake_terminal(None)
        assert out == "None LEFTOVER=b'' RESTORED=True"
