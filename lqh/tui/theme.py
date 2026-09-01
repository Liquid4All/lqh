"""Light/dark palette for the TUI, selected from the terminal's background.

Most of the TUI is styled with *named* colors ("cyan", "dim", "bold red"),
which resolve through the terminal's own palette and therefore already follow
whatever theme the user runs. A handful of places use absolute truecolor hex
values instead, and those are the ones that break on a light terminal: the
Liquid mark is near-white, the slash-command menu is light grey on the
terminal background, and code blocks are highlighted with a dark theme.

This module resolves the background once and hands out the matching palette.
Resolution order:

1. ``LQH_THEME=light|dark`` — an explicit override, for terminals that answer
   neither of the probes below (or answer them wrongly).
2. An OSC 11 query to the terminal ("what is your background color?"). This
   is the ground truth where it is supported, which is most modern terminals.
3. ``COLORFGBG``, set by rxvt/urxvt/konsole and friends.
4. Dark, the historical default.
"""

from __future__ import annotations

import os
import re
import select
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

DARK = "dark"
LIGHT = "light"


@dataclass(frozen=True)
class Palette:
    """The absolute colors the TUI cannot inherit from the terminal."""

    name: str
    # Rich styles.
    logo_mark: str  # the Liquid droplet mark
    logo_lqh: str  # the "LQH" block letters' base
    logo_glyphs: tuple[str, str, str]  # per-letter L / Q / H accents
    accent: str  # the tagline under the banner
    resume_command: str  # the `lqh --resume …` farewell hint
    code_theme: str  # Pygments theme for code blocks and file views
    # prompt_toolkit style rules.
    tui_style: dict[str, str]


DARK_PALETTE = Palette(
    name=DARK,
    logo_mark="bold #f8fafc",
    logo_lqh="bold #94a3b8",
    logo_glyphs=("bold #38bdf8", "bold #a78bfa", "bold #fbbf24"),
    accent="dim #94a3b8",
    resume_command="bold #60a5fa",
    code_theme="monokai",
    tui_style={
        "status": "bg:#1a1a2e #e0e0e0",
        "status.spinner": "bg:#1a1a2e #00ff88 bold",
        "status.separator": "bg:#1a1a2e #555555",
        "status.warning": "bg:#1a1a2e #ff4444 bold",
        "status.caution": "bg:#1a1a2e #ffaa00",
        "status.dim": "bg:#1a1a2e #666666",
        "input-border": "#444444",
        "input-prompt": "bold #888888",
        "input-area": "bg:#16202a #f5f7fa",
        # NB: deliberately NOT named "completion-menu" — that class exists in
        # prompt_toolkit's default UI style (bg:#bbbbbb) and would bleed a light
        # background through any partial override.
        "slash-menu": "#c0c8d0",
        "slash-menu.selected": "bg:#16202a #00ff88 bold",
        "slash-menu.meta": "#777777",
        "slash-menu.meta.selected": "bg:#16202a #aaaaaa",
        "slash-menu.hint": "#555555 italic",
    },
)

LIGHT_PALETTE = Palette(
    name=LIGHT,
    logo_mark="bold #0f172a",
    logo_lqh="bold #475569",
    logo_glyphs=("bold #0369a1", "bold #6d28d9", "bold #b45309"),
    accent="dim #475569",
    resume_command="bold #1d4ed8",
    code_theme="friendly",
    tui_style={
        "status": "bg:#dde3ea #1f2937",
        "status.spinner": "bg:#dde3ea #047857 bold",
        "status.separator": "bg:#dde3ea #94a3b8",
        "status.warning": "bg:#dde3ea #b91c1c bold",
        "status.caution": "bg:#dde3ea #b45309",
        "status.dim": "bg:#dde3ea #64748b",
        "input-border": "#94a3b8",
        "input-prompt": "bold #475569",
        "input-area": "bg:#f1f5f9 #0f172a",
        "slash-menu": "#1e293b",
        "slash-menu.selected": "bg:#dbe4ef #065f46 bold",
        "slash-menu.meta": "#64748b",
        "slash-menu.meta.selected": "bg:#dbe4ef #475569",
        "slash-menu.hint": "#64748b italic",
    },
)

_PALETTES = {DARK: DARK_PALETTE, LIGHT: LIGHT_PALETTE}

# ``rgb:RRRR/GGGG/BBBB`` (xterm) or ``rgba:…`` — each component is 1-4 hex
# digits, so the divisor depends on the width the terminal chose to answer in.
_OSC11_RESPONSE = re.compile(rb"rgba?:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})")

# A primary device-attributes reply, which every terminal answers. It is sent
# right behind the OSC 11 query so the read has a definite end: once the DA1
# reply lands, a terminal that was going to answer OSC 11 already has. Without
# it we would have to guess a timeout, and a terminal that answered just after
# we gave up would spill its reply into the input buffer as junk keystrokes.
_DA1_RESPONSE = re.compile(rb"\x1b\[\?[0-9;]*c")

# Only reached if the terminal answers neither query.
_PROBE_TIMEOUT_SEC = 0.2


def _from_env() -> str | None:
    value = os.environ.get("LQH_THEME", "").strip().lower()
    return value if value in _PALETTES else None


def _from_colorfgbg() -> str | None:
    """Read ``COLORFGBG`` — "fg;bg" or, under rxvt, "fg;<extra>;bg"."""
    raw = os.environ.get("COLORFGBG", "")
    fields = [f.strip() for f in raw.split(";") if f.strip()]
    if not fields or not fields[-1].isdigit():
        return None
    background = int(fields[-1])
    # ANSI 7 (white) and the bright range 9-15 are the light backgrounds;
    # 0-6 and 8 are dark.
    return LIGHT if background == 7 or 9 <= background <= 15 else DARK


def _from_osc11(timeout: float = _PROBE_TIMEOUT_SEC) -> str | None:
    """Ask the terminal for its background color and classify the answer."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        import termios
        import tty
    except ImportError:  # pragma: no cover - Windows
        return None

    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except (termios.error, ValueError, OSError):
        return None

    buffer = b""
    try:
        tty.setraw(fd)
        sys.stdout.write("\x1b]11;?\x07\x1b[c")
        sys.stdout.flush()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                break
            chunk = os.read(fd, 64)
            if not chunk:
                break
            buffer += chunk
            if _DA1_RESPONSE.search(buffer):
                break
    except Exception:
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except (termios.error, ValueError, OSError):  # pragma: no cover
            pass

    return classify_osc11(buffer)


def classify_osc11(response: bytes) -> str | None:
    """Classify an OSC 11 reply as light or dark; None if it isn't one."""
    match = _OSC11_RESPONSE.search(response)
    if not match:
        return None
    channels = []
    for component in match.groups():
        scale = float(16 ** len(component) - 1)
        channels.append(int(component, 16) / scale)
    red, green, blue = channels
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return LIGHT if luminance > 0.5 else DARK


def detect_background() -> str:
    """Resolve the terminal background: "light" or "dark"."""
    for probe in (_from_env, _from_osc11, _from_colorfgbg):
        try:
            resolved = probe()
        except Exception:  # never let detection break startup
            resolved = None
        if resolved:
            return resolved
    return DARK


@lru_cache(maxsize=1)
def active_palette() -> Palette:
    """The palette for this session's terminal, detected once."""
    return _PALETTES[detect_background()]
