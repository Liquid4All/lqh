"""End-to-end lifecycle test under a real PTY.

Verifies the architectural bet the unit tests cannot: a bottom-docked
prompt_toolkit application suspends via ``in_terminal()``, the full-screen
dataset viewer runs on the alternate screen, and control returns to the host
app with the summary. Skipped where PTYs are unavailable.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="requires a POSIX pty"
)

_CHILD_SCRIPT = r"""
import asyncio, json, sys, tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from lqh.tui.dataset_viewer import DatasetViewer
from lqh.tui.dataset_viewer_app import run_dataset_viewer


async def main():
    tmp = Path(tempfile.mkdtemp())
    rows = [{"messages": [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "content": "\n\n".join(f"para {j}" for j in range(30))},
    ]} for i in range(3)]
    pq.write_table(pa.table({
        "messages": [json.dumps(r["messages"]) for r in rows],
        "audio": [None] * len(rows),
    }), tmp / "data.parquet")

    state = {"summary": None}
    kb = KeyBindings()

    @kb.add("v")
    def _open(event):
        async def run():
            viewer = DatasetViewer(tmp / "data.parquet", agent_message="PTY lifecycle")
            async with in_terminal():
                state["summary"] = await run_dataset_viewer(viewer)
            event.app.invalidate()
        asyncio.get_event_loop().create_task(run())

    @kb.add("x")
    def _exit(event):
        event.app.exit()

    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(
            lambda: f"HOST-APP summary={state['summary']!r}"
        ), height=1)])),
        key_bindings=kb,
        full_screen=False,
    )
    await app.run_async()
    print("HOST-EXITED")
    print("SUMMARY:", state["summary"])
    assert state["summary"] and "3 total rows" in state["summary"]
    print("PTY-LIFECYCLE-OK")


asyncio.run(main())
"""


def _drain(fd: int, timeout: float) -> bytes:
    out = b""
    while True:
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return out
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            return out
        if not chunk:
            return out
        out += chunk


def test_in_terminal_lifecycle(tmp_path):
    import pty

    script = tmp_path / "child.py"
    script.write_text(_CHILD_SCRIPT, encoding="utf-8")

    # openpty + subprocess (not pty.fork): forking a multi-threaded pytest
    # process is a documented deadlock risk.
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "TERM": "xterm-256color"},
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave)

    out = b""
    try:
        # open viewer, scroll, next sample, quit viewer, exit host app
        for key in ["v", "j", "j", "n", "q", "x"]:
            time.sleep(0.6)
            out += _drain(master, 0.05)
            os.write(master, key.encode())

        deadline = time.time() + 30
        while time.time() < deadline and proc.poll() is None:
            out += _drain(master, 0.5)
        out += _drain(master, 0.5)
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)

    text = out.decode(errors="replace")
    assert "\x1b[?1049h" in text, "viewer never entered the alternate screen"
    assert "\x1b[?1049l" in text, "alternate screen never restored"
    assert "PTY lifecycle" in text, "viewer header not rendered"
    assert "PTY-LIFECYCLE-OK" in text, f"lifecycle failed; output tail: {text[-2000:]}"
