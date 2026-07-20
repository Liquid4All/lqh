"""`lqh login` — machine-readable output, exit codes."""

from __future__ import annotations

import argparse
import json

from lqh.cli_cmds.login_cmd import cmd_login


def _ns(no_browser: bool = False) -> argparse.Namespace:
    return argparse.Namespace(command="login", no_browser=no_browser)


def test_already_logged_in(monkeypatch, capsys) -> None:
    monkeypatch.setattr("lqh.auth.get_token", lambda: "tok")
    assert cmd_login(_ns()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": 1, "ok": True, "status": "already_logged_in",
    }


def test_successful_login(monkeypatch, capsys) -> None:
    monkeypatch.setattr("lqh.auth.get_token", lambda: None)
    seen: dict = {}

    async def fake_login(on_user_code=None, open_browser=True):
        seen["open_browser"] = open_browser
        if on_user_code is not None:
            await on_user_code("https://lqh.ai/device", "ABCD-1234")
        return {"id": "u1", "email": "a@b.c"}

    monkeypatch.setattr("lqh.auth.login_device_code", fake_login)
    assert cmd_login(_ns(no_browser=True)) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["status"] == "logged_in"
    assert payload["user"] == {"id": "u1", "email": "a@b.c"}
    assert seen["open_browser"] is False
    # Human progress rides stderr, not stdout.
    assert "https://lqh.ai/device" in err
    assert "ABCD-1234" in err


def test_expired_login(monkeypatch, capsys) -> None:
    from lqh.auth import LoginExpired

    monkeypatch.setattr("lqh.auth.get_token", lambda: None)

    async def fake_login(on_user_code=None, open_browser=True):
        raise LoginExpired("device code expired")

    monkeypatch.setattr("lqh.auth.login_device_code", fake_login)
    assert cmd_login(_ns()) == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["status"] == "expired"


def test_network_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr("lqh.auth.get_token", lambda: None)

    async def fake_login(on_user_code=None, open_browser=True):
        raise ConnectionError("no route")

    monkeypatch.setattr("lqh.auth.login_device_code", fake_login)
    assert cmd_login(_ns()) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
