"""`lqh account` — spend/budget readout for headless callers."""

from __future__ import annotations

import argparse
import json

from lqh.cli_cmds.account_cmd import cmd_account

ME = {
    "user": {
        "email": "a@b.c",
        "current_period_spend_cents": 1234,
        "monthly_cost_limit_cents": 5000,
    },
    "organization": {
        "name": "Acme",
        "tier": "sponsored",
        "current_period_spend_cents": 4000,
        "credits": {
            "included_cents": 0,
            "sponsored_cents": 10000,
            "available_cents": 10000,
            "remaining_cents": 6000,
        },
    },
}


def _ns(json_out: bool = False) -> argparse.Namespace:
    return argparse.Namespace(command="account", json_out=json_out)


def test_not_logged_in(monkeypatch, capsys) -> None:
    monkeypatch.setattr("lqh.auth.get_token", lambda: None)
    assert cmd_account(_ns()) == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["status"] == "not_logged_in"


def test_json_output_carries_credits(monkeypatch, capsys) -> None:
    monkeypatch.setattr("lqh.auth.get_token", lambda: "tok")

    async def fake_info():
        return ME

    monkeypatch.setattr("lqh.auth.account_info", fake_info)
    assert cmd_account(_ns(json_out=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["user"]["current_period_spend_cents"] == 1234
    assert payload["organization"]["credits"]["remaining_cents"] == 6000


def test_human_output_shows_dollars(monkeypatch, capsys) -> None:
    monkeypatch.setattr("lqh.auth.get_token", lambda: "tok")

    async def fake_info():
        return ME

    monkeypatch.setattr("lqh.auth.account_info", fake_info)
    assert cmd_account(_ns()) == 0
    out = capsys.readouterr().out
    assert "a@b.c" in out
    assert "$12.34" in out
    assert "$60.00" in out


def test_expired_token_exits_auth(monkeypatch, capsys) -> None:
    from lqh.auth import LoginError

    monkeypatch.setattr("lqh.auth.get_token", lambda: "tok")

    async def fake_info():
        err = LoginError("account lookup failed (401): unauthenticated")
        err.status_code = 401
        raise err

    monkeypatch.setattr("lqh.auth.account_info", fake_info)
    assert cmd_account(_ns()) == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "expired"


def test_server_error_exits_failure(monkeypatch, capsys) -> None:
    from lqh.auth import LoginError

    monkeypatch.setattr("lqh.auth.get_token", lambda: "tok")

    async def fake_info():
        err = LoginError("account lookup failed (500): boom")
        err.status_code = 500
        raise err

    monkeypatch.setattr("lqh.auth.account_info", fake_info)
    assert cmd_account(_ns()) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
