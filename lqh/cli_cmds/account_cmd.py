"""`lqh account` — the logged-in account's spend and remaining budget.

Headless surfaces (an unattended `lqh run` battery) need to pace themselves
against the monthly cap; `lqh login` only reports the user object on a fresh
device-flow login. This command reads it any time from `/api/auth/me`: a
short human summary by default, one JSON document on stdout with --json.
"""

from __future__ import annotations

import argparse
import asyncio
import json


def _result(status: str, *, ok: bool, message: str | None = None,
            **extra) -> str:
    payload: dict = {"schema_version": 1, "ok": ok, "status": status}
    payload.update(extra)
    if message is not None:
        payload["message"] = message
    return json.dumps(payload, default=str)


def _dollars(cents: object) -> str:
    if not isinstance(cents, (int, float)):
        return "n/a"
    return f"${cents / 100:,.2f}"


def _print_human(me: dict) -> None:
    user = me.get("user") or {}
    org = me.get("organization") or {}
    credits = org.get("credits") or {}

    print(f"account       {user.get('email', 'unknown')}")
    org_name = org.get("name") or "—"
    tier = org.get("tier")
    print(f"organization  {org_name}" + (f" ({tier})" if tier else ""))
    print(
        f"your spend    {_dollars(user.get('current_period_spend_cents'))}"
        f"  (limit {_dollars(user.get('monthly_cost_limit_cents'))})"
    )
    print(f"org spend     {_dollars(org.get('current_period_spend_cents'))}")
    if credits:
        print(
            f"credits left  {_dollars(credits.get('remaining_cents'))}"
            f" of {_dollars(credits.get('available_cents'))} available"
            f" (included {_dollars(credits.get('included_cents'))},"
            f" sponsored {_dollars(credits.get('sponsored_cents'))})"
        )
    period = org.get("current_period_start")
    if period:
        print(f"period start  {period}")


def cmd_account(args: argparse.Namespace) -> int:
    from lqh.auth import LoginError, account_info, get_token

    json_out = getattr(args, "json_out", False)

    if not get_token():
        print(_result("not_logged_in", ok=False,
                      message="Not logged in. Run `lqh login`."))
        return 4

    try:
        me = asyncio.run(account_info())
    except KeyboardInterrupt:
        print(_result("error", ok=False, message="interrupted"))
        return 6
    except LoginError as e:
        expired = getattr(e, "status_code", None) in (401, 403)
        print(_result("expired" if expired else "error", ok=False,
                      message=str(e)))
        return 4 if expired else 1
    except Exception as e:  # noqa: BLE001 - network and friends
        print(_result("error", ok=False, message=f"{type(e).__name__}: {e}"))
        return 1

    if json_out:
        print(_result(
            "ok", ok=True,
            user=me.get("user"),
            organization=me.get("organization"),
        ))
    else:
        _print_human(me)
    return 0
