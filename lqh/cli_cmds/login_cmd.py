"""`lqh login` — device-flow auth without the TUI.

Human-readable progress (verification URL + code) on stderr; exactly one
machine-readable JSON result on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def _result(status: str, *, ok: bool, user: dict | None = None,
            message: str | None = None) -> str:
    payload: dict = {"schema_version": 1, "ok": ok, "status": status}
    if user is not None:
        payload["user"] = user
    if message is not None:
        payload["message"] = message
    return json.dumps(payload)


def cmd_login(args: argparse.Namespace) -> int:
    from lqh.auth import LoginError, LoginExpired, get_token, login_device_code

    if get_token():
        print(_result("already_logged_in", ok=True))
        return 0

    async def _show(verification_uri: str, user_code: str) -> None:
        print(f"Open {verification_uri}", file=sys.stderr)
        print(f"Code: {user_code}", file=sys.stderr)

    try:
        user = asyncio.run(
            login_device_code(
                on_user_code=_show, open_browser=not args.no_browser
            )
        )
    except KeyboardInterrupt:
        print(_result("error", ok=False, message="interrupted"))
        return 6
    except LoginExpired as e:
        print(_result("expired", ok=False, message=str(e)))
        return 4
    except LoginError as e:
        print(_result("error", ok=False, message=str(e)))
        return 4
    except Exception as e:  # noqa: BLE001 - network and friends
        print(_result("error", ok=False, message=f"{type(e).__name__}: {e}"))
        return 1

    print(_result("logged_in", ok=True, user=user))
    return 0
