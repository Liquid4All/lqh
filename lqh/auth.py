from __future__ import annotations

import asyncio
import os
import socket
import time
import webbrowser
from typing import Any, Awaitable, Callable

import httpx

from lqh import __version__
from lqh.config import default_api_base_url, load_config, load_credentials, save_credentials

DEVICE_CODE_PATH = "/api/cli/device/code"
DEVICE_TOKEN_PATH = "/api/cli/device/token"


def api_root() -> str:
    """Root URL for non-OpenAI endpoints (e.g. device-code auth).

    Derived from ``LQH_BASE_URL`` (or the default) by stripping a trailing
    ``/v1`` segment so callers can hit ``/api/...`` paths.
    """
    base = default_api_base_url().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


class LoginError(RuntimeError):
    pass


class LoginExpired(LoginError):
    pass


def get_token() -> str | None:
    """Load the API token. Resolution order: env var → new credentials file → legacy config."""
    env_key = os.environ.get("LQH_DEBUG_API_KEY")
    if env_key:
        return env_key
    token = load_credentials()
    if token:
        return token
    return load_config().api_key


def require_token() -> str:
    """Load the API token, raising if not set."""
    token = get_token()
    if token is None:
        raise RuntimeError(
            "Not logged in. Run /login to authenticate with lqh.ai."
        )
    return token


UserCodeCallback = Callable[[str, str], Awaitable[None]]


async def login_device_code(
    on_user_code: UserCodeCallback | None = None,
    open_browser: bool = True,
) -> dict[str, Any]:
    """Run the device-code flow against api.lqh.ai.

    Returns the ``user`` dict on success and persists the bearer token to
    the credentials file. Raises LoginExpired if the device code expires
    before the user approves, or LoginError on any other backend failure.
    """
    client_name = f"lqh-cli/{__version__} {socket.gethostname()}"

    async with httpx.AsyncClient(base_url=api_root(), timeout=30.0) as http:
        r = await http.post(DEVICE_CODE_PATH, json={"client_name": client_name})
        r.raise_for_status()
        data = r.json()

        verification_uri = data["verification_uri"]
        user_code = data["user_code"]
        device_code = data["device_code"]
        interval = max(int(data.get("interval", 5)), 1)
        deadline = time.time() + int(data.get("expires_in", 900))

        if on_user_code is not None:
            await on_user_code(verification_uri, user_code)

        if open_browser:
            try:
                webbrowser.open(verification_uri)
            except Exception:
                pass

        while time.time() < deadline:
            await asyncio.sleep(interval)
            rr = await http.post(DEVICE_TOKEN_PATH, json={"device_code": device_code})
            if rr.status_code == 200:
                payload = rr.json()
                save_credentials(payload["token"])
                user = payload.get("user")
                return user if isinstance(user, dict) else {}

            try:
                body = rr.json()
            except ValueError:
                raise LoginError(f"unexpected response (status {rr.status_code}): {rr.text[:200]}")

            err = body.get("error")
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval = max(interval, int(body.get("retry_after_seconds", interval)))
                continue
            if err == "expired_token":
                raise LoginExpired("device code expired")
            raise LoginError(f"login failed: {body}")

    raise LoginExpired("timed out waiting for approval")
