"""Best-effort notification when a newer lqh release is on PyPI."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from lqh import __version__
from lqh.config import config_dir

PYPI_URL = "https://pypi.org/pypi/lqh/json"
CACHE_TTL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 2.0
_DISABLE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class UpdateInfo:
    current: str
    latest: str


def _updates_disabled() -> bool:
    return os.environ.get("LQH_NO_UPDATE_CHECK", "").strip().lower() in _DISABLE_VALUES


def _read_cache(path: Path, now: float) -> str | None:
    try:
        data = json.loads(path.read_text())
        checked_at = float(data["checked_at"])
        latest = data["latest_version"]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if now - checked_at >= CACHE_TTL_SECONDS or not isinstance(latest, str):
        return None
    return latest


def _write_cache(path: Path, latest: str, now: float) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"checked_at": now, "latest_version": latest}))
        tmp.replace(path)
    except OSError:
        # A read-only home directory should never affect CLI startup.
        pass


def _newer_release(current: str, latest: str) -> UpdateInfo | None:
    try:
        current_version = Version(current)
        latest_version = Version(latest)
    except InvalidVersion:
        return None

    # Do not advertise prereleases to users on the stable channel.
    if latest_version.is_prerelease and not current_version.is_prerelease:
        return None
    if latest_version <= current_version:
        return None
    return UpdateInfo(current=current, latest=latest)


async def check_for_update(
    *,
    current_version: str = __version__,
    cache_path: Path | None = None,
) -> UpdateInfo | None:
    """Return update metadata, silently ignoring network and cache failures."""
    if _updates_disabled():
        return None

    now = time.time()
    if cache_path is None:
        try:
            cache_path = config_dir() / "update-check.json"
        except OSError:
            cache_path = None

    latest = _read_cache(cache_path, now) if cache_path is not None else None
    if latest is None:
        try:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
                headers={"User-Agent": f"lqh/{current_version}"},
            ) as client:
                response = await client.get(PYPI_URL)
                response.raise_for_status()
                payload: Any = response.json()
                latest = payload["info"]["version"]
                if not isinstance(latest, str):
                    return None
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            return None
        if cache_path is not None:
            _write_cache(cache_path, latest, now)

    return _newer_release(current_version, latest)
