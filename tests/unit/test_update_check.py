"""Tests for the cached, best-effort PyPI update check."""

from __future__ import annotations

import json
import time

import pytest

from lqh.update_check import CACHE_TTL_SECONDS, check_for_update


@pytest.mark.asyncio
async def test_fresh_cache_reports_newer_version(tmp_path) -> None:
    cache = tmp_path / "update-check.json"
    cache.write_text(
        json.dumps({"checked_at": time.time(), "latest_version": "0.5.0"})
    )

    update = await check_for_update(current_version="0.4.13", cache_path=cache)

    assert update is not None
    assert update.latest == "0.5.0"


@pytest.mark.asyncio
async def test_fresh_cache_ignores_older_version(tmp_path) -> None:
    cache = tmp_path / "update-check.json"
    cache.write_text(
        json.dumps({"checked_at": time.time(), "latest_version": "0.2.2"})
    )

    assert (
        await check_for_update(current_version="0.4.13", cache_path=cache) is None
    )


@pytest.mark.asyncio
async def test_update_check_can_be_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LQH_NO_UPDATE_CHECK", "1")
    stale_cache = tmp_path / "update-check.json"
    stale_cache.write_text(
        json.dumps(
            {
                "checked_at": time.time() - CACHE_TTL_SECONDS - 1,
                "latest_version": "99.0.0",
            }
        )
    )

    assert await check_for_update(cache_path=stale_cache) is None


@pytest.mark.asyncio
async def test_stable_version_does_not_offer_prerelease(tmp_path) -> None:
    cache = tmp_path / "update-check.json"
    cache.write_text(
        json.dumps({"checked_at": time.time(), "latest_version": "1.0.0rc1"})
    )

    assert await check_for_update(current_version="0.9.0", cache_path=cache) is None
