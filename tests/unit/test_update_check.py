"""Tests for the cached, best-effort PyPI update check."""

from __future__ import annotations

import json
import sys
import time

import pytest

from lqh.update_check import (
    CACHE_TTL_SECONDS,
    check_for_update,
    install_extras_command,
    upgrade_command,
)


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


def test_upgrade_command_detects_uv_tool(tmp_path) -> None:
    (tmp_path / "uv-receipt.toml").write_text("")

    assert upgrade_command(str(tmp_path)) == "uv tool upgrade lqh"
    assert install_extras_command("train", str(tmp_path)) == (
        'uv tool install "lqh[train]"'
    )


def test_upgrade_command_detects_pipx(tmp_path) -> None:
    (tmp_path / "pipx_metadata.json").write_text("{}")

    assert upgrade_command(str(tmp_path)) == "pipx upgrade lqh"
    assert install_extras_command("train", str(tmp_path)) == (
        'pipx install --force "lqh[train]"'
    )


def test_upgrade_command_detects_uv_venv(tmp_path) -> None:
    # A `uv venv` environment ships no pip, so pip advice cannot work there.
    (tmp_path / "pyvenv.cfg").write_text(
        "home = /somewhere/bin\nuv = 0.12.3\nversion_info = 3.13\n"
    )

    assert upgrade_command(str(tmp_path)) == "uv pip install -U lqh"
    assert install_extras_command("train", str(tmp_path)) == (
        'uv pip install "lqh[train]"'
    )


def test_upgrade_command_falls_back_to_pip(tmp_path) -> None:
    (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\nversion_info = 3.13\n")

    assert upgrade_command(str(tmp_path)) == "pip install -U lqh"
    assert install_extras_command("train", str(tmp_path)) == "pip install lqh[train]"


def test_upgrade_command_falls_back_to_pip_without_pyvenv_cfg(tmp_path) -> None:
    assert upgrade_command(str(tmp_path)) == "pip install -U lqh"


def test_upgrade_command_reads_sys_prefix_by_default(tmp_path, monkeypatch) -> None:
    (tmp_path / "uv-receipt.toml").write_text("")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    assert upgrade_command() == "uv tool upgrade lqh"


def test_torch_missing_hint_matches_the_installer(tmp_path, monkeypatch) -> None:
    """The train-extras hint must name the manager that owns this env."""
    from lqh.tools.handlers import _check_torch_available

    (tmp_path / "uv-receipt.toml").write_text("")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise

    message = _check_torch_available()

    assert message is not None
    assert 'uv tool install "lqh[train]"' in message
