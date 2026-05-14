"""Shared fixtures for remote tests."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from lqh.remote.backend import RemoteConfig


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--remote-host",
        default=None,
        help="SSH hostname for remote e2e tests",
    )


@pytest.fixture
def remote_host(request: pytest.FixtureRequest) -> str:
    """SSH hostname for e2e tests.  Skips if not provided."""
    host = request.config.getoption("--remote-host") or os.environ.get(
        "LQH_TEST_REMOTE_HOST"
    )
    if not host:
        pytest.skip("No --remote-host provided")
    return host


@pytest.fixture
def sample_remote_config() -> RemoteConfig:
    """A sample RemoteConfig for unit tests (no real SSH)."""
    return RemoteConfig(
        name="test-gpu",
        type="ssh_direct",
        hostname="test-host",
        remote_root="/home/testuser/lqh/test-project",
        gpu_ids=[0, 1],
    )


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with .lqh/ structure."""
    lqh_dir = tmp_path / ".lqh"
    lqh_dir.mkdir()
    return tmp_path


@pytest.fixture
def remote_project_dir(remote_host: str, tmp_path: Path) -> tuple[str, Path]:
    """Create a temp project dir for e2e tests.

    Returns ``(remote_root, local_project_dir)``.
    """
    unique = uuid4().hex[:8]
    remote_root = f"/tmp/lqh-test-{unique}"
    local = tmp_path / "project"
    local.mkdir()
    (local / ".lqh").mkdir()
    return remote_root, local
