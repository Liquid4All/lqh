"""Shared fixtures for function tests that target a real SSH remote.

The ``--remote-host`` option itself is registered in the root
``tests/conftest.py`` so it works regardless of which directory is passed
on the pytest command line.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def remote_host(request: pytest.FixtureRequest) -> str:
    """SSH hostname for remote workflow tests.  Skips if not provided."""
    host = request.config.getoption("--remote-host") or os.environ.get(
        "LQH_TEST_REMOTE_HOST"
    )
    if not host:
        pytest.skip("No --remote-host provided")
    return host


@pytest.fixture
def remote_project_dir(remote_host: str, tmp_path: Path) -> tuple[str, Path]:
    """Create a temp project dir for remote workflow tests.

    Returns ``(remote_root, local_project_dir)``.
    """
    unique = uuid4().hex[:8]
    remote_root = f"/tmp/lqh-test-{unique}"
    local = tmp_path / "project"
    local.mkdir()
    (local / ".lqh").mkdir()
    return remote_root, local
