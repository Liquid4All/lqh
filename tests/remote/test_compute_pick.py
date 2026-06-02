"""Tests for the project compute picker gating (_compute_pick_options).

The compute target is a fixed, per-project decision. The picker only
fires when a project has >=1 bring-your-own-compute remote bound but
neither a project nor a global default set — otherwise LQH Cloud is the
silent default and no dialog is shown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import lqh.tools.handlers as handlers
from lqh.remote.backend import ProjectBinding, RemoteMachine
from lqh.remote.compute import save_global_default, save_project_default
from lqh.remote.config import add_binding, add_machine
from lqh.tools.handlers import _MAX_PICKER_REMOTES, _compute_pick_options


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect global config dirs and default the local GPU to absent.

    Local GPU availability is environment-dependent, so we pin it to
    False by default; tests that want the "Local (this machine)" option
    re-patch it to True.
    """
    import lqh.config as config_mod
    import lqh.remote.config as remote_config_mod

    global_dir = tmp_path / "global_lqh"
    global_dir.mkdir()
    monkeypatch.setattr(remote_config_mod, "GLOBAL_CONFIG_DIR", global_dir)
    monkeypatch.setattr(config_mod, "config_dir", lambda: global_dir)
    monkeypatch.setattr(handlers, "_local_gpu_available", lambda: False)


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / ".lqh").mkdir(parents=True)
    return proj


def _bind_remote(project_dir: Path, name: str, hostname: str) -> None:
    """Register a global machine and bind it to the project."""
    add_machine(RemoteMachine(name=name, type="ssh_direct", hostname=hostname))
    add_binding(project_dir, ProjectBinding(name=name, remote_root=f"/home/u/{name}"))


def test_no_remotes_no_local_gpu_returns_none(project_dir: Path):
    """Cloud-only project with no local GPU: no picker, silent default."""
    assert _compute_pick_options(project_dir) is None


def test_local_gpu_triggers_picker_without_remotes(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A GPU box with no BYOC remotes still gets a Cloud-vs-Local choice."""
    monkeypatch.setattr(handlers, "_local_gpu_available", lambda: True)
    options = _compute_pick_options(project_dir)
    assert options is not None
    assert options[0] == "LQH Cloud (recommended)"
    assert options[1] == "Local (this machine)"
    assert options[-1].startswith("Something else")


def test_local_option_included_alongside_remotes(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(handlers, "_local_gpu_available", lambda: True)
    _bind_remote(project_dir, "lab-gpu", "lab-gpu-01")
    options = _compute_pick_options(project_dir)
    assert options is not None
    assert "Local (this machine)" in options
    assert any(o.startswith("lab-gpu") for o in options)


def test_remote_but_nothing_chosen_returns_options(project_dir: Path):
    _bind_remote(project_dir, "lab-gpu", "lab-gpu-01")
    options = _compute_pick_options(project_dir)
    assert options is not None
    assert options[0] == "LQH Cloud (recommended)"
    assert options[1].startswith("lab-gpu")
    assert "lab-gpu-01" in options[1]
    assert options[-1].startswith("Something else")


def test_project_default_set_returns_none(project_dir: Path):
    _bind_remote(project_dir, "lab-gpu", "lab-gpu-01")
    save_project_default(project_dir, "cloud")
    assert _compute_pick_options(project_dir) is None


def test_global_default_set_returns_none(project_dir: Path):
    _bind_remote(project_dir, "lab-gpu", "lab-gpu-01")
    save_global_default("ssh:lab-gpu")
    assert _compute_pick_options(project_dir) is None


def test_local_default_resolves_to_local_branch(project_dir: Path):
    """A persisted 'local' default maps to None so callers run in-process."""
    from lqh.tools.handlers import _resolve_compute_target

    save_project_default(project_dir, "local")
    assert _resolve_compute_target(project_dir) is None


def test_cloud_default_passes_through(project_dir: Path):
    from lqh.tools.handlers import _resolve_compute_target

    save_project_default(project_dir, "cloud")
    assert _resolve_compute_target(project_dir) == "cloud"


def test_remotes_capped_at_max(project_dir: Path):
    for i in range(_MAX_PICKER_REMOTES + 3):
        _bind_remote(project_dir, f"gpu{i}", f"host{i}")
    options = _compute_pick_options(project_dir)
    assert options is not None
    # 1 cloud + capped remotes + 1 "something else"
    assert len(options) == _MAX_PICKER_REMOTES + 2
    assert options[0] == "LQH Cloud (recommended)"
    assert options[-1].startswith("Something else")
