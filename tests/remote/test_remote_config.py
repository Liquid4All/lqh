"""Test remotes.json CRUD operations (global + project layers)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lqh.remote.backend import ProjectBinding, RemoteConfig, RemoteMachine
from lqh.remote.config import (
    add_binding,
    add_machine,
    add_remote,
    get_binding,
    get_machine,
    get_remote,
    load_bindings,
    load_machines,
    load_remotes,
    remove_binding,
    remove_machine,
    remove_remote,
    save_remotes,
)


@pytest.fixture(autouse=True)
def _isolate_global_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the global config dir to a temp directory for all tests."""
    import lqh.remote.config as config_mod

    global_dir = tmp_path / "global_lqh"
    global_dir.mkdir()
    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_DIR", global_dir)


class TestGlobalMachines:
    """Test global machine CRUD (~/.lqh/remotes.json)."""

    def test_empty_when_no_file(self):
        assert load_machines() == {}

    def test_add_and_get(self):
        add_machine(RemoteMachine(
            name="lab-gpu", type="ssh_direct", hostname="lab-gpu-01",
        ))
        result = get_machine("lab-gpu")
        assert result is not None
        assert result.hostname == "lab-gpu-01"

    def test_get_nonexistent(self):
        assert get_machine("nope") is None

    def test_remove(self):
        add_machine(RemoteMachine(
            name="rm-me", type="ssh_direct", hostname="h",
        ))
        remove_machine("rm-me")
        assert get_machine("rm-me") is None

    def test_remove_nonexistent(self):
        with pytest.raises(KeyError, match="not found"):
            remove_machine("nope")

    def test_validation_empty_name(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            add_machine(RemoteMachine(name="", type="ssh_direct", hostname="h"))

    def test_validation_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid remote type"):
            add_machine(RemoteMachine(name="x", type="kubernetes", hostname="h"))

    def test_validation_empty_hostname(self):
        with pytest.raises(ValueError, match="hostname cannot be empty"):
            add_machine(RemoteMachine(name="x", type="ssh_direct", hostname=""))


class TestProjectBindings:
    """Test project binding CRUD (.lqh/remotes.json)."""

    def test_empty_when_no_file(self, project_dir: Path):
        assert load_bindings(project_dir) == {}

    def test_add_and_get(self, project_dir: Path):
        add_binding(project_dir, ProjectBinding(
            name="lab-gpu", remote_root="/home/user/lqh/proj",
        ))
        result = get_binding(project_dir, "lab-gpu")
        assert result is not None
        assert result.remote_root == "/home/user/lqh/proj"

    def test_remove(self, project_dir: Path):
        add_binding(project_dir, ProjectBinding(
            name="rm-me", remote_root="/r",
        ))
        remove_binding(project_dir, "rm-me")
        assert get_binding(project_dir, "rm-me") is None

    def test_remove_nonexistent(self, project_dir: Path):
        with pytest.raises(KeyError, match="not found"):
            remove_binding(project_dir, "nope")

    def test_validation_empty_remote_root(self, project_dir: Path):
        with pytest.raises(ValueError, match="root path cannot be empty"):
            add_binding(project_dir, ProjectBinding(name="x", remote_root=""))


class TestMergedRemotes:
    """Test the high-level merged API (global + project)."""

    def test_add_and_get(self, project_dir: Path):
        config = RemoteConfig(
            name="lab-gpu",
            type="ssh_direct",
            hostname="lab-gpu-01",
            remote_root="/home/user/lqh/project",
        )
        add_remote(project_dir, config)

        result = get_remote(project_dir, "lab-gpu")
        assert result is not None
        assert result.hostname == "lab-gpu-01"
        assert result.type == "ssh_direct"
        assert result.remote_root == "/home/user/lqh/project"

    def test_get_nonexistent(self, project_dir: Path):
        assert get_remote(project_dir, "nope") is None

    def test_get_requires_both_layers(self, project_dir: Path):
        """get_remote returns None if machine exists but no binding."""
        add_machine(RemoteMachine(
            name="orphan", type="ssh_direct", hostname="h",
        ))
        assert get_remote(project_dir, "orphan") is None

    def test_add_multiple(self, project_dir: Path):
        add_remote(
            project_dir,
            RemoteConfig(name="a", type="ssh_direct", hostname="h1", remote_root="/r1"),
        )
        add_remote(
            project_dir,
            RemoteConfig(name="b", type="ssh_slurm", hostname="h2", remote_root="/r2"),
        )
        remotes = load_remotes(project_dir)
        assert len(remotes) == 2
        assert "a" in remotes
        assert "b" in remotes

    def test_update_existing(self, project_dir: Path):
        add_remote(
            project_dir,
            RemoteConfig(name="x", type="ssh_direct", hostname="old", remote_root="/r"),
        )
        add_remote(
            project_dir,
            RemoteConfig(name="x", type="ssh_direct", hostname="new", remote_root="/r"),
        )
        result = get_remote(project_dir, "x")
        assert result is not None
        assert result.hostname == "new"

    def test_remove_unbinds_only(self, project_dir: Path):
        """remove_remote only removes project binding, not global machine."""
        add_remote(
            project_dir,
            RemoteConfig(name="rm-me", type="ssh_direct", hostname="h", remote_root="/r"),
        )
        remove_remote(project_dir, "rm-me")
        # Merged view is gone
        assert get_remote(project_dir, "rm-me") is None
        # But global machine still exists
        assert get_machine("rm-me") is not None

    def test_remove_nonexistent(self, project_dir: Path):
        with pytest.raises(KeyError, match="not found"):
            remove_remote(project_dir, "nope")

    def test_project_gpu_override(self, project_dir: Path):
        """Project binding gpu_ids override machine-level."""
        add_machine(RemoteMachine(
            name="gpu-box", type="ssh_direct", hostname="h", gpu_ids=[0, 1, 2, 3],
        ))
        add_binding(project_dir, ProjectBinding(
            name="gpu-box", remote_root="/r", gpu_ids=[0, 1],
        ))
        result = get_remote(project_dir, "gpu-box")
        assert result is not None
        assert result.gpu_ids == [0, 1]

    def test_machine_gpu_used_when_no_override(self, project_dir: Path):
        add_machine(RemoteMachine(
            name="gpu-box", type="ssh_direct", hostname="h", gpu_ids=[0, 1],
        ))
        add_binding(project_dir, ProjectBinding(
            name="gpu-box", remote_root="/r",
        ))
        result = get_remote(project_dir, "gpu-box")
        assert result is not None
        assert result.gpu_ids == [0, 1]

    def test_json_file_format(self, project_dir: Path):
        add_remote(
            project_dir,
            RemoteConfig(
                name="gpu",
                type="ssh_direct",
                hostname="host",
                remote_root="/root",
                gpu_ids=[0],
            ),
        )
        # Check project-level file
        raw = json.loads((project_dir / ".lqh" / "remotes.json").read_text())
        assert "gpu" in raw
        assert raw["gpu"]["remote_root"] == "/root"
        # Check global file
        import lqh.remote.config as config_mod
        global_raw = json.loads(
            (config_mod.GLOBAL_CONFIG_DIR / "remotes.json").read_text()
        )
        assert "gpu" in global_raw
        assert global_raw["gpu"]["type"] == "ssh_direct"
        assert global_raw["gpu"]["gpu_ids"] == [0]

    def test_orphan_bindings_excluded(self, project_dir: Path):
        """Bindings without a matching global machine are excluded."""
        # Write a binding directly (no global machine)
        path = project_dir / ".lqh" / "remotes.json"
        path.write_text(json.dumps({
            "ghost": {"remote_root": "/r"}
        }))
        remotes = load_remotes(project_dir)
        assert len(remotes) == 0

    def test_save_remotes_preserves_other_globals(self, project_dir: Path):
        """save_remotes doesn't clobber machines used by other projects."""
        add_machine(RemoteMachine(
            name="other-project-machine", type="ssh_direct", hostname="h",
        ))
        save_remotes(project_dir, {
            "new": RemoteConfig(
                name="new", type="ssh_direct", hostname="h2", remote_root="/r",
            ),
        })
        # Both should exist in global config
        machines = load_machines()
        assert "other-project-machine" in machines
        assert "new" in machines
