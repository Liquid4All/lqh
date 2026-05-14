"""Test that RemoteBackend ABC contract is properly enforced."""

from __future__ import annotations

import inspect

import pytest

from lqh.remote.backend import RemoteBackend, RemoteConfig, JobStatus


class TestRemoteBackendABC:
    """Verify the ABC defines the expected abstract methods."""

    def test_abstract_methods(self):
        expected = {
            "setup",
            "submit_run",
            "poll_status",
            "sync_progress",
            "sync_file_to_remote",
            "sync_file_from_remote",
            "is_job_alive",
            "teardown",
        }
        abstract = set(RemoteBackend.__abstractmethods__)
        assert abstract == expected

    def test_cannot_instantiate_abc(self):
        config = RemoteConfig(
            name="test", type="ssh_direct", hostname="h", remote_root="/r",
        )
        with pytest.raises(TypeError, match="abstract method"):
            RemoteBackend(config)

    def test_ssh_direct_implements_interface(self):
        from lqh.remote.ssh_direct import SSHDirectBackend
        # Verify all abstract methods are implemented (not abstract)
        assert not getattr(SSHDirectBackend, "__abstractmethods__", set())

    def test_ssh_slurm_implements_interface(self):
        from lqh.remote.ssh_slurm import SSHSlurmBackend
        assert not getattr(SSHSlurmBackend, "__abstractmethods__", set())


class TestRemoteConfig:
    """Test RemoteConfig serialization."""

    def test_round_trip(self):
        config = RemoteConfig(
            name="lab-gpu",
            type="ssh_direct",
            hostname="lab-gpu-01",
            remote_root="/home/user/lqh/project",
            gpu_ids=[0, 1],
            hf_token_configured=True,
            extra={"custom_key": "value"},
        )
        d = config.to_dict()
        restored = RemoteConfig.from_dict("lab-gpu", d)
        assert restored.name == config.name
        assert restored.type == config.type
        assert restored.hostname == config.hostname
        assert restored.remote_root == config.remote_root
        assert restored.gpu_ids == config.gpu_ids
        assert restored.hf_token_configured == config.hf_token_configured
        assert restored.extra == {"custom_key": "value"}

    def test_minimal_config(self):
        d = {"type": "ssh_direct", "hostname": "h", "remote_root": "/r"}
        config = RemoteConfig.from_dict("test", d)
        assert config.instructions_file is None
        assert config.gpu_ids is None
        assert config.hf_token_configured is False
        assert config.extra == {}


class TestJobStatus:
    """Test JobStatus serialization."""

    def test_from_status_json(self):
        data = {
            "state": "running",
            "pid": 12345,
            "current_step": 500,
            "total_steps": 2000,
            "started_at": "2026-04-01T10:00:00Z",
            "last_update": "2026-04-01T10:05:00Z",
        }
        status = JobStatus.from_status_json(data)
        assert status.state == "running"
        assert status.pid == 12345
        assert status.current_step == 500

    def test_to_dict_omits_none(self):
        status = JobStatus(state="completed")
        d = status.to_dict()
        assert d == {"state": "completed"}
        assert "pid" not in d

    def test_unknown_state_default(self):
        status = JobStatus.from_status_json({})
        assert status.state == "unknown"
