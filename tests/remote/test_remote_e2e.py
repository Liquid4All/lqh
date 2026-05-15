"""E2E tests for remote fine-tuning.

These tests require a real SSH-accessible GPU host.  Skip when no
``--remote-host`` is provided.

Usage::

    pytest tests/remote/test_remote_e2e.py --remote-host=lab-gpu-01
    LQH_TEST_REMOTE_HOST=lab-gpu-01 pytest tests/remote/test_remote_e2e.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio

from lqh.remote.backend import RemoteConfig
from lqh.remote.ssh_direct import SSHDirectBackend
from lqh.remote.ssh_helpers import ssh_check, ssh_run


@pytest_asyncio.fixture
async def ssh_backend(
    remote_host: str, remote_project_dir: tuple[str, Path],
) -> SSHDirectBackend:
    """Provision a clean SSHDirect backend for the test."""
    remote_root, local_dir = remote_project_dir
    config = RemoteConfig(
        name="e2e-test",
        type="ssh_direct",
        hostname=remote_host,
        remote_root=remote_root,
    )
    backend = SSHDirectBackend(config, local_dir)
    yield backend
    # Cleanup: remove remote directory
    await ssh_run(remote_host, f"rm -rf {remote_root}", timeout=30.0)


class TestRemoteSetup:
    """Test bootstrapping a fresh remote environment."""

    @pytest.mark.asyncio
    async def test_ssh_reachable(self, remote_host: str):
        """Sanity check: can we reach the remote host?"""
        assert await ssh_check(remote_host), f"Cannot reach {remote_host}"

    @pytest.mark.asyncio
    async def test_bootstrap(self, ssh_backend: SSHDirectBackend, remote_host: str):
        """Full bootstrap: create venv, install deps."""
        log = await ssh_backend.setup()
        assert "Setup complete" in log

        # Verify directory structure
        remote_root = ssh_backend._remote_root
        stdout, _, rc = await ssh_run(
            remote_host, f"ls {remote_root}/", timeout=10.0,
        )
        assert rc == 0
        assert "datasets" in stdout
        assert "runs" in stdout

        # Verify venv exists
        stdout, _, rc = await ssh_run(
            remote_host,
            f"test -f {remote_root}/.lqh-env/bin/python && echo yes",
            timeout=10.0,
        )
        assert "yes" in stdout

    @pytest.mark.asyncio
    async def test_detect_gpu(self, remote_host: str):
        """Check that the remote host has a GPU."""
        from lqh.remote.bootstrap import detect_environment

        env = await detect_environment(remote_host)
        assert env["python3"], "python3 not found on remote"
        # Log GPU info for visibility
        if env["nvidia_smi"]:
            stdout, _, _ = await ssh_run(
                remote_host,
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                timeout=10.0,
            )
            print(f"GPUs on {remote_host}: {stdout}")

    @pytest.mark.asyncio
    async def test_ssh_run_basic(self, remote_host: str):
        """Test basic command execution."""
        stdout, stderr, rc = await ssh_run(remote_host, "echo hello", timeout=10.0)
        assert rc == 0
        assert stdout == "hello"

    @pytest.mark.asyncio
    async def test_ssh_run_multicommand(self, remote_host: str):
        """Test multi-command execution."""
        stdout, _, rc = await ssh_run(
            remote_host, "echo one && echo two", timeout=10.0,
        )
        assert rc == 0
        assert "one" in stdout
        assert "two" in stdout


class TestRemoteFileSync:
    """Test file sync operations."""

    @pytest.mark.asyncio
    async def test_push_and_pull_file(
        self, ssh_backend: SSHDirectBackend, remote_host: str, tmp_path: Path,
    ):
        """Push a file to remote, pull it back, verify contents."""
        remote_root = ssh_backend._remote_root
        # Create remote dir
        await ssh_run(remote_host, f"mkdir -p {remote_root}/test_sync", timeout=10.0)

        # Create local file
        local_file = tmp_path / "test.txt"
        local_file.write_text("hello from local\n")

        # Push
        await ssh_backend.sync_file_to_remote(
            str(local_file),
            f"{remote_root}/test_sync/test.txt",
        )

        # Verify on remote
        stdout, _, rc = await ssh_run(
            remote_host, f"cat {remote_root}/test_sync/test.txt", timeout=10.0,
        )
        assert rc == 0
        assert "hello from local" in stdout

        # Pull back to different location
        pulled = tmp_path / "pulled.txt"
        await ssh_backend.sync_file_from_remote(
            f"{remote_root}/test_sync/test.txt",
            str(pulled),
        )
        assert pulled.read_text().strip() == "hello from local"

    @pytest.mark.asyncio
    async def test_sync_progress(
        self, ssh_backend: SSHDirectBackend, remote_host: str, tmp_path: Path,
    ):
        """Test syncing progress.jsonl from remote."""
        remote_root = ssh_backend._remote_root
        remote_run = f"{remote_root}/runs/test_run"

        # Create remote run dir with progress
        await ssh_run(
            remote_host,
            f'mkdir -p {remote_run} && echo \'{{"step":10,"loss":2.5}}\' > {remote_run}/progress.jsonl',
            timeout=10.0,
        )

        # Sync to local
        local_run = tmp_path / "runs" / "test_run"
        local_run.mkdir(parents=True)
        await ssh_backend.sync_progress(remote_run, str(local_run))

        # Verify local progress file
        progress = local_run / "progress.jsonl"
        assert progress.exists()
        data = json.loads(progress.read_text().strip())
        assert data["step"] == 10
        assert data["loss"] == 2.5


class TestRemoteJobLifecycle:
    """Test submitting and managing a remote job."""

    @pytest.mark.asyncio
    async def test_submit_simple_script(
        self, ssh_backend: SSHDirectBackend, remote_host: str, tmp_path: Path,
    ):
        """Submit a simple Python script (not lqh.train) to verify the launch mechanism."""
        remote_root = ssh_backend._remote_root

        # Bootstrap first
        await ssh_backend.setup()

        # Create a simple script that writes progress and exits
        script = (
            "import json, os, time\n"
            "run_dir = os.path.dirname(os.path.abspath(__file__))\n"
            "# Write PID\n"
            "with open(os.path.join(run_dir, 'pid'), 'w') as f:\n"
            "    f.write(str(os.getpid()))\n"
            "# Write progress\n"
            "with open(os.path.join(run_dir, 'progress.jsonl'), 'w') as f:\n"
            "    for i in range(5):\n"
            "        f.write(json.dumps({'step': i, 'loss': 2.0 - i * 0.3}) + '\\n')\n"
            "        f.flush()\n"
            "        time.sleep(0.5)\n"
            "    f.write(json.dumps({'status': 'completed', 'step': 4}) + '\\n')\n"
        )

        # Write script to remote
        remote_run = f"{remote_root}/runs/test_job"
        await ssh_run(remote_host, f"mkdir -p {remote_run}", timeout=10.0)

        local_script = tmp_path / "test_script.py"
        local_script.write_text(script)
        await ssh_backend.sync_file_to_remote(
            str(local_script), f"{remote_run}/test_script.py",
        )

        # Launch via nohup
        activate = f"source {remote_root}/.lqh-env/bin/activate"
        cmd = (
            f"{activate} && "
            f"nohup python {remote_run}/test_script.py "
            f"> {remote_run}/stdout.log 2> {remote_run}/stderr.log &"
        )
        _, _, rc = await ssh_run(remote_host, cmd, timeout=10.0)
        assert rc == 0

        # Wait for PID file
        for _ in range(10):
            stdout, _, rc = await ssh_run(
                remote_host, f"cat {remote_run}/pid 2>/dev/null", timeout=5.0,
            )
            if rc == 0 and stdout.strip().isdigit():
                pid = stdout.strip()
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("PID file never appeared")

        # Check liveness (may or may not still be alive — script is short)
        await ssh_backend.is_job_alive(pid)

        # Wait for completion
        for _ in range(20):
            stdout, _, rc = await ssh_run(
                remote_host, f"tail -1 {remote_run}/progress.jsonl 2>/dev/null",
                timeout=5.0,
            )
            if rc == 0 and "completed" in stdout:
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Job never completed")

        # Sync progress back and verify
        local_run = tmp_path / "local_run"
        local_run.mkdir()
        await ssh_backend.sync_progress(remote_run, str(local_run))

        progress_file = local_run / "progress.jsonl"
        assert progress_file.exists()
        lines = progress_file.read_text().strip().split("\n")
        assert len(lines) == 6  # 5 steps + 1 completed
        last = json.loads(lines[-1])
        assert last["status"] == "completed"

    @pytest.mark.asyncio
    async def test_is_job_alive_dead_pid(
        self, ssh_backend: SSHDirectBackend, remote_host: str,
    ):
        """is_job_alive should return False for a non-existent PID."""
        alive = await ssh_backend.is_job_alive("999999999")
        assert alive is False
