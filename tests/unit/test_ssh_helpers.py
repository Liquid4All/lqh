"""Test SSH helper functions with mocked subprocess calls."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lqh.remote.ssh_helpers import (
    _multiplex_args,
    rsync_pull,
    rsync_push,
    ssh_check,
    ssh_run,
)


class TestMultiplexArgs:
    """Test SSH ControlMaster argument generation."""

    def test_includes_control_master(self):
        args = _multiplex_args("myhost")
        assert "-o" in args
        assert "ControlMaster=auto" in args

    def test_includes_control_path_with_hostname(self):
        args = _multiplex_args("gpu-box-01")
        # Find the ControlPath argument
        control_path = None
        for i, arg in enumerate(args):
            if arg.startswith("ControlPath="):
                control_path = arg
                break
        assert control_path is not None
        assert "gpu-box-01.sock" in control_path

    def test_batch_mode(self):
        args = _multiplex_args("host")
        assert "BatchMode=yes" in args


class TestSSHRun:
    """Test ssh_run with mocked asyncio.create_subprocess_exec."""

    @pytest.mark.asyncio
    async def test_successful_command(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))
        mock_proc.returncode = 0

        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            stdout, stderr, rc = await ssh_run("host", "echo hello")

        assert stdout == "hello"
        assert rc == 0
        # Verify ssh was called with the hostname and bash-wrapped command
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "ssh"
        assert "host" in call_args
        # Command is wrapped: bash -lc 'echo hello'
        bash_arg = [a for a in call_args if "echo hello" in a]
        assert bash_arg, f"Expected 'echo hello' in args: {call_args}"

    @pytest.mark.asyncio
    async def test_command_failure(self, caplog):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"not found\n"))
        mock_proc.returncode = 127

        caplog.set_level(logging.DEBUG, logger="lqh.remote.ssh_helpers")
        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec", return_value=mock_proc):
            stdout, stderr, rc = await ssh_run("host", "bad-command")

        assert rc == 127
        assert "not found" in stderr
        assert "SSH host exit 127: stderr=not found" in caplog.text

    @pytest.mark.asyncio
    async def test_timeout(self):
        mock_proc = AsyncMock()

        async def slow_communicate():
            await asyncio.sleep(10)
            return b"", b""

        mock_proc.communicate = slow_communicate
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(TimeoutError, match="timed out"):
                await ssh_run("host", "sleep 100", timeout=0.1)


class TestSSHCheck:
    """Test ssh_check."""

    @pytest.mark.asyncio
    async def test_reachable(self):
        with patch("lqh.remote.ssh_helpers.ssh_run", return_value=("ok", "", 0)):
            assert await ssh_check("host") is True

    @pytest.mark.asyncio
    async def test_unreachable(self):
        with patch("lqh.remote.ssh_helpers.ssh_run", return_value=("", "timeout", 255)):
            assert await ssh_check("host") is False

    @pytest.mark.asyncio
    async def test_timeout_exception(self):
        with patch("lqh.remote.ssh_helpers.ssh_run", side_effect=TimeoutError):
            assert await ssh_check("host") is False


class TestRsyncPush:
    """Test rsync_push command construction."""

    @pytest.mark.asyncio
    async def test_push_files(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "data.parquet"
        test_file.write_text("data")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await rsync_push("host", [str(test_file)], "/remote/dir")

        call_args = mock_exec.call_args[0]
        assert call_args[0] == "rsync"
        assert str(test_file) in call_args
        assert "host:/remote/dir/" in call_args

    @pytest.mark.asyncio
    async def test_push_empty_list(self):
        """Empty path list should be a no-op."""
        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec") as mock_exec:
            await rsync_push("host", [], "/remote/dir")
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_failure(self, tmp_path):
        test_file = tmp_path / "f.txt"
        test_file.write_text("x")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error\n"))
        mock_proc.returncode = 1

        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="rsync push failed"):
                await rsync_push("host", [str(test_file)], "/remote")


class TestRsyncPull:
    """Test rsync_pull command construction."""

    @pytest.mark.asyncio
    async def test_pull_basic(self, tmp_path):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await rsync_pull("host", "/remote/run", str(tmp_path / "local"))

        call_args = mock_exec.call_args[0]
        assert "host:/remote/run/" in call_args

    @pytest.mark.asyncio
    async def test_pull_with_patterns(self, tmp_path):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await rsync_pull(
                "host", "/remote/run", str(tmp_path / "local"),
                include_patterns=["progress.jsonl", "*.json"],
            )

        call_args = mock_exec.call_args[0]
        assert "--include" in call_args
        assert "progress.jsonl" in call_args
        assert "--exclude" in call_args

    @pytest.mark.asyncio
    async def test_pull_failure(self, tmp_path):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error\n"))
        mock_proc.returncode = 1

        with patch("lqh.remote.ssh_helpers.asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="rsync pull failed"):
                await rsync_pull("host", "/remote", str(tmp_path / "local"))
