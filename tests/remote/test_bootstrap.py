"""Test bootstrap logic with mocked SSH commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from lqh.remote.bootstrap import (
    bootstrap_remote,
    check_hf_token,
    configure_hf_token,
    detect_environment,
)


class TestDetectEnvironment:
    """Test remote environment detection."""

    @pytest.mark.asyncio
    async def test_all_tools_available(self):
        async def mock_ssh(hostname, cmd, **kw):
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh):
            env = await detect_environment("host")

        assert env["python3"] is True
        assert env["uv"] is True
        assert env["pip"] is True
        assert env["gpu_vendor"] == "nvidia"
        assert env["module"] is True

    @pytest.mark.asyncio
    async def test_only_python_available(self):
        async def mock_ssh(hostname, cmd, **kw):
            if "python3" in cmd:
                return ("/usr/bin/python3", "", 0)
            return ("", "", 1)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh):
            env = await detect_environment("host")

        assert env["python3"] is True
        assert env["uv"] is False
        assert env["pip"] is False
        assert env["gpu_vendor"] is None


def _mock_rsync_noop(*args, **kwargs):
    """No-op async mock for rsync and subprocess calls in bootstrap."""

    async def _noop():
        pass

    return _noop()


class TestBootstrapRemote:
    """Test the full bootstrap flow."""

    @pytest.mark.asyncio
    async def test_bootstrap_with_uv(self):
        calls: list[str] = []

        async def mock_ssh(hostname, cmd, **kw):
            calls.append(cmd)
            if "command -v python3" in cmd:
                return ("/usr/bin/python3", "", 0)
            if "command -v uv" in cmd:
                return ("/usr/bin/uv", "", 0)
            if "command -v nvidia-smi" in cmd:
                return ("/usr/bin/nvidia-smi", "", 0)
            if "nvidia-smi --query-gpu=index,name,memory.total" in cmd:
                return ("0, NVIDIA A100-SXM4-80GB, 81920", "", 0)
            if cmd.startswith("test -x ") and "/bin/python" in cmd:
                return ("", "", 1)  # venv does not exist yet → exercise creation path
            return ("", "", 0)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.bootstrap._find_lqh_package_root", return_value=None):
            log = await bootstrap_remote("host", "/remote/root")

        assert "uv" in log.lower() or "venv" in log.lower()
        # Should have used uv venv, not python3 -m venv
        venv_cmds = [c for c in calls if "venv" in c]
        assert any("uv venv" in c for c in venv_cmds)

    @pytest.mark.asyncio
    async def test_bootstrap_without_uv(self):
        calls: list[str] = []

        async def mock_ssh(hostname, cmd, **kw):
            calls.append(cmd)
            if "command -v python3" in cmd:
                return ("/usr/bin/python3", "", 0)
            if "command -v uv" in cmd:
                return ("", "", 1)  # uv not available
            if "command -v nvidia-smi" in cmd:
                return ("", "", 1)
            if "command -v amd-smi" in cmd or "command -v rocm-smi" in cmd:
                return ("", "", 1)
            if cmd.startswith("test -x ") and "/bin/python" in cmd:
                return ("", "", 1)  # venv does not exist yet → exercise creation path
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.bootstrap._find_lqh_package_root", return_value=None):
            await bootstrap_remote("host", "/remote/root")

        venv_cmds = [c for c in calls if "venv" in c]
        assert any("python3 -m venv" in c for c in venv_cmds)

    @pytest.mark.asyncio
    async def test_bootstrap_no_python(self):
        async def mock_ssh(hostname, cmd, **kw):
            if "python3" in cmd:
                return ("", "", 1)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh):
            with pytest.raises(RuntimeError, match="python3 not found"):
                await bootstrap_remote("host", "/remote/root")


class TestHFTokenConfig:
    """Test HF token configuration."""

    @pytest.mark.asyncio
    async def test_configure_hf_token(self):
        cmds: list[str] = []

        async def mock_ssh(hostname, cmd, **kw):
            cmds.append(cmd)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            await configure_hf_token("host", "/remote", "hf_test_token")

        # Should write to .env file
        assert any("HF_TOKEN=hf_test_token" in c for c in cmds)

    @pytest.mark.asyncio
    async def test_check_hf_token_in_env_file(self):
        async def mock_ssh(hostname, cmd, **kw):
            if "grep" in cmd:
                return ("yes", "", 0)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await check_hf_token("host", "/remote") is True

    @pytest.mark.asyncio
    async def test_check_hf_token_in_shell(self):
        async def mock_ssh(hostname, cmd, **kw):
            if "grep" in cmd:
                return ("", "", 1)  # Not in .env
            if "echo $HF_TOKEN" in cmd:
                return ("hf_abc123", "", 0)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await check_hf_token("host", "/remote") is True

    @pytest.mark.asyncio
    async def test_check_hf_token_missing(self):
        async def mock_ssh(hostname, cmd, **kw):
            if "grep" in cmd:
                return ("", "", 1)
            if "echo $HF_TOKEN" in cmd:
                return ("", "", 0)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await check_hf_token("host", "/remote") is False
