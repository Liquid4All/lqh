"""Test bootstrap logic with mocked SSH commands."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lqh.remote.bootstrap import (
    _UV_CANDIDATE_PATHS,
    _locate_uv,
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
            if "command -v uv" in cmd:
                return ("/usr/local/bin/uv", "", 0)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh):
            env = await detect_environment("host")

        assert env["python3"] is True
        assert env["uv"] == "/usr/local/bin/uv"
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
        assert env["uv"] is None
        assert env["pip"] is False
        assert env["gpu_vendor"] is None


class TestLocateUv:
    """Test the uv discovery helper."""

    @pytest.mark.asyncio
    async def test_found_on_path(self):
        async def mock_ssh(hostname, cmd, **kw):
            if "command -v uv" in cmd:
                return ("/usr/bin/uv", "", 0)
            pytest.fail(f"unexpected probe after PATH hit: {cmd}")

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _locate_uv("host") == "/usr/bin/uv"

    @pytest.mark.asyncio
    async def test_found_via_snap(self):
        async def mock_ssh(hostname, cmd, **kw):
            if "command -v uv" in cmd:
                return ("", "", 1)
            # Probe runs the for-loop and finds /snap/bin/uv.
            assert "/snap/bin/uv" in cmd
            return ("/snap/bin/uv", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _locate_uv("host") == "/snap/bin/uv"

    @pytest.mark.asyncio
    async def test_probe_covers_canonical_locations(self):
        """The probe script must reference every documented install dir."""
        captured: dict[str, str] = {}

        async def mock_ssh(hostname, cmd, **kw):
            if "command -v uv" in cmd:
                return ("", "", 1)
            captured["probe"] = cmd
            return ("", "", 1)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _locate_uv("host") is None

        for expected in _UV_CANDIDATE_PATHS:
            assert expected in captured["probe"]

    @pytest.mark.asyncio
    async def test_not_found(self):
        async def mock_ssh(hostname, cmd, **kw):
            return ("", "", 1)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _locate_uv("host") is None


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
                return ("/opt/uv-build/uv", "", 0)
            if "command -v nvidia-smi" in cmd:
                return ("/usr/bin/nvidia-smi", "", 0)
            if "nvidia-smi --query-gpu=index,name,memory.total" in cmd:
                return ("0, NVIDIA A100-SXM4-80GB, 81920", "", 0)
            # Force venv-doesn't-exist so the creation branch runs.
            if cmd.startswith("test -x") and "/bin/python" in cmd:
                return ("", "", 1)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.bootstrap._find_lqh_package_root", return_value=None):
            await bootstrap_remote("host", "/remote/root")

        # The absolute uv path returned by detection must be the one used
        # for venv creation — not the bare `uv` token.
        venv_cmds = [c for c in calls if "venv" in c]
        assert any("/opt/uv-build/uv venv" in c for c in venv_cmds), venv_cmds
        # And for `uv pip install` of lqh[train].
        install_cmds = [c for c in calls if "lqh[train]" in c]
        assert any("/opt/uv-build/uv pip install" in c for c in install_cmds), install_cmds

    @pytest.mark.asyncio
    async def test_bootstrap_without_uv(self):
        """When uv isn't installed we fall back to `python3 -m venv`
        and `pip install`."""
        calls: list[str] = []

        async def mock_ssh(hostname, cmd, **kw):
            calls.append(cmd)
            if "command -v python3" in cmd:
                return ("/usr/bin/python3", "", 0)
            if "command -v uv" in cmd:
                return ("", "", 1)
            if "command -v nvidia-smi" in cmd:
                return ("", "", 1)
            if "command -v amd-smi" in cmd or "command -v rocm-smi" in cmd:
                return ("", "", 1)
            if cmd.startswith("test -x") and "/bin/python" in cmd:
                return ("", "", 1)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.bootstrap._find_lqh_package_root", return_value=None):
            await bootstrap_remote("host", "/remote/root")

        venv_cmds = [c for c in calls if "venv" in c]
        assert any("python3 -m venv" in c for c in venv_cmds), venv_cmds
        install_cmds = [c for c in calls if "lqh[train]" in c]
        assert any("pip install" in c and "uv" not in c for c in install_cmds), install_cmds

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
