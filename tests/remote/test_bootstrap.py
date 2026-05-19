"""Test bootstrap logic with mocked SSH commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from lqh.remote.bootstrap import (
    _detect_login_shell,
    _locate_tool,
    _UV_CANDIDATE_PATHS,
    _which_via_login_shell,
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
            if "echo $SHELL" in cmd:
                return ("/bin/bash", "", 0)
            if "command -v python3" in cmd:
                return ("/usr/bin/python3", "", 0)
            if "command -v uv" in cmd:
                return ("/usr/local/bin/uv", "", 0)
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh):
            env = await detect_environment("host")

        assert env["login_shell"] == "/bin/bash"
        assert env["python3"] == "/usr/bin/python3"
        assert env["uv"] == "/usr/local/bin/uv"
        assert env["pip"] is True
        assert env["gpu_vendor"] == "nvidia"
        assert env["module"] is True

    @pytest.mark.asyncio
    async def test_only_python_available(self):
        async def mock_ssh(hostname, cmd, **kw):
            if "echo $SHELL" in cmd:
                return ("/bin/bash", "", 0)
            if "python3" in cmd:
                return ("/usr/bin/python3", "", 0)
            return ("", "", 1)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh), \
             patch("lqh.remote.gpu.ssh_run", side_effect=mock_ssh):
            env = await detect_environment("host")

        assert env["python3"] == "/usr/bin/python3"
        assert env["uv"] is None
        assert env["pip"] is False
        assert env["gpu_vendor"] is None


class TestDetectLoginShell:
    """Test login-shell detection."""

    @pytest.mark.asyncio
    async def test_returns_shell_path(self):
        async def mock_ssh(hostname, cmd, **kw):
            return ("/usr/bin/fish", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _detect_login_shell("host") == "/usr/bin/fish"

    @pytest.mark.asyncio
    async def test_skips_nologin_accounts(self):
        async def mock_ssh(hostname, cmd, **kw):
            return ("/sbin/nologin", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _detect_login_shell("host") is None

    @pytest.mark.asyncio
    async def test_skips_false(self):
        async def mock_ssh(hostname, cmd, **kw):
            return ("/bin/false", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _detect_login_shell("host") is None

    @pytest.mark.asyncio
    async def test_empty_returns_none(self):
        async def mock_ssh(hostname, cmd, **kw):
            return ("", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _detect_login_shell("host") is None


class TestWhichViaLoginShell:
    """Test resolving tools through the user's actual shell."""

    @pytest.mark.asyncio
    async def test_invokes_user_shell_with_lc(self):
        seen: dict[str, str] = {}

        async def mock_ssh(hostname, cmd, **kw):
            seen["cmd"] = cmd
            return ("/home/me/.local/bin/uv", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            result = await _which_via_login_shell(
                "host", "/usr/bin/fish", "uv",
            )

        assert result == "/home/me/.local/bin/uv"
        # The wrapping bash -lc is supplied by ssh_run; this command
        # should invoke fish with -lc so fish's own config files are
        # sourced (which is the whole point — they're where fish users'
        # PATH additions live).
        assert "/usr/bin/fish" in seen["cmd"]
        assert "-lc" in seen["cmd"]
        assert "command -v uv" in seen["cmd"]

    @pytest.mark.asyncio
    async def test_strips_motd_greeting(self):
        """A login shell may print a banner before the command's output."""
        banner = "Welcome to Acme HPC, last login Tue\n/snap/bin/uv\n"

        async def mock_ssh(hostname, cmd, **kw):
            return (banner, "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _which_via_login_shell(
                "host", "/usr/bin/fish", "uv",
            ) == "/snap/bin/uv"

    @pytest.mark.asyncio
    async def test_rejects_non_absolute_output(self):
        """`command -v` echoes bare names for aliases/builtins — skip."""
        async def mock_ssh(hostname, cmd, **kw):
            return ("uv: aliased to /weird\n", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _which_via_login_shell(
                "host", "/usr/bin/fish", "uv",
            ) is None

    @pytest.mark.asyncio
    async def test_shell_error_returns_none(self):
        async def mock_ssh(hostname, cmd, **kw):
            return ("", "syntax error", 2)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _which_via_login_shell(
                "host", "/usr/bin/fish", "uv",
            ) is None


class TestLocateTool:
    """Test the layered tool-resolution helper."""

    @pytest.mark.asyncio
    async def test_login_shell_finds_it(self):
        """If fish/zsh resolves uv, we use that and stop."""
        calls: list[str] = []

        async def mock_ssh(hostname, cmd, **kw):
            calls.append(cmd)
            # First call is the fish -lc invocation — return a hit.
            if "/usr/bin/fish" in cmd:
                return ("/home/me/.local/bin/uv", "", 0)
            pytest.fail(f"should have stopped after shell hit: {cmd}")

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            result = await _locate_tool(
                "host", "uv",
                candidates=_UV_CANDIDATE_PATHS,
                login_shell="/usr/bin/fish",
            )

        assert result == "/home/me/.local/bin/uv"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_bash_path(self):
        """Shell can't find it → ask bash directly."""
        async def mock_ssh(hostname, cmd, **kw):
            if "/usr/bin/fish" in cmd:
                return ("", "", 1)
            if cmd.startswith("command -v uv"):
                return ("/usr/local/bin/uv", "", 0)
            pytest.fail(f"unexpected probe: {cmd}")

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _locate_tool(
                "host", "uv",
                candidates=_UV_CANDIDATE_PATHS,
                login_shell="/usr/bin/fish",
            ) == "/usr/local/bin/uv"

    @pytest.mark.asyncio
    async def test_falls_back_to_canonical_probe(self):
        """Neither shell nor bash knows → walk the candidate list."""
        captured: dict[str, str] = {}

        async def mock_ssh(hostname, cmd, **kw):
            if "/usr/bin/fish" in cmd:
                return ("", "", 1)
            if cmd.startswith("command -v uv"):
                return ("", "", 1)
            captured["probe"] = cmd
            return ("/snap/bin/uv", "", 0)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _locate_tool(
                "host", "uv",
                candidates=_UV_CANDIDATE_PATHS,
                login_shell="/usr/bin/fish",
            ) == "/snap/bin/uv"

        # Probe must enumerate every documented location.
        for expected in _UV_CANDIDATE_PATHS:
            assert expected in captured["probe"]

    @pytest.mark.asyncio
    async def test_no_login_shell_skips_first_probe(self):
        """When $SHELL is /bin/false we shouldn't try to exec it."""
        calls: list[str] = []

        async def mock_ssh(hostname, cmd, **kw):
            calls.append(cmd)
            if cmd.startswith("command -v uv"):
                return ("/usr/bin/uv", "", 0)
            return ("", "", 1)

        with patch("lqh.remote.bootstrap.ssh_run", side_effect=mock_ssh):
            assert await _locate_tool(
                "host", "uv",
                candidates=_UV_CANDIDATE_PATHS,
                login_shell=None,
            ) == "/usr/bin/uv"

        assert not any("/bin/false" in c for c in calls)
        assert not any("-lc" in c and "fish" in c for c in calls)


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
