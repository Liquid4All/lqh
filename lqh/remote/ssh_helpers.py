"""Shared SSH and rsync utilities for remote backends.

All functions are async and use ``asyncio.create_subprocess_exec`` so they
integrate cleanly with the main event loop.  SSH ControlMaster multiplexing
is used to avoid repeated handshakes within a session.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "ssh_run",
    "ssh_check",
    "rsync_push",
    "rsync_pull",
]

# Directory for ControlMaster sockets.
_CONTROL_DIR = Path.home() / ".lqh" / "ssh"


def _shell_quote(s: str) -> str:
    """Quote a string for use as a single bash -c argument."""
    import shlex
    return shlex.quote(s)


def _ensure_control_dir() -> None:
    _CONTROL_DIR.mkdir(parents=True, exist_ok=True)


def _multiplex_args(hostname: str) -> list[str]:
    """SSH options for ControlMaster connection reuse."""
    _ensure_control_dir()
    socket_path = _CONTROL_DIR / f"{hostname}.sock"
    return [
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={socket_path}",
        "-o", "ControlPersist=600",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
    ]


async def ssh_run(
    hostname: str,
    command: str,
    *,
    timeout: float = 30.0,
    env: dict[str, str] | None = None,
) -> tuple[str, str, int]:
    """Execute a command on the remote host via SSH.

    Returns ``(stdout, stderr, returncode)``.
    """
    # Force bash as the remote shell to avoid fish/zsh compatibility issues
    # with venv activation and other bash-isms.
    bash_command = f"bash -lc {_shell_quote(command)}"
    ssh_cmd = ["ssh"] + _multiplex_args(hostname) + [hostname, bash_command]

    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    logger.debug("SSH %s: %s", hostname, command)

    proc = await asyncio.create_subprocess_exec(
        *ssh_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=process_env,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(
            f"SSH command timed out after {timeout}s: {command}"
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    returncode = proc.returncode or 0

    if returncode != 0:
        logger.debug(
            "SSH %s exit %d: stderr=%s", hostname, returncode, stderr[:200],
        )

    return stdout, stderr, returncode


async def ssh_check(hostname: str) -> bool:
    """Check if the remote host is reachable via SSH."""
    try:
        _, _, rc = await ssh_run(hostname, "echo ok", timeout=15.0)
        return rc == 0
    except (TimeoutError, OSError):
        return False


async def rsync_push(
    hostname: str,
    local_paths: list[str],
    remote_dir: str,
    *,
    delete: bool = False,
) -> None:
    """Push local files/directories to a remote directory via rsync.

    Each path in *local_paths* is synced into *remote_dir* on the remote.
    Directories are synced recursively.  Uses SSH ControlMaster for the
    transport.
    """
    if not local_paths:
        return

    _ensure_control_dir()
    ssh_opts = " ".join(_multiplex_args(hostname))

    cmd: list[str] = [
        "rsync",
        "-az",
        "--partial",
        "-e", f"ssh {ssh_opts}",
    ]
    if delete:
        cmd.append("--delete")

    # No trailing slash on directories — rsync copies the directory itself
    # (preserving its name) into the destination.
    sources: list[str] = [str(Path(p)) for p in local_paths]

    cmd.extend(sources)
    cmd.append(f"{hostname}:{remote_dir}/")

    logger.debug("rsync push: %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"rsync push failed (exit {proc.returncode}): {err}")


async def rsync_pull(
    hostname: str,
    remote_path: str,
    local_dir: str,
    *,
    include_patterns: list[str] | None = None,
) -> None:
    """Pull files from a remote path into a local directory via rsync.

    If *include_patterns* is provided, only files matching those patterns
    are transferred (using rsync ``--include``/``--exclude`` rules).
    """
    _ensure_control_dir()
    ssh_opts = " ".join(_multiplex_args(hostname))

    cmd: list[str] = [
        "rsync",
        "-az",
        "--partial",
        "-e", f"ssh {ssh_opts}",
    ]

    if include_patterns:
        for pattern in include_patterns:
            cmd.extend(["--include", pattern])
        # Include parent directories so rsync can traverse
        cmd.extend(["--include", "*/"])
        cmd.extend(["--exclude", "*"])

    # Ensure remote path has trailing slash for content sync
    remote = f"{hostname}:{remote_path.rstrip('/')}/"
    local = str(Path(local_dir)) + "/"

    cmd.extend([remote, local])

    logger.debug("rsync pull: %s", " ".join(cmd))

    # Ensure local directory exists
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"rsync pull failed (exit {proc.returncode}): {err}")
