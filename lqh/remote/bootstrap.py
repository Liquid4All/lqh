"""Remote environment detection and provisioning.

Connects via SSH to detect available tools (python3, uv, pip, nvidia-smi)
and installs a Python environment with ``lqh[train]`` dependencies.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from lqh.remote.ssh_helpers import ssh_run

logger = logging.getLogger(__name__)

__all__ = [
    "detect_environment",
    "bootstrap_remote",
    "configure_hf_token",
]


# Fallback install locations searched when neither the user's login
# shell nor bash knows about the tool on PATH. These cover the layouts
# real GPU boxes actually use: Astral's installer (~/.local/bin), cargo
# (~/.cargo/bin), snap (/snap/bin), manual installs, and Homebrew on
# macOS / Linux.
_UV_CANDIDATE_PATHS: tuple[str, ...] = (
    "$HOME/.local/bin/uv",
    "$HOME/.cargo/bin/uv",
    "/snap/bin/uv",
    "/usr/local/bin/uv",
    "/opt/homebrew/bin/uv",
    "/home/linuxbrew/.linuxbrew/bin/uv",
    "/usr/bin/uv",
)

_PYTHON3_CANDIDATE_PATHS: tuple[str, ...] = (
    "/usr/bin/python3",
    "/usr/local/bin/python3",
    "/opt/homebrew/bin/python3",
    "$HOME/.local/bin/python3",
    "/snap/bin/python3",
)

# Login-shell entries that aren't real shells we can exec for a `-lc`
# lookup. Service accounts and locked accounts frequently use these.
_NON_INTERACTIVE_SHELLS: tuple[str, ...] = ("/false", "/nologin", "/true")


async def _detect_login_shell(hostname: str) -> str | None:
    """Return the remote user's login shell (e.g. ``/usr/bin/fish``).

    Reads ``$SHELL``, which SSH sets from the user's ``/etc/passwd``
    entry on session start. Returns ``None`` if it's unset or points
    to a stub shell we can't usefully exec.
    """
    stdout, _, rc = await ssh_run(hostname, "echo $SHELL", timeout=10.0)
    shell = stdout.strip()
    if rc != 0 or not shell:
        return None
    if shell.endswith(_NON_INTERACTIVE_SHELLS):
        return None
    return shell


async def _which_via_login_shell(
    hostname: str, shell: str, tool: str,
) -> str | None:
    """Resolve ``tool`` by asking the user's login shell.

    This picks up PATH set by shell-specific config — e.g.
    ``~/.config/fish/conf.d/uv.env.fish`` for fish, ``~/.zprofile`` for
    zsh — that ``bash -lc`` never sources. The ``-lc`` invocation form
    is accepted by bash, zsh, fish, and dash, which covers everything
    short of csh / nushell.
    """
    inner = f"command -v {shlex.quote(tool)} 2>/dev/null"
    cmd = f"{shlex.quote(shell)} -lc {shlex.quote(inner)}"
    stdout, _, rc = await ssh_run(hostname, cmd, timeout=15.0)
    if rc != 0:
        return None
    # A login shell may print a MOTD/greeting before our command's
    # output, so take the last absolute-path-looking line. Filters out
    # aliases / builtins that `command -v` would echo as bare names.
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("/"):
            return stripped
    return None


async def _locate_tool(
    hostname: str,
    tool: str,
    *,
    candidates: tuple[str, ...],
    login_shell: str | None,
) -> str | None:
    """Find ``tool`` on the remote host, returning its absolute path.

    Tries in order:
    1. The user's actual login shell (picks up fish/zsh PATH config).
    2. ``bash -lc`` PATH (the default ``ssh_run`` shell).
    3. Direct ``test -x`` over ``candidates`` (catches installs no
       shell rc has been taught about — e.g. service accounts).
    """
    if login_shell:
        found = await _which_via_login_shell(hostname, login_shell, tool)
        if found:
            return found

    stdout, _, rc = await ssh_run(
        hostname, f"command -v {shlex.quote(tool)} 2>/dev/null", timeout=10.0,
    )
    if rc == 0 and stdout.strip():
        return stdout.strip()

    quoted = " ".join(f'"{p}"' for p in candidates)
    probe = (
        f'for p in {quoted}; do '
        f'[ -x "$p" ] && {{ echo "$p"; exit 0; }}; '
        f'done; exit 1'
    )
    stdout, _, rc = await ssh_run(hostname, probe, timeout=10.0)
    if rc == 0 and stdout.strip():
        return stdout.strip()
    return None


async def detect_environment(hostname: str) -> dict[str, bool | str | None]:
    """Probe the remote host for available tools.

    Returns a dict with:
    - ``login_shell`` — the user's login shell path (or ``None``)
    - ``python3``, ``uv`` — absolute paths, resolved against the user's
      real shell PATH (not just bash's), or ``None`` if not found
    - ``pip``, ``module`` — bool
    - ``gpu_vendor`` — ``"nvidia"`` | ``"amd"`` | ``None``
    """
    from lqh.remote.gpu import detect_gpu_vendor

    result: dict[str, bool | str | None] = {}
    login_shell = await _detect_login_shell(hostname)
    result["login_shell"] = login_shell

    result["python3"] = await _locate_tool(
        hostname, "python3",
        candidates=_PYTHON3_CANDIDATE_PATHS,
        login_shell=login_shell,
    )
    result["uv"] = await _locate_tool(
        hostname, "uv",
        candidates=_UV_CANDIDATE_PATHS,
        login_shell=login_shell,
    )

    checks = {
        "pip": "command -v pip3 || command -v pip",
        "module": "type module 2>/dev/null",  # Lmod / Environment Modules
    }
    for tool, cmd in checks.items():
        _, _, rc = await ssh_run(hostname, cmd, timeout=10.0)
        result[tool] = rc == 0
        logger.debug("detect %s on %s: %s", tool, hostname, result[tool])

    result["gpu_vendor"] = await detect_gpu_vendor(hostname)
    logger.debug("detect gpu_vendor on %s: %s", hostname, result["gpu_vendor"])
    return result


async def bootstrap_remote(
    hostname: str,
    remote_root: str,
    *,
    hf_token: str | None = None,
) -> str:
    """Provision a remote environment from scratch.

    1. Detect available tools
    2. Create directory structure
    3. Create Python venv (prefer uv, fallback to python3 -m venv)
    4. Install lqh[train] dependencies
    5. Optionally configure HF_TOKEN

    Returns a human-readable setup log.
    """
    log_lines: list[str] = []

    def log(msg: str) -> None:
        logger.info(msg)
        log_lines.append(msg)

    # Step 1: Detect tools
    log(f"Detecting environment on {hostname}...")
    env = await detect_environment(hostname)
    gpu_vendor = env["gpu_vendor"]
    shell = env["login_shell"] if isinstance(env["login_shell"], str) else None
    python3_path = env["python3"] if isinstance(env["python3"], str) else None
    uv_path = env["uv"] if isinstance(env["uv"], str) else None
    log(
        f"  shell: {shell or 'unknown'}, "
        f"python3: {python3_path or 'not found'}, "
        f"uv: {uv_path or 'not found'}, "
        f"pip: {env['pip']}, gpu: {gpu_vendor or 'none'}"
    )

    if not python3_path:
        raise RuntimeError(
            f"python3 not found on {hostname}. "
            "Please install Python 3.11+ before running setup."
        )

    # Step 2: Create directory structure
    log(f"Creating directory structure at {remote_root}...")
    dirs = f"{remote_root}/datasets {remote_root}/runs {remote_root}/.lqh-env"
    stdout, stderr, rc = await ssh_run(
        hostname, f"mkdir -p {dirs}", timeout=15.0,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to create directories: {stderr}")

    # Step 3: Create venv (idempotent — re-runs of remote_setup are
    # the supported way to push local lqh code changes to the remote)
    venv_path = f"{remote_root}/.lqh-env"
    _, _, exists_rc = await ssh_run(
        hostname, f"test -x {venv_path}/bin/python", timeout=10.0,
    )
    if exists_rc == 0:
        log("  venv already exists, reusing.")
    else:
        if uv_path:
            log(f"Creating venv with uv ({uv_path})...")
            cmd = f"{uv_path} venv {venv_path}"
        else:
            log(f"Creating venv with {python3_path} -m venv...")
            cmd = f"{python3_path} -m venv {venv_path}"

        stdout, stderr, rc = await ssh_run(hostname, cmd, timeout=60.0)
        if rc != 0:
            raise RuntimeError(f"Failed to create venv: {stderr}")
        log("  venv created.")

    # Step 4: Sync local lqh source and install with train extras
    activate = f"source {venv_path}/bin/activate"

    # Find the local lqh package root (directory containing pyproject.toml)
    lqh_pkg_root = _find_lqh_package_root()
    if lqh_pkg_root:
        log(f"Syncing lqh source from {lqh_pkg_root}...")
        lqh_remote_src = f"{remote_root}/.lqh-src"
        await ssh_run(hostname, f"mkdir -p {lqh_remote_src}", timeout=10.0)
        # Sync the full project root using rsync with include/exclude filters
        # to get just lqh/, pyproject.toml, README.md
        from lqh.remote.ssh_helpers import _multiplex_args, _ensure_control_dir
        _ensure_control_dir()
        import asyncio as _asyncio
        ssh_opts = " ".join(_multiplex_args(hostname))
        rsync_cmd = [
            "rsync", "-az", "--partial",
            "-e", f"ssh {ssh_opts}",
            # Drop build artifacts so the install_hash matches what arrives.
            "--exclude", "__pycache__/",
            "--exclude", "*.pyc",
            "--exclude", "*.pyo",
            "--exclude", ".pytest_cache/",
            "--exclude", ".mypy_cache/",
            "--include", "lqh/***",
            "--include", "pyproject.toml",
            "--include", "README.md",
            "--exclude", "*",
            str(lqh_pkg_root) + "/",
            f"{hostname}:{lqh_remote_src}/",
        ]
        proc = await _asyncio.create_subprocess_exec(
            *rsync_cmd,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to sync lqh source: {stderr_bytes.decode('utf-8', errors='replace')}"
            )

        if uv_path:
            log("Installing lqh[train] from source with uv pip...")
            install_cmd = f"{activate} && {uv_path} pip install --upgrade '{lqh_remote_src}[train]'"
        else:
            log("Installing lqh[train] from source with pip...")
            install_cmd = f"{activate} && pip install --upgrade '{lqh_remote_src}[train]'"
    else:
        # Fallback: try installing from PyPI
        log("Local lqh source not found, trying PyPI...")
        if uv_path:
            install_cmd = f"{activate} && {uv_path} pip install --upgrade 'lqh[train]'"
        else:
            install_cmd = f"{activate} && pip install --upgrade 'lqh[train]'"

    stdout, stderr, rc = await ssh_run(hostname, install_cmd, timeout=600.0)
    if rc != 0:
        raise RuntimeError(f"Failed to install lqh[train]: {stderr}")
    log("  lqh[train] installed.")

    # Write the install-hash sentinel so remote_status can detect drift.
    if lqh_pkg_root:
        digest = compute_local_lqh_hash(lqh_pkg_root)
        if digest:
            sentinel = f"{remote_root}/.lqh-src/.install_hash"
            await ssh_run(
                hostname, f"printf '%s' '{digest}' > {sentinel}", timeout=10.0,
            )
            log(f"  lqh version: {short_hash(digest)}")

    # Step 5: Configure HF_TOKEN
    if hf_token:
        await configure_hf_token(hostname, remote_root, hf_token)
        log("  HF_TOKEN configured.")

    # Step 6: Verify GPU
    if gpu_vendor:
        from lqh.remote.gpu import query_gpu_info
        gpus = await query_gpu_info(hostname)
        if gpus:
            names = ", ".join(g.name for g in gpus)
            log(f"  GPUs detected: {len(gpus)} ({names})")
        else:
            log(f"  {gpu_vendor} tools available but no GPUs detected.")
    else:
        log("  No GPU tools found — GPU training may not be available.")

    log("Setup complete.")
    return "\n".join(log_lines)


def _find_lqh_package_root() -> Path | None:
    """Find the root of the local lqh package (directory with pyproject.toml).

    Walks up from the lqh package's __file__ looking for pyproject.toml.
    Returns None if not found (e.g., installed from PyPI).
    """
    try:
        import lqh
        pkg_dir = Path(lqh.__file__).parent  # lqh/
        candidate = pkg_dir.parent  # parent of lqh/
        if (candidate / "pyproject.toml").exists():
            return candidate
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Code-version sentinel
# ---------------------------------------------------------------------------
#
# After remote_setup syncs and installs lqh, we write a content hash of the
# *synced* tree to ``<remote_root>/.lqh-src/.install_hash``. remote_status
# reads it back and compares against a freshly-computed local hash, so the
# agent can detect "remote runs an older lqh than the local CLI" and tell
# the user to re-run remote_setup. Idempotent: re-running setup overwrites
# the sentinel.

_HASH_INCLUDE_TOP = ("lqh", "pyproject.toml")
_HASH_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
_HASH_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _iter_hashable_files(pkg_root: Path) -> list[tuple[str, Path]]:
    """Return (relpath, abspath) pairs for files that contribute to the hash."""
    out: list[tuple[str, Path]] = []
    for top in _HASH_INCLUDE_TOP:
        target = pkg_root / top
        if not target.exists():
            continue
        if target.is_file():
            out.append((top, target))
            continue
        for path in target.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in _HASH_EXCLUDE_SUFFIXES:
                continue
            if any(part in _HASH_EXCLUDE_DIRS for part in path.parts):
                continue
            out.append((str(path.relative_to(pkg_root)), path))
    return out


def compute_local_lqh_hash(pkg_root: Path | None = None) -> str | None:
    """Hash the local lqh source tree to identify the installed version.

    Hashes ``lqh/**`` (excluding pycache / build artifacts) plus
    ``pyproject.toml``. Returns a hex digest, or ``None`` if the local
    package root can't be found (e.g. lqh was installed from PyPI rather
    than a source checkout).
    """
    import hashlib

    root = pkg_root or _find_lqh_package_root()
    if root is None:
        return None

    h = hashlib.sha256()
    files = sorted(_iter_hashable_files(root))
    for relpath, abspath in files:
        h.update(relpath.encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(abspath.read_bytes())
        except OSError:
            continue
        h.update(b"\0")
    return h.hexdigest()


def short_hash(digest: str | None) -> str:
    """Format a long hex hash for human display."""
    if not digest:
        return "unknown"
    return digest[:12]


async def read_remote_lqh_hash(hostname: str, remote_root: str) -> str | None:
    """Read the install-hash sentinel from a remote, or return None."""
    sentinel = f"{remote_root}/.lqh-src/.install_hash"
    stdout, _, rc = await ssh_run(
        hostname, f"cat {sentinel} 2>/dev/null", timeout=10.0,
    )
    if rc != 0:
        return None
    s = stdout.strip()
    return s or None


async def configure_hf_token(
    hostname: str,
    remote_root: str,
    token: str,
) -> None:
    """Write ``HF_TOKEN`` to the remote environment's ``.env`` file."""
    env_file = f"{remote_root}/.lqh-env/.env"

    # Create or update the .env file
    # Remove existing HF_TOKEN line if present, then append
    cmd = (
        f"touch {env_file} && "
        f"grep -v '^HF_TOKEN=' {env_file} > {env_file}.tmp 2>/dev/null; "
        f"echo 'HF_TOKEN={token}' >> {env_file}.tmp && "
        f"mv {env_file}.tmp {env_file}"
    )
    stdout, stderr, rc = await ssh_run(hostname, cmd, timeout=10.0)
    if rc != 0:
        raise RuntimeError(f"Failed to configure HF_TOKEN: {stderr}")


async def check_hf_token(hostname: str, remote_root: str) -> bool:
    """Check if ``HF_TOKEN`` is already configured on the remote."""
    env_file = f"{remote_root}/.lqh-env/.env"
    stdout, _, rc = await ssh_run(
        hostname, f"grep -q '^HF_TOKEN=' {env_file} 2>/dev/null && echo yes",
        timeout=10.0,
    )
    if rc == 0 and "yes" in stdout:
        return True

    # Also check the user's shell environment
    stdout, _, rc = await ssh_run(
        hostname, "echo $HF_TOKEN", timeout=10.0,
    )
    return bool(stdout.strip())
