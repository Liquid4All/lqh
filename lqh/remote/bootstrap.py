"""Remote environment detection and provisioning.

Connects via SSH to detect available tools (python3, uv, pip, nvidia-smi)
and installs a Python environment with ``lqh[train]`` dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lqh.remote.ssh_helpers import ssh_run

logger = logging.getLogger(__name__)

__all__ = [
    "detect_environment",
    "bootstrap_remote",
    "configure_hf_token",
]


# Where uv tends to live. Astral's installer lands in ~/.local/bin;
# `cargo install` in ~/.cargo/bin; `snap install` in /snap/bin; the
# rest cover manual installs and Homebrew on macOS / Linux. We probe
# these directly so a user whose login shell is fish/zsh (and whose
# uv PATH config lives in ~/.config/fish/conf.d/ rather than ~/.bashrc)
# still gets detected from inside our `bash -lc` wrapper.
_UV_CANDIDATE_PATHS: tuple[str, ...] = (
    "$HOME/.local/bin/uv",
    "$HOME/.cargo/bin/uv",
    "/snap/bin/uv",
    "/usr/local/bin/uv",
    "/opt/homebrew/bin/uv",
    "/home/linuxbrew/.linuxbrew/bin/uv",
    "/usr/bin/uv",
)


async def _locate_uv(hostname: str) -> str | None:
    """Return the absolute path to ``uv`` on the remote host, or ``None``.

    Tries bash's PATH first, then probes ``_UV_CANDIDATE_PATHS`` directly
    so installs that live outside bash's idea of PATH (snap, Astral's
    installer for a fish user, …) still get found.
    """
    stdout, _, rc = await ssh_run(
        hostname, "command -v uv 2>/dev/null", timeout=10.0,
    )
    if rc == 0 and stdout.strip():
        return stdout.strip()

    candidates = " ".join(f'"{p}"' for p in _UV_CANDIDATE_PATHS)
    probe = (
        f'for p in {candidates}; do '
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
    - ``uv`` — absolute path (or ``None`` if not installed anywhere we
      know to look)
    - ``python3``, ``pip``, ``module`` — bool
    - ``gpu_vendor`` — ``"nvidia"`` | ``"amd"`` | ``None``
    """
    from lqh.remote.gpu import detect_gpu_vendor

    result: dict[str, bool | str | None] = {}
    result["uv"] = await _locate_uv(hostname)

    checks = {
        "python3": "command -v python3",
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
    uv_path: str | None = env["uv"] if isinstance(env["uv"], str) else None
    log(f"  python3: {env['python3']}, uv: {uv_path or 'not found'}, "
        f"pip: {env['pip']}, gpu: {gpu_vendor or 'none'}")

    if not env["python3"]:
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
            log("Creating venv with python3 -m venv...")
            cmd = f"python3 -m venv {venv_path}"

        stdout, stderr, rc = await ssh_run(hostname, cmd, timeout=60.0)
        if rc != 0:
            raise RuntimeError(f"Failed to create venv: {stderr}")
        log("  venv created.")

    # Step 4: Sync local lqh source and install with train extras
    activate = f"source {venv_path}/bin/activate"
    pip_cmd = f"{uv_path} pip" if uv_path else "pip"

    src_root = _local_lqh_root()
    if src_root is None:
        raise RuntimeError(
            "Could not locate the local lqh package. remote_setup must "
            "run in an environment where `import lqh` succeeds."
        )

    lqh_remote_src = f"{remote_root}/.lqh-src"
    await ssh_run(hostname, f"mkdir -p {lqh_remote_src}", timeout=10.0)
    log(f"Syncing lqh source from {src_root}...")
    await _rsync_lqh_source(hostname, src_root, lqh_remote_src)

    # pip needs a pyproject.toml to build the synced tree. A source
    # checkout ships its own; a wheel-installed lqh has none, so
    # synthesize one from the installed distribution's metadata. Either
    # way the remote installs the exact lqh that's running locally —
    # never a same-named package off PyPI.
    if not (src_root / "pyproject.toml").exists():
        log("  no local pyproject.toml — synthesizing from installed metadata.")
        synthetic = _synthesize_pyproject()
        if synthetic is None:
            raise RuntimeError(
                "lqh has neither a pyproject.toml nor installed metadata; "
                "cannot resolve dependencies for the remote install."
            )
        await ssh_run(
            hostname,
            f"cat > {lqh_remote_src}/pyproject.toml << 'LQH_PYPROJECT_EOF'\n"
            f"{synthetic}\nLQH_PYPROJECT_EOF",
            timeout=10.0,
        )

    log("Installing lqh[train] from synced source...")
    install_cmd = (
        f"{activate} && {pip_cmd} install --upgrade '{lqh_remote_src}[train]'"
    )
    stdout, stderr, rc = await ssh_run(hostname, install_cmd, timeout=600.0)
    if rc != 0:
        raise RuntimeError(f"Failed to install lqh[train]: {stderr}")
    log("  lqh[train] installed.")

    # Write the install-hash sentinel so remote_status can detect drift.
    digest = compute_local_lqh_hash(src_root)
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


def _local_lqh_root() -> Path | None:
    """Return the directory that contains the importable ``lqh`` package.

    This is the parent of ``lqh/__init__.py`` — always resolvable while
    lqh is running, whether it was installed as a wheel (then this is
    site-packages), installed editable, or run straight from a source
    checkout. Returns ``None`` only if the location can't be determined
    (e.g. a frozen / zipapp build).
    """
    try:
        import lqh
        if not lqh.__file__:
            return None
        return Path(lqh.__file__).parent.parent
    except Exception:
        return None


async def _rsync_lqh_source(
    hostname: str, src_root: Path, remote_dest: str,
) -> None:
    """rsync the local lqh package tree to ``remote_dest`` on the host.

    Transfers ``lqh/`` plus ``pyproject.toml`` / ``README.md`` when those
    sit alongside it (a source checkout). For a wheel-installed lqh,
    ``src_root`` is site-packages and only ``lqh/`` matches the filter —
    which is all the remote needs.
    """
    import asyncio

    from lqh.remote.ssh_helpers import _ensure_control_dir, _multiplex_args

    _ensure_control_dir()
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
        str(src_root) + "/",
        f"{hostname}:{remote_dest}/",
    ]
    proc = await asyncio.create_subprocess_exec(
        *rsync_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr_bytes = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to sync lqh source: "
            f"{stderr_bytes.decode('utf-8', errors='replace')}"
        )


def _synthesize_pyproject() -> str | None:
    """Build a minimal ``pyproject.toml`` for the installed lqh distribution.

    Used by ``bootstrap_remote`` when lqh was pip-installed without a
    source checkout: the synced ``lqh/`` tree then has no pyproject.toml
    for pip to build from. Version, ``requires-python``, and dependencies
    (including the ``train`` extra) are read from the installed
    distribution metadata.

    Returns ``None`` if lqh has no installed metadata — which only
    happens when it's run purely from source, in which case a real
    pyproject.toml exists and this isn't needed.
    """
    import importlib.metadata as md

    try:
        dist = md.distribution("lqh")
    except md.PackageNotFoundError:
        return None

    requires_python = dist.metadata.get("Requires-Python", ">=3.11")
    base: list[str] = []
    train: list[str] = []
    for raw in dist.requires or []:
        # Entries look like 'rich>=13.0' or 'torch; extra == "train"'.
        spec, _, marker = raw.partition(";")
        spec, marker = spec.strip(), marker.strip()
        if not marker:
            base.append(spec)
        elif 'extra == "train"' in marker or "extra == 'train'" in marker:
            # Other extras (dev, …) are intentionally dropped — the
            # remote needs only runtime + train dependencies.
            train.append(spec)

    def _toml_list(items: list[str]) -> str:
        return "".join(f'    "{item}",\n' for item in items)

    return (
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
        "\n"
        "[project]\n"
        'name = "lqh"\n'
        f'version = "{dist.version}"\n'
        f'requires-python = "{requires_python}"\n'
        "dependencies = [\n"
        f"{_toml_list(base)}"
        "]\n"
        "\n"
        "[project.optional-dependencies]\n"
        "train = [\n"
        f"{_toml_list(train)}"
        "]\n"
        "\n"
        "[tool.hatch.build.targets.wheel]\n"
        'packages = ["lqh"]\n'
    )


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
    ``pyproject.toml`` when one sits alongside the package. Returns a hex
    digest, or ``None`` only if the package location can't be determined.
    """
    import hashlib

    root = pkg_root or _local_lqh_root()
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
