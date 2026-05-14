"""SSHDirect remote backend — SSH + direct exec on a GPU box."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any

from lqh.remote.backend import JobStatus, RemoteBackend, RemoteConfig
from lqh.remote.bootstrap import bootstrap_remote, check_hf_token
from lqh.remote.ssh_helpers import rsync_pull, rsync_push, ssh_run

logger = logging.getLogger(__name__)

__all__ = ["SSHDirectBackend"]


class SSHDirectBackend(RemoteBackend):
    """Remote backend using SSH for direct command execution and rsync
    for file transfer.

    The remote subprocess is launched via ``nohup`` and runs independently
    of the SSH session.  Progress and signal files are synced periodically
    via rsync.
    """

    def __init__(
        self,
        config: RemoteConfig,
        project_dir: Path,
    ) -> None:
        super().__init__(config)
        self.project_dir = project_dir
        self._hostname = config.hostname
        self._remote_root = config.remote_root

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> str:
        """Bootstrap the remote environment."""
        hf_token: str | None = None
        # Pass local HF_TOKEN if remote doesn't have one
        import os
        local_token = os.environ.get("HF_TOKEN")
        if local_token:
            has_remote = await check_hf_token(self._hostname, self._remote_root)
            if not has_remote:
                hf_token = local_token

        return await bootstrap_remote(
            self._hostname,
            self._remote_root,
            hf_token=hf_token,
        )

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit_run(
        self,
        local_run_dir: str,
        config: dict[str, Any],
        *,
        module: str = "lqh.train",
    ) -> str:
        """Sync data to remote, launch the subprocess, return PID as job_id."""
        run_dir = Path(local_run_dir)
        run_name = run_dir.name

        remote_run_dir = f"{self._remote_root}/runs/{run_name}"

        # 1. Rewrite config paths for the remote filesystem
        remote_config = _rewrite_config_paths(
            config, self.project_dir, self._remote_root,
        )

        # 2. Create remote run directory
        await ssh_run(self._hostname, f"mkdir -p {remote_run_dir}", timeout=10.0)

        # 3. Sync dataset and other manifest files preserving relative paths
        manifest_paths = _resolve_local_manifest(config, self.project_dir)
        for local_path in manifest_paths:
            # Compute the relative path within the project
            try:
                rel = local_path.resolve().relative_to(self.project_dir.resolve())
            except ValueError:
                # Path is outside project dir — sync to remote_root directly
                rel = Path(local_path.name)
            remote_dest = f"{self._remote_root}/{rel.parent}" if rel.parent != Path(".") else self._remote_root
            await ssh_run(self._hostname, f"mkdir -p {remote_dest}", timeout=10.0)
            await rsync_push(
                self._hostname,
                [str(local_path)],
                remote_dest,
            )

        # 4. Write config.json to local run dir then sync it
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(remote_config, indent=2) + "\n")
        await rsync_push(
            self._hostname,
            [str(config_path)],
            remote_run_dir,
        )

        # 5. Launch the subprocess on the remote
        remote_config_path = f"{remote_run_dir}/config.json"
        activate = f"source {self._remote_root}/.lqh-env/bin/activate"

        # Source .env for HF_TOKEN
        env_file = f"{self._remote_root}/.lqh-env/.env"
        source_env = f"[ -f {env_file} ] && set -a && source {env_file} && set +a"

        # CUDA_VISIBLE_DEVICES
        cuda_env = ""
        if self.config.gpu_ids is not None:
            ids = ",".join(str(i) for i in self.config.gpu_ids)
            cuda_env = f"CUDA_VISIBLE_DEVICES={ids} "

        # Write a launcher script on the remote to avoid quoting issues.
        launcher_script = (
            f"#!/bin/bash\n"
            f"{activate}\n"
            f"{source_env}\n"
            f"cd {self._remote_root}\n"
            f"{cuda_env}python -m {module} {remote_config_path} "
            f"> {remote_run_dir}/stdout.log 2> {remote_run_dir}/stderr.log\n"
        )
        launcher_path = f"{remote_run_dir}/launcher.sh"
        await ssh_run(
            self._hostname,
            f"cat > {launcher_path} << 'LAUNCHER_EOF'\n{launcher_script}LAUNCHER_EOF",
            timeout=10.0,
        )
        await ssh_run(self._hostname, f"chmod +x {launcher_path}", timeout=5.0)

        # Launch with nohup + full stdio redirect. The `echo $!` prints
        # the PID immediately, and the backgrounded process persists
        # after the SSH session ends.
        stdout, stderr, rc = await ssh_run(
            self._hostname,
            f"nohup bash {launcher_path} < /dev/null > /dev/null 2>&1 & echo $!",
            timeout=30.0,
        )
        if rc != 0:
            raise RuntimeError(f"Failed to launch remote job: {stderr}")

        # Capture PID from echo output
        shell_pid = stdout.strip()
        if shell_pid.isdigit():
            logger.info("Launcher shell PID: %s", shell_pid)
        else:
            raise RuntimeError(
                f"Failed to get launcher PID from output: {stdout!r}"
            )

        # 6. Wait for PID file (written by lqh.train __main__.py)
        # The launcher shell PID is the bash process; the actual training
        # PID is written by the subprocess after it starts.
        pid = await self._wait_for_pid(remote_run_dir, fallback_pid=shell_pid)
        logger.info(
            "Remote job launched on %s: PID %s, run_dir=%s",
            self._hostname, pid, remote_run_dir,
        )

        # Write job metadata locally for reconnection
        meta = {
            "job_id": pid,
            "remote_name": self.config.name,
            "remote_run_dir": remote_run_dir,
            "module": module,
        }
        (run_dir / "remote_job.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

        return pid

    async def _wait_for_pid(
        self,
        remote_run_dir: str,
        *,
        fallback_pid: str | None = None,
        max_attempts: int = 30,
        interval: float = 2.0,
    ) -> str:
        """Poll for the PID file on the remote.

        The subprocess writes its own PID to ``pid`` on startup.  If the
        file doesn't appear (e.g. the process crashes during import before
        writing it), we fall back to the launcher shell PID.
        """
        pid_path = f"{remote_run_dir}/pid"
        for attempt in range(max_attempts):
            stdout, _, rc = await ssh_run(
                self._hostname, f"cat {pid_path} 2>/dev/null", timeout=5.0,
            )
            if rc == 0 and stdout.strip().isdigit():
                return stdout.strip()

            # Check if the launcher is still alive — if not, the process
            # crashed before writing the PID file.
            if fallback_pid and attempt > 3:
                alive = await self.is_job_alive(fallback_pid)
                if not alive:
                    break

            await asyncio.sleep(interval)

        # Check stderr for clues
        stderr_path = f"{remote_run_dir}/stderr.log"
        stderr_out, _, _ = await ssh_run(
            self._hostname, f"tail -30 {stderr_path} 2>/dev/null", timeout=5.0,
        )
        hint = f"\nRemote stderr:\n{stderr_out}" if stderr_out else ""

        raise RuntimeError(
            f"PID file not found at {self._hostname}:{pid_path} "
            f"after {max_attempts} attempts{hint}"
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def poll_status(self, job_id: str) -> JobStatus:
        """Read status.json from the remote, fall back to PID check."""
        # Try to read the latest progress entry
        # (progress.jsonl is authoritative for terminal states)
        run_dir = await self._find_remote_run_dir(job_id)
        if run_dir:
            stdout, _, rc = await ssh_run(
                self._hostname,
                f"tail -1 {run_dir}/progress.jsonl 2>/dev/null",
                timeout=10.0,
            )
            if rc == 0 and stdout.strip():
                try:
                    entry = json.loads(stdout.strip())
                    if entry.get("status") == "completed":
                        return JobStatus(
                            state="completed",
                            pid=int(job_id),
                            current_step=entry.get("step"),
                            last_update=entry.get("timestamp"),
                        )
                    if entry.get("status") == "failed":
                        return JobStatus(
                            state="failed",
                            pid=int(job_id),
                            error=entry.get("error"),
                            current_step=entry.get("step"),
                            last_update=entry.get("timestamp"),
                        )
                    # Running — extract metrics
                    return JobStatus(
                        state="running" if await self.is_job_alive(job_id) else "failed",
                        pid=int(job_id),
                        current_step=entry.get("step"),
                        last_update=entry.get("timestamp"),
                    )
                except json.JSONDecodeError:
                    pass

        # Fallback: just check if PID is alive
        alive = await self.is_job_alive(job_id)
        return JobStatus(
            state="running" if alive else "unknown",
            pid=int(job_id),
        )

    async def _find_remote_run_dir(self, job_id: str) -> str | None:
        """Find the run directory for a job by scanning for its PID."""
        runs_path = f"{self._remote_root}/runs"
        stdout, _, rc = await ssh_run(
            self._hostname,
            f"grep -rl '^{job_id}$' {runs_path}/*/pid 2>/dev/null | head -1",
            timeout=10.0,
        )
        if rc == 0 and stdout.strip():
            # /path/to/runs/run_001/pid -> /path/to/runs/run_001
            return str(PurePosixPath(stdout.strip()).parent)
        return None

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    async def sync_progress(
        self,
        remote_run_dir: str,
        local_run_dir: str,
    ) -> None:
        """Pull progress.jsonl and signal files from remote."""
        await rsync_pull(
            self._hostname,
            remote_run_dir,
            local_run_dir,
            include_patterns=[
                "progress.jsonl",
                "pid",
                # Run-root artifacts for one-shot inference (start_local_eval).
                "eval_request.json",
                "eval_result.json",
                "predictions.parquet",
                "checkpoints/",
                "checkpoints/*/",
                "checkpoints/*/eval_request.json",
                "checkpoints/*/predictions.parquet",
                "iterations/",
                "iterations/*/",
                "iterations/*/iter_request.json",
                "iterations/*/predictions.parquet",
            ],
        )

    async def sync_file_to_remote(
        self,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Push a single file to the remote."""
        # Ensure remote parent directory exists
        remote_parent = str(PurePosixPath(remote_path).parent)
        await ssh_run(
            self._hostname, f"mkdir -p {remote_parent}", timeout=10.0,
        )
        await rsync_push(
            self._hostname,
            [local_path],
            remote_parent,
        )

    async def sync_file_from_remote(
        self,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Pull a single file from the remote."""
        local_parent = str(Path(local_path).parent)
        Path(local_parent).mkdir(parents=True, exist_ok=True)

        # For single file pull, use the file directly
        _hostname = self._hostname
        proc = await asyncio.create_subprocess_exec(
            "rsync", "-az", "--partial",
            f"{_hostname}:{remote_path}", local_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"rsync pull file failed: {err}")

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    async def is_job_alive(self, job_id: str) -> bool:
        """Check if the remote process is still running."""
        _, _, rc = await ssh_run(
            self._hostname,
            f"kill -0 {job_id} 2>/dev/null",
            timeout=10.0,
        )
        return rc == 0

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def teardown(self, job_id: str) -> None:
        """Kill the remote process."""
        # Try graceful SIGTERM first
        await ssh_run(
            self._hostname,
            f"kill {job_id} 2>/dev/null",
            timeout=10.0,
        )

        # Wait briefly, then check if still alive
        await asyncio.sleep(2.0)
        if await self.is_job_alive(job_id):
            # Force kill
            await ssh_run(
                self._hostname,
                f"kill -9 {job_id} 2>/dev/null",
                timeout=10.0,
            )
            logger.info("Force-killed remote job %s on %s", job_id, self._hostname)
        else:
            logger.info("Remote job %s on %s terminated", job_id, self._hostname)


# ------------------------------------------------------------------
# Path utilities
# ------------------------------------------------------------------


def _rewrite_config_paths(
    config: dict[str, Any],
    project_dir: Path,
    remote_root: str,
) -> dict[str, Any]:
    """Replace local project_dir prefixes with remote_root in config values.

    Handles both absolute paths and relative paths.  Relative paths within
    the project are preserved as-is (they work on both sides since the
    directory structure is mirrored).

    Recurses one level into ``base_config`` for hyperparameter-sweep
    payloads (``{"type": "sweep", "base_config": {...}}``) so the
    nested dataset/eval_dataset/scorer/base_model paths get rewritten
    too.
    """
    project_str = str(project_dir)
    path_keys = (
        "dataset", "eval_dataset", "preference_dataset", "base_model", "scorer",
    )

    def _rewrite_level(d: dict[str, Any]) -> dict[str, Any]:
        out = dict(d)
        for key in path_keys:
            value = out.get(key)
            if not value or not isinstance(value, str):
                continue
            if value.startswith(project_str):
                out[key] = remote_root + value[len(project_str):]
        return out

    result = _rewrite_level(config)
    if isinstance(result.get("base_config"), dict):
        result["base_config"] = _rewrite_level(result["base_config"])
    return result


def _resolve_local_manifest(
    config: dict[str, Any],
    project_dir: Path,
) -> list[Path]:
    """Find local files referenced by the config that need to be synced.

    Similar to ``sync.resolve_manifest`` but also handles paths not listed
    in a ``manifest`` key. Recurses one level into ``base_config`` so
    hyperparameter-sweep payloads sync their datasets correctly.
    """
    path_keys = ("dataset", "eval_dataset", "preference_dataset", "scorer")

    def _collect(d: dict[str, Any], into: list[Path]) -> None:
        for key in path_keys:
            value = d.get(key)
            if not value or not isinstance(value, str):
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = project_dir / candidate
            if candidate.exists():
                # For parquet files in a dataset dir, sync the parent
                if candidate.is_file() and candidate.suffix == ".parquet":
                    into.append(candidate.parent)
                else:
                    into.append(candidate)

    paths: list[Path] = []
    _collect(config, paths)
    inner = config.get("base_config")
    if isinstance(inner, dict):
        _collect(inner, paths)

    # Deduplicate
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)

    return unique
