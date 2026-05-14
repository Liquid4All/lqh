"""SSHSlurm remote backend — SSH to headnode + sbatch/srun scheduling.

This is a **stub** — all methods raise ``NotImplementedError``.
The docstrings document the planned behavior for future implementation.

Key differences from SSHDirect:

- ``setup()`` reads the per-remote ``INSTRUCTIONS.md`` for cluster-specific
  steps (module loads, partition info, container setup, scratch paths).
- ``submit_run()`` generates an ``sbatch`` script with the appropriate
  ``#SBATCH`` directives, submits it via ``sbatch``, and returns the
  Slurm job ID (not a PID).
- ``poll_status()`` combines ``squeue``/``sacct`` output with the
  file-based ``status.json`` protocol.
- ``is_job_alive()`` uses ``squeue -j <job_id>``.
- ``teardown()`` uses ``scancel <job_id>``.
- SSH helpers (``ssh_run``, ``rsync_push``, ``rsync_pull``) are shared
  with ``SSHDirectBackend`` via ``ssh_helpers.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lqh.remote.backend import JobStatus, RemoteBackend, RemoteConfig

__all__ = ["SSHSlurmBackend"]


class SSHSlurmBackend(RemoteBackend):
    """SSH to Slurm headnode + sbatch/srun scheduling.

    Not yet implemented.  See module docstring for planned behavior.
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

    async def setup(self) -> str:
        """Provision the remote Slurm environment.

        Planned behavior:
        1. SSH to headnode, detect tools (python3, uv, pip, module, nvidia-smi)
        2. Read INSTRUCTIONS.md for cluster-specific module loads and paths
        3. Create directory structure on scratch storage (per INSTRUCTIONS.md)
        4. Install venv + lqh[train] (or use Apptainer/Singularity container)
        5. Configure HF_TOKEN
        """
        raise NotImplementedError("SSHSlurm backend is not yet implemented")

    async def submit_run(
        self,
        local_run_dir: str,
        config: dict[str, Any],
        *,
        module: str = "lqh.train",
    ) -> str:
        """Submit a training job via sbatch.

        Planned behavior:
        1. Sync dataset and config to headnode (rsync over SSH)
        2. Generate sbatch script with:
           - Partition from INSTRUCTIONS.md or config
           - GPU resource request (--gres=gpu:N)
           - Module loads from INSTRUCTIONS.md
           - Venv activation
           - ``python -m lqh.train <config_path>``
        3. Submit via ``sbatch`` on the headnode
        4. Parse and return the Slurm job ID

        Returns the Slurm job ID (e.g. "12345678").
        """
        raise NotImplementedError("SSHSlurm backend is not yet implemented")

    async def poll_status(self, job_id: str) -> JobStatus:
        """Check Slurm job state.

        Planned behavior:
        1. Run ``squeue -j <job_id> -o %T`` to get Slurm state
           (PENDING, RUNNING, COMPLETING, etc.)
        2. If job is not in squeue, check ``sacct -j <job_id>``
           for COMPLETED, FAILED, TIMEOUT, CANCELLED
        3. Combine with status.json from the run directory
           (which has step-level progress from the subprocess)
        4. Map to JobStatus states
        """
        raise NotImplementedError("SSHSlurm backend is not yet implemented")

    async def sync_progress(
        self,
        remote_run_dir: str,
        local_run_dir: str,
    ) -> None:
        """Pull progress files from the compute node via the headnode.

        Planned behavior:
        Same as SSHDirect — rsync progress.jsonl and signal files.
        Note: on some clusters, compute nodes are not directly reachable;
        files must be on shared storage (NFS/Lustre) accessible from the
        headnode.
        """
        raise NotImplementedError("SSHSlurm backend is not yet implemented")

    async def sync_file_to_remote(
        self,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Push a file to the cluster's shared storage via the headnode."""
        raise NotImplementedError("SSHSlurm backend is not yet implemented")

    async def sync_file_from_remote(
        self,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Pull a file from the cluster's shared storage via the headnode."""
        raise NotImplementedError("SSHSlurm backend is not yet implemented")

    async def is_job_alive(self, job_id: str) -> bool:
        """Check if the Slurm job is still active.

        Planned behavior:
        Run ``squeue -j <job_id> -h``.  If output is non-empty, the job
        is alive (PENDING or RUNNING).  Empty output means the job has
        finished.
        """
        raise NotImplementedError("SSHSlurm backend is not yet implemented")

    async def teardown(self, job_id: str) -> None:
        """Cancel a Slurm job.

        Planned behavior:
        Run ``scancel <job_id>`` on the headnode.  Then check ``squeue``
        to confirm the job was cancelled.
        """
        raise NotImplementedError("SSHSlurm backend is not yet implemented")
