"""RemoteBackend ABC and shared data types for remote fine-tuning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


__all__ = [
    "RemoteBackend",
    "RemoteConfig",
    "RemoteMachine",
    "ProjectBinding",
    "JobStatus",
]


@dataclass
class RemoteMachine:
    """Global machine definition stored in ``~/.lqh/remotes.json``.

    Describes the machine itself — hostname, type, GPUs.  Shared across
    all projects.
    """

    name: str
    type: str  # "ssh_direct" | "ssh_slurm"
    hostname: str
    instructions_file: str | None = None
    gpu_ids: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type,
            "hostname": self.hostname,
        }
        if self.instructions_file is not None:
            d["instructions_file"] = self.instructions_file
        if self.gpu_ids is not None:
            d["gpu_ids"] = self.gpu_ids
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> RemoteMachine:
        known_keys = {
            "type", "hostname", "instructions_file", "gpu_ids",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            name=name,
            type=data["type"],
            hostname=data["hostname"],
            instructions_file=data.get("instructions_file"),
            gpu_ids=data.get("gpu_ids"),
            extra=extra,
        )


@dataclass
class ProjectBinding:
    """Per-project binding for a global remote machine.

    Stored in ``<project>/.lqh/remotes.json``.  Maps a machine name to
    the project-specific ``remote_root`` and optional overrides.
    """

    name: str
    remote_root: str
    hf_token_configured: bool = False
    gpu_ids: list[int] | None = None  # override machine-level gpu_ids

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"remote_root": self.remote_root}
        if self.hf_token_configured:
            d["hf_token_configured"] = True
        if self.gpu_ids is not None:
            d["gpu_ids"] = self.gpu_ids
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> ProjectBinding:
        return cls(
            name=name,
            remote_root=data["remote_root"],
            hf_token_configured=data.get("hf_token_configured", False),
            gpu_ids=data.get("gpu_ids"),
        )


@dataclass
class RemoteConfig:
    """Merged view: global machine + project binding.

    This is what callers (backends, handlers) work with.  Created by
    ``get_remote()`` which merges the two layers.
    """

    name: str
    type: str  # "ssh_direct" | "ssh_slurm"
    hostname: str
    remote_root: str
    instructions_file: str | None = None
    gpu_ids: list[int] | None = None
    hf_token_configured: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type,
            "hostname": self.hostname,
            "remote_root": self.remote_root,
        }
        if self.instructions_file is not None:
            d["instructions_file"] = self.instructions_file
        if self.gpu_ids is not None:
            d["gpu_ids"] = self.gpu_ids
        if self.hf_token_configured:
            d["hf_token_configured"] = True
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> RemoteConfig:
        known_keys = {
            "type", "hostname", "remote_root", "instructions_file",
            "gpu_ids", "hf_token_configured",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            name=name,
            type=data["type"],
            hostname=data["hostname"],
            remote_root=data["remote_root"],
            instructions_file=data.get("instructions_file"),
            gpu_ids=data.get("gpu_ids"),
            hf_token_configured=data.get("hf_token_configured", False),
            extra=extra,
        )

    @classmethod
    def merge(cls, machine: RemoteMachine, binding: ProjectBinding) -> RemoteConfig:
        """Merge a global machine definition with a project binding."""
        return cls(
            name=machine.name,
            type=machine.type,
            hostname=machine.hostname,
            remote_root=binding.remote_root,
            instructions_file=machine.instructions_file,
            # Project binding gpu_ids override machine-level
            gpu_ids=binding.gpu_ids if binding.gpu_ids is not None else machine.gpu_ids,
            hf_token_configured=binding.hf_token_configured,
            extra=machine.extra,
        )


@dataclass
class JobStatus:
    """Status of a remote training/inference job."""

    state: str  # running | completed | failed | waiting_for_scoring | waiting_timeout_expired | unknown
    pid: int | None = None
    current_step: int | None = None
    total_steps: int | None = None
    started_at: str | None = None
    last_update: str | None = None
    error: str | None = None

    @classmethod
    def from_status_json(cls, data: dict[str, Any]) -> JobStatus:
        return cls(
            state=data.get("state", "unknown"),
            pid=data.get("pid"),
            current_step=data.get("current_step"),
            total_steps=data.get("total_steps"),
            started_at=data.get("started_at"),
            last_update=data.get("last_update"),
            error=data.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"state": self.state}
        if self.pid is not None:
            d["pid"] = self.pid
        if self.current_step is not None:
            d["current_step"] = self.current_step
        if self.total_steps is not None:
            d["total_steps"] = self.total_steps
        if self.started_at is not None:
            d["started_at"] = self.started_at
        if self.last_update is not None:
            d["last_update"] = self.last_update
        if self.error is not None:
            d["error"] = self.error
        return d


VALID_REMOTE_TYPES = {"ssh_direct", "ssh_slurm"}


class RemoteBackend(ABC):
    """Abstract base class for remote fine-tuning backends.

    Each backend encapsulates both process management (submit, poll, kill)
    and file transfer (sync progress, push/pull files).  The remote
    subprocess runs the same ``lqh.train`` / ``lqh.infer`` code as local —
    only the transport changes.

    All API communication (scoring, golden generation) happens on the host
    machine.  The remote never calls ``api.lqh.ai``.
    """

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config

    @abstractmethod
    async def setup(self) -> str:
        """Provision the remote environment (install deps, sync code).

        Returns a human-readable setup log.
        """

    @abstractmethod
    async def submit_run(
        self,
        local_run_dir: str,
        config: dict[str, Any],
        *,
        module: str = "lqh.train",
    ) -> str:
        """Submit a training/inference job.  Returns a job ID (e.g. PID)."""

    @abstractmethod
    async def poll_status(self, job_id: str) -> JobStatus:
        """Check job state."""

    @abstractmethod
    async def sync_progress(
        self,
        remote_run_dir: str,
        local_run_dir: str,
    ) -> None:
        """Pull progress.jsonl and signal files from remote to local mirror."""

    @abstractmethod
    async def sync_file_to_remote(
        self,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Push a single file to the remote."""

    @abstractmethod
    async def sync_file_from_remote(
        self,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Pull a single file from the remote."""

    @abstractmethod
    async def is_job_alive(self, job_id: str) -> bool:
        """Check if the job process is still running."""

    @abstractmethod
    async def teardown(self, job_id: str) -> None:
        """Kill a running job and clean up."""
