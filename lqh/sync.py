"""Sync backends for local and remote training runs.

The ``SyncBackend`` protocol abstracts file transfer so the same
subprocess file protocol works whether training runs locally, on a
remote GPU box via SSH, or in the cloud via S3.

For now only ``LocalSync`` (a no-op) is implemented.  Future backends:
``RsyncSync(host, ssh_key)``, ``S3Sync(bucket, prefix)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


__all__ = [
    "SyncBackend",
    "LocalSync",
    "resolve_manifest",
]


@runtime_checkable
class SyncBackend(Protocol):
    """Protocol for syncing files between the main process and a
    (potentially remote) training subprocess."""

    async def push(self, local_paths: list[Path], remote_dir: Path) -> None:
        """Ensure *local_paths* are available at *remote_dir*."""
        ...

    async def pull(
        self,
        remote_dir: Path,
        patterns: list[str],
        local_dir: Path,
    ) -> None:
        """Fetch files matching *patterns* from *remote_dir* into *local_dir*."""
        ...


class LocalSync:
    """No-op backend — everything is on the same filesystem."""

    async def push(self, local_paths: list[Path], remote_dir: Path) -> None:
        pass

    async def pull(
        self,
        remote_dir: Path,
        patterns: list[str],
        local_dir: Path,
    ) -> None:
        pass


def resolve_manifest(
    config: dict[str, Any],
    project_dir: Path,
) -> list[Path]:
    """Extract the list of local paths referenced by the config's ``manifest``.

    The ``manifest`` field is a list of config keys whose values are relative
    paths.  This function resolves them against *project_dir* and returns
    absolute ``Path`` objects (skipping any that don't exist on disk, such
    as HuggingFace Hub model IDs).
    """
    manifest_keys: list[str] = config.get("manifest", [])
    paths: list[Path] = []
    for key in manifest_keys:
        value = config.get(key)
        if value is None:
            continue
        candidate = project_dir / value
        if candidate.exists():
            paths.append(candidate.resolve())
    return paths
