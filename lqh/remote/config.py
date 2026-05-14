"""Two-layer remote configuration.

**Global** (``~/.lqh/remotes.json``): machine definitions shared across all
projects — hostname, type, GPU IDs.

**Project** (``<project>/.lqh/remotes.json``): per-project bindings that map a
global machine name to a ``remote_root`` on that machine, plus optional
overrides (e.g. restrict to specific GPU IDs for this project).

``get_remote()`` merges the two layers into a single ``RemoteConfig`` that
callers (backends, handlers) work with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lqh.remote.backend import (
    VALID_REMOTE_TYPES,
    ProjectBinding,
    RemoteConfig,
    RemoteMachine,
)

__all__ = [
    "load_machines",
    "save_machines",
    "add_machine",
    "remove_machine",
    "get_machine",
    "load_bindings",
    "save_bindings",
    "add_binding",
    "remove_binding",
    "get_binding",
    # High-level API (merges both layers) — backwards-compatible names
    "load_remotes",
    "save_remotes",
    "add_remote",
    "remove_remote",
    "get_remote",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Override this for testing (e.g. monkeypatch to a tmp dir).
GLOBAL_CONFIG_DIR: Path = Path.home() / ".lqh"


def _global_remotes_path() -> Path:
    return GLOBAL_CONFIG_DIR / "remotes.json"


def _project_remotes_path(project_dir: Path) -> Path:
    return project_dir / ".lqh" / "remotes.json"


# ---------------------------------------------------------------------------
# Global machine definitions (~/.lqh/remotes.json)
# ---------------------------------------------------------------------------


def load_machines() -> dict[str, RemoteMachine]:
    """Load all global machine definitions."""
    path = _global_remotes_path()
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return {name: RemoteMachine.from_dict(name, cfg) for name, cfg in data.items()}


def save_machines(machines: dict[str, RemoteMachine]) -> None:
    """Write all global machine definitions."""
    path = _global_remotes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: m.to_dict() for name, m in machines.items()}
    path.write_text(json.dumps(data, indent=2) + "\n")


def add_machine(machine: RemoteMachine) -> None:
    """Add or update a global machine definition."""
    _validate_machine(machine)
    machines = load_machines()
    machines[machine.name] = machine
    save_machines(machines)


def remove_machine(name: str) -> None:
    """Remove a global machine definition.  Raises ``KeyError`` if not found."""
    machines = load_machines()
    if name not in machines:
        raise KeyError(f"Machine '{name}' not found in global config")
    del machines[name]
    save_machines(machines)


def get_machine(name: str) -> RemoteMachine | None:
    """Get a single global machine definition, or ``None``."""
    return load_machines().get(name)


def _validate_machine(machine: RemoteMachine) -> None:
    if not machine.name:
        raise ValueError("Remote name cannot be empty")
    if machine.type not in VALID_REMOTE_TYPES:
        raise ValueError(
            f"Invalid remote type '{machine.type}', "
            f"must be one of: {', '.join(sorted(VALID_REMOTE_TYPES))}"
        )
    if not machine.hostname:
        raise ValueError("Remote hostname cannot be empty")


# ---------------------------------------------------------------------------
# Project bindings (<project>/.lqh/remotes.json)
# ---------------------------------------------------------------------------


def load_bindings(project_dir: Path) -> dict[str, ProjectBinding]:
    """Load all project bindings."""
    path = _project_remotes_path(project_dir)
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return {name: ProjectBinding.from_dict(name, cfg) for name, cfg in data.items()}


def save_bindings(project_dir: Path, bindings: dict[str, ProjectBinding]) -> None:
    """Write all project bindings."""
    path = _project_remotes_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: b.to_dict() for name, b in bindings.items()}
    path.write_text(json.dumps(data, indent=2) + "\n")


def add_binding(project_dir: Path, binding: ProjectBinding) -> None:
    """Add or update a project binding."""
    if not binding.remote_root:
        raise ValueError("Remote root path cannot be empty")
    bindings = load_bindings(project_dir)
    bindings[binding.name] = binding
    save_bindings(project_dir, bindings)


def remove_binding(project_dir: Path, name: str) -> None:
    """Remove a project binding.  Raises ``KeyError`` if not found."""
    bindings = load_bindings(project_dir)
    if name not in bindings:
        raise KeyError(f"Project binding '{name}' not found")
    del bindings[name]
    save_bindings(project_dir, bindings)


def get_binding(project_dir: Path, name: str) -> ProjectBinding | None:
    """Get a single project binding, or ``None``."""
    return load_bindings(project_dir).get(name)


# ---------------------------------------------------------------------------
# High-level API — merges global + project layers
# ---------------------------------------------------------------------------


def load_remotes(project_dir: Path) -> dict[str, RemoteConfig]:
    """Load all remotes visible to this project.

    Returns merged ``RemoteConfig`` for every machine that has a project
    binding.  Machines without a binding are not included (use
    ``load_machines()`` to see all available machines).
    """
    machines = load_machines()
    bindings = load_bindings(project_dir)
    result: dict[str, RemoteConfig] = {}
    for name, binding in bindings.items():
        machine = machines.get(name)
        if machine is not None:
            result[name] = RemoteConfig.merge(machine, binding)
        else:
            # Orphan binding — machine was removed globally.  Skip silently
            # so the project doesn't break, but don't include it.
            pass
    return result


def save_remotes(project_dir: Path, remotes: dict[str, RemoteConfig]) -> None:
    """Write remotes — updates both global machines and project bindings."""
    machines: dict[str, RemoteMachine] = {}
    bindings: dict[str, ProjectBinding] = {}
    for name, cfg in remotes.items():
        machines[name] = RemoteMachine(
            name=name,
            type=cfg.type,
            hostname=cfg.hostname,
            instructions_file=cfg.instructions_file,
            gpu_ids=cfg.gpu_ids,
            extra=cfg.extra,
        )
        bindings[name] = ProjectBinding(
            name=name,
            remote_root=cfg.remote_root,
            hf_token_configured=cfg.hf_token_configured,
        )
    # Merge with existing globals (don't clobber machines used by other projects)
    existing_machines = load_machines()
    existing_machines.update(machines)
    save_machines(existing_machines)
    save_bindings(project_dir, bindings)


def add_remote(project_dir: Path, config: RemoteConfig) -> None:
    """Add or update a remote (writes both global machine + project binding)."""
    machine = RemoteMachine(
        name=config.name,
        type=config.type,
        hostname=config.hostname,
        instructions_file=config.instructions_file,
        gpu_ids=config.gpu_ids,
        extra=config.extra,
    )
    binding = ProjectBinding(
        name=config.name,
        remote_root=config.remote_root,
        hf_token_configured=config.hf_token_configured,
    )
    add_machine(machine)
    add_binding(project_dir, binding)


def remove_remote(project_dir: Path, name: str) -> None:
    """Remove a project binding.  Does NOT remove the global machine.

    Raises ``KeyError`` if the binding is not found.
    """
    remove_binding(project_dir, name)


def get_remote(project_dir: Path, name: str) -> RemoteConfig | None:
    """Get a merged remote config, or ``None`` if not found.

    Requires both a global machine definition and a project binding.
    """
    machine = get_machine(name)
    if machine is None:
        return None
    binding = get_binding(project_dir, name)
    if binding is None:
        return None
    return RemoteConfig.merge(machine, binding)
