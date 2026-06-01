"""Default compute target — what backend ``/train`` etc. use when the
agent doesn't pass an explicit ``remote=...``.

Two layers, in precedence order (highest first):

  1. **Explicit** flag the agent passed (e.g. ``remote="lab-gpu"`` or
     ``"cloud"``). Wins always.
  2. **Per-project** override in ``<project>/.lqh/compute.json``.
  3. **Global** default in ``~/.lqh/config.json`` (``default_compute``).
  4. ``None`` — first-run picker fires.

Values are strings:

  ``"cloud"``                — LQH Cloud (api.lqh.ai, GPU provider backend-implemented)
  ``"ssh:<remote_name>"``    — a previously-bound SSH remote
  ``None``                   — no decision yet

Kept in a dedicated module to keep the picker policy in one place.
``remote_name`` strings without an ``ssh:`` prefix are accepted as a
shorthand for backwards-compat with existing agent calls that pass
``remote="lab-gpu"``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from lqh.config import LqhConfig, config_path, load_config, save_config

__all__ = [
    "ComputeTarget",
    "Scope",
    "load_project_default",
    "save_project_default",
    "load_global_default",
    "save_global_default",
    "resolve_compute",
    "is_cloud",
    "ssh_remote_name",
    "compute_file_path",
]

# Scope of a default-compute write.
Scope = Literal["project", "global"]

# A resolved compute target — one of "cloud", "ssh:<name>", or None.
ComputeTarget = str


def compute_file_path(project_dir: Path) -> Path:
    """Path to the project-level compute.json file."""
    return project_dir / ".lqh" / "compute.json"


def load_project_default(project_dir: Path) -> ComputeTarget | None:
    """Read the per-project default. None if unset or unreadable."""
    path = compute_file_path(project_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    value = data.get("default")
    return value if isinstance(value, str) and value else None


def save_project_default(project_dir: Path, value: ComputeTarget | None) -> None:
    """Write the per-project default. Passing None clears the file."""
    path = compute_file_path(project_dir)
    if value is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"default": value}, indent=2) + "\n")
    os.replace(tmp, path)


def load_global_default() -> ComputeTarget | None:
    """Read ``~/.lqh/config.json``'s ``default_compute`` field."""
    return load_config().default_compute


def save_global_default(value: ComputeTarget | None) -> None:
    """Update ``~/.lqh/config.json``'s ``default_compute`` field.

    Idempotent on identical writes; reads-modify-writes so we don't
    clobber unrelated config fields (e.g. api_key)."""
    cfg = load_config()
    cfg.default_compute = value
    save_config(cfg)


def resolve_compute(
    project_dir: Path,
    *,
    explicit: ComputeTarget | None = None,
) -> ComputeTarget:
    """Compute the effective target, applying the precedence rules.

    Returns ``"cloud"`` / ``"ssh:<name>"``. **LQH Cloud is the default
    when nothing has been configured anywhere** — the agent should
    never need to ask the user where to train just to get started.
    Users who want a different default set it via ``compute_set``.

    Bare remote names without an ``ssh:`` prefix (eg. ``explicit="lab"``)
    are passed through unchanged so existing agent calls keep working;
    ``is_cloud`` and ``ssh_remote_name`` normalize the lookup.
    """
    if explicit:
        return explicit
    proj = load_project_default(project_dir)
    if proj:
        return proj
    glob = load_global_default()
    if glob:
        return glob
    # No layer set anything → LQH Cloud is the product default.
    return "cloud"


def is_cloud(target: ComputeTarget | None) -> bool:
    """True iff ``target`` requests LQH Cloud."""
    return target == "cloud"


def ssh_remote_name(target: ComputeTarget | None) -> str | None:
    """Extract the SSH remote name from a target, or None.

    Accepts both the canonical ``"ssh:<name>"`` form and the legacy
    bare ``"<name>"`` form (anything that isn't ``"cloud"``). Returns
    ``None`` if the input is None / "cloud" / empty.
    """
    if not target or target == "cloud":
        return None
    if target.startswith("ssh:"):
        return target[len("ssh:"):]
    return target
