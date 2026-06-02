"""Default compute target — what training/eval routes to.

The compute target is a **fixed, per-project decision**, not a per-call
argument: the agent no longer passes ``remote=...``. Layers, in
precedence order (highest first):

  1. **Explicit** value (internal/legacy callers only — the agent-facing
     tool schemas no longer expose it). Wins always.
  2. **Per-project** default in ``<project>/.lqh/compute.json``.
  3. **Global** default in ``~/.lqh/config.json`` (``default_compute``).
  4. LQH Cloud — the silent product default.

Values are strings:

  ``"cloud"``                — LQH Cloud (api.lqh.ai, GPU provider backend-implemented)
  ``"ssh:<remote_name>"``    — a previously-bound SSH remote
  ``"local"``                — in-process training on this machine (needs a local CUDA GPU)

The one-time project picker does NOT live here — it is driven from the
handler layer (``handlers._compute_pick_options`` + the agent loop) and
only fires when a project has ≥1 bring-your-own-compute remote bound but
neither a project nor a global default set. When nothing is configured
and no BYOC remote exists, ``resolve_compute`` silently returns
``"cloud"`` with no prompt. ``remote_name`` strings without an ``ssh:``
prefix are accepted as a shorthand for backwards-compat.
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

    Returns ``"cloud"`` / ``"ssh:<name>"`` — never ``None``. **LQH Cloud
    is the default when nothing has been configured anywhere.** This
    function does not prompt; the one-time project picker is gated and
    fired by the handler layer (``handlers._compute_pick_options``)
    before a launch tool ever calls ``resolve_compute``. Users change a
    persisted default via the picker or ``compute_set``.

    Bare remote names without an ``ssh:`` prefix (eg. ``explicit="lab"``)
    are passed through unchanged; ``is_cloud`` and ``ssh_remote_name``
    normalize the lookup.
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
    bare ``"<name>"`` form (anything that isn't ``"cloud"``/``"local"``).
    Returns ``None`` if the input is None / "cloud" / "local" / empty.
    """
    if not target or target in ("cloud", "local"):
        return None
    if target.startswith("ssh:"):
        return target[len("ssh:"):]
    return target
