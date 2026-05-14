from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

PERMISSIONS_FILE = ".lqh/permissions.json"


@dataclass
class PermissionStore:
    project_allow_all: bool = False
    allowed_files: list[str] = field(default_factory=list)
    hf_push_allow_all: bool = False
    hf_allowed_repos: list[str] = field(default_factory=list)


def load_permissions(project_dir: Path) -> PermissionStore:
    path = project_dir / PERMISSIONS_FILE
    if not path.exists():
        return PermissionStore()
    try:
        data = json.loads(path.read_text())
        return PermissionStore(
            project_allow_all=data.get("project_allow_all", False),
            allowed_files=data.get("allowed_files", []),
            hf_push_allow_all=data.get("hf_push_allow_all", False),
            hf_allowed_repos=data.get("hf_allowed_repos", []),
        )
    except (json.JSONDecodeError, OSError):
        return PermissionStore()


def save_permissions(project_dir: Path, perms: PermissionStore) -> None:
    path = project_dir / PERMISSIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(perms), indent=2) + "\n")


def check_permission(project_dir: Path, script_path: str) -> bool:
    perms = load_permissions(project_dir)
    return perms.project_allow_all or script_path in perms.allowed_files


def grant_permission(
    project_dir: Path,
    script_path: str | None = None,
    project_wide: bool = False,
) -> None:
    perms = load_permissions(project_dir)
    if project_wide:
        perms.project_allow_all = True
    elif script_path is not None and script_path not in perms.allowed_files:
        perms.allowed_files.append(script_path)
    save_permissions(project_dir, perms)


def check_hf_permission(project_dir: Path, repo_id: str) -> bool:
    perms = load_permissions(project_dir)
    return perms.hf_push_allow_all or repo_id in perms.hf_allowed_repos


def grant_hf_permission(
    project_dir: Path,
    repo_id: str | None = None,
    project_wide: bool = False,
) -> None:
    perms = load_permissions(project_dir)
    if project_wide:
        perms.hf_push_allow_all = True
    elif repo_id is not None and repo_id not in perms.hf_allowed_repos:
        perms.hf_allowed_repos.append(repo_id)
    save_permissions(project_dir, perms)
