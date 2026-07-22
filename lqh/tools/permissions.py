from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

from lqh.fsio import atomic_write_json, file_lock

PERMISSIONS_FILE = ".lqh/permissions.json"
PERMISSIONS_LOCK = ".lqh/permissions.lock"


@dataclass
class PermissionStore:
    # Pipeline / script execution (run_data_gen_pipeline).
    project_allow_all: bool = False
    allowed_files: list[str] = field(default_factory=list)
    # HF push.
    hf_push_allow_all: bool = False
    hf_allowed_repos: list[str] = field(default_factory=list)
    # Training launches (start_training). Kept SEPARATE from
    # project_allow_all so that approving a training run never silently
    # grants arbitrary pipeline/script execution, and vice versa.
    training_allow_all: bool = False
    allowed_training: list[str] = field(default_factory=list)
    # Cloud data-gen submission (run_data_gen_pipeline execution="cloud").
    # Separate domain again: local script-exec approval must not imply
    # spending cloud compute, and vice versa.
    cloud_data_gen_allow_all: bool = False
    # Cloud HF-eval submission (eval_hf_model). Its own domain: GPU
    # wall-clock spend, gated by the same consent shape as data_gen.
    cloud_eval_hf_allow_all: bool = False


def _permissions_lock(project_dir: Path):
    return file_lock(project_dir / PERMISSIONS_LOCK)


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
            training_allow_all=data.get("training_allow_all", False),
            allowed_training=data.get("allowed_training", []),
            cloud_data_gen_allow_all=data.get("cloud_data_gen_allow_all", False),
            cloud_eval_hf_allow_all=data.get("cloud_eval_hf_allow_all", False),
        )
    except (json.JSONDecodeError, OSError):
        return PermissionStore()


def save_permissions(project_dir: Path, perms: PermissionStore) -> None:
    atomic_write_json(project_dir / PERMISSIONS_FILE, asdict(perms))


def check_permission(project_dir: Path, script_path: str) -> bool:
    perms = load_permissions(project_dir)
    return perms.project_allow_all or script_path in perms.allowed_files


def grant_permission(
    project_dir: Path,
    script_path: str | None = None,
    project_wide: bool = False,
) -> None:
    with _permissions_lock(project_dir):
        perms = load_permissions(project_dir)
        if project_wide:
            perms.project_allow_all = True
        elif script_path is not None and script_path not in perms.allowed_files:
            perms.allowed_files.append(script_path)
        save_permissions(project_dir, perms)


def check_training_permission(project_dir: Path, run_name: str) -> bool:
    """Whether a training run may launch.

    Deliberately does NOT consult ``project_allow_all`` (pipeline/script
    execution): the two domains are independent so granting one never
    implies the other.
    """
    perms = load_permissions(project_dir)
    return perms.training_allow_all or f"training:{run_name}" in perms.allowed_training


def grant_training_permission(
    project_dir: Path,
    key: str | None = None,
    project_wide: bool = False,
) -> None:
    """Grant a training launch.

    ``project_wide=True`` approves all future training in the project (used
    by autonomous auto mode so it never re-prompts). Otherwise ``key`` — a
    ``"training:<run_name>"`` string — grants exactly that one run.
    """
    with _permissions_lock(project_dir):
        perms = load_permissions(project_dir)
        if project_wide:
            perms.training_allow_all = True
        elif key is not None and key not in perms.allowed_training:
            perms.allowed_training.append(key)
        save_permissions(project_dir, perms)


def check_cloud_data_gen_permission(project_dir: Path) -> bool:
    """Whether cloud data-gen submits may proceed without a prompt."""
    return load_permissions(project_dir).cloud_data_gen_allow_all


def grant_cloud_data_gen_permission(project_dir: Path) -> None:
    """Project-wide grant — used by "don't ask again" and by auto mode."""
    with _permissions_lock(project_dir):
        perms = load_permissions(project_dir)
        perms.cloud_data_gen_allow_all = True
        save_permissions(project_dir, perms)


def check_cloud_eval_hf_permission(project_dir: Path) -> bool:
    """Whether cloud HF-eval submits may proceed without a prompt."""
    return load_permissions(project_dir).cloud_eval_hf_allow_all


def grant_cloud_eval_hf_permission(project_dir: Path) -> None:
    """Project-wide grant — used by "don't ask again" and by auto mode."""
    with _permissions_lock(project_dir):
        perms = load_permissions(project_dir)
        perms.cloud_eval_hf_allow_all = True
        save_permissions(project_dir, perms)


def check_hf_permission(project_dir: Path, repo_id: str) -> bool:
    perms = load_permissions(project_dir)
    return perms.hf_push_allow_all or repo_id in perms.hf_allowed_repos


def grant_hf_permission(
    project_dir: Path,
    repo_id: str | None = None,
    project_wide: bool = False,
) -> None:
    with _permissions_lock(project_dir):
        perms = load_permissions(project_dir)
        if project_wide:
            perms.hf_push_allow_all = True
        elif repo_id is not None and repo_id not in perms.hf_allowed_repos:
            perms.hf_allowed_repos.append(repo_id)
        save_permissions(project_dir, perms)


# ---------------------------------------------------------------------------
# PermissionContext — invocation-scoped consent (CLI_PLAN §3.4)
# ---------------------------------------------------------------------------

# Consent domain names. Keep aligned with the `permission_domain` tool
# metadata in lqh/tools/definitions.py.
PERMISSION_DOMAINS = frozenset(
    {"script", "cloud_data_gen", "cloud_eval_hf", "training", "hf_push"}
)


@dataclass(frozen=True)
class PermissionContext:
    """Invocation-scoped consent, consulted by all permission-gated handlers.

    Precedence: ``full_consent`` -> invocation ``grants`` -> durable store
    -> deny (the handler returns its PERMISSION_REQUIRED sentinel).

    Threaded to handlers as the ``_permissions`` extra-kwarg — the
    underscore channel is already stripped from model-controlled arguments
    (``execute_tool``), so a model call can never smuggle consent in.
    Invocation grants never persist anything; durable grants remain the
    agent loop's job at its grant sites.
    """

    full_consent: bool = False  # CLI `lqh tool call` surface (invocation-is-consent)
    grants: frozenset[str] = frozenset()  # domains granted for THIS invocation

    @classmethod
    def granting(cls, *domains: str) -> "PermissionContext":
        unknown = set(domains) - PERMISSION_DOMAINS
        if unknown:
            raise ValueError(f"unknown permission domain(s): {sorted(unknown)}")
        return cls(grants=frozenset(domains))

    def with_grants(self, *domains: str) -> "PermissionContext":
        unknown = set(domains) - PERMISSION_DOMAINS
        if unknown:
            raise ValueError(f"unknown permission domain(s): {sorted(unknown)}")
        return replace(self, grants=self.grants | frozenset(domains))

    def allows_script(self, project_dir: Path, script_path: str) -> bool:
        return (
            self.full_consent
            or "script" in self.grants
            or check_permission(project_dir, script_path)
        )

    def allows_cloud_data_gen(self, project_dir: Path) -> bool:
        return (
            self.full_consent
            or "cloud_data_gen" in self.grants
            or check_cloud_data_gen_permission(project_dir)
        )

    def allows_cloud_eval_hf(self, project_dir: Path) -> bool:
        return (
            self.full_consent
            or "cloud_eval_hf" in self.grants
            or check_cloud_eval_hf_permission(project_dir)
        )

    def allows_training(self, project_dir: Path, run_name: str) -> bool:
        return (
            self.full_consent
            or "training" in self.grants
            or check_training_permission(project_dir, run_name)
        )

    def allows_hf_push(self, project_dir: Path, repo_id: str) -> bool:
        return (
            self.full_consent
            or "hf_push" in self.grants
            or check_hf_permission(project_dir, repo_id)
        )
