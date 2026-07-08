"""Permission-store domain isolation.

The training permission domain is intentionally separate from the pipeline/
script-execution domain (``project_allow_all``) and the HF-push domain, so
that approving one action never silently grants another. See
``lqh/tools/permissions.py``.
"""

from __future__ import annotations

from pathlib import Path

from lqh.tools.permissions import (
    check_hf_permission,
    check_permission,
    check_training_permission,
    grant_hf_permission,
    grant_permission,
    grant_training_permission,
)


def test_training_grant_does_not_grant_pipeline(tmp_path: Path) -> None:
    grant_training_permission(tmp_path, project_wide=True)
    # Training is now allowed...
    assert check_training_permission(tmp_path, "sft_1") is True
    # ...but pipeline/script execution is NOT.
    assert check_permission(tmp_path, "data_gen/foo.py") is False


def test_pipeline_grant_does_not_grant_training(tmp_path: Path) -> None:
    grant_permission(tmp_path, None, project_wide=True)
    assert check_permission(tmp_path, "data_gen/foo.py") is True
    # Pipeline approval must not leak into the training domain.
    assert check_training_permission(tmp_path, "sft_1") is False


def test_per_run_training_grant_is_scoped(tmp_path: Path) -> None:
    grant_training_permission(tmp_path, key="training:sft_1")
    assert check_training_permission(tmp_path, "sft_1") is True
    # A different run is still gated.
    assert check_training_permission(tmp_path, "sft_2") is False
    # And it did not flip the project-wide training flag.
    assert check_training_permission(tmp_path, "dpo_9") is False


def test_training_grant_does_not_grant_hf(tmp_path: Path) -> None:
    grant_training_permission(tmp_path, project_wide=True)
    assert check_hf_permission(tmp_path, "org/repo") is False


def test_grants_persist_across_loads(tmp_path: Path) -> None:
    grant_training_permission(tmp_path, key="training:sft_1")
    grant_hf_permission(tmp_path, repo_id="org/repo")
    # Independent helpers read the same on-disk store.
    assert check_training_permission(tmp_path, "sft_1") is True
    assert check_hf_permission(tmp_path, "org/repo") is True
    assert check_permission(tmp_path, "anything.py") is False
