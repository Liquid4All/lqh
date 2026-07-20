"""Permission-store domain isolation.

The training permission domain is intentionally separate from the pipeline/
script-execution domain (``project_allow_all``) and the HF-push domain, so
that approving one action never silently grants another. See
``lqh/tools/permissions.py``.
"""

from __future__ import annotations

from pathlib import Path

from lqh.tools.permissions import (
    PERMISSIONS_FILE,
    PermissionContext,
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


# ---------------------------------------------------------------------------
# PermissionContext (CLI_PLAN §3.4)
# ---------------------------------------------------------------------------


def test_context_defaults_to_store(tmp_path: Path) -> None:
    ctx = PermissionContext()
    assert ctx.allows_script(tmp_path, "data_gen/x.py") is False
    grant_permission(tmp_path, "data_gen/x.py")
    assert ctx.allows_script(tmp_path, "data_gen/x.py") is True
    assert ctx.allows_script(tmp_path, "data_gen/other.py") is False


def test_context_invocation_grant_is_domain_scoped(tmp_path: Path) -> None:
    ctx = PermissionContext.granting("script")
    assert ctx.allows_script(tmp_path, "data_gen/x.py") is True
    assert ctx.allows_training(tmp_path, "sft_1") is False
    assert ctx.allows_hf_push(tmp_path, "org/repo") is False
    assert ctx.allows_cloud_data_gen(tmp_path) is False


def test_full_consent_allows_all_domains(tmp_path: Path) -> None:
    ctx = PermissionContext(full_consent=True)
    assert ctx.allows_script(tmp_path, "data_gen/x.py") is True
    assert ctx.allows_training(tmp_path, "sft_1") is True
    assert ctx.allows_hf_push(tmp_path, "org/repo") is True
    assert ctx.allows_cloud_data_gen(tmp_path) is True


def test_invocation_grants_do_not_persist(tmp_path: Path) -> None:
    ctx = PermissionContext.granting("script", "cloud_data_gen")
    assert ctx.allows_cloud_data_gen(tmp_path) is True
    assert not (tmp_path / PERMISSIONS_FILE).exists()
    # A fresh store-only context still denies.
    assert PermissionContext().allows_cloud_data_gen(tmp_path) is False


def test_granting_rejects_unknown_domain() -> None:
    import pytest

    with pytest.raises(ValueError):
        PermissionContext.granting("scripts")  # typo'd domain


def test_with_grants_extends_immutably(tmp_path: Path) -> None:
    base = PermissionContext.granting("script")
    extended = base.with_grants("training")
    assert extended.allows_training(tmp_path, "sft_1") is True
    # Original unchanged.
    assert base.allows_training(tmp_path, "sft_1") is False


def test_concurrent_grants_do_not_lose_updates(tmp_path: Path) -> None:
    import threading

    n = 16
    threads = [
        threading.Thread(target=grant_permission, args=(tmp_path, f"data_gen/s{i}.py"))
        for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(n):
        assert check_permission(tmp_path, f"data_gen/s{i}.py") is True
