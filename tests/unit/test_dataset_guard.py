"""Tests for the dataset overwrite/claim guards (round-6 hardening).

The logical output is more than ``data.parquet``; claims fail CLOSED
when the lock infrastructure is unavailable; run-name reservation is
atomic instead of check-then-create.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lqh.dataset_guard import (
    claim_output,
    existing_output,
    overwrite_refusal,
    release_output,
)


def test_existing_output_recognizes_any_parquet(project_dir: Path) -> None:
    ds = project_dir / "datasets" / "mixed"
    ds.mkdir(parents=True)
    (ds / "train.parquet").write_bytes(b"x")

    assert existing_output(project_dir, "mixed") == "train.parquet"
    refusal = overwrite_refusal(project_dir, "mixed")
    assert refusal is not None and "train.parquet" in refusal
    # claim_output must refuse too — generating data.parquet into this
    # directory would mix unrelated artifacts and replace its manifest.
    assert claim_output(project_dir, "mixed") is not None


def test_existing_output_recognizes_manifest_only(project_dir: Path) -> None:
    ds = project_dir / "datasets" / "manifest_only"
    ds.mkdir(parents=True)
    (ds / "manifest.json").write_text("{}")

    assert existing_output(project_dir, "manifest_only") == "manifest.json"
    assert claim_output(project_dir, "manifest_only") is not None


def test_fresh_name_is_claimable(project_dir: Path) -> None:
    try:
        assert claim_output(project_dir, "fresh_v1") is None
    finally:
        release_output(project_dir, "fresh_v1")


def test_claim_fails_closed_when_lock_unavailable(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken claim lock must REFUSE, not fall back to an unlocked
    check — that would restore the exact cross-process overwrite race
    the claim exists to remove."""
    from contextlib import contextmanager

    @contextmanager
    def broken_lock(path):
        raise OSError("lock filesystem readonly")
        yield  # pragma: no cover

    monkeypatch.setattr("lqh.dataset_guard.file_lock", broken_lock)

    refusal = claim_output(project_dir, "anything")

    assert refusal is not None
    assert "cannot reserve" in refusal


def test_pending_cloud_target_needs_confirmation_even_with_overwrite(
    project_dir: Path,
) -> None:
    """overwrite=true alone must not bypass the pending-cloud-output
    protection: the data-gen handler routes it through the human
    confirmation gate (asserted here via the sentinel)."""
    import asyncio

    from lqh.tools.handlers import handle_run_data_gen_pipeline

    # A pending cloud data-gen job targeting datasets/claimed/ …
    run = project_dir / "runs" / "cloud_gen"
    run.mkdir(parents=True)
    (run / ".lqh_data_gen.json").write_text(
        json.dumps({"output_dataset": "claimed", "job_id": "j1"})
    )
    script = project_dir / "data_gen" / "pipe.py"
    script.parent.mkdir(parents=True)
    script.write_text("from lqh.pipeline import Pipeline\n")

    result = asyncio.run(handle_run_data_gen_pipeline(
        project_dir,
        script_path="data_gen/pipe.py",
        num_samples=3,
        output_dataset="claimed",
        overwrite=True,
        _script_consent=True,
    ))

    assert result.content == "OVERWRITE_CONFIRMATION_REQUIRED"
    assert result.requires_user_input
    assert "cloud data-gen job 'cloud_gen'" in (result.question or "")


def _stub_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lqh.tools.handlers._get_hf_api", lambda: object())
    monkeypatch.setattr(
        "lqh.tools.handlers._resolve_hf_pull_repo_type",
        lambda api, repo_id, repo_type: ("dataset", None),
    )


async def test_hf_pull_refuses_nested_parquet_without_overwrite(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existence check is recursive — a nested data/train.parquet
    must not slip past a top-level glob."""
    from lqh.tools.handlers import handle_hf_pull

    _stub_hf(monkeypatch)
    target = project_dir / "datasets" / "pulled"
    (target / "data").mkdir(parents=True)
    (target / "data" / "train.parquet").write_bytes(b"x")

    result = await handle_hf_pull(
        project_dir, repo_id="org/things", local_path="datasets/pulled"
    )

    assert result.content.startswith("Error:")
    assert "data/train.parquet" in result.content


async def test_hf_pull_overwrite_needs_human_consent(
    project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """overwrite=true from the model is a REQUEST — replacing existing
    dataset files goes through the confirmation round-trip."""
    from lqh.tools.handlers import handle_hf_pull

    _stub_hf(monkeypatch)
    target = project_dir / "datasets" / "pulled"
    target.mkdir(parents=True)
    (target / "data.parquet").write_bytes(b"x")

    result = await handle_hf_pull(
        project_dir, repo_id="org/things", local_path="datasets/pulled",
        overwrite=True,
    )

    assert result.content == "OVERWRITE_CONFIRMATION_REQUIRED"
    assert result.requires_user_input
    assert "data.parquet" in (result.question or "")


def test_run_name_claim_is_atomic(project_dir: Path) -> None:
    from lqh.tools.handlers import _claim_run_name

    name, err = _claim_run_name(project_dir, "sft_007", "sft")
    assert (name, err) == ("sft_007", None)
    assert (project_dir / "runs" / "sft_007").is_dir()

    # Explicit duplicate → error (never truncate an existing run).
    name2, err2 = _claim_run_name(project_dir, "sft_007", "sft")
    assert name2 is None and "already exists" in err2

    # Auto-generated names skip claimed numbers instead of erroring.
    auto, err3 = _claim_run_name(project_dir, None, "sft")
    assert err3 is None
    assert auto != "sft_007"
    assert (project_dir / "runs" / auto).is_dir()
