"""Tests for artifact provenance manifests and no-overwrite guards (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path

from lqh.manifest import (
    inherit_purpose,
    write_dataset_manifest,
    write_run_manifest,
)
from lqh.project_meta import compute_spec_sha256
from lqh.tools.handlers import handle_run_data_filter, handle_run_data_gen_pipeline
from lqh.tools.permissions import PermissionContext


# ---------------------------------------------------------------------------
# Dataset manifests
# ---------------------------------------------------------------------------


def test_dataset_manifest_records_provenance(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    (project_dir / "SPEC.md").write_text("# spec\n")
    (project_dir / "data_gen").mkdir()
    (project_dir / "data_gen" / "pipe_v1.py").write_text("# pipeline\n")
    ds = project_dir / "datasets" / "train_v1"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(5))

    (project_dir / "seed_data").mkdir()
    (project_dir / "seed_data" / "seeds.jsonl").write_text('{"x": 1}\n')

    path = write_dataset_manifest(
        project_dir,
        ds,
        purpose="training",
        rows=5,
        pipeline_path="data_gen/pipe_v1.py",
        source_paths=["seed_data/seeds.jsonl"],
    )

    manifest = json.loads(path.read_text())
    assert manifest["purpose"] == "training"
    assert manifest["rows"] == 5
    assert manifest["name"] == "train_v1"
    assert manifest["spec_sha256"] == compute_spec_sha256(project_dir)
    # No captured hash was passed → the finalization-time fallback is marked.
    assert manifest["spec_sha256_source"] == "finalization"
    assert len(manifest["content_sha256"]) == 64
    assert manifest["pipeline_path"] == "data_gen/pipe_v1.py"
    assert len(manifest["pipeline_hash"]) == 12
    # Source inputs are hashed so provenance survives later edits.
    assert manifest["sources"] == [
        {"path": "seed_data/seeds.jsonl", "hash": manifest["sources"][0]["hash"]}
    ]
    assert len(manifest["sources"][0]["hash"]) == 12
    assert manifest["version"] == 1
    assert manifest["dataset_id"]


def test_dataset_manifest_prefers_captured_hashes(project_dir: Path) -> None:
    """Hashes captured when the work STARTED must win over the current
    files — a spec edited during a long run must not be attributed."""
    (project_dir / "SPEC.md").write_text("# spec v2, edited mid-run\n")
    ds = project_dir / "datasets" / "train_v1"
    ds.mkdir(parents=True)

    manifest = json.loads(write_dataset_manifest(
        project_dir,
        ds,
        purpose="training",
        spec_sha256="a" * 64,
        pipeline_path="data_gen/pipe.py",
        pipeline_hash="b" * 12,
    ).read_text())

    assert manifest["spec_sha256"] == "a" * 64
    assert "spec_sha256_source" not in manifest  # captured, not fallback
    assert manifest["pipeline_hash"] == "b" * 12
    assert "pipeline_hash_source" not in manifest


def test_dataset_manifest_relativizes_absolute_paths(project_dir: Path) -> None:
    ds = project_dir / "datasets" / "x"
    ds.mkdir(parents=True)

    manifest = json.loads(write_dataset_manifest(
        project_dir,
        ds,
        derived_from=str(project_dir / "datasets" / "raw" / "data.parquet"),
    ).read_text())

    assert manifest["derived_from"] == "datasets/raw/data.parquet"


def test_dataset_manifest_rewrite_keeps_id_and_bumps_version(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    ds = project_dir / "datasets" / "train_v1"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(3))

    first = json.loads(
        write_dataset_manifest(project_dir, ds, purpose="training").read_text()
    )
    second = json.loads(
        write_dataset_manifest(project_dir, ds, purpose="training").read_text()
    )

    assert second["dataset_id"] == first["dataset_id"]
    assert second["version"] == first["version"] + 1


def test_dataset_manifest_normalizes_unknown_purpose(project_dir: Path) -> None:
    ds = project_dir / "datasets" / "x"
    ds.mkdir(parents=True)

    manifest = json.loads(
        write_dataset_manifest(project_dir, ds, purpose="whatever").read_text()
    )
    assert manifest["purpose"] == "unspecified"


def test_inherit_purpose(project_dir: Path) -> None:
    ds = project_dir / "datasets" / "failures_v1"
    ds.mkdir(parents=True)
    write_dataset_manifest(project_dir, ds, purpose="failures")

    assert inherit_purpose(ds) == "failures"
    # Unknown provenance stays unknown — never invented.
    assert inherit_purpose(project_dir / "datasets" / "nope") == "unspecified"


# ---------------------------------------------------------------------------
# Run manifests
# ---------------------------------------------------------------------------


def test_run_manifest_prefers_submitted_spec_hash(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# spec v2 (edited mid-run)\n")
    run = project_dir / "runs" / "sft_v1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "type": "sft",
        "base_model": "lfm-1b",
        "dataset": [{"path": "datasets/train_v1/data.parquet", "repeat": 2}],
        "spec_sha256": "a" * 64,  # as submitted, before the edit
    }))
    (run / "checkpoints" / "final").mkdir(parents=True)

    path = write_run_manifest(project_dir, run, state="completed")

    manifest = json.loads(path.read_text())
    assert manifest["spec_sha256"] == "a" * 64
    assert manifest["kind"] == "sft"
    assert manifest["base_model"] == "lfm-1b"
    assert manifest["dataset"] == [
        {"path": "datasets/train_v1/data.parquet", "repeat": 2}
    ]
    assert manifest["checkpoints"] == ["final"]
    assert manifest["state"] == "completed"


def test_run_manifest_includes_results_and_model_identity(
    project_dir: Path,
) -> None:
    run = project_dir / "runs" / "eval_hf_1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "type": "eval_hf",
        "hf_repo": "org/model-sft",
        "revision": "abc123",
        "training": {"learning_rate": 2e-5},
        "spec_sha256": "b" * 64,
    }))
    (run / "progress.jsonl").write_text(
        json.dumps({"step": 40, "loss": 0.31}) + "\n"
        + json.dumps({"status": "completed"}) + "\n"
    )
    (run / "eval_result.json").write_text(json.dumps({
        "scores": {"mean": 7.2}, "num_samples": 50,
    }))

    manifest = json.loads(
        write_run_manifest(project_dir, run, state="completed").read_text()
    )

    assert manifest["hf_repo"] == "org/model-sft"
    assert manifest["revision"] == "abc123"
    assert manifest["training"] == {"learning_rate": 2e-5}
    assert manifest["final_metrics"] == {"step": 40, "loss": 0.31}
    assert manifest["result_summary"] == {"scores": {"mean": 7.2}, "num_samples": 50}
    assert manifest["spec_sha256_source"] == "submission"
    assert len(manifest["config_sha256"]) == 12


async def test_get_eval_failures_export(project_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lqh.tools.handlers import handle_get_eval_failures

    run = project_dir / "evals" / "runs" / "baseline"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "hf_repo": "org/model-sft", "revision": "abc123", "type": "model_eval",
    }))
    msgs = json.dumps([{"role": "user", "content": "long input " * 200}])
    pq.write_table(pa.table({
        "sample_index": [0, 1],
        "messages": [msgs, msgs],
        "score": [2.0, 9.0],
        "reasoning": ["missed the point", "good"],
    }), run / "results.parquet")

    result = await handle_get_eval_failures(
        project_dir,
        eval_run="evals/runs/baseline",
        threshold=6.0,
        min_failures=1,
        max_failures=5,
        export_path="feedback/baseline_failures.jsonl",
    )

    assert "Exported" in result.content
    exported = [
        json.loads(line)
        for line in (project_dir / "feedback" / "baseline_failures.jsonl")
        .read_text().splitlines()
    ]
    failure_rows = [e for e in exported if not e["scoring_error"]]
    assert len(failure_rows) >= 1
    row = failure_rows[0]
    assert row["eval_run"] == "evals/runs/baseline"
    assert row["model"]["hf_repo"] == "org/model-sft"
    # Full messages — not the 500-char display truncation.
    assert len(row["messages"][0]["content"]) > 500


def test_run_manifest_without_config_falls_back(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# spec\n")
    run = project_dir / "runs" / "broken"
    run.mkdir(parents=True)

    manifest = json.loads(
        write_run_manifest(project_dir, run, state="failed", error="x" * 900)
        .read_text()
    )
    assert manifest["spec_sha256"] == compute_spec_sha256(project_dir)
    assert len(manifest["error"]) == 500


# ---------------------------------------------------------------------------
# No-overwrite guards (R5: expensive outputs are immutable by default)
# ---------------------------------------------------------------------------


async def test_data_gen_refuses_overwrite_of_finalized_dataset(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    ds = project_dir / "datasets" / "train_v1"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(3))

    result = await handle_run_data_gen_pipeline(
        project_dir,
        script_path="data_gen/pipe.py",
        num_samples=10,
        output_dataset="train_v1",
    )

    assert "refusing to overwrite" in result.content
    assert "train_v1_v2" in result.content


async def test_stale_partial_does_not_bypass_immutability(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    """Finalized data + a leftover partial marker is NOT a resume — the
    engine deletes the partial on finalization, so this combination means
    stale/unrelated state and must still require overwrite=true."""
    ds = project_dir / "datasets" / "train_v1"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(3))
    (ds / "data.partial.jsonl").write_text("{}\n")

    result = await handle_run_data_gen_pipeline(
        project_dir,
        script_path="data_gen/missing.py",
        num_samples=10,
        output_dataset="train_v1",
    )

    assert "refusing to overwrite" in result.content


async def test_data_gen_allows_resume_of_unfinalized_partial(
    project_dir: Path,
) -> None:
    """A partial WITHOUT finalized data is a genuine interrupted run —
    resuming needs no overwrite (here it proceeds to the next validation:
    the missing script)."""
    ds = project_dir / "datasets" / "train_v1"
    ds.mkdir(parents=True)
    (ds / "data.partial.jsonl").write_text("{}\n")

    result = await handle_run_data_gen_pipeline(
        project_dir,
        script_path="data_gen/missing.py",
        num_samples=10,
        output_dataset="train_v1",
    )

    assert "does not exist" in result.content  # passed the guard


async def test_overwrite_flag_requires_human_confirmation(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    """overwrite=true from the model is a request, not consent: the
    handler returns a confirmation prompt; only the consent flag (set by
    the agent loop after the user says yes) proceeds."""
    ds = project_dir / "datasets" / "train_v1"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(3))
    script = project_dir / "data_gen" / "pipe.py"
    script.parent.mkdir()
    script.write_text("# pipeline\n")

    result = await handle_run_data_gen_pipeline(
        project_dir,
        script_path="data_gen/pipe.py",
        num_samples=10,
        output_dataset="train_v1",
        overwrite=True,
        _permissions=PermissionContext.granting("script"),
    )

    assert result.content == "OVERWRITE_CONFIRMATION_REQUIRED"
    assert result.requires_user_input
    assert "destroyed" in (result.question or "")
    # Data untouched by the prompt round-trip.
    assert (ds / "data.parquet").exists()

    confirmed = await handle_run_data_gen_pipeline(
        project_dir,
        script_path="data_gen/missing.py",
        num_samples=10,
        output_dataset="train_v1",
        overwrite=True,
        _permissions=PermissionContext.granting("script"),
        _overwrite_consent=True,
    )
    assert "does not exist" in confirmed.content  # passed both gates


def test_claim_blocks_other_live_process(project_dir: Path) -> None:
    """Concurrency: a second task in the SAME process is refused while the
    claim is held; another LIVE process is refused; a dead claimant never
    blocks; release makes the name reusable."""
    import json as _json

    from lqh.dataset_guard import claim_output, release_output

    assert claim_output(project_dir, "train_v1") is None
    # A concurrent task in this process (pid alone can't tell them apart).
    refusal = claim_output(project_dir, "train_v1")
    assert refusal is not None and "another task in this session" in refusal
    release_output(project_dir, "train_v1")
    claim_path = project_dir / "datasets" / "train_v1" / ".lqh_claim.json"
    assert not claim_path.exists()

    # Another LIVE process (pid 1 is always alive) → refused.
    assert claim_output(project_dir, "train_v1") is None
    claim = _json.loads(claim_path.read_text())
    release_output(project_dir, "train_v1")
    claim["pid"] = 1
    claim_path.write_text(_json.dumps(claim))
    refusal = claim_output(project_dir, "train_v1")
    assert refusal is not None and "another lqh process" in refusal

    # A DEAD claimant never blocks the name forever.
    claim["pid"] = 2**22 + 5
    claim_path.write_text(_json.dumps(claim))
    assert claim_output(project_dir, "train_v1") is None
    release_output(project_dir, "train_v1")


def test_pending_cloud_job_blocks_same_output(project_dir: Path) -> None:
    """A durable cloud data-gen marker targeting the name blocks a second
    submission/generation into it — 'newest submission wins' must not be
    reachable by accident."""
    import json as _json

    from lqh.dataset_guard import overwrite_refusal

    run = project_dir / "runs" / "datagen_v1"
    run.mkdir(parents=True)
    (run / ".lqh_data_gen.json").write_text(_json.dumps({
        "output_dataset": "train_v1", "job_id": "j1",
    }))

    refusal = overwrite_refusal(project_dir, "train_v1")
    assert refusal is not None and "pending cloud data-gen job" in refusal
    assert overwrite_refusal(project_dir, "other") is None
    assert overwrite_refusal(project_dir, "train_v1", overwrite=True) is None


async def test_data_filter_rejects_path_separators(project_dir: Path) -> None:
    result = await handle_run_data_filter(
        project_dir,
        input_path="datasets/raw/data.parquet",
        scorer_path="evals/scorers/q.md",
        output_dataset="../escape",
    )
    assert "plain name" in result.content


async def test_data_filter_refuses_overwrite(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    src = project_dir / "datasets" / "raw"
    write_chatml_parquet(src / "data.parquet", sample_conversations(3))
    scorer = project_dir / "evals" / "scorers"
    scorer.mkdir(parents=True)
    (scorer / "quality.md").write_text("judge it\n")
    out = project_dir / "datasets" / "raw_filtered"
    write_chatml_parquet(out / "data.parquet", sample_conversations(1))

    result = await handle_run_data_filter(
        project_dir,
        input_path="datasets/raw/data.parquet",
        scorer_path="evals/scorers/quality.md",
        output_dataset="raw_filtered",
    )

    assert "refusing to overwrite" in result.content