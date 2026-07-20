"""Characterization tests for the `summary` tool (handle_summary).

Phase 0 of the persistency work (see PERSISTENCY_PLAN.md). Documents the
CURRENT shallow-inventory behavior: no prompts/ section, run names with
no status, silent truncation caps. ``CURRENT:`` cases flip when the
summary overhaul (Phase 2) lands; names stay stable.
"""

from __future__ import annotations

import json
from pathlib import Path

from lqh.project_identity import cloud_project_key
from lqh.tools.handlers import handle_summary


async def test_header_and_spec_line(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# spec\n")

    result = await handle_summary(project_dir)

    assert f"## Project: {project_dir.name}" in result.content
    assert "- **SPEC.md**:" in result.content


async def test_missing_spec_reports_new_project(project_dir: Path) -> None:
    result = await handle_summary(project_dir)
    assert "not found (new project)" in result.content


async def test_datasets_show_row_counts_and_scores(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    ds = project_dir / "datasets" / "train_v1"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(8))

    result = await handle_summary(project_dir)

    assert "- **datasets/**: 1 dataset(s)" in result.content
    assert "data.parquet: 8 rows" in result.content


async def test_malformed_eval_summary_degrades_gracefully(
    project_dir: Path,
) -> None:
    """{"scores": null} and similar malformed artifacts must degrade to a
    bare name — never crash the summary (and with it, startup)."""
    for name, payload in [
        ("null_scores", '{"scores": null}'),
        ("scores_list", '{"scores": [1, 2]}'),
        ("not_json", "{{{"),
    ]:
        er = project_dir / "evals" / "runs" / name
        er.mkdir(parents=True)
        (er / "summary.json").write_text(payload)

    result = await handle_summary(project_dir)

    for name in ("null_scores", "scores_list", "not_json"):
        assert f"  - {name}" in result.content


async def test_eval_runs_show_mean_scores(project_dir: Path) -> None:
    er = project_dir / "evals" / "runs" / "baseline"
    er.mkdir(parents=True)
    (er / "summary.json").write_text(
        json.dumps({"scores": {"mean": 6.5}, "num_samples": 20})
    )

    result = await handle_summary(project_dir)

    assert "baseline: mean 6.5/10 (20 samples)" in result.content


async def test_prompts_directory_is_reported(project_dir: Path) -> None:
    """Flipped from Phase 0: prompts/ now has its own section."""
    prompts = project_dir / "prompts"
    prompts.mkdir()
    (prompts / "triage_v1.md").write_text("You are a triage assistant.\n")
    (prompts / "triage.schema.json").write_text("{}")

    result = await handle_summary(project_dir)

    assert "- **prompts/**: 2 file(s)" in result.content
    assert "triage_v1.md" in result.content
    assert "triage.schema.json" in result.content


async def test_runs_listed_with_semantic_status(project_dir: Path) -> None:
    """Flipped from Phase 0: runs show state, base model, and failure
    reasons instead of bare names."""
    run = project_dir / "runs" / "sft_v1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"base_model": "lfm-1b"}))
    (run / "progress.jsonl").write_text(
        json.dumps({"status": "completed"}) + "\n"
    )
    failed = project_dir / "runs" / "sft_v2"
    failed.mkdir(parents=True)
    (failed / "config.json").write_text(json.dumps({"base_model": "lfm-1b"}))
    (failed / "progress.jsonl").write_text(
        json.dumps({"status": "failed", "error": "OOM on step 12"}) + "\n"
    )

    result = await handle_summary(project_dir)

    assert "sft_v1: completed (lfm-1b)" in result.content
    assert "sft_v2: failed" in result.content
    assert "OOM on step 12" in result.content


def _cloud_remote_job(job_id: str) -> dict:
    """The exact shape lqh/remote/cloud.py writes."""
    return {
        "job_id": job_id,
        "remote_name": "lqh-cloud",
        "remote_run_dir": f"cloud:{job_id}",
        "module": "lqh.train",
        "kind": "sft",
        "backend": "cloud",
    }


def _ssh_remote_job(pid: int) -> dict:
    """The exact shape lqh/remote/ssh_direct.py writes (no backend key)."""
    return {
        "job_id": pid,
        "remote_name": "lambda1",
        "remote_run_dir": "/tmp/lqh/runs/x",
        "module": "lqh.train",
    }


async def test_cloud_run_status_from_state_files(project_dir: Path) -> None:
    run = project_dir / "runs" / "cloud_sft"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"base_model": "lfm-1b"}))
    (run / "remote_job.json").write_text(json.dumps(_cloud_remote_job("j1")))
    (run / "cloud_state.json").write_text(json.dumps({"job_id": "j1", "status": "running"}))

    done = project_dir / "runs" / "cloud_done"
    done.mkdir(parents=True)
    (done / "config.json").write_text("{}")
    (done / "remote_job.json").write_text(json.dumps(_cloud_remote_job("j2")))
    (done / "cloud_state.json").write_text(json.dumps({"job_id": "j2", "status": "completed"}))

    orphan = project_dir / "runs" / "lost_submit"
    orphan.mkdir(parents=True)
    (orphan / "config.json").write_text("{}")
    (orphan / "submit_intent.json").write_text(json.dumps({"idempotency_key": "k"}))

    result = await handle_summary(project_dir)

    assert "cloud_sft: cloud, running (as of last sync)" in result.content
    assert "cloud_done: cloud, completed" in result.content
    assert "lost_submit: submitted, fate unknown" in result.content


async def test_ssh_run_status_from_synced_progress(project_dir: Path) -> None:
    """SSH runs (remote_job.json without a backend marker) get semantic
    status from the rsynced progress.jsonl, including failure reasons."""
    run = project_dir / "runs" / "ssh_sft"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"base_model": "lfm-1b"}))
    (run / "remote_job.json").write_text(json.dumps(_ssh_remote_job(4242)))
    (run / "progress.jsonl").write_text(
        json.dumps({"step": 10}) + "\n"
        + json.dumps({"status": "failed", "error": "CUDA OOM"}) + "\n"
    )

    live = project_dir / "runs" / "ssh_live"
    live.mkdir(parents=True)
    (live / "config.json").write_text("{}")
    (live / "remote_job.json").write_text(json.dumps(_ssh_remote_job(4243)))

    result = await handle_summary(project_dir)

    assert "ssh_sft: ssh, failed — CUDA OOM" in result.content
    assert "ssh_live: ssh, running (as of last sync)" in result.content


async def test_run_shows_checkpoint_and_dict_datasets(project_dir: Path) -> None:
    run = project_dir / "runs" / "multi_src"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "base_model": "lfm-1b",
        "datasets": [
            {"path": "datasets/base_v1/data.parquet", "repeat": 2},
            "datasets/failures_v1/data.parquet",
        ],
    }))
    (run / "progress.jsonl").write_text(json.dumps({"status": "completed"}) + "\n")
    (run / "checkpoints" / "final").mkdir(parents=True)

    result = await handle_summary(project_dir)

    assert "multi_src: completed" in result.content
    # Canonical paths end in data.parquet — display the dataset dir name.
    assert "data: base_v1×2, failures_v1" in result.content
    assert "ckpt ✓" in result.content


async def test_sweep_config_unwraps_base_config(project_dir: Path) -> None:
    run = project_dir / "runs" / "sweep_1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "type": "sweep",
        "base_config": {
            "base_model": "lfm-1b",
            "dataset_path": "datasets/train_v1/data.parquet",
        },
    }))
    (run / "progress.jsonl").write_text(json.dumps({"step": 3}) + "\n")

    result = await handle_summary(project_dir)

    assert "lfm-1b" in result.content
    assert "data: train_v1" in result.content


async def test_cloud_failure_reason_from_progress(project_dir: Path) -> None:
    """cloud_state.json records the terminal status but not the error —
    the replayed progress row carries it."""
    run = project_dir / "runs" / "cloud_fail"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")
    (run / "remote_job.json").write_text(json.dumps(_cloud_remote_job("j3")))
    (run / "cloud_state.json").write_text(json.dumps({"job_id": "j3", "status": "failed"}))
    (run / "progress.jsonl").write_text(
        json.dumps({"status": "failed", "error": "OOM in trainer"}) + "\n"
    )

    result = await handle_summary(project_dir)

    assert "cloud_fail: cloud, failed — OOM in trainer" in result.content


async def test_filtered_dataset_keeps_manifest_provenance(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    """summary.json (filter stats) must append to, not replace, the
    manifest provenance."""
    ds = project_dir / "datasets" / "combo"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(4))
    (ds / "manifest.json").write_text(json.dumps({
        "purpose": "training",
        "parent_dataset": "datasets/base_v1",
    }))
    (ds / "summary.json").write_text(json.dumps({"kept": 4, "total": 9, "threshold": 6}))

    result = await handle_summary(project_dir)

    line = [l for l in result.content.splitlines() if "combo" in l][0]
    assert "training" in line
    assert "supplements base_v1" in line
    assert "filtered 4/9 @ ≥6" in line


async def test_derived_dataset_renders_filtered_from(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    ds = project_dir / "datasets" / "clean_v1"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(2))
    (ds / "manifest.json").write_text(json.dumps({
        "derived_from": "datasets/raw_v1/data.parquet",
    }))

    result = await handle_summary(project_dir)

    assert "filtered from raw_v1" in result.content


async def test_stale_empty_cloud_sections_are_reported(
    project_dir: Path,
) -> None:
    (project_dir / ".lqh" / "snapshot.json").write_text(json.dumps({
        "schema_version": 1,
        "fetched_at": "2026-07-16T00:00:00+00:00",
        "snapshot": {"jobs": [{"job_id": "j1", "status": "completed"}]},
        "artifacts": None,
        "deployments": None,
        "stale_sections": ["artifacts", "deployments"],
    }))

    result = await handle_summary(project_dir)

    assert "artifact list unavailable (last refresh failed" in result.content
    assert "deployment state unavailable (last refresh failed" in result.content


async def test_cloud_section_renders_wrapper_deployments_without_core(
    project_dir: Path,
) -> None:
    """The exact shape a core 404 with live deployments produces: empty
    snapshot, wrapper-level deployments. They must not be omitted."""
    (project_dir / ".lqh" / "snapshot.json").write_text(json.dumps({
        "schema_version": 1,
        "fetched_at": "2026-07-17T00:00:00+00:00",
        "project_key": cloud_project_key(project_dir),
        "snapshot": {},
        "artifacts": [{"artifact_id": "a1", "kind": "checkpoint"}],
        "deployments": [{"id": "d1", "name": "triage-prod", "status": "running"}],
        "stale_sections": [],
    }))

    result = await handle_summary(project_dir)

    assert "triage-prod: running" in result.content
    assert "a1 [checkpoint]" in result.content


async def test_run_manifest_spec_note_in_summary(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# current spec\n")
    run = project_dir / "runs" / "sft_old"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")
    (run / "progress.jsonl").write_text(json.dumps({"status": "completed"}) + "\n")
    (run / "manifest.json").write_text(json.dumps({"spec_sha256": "d" * 64}))

    result = await handle_summary(project_dir)

    assert "sft_old: completed" in result.content
    assert "built against an OLDER spec" in result.content


async def test_cloud_stale_sections_are_labeled(project_dir: Path) -> None:
    (project_dir / ".lqh" / "snapshot.json").write_text(json.dumps({
        "schema_version": 1,
        "fetched_at": "2026-07-16T00:00:00+00:00",
        "snapshot": {"jobs": [{"job_id": "j1", "status": "completed"}]},
        "artifacts": [{"artifact_id": "a1", "kind": "checkpoint"}],
        "deployments": [{"id": "d1", "name": "prod", "status": "running"}],
        "stale_sections": ["artifacts", "deployments"],
    }))

    result = await handle_summary(project_dir)

    assert result.content.count("STALE — last refresh failed") == 2


async def test_run_truncation_is_honest(project_dir: Path) -> None:
    """Flipped from Phase 0: truncated sections say what was omitted."""
    for i in range(12):
        (project_dir / "runs" / f"run_{i:02d}").mkdir(parents=True)

    result = await handle_summary(project_dir)

    assert "- **runs/**: 12 run(s)" in result.content
    listed = [l for l in result.content.splitlines() if l.startswith("  - run_")]
    assert len(listed) == 10
    assert "…2 older runs not shown" in result.content


async def test_cloud_section_renders_from_cached_snapshot(
    project_dir: Path,
) -> None:
    """The summary's cloud section reads only .lqh/snapshot.json — works
    offline, never fetches."""
    (project_dir / ".lqh" / "snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fetched_at": "2026-07-15T18:02:00+00:00",
                "project_key": cloud_project_key(project_dir),
                "snapshot": {
                    "jobs": [
                        {"job_id": "j-123", "status": "completed", "kind": "sft"}
                    ],
                    "lifetime_spend_micros": 2_500_000,
                    "best_checkpoint": {"artifact_id": "ckpt-9"},
                    "deployments": [{"name": "triage-prod", "status": "running"}],
                },
            }
        )
    )

    result = await handle_summary(project_dir)

    assert "**Cloud** (cached snapshot from 2026-07-15T18:02:00+00:00" in result.content
    assert "may lag live state" in result.content
    assert "j-123 sft: completed" in result.content
    assert "lifetime cloud spend: $2.50" in result.content
    assert "selected best checkpoint: ckpt-9" in result.content
    assert "triage-prod: running" in result.content


async def test_no_cloud_section_without_cache(project_dir: Path) -> None:
    result = await handle_summary(project_dir)
    assert "**Cloud**" not in result.content


async def test_notes_md_line(project_dir: Path) -> None:
    (project_dir / "NOTES.md").write_text("objective: triage model\n")

    result = await handle_summary(project_dir)

    assert "- **NOTES.md**:" in result.content


async def test_all_capped_sections_are_honest_about_truncation(
    project_dir: Path,
) -> None:
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True)
    for i in range(12):
        (scorers / f"scorer_{i:02d}.md").write_text("criteria")
    eval_runs = project_dir / "evals" / "runs"
    for i in range(13):
        (eval_runs / f"eval_{i:02d}").mkdir(parents=True)
    (project_dir / ".lqh" / "snapshot.json").write_text(
        json.dumps({
            "schema_version": 1,
            "fetched_at": "2026-07-16T00:00:00+00:00",
            "snapshot": {"jobs": [{"job_id": f"j{i}", "status": "completed"} for i in range(8)]},
            "deployments": [
                {"id": f"d{i}", "name": f"dep-{i}", "status": "running"}
                for i in range(7)
            ],
        })
    )

    result = await handle_summary(project_dir)

    assert "…2 more not shown (use list_files evals/scorers/)" in result.content
    assert "…3 older eval runs not shown" in result.content
    assert "…2 more not shown (use list_deployments)" in result.content
    assert "…3 more not shown" in result.content  # cloud jobs beyond 5


async def test_dataset_manifest_provenance_and_spec_match(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    from lqh.project_meta import compute_spec_sha256

    (project_dir / "SPEC.md").write_text("# spec v2\n")
    current_hash = compute_spec_sha256(project_dir)

    stale = project_dir / "datasets" / "old_batch"
    write_chatml_parquet(stale / "data.parquet", sample_conversations(3))
    (stale / "manifest.json").write_text(json.dumps({
        "purpose": "training",
        "spec_sha256": "e" * 64,
        "parent_dataset": "datasets/base_v1",
    }))
    fresh = project_dir / "datasets" / "new_batch"
    write_chatml_parquet(fresh / "data.parquet", sample_conversations(3))
    (fresh / "manifest.json").write_text(json.dumps({
        "purpose": "failures",
        "spec_sha256": current_hash,
    }))

    result = await handle_summary(project_dir)

    assert "training" in result.content
    assert "supplements base_v1" in result.content
    assert "built against an OLDER spec" in result.content
    assert "failures" in result.content
    assert "spec ✓" in result.content


async def test_dataset_provenance_from_sidecars(
    project_dir: Path, sample_conversations, write_chatml_parquet
) -> None:
    ds = project_dir / "datasets" / "train_filtered"
    write_chatml_parquet(ds / "data.parquet", sample_conversations(5))
    (ds / "summary.json").write_text(
        json.dumps({"kept": 5, "total": 8, "threshold": 7})
    )
    src = project_dir / "datasets" / "cloud_batch"
    write_chatml_parquet(src / "data.parquet", sample_conversations(3))
    (src / ".lqh_source.json").write_text(
        json.dumps({"run_name": "datagen_v2", "job_id": "j-9"})
    )

    result = await handle_summary(project_dir)

    assert "filtered 5/8 @ ≥7" in result.content
    assert "cloud output of datagen_v2" in result.content


async def test_conversations_listed_from_legacy_files(project_dir: Path) -> None:
    convos = project_dir / ".lqh" / "conversations"
    convos.mkdir(parents=True)
    (convos / "abc.jsonl").write_text(
        json.dumps(
            {
                "__metadata__": True,
                "id": "abc",
                "created_at": "2026-07-01T00:00:00+00:00",
                "preview": "train a model",
            }
        )
        + "\n"
    )

    result = await handle_summary(project_dir)

    assert "- **Conversations**: 1 session(s)" in result.content
    assert "train a model" in result.content
