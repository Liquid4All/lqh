"""Tests for the study's design matrix, grid, launch payload and report.

These are the parts that decide *what gets measured*. A mistake here is
expensive in a different way from an analysis bug: it burns GPU hours before
anyone notices the matrix was unbalanced or the sweep was launched without
per-config eval.
"""

from __future__ import annotations

import json

import pytest

from tests.benchmarks.hp_defaults import runner
from tests.benchmarks.hp_defaults.analyze import Observation
from tests.benchmarks.hp_defaults.cells import (
    MODELS,
    TASKS,
    TRAIN_SIZES,
    anchor_cells,
    build_cells,
    resolve_cells,
)
from tests.benchmarks.hp_defaults.grid import (
    GridPoint,
    chunk_points,
    parse_points,
    replicate_grid,
    study_grid,
)
from tests.benchmarks.hp_defaults.report import build_results, render_markdown


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def test_full_matrix_is_the_product_of_the_three_axes():
    cells = build_cells()
    assert len(cells) == len(TASKS) * len(TRAIN_SIZES) * len(MODELS)
    assert len({c.id for c in cells}) == len(cells), "cell ids must be unique"


def test_models_cover_both_kinds_at_both_sizes():
    """The base-vs-instruct question is only answerable if the comparison is
    available at more than one model size."""
    by_size: dict[float, set[str]] = {}
    for _, kind, params in MODELS.values():
        by_size.setdefault(params, set()).add(kind)
    assert len(by_size) >= 2
    for kinds in by_size.values():
        assert kinds == {"base", "instruct"}


def test_anchor_subset_covers_every_level_of_every_dimension():
    """The stage-A screen must not be blind to a whole level — a grid narrowed
    down on instruct-only cells would be a bad shortlist for base models."""
    cells = build_cells()
    anchors = anchor_cells(cells, per_dimension=2)

    assert 0 < len(anchors) < len(cells), "anchors should be a real subset"
    for dim in ("task", "train_size", "model_kind", "param_count"):
        full = {c.dimensions()[dim] for c in cells}
        covered = {c.dimensions()[dim] for c in anchors}
        assert covered == full, f"anchor set misses levels of {dim}: {full - covered}"


def test_anchor_subset_is_deterministic():
    """A study you cannot re-run identically is not a study."""
    cells = build_cells()
    assert [c.id for c in anchor_cells(cells)] == [c.id for c in anchor_cells(cells)]


def test_resolve_cells_narrows_on_every_axis():
    cells = resolve_cells(
        tasks="translation", models="350M-Base", sizes="500,2000",
    )
    assert len(cells) == 2
    assert {c.train_size for c in cells} == {500, 2000}
    assert all(c.model_kind == "base" for c in cells)


def test_resolve_cells_rejects_unknown_selections():
    with pytest.raises(SystemExit, match="unknown task"):
        resolve_cells(tasks="nonexistent")
    with pytest.raises(SystemExit, match="unknown model"):
        resolve_cells(models="GPT-9")


def test_cell_dimensions_are_the_ones_the_analysis_groups_on():
    from tests.benchmarks.hp_defaults.analyze import DIMENSIONS

    cell = build_cells()[0]
    assert set(cell.dimensions()) == set(DIMENSIONS)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def test_study_grid_brackets_the_shipped_default():
    """The point of a wider grid: a search inside the current range could
    never discover that the whole range sits too low."""
    from lqh.train import defaults

    shipped = defaults.recommended(run_type="sft").learning_rate
    rates = sorted({p.learning_rate for p in study_grid()})
    assert min(rates) < shipped < max(rates)


def test_grid_points_render_as_sweep_overrides():
    override = GridPoint(learning_rate=5e-5, num_epochs=2).to_override()
    assert override["id"] == "lr5e-05_e2"
    assert override["overrides"]["training"] == {
        "learning_rate": 5e-5, "num_epochs": 2,
    }
    assert "seed" not in override["overrides"]["training"]


def test_replicates_vary_both_seeds_together():
    """Init and shuffling must both move, or the replicate underestimates the
    run-to-run variance it exists to measure."""
    point = replicate_grid([GridPoint(5e-5, 2)], (7,))[0]
    training = point.to_override()["overrides"]["training"]
    assert training["seed"] == 7
    assert training["data_seed"] == 7
    assert point.id.endswith("_s7")


def test_replicate_ids_stay_distinct_from_the_base_config():
    base = GridPoint(5e-5, 2)
    reps = replicate_grid([base], (1, 2))
    assert len({p.id for p in reps} | {base.id}) == 3


def test_chunking_preserves_every_point_exactly_once():
    points = study_grid()
    chunks = chunk_points(points, 6)
    assert [p for chunk in chunks for p in chunk] == points
    assert all(len(chunk) <= 6 for chunk in chunks)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        chunk_points(study_grid(), 0)


def test_parse_points_round_trips():
    points = parse_points("lr=2e-5:e2,lr=1e-4:e1")
    assert [(p.learning_rate, p.num_epochs) for p in points] == [(2e-5, 2), (1e-4, 1)]


def test_parse_points_rejects_a_malformed_spec():
    with pytest.raises(SystemExit, match="bad grid point"):
        parse_points("lr=2e-5")


# ---------------------------------------------------------------------------
# Launch payload
# ---------------------------------------------------------------------------


def _launch():
    return runner.build_launch_config(
        base_model="LiquidAI/LFM2.5-350M",
        dataset_rel="datasets/t_train_n500/data.parquet",
        eval_rel="datasets/t_eval/data.parquet",
        scorer_rel="scorers/t.md",
        grid_override=[GridPoint(5e-5, 2).to_override()],
    )


def test_launch_config_turns_on_per_config_eval():
    """Without eval_all the study has no real metric for anything but the
    winner, and cannot compute a cell's oracle at all."""
    cfg = _launch()
    assert cfg["eval_all"] is True
    # Evaluating the winner again would just duplicate one of those.
    assert cfg["eval_best"] is False
    assert cfg["type"] == "sweep"


def test_launch_config_lets_the_grid_own_the_swept_hyperparameters():
    """A learning rate left in base_config would be silently overridden — or
    worse, silently NOT overridden if the grid entry were ever dropped."""
    training = _launch()["base_config"]["training"]
    assert "learning_rate" not in training
    assert "num_epochs" not in training


def test_launch_config_keeps_the_product_defaults_for_everything_else():
    """The study must measure the configuration the product actually ships,
    so batch size and LoRA shape come from defaults.py, not from here."""
    from lqh.train import defaults

    expected = defaults.recommended(run_type="sft", lora=True)
    cfg = _launch()["base_config"]
    assert cfg["training"]["per_device_batch_size"] == expected.per_device_batch_size
    assert cfg["training"]["effective_batch_size"] == expected.effective_batch_size
    assert cfg["lora"]["r"] == expected.lora["r"]
    assert cfg["lora"]["target_modules"] == expected.lora["target_modules"]


def test_launch_config_derives_the_batch_from_the_cell_size():
    """The product derives the effective batch from the dataset, so a study
    that pinned 256 would measure a recipe the product does not ship."""
    from lqh.train import defaults

    cfg = runner.build_launch_config(
        base_model="LiquidAI/LFM2.5-350M",
        dataset_rel="datasets/t_train_n500/data.parquet",
        eval_rel="datasets/t_eval/data.parquet",
        scorer_rel="scorers/t.md",
        grid_override=[GridPoint(5e-5, 2).to_override()],
        train_rows=500,
    )
    assert cfg["base_config"]["training"]["effective_batch_size"] == (
        defaults.sft_effective_batch(500, defaults.DEFAULT_SFT_EPOCHS)
    )


def test_launch_config_batch_is_invariant_to_the_epochs_axis():
    """The batch must be constant across a cell's configs: if it moved with the
    grid's epochs axis, an LR difference and a batch difference would be
    confounded and the study's answer would be uninterpretable."""
    batches = set()
    for epochs in (1, 2, 3):
        cfg = runner.build_launch_config(
            base_model="LiquidAI/LFM2.5-350M",
            dataset_rel="datasets/t_train_n8000/data.parquet",
            eval_rel="datasets/t_eval/data.parquet",
            scorer_rel="scorers/t.md",
            grid_override=[GridPoint(5e-5, epochs).to_override()],
            train_rows=8_000,
        )
        batches.add(cfg["base_config"]["training"]["effective_batch_size"])
    assert len(batches) == 1


def test_launch_config_disables_checkpoint_eval():
    """eval_on_checkpoints would judge-score at every save on top of the
    per-config eval — the same cost again, for data nothing reads."""
    assert _launch()["base_config"]["eval_on_checkpoints"] is False


def test_launch_config_manifests_every_path_the_bundle_must_carry():
    cfg = _launch()["base_config"]
    assert set(cfg["manifest"]) == {
        "base_model", "dataset", "eval_dataset", "scorer",
    }


# ---------------------------------------------------------------------------
# Reading results back
# ---------------------------------------------------------------------------


def test_read_rows_returns_nothing_for_a_missing_or_corrupt_summary(tmp_path):
    assert runner.read_rows(tmp_path, "nope") == []
    run_dir = tmp_path / "runs" / "broken"
    run_dir.mkdir(parents=True)
    (run_dir / "sweep_summary.json").write_text("{not json")
    assert runner.read_rows(tmp_path, "broken") == []


def test_job_complete_requires_every_config_to_have_a_row(tmp_path):
    """Resume matters: a 48-cell study loses jobs to preemption, and re-running
    a finished cell costs an hour of GPU."""
    run_dir = tmp_path / "runs" / "job"
    run_dir.mkdir(parents=True)
    (run_dir / "sweep_summary.json").write_text(
        json.dumps({"rows": [{"config_id": "a"}, {"config_id": "b"}]})
    )
    assert runner.job_complete(tmp_path, "job", 2) is True
    assert runner.job_complete(tmp_path, "job", 3) is False


def test_canonical_config_id_folds_replicates_together():
    """Replicates are the same hyperparameters — if the seed stayed in the id
    they would look like distinct configs and never form a replicate group."""
    from tests.benchmarks.hp_defaults.run import _canonical_config_id

    assert _canonical_config_id("lr5e-05_e2_s3") == "lr5e-05_e2"
    assert _canonical_config_id("lr5e-05_e2") == "lr5e-05_e2"
    # Not a seed suffix — must be left alone.
    assert _canonical_config_id("lr5e-05_e2_slow") == "lr5e-05_e2_slow"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _observations(judge_by_config: dict[str, float], cells: int = 4):
    out = []
    for i in range(cells):
        dims = {
            "task": "translation" if i % 2 else "extraction",
            "train_size": 500 if i < 2 else 2000,
            "model_kind": "base" if i % 2 else "instruct",
            "param_count": 1.2,
        }
        for config, judge in judge_by_config.items():
            out.append(Observation(
                cell_id=f"c{i}", config_id=config, dimensions=dims,
                judge_mean=judge, eval_loss=10.0 - judge,
                num_epochs=2, best_epoch=2.0, elapsed_s=100.0,
                judge_num_scored=100, judge_std=2.0,
            ))
    return out


def test_report_leads_with_the_recommendation_and_the_snippet():
    results = build_results(
        {"run_name": "r1", "compute": "cloud", "judge_size": "small"},
        _observations({"lr5e-05_e2": 8.0, "lr1e-05_e2": 6.0}),
    )
    md = render_markdown(results)
    assert "## Recommended default" in md
    assert "`lr5e-05_e2`" in md
    assert "## Install this" in md
    assert "PROVENANCE" in md
    assert "learning_rate = 5e-05" in md


def test_report_flags_a_judge_sem_floor_as_a_lower_bound_only():
    """Judge SEM captures judging variance but not seed-to-seed training
    variance, so it must not be presented as a validated floor — silence here
    would let a reader treat an unguarded split as confirmed."""
    results = build_results({}, _observations({"a": 8.0, "b": 7.0}))
    assert results["noise_floor"]["basis"] == "judge_sem"
    md = render_markdown(results)
    assert "lower bound only" in md
    assert "--replicate-seeds" in md


def test_report_states_a_replicate_backed_floor_plainly():
    obs = _observations({"a": 8.0, "b": 7.0})
    obs += [
        Observation(cell_id="c0", config_id="a", dimensions={}, judge_mean=8.0, seed=1),
        Observation(cell_id="c0", config_id="a", dimensions={}, judge_mean=8.3, seed=2),
    ]
    results = build_results({}, obs)
    assert results["noise_floor"]["basis"] == "seed_replicates"
    md = render_markdown(results)
    assert "seed-replicate group" in md
    assert "lower bound only" not in md


def test_report_says_so_when_there_is_no_noise_floor_at_all():
    obs = [
        Observation(cell_id=f"c{i}", config_id=cfg, dimensions={}, judge_mean=j)
        for i in range(3) for cfg, j in (("a", 8.0), ("b", 7.0))
    ]
    results = build_results({}, obs)
    assert results["noise_floor"]["basis"] == "none"
    assert "No noise floor at all" in render_markdown(results)


def test_report_surfaces_failed_jobs_rather_than_hiding_the_gap():
    results = build_results(
        {}, _observations({"a": 8.0, "b": 7.0}), ["cell_x#0: timed out"],
    )
    md = render_markdown(results)
    assert "Jobs that did not complete" in md
    assert "cell_x#0: timed out" in md


def test_report_declines_to_recommend_when_nothing_scored():
    results = build_results({}, [
        Observation(cell_id="c1", config_id="a", dimensions={}, judge_mean=None),
    ])
    md = render_markdown(results)
    assert "## No recommendation" in md
    assert "## Install this" not in md


def test_results_json_round_trips():
    """The raw observations ship with the report so the analysis can be
    re-run — and argued with — without touching a GPU."""
    results = build_results({}, _observations({"a": 8.0, "b": 7.0}))
    reloaded = json.loads(json.dumps(results, default=str))
    assert len(reloaded["observations"]) == 8
    assert reloaded["recommendation"]["config_id"] == "a"


# ---------------------------------------------------------------------------
# Fetching results back from the cloud
# ---------------------------------------------------------------------------


async def test_fetch_result_artifacts_downloads_the_sweep_leaderboard(
    tmp_path, monkeypatch,
):
    """sync_progress writes progress/status/artifacts.json but does NOT pull
    artifact bytes from R2. Without this step a cloud job completes perfectly
    and still leaves no sweep_summary.json on the laptop — which reads
    downstream as a job that produced nothing.
    """
    run_dir = tmp_path / "runs" / "job"
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts.json").write_text(json.dumps({"artifacts": [
        {"artifact_id": "a1", "relpath": "sweep_summary.json", "kind": "metrics"},
        {"artifact_id": "a2", "relpath": "runs.jsonl", "kind": "metrics"},
        {"artifact_id": "a3", "relpath": "sweep_x/model-lora", "kind": "checkpoint"},
    ]}))

    downloaded: list[tuple[str, str]] = []

    class FakeStore:
        async def download(self, artifact_id, dest):
            downloaded.append((artifact_id, dest.name))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("{}")

    monkeypatch.setattr("lqh.artifacts.BackendArtifactStore", lambda: FakeStore())

    fetched = await runner.fetch_result_artifacts(run_dir)

    assert sorted(fetched) == ["runs.jsonl", "sweep_summary.json"]
    assert (run_dir / "sweep_summary.json").exists()
    # Multi-GB checkpoints must not be dragged down for every cell.
    assert "a3" not in [a for a, _ in downloaded]


async def test_fetch_result_artifacts_survives_a_missing_listing(tmp_path):
    run_dir = tmp_path / "runs" / "job"
    run_dir.mkdir(parents=True)
    assert await runner.fetch_result_artifacts(run_dir) == []
    (run_dir / "artifacts.json").write_text("{not json")
    assert await runner.fetch_result_artifacts(run_dir) == []


async def test_fetch_result_artifacts_reports_a_download_failure_as_absent(
    tmp_path, monkeypatch,
):
    """A failed fetch must not look like a successful one — the caller uses the
    returned list to decide whether the job really produced results."""
    run_dir = tmp_path / "runs" / "job"
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts.json").write_text(json.dumps({"artifacts": [
        {"artifact_id": "a1", "relpath": "sweep_summary.json"},
    ]}))

    class BrokenStore:
        async def download(self, artifact_id, dest):
            raise RuntimeError("R2 said no")

    monkeypatch.setattr("lqh.artifacts.BackendArtifactStore", lambda: BrokenStore())
    assert await runner.fetch_result_artifacts(run_dir) == []


# ---------------------------------------------------------------------------
# Auth is a precondition, not a per-cell failure
# ---------------------------------------------------------------------------


def test_auth_errors_are_told_apart_from_job_errors():
    from lqh.remote.cloud import CloudError

    assert runner.is_auth_error(CloudError("401: authentication required"))
    assert runner.is_auth_error(CloudError("403: forbidden"))
    # A real job failure must still be handled per-cell, not abort the study.
    assert not runner.is_auth_error(CloudError("500: internal error"))
    assert not runner.is_auth_error(RuntimeError("401"))


def test_auth_help_names_the_usual_cause():
    """LQH_DEBUG_API_KEY is accepted on /v1/* so datagen succeeds, but it
    carries no user or org and every cloud submit is rejected. A bare
    '401: authentication required' sends people looking in the wrong place.
    """
    help_text = runner._auth_help("401: authentication required")
    assert "LQH_DEBUG_API_KEY" in help_text
    assert "unset LQH_DEBUG_API_KEY" in help_text
    assert "/login" in help_text


async def test_cloud_auth_preflight_raises_on_a_rejected_token(tmp_path, monkeypatch):
    """It must fail here — before data generation — not hours later."""
    from lqh.remote.cloud import CloudError

    class RejectingBackend:
        def __init__(self, *a, **kw):
            pass

        async def plan_job(self, kind, **kw):
            raise CloudError("401: authentication required")

    monkeypatch.setattr("lqh.remote.cloud.CloudBackend", RejectingBackend)

    with pytest.raises(runner.CloudAuthError, match="LQH_DEBUG_API_KEY"):
        await runner.check_cloud_auth(tmp_path)


async def test_cloud_auth_preflight_tolerates_a_non_auth_planner_failure(
    tmp_path, monkeypatch,
):
    """A planner blip must not block a study whose credentials are fine —
    submit will surface any real problem with a better message."""
    from lqh.remote.cloud import CloudError

    class FlakyBackend:
        def __init__(self, *a, **kw):
            pass

        async def plan_job(self, kind, **kw):
            raise CloudError("503: planner unavailable")

    monkeypatch.setattr("lqh.remote.cloud.CloudBackend", FlakyBackend)
    await runner.check_cloud_auth(tmp_path)  # must not raise


async def test_cloud_auth_preflight_passes_on_a_good_token(tmp_path, monkeypatch):
    class OkBackend:
        def __init__(self, *a, **kw):
            pass

        async def plan_job(self, kind, **kw):
            return {"fits": True, "gpu_type": "A100-40GB"}

    monkeypatch.setattr("lqh.remote.cloud.CloudBackend", OkBackend)
    await runner.check_cloud_auth(tmp_path)


# ---------------------------------------------------------------------------
# Jobs that report success and publish nothing
#
# Stage A, 2026-08-06: two sandboxes went quiet mid-eval and the next backend
# restart ("job pump reattached") wrote exit_code=0 for both. Zero artifacts,
# full GPU time billed, and the harness recorded them as permanent failures
# against a message blaming the sandbox image. Both halves were wrong.
# ---------------------------------------------------------------------------


def _spec(run_name: str = "cell__c0", n_configs: int = 6) -> runner.JobSpec:
    return runner.JobSpec(
        cell_id="cell", chunk_index=0, run_name=run_name,
        launch_config={"grid_override": [{"training": {}} for _ in range(n_configs)]},
    )


def test_no_artifacts_at_all_reads_as_an_orphan_not_a_stale_image(tmp_path):
    (tmp_path / "artifacts.json").write_text(json.dumps({"artifacts": []}))
    msg = runner._no_result_message(_spec(), "job-1", tmp_path)
    assert "orphan" in msg
    assert "Retrying" in msg
    assert "Modal" not in msg  # the image is not the suspect here


def test_a_published_run_missing_only_the_leaderboard_blames_the_image(tmp_path):
    (tmp_path / "artifacts.json").write_text(json.dumps({"artifacts": [
        {"relpath": "model-lora/adapter.safetensors", "artifact_id": "a"},
        {"relpath": "stdout.log", "artifact_id": "b"},
    ]}))
    msg = runner._no_result_message(_spec(), "job-1", tmp_path)
    assert "eval_all" in msg
    assert "rebuild the Modal training image" in msg
    assert "orphan" not in msg


def test_reset_clears_the_terminal_state_that_would_block_a_resubmit(tmp_path):
    run_dir = tmp_path / "runs" / "cell__c0"
    run_dir.mkdir(parents=True)
    for name in ("cloud_state.json", "artifacts.json", "status.json",
                 "progress.jsonl", "stdout.log"):
        (run_dir / name).write_text("stale")
    (run_dir / "config.json").write_text("keep me")

    runner.reset_run_dir(tmp_path, "cell__c0")

    assert not (run_dir / "cloud_state.json").exists()
    assert not (run_dir / "artifacts.json").exists()
    assert (run_dir / "config.json").exists()


def _cloud_backend(monkeypatch, *, state: str, failure=None):
    from lqh.remote.backend import JobStatus

    class Backend:
        def __init__(self, *a, **kw):
            pass

        async def submit_run(self, run_dir, config, module="lqh.train"):
            return "job-1"

        async def sync_progress(self, remote, local):
            return None

        async def poll_status(self, job_id):
            return JobStatus(state=state, error="boom", failure=failure)

    monkeypatch.setattr("lqh.remote.cloud.CloudBackend", Backend)


async def test_a_completed_job_with_no_leaderboard_is_retryable(tmp_path, monkeypatch):
    _cloud_backend(monkeypatch, state="completed")
    monkeypatch.setattr(runner, "fetch_result_artifacts", _returns([]))

    with pytest.raises(runner.EmptyResultError):
        await runner.run_cloud(_spec(), workdir=tmp_path, timeout=60)


async def test_an_infra_classified_failure_is_retryable(tmp_path, monkeypatch):
    _cloud_backend(monkeypatch, state="failed", failure={"cls": "preempted", "infra": True})

    with pytest.raises(runner.InfraJobError):
        await runner.run_cloud(_spec(), workdir=tmp_path, timeout=60)


async def test_a_config_failure_is_not_retryable(tmp_path, monkeypatch):
    """An OOM or a bad config fails identically on the second submit — the
    only thing a retry buys is another GPU bill."""
    _cloud_backend(monkeypatch, state="failed", failure={"cls": "oom", "infra": False})

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_cloud(_spec(), workdir=tmp_path, timeout=60)
    assert not isinstance(excinfo.value, runner.RetryableJobError)


def _returns(value):
    async def _f(*a, **kw):
        return value
    return _f


# ---------------------------------------------------------------------------
# The retry loop in _run_cell
# ---------------------------------------------------------------------------


def _cell_args(**over):
    import argparse
    base = dict(no_resume=True, compute="cloud", job_timeout=60)
    base.update(over)
    return argparse.Namespace(**base)


async def _run_one_cell(monkeypatch, tmp_path, outcomes: list):
    """Drive _run_cell with a scripted sequence of run_job outcomes."""
    import asyncio

    from tests.benchmarks.hp_defaults import run as run_mod
    from tests.benchmarks.hp_defaults.cells import build_cells

    attempts: list[str] = []

    async def fake_run_job(spec, **kw):
        attempts.append(spec.run_name)
        outcome = outcomes[len(attempts) - 1]
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(runner, "run_job", fake_run_job)
    notes = await run_mod._run_cell(
        build_cells()[0], [_spec()], workdir=tmp_path, args=_cell_args(),
        semaphore=asyncio.Semaphore(1),
    )
    return attempts, notes


async def test_an_orphaned_chunk_is_resubmitted_once(tmp_path, monkeypatch):
    attempts, notes = await _run_one_cell(
        monkeypatch, tmp_path, [runner.EmptyResultError("orphan"), None],
    )
    assert len(attempts) == 2
    assert notes == []  # the retry succeeded, so the cell is whole


async def test_a_chunk_lost_twice_is_reported_not_retried_forever(
    tmp_path, monkeypatch,
):
    attempts, notes = await _run_one_cell(
        monkeypatch, tmp_path,
        [runner.EmptyResultError("orphan"), runner.EmptyResultError("orphan again")],
    )
    assert len(attempts) == 2
    assert len(notes) == 1 and "orphan again" in notes[0]


async def test_a_config_failure_is_recorded_on_the_first_attempt(
    tmp_path, monkeypatch,
):
    attempts, notes = await _run_one_cell(
        monkeypatch, tmp_path, [RuntimeError("CUDA out of memory"), None],
    )
    assert len(attempts) == 1
    assert len(notes) == 1 and "out of memory" in notes[0]


async def test_a_retry_clears_the_previous_attempts_state(tmp_path, monkeypatch):
    """Without this the resubmit inherits a terminal cloud_state.json and
    sync_progress returns instantly, so the new job is never watched."""
    run_dir = tmp_path / "runs" / "cell__c0"
    run_dir.mkdir(parents=True)
    (run_dir / "cloud_state.json").write_text('{"status": "completed"}')

    await _run_one_cell(
        monkeypatch, tmp_path, [runner.EmptyResultError("orphan"), None],
    )
    assert not (run_dir / "cloud_state.json").exists()


async def test_auth_failure_still_aborts_instead_of_retrying(tmp_path, monkeypatch):
    from lqh.remote.cloud import CloudError

    with pytest.raises(runner.CloudAuthError):
        await _run_one_cell(
            monkeypatch, tmp_path,
            [CloudError("401: authentication required"), None],
        )
