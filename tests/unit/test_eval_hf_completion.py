"""eval_hf completion contract, predictions resume, timeout consent,
client completion gating, and the stale-progress marker.

Covers the ISSUE-4 P0 fixes: an eval without an eval_result.json must
never present as completed, large evals resume from partial predictions
on continuation, and the submit path surfaces the (configurable) timeout
as a hard cost cap behind a consent prompt.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lqh.infer.__main__ import (
    PREDICTIONS_PARTIAL,
    _append_prediction_partial,
    _init_prediction_partial,
    _load_prediction_partial,
    _predictions_digest,
)

# ---------------------------------------------------------------------------
# Partial predictions: append/load/init
# ---------------------------------------------------------------------------


def _entry(i: int) -> dict:
    return {
        "sample_index": i,
        "messages": json.dumps([{"role": "assistant", "content": f"p{i}"}]),
        "source": "evals/x.parquet",
    }


def test_prediction_partial_roundtrip(tmp_path: Path) -> None:
    digest = "d" * 64
    resumed = _init_prediction_partial(tmp_path, 5, digest)
    assert resumed == {}
    path = tmp_path / PREDICTIONS_PARTIAL
    for i in (0, 1, 3):
        _append_prediction_partial(path, i, _entry(i))
    rows = _load_prediction_partial(path, 5, digest)
    assert rows is not None
    assert set(rows) == {0, 1, 3}
    assert rows[3]["sample_index"] == 3
    assert "index" not in rows[3]


def test_prediction_partial_tolerates_truncated_tail(tmp_path: Path) -> None:
    digest = "d" * 64
    _init_prediction_partial(tmp_path, 5, digest)
    path = tmp_path / PREDICTIONS_PARTIAL
    _append_prediction_partial(path, 0, _entry(0))
    with open(path, "a") as f:
        f.write('{"index": 1, "sample_ind')  # killed mid-write
    rows = _load_prediction_partial(path, 5, digest)
    assert rows is not None and set(rows) == {0}


def test_prediction_partial_ignores_out_of_range(tmp_path: Path) -> None:
    digest = "d" * 64
    _init_prediction_partial(tmp_path, 3, digest)
    path = tmp_path / PREDICTIONS_PARTIAL
    _append_prediction_partial(path, 0, _entry(0))
    _append_prediction_partial(path, 7, _entry(7))  # beyond total
    rows = _load_prediction_partial(path, 3, digest)
    assert rows is not None and set(rows) == {0}


def test_prediction_partial_digest_mismatch_restarts(tmp_path: Path) -> None:
    _init_prediction_partial(tmp_path, 5, "old" * 22)
    path = tmp_path / PREDICTIONS_PARTIAL
    _append_prediction_partial(path, 0, _entry(0))

    resumed = _init_prediction_partial(tmp_path, 5, "new" * 22)
    assert resumed == {}
    # Paid-for GPU output preserved, not destroyed.
    stale = tmp_path / "predictions.partial.stale.jsonl"
    assert stale.exists()
    assert '"index": 0' in stale.read_text()
    # Fresh header bound to the new digest.
    header = json.loads(path.read_text().splitlines()[0])
    assert header["_meta"] and header["digest"] == "new" * 22


def test_prediction_partial_total_mismatch_restarts(tmp_path: Path) -> None:
    _init_prediction_partial(tmp_path, 5, "d" * 64)
    path = tmp_path / PREDICTIONS_PARTIAL
    _append_prediction_partial(path, 0, _entry(0))
    assert _load_prediction_partial(path, 9, "d" * 64) is None
    assert _init_prediction_partial(tmp_path, 9, "d" * 64) == {}


def test_predictions_digest_sensitivity() -> None:
    base = {"base_model": "m", "dataset": "d", "max_new_tokens": 4096}
    assert _predictions_digest(base) == _predictions_digest(dict(reversed(list(base.items()))))
    for key, val in (
        ("base_model", "other"),
        ("dataset", "other.parquet"),
        ("max_new_tokens", 8192),
        ("system_prompt", "be brief"),
        ("spec_sha256", "abc"),
    ):
        changed = {**base, key: val}
        assert _predictions_digest(changed) != _predictions_digest(base), key


# ---------------------------------------------------------------------------
# eval_hf terminal-status ownership
# ---------------------------------------------------------------------------


def _eval_hf_config(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "type": "eval_hf",
        "hf_repo": "org/model",
        "training_method": "full",
        "eval_dataset": "evals/x.parquet",
        "scorer": "scorers/x.md",
    }
    path = run_dir / "config.json"
    path.write_text(json.dumps(config))
    return path


def _last_status(run_dir: Path) -> str | None:
    status = None
    for line in (run_dir / "progress.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "status" in row:
            status = row["status"]
    return status


def _run_eval_hf_main(monkeypatch, config_path: Path, scoring_error: str | None):
    import lqh.infer.eval_hf as eval_hf

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        eval_hf, "_download_checkpoint",
        lambda repo, rev, root: config_path.parent / "ckpt",
    )

    def fake_run_inference(run_dir: Path, infer_config: dict) -> None:
        captured["infer_config"] = infer_config

    monkeypatch.setattr(
        "lqh.infer.__main__._run_inference", fake_run_inference,
    )
    monkeypatch.setattr(
        eval_hf, "_run_inline_scoring",
        lambda run_dir, infer_config: scoring_error,
    )
    monkeypatch.setattr("sys.argv", ["eval_hf", str(config_path)])
    eval_hf.main()
    return captured


def test_eval_hf_defers_terminal_status_to_scoring(tmp_path, monkeypatch) -> None:
    config_path = _eval_hf_config(tmp_path)
    # A leftover resume scratch must be cleaned up only on full success.
    (config_path.parent / PREDICTIONS_PARTIAL).write_text("{}\n")
    captured = _run_eval_hf_main(monkeypatch, config_path, scoring_error=None)
    assert captured["infer_config"]["defer_terminal_status"] is True
    assert _last_status(config_path.parent) == "completed"
    assert not (config_path.parent / PREDICTIONS_PARTIAL).exists()


def test_eval_hf_scoring_failure_exits_nonzero(tmp_path, monkeypatch) -> None:
    config_path = _eval_hf_config(tmp_path)
    (config_path.parent / PREDICTIONS_PARTIAL).write_text("{}\n")
    with pytest.raises(SystemExit) as exc_info:
        _run_eval_hf_main(monkeypatch, config_path, scoring_error="judge exploded")
    assert exc_info.value.code == 4
    run_dir = config_path.parent
    assert _last_status(run_dir) == "failed"
    marker = json.loads((run_dir / "eval_error.json").read_text())
    assert "judge exploded" in marker["error"]
    # Resume state survives a scoring failure so a retry/continuation
    # skips regeneration and only redoes scoring.
    assert (run_dir / PREDICTIONS_PARTIAL).exists()


def test_eval_hf_keeps_existing_eval_error_marker(tmp_path, monkeypatch) -> None:
    config_path = _eval_hf_config(tmp_path)
    run_dir = config_path.parent
    (run_dir / "eval_error.json").write_text(
        json.dumps({"error": "all judge scoring attempts failed"})
    )
    with pytest.raises(SystemExit):
        _run_eval_hf_main(monkeypatch, config_path, scoring_error="generic")
    marker = json.loads((run_dir / "eval_error.json").read_text())
    # The more specific marker written by cloud_score is not clobbered.
    assert marker["error"] == "all judge scoring attempts failed"


def test_run_inference_writes_completed_without_defer_flag(tmp_path) -> None:
    # Plain infer runs keep the historical unconditional status write —
    # asserted at the source so the eval_hf gate can't leak onto them.
    # Both engines finish through _finalize_predictions, so the gate
    # lives (only) there.
    import inspect

    from lqh.infer.__main__ import _finalize_predictions

    src = inspect.getsource(_finalize_predictions)
    assert 'if not config.get("defer_terminal_status")' in src


# ---------------------------------------------------------------------------
# _run_inline_scoring failure modes
# ---------------------------------------------------------------------------


def _score(monkeypatch, tmp_path: Path, *, cloud=True, summary=None,
           raise_exc=None, result_file=False, stamp_raises=False):
    from lqh.infer import eval_hf

    monkeypatch.setattr("lqh.train.cloud_score.is_cloud_mode", lambda: cloud)

    def fake_score(run_dir, infer_config):
        if raise_exc is not None:
            raise raise_exc
        return summary

    monkeypatch.setattr("lqh.train.cloud_score.score_run_eval_inline", fake_score)
    if result_file:
        (tmp_path / "eval_result.json").write_text(json.dumps({
            "scores": {"mean": 5.0, "median": 5.0},
            "num_scored": 3, "num_failed": 0,
        }))
    if stamp_raises:
        monkeypatch.setattr(
            eval_hf, "_stamp_real_metric",
            lambda *a: (_ for _ in ()).throw(RuntimeError("stamp boom")),
        )
    return eval_hf._run_inline_scoring(tmp_path, {"scorer": "scorers/x.md"})


def test_inline_scoring_not_cloud_mode_is_failure(tmp_path, monkeypatch) -> None:
    err = _score(monkeypatch, tmp_path, cloud=False)
    assert err is not None and "cloud" in err


def test_inline_scoring_exception_is_failure(tmp_path, monkeypatch) -> None:
    err = _score(monkeypatch, tmp_path, raise_exc=ValueError("bad rubric"))
    assert err is not None and "bad rubric" in err


def test_inline_scoring_none_summary_is_failure(tmp_path, monkeypatch) -> None:
    err = _score(monkeypatch, tmp_path, summary=None)
    assert err is not None and "no summary" in err


def test_inline_scoring_summary_without_file_is_failure(tmp_path, monkeypatch) -> None:
    err = _score(monkeypatch, tmp_path, summary={"scores": {"mean": 5.0}})
    assert err is not None and "eval_result.json" in err


def test_inline_scoring_success(tmp_path, monkeypatch) -> None:
    err = _score(
        monkeypatch, tmp_path,
        summary={"scores": {"mean": 5.0}, "num_scored": 3}, result_file=True,
    )
    assert err is None


def test_inline_scoring_stamp_failure_does_not_fail(tmp_path, monkeypatch) -> None:
    err = _score(
        monkeypatch, tmp_path,
        summary={"scores": {"mean": 5.0}, "num_scored": 3},
        result_file=True, stamp_raises=True,
    )
    assert err is None


def test_inline_scoring_invalid_json_result_is_failure(tmp_path, monkeypatch) -> None:
    (tmp_path / "eval_result.json").write_text("{not json")
    err = _score(monkeypatch, tmp_path, summary={"scores": {"mean": 5.0}})
    assert err is not None and "invalid JSON" in err


def test_inline_scoring_result_without_mean_is_failure(tmp_path, monkeypatch) -> None:
    (tmp_path / "eval_result.json").write_text(json.dumps({"num_scored": 3}))
    err = _score(monkeypatch, tmp_path, summary={"scores": {"mean": 5.0}})
    assert err is not None and "scores.mean" in err


def test_inline_scoring_zero_scored_result_is_failure(tmp_path, monkeypatch) -> None:
    (tmp_path / "eval_result.json").write_text(json.dumps({
        "scores": {"mean": 0.0, "median": 0.0}, "num_scored": 0,
    }))
    err = _score(monkeypatch, tmp_path, summary={"scores": {"mean": 0.0}})
    assert err is not None and "zero scored samples" in err


# ---------------------------------------------------------------------------
# publish gate
# ---------------------------------------------------------------------------


def _publish_main(monkeypatch, tmp_path: Path, handles, failed=()):
    import lqh.remote.publish as publish

    async def fake_publish_run(run_dir, **kwargs):
        return publish.PublishResult(artifacts=list(handles), failed=list(failed))

    monkeypatch.setattr(publish, "publish_run", fake_publish_run)
    return publish.main([str(tmp_path), "--project-id", "p", "--token", "t"])


def test_publish_gate_eval_hf_requires_eval_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LQH_KIND", "eval_hf")
    request_only = SimpleNamespace(kind="eval_result", r2_key="a/eval_result/ff-eval_request.json")
    assert _publish_main(monkeypatch, tmp_path, [request_only]) == 1

    result = SimpleNamespace(kind="eval_result", r2_key="a/eval_result/ff-eval_result.json")
    assert _publish_main(monkeypatch, tmp_path, [request_only, result]) == 0


def test_publish_gate_eval_hf_ignores_failed_log_uploads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LQH_KIND", "eval_hf")
    result = SimpleNamespace(kind="eval_result", r2_key="a/eval_result/ff-eval_result.json")
    rc = _publish_main(
        monkeypatch, tmp_path, [result], failed=[("stdout.log", "boom")],
    )
    assert rc == 0


def test_publish_gate_other_kinds_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LQH_KIND", raising=False)
    ok = SimpleNamespace(kind="metrics", r2_key="a/metrics/ff-progress.jsonl")
    assert _publish_main(monkeypatch, tmp_path, [ok]) == 0
    assert _publish_main(monkeypatch, tmp_path, [ok], failed=[("x", "e")]) == 1


# ---------------------------------------------------------------------------
# eval_hf_model handler: timeout + consent
# ---------------------------------------------------------------------------


def _eval_project(tmp_path: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    project = tmp_path / "project"
    (project / "evals" / "x").mkdir(parents=True)
    (project / "scorers").mkdir()
    messages = [
        json.dumps([
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ])
        for i in range(4)
    ]
    pq.write_table(
        pa.table({"messages": messages}), project / "evals" / "x" / "data.parquet",
    )
    (project / "scorers" / "x.md").write_text("Score 0-10.")
    return project


async def _none(*args, **kwargs):
    return None


async def _plan_unavailable(self, kind, *, base_model=None, config=None):
    raise RuntimeError("older backend")


@pytest.mark.asyncio
async def test_eval_hf_consent_prompt_shows_timeout_and_cost(tmp_path, monkeypatch) -> None:
    from lqh.remote.cloud import CloudBackend
    from lqh.tools.handlers import handle_eval_hf_model

    # Plan preview unavailable → default-GPU estimate with caveat.
    monkeypatch.setattr(CloudBackend, "plan_job", _plan_unavailable)
    monkeypatch.setattr("lqh.tools.handlers._fetch_eval_hf_rate_usd", _none)
    project = _eval_project(tmp_path)
    result = await handle_eval_hf_model(
        project, repo="org/model", eval_dataset="evals/x",
        scorer="scorers/x.md", training_method="full", timeout_minutes=180,
    )
    assert result.content == "PERMISSION_REQUIRED"
    assert result.requires_user_input
    assert result.permission_key == "cloud_eval_hf:org/model"
    q = result.question or ""
    assert "3-hour timeout" in q
    assert "$" in q
    assert "4 samples" in q
    assert "default GPU" in q  # honest caveat when only estimating


@pytest.mark.asyncio
async def test_eval_hf_consent_uses_planned_gpu_and_cap(tmp_path, monkeypatch) -> None:
    from lqh.remote.cloud import CloudBackend
    from lqh.tools.handlers import handle_eval_hf_model

    async def fake_plan(self, kind, *, base_model=None, config=None):
        assert kind == "eval_hf"
        assert config["hf_repo"] == "org/big-model"
        return {
            "fits": True, "gpu_type": "A100-80GB", "gpu_count": 1,
            "timeout_minutes": 120,
            "worst_case_cost_billed_micros": 18_000_000,
            "selection_reason": "eval: picked A100-80GB (80GB) for required 56GB",
        }

    monkeypatch.setattr(CloudBackend, "plan_job", fake_plan)
    project = _eval_project(tmp_path)
    result = await handle_eval_hf_model(
        project, repo="org/big-model", eval_dataset="evals/x",
        scorer="scorers/x.md", training_method="full",
    )
    assert result.content == "PERMISSION_REQUIRED"
    q = result.question or ""
    # Consent covers the ACTUAL planned GPU and billed cap.
    assert "A100-80GB GPU" in q
    assert "$18.00" in q


@pytest.mark.asyncio
async def test_eval_hf_no_fit_model_rejected_before_consent(tmp_path, monkeypatch) -> None:
    from lqh.remote.cloud import CloudBackend
    from lqh.tools.handlers import handle_eval_hf_model

    async def fake_plan(self, kind, *, base_model=None, config=None):
        return {"fits": False, "no_fit_reason": "org/huge needs ~250 GB VRAM for inference"}

    monkeypatch.setattr(CloudBackend, "plan_job", fake_plan)
    project = _eval_project(tmp_path)
    result = await handle_eval_hf_model(
        project, repo="org/huge", eval_dataset="evals/x",
        scorer="scorers/x.md", training_method="full",
    )
    assert result.content != "PERMISSION_REQUIRED"
    assert "fits no supported GPU" in result.content
    assert "250 GB" in result.content


@pytest.mark.asyncio
async def test_eval_hf_submit_clamps_timeout_into_config(tmp_path, monkeypatch) -> None:
    from lqh.remote.cloud import CloudBackend
    from lqh.tools.handlers import handle_eval_hf_model
    from lqh.tools.permissions import PermissionContext

    submitted: dict = {}

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        submitted["config"] = config
        submitted["module"] = module
        return "job-7"

    async def fake_snapshot(self, job_id):
        return {"resource": {
            "gpu_type": "A100-80GB", "timeout_minutes": 1440,
            "worst_case_cost_billed_micros": 216_000_000,
        }}

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    monkeypatch.setattr(CloudBackend, "job_snapshot", fake_snapshot)
    monkeypatch.setattr(CloudBackend, "plan_job", _plan_unavailable)
    project = _eval_project(tmp_path)
    result = await handle_eval_hf_model(
        project, repo="org/model", eval_dataset="evals/x",
        scorer="scorers/x.md", training_method="full",
        timeout_minutes=5000,
        _permissions=PermissionContext.granting("cloud_eval_hf"),
    )
    assert "HF eval submitted" in result.content
    # Post-submit line reflects the planner's ACTUAL selection (upsized
    # GPU + billed cap), not the consent-time default estimate.
    assert "A100-80GB GPU, 1440 min timeout" in result.content
    assert "$216.00" in result.content
    assert submitted["module"] == "lqh.infer.eval_hf"
    assert submitted["config"]["timeout_minutes"] == 1440


@pytest.mark.asyncio
async def test_eval_hf_default_timeout_in_config(tmp_path, monkeypatch) -> None:
    from lqh.remote.cloud import CloudBackend
    from lqh.tools.handlers import handle_eval_hf_model
    from lqh.tools.permissions import grant_cloud_eval_hf_permission

    submitted: dict = {}

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        submitted["config"] = config
        return "job-8"

    async def failing_snapshot(self, job_id):
        raise RuntimeError("backend unreachable")

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    monkeypatch.setattr(CloudBackend, "job_snapshot", failing_snapshot)
    monkeypatch.setattr(CloudBackend, "plan_job", _plan_unavailable)
    project = _eval_project(tmp_path)
    # Durable project-wide grant (the "don't ask again" path).
    grant_cloud_eval_hf_permission(project)
    result = await handle_eval_hf_model(
        project, repo="org/model", eval_dataset="evals/x",
        scorer="scorers/x.md", training_method="full",
    )
    assert "HF eval submitted" in result.content
    # Snapshot fetch failure degrades to the requested-timeout line.
    assert "Timeout: 120 min" in result.content
    assert submitted["config"]["timeout_minutes"] == 120


# ---------------------------------------------------------------------------
# Client completion gating
# ---------------------------------------------------------------------------


def _jobs_project(tmp_path: Path, run_name: str, artifacts: list[dict] | None):
    run_dir = tmp_path / "runs" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"type": "eval_hf"}))
    if artifacts is not None:
        (run_dir / "artifacts.json").write_text(
            json.dumps({"artifacts": artifacts, "failed": []})
        )
    return run_dir


def test_finalize_eval_hf_missing_artifact_is_failure_notice(tmp_path) -> None:
    from lqh.jobs import JobSupervisor

    _jobs_project(tmp_path, "ev1", artifacts=[
        {"artifact_id": "a1", "kind": "eval_result", "relpath": "eval_request.json"},
    ])
    sup = JobSupervisor(tmp_path)
    text = asyncio.run(sup.finalize_eval_hf_run("ev1", "completed", None, "cloud"))
    assert text is not None
    assert "no eval_result.json artifact" in text
    assert "treat it as failed" in text


def test_finalize_eval_hf_downloads_and_reports_scores(tmp_path, monkeypatch) -> None:
    from lqh.jobs import JobSupervisor

    run_dir = _jobs_project(tmp_path, "ev2", artifacts=[
        {"artifact_id": "a2", "kind": "eval_result", "relpath": "eval_result.json"},
    ])

    async def fake_download(self, handle, dest):
        dest.write_text(json.dumps({"scores": {"mean": 7.25}, "num_scored": 4}))

    monkeypatch.setattr("lqh.artifacts.BackendArtifactStore.download", fake_download)
    sup = JobSupervisor(tmp_path)
    text = asyncio.run(sup.finalize_eval_hf_run("ev2", "completed", None, "cloud"))
    assert text is not None and "completed" in text
    assert "7.250" in text
    assert (run_dir / "eval_result.json").exists()


def test_finalize_eval_hf_failed_state_uses_generic_message(tmp_path) -> None:
    from lqh.jobs import JobSupervisor

    _jobs_project(tmp_path, "ev3", artifacts=None)
    sup = JobSupervisor(tmp_path)
    text = asyncio.run(sup.finalize_eval_hf_run("ev3", "failed", "sigkill", "cloud"))
    assert text is not None and "failed" in text


def _handle(**kw):
    defaults = dict(
        id="a9", kind="eval_result", project_id="p", size_bytes=10,
        r2_key="u/p/j/outputs/eval_result/ff-eval_result.json", job_id="job-9",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_resolve_eval_hf_result_falls_back_to_backend_api(tmp_path, monkeypatch) -> None:
    # A backend restart means artifact events were never streamed: the
    # local manifest is empty, but the artifact API knows the truth.
    from lqh.jobs import JobSupervisor

    run_dir = _jobs_project(tmp_path, "ev4", artifacts=None)
    (run_dir / "remote_job.json").write_text(json.dumps({"job_id": "job-9"}))

    async def fake_list(self, project_id, *, kind=None, job_id=None, limit=100):
        assert kind == "eval_result" and job_id == "job-9"
        return [
            _handle(id="req", r2_key="u/p/j/outputs/eval_result/aa-eval_request.json"),
            _handle(id="res"),
        ]

    monkeypatch.setattr(
        "lqh.artifacts.BackendArtifactStore.list_for_project", fake_list,
    )
    sup = JobSupervisor(tmp_path)
    entry, verified = asyncio.run(sup.resolve_eval_hf_result_artifact("ev4"))
    assert verified
    assert entry is not None and entry["artifact_id"] == "res"
    # Manifest backfilled so later checks stay local.
    assert sup.eval_hf_result_artifact("ev4") is not None


def test_finalize_eval_hf_unreachable_api_keeps_completed(tmp_path, monkeypatch) -> None:
    # Can't reach the artifact API → absence is inconclusive: no failure
    # claim, completed-with-caveat instead.
    from lqh.jobs import JobSupervisor

    run_dir = _jobs_project(tmp_path, "ev5", artifacts=None)
    (run_dir / "remote_job.json").write_text(json.dumps({"job_id": "job-5"}))

    async def broken_list(self, project_id, *, kind=None, job_id=None, limit=100):
        raise RuntimeError("api down")

    monkeypatch.setattr(
        "lqh.artifacts.BackendArtifactStore.list_for_project", broken_list,
    )
    sup = JobSupervisor(tmp_path)
    text = asyncio.run(sup.finalize_eval_hf_run("ev5", "completed", None, "cloud"))
    assert text is not None
    assert "could not be verified" in text
    assert "treat it as failed" not in text
    assert sup.eval_hf_verdicts["ev5"] == "unverified"


def test_finalize_eval_hf_verified_absence_is_failure(tmp_path, monkeypatch) -> None:
    from lqh.jobs import JobSupervisor

    run_dir = _jobs_project(tmp_path, "ev6", artifacts=None)
    (run_dir / "remote_job.json").write_text(json.dumps({"job_id": "job-6"}))

    async def empty_list(self, project_id, *, kind=None, job_id=None, limit=100):
        return []

    monkeypatch.setattr(
        "lqh.artifacts.BackendArtifactStore.list_for_project", empty_list,
    )
    sup = JobSupervisor(tmp_path)
    text = asyncio.run(sup.finalize_eval_hf_run("ev6", "completed", None, "cloud"))
    assert text is not None and "treat it as failed" in text
    assert sup.eval_hf_verdicts["ev6"] == "missing_result"


def test_format_cloud_resource_lines() -> None:
    from lqh.tools.handlers import _format_cloud_resource_lines

    lines = _format_cloud_resource_lines({
        "started_at": "2026-07-22T10:00:00Z",
        "ended_at": "2026-07-22T10:37:00Z",
        "resource": {
            "gpu_type": "L4", "timeout_minutes": 120,
            "worst_case_cost_billed_micros": 4_800_000,
        },
    })
    assert lines == ["  Compute: L4 GPU · 37/120 min used · hard cap ≈ $4.80"]
    assert _format_cloud_resource_lines({}) == []


# ---------------------------------------------------------------------------
# Stale-progress marker
# ---------------------------------------------------------------------------


def _cloud_backend(project_dir: Path):
    from lqh.remote.backend import RemoteConfig
    from lqh.remote.cloud import CloudBackend

    cfg = RemoteConfig(
        name="cloud", type="cloud", hostname="api.lqh.ai", remote_root="cloud:lqh",
    )
    return CloudBackend(cfg, project_dir, token="t")


def _reattach_event(seq: int):
    from lqh.remote.cloud import _SSEEvent

    return _SSEEvent(kind="log", payload={
        "seq": seq,
        "ts": "2026-07-22T10:00:00Z",
        "payload": {
            "stream": "system",
            "line": "backend restarted; job pump reattached",
        },
    })


def test_stale_progress_marker_appended_on_reattach(tmp_path) -> None:
    from lqh.progress import format_event_oneline
    from lqh.remote.cloud import _CloudState

    run_dir = tmp_path / "runs" / "ev"
    run_dir.mkdir(parents=True)
    seed = {
        "overall_fraction": 0.25, "phase": "inference",
        "completed": 40, "total": 445, "unit": "samples", "attempt_id": "att-1",
    }
    (run_dir / "progress.jsonl").write_text(json.dumps(seed) + "\n")

    backend = _cloud_backend(tmp_path)
    state = _CloudState(job_id="j1")
    asyncio.run(backend._apply_event(run_dir, state, _reattach_event(1)))

    rows = [
        json.loads(line)
        for line in (run_dir / "progress.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    marker = rows[-1]
    assert marker["overall_fraction"] == 0.25
    assert marker["attempt_id"] == "att-1"
    assert marker["completed"] == 40
    assert "stale" in marker["detail"]
    oneline_text, _pct = format_event_oneline(marker)
    assert "stale" in oneline_text
    # The system line still lands in stdout.log.
    assert "job pump reattached" in (run_dir / "stdout.log").read_text()


def test_stale_progress_marker_skipped_without_v1_row(tmp_path) -> None:
    from lqh.remote.cloud import _CloudState

    run_dir = tmp_path / "runs" / "ev"
    run_dir.mkdir(parents=True)
    backend = _cloud_backend(tmp_path)
    asyncio.run(backend._apply_event(run_dir, _CloudState(job_id="j1"), _reattach_event(1)))
    assert not (run_dir / "progress.jsonl").exists()


def test_stale_progress_marker_deduplicated(tmp_path) -> None:
    from lqh.remote.cloud import _CloudState

    run_dir = tmp_path / "runs" / "ev"
    run_dir.mkdir(parents=True)
    (run_dir / "progress.jsonl").write_text(
        json.dumps({"overall_fraction": 0.5, "attempt_id": "a"}) + "\n"
    )
    backend = _cloud_backend(tmp_path)
    state = _CloudState(job_id="j1")
    asyncio.run(backend._apply_event(run_dir, state, _reattach_event(1)))
    asyncio.run(backend._apply_event(run_dir, state, _reattach_event(2)))
    rows = (run_dir / "progress.jsonl").read_text().splitlines()
    assert len(rows) == 2  # seed + exactly one marker
