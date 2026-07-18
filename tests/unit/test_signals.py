"""Tests for the startup attention signals (lqh/signals.py)."""

from __future__ import annotations

import json
from pathlib import Path

from lqh.signals import (
    Signal,
    collect_signals,
    finished_while_away_signals,
    format_signal_block,
    load_seen_states,
    observe_run_states,
    record_seen_states,
)


def _make_run(
    project_dir: Path,
    name: str,
    *,
    config: dict | None = None,
    write_config: bool = True,
    progress: list[dict] | None = None,
    remote_job: dict | None = None,
    cloud_state: dict | None = None,
    submit_intent: dict | None = None,
) -> Path:
    run = project_dir / "runs" / name
    run.mkdir(parents=True)
    if write_config:
        (run / "config.json").write_text(json.dumps(config or {}))
    if progress is not None:
        (run / "progress.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in progress)
        )
    if remote_job is not None:
        (run / "remote_job.json").write_text(json.dumps(remote_job))
    if cloud_state is not None:
        (run / "cloud_state.json").write_text(json.dumps(cloud_state))
    if submit_intent is not None:
        (run / "submit_intent.json").write_text(json.dumps(submit_intent))
    return run


# ---------------------------------------------------------------------------
# observe_run_states
# ---------------------------------------------------------------------------


def test_observe_run_states_ssh_uses_synced_progress(project_dir: Path) -> None:
    """SSH runs have remote_job.json (job_id but no backend marker) and no
    cloud_state.json — the rsynced progress.jsonl carries their terminal
    state. Local PID liveness must not be consulted."""
    _make_run(
        project_dir,
        "ssh_done",
        remote_job={"job_id": 4242, "remote_name": "lambda1", "module": "lqh.train"},
        progress=[{"step": 5}, {"status": "completed"}],
    )
    _make_run(
        project_dir,
        "ssh_live",
        remote_job={"job_id": 4243, "remote_name": "lambda1", "module": "lqh.train"},
        progress=[{"step": 5}],
    )

    states = observe_run_states(project_dir)

    assert states["ssh_done"] == "completed"
    assert states["ssh_live"] == "running"


def test_observe_run_states_cloud_stale_state_defers_to_progress(
    project_dir: Path,
) -> None:
    """A cloud run whose cloud_state.json is stale ('running') but whose
    replayed progress log has the terminal row is reported terminal."""
    _make_run(
        project_dir,
        "cloud_stale",
        remote_job={"job_id": "j9", "backend": "cloud"},
        cloud_state={"job_id": "j9", "status": "running"},
        progress=[{"status": "failed", "error": "boom"}],
    )

    assert observe_run_states(project_dir)["cloud_stale"] == "failed"


def test_observe_run_states_covers_local_cloud_and_orphans(
    project_dir: Path,
) -> None:
    _make_run(project_dir, "done", progress=[{"status": "completed"}])
    _make_run(
        project_dir,
        "cloudy",
        remote_job={"type": "cloud", "job_id": "j1"},
        cloud_state={"job_id": "j1", "status": "running"},
    )
    _make_run(
        project_dir,
        "cloud_done",
        remote_job={"type": "cloud", "job_id": "j2"},
        cloud_state={"job_id": "j2", "status": "completed"},
    )
    _make_run(project_dir, "orphan", submit_intent={"idempotency_key": "k"})
    (project_dir / "runs" / "not_a_run").mkdir()  # no config.json → ignored

    states = observe_run_states(project_dir)

    assert states["done"] == "completed"
    assert states["cloudy"] == "running"
    assert states["cloud_done"] == "completed"
    assert states["orphan"] == "submit_fate_unknown"
    assert "not_a_run" not in states


# ---------------------------------------------------------------------------
# seen-state persistence
# ---------------------------------------------------------------------------


def test_record_and_load_seen_states_merge(project_dir: Path) -> None:
    record_seen_states(project_dir, {"a": "running"})
    record_seen_states(project_dir, {"b": "completed"})

    assert load_seen_states(project_dir) == {"a": "running", "b": "completed"}


# ---------------------------------------------------------------------------
# collect_signals
# ---------------------------------------------------------------------------


def test_running_jobs_are_signaled(project_dir: Path) -> None:
    signals = collect_signals(
        project_dir,
        snapshot=None,
        snapshot_fresh=True,
        run_states={"sft_v1": "running"},
    )

    assert [s.kind for s in signals] == ["running"]
    assert "runs/sft_v1" in signals[0].text


def test_finished_while_away_requires_previously_seen_running(
    project_dir: Path,
) -> None:
    # Previously seen running, now completed → signal.
    record_seen_states(project_dir, {"sft_v1": "running", "old": "completed"})

    signals = finished_while_away_signals(
        project_dir,
        {"sft_v1": "completed", "old": "completed", "new": "completed"},
    )

    kinds = [s.kind for s in signals]
    assert kinds == ["finished_while_away"]
    assert "runs/sft_v1 → completed" in signals[0].text
    # "old" (already seen terminal) and "new" (never seen — first startup
    # must not announce ancient runs) stay silent.


def test_finished_while_away_is_pure_and_repeatable(project_dir: Path) -> None:
    """The diff does not consume the baseline itself — the caller decides
    when to record, so /clear and /resume can re-inject the same list."""
    record_seen_states(project_dir, {"sft_v1": "running"})
    states = {"sft_v1": "completed"}

    first = finished_while_away_signals(project_dir, states)
    second = finished_while_away_signals(project_dir, states)
    assert [s.text for s in first] == [s.text for s in second]

    record_seen_states(project_dir, states)
    assert finished_while_away_signals(project_dir, states) == []


def test_launch_baseline_from_subprocess_manager(
    project_dir: Path, monkeypatch,
) -> None:
    """SubprocessManager.start records a durable 'running' baseline, so a
    job submitted and finished entirely between watcher scans still
    produces a finished-while-closed signal on the next open."""
    from types import SimpleNamespace

    from lqh.subprocess_manager import SubprocessManager

    monkeypatch.setattr(
        "lqh.subprocess_manager.subprocess.Popen",
        lambda *a, **k: SimpleNamespace(pid=12345),
    )
    run_dir = project_dir / "runs" / "sft_v1"
    SubprocessManager().start(run_dir, {"type": "sft"}, project_dir=project_dir)

    assert load_seen_states(project_dir) == {"sft_v1": "running"}

    # The job completes while the CLI is closed…
    (run_dir / "progress.jsonl").write_text(
        json.dumps({"status": "completed"}) + "\n"
    )
    states = observe_run_states(project_dir)
    signals = finished_while_away_signals(project_dir, states)
    assert [s.kind for s in signals] == ["finished_while_away"]


def test_orphan_submit_intent_is_signaled(project_dir: Path) -> None:
    signals = collect_signals(
        project_dir,
        snapshot=None,
        snapshot_fresh=True,
        run_states={"lost": "submit_fate_unknown"},
    )

    assert [s.kind for s in signals] == ["submit_fate_unknown"]
    assert "billing-relevant" in signals[0].text


def test_real_orphan_has_only_submit_intent(project_dir: Path) -> None:
    """The response-loss state on disk is a run dir containing ONLY
    submit_intent.json — cloud submission writes config.json only after
    acceptance. The scanner must not require config.json."""
    _make_run(
        project_dir,
        "lost",
        write_config=False,
        submit_intent={"idempotency_key": "k"},
    )
    (project_dir / "runs" / "junk").mkdir()  # truly empty dir → ignored

    states = observe_run_states(project_dir)

    assert states == {"lost": "submit_fate_unknown"}


def test_failed_refresh_is_signaled(project_dir: Path) -> None:
    signals = collect_signals(
        project_dir,
        snapshot=None,
        snapshot_fresh=True,
        run_states={"sft_v1": "running"},
        jobs_refreshed=False,
    )

    kinds = [s.kind for s in signals]
    assert kinds[0] == "refresh_failed"
    assert "possibly-stale" in signals[0].text
    assert "running" in kinds


def test_unknown_states_are_signaled(project_dir: Path) -> None:
    signals = collect_signals(
        project_dir,
        snapshot=None,
        snapshot_fresh=True,
        run_states={"weird": "unknown"},
    )

    assert [s.kind for s in signals] == ["state_unknown"]
    assert "runs/weird" in signals[0].text


def test_spec_drift_against_cloud_snapshot(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# new spec\n")
    snapshot = {
        "schema_version": 1,
        "fetched_at": "2026-07-15T00:00:00+00:00",
        "snapshot": {"spec_sha256": "0" * 64},
    }

    signals = collect_signals(
        project_dir, snapshot=snapshot, snapshot_fresh=True, run_states={}
    )

    assert [s.kind for s in signals] == ["spec_drift"]


def test_no_spec_drift_when_hashes_match(project_dir: Path) -> None:
    from lqh.project_meta import compute_spec_sha256

    (project_dir / "SPEC.md").write_text("# same spec\n")
    snapshot = {
        "schema_version": 1,
        "snapshot": {"spec_sha256": compute_spec_sha256(project_dir)},
    }

    signals = collect_signals(
        project_dir, snapshot=snapshot, snapshot_fresh=True, run_states={}
    )

    assert signals == []


def test_stale_snapshot_is_signaled(project_dir: Path) -> None:
    snapshot = {
        "schema_version": 1,
        "fetched_at": "2026-07-15T18:02:00+00:00",
        "snapshot": {},
    }

    signals = collect_signals(
        project_dir, snapshot=snapshot, snapshot_fresh=False, run_states={}
    )

    assert [s.kind for s in signals] == ["snapshot_stale"]
    assert "2026-07-15T18:02:00+00:00" in signals[0].text


def test_unavailable_snapshot_without_cache_is_signaled(
    project_dir: Path,
) -> None:
    """Offline/auth failure with no cache must warn — silence would let
    the agent assume there is no cloud state at all."""
    signals = collect_signals(
        project_dir, snapshot=None, snapshot_fresh=False, run_states={}
    )

    assert [s.kind for s in signals] == ["snapshot_unavailable"]
    assert "no cached snapshot" in signals[0].text


def test_manifest_spec_drift_is_signaled(project_dir: Path) -> None:
    (project_dir / "SPEC.md").write_text("# current spec\n")
    ds = project_dir / "datasets" / "train_v1"
    ds.mkdir(parents=True)
    (ds / "manifest.json").write_text(json.dumps({"spec_sha256": "f" * 64}))
    from lqh.project_meta import compute_spec_sha256

    current = project_dir / "datasets" / "train_v2"
    current.mkdir(parents=True)
    (current / "manifest.json").write_text(
        json.dumps({"spec_sha256": compute_spec_sha256(project_dir)})
    )

    signals = collect_signals(
        project_dir, snapshot=None, snapshot_fresh=True, run_states={}
    )

    assert [s.kind for s in signals] == ["artifact_spec_drift"]
    assert "train_v1" in signals[0].text
    assert "train_v2" not in signals[0].text


def test_partial_snapshot_refresh_is_signaled(project_dir: Path) -> None:
    """A fresh core snapshot with stale enrichment sections is an
    unknown-unknown — the agent must be told the artifact/deployment
    lists may be old."""
    snapshot = {
        "schema_version": 1,
        "fetched_at": "2026-07-17T00:00:00+00:00",
        "snapshot": {"jobs": []},
        "stale_sections": ["artifacts"],
    }

    signals = collect_signals(
        project_dir, snapshot=snapshot, snapshot_fresh=True, run_states={}
    )

    assert [s.kind for s in signals] == ["snapshot_partial"]
    assert "artifacts" in signals[0].text


def test_quiet_project_has_no_signals(project_dir: Path) -> None:
    signals = collect_signals(
        project_dir, snapshot=None, snapshot_fresh=True, run_states={}
    )
    assert signals == []
    assert format_signal_block(signals) is None


def test_format_signal_block_includes_investigation_instruction() -> None:
    block = format_signal_block([Signal("running", "1 job still running")])

    assert block is not None
    assert block.startswith("⚡ Attention signals")
    assert "- 1 job still running" in block
    assert "Investigate with your tools" in block


# ---------------------------------------------------------------------------
# End-to-end: signals land in the agent's ephemeral context
# ---------------------------------------------------------------------------


async def test_prepare_context_injects_signals_and_records_seen(
    project_dir: Path,
) -> None:
    from lqh.agent import Agent
    from lqh.session import Session

    (project_dir / "SPEC.md").write_text("# spec\n")
    record_seen_states(project_dir, {"sft_v1": "running"})
    _make_run(project_dir, "sft_v1", progress=[{"status": "completed"}])

    agent = Agent(project_dir, Session.create(project_dir))
    await agent.prepare_context()

    contents = [m["content"] for m in agent.context_messages]
    signal_blocks = [c for c in contents if c.startswith("⚡ Attention signals")]
    assert len(signal_blocks) == 1
    assert "runs/sft_v1 → completed" in signal_blocks[0]

    # Observed states were recorded: a second open stays quiet.
    assert load_seen_states(project_dir)["sft_v1"] == "completed"
    await agent.prepare_context()
    contents = [m["content"] for m in agent.context_messages]
    assert not any(c.startswith("⚡ Attention signals") for c in contents)