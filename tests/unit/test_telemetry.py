from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lqh.config import LqhConfig, load_config, save_config, save_credentials, telemetry_enabled, update_config
from lqh.telemetry import QUEUE_MAX_BYTES, TelemetryClient, ensure_project_identity, notice_needed


@pytest.fixture(autouse=True)
def authenticated(monkeypatch):
    monkeypatch.setattr("lqh.auth.get_token", lambda: "account-a-token")


def test_config_and_environment_precedence(monkeypatch):
    assert telemetry_enabled(LqhConfig(telemetry_enabled=True)) is True
    assert telemetry_enabled(LqhConfig(telemetry_enabled=False)) is False
    monkeypatch.setenv("LQH_TELEMETRY", "0")
    assert telemetry_enabled(LqhConfig(telemetry_enabled=True)) is False
    monkeypatch.setenv("LQH_TELEMETRY", "1")
    assert telemetry_enabled(LqhConfig(telemetry_enabled=False)) is True


def test_config_write_is_private_and_stale_writer_cannot_reenable_telemetry(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    save_config(LqhConfig(api_key="secret"))
    stale = load_config()

    def opt_out(config: LqhConfig) -> None:
        config.telemetry_enabled = False
        config.telemetry_consent_epoch += 1

    update_config(opt_out)
    stale.default_compute = "cloud"
    save_config(stale)

    persisted = load_config()
    assert persisted.telemetry_enabled is False
    assert persisted.telemetry_consent_epoch == 1
    assert persisted.default_compute == "cloud"
    assert stat.S_IMODE((tmp_path / ".lqh" / "config.json").stat().st_mode) == 0o600


def test_config_field_updates_do_not_clobber_each_other(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    save_config(LqhConfig(api_key="secret"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(
            lambda update: update_config(update),
            [
                lambda config: setattr(config, "default_compute", "ssh:gpu"),
                lambda config: setattr(config, "api_base_url", "https://staging.example/v1"),
            ],
        ))

    persisted = load_config()
    assert persisted.api_key == "secret"
    assert persisted.default_compute == "ssh:gpu"
    assert persisted.api_base_url == "https://staging.example/v1"


def test_project_identity_new_preexisting_reopened(tmp_path: Path):
    fresh = tmp_path / "fresh"; fresh.mkdir()
    project_id, state = ensure_project_identity(fresh)
    assert state == "new"
    again, state = ensure_project_identity(fresh)
    assert again == project_id and state == "reopened"
    old = tmp_path / "old"; old.mkdir(); (old / "SPEC.md").write_text("spec")
    _, state = ensure_project_identity(old)
    assert state == "pre_existing"


def test_project_identity_is_atomic_across_concurrent_clients(tmp_path: Path):
    project = tmp_path / "concurrent"; project.mkdir()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: ensure_project_identity(project), range(24)))
    assert len({project_id for project_id, _state in results}) == 1
    assert sum(state == "new" for _project_id, state in results) == 1


def test_project_identity_failure_disables_telemetry_without_blocking_startup(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "read-only-project"; project.mkdir()
    monkeypatch.setattr("lqh.telemetry.ensure_project_identity", lambda _path: (_ for _ in ()).throw(PermissionError("read-only")))

    client = TelemetryClient(project)

    assert client.enabled is False
    assert client.project_id is None
    assert client.correlation_project_id() is None


def test_project_identity_does_not_require_chmod_support(monkeypatch, tmp_path: Path):
    project = tmp_path / "non-posix-project"; project.mkdir()
    monkeypatch.setattr(
        "lqh.telemetry.os.chmod",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )

    project_id, state = ensure_project_identity(project)
    reopened_id, reopened_state = ensure_project_identity(project)

    assert state == "new"
    assert reopened_state == "reopened"
    assert reopened_id == project_id


def test_session_starts_after_first_login_without_losing_new_project(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    token: list[str | None] = [None]
    monkeypatch.setattr("lqh.auth.get_token", lambda: token[0])
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    assert client.start_session() is False
    assert not (project / ".lqh" / "project.json").exists()

    token[0] = "first-login-token"
    assert client.start_session() is True
    assert client.start_session() is False
    events = [json.loads(line) for line in client.queue_path.read_text().splitlines()]
    assert [event["event_name"] for event in events] == ["session_started", "project_opened"]
    assert events[1]["metadata"]["project_state"] == "new"


def test_concurrent_clients_join_project_workflow_state(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    first = TelemetryClient(project)
    second = TelemetryClient(project)
    first.record_user_turn()
    second.record_user_turn()
    assert first.spec_workflow_id == second.spec_workflow_id
    events = [json.loads(line) for line in first.queue_path.read_text().splitlines()]
    assert sum(event["event_name"] == "spec_capture_started" for event in events) == 1
    first.maybe_spec_completed("SPEC.md", True)
    second.record_agent_turn()
    state = json.loads((project / ".lqh" / "project.json").read_text())
    assert "spec_capture" not in state


def test_queue_bounds_and_opt_out_clear(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    client = TelemetryClient(tmp_path / "project")
    client.queue_path.parent.mkdir(parents=True, exist_ok=True)
    # Use valid bounded events; the queue drops oldest complete JSONL records.
    for _ in range(12_000):
        client.event("agent_turn")
    assert client.queue_path.stat().st_size <= QUEUE_MAX_BYTES
    assert all(json.loads(line)["event_name"] == "agent_turn" for line in client.queue_path.read_text().splitlines())
    client.clear_queue()
    assert not client.queue_path.exists()


def test_queue_trim_preserves_session_and_project_anchors(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("lqh.telemetry.QUEUE_MAX_BYTES", 4_000)
    monkeypatch.setattr("lqh.telemetry.QUEUE_TRIM_TARGET_BYTES", 3_000)
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.start_session()
    for _ in range(100):
        client.event("agent_turn")

    names = [json.loads(line)["event_name"] for line in client.queue_path.read_text().splitlines()]
    assert "session_started" in names
    assert "project_opened" in names
    assert "agent_turn" in names


def test_notice_failure_is_best_effort(monkeypatch):
    monkeypatch.setattr("lqh.telemetry.config_dir", lambda: (_ for _ in ()).throw(FileNotFoundError("unavailable")))
    assert notice_needed() is False


async def test_zero_success_pipeline_emits_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; (project / "data_gen").mkdir(parents=True)
    (project / "data_gen" / "pipeline.py").write_text("# test\n")
    telemetry = MagicMock(active_seconds=0.0, consent_epoch=1)
    telemetry.state_snapshot.return_value = (True, 1, 0.0, "account-a")
    telemetry.consent_active.return_value = True
    async def run_deferred(callback, *args):
        return callback(*args)
    telemetry.run_deferred = AsyncMock(side_effect=run_deferred)
    monkeypatch.setattr("lqh.telemetry.active_telemetry", lambda: telemetry)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "token")
    monkeypatch.setattr("lqh.client.create_client", lambda *_args: object())

    async def failed_pipeline(**_kwargs):
        return SimpleNamespace(succeeded=0, failed=2, total=2)

    monkeypatch.setattr("lqh.engine.run_pipeline", failed_pipeline)
    from lqh.tools.handlers import _execute_pipeline

    result = await _execute_pipeline(
        project, "data_gen/pipeline.py", 2, "empty", None,
    )

    assert result.content.startswith("❌ Pipeline failed")
    event_names = [call.args[0] for call in telemetry.event.call_args_list]
    assert event_names == ["data_generation_started", "data_generation_failed"]
    telemetry.record_generation_succeeded.assert_not_called()


async def test_deferred_telemetry_work_is_ordered(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    seen: list[int] = []
    client.defer(seen.append, 1)
    client.defer(seen.append, 2)

    await client.run_deferred(seen.append, 3)

    assert seen == [1, 2, 3]


async def test_deferred_telemetry_timeout_detaches_but_preserves_order(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    entered = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    def block_worker() -> None:
        entered.set()
        release.wait()

    client.defer(block_worker)
    assert entered.wait(timeout=1)
    started = time.monotonic()
    try:
        await client.run_deferred(seen.append, "after-blocker")
        assert time.monotonic() - started < 0.75
        assert seen == []
    finally:
        release.set()
    await client.run_deferred(lambda: None, timeout=1)
    assert seen == ["after-blocker"]


async def test_deferred_privacy_barrier_waits_for_full_queue(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    entered = threading.Event()
    release = threading.Event()
    barrier_ran = threading.Event()

    def block_worker() -> None:
        entered.set()
        release.wait()

    client.defer(block_worker)
    assert entered.wait(timeout=1)
    for _ in range(client._work_queue.maxsize):
        assert client.defer(lambda: None)

    barrier = asyncio.create_task(
        client.run_deferred(barrier_ran.set, timeout=None),
    )
    await asyncio.sleep(0.03)
    assert not barrier.done()
    assert not barrier_ran.is_set()

    release.set()
    await asyncio.wait_for(barrier, timeout=2)
    assert barrier_ran.is_set()


async def test_cached_consent_check_never_resets_measurement_state(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    epoch = client.consent_epoch
    client.active_seconds = 42
    config = load_config()
    config.telemetry_enabled = False
    config.telemetry_consent_epoch += 1
    save_config(config)

    assert client.cached_consent_active(epoch) is True
    assert client.active_seconds == 42
    assert await client.run_deferred(client.consent_active, epoch) is False
    assert client.active_seconds == 0


def test_unreadable_consent_fails_closed(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.active_seconds = 42
    monkeypatch.setattr(
        "lqh.telemetry.load_config",
        lambda: (_ for _ in ()).throw(json.JSONDecodeError("partial", "{", 1)),
    )

    assert client.is_enabled() is False
    assert client.enabled is False
    assert client.active_seconds == 0


def test_exact_turns_and_spec_lifecycle(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.record_user_turn("message")
    client.record_user_turn("ask_user_answer")
    client.record_agent_turn(); client.record_agent_turn()
    assert client.user_turns == 2 and client.agent_turns == 2
    workflow = client.spec_workflow_id
    assert workflow
    client.maybe_spec_completed("nested/SPEC.md", True)
    assert client.spec_workflow_id == workflow
    client.maybe_spec_completed("SPEC.md", True)
    assert client.spec_workflow_id is None
    names = [json.loads(x)["event_name"] for x in client.queue_path.read_text().splitlines()]
    assert names.count("user_turn") == 2
    assert names.count("agent_turn") == 2
    assert names.count("spec_capture_started") == 1
    assert names.count("spec_capture_completed") == 1


def test_network_failure_keeps_queue(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.event("agent_turn")
    before = client.queue_path.read_text()
    monkeypatch.setattr("lqh.auth.get_token", lambda: "token")
    def fail(*_args, **_kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr("httpx.post", fail)
    try:
        client._flush_sync()
    except RuntimeError:
        pass
    assert client.queue_path.read_text() == before


def test_legacy_custom_api_key_is_never_used_for_telemetry(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = LqhConfig(api_key="third-party-secret", api_base_url="https://custom.example/v1")
    save_config(config)
    monkeypatch.setattr("lqh.auth.get_token", lambda: config.api_key)
    client = TelemetryClient(tmp_path / "project")
    client.event("agent_turn")
    post = MagicMock()
    monkeypatch.setattr("httpx.post", post)
    client._flush_sync()

    assert client.account_key is None
    assert not client.queue_path.exists()
    post.assert_not_called()


def test_device_credential_stays_bound_to_lqh_control_plane(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    save_config(LqhConfig(api_base_url="https://custom.example/v1"))
    save_credentials("lqh-device-token", "user-123")
    client = TelemetryClient(tmp_path / "project")
    client.event("agent_turn")
    sent: dict[str, object] = {}

    class Response:
        status_code = 202

    def accept(url, **kwargs):
        sent["url"] = url
        sent["authorization"] = kwargs["headers"]["Authorization"]
        return Response()

    monkeypatch.setattr("httpx.post", accept)
    client._flush_sync()

    assert sent == {
        "url": "https://api.lqh.ai/v1/telemetry/events",
        "authorization": "Bearer lqh-device-token",
    }


def test_device_credential_is_not_sent_to_untrusted_base_url(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LQH_BASE_URL", "https://third-party.example/v1")
    save_credentials("lqh-device-token", "user-123")
    post = MagicMock()
    get = MagicMock()
    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr("httpx.get", get)

    client = TelemetryClient(tmp_path / "project")
    client.event("agent_turn")
    client._flush_sync()

    assert client.account_key is None
    assert not client.queue_path.exists()
    post.assert_not_called()
    get.assert_not_called()


def test_debug_alias_does_not_create_or_retry_attributed_telemetry(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LQH_DEBUG_API_KEY", "debug-secret")
    client = TelemetryClient(tmp_path / "project")
    client.event("agent_turn")
    post = MagicMock()
    monkeypatch.setattr("httpx.post", post)

    client._flush_sync()

    assert client.account_key is None
    assert not client.queue_path.exists()
    post.assert_not_called()


@pytest.mark.parametrize("message,dropped", [
    ("telemetry daily admission limit reached", True),
    ("rate limit exceeded", False),
])
def test_only_daily_quota_429_drops_batch(monkeypatch, tmp_path: Path, message: str, dropped: bool):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.event("agent_turn")

    class Response:
        status_code = 429

        @staticmethod
        def json():
            return {"error": {"message": message}}

    monkeypatch.setattr("httpx.post", lambda *_args, **_kwargs: Response())
    client._flush_sync()
    assert (not client.queue_path.read_text()) is dropped


def test_readiness_survives_reopen_and_completes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    first = TelemetryClient(project)
    first.record_generation_attempt()
    dataset = project / "datasets" / "generated"; dataset.mkdir(parents=True)
    first.record_generation_succeeded(dataset)
    workflow_id = first.readiness_workflow_id
    reopened = TelemetryClient(project)
    assert reopened.readiness_workflow_id == workflow_id
    assert reopened.generation_attempts == 1
    reopened.complete_readiness({"dataset": "datasets/generated"})
    state = json.loads((project / ".lqh" / "project.json").read_text())
    assert "pipeline_readiness" not in state


def test_readiness_does_not_complete_before_generation_attempt(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.start_readiness()
    workflow_id = client.readiness_workflow_id

    client.complete_readiness()

    assert client.readiness_workflow_id == workflow_id
    events = [json.loads(line) for line in client.queue_path.read_text().splitlines()]
    assert not any(event["event_name"] == "pipeline_readiness_completed" for event in events)


def test_readiness_does_not_complete_after_failed_generation(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.record_generation_attempt()

    client.complete_readiness({"dataset": "datasets/generated"})

    assert client.readiness_workflow_id is not None


def test_readiness_rejects_unrelated_fresh_data(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    generated = project / "datasets" / "generated"; generated.mkdir(parents=True)
    unrelated = project / "datasets" / "unrelated"; unrelated.mkdir(parents=True)
    client = TelemetryClient(project)
    client.record_generation_attempt()
    client.record_generation_succeeded(generated)

    client.complete_readiness({"dataset": "datasets/unrelated"})

    assert client.readiness_workflow_id is not None


def test_explicit_spec_capture_starts_without_counting_user_turn(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)

    assert client.start_spec_capture() is True
    client.record_agent_turn()
    client.maybe_spec_completed("SPEC.md", True)

    events = [json.loads(line) for line in client.queue_path.read_text().splitlines()]
    completed = next(event for event in events if event["event_name"] == "spec_capture_completed")
    assert completed["metadata"]["user_turns"] == 0
    assert completed["metadata"]["agent_turns"] == 1


def test_spec_capture_survives_reopen(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    first = TelemetryClient(project)
    first.record_user_turn()
    workflow_id = first.spec_workflow_id
    first.end_session()
    reopened = TelemetryClient(project)
    assert reopened.spec_workflow_id == workflow_id
    assert reopened.spec_user_turns == 1
    reopened.record_agent_turn()
    reopened.maybe_spec_completed("SPEC.md", True)
    state = json.loads((project / ".lqh" / "project.json").read_text())
    assert "spec_capture" not in state


def test_queue_is_account_bound(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    token = ["account-a-token"]
    monkeypatch.setattr("lqh.auth.get_token", lambda: token[0])
    project = tmp_path / "project"; project.mkdir()
    account_a = TelemetryClient(project)
    account_a.event("agent_turn")
    a_path = account_a.queue_path
    token[0] = "account-b-token"
    account_a.event("agent_turn")
    assert len(a_path.read_text().splitlines()) == 1
    account_b = TelemetryClient(project)
    account_b.event("agent_turn")
    assert account_b.queue_path != a_path
    assert len(account_b.queue_path.read_text().splitlines()) == 1


def test_flush_removes_only_sent_records_when_new_event_arrives(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.event("agent_turn")

    class Response:
        status_code = 202

    def append_during_send(*_args, **_kwargs):
        client.event("user_turn", {"source": "message"})
        return Response()

    monkeypatch.setattr("httpx.post", append_during_send)
    client._flush_sync()
    queued = [json.loads(line) for line in client.queue_path.read_text().splitlines()]
    assert [event["event_name"] for event in queued] == ["user_turn"]


def test_flush_drains_multiple_batches(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    for _ in range(250):
        client.event("agent_turn")

    calls: list[int] = []

    class Response:
        status_code = 202

    def accept(*_args, **kwargs):
        calls.append(len(kwargs["json"]["events"]))
        return Response()

    monkeypatch.setattr("httpx.post", accept)
    client._flush_sync()
    assert calls == [100, 100, 50]
    assert client.queue_path.read_text() == ""


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_telemetry_files_are_owner_only(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.record_user_turn()
    assert client.queue_path.stat().st_mode & 0o777 == 0o600
    assert (project / ".lqh" / "project.json").stat().st_mode & 0o777 == 0o600
    assert (project / ".lqh").stat().st_mode & 0o777 == 0o700
    assert client.queue_path.parent.stat().st_mode & 0o777 == 0o700
    save_credentials("bearer-token", "user-123")
    credentials = tmp_path / ".config" / "lqh" / "credentials"
    assert credentials.stat().st_mode & 0o777 == 0o600
    assert credentials.parent.stat().st_mode & 0o777 == 0o700
    assert list(credentials.parent.glob(".credentials.*")) == []


def test_wall_clock_regression_clamps_session_duration(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.start_session()
    client.started_wall = time.time() + 60

    client.end_session()

    events = [json.loads(line) for line in client.queue_path.read_text().splitlines()]
    ended = next(event for event in events if event["event_name"] == "session_ended")
    assert ended["metadata"]["wall_duration_ms"] == 0


def test_opt_out_creates_no_identity_or_cloud_correlation(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LQH_TELEMETRY", "0")
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    assert client.project_id is None
    assert client.correlation_project_id() is None
    assert not (project / ".lqh" / "project.json").exists()


def test_opt_out_resets_measurements_and_reenable_starts_clean(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.start_session()
    client.active_seconds = 120
    client.user_turns = 4
    client.agent_turns = 7
    client.set_enabled(False)
    assert client.active_seconds == 0
    assert client.user_turns == 0 and client.agent_turns == 0
    assert not client.queue_path.exists()

    client.record_user_turn()
    assert client.active_seconds == 0 and client.user_turns == 0
    client.set_enabled(True)
    client.end_session()
    events = [json.loads(line) for line in client.queue_path.read_text().splitlines()]
    ended = next(event for event in events if event["event_name"] == "session_ended")
    assert ended["metadata"]["active_duration_ms"] < 1_000


def test_opt_out_epoch_stops_other_running_client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    controller = TelemetryClient(project)
    other = TelemetryClient(project)
    other.start_session()
    other.active_seconds = 120

    config = load_config()
    config.telemetry_enabled = False
    config.telemetry_consent_epoch += 1
    save_config(config)
    controller.set_enabled(False)

    other.record_user_turn()
    assert other.enabled is False
    assert other.active_seconds == 0 and other.user_turns == 0
    assert not other.queue_path.exists()

    config = load_config()
    config.telemetry_enabled = True
    config.telemetry_consent_epoch += 1
    save_config(config)
    other.record_user_turn()
    events = [json.loads(line) for line in other.queue_path.read_text().splitlines()]
    assert [event["event_name"] for event in events[:2]] == ["session_started", "project_opened"]
    assert events[-1]["event_name"] == "spec_capture_started"
    assert other.active_seconds < 1


def test_consent_epoch_discards_queue_left_by_crash_before_opt_out_barrier(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    before_opt_out = TelemetryClient(project)
    before_opt_out.event("agent_turn")

    # Simulate a process dying after /telemetry off persisted the new epoch,
    # but before set_enabled(False) acquired its locks and cleared the queue.
    config = load_config()
    config.telemetry_enabled = False
    config.telemetry_consent_epoch += 1
    save_config(config)

    # A later re-enable advances the epoch again. The abandoned epoch-zero
    # record must be discarded locally, while a new record is sent normally.
    config.telemetry_enabled = True
    config.telemetry_consent_epoch += 1
    save_config(config)
    resumed = TelemetryClient(project)
    resumed.event("agent_turn")
    submitted: list[dict] = []

    class Response:
        status_code = 202

    def accept(*_args, **kwargs):
        submitted.extend(kwargs["json"]["events"])
        return Response()

    monkeypatch.setattr("httpx.post", accept)
    resumed._flush_sync()

    assert [event["event_name"] for event in submitted] == ["agent_turn"]
    assert all("_consent_epoch" not in event for event in submitted)
    assert resumed.queue_path.read_text() == ""


def test_account_refresh_starts_a_new_session(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    token = ["account-a-token"]
    monkeypatch.setattr("lqh.auth.get_token", lambda: token[0])
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    assert client.start_session()
    old_queue = client.queue_path

    token[0] = "account-b-token"
    client.refresh_account_binding()
    assert client.start_session()
    assert client.queue_path != old_queue
    events = [json.loads(line) for line in client.queue_path.read_text().splitlines()]
    assert [event["event_name"] for event in events] == ["session_started", "project_opened"]


def test_heartbeat_only_queues_when_activity_advances(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.start_session()
    assert client.heartbeat() is False
    client.active_seconds = 1.25
    assert client.heartbeat() is True
    assert client.heartbeat() is False
    names = [json.loads(line)["event_name"] for line in client.queue_path.read_text().splitlines()]
    assert names.count("session_heartbeat") == 1


def test_queue_identity_survives_token_rotation_for_same_account(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    token = ["token-a"]
    monkeypatch.setattr("lqh.auth.get_token", lambda: token[0])
    save_credentials(token[0], "user-123")
    project = tmp_path / "project"; project.mkdir()
    client = TelemetryClient(project)
    client.event("agent_turn")
    original_path = client.queue_path

    token[0] = "token-b"
    save_credentials(token[0], "user-123")
    client.event("agent_turn")
    assert client.queue_path == original_path
    assert len(original_path.read_text().splitlines()) == 2
