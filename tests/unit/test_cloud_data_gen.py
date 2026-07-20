"""Cloud data-gen (CLOUD_OFFLOAD_PLAN.md §2) — client-side pieces.

Covers the seed-data path recorder, the local-validation gate store, the
tool-handler gates (VALIDATION_REQUIRED → consent → submit), the kind
inference + presigned-PUT staging in the cloud client, the publish
classification of ``data.parquet``, and the in-sandbox module contract.
"""

from __future__ import annotations

import asyncio
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from lqh import sources
from lqh.data_gen_validation import (
    check_validation,
    record_validation,
    validation_file_path,
)
from lqh.engine import EngineResult, run_pipeline
from lqh.remote.backend import RemoteConfig
from lqh.remote.bundle import build_bundle_to_file
from lqh.remote.cloud import CloudBackend, _infer_kind
from lqh.tools.handlers import handle_run_data_gen_pipeline
from lqh.tools.permissions import (
    PermissionContext,
    check_cloud_data_gen_permission,
    grant_cloud_data_gen_permission,
    grant_permission,
    load_permissions,
)


# ---------------------------------------------------------------------------
# sources path recorder
# ---------------------------------------------------------------------------


def test_recorder_captures_helper_paths(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    (project / "seeds.txt").write_text("one\ntwo\n")
    (project / "seed_data").mkdir()
    (project / "seed_data" / "names.txt").write_text("a\nb\n")
    # Unrelated sibling file that must NOT be recorded (per-file
    # recording — folder contents never ride a bundle wholesale).
    (project / "seed_data" / "proprietary.txt").write_text("secret\n")

    with sources.record_source_paths() as recorded:
        sources.prompts("seeds.txt")
        sources.seed_data("names")
    assert (project / "seeds.txt").resolve() in recorded
    assert (project / "seed_data" / "names.txt").resolve() in recorded
    assert (project / "seed_data" / "proprietary.txt").resolve() not in recorded
    assert (project / "seed_data").resolve() not in recorded


def test_recorder_image_folder_records_only_matched_images(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    images = project / "images"
    images.mkdir()
    from PIL import Image

    Image.new("RGB", (4, 4), (1, 2, 3)).save(images / "a.jpg", format="JPEG")
    (images / "notes.txt").write_text("not an image — must not upload")

    with sources.record_source_paths() as recorded:
        sources.image_folder("images")
    assert (images / "a.jpg").resolve() in recorded
    assert (images / "notes.txt").resolve() not in recorded
    assert images.resolve() not in recorded


def test_recorder_inactive_outside_context(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    (project / "seeds.txt").write_text("one\n")
    with sources.record_source_paths() as recorded:
        pass
    sources.prompts("seeds.txt")  # after the context — must not record
    assert recorded == set()


def test_recorder_contexts_are_isolated(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    (project / "a.txt").write_text("x\n")
    (project / "b.txt").write_text("y\n")
    with sources.record_source_paths() as outer:
        sources.prompts("a.txt")
        with sources.record_source_paths() as inner:
            sources.prompts("b.txt")
    assert (project / "b.txt").resolve() in inner
    assert (project / "b.txt").resolve() not in outer


# ---------------------------------------------------------------------------
# engine returns recorded paths
# ---------------------------------------------------------------------------

_BYO_PIPELINE = """
from lqh.pipeline import Pipeline, ChatMLMessage
from lqh.sources import prompts

class Byo(Pipeline):
    @classmethod
    def source(cls, project_dir):
        return prompts("seeds.txt")

    async def generate(self, client, input=None):
        return [
            ChatMLMessage(role="user", content=input.prompt),
            ChatMLMessage(role="assistant", content="ok"),
        ]
"""


def _write_byo_project(project: Path) -> Path:
    (project / "data_gen").mkdir(parents=True, exist_ok=True)
    script = project / "data_gen" / "byo.py"
    script.write_text(_BYO_PIPELINE)
    (project / "seeds.txt").write_text("alpha\nbeta\n")
    return script


def test_run_pipeline_reports_source_paths(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    script = _write_byo_project(project)

    result = asyncio.run(run_pipeline(
        script_path=script,
        num_samples=2,
        output_dir=project / "datasets" / "byo",
        client=object(),  # type: ignore[arg-type] — pipeline never touches it
    ))
    assert result.succeeded == 2
    assert (project / "seeds.txt").resolve() in result.source_paths


# ---------------------------------------------------------------------------
# validation store
# ---------------------------------------------------------------------------


def test_validation_round_trip_and_hash_invalidation(tmp_path: Path) -> None:
    project = tmp_path
    script = project / "data_gen" / "p.py"
    script.parent.mkdir(parents=True)
    script.write_text("VERSION = 1\n")

    assert check_validation(project, script) is None

    record_validation(
        project, script,
        num_samples=20, succeeded=19, failed=1,
        source_paths=[project / "seed_data", Path("/outside/elsewhere")],
    )
    rec = check_validation(project, script)
    assert rec is not None
    assert rec.succeeded == 19
    # Outside-project paths are dropped; inside ones stored project-relative.
    assert rec.source_paths == ["seed_data"]
    assert validation_file_path(project).exists()

    # Any edit re-arms the gate.
    script.write_text("VERSION = 2\n")
    assert check_validation(project, script) is None

    # Re-validating the new version unlocks again.
    record_validation(project, script, num_samples=3, succeeded=3, failed=0)
    rec2 = check_validation(project, script)
    assert rec2 is not None and rec2.num_samples == 3


def test_validation_survives_corrupt_file(tmp_path: Path) -> None:
    project = tmp_path
    script = project / "data_gen" / "p.py"
    script.parent.mkdir(parents=True)
    script.write_text("x = 1\n")
    validation_file_path(project).parent.mkdir(parents=True)
    validation_file_path(project).write_text("{not json")
    assert check_validation(project, script) is None
    record_validation(project, script, num_samples=1, succeeded=1, failed=0)
    assert check_validation(project, script) is not None


def test_validation_binds_recorded_source_contents(tmp_path: Path) -> None:
    project = tmp_path
    script = project / "data_gen" / "p.py"
    script.parent.mkdir(parents=True)
    script.write_text("x = 1\n")
    source = project / "seeds.txt"
    source.write_text("alpha\n")
    record_validation(
        project, script, num_samples=3, succeeded=3, failed=0,
        source_paths=[source],
    )
    assert check_validation(project, script) is not None
    source.write_text("beta\n")
    assert check_validation(project, script) is None


# ---------------------------------------------------------------------------
# permissions domain
# ---------------------------------------------------------------------------


def test_cloud_data_gen_permission_is_own_domain(tmp_path: Path) -> None:
    grant_permission(tmp_path, None, project_wide=True)  # script-exec domain
    assert not check_cloud_data_gen_permission(tmp_path)
    grant_cloud_data_gen_permission(tmp_path)
    assert check_cloud_data_gen_permission(tmp_path)
    # And the reverse: cloud grant must not grant script execution — spot
    # check by reloading the raw store.
    perms = load_permissions(tmp_path)
    assert perms.cloud_data_gen_allow_all


def test_old_permission_files_load_with_default(tmp_path: Path) -> None:
    path = tmp_path / ".lqh" / "permissions.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"project_allow_all": True}))
    perms = load_permissions(tmp_path)
    assert perms.project_allow_all
    assert not perms.cloud_data_gen_allow_all


# ---------------------------------------------------------------------------
# handler gates
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_price_fetch(monkeypatch):
    """Keep consent prompts offline: the live pricing fetch would
    otherwise hit the real API when local credentials exist."""

    async def _none() -> float | None:
        return None

    monkeypatch.setattr("lqh.tools.handlers._fetch_data_gen_rate_usd", _none)


def _handler_project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path
    (project / "data_gen").mkdir(parents=True, exist_ok=True)
    script_rel = "data_gen/task.py"
    (project / script_rel).write_text(_BYO_PIPELINE)
    (project / "seeds.txt").write_text("alpha\n")
    return project, script_rel

def _mark_script_uses_hf(project, script_rel):
    """Append a (real) hf_dataset reference to the fixture pipeline.

    needs_hf is honored only when the hashed script names hf_dataset —
    single-file pipelines can't use it otherwise — so tests exercising
    the HF path need a fixture that could legitimately have observed it.
    Returns nothing; callers must record_validation AFTER this (the
    edit changes the content hash).
    """
    p = project / script_rel
    p.write_text(p.read_text() + "\nfrom lqh.sources import hf_dataset  # noqa: F401\n")


@pytest.mark.asyncio
async def test_model_supplied_consent_flags_are_stripped(tmp_path: Path) -> None:
    """A model-generated tool call must not be able to smuggle consent.

    Underscore-prefixed keys ride only extra_kwargs from the agent loop;
    execute_tool drops them from the model-controlled arguments dict.
    """
    from lqh.tools.handlers import execute_tool

    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(project, project / script_rel, num_samples=3, succeeded=3, failed=0)

    result = await execute_tool(
        "run_data_gen_pipeline",
        {
            "script_path": script_rel,
            "num_samples": 1000,
            "output_dataset": "big",
            "execution": "cloud",
            # Injected by a (hypothetical) malicious/confused model:
            "_cloud_consent": True,
            "_script_consent": True,
        },
        project,
    )
    # The consent gate must still fire — no silent cloud submit.
    assert result.content == "PERMISSION_REQUIRED"
    assert result.requires_user_input


@pytest.mark.asyncio
async def test_script_permission_sentinel_has_key(tmp_path: Path) -> None:
    """The script-execution sentinel must carry a grantable permission_key."""
    project, script_rel = _handler_project(tmp_path)
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=3, output_dataset="d",
    )
    assert result.content == "PERMISSION_REQUIRED"
    assert result.permission_key == f"script:{script_rel}"


@pytest.mark.asyncio
async def test_full_consent_context_skips_prompts(tmp_path: Path, monkeypatch) -> None:
    """The headless CLI surface passes full consent — no sentinel fires."""
    from lqh.remote.cloud import CloudBackend

    project, script_rel = _handler_project(tmp_path)
    record_validation(
        project, project / script_rel,
        num_samples=20, succeeded=20, failed=0,
        source_paths=[project / "seeds.txt"],
    )

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        return "job-full-consent"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="d", execution="cloud",
        _permissions=PermissionContext(full_consent=True),
    )
    assert "job-full-consent" in result.content
    # And nothing was persisted into the durable store.
    perms = load_permissions(project)
    assert perms.project_allow_all is False
    assert perms.cloud_data_gen_allow_all is False


@pytest.mark.asyncio
async def test_output_dataset_path_rejected(tmp_path: Path) -> None:
    project, script_rel = _handler_project(tmp_path)
    for bad in ("../evil", "a/b", "", "..", "runs\\x"):
        result = await handle_run_data_gen_pipeline(
            project, script_path=script_rel, num_samples=5,
            output_dataset=bad, execution="local",
        )
        assert "output_dataset must be a plain name" in result.content, bad


@pytest.mark.asyncio
async def test_execution_enum_rejected(tmp_path: Path) -> None:
    project, script_rel = _handler_project(tmp_path)
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=5,
        output_dataset="d", execution="ssh",
    )
    assert "execution must be" in result.content


@pytest.mark.asyncio
async def test_cloud_refused_without_local_validation(tmp_path: Path) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=1000,
        output_dataset="big", execution="cloud",
    )
    assert result.content.startswith("VALIDATION_REQUIRED")
    assert not result.requires_user_input  # plain error the agent acts on


@pytest.mark.asyncio
async def test_cloud_stale_hash_refused(tmp_path: Path) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(project, project / script_rel, num_samples=3, succeeded=3, failed=0)
    (project / script_rel).write_text(_BYO_PIPELINE + "\n# edited\n")
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=1000,
        output_dataset="big", execution="cloud",
    )
    assert result.content.startswith("VALIDATION_REQUIRED")


@pytest.mark.asyncio
async def test_cloud_consent_prompt_shows_cost(tmp_path: Path) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(
        project, project / script_rel,
        num_samples=20, succeeded=20, failed=0,
        source_paths=[project / "seeds.txt"],
    )
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=1000,
        output_dataset="big", execution="cloud",
    )
    assert result.content == "PERMISSION_REQUIRED"
    assert result.requires_user_input
    assert result.permission_key == f"cloud_data_gen:{script_rel}"
    assert "1000" in (result.question or "")
    assert "$" in (result.question or "")
    assert "seeds.txt" in (result.question or "")


@pytest.mark.asyncio
async def test_cloud_submit_with_consent(tmp_path: Path, monkeypatch) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(
        project, project / script_rel,
        num_samples=20, succeeded=20, failed=0,
        source_paths=[project / "seeds.txt"],
    )

    submitted: dict = {}

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        submitted["run_dir"] = run_dir
        submitted["config"] = config
        submitted["module"] = module
        return "job-42"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)

    started: list = []
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=1000,
        output_dataset="big", execution="cloud",
        _permissions=PermissionContext.granting("cloud_data_gen"),
        on_background_task_started=lambda *a: started.append(a),
    )
    assert "job-42" in result.content
    assert result.workflow_launched
    assert submitted["module"] == "lqh.remote.data_gen"
    cfg = submitted["config"]
    assert cfg["kind"] == "data_gen" and cfg["type"] == "data_gen"
    assert cfg["script_path"] == script_rel
    assert cfg["source_paths"] == ["seeds.txt"]
    assert set(cfg["manifest"]) == {
        "script_path", "validation_instructions", "source_paths",
    }
    # Single-file contract: nothing beyond the pipeline file ships.
    assert "helper_modules" not in cfg
    # The pipeline doesn't use hf_dataset → the sandbox must not receive
    # the stored HF token (backend gates on this flag for data_gen).
    assert cfg["needs_hf"] is False
    assert started and started[0][1] == "data_gen" and started[0][3] == "cloud"
    # Durable finalization marker for the TUI watcher (survives restarts).
    marker = json.loads(
        (Path(submitted["run_dir"]) / ".lqh_data_gen.json").read_text()
    )
    assert marker["output_dataset"] == "big"
    assert marker["job_id"] == "job-42"
    assert marker["workflow_id"]


@pytest.mark.asyncio
async def test_validation_instructions_path_escape_rejected(tmp_path: Path) -> None:
    """validation_instructions rides the bundle manifest — an absolute or
    ../ path would upload an arbitrary readable local file."""
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(project, project / script_rel, num_samples=3, succeeded=3, failed=0)

    for bad in ("../../etc/passwd", "/etc/passwd"):
        result = await handle_run_data_gen_pipeline(
            project, script_path=script_rel, num_samples=600,
            output_dataset="d", execution="cloud",
            validation_instructions=bad, _permissions=PermissionContext.granting("cloud_data_gen"),
        )
        assert "Error" in result.content and "validation_instructions" in result.content, bad

    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="d", execution="cloud",
        validation_instructions="missing.md", _permissions=PermissionContext.granting("cloud_data_gen"),
    )
    assert "does not exist" in result.content


@pytest.mark.asyncio
async def test_validation_instructions_stored_project_relative(
    tmp_path: Path, monkeypatch,
) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(project, project / script_rel, num_samples=3, succeeded=3, failed=0)
    (project / "data_gen" / "rubric.md").write_text("be good\n")

    seen: dict = {}

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        seen["config"] = config
        return "job-v"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    # Absolute-but-inside-project path must be normalized: the sandbox
    # resolves config paths against the extracted bundle root.
    await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="d", execution="cloud",
        validation_instructions=str(project / "data_gen" / "rubric.md"),
        _permissions=PermissionContext.granting("cloud_data_gen"),
    )
    assert seen["config"]["validation_instructions"] == "data_gen/rubric.md"


def test_partial_resume_bound_to_pipeline_digest(chdir_to_tmp: Path) -> None:
    """A leftover partial from an OLD pipeline version must not let an
    edited pipeline 'complete' (and validate) without executing."""
    project = chdir_to_tmp
    script = _write_byo_project(project)
    out_dir = project / "datasets" / "d"
    out_dir.mkdir(parents=True)

    from lqh.data_gen_validation import pipeline_digest

    # Forge a partial claiming both samples are done — written by a
    # DIFFERENT pipeline version (wrong digest).
    row = json.dumps({"index": 0, "messages": "[]", "audio": None, "tools": None})
    (out_dir / "data.partial.jsonl").write_text(
        json.dumps({"_meta": True, "total": 2, "digest": "stale"}) + "\n"
        + row + "\n"
        + json.dumps({"index": 1, "messages": "[]", "audio": None, "tools": None}) + "\n"
    )

    result = asyncio.run(run_pipeline(
        script_path=script, num_samples=2, output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
    ))
    # All samples regenerated (the pipeline writes real messages, not "[]").
    assert result.succeeded == 2
    import pyarrow.parquet as pq

    table = pq.read_table(out_dir / "data.parquet")
    assert all(m != "[]" for m in table.column("messages").to_pylist())

    # The discarded legacy samples were preserved, not destroyed.
    stale = out_dir / "data.partial.stale.jsonl"
    assert stale.exists()
    assert '"index": 0' in stale.read_text()

    # Matching digest resumes as before.
    digest = pipeline_digest(script)
    (out_dir / "data.partial.jsonl").write_text(
        json.dumps({"_meta": True, "total": 2, "digest": digest}) + "\n" + row + "\n"
    )
    result2 = asyncio.run(run_pipeline(
        script_path=script, num_samples=2, output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
    ))
    assert result2.succeeded == 2  # 1 resumed + 1 generated


@pytest.mark.asyncio
async def test_cloud_concurrency_accounts_for_samples_per_item(
    tmp_path: Path, monkeypatch,
) -> None:
    """1 item × 100 variants must not run at concurrency 1 (cloud bills
    wall-clock)."""
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(project, project / script_rel, num_samples=3, succeeded=3, failed=0)

    seen: dict = {}

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        seen["config"] = config
        return "job-c"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=5, samples_per_item=10,
        output_dataset="d", execution="cloud", _permissions=PermissionContext.granting("cloud_data_gen"),
    )
    assert seen["config"]["concurrency"] == 50


@pytest.mark.asyncio
async def test_project_local_import_blocks_cloud_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    """A pipeline that imports a project-local module runs locally but
    that module won't exist in the cloud bundle — the run must not
    validate for cloud execution."""
    import sys
    import types

    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)

    async def fake_run_pipeline(*, script_path, num_samples, output_dir, client, **kw):
        mod = types.ModuleType("sneaky_helper")
        mod.__file__ = str(project / "data_gen" / "sneaky_helper.py")
        sys.modules["sneaky_helper"] = mod
        return EngineResult(
            total=num_samples, succeeded=num_samples, failed=0,
            output_path=output_dir / "data.parquet",
        )

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "tok")
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())
    try:
        result = await handle_run_data_gen_pipeline(
            project, script_path=script_rel, num_samples=3,
            output_dataset="d", execution="local",
        )
    finally:
        sys.modules.pop("sneaky_helper", None)
    assert "Not validated for cloud execution" in result.content
    assert "sneaky_helper" in result.content
    assert check_validation(project, project / script_rel) is None


@pytest.mark.asyncio
async def test_resumed_run_does_not_validate(tmp_path: Path, monkeypatch) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)

    async def fake_run_pipeline(*, script_path, num_samples, output_dir, client, **kw):
        return EngineResult(
            total=num_samples, succeeded=num_samples, failed=0,
            output_path=output_dir / "data.parquet",
            resumed_samples=2,
        )

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "tok")
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=3,
        output_dataset="d", execution="local",
    )
    assert "resumed" in result.content
    assert check_validation(project, project / script_rel) is None


@pytest.mark.asyncio
async def test_missing_source_input_blocks_submit(tmp_path: Path) -> None:
    """A recorded seed input deleted since validation would silently
    vanish from the bundle and only fail in the paid sandbox."""
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(
        project, project / script_rel,
        num_samples=3, succeeded=3, failed=0,
        source_paths=[project / "seeds.txt"],
    )
    (project / "seeds.txt").unlink()

    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="d", execution="cloud", _permissions=PermissionContext.granting("cloud_data_gen"),
    )
    assert result.content.startswith("VALIDATION_REQUIRED")
    assert "seeds.txt" in result.content


@pytest.mark.asyncio
async def test_consent_prompt_uses_live_pricing(tmp_path: Path, monkeypatch) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(project, project / script_rel, num_samples=3, succeeded=3, failed=0)

    async def _live_rate() -> float | None:
        return 2.0  # operator doubled the rate

    monkeypatch.setattr("lqh.tools.handlers._fetch_data_gen_rate_usd", _live_rate)
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="d", execution="cloud",
    )
    q = result.question or ""
    assert "$2.00/hr" in q
    assert "$16" in q         # 8h overnight at the live rate
    assert "$24" in q         # 12h cap at the live rate
    assert "default rates" not in q


def test_publish_retries_and_isolates_failures(tmp_path: Path, monkeypatch) -> None:
    """The dataset uploads first, transient errors are retried, and a
    persistently failing aux artifact doesn't fail the dataset."""
    import lqh.remote.publish as pub

    (tmp_path / "data.parquet").write_bytes(b"PAR1")
    (tmp_path / "stdout.log").write_text("log\n")
    (tmp_path / "config.json").write_text("{}")
    # The dataset candidate is gated on the run reporting success.
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "completed", "task": "data_gen"})
    )

    uploads: list[str] = []

    from lqh.artifacts import ArtifactHandle

    def _handle(kind):
        return ArtifactHandle(
            id=f"art-{len(uploads)}", kind=kind, project_id="p",
            size_bytes=4, r2_key=f"p/j/{kind}", job_id="j",
        )

    class _FlakyStore:
        def __init__(self, **_kw): ...

        async def upload_file(self, path, *, project_id, kind, job_id,
                              lineage=None, checkpoint_role=None):
            uploads.append(f"{kind}:{Path(path).name}")
            if kind == "dataset" and uploads.count(f"{kind}:{Path(path).name}") == 1:
                raise RuntimeError("transient blip")  # not ArtifactError/OSError
            if kind == "logs":
                raise RuntimeError("permanently down")
            return _handle(kind)

    monkeypatch.setattr(pub, "BackendArtifactStore", _FlakyStore)
    monkeypatch.setattr("asyncio.sleep", _fast_sleep := (lambda *_a: _instant()))

    async def _run():
        return await pub.publish_run(tmp_path, project_id="p", job_id="j")

    result = asyncio.run(_run())
    # Dataset went FIRST, survived the transient error via retry.
    assert uploads[0].startswith("dataset:")
    assert any(h.kind == "dataset" for h in result.artifacts)
    # The permanently failing log is a recorded failure, nothing raised.
    assert any("stdout.log" in f[0] for f in result.failed)


async def _instant() -> None:
    return None


@pytest.mark.asyncio
async def test_rapid_double_submit_gets_distinct_run_dirs(
    tmp_path: Path, monkeypatch,
) -> None:
    """Second-resolution timestamps collide; the random suffix must not."""
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(project, project / script_rel, num_samples=3, succeeded=3, failed=0)

    run_dirs: list[str] = []

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        run_dirs.append(run_dir)
        return f"job-{len(run_dirs)}"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    for name in ("name_a", "name_b"):
        result = await handle_run_data_gen_pipeline(
            project, script_path=script_rel, num_samples=600,
            output_dataset=name, execution="cloud", _permissions=PermissionContext.granting("cloud_data_gen"),
        )
        assert "job-" in result.content
    assert len(set(run_dirs)) == 2, run_dirs

    # A second submission targeting an output a PENDING job already owns
    # is refused — "newest submission wins" must not be reachable by
    # accident (see lqh/dataset_guard.py).
    refused = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="name_a", execution="cloud", _permissions=PermissionContext.granting("cloud_data_gen"),
    )
    assert "pending cloud data-gen job" in refused.content
    assert len(run_dirs) == 2  # no third submission happened


@pytest.mark.asyncio
async def test_observed_hf_usage_sets_needs_hf(tmp_path: Path, monkeypatch) -> None:
    """needs_hf comes from OBSERVED hf_dataset usage in the validated
    run (stored on the validation record), not from source-text scans."""
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    _mark_script_uses_hf(project, script_rel)
    record_validation(
        project, project / script_rel,
        num_samples=3, succeeded=3, failed=0, needs_hf=True,
    )

    seen: dict = {}

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        seen["config"] = config
        return "job-hf"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="d", execution="cloud", _permissions=PermissionContext.granting("cloud_data_gen"),
    )
    assert seen["config"]["needs_hf"] is True


def test_recorder_observes_hf_dataset_usage(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules, "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: iter([{"x": 1}])),
    )
    with sources.record_source_paths():
        assert not sources.hf_dataset_was_used()
        next(iter(sources.hf_dataset("user/repo")))
        assert sources.hf_dataset_was_used()
    # Flag resets with the context.
    assert not sources.hf_dataset_was_used()


def test_hf_dataset_passes_cloud_token_and_revision(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    seen: dict = {}

    def fake_load_dataset(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return iter([{"x": 1}])

    monkeypatch.setenv("HF_TOKEN", "hf_private")
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    assert next(iter(sources.hf_dataset(
        "user/private", revision="deadbeef", streaming=True,
    ))) == {"x": 1}
    assert seen["kwargs"]["token"] == "hf_private"
    assert seen["kwargs"]["revision"] == "deadbeef"


def test_sibling_edit_does_not_rearm_gate(tmp_path: Path) -> None:
    """Pipelines are self-contained single files (sibling imports are
    unsupported), so editing an UNRELATED sibling pipeline must not
    re-arm this pipeline's validation gate."""
    project = tmp_path
    (project / "data_gen").mkdir()
    script = project / "data_gen" / "p.py"
    script.write_text("x = 1\n")
    other = project / "data_gen" / "other_pipeline.py"
    other.write_text("y = 1\n")

    record_validation(project, script, num_samples=3, succeeded=3, failed=0)
    assert check_validation(project, script) is not None

    other.write_text("y = 2\n")
    assert check_validation(project, script) is not None

    script.write_text("x = 2\n")
    assert check_validation(project, script) is None


def test_zero_success_run_does_not_validate(tmp_path: Path) -> None:
    project = tmp_path
    script = project / "data_gen" / "p.py"
    script.parent.mkdir(parents=True)
    script.write_text("x = 1\n")
    record_validation(project, script, num_samples=3, succeeded=0, failed=3)
    assert check_validation(project, script) is None


@pytest.mark.asyncio
async def test_persisted_cloud_grant_skips_prompt(tmp_path: Path, monkeypatch) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    grant_cloud_data_gen_permission(project)
    record_validation(project, project / script_rel, num_samples=3, succeeded=3, failed=0)

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        return "job-7"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="d2", execution="cloud",
    )
    assert "job-7" in result.content


@pytest.mark.asyncio
async def test_local_success_records_validation_and_tips_cloud(
    tmp_path: Path, monkeypatch,
) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)

    async def fake_run_pipeline(*, script_path, num_samples, output_dir, client, **kw):
        return EngineResult(
            total=num_samples, succeeded=num_samples, failed=0,
            output_path=output_dir / "data.parquet",
            source_paths=[project / "seeds.txt"],
        )

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "tok")
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())

    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=600,
        output_dataset="big", execution="local",
    )
    assert "✅" in result.content
    assert "execution='cloud'" in result.content  # ≥500 soft tip
    rec = check_validation(project, project / script_rel)
    assert rec is not None
    assert rec.source_paths == ["seeds.txt"]


@pytest.mark.asyncio
async def test_local_self_modifying_pipeline_does_not_validate(
    tmp_path: Path, monkeypatch,
) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)

    async def fake_run_pipeline(*, script_path, num_samples, output_dir, client, **kw):
        script_path.write_text(script_path.read_text() + "\n# changed while running\n")
        return EngineResult(
            total=num_samples, succeeded=num_samples, failed=0,
            output_path=output_dir / "data.parquet",
            source_paths=[project / "seeds.txt"],
        )

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "tok")
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())

    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=3,
        output_dataset="d", execution="local",
    )
    assert "changed while it was running" in result.content
    assert check_validation(project, project / script_rel) is None


# ---------------------------------------------------------------------------
# cloud client: kind inference + presigned staging
# ---------------------------------------------------------------------------


def test_infer_kind_data_gen() -> None:
    assert _infer_kind({}, "lqh.remote.data_gen") == "data_gen"
    assert _infer_kind({"type": "data_gen"}, "lqh.train") == "data_gen"
    assert _infer_kind({}, "lqh.train") == "train_sft"


def test_bundle_to_file_contains_sources(tmp_path: Path) -> None:
    project = tmp_path
    (project / "data_gen").mkdir()
    (project / "data_gen" / "p.py").write_text("x = 1\n")
    (project / "images" / "sub").mkdir(parents=True)
    (project / "images" / "sub" / "a.jpg").write_bytes(b"jpegish")
    config = {
        "script_path": "data_gen/p.py",
        "source_paths": ["images"],
        "manifest": ["script_path", "source_paths"],
    }
    dest = tmp_path / "bundle.tar.gz"
    size = build_bundle_to_file(config, project, dest)
    assert size == dest.stat().st_size > 0
    with tarfile.open(dest, "r:gz") as tar:
        names = set(tar.getnames())
    assert "config.json" in names
    assert "data_gen/p.py" in names
    assert "images/sub/a.jpg" in names


def test_large_bundle_goes_via_presigned_put(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    (project / "data_gen").mkdir(parents=True)
    (project / "data_gen" / "p.py").write_text("x = 1\n")

    calls: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/cloud/bundles/upload-url" and request.method == "POST":
            body = json.loads(request.content)
            calls["upload_url_req"] = body
            return httpx.Response(200, json={
                "bundle_key": "u/p/staging/abc/bundle.tar.gz",
                "upload_url": "https://r2.mock/staging-put",
                "max_bytes": 2 << 30,
            })
        if request.url.host == "r2.mock" and request.method == "PUT":
            calls["put_bytes"] = len(request.read())
            calls["put_headers"] = dict(request.headers)
            return httpx.Response(200)
        if path == "/v1/cloud/jobs" and request.method == "POST":
            # Multipart body: assert the meta part carries bundle_key and
            # no bundle file part rides along.
            body = request.read()
            calls["submit_has_bundle_part"] = b'name="bundle"' in body
            calls["submit_body"] = body
            return httpx.Response(201, json={"job_id": "job-9", "status": "pending"})
        return httpx.Response(404, json={"error": {"message": f"not mocked: {path}"}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("lqh.remote.cloud.httpx.AsyncClient", _patched)
    monkeypatch.setattr("lqh.remote.cloud._MULTIPART_BUNDLE_MAX", 10)  # force staging

    cfg = RemoteConfig(name="cloud", type="cloud", hostname="h", remote_root="cloud:x")
    backend = CloudBackend(cfg, project, api_base="https://mock.lqh.test", token="t")
    job_id = asyncio.run(backend.submit_run(
        str(project / "runs" / "r1"),
        {"kind": "data_gen", "type": "data_gen",
         "script_path": "data_gen/p.py", "manifest": ["script_path"]},
        module="lqh.remote.data_gen",
    ))
    assert job_id == "job-9"
    assert calls["upload_url_req"]["project_id"] == "proj"
    # Kind rides the upload-url request so kind-level submit gates
    # (data_gen rollout flag) fail before the upload, not after.
    assert calls["upload_url_req"]["kind"] == "data_gen"
    assert calls["put_bytes"] == calls["upload_url_req"]["size_bytes"] > 0
    assert calls["put_headers"]["content-length"] == str(calls["put_bytes"])
    assert not calls["submit_has_bundle_part"]
    assert b"u/p/staging/abc/bundle.tar.gz" in calls["submit_body"]
    # The on-disk temp bundle is cleaned up.
    assert not (project / "runs" / "r1" / ".bundle.tar.gz.tmp").exists()


# ---------------------------------------------------------------------------
# publish classification
# ---------------------------------------------------------------------------


def test_publish_classifies_data_parquet_as_dataset(tmp_path: Path) -> None:
    from lqh.remote.publish import _resolve_candidates

    (tmp_path / "data.parquet").write_bytes(b"PAR1")
    (tmp_path / "data.partial.jsonl").write_text("{}\n")
    # The dataset candidate is additionally gated on the run reporting
    # success (see test_publish_gates_dataset_on_completed_status).
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "completed", "task": "data_gen"})
    )
    candidates = _resolve_candidates(tmp_path)
    by_rel = {c.relpath: c.kind for c in candidates}
    assert by_rel.get("data.parquet") == "dataset"
    assert "data.partial.jsonl" not in by_rel


# ---------------------------------------------------------------------------
# in-sandbox module
# ---------------------------------------------------------------------------


def _sandbox_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "job-1"
    inputs = run_dir / "inputs"
    (inputs / "data_gen").mkdir(parents=True)
    (inputs / "data_gen" / "p.py").write_text(_BYO_PIPELINE)
    (inputs / "seeds.txt").write_text("alpha\n")
    config = {
        "script_path": "data_gen/p.py",
        "num_samples": 2,
        "samples_per_item": 1,
        "concurrency": 4,
        "output_dataset": "d",
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    return run_dir


def test_openai_base_normalizes_both_url_forms() -> None:
    """LQH_BASE_URL is an origin in sandboxes but .../v1 on laptops —
    the OpenAI client must get a /v1 base either way."""
    from lqh.remote.data_gen import _openai_base

    assert _openai_base("https://api.lqh.ai") == "https://api.lqh.ai/v1"
    assert _openai_base("https://api.lqh.ai/") == "https://api.lqh.ai/v1"
    assert _openai_base("https://api.lqh.ai/v1") == "https://api.lqh.ai/v1"
    assert _openai_base("http://localhost:8000") == "http://localhost:8000/v1"
    assert _openai_base(None) is None
    assert _openai_base("") is None


def test_remote_data_gen_main_success(tmp_path: Path, monkeypatch) -> None:
    import lqh.remote.data_gen as dg

    run_dir = _sandbox_run_dir(tmp_path)
    seen: dict = {}

    async def fake_run_pipeline(*, script_path, num_samples, output_dir, client, **kw):
        seen["cwd"] = Path.cwd()
        seen["script_path"] = script_path
        seen["output_dir"] = output_dir
        return EngineResult(total=2, succeeded=2, failed=0,
                            output_path=output_dir / "data.parquet")

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())
    monkeypatch.setenv("LQH_API_TOKEN", "tok")
    monkeypatch.setenv("LQH_BASE_URL", "https://mock.lqh.test/v1")
    monkeypatch.delenv("LQH_JOB_ID", raising=False)  # no sentinel noise
    monkeypatch.setattr("sys.argv", ["lqh.remote.data_gen", str(run_dir / "config.json")])

    with pytest.raises(SystemExit) as exc:
        dg.main()
    assert exc.value.code == 0
    assert seen["cwd"] == (run_dir / "inputs").resolve()
    assert seen["output_dir"] == run_dir
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed" and status["succeeded"] == 2


def test_remote_data_gen_main_no_samples_fails(tmp_path: Path, monkeypatch) -> None:
    import lqh.remote.data_gen as dg

    run_dir = _sandbox_run_dir(tmp_path)

    async def fake_run_pipeline(**kw):
        return EngineResult(total=2, succeeded=0, failed=2,
                            output_path=kw["output_dir"] / "data.parquet")

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())
    monkeypatch.setenv("LQH_API_TOKEN", "tok")
    monkeypatch.delenv("LQH_JOB_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["lqh.remote.data_gen", str(run_dir / "config.json")])

    with pytest.raises(SystemExit) as exc:
        dg.main()
    assert exc.value.code == 1
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed"


# ---------------------------------------------------------------------------
# publish gating: dataset artifact existence must imply success
# ---------------------------------------------------------------------------


def test_publish_gates_dataset_on_completed_status(tmp_path: Path) -> None:
    """Downstream recovery (backend reconciler, TUI restart path) treats a
    registered dataset artifact as proof the run succeeded — so a run
    that did NOT report completed must never publish data.parquet, even
    though the engine writes one for zero-success runs."""
    import lqh.remote.publish as pub

    (tmp_path / "data.parquet").write_bytes(b"PAR1")

    def dataset_candidates() -> list:
        return [c for c in pub._resolve_candidates(tmp_path) if c.kind == "dataset"]

    # No status.json — process died before finishing.
    assert dataset_candidates() == []

    # Explicit failure (e.g. zero successful samples).
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "failed", "task": "data_gen"})
    )
    assert dataset_candidates() == []

    # Success: published.
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "completed", "task": "data_gen", "succeeded": 3})
    )
    assert len(dataset_candidates()) == 1


# ---------------------------------------------------------------------------
# HF-token donation for pipelines that stream HF datasets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_needs_hf_configures_token_donation(tmp_path: Path, monkeypatch) -> None:
    """A pipeline whose validated local run used hf_dataset must submit
    with hf_token_configured, so a locally-working private dataset also
    works in the sandbox (submit_run donates the env HF_TOKEN)."""
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    _mark_script_uses_hf(project, script_rel)
    record_validation(
        project, project / script_rel,
        num_samples=3, succeeded=3, failed=0, needs_hf=True,
    )

    seen: dict = {}

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        seen["hf_flag"] = self.config.hf_token_configured
        return "job-hf"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=5,
        output_dataset="d", execution="cloud", _permissions=PermissionContext.granting("cloud_data_gen"),
    )
    assert seen["hf_flag"] is True


@pytest.mark.asyncio
async def test_no_hf_usage_means_no_token_donation(tmp_path: Path, monkeypatch) -> None:
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    record_validation(
        project, project / script_rel,
        num_samples=3, succeeded=3, failed=0, needs_hf=False,
    )

    seen: dict = {}

    async def fake_submit(self, run_dir, config, *, module="lqh.train",
                          telemetry_workflow_id=None):
        seen["hf_flag"] = self.config.hf_token_configured
        return "job-nohf"

    monkeypatch.setattr(CloudBackend, "submit_run", fake_submit)
    await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=5,
        output_dataset="d", execution="cloud", _permissions=PermissionContext.granting("cloud_data_gen"),
    )
    assert seen["hf_flag"] is False


@pytest.mark.asyncio
async def test_consent_prompt_discloses_hf_token_donation(
    tmp_path: Path, monkeypatch,
) -> None:
    """Sending a credential with the job must be visible in the consent
    prompt, not silent."""
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    _mark_script_uses_hf(project, script_rel)
    record_validation(
        project, project / script_rel,
        num_samples=3, succeeded=3, failed=0, needs_hf=True,
    )
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=5,
        output_dataset="d", execution="cloud",
    )
    assert result.content == "PERMISSION_REQUIRED"
    assert "HF_TOKEN is sent" in result.question


# ---------------------------------------------------------------------------
# cached project-local imports must not evade bundle validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cached_project_module_is_evicted_before_run(
    tmp_path: Path, monkeypatch,
) -> None:
    """A project-local module cached in sys.modules by an earlier run
    makes its import a no-op — invisible to newly-loaded detection. The
    handler must evict it pre-run so the pipeline's (re)import shows up
    and blocks cloud validation."""
    import sys
    import types

    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)

    helper_file = project / "data_gen" / "cached_helper.py"
    helper_file.write_text("X = 1\n")
    stale = types.ModuleType("cached_helper")
    stale.__file__ = str(helper_file)
    sys.modules["cached_helper"] = stale

    observed: dict = {}

    async def fake_run_pipeline(*, script_path, num_samples, output_dir, client, **kw):
        observed["evicted"] = "cached_helper" not in sys.modules
        # What a real `import cached_helper` does after eviction:
        mod = types.ModuleType("cached_helper")
        mod.__file__ = str(helper_file)
        sys.modules["cached_helper"] = mod
        return EngineResult(
            total=num_samples, succeeded=num_samples, failed=0,
            output_path=output_dir / "data.parquet",
        )

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "tok")
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())
    try:
        result = await handle_run_data_gen_pipeline(
            project, script_path=script_rel, num_samples=3,
            output_dataset="d", execution="local",
        )
    finally:
        sys.modules.pop("cached_helper", None)
    assert observed["evicted"] is True
    assert "Not validated for cloud execution" in result.content
    assert check_validation(project, project / script_rel) is None


# ---------------------------------------------------------------------------
# local runs clear the cloud-download sidecar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_run_clears_cloud_sidecar(tmp_path: Path, monkeypatch) -> None:
    """A locally regenerated dataset is local work — the stale download
    attribution must go, or a later cloud completion would treat the
    fresh file as an old download and clobber it."""
    project, script_rel = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)
    ds_dir = project / "datasets" / "d"
    ds_dir.mkdir(parents=True)
    (ds_dir / ".lqh_source.json").write_text("{}")

    async def fake_run_pipeline(*, script_path, num_samples, output_dir, client, **kw):
        return EngineResult(
            total=num_samples, succeeded=num_samples, failed=0,
            output_path=output_dir / "data.parquet",
        )

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "tok")
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())
    result = await handle_run_data_gen_pipeline(
        project, script_path=script_rel, num_samples=3,
        output_dataset="d", execution="local",
    )
    assert "Pipeline completed" in result.content
    assert not (ds_dir / ".lqh_source.json").exists()


@pytest.mark.asyncio
async def test_script_outside_data_gen_rejected(tmp_path: Path) -> None:
    """The engine derives the project root as script_path.parent.parent,
    so scripts anywhere but directly under data_gen/ resolve paths
    against the wrong directory — reject them up front."""
    project, _ = _handler_project(tmp_path)
    grant_permission(project, None, project_wide=True)

    # Nested under data_gen/ — parent.parent would be data_gen/, not the
    # project root.
    nested = project / "data_gen" / "sub"
    nested.mkdir()
    (nested / "task.py").write_text(_BYO_PIPELINE)
    result = await handle_run_data_gen_pipeline(
        project, script_path="data_gen/sub/task.py", num_samples=1,
        output_dataset="d",
    )
    assert "directly under data_gen/" in result.content

    # At the project root — parent.parent would be the project's PARENT.
    (project / "task.py").write_text(_BYO_PIPELINE)
    result = await handle_run_data_gen_pipeline(
        project, script_path="task.py", num_samples=1, output_dataset="d",
    )
    assert "directly under data_gen/" in result.content

    # Not a .py file.
    (project / "data_gen" / "task.txt").write_text("x")
    result = await handle_run_data_gen_pipeline(
        project, script_path="data_gen/task.txt", num_samples=1,
        output_dataset="d",
    )
    assert "directly under data_gen/" in result.content


# ---------------------------------------------------------------------------
# .lqh trust boundary: agent-writable files must not be security state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_tools_cannot_write_lqh_state(tmp_path: Path) -> None:
    """.lqh/ holds consent grants and the cloud-validation gate — if the
    model could write there through file tools, every handler-enforced
    gate would become prompt-trusted."""
    from lqh.tools.handlers import (
        handle_create_file,
        handle_edit_file,
        handle_write_file,
    )

    (tmp_path / ".lqh").mkdir()
    perms = tmp_path / ".lqh" / "permissions.json"
    perms.write_text(json.dumps({"cloud_data_gen_allow_all": False}))

    for call in (
        handle_create_file(tmp_path, path=".lqh/new.json", content="{}"),
        handle_write_file(
            tmp_path, path=".lqh/permissions.json",
            content=json.dumps({"cloud_data_gen_allow_all": True}),
        ),
        handle_edit_file(
            tmp_path, path=".lqh/permissions.json",
            old_string="false", new_string="true",
        ),
        handle_write_file(
            tmp_path, path="datasets/../.lqh/data_gen_validation.json",
            content="{}",
        ),
    ):
        with pytest.raises(ValueError, match="lqh"):
            await call

    # The on-disk grant is untouched.
    assert json.loads(perms.read_text()) == {"cloud_data_gen_allow_all": False}
    # Ordinary project writes still work.
    result = await handle_write_file(tmp_path, path="notes.md", content="ok")
    assert "Wrote" in result.content


def test_forged_absolute_source_paths_invalidate_record(tmp_path: Path) -> None:
    """A validation record with absolute or traversal source_paths cannot
    be produced by record_validation — it is a forgery, and honoring it
    would bundle arbitrary readable local files (sync.resolve_manifest
    accepts absolute paths; the bundler ships them under extern/)."""
    project = tmp_path
    script = project / "data_gen" / "p.py"
    script.parent.mkdir(parents=True)
    script.write_text("x = 1\n")
    record_validation(project, script, num_samples=3, succeeded=3, failed=0)
    assert check_validation(project, script) is not None

    path = validation_file_path(project)
    data = json.loads(path.read_text())
    entry = data["pipelines"]["data_gen/p.py"]

    for bad in ("/etc/passwd", "../secrets.txt", "seed/../../x", ""):
        entry["source_paths"] = [bad]
        path.write_text(json.dumps(data))
        assert check_validation(project, script) is None, bad

    # Legitimate relative paths still pass.
    entry["source_paths"] = ["seeds.txt"]
    path.write_text(json.dumps(data))
    assert check_validation(project, script) is not None


def test_forged_needs_hf_requires_hf_dataset_in_source(tmp_path: Path) -> None:
    """needs_hf feeds HF-credential injection into arbitrary sandbox
    code. A forged flag on a pipeline that never names hf_dataset must
    not be honored; a pipeline that does keeps it."""
    project = tmp_path
    script = project / "data_gen" / "p.py"
    script.parent.mkdir(parents=True)
    script.write_text("x = 1  # no HF anywhere\n")
    record_validation(project, script, num_samples=3, succeeded=3, failed=0)

    path = validation_file_path(project)
    data = json.loads(path.read_text())
    data["pipelines"]["data_gen/p.py"]["needs_hf"] = True
    path.write_text(json.dumps(data))

    rec = check_validation(project, script)
    assert rec is not None and rec.needs_hf is False

    # A pipeline that genuinely references hf_dataset keeps the flag.
    script2 = project / "data_gen" / "q.py"
    script2.write_text("from lqh.sources import hf_dataset\n")
    record_validation(
        project, script2, num_samples=3, succeeded=3, failed=0, needs_hf=True,
    )
    rec2 = check_validation(project, script2)
    assert rec2 is not None and rec2.needs_hf is True


def test_remote_data_gen_clamps_nonpositive_concurrency(
    tmp_path: Path, monkeypatch,
) -> None:
    """config.json is client-authored; concurrency <= 0 must not hang
    the in-sandbox worker pool."""
    import lqh.remote.data_gen as dg

    run_dir = _sandbox_run_dir(tmp_path)
    config = json.loads((run_dir / "config.json").read_text())
    config["concurrency"] = 0
    (run_dir / "config.json").write_text(json.dumps(config))

    seen: dict = {}

    async def fake_run_pipeline(**kw):
        seen["concurrency"] = kw["concurrency"]
        return EngineResult(total=2, succeeded=2, failed=0,
                            output_path=kw["output_dir"] / "data.parquet")

    monkeypatch.setattr("lqh.engine.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("lqh.client.create_client", lambda *a, **k: object())
    monkeypatch.setenv("LQH_API_TOKEN", "tok")
    monkeypatch.delenv("LQH_JOB_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["lqh.remote.data_gen", str(run_dir / "config.json")])

    with pytest.raises(SystemExit) as exc:
        dg.main()
    assert exc.value.code == 0
    assert seen["concurrency"] >= 1
