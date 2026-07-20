"""Production E2E smoke for cloud data generation.

Runs three samples locally to satisfy the pipeline-revision validation gate,
submits the same unchanged pipeline for three samples in LQH Cloud, downloads
the resulting dataset artifact from R2, and validates the Parquet contents.

This test spends real LLM and cloud-compute credits, so it is opt-in::

    LQH_E2E=1 python -m pytest \
        tests/function/test_data_gen_clould_smoke.py -v -s

The lqh CLI must already be logged in. Set ``LQH_E2E_PROJECT_DIR`` to choose
where the downloaded Parquet and run logs are preserved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import pytest

from lqh.artifacts import BackendArtifactStore
from lqh.auth import get_token
from lqh.config import default_api_base_url
from lqh.project_identity import cloud_project_key
from lqh.remote.backend import RemoteConfig
from lqh.remote.cloud import CloudBackend
from lqh.tools.handlers import handle_run_data_gen_pipeline
from lqh.tools.permissions import PermissionContext

logger = logging.getLogger(__name__)

SAMPLES = 3
TIMEOUT_SECONDS = int(os.environ.get("LQH_CLOUD_DATA_GEN_SMOKE_TIMEOUT", "900"))
POLL_INTERVAL_SECONDS = 2.0

PIPELINE = '''\
from lqh.pipeline import (
    Pipeline,
    ChatMLMessage,
    Conversation,
    GenerationError,
    safe_content,
)


class CloudDataGenSmoke(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        prompt = (
            "Invent one fictional customer-support request. "
            "Return exactly one short sentence."
        )
        response = await client.chat.completions.create(
            model="small",
            messages=[
                {
                    "role": "system",
                    "content": "Generate concise synthetic training examples.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = safe_content(response).strip()
        if not answer:
            raise GenerationError("empty model response")
        return [
            ChatMLMessage(role="user", content=prompt),
            ChatMLMessage(role="assistant", content=answer),
        ]
'''


def _e2e_enabled() -> tuple[bool, str]:
    if os.environ.get("LQH_E2E") != "1":
        return False, "LQH_E2E != 1 (this test spends real credits)"
    if get_token() is None:
        return False, "no lqh auth token (run `lqh login` or use /login in the TUI)"
    base = default_api_base_url()
    if not base:
        return False, "LQH API base URL is empty"
    return True, ""


def _assert_dataset(path: Path, expected_rows: int) -> None:
    import pyarrow.parquet as pq

    assert path.exists(), f"dataset was not downloaded: {path}"
    table = pq.read_table(path)
    assert table.num_rows == expected_rows
    assert table.column_names == ["messages", "audio", "tools"]

    for row in table.to_pylist():
        messages = json.loads(row["messages"])
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[1]["content"].strip()


def _failure_logs(run_dir: Path) -> str:
    parts: list[str] = []
    for name in ("stdout.log", "stderr.log"):
        path = run_dir / name
        if path.exists():
            parts.append(f"--- {name} ---\n{path.read_text(errors='replace')[-4000:]}")
    return "\n".join(parts)


@pytest.mark.asyncio
@pytest.mark.skipif(not _e2e_enabled()[0], reason=_e2e_enabled()[1])
async def test_data_gen_cloud_smoke() -> None:
    project_dir = Path(
        os.environ.get("LQH_E2E_PROJECT_DIR")
        or Path.home() / f".lqh-e2e-cloud-data-gen-smoke-{int(time.time())}"
    ).resolve()
    script = project_dir / "data_gen" / "cloud_smoke.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(PIPELINE, encoding="utf-8")

    consent = PermissionContext(full_consent=True)

    # The real handler records the successful pipeline digest. Cloud submit
    # must consume that record, proving the local-first correctness gate.
    local_result = await handle_run_data_gen_pipeline(
        project_dir,
        script_path="data_gen/cloud_smoke.py",
        num_samples=SAMPLES,
        output_dataset="cloud_smoke_local_validation",
        purpose="smoke",
        execution="local",
        _permissions=consent,
    )
    assert local_result.ok is not False, local_result.content
    local_parquet = (
        project_dir / "datasets" / "cloud_smoke_local_validation" / "data.parquet"
    )
    _assert_dataset(local_parquet, SAMPLES)

    cloud_result = await handle_run_data_gen_pipeline(
        project_dir,
        script_path="data_gen/cloud_smoke.py",
        num_samples=SAMPLES,
        output_dataset="cloud_smoke_cloud",
        purpose="smoke",
        execution="cloud",
        timeout_minutes=15,
        _permissions=consent,
    )
    assert cloud_result.workflow_launched, cloud_result.content

    job_match = re.search(r"Job ID:\s+(\S+)", cloud_result.content)
    run_match = re.search(r"Run:\s+(\S+)", cloud_result.content)
    assert job_match and run_match, cloud_result.content
    job_id = job_match.group(1)
    run_name = run_match.group(1)
    run_dir = project_dir / "runs" / run_name

    backend = CloudBackend(
        RemoteConfig(
            name="cloud",
            type="cloud",
            hostname="api.lqh.ai",
            remote_root="cloud:lqh",
        ),
        project_dir,
    )

    reached_terminal = False
    try:
        meta = json.loads((run_dir / "remote_job.json").read_text())
        remote_run_dir = meta["remote_run_dir"]
        deadline = time.monotonic() + TIMEOUT_SECONDS
        last_state = "pending"

        while time.monotonic() < deadline:
            await backend.sync_progress(remote_run_dir, str(run_dir))
            status = await backend.poll_status(job_id)
            if status.state != last_state:
                last_state = status.state
                logger.info("cloud data-gen %s: %s", job_id, last_state)
            if status.state in {"completed", "failed", "cancelled"}:
                reached_terminal = True
                break
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        else:
            pytest.fail(
                f"cloud data-gen job {job_id} timed out after "
                f"{TIMEOUT_SECONDS}s (last state: {last_state})"
            )

        assert last_state == "completed", (
            f"cloud data-gen job ended as {last_state}\n{_failure_logs(run_dir)}"
        )

        # Job-filtered listing avoids accidentally downloading an older
        # dataset when the project has several cloud data-gen runs.
        store = BackendArtifactStore()
        artifact_deadline = time.monotonic() + 60
        datasets = []
        while time.monotonic() < artifact_deadline:
            datasets = await store.list_for_project(
                cloud_project_key(project_dir),
                kind="dataset",
                job_id=job_id,
            )
            if datasets:
                break
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        assert datasets, f"job {job_id} completed without a dataset artifact"

        cloud_parquet = project_dir / "datasets" / "cloud_smoke_cloud" / "data.parquet"
        await store.download(datasets[0], cloud_parquet)
        _assert_dataset(cloud_parquet, SAMPLES)

        print(f"\nCloud data-gen job: {job_id}")
        print(f"Downloaded Parquet: {cloud_parquet}")
        print(f"Preserved E2E project: {project_dir}")
    finally:
        if not reached_terminal:
            try:
                await backend.teardown(job_id)
            except Exception as exc:  # best-effort cleanup after a failed test
                logger.warning("failed to cancel cloud data-gen job %s: %s", job_id, exc)

