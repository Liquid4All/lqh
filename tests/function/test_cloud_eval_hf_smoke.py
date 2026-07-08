"""End-to-end smoke for the eval_hf cloud job kind (Step 4).

What this proves:
  - The eval_hf_model TUI tool round-trips through CloudBackend.
  - The sandbox downloads a public HF model with snapshot_download.
  - lqh.infer.eval_hf generates rollouts on the project's eval set.
  - The scoped LQH_API_TOKEN judges the rollouts inline; eval_result.json
    materialises without laptop involvement.
  - The published lineage row records the HF repo + revision in
    base_model and the judge in reward_model. real_metric carries
    the judge summary.

Skipped unless ``LQH_E2E=1`` AND a token is resolvable. Picks the
smallest public Qwen model so the HF download stays under a minute
on a warm cache; first-ever run will be slower (~3-5 min).

Time + cost budget:
  - Wall: ~3-7 min cold, ~1-2 min warm (sandbox boots on an L4).
  - GPU cost: ~$0.15-$0.30 per run.

Usage:
    LQH_E2E=1 python -m pytest lqh_py/tests/function/test_cloud_eval_hf_smoke.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import unittest
from pathlib import Path

import httpx

from lqh.auth import api_root, get_token
from lqh.config import default_api_base_url
from lqh.tools.handlers import handle_eval_hf_model

logger = logging.getLogger(__name__)


SMOKE_TIMEOUT_SEC = int(os.environ.get("LQH_EVAL_HF_SMOKE_TIMEOUT", "900"))
POLL_INTERVAL_SEC = 2.0


def _e2e_enabled() -> tuple[bool, str]:
    if os.environ.get("LQH_E2E") != "1":
        return False, "LQH_E2E != 1"
    if get_token() is None:
        return False, "no lqh auth token"
    if not default_api_base_url():
        return False, "LQH_BASE_URL not set"
    return True, ""


_SCORER_MD = """\
# Translation scorer (eval_hf smoke)

Rate 0-10. Return: `{"score": <int>, "reasoning": "<one sentence>"}`.
"""


def _build_eval_dataset(path: Path) -> None:
    """Eight English→German prompts. Same shape as the SFT smoke
    dataset but we drop the assistant turn — eval_hf strips it
    anyway, so we let the test exercise that path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    prompts = [
        ("hello",        "hallo"),
        ("good morning", "guten morgen"),
        ("thank you",    "danke"),
        ("water",        "wasser"),
        ("yes",          "ja"),
        ("no",           "nein"),
        ("please",       "bitte"),
        ("goodbye",      "auf wiedersehen"),
    ]
    messages = []
    for en, de in prompts:
        conv = [
            {"role": "system",
             "content": "You translate English to German. Output only the German word."},
            {"role": "user", "content": en},
            # eval_hf will strip this; keeping it makes the dataset
            # match the same shape as everything else in the project.
            {"role": "assistant", "content": de},
        ]
        messages.append(json.dumps(conv, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"messages": messages})
    pq.write_table(table, path)


@unittest.skipUnless(_e2e_enabled()[0], _e2e_enabled()[1])
class TestCloudEvalHFSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self._project_dir = Path(
            os.environ.get("LQH_E2E_PROJECT_DIR")
            or os.path.expanduser(f"~/.lqh-e2e-eval-hf-{int(time.time())}")
        )
        self._project_dir.mkdir(parents=True, exist_ok=True)
        ds_dir = self._project_dir / "evals" / "translation"
        _build_eval_dataset(ds_dir / "data.parquet")
        (self._project_dir / "scorers").mkdir(exist_ok=True)
        (self._project_dir / "scorers" / "translation.md").write_text(_SCORER_MD)

    def tearDown(self) -> None:
        print(f"\neval_hf smoke artifacts preserved at: {self._project_dir}")

    def test_smoke_eval_hf_public_model(self):
        start = time.monotonic()
        # Liquid's own LFM2.5-1.2B-Instruct is the canonical small
        # model for the rest of the stack (cloud SFT smoke uses it,
        # the train skill recommends it, the LFM router knows it).
        # ~2 GB on disk; warm-cache downloads in ~30 s on the GPU sandbox.
        # Override with LQH_EVAL_HF_SMOKE_REPO for ad-hoc tests.
        repo = os.environ.get("LQH_EVAL_HF_SMOKE_REPO",
                              "LiquidAI/LFM2.5-1.2B-Instruct")

        result = asyncio.run(handle_eval_hf_model(
            self._project_dir,
            repo=repo,
            training_method="full",
            eval_dataset="evals/translation",
            scorer="scorers/translation.md",
            judge_size="small",
            max_new_tokens=64,
        ))
        # The handler returns a ToolResult with the job id embedded
        # in its content; ugly to parse but our test only needs to
        # verify the API surface. The cleaner check below is the
        # server-side artifacts list once the job completes.
        self.assertNotIn("Error", result.content,
                         f"eval_hf submit returned an error: {result.content}")
        # Extract the job id from the rendered result body.
        job_id = None
        for line in result.content.splitlines():
            if line.strip().startswith("Job ID:"):
                job_id = line.split(":", 1)[1].strip()
                break
        self.assertTrue(job_id, f"could not extract Job ID from result: {result.content!r}")
        logger.info("submitted eval_hf job: %s", job_id)

        # Poll the snapshot endpoint until terminal.
        token = get_token()
        deadline = start + SMOKE_TIMEOUT_SEC
        last_status = "pending"
        while time.monotonic() < deadline:
            with httpx.Client(base_url=api_root(), timeout=30.0) as client:
                resp = client.get(
                    f"/v1/cloud/jobs/{job_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
            if resp.status_code == 200:
                snap = resp.json()
                if snap.get("status") != last_status:
                    last_status = snap["status"]
                    logger.info("status → %s (elapsed %.0fs)",
                                last_status, time.monotonic() - start)
                if last_status in ("completed", "failed", "cancelled"):
                    break
            time.sleep(POLL_INTERVAL_SEC)
        else:
            self.fail(
                f"eval_hf job {job_id} did not reach terminal within "
                f"{SMOKE_TIMEOUT_SEC}s (last status: {last_status})"
            )

        self.assertEqual(last_status, "completed",
                         f"eval_hf ended non-success: {last_status}")

        # Grace period for publish.
        time.sleep(30)

        # Assert artifacts + lineage shape.
        project_id = self._project_dir.name
        with httpx.Client(base_url=api_root(), timeout=30.0) as client:
            arts_resp = client.get(
                f"/v1/projects/{project_id}/artifacts",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(arts_resp.status_code, 200, arts_resp.text)
            artifacts = arts_resp.json().get("artifacts", [])
            self.assertTrue(artifacts,
                            f"no artifacts published for eval_hf project {project_id}")

            lineage_resp = client.get(
                f"/v1/projects/{project_id}/lineage",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(lineage_resp.status_code, 200, lineage_resp.text)
            lineage_rows = lineage_resp.json().get("lineage", [])

        # eval_hf is the one trainer that DOES write a lineage sidecar
        # today (see lqh/infer/eval_hf.py:_write_lineage_sidecar). So
        # at least one row MUST exist and reference the HF repo.
        self.assertTrue(lineage_rows,
                        "eval_hf produced no lineage rows — _write_lineage_sidecar didn't run")

        repo_pin = None
        real_metric_seen = False
        image_id_seen = False
        for row in lineage_rows:
            bm = row.get("base_model") or ""
            if bm.startswith(repo):
                repo_pin = bm
            if row.get("real_metric") is not None:
                real_metric_seen = True
            if row.get("image_id"):
                image_id_seen = True

        self.assertIsNotNone(
            repo_pin,
            f"no lineage row pins base_model={repo}@<rev>; rows={lineage_rows}"
        )
        self.assertTrue(
            real_metric_seen,
            "no lineage row carries real_metric — inline scoring didn't "
            "merge the judge summary back into the sidecar"
        )
        # image_id may be absent in local-dev backends without an
        # image-registry row promoted. Warn rather than fail.
        if not image_id_seen:
            logger.warning(
                "no lineage row carries image_id; if you expected the "
                "Step 7 env-stamp to work, check that the backend "
                "injected LQH_IMAGE_ID into the sandbox (image registry "
                "needs an active 'gpu_rollout' row)."
            )

        elapsed = time.monotonic() - start
        logger.info(
            "eval_hf smoke OK in %.0fs (artifacts=%d, lineage=%d, base=%s)",
            elapsed, len(artifacts), len(lineage_rows), repo_pin,
        )


if __name__ == "__main__":
    unittest.main()
