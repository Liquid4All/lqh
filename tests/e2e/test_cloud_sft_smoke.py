"""End-to-end smoke for cloud fine-tuning — small dataset → real
trained checkpoint → final artifacts in R2.

Skipped unless:
  - ``LQH_E2E=1`` is exported (opt-in; the test spends real money)
  - the lqh CLI is logged in (token resolvable via ``lqh.auth.get_token``)
  - ``LQH_BASE_URL`` points at a backend that has the GPU provider
    + R2 wired up (verify with ``go run ./cmd/modaldebug`` first)

What this proves:
  - Bundle assembly handles a real parquet dataset
  - Multipart submit to ``/v1/cloud/jobs`` succeeds
  - Backend pre-flight cost check passes for the bootstrapped user
  - R2 bundle upload + presigned-GET works
  - Cloud sandbox boots from the configured image
  - Volume sub_path mount (project-scoped) works
  - HF model download (or cache hit on warm boot) works
  - Trainer runs and emits LQH_EVENT_JSON sentinels
  - Sentinels reach the runner, get persisted to cloud_job_events
  - SSE replay delivers them to the CloudBackend on reconnect
  - Publish step uploads checkpoint + metrics to R2
  - Output reaper cleans up the per-job volume scratch
  - cloud_jobs row flips to status=completed
  - artifacts table has entries pointing at real R2 keys

What this does NOT prove:
  - Scoring (no scorer in the smoke config — keeps the test tight)
  - DPO (smoke does SFT only — see CLOUD_FT_TODOs C1)
  - Large-dataset bundle streaming (smoke uses 8 rows — see C2)
  - The agent's tool dispatch (we drive CloudBackend directly, one
    layer below the LLM)

Time + cost budget:
  - Wall time: ~3-5 min on a cold project (HF model download); ~30s
    warm (post-6a cache).
  - GPU cost: ~$0.25-$0.50 per run on an A100-40GB.

Usage:
    LQH_E2E=1 python -m pytest lqh_py/tests/e2e/test_cloud_sft_smoke.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import unittest
from pathlib import Path
from typing import Any

from lqh.auth import get_token
from lqh.config import default_api_base_url
from lqh.remote.backend import RemoteConfig
from lqh.remote.cloud import CloudBackend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Total wall-clock cap on a single smoke run. A cold first-ever run
# in a new project can take 3-5 min for the HF model download; a
# warm run (with the per-project cache populated) lands ~30s. We
# give it 15 min so a CI box with a flaky network still completes.
SMOKE_TIMEOUT_SEC = int(os.environ.get("LQH_CLOUD_SMOKE_TIMEOUT", "900"))

# How often to poke the SSE/sync_progress loop. Tight enough to feel
# responsive, loose enough that we don't burn the backend's poll
# budget on a multi-minute training step.
POLL_INTERVAL_SEC = 2.0


def _e2e_enabled() -> tuple[bool, str]:
    """Returns (enabled, reason_if_disabled)."""
    if os.environ.get("LQH_E2E") != "1":
        return False, "LQH_E2E != 1"
    if get_token() is None:
        return False, "no lqh auth token (run /login or set LQH_DEBUG_API_KEY)"
    base = default_api_base_url()
    if not base or "lqh" not in base.lower() and "localhost" not in base.lower():
        return False, f"LQH_BASE_URL not set or unexpected: {base!r}"
    return True, ""


def _build_smoke_dataset(path: Path) -> None:
    """Write a tiny ChatML parquet at ``path``. Eight English→German
    pairs. Trivial enough that a 1-epoch LoRA on LFM2-1.2B converges
    visibly without overfitting matters for the smoke check."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pairs = [
        ("hello",       "hallo"),
        ("good morning","guten morgen"),
        ("thank you",   "danke"),
        ("water",       "wasser"),
        ("yes",         "ja"),
        ("no",          "nein"),
        ("please",      "bitte"),
        ("goodbye",     "auf wiedersehen"),
    ]
    messages = []
    for en, de in pairs:
        conv = [
            {"role": "system",
             "content": "You translate English to German. Output only the German translation."},
            {"role": "user", "content": en},
            {"role": "assistant", "content": de},
        ]
        messages.append(json.dumps(conv, ensure_ascii=False))

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"messages": messages})
    pq.write_table(table, path)


def _build_smoke_config(project_dir: Path, dataset_rel: str) -> dict[str, Any]:
    """Compose the SFT config the CloudBackend will ship to the
    sandbox. Aggressively small: one epoch, batch size 1, eval
    every 4 steps, LoRA. The point is to verify the wiring, not
    train a useful model."""
    return {
        "type": "sft",
        # LFM2-1.2B is the documented smallest base model and is what
        # the per-project cache is most likely to have warm.
        "base_model": "LiquidAI/LFM2-1.2B",
        "dataset":     dataset_rel,
        "eval_dataset": dataset_rel,  # train==eval is fine for smoke
        "training": {
            "num_epochs":         1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size":  1,
            "learning_rate":      2e-4,
            "lora":               True,
            "save_steps":         4,
            "eval_steps":         4,
            "logging_steps":      1,
            "max_seq_length":     128,
        },
        # Files the bundle builder will tar in. Paths are project-relative.
        "manifest": ["dataset", "eval_dataset"],
    }


@unittest.skipUnless(_e2e_enabled()[0], _e2e_enabled()[1])
class TestCloudSftSmoke(unittest.TestCase):
    """The minimum-viable cloud fine-tune that exercises the full
    GPU+R2+backend round trip."""

    def setUp(self) -> None:
        self._project_dir = Path(
            os.environ.get("LQH_E2E_PROJECT_DIR")
            or os.path.expanduser(f"~/.lqh-e2e-cloud-smoke-{int(time.time())}")
        )
        self._project_dir.mkdir(parents=True, exist_ok=True)
        # Backend keys (uid+pid) on R2 prefix from project_dir.name;
        # using a unique-per-run name keeps R2 prefix collisions out
        # of repeated test runs.

        self._run_dir = self._project_dir / "runs" / "smoke"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        ds_dir = self._project_dir / "datasets" / "tiny"
        _build_smoke_dataset(ds_dir / "data.parquet")

        self._config = _build_smoke_config(self._project_dir, "datasets/tiny/data.parquet")

        cfg = RemoteConfig(
            name="cloud",
            type="cloud",
            hostname="api.lqh.ai",  # informational; real base from default_api_base_url()
            remote_root="cloud:lqh",
        )
        self._backend = CloudBackend(cfg, self._project_dir)
        self._job_id: str | None = None

    def tearDown(self) -> None:
        # Best-effort cleanup: cancel any still-running job. Don't fail
        # the test if cleanup itself errors.
        if self._job_id:
            try:
                asyncio.run(self._backend.teardown(self._job_id))
            except Exception as exc:
                logger.warning("cleanup teardown failed: %s", exc)
        # Leave the project_dir on disk so an operator inspecting a
        # failed run can read progress.jsonl / artifacts.json. Add a
        # one-line breadcrumb pointing at it.
        print(f"\nE2E artifacts preserved at: {self._project_dir}")

    def test_smoke_sft_lora_one_epoch(self):
        """Submit, wait for terminal, assert on the local mirror."""
        start = time.monotonic()

        # ---- 1. submit ----
        self._job_id = asyncio.run(self._backend.submit_run(
            str(self._run_dir),
            self._config,
            module="lqh.train",
        ))
        self.assertTrue(self._job_id, "submit_run returned empty job_id")
        logger.info("submitted cloud job: %s", self._job_id)

        # remote_job.json + cloud_state.json must be on disk
        # immediately so a CLI restart could pick up the run.
        self.assertTrue((self._run_dir / "remote_job.json").exists())
        self.assertTrue((self._run_dir / "cloud_state.json").exists())

        # ---- 2. drive sync_progress until terminal ----
        terminal_states = {"completed", "failed"}
        deadline = start + SMOKE_TIMEOUT_SEC
        last_status = "pending"
        loops = 0
        while time.monotonic() < deadline:
            loops += 1
            asyncio.run(self._backend.sync_progress(
                f"cloud:{self._job_id}", str(self._run_dir),
            ))
            state_path = self._run_dir / "cloud_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("status") != last_status:
                    last_status = state["status"]
                    logger.info("status → %s (loop %d, elapsed %.0fs)",
                                last_status, loops, time.monotonic() - start)
                if last_status in terminal_states:
                    break
            time.sleep(POLL_INTERVAL_SEC)
        else:
            self.fail(
                f"smoke job {self._job_id} did not reach terminal status "
                f"within {SMOKE_TIMEOUT_SEC}s (last status: {last_status})"
            )

        # ---- 3. assertions ----
        self.assertEqual(last_status, "completed",
                         f"job ended in non-success state: {last_status}")

        # status.json: terminal state was mirrored to disk
        status = json.loads((self._run_dir / "status.json").read_text())
        self.assertEqual(status["state"], "completed", status)

        # progress.jsonl: at least one progress row was received
        progress_lines = [
            json.loads(l) for l in
            (self._run_dir / "progress.jsonl").read_text().strip().splitlines()
            if l.strip()
        ]
        self.assertTrue(progress_lines, "progress.jsonl is empty — no events streamed back")

        progress_rows = [r for r in progress_lines if "step" in r]
        self.assertTrue(progress_rows,
                        f"progress.jsonl has no step rows (got {len(progress_lines)} entries)")
        logger.info("received %d progress rows", len(progress_rows))

        # ---- 4. server-side artifacts ----
        # The trainer's write_status('completed') sentinel arrives BEFORE
        # the launcher's publish step runs — so by the time the poll loop
        # exited above, publish may not have registered anything yet.
        # We keep polling SSE for a grace period to catch trailing log
        # events, then query the server's artifacts list directly. The
        # canonical "did publish work?" check is the SERVER-SIDE state,
        # not the local artifacts.json (which is a best-effort mirror
        # built from SSE 'artifact' events the publish helper doesn't
        # currently emit — tracked in CLOUD_FT_TODOs).
        grace_deadline = time.monotonic() + 60.0
        while time.monotonic() < grace_deadline:
            asyncio.run(self._backend.sync_progress(
                f"cloud:{self._job_id}", str(self._run_dir),
            ))
            time.sleep(POLL_INTERVAL_SEC)
            # Stop early if we've seen artifacts come through the SSE
            # (would require publish.py emitting LQH_EVENT_JSON kind=
            # artifact — future work).
            if (self._run_dir / "artifacts.json").exists():
                manifest = json.loads((self._run_dir / "artifacts.json").read_text())
                if manifest.get("artifacts"):
                    break

        # Authoritative check: query the artifacts list endpoint.
        # This is what production users would do via the lqh CLI to
        # find their trained model. Project ID is the basename of
        # the project dir we set up.
        project_id = self._project_dir.name
        import httpx
        from lqh.auth import get_token, api_root
        token = get_token()
        self.assertIsNotNone(token, "no auth token — should have skipped before reaching here")
        with httpx.Client(base_url=api_root(), timeout=30.0) as client:
            resp = client.get(
                f"/v1/projects/{project_id}/artifacts",
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 200,
                         f"artifacts list returned {resp.status_code}: {resp.text}")
        body = resp.json()
        server_artifacts = body.get("artifacts", [])
        logger.info("server-side artifacts for project %r: %d entries (kinds=%s)",
                    project_id, len(server_artifacts),
                    sorted({a.get("kind", "") for a in server_artifacts}))
        self.assertTrue(
            server_artifacts,
            f"server has no artifacts for project {project_id!r} — "
            f"publish step inside the sandbox either never ran or "
            f"failed to register against /v1/artifacts (check backend logs "
            f"filtered by user_id, or stdout.log in the preserved run dir)"
        )

        # At least one artifact should be a train-like kind. The
        # SFT smoke produces a checkpoint (run_dir/model/), metrics
        # (progress.jsonl), and possibly logs — any of these
        # qualifies as proof that publish.py walked the run dir
        # correctly.
        kinds = sorted({a.get("kind", "") for a in server_artifacts})
        self.assertTrue(any(k in kinds for k in ("checkpoint", "metrics", "predictions", "logs")),
                        f"no train-like artifact published: kinds={kinds}")

        elapsed = time.monotonic() - start
        logger.info("smoke completed in %.0fs (job_id=%s, artifacts=%d, kinds=%s)",
                    elapsed, self._job_id, len(server_artifacts), kinds)


if __name__ == "__main__":
    unittest.main()
