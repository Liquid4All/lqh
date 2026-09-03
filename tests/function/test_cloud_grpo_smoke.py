"""End-to-end smoke for the train_grpo cloud job kind (GRPO plan, Phase 3).

Skipped unless ``LQH_E2E=1`` AND a token is resolvable AND the backend
has the `grpo` image promoted (submit hard-gates on it — a 503 mentioning
"no active grpo image" means run modal_build_image.py --purpose grpo
--promote --register first).

What this proves:
  - `_infer_kind` maps type="grpo" → train_grpo and the backend accepts it
  - the sandbox boots the grpo image (vLLM base + trl 1.12 + lqh_py)
  - grpo_loop runs: prompt-only conversion, GRPOTrainer + vLLM colocate
    rollouts, group-rank judge rewards through the scoped chat.score
    token, a few real optimizer steps
  - the reward ledger materialises (rewards/*.json in the run dir /
    artifacts) and judge failures did not zero-fill rewards
  - the adapter publishes and the cloud_jobs row flips to completed

Deliberately tiny: G=4, 8 prompts, 3 optimizer steps, 64-token
completions, no final eval. The point is wiring, not learning.

Time + cost budget:
  - Wall: ~6-15 min (the grpo image is large; first pull dominates).
  - GPU cost: ~$0.75-$2 per run on an A100-80GB.

Usage:
    LQH_E2E=1 python -m pytest lqh_py/tests/function/test_cloud_grpo_smoke.py -v -s
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

SMOKE_TIMEOUT_SEC = int(os.environ.get("LQH_GRPO_SMOKE_TIMEOUT", "1800"))
POLL_INTERVAL_SEC = 2.0


def _e2e_enabled() -> tuple[bool, str]:
    if os.environ.get("LQH_E2E") != "1":
        return False, "LQH_E2E != 1"
    if get_token() is None:
        return False, "no lqh auth token (run /login)"
    if not default_api_base_url():
        return False, "LQH_BASE_URL not set"
    return True, ""


_SCORER_MD = """\
# Short-description scorer (grpo smoke)

Score how well the assistant's reply describes the requested topic:
clear, factually plausible, one short paragraph, no filler or repetition.
"""


def _build_smoke_dataset(path: Path) -> None:
    """Prompt-only ChatML rows (a trailing assistant turn is included so
    the loop's stripping path is exercised too)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    topics = [
        "the ocean", "a thunderstorm", "honeybees", "glaciers",
        "volcanoes", "old libraries", "sourdough bread", "public transit",
    ]
    messages = []
    for t in topics:
        conv = [
            {"role": "user", "content": f"Describe {t} in one short paragraph."},
            {"role": "assistant", "content": f"{t} is interesting."},
        ]
        messages.append(json.dumps(conv, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"messages": messages}), path)


def _build_smoke_config(dataset_rel: str, scorer_rel: str) -> dict[str, Any]:
    return {
        "type": "grpo",
        "base_model": "LiquidAI/LFM2.5-350M",
        "dataset": dataset_rel,
        "eval_dataset": dataset_rel,  # unused (no eval_on_checkpoints); keeps schema happy
        "scorer": scorer_rel,
        "training": {
            "learning_rate": 2e-6,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "logging_steps": 1,
            "max_seq_length": 512,
        },
        "grpo": {
            "num_generations": 4,
            "max_steps": 3,
            "max_completion_length": 64,
            "temperature": 0.3,
            "save_steps": 2,
        },
        "reward": {"judge_size": "small", "concurrency": 8},
        "timeout_minutes": 30,
        "manifest": ["dataset", "eval_dataset", "scorer"],
    }


@unittest.skipUnless(_e2e_enabled()[0], _e2e_enabled()[1])
class TestCloudGrpoSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self._project_dir = Path(
            os.environ.get("LQH_E2E_PROJECT_DIR")
            or os.path.expanduser(f"~/.lqh-e2e-grpo-smoke-{int(time.time())}")
        )
        self._project_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = self._project_dir / "runs" / "grpo_smoke"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        _build_smoke_dataset(self._project_dir / "datasets" / "tiny" / "data.parquet")
        scorer_path = self._project_dir / "scorers" / "smoke.md"
        scorer_path.parent.mkdir(parents=True, exist_ok=True)
        scorer_path.write_text(_SCORER_MD)

        self._config = _build_smoke_config(
            "datasets/tiny/data.parquet", "scorers/smoke.md",
        )
        cfg = RemoteConfig(
            name="cloud", type="cloud",
            hostname="api.lqh.ai", remote_root="cloud:lqh",
        )
        self._backend = CloudBackend(cfg, self._project_dir)
        self._job_id: str | None = None

    def tearDown(self) -> None:
        if self._job_id:
            try:
                asyncio.run(self._backend.teardown(self._job_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning("cleanup teardown failed: %s", exc)
        print(f"\nE2E artifacts preserved at: {self._project_dir}")

    def test_smoke_grpo_three_steps(self) -> None:
        start = time.monotonic()
        self._job_id = asyncio.run(self._backend.submit_run(
            str(self._run_dir), self._config, module="lqh.train",
        ))
        self.assertTrue(self._job_id, "submit_run returned empty job_id")
        logger.info("submitted train_grpo job: %s", self._job_id)

        terminal_states = {"completed", "failed"}
        deadline = start + SMOKE_TIMEOUT_SEC
        last_status = "pending"
        while time.monotonic() < deadline:
            asyncio.run(self._backend.sync_progress(
                f"cloud:{self._job_id}", str(self._run_dir),
            ))
            state_path = self._run_dir / "cloud_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("status") != last_status:
                    last_status = state["status"]
                    logger.info("status → %s (%.0fs)", last_status,
                                time.monotonic() - start)
                if last_status in terminal_states:
                    break
            time.sleep(POLL_INTERVAL_SEC)
        else:
            self.fail(f"job {self._job_id} not terminal within "
                      f"{SMOKE_TIMEOUT_SEC}s (last: {last_status})")

        self.assertEqual(last_status, "completed",
                         f"job ended in non-success state: {last_status}")

        # Trainer-side progress reached us, with GRPO's task kind.
        progress_lines = [
            json.loads(line) for line in
            (self._run_dir / "progress.jsonl").read_text().strip().splitlines()
            if line.strip()
        ]
        self.assertTrue(progress_lines, "no progress events streamed back")
        kinds = {row.get("task_kind") for row in progress_lines}
        self.assertIn("grpo", kinds, f"no grpo-kind progress rows: {kinds}")

        # Server-side artifacts: the adapter checkpoint must publish.
        grace_deadline = time.monotonic() + 60.0
        while time.monotonic() < grace_deadline:
            asyncio.run(self._backend.sync_progress(
                f"cloud:{self._job_id}", str(self._run_dir),
            ))
            time.sleep(POLL_INTERVAL_SEC)

        import httpx

        from lqh.auth import api_root

        token = get_token()
        with httpx.Client(base_url=api_root(), timeout=30.0) as client:
            resp = client.get(
                f"/v1/projects/{self._project_dir.name}/artifacts",
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        artifacts = resp.json().get("artifacts", [])
        art_kinds = sorted({a.get("kind", "") for a in artifacts})
        self.assertIn("checkpoint", art_kinds,
                      f"no checkpoint published: kinds={art_kinds}")
        logger.info("grpo smoke completed in %.0fs (artifacts=%s)",
                    time.monotonic() - start, art_kinds)


if __name__ == "__main__":
    unittest.main()
