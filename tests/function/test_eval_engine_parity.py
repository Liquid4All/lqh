"""HF-vs-sglang eval engine parity fixture (ISSUE 4 P1 validation).

Runs the SAME small eval twice in the cloud — once on the default
engine (sglang on the gpu_eval image) and once with force_hf_engine —
and compares the judged outcomes. This is the repeatable version of the
one-off parity evidence gathered at P1 rollout: both engines must
complete, score every sample, and land in the same quality band.

Judge scores are stochastic-adjacent (LLM judge), so the assertion is a
band (|Δmean| <= 2.0), not equality; the hard guarantees are structural:
both complete, both score n == len(dataset).

Skipped unless ``LQH_E2E=1`` AND a token is resolvable. Costs two short
L4 runs (~$0.30-0.60 total).

Usage:
    LQH_E2E=1 python -m pytest tests/function/test_eval_engine_parity.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import unittest
from pathlib import Path

import httpx

from lqh.auth import api_root, get_token
from lqh.config import default_api_base_url
from lqh.tools.handlers import handle_eval_hf_model
from lqh.tools.permissions import PermissionContext

PARITY_TIMEOUT_SEC = int(os.environ.get("LQH_EVAL_PARITY_TIMEOUT", "1800"))
REPO = os.environ.get("LQH_EVAL_PARITY_REPO", "LiquidAI/LFM2.5-1.2B-Instruct")
MAX_MEAN_DELTA = 2.0


def _e2e_enabled() -> tuple[bool, str]:
    if os.environ.get("LQH_E2E") != "1":
        return False, "LQH_E2E != 1"
    if get_token() is None:
        return False, "no lqh auth token"
    if not default_api_base_url():
        return False, "LQH_BASE_URL not set"
    return True, ""


_SCORER_MD = """\
# Capital-city scorer (engine parity)

Rate 0-10: is the named capital correct and the answer concise?
Return: `{"score": <int>, "reasoning": "<one sentence>"}`.
"""

_CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Norway", "Oslo"),
    ("Peru", "Lima"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
    ("Chile", "Santiago"), ("Poland", "Warsaw"),
]


def _build_project(root: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "scorers").mkdir(parents=True)
    (root / "scorers" / "capitals.md").write_text(_SCORER_MD)
    messages = [
        json.dumps([{"role": "user",
                     "content": f"What is the capital of {country}? "
                                "Answer with just the city name."}])
        for country, _ in _CAPITALS
    ]
    ds = root / "evals" / "capitals"
    ds.mkdir(parents=True)
    pq.write_table(pa.table({"messages": messages}), ds / "data.parquet")
    return root


def _submit_and_wait(project_dir: Path, *, force_hf: bool) -> dict:
    """Submit one eval, poll to terminal, return {status, mean, n}."""
    kwargs: dict = dict(
        repo=REPO, training_method="full", eval_dataset="evals/capitals",
        scorer="scorers/capitals.md", max_new_tokens=32, timeout_minutes=30,
        _permissions=PermissionContext.granting("cloud_eval_hf"),
    )
    if force_hf:
        kwargs["force_hf_engine"] = True
    result = asyncio.run(handle_eval_hf_model(project_dir, **kwargs))
    assert "Error" not in result.content, result.content
    job_id = next(ln.split(":", 1)[1].strip()
                  for ln in result.content.splitlines()
                  if ln.strip().startswith("Job ID:"))

    token = get_token()
    deadline = time.monotonic() + PARITY_TIMEOUT_SEC
    status = "pending"
    while time.monotonic() < deadline:
        with httpx.Client(base_url=api_root(), timeout=30.0) as c:
            r = c.get(f"/v1/cloud/jobs/{job_id}",
                      headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            status = r.json().get("status", status)
            if status in ("completed", "failed", "cancelled"):
                break
        time.sleep(5)

    mean, n = None, None
    if status == "completed":
        time.sleep(20)  # publish grace
        with httpx.Client(base_url=api_root(), timeout=30.0) as c:
            rows = c.get(f"/v1/projects/{project_dir.name}/lineage",
                         headers={"Authorization": f"Bearer {token}"}
                         ).json().get("lineage", [])
        for row in rows:
            rm = row.get("real_metric") or {}
            if rm.get("name") == "judge_score_mean":
                mean, n = rm.get("value"), rm.get("n")
    return {"status": status, "mean": mean, "n": n, "job_id": job_id}


@unittest.skipUnless(_e2e_enabled()[0], _e2e_enabled()[1])
class TestEvalEngineParity(unittest.TestCase):
    def test_sglang_and_hf_engines_agree(self):
        ts = int(time.time())
        proj_sgl = _build_project(
            Path(os.path.expanduser(f"~/.lqh-e2e-parity-sglang-{ts}")))
        proj_hf = _build_project(
            Path(os.path.expanduser(f"~/.lqh-e2e-parity-hf-{ts}")))

        sgl = _submit_and_wait(proj_sgl, force_hf=False)
        hf = _submit_and_wait(proj_hf, force_hf=True)
        print(f"\nsglang: {sgl}\nhf:     {hf}")

        self.assertEqual(sgl["status"], "completed", sgl)
        self.assertEqual(hf["status"], "completed", hf)
        self.assertEqual(sgl["n"], len(_CAPITALS), sgl)
        self.assertEqual(hf["n"], len(_CAPITALS), hf)
        self.assertIsNotNone(sgl["mean"])
        self.assertIsNotNone(hf["mean"])
        self.assertLessEqual(
            abs(sgl["mean"] - hf["mean"]), MAX_MEAN_DELTA,
            f"engine quality bands diverge: sglang={sgl['mean']} hf={hf['mean']}",
        )


if __name__ == "__main__":
    unittest.main()
