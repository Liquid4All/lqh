"""End-to-end validation that a cloud SFT run actually improves the
model on a held-out eval — not just "the job ran to completion".

Three phases:

  1. **Baseline** (~30 s, ~$0.05). Run inference + judge scoring via
     the API against ``LFM2.5-1.2B-Instruct`` on a held-out eval set.
     ``lqh.scoring.run_scoring`` with ``run_inference=True`` does the
     whole thing in one call: strips the trailing assistant turn from
     each conversation, calls ``/v1/chat/completions`` with the base
     model to get a prediction, then judges with ``judge:small``.
     Result: ``mean_pre``.

  2. **Training** (~20-40 min, ~$1-2). Submit a 1-config ``train_sft_sweep``
     with ``eval_best=True``. The sweep trains, runs the winner against
     the same eval set, and ``score_run_eval_inline`` writes
     ``real_metric`` onto the rollout-lineage row. The sweep is used
     rather than plain ``train_sft`` because the sweep path is the
     one that already does in-sandbox post-training scoring (Step 3
     of the training-infra plan). Plain ``train_sft`` would require
     a follow-up laptop scoring round.

  3. **Compare**. Pull the lineage row for the winning checkpoint;
     read ``real_metric.value`` → ``mean_post``. Assert ``mean_post``
     is strictly greater than ``mean_pre`` by at least
     ``MIN_IMPROVEMENT`` (default 0.5 score points out of 10).

Dataset:
  Format-discipline en→de translation. The instruction says "output
  ONLY the German word, no preamble, no quotes". ``LFM2.5-1.2B-Instruct``
  out of the box tends to add prefatory chat ("Sure, the German is
  '<x>'."), which the scorer's rubric punishes. After SFT on ~80 clean
  pairs, the model should snap to the bare-word format → big delta.

Gating:
  Opt-in via ``LQH_E2E_BEFORE_AFTER=1`` (separate from ``LQH_E2E``
  because this one spends measurably more $ and time than the smoke
  tests). Requires lqh login + an active image-registry sft row.

Usage:
    LQH_E2E_BEFORE_AFTER=1 python -m pytest \
        lqh_py/tests/function/test_cloud_sft_before_after.py -v -s
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

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from lqh.auth import api_root, get_token
from lqh.client import create_client
from lqh.remote.backend import RemoteConfig
from lqh.remote.cloud import CloudBackend
from lqh.scoring import run_scoring

logger = logging.getLogger(__name__)


# -- tuning knobs ----------------------------------------------------

# Hard wall-time cap on the cloud training job. Conservative — the
# sweep includes an eval-of-best step after training; A100-80GB
# starts cold ~60s, trains 80×3 epochs LoRA in 5-10 min, then runs
# eval on 30 prompts (~30s) + judge scoring (~30s).
TRAIN_TIMEOUT_SEC = int(os.environ.get("LQH_BA_TRAIN_TIMEOUT", "2700"))

# Poll cadence for the snapshot endpoint.
POLL_INTERVAL_SEC = 5.0

# Minimum score improvement (out of 10) required for a passing test.
# 0.5 is a low bar but rules out noise on 30-sample evals; the
# format-discipline dataset typically gives +3.0 or more.
MIN_IMPROVEMENT = float(os.environ.get("LQH_BA_MIN_IMPROVEMENT", "0.5"))

# Base model. Pinned to the one the LFM router serves so the
# baseline API call lands on it without a snapshot_download.
BASE_MODEL_HF = "LiquidAI/LFM2.5-1.2B-Instruct"
BASE_MODEL_LFM = "lfm2.5-1.2b-instruct"

# Judge.
JUDGE_SIZE = "small"


# -- gating ----------------------------------------------------------


def _enabled() -> tuple[bool, str]:
    if os.environ.get("LQH_E2E_BEFORE_AFTER") != "1":
        return False, "LQH_E2E_BEFORE_AFTER != 1 (opt-in; ~30 min, ~$2)"
    if not get_token():
        return False, "no lqh auth token (run /login or set LQH_API_TOKEN)"
    return True, ""


# -- fixtures --------------------------------------------------------


# 80 en→de pairs — common nouns + short phrases. Same distribution
# as the eval set so a 1-config sweep can actually fit.
_TRAIN_PAIRS = [
    ("hello", "hallo"), ("good morning", "guten morgen"),
    ("thank you", "danke"), ("water", "wasser"),
    ("yes", "ja"), ("no", "nein"), ("please", "bitte"),
    ("goodbye", "auf wiedersehen"),
    ("house", "haus"), ("car", "auto"), ("dog", "hund"),
    ("cat", "katze"), ("book", "buch"), ("tree", "baum"),
    ("sun", "sonne"), ("moon", "mond"), ("star", "stern"),
    ("rain", "regen"), ("snow", "schnee"), ("fire", "feuer"),
    ("table", "tisch"), ("chair", "stuhl"), ("window", "fenster"),
    ("door", "tür"), ("street", "straße"), ("city", "stadt"),
    ("country", "land"), ("ocean", "ozean"), ("river", "fluss"),
    ("mountain", "berg"), ("forest", "wald"), ("flower", "blume"),
    ("bread", "brot"), ("cheese", "käse"), ("milk", "milch"),
    ("coffee", "kaffee"), ("tea", "tee"), ("wine", "wein"),
    ("beer", "bier"), ("apple", "apfel"), ("orange", "orange"),
    ("banana", "banane"), ("egg", "ei"), ("salt", "salz"),
    ("sugar", "zucker"), ("butter", "butter"), ("rice", "reis"),
    ("morning", "morgen"), ("evening", "abend"), ("night", "nacht"),
    ("today", "heute"), ("tomorrow", "morgen"), ("yesterday", "gestern"),
    ("father", "vater"), ("mother", "mutter"), ("son", "sohn"),
    ("daughter", "tochter"), ("brother", "bruder"), ("sister", "schwester"),
    ("friend", "freund"), ("teacher", "lehrer"), ("doctor", "arzt"),
    ("school", "schule"), ("hospital", "krankenhaus"),
    ("library", "bibliothek"), ("park", "park"),
    ("museum", "museum"), ("church", "kirche"),
    ("hand", "hand"), ("foot", "fuß"), ("head", "kopf"),
    ("eye", "auge"), ("ear", "ohr"), ("mouth", "mund"),
    ("red", "rot"), ("blue", "blau"), ("green", "grün"),
    ("yellow", "gelb"), ("black", "schwarz"), ("white", "weiß"),
    ("one", "eins"),
]

# 30 held-out pairs — same domain, different words. Trained model
# should generalise format discipline (just-the-word output) even on
# words it hasn't seen.
_EVAL_PAIRS = [
    ("airplane", "flugzeug"), ("train", "zug"), ("bicycle", "fahrrad"),
    ("ship", "schiff"), ("bus", "bus"), ("truck", "lastwagen"),
    ("garden", "garten"), ("kitchen", "küche"),
    ("bedroom", "schlafzimmer"), ("bathroom", "badezimmer"),
    ("computer", "computer"), ("phone", "telefon"), ("camera", "kamera"),
    ("music", "musik"), ("dance", "tanz"), ("song", "lied"),
    ("game", "spiel"), ("ball", "ball"),
    ("knife", "messer"), ("spoon", "löffel"), ("fork", "gabel"),
    ("plate", "teller"), ("cup", "tasse"), ("bottle", "flasche"),
    ("week", "woche"), ("month", "monat"), ("year", "jahr"),
    ("summer", "sommer"), ("winter", "winter"), ("spring", "frühling"),
]


_SYSTEM_PROMPT = (
    "You translate English to German. Output ONLY the German word "
    "or short phrase. No preamble, no quotes, no explanation, just "
    "the German."
)


_SCORER_MD = """\
# en→de translation scorer

You are scoring an English-to-German translation. The user gave the
English word/phrase. The assistant should output ONLY the German
translation — no preamble, no surrounding quotes, no commentary, just
the German word(s). Case differences are fine; trailing punctuation
is fine.

Score 0-10:

- 10: Exactly the right German word/phrase, no extra text.
- 8-9: Right German but with minor extras (e.g. trailing period,
       leading uppercase that didn't belong).
- 5-7: Right German embedded in chatty prose ("The German is 'x'.").
- 3-4: Wrong German but at least an attempt.
- 0-2: Empty, English, or unrelated.

Return JSON exactly: `{"score": <int 0-10>, "reasoning": "<one sentence>"}`.
"""


def _build_chatml_parquet(path: Path, pairs: list[tuple[str, str]]) -> None:
    """Write a ChatML parquet at ``path`` with one row per pair. Each
    conversation includes the system instruction (so train + eval
    apply the same prompt), the user's English word, and the gold
    German assistant turn (the eval path strips this; the train path
    keeps it)."""
    messages: list[str] = []
    for en, de in pairs:
        conv = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": en},
            {"role": "assistant", "content": de},
        ]
        messages.append(json.dumps(conv, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"messages": messages}), path)


def _build_sweep_config(dataset_rel: str, eval_rel: str, scorer_rel: str) -> dict[str, Any]:
    """1-config sweep wrapped around the SFT base config. ``eval_best``
    drives the post-training inline eval that produces the
    ``real_metric`` we're comparing against the baseline."""
    base: dict[str, Any] = {
        "type": "sft",
        "base_model": BASE_MODEL_HF,
        "dataset": dataset_rel,
        "eval_dataset": eval_rel,
        "scorer": scorer_rel,
        "system_prompt": _SYSTEM_PROMPT,
        "training": {
            "num_epochs":                     3,
            "per_device_train_batch_size":    4,
            "per_device_eval_batch_size":     4,
            "learning_rate":                  2e-4,
            "lora":                           True,
            "save_steps":                     20,
            "eval_steps":                     20,
            "logging_steps":                  5,
            "max_seq_length":                 128,
        },
        # eval_hf-style judge: the LFM-routed small judge. Picked
        # so a flaky prod scorer doesn't make the test look broken.
        "judge_size":               JUDGE_SIZE,
        "max_new_tokens":           32,
        "manifest": ["dataset", "eval_dataset", "scorer"],
    }
    return {
        "type":            "sweep",
        "base_config":     base,
        "grid_size":       "tiny",
        # The "tiny" SFT grid is normally 3 configs (different
        # learning rates) — we explicitly override to ONE config
        # so the test stays bounded to <30 min wall.
        "grid_override": [
            {
                "id": "ba_lr2e-4_e3",
                "overrides": {
                    # Empty override — base_config already sets these,
                    # but the sweep needs an entry to know which point
                    # to run. The id is descriptive.
                },
            },
        ],
        "eval_best": True,
    }


# -- helpers ---------------------------------------------------------


async def _baseline_score(
    eval_parquet: Path,
    scorer_md: Path,
    output_dir: Path,
) -> float | None:
    """Run inference + judge against the BASE model via the API; return
    the judge's mean score (or None on no scored samples).

    Uses ``lqh.scoring.run_scoring(run_inference=True)`` so the same
    code path that scores the trained checkpoint also produces the
    baseline — keeps the apples-to-apples honest.
    """
    token = get_token()
    api_base = api_root() + "/v1"   # the OpenAI SDK expects /v1
    client = create_client(token, api_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    await run_scoring(
        dataset_path=eval_parquet,
        scorer_path=scorer_md,
        output_dir=output_dir,
        client=client,
        model_size=JUDGE_SIZE,
        run_inference=True,
        inference_model=BASE_MODEL_LFM,
        inference_system_prompt=_SYSTEM_PROMPT,
    )
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return _extract_mean(summary)


def _extract_mean(summary: dict[str, Any]) -> float | None:
    """summary.json from run_scoring has ``scores.mean`` per
    lqh/scoring.py:563. Defensive: tolerate older shapes."""
    if not isinstance(summary, dict):
        return None
    scores = summary.get("scores")
    if isinstance(scores, dict) and "mean" in scores:
        try:
            return float(scores["mean"])
        except (TypeError, ValueError):
            return None
    # Old shape used flat keys; fall back.
    if "mean" in summary:
        try:
            return float(summary["mean"])
        except (TypeError, ValueError):
            return None
    return None


# -- the test --------------------------------------------------------


@unittest.skipUnless(_enabled()[0], _enabled()[1])
class TestCloudSftBeforeAfter(unittest.TestCase):
    """Train and prove the score actually improved."""

    def setUp(self) -> None:
        self._project_dir = Path(
            os.environ.get("LQH_BA_PROJECT_DIR")
            or os.path.expanduser(f"~/.lqh-e2e-ba-{int(time.time())}")
        )
        self._project_dir.mkdir(parents=True, exist_ok=True)

        # On-disk layout: train + eval ChatML + scorer.
        ds_train = self._project_dir / "datasets" / "train.parquet"
        ds_eval  = self._project_dir / "evals" / "translation" / "data.parquet"
        _build_chatml_parquet(ds_train, _TRAIN_PAIRS)
        _build_chatml_parquet(ds_eval,  _EVAL_PAIRS)
        scorer_path = self._project_dir / "scorers" / "translation.md"
        scorer_path.parent.mkdir(parents=True, exist_ok=True)
        scorer_path.write_text(_SCORER_MD)

        self._run_dir = self._project_dir / "runs" / "ba_sweep"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._sweep_config = _build_sweep_config(
            "datasets/train.parquet",
            "evals/translation/data.parquet",
            "scorers/translation.md",
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
            except Exception as exc:
                logger.warning("cleanup teardown failed: %s", exc)
        print(f"\nBefore/after E2E artifacts at: {self._project_dir}")

    def test_training_improves_score(self):
        start = time.monotonic()

        # ---- Phase 1: baseline via the LFM router ----
        print("=" * 60, flush=True)
        print(f"Phase 1: baseline scoring {BASE_MODEL_LFM} on "
              f"{len(_EVAL_PAIRS)} held-out pairs", flush=True)
        print("=" * 60, flush=True)
        baseline_dir = self._project_dir / "baseline"
        mean_pre = asyncio.run(_baseline_score(
            self._project_dir / "evals" / "translation" / "data.parquet",
            self._project_dir / "scorers" / "translation.md",
            baseline_dir,
        ))
        self.assertIsNotNone(
            mean_pre,
            "baseline scoring produced no summary — check the run_scoring "
            "call (model id, scorer path, token)."
        )
        print(f"  → mean_pre = {mean_pre:.3f}", flush=True)

        # ---- Phase 2: cloud SFT-sweep ----
        print("=" * 60, flush=True)
        print(f"Phase 2: train_sft_sweep (1 config, 3 epochs) — "
              f"timeout {TRAIN_TIMEOUT_SEC}s", flush=True)
        print("=" * 60, flush=True)
        # Retry the submit on transient 502s. The Cloudflare edge in
        # front of api.lqh.ai sometimes returns a generic gateway
        # error during a slow sandbox-create call; the
        # backend's cloud_jobs row was already inserted (the
        # reconciler will reap it if no sandbox materialises), so a
        # retry just creates a NEW job. We tolerate up to 3 tries
        # over ~30 s before giving up.
        from lqh.remote.cloud import CloudError

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                self._job_id = asyncio.run(self._backend.submit_run(
                    str(self._run_dir),
                    self._sweep_config,
                    module="lqh.train.sweep",
                ))
                break
            except CloudError as exc:
                last_err = exc
                # 502 / 504 are the transient-gateway flavors; retry
                # those. Anything else (400 invalid config, 401 auth,
                # 402 balance) is a real error and shouldn't loop.
                msg = str(exc)
                if not (msg.startswith("502") or msg.startswith("504")):
                    raise
                print(f"  submit attempt {attempt + 1}/3 hit {msg[:60]}; "
                      f"retrying in {2 ** attempt}s...", flush=True)
                time.sleep(2 ** attempt)
        else:
            raise AssertionError(
                f"submit_run failed after 3 retries: {last_err}"
            )
        self.assertTrue(self._job_id, "submit_run returned empty job_id")
        print(f"  submitted cloud sweep: {self._job_id}", flush=True)

        # Drive sync_progress until terminal.
        terminal_states = {"completed", "failed", "cancelled"}
        deadline = start + TRAIN_TIMEOUT_SEC
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
                    print(
                        f"  status → {last_status} "
                        f"(elapsed {time.monotonic() - start:.0f}s)",
                        flush=True,
                    )
                if last_status in terminal_states:
                    break
            time.sleep(POLL_INTERVAL_SEC)
        else:
            self.fail(
                f"sweep {self._job_id} did not reach terminal within "
                f"{TRAIN_TIMEOUT_SEC}s (last status: {last_status})"
            )
        self.assertEqual(last_status, "completed",
                         f"sweep ended non-success: {last_status}")

        # Grace period for publish.
        print("  waiting 60s for publish step ...", flush=True)
        time.sleep(60)

        # ---- Phase 3: pull lineage, find real_metric ----
        print("=" * 60, flush=True)
        print("Phase 3: extracting real_metric from lineage", flush=True)
        print("=" * 60, flush=True)
        token = get_token()
        project_id = self._project_dir.name
        with httpx.Client(base_url=api_root(), timeout=30.0) as client:
            lineage_resp = client.get(
                f"/v1/projects/{project_id}/lineage",
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(lineage_resp.status_code, 200, lineage_resp.text)
        lineage_rows = lineage_resp.json().get("lineage", [])
        print(f"  {len(lineage_rows)} lineage rows", flush=True)

        # The eval-of-best in lqh.train.sweep produces a rollout
        # artifact at the run-dir level whose lineage row carries the
        # real_metric. Find the most recent one with a non-null
        # real_metric.
        mean_post: float | None = None
        for row in lineage_rows:
            rm = row.get("real_metric")
            if rm and isinstance(rm, dict) and "value" in rm:
                mean_post = float(rm["value"])
                print(
                    f"  found lineage row with real_metric: "
                    f"name={rm.get('name')}, value={mean_post:.3f}, "
                    f"artifact={row.get('artifact_id')}",
                    flush=True,
                )
                break

        if mean_post is None:
            # Fall back to pulling the eval_result artifact and
            # extracting its summary.scores.mean — sweep.py's
            # eval-of-best writes the same data via two paths and
            # we want the test to be resilient to whichever lands.
            print(
                "  no lineage row carried real_metric; falling back to "
                "eval_result artifact",
                flush=True,
            )
            with httpx.Client(base_url=api_root(), timeout=30.0) as client:
                arts = client.get(
                    f"/v1/projects/{project_id}/artifacts?kind=eval_result",
                    headers={"Authorization": f"Bearer {token}"},
                ).json().get("artifacts", [])
                # newest-first; the sweep's eval-of-best is the last
                # eval_result published.
                for art in arts:
                    url = client.get(
                        f"/v1/artifacts/{art['id']}/url",
                        headers={"Authorization": f"Bearer {token}"},
                    ).json()["url"]
                    body = httpx.get(url, timeout=30.0).text
                    try:
                        payload = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    summary = payload.get("summary")
                    mean_post = _extract_mean(summary) if isinstance(summary, dict) else None
                    if mean_post is not None:
                        print(
                            f"  pulled eval_result artifact {art['id']}: "
                            f"mean_post={mean_post:.3f}",
                            flush=True,
                        )
                        break

        self.assertIsNotNone(
            mean_post,
            "no real_metric on any lineage row AND no eval_result artifact "
            "with a parseable summary. Inline scoring in lqh.train.sweep "
            "didn't produce the eval-of-best metric — check stdout.log "
            "from the sandbox.",
        )

        # ---- assertion ----
        delta = mean_post - mean_pre
        print("=" * 60, flush=True)
        print(f"  mean_pre  = {mean_pre:.3f}", flush=True)
        print(f"  mean_post = {mean_post:.3f}", flush=True)
        print(f"  Δ         = {delta:+.3f}  "
              f"(required ≥ +{MIN_IMPROVEMENT})", flush=True)
        print("=" * 60, flush=True)

        self.assertGreater(
            mean_post, mean_pre + MIN_IMPROVEMENT,
            f"SFT did not improve the score by ≥ {MIN_IMPROVEMENT}: "
            f"mean_pre={mean_pre:.3f}, mean_post={mean_post:.3f}, "
            f"Δ={delta:+.3f}. Either training under-fit, the judge is "
            f"noisy on this rubric, or the eval set drift is too high. "
            f"Inspect: baseline at {baseline_dir}, run at {self._run_dir}",
        )

        elapsed = time.monotonic() - start
        print(f"\n✅ training improved score from {mean_pre:.3f} → "
              f"{mean_post:.3f} (Δ {delta:+.3f}) in {elapsed:.0f}s",
              flush=True)


if __name__ == "__main__":
    unittest.main()
