"""End-to-end cloud fine-tuning on a real HF dataset, with API-based
zero-shot baseline, post-train eval and a local checkpoint fetch.

Heavier counterpart to ``test_cloud_sft_smoke.py``. Where the smoke
test verifies the round-trip with 8 hand-written pairs, this one
exercises:

  - Real HF dataset download (mlech26l/shell-helper, 39,844 rows on
    HF; we slice 30 train + 8 eval).
  - **API-based zero-shot baseline** locally: call ``/v1/chat/completions``
    for each eval prompt to get a "no fine-tuning" reference. Stands
    in for "HF eval in the cloud" since lqh.train has no eval-only
    run type today; the substitute makes the comparison machine-local
    instead of cloud-side.
  - SFT in the cloud with ``eval_on_checkpoints=True`` so the trainer
    runs inference on the eval set at each save step and emits
    ``checkpoints/step_N/predictions.parquet``.
  - publish.py uploads all per-checkpoint artifacts (predictions,
    eval requests) plus the final ``model/`` checkpoint tar.
  - The test fetches both a per-checkpoint predictions.parquet AND
    the final checkpoint tar to local paths, then compares
    baseline vs trained predictions side-by-side.

Skipped unless ``LQH_E2E=1`` is set and a token is resolvable —
same gate as the smoke test (this one spends more real money).

Time + cost budget:
  - Wall time: ~6-10 min cold (HF model download + 1 epoch over ~30
    samples + per-checkpoint eval). ~2-3 min warm.
  - GPU cost: ~$0.50-$1.00 per run on an A100-40GB.

Usage:
    LQH_E2E=1 python -m pytest lqh_py/tests/function/test_cloud_sft_hf_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tarfile
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any

from lqh.artifacts import ArtifactHandle, BackendArtifactStore
from lqh.auth import get_token
from lqh.client import chat_with_retry, create_client
from lqh.config import default_api_base_url
from lqh.remote.backend import RemoteConfig
from lqh.remote.cloud import CloudBackend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


DATASET_URL = (
    "https://huggingface.co/datasets/mlech26l/shell-helper/resolve/main/"
    "data/shell_helper_dataset_many_opt.parquet?download=true"
)

# Keep these tight enough to stay under the cost budget but big enough
# that a real HF dataset round-trip is exercised (smoke uses 8 pairs).
TRAIN_SAMPLES = 30
EVAL_SAMPLES = 8

# LFM2-1.2B is the smallest documented base and is what the per-project
# cache is most likely to have warm.
MODEL_ID = "LiquidAI/LFM2-1.2B"

# Wall-clock cap. Cold first-ever run can take ~10 min for HF model
# download + 1 epoch + per-checkpoint eval inference. Warm runs land
# ~3 min. 20 min cap lets a flaky CI box finish.
E2E_TIMEOUT_SEC = int(os.environ.get("LQH_CLOUD_HF_TIMEOUT", "1200"))

POLL_INTERVAL_SEC = 2.0

# Grace window we keep polling after the trainer's ``completed``
# sentinel arrives, so publish.py has time to finish uploading the
# checkpoint tar and register it. Generous because the trainer's
# completed-status sentinel fires BEFORE the launcher's publish step
# runs — and publish has to tar the merged 1.2B model (~2.4 GB in
# bf16) on a single sandbox CPU, then upload to R2. Cold paths land
# at 3-6 minutes; 600s is a safety margin.
PUBLISH_GRACE_SEC = float(os.environ.get("LQH_CLOUD_HF_PUBLISH_GRACE", "600"))

# Model the API-baseline goes to. "small" lands on the cheapest pool;
# the baseline is a wiring check, not a quality bake-off — a frontier
# model would dominate a 1-epoch LoRA on 30 samples anyway.
BASELINE_MODEL = "small"

# Hard cap on response length for the baseline. Shell-helper answers
# are short (a one-liner + a small bash block); 256 tokens is plenty.
BASELINE_MAX_TOKENS = 256

# Shell-helper rows in HF are user → assistant only, no system turn.
# The baseline matches that shape so it stays comparable to what the
# trained model sees during eval_on_checkpoints. (No system prompt =
# zero-shot in the "you only told the model what to do via examples"
# sense — the fine-tune is the only "teaching".)


def _e2e_enabled() -> tuple[bool, str]:
    if os.environ.get("LQH_E2E") != "1":
        return False, "LQH_E2E != 1"
    if get_token() is None:
        return False, "no lqh auth token (run /login or set LQH_DEBUG_API_KEY)"
    base = default_api_base_url()
    if not base or ("lqh" not in base.lower() and "localhost" not in base.lower()):
        return False, f"LQH_BASE_URL not set or unexpected: {base!r}"
    return True, ""


def _download_hf_dataset(cache_dir: Path) -> Path:
    """Fetch shell_helper_dataset_many_opt.parquet from HF.

    Cached at ``cache_dir/shell_helper_raw.parquet`` across test runs
    so the second run doesn't re-download ~1MB.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = cache_dir / "shell_helper_raw.parquet"
    if raw.exists() and raw.stat().st_size > 0:
        logger.info("HF dataset cache hit: %s (%.0f KB)",
                    raw, raw.stat().st_size / 1024)
        return raw
    logger.info("Downloading HF dataset from %s ...", DATASET_URL)
    urllib.request.urlretrieve(DATASET_URL, raw)
    logger.info("Downloaded %.0f KB", raw.stat().st_size / 1024)
    return raw


def _prepare_train_eval(raw: Path, project_dir: Path) -> tuple[Path, Path]:
    """Slice the raw HF parquet into train + eval parquets under the
    project dir, both in the ChatML-string format the SFT trainer
    expects (one row = one ``messages`` JSON string).

    Returns (train_relpath, eval_relpath) as project-relative posix
    strings so the trainer config can point at them through the
    bundle's manifest mechanism.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(str(raw))
    total = len(table)
    if total < EVAL_SAMPLES + TRAIN_SAMPLES:
        raise unittest.SkipTest(
            f"HF dataset too small: have {total}, need "
            f"{EVAL_SAMPLES + TRAIN_SAMPLES}"
        )

    def slice_to_chatml(t: pa.Table, out_path: Path) -> int:
        messages_col = t.column("messages")
        rows: list[str] = []
        for i in range(len(t)):
            raw_row = messages_col[i].as_py()
            if isinstance(raw_row, list) and raw_row and isinstance(raw_row[0], dict):
                rows.append(json.dumps(raw_row))
            elif isinstance(raw_row, list):
                rows.append(json.dumps([dict(m) for m in raw_row]))
            else:
                rows.append(json.dumps(raw_row))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {"messages": rows},
                schema=pa.schema([pa.field("messages", pa.string())]),
            ),
            out_path,
        )
        return len(t)

    eval_rel = Path("datasets/hf_eval/data.parquet")
    train_rel = Path("datasets/hf_train/data.parquet")
    n_eval = slice_to_chatml(
        table.slice(0, EVAL_SAMPLES),
        project_dir / eval_rel,
    )
    n_train = slice_to_chatml(
        table.slice(EVAL_SAMPLES, TRAIN_SAMPLES),
        project_dir / train_rel,
    )
    logger.info("dataset prepared: train=%d, eval=%d", n_train, n_eval)
    return train_rel, eval_rel


def _load_eval_prompts(eval_parquet: Path) -> list[tuple[list[dict[str, str]], str]]:
    """Read the eval parquet and return (prompt_messages, gold_answer)
    pairs. ``prompt_messages`` is the conversation with the final
    assistant turn stripped — what we feed to both the API baseline
    and (effectively) what the trainer feeds itself during
    eval_on_checkpoints.
    """
    import pyarrow.parquet as pq

    out: list[tuple[list[dict[str, str]], str]] = []
    table = pq.read_table(str(eval_parquet))
    msgs_col = table.column("messages")
    for i in range(len(table)):
        conv = json.loads(msgs_col[i].as_py())
        if not conv or conv[-1].get("role") != "assistant":
            continue
        prompt = conv[:-1]
        gold = conv[-1].get("content", "")
        out.append((prompt, gold))
    return out


async def _run_api_baseline(
    eval_prompts: list[tuple[list[dict[str, str]], str]],
    *,
    api_base: str,
    token: str,
) -> list[dict[str, Any]]:
    """Run zero-shot baseline inference against the LQH API in parallel.

    Returns one row per eval prompt with the prompt, gold answer, and
    the API's response. Used to compare against the fine-tuned
    model's predictions.parquet fetched from R2.
    """
    client = create_client(token, base_url=api_base)
    sem = asyncio.Semaphore(4)  # be polite to the API rate limit

    async def one(idx: int, prompt: list[dict[str, str]], gold: str) -> dict[str, Any]:
        async with sem:
            try:
                resp = await chat_with_retry(
                    client,
                    model=BASELINE_MODEL,
                    messages=prompt,
                    max_tokens=BASELINE_MAX_TOKENS,
                    temperature=0.0,
                )
                content = (resp.choices[0].message.content or "") if resp.choices else ""
            except Exception as exc:
                content = f"[baseline error: {exc}]"
            return {
                "sample_index": idx,
                "prompt": json.dumps(prompt),
                "gold": gold,
                "baseline": content,
            }

    return await asyncio.gather(*(
        one(i, p, g) for i, (p, g) in enumerate(eval_prompts)
    ))


def _build_sft_config(train_rel: Path, eval_rel: Path) -> dict[str, Any]:
    """SFT config: 1 epoch, LoRA, eval_on_checkpoints so the trainer
    emits predictions at each save step (=> per-checkpoint
    predictions.parquet on R2 + final eval after training)."""
    return {
        "type": "sft",
        "base_model": MODEL_ID,
        "dataset": str(train_rel),
        "eval_dataset": str(eval_rel),
        "eval_on_checkpoints": True,
        "lora": {
            "enabled": True,
            # Don't merge the adapter into the base model at save
            # time. Merged 1.2B = ~2.4 GB on disk, and publish.py's
            # tar of that on the cloud sandbox hits a memory cap and gets
            # SIGKILL'd (exit 137). Saving the adapter alone keeps
            # the published checkpoint at ~tens of MB. Downstream
            # consumers reload via PeftModel.from_pretrained.
            "merge": False,
            "r": 16,
            "alpha": 32,
            "dropout": 0.02,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj", "out_proj", "w1", "w2", "w3",
            ],
        },
        "training": {
            "num_epochs": 1,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.1,
            "logging_steps": 1,
            # The SFT trainer needs eval_split_ratio big enough that
            # `split_train_eval` clears its min_eval=10 floor, or it
            # silently returns ([], []) and `has_eval` flips to
            # False — which means NO eval_strategy, NO save_strategy,
            # and NO on_save callbacks (so eval_on_checkpoints is a
            # no-op too). With 30 train samples, 0.4 → 12 eval ≥ 10.
            "eval_split_ratio": 0.4,
            # 30 - 12 = 18 train / batch 2 = 9 steps. eval_steps=4
            # fires eval at steps 4 + 8 → two predictions.parquet
            # files plus two eval_loss rows for a clean trend.
            "save_steps": 4,
            "eval_steps": 4,
            "gradient_checkpointing": True,
            "bf16": True,
            "max_seq_length": 512,
            "dataloader_num_workers": 0,
        },
        # The bundle helper resolves these keys → file paths to tar in.
        "manifest": ["dataset", "eval_dataset"],
    }


@unittest.skipUnless(_e2e_enabled()[0], _e2e_enabled()[1])
class TestCloudSftHFE2E(unittest.TestCase):
    """Cloud fine-tune on a real HF dataset with checkpoint fetch."""

    def setUp(self) -> None:
        # Project dir is unique per run so R2 prefixes don't collide
        # across repeated invocations. Keep cache_dir stable so the
        # HF parquet only downloads once.
        ts = int(time.time())
        self._project_dir = Path(
            os.environ.get("LQH_E2E_PROJECT_DIR")
            or os.path.expanduser(f"~/.lqh-e2e-cloud-hf-{ts}")
        )
        self._project_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = Path(os.path.expanduser("~/.lqh-e2e-cache"))

        self._run_dir = self._project_dir / "runs" / "hf_sft"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        raw = _download_hf_dataset(self._cache_dir)
        train_rel, eval_rel = _prepare_train_eval(raw, self._project_dir)
        self._config = _build_sft_config(train_rel, eval_rel)

        cfg = RemoteConfig(
            name="cloud",
            type="cloud",
            hostname="api.lqh.ai",
            remote_root="cloud:lqh",
        )
        self._backend = CloudBackend(cfg, self._project_dir)
        self._job_id: str | None = None

    def tearDown(self) -> None:
        if self._job_id:
            try:
                asyncio.run(self._backend.teardown(self._job_id))
            except Exception as exc:
                logger.warning("cleanup teardown failed: %s", exc)
        # Preserve artifacts for inspection; cache_dir is reusable.
        print(f"\nE2E artifacts preserved at: {self._project_dir}")

    def test_hf_sft_with_checkpoint_fetch(self):
        start = time.monotonic()

        # ---- 0. zero-shot API baseline on the eval slice ----
        # This stands in for "HF eval before training": instead of
        # spinning up a cloud sandbox to run the base model, we just
        # call the LQH API with the same prompts. Cheap (~$0.001),
        # fast (~5-10s), and the result is directly comparable to the
        # predictions.parquet that publish.py uploads after training.
        token = get_token()
        self.assertIsNotNone(token)
        eval_parquet = (
            self._project_dir / self._config["eval_dataset"]
        )
        eval_prompts = _load_eval_prompts(eval_parquet)
        self.assertEqual(
            len(eval_prompts), EVAL_SAMPLES,
            f"expected {EVAL_SAMPLES} eval prompts, got {len(eval_prompts)}"
        )
        logger.info("running API baseline (model=%s) on %d eval prompts...",
                    BASELINE_MODEL, len(eval_prompts))
        baseline_rows = asyncio.run(_run_api_baseline(
            eval_prompts,
            api_base=default_api_base_url(),
            token=token,
        ))
        baseline_path = self._run_dir / "baseline_predictions.jsonl"
        with baseline_path.open("w") as fh:
            for row in baseline_rows:
                fh.write(json.dumps(row) + "\n")
        non_empty_baseline = sum(
            1 for r in baseline_rows
            if r["baseline"] and not r["baseline"].startswith("[baseline error:")
        )
        self.assertGreater(
            non_empty_baseline, 0,
            "API baseline returned no successful responses — "
            "the rest of the test won't have anything to compare against"
        )
        logger.info("API baseline: %d/%d non-empty responses",
                    non_empty_baseline, len(baseline_rows))

        # ---- 1. submit ----
        self._job_id = asyncio.run(self._backend.submit_run(
            str(self._run_dir),
            self._config,
            module="lqh.train",
        ))
        self.assertTrue(self._job_id, "submit_run returned empty job_id")
        logger.info("submitted cloud HF job: %s", self._job_id)

        self.assertTrue((self._run_dir / "remote_job.json").exists())
        self.assertTrue((self._run_dir / "cloud_state.json").exists())

        # ---- 2. drive SSE until terminal ----
        terminal = {"completed", "failed"}
        deadline = start + E2E_TIMEOUT_SEC
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
                if last_status in terminal:
                    break
            time.sleep(POLL_INTERVAL_SEC)
        else:
            self.fail(
                f"cloud HF job {self._job_id} did not reach terminal "
                f"within {E2E_TIMEOUT_SEC}s (last status: {last_status})"
            )

        self.assertEqual(last_status, "completed",
                         f"job ended non-success: {last_status}")

        # status.json mirrored to disk
        status = json.loads((self._run_dir / "status.json").read_text())
        self.assertEqual(status["state"], "completed", status)

        # ---- 3. progress + training step rows ----
        progress_lines = [
            json.loads(line) for line in
            (self._run_dir / "progress.jsonl").read_text().strip().splitlines()
            if line.strip()
        ]
        self.assertTrue(progress_lines, "progress.jsonl is empty")
        step_rows = [r for r in progress_lines if "step" in r and "status" not in r]
        self.assertTrue(step_rows, f"no step rows in progress (got {len(progress_lines)})")
        logger.info("received %d step rows; final step %s",
                    len(step_rows), step_rows[-1].get("step"))

        # Eval-loss rows are emitted by HF Trainer at eval_steps; they
        # carry an 'eval_loss' key. Asserting they exist is the
        # cheapest proof that eval-after-training actually ran in the
        # cloud sandbox.
        eval_rows = [r for r in progress_lines if "eval_loss" in r]
        self.assertTrue(
            eval_rows,
            f"no eval_loss rows in progress.jsonl — eval-on-checkpoints "
            f"did not fire (eval_steps={self._config['training']['eval_steps']}, "
            f"step rows={len(step_rows)})"
        )
        logger.info("received %d eval rows; final eval_loss=%.4f",
                    len(eval_rows), float(eval_rows[-1]["eval_loss"]))

        # ---- 4. wait for publish to register artifacts ----
        # The trainer's completed sentinel arrives BEFORE publish
        # finishes uploading the checkpoint tar. Poll the server-side
        # artifacts list until a checkpoint shows up, capped by
        # PUBLISH_GRACE_SEC.
        project_id = self._project_dir.name
        store = BackendArtifactStore(token=token)

        publish_deadline = time.monotonic() + PUBLISH_GRACE_SEC
        checkpoint: ArtifactHandle | None = None
        predictions: list[ArtifactHandle] = []
        kinds_seen: set[str] = set()
        last_count = -1
        publish_start = time.monotonic()
        while time.monotonic() < publish_deadline:
            handles = asyncio.run(store.list_for_project(project_id, limit=200))
            kinds_seen = {h.kind for h in handles}
            # Log when new artifacts land so an operator watching the
            # test in -s mode can see publish making progress instead
            # of staring at a silent 10-minute spinner.
            if len(handles) != last_count:
                last_count = len(handles)
                logger.info(
                    "  publish progress: %d artifacts so far "
                    "(kinds=%s, elapsed=%.0fs)",
                    len(handles), sorted(kinds_seen),
                    time.monotonic() - publish_start,
                )
            ckpts = [h for h in handles if h.kind == "checkpoint"]
            if ckpts:
                # Largest checkpoint = the final merged model dir
                # (per-step LoRA adapter dirs are tiny relative to the
                # full safetensors output).
                checkpoint = max(ckpts, key=lambda h: h.size_bytes)
                predictions = [h for h in handles if h.kind == "predictions"]
                # Once the checkpoint is up, give predictions a short
                # tail-window — they're tiny parquets and typically
                # finish within seconds of the big tar. Don't wait for
                # them past the publish deadline though.
                tail_deadline = min(
                    publish_deadline,
                    time.monotonic() + 30.0,
                )
                while not predictions and time.monotonic() < tail_deadline:
                    time.sleep(POLL_INTERVAL_SEC)
                    handles = asyncio.run(
                        store.list_for_project(project_id, limit=200))
                    predictions = [h for h in handles if h.kind == "predictions"]
                break
            time.sleep(POLL_INTERVAL_SEC)

        self.assertIsNotNone(
            checkpoint,
            f"no checkpoint artifact published within {PUBLISH_GRACE_SEC}s "
            f"(kinds seen: {sorted(kinds_seen)})"
        )
        logger.info("checkpoint artifact: id=%s size=%.1f MB",
                    checkpoint.id, checkpoint.size_bytes / 1e6)

        # eval_on_checkpoints=True should have produced at least one
        # predictions.parquet per checkpoint. Don't be strict about
        # the count — but at least one must exist.
        self.assertTrue(
            predictions,
            f"no predictions artifacts published — eval-on-checkpoints "
            f"did not emit eval predictions (kinds seen: {sorted(kinds_seen)})"
        )
        logger.info("predictions artifacts: %d (sizes: %s)",
                    len(predictions),
                    [h.size_bytes for h in predictions])

        # ---- 5. fetch the checkpoint to a local path ----
        # This is the user-visible deliverable: "I trained a model in
        # the cloud, now give me the weights on my laptop".
        local_tar = self._project_dir / "fetched" / "checkpoint.tar.gz"
        local_tar.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(store.download(checkpoint, local_tar))

        self.assertTrue(local_tar.exists(), "checkpoint tar not written")
        # A LoRA adapter for LFM2-1.2B with r=16 lands ~10-50 MB
        # (adapter weights + tokenizer + adapter_config.json). Bound
        # liberally; a merged model would be 100x larger but that's
        # also acceptable proof of a real artifact.
        self.assertGreater(
            local_tar.stat().st_size, 100_000,
            f"checkpoint tar suspiciously small: {local_tar.stat().st_size} bytes",
        )
        logger.info("fetched checkpoint to %s (%.1f MB)",
                    local_tar, local_tar.stat().st_size / 1e6)

        # ---- 6. verify the tar is a usable HF model dir ----
        extract_root = self._project_dir / "fetched" / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local_tar, "r:gz") as tar:
            members = tar.getnames()
            tar.extractall(extract_root)

        # publish.py rooted the tar at the source dir's basename. With
        # lora.merge=False the dir contains adapter_config.json (peft
        # adapter manifest) + adapter_model.safetensors. With merge=
        # True it'd be a full HF model dir with config.json + model
        # safetensors. Accept either layout.
        adapter_configs = list(extract_root.rglob("adapter_config.json"))
        full_configs = list(extract_root.rglob("config.json"))
        self.assertTrue(
            adapter_configs or full_configs,
            f"no adapter_config.json or config.json in extracted "
            f"checkpoint (members sample: {members[:10]})"
        )

        # Weight files: full merge writes model*.safetensors;
        # adapter-only writes adapter_model.safetensors. Either is
        # proof the trainer produced something downloadable.
        weight_files = (
            list(extract_root.rglob("*.safetensors"))
            + list(extract_root.rglob("adapter_model.bin"))
        )
        self.assertTrue(
            weight_files,
            f"no weight files in checkpoint "
            f"(members sample: {members[:20]})"
        )

        # ---- 6b. loader handoff: prove the fetched dir is usable ----
        # The user-visible deliverable is "I trained in the cloud and
        # now PeftModel.from_pretrained-equivalent loads on my laptop".
        # load_for_inference auto-detects adapter vs merged and dispatches.
        # Gated by LQH_E2E_SKIP_LOAD so CI machines without GPU/RAM can
        # opt out of the ~1-2 GB base-model download.
        if os.environ.get("LQH_E2E_SKIP_LOAD") == "1":
            logger.info("LQH_E2E_SKIP_LOAD=1; skipping loader handoff step")
        else:
            # The tar's top-level dir is either ``model`` or ``model-lora``.
            # Pick whichever has the (adapter|full) config we found above.
            handoff_dir = (
                adapter_configs[0].parent if adapter_configs
                else full_configs[0].parent
            )
            logger.info("loader handoff: loading %s ...", handoff_dir)

            from lqh.train.load_model import load_for_inference
            import torch

            loaded_model, loaded_tok = load_for_inference(
                str(handoff_dir),
                dtype=torch.float32,  # CPU-safe
                device_map="cpu",
            )
            self.assertIsNotNone(loaded_model, "loader returned None for model")
            self.assertIsNotNone(loaded_tok, "loader returned None for tokenizer")

            # One forward pass to prove the weights actually loaded into
            # the graph (not just instantiated as empty shells).
            inputs = loaded_tok("hello", return_tensors="pt")
            with torch.no_grad():
                out = loaded_model(**inputs)
            self.assertEqual(
                out.logits.ndim, 3,
                f"unexpected logits shape: {tuple(out.logits.shape)}"
            )
            logger.info(
                "loader handoff OK (logits shape %s)",
                tuple(out.logits.shape),
            )

        # ---- 7. fetch trained-model predictions, compare vs baseline ----
        # publish.py uploads one predictions.parquet per checkpoint
        # step. Take the largest (= most rows OR latest run; both pick
        # the post-training one for a single-epoch SFT). The local
        # comparison stands in for the missing "HF eval in the cloud"
        # before-vs-after view.
        trained_path = self._project_dir / "fetched" / "predictions_trained.parquet"
        trained_path.parent.mkdir(parents=True, exist_ok=True)
        chosen_pred = max(predictions, key=lambda h: h.size_bytes)
        asyncio.run(store.download(chosen_pred, trained_path))
        self.assertTrue(trained_path.exists())

        import pyarrow.parquet as pq
        trained_table = pq.read_table(str(trained_path))
        self.assertIn("messages", trained_table.column_names,
                      f"predictions parquet missing 'messages' column: "
                      f"{trained_table.column_names}")
        trained_rows = trained_table.column("messages").to_pylist()
        self.assertEqual(
            len(trained_rows), len(baseline_rows),
            f"trained predictions row count ({len(trained_rows)}) "
            f"differs from baseline ({len(baseline_rows)}) — eval set "
            f"shape mismatch"
        )

        # Pull the assistant turn out of the trained predictions, line
        # it up with baseline + gold, write a comparison jsonl, and
        # log a few examples. We DON'T assert "trained beats baseline"
        # — a 1-epoch LoRA on 30 samples vs a frontier API model is
        # noise. We assert both sides produced non-empty answers, and
        # surface the deltas so a human reviewer can eyeball quality.
        compare_path = self._run_dir / "baseline_vs_trained.jsonl"
        non_empty_trained = 0
        avg_baseline_len = 0
        avg_trained_len = 0
        with compare_path.open("w") as fh:
            for i, (base, trained_msgs_json) in enumerate(zip(baseline_rows, trained_rows)):
                trained_conv = json.loads(trained_msgs_json)
                trained_answer = next(
                    (m["content"] for m in reversed(trained_conv)
                     if m.get("role") == "assistant"),
                    "",
                )
                if trained_answer:
                    non_empty_trained += 1
                avg_baseline_len += len(base["baseline"])
                avg_trained_len += len(trained_answer)
                fh.write(json.dumps({
                    "sample_index": i,
                    "gold": base["gold"],
                    "baseline": base["baseline"],
                    "trained": trained_answer,
                }) + "\n")

        self.assertGreater(
            non_empty_trained, 0,
            f"all {len(trained_rows)} trained predictions were empty — "
            f"the LoRA fine-tune ran but produced no usable output"
        )
        logger.info(
            "comparison written to %s (baseline non-empty=%d/%d, "
            "trained non-empty=%d/%d, avg len baseline=%.0f trained=%.0f)",
            compare_path,
            non_empty_baseline, len(baseline_rows),
            non_empty_trained, len(trained_rows),
            avg_baseline_len / max(1, len(baseline_rows)),
            avg_trained_len / max(1, len(trained_rows)),
        )
        # Log a sample for eyeball quality check.
        for i in range(min(2, len(baseline_rows))):
            logger.info(
                "  sample %d:\n    GOLD:     %s\n    BASELINE: %s\n    TRAINED:  %s",
                i,
                baseline_rows[i]["gold"][:160],
                baseline_rows[i]["baseline"][:160],
                json.loads(trained_rows[i])[-1].get("content", "")[:160],
            )

        elapsed = time.monotonic() - start
        logger.info(
            "HF cloud SFT e2e completed in %.0fs "
            "(job=%s, steps=%d, evals=%d, ckpt=%.1fMB, "
            "preds=%d, baseline=%d/%d, trained=%d/%d)",
            elapsed, self._job_id, len(step_rows), len(eval_rows),
            checkpoint.size_bytes / 1e6, len(predictions),
            non_empty_baseline, len(baseline_rows),
            non_empty_trained, len(trained_rows),
        )


if __name__ == "__main__":
    unittest.main()
