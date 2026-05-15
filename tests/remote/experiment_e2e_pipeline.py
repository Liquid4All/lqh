"""Generic e2e harness: spec → datagen → filter → SFT → eval (→ DPO).

Driven by ``e2e_config.json`` in a per-task project under
``tests/e2e_projects/<task>/``. The same script handles JSON-constrained
tasks, tool-calling tasks (Phase 4), and open-ended tasks (Phase 3
adds DPO support); per-task differences are declared in the config.

Stage order (each skippable via ``--skip-<stage>``)::

    1. generate-train      — run the task's pipeline, write raw train parquet
    2. generate-eval       — same, eval split
    3. filter-train        — LLM-judge filter at config.filter_threshold
    4. filter-eval         — same, eval split
    5. baseline-eval       — zero-shot infer + score on the filtered eval
    6. sft                 — `lqh.train` subprocess with config.sft
    7. post-sft-eval       — same as baseline, against the trained checkpoint
    8. dpo                 — only if ``config.dpo`` is set (Phase 3+)
    9. post-dpo-eval       — only if dpo ran

Final report: a single table with baseline / post-SFT (/ post-DPO)
mean+median+deltas + PASS/WARN/FAIL verdict, written to
``tests/e2e_projects/<task>/runs/e2e_<timestamp>/report.md``.

Usage::

    python -m tests.remote.experiment_e2e_pipeline --task translation
    python -m tests.remote.experiment_e2e_pipeline --task translation \\
        --train-samples 8 --eval-samples 4 --skip-sft --skip-post-sft-eval

Designed to run on a GPU host (toka). Needs an lqh API token (in
``~/.lqh/config.json`` or ``LQH_API_KEY``) for the judge and the data
generation calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECTS_ROOT = REPO_ROOT / "tests" / "e2e_projects"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class TaskConfig:
    task_name: str
    config_name: str  # "default" | "dpo_only" | etc — used in exp_name
    project_dir: Path
    task_kind: str  # "json" | "tools" | "open"
    base_model: str
    train_samples: int
    eval_samples: int
    filter_threshold: float
    datagen_pipeline: Path
    system_prompt_path: Path | None
    schema_path: Path | None
    scorer_path: Path
    sft: dict[str, Any] | None  # None = skip SFT entirely (DPO from base_model)
    dpo: dict[str, Any] | None
    max_new_tokens: int
    regression_threshold: float

    # Loaded blobs
    system_prompt_text: str | None = None
    schema_envelope: dict[str, Any] | None = None


def _resolve_in_project(project_dir: Path, p: str | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else (project_dir / path).resolve()


def load_task_config(
    task: str,
    *,
    override_train: int | None,
    override_eval: int | None,
    config_name: str = "default",
) -> TaskConfig:
    project_dir = (PROJECTS_ROOT / task).resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(
            f"Task project not found: {project_dir}. "
            f"Existing tasks: {sorted(p.name for p in PROJECTS_ROOT.iterdir() if p.is_dir())}"
        )
    cfg_filename = "e2e_config.json" if config_name == "default" else f"e2e_config_{config_name}.json"
    cfg_path = project_dir / cfg_filename
    if not cfg_path.exists():
        existing = sorted(p.name for p in project_dir.glob("e2e_config*.json"))
        raise FileNotFoundError(
            f"Missing {cfg_path}. Available configs in this task: {existing}"
        )
    raw = json.loads(cfg_path.read_text())

    pipeline = _resolve_in_project(project_dir, raw["datagen_pipeline"])
    if pipeline is None or not pipeline.exists():
        raise FileNotFoundError(f"datagen_pipeline missing: {pipeline}")
    scorer = _resolve_in_project(project_dir, raw["scorer"])
    if scorer is None or not scorer.exists():
        raise FileNotFoundError(f"scorer missing: {scorer}")
    sys_prompt = _resolve_in_project(project_dir, raw.get("system_prompt"))
    schema = _resolve_in_project(project_dir, raw.get("schema"))

    cfg = TaskConfig(
        task_name=task,
        config_name=config_name,
        project_dir=project_dir,
        task_kind=raw["task_kind"],
        base_model=raw["base_model"],
        train_samples=override_train or raw["train_samples"],
        eval_samples=override_eval or raw["eval_samples"],
        filter_threshold=raw.get("filter_threshold", 7.0),
        datagen_pipeline=pipeline,
        system_prompt_path=sys_prompt,
        schema_path=schema,
        scorer_path=scorer,
        sft=raw.get("sft"),  # None = skip SFT entirely
        dpo=raw.get("dpo"),
        max_new_tokens=raw.get("max_new_tokens", 8192),
        regression_threshold=raw.get("regression_threshold", -0.5),
    )

    if cfg.system_prompt_path is not None:
        cfg.system_prompt_text = cfg.system_prompt_path.read_text(encoding="utf-8")
    if cfg.schema_path is not None:
        cfg.schema_envelope = json.loads(cfg.schema_path.read_text(encoding="utf-8"))

    if cfg.task_kind not in ("json", "tools", "open"):
        raise ValueError(
            f"unknown task_kind: {cfg.task_kind!r} "
            f"(expected one of: json, tools, open)"
        )
    if cfg.sft is None and cfg.dpo is None:
        raise ValueError(
            f"{cfg_filename}: at least one of 'sft' or 'dpo' must be set "
            f"(both null = nothing to train)"
        )
    # task_kind=tools: tools live PER-SAMPLE inside the pipeline-emitted
    # parquet (the engine extracts them into a "tools" column from
    # ChatMLMessage(..., tools=...)). The harness has nothing to wire —
    # lqh.infer threads them through apply_chat_template automatically
    # and the judge picks the tool-call-aware system prompt when it
    # sees [Tool Calls] blocks. We just don't pass response_format.
    return cfg


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class EvalScore:
    label: str
    mean: float
    median: float
    scored: int
    failed: int
    run_dir: Path


@dataclass
class FilterReport:
    label: str
    input_path: Path
    output_path: Path
    kept: int
    dropped: int
    mean: float


@dataclass
class DpoIterStat:
    iteration: int
    on_policy_mean: float
    on_policy_median: float
    held_out_mean: float | None = None
    held_out_median: float | None = None


# --------------------------------------------------------------------------
# Stage: generate
# --------------------------------------------------------------------------


async def stage_generate(
    cfg: TaskConfig,
    label: str,
    num_samples: int,
    output_dir: Path,
) -> Path:
    """Run the task's datagen pipeline. Returns the path to data.parquet."""
    from lqh.auth import get_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.engine import run_pipeline

    print(f"\n=== generate-{label}: {num_samples} samples → {output_dir.name} ===", flush=True)

    api_key = os.environ.get("LQH_API_KEY") or get_token() or ""
    if not api_key:
        raise RuntimeError("No lqh API key (set LQH_API_KEY or run `lqh /login`).")
    client = create_client(api_key, load_config().api_base_url)

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = {"n": 0}

    def _on_progress(done: int, total: int) -> None:
        if done >= completed["n"] + max(1, total // 20):
            completed["n"] = done
            print(f"  [{label}] {done}/{total} samples", flush=True)

    result = await run_pipeline(
        script_path=cfg.datagen_pipeline,
        num_samples=num_samples,
        output_dir=output_dir,
        client=client,
        on_progress=_on_progress,
    )
    parquet = output_dir / "data.parquet"
    if not parquet.exists():
        raise RuntimeError(f"datagen produced no parquet at {parquet}")
    print(f"  [{label}] generated {result.succeeded} ({result.failed} failed)", flush=True)
    return parquet


# --------------------------------------------------------------------------
# Stage: filter
# --------------------------------------------------------------------------


async def stage_filter(
    cfg: TaskConfig,
    label: str,
    input_parquet: Path,
    output_dir: Path,
) -> FilterReport:
    """Filter the dataset by LLM judge score. Returns FilterReport."""
    from lqh.auth import get_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_data_filter

    print(f"\n=== filter-{label}: threshold={cfg.filter_threshold} ===", flush=True)

    api_key = os.environ.get("LQH_API_KEY") or get_token() or ""
    client = create_client(api_key, load_config().api_base_url)

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = {"n": 0}

    def _on_progress(done: int, total: int) -> None:
        if done >= completed["n"] + max(1, total // 20):
            completed["n"] = done
            print(f"  [{label}] scored {done}/{total}", flush=True)

    result = await run_data_filter(
        input_path=input_parquet,
        scorer_path=cfg.scorer_path,
        output_dataset_dir=output_dir,
        client=client,
        threshold=cfg.filter_threshold,
        on_progress=_on_progress,
    )
    out_parquet = output_dir / "data.parquet"
    if not out_parquet.exists():
        raise RuntimeError(f"filter produced no parquet at {out_parquet}")
    kept_pct = (result.kept / max(result.total, 1)) * 100
    print(
        f"  [{label}] kept {result.kept}/{result.total} ({kept_pct:.1f}%, "
        f"mean {result.mean_score:.2f})",
        flush=True,
    )
    if result.kept < 0.5 * result.total:
        print(
            "  ⚠️  WARN: kept ratio is below 50%. "
            "Either the data is bad or the scorer is off-axis with the spec. "
            "Continuing, but consider investigating before relying on the result.",
            flush=True,
        )
    return FilterReport(
        label=label,
        input_path=input_parquet,
        output_path=out_parquet,
        kept=result.kept,
        dropped=result.dropped,
        mean=result.mean_score,
    )


# --------------------------------------------------------------------------
# Stage: eval (infer + score)
# --------------------------------------------------------------------------


def _build_infer_config(cfg: TaskConfig, model: str, eval_dataset: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "infer",
        "base_model": model,
        "dataset": str(eval_dataset.resolve()),
        "max_new_tokens": cfg.max_new_tokens,
        "manifest": ["base_model", "dataset"],
    }
    if cfg.system_prompt_text is not None:
        out["system_prompt"] = cfg.system_prompt_text
    if cfg.schema_envelope is not None:
        out["response_format"] = cfg.schema_envelope
    return out


def _run_subprocess(module: str, config_path: Path) -> None:
    print(f"\n=== {module} {config_path.relative_to(REPO_ROOT)} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", module, str(config_path)],
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{module} exited with status {proc.returncode}; see output above."
        )


async def stage_eval(
    cfg: TaskConfig,
    label: str,
    model: str,
    eval_dataset: Path,
    run_dir: Path,
    *,
    skip_inference: bool,
    judge_size: str,
    judge_concurrency: int,
) -> EvalScore:
    from lqh.auth import get_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import run_scoring

    print(f"\n=== eval-{label}: model={model} ===", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    predictions = run_dir / "predictions.parquet"

    if skip_inference and predictions.exists():
        print(f"  [{label}] reusing existing predictions at {predictions.relative_to(REPO_ROOT)}")
    else:
        config = _build_infer_config(cfg, model, eval_dataset)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        _run_subprocess("lqh.infer", config_path)

    if not predictions.exists():
        raise RuntimeError(f"[{label}] inference produced no predictions.parquet")

    api_key = os.environ.get("LQH_API_KEY") or get_token() or ""
    client = create_client(api_key, load_config().api_base_url)

    completed = {"n": 0}

    def _on_progress(done: int, total: int) -> None:
        if done >= completed["n"] + max(1, total // 20):
            completed["n"] = done
            print(f"  [{label}] judged {done}/{total}", flush=True)

    result = await run_scoring(
        dataset_path=predictions,
        scorer_path=cfg.scorer_path,
        output_dir=run_dir / "scoring",
        client=client,
        model_size=judge_size,
        concurrency=judge_concurrency,
        run_inference=False,
        on_progress=_on_progress,
    )
    print(
        f"  [{label}] mean={result.mean_score:.3f} median={result.median_score:.1f} "
        f"({result.scored} scored, {result.failed} failed)",
        flush=True,
    )
    return EvalScore(
        label=label,
        mean=result.mean_score, median=result.median_score,
        scored=result.scored, failed=result.failed,
        run_dir=run_dir,
    )


# --------------------------------------------------------------------------
# Stage: SFT
# --------------------------------------------------------------------------


def stage_sft(
    cfg: TaskConfig,
    train_dataset: Path,
    run_dir: Path,
    *,
    skip: bool,
) -> Path:
    """Train SFT subprocess. Returns path to the trained model dir."""
    print(f"\n=== sft on {train_dataset.relative_to(REPO_ROOT)} ===", flush=True)
    model_dir = run_dir / "model"
    if skip and model_dir.exists():
        print(f"  [sft] reusing existing checkpoint at {model_dir.relative_to(REPO_ROOT)}")
        return model_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "type": "sft",
        "base_model": cfg.base_model,
        "dataset": str(train_dataset.resolve()),
        "training": {
            "num_epochs": cfg.sft.get("num_epochs", 3),
            "learning_rate": cfg.sft.get("learning_rate", 2e-5),
            "per_device_batch_size": cfg.sft.get("per_device_batch_size", 4),
            "gradient_accumulation_steps": cfg.sft.get("gradient_accumulation_steps", 4),
            "max_seq_length": cfg.sft.get("max_seq_length", 2048),
        },
        "lora": {
            "enabled": True,
            "r": cfg.sft.get("lora_r", 32),
            "alpha": cfg.sft.get("lora_alpha", 64),
        },
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    _run_subprocess("lqh.train", config_path)

    if not model_dir.exists():
        raise RuntimeError(f"SFT produced no model at {model_dir}")
    return model_dir


# --------------------------------------------------------------------------
# Stage: DPO (on-policy, ping-pong with the subprocess)
# --------------------------------------------------------------------------


async def _wait_for_iter_artifacts(
    request_path: Path,
    predictions_path: Path,
    proc: "asyncio.subprocess.Process",
    *,
    iteration: int,
    poll_interval: float = 3.0,
    no_progress_timeout: float = 600.0,
    progress_dir: Path,
) -> None:
    """Block until both files exist, or the subprocess dies, or timeout.

    Resets the deadline whenever ``progress.jsonl`` is written to.
    Uses the file's mtime rather than the ``step`` field — across
    phase transitions (iter N held-out → iter N+1 generation), the
    step counter resets to 10 from a much higher value, and a
    monotonic step comparison would fail to reset the timer for the
    next ~step samples (causing iter N+1 to time out spuriously
    before any predictions are produced).
    """
    progress_file = progress_dir / "progress.jsonl"

    def _mtime() -> float:
        try:
            return progress_file.stat().st_mtime if progress_file.exists() else 0.0
        except OSError:
            return 0.0

    last_mtime = _mtime()
    deadline = time.monotonic() + no_progress_timeout
    while time.monotonic() < deadline:
        if request_path.exists() and predictions_path.exists():
            return
        mtime = _mtime()
        if mtime > last_mtime:
            last_mtime = mtime
            deadline = time.monotonic() + no_progress_timeout
        if proc.returncode is not None:
            # Subprocess exited; one final FS check, then declare failure.
            if request_path.exists() and predictions_path.exists():
                return
            raise RuntimeError(
                f"DPO subprocess exited (code {proc.returncode}) before "
                f"iteration {iteration} produced predictions. "
                f"See stderr above."
            )
        await asyncio.sleep(poll_interval)
    raise RuntimeError(
        f"Timeout waiting for iteration {iteration} predictions "
        f"(no progress for {no_progress_timeout:.0f}s)"
    )


async def _wait_for_eval_predictions(
    eval_preds_file: Path,
    ready_file: Path,
    proc: "asyncio.subprocess.Process",
    *,
    iteration: int,
    no_progress_timeout: float = 1800.0,
    poll_interval: float = 3.0,
) -> bool:
    """Block until per-iter eval_predictions appear, subprocess dies, or timeout.

    The DPO subprocess writes eval_predictions.parquet AFTER the DPO
    step completes (so model has been updated for this iter). The ready
    marker file is the trailing write — gating on it avoids racing on
    a half-written parquet.
    """
    start = time.monotonic()
    while time.monotonic() - start < no_progress_timeout:
        if eval_preds_file.exists() and ready_file.exists():
            return True
        if proc.returncode is not None:
            # Subprocess exited; one final FS check.
            return eval_preds_file.exists() and ready_file.exists()
        await asyncio.sleep(poll_interval)
    return False


async def stage_dpo(
    cfg: TaskConfig,
    starting_model: str,
    train_dataset: Path,
    eval_dataset: Path,
    baseline_mean: float,
    run_dir: Path,
    *,
    skip: bool,
    judge_size: str,
    judge_concurrency: int,
    early_abort_delta: float,
    trend_abort_delta: float,
) -> tuple[Path, list[DpoIterStat], float | None]:
    """Run on-policy DPO ping-pong. Returns (model_dir, per_iter_scores).

    ``starting_model`` is either the path to an SFT checkpoint or the
    base-model HF id (when SFT was skipped). The subprocess (``lqh.train``
    with ``type=on_policy_dpo``) generates predictions per iteration into
    ``iterations/iter_NNN/``; this main process polls for those files,
    runs the API judge, calls ``lqh.golden.generate_golden`` to assemble
    preference pairs, writes them back, and the subprocess proceeds with
    the DPO step. Loops until ``num_iterations`` complete.
    """
    from lqh.auth import get_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.golden import generate_golden
    from lqh.scoring import run_scoring

    assert cfg.dpo is not None, "stage_dpo called without a dpo config block"
    num_iterations = int(cfg.dpo.get("num_iterations", 3))
    beta = float(cfg.dpo.get("beta", 0.1))
    rejection_threshold = float(cfg.dpo.get("rejection_threshold", 6.0))
    learning_rate = float(cfg.dpo.get("learning_rate", 5e-6))
    # When SFT was skipped, allow the dpo block to specify lora/max_seq_length;
    # otherwise inherit them from the sft block. Either way we have defaults.
    sft = cfg.sft or {}
    def _hyperparam(name: str, default: Any) -> Any:
        return sft.get(name, cfg.dpo.get(name, default)) if cfg.dpo else sft.get(name, default)

    print(f"\n=== dpo: {num_iterations} iterations from {starting_model!r} ===", flush=True)
    print(f"  baseline mean={baseline_mean:.3f}, early-abort delta={early_abort_delta:+.3f}")
    model_dir = run_dir / "model"
    if skip and model_dir.exists():
        print(f"  [dpo] reusing existing checkpoint at {model_dir.relative_to(REPO_ROOT)}")
        return model_dir, []

    run_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale early-abort signal from a previous run in this dir.
    abort_signal = run_dir / "early_abort.json"
    if abort_signal.exists():
        abort_signal.unlink()

    api_key = os.environ.get("LQH_API_KEY") or get_token() or ""
    client = create_client(api_key, load_config().api_base_url)

    # ---- Score the chosen pool ONCE upfront ----
    # The chosen responses are the assistant turns in the training set
    # (assumed-good after Phase 3.5 filtering). Re-scoring them with the
    # same judge gives us per-pair gap = chosen - rejected later, which
    # is a better selector than absolute thresholds. The chosen text
    # doesn't change across DPO iters, so we score once and cache.
    #
    # The mean is the *ceiling* for SFT/DPO — the trained model can't
    # exceed the quality of its training data on this judge. We surface
    # it prominently so the user/agent can see whether DPO has any
    # headroom to work with vs the current model state.
    chosen_scores: list[float | None] | None = None
    chosen_pool_mean: float | None = None
    if cfg.dpo and "selection" in cfg.dpo:
        chosen_scores_path = run_dir / "chosen_scores.parquet"
        if chosen_scores_path.exists():
            print(f"  reusing cached chosen scores at {chosen_scores_path.relative_to(REPO_ROOT)}")
            import pyarrow.parquet as pq_mod
            tbl = pq_mod.read_table(str(chosen_scores_path))
            chosen_scores = [tbl.column("score")[i].as_py() for i in range(len(tbl))]
            non_null = [s for s in chosen_scores if s is not None and s > 0]
            if non_null:
                chosen_pool_mean = sum(non_null) / len(non_null)
        else:
            print(f"\n  scoring chosen pool ({train_dataset.name}) once upfront…", flush=True)
            chosen_score_dir = run_dir / "chosen_scoring"
            chosen_score_dir.mkdir(parents=True, exist_ok=True)
            chosen_eval = await run_scoring(
                dataset_path=train_dataset,
                scorer_path=cfg.scorer_path,
                output_dir=chosen_score_dir,
                client=client,
                model_size=judge_size,
                concurrency=judge_concurrency,
                run_inference=False,
            )
            chosen_pool_mean = chosen_eval.mean_score
            print(
                f"  chosen-pool mean={chosen_pool_mean:.3f} "
                f"median={chosen_eval.median_score:.1f} "
                f"({chosen_eval.scored} scored, {chosen_eval.failed} failed)",
                flush=True,
            )
            # results.parquet is the per-sample score table.
            import pyarrow.parquet as pq_mod
            results_path = chosen_score_dir / "results.parquet"
            if not results_path.exists():
                raise RuntimeError(f"chosen scoring produced no results.parquet at {results_path}")
            tbl = pq_mod.read_table(str(results_path))
            chosen_scores = [tbl.column("score")[i].as_py() for i in range(len(tbl))]
            # Persist a clean copy for resume / inspection. Also stash
            # the mean as a sidecar so the agent's training_status tool
            # can read it without recomputing.
            import pyarrow as pa_mod
            cs_table = pa_mod.table({"score": chosen_scores})
            pq_mod.write_table(cs_table, chosen_scores_path)
            (run_dir / "chosen_pool_summary.json").write_text(json.dumps({
                "mean": chosen_pool_mean,
                "median": chosen_eval.median_score,
                "scored": chosen_eval.scored,
                "failed": chosen_eval.failed,
            }, indent=2) + "\n")

        if chosen_pool_mean is not None:
            headroom = chosen_pool_mean - baseline_mean
            print(
                f"  📊 ceiling: chosen-pool mean={chosen_pool_mean:.3f} | "
                f"baseline={baseline_mean:.3f} | headroom={headroom:+.3f}",
                flush=True,
            )
    config: dict[str, Any] = {
        "type": "on_policy_dpo",
        "base_model": starting_model,
        "dataset": str(train_dataset.resolve()),
        "scorer": str(cfg.scorer_path.resolve()),
        "num_iterations": num_iterations,
        "dpo_beta": beta,
        "golden_source": "dataset",
        "rejection_threshold": rejection_threshold,
        "lora": {
            "enabled": True,
            "r": _hyperparam("lora_r", 32),
            "alpha": _hyperparam("lora_alpha", 64),
        },
        "training": {
            "learning_rate": learning_rate,
            "per_device_batch_size": cfg.dpo.get("per_device_batch_size", 2),
            "gradient_checkpointing": True,
            "bf16": True,
            "max_seq_length": _hyperparam("max_seq_length", 2048),
            "dataloader_num_workers": 0,
        },
    }
    if cfg.system_prompt_text is not None:
        config["system_prompt"] = cfg.system_prompt_text
    if cfg.schema_envelope is not None:
        config["response_format"] = cfg.schema_envelope
    # Held-out eval set — subprocess runs inference on it after each
    # DPO step, harness scores. Same dataset that stage_eval will use
    # at the end, so per-iter and final scores are comparable.
    config["held_out_eval_dataset"] = str(eval_dataset.resolve())
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print("  spawning lqh.train (DPO) subprocess…", flush=True)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "lqh.train", str(config_path),
        cwd=str(REPO_ROOT),
    )

    iter_stats: list[DpoIterStat] = []
    aborted = False
    try:
        for i in range(num_iterations):
            iter_name = f"iter_{i:03d}"
            iter_dir = run_dir / "iterations" / iter_name
            request_file = iter_dir / "iter_request.json"
            preds_file = iter_dir / "predictions.parquet"

            print(f"\n  --- iteration {i + 1}/{num_iterations} ---", flush=True)
            print("  waiting for predictions…", flush=True)
            await _wait_for_iter_artifacts(
                request_file, preds_file, proc,
                iteration=i, progress_dir=run_dir,
            )

            import pyarrow.parquet as pq
            n = pq.read_metadata(str(preds_file)).num_rows
            print(f"  got {n} predictions, scoring…", flush=True)

            score = await run_scoring(
                dataset_path=preds_file,
                scorer_path=cfg.scorer_path,
                output_dir=iter_dir,
                client=client,
                model_size=judge_size,
                concurrency=judge_concurrency,
                run_inference=False,
            )
            print(
                f"  iter {i} on-policy: mean={score.mean_score:.3f} "
                f"median={score.median_score:.1f}",
                flush=True,
            )

            await generate_golden(
                predictions_path=preds_file,
                scores_path=iter_dir / "results.parquet",
                dataset_path=str(train_dataset),
                config=config,
                client=client,
                output_dir=iter_dir,
                chosen_scores=chosen_scores,
            )
            prefs = iter_dir / "preferences.parquet"
            if not prefs.exists():
                raise RuntimeError(
                    f"generate_golden did not write preferences.parquet for "
                    f"iter {i}. Subprocess will block waiting for it."
                )
            n_prefs = pq.read_metadata(str(prefs)).num_rows
            print(f"  {n_prefs} preference pairs assembled, DPO step running…", flush=True)

            # ---- Per-iter held-out eval (after DPO step finishes) ----
            eval_preds_file = iter_dir / "eval_predictions.parquet"
            ready_file = iter_dir / "eval_predictions_ready.json"
            print("  waiting for held-out eval predictions…", flush=True)
            got_eval = await _wait_for_eval_predictions(
                eval_preds_file, ready_file, proc, iteration=i,
            )
            held_out_mean: float | None = None
            held_out_median: float | None = None
            if got_eval:
                eval_n = pq.read_metadata(str(eval_preds_file)).num_rows
                print(f"  scoring held-out eval ({eval_n} predictions)…", flush=True)
                eval_score = await run_scoring(
                    dataset_path=eval_preds_file,
                    scorer_path=cfg.scorer_path,
                    output_dir=iter_dir / "held_out_scoring",
                    client=client,
                    model_size=judge_size,
                    concurrency=judge_concurrency,
                    run_inference=False,
                )
                held_out_mean = eval_score.mean_score
                held_out_median = eval_score.median_score
                (iter_dir / "held_out_eval.json").write_text(json.dumps({
                    "iteration": i,
                    "mean": held_out_mean,
                    "median": held_out_median,
                    "scored": eval_score.scored,
                    "failed": eval_score.failed,
                    "baseline_mean": baseline_mean,
                    "delta_vs_baseline": held_out_mean - baseline_mean,
                }, indent=2) + "\n")
                delta = held_out_mean - baseline_mean
                # Trend: drop relative to the previous iter. None for iter 0.
                prev_held_out = (
                    iter_stats[-1].held_out_mean if iter_stats else None
                )
                trend_delta: float | None = None
                if prev_held_out is not None:
                    trend_delta = held_out_mean - prev_held_out
                print(
                    f"  iter {i} held-out: mean={held_out_mean:.3f} "
                    f"median={held_out_median:.1f}  "
                    f"Δ vs baseline {delta:+.3f}"
                    + (f"  Δ vs iter {i-1} {trend_delta:+.3f}"
                       if trend_delta is not None else ""),
                    flush=True,
                )
                # Early-abort: two independent checks. (A) absolute regression
                # past baseline. (B) iter-over-iter trend heading down. Either
                # fires → write early_abort.json with the specific reason and
                # break the loop.
                abort_reason: str | None = None
                if delta < early_abort_delta:
                    abort_reason = (
                        f"held-out eval at iter {i} is {delta:+.3f} below "
                        f"baseline (threshold {early_abort_delta:+.3f})"
                    )
                elif trend_delta is not None and trend_delta < trend_abort_delta:
                    abort_reason = (
                        f"held-out at iter {i} dropped {trend_delta:+.3f} "
                        f"from iter {i-1} (trend threshold "
                        f"{trend_abort_delta:+.3f}); trajectory is heading "
                        f"down — best-so-far is iter {i-1}"
                    )
                if abort_reason is not None:
                    abort_signal.write_text(json.dumps({
                        "iteration": i,
                        "held_out_mean": held_out_mean,
                        "prev_held_out_mean": prev_held_out,
                        "baseline_mean": baseline_mean,
                        "delta_vs_baseline": delta,
                        "delta_vs_prev_iter": trend_delta,
                        "threshold_baseline": early_abort_delta,
                        "threshold_trend": trend_abort_delta,
                        "reason": abort_reason,
                    }, indent=2) + "\n")
                    print(f"  ❌ EARLY ABORT: {abort_reason}", flush=True)
                    print(
                        "  Subprocess will exit on next loop check.",
                        flush=True,
                    )
                    iter_stats.append(DpoIterStat(
                        iteration=i,
                        on_policy_mean=score.mean_score,
                        on_policy_median=score.median_score,
                        held_out_mean=held_out_mean,
                        held_out_median=held_out_median,
                    ))
                    aborted = True
                    break
            else:
                print(
                    f"  WARN: no held-out eval predictions for iter {i} "
                    f"(timeout or subprocess died)",
                    flush=True,
                )

            iter_stats.append(DpoIterStat(
                iteration=i,
                on_policy_mean=score.mean_score,
                on_policy_median=score.median_score,
                held_out_mean=held_out_mean,
                held_out_median=held_out_median,
            ))

        print(
            f"\n  waiting for DPO subprocess to finish "
            f"({'after early-abort' if aborted else 'normal completion'})…",
            flush=True,
        )
        await proc.wait()
        if proc.returncode != 0 and not aborted:
            raise RuntimeError(f"DPO subprocess exited with code {proc.returncode}")
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

    if not model_dir.exists():
        raise RuntimeError(f"DPO produced no model at {model_dir}")
    return model_dir, iter_stats, chosen_pool_mean


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _verdict(delta: float, threshold: float, *, label: str = "SFT") -> tuple[str, int]:
    if delta < threshold:
        return f"❌ FAIL — {label} regressed beyond threshold ({delta:+.3f} < {threshold:+.3f})", 1
    if delta < 0:
        return f"⚠️  WARN — {label} regressed slightly ({delta:+.3f}); within threshold {threshold:+.3f}", 0
    return f"✅ PASS — {label} improved by {delta:+.3f}", 0


def _render_report(
    cfg: TaskConfig,
    baseline: EvalScore,
    post_sft: EvalScore | None,
    *,
    post_dpo: EvalScore | None,
    dpo_iter_stats: list["DpoIterStat"],
    chosen_pool_mean: float | None,
    train_filter: FilterReport | None,
    eval_filter: FilterReport | None,
    elapsed_s: float,
    verdict_text: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# E2E pipeline report — {cfg.task_name} ({cfg.config_name})")
    lines.append("")
    lines.append(f"- task_kind: `{cfg.task_kind}`")
    lines.append(f"- base_model: `{cfg.base_model}`")
    lines.append(f"- stages: "
                 f"baseline → "
                 f"{'SFT → ' if cfg.sft is not None else ''}"
                 f"{'DPO' if cfg.dpo is not None else ''}".rstrip(" →"))
    lines.append(f"- train_samples (post-filter target): {cfg.train_samples}")
    lines.append(f"- eval_samples (post-filter target):  {cfg.eval_samples}")
    lines.append(f"- filter_threshold: {cfg.filter_threshold}")
    lines.append(f"- elapsed: {elapsed_s/60:.1f} min")
    lines.append("")
    if train_filter is not None:
        lines.append(
            f"**Train filter**: kept {train_filter.kept}/"
            f"{train_filter.kept + train_filter.dropped} (mean {train_filter.mean:.2f})"
        )
    if eval_filter is not None:
        lines.append(
            f"**Eval filter**:  kept {eval_filter.kept}/"
            f"{eval_filter.kept + eval_filter.dropped} (mean {eval_filter.mean:.2f})"
        )
    if chosen_pool_mean is not None:
        headroom = chosen_pool_mean - baseline.mean
        lines.append(
            f"**Chosen-pool ceiling**: {chosen_pool_mean:.3f} "
            f"(baseline {baseline.mean:.3f}, headroom {headroom:+.3f}) — "
            f"the model can't exceed chosen-pool quality on this judge."
        )
    lines.append("")
    lines.append("## Scores")
    lines.append("")
    lines.append("| stage     | mean   | median | scored | failed |")
    lines.append("|-----------|--------|--------|--------|--------|")
    lines.append(
        f"| baseline  | {baseline.mean:6.3f} | {baseline.median:5.1f}  "
        f"| {baseline.scored:>6} | {baseline.failed:>6} |"
    )
    if post_sft is not None:
        lines.append(
            f"| post-SFT  | {post_sft.mean:6.3f} | {post_sft.median:5.1f}  "
            f"| {post_sft.scored:>6} | {post_sft.failed:>6} |"
        )
        delta_sft = post_sft.mean - baseline.mean
        lines.append(f"| **Δ SFT** | **{delta_sft:+.3f}** |       |        |        |")
    if post_dpo is not None:
        lines.append(
            f"| post-DPO  | {post_dpo.mean:6.3f} | {post_dpo.median:5.1f}  "
            f"| {post_dpo.scored:>6} | {post_dpo.failed:>6} |"
        )
        ref = post_sft if post_sft is not None else baseline
        delta_dpo = post_dpo.mean - ref.mean
        lines.append(
            f"| **Δ DPO** | **{delta_dpo:+.3f}** |       |        |        |"
            f"  *(vs {ref.label})*"
        )
    lines.append("")
    if dpo_iter_stats:
        any_held_out = any(s.held_out_mean is not None for s in dpo_iter_stats)
        lines.append("**DPO per-iter scores**:")
        lines.append("")
        if any_held_out:
            lines.append("| iter | on-policy mean | held-out mean | Δ vs baseline |")
            lines.append("|------|---------------|---------------|----------------|")
            for s in dpo_iter_stats:
                ho = f"{s.held_out_mean:.3f}" if s.held_out_mean is not None else "—"
                if s.held_out_mean is not None:
                    delta = s.held_out_mean - baseline.mean
                    delta_str = f"{delta:+.3f}"
                else:
                    delta_str = "—"
                lines.append(
                    f"| {s.iteration} | {s.on_policy_mean:.3f} | {ho} | {delta_str} |"
                )
        else:
            lines.append("| iter | on-policy mean |")
            lines.append("|------|---------------|")
            for s in dpo_iter_stats:
                lines.append(f"| {s.iteration} | {s.on_policy_mean:.3f} |")
        lines.append("")
    lines.append(f"**Verdict**: {verdict_text}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True,
                   help="Task project name under tests/e2e_projects/")
    p.add_argument("--config-name", default="default",
                   help="Picks <project>/e2e_config_<NAME>.json. Default loads "
                        "<project>/e2e_config.json. Use this to run multiple "
                        "configs (e.g. SFT-only vs DPO-only) against the same "
                        "task without duplicating the pipeline/scorer files.")
    p.add_argument("--exp-name", default=None,
                   help="Run dir prefix; defaults to e2e[_<config-name>]_<timestamp>")
    p.add_argument("--train-samples", type=int, default=None,
                   help="Override config train_samples (small ints handy for shake-out)")
    p.add_argument("--eval-samples", type=int, default=None,
                   help="Override config eval_samples")
    p.add_argument("--train-dataset-suffix", default="",
                   help="Suffix appended to train_raw/train_filtered dataset "
                        "directory names. Use to separate per-scale datasets "
                        "in scaling experiments without colliding (e.g. "
                        "--train-dataset-suffix n200 → "
                        "<task>_train_raw_n200/, <task>_train_filtered_n200/). "
                        "Eval datasets are unaffected — they stay shared so "
                        "scaling runs are comparable on the same eval set.")
    p.add_argument("--judge-size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--judge-concurrency", type=int, default=5)
    p.add_argument("--dpo-early-abort-delta", type=float, default=-0.5,
                   help="Per-iter held-out eval delta vs baseline below which "
                        "DPO is killed early. Default -0.5 (abort if held-out "
                        "mean drops 0.5+ points below baseline at any iter). "
                        "Set very negative (e.g. -10) to disable.")
    p.add_argument("--dpo-trend-abort-delta", type=float, default=-0.4,
                   help="Per-iter held-out delta vs the PREVIOUS iter's "
                        "held-out below which DPO is killed early. Default "
                        "-0.4 (abort if held-out drops 0.4+ from one iter to "
                        "the next, even if still near baseline). Catches "
                        "downward trajectories before they cross the absolute "
                        "abort threshold. Set very negative to disable.")

    # Skip flags
    for stage in ("generate-train", "generate-eval", "filter-train", "filter-eval",
                  "baseline-eval", "sft", "post-sft-eval",
                  "dpo", "post-dpo-eval"):
        p.add_argument(f"--skip-{stage}", action="store_true")
    args = p.parse_args()

    cfg = load_task_config(
        args.task,
        override_train=args.train_samples,
        override_eval=args.eval_samples,
        config_name=args.config_name,
    )
    if args.exp_name:
        exp_name = args.exp_name
    elif args.config_name != "default":
        exp_name = f"e2e_{args.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_name = f"e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runs_root = cfg.project_dir / "runs" / exp_name
    runs_root.mkdir(parents=True, exist_ok=True)

    print(f"task:       {cfg.task_name}")
    print(f"project:    {cfg.project_dir.relative_to(REPO_ROOT)}")
    print(f"exp:        {exp_name}")
    print(f"base_model: {cfg.base_model}")

    started = time.monotonic()

    # ------------------------------------------------------------------
    # 1+2. Generate train and eval
    # ------------------------------------------------------------------
    train_suffix = (
        f"_{args.train_dataset_suffix}" if args.train_dataset_suffix else ""
    )
    train_raw_dir = cfg.project_dir / "datasets" / f"{cfg.task_name}_train_raw{train_suffix}"
    eval_raw_dir = cfg.project_dir / "datasets" / f"{cfg.task_name}_eval_raw"

    if args.skip_generate_train and (train_raw_dir / "data.parquet").exists():
        print(f"  [generate-train] reusing {train_raw_dir.relative_to(REPO_ROOT)}")
        train_raw = train_raw_dir / "data.parquet"
    else:
        train_raw = await stage_generate(cfg, "train", cfg.train_samples, train_raw_dir)

    if args.skip_generate_eval and (eval_raw_dir / "data.parquet").exists():
        print(f"  [generate-eval]  reusing {eval_raw_dir.relative_to(REPO_ROOT)}")
        eval_raw = eval_raw_dir / "data.parquet"
    else:
        eval_raw = await stage_generate(cfg, "eval", cfg.eval_samples, eval_raw_dir)

    # ------------------------------------------------------------------
    # 3+4. Filter train and eval
    # ------------------------------------------------------------------
    train_filtered_dir = cfg.project_dir / "datasets" / f"{cfg.task_name}_train_filtered{train_suffix}"
    eval_filtered_dir = cfg.project_dir / "datasets" / f"{cfg.task_name}_eval_filtered"

    train_filter: FilterReport | None = None
    eval_filter: FilterReport | None = None

    if args.skip_filter_train and (train_filtered_dir / "data.parquet").exists():
        print(f"  [filter-train] reusing {train_filtered_dir.relative_to(REPO_ROOT)}")
        train_filtered = train_filtered_dir / "data.parquet"
    else:
        train_filter = await stage_filter(cfg, "train", train_raw, train_filtered_dir)
        train_filtered = train_filter.output_path

    if args.skip_filter_eval and (eval_filtered_dir / "data.parquet").exists():
        print(f"  [filter-eval]  reusing {eval_filtered_dir.relative_to(REPO_ROOT)}")
        eval_filtered = eval_filtered_dir / "data.parquet"
    else:
        eval_filter = await stage_filter(cfg, "eval", eval_raw, eval_filtered_dir)
        eval_filtered = eval_filter.output_path

    # ------------------------------------------------------------------
    # 5. Baseline eval
    # ------------------------------------------------------------------
    baseline = await stage_eval(
        cfg, "baseline", cfg.base_model, eval_filtered,
        runs_root / "baseline_eval",
        skip_inference=args.skip_baseline_eval,
        judge_size=args.judge_size, judge_concurrency=args.judge_concurrency,
    )

    # ------------------------------------------------------------------
    # 6 + 7. SFT and post-SFT eval (only if config.sft is set; null = skip)
    # ------------------------------------------------------------------
    post_sft: EvalScore | None = None
    sft_model_dir: Path | None = None
    if cfg.sft is not None:
        sft_run_dir = runs_root / "sft"
        sft_model_dir = stage_sft(cfg, train_filtered, sft_run_dir, skip=args.skip_sft)
        post_sft = await stage_eval(
            cfg, "post-sft", str(sft_model_dir), eval_filtered,
            runs_root / "post_sft_eval",
            skip_inference=args.skip_post_sft_eval,
            judge_size=args.judge_size, judge_concurrency=args.judge_concurrency,
        )

    # ------------------------------------------------------------------
    # 8 + 9. DPO + post-DPO eval (only if config.dpo is set)
    # When SFT was skipped, DPO starts from cfg.base_model directly.
    # ------------------------------------------------------------------
    post_dpo: EvalScore | None = None
    dpo_iter_stats: list[DpoIterStat] = []
    chosen_pool_mean: float | None = None
    dpo_run_dir = runs_root / "dpo"
    if cfg.dpo is not None:
        starting_model = (
            str(sft_model_dir.resolve()) if sft_model_dir is not None
            else cfg.base_model
        )
        dpo_model_dir, dpo_iter_stats, chosen_pool_mean = await stage_dpo(
            cfg, starting_model, train_filtered, eval_filtered,
            baseline.mean,  # for per-iter early-abort comparison
            dpo_run_dir,
            skip=args.skip_dpo,
            judge_size=args.judge_size, judge_concurrency=args.judge_concurrency,
            early_abort_delta=args.dpo_early_abort_delta,
            trend_abort_delta=args.dpo_trend_abort_delta,
        )
        post_dpo = await stage_eval(
            cfg, "post-dpo", str(dpo_model_dir), eval_filtered,
            runs_root / "post_dpo_eval",
            skip_inference=args.skip_post_dpo_eval,
            judge_size=args.judge_size, judge_concurrency=args.judge_concurrency,
        )

    elapsed = time.monotonic() - started

    # Verdict: compare the FINAL trained stage to baseline. Label varies
    # by which stages actually ran (SFT only, DPO only, or SFT+DPO).
    if post_dpo is not None and post_sft is not None:
        verdict_label = "SFT+DPO"
        delta_final = post_dpo.mean - baseline.mean
    elif post_dpo is not None:
        verdict_label = "DPO"
        delta_final = post_dpo.mean - baseline.mean
    elif post_sft is not None:
        verdict_label = "SFT"
        delta_final = post_sft.mean - baseline.mean
    else:
        # Should not happen — load_task_config rejects sft=null && dpo=null.
        raise RuntimeError("neither SFT nor DPO produced an eval score")

    verdict_text, exit_code = _verdict(
        delta_final, cfg.regression_threshold, label=verdict_label,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  baseline  mean={baseline.mean:6.3f}  median={baseline.median:5.1f}")
    if post_sft is not None:
        delta_sft = post_sft.mean - baseline.mean
        print(f"  post-SFT  mean={post_sft.mean:6.3f}  median={post_sft.median:5.1f}")
        print(f"  Δ SFT     {delta_sft:+.3f}")
    if post_dpo is not None:
        ref = post_sft if post_sft is not None else baseline
        delta_dpo = post_dpo.mean - ref.mean
        print(f"  post-DPO  mean={post_dpo.mean:6.3f}  median={post_dpo.median:5.1f}")
        print(f"  Δ DPO     {delta_dpo:+.3f}  (vs {ref.label})")
    print(f"  elapsed   {elapsed/60:.1f} min")
    print(f"  {verdict_text}")

    report = _render_report(
        cfg, baseline, post_sft,
        post_dpo=post_dpo, dpo_iter_stats=dpo_iter_stats,
        chosen_pool_mean=chosen_pool_mean,
        train_filter=train_filter, eval_filter=eval_filter,
        elapsed_s=elapsed, verdict_text=verdict_text,
    )
    report_path = runs_root / "report.md"
    report_path.write_text(report)
    print(f"\nreport: {report_path.relative_to(REPO_ROOT)}")

    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
