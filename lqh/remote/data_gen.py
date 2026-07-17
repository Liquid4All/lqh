"""In-sandbox entry point for cloud data-gen jobs.

``python -m lqh.remote.data_gen <config.json>`` — launched by the
backend's default bundle launcher inside a CPU Modal sandbox
(CLOUD_OFFLOAD_PLAN.md §2). The launcher has already:

* extracted the submit bundle (project-relative layout: the pipeline
  script under ``data_gen/``, seed data folders, ``config.json``) into
  ``$RUN_DIR/inputs/``,
* copied ``config.json`` up to ``$RUN_DIR/config.json`` (our argv), and
* set cwd to ``$RUN_DIR/inputs`` — the extracted project root, which is
  what ``lqh.sources`` path validation anchors on.

This module is a thin shell around the unchanged pipeline engine: build
the OpenAI client from the injected job token (scoped ``chat.gen``),
run ``lqh.engine.run_pipeline``, stream progress via the
``LQH_EVENT_JSON:`` sentinel, and leave outputs at the run-dir root
where the launcher's ``lqh.remote.publish`` step picks up
``data.parquet`` as a ``dataset`` artifact.

``data.partial.jsonl`` is written to the run dir on the per-project
volume, so a worker continuation of the *same* job resumes via the
engine's index-skip logic. A fresh submit gets a fresh run dir — no
cross-submit resume (v1 limitation, documented in the plan).

Env contract (injected by the backend):
    LQH_API_TOKEN   scoped job token (chat.gen + artifacts.write + ...)
    LQH_BASE_URL    API origin WITHOUT /v1 (the backend injects its
                    cfg.APIBaseURL; the publish/artifacts helpers append
                    /v1 themselves) — _openai_base() adds the /v1 the
                    OpenAI client needs, tolerating either form
    LQH_JOB_ID      gates sentinel emission in lqh.progress
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


def _openai_base(url: str | None) -> str | None:
    """OpenAI-client base from LQH_BASE_URL, whichever form it arrives in.

    The sandbox env carries the backend origin (https://api.lqh.ai);
    the laptop convention carries .../v1. AsyncOpenAI resolves
    ``/chat/completions`` relative to its base, so it needs the /v1.
    """
    if not url:
        return None  # create_client falls back to its default
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lqh.remote.data_gen <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    run_dir = config_path.parent
    inputs_dir = run_dir / "inputs"
    if not inputs_dir.is_dir():
        print(f"No inputs/ dir under {run_dir} — bundle missing?", file=sys.stderr)
        sys.exit(1)
    # The launcher already runs us with cwd=inputs; chdir defensively —
    # lqh.sources helpers resolve every path against Path.cwd().
    os.chdir(inputs_dir)

    token = os.environ.get("LQH_API_TOKEN", "")
    if not token:
        print("LQH_API_TOKEN not set — cannot reach the LLM API", file=sys.stderr)
        sys.exit(1)

    script_rel = str(config.get("script_path", ""))
    if not script_rel:
        print("config.script_path is required", file=sys.stderr)
        sys.exit(1)
    script = (inputs_dir / script_rel).resolve()
    if not script.exists():
        print(f"Pipeline script not in bundle: {script_rel}", file=sys.stderr)
        sys.exit(1)

    num_samples = int(config.get("num_samples", 0))
    if num_samples <= 0:
        print(f"config.num_samples must be positive, got {num_samples}", file=sys.stderr)
        sys.exit(1)
    samples_per_item = max(1, int(config.get("samples_per_item", 1)))
    # Total work is num_samples × samples_per_item — clamping to
    # num_samples alone would serialize iterate-N× pipelines, and cloud
    # CPU bills by wall-clock. Clamp BOTH bounds: the CLI sends sane
    # values, but config.json is client-authored, and a raw API caller
    # could pass concurrency <= 0, which would hang the worker pool.
    concurrency = min(
        max(1, int(config.get("concurrency", 100))),
        max(1, num_samples * samples_per_item),
    )

    val_text: str | None = None
    val_rel = config.get("validation_instructions")
    if val_rel:
        val_path = inputs_dir / str(val_rel)
        if val_path.exists():
            val_text = val_path.read_text(encoding="utf-8")
        else:
            print(f"validation instructions not in bundle: {val_rel} (continuing without)")

    from lqh.client import create_client
    from lqh.engine import run_pipeline
    from lqh.progress import ProgressReporter, emit_sentinel

    client = create_client(token, _openai_base(os.environ.get("LQH_BASE_URL")))

    # run_dir on the reporter is what routes updates through
    # write_progress_event → LQH_EVENT_JSON sentinel (cloud SSE) +
    # progress.jsonl (published as a metrics artifact).
    reporter = ProgressReporter(
        task_kind="data_gen",
        label="Data generation",
        run_dir=run_dir,
    )
    # Best pre-run estimate of the expanded total (BYO sources may yield
    # fewer items); the engine's first callback corrects it.
    reporter.update(
        phase="generation", phase_label="generating", completed=0,
        total=num_samples * samples_per_item, unit="samples", overall_fraction=0,
        concurrency=concurrency, force=True,
    )

    def on_progress(completed: int, total: int) -> None:
        reporter.update(
            phase="generation", phase_label="generating",
            completed=completed, total=total, unit="samples",
            overall_fraction=completed / max(total, 1),
            concurrency=concurrency,
        )

    def write_status(status: str, **extra: object) -> None:
        (run_dir / "status.json").write_text(
            json.dumps({"status": status, "task": "data_gen", **extra}) + "\n"
        )

    try:
        result = asyncio.run(
            run_pipeline(
                script_path=script,
                num_samples=num_samples,
                # Outputs at the run-dir root: data.parquet is what
                # publish registers; data.partial.jsonl persists on the
                # volume for continuation resume.
                output_dir=run_dir,
                client=client,
                concurrency=concurrency,
                samples_per_item=samples_per_item,
                validation_instructions=val_text,
                on_progress=on_progress,
            )
        )
    except Exception as exc:
        # Deterministic pipeline bugs (the engine aborts fast on those)
        # and unexpected crashes both land here. status.json + logs are
        # published by the launcher regardless of our exit code.
        if isinstance(exc, FileNotFoundError):
            # The most common cloud-only failure: the pipeline read a
            # file the validated local run never touched, so it wasn't
            # recorded into the bundle manifest.
            print(
                "data_gen: a file the pipeline tried to read is not in the "
                "bundle. Cloud bundles contain only the files the local "
                "validation run actually read via lqh.sources — make "
                "source()/generate() read the same inputs on every run "
                "(no conditional or random file access), then re-validate "
                "locally and resubmit.",
                file=sys.stderr,
            )
        write_status("failed", error=f"{type(exc).__name__}: {exc}")
        raise

    ok = result.succeeded > 0
    write_status(
        "completed" if ok else "failed",
        total=result.total,
        succeeded=result.succeeded,
        failed=result.failed,
        **({} if ok else {"error": "no successful samples"}),
    )
    # Sample counts for the client: the SSE status mirror rewrites the
    # local status.json with state-only payloads, so ship the summary as
    # a progress row (lands in the client's progress.jsonl) instead.
    emit_sentinel("progress", {
        "summary": True,
        "total": result.total,
        "succeeded": result.succeeded,
        "failed": result.failed,
    })
    reporter.update(
        phase="completed" if ok else "failed",
        phase_label="dataset ready" if ok else "no samples generated",
        completed=result.total, total=result.total, unit="samples",
        overall_fraction=1.0, result_ready=ok, force=True,
    )
    print(
        f"data_gen: {result.succeeded}/{result.total} samples succeeded"
        + (f", {result.failed} failed" if result.failed else "")
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
