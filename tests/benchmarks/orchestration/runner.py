"""Orchestration benchmark runner.

Iterates all (model, scenario) pairs, collects results, and generates
an aggregate report.

Usage:
    python -m tests.benchmarks.orchestration.runner
    python -m tests.benchmarks.orchestration.runner --models orchestration:12,orchestration:3
    python -m tests.benchmarks.orchestration.runner --categories datagen_pipeline,error_recovery
    python -m tests.benchmarks.orchestration.runner --timeout=900
    python -m tests.benchmarks.orchestration.runner --parallel=3
    python -m tests.benchmarks.orchestration.runner --resume results/run_20260414/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from lqh.auth import require_token
from lqh.client import create_client
from lqh.config import load_config

from tests.benchmarks.orchestration.scoring import BenchmarkScore, score_result
from tests.benchmarks.orchestration.aggregate_report import generate_aggregate_report
from tests.harness.harness import E2EHarness
from tests.harness.report import generate_report
from tests.harness.scenarios import Scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Silence noisy HTTP request logs from the openai/httpx stack
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DEFAULT_MODELS = [f"orchestration:{i}" for i in range(1, 15)]  # 1..14

RESULTS_DIR = Path(__file__).parent / "results"

# Categories excluded from the default "all" sweep because they are expensive
# and/or can launch real training. Run them explicitly with
# `--categories auto_mode`.
EXPENSIVE_CATEGORIES = {"auto_mode"}


def _load_all_scenarios() -> list[tuple[str, Scenario]]:
    """Import and collect all benchmark scenarios with their category names."""
    scenarios: list[tuple[str, Scenario]] = []

    from tests.benchmarks.orchestration.categories.spec_capture import SCENARIOS as sc
    scenarios.extend(("spec_capture", s) for s in sc)

    from tests.benchmarks.orchestration.categories.spec_generation import SCENARIOS as sg
    scenarios.extend(("spec_generation", s) for s in sg)

    from tests.benchmarks.orchestration.categories.datagen_pipeline import SCENARIOS as dp
    scenarios.extend(("datagen_pipeline", s) for s in dp)

    from tests.benchmarks.orchestration.categories.scorer_validation import SCENARIOS as sv
    scenarios.extend(("scorer_validation", s) for s in sv)

    from tests.benchmarks.orchestration.categories.data_filtering import SCENARIOS as df
    scenarios.extend(("data_filtering", s) for s in df)

    from tests.benchmarks.orchestration.categories.error_recovery import SCENARIOS as er
    scenarios.extend(("error_recovery", s) for s in er)

    from tests.benchmarks.orchestration.categories.next_steps import SCENARIOS as ns
    scenarios.extend(("next_steps", s) for s in ns)

    from tests.benchmarks.orchestration.categories.edit import SCENARIOS as ed
    scenarios.extend(("edit", s) for s in ed)

    from tests.benchmarks.orchestration.categories.context_management import SCENARIOS as cm
    scenarios.extend(("context_management", s) for s in cm)

    from tests.benchmarks.orchestration.categories.auto_mode import SCENARIOS as am
    scenarios.extend(("auto_mode", s) for s in am)

    return scenarios


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run orchestration benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of orchestration models to test",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="",
        help=(
            "Comma-separated list of categories to run (empty = all non-expensive). "
            "Expensive/opt-in categories (auto_mode) only run when named explicitly."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout per scenario in seconds (default: 900)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to existing run directory to resume from",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of (model, scenario) runs to execute concurrently (default: 1)",
    )
    return parser.parse_args()


async def _run_single(
    model: str,
    category: str,
    scenario: Scenario,
    timeout: int,
    client,
) -> tuple[BenchmarkScore, Path | None]:
    """Run a single (model, scenario) pair and return the score + report path."""
    logger.info("=" * 60)
    logger.info("Running: model=%s category=%s scenario=%s", model, category, scenario.name)
    logger.info("=" * 60)

    harness = E2EHarness(scenario, orchestration_model=model)
    result = None
    start = time.time()

    try:
        if timeout > 0:
            try:
                result = await asyncio.wait_for(harness.run(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Timed out after %ds: %s / %s", timeout, model, scenario.name)
                result = harness._build_result(duration=time.time() - start)
                result.errors.append(f"Timed out after {timeout}s")
        else:
            result = await harness.run()
    except KeyboardInterrupt:
        raise  # Let the outer loop handle it
    except Exception as exc:
        logger.error("Run crashed: %s", exc)
        result = harness._build_result(duration=time.time() - start)
        result.errors.append(f"Crashed: {type(exc).__name__}: {exc}")

    if result is None:
        result = harness._build_result(duration=time.time() - start)

    # Generate per-run report
    report_path = generate_report(result)
    logger.info("Report: %s", report_path)

    # Score the result
    score = await score_result(result, scenario, category, client)
    logger.info(
        "Score: %.1f/100 | catastrophic=%s | duration=%.1fs",
        score.composite_score,
        score.is_catastrophic_failure,
        score.duration_seconds,
    )

    return score, report_path


async def main() -> None:
    args = _parse_args()
    models = [m.strip() for m in args.models.split(",")]
    category_filter = set(c.strip() for c in args.categories.split(",") if c.strip())

    # Check API access
    try:
        token = require_token()
    except Exception:
        logger.error("No API access. Set LQH_DEBUG_API_KEY or run /login.")
        sys.exit(1)

    config = load_config()
    client = create_client(token, config.api_base_url)

    # Load scenarios
    all_scenarios = _load_all_scenarios()
    if category_filter:
        all_scenarios = [(c, s) for c, s in all_scenarios if c in category_filter]
    else:
        # Default sweep excludes expensive, opt-in categories (e.g. auto_mode,
        # which can launch real training). Request them explicitly via
        # --categories auto_mode.
        skipped = sorted({c for c, _ in all_scenarios if c in EXPENSIVE_CATEGORIES})
        if skipped:
            logger.info("Excluding expensive categories from default run: %s "
                        "(run explicitly with --categories %s)",
                        ", ".join(skipped), ",".join(skipped))
        all_scenarios = [
            (c, s) for c, s in all_scenarios if c not in EXPENSIVE_CATEGORIES
        ]

    logger.info("Benchmark: %d scenarios x %d models = %d runs",
                len(all_scenarios), len(models), len(all_scenarios) * len(models))

    # Setup results directory
    if args.resume:
        run_dir = Path(args.resume)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scores_path = run_dir / "scores.json"
    progress_path = run_dir / "progress.json"

    # Load existing scores for resume
    existing_scores: list[dict] = []
    completed_pairs: set[tuple[str, str]] = set()
    if scores_path.exists():
        existing_scores = json.loads(scores_path.read_text())
        for s in existing_scores:
            completed_pairs.add((s["model"], s["scenario_name"]))
        logger.info("Resuming: %d runs already completed", len(completed_pairs))

    scores: list[BenchmarkScore] = [
        BenchmarkScore(**s) for s in existing_scores
    ] if existing_scores else []

    # Build work list
    work: list[tuple[str, str, Scenario]] = []
    for model in models:
        for category, scenario in all_scenarios:
            if (model, scenario.name) not in completed_pairs:
                work.append((model, category, scenario))

    logger.info("Work remaining: %d runs", len(work))

    # Update progress
    def _save_progress() -> None:
        progress = {
            "total": len(all_scenarios) * len(models),
            "completed": len(scores),
            "remaining": len(work) - (len(scores) - len(completed_pairs)),
            "models": models,
            "categories": sorted(set(c for c, _ in all_scenarios)),
        }
        progress_path.write_text(json.dumps(progress, indent=2))

    def _save_scores() -> None:
        scores_path.write_text(
            json.dumps([s.to_dict() for s in scores], indent=2, ensure_ascii=False)
        )

    _save_progress()

    parallel = max(1, args.parallel)
    semaphore = asyncio.Semaphore(parallel)
    save_lock = asyncio.Lock()
    completed_count = 0

    async def _run_with_limit(idx: int, model: str, category: str, scenario: Scenario) -> None:
        nonlocal completed_count
        async with semaphore:
            logger.info("[start %d/%d] %s / %s", idx + 1, len(work), model, scenario.name)
            try:
                score, _ = await _run_single(model, category, scenario, args.timeout, client)
            except Exception as exc:
                logger.error("Unhandled exception in worker: %s", exc)
                return
            async with save_lock:
                scores.append(score)
                completed_count += 1
                _save_scores()
                _save_progress()
                logger.info("[done %d/%d] %s / %s -> %.1f",
                            completed_count, len(work), model, scenario.name, score.composite_score)

    logger.info("Running with parallelism=%d", parallel)

    try:
        await asyncio.gather(
            *[_run_with_limit(i, m, c, s) for i, (m, c, s) in enumerate(work)],
            return_exceptions=False,
        )
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted. Saving partial results...")
        async with save_lock:
            _save_scores()
            _save_progress()

    # Generate aggregate report
    logger.info("Generating aggregate report...")
    report_path = generate_aggregate_report(scores, run_dir)
    logger.info("Aggregate report: %s", report_path)
    logger.info("Scores: %s", scores_path)


if __name__ == "__main__":
    asyncio.run(main())
