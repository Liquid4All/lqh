"""E2E test: Model evaluation + prompt optimization on pre-seeded project.

Starts with an existing translation project (spec, 50 eval samples, scorer)
and tests model evaluation on lfm2.5-1.2b-instruct followed by prompt optimization.

Usage:
    python -m tests.e2e.test_eval_promptopt
    python -m tests.e2e.test_eval_promptopt orchestration:3
    python -m tests.e2e.test_eval_promptopt orchestration:3 --timeout=1200
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import unittest

from tests.e2e.harness import E2EHarness
from tests.e2e.judge import judge_artifacts
from tests.e2e.report import generate_report
from tests.e2e.scenarios import TRANSLATION_EVAL_PROMPTOPT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# CLI args
_ORCHESTRATION_MODEL = "orchestration:3"
_TIMEOUT_SECONDS = 0
_remaining_args = []
for arg in sys.argv[1:]:
    if arg.startswith("orchestration"):
        _ORCHESTRATION_MODEL = arg
    elif arg.startswith("--timeout="):
        _TIMEOUT_SECONDS = int(arg.split("=")[1])
    else:
        _remaining_args.append(arg)
sys.argv = [sys.argv[0]] + _remaining_args


def _has_api_access() -> bool:
    try:
        from lqh.auth import get_token
        return get_token() is not None
    except Exception:
        return False


def _run_e2e(model: str, timeout: int):
    """Run the E2E test with report generation on any exit."""
    harness = E2EHarness(TRANSLATION_EVAL_PROMPTOPT, orchestration_model=model)
    result = None
    start = time.time()

    async def _run():
        nonlocal result
        if timeout > 0:
            try:
                result = await asyncio.wait_for(harness.run(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("E2E test timed out after %ds", timeout)
                result = harness._build_result(duration=time.time() - start)
                result.errors.append(f"Timed out after {timeout}s")
        else:
            result = await harness.run()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.warning("E2E test interrupted by user")
        result = harness._build_result(duration=time.time() - start)
        result.errors.append("Interrupted by user (KeyboardInterrupt)")
    except BaseException as exc:
        logger.error("E2E run crashed: %s", exc)
        result = harness._build_result(duration=time.time() - start)
        result.errors.append(f"Crashed: {type(exc).__name__}: {exc}")

    if result is None:
        result = harness._build_result(duration=time.time() - start)

    logger.info(
        "E2E completed: %d turns, %d tool calls, %.1fs, model=%s",
        result.total_turns, result.total_tool_calls,
        result.duration_seconds, result.orchestration_model,
    )
    report_path = generate_report(result)
    logger.info("Report: %s", report_path)
    logger.info("Project dir: %s", result.project_dir)

    return result


@unittest.skipUnless(_has_api_access(), "No API access")
class TestEvalPromptOpt(unittest.TestCase):
    """E2E test: model eval + prompt optimization on pre-seeded project."""

    def test_eval_and_prompt_opt(self) -> None:
        """Evaluate lfm2.5-1.2b-instruct then optimize the prompt."""
        result = _run_e2e(_ORCHESTRATION_MODEL, _TIMEOUT_SECONDS)

        # --- Heuristic checks ---
        tools = result.tools_called()

        # Agent should have used run_scoring (for eval and/or prompt opt)
        self.assertIn("run_scoring", tools, "Agent never ran scoring")

        # Check for eval runs or prompts created
        artifacts = result.artifacts
        eval_runs = [k for k in artifacts if k.startswith("evals/runs/")]
        prompts = [k for k in artifacts if k.startswith("prompts/")]

        logger.info("Eval runs created: %s", eval_runs)
        logger.info("Prompts created: %s", prompts)
        logger.info("Pipeline attempts: %s", result.pipeline_attempts)

        # At minimum, an eval run should exist
        self.assertTrue(
            len(eval_runs) > 0 or len(prompts) > 0,
            "No eval runs or prompts were created",
        )

        # --- LLM Judge on prompts ---
        if prompts:
            async def _judge():
                from lqh.auth import require_token
                from lqh.client import create_client
                from lqh.config import load_config

                config = load_config()
                token = require_token()
                client = create_client(token, config.api_base_url)

                judge_results = await judge_artifacts(
                    client, TRANSLATION_EVAL_PROMPTOPT, artifacts,
                )
                for jr in judge_results:
                    logger.info("Judge %s: %d/10 — %s", jr.artifact, jr.score, jr.reasoning)

            asyncio.run(_judge())


if __name__ == "__main__":
    unittest.main()
