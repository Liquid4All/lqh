"""E2E test: Spec capture only — email summarization

Usage:
    python -m tests.e2e.test_spec_only
    python -m tests.e2e.test_spec_only orchestration:2
    python -m tests.e2e.test_spec_only orchestration:2 --timeout=600
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
from tests.e2e.scenarios import SPEC_ONLY_EMAIL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_ORCHESTRATION_MODEL = "orchestration:2"
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


def _run_e2e(model: str, timeout: int) -> None:
    """Run the E2E test with report generation on any exit (success, failure, kill)."""
    harness = E2EHarness(SPEC_ONLY_EMAIL, orchestration_model=model)
    result = None
    start = time.time()

    async def _run() -> None:
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


@unittest.skipUnless(_has_api_access(), "No API access (set LQH_DEBUG_API_KEY or run /login)")
class TestSpecOnlyE2E(unittest.TestCase):
    """End-to-end test for spec capture only — email summarization."""

    def test_spec_capture(self) -> None:
        """Run spec capture for the email summarization task."""
        result = _run_e2e(_ORCHESTRATION_MODEL, _TIMEOUT_SECONDS)

        # --- Heuristic checks ---
        tools = result.tools_called()
        self.assertIn("ask_user", tools, "Agent never asked the user anything")
        self.assertIn("create_file", tools, "Agent never created any files")

        artifacts = result.artifacts
        self.assertIn("SPEC.md", artifacts, "SPEC.md was not created")

        spec = artifacts["SPEC.md"].lower()
        self.assertTrue(
            "email" in spec or "summar" in spec,
            f"SPEC.md doesn't mention email or summarization: {spec[:200]}",
        )

        critical_errors = [e for e in result.errors if "Internal error" in e]
        self.assertEqual(critical_errors, [], f"Critical errors: {critical_errors}")

        ask_user_turns = [t for t in result.transcript if t.role == "ask_user_q"]
        self.assertGreaterEqual(
            len(ask_user_turns), 3,
            f"Agent only asked {len(ask_user_turns)} questions, expected at least 3",
        )

        # --- LLM Judge ---
        async def _judge() -> None:
            from lqh.auth import require_token
            from lqh.client import create_client
            from lqh.config import load_config

            config = load_config()
            token = require_token()
            client = create_client(token, config.api_base_url)

            judge_results = await judge_artifacts(
                client, SPEC_ONLY_EMAIL, artifacts,
            )

            for jr in judge_results:
                logger.info("Judge %s: %d/10 — %s", jr.artifact, jr.score, jr.reasoning)

            spec_scores = [jr for jr in judge_results if jr.artifact == "SPEC.md"]
            if spec_scores:
                self.assertGreaterEqual(
                    spec_scores[0].score, 6,
                    f"SPEC.md judge score too low: {spec_scores[0].reasoning}",
                )

        asyncio.run(_judge())


if __name__ == "__main__":
    unittest.main()
