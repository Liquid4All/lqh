"""E2E test: observability action model, trained from scratch → GGUF Q4_K.

Runs the full app flow with an LLM-simulated human: spec capture →
data generation → cloud fine-tuning of LFM2.5-350M → GGUF export with
Q4_K quantization. This is the holistic template scenario from
E2E_TEST2.md — it exercises every stage of the product in one run.

Requires platform auth AND ``LQH_E2E=1`` (it spends real training and
CPU-sandbox money and can run for hours).

Usage:
    # Default orchestration model, no timeout:
    LQH_E2E=1 python -m tests.e2e.test_observability_gguf

    # Compare orchestration models / bound the runtime:
    LQH_E2E=1 python -m tests.e2e.test_observability_gguf orchestration:12 --timeout=7200

    # Via pytest:
    LQH_E2E=1 pytest tests/e2e/test_observability_gguf.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import unittest

from tests.harness.harness import E2EHarness
from tests.harness.judge import judge_artifacts
from tests.harness.report import generate_report
from tests.harness.scenarios import OBSERVABILITY_GGUF_350M

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# How long to wait for the async GGUF conversion job to register its
# artifacts after the agent conversation ends.
_GGUF_POLL_TIMEOUT = 900  # seconds
_GGUF_POLL_INTERVAL = 30

# CLI args
_ORCHESTRATION_MODEL = "orchestration:12"
_TIMEOUT_SECONDS = 0  # 0 = no timeout
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


def _e2e_enabled() -> bool:
    return os.environ.get("LQH_E2E") == "1"


def _run_e2e(model: str, timeout: int):
    """Run the E2E scenario, generating a report on any exit."""
    harness = E2EHarness(OBSERVABILITY_GGUF_350M, orchestration_model=model)
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


async def _wait_for_gguf_artifacts(project_id: str) -> list:
    """Poll the artifact registry until the GGUF conversion job registers
    its output (kind='gguf'), or the bounded timeout elapses.

    The conversion runs as an async cloud job, so the .gguf artifacts may
    land after the agent's final status poll. The registry — not a local
    file — is the source of truth for the export.
    """
    from lqh.artifacts import BackendArtifactStore

    store = BackendArtifactStore()
    deadline = time.time() + _GGUF_POLL_TIMEOUT
    while True:
        handles = await store.list_for_project(project_id, kind="gguf")
        if handles:
            return handles
        if time.time() >= deadline:
            return []
        logger.info("No gguf artifacts yet for %s; polling again…", project_id)
        await asyncio.sleep(_GGUF_POLL_INTERVAL)


@unittest.skipUnless(_has_api_access(), "No API access (set LQH_DEBUG_API_KEY or run /login)")
@unittest.skipUnless(_e2e_enabled(), "Set LQH_E2E=1 to run (spends real training money)")
class TestObservabilityGGUFE2E(unittest.TestCase):
    """Full app flow: spec → datagen → train LFM2.5-350M → GGUF Q4_K."""

    def test_full_flow_to_gguf(self) -> None:
        result = _run_e2e(_ORCHESTRATION_MODEL, _TIMEOUT_SECONDS)

        # --- Heuristic checks ---
        tools = result.tools_called()
        for tool in OBSERVABILITY_GGUF_350M.expected_tools:
            self.assertIn(tool, tools, f"Agent never called {tool}")

        artifacts = result.artifacts
        self.assertIn("SPEC.md", artifacts, "SPEC.md was not created")

        spec = artifacts["SPEC.md"].lower()
        self.assertTrue(
            "process" in spec or "cpu" in spec,
            f"SPEC.md doesn't mention the observability task: {spec[:200]}",
        )

        critical_errors = [e for e in result.errors if "Internal error" in e]
        self.assertEqual(critical_errors, [], f"Critical errors: {critical_errors}")

        # --- The requested base model (350M) reached start_training ---
        train_calls = [t for t in result.tool_calls if t.tool_name == "start_training"]
        self.assertTrue(train_calls, "start_training was never called")
        self.assertTrue(
            any("350" in json.dumps(t.tool_args or {}) for t in train_calls),
            f"No start_training call references a 350M base model: "
            f"{[t.tool_args for t in train_calls]}",
        )

        # --- The GGUF export was requested with a Q4_K quant ---
        gguf_calls = [t for t in result.tool_calls if t.tool_name == "gguf_convert"]
        self.assertTrue(gguf_calls, "gguf_convert was never called")
        quants = [
            q
            for t in gguf_calls
            for q in (t.tool_args or {}).get("quant_types", [])
        ]
        self.assertTrue(
            any(str(q).upper().startswith("Q4_K") for q in quants),
            f"No Q4_K quant requested; got quant_types={quants}",
        )

        # --- The .gguf artifact actually registered (async cloud job) ---
        handles = asyncio.run(_wait_for_gguf_artifacts(result.project_dir.name))
        self.assertTrue(
            handles,
            f"No artifacts of kind 'gguf' registered for project "
            f"{result.project_dir.name} within {_GGUF_POLL_TIMEOUT}s",
        )
        logger.info("GGUF artifacts: %s", [h.id for h in handles])

        # --- LLM Judge on the spec ---
        async def _judge() -> None:
            from lqh.auth import require_token
            from lqh.client import create_client
            from lqh.config import load_config

            config = load_config()
            client = create_client(require_token(), config.api_base_url)

            judge_results = await judge_artifacts(
                client, OBSERVABILITY_GGUF_350M, artifacts,
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
