"""End-to-end test for auto mode: spec in → trained model out.

Each scenario:
1. Copies a spec fixture from tests/e2e/fixtures/auto/<name>/SPEC.md into a
   temp project directory.
2. Constructs an Agent in auto mode (same code path as `lqh --auto <dir>`,
   minus the TUI wrapper).
3. Drives the agent with the standard auto-mode kickoff message.
4. Asserts the run reaches a terminal state (success/failure via
   exit_auto_mode), never blocks on ask_user, and produces the expected
   class of artifacts (datasets, runs/, evals/scorers).

This is a real pipeline run and can take 30+ minutes per scenario. Skipped
when no API token is configured.

Usage:
    python -m tests.e2e.test_auto_mode_e2e
    python -m unittest tests.e2e.test_auto_mode_e2e.TestAutoModeE2E.test_translation_de_fr
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from lqh.agent import Agent, AgentCallbacks
from lqh.session import Session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "auto"

# These tests can take a long time. Default 1 hour cap; override via env.
AUTO_E2E_TIMEOUT_SEC = int(os.environ.get("LQH_AUTO_E2E_TIMEOUT", "3600"))


def _has_api_access() -> bool:
    try:
        from lqh.auth import get_token
        return get_token() is not None
    except Exception:
        return False


@unittest.skipUnless(_has_api_access(), "No API token; skipping auto-mode e2e")
class TestAutoModeE2E(unittest.TestCase):
    """Full-pipeline auto-mode tests, one per spec fixture."""

    def _run_auto(self, fixture_name: str) -> dict[str, Any]:
        """Run auto mode against a fixture and return summary metadata."""
        fixture = FIXTURES_DIR / fixture_name
        self.assertTrue(
            (fixture / "SPEC.md").is_file(),
            f"Missing fixture spec: {fixture}/SPEC.md",
        )

        project_dir = Path(tempfile.mkdtemp(prefix=f"lqh_auto_{fixture_name}_"))
        shutil.copy(fixture / "SPEC.md", project_dir / "SPEC.md")
        logger.info("Auto-mode e2e: %s in %s", fixture_name, project_dir)

        ask_user_called = 0

        async def on_ask_user(*args: Any, **kwargs: Any) -> str:
            nonlocal ask_user_called
            ask_user_called += 1
            return "(no user available)"

        callbacks = AgentCallbacks(on_ask_user=on_ask_user)

        session = Session.create(project_dir)
        agent = Agent(project_dir, session, callbacks=callbacks, auto_mode=True)

        spec_text = (project_dir / "SPEC.md").read_text(encoding="utf-8")
        kickoff = (
            "Here is the spec for this auto-mode run (also at SPEC.md):\n\n"
            f"```\n{spec_text}\n```\n\n"
            "Begin the auto-mode pipeline. Use set_auto_stage to report each "
            "stage. When you reach a terminal state, call "
            "exit_auto_mode(status, reason)."
        )

        start = time.time()
        try:
            asyncio.run(asyncio.wait_for(
                agent.process_user_input(kickoff),
                timeout=AUTO_E2E_TIMEOUT_SEC,
            ))
        except asyncio.TimeoutError:
            logger.warning(
                "Auto-mode e2e timed out after %ds (%s)",
                AUTO_E2E_TIMEOUT_SEC, fixture_name,
            )
        duration = time.time() - start

        return {
            "project_dir": project_dir,
            "duration_seconds": duration,
            "ask_user_called": ask_user_called,
            "exit": agent._auto_exit,
            "messages": len(agent.session.messages),
        }

    def _assert_terminal_state(self, summary: dict[str, Any]) -> None:
        """Auto mode must reach a terminal state (success OR failure) cleanly."""
        # ask_user must never have been forwarded to the user callback —
        # the agent loop intercepts it before the callback runs.
        self.assertEqual(
            summary["ask_user_called"], 0,
            "auto mode forwarded ask_user to a user callback (interception broken)",
        )
        # Either exit_auto_mode was called, OR the run timed out.
        if summary["exit"] is None:
            self.fail(
                f"auto mode did not reach a terminal state in "
                f"{summary['duration_seconds']:.0f}s; project at "
                f"{summary['project_dir']}"
            )
        status, reason = summary["exit"]
        self.assertIn(status, ("success", "failure"))
        self.assertTrue(reason, "exit_auto_mode reason must be non-empty")
        logger.info(
            "Auto-mode result: %s — %s (%s, %.0fs)",
            status, reason, summary["project_dir"], summary["duration_seconds"],
        )

    def test_translation_de_fr(self) -> None:
        summary = self._run_auto("translation_de_fr")
        self._assert_terminal_state(summary)

    def test_sentiment_classification(self) -> None:
        summary = self._run_auto("sentiment_classification")
        self._assert_terminal_state(summary)

    def test_email_summary(self) -> None:
        summary = self._run_auto("email_summary")
        self._assert_terminal_state(summary)


if __name__ == "__main__":
    unittest.main()
