"""Tests for the scoring engine (lqh/scoring.py).

Unit tests verify prompt building, response parsing, and system prompt injection.
Integration tests run actual scoring against api.lqh.ai.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pyarrow.parquet as pq

from lqh.runner import APIModelRunner, RunnerResponse, RunnerUsage
from lqh.scoring import (
    _build_scoring_prompt,
    _parse_score_response,
    _strip_trailing_assistant,
    extract_failures,
    run_data_scoring,
    run_scoring,
)


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------


class TestStripTrailingAssistant(unittest.TestCase):

    def test_strips_single_trailing_assistant(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _strip_trailing_assistant(msgs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")

    def test_strips_multiple_trailing_assistants(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
        result = _strip_trailing_assistant(msgs)
        self.assertEqual(len(result), 1)

    def test_preserves_non_trailing_assistant(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "bye"},
        ]
        result = _strip_trailing_assistant(msgs)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[-1]["role"], "user")

    def test_empty_list(self) -> None:
        self.assertEqual(_strip_trailing_assistant([]), [])

    def test_no_trailing_assistant(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = _strip_trailing_assistant(msgs)
        self.assertEqual(len(result), 1)

    def test_does_not_mutate_original(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        original_len = len(msgs)
        _strip_trailing_assistant(msgs)
        self.assertEqual(len(msgs), original_len)


class TestBuildScoringPrompt(unittest.TestCase):

    def test_contains_scorer_text_and_sample(self) -> None:
        msgs = [
            {"role": "user", "content": "summarize this"},
            {"role": "assistant", "content": "here is the summary"},
        ]
        prompt = _build_scoring_prompt("Score clarity 1-10", msgs)
        self.assertEqual(len(prompt), 2)
        self.assertEqual(prompt[0]["role"], "system")
        self.assertIn("evaluator", prompt[0]["content"])
        self.assertIn("Score clarity 1-10", prompt[1]["content"])
        self.assertIn("summarize this", prompt[1]["content"])

    def test_includes_reference_when_provided(self) -> None:
        msgs = [{"role": "assistant", "content": "generated"}]
        ref = [{"role": "assistant", "content": "reference"}]
        prompt = _build_scoring_prompt("criteria", msgs, reference_messages=ref)
        self.assertIn("Reference (ground truth)", prompt[1]["content"])
        self.assertIn("reference", prompt[1]["content"])


class TestParseScoreResponse(unittest.TestCase):

    def test_valid_json(self) -> None:
        score, reasoning = _parse_score_response('{"reasoning": "Good work", "score": 8}')
        self.assertEqual(score, 8.0)
        self.assertEqual(reasoning, "Good work")

    def test_invalid_json_returns_zero(self) -> None:
        score, reasoning = _parse_score_response("not json")
        self.assertEqual(score, 0.0)
        self.assertIn("Parse error", reasoning)

    def test_missing_score_returns_zero(self) -> None:
        score, _ = _parse_score_response('{"reasoning": "ok"}')
        self.assertEqual(score, 0.0)

    def test_whitespace_handling(self) -> None:
        score, _ = _parse_score_response('  {"reasoning": "ok", "score": 7}  ')
        self.assertEqual(score, 7.0)


class TestScoringWithMockRunner(unittest.TestCase):
    """Verify that run_scoring correctly uses the ModelRunner for inference."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.dataset_path = Path(self.tmpdir) / "data.parquet"
        self.scorer_path = Path(self.tmpdir) / "scorer.md"
        self.output_dir = Path(self.tmpdir) / "output"

        # Create a small dataset with 2 samples
        messages = [
            json.dumps([
                {"role": "system", "content": "You summarize things."},
                {"role": "user", "content": "Summarize: The cat sat on the mat."},
                {"role": "assistant", "content": "A cat was on a mat."},
            ]),
            json.dumps([
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]),
        ]
        table = pa.table(
            {"messages": messages, "audio": [None, None]},
            schema=pa.schema([pa.field("messages", pa.string()), pa.field("audio", pa.string())]),
        )
        pq.write_table(table, self.dataset_path)
        self.scorer_path.write_text("Score the response for accuracy and clarity on a 1-10 scale.")

    def _make_mock_client(self) -> MagicMock:
        """Create a mock AsyncOpenAI that returns valid scoring responses."""
        mock = MagicMock()
        # The judge scoring call goes through the raw client
        scoring_response = MagicMock()
        scoring_response.choices = [
            MagicMock(message=MagicMock(content='{"reasoning": "Good", "score": 8}'))
        ]
        mock.chat.completions.create = AsyncMock(return_value=scoring_response)
        return mock

    def test_inference_runner_is_called(self) -> None:
        """When run_inference=True, the runner.complete() is used instead of client."""
        mock_client = self._make_mock_client()
        mock_runner = MagicMock()
        mock_runner.complete = AsyncMock(
            return_value=RunnerResponse(
                content="Mocked inference output",
                model="test-model",
                usage=RunnerUsage(prompt_tokens=10, completion_tokens=5),
            )
        )

        result = asyncio.run(
            run_scoring(
                dataset_path=self.dataset_path,
                scorer_path=self.scorer_path,
                output_dir=self.output_dir,
                client=mock_client,
                run_inference=True,
                inference_model="test-model",
                inference_runner=mock_runner,
            )
        )

        # Runner should have been called for inference (2 samples)
        self.assertEqual(mock_runner.complete.call_count, 2)
        # Verify model was passed correctly
        for call in mock_runner.complete.call_args_list:
            self.assertEqual(call[1]["model"], "test-model")

        self.assertEqual(result.total, 2)

    def test_system_prompt_injected_via_runner(self) -> None:
        """inference_system_prompt is prepended to messages sent to the runner."""
        mock_client = self._make_mock_client()
        mock_runner = MagicMock()
        mock_runner.complete = AsyncMock(
            return_value=RunnerResponse(content="output", model="m", usage=None)
        )

        asyncio.run(
            run_scoring(
                dataset_path=self.dataset_path,
                scorer_path=self.scorer_path,
                output_dir=self.output_dir,
                client=mock_client,
                run_inference=True,
                inference_model="small",
                inference_system_prompt="You are a concise summarizer.",
                inference_runner=mock_runner,
            )
        )

        # Check that the second sample (no system message) got the prompt injected
        # Sample 2: [{"role":"user","content":"What is 2+2?"}] -> should get system prepended
        calls = mock_runner.complete.call_args_list
        for call in calls:
            messages = call[0][0]
            # All calls should have a system message (either original or injected)
            system_msgs = [m for m in messages if m["role"] == "system"]
            self.assertGreater(len(system_msgs), 0)

    def test_system_prompt_replaces_existing(self) -> None:
        """Existing system messages are stripped and replaced by inference_system_prompt."""
        mock_client = self._make_mock_client()
        mock_runner = MagicMock()
        mock_runner.complete = AsyncMock(
            return_value=RunnerResponse(content="output", model="m", usage=None)
        )

        asyncio.run(
            run_scoring(
                dataset_path=self.dataset_path,
                scorer_path=self.scorer_path,
                output_dir=self.output_dir,
                client=mock_client,
                run_inference=True,
                inference_model="small",
                inference_system_prompt="Custom prompt",
                inference_runner=mock_runner,
            )
        )

        # Sample 1 had "You summarize things." as system — should be replaced
        call_0_messages = mock_runner.complete.call_args_list[0][0][0]
        system_msgs = [m for m in call_0_messages if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(system_msgs[0]["content"], "Custom prompt")

    def test_response_format_passed_to_runner(self) -> None:
        """inference_response_format is passed through to runner.complete()."""
        mock_client = self._make_mock_client()
        mock_runner = MagicMock()
        mock_runner.complete = AsyncMock(
            return_value=RunnerResponse(content='{"de":"a","fr":"b"}', model="m", usage=None)
        )

        test_schema = {"type": "json_object"}

        asyncio.run(
            run_scoring(
                dataset_path=self.dataset_path,
                scorer_path=self.scorer_path,
                output_dir=self.output_dir,
                client=mock_client,
                run_inference=True,
                inference_model="small",
                inference_response_format=test_schema,
                inference_runner=mock_runner,
            )
        )

        # Every runner.complete() call should have response_format
        for call in mock_runner.complete.call_args_list:
            self.assertEqual(call[1]["response_format"], test_schema)

    def test_fallback_to_api_runner_when_no_runner_provided(self) -> None:
        """When inference_runner is None, an APIModelRunner wrapping client is used."""
        mock_client = self._make_mock_client()
        # The same mock handles both inference and scoring calls

        result = asyncio.run(
            run_scoring(
                dataset_path=self.dataset_path,
                scorer_path=self.scorer_path,
                output_dir=self.output_dir,
                client=mock_client,
                run_inference=True,
                inference_model="small",
                # No inference_runner -> falls back to APIModelRunner(client)
            )
        )

        # Should succeed using the mock client directly
        self.assertEqual(result.total, 2)

    def test_summary_json_includes_system_prompt(self) -> None:
        """summary.json should record the inference_system_prompt if provided."""
        mock_client = self._make_mock_client()
        mock_runner = MagicMock()
        mock_runner.complete = AsyncMock(
            return_value=RunnerResponse(content="output", model="m", usage=None)
        )

        asyncio.run(
            run_scoring(
                dataset_path=self.dataset_path,
                scorer_path=self.scorer_path,
                output_dir=self.output_dir,
                client=mock_client,
                run_inference=True,
                inference_model="test-model",
                inference_system_prompt="Be concise.",
                inference_runner=mock_runner,
            )
        )

        summary = json.loads((self.output_dir / "summary.json").read_text())
        self.assertEqual(summary["inference_model"], "test-model")
        self.assertEqual(summary["inference_system_prompt"], "Be concise.")

    def test_summary_json_omits_system_prompt_when_none(self) -> None:
        mock_client = self._make_mock_client()
        mock_runner = MagicMock()
        mock_runner.complete = AsyncMock(
            return_value=RunnerResponse(content="output", model="m", usage=None)
        )

        asyncio.run(
            run_scoring(
                dataset_path=self.dataset_path,
                scorer_path=self.scorer_path,
                output_dir=self.output_dir,
                client=mock_client,
                run_inference=True,
                inference_model="test-model",
                inference_runner=mock_runner,
            )
        )

        summary = json.loads((self.output_dir / "summary.json").read_text())
        self.assertNotIn("inference_system_prompt", summary)

    def test_results_parquet_written(self) -> None:
        mock_client = self._make_mock_client()
        mock_runner = MagicMock()
        mock_runner.complete = AsyncMock(
            return_value=RunnerResponse(content="output", model="m", usage=None)
        )

        asyncio.run(
            run_scoring(
                dataset_path=self.dataset_path,
                scorer_path=self.scorer_path,
                output_dir=self.output_dir,
                client=mock_client,
                run_inference=True,
                inference_model="small",
                inference_runner=mock_runner,
            )
        )

        results_path = self.output_dir / "results.parquet"
        self.assertTrue(results_path.exists())
        table = pq.read_table(results_path)
        self.assertEqual(len(table), 2)
        self.assertIn("score", table.column_names)
        self.assertIn("reasoning", table.column_names)
        self.assertIn("messages", table.column_names)


# ---------------------------------------------------------------------------
# extract_failures tests (no network)
# ---------------------------------------------------------------------------


class TestExtractFailures(unittest.TestCase):
    """Unit tests for the extract_failures() function."""

    def _make_results(self, scores: list[float], tmpdir: str) -> Path:
        """Create a results.parquet with given scores."""
        rows = []
        for i, score in enumerate(scores):
            rows.append({
                "sample_index": i,
                "messages": json.dumps([
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ]),
                "score": score,
                "reasoning": f"Score reasoning for sample {i}",
            })
        table = pa.table(
            {
                "sample_index": [r["sample_index"] for r in rows],
                "messages": [r["messages"] for r in rows],
                "score": [r["score"] for r in rows],
                "reasoning": [r["reasoning"] for r in rows],
            },
            schema=pa.schema([
                pa.field("sample_index", pa.int64()),
                pa.field("messages", pa.string()),
                pa.field("score", pa.float64()),
                pa.field("reasoning", pa.string()),
            ]),
        )
        path = Path(tmpdir) / "results.parquet"
        pq.write_table(table, path)
        return path

    def test_below_threshold_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_results([2.0, 4.0, 7.0, 9.0, 8.0], tmpdir)
            failures, _ = extract_failures(path, threshold=6.0, min_failures=0)
            scores = [f["score"] for f in failures]
            self.assertEqual(scores, [2.0, 4.0])

    def test_padding_with_bottom_n(self) -> None:
        """If only 1 sample below threshold but min_failures=3, pad from bottom."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_results([3.0, 7.0, 8.0, 6.5, 9.0], tmpdir)
            failures, _ = extract_failures(path, threshold=6.0, min_failures=3)
            self.assertEqual(len(failures), 3)
            # Should include the one below threshold + 2 next lowest
            scores = [f["score"] for f in failures]
            self.assertEqual(scores, [3.0, 6.5, 7.0])

    def test_max_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_results([1.0, 2.0, 3.0, 4.0, 5.0], tmpdir)
            failures, _ = extract_failures(path, threshold=10.0, min_failures=0, max_failures=3)
            self.assertEqual(len(failures), 3)

    def test_sorted_ascending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_results([5.0, 2.0, 4.0, 1.0, 3.0], tmpdir)
            failures, _ = extract_failures(path, threshold=6.0, min_failures=0)
            scores = [f["score"] for f in failures]
            self.assertEqual(scores, sorted(scores))

    def test_empty_when_all_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_results([8.0, 9.0, 10.0], tmpdir)
            failures, _ = extract_failures(path, threshold=6.0, min_failures=0)
            self.assertEqual(failures, [])

    def test_messages_parsed(self) -> None:
        """Messages should be parsed from JSON, not raw strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_results([1.0], tmpdir)
            failures, _ = extract_failures(path, threshold=6.0, min_failures=0)
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0]["messages"], list)
            self.assertEqual(failures[0]["messages"][0]["role"], "user")

    def test_min_failures_exceeding_dataset_size(self) -> None:
        """min_failures > dataset size should return all samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_results([8.0, 9.0], tmpdir)
            failures, _ = extract_failures(path, threshold=6.0, min_failures=10)
            self.assertEqual(len(failures), 2)


# ---------------------------------------------------------------------------
# Integration tests (require API access)
# ---------------------------------------------------------------------------


def _has_api_access() -> bool:
    try:
        from lqh.auth import get_token
        return get_token() is not None
    except Exception:
        return False


@unittest.skipUnless(_has_api_access(), "No API access (set LQH_DEBUG_API_KEY or run /login)")
class TestScoringIntegration(unittest.TestCase):
    """Integration tests that actually call the scoring API."""

    def setUp(self) -> None:
        from lqh.auth import require_token
        from lqh.client import create_client
        from lqh.config import load_config

        config = load_config()
        token = require_token()
        self.client = create_client(token, config.api_base_url)

    def _make_dataset(self, tmpdir: str, samples: list[list[dict]]) -> Path:
        """Write a small parquet dataset from ChatML conversations."""
        messages = [json.dumps(s) for s in samples]
        table = pa.table(
            {"messages": messages, "audio": [None] * len(messages)},
            schema=pa.schema([pa.field("messages", pa.string()), pa.field("audio", pa.string())]),
        )
        path = Path(tmpdir) / "data.parquet"
        pq.write_table(table, path)
        return path

    def test_data_quality_scoring(self) -> None:
        """Score data quality on a tiny dataset (no inference, judge only)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "ds"
            dataset_dir.mkdir()

            samples = [
                [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."},
                ],
            ]
            # _make_dataset writes to tmpdir/data.parquet, we need it in dataset_dir
            messages = [json.dumps(s) for s in samples]
            table = pa.table(
                {"messages": messages, "audio": [None] * len(messages)},
                schema=pa.schema([pa.field("messages", pa.string()), pa.field("audio", pa.string())]),
            )
            pq.write_table(table, dataset_dir / "data.parquet")

            scorer_path = Path(tmpdir) / "scorer.md"
            scorer_path.write_text(
                "Score the assistant's response for factual accuracy.\n"
                "- 10: Perfectly correct\n"
                "- 5: Partially correct\n"
                "- 1: Completely wrong\n"
            )

            result = asyncio.run(
                run_data_scoring(
                    dataset_dir=dataset_dir,
                    scorer_path=scorer_path,
                    client=self.client,
                    model_size="small",
                )
            )

            self.assertEqual(result.total, 1)
            self.assertEqual(result.scored, 1)
            self.assertEqual(result.failed, 0)
            # Paris is correct, should score high
            self.assertGreaterEqual(result.mean_score, 6.0)

            # Verify output file
            scores_file = dataset_dir / "scores.parquet"
            self.assertTrue(scores_file.exists())

    def test_model_eval_with_lfm(self) -> None:
        """Run model eval: strip assistant turns, run LFM inference, score output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Eval dataset: questions with reference answers
            samples = [
                [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "4"},
                ],
                [
                    {"role": "user", "content": "Name the largest planet in our solar system."},
                    {"role": "assistant", "content": "Jupiter is the largest planet."},
                ],
            ]
            dataset_path = self._make_dataset(tmpdir, samples)

            scorer_path = Path(tmpdir) / "scorer.md"
            scorer_path.write_text(
                "Score the assistant's response for correctness.\n"
                "- 10: Correct and complete\n"
                "- 5: Partially correct\n"
                "- 1: Wrong\n"
            )

            output_dir = Path(tmpdir) / "eval_run"

            result = asyncio.run(
                run_scoring(
                    dataset_path=dataset_path,
                    scorer_path=scorer_path,
                    output_dir=output_dir,
                    client=self.client,
                    model_size="small",
                    run_inference=True,
                    inference_model="small",
                    inference_system_prompt="Answer questions accurately and concisely.",
                )
            )

            self.assertEqual(result.total, 2)
            self.assertGreater(result.scored, 0)

            # Verify outputs
            self.assertTrue((output_dir / "results.parquet").exists())
            self.assertTrue((output_dir / "summary.json").exists())

            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["inference_model"], "small")
            self.assertEqual(summary["inference_system_prompt"], "Answer questions accurately and concisely.")
            self.assertIn("scores", summary)

    def test_model_eval_with_runner(self) -> None:
        """Same as above but explicitly passing an APIModelRunner."""
        runner = APIModelRunner(self.client)

        with tempfile.TemporaryDirectory() as tmpdir:
            samples = [
                [
                    {"role": "user", "content": "What color is the sky on a clear day?"},
                    {"role": "assistant", "content": "Blue."},
                ],
            ]
            dataset_path = self._make_dataset(tmpdir, samples)

            scorer_path = Path(tmpdir) / "scorer.md"
            scorer_path.write_text("Score for correctness, 1-10.")

            output_dir = Path(tmpdir) / "eval_run"

            result = asyncio.run(
                run_scoring(
                    dataset_path=dataset_path,
                    scorer_path=scorer_path,
                    output_dir=output_dir,
                    client=self.client,
                    run_inference=True,
                    inference_model="small",
                    inference_runner=runner,
                )
            )

            self.assertEqual(result.total, 1)
            self.assertGreater(result.scored, 0)


if __name__ == "__main__":
    unittest.main()
