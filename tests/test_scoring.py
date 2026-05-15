"""Tests for the scoring engine (lqh/scoring.py).

Unit tests verify prompt building, response parsing, and system prompt
injection.  Integration tests run actual scoring against ``api.lqh.ai``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
# Local fixtures
# ---------------------------------------------------------------------------


SAMPLE_CONVERSATIONS = [
    [
        {"role": "system", "content": "You summarize things."},
        {"role": "user", "content": "Summarize: The cat sat on the mat."},
        {"role": "assistant", "content": "A cat was on a mat."},
    ],
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ],
]


@pytest.fixture
def judge_client() -> MagicMock:
    """A mock ``AsyncOpenAI`` whose judge call returns a fixed JSON score."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content='{"reasoning": "Good", "score": 8}'))
    ]
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.fixture
def inference_runner() -> MagicMock:
    """A mock ``ModelRunner`` that always returns a canned ``RunnerResponse``."""
    runner = MagicMock()
    runner.complete = AsyncMock(
        return_value=RunnerResponse(
            content="output", model="m", usage=None,
        )
    )
    return runner


@pytest.fixture
def scoring_workspace(
    tmp_path: Path,
    write_chatml_parquet,
) -> dict[str, Path]:
    """Set up the parquet + scorer + output dir layout used by run_scoring."""
    dataset_path = write_chatml_parquet(
        tmp_path / "data.parquet", SAMPLE_CONVERSATIONS, audio=True,
    )
    scorer_path = tmp_path / "scorer.md"
    scorer_path.write_text("Score the response for accuracy and clarity on a 1-10 scale.")
    output_dir = tmp_path / "output"
    return {
        "dataset_path": dataset_path,
        "scorer_path": scorer_path,
        "output_dir": output_dir,
    }


@pytest.fixture
def run_scoring_call(
    scoring_workspace: dict[str, Path],
    judge_client: MagicMock,
    inference_runner: MagicMock,
) -> Callable[..., Any]:
    """Invoke ``run_scoring`` with the standard workspace + defaults.

    Tests can override any kwarg, e.g. swap the runner or drop ``run_inference``.
    """

    async def _call(**overrides: Any) -> Any:
        kwargs = dict(
            client=judge_client,
            run_inference=True,
            inference_model="small",
            inference_runner=inference_runner,
            **scoring_workspace,
        )
        kwargs.update(overrides)
        return await run_scoring(**kwargs)

    return _call


# ---------------------------------------------------------------------------
# _strip_trailing_assistant
# ---------------------------------------------------------------------------


class TestStripTrailingAssistant:
    def test_strips_single_trailing_assistant(self) -> None:
        result = _strip_trailing_assistant([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_strips_multiple_trailing_assistants(self) -> None:
        result = _strip_trailing_assistant([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ])
        assert len(result) == 1

    def test_preserves_non_trailing_assistant(self) -> None:
        result = _strip_trailing_assistant([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "bye"},
        ])
        assert len(result) == 3
        assert result[-1]["role"] == "user"

    def test_empty_list(self) -> None:
        assert _strip_trailing_assistant([]) == []

    def test_no_trailing_assistant(self) -> None:
        result = _strip_trailing_assistant([{"role": "user", "content": "hi"}])
        assert len(result) == 1

    def test_does_not_mutate_original(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        _strip_trailing_assistant(msgs)
        assert len(msgs) == 2


# ---------------------------------------------------------------------------
# _build_scoring_prompt
# ---------------------------------------------------------------------------


class TestBuildScoringPrompt:
    def test_contains_scorer_text_and_sample(self) -> None:
        prompt = _build_scoring_prompt(
            "Score clarity 1-10",
            [
                {"role": "user", "content": "summarize this"},
                {"role": "assistant", "content": "here is the summary"},
            ],
        )
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert "evaluator" in prompt[0]["content"]
        assert "Score clarity 1-10" in prompt[1]["content"]
        assert "summarize this" in prompt[1]["content"]

    def test_includes_reference_when_provided(self) -> None:
        prompt = _build_scoring_prompt(
            "criteria",
            [{"role": "assistant", "content": "generated"}],
            reference_messages=[{"role": "assistant", "content": "reference"}],
        )
        assert "Reference (ground truth)" in prompt[1]["content"]
        assert "reference" in prompt[1]["content"]


# ---------------------------------------------------------------------------
# _parse_score_response
# ---------------------------------------------------------------------------


class TestParseScoreResponse:
    @pytest.mark.parametrize(
        "payload,expected_score,expected_reasoning_contains",
        [
            ('{"reasoning": "Good work", "score": 8}', 8.0, "Good work"),
            ('  {"reasoning": "ok", "score": 7}  ', 7.0, "ok"),
        ],
    )
    def test_valid_payloads(
        self, payload: str, expected_score: float, expected_reasoning_contains: str,
    ) -> None:
        score, reasoning = _parse_score_response(payload)
        assert score == expected_score
        assert expected_reasoning_contains in reasoning

    def test_invalid_json_returns_zero(self) -> None:
        score, reasoning = _parse_score_response("not json")
        assert score == 0.0
        assert "Parse error" in reasoning

    def test_missing_score_returns_zero(self) -> None:
        score, _ = _parse_score_response('{"reasoning": "ok"}')
        assert score == 0.0


# ---------------------------------------------------------------------------
# run_scoring with mocked client + runner
# ---------------------------------------------------------------------------


class TestScoringWithMockRunner:
    """Verify that run_scoring correctly uses the ModelRunner for inference."""

    async def test_inference_runner_is_called(
        self, run_scoring_call, inference_runner: MagicMock,
    ) -> None:
        """``run_inference=True`` routes through ``runner.complete()``."""
        inference_runner.complete = AsyncMock(
            return_value=RunnerResponse(
                content="Mocked inference output",
                model="test-model",
                usage=RunnerUsage(prompt_tokens=10, completion_tokens=5),
            )
        )

        result = await run_scoring_call(inference_model="test-model")

        assert inference_runner.complete.call_count == 2
        for call in inference_runner.complete.call_args_list:
            assert call.kwargs["model"] == "test-model"
        assert result.total == 2

    async def test_system_prompt_injected_via_runner(
        self, run_scoring_call, inference_runner: MagicMock,
    ) -> None:
        """``inference_system_prompt`` is prepended to messages sent to the runner."""
        await run_scoring_call(
            inference_system_prompt="You are a concise summarizer.",
        )

        for call in inference_runner.complete.call_args_list:
            messages = call.args[0]
            system_msgs = [m for m in messages if m["role"] == "system"]
            assert system_msgs, "every call should carry a system message"

    async def test_system_prompt_replaces_existing(
        self, run_scoring_call, inference_runner: MagicMock,
    ) -> None:
        """Existing system messages are stripped and replaced by ``inference_system_prompt``."""
        await run_scoring_call(inference_system_prompt="Custom prompt")

        first_call_messages = inference_runner.complete.call_args_list[0].args[0]
        system_msgs = [m for m in first_call_messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Custom prompt"

    async def test_response_format_passed_to_runner(
        self, run_scoring_call, inference_runner: MagicMock,
    ) -> None:
        """``inference_response_format`` is passed through to ``runner.complete()``."""
        schema = {"type": "json_object"}
        inference_runner.complete = AsyncMock(
            return_value=RunnerResponse(
                content='{"de":"a","fr":"b"}', model="m", usage=None,
            )
        )

        await run_scoring_call(inference_response_format=schema)

        for call in inference_runner.complete.call_args_list:
            assert call.kwargs["response_format"] == schema

    async def test_fallback_to_api_runner_when_no_runner_provided(
        self, run_scoring_call,
    ) -> None:
        """With ``inference_runner=None``, scoring wraps the client in ``APIModelRunner``."""
        result = await run_scoring_call(inference_runner=None)
        assert result.total == 2

    async def test_summary_json_includes_system_prompt(
        self, run_scoring_call, scoring_workspace: dict[str, Path],
    ) -> None:
        await run_scoring_call(
            inference_model="test-model",
            inference_system_prompt="Be concise.",
        )

        summary = json.loads((scoring_workspace["output_dir"] / "summary.json").read_text())
        assert summary["inference_model"] == "test-model"
        assert summary["inference_system_prompt"] == "Be concise."

    async def test_summary_json_omits_system_prompt_when_none(
        self, run_scoring_call, scoring_workspace: dict[str, Path],
    ) -> None:
        await run_scoring_call(inference_model="test-model")

        summary = json.loads((scoring_workspace["output_dir"] / "summary.json").read_text())
        assert "inference_system_prompt" not in summary

    async def test_results_parquet_written(
        self, run_scoring_call, scoring_workspace: dict[str, Path],
    ) -> None:
        await run_scoring_call()

        results_path = scoring_workspace["output_dir"] / "results.parquet"
        assert results_path.exists()
        table = pq.read_table(results_path)
        assert len(table) == 2
        assert {"score", "reasoning", "messages"} <= set(table.column_names)


# ---------------------------------------------------------------------------
# extract_failures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_results_parquet(tmp_path: Path) -> Callable[[list[float]], Path]:
    """Factory writing a ``results.parquet`` with the given score column."""

    def _factory(scores: list[float]) -> Path:
        rows = {
            "sample_index": list(range(len(scores))),
            "messages": [
                json.dumps([
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ])
                for i in range(len(scores))
            ],
            "score": scores,
            "reasoning": [f"Score reasoning for sample {i}" for i in range(len(scores))],
        }
        table = pa.table(
            rows,
            schema=pa.schema([
                pa.field("sample_index", pa.int64()),
                pa.field("messages", pa.string()),
                pa.field("score", pa.float64()),
                pa.field("reasoning", pa.string()),
            ]),
        )
        path = tmp_path / "results.parquet"
        pq.write_table(table, path)
        return path

    return _factory


class TestExtractFailures:
    def test_below_threshold_returned(self, make_results_parquet) -> None:
        path = make_results_parquet([2.0, 4.0, 7.0, 9.0, 8.0])
        failures, _ = extract_failures(path, threshold=6.0, min_failures=0)
        assert [f["score"] for f in failures] == [2.0, 4.0]

    def test_padding_with_bottom_n(self, make_results_parquet) -> None:
        """1 below threshold but ``min_failures=3`` → pad from the bottom."""
        path = make_results_parquet([3.0, 7.0, 8.0, 6.5, 9.0])
        failures, _ = extract_failures(path, threshold=6.0, min_failures=3)
        assert len(failures) == 3
        assert [f["score"] for f in failures] == [3.0, 6.5, 7.0]

    def test_max_cap(self, make_results_parquet) -> None:
        path = make_results_parquet([1.0, 2.0, 3.0, 4.0, 5.0])
        failures, _ = extract_failures(
            path, threshold=10.0, min_failures=0, max_failures=3,
        )
        assert len(failures) == 3

    def test_sorted_ascending(self, make_results_parquet) -> None:
        path = make_results_parquet([5.0, 2.0, 4.0, 1.0, 3.0])
        failures, _ = extract_failures(path, threshold=6.0, min_failures=0)
        scores = [f["score"] for f in failures]
        assert scores == sorted(scores)

    def test_empty_when_all_high(self, make_results_parquet) -> None:
        path = make_results_parquet([8.0, 9.0, 10.0])
        failures, _ = extract_failures(path, threshold=6.0, min_failures=0)
        assert failures == []

    def test_messages_parsed(self, make_results_parquet) -> None:
        path = make_results_parquet([1.0])
        failures, _ = extract_failures(path, threshold=6.0, min_failures=0)
        assert len(failures) == 1
        assert isinstance(failures[0]["messages"], list)
        assert failures[0]["messages"][0]["role"] == "user"

    def test_min_failures_exceeding_dataset_size(self, make_results_parquet) -> None:
        path = make_results_parquet([8.0, 9.0])
        failures, _ = extract_failures(path, threshold=6.0, min_failures=10)
        assert len(failures) == 2


# ---------------------------------------------------------------------------
# Integration tests (require API access)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestScoringIntegration:
    """Integration tests that actually call the scoring API."""

    @pytest.fixture
    def make_dataset(
        self, tmp_path: Path, write_chatml_parquet,
    ) -> Callable[..., Path]:
        def _factory(samples: list[list[dict]], *, name: str = "data.parquet") -> Path:
            return write_chatml_parquet(tmp_path / name, samples, audio=True)

        return _factory

    async def test_data_quality_scoring(
        self, tmp_path: Path, api_client: Any, write_chatml_parquet,
    ) -> None:
        """Score data quality on a tiny dataset (no inference, judge only)."""
        dataset_dir = tmp_path / "ds"
        write_chatml_parquet(
            dataset_dir / "data.parquet",
            [[
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]],
            audio=True,
        )

        scorer_path = tmp_path / "scorer.md"
        scorer_path.write_text(
            "Score the assistant's response for factual accuracy.\n"
            "- 10: Perfectly correct\n"
            "- 5: Partially correct\n"
            "- 1: Completely wrong\n"
        )

        result = await run_data_scoring(
            dataset_dir=dataset_dir,
            scorer_path=scorer_path,
            client=api_client,
            model_size="small",
        )

        assert result.total == 1
        assert result.scored == 1
        assert result.failed == 0
        assert result.mean_score >= 6.0  # Paris is correct
        assert (dataset_dir / "scores.parquet").exists()

    async def test_model_eval_with_lfm(
        self, tmp_path: Path, api_client: Any, make_dataset,
    ) -> None:
        """Run model eval: strip assistant turns, run LFM inference, score output."""
        dataset_path = make_dataset([
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
            ],
            [
                {"role": "user", "content": "Name the largest planet in our solar system."},
                {"role": "assistant", "content": "Jupiter is the largest planet."},
            ],
        ])

        scorer_path = tmp_path / "scorer.md"
        scorer_path.write_text(
            "Score the assistant's response for correctness.\n"
            "- 10: Correct and complete\n"
            "- 5: Partially correct\n"
            "- 1: Wrong\n"
        )

        output_dir = tmp_path / "eval_run"
        result = await run_scoring(
            dataset_path=dataset_path,
            scorer_path=scorer_path,
            output_dir=output_dir,
            client=api_client,
            model_size="small",
            run_inference=True,
            inference_model="small",
            inference_system_prompt="Answer questions accurately and concisely.",
        )

        assert result.total == 2
        assert result.scored > 0
        assert (output_dir / "results.parquet").exists()
        assert (output_dir / "summary.json").exists()

        summary = json.loads((output_dir / "summary.json").read_text())
        assert summary["inference_model"] == "small"
        assert summary["inference_system_prompt"] == "Answer questions accurately and concisely."
        assert "scores" in summary

    async def test_model_eval_with_runner(
        self, tmp_path: Path, api_client: Any, make_dataset,
    ) -> None:
        """Same as above but explicitly passing an ``APIModelRunner``."""
        dataset_path = make_dataset([[
            {"role": "user", "content": "What color is the sky on a clear day?"},
            {"role": "assistant", "content": "Blue."},
        ]])

        scorer_path = tmp_path / "scorer.md"
        scorer_path.write_text("Score for correctness, 1-10.")

        result = await run_scoring(
            dataset_path=dataset_path,
            scorer_path=scorer_path,
            output_dir=tmp_path / "eval_run",
            client=api_client,
            run_inference=True,
            inference_model="small",
            inference_runner=APIModelRunner(api_client),
        )
        assert result.total == 1
        assert result.scored > 0
