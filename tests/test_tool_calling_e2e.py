"""API-only integration test for the tool-calling pipeline.

Exercises the full data generation + scoring flow for tool-calling data
without requiring a GPU:

1. Generate ~10 samples with tool calls using the pipeline engine.
2. Verify the parquet has the ``tools`` column and messages contain
   ``tool_calls``.
3. Score the generated data using the tool-call-aware scorer.
4. Verify scoring produces meaningful results.

Requires LQH API access (``api.lqh.ai``).  Opt in via
``@pytest.mark.integration``; the suite skips when no credentials are
present.

Usage::

    pytest tests/test_tool_calling_e2e.py -v -s
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from lqh.engine import load_dataset_with_tools, run_pipeline
from lqh.scoring import run_data_scoring

logger = logging.getLogger(__name__)

NUM_SAMPLES = 10

_TOOL_CALLING_SCORER = """\
# Scorer: Tool Calling Quality

## Task Description
The model should correctly call tools in response to user queries.
Each conversation has available tools defined, and the assistant should
select the appropriate tool with correct arguments.

## Conversation Format
The conversation includes:
- A system prompt with available tools
- A user query
- An assistant response with tool calls
- Tool results
- A final assistant response summarizing the result

## Scoring Scale
- **9-10**: Correct tool selected, arguments accurate, natural response
- **7-8**: Correct tool but minor argument issues (e.g., formatting)
- **5-6**: Correct tool but significantly wrong arguments
- **3-4**: Wrong tool selected or missing tool call
- **1-2**: No tool call when one was needed, or completely wrong behavior

## Critical Failures (automatic score <= 3)
- No tool call present when the user clearly needs one
- Calling a tool that doesn't exist in the available tools
"""


@pytest.mark.integration
class TestToolCallingE2E:
    """End-to-end test for tool calling data generation and scoring."""

    async def test_generate_and_score_tool_calling_data(
        self, tmp_path: Path, api_client: Any,
    ) -> None:
        # ---- 1. Generate tool-calling samples ----
        pipeline_path = Path(__file__).parent.parent / "data_gen" / "tool_calling.py"
        assert pipeline_path.exists(), f"Pipeline not found: {pipeline_path}"

        output_dir = tmp_path / "datasets" / "tool_calling_test"
        result = await run_pipeline(
            script_path=pipeline_path,
            num_samples=NUM_SAMPLES,
            output_dir=output_dir,
            client=api_client,
            concurrency=3,
            max_retries=5,
        )
        assert result.succeeded >= NUM_SAMPLES * 0.7, (
            f"Too many failures: {result.failed}/{result.total}"
        )

        # ---- 2. Verify parquet structure ----
        data_path = output_dir / "data.parquet"
        assert data_path.exists()

        table = pq.read_table(str(data_path))
        assert "messages" in table.column_names
        assert "tools" in table.column_names, "Missing 'tools' column in parquet"

        conversations, tools_list = load_dataset_with_tools(data_path)
        assert len(conversations) == result.succeeded

        samples_with_tools = sum(1 for t in tools_list if t is not None)
        samples_with_tool_calls = sum(
            1 for conv in conversations if any(msg.get("tool_calls") for msg in conv)
        )

        assert samples_with_tools == len(conversations)
        assert samples_with_tool_calls == len(conversations)

        # Spot-check a conversation.
        conv = conversations[0]
        tools = tools_list[0]
        roles = [msg["role"] for msg in conv]
        assert "system" in roles
        assert "user" in roles
        assert "tool" in roles
        assert roles.count("assistant") >= 2  # pre-tool and post-tool

        assert tools is not None
        assert all("function" in t for t in tools)
        assert all("name" in t["function"] for t in tools)

        # ---- 3. Score the data ----
        scorer_path = tmp_path / "scorer.md"
        scorer_path.write_text(_TOOL_CALLING_SCORER)

        scoring_result = await run_data_scoring(
            dataset_dir=output_dir,
            scorer_path=scorer_path,
            client=api_client,
            model_size="small",
            concurrency=3,
        )

        # ---- 4. Verify results ----
        assert scoring_result.scored > 0, "No samples were scored"
        assert scoring_result.mean_score > 0, "Mean score should be positive"

        scores_path = output_dir / "scores.parquet"
        assert scores_path.exists()
        assert len(pq.read_table(str(scores_path))) == scoring_result.total
