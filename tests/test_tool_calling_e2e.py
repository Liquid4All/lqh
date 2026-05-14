"""API-only integration test for tool calling pipeline.

Tests the full data generation + scoring flow for tool-calling data
without requiring a GPU.  Exercises:

1. Generate ~10 samples with tool calls using the pipeline engine
2. Verify parquet has the ``tools`` column and messages contain ``tool_calls``
3. Score the generated data using the tool-call-aware scorer
4. Verify scoring produces meaningful results

Requires:
  - LQH API access (api.lqh.ai)

Usage::

    pytest tests/test_tool_calling_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

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


def _has_api_access() -> bool:
    try:
        from lqh.auth import get_token
        return get_token() is not None
    except Exception:
        return False


@pytest.fixture
def api_client():
    """Create an authenticated API client."""
    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config

    config = load_config()
    token = require_token()
    return create_client(token, config.api_base_url)


@pytest.mark.skipif(not _has_api_access(), reason="No API access")
class TestToolCallingE2E:
    """End-to-end test for tool calling data generation and scoring."""

    @pytest.mark.asyncio
    async def test_generate_and_score_tool_calling_data(
        self, tmp_path: Path, api_client,
    ):
        # ---- Step 1: Generate tool-calling samples ----
        print(f"\n[1/4] Generating {NUM_SAMPLES} tool-calling samples...")

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
        print(f"  Generated: {result.succeeded}/{result.total} succeeded")
        assert result.succeeded >= NUM_SAMPLES * 0.7, (
            f"Too many failures: {result.failed}/{result.total}"
        )

        # ---- Step 2: Verify parquet structure ----
        print("\n[2/4] Verifying parquet structure...")

        data_path = output_dir / "data.parquet"
        assert data_path.exists()

        table = pq.read_table(str(data_path))
        assert "messages" in table.column_names
        assert "tools" in table.column_names, "Missing 'tools' column in parquet"

        conversations, tools_list = load_dataset_with_tools(data_path)
        assert len(conversations) == result.succeeded

        # Check that samples have tool calls and tool definitions
        samples_with_tools = sum(1 for t in tools_list if t is not None)
        samples_with_tool_calls = 0
        for conv in conversations:
            has_tc = any(msg.get("tool_calls") for msg in conv)
            if has_tc:
                samples_with_tool_calls += 1

        print(f"  Samples with tools column: {samples_with_tools}/{len(conversations)}")
        print(f"  Samples with tool_calls: {samples_with_tool_calls}/{len(conversations)}")

        assert samples_with_tools == len(conversations), "All samples should have tools"
        assert samples_with_tool_calls == len(conversations), "All samples should have tool_calls"

        # Spot-check a conversation
        conv = conversations[0]
        tools = tools_list[0]

        # Should have system, user, assistant (with tool call), tool result, assistant
        roles = [msg["role"] for msg in conv]
        assert "system" in roles
        assert "user" in roles
        assert "tool" in roles
        assert roles.count("assistant") >= 2  # pre-tool and post-tool

        # Tools should be in OpenAI format
        assert tools is not None
        assert all("function" in t for t in tools)
        assert all("name" in t["function"] for t in tools)

        print(f"  Sample tools: {[t['function']['name'] for t in tools[:3]]}...")
        print(f"  Sample roles: {roles}")

        # ---- Step 3: Score the data ----
        print("\n[3/4] Scoring tool-calling data...")

        scorer_path = tmp_path / "scorer.md"
        scorer_path.write_text(_TOOL_CALLING_SCORER)

        scoring_result = await run_data_scoring(
            dataset_dir=output_dir,
            scorer_path=scorer_path,
            client=api_client,
            model_size="small",
            concurrency=3,
        )
        print(f"  Scored: {scoring_result.scored}/{scoring_result.total}")
        print(f"  Mean score: {scoring_result.mean_score:.2f}/10")
        print(f"  Median score: {scoring_result.median_score:.2f}/10")

        # ---- Step 4: Verify results ----
        print("\n[4/4] Verifying results...")

        assert scoring_result.scored > 0, "No samples were scored"
        assert scoring_result.mean_score > 0, "Mean score should be positive"

        # Check scores.parquet was written
        scores_path = output_dir / "scores.parquet"
        assert scores_path.exists(), "scores.parquet not written"

        scores_table = pq.read_table(str(scores_path))
        assert len(scores_table) == scoring_result.total

        print(f"\n  All checks passed!")
        print(f"  Summary: {result.succeeded} samples generated, "
              f"mean score = {scoring_result.mean_score:.2f}/10")
