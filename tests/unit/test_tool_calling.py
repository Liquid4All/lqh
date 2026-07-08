"""Unit tests for tool-calling support.

Tests serialization round-trips, LFM2 format conversion, and
tool-call-aware scoring/judge prompt generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    FunctionCall,
    ToolCall,
    ToolDef,
)
from lqh.engine import (
    _extract_tools,
    _serialize_conversation,
    _serialize_message,
    load_dataset_with_tools,
)
from lqh.scoring import (
    _build_scoring_prompt,
    _format_conversation,
    _format_tool_calls,
    _has_tool_calls,
)
from lqh.train.tool_format import LFM2ToolFormatter, get_tool_formatter
from lqh.train.data_utils import (
    chatml_to_sft_dataset,
    load_chatml_dataset_with_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_TOOLS: list[ToolDef] = [
    ToolDef(
        name="get_weather",
        description="Get current weather for a location.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    ),
    ToolDef(
        name="search_products",
        description="Search for products.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
]


def _tc(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """Build an OpenAI-shaped tool-call dict."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.fixture
def tool_call_conversation() -> Conversation:
    """Sample conversation containing a complete tool-call cycle."""
    return [
        ChatMLMessage(
            role="system",
            content="You are a helpful assistant.",
            tools=SAMPLE_TOOLS,
        ),
        ChatMLMessage(role="user", content="What's the weather in SF?"),
        ChatMLMessage(
            role="assistant",
            content="Let me check that for you.",
            tool_calls=[
                ToolCall(
                    id="call_weather_0",
                    function=FunctionCall(
                        name="get_weather",
                        arguments='{"location": "San Francisco"}',
                    ),
                )
            ],
        ),
        ChatMLMessage(
            role="tool",
            content='{"temperature": 65, "condition": "foggy"}',
            tool_call_id="call_weather_0",
            name="get_weather",
        ),
        ChatMLMessage(
            role="assistant",
            content="It's 65F and foggy in San Francisco right now.",
        ),
    ]


@pytest.fixture
def formatter() -> LFM2ToolFormatter:
    return LFM2ToolFormatter()


def _write_tool_parquet(
    path: Path,
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    include_audio: bool = True,
    include_tools_column: bool = True,
) -> Path:
    fields = [pa.field("messages", pa.string())]
    columns: dict[str, list[Any]] = {"messages": [json.dumps(messages)]}
    if include_audio:
        fields.append(pa.field("audio", pa.string()))
        columns["audio"] = [None]
    if include_tools_column:
        fields.append(pa.field("tools", pa.string()))
        columns["tools"] = [json.dumps(tools) if tools is not None else None]

    pq.write_table(pa.table(columns, schema=pa.schema(fields)), path)
    return path


# ---------------------------------------------------------------------------
# Engine serialization
# ---------------------------------------------------------------------------


class TestToolSerialization:
    """Tool-call serialization and parquet round-trip."""

    def test_serialize_message_with_tool_calls(self) -> None:
        msg = ChatMLMessage(
            role="assistant",
            content="Let me check.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="get_weather", arguments='{"location": "NYC"}'),
                )
            ],
        )
        d = _serialize_message(msg)
        assert d["role"] == "assistant"
        assert d["content"] == "Let me check."
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["function"]["name"] == "get_weather"
        assert d["tool_calls"][0]["id"] == "call_1"
        assert d["tool_calls"][0]["type"] == "function"

    def test_serialize_message_with_tool_result(self) -> None:
        msg = ChatMLMessage(
            role="tool",
            content='{"temp": 72}',
            tool_call_id="call_1",
            name="get_weather",
        )
        d = _serialize_message(msg)
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "get_weather"

    def test_serialize_message_with_tools(self) -> None:
        msg = ChatMLMessage(role="system", content="Hello", tools=SAMPLE_TOOLS)
        d = _serialize_message(msg)
        assert len(d["tools"]) == 2
        assert d["tools"][0]["type"] == "function"
        assert d["tools"][0]["function"]["name"] == "get_weather"

    def test_extract_tools(self, tool_call_conversation: Conversation) -> None:
        tools = _extract_tools(tool_call_conversation)
        assert tools is not None
        assert [t["function"]["name"] for t in tools] == ["get_weather", "search_products"]

    def test_extract_tools_none_when_no_tools(self) -> None:
        conv = [
            ChatMLMessage(role="user", content="Hi"),
            ChatMLMessage(role="assistant", content="Hello!"),
        ]
        assert _extract_tools(conv) is None

    def test_extract_tools_deduplicates(self) -> None:
        conv = [
            ChatMLMessage(role="system", content="a", tools=SAMPLE_TOOLS),
            ChatMLMessage(role="system", content="b", tools=SAMPLE_TOOLS),
        ]
        assert len(_extract_tools(conv)) == 2  # not 4

    def test_serialize_conversation_includes_tools(
        self, tool_call_conversation: Conversation,
    ) -> None:
        result = _serialize_conversation(tool_call_conversation)
        assert result["tools"] is not None
        assert len(result["tools"]) == 2

    def test_serialize_conversation_no_tools(self) -> None:
        result = _serialize_conversation([
            ChatMLMessage(role="user", content="Hi"),
            ChatMLMessage(role="assistant", content="Hello!"),
        ])
        assert result["tools"] is None

    def test_parquet_round_trip(
        self, tmp_path: Path, tool_call_conversation: Conversation,
    ) -> None:
        """Write a tool-calling conversation to parquet, read it back."""
        serialized = _serialize_conversation(tool_call_conversation)
        path = _write_tool_parquet(
            tmp_path / "data.parquet",
            messages=serialized["messages"],
            tools=serialized["tools"],
        )

        conversations, tools_list = load_dataset_with_tools(path)
        assert len(conversations) == 1
        assert len(tools_list) == 1

        msgs = conversations[0]
        tools = tools_list[0]

        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert "tool_calls" in msgs[2]
        assert msgs[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "call_weather_0"
        assert msgs[4]["role"] == "assistant"

        assert tools is not None
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "get_weather"

    def test_parquet_round_trip_no_tools(self, tmp_path: Path) -> None:
        """Parquet files without a ``tools`` column should still load."""
        path = _write_tool_parquet(
            tmp_path / "data.parquet",
            messages=[{"role": "user", "content": "Hi"}],
            include_tools_column=False,
        )

        conversations, tools_list = load_dataset_with_tools(path)
        assert len(conversations) == 1
        assert tools_list[0] is None


# ---------------------------------------------------------------------------
# LFM2 tool format
# ---------------------------------------------------------------------------


class TestLFM2ToolFormat:
    """LFM2/LFM2.5 tool call materialization and parsing."""

    def test_materialize_simple(self, formatter: LFM2ToolFormatter) -> None:
        result = formatter.materialize_tool_calls([
            _tc("call_0", "get_weather", '{"location": "San Francisco"}'),
        ])
        assert result == '<|tool_call_start|>[get_weather(location="San Francisco")]<|tool_call_end|>'

    def test_materialize_multiple_calls(self, formatter: LFM2ToolFormatter) -> None:
        result = formatter.materialize_tool_calls([
            _tc("call_0", "get_weather", '{"location": "NYC"}'),
            _tc("call_1", "search_products", '{"query": "umbrella"}'),
        ])
        assert "<|tool_call_start|>" in result
        assert "<|tool_call_end|>" in result
        assert "get_weather" in result
        assert "search_products" in result

    @pytest.mark.parametrize(
        "arguments,expected_substring",
        [
            ('{"query": "laptop", "max_results": 5}', "max_results=5"),
            ('{"flag": true}', "flag=true"),
        ],
    )
    def test_materialize_typed_args(
        self,
        formatter: LFM2ToolFormatter,
        arguments: str,
        expected_substring: str,
    ) -> None:
        result = formatter.materialize_tool_calls([
            _tc("call_0", "test_func", arguments),
        ])
        assert expected_substring in result

    def test_parse_simple(self, formatter: LFM2ToolFormatter) -> None:
        text = '<|tool_call_start|>[get_weather(location="San Francisco")]<|tool_call_end|>'
        calls = formatter.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"])["location"] == "San Francisco"

    def test_parse_multiple(self, formatter: LFM2ToolFormatter) -> None:
        text = (
            '<|tool_call_start|>[get_weather(location="NYC"),'
            'search_products(query="umbrella")]<|tool_call_end|>'
        )
        calls = formatter.parse_tool_calls(text)
        assert [c["function"]["name"] for c in calls] == ["get_weather", "search_products"]

    def test_parse_with_surrounding_text(self, formatter: LFM2ToolFormatter) -> None:
        text = 'Let me check that. <|tool_call_start|>[get_weather(location="LA")]<|tool_call_end|>'
        calls = formatter.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"

    def test_parse_no_tool_calls(self, formatter: LFM2ToolFormatter) -> None:
        assert formatter.parse_tool_calls("I don't need to call any tools for this.") == []

    def test_parse_numeric_args(self, formatter: LFM2ToolFormatter) -> None:
        text = '<|tool_call_start|>[search_products(query="laptop", max_results=5)]<|tool_call_end|>'
        calls = formatter.parse_tool_calls(text)
        assert json.loads(calls[0]["function"]["arguments"])["max_results"] == 5

    def test_round_trip(self, formatter: LFM2ToolFormatter) -> None:
        """Materialize then parse should recover the same tool calls."""
        original = [_tc("call_0", "get_weather", '{"location": "Tokyo", "unit": "celsius"}')]
        parsed = formatter.parse_tool_calls(formatter.materialize_tool_calls(original))
        assert len(parsed) == 1
        assert parsed[0]["function"]["name"] == "get_weather"
        assert json.loads(parsed[0]["function"]["arguments"]) == json.loads(
            original[0]["function"]["arguments"]
        )

    def test_materialize_assistant_content(self, formatter: LFM2ToolFormatter) -> None:
        result = formatter.materialize_assistant_content(
            "Let me check.",
            [_tc("c0", "get_weather", '{"location": "SF"}')],
        )
        assert result.startswith("Let me check.")
        assert "<|tool_call_start|>" in result

    def test_parse_assistant_output(self, formatter: LFM2ToolFormatter) -> None:
        text = 'Sure! <|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>'
        content, calls = formatter.parse_assistant_output(text)
        assert content == "Sure!"
        assert len(calls) == 1

    def test_strip_markers(self, formatter: LFM2ToolFormatter) -> None:
        text = 'Hello <|tool_call_start|>[func()]<|tool_call_end|> world'
        stripped = formatter._strip_tool_call_markers(text)
        assert "<|tool_call_start|>" not in stripped
        assert "<|tool_call_end|>" not in stripped

    @pytest.mark.parametrize(
        "model_id",
        ["LiquidAI/LFM2-1.2B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct"],
    )
    def test_get_tool_formatter_lfm2(self, model_id: str) -> None:
        assert isinstance(get_tool_formatter(model_id), LFM2ToolFormatter)

    def test_get_tool_formatter_unknown(self) -> None:
        assert get_tool_formatter("meta-llama/Llama-3-8B") is None


# ---------------------------------------------------------------------------
# Scoring format
# ---------------------------------------------------------------------------


class TestScoringFormat:
    """Tool-call-aware scoring formatting and judge prompts."""

    def test_has_tool_calls_true(self) -> None:
        msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}]
        assert _has_tool_calls(msgs) is True

    def test_has_tool_calls_false(self) -> None:
        msgs = [{"role": "assistant", "content": "Hello"}]
        assert _has_tool_calls(msgs) is False

    def test_format_tool_calls(self) -> None:
        result = _format_tool_calls([
            {"function": {"name": "get_weather", "arguments": '{"location": "SF"}'}}
        ])
        assert "get_weather" in result
        assert '{"location": "SF"}' in result

    def test_format_conversation_with_tool_calls(self) -> None:
        msgs = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"location": "SF"}'}}
                ],
            },
            {"role": "tool", "name": "get_weather", "content": '{"temp": 65}'},
            {"role": "assistant", "content": "It's 65F."},
        ]
        result = _format_conversation(msgs)
        assert "[Tool Calls]" in result
        assert "get_weather" in result
        assert "[Tool Result: get_weather]" in result

    def test_format_conversation_with_tools_header(self) -> None:
        result = _format_conversation(
            [{"role": "user", "content": "Hi"}],
            tools=[
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "function", "function": {"name": "search_products"}},
            ],
        )
        assert "[Available Tools: get_weather, search_products]" in result

    def test_format_conversation_assistant_only_tool_calls(self) -> None:
        result = _format_conversation([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"q": "test"}'}}
                ],
            },
        ])
        assert "[Tool Calls]" in result
        assert "[Assistant]" in result

    def test_build_scoring_prompt_detects_tool_calls(self) -> None:
        prompt = _build_scoring_prompt(
            "Score well.",
            [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": '{"location": "SF"}'}}
                    ],
                },
            ],
        )
        # Tool-calling judge prompt
        assert "tool" in prompt[0]["content"].lower()
        assert "tool" in prompt[1]["content"].lower()

    def test_build_scoring_prompt_with_tools_kwarg(self) -> None:
        prompt = _build_scoring_prompt(
            "Score.",
            [{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert "tool" in prompt[0]["content"].lower()

    def test_build_scoring_prompt_no_tools(self) -> None:
        prompt = _build_scoring_prompt(
            "Score.",
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert "tool-calling" not in prompt[0]["content"].lower()


# ---------------------------------------------------------------------------
# Training data utils
# ---------------------------------------------------------------------------


class TestTrainingDataUtils:
    """Training data loading with tools."""

    def test_chatml_to_sft_with_tools(self) -> None:
        result = chatml_to_sft_dataset(
            [[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}]],
            tools_per_sample=[[{"type": "function", "function": {"name": "f1"}}]],
        )
        assert len(result) == 1
        assert "messages" in result[0]
        assert result[0]["tools"][0]["function"]["name"] == "f1"

    def test_chatml_to_sft_without_tools(self) -> None:
        result = chatml_to_sft_dataset([[{"role": "user", "content": "Hi"}]])
        assert len(result) == 1
        assert "tools" not in result[0]

    def test_chatml_to_sft_mixed_tools(self) -> None:
        result = chatml_to_sft_dataset(
            [
                [{"role": "user", "content": "Hi"}],
                [{"role": "user", "content": "Bye"}],
            ],
            tools_per_sample=[
                [{"type": "function", "function": {"name": "f1"}}],
                None,
            ],
        )
        assert "tools" in result[0]
        assert "tools" not in result[1]

    def test_load_chatml_dataset_with_tools(self, tmp_path: Path) -> None:
        """Write and read back a parquet with a ``tools`` column."""
        path = _write_tool_parquet(
            tmp_path / "data.parquet",
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        conversations, tools_list = load_chatml_dataset_with_tools(path)
        assert len(conversations) == 1
        assert tools_list[0] is not None
        assert tools_list[0][0]["function"]["name"] == "get_weather"

    def test_load_chatml_dataset_no_tools_column(self, tmp_path: Path) -> None:
        """Old parquet without a ``tools`` column should still work."""
        path = tmp_path / "data.parquet"
        schema = pa.schema([pa.field("messages", pa.string())])
        table = pa.table(
            {"messages": [json.dumps([{"role": "user", "content": "Hi"}])]},
            schema=schema,
        )
        pq.write_table(table, path)

        conversations, tools_list = load_chatml_dataset_with_tools(path)
        assert len(conversations) == 1
        assert tools_list[0] is None
