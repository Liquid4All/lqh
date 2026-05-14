"""Unit tests for tool-calling support.

Tests serialization round-trips, LFM2 format conversion, and
tool-call-aware scoring/judge prompt generation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
from lqh.train.tool_format import (
    LFM2ToolFormatter,
    get_tool_formatter,
)
from lqh.train.data_utils import (
    chatml_to_sft_dataset,
    load_chatml_dataset_with_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    ToolDef(
        name="get_weather",
        description="Get current weather for a location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    ),
    ToolDef(
        name="search_products",
        description="Search for products.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    ),
]


def _make_tool_call_conversation() -> Conversation:
    """Build a sample conversation with tool calls."""
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


# ---------------------------------------------------------------------------
# Engine serialization tests
# ---------------------------------------------------------------------------


class TestToolSerialization:
    """Test tool call serialization and parquet round-trip."""

    def test_serialize_message_with_tool_calls(self):
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

    def test_serialize_message_with_tool_result(self):
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

    def test_serialize_message_with_tools(self):
        msg = ChatMLMessage(
            role="system",
            content="Hello",
            tools=SAMPLE_TOOLS,
        )
        d = _serialize_message(msg)
        assert len(d["tools"]) == 2
        assert d["tools"][0]["type"] == "function"
        assert d["tools"][0]["function"]["name"] == "get_weather"

    def test_extract_tools(self):
        conv = _make_tool_call_conversation()
        tools = _extract_tools(conv)
        assert tools is not None
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[1]["function"]["name"] == "search_products"

    def test_extract_tools_none_when_no_tools(self):
        conv = [
            ChatMLMessage(role="user", content="Hi"),
            ChatMLMessage(role="assistant", content="Hello!"),
        ]
        assert _extract_tools(conv) is None

    def test_extract_tools_deduplicates(self):
        conv = [
            ChatMLMessage(role="system", content="a", tools=SAMPLE_TOOLS),
            ChatMLMessage(role="system", content="b", tools=SAMPLE_TOOLS),
        ]
        tools = _extract_tools(conv)
        assert len(tools) == 2  # not 4

    def test_serialize_conversation_includes_tools(self):
        conv = _make_tool_call_conversation()
        result = _serialize_conversation(conv)
        assert "tools" in result
        assert result["tools"] is not None
        assert len(result["tools"]) == 2

    def test_serialize_conversation_no_tools(self):
        conv = [
            ChatMLMessage(role="user", content="Hi"),
            ChatMLMessage(role="assistant", content="Hello!"),
        ]
        result = _serialize_conversation(conv)
        assert result["tools"] is None

    def test_parquet_round_trip(self, tmp_path: Path):
        """Write a tool-calling conversation to parquet, read it back."""
        import pyarrow as pa

        conv = _make_tool_call_conversation()
        serialized = _serialize_conversation(conv)

        schema = pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ])
        table = pa.table(
            {
                "messages": [json.dumps(serialized["messages"])],
                "audio": [None],
                "tools": [json.dumps(serialized["tools"])],
            },
            schema=schema,
        )
        path = tmp_path / "data.parquet"
        pq.write_table(table, path)

        # Read back
        conversations, tools_list = load_dataset_with_tools(path)
        assert len(conversations) == 1
        assert len(tools_list) == 1

        msgs = conversations[0]
        tools = tools_list[0]

        # Check messages
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert "tool_calls" in msgs[2]
        assert msgs[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "call_weather_0"
        assert msgs[4]["role"] == "assistant"

        # Check tools
        assert tools is not None
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "get_weather"

    def test_parquet_round_trip_no_tools(self, tmp_path: Path):
        """Parquet files without a tools column should still load."""
        import pyarrow as pa

        schema = pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
        ])
        table = pa.table(
            {
                "messages": [json.dumps([{"role": "user", "content": "Hi"}])],
                "audio": [None],
            },
            schema=schema,
        )
        path = tmp_path / "data.parquet"
        pq.write_table(table, path)

        conversations, tools_list = load_dataset_with_tools(path)
        assert len(conversations) == 1
        assert tools_list[0] is None


# ---------------------------------------------------------------------------
# LFM2 tool format tests
# ---------------------------------------------------------------------------


class TestLFM2ToolFormat:
    """Test LFM2/LFM2.5 tool call materialization and parsing."""

    def setup_method(self):
        self.formatter = LFM2ToolFormatter()

    def test_materialize_simple(self):
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
            }
        ]
        result = self.formatter.materialize_tool_calls(tool_calls)
        assert result == '<|tool_call_start|>[get_weather(location="San Francisco")]<|tool_call_end|>'

    def test_materialize_multiple_calls(self):
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
            },
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_products",
                    "arguments": '{"query": "umbrella"}',
                },
            },
        ]
        result = self.formatter.materialize_tool_calls(tool_calls)
        assert "<|tool_call_start|>" in result
        assert "<|tool_call_end|>" in result
        assert "get_weather" in result
        assert "search_products" in result

    def test_materialize_numeric_args(self):
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "search_products",
                    "arguments": '{"query": "laptop", "max_results": 5}',
                },
            }
        ]
        result = self.formatter.materialize_tool_calls(tool_calls)
        assert "max_results=5" in result

    def test_materialize_boolean_args(self):
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": '{"flag": true}',
                },
            }
        ]
        result = self.formatter.materialize_tool_calls(tool_calls)
        assert "flag=true" in result

    def test_parse_simple(self):
        text = '<|tool_call_start|>[get_weather(location="San Francisco")]<|tool_call_end|>'
        calls = self.formatter.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["location"] == "San Francisco"

    def test_parse_multiple(self):
        text = '<|tool_call_start|>[get_weather(location="NYC"),search_products(query="umbrella")]<|tool_call_end|>'
        calls = self.formatter.parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[1]["function"]["name"] == "search_products"

    def test_parse_with_surrounding_text(self):
        text = 'Let me check that. <|tool_call_start|>[get_weather(location="LA")]<|tool_call_end|>'
        calls = self.formatter.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"

    def test_parse_no_tool_calls(self):
        text = "I don't need to call any tools for this."
        calls = self.formatter.parse_tool_calls(text)
        assert calls == []

    def test_parse_numeric_args(self):
        text = '<|tool_call_start|>[search_products(query="laptop", max_results=5)]<|tool_call_end|>'
        calls = self.formatter.parse_tool_calls(text)
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["max_results"] == 5

    def test_round_trip(self):
        """Materialize then parse should recover the same tool calls."""
        original = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Tokyo", "unit": "celsius"}',
                },
            }
        ]
        text = self.formatter.materialize_tool_calls(original)
        parsed = self.formatter.parse_tool_calls(text)
        assert len(parsed) == 1
        assert parsed[0]["function"]["name"] == "get_weather"
        orig_args = json.loads(original[0]["function"]["arguments"])
        parsed_args = json.loads(parsed[0]["function"]["arguments"])
        assert orig_args == parsed_args

    def test_materialize_assistant_content(self):
        result = self.formatter.materialize_assistant_content(
            "Let me check.",
            [{"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "SF"}'}}],
        )
        assert result.startswith("Let me check.")
        assert "<|tool_call_start|>" in result

    def test_parse_assistant_output(self):
        text = 'Sure! <|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>'
        content, calls = self.formatter.parse_assistant_output(text)
        assert content == "Sure!"
        assert len(calls) == 1

    def test_strip_markers(self):
        text = 'Hello <|tool_call_start|>[func()]<|tool_call_end|> world'
        stripped = self.formatter._strip_tool_call_markers(text)
        assert "<|tool_call_start|>" not in stripped
        assert "<|tool_call_end|>" not in stripped

    def test_get_tool_formatter_lfm2(self):
        assert isinstance(get_tool_formatter("LiquidAI/LFM2-1.2B-Instruct"), LFM2ToolFormatter)
        assert isinstance(get_tool_formatter("LiquidAI/LFM2.5-1.2B-Instruct"), LFM2ToolFormatter)

    def test_get_tool_formatter_unknown(self):
        assert get_tool_formatter("meta-llama/Llama-3-8B") is None


# ---------------------------------------------------------------------------
# Scoring format tests
# ---------------------------------------------------------------------------


class TestScoringFormat:
    """Test tool-call-aware scoring formatting and judge prompts."""

    def test_has_tool_calls_true(self):
        msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}]
        assert _has_tool_calls(msgs) is True

    def test_has_tool_calls_false(self):
        msgs = [{"role": "assistant", "content": "Hello"}]
        assert _has_tool_calls(msgs) is False

    def test_format_tool_calls(self):
        tc = [{"function": {"name": "get_weather", "arguments": '{"location": "SF"}'}}]
        result = _format_tool_calls(tc)
        assert "get_weather" in result
        assert '{"location": "SF"}' in result

    def test_format_conversation_with_tool_calls(self):
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

    def test_format_conversation_with_tools_header(self):
        tools = [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "search_products"}},
        ]
        msgs = [{"role": "user", "content": "Hi"}]
        result = _format_conversation(msgs, tools=tools)
        assert "[Available Tools: get_weather, search_products]" in result

    def test_format_conversation_assistant_only_tool_calls(self):
        """Assistant message with tool_calls but no text content."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"q": "test"}'}}
                ],
            },
        ]
        result = _format_conversation(msgs)
        assert "[Tool Calls]" in result
        assert "[Assistant]" in result

    def test_build_scoring_prompt_detects_tool_calls(self):
        msgs = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"location": "SF"}'}}
                ],
            },
        ]
        prompt = _build_scoring_prompt("Score well.", msgs)
        # Should use tool-calling judge system prompt
        assert "tool-calling" in prompt[0]["content"].lower() or "tool" in prompt[0]["content"].lower()
        assert "tool" in prompt[1]["content"].lower()

    def test_build_scoring_prompt_with_tools_kwarg(self):
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        prompt = _build_scoring_prompt("Score.", msgs, tools=tools)
        assert "tool" in prompt[0]["content"].lower()

    def test_build_scoring_prompt_no_tools(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        prompt = _build_scoring_prompt("Score.", msgs)
        # Should use default judge (not tool-specific)
        assert "tool-calling" not in prompt[0]["content"].lower()


# ---------------------------------------------------------------------------
# Training data_utils tests
# ---------------------------------------------------------------------------


class TestTrainingDataUtils:
    """Test training data loading with tools."""

    def test_chatml_to_sft_with_tools(self):
        convs = [
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}],
        ]
        tools = [[{"type": "function", "function": {"name": "f1"}}]]
        result = chatml_to_sft_dataset(convs, tools_per_sample=tools)
        assert len(result) == 1
        assert "messages" in result[0]
        assert "tools" in result[0]
        assert result[0]["tools"][0]["function"]["name"] == "f1"

    def test_chatml_to_sft_without_tools(self):
        convs = [
            [{"role": "user", "content": "Hi"}],
        ]
        result = chatml_to_sft_dataset(convs)
        assert len(result) == 1
        assert "tools" not in result[0]

    def test_chatml_to_sft_mixed_tools(self):
        convs = [
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "Bye"}],
        ]
        tools = [
            [{"type": "function", "function": {"name": "f1"}}],
            None,
        ]
        result = chatml_to_sft_dataset(convs, tools_per_sample=tools)
        assert "tools" in result[0]
        assert "tools" not in result[1]

    def test_load_chatml_dataset_with_tools(self, tmp_path: Path):
        """Write and read back a parquet with tools column."""
        import pyarrow as pa

        msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        schema = pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ])
        table = pa.table(
            {
                "messages": [json.dumps(msgs)],
                "audio": [None],
                "tools": [json.dumps(tools)],
            },
            schema=schema,
        )
        path = tmp_path / "data.parquet"
        pq.write_table(table, path)

        conversations, tools_list = load_chatml_dataset_with_tools(path)
        assert len(conversations) == 1
        assert tools_list[0] is not None
        assert tools_list[0][0]["function"]["name"] == "get_weather"

    def test_load_chatml_dataset_no_tools_column(self, tmp_path: Path):
        """Old parquet without tools column should still work."""
        import pyarrow as pa

        schema = pa.schema([
            pa.field("messages", pa.string()),
        ])
        table = pa.table(
            {"messages": [json.dumps([{"role": "user", "content": "Hi"}])]},
            schema=schema,
        )
        path = tmp_path / "data.parquet"
        pq.write_table(table, path)

        conversations, tools_list = load_chatml_dataset_with_tools(path)
        assert len(conversations) == 1
        assert tools_list[0] is None
