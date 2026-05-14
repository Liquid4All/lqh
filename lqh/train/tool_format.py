"""Model-specific tool-calling format conversion.

Converts between the canonical OpenAI JSON tool-call format (used in
parquet storage) and the text-based formats used by specific models for
local training and inference via HuggingFace transformers.

The abstract ``ToolFormatter`` base class allows adding new model formats
by subclassing.  Currently only LFM2/LFM2.5 is implemented.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "ToolFormatter",
    "LFM2ToolFormatter",
    "get_tool_formatter",
]


class ToolFormatter(ABC):
    """Abstract base for model-specific tool-call format conversion."""

    @abstractmethod
    def materialize_tool_calls(self, tool_calls: list[dict[str, Any]]) -> str:
        """Convert OpenAI-format tool calls to the model's text representation.

        Parameters
        ----------
        tool_calls:
            List of dicts with ``id``, ``type``, ``function`` (name, arguments).

        Returns
        -------
        str
            The text representation that the model would generate.
        """
        ...

    @abstractmethod
    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse the model's text output into OpenAI-format tool calls.

        Parameters
        ----------
        text:
            Raw decoded text from model generation.

        Returns
        -------
        list
            List of tool call dicts in OpenAI format, or empty list if
            no tool calls were found.
        """
        ...

    def materialize_assistant_content(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None,
    ) -> str:
        """Build the full assistant text including any tool calls.

        Combines optional text content with tool call text in the
        model's expected format.
        """
        parts: list[str] = []
        if content:
            parts.append(content)
        if tool_calls:
            parts.append(self.materialize_tool_calls(tool_calls))
        return "".join(parts)

    def parse_assistant_output(
        self, text: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Split model output into text content and parsed tool calls.

        Returns ``(content, tool_calls)`` where *content* is the text
        outside tool-call markers and *tool_calls* is a (possibly empty)
        list in OpenAI format.
        """
        tool_calls = self.parse_tool_calls(text)
        # Remove tool call markers from content
        content = self._strip_tool_call_markers(text)
        return content, tool_calls

    def _strip_tool_call_markers(self, text: str) -> str:
        """Remove tool call marker tokens/text, returning plain content."""
        return text


# ---------------------------------------------------------------------------
# LFM2 / LFM2.5 format
# ---------------------------------------------------------------------------

# Matches: <|tool_call_start|>[...]<|tool_call_end|>
_LFM2_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>",
    re.DOTALL,
)

# Matches a single function call: func_name(arg1="val1", arg2="val2")
_LFM2_FUNC_RE = re.compile(
    r"(\w+)\(([^)]*)\)",
)


def _parse_lfm2_args(args_str: str) -> dict[str, Any]:
    """Parse LFM2 pythonic keyword arguments into a dict.

    Handles: key="string", key=123, key=true, key=null, key='string'
    """
    result: dict[str, Any] = {}
    if not args_str.strip():
        return result

    # Split on commas that are not inside quotes
    # Simple approach: iterate character by character
    parts: list[str] = []
    current: list[str] = []
    in_quote: str | None = None
    for ch in args_str:
        if ch in ('"', "'") and in_quote is None:
            in_quote = ch
            current.append(ch)
        elif ch == in_quote:
            in_quote = None
            current.append(ch)
        elif ch == "," and in_quote is None:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())

    for part in parts:
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        key = key.strip()
        val = val.strip()

        # Parse the value
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            result[key] = val[1:-1]
        elif val.lower() == "true":
            result[key] = True
        elif val.lower() == "false":
            result[key] = False
        elif val.lower() == "null" or val.lower() == "none":
            result[key] = None
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val

    return result


def _format_lfm2_args(arguments: dict[str, Any]) -> str:
    """Format arguments as LFM2 pythonic keyword arguments."""
    parts: list[str] = []
    for key, val in arguments.items():
        if isinstance(val, str):
            # Escape quotes in the value
            escaped = val.replace('"', '\\"')
            parts.append(f'{key}="{escaped}"')
        elif isinstance(val, bool):
            parts.append(f"{key}={str(val).lower()}")
        elif val is None:
            parts.append(f"{key}=null")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


class LFM2ToolFormatter(ToolFormatter):
    """Tool format for LFM2 and LFM2.5 models.

    Format::

        <|tool_call_start|>[func(arg="val"), other(x=1)]<|tool_call_end|>

    The content inside the markers is a Python-like function call list.
    Text content may appear before or after the tool call block.
    """

    def materialize_tool_calls(self, tool_calls: list[dict[str, Any]]) -> str:
        calls: list[str] = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args_raw = func.get("arguments", "{}")
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            calls.append(f"{name}({_format_lfm2_args(args)})")
        return f"<|tool_call_start|>[{','.join(calls)}]<|tool_call_end|>"

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        match = _LFM2_TOOL_CALL_RE.search(text)
        if not match:
            return []

        inner = match.group(1)
        results: list[dict[str, Any]] = []
        for func_match in _LFM2_FUNC_RE.finditer(inner):
            name = func_match.group(1)
            args_str = func_match.group(2)
            args = _parse_lfm2_args(args_str)
            results.append({
                "id": f"call_{name}_{len(results)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            })
        return results

    def _strip_tool_call_markers(self, text: str) -> str:
        return _LFM2_TOOL_CALL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FORMATTERS: dict[str, type[ToolFormatter]] = {
    "lfm2": LFM2ToolFormatter,
    "lfm2.5": LFM2ToolFormatter,
}


def get_tool_formatter(model_name: str) -> ToolFormatter | None:
    """Return a ToolFormatter for the given model, or None if unknown.

    Matches by checking if any known key is a substring of the model name
    (case-insensitive).
    """
    model_lower = model_name.lower()
    for key, cls in _FORMATTERS.items():
        if key in model_lower:
            return cls()
    return None
