"""Model runner abstraction for inference.

Provides a thin Protocol that all model inference (for evals, scoring,
prompt optimization) goes through.  The protocol is intentionally minimal
— just enough to run completions and inspect the result.

Implementations:
  - ``APIModelRunner`` — wraps ``AsyncOpenAI`` pointed at api.lqh.ai (or any
    OpenAI-compatible endpoint).
  - Future: ``HFModelRunner`` (local transformers), ``QuantModelRunner``
    (llama.cpp server).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI

__all__ = [
    "ModelRunner",
    "APIModelRunner",
    "RunnerResponse",
    "RunnerUsage",
    "RunnerToolCall",
]


@dataclass
class RunnerUsage:
    """Token usage from a single completion."""

    prompt_tokens: int
    completion_tokens: int


@dataclass
class RunnerToolCall:
    """A tool call returned by the model."""

    id: str
    function_name: str
    function_arguments: str  # JSON string


@dataclass
class RunnerResponse:
    """Result of a single model completion."""

    content: str
    model: str
    usage: RunnerUsage | None = None
    tool_calls: list[RunnerToolCall] = field(default_factory=list)


@runtime_checkable
class ModelRunner(Protocol):
    """Protocol for running model inference.

    ``model`` is a parameter on ``complete()``, not on the constructor.
    This lets a single runner serve multiple model names (the API routes
    them server-side; a local runner can load different checkpoints).
    """

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> RunnerResponse: ...


class APIModelRunner:
    """ModelRunner backed by an OpenAI-compatible API."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> RunnerResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        usage = None
        if response.usage:
            usage = RunnerUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        tc: list[RunnerToolCall] = []
        if choice.message.tool_calls:
            for call in choice.message.tool_calls:
                tc.append(
                    RunnerToolCall(
                        id=call.id,
                        function_name=call.function.name,
                        function_arguments=call.function.arguments,
                    )
                )

        return RunnerResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            tool_calls=tc,
        )
