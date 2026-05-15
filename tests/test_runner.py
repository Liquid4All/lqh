"""Tests for the ModelRunner abstraction (lqh/runner.py).

Unit tests use a mock ``AsyncOpenAI`` client.  Integration tests hit
``api.lqh.ai`` and require authentication (``LQH_DEBUG_API_KEY`` env var
or ``~/.lqh/config.json``).  They opt in via ``@pytest.mark.integration``
and skip automatically when no credentials are present.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from lqh.runner import (
    APIModelRunner,
    ModelRunner,
    RunnerResponse,
    RunnerToolCall,
    RunnerUsage,
)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner_with_mock(
    make_chat_completion: Callable[..., SimpleNamespace],
) -> Callable[..., tuple[APIModelRunner, AsyncMock]]:
    """Build an APIModelRunner wired to a mocked completions endpoint.

    Returns ``(runner, create_mock)`` so tests can inspect the call args.
    """

    def _factory(**completion_kwargs: Any) -> tuple[APIModelRunner, AsyncMock]:
        client = MagicMock()
        response = make_chat_completion(**completion_kwargs)
        create_mock = AsyncMock(return_value=response)
        client.chat.completions.create = create_mock
        return APIModelRunner(client), create_mock

    return _factory


# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------


class TestRunnerTypes:
    """Verify dataclass constructors and protocol shape."""

    def test_runner_usage_fields(self) -> None:
        u = RunnerUsage(prompt_tokens=10, completion_tokens=20)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20

    def test_runner_response_defaults(self) -> None:
        r = RunnerResponse(content="hello", model="small")
        assert r.usage is None
        assert r.tool_calls == []

    def test_runner_tool_call_fields(self) -> None:
        tc = RunnerToolCall(
            id="call_1",
            function_name="get_weather",
            function_arguments='{"loc":"NYC"}',
        )
        assert tc.function_name == "get_weather"

    def test_api_model_runner_satisfies_protocol(self) -> None:
        assert issubclass(APIModelRunner, ModelRunner)


# ---------------------------------------------------------------------------
# APIModelRunner against a mocked AsyncOpenAI
# ---------------------------------------------------------------------------


class TestAPIModelRunnerUnit:
    """Unit tests for ``APIModelRunner`` with a mocked client."""

    async def test_basic_completion(self, runner_with_mock) -> None:
        runner, _ = runner_with_mock(
            content="response text",
            model="small",
            prompt_tokens=10,
            completion_tokens=5,
        )

        result = await runner.complete(
            [{"role": "user", "content": "hi"}], model="small",
        )

        assert result.content == "response text"
        assert result.model == "small"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.tool_calls == []

    async def test_passes_optional_params(self, runner_with_mock) -> None:
        """``temperature``, ``max_tokens``, ``response_format``, ``tools`` are forwarded."""
        runner, create_mock = runner_with_mock()

        response_format = {"type": "json_object"}
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

        await runner.complete(
            [{"role": "user", "content": "hi"}],
            model="medium",
            temperature=0.7,
            max_tokens=100,
            response_format=response_format,
            tools=tools,
            tool_choice="auto",
        )

        kwargs = create_mock.call_args.kwargs
        assert kwargs["model"] == "medium"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["response_format"] == response_format
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"

    async def test_omits_none_optional_params(self, runner_with_mock) -> None:
        """``None``-valued optional params are not sent to the API."""
        runner, create_mock = runner_with_mock()

        await runner.complete([{"role": "user", "content": "hi"}], model="small")

        kwargs = create_mock.call_args.kwargs
        for omitted in ("max_tokens", "response_format", "tools", "tool_choice"):
            assert omitted not in kwargs

    async def test_tool_calls_parsed(self, runner_with_mock) -> None:
        """Tool calls from the API response are correctly mapped."""
        tc = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
        )
        runner, _ = runner_with_mock(tool_calls=[tc])

        result = await runner.complete(
            [{"role": "user", "content": "weather?"}], model="orchestration",
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].function_name == "get_weather"
        assert result.tool_calls[0].function_arguments == '{"city":"Paris"}'

    async def test_null_content_returns_empty_string(self, runner_with_mock) -> None:
        runner, _ = runner_with_mock(content=None)

        result = await runner.complete(
            [{"role": "user", "content": "hi"}], model="small",
        )
        assert result.content == ""

    async def test_null_usage_returns_none(self, runner_with_mock) -> None:
        runner, create_mock = runner_with_mock()
        # The make_chat_completion fixture always builds a usage object;
        # clobber it on the canned response to exercise the None branch.
        create_mock.return_value.usage = None

        result = await runner.complete(
            [{"role": "user", "content": "hi"}], model="small",
        )
        assert result.usage is None


# ---------------------------------------------------------------------------
# Integration tests (require API access)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAPIModelRunnerIntegration:
    """Integration tests that actually hit ``api.lqh.ai``."""

    @pytest.fixture
    def runner(self, api_client: Any) -> APIModelRunner:
        return APIModelRunner(api_client)

    async def test_simple_completion_small(self, runner: APIModelRunner) -> None:
        """Basic completion with the ``small`` pool model."""
        result = await runner.complete(
            [{"role": "user", "content": "Reply with exactly: PONG"}],
            model="small",
            temperature=0.0,
            max_tokens=10,
        )
        assert "PONG" in result.content.upper()
        assert result.usage is not None
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0

    async def test_json_mode(self, runner: APIModelRunner) -> None:
        """Structured output with ``response_format``."""
        result = await runner.complete(
            [{"role": "user", "content": "Return a JSON object with key 'color' and value 'blue'."}],
            model="small",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        assert json.loads(result.content)["color"] == "blue"

    async def test_lfm_model_direct(
        self, runner: APIModelRunner, api_client: Any,
    ) -> None:
        """Call a specific LFM model by ID."""
        models = await api_client.models.list()
        if not models.data:
            pytest.skip("No LFM models available on the API")

        result = await runner.complete(
            [{"role": "user", "content": "Say hello."}],
            model=models.data[0].id,
            temperature=0.0,
            max_tokens=20,
        )
        assert len(result.content) > 0
        assert result.usage is not None

    async def test_list_models_api(self, api_client: Any) -> None:
        """Verify the ``/v1/models`` endpoint returns model data."""
        models = await api_client.models.list()
        assert len(models.data) > 0
        assert hasattr(models.data[0], "id")
