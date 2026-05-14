"""Tests for the ModelRunner abstraction (lqh/runner.py).

Unit tests use a mock AsyncOpenAI client.
Integration tests hit api.lqh.ai and require authentication
(LQH_DEBUG_API_KEY env var or ~/.lqh/config.json).
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from lqh.runner import (
    APIModelRunner,
    ModelRunner,
    RunnerResponse,
    RunnerToolCall,
    RunnerUsage,
)


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------


class TestRunnerTypes(unittest.TestCase):
    """Verify dataclass constructors and protocol shape."""

    def test_runner_usage_fields(self) -> None:
        u = RunnerUsage(prompt_tokens=10, completion_tokens=20)
        self.assertEqual(u.prompt_tokens, 10)
        self.assertEqual(u.completion_tokens, 20)

    def test_runner_response_defaults(self) -> None:
        r = RunnerResponse(content="hello", model="small")
        self.assertIsNone(r.usage)
        self.assertEqual(r.tool_calls, [])

    def test_runner_tool_call_fields(self) -> None:
        tc = RunnerToolCall(id="call_1", function_name="get_weather", function_arguments='{"loc":"NYC"}')
        self.assertEqual(tc.function_name, "get_weather")

    def test_api_model_runner_satisfies_protocol(self) -> None:
        self.assertTrue(issubclass(APIModelRunner, ModelRunner))


class TestAPIModelRunnerUnit(unittest.TestCase):
    """Unit tests for APIModelRunner with a mocked AsyncOpenAI client."""

    def setUp(self) -> None:
        self.mock_client = MagicMock()

    def _make_mock_response(
        self,
        content: str = "response text",
        model: str = "small",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        tool_calls: list | None = None,
    ) -> SimpleNamespace:
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        return SimpleNamespace(choices=[choice], model=model, usage=usage)

    def test_basic_completion(self) -> None:
        mock_resp = self._make_mock_response()
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        runner = APIModelRunner(self.mock_client)

        result = asyncio.run(
            runner.complete(
                [{"role": "user", "content": "hi"}],
                model="small",
            )
        )

        self.assertEqual(result.content, "response text")
        self.assertEqual(result.model, "small")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.prompt_tokens, 10)
        self.assertEqual(result.usage.completion_tokens, 5)
        self.assertEqual(result.tool_calls, [])

    def test_passes_optional_params(self) -> None:
        """Verify temperature, max_tokens, response_format, tools are forwarded."""
        mock_resp = self._make_mock_response()
        mock_create = AsyncMock(return_value=mock_resp)
        self.mock_client.chat.completions.create = mock_create
        runner = APIModelRunner(self.mock_client)

        response_format = {"type": "json_object"}
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

        asyncio.run(
            runner.complete(
                [{"role": "user", "content": "hi"}],
                model="medium",
                temperature=0.7,
                max_tokens=100,
                response_format=response_format,
                tools=tools,
                tool_choice="auto",
            )
        )

        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["model"], "medium")
        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertEqual(call_kwargs["max_tokens"], 100)
        self.assertEqual(call_kwargs["response_format"], response_format)
        self.assertEqual(call_kwargs["tools"], tools)
        self.assertEqual(call_kwargs["tool_choice"], "auto")

    def test_omits_none_optional_params(self) -> None:
        """None optional params should not be sent to the API."""
        mock_resp = self._make_mock_response()
        mock_create = AsyncMock(return_value=mock_resp)
        self.mock_client.chat.completions.create = mock_create
        runner = APIModelRunner(self.mock_client)

        asyncio.run(
            runner.complete(
                [{"role": "user", "content": "hi"}],
                model="small",
            )
        )

        call_kwargs = mock_create.call_args[1]
        self.assertNotIn("max_tokens", call_kwargs)
        self.assertNotIn("response_format", call_kwargs)
        self.assertNotIn("tools", call_kwargs)
        self.assertNotIn("tool_choice", call_kwargs)

    def test_tool_calls_parsed(self) -> None:
        """Tool calls from the API response are correctly mapped."""
        tc = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
        )
        mock_resp = self._make_mock_response(tool_calls=[tc])
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        runner = APIModelRunner(self.mock_client)

        result = asyncio.run(
            runner.complete(
                [{"role": "user", "content": "weather?"}],
                model="orchestration",
            )
        )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].id, "call_abc")
        self.assertEqual(result.tool_calls[0].function_name, "get_weather")
        self.assertEqual(result.tool_calls[0].function_arguments, '{"city":"Paris"}')

    def test_null_content_returns_empty_string(self) -> None:
        mock_resp = self._make_mock_response(content=None)
        # Manually set content to None since SimpleNamespace doesn't enforce types
        mock_resp.choices[0].message.content = None
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        runner = APIModelRunner(self.mock_client)

        result = asyncio.run(
            runner.complete([{"role": "user", "content": "hi"}], model="small")
        )
        self.assertEqual(result.content, "")

    def test_null_usage_returns_none(self) -> None:
        mock_resp = self._make_mock_response()
        mock_resp.usage = None
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        runner = APIModelRunner(self.mock_client)

        result = asyncio.run(
            runner.complete([{"role": "user", "content": "hi"}], model="small")
        )
        self.assertIsNone(result.usage)


# ---------------------------------------------------------------------------
# Integration tests (require API access)
# ---------------------------------------------------------------------------


def _has_api_access() -> bool:
    """Check if we can authenticate with the API."""
    try:
        from lqh.auth import get_token
        return get_token() is not None
    except Exception:
        return False


@unittest.skipUnless(_has_api_access(), "No API access (set LQH_DEBUG_API_KEY or run /login)")
class TestAPIModelRunnerIntegration(unittest.TestCase):
    """Integration tests that hit api.lqh.ai."""

    def setUp(self) -> None:
        from lqh.auth import require_token
        from lqh.client import create_client
        from lqh.config import load_config

        config = load_config()
        token = require_token()
        self.client = create_client(token, config.api_base_url)
        self.runner = APIModelRunner(self.client)

    def test_simple_completion_small(self) -> None:
        """Basic completion with the 'small' pool model."""
        result = asyncio.run(
            self.runner.complete(
                [{"role": "user", "content": "Reply with exactly: PONG"}],
                model="small",
                temperature=0.0,
                max_tokens=10,
            )
        )
        self.assertIn("PONG", result.content.upper())
        self.assertIsNotNone(result.usage)
        self.assertGreater(result.usage.prompt_tokens, 0)
        self.assertGreater(result.usage.completion_tokens, 0)

    def test_json_mode(self) -> None:
        """Structured output with response_format."""
        import json

        result = asyncio.run(
            self.runner.complete(
                [{"role": "user", "content": "Return a JSON object with key 'color' and value 'blue'."}],
                model="small",
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
        data = json.loads(result.content)
        self.assertEqual(data["color"], "blue")

    def test_lfm_model_direct(self) -> None:
        """Call a specific LFM model by name using the model ID."""
        models = asyncio.run(self.client.models.list())
        if not models.data:
            self.skipTest("No LFM models available on the API")

        model_id = models.data[0].id
        result = asyncio.run(
            self.runner.complete(
                [{"role": "user", "content": "Say hello."}],
                model=model_id,
                temperature=0.0,
                max_tokens=20,
            )
        )
        self.assertTrue(len(result.content) > 0)
        self.assertIsNotNone(result.usage)

    def test_list_models_api(self) -> None:
        """Verify the /v1/models endpoint returns model data."""
        models = asyncio.run(self.client.models.list())
        self.assertGreater(len(models.data), 0)
        first = models.data[0]
        self.assertTrue(hasattr(first, "id"))


if __name__ == "__main__":
    unittest.main()
