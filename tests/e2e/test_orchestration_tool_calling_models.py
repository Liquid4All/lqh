"""Smoke-test tool calling for orchestration model variants.

This file can be used two ways:

    pytest tests/e2e/test_orchestration_tool_calling_models.py -q
    python -m tests.e2e.test_orchestration_tool_calling_models

The pytest cases are marked ``integration`` and are skipped automatically when
no API token is configured. The direct runner prints a table that is useful when
diagnosing backend regressions across orchestration model versions.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from lqh.auth import get_token
from lqh.client import chat_with_retry, create_client
from lqh.config import load_config


TOOL_NAME = "list_user_data"
ORCHESTRATION_MODELS = ("orchestration",) + tuple(
    f"orchestration:{i}" for i in range(1, 15)
)
TOOL_CALL_MESSAGES: list[dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "You are an API tool-calling smoke test. When the user asks you to "
            "inspect project data, call list_user_data exactly once. Do not "
            "answer in prose instead of using the tool."
        ),
    },
    {
        "role": "user",
        "content": (
            "I have provided examples in example_inputs.jsonl and I do not have "
            "outputs yet. Inspect the project data now."
        ),
    },
]

LIST_USER_DATA_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Report user-brought data in the project directory. Returns filenames, "
            "row counts, and detected schemas."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


@dataclass
class ToolCallingResult:
    model: str
    ok: bool
    status: str
    finish_reason: str | None = None
    tool_call_count: int = 0
    tool_names: tuple[str, ...] = ()
    content_preview: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    def detail(self) -> str:
        if self.error:
            return self.error
        if self.tool_names:
            return ",".join(self.tool_names)
        if self.content_preview:
            return self.content_preview.replace("\n", " ")[:120]
        return ""


def _make_client() -> Any:
    token = get_token()
    if not token:
        raise RuntimeError("No API access. Set LQH_DEBUG_API_KEY or run /login.")

    config = load_config()
    return create_client(token, config.api_base_url)


async def check_model_tool_calling(model: str) -> ToolCallingResult:
    """Return whether *model* emits a real OpenAI-style tool call."""
    client = _make_client()
    try:
        response = await chat_with_retry(
            client,
            max_retries=1,
            model=model,
            messages=TOOL_CALL_MESSAGES,
            tools=[LIST_USER_DATA_TOOL],
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
        )
    except Exception as exc:
        return ToolCallingResult(
            model=model,
            ok=False,
            status="api_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None
    tool_calls = list(message.tool_calls or []) if message else []
    tool_names = tuple(
        tc.function.name
        for tc in tool_calls
        if getattr(tc, "function", None) is not None
    )
    content = (message.content or "") if message else ""
    finish_reason = getattr(choice, "finish_reason", None) if choice else None

    if TOOL_NAME in tool_names:
        status = "ok"
        ok = True
    elif finish_reason == "tool_calls" and not tool_calls:
        status = "empty_tool_calls"
        ok = False
    elif tool_calls:
        status = "wrong_tool"
        ok = False
    else:
        status = "no_tool_call"
        ok = False

    return ToolCallingResult(
        model=model,
        ok=ok,
        status=status,
        finish_reason=finish_reason,
        tool_call_count=len(tool_calls),
        tool_names=tool_names,
        content_preview=content[:200],
        prompt_tokens=(
            getattr(response.usage, "prompt_tokens", None) if response.usage else None
        ),
        completion_tokens=(
            getattr(response.usage, "completion_tokens", None) if response.usage else None
        ),
    )


def _pytest_params() -> list[Any]:
    params = []
    for model in ORCHESTRATION_MODELS:
        params.append(pytest.param(model, marks=[pytest.mark.integration], id=model))
    return params


@pytest.mark.parametrize("model", _pytest_params())
async def test_orchestration_model_emits_list_user_data_tool_call(model: str) -> None:
    result = await check_model_tool_calling(model)

    assert result.ok, (
        f"{result.model} did not emit a usable {TOOL_NAME} call: "
        f"status={result.status}, finish_reason={result.finish_reason}, "
        f"tool_call_count={result.tool_call_count}, detail={result.detail()}"
    )


def _parse_models(raw: str) -> tuple[str, ...]:
    if not raw:
        return ORCHESTRATION_MODELS
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _print_results(results: list[ToolCallingResult]) -> None:
    print(
        "model              ok   status            finish        "
        "calls  tokens       detail"
    )
    print(
        "-----------------  ---  ----------------  ------------  "
        "-----  -----------  ------"
    )
    for result in results:
        tokens = (
            f"{result.prompt_tokens or 0}/{result.completion_tokens or 0}"
            if result.prompt_tokens is not None or result.completion_tokens is not None
            else "-"
        )
        print(
            f"{result.model:<17}  "
            f"{'yes' if result.ok else 'no':<3}  "
            f"{result.status:<16}  "
            f"{str(result.finish_reason or '-'):<12}  "
            f"{result.tool_call_count:>5}  "
            f"{tokens:<11}  "
            f"{result.detail()}"
        )


async def _run_cli(models: tuple[str, ...]) -> list[ToolCallingResult]:
    results: list[ToolCallingResult] = []
    for model in models:
        results.append(await check_model_tool_calling(model))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check tool-call output shape for orchestration model variants.",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated models to test. Defaults to orchestration and "
            "orchestration:1..14."
        ),
    )
    parser.add_argument(
        "--fail-on-broken",
        action="store_true",
        help="Exit nonzero if any tested model fails to emit list_user_data.",
    )
    args = parser.parse_args()

    try:
        results = asyncio.run(_run_cli(_parse_models(args.models)))
    except RuntimeError as exc:
        print(str(exc))
        return 2

    _print_results(results)
    if args.fail_on_broken and any(not result.ok for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
