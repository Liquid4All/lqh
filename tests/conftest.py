"""Shared fixtures and markers for the lqh test suite.

This file is auto-discovered by pytest at collection time. Test modules
should NOT import from it — fixtures are injected via argument names and
markers are registered in ``pyproject.toml``.

Fixtures provided here cover the patterns that recur across the unit and
integration tests:

* environment gates (``has_api_access``, ``has_cuda``, ``has_torch``)
* an authenticated API client (``api_client``) that auto-skips when no
  credentials are present
* factories for building ChatML / parquet datasets used by training,
  scoring, and pipeline tests
* a configurable async OpenAI client double (``mock_openai_client``)
* a helper for assembling OpenAI ``ChatCompletion`` response shapes

Tests should compose these fixtures rather than re-rolling local
helpers — see ``tests/test_scoring.py`` and ``tests/test_training.py``
for canonical usage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


def _check_api_access() -> bool:
    try:
        from lqh.auth import get_token

        return get_token() is not None
    except Exception:
        return False


def _check_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_api_access() -> bool:
    """True when ``LQH_DEBUG_API_KEY`` or ``~/.lqh/config.json`` is set."""
    return _check_api_access()


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    """True when a CUDA-capable GPU is visible to torch."""
    return _check_cuda()


@pytest.fixture(scope="session")
def has_torch() -> bool:
    """True when ``torch`` is importable (``pip install lqh[train]``)."""
    return _check_torch()


@pytest.fixture
def api_client(has_api_access: bool) -> Any:
    """Authenticated ``AsyncOpenAI`` client.  Skips the test on missing auth."""
    if not has_api_access:
        pytest.skip("No API access (set LQH_DEBUG_API_KEY or run /login)")

    from lqh.auth import require_token
    from lqh.client import create_client
    from lqh.config import load_config

    config = load_config()
    return create_client(require_token(), config.api_base_url)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A throw-away project directory with the ``.lqh/`` skeleton."""
    (tmp_path / ".lqh").mkdir()
    return tmp_path


@pytest.fixture
def chdir_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Change cwd to ``tmp_path`` for the duration of the test.

    Several lqh subsystems (``lqh.sources``, ``lqh.engine``) resolve paths
    against the process working directory.  Tests that exercise those code
    paths should depend on this fixture instead of calling ``os.chdir``.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# ChatML / parquet builders
# ---------------------------------------------------------------------------


def _sample_conversation(idx: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"Convert {idx}.mp4 to mp3"},
        {"role": "assistant", "content": f"ffmpeg -i {idx}.mp4 {idx}.mp3"},
    ]


@pytest.fixture
def sample_conversations() -> Callable[[int], list[list[dict[str, str]]]]:
    """Factory returning N synthetic ChatML conversations."""

    def _factory(n: int = 5) -> list[list[dict[str, str]]]:
        return [_sample_conversation(i) for i in range(n)]

    return _factory


ChatmlWriter = Callable[..., Path]


@pytest.fixture
def write_chatml_parquet() -> ChatmlWriter:
    """Factory that writes a ChatML parquet and returns its path.

    Signature::

        write_chatml_parquet(path, conversations, *, num=None, audio=False, tools=None)

    ``num`` truncates/repeats ``conversations`` to the requested size.
    ``audio`` adds a null-valued audio column to match the legacy schema.
    ``tools`` adds a JSON-encoded tools column (one entry per conversation,
    ``None`` for conversations without tools).
    """

    def _factory(
        path: Path,
        conversations: list[list[dict[str, Any]]],
        *,
        num: int | None = None,
        audio: bool = False,
        tools: list[list[dict[str, Any]] | None] | None = None,
    ) -> Path:
        if num is not None:
            while len(conversations) < num:
                conversations = conversations + conversations
            conversations = conversations[:num]

        columns: dict[str, list[Any]] = {
            "messages": [json.dumps(conv) for conv in conversations],
        }
        fields = [pa.field("messages", pa.string())]

        if audio:
            columns["audio"] = [None] * len(conversations)
            fields.append(pa.field("audio", pa.string()))

        if tools is not None:
            if len(tools) != len(conversations):
                raise ValueError("tools length must match conversations length")
            columns["tools"] = [json.dumps(t) if t is not None else None for t in tools]
            fields.append(pa.field("tools", pa.string()))

        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns, schema=pa.schema(fields)), path)
        return path

    return _factory


# ---------------------------------------------------------------------------
# Async OpenAI doubles
# ---------------------------------------------------------------------------


def _make_chat_completion(
    *,
    content: str | None = "ok",
    model: str = "small",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: Iterable[Any] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Build a duck-typed OpenAI ``ChatCompletion`` object.

    Returns a ``SimpleNamespace`` that responds to ``.choices[0].message``,
    ``.usage``, and ``.model`` exactly like the real SDK object.
    """
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls) if tool_calls else None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


@pytest.fixture
def make_chat_completion() -> Callable[..., SimpleNamespace]:
    """Builder for OpenAI ``ChatCompletion`` response objects."""
    return _make_chat_completion


@pytest.fixture
def mock_openai_client() -> Callable[..., MagicMock]:
    """Factory for ``AsyncOpenAI`` doubles.

    The default returns a single canned completion. Pass ``content=`` for a
    single reply, ``contents=[...]`` for a queue, or ``scores=[...]`` for
    judge-style JSON responses.

    Examples::

        client = mock_openai_client()                           # always "ok"
        client = mock_openai_client(content="greetings")        # canned reply
        client = mock_openai_client(scores=[8, 3, 7])           # judge queue
        client = mock_openai_client(contents=["one", "two"])    # reply queue
    """

    def _factory(
        *,
        content: str | None = None,
        contents: list[str] | None = None,
        scores: list[int | float] | None = None,
    ) -> MagicMock:
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        if scores is not None:
            replies = [json.dumps({"reasoning": f"r{s}", "score": s}) for s in scores]
        elif contents is not None:
            replies = list(contents)
        elif content is not None:
            replies = [content]
        else:
            replies = ["ok"]

        queue = list(replies)
        default = replies[-1] if replies else "ok"

        async def _create(**_: Any) -> SimpleNamespace:
            reply = queue.pop(0) if queue else default
            return _make_chat_completion(content=reply)

        client.chat.completions.create = AsyncMock(side_effect=_create)
        return client

    return _factory


# ---------------------------------------------------------------------------
# CLI options & auto-skipping
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip GPU- and API-gated tests automatically when the env is missing.

    Tests opt in by applying ``@pytest.mark.gpu`` or
    ``@pytest.mark.integration``.  Authors keep the marker; the suite skips
    when the environment cannot satisfy it.
    """
    gpu_available = _check_cuda()
    api_available = _check_api_access()

    skip_gpu = pytest.mark.skip(reason="Requires CUDA GPU")
    skip_api = pytest.mark.skip(reason="No API access (set LQH_DEBUG_API_KEY or run /login)")

    for item in items:
        if "gpu" in item.keywords and not gpu_available:
            item.add_marker(skip_gpu)
        if "integration" in item.keywords and not api_available:
            item.add_marker(skip_api)
