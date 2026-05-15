"""Data generation pipeline base classes.

Provides the abstract Pipeline interface, message dataclasses, and the step
decorator used by user-authored pipeline scripts in ``data_gen/``.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import AsyncOpenAI

__all__ = [
    "Pipeline",
    "ChatMLMessage",
    "Conversation",
    "GenerationError",
    "step",
    "safe_content",
    "ToolCall",
    "FunctionCall",
    "ToolDef",
]


class GenerationError(Exception):
    """Raise to signal a retryable failure."""
    pass


def safe_content(resp: Any) -> str:
    """Extract message content from a chat-completion response, robustly.

    Returns ``""`` if the response is missing, has no choices, the
    first choice has no message, or the content is ``None``. Never
    raises ``AttributeError`` / ``TypeError`` / ``IndexError`` —
    callers can ``if not safe_content(resp): raise GenerationError``
    and rely on the engine's retry path. Without this helper, an
    ``AttributeError`` from a malformed API response would be treated
    by ``lqh.engine`` as a deterministic bug and abort the whole run
    (see ``engine.py``: TypeError/AttributeError/ValueError are in the
    abort list because they typically indicate code bugs, not
    transient data).

    Use as the first line after every ``client.chat.completions.create``::

        resp = await client.chat.completions.create(...)
        text = safe_content(resp).strip()
        if not text:
            raise GenerationError("response was empty")
    """
    try:
        if not resp or not getattr(resp, "choices", None):
            return ""
        msg = resp.choices[0].message
        if msg is None:
            return ""
        return msg.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


@dataclass
class FunctionCall:
    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    id: str
    function: FunctionCall


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict


@dataclass
class ChatMLMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | list[dict] | None = None
    audio: bytes | None = None
    tools: list[ToolDef] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


Conversation = list[ChatMLMessage]


def step(retries: int = 0):
    """Decorator for pipeline steps with retry logic on GenerationError."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except GenerationError as e:
                    last_err = e
                    if attempt < retries:
                        continue
            raise last_err
        return wrapper
    return decorator


class Pipeline(ABC):
    @classmethod
    def source(cls, project_dir: Path) -> Iterable[Any] | None:
        return None

    @abstractmethod
    async def generate(self, client: AsyncOpenAI, input: Any = None) -> Conversation:
        ...
