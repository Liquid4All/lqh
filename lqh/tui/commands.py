"""Command palette and slash command handling for the lqh TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Awaitable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


@dataclass
class SlashCommand:
    """A slash command definition."""
    name: str
    description: str
    handler: Callable[..., Awaitable[None]] | None = None


# All available slash commands
COMMANDS: list[SlashCommand] = [
    SlashCommand("/login", "Log in to lqh.ai"),
    SlashCommand("/hf_login", "Store a Hugging Face token for cloud jobs"),
    SlashCommand("/clear", "Start a fresh conversation"),
    SlashCommand("/resume", "Resume a previous conversation"),
    SlashCommand("/spec", "Start specification capture mode"),
    SlashCommand("/datagen", "Start data generation mode"),
    SlashCommand("/validate", "Start data validation mode"),
    SlashCommand("/train", "Start training mode (requires torch)"),
    SlashCommand("/eval", "Start evaluation mode"),
    SlashCommand("/prompt", "Start prompt optimization mode"),
    SlashCommand("/reconnect", "Retry a failed network/API operation"),
    SlashCommand("/feedback", "Send feedback to the lqh team"),
    SlashCommand("/help", "Show available commands"),
    SlashCommand("/quit", "Exit lqh"),
]


class SlashCommandCompleter(Completer):
    """Completer for slash commands.

    Only completes while the buffer is a single line whose first word is
    still being typed (``/spe``); as soon as the user is past the command
    word (``/spec my task``) or composing a multiline message, it yields
    nothing so the menu closes. ``enabled`` gates completion entirely —
    the TUI uses it to suppress the menu while an ask_user prompt or the
    dataset viewer owns the input buffer.
    """

    def __init__(self, enabled: Callable[[], bool] | None = None) -> None:
        self._enabled = enabled

    def get_completions(self, document: Document, complete_event):
        if self._enabled is not None and not self._enabled():
            return

        if "\n" in document.text:
            return

        text = document.text_before_cursor.lstrip()

        # Past the command word (a space was typed) — the user is writing
        # arguments or free text; keep the menu closed.
        if not text.startswith("/") or any(ch.isspace() for ch in text):
            return

        for cmd in COMMANDS:
            if cmd.name.startswith(text):
                yield Completion(
                    cmd.name,
                    start_position=-len(text),
                    display_meta=cmd.description,
                )


def is_command(text: str) -> bool:
    """Check if the input text is a slash command."""
    return text.strip().startswith("/")


def parse_command(text: str) -> tuple[str, str]:
    """Parse a slash command into (command_name, args)."""
    text = text.strip()
    parts = text.split(None, 1)
    command = parts[0]
    args = parts[1] if len(parts) > 1 else ""
    return command, args
