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
    SlashCommand("/clear", "Start a fresh conversation"),
    SlashCommand("/resume", "Resume a previous conversation"),
    SlashCommand("/spec", "Start specification capture mode"),
    SlashCommand("/datagen", "Start data generation mode"),
    SlashCommand("/validate", "Start data validation mode"),
    SlashCommand("/train", "Start training mode (requires torch)"),
    SlashCommand("/eval", "Start evaluation mode"),
    SlashCommand("/prompt", "Start prompt optimization mode"),
    SlashCommand("/help", "Show available commands"),
    SlashCommand("/quit", "Exit lqh"),
]


class SlashCommandCompleter(Completer):
    """Completer for slash commands."""

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor.strip()

        if not text.startswith("/"):
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
