"""Interactive dataset viewer for parquet files with ChatML conversations."""

from __future__ import annotations

import json
import random
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


def _make_console(width: int = 100) -> tuple[StringIO, Console]:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=width, color_system="truecolor"
    )
    return buf, console


# Role styling
_ROLE_STYLE = {
    "system": ("dim", "🔧 System"),
    "user": ("bold cyan", "👤 User"),
    "assistant": ("bold magenta", "🧪 Assistant"),
    "tool": ("yellow", "🔧 Tool"),
}


class DatasetViewer:
    """Loads a parquet dataset and renders individual ChatML samples."""

    def __init__(self, parquet_path: Path) -> None:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        self.total_rows = len(table)
        self.current_index = 0
        self.viewed_indices: set[int] = set()
        self._path = parquet_path

        # Parse rows: each row has "messages" (JSON string) and "audio" (JSON string | None)
        messages_col = table.column("messages")
        audio_col = table.column("audio") if "audio" in table.schema.names else None

        self._rows: list[dict] = []
        for i in range(self.total_rows):
            messages_json = messages_col[i].as_py()
            audio_json = audio_col[i].as_py() if audio_col is not None else None
            self._rows.append({
                "messages": json.loads(messages_json) if messages_json else [],
                "audio": json.loads(audio_json) if audio_json else None,
            })

    @property
    def empty(self) -> bool:
        return self.total_rows == 0

    def go_next(self) -> None:
        if self.current_index < self.total_rows - 1:
            self.current_index += 1
            self.viewed_indices.add(self.current_index)

    def go_prev(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.viewed_indices.add(self.current_index)

    def go_random(self) -> None:
        if self.total_rows > 1:
            self.current_index = random.randint(0, self.total_rows - 1)
            self.viewed_indices.add(self.current_index)

    def render_sample(self, width: int = 100) -> str:
        """Render the current sample as an ANSI string."""
        if self.empty:
            buf, console = _make_console(width)
            console.print(Text("Dataset is empty (0 rows)", style="dim italic"))
            return buf.getvalue()

        self.viewed_indices.add(self.current_index)
        row = self._rows[self.current_index]
        messages = row["messages"]
        audio = row["audio"]  # dict mapping message index -> base64 wav, or None

        buf, console = _make_console(width)

        # Header
        console.print()
        header = Text()
        header.append(f" Sample {self.current_index + 1}", style="bold bright_cyan")
        header.append(f" of {self.total_rows} ", style="dim")
        header.append(f"  {self._path.name}", style="dim italic")
        console.print(Panel(header, style="bright_cyan", expand=True, padding=(0, 1)))
        console.print()

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            style, label = _ROLE_STYLE.get(role, ("", f"  {role}"))

            # Tool messages: show tool name if available
            if role == "tool":
                name = msg.get("name", "")
                tool_call_id = msg.get("tool_call_id", "")
                if name:
                    label = f"🔧 Tool ({name})"
                elif tool_call_id:
                    label = f"🔧 Tool [{tool_call_id[:8]}]"

            console.print(Text(f"  {label}", style=style))

            # Render content
            if content:
                if role == "assistant":
                    # Render as markdown for assistant messages
                    md = Markdown(str(content))
                    # Indent the markdown output
                    inner_buf = StringIO()
                    inner_console = Console(
                        file=inner_buf, force_terminal=True,
                        width=width - 4, color_system="truecolor",
                    )
                    inner_console.print(md)
                    for line in inner_buf.getvalue().splitlines():
                        console.print(Text(f"    {line}"))
                elif isinstance(content, list):
                    # Multi-part content (e.g., vision messages)
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                console.print(Text(f"    {part['text']}", style=""))
                            else:
                                console.print(Text(f"    [{part.get('type', '?')}]", style="dim"))
                        else:
                            console.print(Text(f"    {part}", style=""))
                else:
                    for line in str(content).splitlines():
                        console.print(Text(f"    {line}", style=""))

            # Show tool_calls if present
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "?")
                    fn_args = fn.get("arguments", "{}")
                    console.print(Text(f"    -> {fn_name}({fn_args[:60]}{'...' if len(fn_args) > 60 else ''})", style="dim yellow"))

            # Audio indicator
            if audio and str(i) in audio:
                console.print(Text("    🔊 audio attached", style="dim magenta"))

            console.print()  # spacing between messages

        return buf.getvalue()

    def render_nav_bar(self, width: int = 100) -> str:
        """Render the navigation bar with keyboard shortcuts."""
        buf, console = _make_console(width)

        bar = Text()
        bar.append("  [", style="dim")
        bar.append("n", style="bold bright_cyan")
        bar.append("]ext  ", style="dim")
        bar.append("[", style="dim")
        bar.append("p", style="bold bright_cyan")
        bar.append("]rev  ", style="dim")
        bar.append("[", style="dim")
        bar.append("r", style="bold bright_cyan")
        bar.append("]andom  ", style="dim")
        bar.append("[", style="dim")
        bar.append("q", style="bold bright_cyan")
        bar.append("/", style="dim")
        bar.append("Esc", style="bold bright_cyan")
        bar.append("] close", style="dim")

        bar.append("  ", style="")
        bar.append("│", style="dim")
        bar.append(f"  Viewed: {len(self.viewed_indices)} sample{'s' if len(self.viewed_indices) != 1 else ''}", style="dim")

        # Position indicator
        if self.total_rows > 0:
            bar.append("  │  ", style="dim")
            bar.append(f"{self.current_index + 1}/{self.total_rows}", style="bold")

        console.print(Panel(bar, style="dim", expand=True, padding=(0, 0)))
        return buf.getvalue()

    def get_summary(self) -> str:
        """Return a summary string for the agent."""
        if self.empty:
            return f"Dataset {self._path.name} is empty (0 rows)."

        viewed = sorted(self.viewed_indices)
        if len(viewed) <= 10:
            indices_str = ", ".join(str(i) for i in viewed)
        else:
            indices_str = ", ".join(str(i) for i in viewed[:10]) + f" ... ({len(viewed)} total)"

        return (
            f"User viewed {len(viewed)} sample(s) (indices: {indices_str}) "
            f"of {self.total_rows} total rows in {self._path.name}."
        )
