"""Rich -> ANSI string bridge for prompt_toolkit display."""

from __future__ import annotations

from io import StringIO
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

BLOCK_INDENT = 2
MIN_RENDER_WIDTH = 20
WELCOME_LOGO = (
    "██╗     ██╗ ██████╗ ██╗   ██╗██╗██████╗     ██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗",
    "██║     ██║██╔═══██╗██║   ██║██║██╔══██╗    ██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝",
    "██║     ██║██║   ██║██║   ██║██║██║  ██║    ███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗",
    "██║     ██║██║▄▄ ██║██║   ██║██║██║  ██║    ██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║",
    "███████╗██║╚██████╔╝╚██████╔╝██║██████╔╝    ██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║",
    "╚══════╝╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚═════╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝",
)
# Fixed column spans for the large L, Q, and H glyphs in the welcome banner.
WELCOME_LOGO_HIGHLIGHTS = ((0, 8), (11, 20), (40, 52))


def _make_console(width: int = 100) -> Console:
    """Create a Rich console that renders to a string."""
    buf = StringIO()
    return Console(
        file=buf,
        force_terminal=True,
        width=max(MIN_RENDER_WIDTH, width - BLOCK_INDENT),
        color_system="truecolor",
    )


def _render_block(
    render_fn,
    width: int = 100,
    *,
    separated: bool = True,
    indent_body_only: bool = False,
) -> str:
    """Render a message block with optional spacing and body indentation."""
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        width=max(MIN_RENDER_WIDTH, width - BLOCK_INDENT),
        color_system="truecolor",
    )
    render_fn(console)

    prefix = " " * BLOCK_INDENT
    lines = buf.getvalue().splitlines(keepends=True)
    rendered = []
    if separated:
        rendered.append("\n")

    seen_content = False
    for line in lines:
        if line.strip():
            if indent_body_only and not seen_content:
                rendered.append(line)
                seen_content = True
            else:
                rendered.append(prefix + line)
        else:
            rendered.append(line)

    return "".join(rendered)


def render_markdown(text: str, width: int = 100) -> str:
    """Render markdown text to ANSI string."""
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=width, color_system="truecolor"
    )
    console.print(Markdown(text))
    return buf.getvalue()


def _render_welcome_logo_line(line: str, *, base_style: str, accent_style: str) -> Text:
    """Render one welcome banner row with brighter L/Q/H spans."""
    text = Text(f"  {line}", style=base_style)
    for start, end in WELCOME_LOGO_HIGHLIGHTS:
        text.stylize(accent_style, 2 + start, 2 + end)
    return text


def render_agent_message(text: str, width: int = 100) -> str:
    """Render an agent message (markdown) with a prefix."""
    return _render_block(
        lambda console: (
            console.print(Text("💧 Liquid", style="bold magenta")),
            console.print(Markdown(text)),
        ),
        width,
        indent_body_only=True,
    )


def render_user_message(text: str, width: int = 100) -> str:
    """Render a user message."""
    return _render_block(
        lambda console: (
            console.print(Text("👤 You", style="bold cyan")),
            console.print(Text(text)),
        ),
        width,
        indent_body_only=True,
    )


TOOL_DISPLAY_NAMES: dict[str, str] = {
    "summary": "📊 Project Summary",
    "list_files": "📂 List Files",
    "read_file": "📄 Read File",
    "create_file": "📝 Create File",
    "write_file": "✏️ Write File",
    "edit_file": "🔧 Edit File",
    "run_data_gen_pipeline": "🚀 Run Pipeline",
    "run_scoring": "📏 Run Scoring",
    "ask_user": "💬 Ask User",
    "show_file": "👁️ Show File",
    "list_skills": "📋 List Skills",
    "load_skill": "⚡ Load Skill",
}

# Arguments to hide in tool call display (not useful to the user)
_HIDDEN_ARGS = {"content"}

# Max display length per argument value
_ARG_MAX_LEN = 60


def _format_tool_arg(key: str, value: object) -> str | None:
    """Format a single tool argument for display. Returns None to skip."""
    if key in _HIDDEN_ARGS:
        return f"{key}: ({len(str(value)):,} chars)"
    val_str = str(value)
    if len(val_str) > _ARG_MAX_LEN:
        val_str = val_str[: _ARG_MAX_LEN - 3] + "..."
    return f"{key}: {val_str}"


def render_tool_call(tool_name: str, arguments: dict, width: int = 100) -> str:
    """Render a tool call notification."""

    def render(console: Console) -> None:
        display_name = TOOL_DISPLAY_NAMES.get(tool_name, f"🔧 {tool_name}")
        console.print(Text(display_name, style="bold yellow"))

        for key, value in arguments.items():
            formatted = _format_tool_arg(key, value)
            if formatted:
                console.print(Text(f"  {formatted}", style="dim"))

    return _render_block(render, width)


def render_tool_result(tool_name: str, content: str, width: int = 100) -> str:
    """Render a tool result."""

    def render(console: Console) -> None:
        display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)

        # Truncate long results for display.
        display_content = content
        if len(display_content) > 2000:
            display_content = display_content[:2000] + "\n... (truncated)"

        console.print(
            Panel(
                Text(display_content),
                title=f"{display_name}",
                border_style="green",
                expand=False,
                padding=(0, 1),
            )
        )

    return _render_block(render, width)


def render_error(text: str, width: int = 100) -> str:
    """Render an error message."""
    return _render_block(
        lambda console: console.print(Text(f"❌ {text}", style="bold red")),
        width,
        indent_body_only=True,
    )


def render_system_message(
    text: str, width: int = 100, *, separated: bool = True
) -> str:
    """Render a system/info message."""
    return _render_block(
        lambda console: console.print(Text(f"ℹ️ {text}", style="dim italic")),
        width,
        separated=separated,
        indent_body_only=True,
    )


def render_auto_progress(
    *,
    stage: str | None,
    note: str | None,
    history: list[str],
    done: bool,
    width: int = 100,
) -> str:
    """Render the auto-mode progress panel."""
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=width, color_system="truecolor"
    )
    headline_style = "bold green" if done else "bold cyan"
    headline = "🤖 AUTO MODE"
    if done:
        headline += " — finished"
    console.print(Text(headline, style=headline_style))
    if stage:
        console.print(Text(f"  Stage: {stage}", style="bold"))
    else:
        console.print(Text("  Stage: (starting up)", style="dim"))
    if note:
        console.print(Text(f"  {note}", style="cyan"))
    console.print()
    if history:
        console.print(Text("  Progress:", style="dim italic"))
        for line in history[-8:]:
            console.print(Text(f"    {line}", style="dim"))
    return buf.getvalue()


def render_options(
    options: list[str],
    selected: int,
    width: int = 100,
    *,
    checked: set[int] | None = None,
) -> str:
    """Render a selectable options list.

    When *checked* is provided, options are rendered as checkboxes (multi-select
    mode) with Space-to-toggle hint.  Otherwise rendered as single-select radio
    with an arrow marker.
    """
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=width, color_system="truecolor"
    )

    if checked is not None:
        # Multi-select (checkbox) mode
        for i, opt in enumerate(options):
            mark = "✓" if i in checked else " "
            if i == selected:
                console.print(Text(f"  ▶ [{mark}] {opt}", style="bold cyan"))
            else:
                console.print(Text(f"    [{mark}] {opt}", style="dim"))
        console.print(Text("    Space: toggle  Enter: confirm", style="dim italic"))
    else:
        # Single-select (radio) mode
        for i, opt in enumerate(options):
            if i == selected:
                console.print(Text(f"  ▶ {opt}", style="bold cyan"))
            else:
                console.print(Text(f"    {opt}", style="dim"))

    console.print()
    return buf.getvalue()


def render_option_list(options: list[str], width: int = 100) -> str:
    """Render options as a numbered list for append-only terminal output."""
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=width, color_system="truecolor"
    )
    for i, opt in enumerate(options, start=1):
        console.print(Text(f"  {i}. {opt}", style="dim"))
    console.print()
    return buf.getvalue()


def render_file_view(path: str, content: str, width: int = 100) -> str:
    """Render a file for display (with syntax highlighting if possible)."""

    def render(console: Console) -> None:
        # Try to determine language for syntax highlighting.
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".json": "json",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".sh": "bash",
            ".html": "html",
            ".css": "css",
        }
        ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
        lang = ext_to_lang.get(ext)

        if lang:
            console.print(Syntax(content, lang, theme="monokai", line_numbers=True))
        else:
            console.print(Panel(content, title=path))

    return _render_block(render, width)


def render_welcome(width: int = 100) -> str:
    """Render the welcome screen."""
    buf = StringIO()
    logo_width = max(len(line) for line in WELCOME_LOGO) + 2
    console = Console(
        file=buf,
        force_terminal=True,
        width=max(width, logo_width),
        color_system="truecolor",
    )

    logo_style = "bold #60a5fa"
    accent_style = "bold #f8fafc"
    accent = "dim #94a3b8"

    console.print()
    for line in WELCOME_LOGO:
        console.print(
            _render_welcome_logo_line(
                line,
                base_style=logo_style,
                accent_style=accent_style,
            ),
            no_wrap=True,
            overflow="ignore",
        )

    console.print()
    console.print(Text("  Customize Liquid AI foundation models", style=accent))
    console.print()
    console.print(
        Text("  Type a message to get started, or use / for commands.", style="dim")
    )
    console.print(
        Text(
            "  Commands: /login, /clear, /resume, /spec, /datagen, /validate, /prompt",
            style="dim",
        )
    )
    console.print(
        Text("  Scroll:   use your terminal's native scrollback", style="dim")
    )
    console.print(
        Text("  Copy:     use normal click-drag selection in the terminal", style="dim")
    )
    console.print()
    return buf.getvalue()
