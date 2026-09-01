"""Rich -> ANSI string bridge for prompt_toolkit display."""

from __future__ import annotations

import shutil
from io import StringIO
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from lqh import __version__
from lqh.tui.theme import active_palette

BLOCK_INDENT = 2
MIN_RENDER_WIDTH = 20
LIQUID_AI_LOGO = (
    "                LQH",
    "               LQHLQH",
    "               LQHLQHL",
    "            LQ   LQHLQHLQ",
    "          LQHLQ   LQHLQHLQ",
    "         LQHLQHLQ   LQHLQHLQ",
    "       LQHLQHLQ     LQHLQHLQ",
    "      LQHLQHLQ       LQHLQHLQ",
    "    LQHLQHLQ           LQHLQHLQ",
    "   LQHLQHLQ             LQHLQHLQ",
    "   LQHLQHLQ",
    "     LQHLQHLQ           LQHLQH",
    "      LQHLQHLQH   LQHLQHLQHL",
    "        LQHLQ   LQHLQHLQHLQ",
    "         LQ   LQHLQHLQHLQ",
)
WELCOME_LOGO = (
    "██╗      ██████╗  ██╗  ██╗",
    "██║     ██╔═══██╗ ██║  ██║",
    "██║     ██║   ██║ ███████║",
    "██║     ██║▄▄ ██║ ██╔══██║",
    "███████╗╚██████╔╝ ██║  ██║",
    "╚══════╝ ╚══▀▀═╝  ╚═╝  ╚═╝",
)
# Fixed column spans of the large L, Q, and H glyphs; their colors come from
# the active palette (Palette.logo_glyphs) in the same order.
WELCOME_LOGO_GLYPH_SPANS = (
    (0, 8),  # L
    (9, 18),  # Q
    (19, 27),  # H
)


def _display_width(width: int) -> int:
    """Resolve the render width from the terminal, *width* being the fallback.

    Rendered blocks are written to stdout verbatim, so a line wider than the
    terminal gets re-wrapped by the terminal itself — at its own width and
    without the block indent — which is what makes long messages look ragged
    on a narrow terminal. A wider terminal simply gets used.
    """
    # get_terminal_size falls back on its own when there is no tty (piped
    # output, CI), so a fallback of *width* keeps that case at the caller's
    # width.
    columns = shutil.get_terminal_size(fallback=(width, 24)).columns
    return max(MIN_RENDER_WIDTH, columns)


def _make_console(width: int = 100) -> Console:
    """Create a Rich console that renders to a string."""
    buf = StringIO()
    return Console(
        file=buf,
        force_terminal=True,
        width=max(MIN_RENDER_WIDTH, _display_width(width) - BLOCK_INDENT),
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
        width=max(MIN_RENDER_WIDTH, _display_width(width) - BLOCK_INDENT),
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
        file=buf, force_terminal=True, width=_display_width(width), color_system="truecolor"
    )
    console.print(Markdown(text, code_theme=active_palette().code_theme))
    return buf.getvalue()


def _render_welcome_logo_line(line: str, *, base_style: str) -> Text:
    """Render one welcome banner row with a distinct color for each glyph."""
    text = Text(f"  {line}", style=base_style)
    for (start, end), style in zip(
        WELCOME_LOGO_GLYPH_SPANS, active_palette().logo_glyphs
    ):
        text.stylize(style, 2 + start, 2 + end)
    return text


def _render_liquid_ai_logo_line(line: str) -> Text:
    """Render the Liquid AI mark in the palette's strongest foreground."""
    return Text(f"  {line}", style=active_palette().logo_mark)


def render_agent_message(text: str, width: int = 100) -> str:
    """Render an agent message (markdown) with a prefix."""
    return _render_block(
        lambda console: (
            console.print(Text("💧 Liquid", style="bold magenta")),
            console.print(Markdown(text, code_theme=active_palette().code_theme)),
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


def render_secret(text: str, width: int = 100) -> str:
    """Render a one-time secret in a distinct, attention-grabbing panel.

    Used for out-of-band secret delivery (e.g. a freshly minted inference key)
    that must stand apart from normal tool output and is never logged.
    """

    def render(console: Console) -> None:
        console.print(
            Panel(
                Text(text),
                title="🔐 ONE-TIME SECRET — copy it now",
                border_style="bold yellow",
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
        file=buf, force_terminal=True, width=_display_width(width), color_system="truecolor"
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
    warn_empty: bool = False,
    allow_other: bool = False,
    other_index: int | None = None,
    other_text: str = "",
) -> str:
    """Render a selectable options list.

    When *checked* is provided, options are rendered as checkboxes (multi-select
    mode) with Space-to-toggle hint.  Otherwise rendered as single-select radio
    with an arrow marker.

    When *warn_empty* is set (multi-select confirm guard), a prominent banner
    reminds the user that Space toggles options before they confirm an empty
    selection.

    *other_index* marks the auto-appended "Other" row in multi-select mode. It
    has no checkbox to toggle — it is filled by typing — so it echoes whatever
    is currently in the input line (*other_text*) and ticks itself once that is
    non-empty, otherwise the row looks permanently unselectable.
    """
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=_display_width(width), color_system="truecolor"
    )

    if checked is not None:
        # Multi-select (checkbox) mode
        # Collapsed, not just stripped: the echo is a single row, so a
        # newline in the answer must not split the option list.
        typed = " ".join(other_text.split())
        for i, opt in enumerate(options):
            if i == other_index:
                mark = "✓" if typed else " "
                label = f"Other: {typed}" if typed else opt
            else:
                mark = "✓" if i in checked else " "
                label = opt
            if i == selected:
                console.print(Text(f"  ▶ [{mark}] {label}", style="bold cyan"))
            else:
                console.print(Text(f"    [{mark}] {label}", style="dim"))
        if warn_empty:
            console.print(
                Text(
                    "  ⚠ Nothing selected. Press Space to toggle the highlighted "
                    "option,",
                    style="bold yellow",
                )
            )
            console.print(
                Text(
                    "    or press Enter again to answer with none selected.",
                    style="bold yellow",
                )
            )
        else:
            if other_index is not None and typed:
                # Space is a space again while an Other answer is being typed,
                # so promising "Space: toggle" here would be a lie.
                hint = "    Enter: confirm  ·  clear the line to toggle boxes again"
            elif other_index is not None:
                # The Other row has no checkbox to toggle: it is filled by
                # typing, and Enter then submits it together with the ticks.
                hint = "    Space: toggle  Enter: confirm  ·  type your own answer for Other"
            else:
                hint = "    Space: toggle  Enter: confirm"
            console.print(Text(hint, style="dim italic"))
    else:
        # Single-select (radio) mode
        for i, opt in enumerate(options):
            if i == selected:
                console.print(Text(f"  ▶ {opt}", style="bold cyan"))
            else:
                console.print(Text(f"    {opt}", style="dim"))
        hint = "    ↑↓: navigate  Enter: select"
        if allow_other:
            # The auto-appended "Other" row needs Enter first, THEN typing — the
            # single most common point of confusion. Spell out both paths.
            hint += "  ·  pick Other (or just type) for your own answer"
        console.print(Text(hint, style="dim italic"))

    console.print()
    return buf.getvalue()


def render_option_list(options: list[str], width: int = 100) -> str:
    """Render options as a numbered list for append-only terminal output."""
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=_display_width(width), color_system="truecolor"
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
            console.print(
                Syntax(
                    content,
                    lang,
                    theme=active_palette().code_theme,
                    line_numbers=True,
                )
            )
        else:
            console.print(Panel(content, title=path))

    return _render_block(render, width)


def render_welcome(width: int = 100) -> str:
    """Render the welcome screen."""
    buf = StringIO()
    liquid_logo_width = max(len(line) for line in LIQUID_AI_LOGO)
    lqh_logo_width = max(len(line) for line in WELCOME_LOGO)
    logo_gap = 4
    logo_width = 2 + liquid_logo_width + logo_gap + 2 + lqh_logo_width
    text_width = _display_width(width)
    # The ASCII art can't shrink — Rich crops to the console width even with
    # overflow="ignore" — so the logo keeps a console wide enough for it. The
    # lines under it get their own, and wrap to the terminal like every other
    # block.
    console = Console(
        file=buf,
        force_terminal=True,
        width=max(text_width, logo_width),
        color_system="truecolor",
    )

    palette = active_palette()
    logo_style = palette.logo_lqh
    accent = palette.accent

    console.print()
    lqh_start_row = (len(LIQUID_AI_LOGO) - len(WELCOME_LOGO)) // 2
    for row, liquid_line in enumerate(LIQUID_AI_LOGO):
        brand_line = _render_liquid_ai_logo_line(liquid_line)
        lqh_row = row - lqh_start_row
        if 0 <= lqh_row < len(WELCOME_LOGO):
            brand_line.append(" " * (liquid_logo_width - len(liquid_line) + logo_gap))
            brand_line.append_text(
                _render_welcome_logo_line(
                    WELCOME_LOGO[lqh_row],
                    base_style=logo_style,
                )
            )
        console.print(brand_line, no_wrap=True, overflow="ignore")

    console = Console(
        file=buf,
        force_terminal=True,
        width=text_width,
        color_system="truecolor",
    )
    console.print()
    console.print(
        Text(
            f"  Customize Liquid AI foundation models  •  v{__version__}",
            style=accent,
        )
    )
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
    console.print(
        Text("  Exit:     /exit, or press Ctrl+C twice", style="dim")
    )
    console.print()
    return buf.getvalue()


def render_resume_hint(session_id: str, width: int = 100) -> str:
    """Render the farewell hint that brings this conversation back.

    Printed after the application is torn down, so it lands in plain
    scrollback where the user can select and paste it.
    """
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, width=_display_width(width), color_system="truecolor"
    )
    console.print()
    console.print(Text("  Resume this conversation with:", style="dim"))
    console.print(
        Text(f"    lqh --resume {session_id}", style=active_palette().resume_command)
    )
    console.print()
    return buf.getvalue()
