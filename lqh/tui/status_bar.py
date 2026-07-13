"""Status bar widget for the lqh TUI."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl

from lqh.tui.background_tasks import BackgroundTask


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# The session ID (📋 1a2b3c4d) is the hash-like element we hide to make room
# for live background-task progress in the status bar. Flip to True to bring
# it back. The rendering code below is left intact behind this flag.
SHOW_SESSION_ID = False


class StatusBar:
    """Status bar showing session info, tokens, GPU, and spinner."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self.session_id: str = ""
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.spinning: bool = False
        self.pipeline_status: str = ""  # e.g. "🚀 2/10 samples (concurrency 5)"
        self.logged_in: bool = False
        self.active_skill: str = ""
        self.auto_mode: bool = False
        self.bg_tasks: list[BackgroundTask] = []
        self.recent_completion: tuple[str, float] | None = None
        self._spinner_frame: int = 0
        self._spin_start: float = 0.0
        self._hf_token: bool = bool(os.environ.get("HF_TOKEN"))
        # Compute-aware HF indicator. For a cloud project the token that
        # matters is the one STORED on the backend (injected into the
        # sandbox), not the laptop's env. compute_is_cloud +
        # hf_cloud_configured are set by the app after resolving the
        # project's compute target and querying the backend.
        self.compute_is_cloud: bool = True
        self.hf_cloud_configured: bool | None = None  # None = unknown
        self._gpu_info: str = self._detect_gpu()
        self._cwd: str = self._format_cwd(project_dir)

    @staticmethod
    def _format_cwd(project_dir: Path | None) -> str:
        """Format the working directory, replacing $HOME with ~."""
        path = str(project_dir) if project_dir else os.getcwd()
        home = os.path.expanduser("~")
        if path == home or path.startswith(home + os.sep):
            path = "~" + path[len(home):]
        return path

    @staticmethod
    def _detect_gpu() -> str:
        """Detect GPU availability."""
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                return f"🟢 {count} GPU{'s' if count > 1 else ''}"
            return "⚪ CPU only"
        except ImportError:
            return "⚪ No PyTorch"

    def start_spinning(self) -> None:
        """Mark the start of a spin (for timer tracking)."""
        self.spinning = True
        self._spin_start = time.monotonic()

    def stop_spinning(self) -> None:
        """Stop spinning and reset timer."""
        self.spinning = False
        self._spin_start = 0.0

    def _format_elapsed(self) -> str:
        """Format elapsed time since spin started."""
        if self._spin_start <= 0:
            return ""
        elapsed = time.monotonic() - self._spin_start
        if elapsed < 60:
            return f"{elapsed:.0f}s"
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        return f"{minutes}m{seconds:02d}s"

    @staticmethod
    def _format_age(seconds: float) -> str:
        """Compact relative age: 8s / 3m / 1h12m."""
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m"
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"

    def _format_bg_summary(self) -> str:
        """One-line summary of pending background tasks."""
        n = len(self.bg_tasks)
        if not n:
            return ""
        # The most recently *advancing* task is the useful headline. Polling a
        # stale job must not steal the slot from a job doing visible work.
        t = max(self.bg_tasks, key=lambda item: item.updated_at or 0.0)
        label = t.label if len(t.label) <= 32 else t.label[:31] + "…"
        remote = f"@{t.remote}" if t.remote else ""
        prefix = f"{n} active · " if n > 1 else ""
        base = f"{prefix}{t.kind}:{label}{remote}"
        if not t.progress:
            return base
        fresh = ""
        if t.updated_at is not None:
            fresh = f" · ↑{self._format_age(time.time() - t.updated_at)}"
        return f"{base} · {t.progress}{fresh}"

    def advance_spinner(self) -> None:
        """Advance the spinner animation frame."""
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)

    def get_formatted_text(self) -> FormattedText:
        """Build the status bar formatted text."""
        parts: list[tuple[str, str]] = []

        # Spinner / pipeline status / bg-tasks / idle (with elapsed timer)
        bg_count = len(self.bg_tasks)
        bg_suffix = f" +{bg_count} bg" if bg_count else ""
        if self.pipeline_status:
            frame = SPINNER_FRAMES[self._spinner_frame]
            elapsed = self._format_elapsed()
            timer = f" ({elapsed})" if elapsed else ""
            parts.append((
                "class:status.spinner",
                f" {frame} {self.pipeline_status}{timer}{bg_suffix} ",
            ))
        elif self.spinning:
            frame = SPINNER_FRAMES[self._spinner_frame]
            elapsed = self._format_elapsed()
            timer = f" ({elapsed})" if elapsed else ""
            parts.append((
                "class:status.spinner",
                f" {frame} thinking...{timer}{bg_suffix} ",
            ))
        elif self.recent_completion is not None and self.recent_completion[1] > time.time():
            parts.append(("class:status.spinner", f" ✅ {self.recent_completion[0]} completed "))
        elif bg_count:
            parts.append(("class:status.spinner", f" 🟡 {self._format_bg_summary()} "))
        else:
            parts.append(("class:status", " 🔵 ready "))

        parts.append(("class:status.separator", " │ "))

        # Working directory
        parts.append(("class:status.dim", f"📂 {self._cwd}"))

        # Session (hidden by default — see SHOW_SESSION_ID; freed the slot for
        # live background progress). Code kept intact for easy re-enabling.
        if SHOW_SESSION_ID:
            parts.append(("class:status.separator", " │ "))
            short_id = self.session_id[:8] if self.session_id else "none"
            parts.append(("class:status", f"📋 {short_id}"))

        parts.append(("class:status.separator", " │ "))

        # Token usage
        total = self.prompt_tokens + self.completion_tokens
        pct = (total / 200_000) * 100 if total > 0 else 0
        token_style = "class:status"
        if pct > 80:
            token_style = "class:status.warning"
        elif pct > 60:
            token_style = "class:status.caution"
        parts.append((token_style, f"🎯 {total:,}/200k ({pct:.0f}%)"))

        parts.append(("class:status.separator", " │ "))

        # Login status
        if self.logged_in:
            parts.append(("class:status", "🔑 ✓"))
        else:
            parts.append(("class:status.dim", "🔑 ✗"))

        parts.append(("class:status.separator", " │ "))

        # HF token — compute-aware. On a cloud project, the relevant
        # token is the one stored on the backend (used inside the
        # sandbox); on ssh/local, it's the laptop's HF_TOKEN env.
        if self.compute_is_cloud:
            if self.hf_cloud_configured is True:
                parts.append(("class:status", "🤗 HF ✓"))
            elif self.hf_cloud_configured is False:
                parts.append(("class:status.dim", "🤗 HF ✗"))
            else:
                parts.append(("class:status.dim", "🤗 HF ?"))
        elif self._hf_token:
            parts.append(("class:status", "🤗 HF ✓"))
        else:
            parts.append(("class:status.dim", "🤗 HF ✗"))

        parts.append(("class:status.separator", " │ "))

        # GPU
        parts.append(("class:status", self._gpu_info))

        # Auto-mode indicator
        if self.auto_mode:
            parts.append(("class:status.separator", " │ "))
            parts.append(("class:status.spinner", "🤖 AUTO"))

        # Active skill (if any)
        if self.active_skill:
            parts.append(("class:status.separator", " │ "))
            parts.append(("class:status.spinner", f"⚡ {self.active_skill}"))

        # Pad to fill the terminal width
        text_len = sum(len(t) for _, t in parts)
        term_width = shutil.get_terminal_size().columns
        if text_len < term_width:
            parts.append(("class:status", " " * (term_width - text_len)))

        return FormattedText(parts)

    def get_control(self) -> FormattedTextControl:
        """Return a FormattedTextControl for prompt_toolkit layout."""
        return FormattedTextControl(self.get_formatted_text)
