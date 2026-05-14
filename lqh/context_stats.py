"""Context window usage tracker.

Records per-turn token usage to help understand what causes context blowup
and how to be more efficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnStats:
    """Token usage + output shape for a single orchestration API call."""

    turn_number: int
    prompt_tokens: int
    completion_tokens: int
    total_messages: int
    system_message_count: int
    estimated_system_tokens: int  # ~4 chars/token estimate
    skill_active: str | None = None
    compacted: bool = False  # whether compaction happened this turn
    # Output-shape diagnostics. Populated on successful API responses so we
    # can see per-turn what the model actually returned (assistant text vs
    # tool calls vs nothing), and how long the response took end-to-end.
    finish_reason: str | None = None
    tool_call_names: list[str] = field(default_factory=list)
    # Parallel to tool_call_names: the JSON-serialised (truncated) arguments
    # for each tool call in this response. Lets us answer post-hoc questions
    # like "what num_samples did the agent request?" without opening
    # transcript reports.
    tool_call_args: list[str] = field(default_factory=list)
    content_length: int = 0
    content_preview: str = ""  # first ~400 chars of message.content for triage
    duration_s: float | None = None


@dataclass
class ContextStats:
    """Accumulates per-turn token usage across a session."""

    turns: list[TurnStats] = field(default_factory=list)

    def record_turn(self, turn: TurnStats) -> None:
        self.turns.append(turn)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def total_completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def peak_prompt_tokens(self) -> int:
        return max((t.prompt_tokens for t in self.turns), default=0)

    def summary(self) -> dict:
        """Return aggregate stats as a dict."""
        if not self.turns:
            return {"turns": 0}
        return {
            "turns": len(self.turns),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "peak_prompt_tokens": self.peak_prompt_tokens,
            "avg_prompt_tokens": self.total_prompt_tokens // len(self.turns),
            "compactions": sum(1 for t in self.turns if t.compacted),
        }

    def format_report(self) -> str:
        """Human-readable table of per-turn usage."""
        if not self.turns:
            return "No turns recorded."

        lines = [
            "| Turn | Prompt | Completion | Total | Finish | Tools | Content | Dur(s) | Compacted |",
            "|------|--------|------------|-------|--------|-------|---------|--------|-----------|",
        ]
        for t in self.turns:
            total = t.prompt_tokens + t.completion_tokens
            comp = "yes" if t.compacted else ""
            tools = ",".join(t.tool_call_names) if t.tool_call_names else ""
            finish = t.finish_reason or ""
            dur = f"{t.duration_s:.1f}" if t.duration_s is not None else ""
            lines.append(
                f"| {t.turn_number} | {t.prompt_tokens:,} | {t.completion_tokens:,} | "
                f"{total:,} | {finish} | {tools} | {t.content_length} | "
                f"{dur} | {comp} |"
            )

        # Summary row
        s = self.summary()
        lines.append(
            f"| **Total** | **{s['total_prompt_tokens']:,}** | "
            f"**{s['total_completion_tokens']:,}** | **{s['total_tokens']:,}** | "
            f"| | | {s['compactions']} compactions |"
        )
        lines.append(f"\nPeak prompt tokens: {s['peak_prompt_tokens']:,}")
        return "\n".join(lines)
