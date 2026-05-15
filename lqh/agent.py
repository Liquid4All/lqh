"""Main agent loop with tool execution for lqh."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

from lqh.client import chat_with_retry, create_client
from lqh.config import load_config
from lqh.auth import get_token
from lqh.context_stats import ContextStats, TurnStats
from lqh.tools.definitions import get_all_tools
from lqh.tools.handlers import execute_tool, ToolResult
from lqh.tools.permissions import grant_permission, grant_hf_permission
from lqh.session import Session
from lqh.skills import load_skill_content

# Maximum tokens to allow in a single orchestration response. The default is
# set generously so large artifacts (long SPEC.md, detailed tool definitions,
# context summaries) can be written in one call. Backend per-model caps may
# clamp this lower; see the finish_reason / completion-size handling in the
# agent loop for recovery when that happens.
ORCHESTRATION_MAX_TOKENS = 62_000


SYSTEM_PROMPT = """\
You are Liquid Harness (lqh), an AI agent that helps users customize Liquid AI's \
foundation models (LFMs) into task-specific or domain-specific models.

## Customization Pipeline

The full pipeline for customizing an LFM is:

1. **Specification** (`/spec`) — Interview the user, create SPEC.md
2. **Data generation + eval criteria** (`/datagen`) — Create a data pipeline, iterate \
on ~20 draft samples with the user (human-in-the-loop), then create judge/scorer \
criteria while feedback is fresh, then generate a validation set (100-500 samples)
3. **Model evaluation** (`/eval`) — Run zero-shot baselines on different models, compare
4. **Prompt optimization** (`/prompt`) — Iterative system prompt refinement (2-3 rounds)
5. **Training data generation** — Scale up the same pipeline for full training set (thousands)
6. **Fine-tuning** (`/train`) — SFT on training data (requires torch)
7. **On-policy iteration** — Eval fine-tuned model, extract failure cases, generate \
targeted training data for failures, re-train. Repeat until scores plateau.

**You do NOT need to run every step.** If performance is good enough after any step, \
stop and suggest deployment. The user may also jump to a specific step or skip steps. \
Adapt to what the project needs.

### Session start behavior

- If **no SPEC.md exists**: automatically load the `spec_capture` skill and begin \
the specification interview.
- If **SPEC.md exists**: use the `summary` tool to show the current project state \
(specs, datasets, eval runs, prompts) and the activity log. Review what has been \
done and what the logical next step is. Suggest it to the user — don't just wait.

When suggesting next steps, consider the pipeline order above and the current project \
state. For example:
- Spec exists but no datasets → suggest data generation with draft iteration (`/datagen`)
- Validation set exists but no model eval runs → suggest model evaluation (`/eval`)
- Baselines exist but no prompts → suggest prompt optimization (`/prompt`)
- Good prompt exists but no training data → suggest scaling up data generation

## General behavior

Use `ask_user` for structured questions. Be concise and helpful. Use emojis to make \
the interface friendly. Guide users through the workflow step by step.

User messages prefixed with `[System: ...]` are automated notifications from the \
lqh harness — typically training-run completion or failure. Treat them as factual \
status updates: acknowledge briefly and propose the natural next step (e.g. \
`training_status` for details, then `start_local_eval` to score the new model). \
Do not ask the user a clarifying question for these messages.

Always validate your work: after creating files or generating data, read them back \
to verify correctness.

Never ask the user a question that has already been answered earlier in this \
conversation. Scan the conversation history first; if the answer is there, \
use it silently instead of re-asking.

## File naming conventions

Use descriptive, specific names everywhere. Never use generic names like data.py, \
pipeline1.py, output, or test.

### SPEC.md
The main specification. Always at the project root.

### other_specs/
Edge-case or sub-task specifications. Name by what they cover:
- other_specs/multilingual_handling.md
- other_specs/json_output_mode.md
- other_specs/long_document_edge_cases.md

### data_gen/
Pipeline scripts. Name by the spec/task they target and version:
- data_gen/summarization_v1.py
- data_gen/multilingual_edge_v1.py
- data_gen/json_output_v2.py
When a pipeline covers the main spec, use the core task name. When it targets an \
other_spec, reference that spec's topic.

### datasets/
Each pipeline run produces a subdirectory under datasets/. Use the pipeline name \
as the directory name. Inside, the engine writes data.parquet automatically.
- Draft runs (~10 samples for inspection): datasets/{name}_draft/data.parquet
- Final runs (full generation): datasets/{name}/data.parquet
- Eval sets (for model evaluation): datasets/{name}_eval/data.parquet
Examples:
- datasets/summarization_v1_draft/data.parquet  (draft, for review)
- datasets/summarization_v1/data.parquet         (final production set)
- datasets/summarization_v1_eval/data.parquet    (eval set)

Eval datasets are generated by the same data_gen pipelines as training data. \
They are always labelled (full conversations). The _eval suffix signals \
intent. The scoring engine strips assistant turns at score-time when running \
model inference.

When showing generated data to the user for review, use show_file on the parquet \
file (opens an interactive dataset viewer) followed by ask_user to get feedback. \
These can be called together in one response. If no user is attached to the \
session (headless runs), skip show_file and inspect files with read_file instead.

### prompts/
System prompts for model inference, managed separately from the data:
- prompts/{task}_v0.md - baseline prompt derived from spec (for "zero-shot" eval)
- prompts/{task}_v1.md, v2.md, ... - optimized versions from prompt optimization
- prompts/{task}.schema.json - response format schema for structured output tasks (JSON)

Data contains only user+assistant turns (no system messages). The system prompt is \
injected at eval time via the system_prompt_path parameter on run_scoring. \
If prompts/{task}.schema.json exists, it is auto-discovered and used to constrain \
the model's output format (e.g., enforcing exact JSON keys). This allows testing \
different prompts on the same eval data with guaranteed format compliance.

### evals/
Scoring and evaluation artifacts:
- evals/scorers/{name}.md - scoring criteria derived from spec(s)
- evals/runs/{run_name}/ - model evaluation results (config.json, results.parquet, summary.json)

Data quality scores are co-located with datasets: datasets/{name}/scores.parquet.

To score data quality: create a scorer .md file, then use run_scoring with mode='data_quality'. \
To evaluate a model: use run_scoring with mode='model_eval', a run_name, and system_prompt_path.\
"""

MAX_CONTEXT_TOKENS = 200_000

# When True, strip reasoning/thinking content from previous assistant turns
# before sending them in follow-up API calls.  Reduces context usage at the
# cost of losing the model's chain-of-thought in subsequent turns.
DISCARD_THINKING = os.environ.get("LQH_DISCARD_THINKING", "").lower() in ("1", "true", "yes")


@dataclass
class AgentCallbacks:
    """Callbacks for TUI integration."""
    on_agent_message: Callable[[str], Awaitable[None]] | None = None
    on_tool_call: Callable[[str, dict], Awaitable[None]] | None = None
    on_tool_result: Callable[[str, str], Awaitable[None]] | None = None
    on_ask_user: Callable[[str, list[str] | None, bool], Awaitable[str]] | None = None
    on_show_file: Callable[[str], Awaitable[str | None]] | None = None
    on_spinner_start: Callable[[], None] | None = None
    on_spinner_stop: Callable[[], None] | None = None
    on_token_update: Callable[[int, int], None] | None = None
    on_skill_loaded: Callable[[str], Awaitable[None]] | None = None
    on_pipeline_progress: Callable[[int, int, int], None] | None = None  # (completed, total, concurrency)
    on_pipeline_done: Callable[[], None] | None = None
    # Fires when a tool submits a long-running job whose completion will
    # later notify the agent (e.g. start_local_eval, start_training).
    # Signature: (task_id, kind, label, remote_name | None).
    on_background_task_started: Callable[[str, str, str, str | None], None] | None = None
    # Auto-mode: fires when the agent calls set_auto_stage (stage, note?).
    on_auto_stage: Callable[[str, str | None], None] | None = None
    # Auto-mode: fires when the agent calls exit_auto_mode (status, reason).
    on_auto_exit: Callable[[str, str], Awaitable[None]] | None = None


def _strip_thinking(msg: dict[str, Any]) -> dict[str, Any]:
    """Remove reasoning/thinking content from an assistant message.

    Handles two common formats:
    1. Top-level ``reasoning_content`` field (extended thinking API)
    2. Content blocks with ``type: "thinking"`` in a list-style ``content``

    Non-assistant messages are returned unchanged.
    """
    if msg.get("role") != "assistant":
        return msg

    cleaned = dict(msg)

    # Format 1: reasoning_content field
    cleaned.pop("reasoning_content", None)

    # Format 2: content is a list of blocks — filter out thinking blocks
    content = cleaned.get("content")
    if isinstance(content, list):
        filtered = [
            block for block in content
            if not (isinstance(block, dict) and block.get("type") == "thinking")
        ]
        if filtered:
            cleaned["content"] = filtered
        else:
            # All blocks were thinking — remove content entirely
            cleaned.pop("content", None)

    return cleaned


class Agent:
    """Main agent that manages the conversation loop with tool execution."""

    def __init__(
        self,
        project_dir: Path,
        session: Session,
        callbacks: AgentCallbacks | None = None,
        *,
        auto_mode: bool = False,
        extra_spec: str | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.session = session
        self.callbacks = callbacks or AgentCallbacks()
        self.context_stats = ContextStats()
        self.orchestration_model: str = "orchestration:2"
        self.max_tool_calls_per_turn: int = 50  # safety cap to prevent runaway loops
        self._client = None
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._turn_number = 0
        self._active_skill: str | None = None
        self._spec_edit_logged: dict[str, bool] = {}
        # Diagnostics: the string describing what the agent loop is awaiting
        # right now. Read by the harness on CancelledError so the benchmark
        # can report exactly where a scenario was stuck when it timed out.
        self._current_operation: str | None = None

        # Auto mode: sticky system messages live outside session.messages so
        # they survive compaction; _build_messages() prepends them on every
        # turn after SYSTEM_PROMPT.
        self.auto_mode = auto_mode
        self.sticky_system_messages: list[str] = []
        self._auto_exit: tuple[str, str] | None = None
        if auto_mode:
            try:
                self.sticky_system_messages.append(load_skill_content("auto"))
            except FileNotFoundError:
                pass
        if extra_spec:
            self.sticky_system_messages.append(
                "Additional user-provided context (always in scope, never drop):\n\n"
                + extra_spec
            )

    def _get_client(self):
        if self._client is None:
            config = load_config()
            token = get_token()
            if not token:
                raise RuntimeError(
                    "Not logged in. Please run /login to authenticate."
                )
            self._client = create_client(token, config.api_base_url)
        return self._client

    def _build_messages(self) -> list[dict]:
        """Build the messages list for the API call."""
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Sticky system messages (auto-mode framing, --spec context) survive
        # compaction because they're reinjected here on every API call.
        for sticky in self.sticky_system_messages:
            messages.append({"role": "system", "content": sticky})
        if DISCARD_THINKING:
            messages.extend(_strip_thinking(msg) for msg in self.session.messages)
        else:
            messages.extend(self.session.messages)
        return messages

    async def _compact_context(self) -> None:
        """Summarize and compact the conversation to free context space.

        Keeps system messages (SYSTEM_PROMPT injected at call time, plus
        skill/spec injections in session) and the last few turns.  Replaces
        everything else with a summary generated by the orchestration model.
        """
        if len(self.session.messages) < 10:
            return  # too few messages to compact

        try:
            client = self._get_client()
            # Ask the model to summarize the conversation so far
            summary_msgs: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "Summarize the key decisions, requirements, artifacts created, "
                        "and current state from the conversation below. Be concise but "
                        "include all important details (file names, scores, model choices, "
                        "spec requirements). Output only the summary, no preamble."
                    ),
                },
            ]
            # Include last ~20 messages for context (or all if fewer)
            context_msgs = self.session.messages[-20:]
            summary_msgs.extend(context_msgs)

            response = await chat_with_retry(
                client, model=self.orchestration_model, messages=summary_msgs,
                max_tokens=ORCHESTRATION_MAX_TOKENS,
                temperature=0.0,
            )
            summary_text = (response.choices[0].message.content or "").strip()
            if not summary_text:
                return

            # Keep system messages + summary + last 4 turns
            system_msgs = [m for m in self.session.messages if m.get("role") == "system"]
            recent = self.session.messages[-4:]

            self.session.messages = system_msgs + [
                {
                    "role": "system",
                    "content": (
                        f"[Context compacted] Summary of prior conversation:\n\n"
                        f"{summary_text}"
                    ),
                },
            ] + recent

            # Reset token counters (next API call will get fresh counts)
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0

            # Record compaction in stats
            if self.context_stats.turns:
                self.context_stats.turns[-1].compacted = True

            if self.callbacks.on_agent_message:
                await self.callbacks.on_agent_message(
                    "🗜️ Context compacted to free up space."
                )
        except Exception:
            pass  # compaction is best-effort; don't break the agent loop

    async def process_user_input(self, user_input: str) -> None:
        """Process a user message and run the agent loop."""
        self.session.add_message({"role": "user", "content": user_input})
        await self._run_inner_loop(is_first_turn=True)

    async def _run_inner_loop(self, is_first_turn: bool = False) -> None:
        """Inner loop: call agent, execute tools, repeat until no tools."""
        has_tools_or_first = is_first_turn
        tool_calls_this_turn = 0

        while has_tools_or_first:
            # Auto-mode: terminate cleanly once exit_auto_mode was called
            if self._auto_exit is not None:
                break
            # Safety: break if too many tool calls in a single turn
            if tool_calls_this_turn >= self.max_tool_calls_per_turn:
                if self.callbacks.on_agent_message:
                    await self.callbacks.on_agent_message(
                        f"\n⚠️ Reached {self.max_tool_calls_per_turn} tool calls in this turn. "
                        "Breaking to avoid an infinite loop. Please try a different approach."
                    )
                break
            # Call the API
            if self.callbacks.on_spinner_start:
                self.callbacks.on_spinner_start()

            try:
                client = self._get_client()
                self._current_operation = f"awaiting_chat_completion (turn {self._turn_number + 1})"
                _api_call_start = time.monotonic()
                response = await chat_with_retry(
                    client,
                    model=self.orchestration_model,
                    messages=self._build_messages(),
                    tools=get_all_tools(auto_mode=self.auto_mode),
                    tool_choice="auto",
                    max_tokens=ORCHESTRATION_MAX_TOKENS,
                    temperature=0.0,
                )
                _api_call_duration = time.monotonic() - _api_call_start
                self._current_operation = None
            except Exception as e:
                if self.callbacks.on_spinner_stop:
                    self.callbacks.on_spinner_stop()
                # Handle 401 specifically
                from openai import AuthenticationError
                if isinstance(e, AuthenticationError):
                    if self.callbacks.on_agent_message:
                        await self.callbacks.on_agent_message(
                            "❌ Authentication failed. Your API key is invalid or expired. "
                            "Please run `/login` to re-authenticate."
                        )
                    return
                raise
            else:
                if self.callbacks.on_spinner_stop:
                    self.callbacks.on_spinner_stop()

            # Track token usage
            self._turn_number += 1
            if response.usage:
                prompt_tok = response.usage.prompt_tokens or 0
                completion_tok = response.usage.completion_tokens or 0
                # Each API response's usage already reflects the full conversation
                # being sent that turn (prompt) plus what was just generated
                # (completion). Assigning — not summing — gives the actual current
                # context footprint. Summing across turns double-counts the history.
                self._total_prompt_tokens = prompt_tok
                self._total_completion_tokens = completion_tok
                self.session.prompt_tokens = self._total_prompt_tokens
                self.session.completion_tokens = self._total_completion_tokens
                if self.callbacks.on_token_update:
                    self.callbacks.on_token_update(
                        self._total_prompt_tokens,
                        self._total_completion_tokens,
                    )

                # Record context stats + output-shape diagnostics
                msgs = self.session.messages
                sys_msgs = [m for m in msgs if m.get("role") == "system"]
                sys_chars = sum(len(m.get("content", "")) for m in sys_msgs)
                _choice = response.choices[0] if response.choices else None
                _msg = _choice.message if _choice else None
                _tool_names: list[str] = []
                _tool_args: list[str] = []
                if _msg and _msg.tool_calls:
                    for _tc in _msg.tool_calls:
                        _name = _tc.function.name if _tc.function else "unknown"
                        _tool_names.append(_name)
                        _raw_args = (_tc.function.arguments if _tc.function else "") or ""
                        _tool_args.append(_raw_args[:400])
                _content = (_msg.content or "") if _msg else ""
                self.context_stats.record_turn(TurnStats(
                    turn_number=self._turn_number,
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                    total_messages=len(msgs),
                    system_message_count=len(sys_msgs),
                    estimated_system_tokens=sys_chars // 4,
                    skill_active=self._active_skill,
                    finish_reason=getattr(_choice, "finish_reason", None),
                    tool_call_names=_tool_names,
                    tool_call_args=_tool_args,
                    content_length=len(_content),
                    content_preview=_content[:400],
                    duration_s=round(_api_call_duration, 3),
                ))

            if not response.choices:
                if self.callbacks.on_agent_message:
                    await self.callbacks.on_agent_message(
                        "⚠️ Received empty response from the API. Please try again."
                    )
                return

            choice = response.choices[0]
            message = choice.message
            # Some orchestration backends enforce a per-model output cap below
            # our max_tokens setting. When that happens the assistant message
            # (and any tool-call arguments it contained) is truncated mid-
            # content. Record this so we can notify the model after the
            # current turn finishes processing.
            #
            # Two signals can indicate truncation:
            #   1. finish_reason == "length" — the obvious case.
            #   2. finish_reason == "tool_calls" but completion_tokens is
            #      close to our ORCHESTRATION_MAX_TOKENS setting (>=90%).
            #      When a response ends with a tool_call whose arguments were
            #      cut off, the API still reports finish_reason="tool_calls"
            #      and closes the JSON gracefully, so we must detect it via
            #      size. Small tool calls never trip this heuristic.
            finish_reason = getattr(choice, "finish_reason", None)
            _completion_tok = (
                response.usage.completion_tokens
                if response.usage and response.usage.completion_tokens
                else 0
            )
            truncated_by_length = (
                finish_reason == "length"
                or (
                    finish_reason == "tool_calls"
                    and _completion_tok >= int(ORCHESTRATION_MAX_TOKENS * 0.9)
                )
            )

            # Build the assistant message dict
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if message.content:
                assistant_msg["content"] = message.content
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in message.tool_calls
                ]

            self.session.add_message(assistant_msg)

            # Display agent text
            if message.content and self.callbacks.on_agent_message:
                await self.callbacks.on_agent_message(message.content)

            # Process tool calls
            if message.tool_calls:
                has_tools_or_first = True
                tool_calls_this_turn += len(message.tool_calls)
                for tc in message.tool_calls:
                    tool_name = tc.function.name if tc.function else "unknown"
                    try:
                        arguments = json.loads(tc.function.arguments) if tc.function and tc.function.arguments else {}
                    except json.JSONDecodeError:
                        arguments = {}

                    if self.callbacks.on_tool_call:
                        await self.callbacks.on_tool_call(tool_name, arguments)

                    try:
                        self._current_operation = f"executing_tool:{tool_name}"
                        result = await self._handle_tool_call(tool_name, arguments)
                        self._current_operation = None
                    except Exception as e:
                        result = ToolResult(
                            content=f"Internal error executing {tool_name}: {type(e).__name__}: {e}"
                        )

                    # Add tool result to conversation
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.content,
                    }
                    self.session.add_message(tool_msg)

                    if self.callbacks.on_tool_result:
                        await self.callbacks.on_tool_result(tool_name, result.content)

                    # Auto-mode: surface stage updates to the TUI
                    if result.auto_stage and self.callbacks.on_auto_stage:
                        self.callbacks.on_auto_stage(
                            result.auto_stage, result.auto_stage_note,
                        )

                    # Auto-mode: explicit termination tool was called
                    if result.exit_auto_mode and self.auto_mode:
                        self._auto_exit = (
                            result.auto_status or "failure",
                            result.auto_reason or "",
                        )

                    # Project activity log: reset coalescing on non-edit events
                    if tool_name in ("run_data_gen_pipeline", "run_scoring") and not result.content.startswith("❌"):
                        self._spec_edit_logged = {}

                    # Project activity log: spec edit coalescing
                    if tool_name in ("create_file", "write_file", "edit_file"):
                        from lqh.project_log import append_event, file_hash_prefix, is_spec_file

                        file_path = arguments.get("path", "")
                        if is_spec_file(file_path) and not result.content.startswith("Error"):
                            if not self._spec_edit_logged.get(file_path):
                                evt = "spec_created" if tool_name == "create_file" else "spec_updated"
                                append_event(
                                    self.project_dir,
                                    evt,
                                    f"{'Created' if tool_name == 'create_file' else 'Updated'} {file_path}",
                                    path=file_path,
                                    hash=file_hash_prefix(self.project_dir / file_path),
                                )
                                self._spec_edit_logged[file_path] = True

                    # Handle skill loading (with compaction)
                    if result.skill_content:
                        skill_name = arguments.get("skill_name", tool_name)
                        self._active_skill = skill_name
                        # Compact context before loading a new skill
                        if len(self.session.messages) > 10:
                            await self._compact_context()
                        skill_msg = {"role": "system", "content": result.skill_content}
                        self.session.add_message(skill_msg)
                        if self.callbacks.on_skill_loaded:
                            await self.callbacks.on_skill_loaded(skill_name)
            else:
                if self.auto_mode and self._auto_exit is None:
                    # Auto mode: a turn with no tool call would otherwise return
                    # control to the user. There is no user — nudge the agent to
                    # keep going. Per AUTOMODE.md §7.4.
                    self.session.add_message({
                        "role": "user",
                        "content": (
                            "[Auto mode] You produced a turn without a tool call. "
                            "There is no user to wait for. Continue the auto-mode "
                            "pipeline, or call exit_auto_mode if you have reached a "
                            "terminal state."
                        ),
                    })
                    has_tools_or_first = True
                else:
                    has_tools_or_first = False

            # If the model's output was truncated by the per-model token cap,
            # notify it so it can recover on the next iteration. This happens
            # when e.g. a long SPEC.md is written in a single create_file call
            # and the arguments JSON is cut off mid-content. We inject a user-
            # role message because it reads naturally in the history and is
            # always safe to append (no tool_call pairing constraints).
            if truncated_by_length:
                notice = (
                    "⚠️ Your previous response was cut off because it exceeded the "
                    "model's output token limit. If it contained a tool call with "
                    "long arguments (e.g., create_file / write_file on a large file), "
                    "the arguments may be invalid JSON and the tool result will "
                    "reflect that. To recover:\n"
                    "- For long files such as SPEC.md, build incrementally: "
                    "`create_file` with the header + first 1-2 sections, then "
                    "`edit_file` (or subsequent `write_file`) to append each "
                    "remaining section.\n"
                    "- Keep individual tool-call content under ~3000 tokens.\n"
                    "- If the tool call failed, retry it with shorter content."
                )
                self.session.add_message({"role": "user", "content": notice})
                if self.callbacks.on_agent_message:
                    # Surface to the user/transcript as an agent note so it is
                    # visible in reports; it's still in the conversation as a
                    # user turn.
                    await self.callbacks.on_agent_message(
                        "⚠️ Previous response truncated by output token limit — "
                        "agent will retry with a continuation notice."
                    )
                # Ensure the loop continues so the agent can act on the notice.
                has_tools_or_first = True

            # Check context window — compact at 80%, warn at 90%.
            # Use the MOST RECENT prompt_tokens as the context-size proxy —
            # that reflects the conversation footprint actually being sent to
            # the model. Using cumulative _total_prompt_tokens is wrong
            # because it sums invoice totals across API calls and triggers
            # spurious compactions on healthy conversations.
            current_prompt_tokens = 0
            if self.context_stats.turns:
                current_prompt_tokens = self.context_stats.turns[-1].prompt_tokens
            if current_prompt_tokens > MAX_CONTEXT_TOKENS * 0.8:
                await self._compact_context()
            if current_prompt_tokens > MAX_CONTEXT_TOKENS * 0.9:
                if self.callbacks.on_agent_message:
                    await self.callbacks.on_agent_message(
                        "\n⚠️ Context window is almost full "
                        f"({current_prompt_tokens:,}/{MAX_CONTEXT_TOKENS:,} tokens). "
                        "Consider starting a new session with /clear."
                    )

    def _pipeline_kwargs(self) -> dict:
        """Build extra kwargs for pipeline execution with TUI callbacks."""
        return {
            "on_pipeline_progress": self.callbacks.on_pipeline_progress,
            "on_pipeline_done": self.callbacks.on_pipeline_done,
        }

    async def _handle_tool_call(self, tool_name: str, arguments: dict) -> ToolResult:
        """Handle a single tool call, including user interaction tools."""
        # Auto mode: show_file has no audience. Nudge the agent toward read_file
        # instead of running the handler (which would set show_file_path and
        # trigger the blocking TUI viewer callback at the bottom of this method).
        if tool_name == "show_file" and self.auto_mode:
            return ToolResult(
                content=(
                    "[Auto mode] No user is attached to view the file. "
                    "Use `read_file` (supports offset/limit pagination) to "
                    "inspect the contents yourself. Do not call show_file "
                    "again in this run."
                ),
            )
        extra: dict[str, Any] = {}
        if tool_name in ("run_data_gen_pipeline", "run_scoring", "run_data_filter"):
            extra = self._pipeline_kwargs()
        if tool_name in ("start_training", "start_local_eval"):
            # Lets handlers eagerly register a background task with the TUI
            # the moment a job is submitted, so the status bar updates
            # immediately instead of waiting for the next 60-second
            # _watch_jobs poll.
            extra["on_background_task_started"] = self.callbacks.on_background_task_started
        result = await execute_tool(tool_name, arguments, self.project_dir, **extra)

        # Handle ask_user tool
        if result.requires_user_input and tool_name == "ask_user":
            # Auto mode: never block on a user. Inject a synthetic nudge instead.
            # Per AUTOMODE.md §7.2.
            if self.auto_mode:
                return ToolResult(
                    content=(
                        "[Auto mode] You called ask_user, but there is no user. "
                        "Resolve the situation yourself: pick a sensible default, "
                        "log your reasoning, and continue the pipeline. Never call "
                        "ask_user again in this run."
                    ),
                )
            if self.callbacks.on_ask_user:
                user_response = await self.callbacks.on_ask_user(
                    result.question or "", result.options, result.multi_select
                )
                return ToolResult(content=user_response)
            else:
                return ToolResult(content="[No user input handler available]")

        # Handle permission request (pipeline execution or HF push)
        if result.requires_user_input and result.content == "PERMISSION_REQUIRED":
            # Auto mode: auto-grant project-wide so the pipeline never blocks.
            if self.auto_mode:
                synthetic_choice = "Execute and don't ask again for this project"
                return await self._handle_permission_response(
                    synthetic_choice, tool_name, arguments
                )
            if self.callbacks.on_ask_user:
                user_response = await self.callbacks.on_ask_user(
                    result.question or "", result.options
                )
                return await self._handle_permission_response(
                    user_response, tool_name, arguments
                )
            else:
                return ToolResult(content="[No user input handler available]")

        # Handle show_file
        if result.show_file_path and self.callbacks.on_show_file:
            viewer_summary = await self.callbacks.on_show_file(result.show_file_path)
            if viewer_summary:
                return ToolResult(content=viewer_summary)

        return result

    async def _handle_permission_response(
        self, response: str, tool_name: str, tool_args: dict
    ) -> ToolResult:
        """Process the user's permission choice for pipeline execution or HF push."""
        if tool_name == "hf_push":
            return await self._handle_hf_push_permission(response, tool_args)

        # Pipeline execution permission (run_data_gen_pipeline)
        script_path = tool_args.get("script_path", "")

        if "Do not execute" in response:
            return ToolResult(content="Pipeline execution declined by user.")

        # Grant appropriate permission
        if "don't ask again for this project" in response:
            grant_permission(self.project_dir, None, project_wide=True)
        elif "don't ask again for this file" in response:
            grant_permission(self.project_dir, script_path)

        # Execute the pipeline
        from lqh.tools.handlers import _execute_pipeline
        return await _execute_pipeline(
            self.project_dir,
            script_path,
            tool_args.get("num_samples", 1),
            tool_args.get("output_dataset", "output"),
            tool_args.get("validation_instructions"),
            on_pipeline_progress=self.callbacks.on_pipeline_progress,
            on_pipeline_done=self.callbacks.on_pipeline_done,
        )

    async def _handle_hf_push_permission(
        self, response: str, push_args: dict
    ) -> ToolResult:
        """Process the user's permission choice for HF push."""
        if "Do not push" in response:
            return ToolResult(content="HF push declined by user.")

        repo_id = push_args.get("repo_id", "")

        # Grant appropriate permission
        if "don't ask again for this project" in response:
            grant_hf_permission(self.project_dir, project_wide=True)
        elif "don't ask again for this repo" in response:
            grant_hf_permission(self.project_dir, repo_id=repo_id)

        # Execute the push
        from lqh.tools.handlers import _execute_hf_push, _get_hf_api, _validate_path

        api = _get_hf_api()
        local_path = push_args.get("local_path", "")
        target = _validate_path(self.project_dir, local_path)

        # Find parquet file
        if target.is_dir():
            data_parquet = target / "data.parquet"
            parquet_files = list(target.glob("*.parquet"))
            parquet_path = data_parquet if data_parquet.exists() else parquet_files[0]
        else:
            parquet_path = target

        return await _execute_hf_push(
            self.project_dir,
            parquet_path,
            local_path,
            repo_id,
            push_args.get("private", True),
            push_args.get("split", "train"),
            push_args.get("subset"),
            push_args.get("commit_message"),
            api,
        )

    async def prepare_context(self) -> str:
        """Pre-populate the agent context without making any LLM calls.

        Returns a mode string indicating what was loaded:
        - "new_project" if no SPEC.md exists (spec_capture skill loaded)
        - "existing_project" if SPEC.md exists (spec + summary injected)
        """
        spec_path = self.project_dir / "SPEC.md"

        if not spec_path.exists():
            # Load spec capture skill into context
            try:
                content = load_skill_content("spec_capture")
                self.session.add_message({"role": "system", "content": content})
            except FileNotFoundError:
                pass
            mode = "new_project"
        else:
            # Inject SPEC.md content into context
            spec_content = spec_path.read_text(encoding="utf-8")
            self.session.add_message({
                "role": "system",
                "content": (
                    f"The user's main specification (SPEC.md):\n\n{spec_content}"
                ),
            })

            # Run summary tool directly (no LLM) and inject result
            from lqh.tools.handlers import handle_summary
            summary_result = await handle_summary(self.project_dir)
            self.session.add_message({
                "role": "system",
                "content": (
                    f"Current project state:\n\n{summary_result.content}"
                ),
            })

            mode = "existing_project"

        # Inject project activity log (last 50 events)
        from lqh.project_log import format_log_for_context, read_recent

        entries = read_recent(self.project_dir, n=50)
        if entries:
            self.session.add_message({
                "role": "system",
                "content": (
                    f"Project activity log (most recent events):\n\n"
                    f"{format_log_for_context(entries)}"
                ),
            })

        return mode
