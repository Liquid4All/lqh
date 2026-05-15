"""E2E test harness: runs the full agent loop with an LLM-simulated human."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from lqh.agent import Agent, AgentCallbacks
from lqh.auth import require_token
from lqh.client import capture_api_metrics, create_client
from lqh.config import load_config
from lqh.context_stats import ContextStats
from lqh.session import Session

from tests.e2e.scenarios import Scenario

logger = logging.getLogger(__name__)

# Silence noisy HTTP request logs from the openai/httpx stack
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Strict global caps for E2E tests.  If any of these trip, the run is
# considered a failure — the agent is either stuck in a loop or using
# too much context.
MAX_TOTAL_TOOL_CALLS = 50
MAX_TOOL_CALLS_PER_TURN = 20

# Context compaction tolerance. A single compaction at a skill boundary is
# normal production behaviour (keeps context lean across stages). Treat it as
# a warning. Abort only if compactions repeat or the context genuinely grew
# too large, which both indicate a real problem.
MAX_COMPACTIONS_BEFORE_ABORT = 2
COMPACTION_TOKEN_ABORT_THRESHOLD = 150_000


class E2EAbort(Exception):
    """Raised to abort the E2E run immediately on a strict-limit violation."""
    pass


@dataclass
class TurnRecord:
    """A single event in the E2E transcript."""

    role: str  # "user", "agent", "tool_call", "tool_result", "ask_user_q", "ask_user_a", "chat_q", "chat_a", "skill_loaded"
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    timestamp: float = 0.0


@dataclass
class E2EResult:
    """Complete result from an E2E test run."""

    scenario_name: str
    scenario_description: str
    project_dir: Path
    transcript: list[TurnRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    total_turns: int = 0
    total_tool_calls: int = 0
    skills_loaded: list[str] = field(default_factory=list)
    context_stats: ContextStats = field(default_factory=ContextStats)
    orchestration_model: str = "orchestration:2"
    # Files created by scenario.seed_fn (relative paths). Judges should
    # exclude these because they are not the agent's work.
    seeded_files: set[str] = field(default_factory=set)
    # Per-attempt metrics for every chat_with_retry call, populated via
    # lqh.client.capture_api_metrics during the scenario. Each entry is a
    # dict (see capture_api_metrics docstring) with timing, error type,
    # finish_reason, tokens, and tool-call counts.
    api_call_log: list[dict] = field(default_factory=list)
    # Snapshot of agent._current_operation captured at the moment the run
    # ended (most useful on CancelledError / timeout: tells us what the
    # agent loop was awaiting when it was killed).
    last_operation: str | None = None

    @property
    def tool_calls(self) -> list[TurnRecord]:
        return [t for t in self.transcript if t.role == "tool_call"]

    @property
    def artifacts(self) -> dict[str, str]:
        """Collect all text files created in the project directory."""
        result = {}
        for path in self.project_dir.rglob("*"):
            if path.is_file() and not str(path).startswith(str(self.project_dir / ".lqh")):
                rel = str(path.relative_to(self.project_dir))
                try:
                    result[rel] = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    result[rel] = f"<binary, {path.stat().st_size} bytes>"
        return result

    def tools_called(self) -> set[str]:
        return {t.tool_name for t in self.tool_calls if t.tool_name}

    @property
    def pipeline_attempts(self) -> dict[str, int]:
        """Count run_data_gen_pipeline calls: total, failed, succeeded."""
        calls = [t for t in self.transcript if t.role == "tool_call" and t.tool_name == "run_data_gen_pipeline"]
        results = [t for t in self.transcript if t.role == "tool_result" and t.tool_name == "run_data_gen_pipeline"]
        failed = sum(1 for r in results if "❌" in r.content or "Error" in r.content)
        succeeded = len(results) - failed
        return {"total": len(calls), "failed": failed, "succeeded": succeeded}

    def has_errors(self) -> bool:
        return len(self.errors) > 0


class E2EHarness:
    """Runs the agent loop with an LLM-simulated human."""

    def __init__(self, scenario: Scenario, orchestration_model: str = "orchestration:2") -> None:
        self.scenario = scenario
        self.orchestration_model = orchestration_model
        self.project_dir = Path(tempfile.mkdtemp(prefix=f"lqh_e2e_{scenario.name}_"))
        self._transcript: list[TurnRecord] = []
        self._errors: list[str] = []
        self._skills_loaded: list[str] = []
        self._turn_count = 0
        self._tool_call_count = 0
        self._sim_human_history: list[dict[str, str]] = []
        self._current_stage: str | None = None
        self._stage_turns: int = 0
        self._compaction_count = 0
        self._warnings: list[str] = []
        # Paths created by scenario.seed_fn (relative to project_dir). These
        # files existed before the agent ran and should not count against the
        # agent when judged.
        self._seeded_files: set[str] = set()
        # Populated during run() via lqh.client.capture_api_metrics.
        self._api_call_log: list[dict] = []
        self._last_operation: str | None = None

        # Create API client
        config = load_config()
        token = require_token()
        self._client = create_client(token, config.api_base_url)

    def _build_result(self, duration: float = 0.0, agent: Agent | None = None) -> E2EResult:
        """Build an E2EResult from the current state (works even mid-run)."""
        return E2EResult(
            scenario_name=self.scenario.name,
            scenario_description=self.scenario.description,
            project_dir=self.project_dir,
            transcript=self._transcript,
            errors=self._errors,
            warnings=list(self._warnings),
            duration_seconds=duration,
            total_turns=self._turn_count,
            total_tool_calls=self._tool_call_count,
            skills_loaded=self._skills_loaded,
            context_stats=agent.context_stats if agent else ContextStats(),
            orchestration_model=self.orchestration_model,
            seeded_files=set(self._seeded_files),
            api_call_log=list(self._api_call_log),
            last_operation=(agent._current_operation if agent else None) or self._last_operation,
        )

    def _record(self, role: str, content: str, **kwargs: Any) -> None:
        self._transcript.append(TurnRecord(
            role=role,
            content=content,
            timestamp=time.time(),
            **kwargs,
        ))

    @staticmethod
    def _match_option(response: str, options: list[str]) -> str:
        """Match an LLM response to the closest available option."""
        resp_lower = response.lower().strip()
        # Exact match
        for opt in options:
            if opt.lower() == resp_lower:
                return opt
        # Response contains option text (or vice versa)
        for opt in options:
            if opt.lower() in resp_lower or resp_lower in opt.lower():
                return opt
        # Number match ("1", "2", etc.)
        for i, opt in enumerate(options, 1):
            if resp_lower == str(i):
                return opt
        # Fallback: return raw response (treated as "Other")
        return response

    async def _simulated_human(
        self,
        question: str,
        options: list[str] | None,
        multi_select: bool = False,
        source: str = "ask_user",
    ) -> str:
        """LLM-backed simulated human for agent questions.

        ``source`` distinguishes real ``ask_user`` tool calls from open-ended
        chat questions that the agent asked without the tool. The former are
        recorded as ``ask_user_q``/``ask_user_a``; the latter as
        ``chat_q``/``chat_a``. The scorer uses this to tier spec-capture
        quality (guided > chat).
        """
        q_role = "ask_user_q" if source == "ask_user" else "chat_q"
        a_role = "ask_user_a" if source == "ask_user" else "chat_a"
        self._record(q_role, question)

        # Build the prompt for the simulated human
        user_prompt = f"The agent is asking you:\n\n{question}\n"
        if options:
            user_prompt += "\nYou MUST pick one of these options (copy the text exactly):\n"
            for i, opt in enumerate(options, 1):
                user_prompt += f"  {i}. {opt}\n"
            if multi_select:
                user_prompt += "\nYou may select multiple (comma-separated).\n"
            user_prompt += (
                "\nRespond with ONLY the option text, copied verbatim. "
                "Do not add commentary, do not say 'OK', do not rephrase."
            )
        else:
            user_prompt += (
                "\nProvide a short, specific answer. "
                "Do not just say 'OK' — give a real answer."
            )

        self._sim_human_history.append({"role": "user", "content": user_prompt})

        try:
            response = await self._client.chat.completions.create(
                model="orchestration",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are simulating a human user in a conversation with an AI agent.\n\n"
                            f"Your scenario: {self.scenario.description}\n\n"
                            f"Rules:\n"
                            f"- When given numbered options, you MUST pick one and respond with the exact option text only\n"
                            f"- When given free-text prompts, give short practical answers (never just 'OK')\n"
                            f"- Stay in character — you are the user, not the agent\n"
                            f"- If asked for examples, provide realistic ones for your task\n"
                            f"- Respond with ONLY your answer, nothing else"
                        ),
                    },
                    *self._sim_human_history,
                ],
                temperature=0.0,
                max_tokens=500,
            )
            answer = response.choices[0].message.content or ""
            answer = answer.strip()
            if not answer:
                answer = options[0] if options else "I'm not sure, please suggest something."
        except Exception as e:
            logger.error("Simulated human LLM call failed: %s", e)
            answer = options[0] if options else "I'm not sure, please suggest something."

        # When options are provided, fuzzy-match the response to an actual option
        if options and not multi_select:
            answer = self._match_option(answer, options)

        self._sim_human_history.append({"role": "assistant", "content": answer})
        self._record(a_role, answer)
        return answer

    def _make_callbacks(self) -> AgentCallbacks:
        """Create callbacks that record everything and use the simulated human."""

        async def on_agent_message(text: str) -> None:
            self._record("agent", text)
            # Context compaction detection. A single compaction at a skill
            # boundary is normal production behaviour (agent.py compacts when a
            # new skill is loaded and the conversation has >10 messages).
            # Only treat it as a catastrophic abort if compactions repeat or
            # the context genuinely grew past a concerning size — either
            # signals the agent is stuck in a loop or burning context.
            if "Context compacted" in text or "🗜️" in text:
                self._compaction_count += 1
                # Find the most recent peak prompt-token count, if stats are
                # available. We do not have direct access to the Agent here,
                # so we look for compaction token info in the agent message
                # (agent.py reports it) — otherwise rely on the count.
                if self._compaction_count >= MAX_COMPACTIONS_BEFORE_ABORT:
                    self._errors.append(
                        f"Strict limit violation: {self._compaction_count} "
                        "context compactions occurred (agent used too much "
                        "context or is looping)"
                    )
                    raise E2EAbort(
                        f"{self._compaction_count} compactions exceeded limit"
                    )
                # First compaction: warn-only, let the run continue.
                self._warnings.append(
                    f"context compaction #{self._compaction_count} "
                    "(expected at skill boundaries; not aborting)"
                )

        async def on_tool_call(name: str, args: dict) -> None:
            self._tool_call_count += 1
            self._record("tool_call", f"{name}({json.dumps(args, ensure_ascii=False)[:200]})",
                         tool_name=name, tool_args=args)
            # Enforce global tool-call cap.
            if self._tool_call_count > MAX_TOTAL_TOOL_CALLS:
                self._errors.append(
                    f"Strict limit violation: exceeded {MAX_TOTAL_TOOL_CALLS} "
                    f"total tool calls (agent likely stuck in a loop)"
                )
                raise E2EAbort(
                    f"Exceeded {MAX_TOTAL_TOOL_CALLS} total tool calls"
                )

        async def on_tool_result(name: str, content: str) -> None:
            # Check for errors
            if content.startswith("❌") or content.startswith("Error:"):
                self._errors.append(f"{name}: {content[:200]}")
            self._record("tool_result", content[:500], tool_name=name)

        async def on_ask_user(question: str, options: list[str] | None, multi_select: bool = False) -> str:
            return await self._simulated_human(question, options, multi_select)

        async def on_show_file(path: str) -> str | None:
            self._record("tool_result", f"[show_file: {path}]", tool_name="show_file")
            return None

        async def on_skill_loaded(skill_name: str) -> None:
            self._skills_loaded.append(skill_name)
            self._current_stage = skill_name
            self._stage_turns = 0
            self._record("skill_loaded", skill_name)

        return AgentCallbacks(
            on_agent_message=on_agent_message,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_ask_user=on_ask_user,
            on_show_file=on_show_file,
            on_skill_loaded=on_skill_loaded,
        )

    async def run(self, initial_message: str | None = None) -> E2EResult:
        """Run the full E2E scenario.

        The agent drives the conversation. The harness provides simulated
        human responses to ask_user calls. Stops after the agent returns
        without tool calls (natural end) or when max_turns is reached.
        """
        start = time.time()

        # Enable debug mode for E2E tests (generates curl scripts for low-scoring samples)
        os.environ["LQH_DEBUG"] = "1"

        # Pre-seed the project directory if the scenario provides a seed function.
        # Capture the set of files that existed after seeding so the judge can
        # exclude them — they are not the agent's work.
        if self.scenario.seed_fn:
            import inspect
            if inspect.iscoroutinefunction(self.scenario.seed_fn):
                await self.scenario.seed_fn(self.project_dir)
            else:
                self.scenario.seed_fn(self.project_dir)
            for path in self.project_dir.rglob("*"):
                if path.is_file() and not str(path).startswith(str(self.project_dir / ".lqh")):
                    rel = str(path.relative_to(self.project_dir))
                    self._seeded_files.add(rel)

        session = Session.create(self.project_dir)
        agent = Agent(self.project_dir, session, callbacks=self._make_callbacks())
        agent.orchestration_model = self.orchestration_model
        agent.max_tool_calls_per_turn = MAX_TOOL_CALLS_PER_TURN  # strict E2E cap

        # Enable per-attempt API metrics capture — every chat_with_retry call
        # from here on writes a record to self._api_call_log. This survives
        # CancelledError / timeout so we can see exactly where runs hang.
        _metrics_token = capture_api_metrics(self._api_call_log)
        try:
            await agent.prepare_context()

            # First user turn
            msg = initial_message or self.scenario.initial_message
            self._record("user", msg)
            self._turn_count += 1

            await agent.process_user_input(msg)

            # The agent may ask the user what to do next via ask_user.
            # The simulated human's scenario description guides transitions.
            # We run additional turns if the agent's last message suggests
            # a transition and the simulated human responded with a next step.

            # Check if more turns are needed by looking at recent ask_user answers
            while self._turn_count < self.scenario.max_turns:
                # Check per-stage turn limit
                if self._current_stage and self.scenario.stage_limits:
                    limit = self.scenario.stage_limits.get(self._current_stage)
                    if limit is not None and self._stage_turns >= limit:
                        logger.info(
                            "Stage '%s' hit turn limit (%d), injecting done signal",
                            self._current_stage, limit,
                        )
                        self._record("user", "I'm done for now")
                        self._turn_count += 1
                        await agent.process_user_input("I'm done for now")
                        break

                # Find the last simulated-human answer (from either ask_user or
                # chat) and the last agent message, to decide whether the agent
                # is waiting on the user.
                last_answer = None
                last_answer_idx = -1
                last_agent = None
                last_agent_idx = -1
                for idx, rec in enumerate(self._transcript):
                    if rec.role in ("ask_user_a", "chat_a"):
                        last_answer = rec.content
                        last_answer_idx = idx
                    elif rec.role == "agent":
                        last_agent = rec
                        last_agent_idx = idx

                # Check "done" signal from the most recent simulated answer.
                done_signals = ["done for now", "i'm done", "that's all", "exit", "quit"]
                if last_answer is not None and any(s in last_answer.lower() for s in done_signals):
                    break

                # Decide the follow-up:
                # 1) If the last agent message came AFTER the last answer and
                #    contains a question, the agent is asking via plain chat
                #    (not ask_user). Synthesize a simulated reply with
                #    source="chat" so the scorer can tell guided from chat.
                # 2) Else, feed the last ask_user answer back as a follow-up.
                if last_agent is None:
                    break

                agent_asked_open_question = (
                    last_agent_idx > last_answer_idx
                    and "?" in last_agent.content
                )

                if agent_asked_open_question:
                    follow_up = await self._simulated_human(
                        last_agent.content, options=None, multi_select=False,
                        source="chat",
                    )
                elif last_answer is not None:
                    follow_up = last_answer
                else:
                    break  # Agent finished without asking anything

                self._turn_count += 1
                self._stage_turns += 1
                self._record("user", follow_up)
                await agent.process_user_input(follow_up)
        except E2EAbort as exc:
            # Strict limit violation — reason already appended to errors
            logger.warning("E2E aborted by strict limit: %s", exc)
        except (Exception, asyncio.CancelledError) as exc:
            self._errors.append(f"Harness error: {type(exc).__name__}: {exc}")
        finally:
            # Snapshot what the agent was awaiting at the moment this scope
            # exited (most revealing on CancelledError / timeout).
            self._last_operation = getattr(agent, "_current_operation", None)
            try:
                from lqh.client import _api_metrics_log
                _api_metrics_log.reset(_metrics_token)
            except Exception:
                pass

        duration = time.time() - start
        return self._build_result(duration=duration, agent=agent)
