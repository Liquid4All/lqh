"""Benchmark scoring: computes per-run scores from E2E results."""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from tests.e2e.harness import E2EResult
from tests.e2e.judge import judge_artifacts, JudgeResult, JUDGE_SCHEMA, _coerce_judge_payload
from tests.e2e.scenarios import Scenario

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkScore:
    """Unified score for one scenario+model run."""

    scenario_name: str
    category: str
    model: str

    # Raw pass/fail
    completed_without_abort: bool
    expected_tools_called: bool
    expected_files_exist: bool

    # Artifact judge score (1-10, averaged across artifacts)
    artifact_judge_score: float

    # Composite score (0-100, weighted by category)
    composite_score: float

    # Worst-case flag
    is_catastrophic_failure: bool

    # Raw metrics
    duration_seconds: float
    total_tool_calls: int
    total_turns: int

    # Category-specific details
    category_details: dict[str, Any] = field(default_factory=dict)

    # Judge results for reference
    judge_results: list[dict] = field(default_factory=list)

    # Errors from the underlying E2EResult (seed crashes, aborts, timeouts, etc.)
    errors: list[str] = field(default_factory=list)

    # Non-fatal warnings (single compactions, recoverable truncation, etc.).
    warnings: list[str] = field(default_factory=list)

    # Context/token diagnostics, sourced from E2EResult.context_stats.summary().
    # Lets post-hoc analysis inspect why a scenario went catastrophic without
    # having to open per-run reports. Keys: turns, total_prompt_tokens,
    # total_completion_tokens, total_tokens, peak_prompt_tokens,
    # avg_prompt_tokens, compactions.
    context_stats: dict = field(default_factory=dict)

    # Per-turn detail, sourced from context_stats.turns. Lets us see
    # finish_reason, tool_call_names, content_preview, duration per turn
    # directly from scores.json.
    turns_detail: list[dict] = field(default_factory=list)

    # Per-attempt API-call log (from lqh.client.capture_api_metrics).
    # One entry per chat_with_retry attempt (success or failure) with
    # timing, error type, finish_reason, tokens, tool_call_count.
    api_call_log: list[dict] = field(default_factory=list)

    # Whatever the agent loop was awaiting at the moment the scenario ended.
    # Particularly useful after CancelledError / timeout: tells us whether
    # the hang was in an API call, a tool execution, or neither.
    last_operation: str | None = None

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "category": self.category,
            "model": self.model,
            "completed_without_abort": self.completed_without_abort,
            "expected_tools_called": self.expected_tools_called,
            "expected_files_exist": self.expected_files_exist,
            "artifact_judge_score": self.artifact_judge_score,
            "composite_score": self.composite_score,
            "is_catastrophic_failure": self.is_catastrophic_failure,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_tool_calls": self.total_tool_calls,
            "total_turns": self.total_turns,
            "category_details": self.category_details,
            "judge_results": self.judge_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "context_stats": self.context_stats,
            "turns_detail": self.turns_detail,
            "api_call_log": self.api_call_log,
            "last_operation": self.last_operation,
        }


# ---------------------------------------------------------------------------
# Category-specific scoring functions
# ---------------------------------------------------------------------------

async def _judge_transcript(
    client: AsyncOpenAI,
    transcript_text: str,
    criteria: str,
) -> int:
    """Run LLM judge on a transcript for a specific criterion. Returns score 1-10."""
    try:
        response = await client.chat.completions.create(
            model="judge:medium",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluator judging an AI agent's behavior in a conversation. "
                        "First write your reasoning (2-3 sentences), then give a score 1-10. "
                        "Output JSON with keys: reasoning, score."
                    ),
                },
                {
                    "role": "user",
                    "content": f"## Criteria\n\n{criteria}\n\n## Transcript\n\n{transcript_text[:6000]}",
                },
            ],
            temperature=0.0,
            response_format=JUDGE_SCHEMA,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        score, _reasoning = _coerce_judge_payload(data)
        return score
    except Exception as e:
        logger.error("Transcript judge failed: %s", e)
        return 0


def _transcript_text(result: E2EResult) -> str:
    """Convert transcript to readable text for judge input."""
    lines = []
    for rec in result.transcript:
        if rec.role == "user":
            lines.append(f"[User] {rec.content}")
        elif rec.role == "agent":
            lines.append(f"[Agent] {rec.content[:500]}")
        elif rec.role == "tool_call":
            lines.append(f"[Tool Call] {rec.tool_name}({json.dumps(rec.tool_args or {})[:200]})")
        elif rec.role == "tool_result":
            lines.append(f"[Tool Result] ({rec.tool_name}) {rec.content[:200]}")
        elif rec.role == "ask_user_q":
            lines.append(f"[Agent Question] {rec.content}")
        elif rec.role == "ask_user_a":
            lines.append(f"[User Answer] {rec.content}")
        elif rec.role == "skill_loaded":
            lines.append(f"[Skill Loaded] {rec.content}")
    return "\n".join(lines)


async def score_spec_capture(
    result: E2EResult,
    scenario: Scenario,
    client: AsyncOpenAI,
    judge_results: list[JudgeResult],
) -> BenchmarkScore:
    """Score a spec capture benchmark run.

    Tiered evaluation of the capture process:

    - **Ideal**: guided capture via ``ask_user`` with non-redundant questions,
      spec completed only after sufficient dimensions are covered.
    - **Acceptable but suboptimal**: open-ended chat questions (model asked but
      did not use the structured ``ask_user`` tool). Not catastrophic, but
      penalised relative to guided capture because UX/structured output is worse.
    - **Catastrophic**: (a) spec was never created, (b) spec was created
      prematurely (<3 questions of any kind asked), or (c) the agent asked
      redundant/bad questions (re-asks information already stated by the user).
    """
    artifacts = result.artifacts
    spec_created = "SPEC.md" in artifacts
    ask_user_turns = [t for t in result.transcript if t.role == "ask_user_q"]
    chat_q_turns = [t for t in result.transcript if t.role == "chat_q"]
    ask_user_count = len(ask_user_turns)
    chat_question_count = len(chat_q_turns)
    total_questions = ask_user_count + chat_question_count

    # "Already detailed" escape: if the user pasted a full spec in the
    # initial message, the skill explicitly permits only 2-3 confirmation
    # questions before creating SPEC.md. The scoring would otherwise flag
    # this valid shortcut as "premature_conclusion" and tank the composite.
    # Heuristic: initial user message with >= 150 words signals a pasted
    # full spec. (Typical ambiguous scenarios open with 5-15 words.)
    initial_user_msg = next(
        (t.content for t in result.transcript if t.role == "user"),
        "",
    )
    initial_word_count = len(initial_user_msg.split())
    user_pasted_full_spec = initial_word_count >= 150
    # Premature only if the user did NOT already provide the spec upfront.
    premature_conclusion = (
        total_questions < 3 and spec_created and not user_pasted_full_spec
    )

    # Guided ratio: 1.0 = fully guided (ask_user), 0.0 = all open-ended chat.
    guided_ratio = ask_user_count / total_questions if total_questions > 0 else 0.0
    # Turn ratio into a 0-100 score with a 60 floor so chat-only is "okay-ish":
    #   all-guided    -> 100
    #   50/50 mix     -> 80
    #   all-chat      -> 60
    guided_score = 60 + 40 * guided_ratio if total_questions > 0 else 0

    # Judge question quality (redundancy / overall clarity). This governs the
    # "redundant or bad questions" catastrophic flag.
    question_quality = 0
    if total_questions > 0:
        question_quality = await _judge_transcript(
            client,
            _transcript_text(result),
            "Rate the quality of the agent's clarifying questions to the user (1-10). "
            "Good questions: specific, non-redundant, cover different aspects of the task. "
            "Bad questions: vague, repetitive, asking about things the user already stated earlier, "
            "or asking for the same information in different forms. "
            "10 = excellent targeted non-redundant questions, "
            "5 = mix of useful and redundant, "
            "1 = mostly redundant or re-asking stated info.",
        )

    avg_judge = sum(jr.score for jr in judge_results) / len(judge_results) if judge_results else 0

    # Catastrophic conditions
    bad_questions = total_questions > 0 and question_quality <= 3
    is_catastrophic = (not spec_created) or premature_conclusion or bad_questions

    composite = (
        0.35 * (avg_judge / 10 * 100)               # SPEC.md quality
        + 0.25 * (question_quality / 10 * 100)      # non-redundant questions
        + 0.20 * guided_score                        # guided vs chat preference
        + 0.20 * (0 if premature_conclusion else 100)
    )

    return BenchmarkScore(
        scenario_name=scenario.name,
        category="spec_capture",
        model=result.orchestration_model,
        completed_without_abort=not result.has_errors(),
        expected_tools_called=set(scenario.expected_tools).issubset(result.tools_called()),
        expected_files_exist=all(f in artifacts for f in scenario.expected_files),
        artifact_judge_score=avg_judge,
        composite_score=round(composite, 1),
        is_catastrophic_failure=is_catastrophic,
        duration_seconds=result.duration_seconds,
        total_tool_calls=result.total_tool_calls,
        total_turns=result.total_turns,
        category_details={
            "ask_user_count": ask_user_count,
            "chat_question_count": chat_question_count,
            "total_questions": total_questions,
            "guided_ratio": round(guided_ratio, 2),
            "guided_score": round(guided_score, 1),
            "question_quality_score": question_quality,
            "bad_questions": bad_questions,
            "premature_conclusion": premature_conclusion,
            "spec_created": spec_created,
            "initial_word_count": initial_word_count,
            "user_pasted_full_spec": user_pasted_full_spec,
        },
        judge_results=[{"artifact": jr.artifact, "score": jr.score, "reasoning": jr.reasoning} for jr in judge_results],
    )


async def score_spec_generation(
    result: E2EResult,
    scenario: Scenario,
    client: AsyncOpenAI,
    judge_results: list[JudgeResult],
) -> BenchmarkScore:
    """Score a spec generation benchmark run."""
    artifacts = result.artifacts
    spec_created = "SPEC.md" in artifacts

    # Check requirements recall via judge
    planted_requirements = scenario.judge_criteria  # criteria contains the planted requirements list
    requirements_recall = 0
    if spec_created:
        requirements_recall = await _judge_transcript(
            client,
            f"## Generated SPEC.md\n\n{artifacts.get('SPEC.md', '')[:4000]}",
            f"Rate how many of the following requirements are captured in the SPEC.md (1-10). "
            f"Each requirement should be explicitly mentioned or clearly implied.\n\n"
            f"Requirements to check:\n{planted_requirements}\n\n"
            f"10 = all requirements captured, 5 = about half missing, 1 = most missing.",
        )

    avg_judge = sum(jr.score for jr in judge_results) / len(judge_results) if judge_results else 0

    composite = (
        0.5 * (requirements_recall / 10 * 100)
        + 0.5 * (avg_judge / 10 * 100)
    )

    is_catastrophic = not spec_created or requirements_recall <= 3

    return BenchmarkScore(
        scenario_name=scenario.name,
        category="spec_generation",
        model=result.orchestration_model,
        completed_without_abort=not result.has_errors(),
        expected_tools_called=set(scenario.expected_tools).issubset(result.tools_called()),
        expected_files_exist=all(f in artifacts for f in scenario.expected_files),
        artifact_judge_score=avg_judge,
        composite_score=round(composite, 1),
        is_catastrophic_failure=is_catastrophic,
        duration_seconds=result.duration_seconds,
        total_tool_calls=result.total_tool_calls,
        total_turns=result.total_turns,
        category_details={
            "spec_created": spec_created,
            "requirements_recall_score": requirements_recall,
        },
        judge_results=[{"artifact": jr.artifact, "score": jr.score, "reasoning": jr.reasoning} for jr in judge_results],
    )


def _read_first_parquet_rows(parquet_path: Path, n: int = 3) -> list[str]:
    """Return first n rows from a parquet's ``messages`` column as JSON strings.

    Returns an empty list on failure; never raises.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        if "messages" not in table.column_names:
            return []
        col = table.column("messages").to_pylist()
        return [c for c in col[:n] if c]
    except Exception:
        return []


async def _judge_pipeline_and_samples(
    client: AsyncOpenAI,
    scenario: Scenario,
    pipeline_code: str,
    samples: list[str],
) -> tuple[int, str]:
    """Category-specific judge for datagen_pipeline runs.

    Unlike the generic judge_artifacts (which only sees .md/.txt files),
    this loads the actual pipeline .py file plus up to 3 generated sample
    rows and asks the judge whether the pipeline implements the task
    described in the scenario's judge_criteria.

    Returns (score, reasoning).
    """
    PIPELINE_MAX_CHARS = 12_000
    SAMPLE_MAX_CHARS = 2_500
    try:
        # Render samples with a clear truncation marker so the judge never
        # mistakes our display-side cut for a broken artifact.
        sample_blocks: list[str] = []
        for i, s in enumerate(samples):
            shown = s[:SAMPLE_MAX_CHARS]
            note = ""
            if len(s) > SAMPLE_MAX_CHARS:
                note = (
                    f"\n[NOTE: only the first {SAMPLE_MAX_CHARS} of "
                    f"{len(s)} characters are shown — the remainder is "
                    "truncated for display only; the actual stored sample is complete.]"
                )
            sample_blocks.append(f"### Sample {i + 1}\n```json\n{shown}\n```{note}")
        samples_block = "\n\n".join(sample_blocks) or "_No samples were generated._"

        # Same treatment for the pipeline code. Without this note, previous
        # runs got 1/10 from the judge because it flagged OUR 6000-char
        # cut as a syntax error in the code.
        code_shown = pipeline_code[:PIPELINE_MAX_CHARS]
        code_note = ""
        if len(pipeline_code) > PIPELINE_MAX_CHARS:
            code_note = (
                f"\n\n[NOTE: only the first {PIPELINE_MAX_CHARS} of "
                f"{len(pipeline_code)} characters of the pipeline are "
                "shown for display — the actual file is complete. "
                "Do not score down for apparent mid-statement truncation at the end of this block.]"
            )

        user_content = (
            f"## Task Description\n\n{scenario.judge_criteria}\n\n"
            f"## Generated Pipeline Code\n\n```python\n{code_shown}\n```{code_note}\n\n"
            f"## First Generated Samples\n\n{samples_block}\n\n"
            f"Score how well the pipeline + samples together fulfil the task. "
            f"Judge:\n"
            f"- Pipeline structure (imports from `lqh.pipeline`, one `Pipeline` subclass, correct client usage)\n"
            f"- Pipeline logic matches the task\n"
            f"- Generated samples look like valid training examples for the task\n\n"
            f"IMPORTANT: Only score down for code problems you can *confirm* in the code shown. "
            f"If the display block ends mid-expression or mid-string, that is because we truncated "
            f"it for display — the actual file is intact and passed a local syntax check."
        )
        response = await client.chat.completions.create(
            model="judge:medium",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are judging a data-generation pipeline implementation. "
                        "Score 1-10 where 10 = correct pipeline AND good samples; "
                        "5 = runs but poor quality; 1 = broken or wrong task. "
                        "Output JSON with keys: reasoning, score."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format=JUDGE_SCHEMA,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return _coerce_judge_payload(data)
    except Exception as exc:
        logger.error("Pipeline+samples judge failed: %s", exc)
        return 0, f"Judge error: {exc}"


async def score_datagen_pipeline(
    result: E2EResult,
    scenario: Scenario,
    client: AsyncOpenAI,
    judge_results: list[JudgeResult],  # Ignored: we run our own judge.
) -> BenchmarkScore:
    """Score a data generation pipeline benchmark run.

    Runs its OWN judge call over (pipeline code + first 3 generated samples)
    instead of relying on the generic artifact judge, which only sees .md/.txt
    files and therefore scored the wrong artifacts (SPEC.md or a scorer file)
    against pipeline-code criteria.
    """
    artifacts = result.artifacts
    pa = result.pipeline_attempts

    # Find the pipeline file
    pipeline_files = [p for p in artifacts if p.startswith("data_gen/") and p.endswith(".py")]
    pipeline_created = len(pipeline_files) > 0

    syntax_valid = False
    imports_correct = False
    has_pipeline_subclass = False
    pipeline_code = ""

    if pipeline_created:
        pipeline_code = artifacts[pipeline_files[0]]
        try:
            ast.parse(pipeline_code)
            syntax_valid = True
        except SyntaxError:
            pass
        imports_correct = "from lqh.pipeline import" in pipeline_code
        has_pipeline_subclass = "Pipeline)" in pipeline_code or "Pipeline):" in pipeline_code

    pipeline_ran = pa["succeeded"] > 0

    # Locate a generated dataset and pull the first 3 rows for judge input
    dataset_parquets = sorted(
        result.project_dir.glob("datasets/*/data.parquet")
    )
    samples_generated = len(dataset_parquets) > 0
    sample_strs: list[str] = []
    if dataset_parquets:
        sample_strs = _read_first_parquet_rows(dataset_parquets[0], n=3)

    # Category-specific judge (replaces generic judge_artifacts for this run)
    category_judge_score = 0
    category_judge_reasoning = ""
    if pipeline_created:
        category_judge_score, category_judge_reasoning = await _judge_pipeline_and_samples(
            client, scenario, pipeline_code, sample_strs,
        )

    # Composite: 50% ran successfully + 30% local structure checks + 20% judge
    structure_score = (
        (25 if syntax_valid else 0)
        + (25 if imports_correct else 0)
        + (25 if has_pipeline_subclass else 0)
        + (25 if pipeline_created else 0)
    )

    composite = (
        0.5 * (100 if pipeline_ran else 0)
        + 0.3 * structure_score
        + 0.2 * (category_judge_score / 10 * 100)
    )

    # Build a judge_results entry exposing the category-specific judge result
    judge_results_out = []
    if pipeline_created:
        judge_results_out.append({
            "artifact": pipeline_files[0] + (f" + {len(sample_strs)} samples" if sample_strs else ""),
            "score": category_judge_score,
            "reasoning": category_judge_reasoning,
        })

    return BenchmarkScore(
        scenario_name=scenario.name,
        category="datagen_pipeline",
        model=result.orchestration_model,
        completed_without_abort=not result.has_errors(),
        expected_tools_called=set(scenario.expected_tools).issubset(result.tools_called()),
        expected_files_exist=all(f in artifacts for f in scenario.expected_files),
        artifact_judge_score=category_judge_score,
        composite_score=round(composite, 1),
        is_catastrophic_failure=not pipeline_ran,
        duration_seconds=result.duration_seconds,
        total_tool_calls=result.total_tool_calls,
        total_turns=result.total_turns,
        category_details={
            "pipeline_created": pipeline_created,
            "syntax_valid": syntax_valid,
            "imports_correct": imports_correct,
            "has_pipeline_subclass": has_pipeline_subclass,
            "pipeline_ran_successfully": pipeline_ran,
            "pipeline_attempts_total": pa["total"],
            "pipeline_attempts_failed": pa["failed"],
            "samples_generated": samples_generated,
            "samples_judged": len(sample_strs),
        },
        judge_results=judge_results_out,
    )


async def score_error_recovery(
    result: E2EResult,
    scenario: Scenario,
    client: AsyncOpenAI,
    judge_results: list[JudgeResult],
) -> BenchmarkScore:
    """Score an error recovery benchmark run."""
    pa = result.pipeline_attempts
    fixed = pa["succeeded"] > 0

    # Count edit_file calls as fix attempts
    edit_calls = [t for t in result.transcript if t.role == "tool_call" and t.tool_name == "edit_file"]
    fix_attempts = len(edit_calls)

    # Judge: did the agent correctly diagnose the issue?
    diagnosis_score = await _judge_transcript(
        client,
        _transcript_text(result),
        "Rate whether the agent correctly identified the root cause of the pipeline failure (1-10). "
        "10 = identified exact bug and explained it clearly, "
        "5 = partially identified the issue, "
        "1 = never understood what was wrong or made random changes.",
    )

    # Fewer attempts is better (normalize: 1 attempt = 100, 5+ = 0)
    attempt_score = max(0, 100 - (fix_attempts - 1) * 25) if fixed else 0

    composite = (
        0.6 * (100 if fixed else 0)
        + 0.2 * attempt_score
        + 0.2 * (diagnosis_score / 10 * 100)
    )

    return BenchmarkScore(
        scenario_name=scenario.name,
        category="error_recovery",
        model=result.orchestration_model,
        completed_without_abort=not result.has_errors(),
        expected_tools_called=set(scenario.expected_tools).issubset(result.tools_called()),
        expected_files_exist=all(f in artifacts for f in scenario.expected_files) if (artifacts := result.artifacts) else False,
        artifact_judge_score=0,  # No artifact judging for error recovery
        composite_score=round(composite, 1),
        is_catastrophic_failure=not fixed,
        duration_seconds=result.duration_seconds,
        total_tool_calls=result.total_tool_calls,
        total_turns=result.total_turns,
        category_details={
            "fixed_successfully": fixed,
            "fix_attempts": fix_attempts,
            "diagnosis_score": diagnosis_score,
            "pipeline_attempts_total": pa["total"],
            "pipeline_attempts_failed": pa["failed"],
        },
        judge_results=[],
    )


def _detect_next_step(result: E2EResult, expected: str) -> tuple[str, list[str]]:
    """Detect which next-step bucket the agent landed in.

    Returns (actual, signals) where signals is a list of evidence strings for debugging.

    Detection buckets and their signals (any one triggers a match):
    - data_generation: run_data_gen_pipeline; files under data_gen/, evals/scorers/, datasets/;
      load_skill(data_generation|data_validation|data_filtering); skill loaded
    - evaluation: run_scoring, start_local_eval; files under evals/runs/; load_skill(evaluation); skill loaded
    - prompt_optimization: files under prompts/; load_skill(prompt_optimization); skill loaded
    - train: start_training; load_skill(train); skill loaded
    """

    def _args_str(rec: Any) -> str:
        return json.dumps(rec.tool_args or {}, ensure_ascii=False)

    matches: dict[str, list[str]] = {
        "data_generation": [],
        "evaluation": [],
        "prompt_optimization": [],
        "train": [],
    }

    # Skills loaded (any, not just first)
    skill_to_bucket = {
        "data_generation": "data_generation",
        "data_validation": "data_generation",
        "data_filtering": "data_generation",
        "evaluation": "evaluation",
        "prompt_optimization": "prompt_optimization",
        "train": "train",
    }
    for skill in result.skills_loaded:
        bucket = skill_to_bucket.get(skill)
        if bucket:
            matches[bucket].append(f"skill_loaded:{skill}")

    # Tool calls
    file_tool_names = {"create_file", "write_file", "edit_file"}
    for rec in result.transcript:
        if rec.role != "tool_call":
            continue
        name = rec.tool_name
        args = _args_str(rec)

        if name == "run_data_gen_pipeline":
            matches["data_generation"].append("run_data_gen_pipeline")
        elif name in ("run_scoring", "start_local_eval"):
            matches["evaluation"].append(name)
        elif name == "start_training":
            matches["train"].append("start_training")
        elif name == "load_skill":
            skill = (rec.tool_args or {}).get("skill_name", "")
            bucket = skill_to_bucket.get(skill)
            if bucket:
                matches[bucket].append(f"load_skill:{skill}")
        elif name in file_tool_names:
            # Map target path prefix to bucket.
            if any(p in args for p in ("data_gen/", "evals/scorers/", "datasets/")):
                matches["data_generation"].append(f"{name}:datagen_path")
            elif "evals/runs/" in args:
                matches["evaluation"].append(f"{name}:evals/runs/")
            elif "prompts/" in args:
                matches["prompt_optimization"].append(f"{name}:prompts/")

    # Prefer the expected bucket when it has signals; otherwise pick the bucket
    # with the most signals; otherwise unknown.
    if matches.get(expected):
        return expected, matches[expected]
    ranked = sorted(matches.items(), key=lambda kv: -len(kv[1]))
    if ranked and ranked[0][1]:
        return ranked[0][0], ranked[0][1]
    return "unknown", []


async def score_next_steps(
    result: E2EResult,
    scenario: Scenario,
    client: AsyncOpenAI,
    judge_results: list[JudgeResult],
) -> BenchmarkScore:
    """Score a next steps benchmark run."""
    # Determine what the agent actually did
    expected = scenario.judge_criteria  # Expected step encoded in judge_criteria for this category

    actual, signals = _detect_next_step(result, expected)

    # If no deterministic signals, fall back to LLM judge: did the agent attempt `expected`?
    judge_fallback_used = False
    judge_fallback_says_correct = False
    if actual == "unknown":
        judge_fallback_used = True
        fallback_score = await _judge_transcript(
            client,
            _transcript_text(result),
            f"The correct next step in this workflow is: '{expected}'. "
            f"Rate whether the agent correctly attempted or recommended this specific next step (1-10). "
            f"10 = clearly attempted {expected}, 6 = verbally recommended it but did not act, "
            f"1 = did something unrelated or wrong.",
        )
        if fallback_score >= 7:
            actual = expected
            judge_fallback_says_correct = True

    # Ambiguity escape for "post prompt-opt" state: if the expected step is
    # 'train' but the agent chose 'data_generation' with a modest sample count
    # (<=1000), accept that as a legitimate "scale up training data before
    # kicking off training" step rather than penalising it as wrong. This
    # matches reasonable production behaviour where 200 seeded samples is
    # borderline for SFT.  Only a massive unchecked generation counts as
    # wrong (agent skipping ahead without evaluating first).
    scale_up_instead_of_train = False
    if expected == "train" and actual == "data_generation":
        max_samples_requested = 0
        for rec in result.transcript:
            if rec.role == "tool_call" and rec.tool_name == "run_data_gen_pipeline":
                n = (rec.tool_args or {}).get("num_samples", 0)
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    n = 0
                if n > max_samples_requested:
                    max_samples_requested = n
        if 0 < max_samples_requested <= 1000:
            actual = expected  # accept as correct
            scale_up_instead_of_train = True

    correct = actual == expected

    # Judge rationale quality
    rationale_score = await _judge_transcript(
        client,
        _transcript_text(result),
        "Rate whether the agent provided a clear rationale for the next step it chose (1-10). "
        "Did it explain why this is the right next step given the current project state? "
        "10 = clear reasoning about project state, 5 = just picked an action without explanation, "
        "1 = confused about project state or chose wrong action.",
    )

    composite = (
        0.7 * (100 if correct else 0)
        + 0.3 * (rationale_score / 10 * 100)
    )

    return BenchmarkScore(
        scenario_name=scenario.name,
        category="next_steps",
        model=result.orchestration_model,
        completed_without_abort=not result.has_errors(),
        expected_tools_called=True,  # Not applicable
        expected_files_exist=True,  # Not applicable
        artifact_judge_score=0,
        composite_score=round(composite, 1),
        is_catastrophic_failure=not correct,
        duration_seconds=result.duration_seconds,
        total_tool_calls=result.total_tool_calls,
        total_turns=result.total_turns,
        category_details={
            "expected_next_step": expected,
            "actual_next_step": actual,
            "correct": correct,
            "rationale_score": rationale_score,
            "detection_signals": signals,
            "judge_fallback_used": judge_fallback_used,
            "judge_fallback_says_correct": judge_fallback_says_correct,
            "scale_up_instead_of_train": scale_up_instead_of_train,
        },
        judge_results=[],
    )


async def score_edit(
    result: E2EResult,
    scenario: Scenario,
    client: AsyncOpenAI,
    judge_results: list[JudgeResult],
) -> BenchmarkScore:
    """Score an edit spec/pipeline benchmark run."""
    artifacts = result.artifacts
    pa = result.pipeline_attempts

    # Check if spec was updated (edit_file or write_file on SPEC.md)
    spec_updated = any(
        t.role == "tool_call"
        and t.tool_name in ("edit_file", "write_file")
        and "SPEC.md" in json.dumps(t.tool_args or {})
        for t in result.transcript
    )

    # Check if pipeline was updated
    pipeline_updated = any(
        t.role == "tool_call"
        and t.tool_name in ("edit_file", "write_file")
        and "data_gen/" in json.dumps(t.tool_args or {})
        for t in result.transcript
    )

    # Pipeline still runs after edit
    pipeline_still_runs = pa["succeeded"] > 0 if pa["total"] > 0 else True

    # Check for unnecessary recreation (created new v2 files instead of editing)
    unnecessary_recreation = any(
        t.role == "tool_call"
        and t.tool_name == "create_file"
        and ("_v2" in json.dumps(t.tool_args or {}) or "new_" in json.dumps(t.tool_args or {}))
        for t in result.transcript
    )

    avg_judge = sum(jr.score for jr in judge_results) / len(judge_results) if judge_results else 0

    # Scenario-aware weighting: edit_spec_* scenarios target SPEC.md;
    # edit_pipeline_* scenarios target the pipeline file only.
    edit_target = "spec" if scenario.name.startswith("bench_edit_spec_") else "pipeline"

    if edit_target == "spec":
        composite = (
            0.4 * (100 if spec_updated else 0)
            + 0.4 * (100 if pipeline_still_runs else 0)
            + 0.2 * (0 if unnecessary_recreation else 100)
        )
    else:
        composite = (
            0.5 * (100 if pipeline_updated else 0)
            + 0.4 * (100 if pipeline_still_runs else 0)
            + 0.1 * (0 if unnecessary_recreation else 100)
        )

    return BenchmarkScore(
        scenario_name=scenario.name,
        category="edit",
        model=result.orchestration_model,
        completed_without_abort=not result.has_errors(),
        expected_tools_called=set(scenario.expected_tools).issubset(result.tools_called()),
        expected_files_exist=all(f in artifacts for f in scenario.expected_files),
        artifact_judge_score=avg_judge,
        composite_score=round(composite, 1),
        is_catastrophic_failure=not pipeline_still_runs,
        duration_seconds=result.duration_seconds,
        total_tool_calls=result.total_tool_calls,
        total_turns=result.total_turns,
        category_details={
            "edit_target": edit_target,
            "spec_updated": spec_updated,
            "pipeline_updated": pipeline_updated,
            "pipeline_still_runs": pipeline_still_runs,
            "unnecessary_recreation": unnecessary_recreation,
        },
        judge_results=[{"artifact": jr.artifact, "score": jr.score, "reasoning": jr.reasoning} for jr in judge_results],
    )


async def score_context_management(
    result: E2EResult,
    scenario: Scenario,
    client: AsyncOpenAI,
    judge_results: list[JudgeResult],
) -> BenchmarkScore:
    """Score a context management benchmark run."""
    cs = result.context_stats
    compaction_triggered = any("compaction" in e.lower() for e in result.errors)

    # Peak tokens
    peak_tokens = 0
    if cs and cs.turns:
        summary = cs.summary()
        peak_tokens = summary.get("peak_prompt_tokens", 0)

    # Judge coherence at end of conversation
    coherence_score = await _judge_transcript(
        client,
        _transcript_text(result),
        "Rate whether the agent remained coherent and aware of the task throughout the conversation (1-10). "
        "Check: does it repeat questions already answered? Does it lose track of what was discussed? "
        "Does it still understand the original task at the end? "
        "10 = perfectly coherent throughout, 5 = some confusion, 1 = lost track of task.",
    )

    # Token efficiency: penalize if peak tokens > 150k (close to 200k limit)
    token_efficiency = 100
    if peak_tokens > 150000:
        token_efficiency = max(0, 100 - (peak_tokens - 150000) / 500)

    composite = (
        0.4 * (0 if compaction_triggered else 100)
        + 0.3 * token_efficiency
        + 0.3 * (coherence_score / 10 * 100)
    )

    return BenchmarkScore(
        scenario_name=scenario.name,
        category="context_management",
        model=result.orchestration_model,
        completed_without_abort=not result.has_errors(),
        expected_tools_called=True,
        expected_files_exist=True,
        artifact_judge_score=0,
        composite_score=round(composite, 1),
        is_catastrophic_failure=compaction_triggered,
        duration_seconds=result.duration_seconds,
        total_tool_calls=result.total_tool_calls,
        total_turns=result.total_turns,
        category_details={
            "compaction_triggered": compaction_triggered,
            "peak_tokens": peak_tokens,
            "coherence_score": coherence_score,
            "token_efficiency": round(token_efficiency, 1),
        },
        judge_results=[],
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CATEGORY_SCORERS = {
    "spec_capture": score_spec_capture,
    "spec_generation": score_spec_generation,
    "datagen_pipeline": score_datagen_pipeline,
    "error_recovery": score_error_recovery,
    "next_steps": score_next_steps,
    "edit": score_edit,
    "context_management": score_context_management,
}


async def score_result(
    result: E2EResult,
    scenario: Scenario,
    category: str,
    client: AsyncOpenAI,
) -> BenchmarkScore:
    """Score an E2E result using the appropriate category scorer."""
    # Run artifact judge. Exclude files that were pre-seeded by the scenario
    # (they are not the agent's work and shouldn't be judged against the
    # agent's task criteria — e.g., a seeded SPEC.md in a datagen scenario
    # where the agent's task is to write a pipeline).
    artifacts = result.artifacts
    jr = await judge_artifacts(
        client, scenario, artifacts,
        exclude_paths=result.seeded_files,
    )

    scorer = CATEGORY_SCORERS[category]
    score = await scorer(result, scenario, client, jr)
    # Attach raw errors (seed crashes, aborts, timeouts, tool errors) and any
    # non-fatal warnings (e.g. single compaction at skill boundary) for later
    # debugging and aggregate reporting.
    score.errors = list(result.errors)
    score.warnings = list(getattr(result, "warnings", []))
    # Attach per-run context/token diagnostics. Surfaces peak prompt tokens,
    # compaction count, and total tokens directly in scores.json so we can
    # answer questions like "did this run really blow up context, or did
    # compaction fire spuriously on cumulative-vs-current token counts?"
    if result.context_stats and result.context_stats.turns:
        score.context_stats = result.context_stats.summary()
        # Per-turn detail: finish_reason / tool names / content preview /
        # duration so mysterious hangs can be diagnosed from scores.json alone.
        score.turns_detail = [
            {
                "turn": t.turn_number,
                "prompt_tokens": t.prompt_tokens,
                "completion_tokens": t.completion_tokens,
                "finish_reason": t.finish_reason,
                "tool_call_names": list(t.tool_call_names),
                "tool_call_args": list(getattr(t, "tool_call_args", [])),
                "content_length": t.content_length,
                "content_preview": t.content_preview,
                "duration_s": t.duration_s,
                "compacted": t.compacted,
                "skill_active": t.skill_active,
            }
            for t in result.context_stats.turns
        ]
    score.api_call_log = list(getattr(result, "api_call_log", []))
    score.last_operation = getattr(result, "last_operation", None)
    return score
