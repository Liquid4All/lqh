"""Task catalog for the base-vs-instruct benchmark.

A ``Task`` ties together:
  - the datagen pipeline (loaded by file path by the engine),
  - the judge scorer (a 0-10 rubric, written to the workdir and consumed by
    ``lqh.scoring.run_scoring``),
  - the system prompt (baked into the ChatML by the pipeline; mirrored here
    only for reference/reporting — it is NOT re-applied at eval time).

The system prompt lives in each pipeline module (it is baked into every
generated row, which is what train/eval/score all read). We import it here so
there is a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .pipelines import classification as _classification
from .pipelines import extraction as _extraction
from .pipelines import messy_extraction as _messy_extraction
from .pipelines import style_rewrite as _style_rewrite
from .pipelines import translation as _translation
from .pipelines import voice_satisfaction as _voice_satisfaction

_PIPELINE_DIR = Path(__file__).parent / "pipelines"


@dataclass(frozen=True)
class Task:
    name: str
    pipeline_path: Path
    system_prompt: str
    scorer_md: str


# --- scorers -------------------------------------------------------------
# Each returns JSON `{"score": <int 0-10>, "reasoning": "<one sentence>"}`.
# run_scoring strips the trailing assistant turn at inference time; at scoring
# time the assistant turn IS the model's prediction, judged against the user
# message + this rubric.

_TRANSLATION_SCORER = """\
# English→German translation scorer

The user gave an English sentence. The assistant should output ONLY the German
translation — no preamble, no surrounding quotes, no English, no commentary.

Score 0-10 on BOTH meaning fidelity and format discipline:

- 10: Accurate, fluent German AND nothing but the German (no extras).
- 8-9: Accurate German with only a tiny format slip (a trailing period, an
       outer quote).
- 5-7: Correct German but wrapped in chatty prose ("Sure, in German that's
       '...'.") or with leftover English preamble.
- 3-4: German present but meaning is wrong/partial, or heavy extra text.
- 0-2: Empty, still English, a refusal, or unrelated.

Return JSON exactly: `{"score": <int 0-10>, "reasoning": "<one sentence>"}`.
"""

_EXTRACTION_SCORER = """\
# Event-extraction scorer

The user gave a free-text event invitation. The assistant should output ONLY a
single JSON object with exactly these keys: title, date, time, location,
organizer — each value taken from the message.

Score 0-10:

- 10: Valid JSON, exactly those five keys, every value correct, NOTHING else
      (no code fences, no prose).
- 8-9: All values correct but a minor format issue (code fence, extra
       whitespace, key reordering only).
- 5-7: Valid JSON but one field wrong/missing, or wrapped in commentary.
- 3-4: Malformed JSON or several fields wrong.
- 0-2: Not JSON, empty, a refusal, or unrelated.

Return JSON exactly: `{"score": <int 0-10>, "reasoning": "<one sentence>"}`.
"""

_CLASSIFICATION_SCORER = """\
# Sentiment-classification scorer

The user gave a short review. The assistant should output ONLY a JSON object
{"sentiment": "<label>"} where <label> is exactly one of positive, negative,
neutral, and matches the review's overall sentiment.

Score 0-10:

- 10: Valid JSON, single "sentiment" key, correct label, nothing else.
- 8-9: Correct label but a minor format slip (code fence, extra whitespace).
- 5-7: Correct label buried in prose, OR valid JSON with a borderline-but-
       defensible label on a genuinely mixed review.
- 3-4: Wrong label, or malformed JSON.
- 0-2: Not JSON / no label, empty, a refusal, or unrelated.

Judge the label against the review's actual sentiment, not against any label
word in the text. Return JSON exactly:
`{"score": <int 0-10>, "reasoning": "<one sentence>"}`.
"""

_MESSY_EXTRACTION_SCORER = """\
# Messy support-thread extraction scorer

The user gave a noisy support-thread excerpt containing stale values,
corrections, and distractors. The assistant should output ONLY a single JSON
object with exactly these keys: customer_name, account_id, requested_action,
effective_date, priority, owner. Values must reflect the latest corrected
request, not earlier superseded values. Missing values should be null.

Score 0-10:

- 10: Valid JSON, exactly those six keys, all latest values correct, no prose.
- 8-9: All latest values correct with only a minor format issue.
- 5-7: Valid JSON but one or two fields are stale, missing, or wrong.
- 3-4: Malformed JSON or several fields wrong/stale.
- 0-2: Not JSON, empty, refusal, or unrelated.

Return JSON exactly: `{"score": <int 0-10>, "reasoning": "<one sentence>"}`.
"""

_STYLE_REWRITE_SCORER = """\
# Tone/style rewrite scorer

The user gave a flawed support reply, a requested tone/style, required facts,
banned phrases, and length/style constraints. The assistant should output ONLY
the rewritten reply.

Score 0-10 on all of: fact preservation, requested tone, removal of banned
phrases, concision, and natural support-writing quality.

- 10: Preserves every required fact, removes every banned phrase, matches the
      requested tone, obeys length/style constraints, and reads naturally.
- 8-9: Strong rewrite with only a small tone, length, or phrasing issue.
- 5-7: Preserves most facts but is generic, too wordy, slightly off-tone, or
       keeps one minor forbidden-style artifact.
- 3-4: Drops/changes important facts, keeps banned phrases, or ignores tone.
- 0-2: Empty, refusal, unrelated, or mostly copies the flawed draft.

Return JSON exactly: `{"score": <int 0-10>, "reasoning": "<one sentence>"}`.
"""

_VOICE_SATISFACTION_SCORER = """\
# Voice-assistant satisfaction scorer (observability output) scorer

The user gave a voice-assistant interaction (a `User:`/`Assistant:` transcript
OR a JSON event log) and asked for a satisfaction assessment. The assistant
should output ONLY one JSON object with exactly these keys, in this order:
reasoning, score, failure_tags, success_tags, failed_turns, successful_turns —
where `score` is an integer 1 (very dissatisfied) to 5 (very satisfied).

Read the interaction YOURSELF, decide the correct assessment, then grade the
assistant's JSON against it. Score 0-10 on ALL of:

- Format: valid JSON, exactly those six keys, reasoning before score, integer
  score in 1-5, tags only from the allowed taxonomies (failure: misrecognition,
  wrong_action, no_action, noise_trigger, user_correction, incomplete_action,
  unrelated_response, repeated_failure, hallucination, canceled, wrong_entity,
  context_loss; success: success, multi_turn_success, graceful_recovery), turn
  numbers in range, no prose/code fences.
- Score correctness: matches how satisfied the user actually was (±1 tolerated;
  a 4-5 on a clearly failed interaction is not).
- Frustration sensitivity (MOST IMPORTANT): every genuine frustration signal in
  the interaction (a correction, repeat, "ugh", a noise trigger, a wrong entity,
  a hallucination) must be reflected by a low score AND a failure tag. MISSING a
  real frustration signal is the worst error.
- Tag accuracy & coupling: tags correctly describe what happened; score 1-3
  carries >=1 failure tag (unless a pure `canceled` ambiguity); score 4-5
  carries >=1 success tag.
- Reasoning quality: 2-4 sentences referencing specific turns/events, not
  generic ("the user seemed unhappy").

- 10: valid JSON, correct score, accurate tags, specific reasoning, nothing
      genuine missed.
- 8-9: correct assessment with a minor tag or reasoning slip.
- 5-7: defensible score but a wrong/missing tag, an off-by-one turn, or vague
       reasoning.
- 3-4: wrong score direction, malformed JSON, or a tag that contradicts the
       interaction.
- 0-2: MISSED a genuine frustration signal (rated a frustrated user as
       satisfied), not JSON, empty, a refusal, or wrong field structure.

Return JSON exactly: `{"score": <int 0-10>, "reasoning": "<one sentence>"}`.
"""


ALL_TASKS: dict[str, Task] = {
    "translation": Task(
        name="translation",
        pipeline_path=_PIPELINE_DIR / "translation.py",
        system_prompt=_translation.SYSTEM_PROMPT,
        scorer_md=_TRANSLATION_SCORER,
    ),
    "extraction": Task(
        name="extraction",
        pipeline_path=_PIPELINE_DIR / "extraction.py",
        system_prompt=_extraction.SYSTEM_PROMPT,
        scorer_md=_EXTRACTION_SCORER,
    ),
    "classification": Task(
        name="classification",
        pipeline_path=_PIPELINE_DIR / "classification.py",
        system_prompt=_classification.SYSTEM_PROMPT,
        scorer_md=_CLASSIFICATION_SCORER,
    ),
    "messy_extraction": Task(
        name="messy_extraction",
        pipeline_path=_PIPELINE_DIR / "messy_extraction.py",
        system_prompt=_messy_extraction.SYSTEM_PROMPT,
        scorer_md=_MESSY_EXTRACTION_SCORER,
    ),
    "style_rewrite": Task(
        name="style_rewrite",
        pipeline_path=_PIPELINE_DIR / "style_rewrite.py",
        system_prompt=_style_rewrite.SYSTEM_PROMPT,
        scorer_md=_STYLE_REWRITE_SCORER,
    ),
    "voice_satisfaction": Task(
        name="voice_satisfaction",
        pipeline_path=_PIPELINE_DIR / "voice_satisfaction.py",
        system_prompt=_voice_satisfaction.SYSTEM_PROMPT,
        scorer_md=_VOICE_SATISFACTION_SCORER,
    ),
}


def resolve_tasks(names: list[str] | None) -> list[Task]:
    """Resolve a list of task names to Task objects (all if None/empty)."""
    if not names:
        return list(ALL_TASKS.values())
    out: list[Task] = []
    for n in names:
        key = n.strip().lower()
        if key not in ALL_TASKS:
            raise SystemExit(
                f"unknown task {n!r}; choose from {', '.join(ALL_TASKS)}"
            )
        out.append(ALL_TASKS[key])
    return out
