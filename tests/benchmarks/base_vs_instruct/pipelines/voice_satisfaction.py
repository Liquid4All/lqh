"""Voice-assistant satisfaction scorer datagen (the "observability twist").

This is the benchmark's third task and the one that does NOT saturate after
SFT the way translation/extraction/classification do (>8/10 quickly), which is
what limits the DPO comparison. The model is an *observability* model: it reads
a human↔voice-assistant interaction and emits a structured satisfaction
assessment (reasoning, 1-5 score, failure/success tags, per-turn breakdown).
Scoring it well requires genuine judgement — sensitivity to frustration, correct
tag attribution, faithful reasoning — so the judge rarely hands out a 10, which
keeps headroom for DPO to move the needle.

Two input formats are generated (the task must treat them equivalently):
  - a plain ``User:``/``Assistant:`` transcript,
  - a JSON event log, either rich (timestamps/action/entity/intent/confidence)
    or minimal (role+content only).

Each sample is a 3-message ChatML conversation:
  system    -> SYSTEM_PROMPT (the scorer's instructions, baked in)
  user      -> the interaction (one of the formats above)
  assistant -> the canonical satisfaction JSON (reasoning first, then score)

The gold output is LLM-generated against a target score+tag profile and then
**validated** (``validate_output``) so a malformed or rule-violating gold never
enters train/eval — the run-level scorer filter is the second line of defence.

Derived from the standalone ``obs`` project's ``voice_satisfaction_v2``
pipeline, adapted to this benchmark's conventions (plain ``random``,
``safe_content``, ``@step`` retries, system prompt baked into the row).
"""

from __future__ import annotations

import json
import random

from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    GenerationError,
    Pipeline,
    safe_content,
    step,
)

# --- taxonomies (single source of truth, mirrored in the scorer rubric) ----

FAILURE_TAGS = {
    "misrecognition": "ASR did not transcribe the user's speech correctly (the user corrects what they said).",
    "wrong_action": "Assistant understood the request but took the wrong action.",
    "no_action": "Assistant failed to act when it should have.",
    "noise_trigger": "Assistant activated from background noise / non-speech, not an intentional command.",
    "user_correction": "User had to rephrase, repeat, or explicitly correct the assistant.",
    "incomplete_action": "Assistant partially completed the request but missed something.",
    "unrelated_response": "Assistant responded to something the user didn't ask about.",
    "repeated_failure": "The same type of error occurred multiple times in the conversation.",
    "hallucination": "Assistant made up information or claimed a capability it doesn't have.",
    "canceled": "User canceled or said 'never mind' — ambiguous between cancellation and frustration.",
    "wrong_entity": "Assistant operated on the wrong device, room, or entity.",
    "context_loss": "Assistant lost context from earlier in the conversation.",
}
SUCCESS_TAGS = {
    "success": "Request was handled correctly and completely.",
    "multi_turn_success": "A complex multi-turn request was handled correctly across all turns.",
    "graceful_recovery": "Assistant recovered well from an earlier error after user correction.",
}

_VALID_FAILURE = set(FAILURE_TAGS)
_VALID_SUCCESS = set(SUCCESS_TAGS)
_REQUIRED_FIELDS = (
    "reasoning", "score", "failure_tags", "success_tags",
    "failed_turns", "successful_turns",
)

SYSTEM_PROMPT = (
    "You are an on-device observability model that scores user satisfaction "
    "from a voice-assistant interaction. The interaction is given either as a "
    "User:/Assistant: transcript or as a JSON event log (role+content, "
    "sometimes with action/entity/intent/confidence metadata); treat both "
    "formats equivalently. Output ONLY one JSON object with exactly these keys "
    "in this order: reasoning, score, failure_tags, success_tags, "
    "failed_turns, successful_turns. 'reasoning' (2-4 sentences referencing "
    "specific turns) comes first; 'score' is an integer 1 (very dissatisfied) "
    "to 5 (very satisfied). failure_tags come from: "
    + ", ".join(_VALID_FAILURE)
    + ". success_tags come from: " + ", ".join(_VALID_SUCCESS)
    + ". failed_turns/successful_turns are 1-indexed turn numbers (a turn = one "
    "user utterance + the assistant's response). Missing a genuine user "
    "frustration signal is the worst possible error — when in doubt, flag it. "
    "Score 1-3 carries at least one failure tag (unless a pure 'canceled' "
    "ambiguity); score 4-5 carries at least one success tag. No commentary, no "
    "code fences."
)

# --- scenario / diversity knobs --------------------------------------------

_CATEGORIES = [
    "smart_home_lights", "smart_home_thermostat", "smart_home_locks",
    "smart_home_routine", "smart_home_appliance", "music_play", "music_control",
    "podcast_audiobook", "smart_tv", "navigation_route", "navigation_nearby",
    "communication_call", "communication_message", "timer_alarm",
    "reminder", "weather", "news_info", "calendar", "shopping_list",
    "car_controls", "noise_trigger", "misrecognition", "hallucination",
    "context_loss", "multi_intent", "ambiguous_request", "cancellation",
    "wrong_room", "partial_completion", "recovery_after_error",
    "repeated_same_error",
]

# Weighted toward failures: failures are the hard, high-value cases and the
# ones that keep the judge from saturating (and give DPO preference signal).
_SCORE_WEIGHTS = {1: 0.20, 2: 0.25, 3: 0.15, 4: 0.15, 5: 0.25}

_INPUT_FORMAT_WEIGHTS = {"transcript": 0.40, "event_log_rich": 0.35, "event_log_minimal": 0.25}

_USER_PERSONAS = [
    "informal, uses contractions, may trail off",
    "short, clear commands, minimal words",
    "gives extra context, specifies exactly what they want",
    "formal, less familiar with the technology, patient",
    "already annoyed, short temper, uses sarcasm",
    "polite, says please/thank you",
    "rapid-fire commands, switching tasks quickly",
    "unsure what they want, asks questions, may change their mind",
]
_ASSISTANT_PERSONAS = [
    "concise, helpful, neutral",
    "verbose, gives more information than needed",
    "chatty, adds personality and follow-up questions",
    "minimal, sometimes just 'done' or 'ok'",
]


def validate_output(data: dict, num_turns: int) -> None:
    """Raise ``GenerationError`` if *data* is not a well-formed assessment.

    First line of defence (the run-level scorer filter is the second). Enforces
    the SPEC's hard rules: exact field set, integer 1-5 score, tags from the
    allowed taxonomies, turn numbers within 1..num_turns, non-trivial reasoning,
    and the tag/score coupling (score 1-3 -> a failure tag unless a pure
    `canceled` ambiguity; score 4-5 -> a success tag).
    """
    missing = set(_REQUIRED_FIELDS) - set(data)
    if missing:
        raise GenerationError(f"missing fields: {sorted(missing)}")
    extra = set(data) - set(_REQUIRED_FIELDS)
    if extra:
        raise GenerationError(f"unexpected fields: {sorted(extra)}")

    score = data["score"]
    if not isinstance(score, int) or isinstance(score, bool) or not (1 <= score <= 5):
        raise GenerationError(f"invalid score: {score!r}")

    reasoning = data["reasoning"]
    if not isinstance(reasoning, str) or len(reasoning.split()) < 6:
        raise GenerationError("reasoning missing or too short (need a real explanation)")

    for field, allowed in (("failure_tags", _VALID_FAILURE), ("success_tags", _VALID_SUCCESS)):
        val = data[field]
        if not isinstance(val, list) or any(not isinstance(t, str) for t in val):
            raise GenerationError(f"{field} must be a list of strings")
        bad = set(val) - allowed
        if bad:
            raise GenerationError(f"invalid {field}: {sorted(bad)}")

    for field in ("failed_turns", "successful_turns"):
        val = data[field]
        if not isinstance(val, list):
            raise GenerationError(f"{field} must be a list")
        for t in val:
            if not isinstance(t, int) or isinstance(t, bool) or not (1 <= t <= num_turns):
                raise GenerationError(f"{field} has out-of-range turn {t!r} (expected 1..{num_turns})")

    # Tag/score coupling (SPEC rules 6 & 7). The only allowed neutral-score
    # exception is pure ambiguity, which the generator always encodes as a
    # `canceled` tag — so a 1-3 with no failure tag is always a violation here.
    if score <= 3 and not data["failure_tags"]:
        raise GenerationError(f"score {score} requires at least one failure tag")
    if score >= 4 and not data["success_tags"]:
        raise GenerationError(f"score {score} requires at least one success tag")


def _ordered_output(data: dict) -> str:
    """Serialize with the SPEC field order (reasoning first, score second)."""
    return json.dumps({k: data[k] for k in _REQUIRED_FIELDS}, ensure_ascii=False, indent=2)


def _extract_json(text: str):
    """Parse a JSON value from a model response, tolerating ```json fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    t = t.strip().strip("`").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"invalid JSON: {t[:200]}") from exc


class VoiceSatisfactionPipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.category = random.choice(_CATEGORIES)
        self.score = random.choices(
            list(_SCORE_WEIGHTS), weights=list(_SCORE_WEIGHTS.values()), k=1
        )[0]
        self.input_format = random.choices(
            list(_INPUT_FORMAT_WEIGHTS), weights=list(_INPUT_FORMAT_WEIGHTS.values()), k=1
        )[0]
        self.num_turns = random.randint(1, 6)
        self.user_persona = random.choice(_USER_PERSONAS)
        self.assistant_persona = random.choice(_ASSISTANT_PERSONAS)
        self.seed = f"vs-{self.category}-{random.randint(0, 1_000_000)}"
        self._pick_tags()

        await self._make_conversation(client)
        self.formatted_input = self._format_conversation()
        await self._make_output(client)

        return [
            ChatMLMessage("system", SYSTEM_PROMPT),
            ChatMLMessage(
                "user",
                "Analyze this voice-assistant interaction and score user "
                "satisfaction.\n\n" + self.formatted_input,
            ),
            ChatMLMessage("assistant", self.output_json),
        ]

    def _pick_tags(self) -> None:
        """Pick a target failure/success tag profile consistent with the score."""
        if self.score >= 4:
            self.failure_tags: list[str] = []
            self.success_tags = ["success"]
            if self.score == 5 and self.num_turns >= 3:
                self.success_tags.append("multi_turn_success")
            if self.score == 4:
                self.failure_tags = [random.choice(["user_correction", "incomplete_action"])]
                self.success_tags.append("graceful_recovery")
        elif self.score == 3:
            choice = random.choice(["canceled", "noise_trigger", "mixed"])
            if choice == "canceled":
                self.failure_tags = ["canceled"]
                self.success_tags = ["success"]
            elif choice == "noise_trigger":
                self.failure_tags = ["noise_trigger"]
                self.success_tags = []
            else:
                self.failure_tags = [random.choice(["user_correction", "incomplete_action"])]
                self.success_tags = ["success"]
        else:  # 1-2
            n = random.randint(1, 3) if self.score == 1 else random.randint(1, 2)
            pool = [t for t in _VALID_FAILURE if t != "canceled"]
            self.failure_tags = random.sample(pool, min(n, len(pool)))
            if self.score == 1 and self.num_turns >= 3 and "repeated_failure" not in self.failure_tags:
                self.failure_tags.append("repeated_failure")
            self.success_tags = []
            if self.score == 2 and random.random() < 0.4:
                self.success_tags.append("graceful_recovery")
            if self.score == 2 and random.random() < 0.5 and "success" not in self.success_tags:
                self.success_tags.append("success")

    @step(retries=4)
    async def _make_conversation(self, client) -> None:
        tag_lines = [f"- {t}: {FAILURE_TAGS[t]}" for t in self.failure_tags]
        tag_lines += [f"- {t}: {SUCCESS_TAGS[t]}" for t in self.success_tags]
        tags_text = "\n".join(tag_lines) if tag_lines else "No specific issues — a smooth interaction."
        meaning = {1: "very dissatisfied", 2: "dissatisfied", 3: "neutral/ambiguous",
                   4: "mostly satisfied", 5: "very satisfied"}[self.score]
        prompt = (
            f"Write a realistic voice-assistant interaction for the category "
            f"'{self.category}'.\n\n"
            f"- Exactly {self.num_turns} turn(s); a turn = one user utterance + "
            f"one assistant response.\n"
            f"- Overall user satisfaction should read as {self.score}/5 ({meaning}).\n"
            f"- It should exhibit these characteristics:\n{tags_text}\n"
            f"- User style: {self.user_persona}.\n"
            f"- Assistant style: {self.assistant_persona}.\n\n"
            "Guidance: use natural language and concrete names (rooms, devices, "
            "songs, contacts, places). Include realistic frustration markers "
            "('no', 'ugh', corrections, rephrasings) only when the score calls "
            "for them. Do not be over-dramatic. For a noise_trigger, the "
            "assistant acts with no preceding user command. For context_loss the "
            "assistant forgets/contradicts something established earlier. For "
            "hallucination the assistant invents a fact or capability.\n\n"
            "Output ONLY a JSON array of objects, each with 'role' "
            "('user' or 'assistant') and 'content'. No other fields, no prose."
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        data = _extract_json(safe_content(resp))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), None)
        if not isinstance(data, list) or len(data) < 2:
            raise GenerationError("conversation must be a JSON array of >=2 turns")

        turns = []
        for t in data:
            if not isinstance(t, dict) or "role" not in t or "content" not in t:
                raise GenerationError(f"bad turn: {t!r}")
            if t["role"] not in ("user", "assistant"):
                raise GenerationError(f"bad role: {t['role']!r}")
            turns.append({"role": t["role"], "content": str(t["content"]).strip()})
        # A noise_trigger legitimately opens on the assistant; otherwise a real
        # interaction starts with the user.
        if "noise_trigger" not in self.failure_tags and turns[0]["role"] != "user":
            raise GenerationError("interaction should start with the user")
        self.turns = turns
        self.n_user_turns = sum(1 for t in turns if t["role"] == "user") or 1

    def _format_conversation(self) -> str:
        if self.input_format == "transcript":
            return "\n".join(
                f"{'User:' if t['role'] == 'user' else 'Assistant:'} {t['content']}"
                for t in self.turns
            )
        if self.input_format == "event_log_minimal":
            events = [{"role": t["role"], "content": t["content"]} for t in self.turns]
            return json.dumps(events, indent=2)
        return self._rich_event_log()

    def _rich_event_log(self) -> str:
        """A JSON event log with plausibly-inferred metadata on assistant turns."""
        action_kw = [
            ("play", ("playing", "play", "started")),
            ("turn_on", ("turning on", "turned on", "switching on")),
            ("turn_off", ("turning off", "turned off")),
            ("navigate", ("navigat", "routing", "directions")),
            ("call", ("calling", "dial")),
            ("send_message", ("texting", "sending", "message")),
            ("set", ("timer", "alarm", "reminder")),
            ("lock", ("locking", "locked")),
            ("unlock", ("unlocking", "unlocked")),
            ("none", ("sorry", "apolog", "can't", "couldn't", "unable")),
            ("cancel", ("stopping", "stopped", "cancel")),
        ]
        entity_kw = [
            ("living_room_lights", ("living room light",)),
            ("kitchen_lights", ("kitchen light",)),
            ("bedroom_lights", ("bedroom light",)),
            ("thermostat", ("temperature", "thermostat", "degree")),
            ("media:spotify", ("spotify", "song", "playlist", "music")),
            ("phone", ("call", "dial", "phone")),
            ("navigation", ("route", "directions", "miles", "traffic")),
            ("timer", ("timer", "minute")),
            ("alarm", ("alarm", "wake")),
            ("front_door", ("front door",)),
        ]
        events = []
        for i, t in enumerate(self.turns):
            ev = {"timestamp": f"2024-01-15T08:30:{i * 3:02d}Z", "role": t["role"], "content": t["content"]}
            if t["role"] == "assistant":
                low = t["content"].lower()
                ev["action"] = next((a for a, kws in action_kw if any(k in low for k in kws)), "confirm")
                ent = next((e for e, kws in entity_kw if any(k in low for k in kws)), None)
                if ent:
                    ev["entity"] = ent
                unsure = any(w in low for w in ("sorry", "can't", "couldn't", "unable", "i think", "maybe"))
                ev["confidence"] = round(random.uniform(0.5, 0.8) if unsure else random.uniform(0.85, 1.0), 2)
            events.append(ev)
        return json.dumps(events, indent=2)

    @step(retries=4)
    async def _make_output(self, client) -> None:
        n = self.n_user_turns
        target = (
            f"Target assessment to encode (already determined): score={self.score}/5, "
            f"failure_tags={self.failure_tags or '[]'}, success_tags={self.success_tags or '[]'}."
        )
        prompt = (
            "You score voice-assistant interactions for user satisfaction. "
            "Produce the structured assessment for the interaction below.\n\n"
            + target
            + "\n\nOutput a JSON object with EXACTLY these keys in this order:\n"
            "- reasoning: 2-4 sentences citing specific turns/events and WHY/WHERE "
            "things went well or wrong (no generic 'the user seemed unhappy').\n"
            "- score: integer 1-5 (use the target).\n"
            "- failure_tags: array using the target failure tags.\n"
            "- success_tags: array using the target success tags.\n"
            f"- failed_turns: 1-indexed turn numbers (1..{n}) where failures occurred.\n"
            f"- successful_turns: 1-indexed turn numbers (1..{n}) that went well.\n\n"
            f"There are {n} turn(s). Turn numbers must be within 1..{n}. "
            "Output ONLY the JSON object, no prose, no code fences.\n\n"
            f"Interaction:\n{self.formatted_input}"
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}-out",
            messages=[{"role": "user", "content": prompt}],
        )
        data = _extract_json(safe_content(resp))
        if not isinstance(data, dict):
            raise GenerationError("output must be a JSON object")
        # Pin the determined score/tags so the gold matches the target profile
        # the conversation was generated for (the LLM occasionally drifts).
        data["score"] = self.score
        data["failure_tags"] = list(self.failure_tags)
        data["success_tags"] = list(self.success_tags)
        data = {k: data.get(k) for k in _REQUIRED_FIELDS}
        validate_output(data, num_turns=n)
        self.output_json = _ordered_output(data)
