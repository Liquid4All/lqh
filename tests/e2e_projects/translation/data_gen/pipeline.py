"""
Business Chat Translation Pipeline v1
Generates EN↔DE and EN↔JA translation training data with structured JSON output.
"""

import json
import random
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, safe_content, step
import liquidrandom

# Translation directions with source/target language codes
DIRECTIONS = [
    ("en", "de"),  # English → German
    ("de", "en"),  # German → English
    ("en", "ja"),  # English → Japanese
    ("ja", "en"),  # Japanese → English
]

# Common English business loanwords that should be preserved in DE/JA
BUSINESS_LOANWORDS = [
    "Meeting", "Deadline", "Feedback", "Agenda", "Presentation",
    "Project", "Budget", "Report", "Team", "Manager",
    "Workshop", "Benchmark", "Sprint", "Review", "Kickoff",
    "Update", "Briefing", "Proposal", "Pipeline", "Target",
]

# Business scenario types for diversity
SCENARIO_TYPES = [
    "scheduling a meeting",
    "requesting information",
    "providing a status update",
    "asking for approval",
    "giving feedback on work",
    "discussing a deadline",
    "sharing meeting notes",
    "coordinating with a team",
    "following up on a task",
    "announcing a change",
    "confirming attendance",
    "escalating an issue",
    "requesting a review",
    "sending a reminder",
    "introducing a new team member",
]

# Edge case categories to ensure coverage
EDGE_CASE_TYPES = [
    "short_input",        # 1-2 words
    "mixed_language",     # German with English phrases
    "technical_jargon",   # ROI, KPI, SaaS
    "proper_noun",        # Person names, product names
    "ambiguous",          # "Let's schedule it"
    "numbers_dates",      # Currencies, dates
    "emoji",              # Contains emojis
    "question",           # Direct question
]

LANG_NAMES = {"en": "English", "de": "German", "ja": "Japanese"}

# Register instructions per target language
REGISTER_INSTRUCTIONS = {
    "de": "Use formal 'Sie' form (not 'du'). Preserve common English business loanwords like Meeting, Deadline, Feedback, Agenda, Presentation, Project, Budget as-is.",
    "ja": "Use polite 'desu/masu' form (not casual/plain form). Preserve common English business loanwords like Meeting, Deadline, Feedback, Agenda, Presentation, Project, Budget as-is (do not transliterate to katakana unless the term is widely used in katakana form like ミーティング).",
    "en": "Preserve any German or Japanese business loanwords/expressions that are commonly used in English business communication.",
}


class BusinessChatTranslationV1(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        # Pick a random direction
        self.source_lang, self.target_lang = random.choice(DIRECTIONS)
        self.source_name = LANG_NAMES[self.source_lang]
        self.target_name = LANG_NAMES[self.target_lang]

        # Pick scenario type: 60% normal, 30% thread, 10% edge case
        roll = random.random()
        if roll < 0.60:
            self.scenario_type = "single_message"
        elif roll < 0.90:
            self.scenario_type = "thread"
        else:
            self.scenario_type = "edge_case"

        # Generate seed for consistency within sample
        self.persona = liquidrandom.persona()
        self.seed = f"{self.persona.name}-{self.source_lang}{self.target_lang}"

        # Generate source text
        if self.scenario_type == "single_message":
            await self._generate_single_message(client)
        elif self.scenario_type == "thread":
            await self._generate_thread(client)
        else:
            await self._generate_edge_case(client)

        # Generate translation
        await self._generate_translation(client)

        # Build the user prompt
        user_prompt = self._build_user_prompt()

        # Build the assistant response (JSON)
        response = json.dumps(
            {
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "translation": self.translation,
            },
            ensure_ascii=False,
        )

        return [
            ChatMLMessage("user", user_prompt),
            ChatMLMessage("assistant", response),
        ]

    def _build_user_prompt(self) -> str:
        """Build the user-facing translation request prompt."""
        if self.scenario_type == "thread":
            return f"Translate to {self.target_name}:\n{self.source_text}"
        else:
            return f"Translate to {self.target_name}: {self.source_text}"

    @step(retries=3)
    async def _generate_single_message(self, client):
        """Generate a single business chat message in the source language."""
        scenario = random.choice(SCENARIO_TYPES)
        loanword = random.choice(BUSINESS_LOANWORDS)

        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write a single business chat message in {self.source_name} "
                        f"about {scenario}. "
                        f"Include the word '{loanword}' naturally. "
                        f"Keep it to 1-3 sentences. Sound like a real Slack/Teams message. "
                        f"Write ONLY the message, nothing else."
                    ),
                }
            ],
        )
        content = safe_content(resp) or None
        if not content:
            raise GenerationError("Source text was empty (None content)")
        self.source_text = content.strip()
        if len(self.source_text) < 5:
            raise GenerationError("Source text too short")

    @step(retries=3)
    async def _generate_thread(self, client):
        """Generate a short conversation thread in the source language."""
        scenario = random.choice(SCENARIO_TYPES)
        loanword = random.choice(BUSINESS_LOANWORDS)

        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write a short business chat thread in {self.source_name} "
                        f"about {scenario}. "
                        f"Include the word '{loanword}' naturally. "
                        f"Format as 2-4 messages from different people, each on a new line "
                        f"prefixed with the speaker name and a colon (e.g., 'Alex: ...'). "
                        f"Keep each message to 1-2 sentences. Sound like real Slack/Teams messages. "
                        f"Write ONLY the thread, nothing else."
                    ),
                }
            ],
        )
        content = safe_content(resp) or None
        if not content:
            raise GenerationError("Thread was empty (None content)")
        self.source_text = content.strip()
        if len(self.source_text) < 20:
            raise GenerationError("Thread too short")

    @step(retries=3)
    async def _generate_edge_case(self, client):
        """Generate an edge-case input."""
        edge_type = random.choice(EDGE_CASE_TYPES)

        edge_prompts = {
            "short_input": (
                f"Write a very short business message in {self.source_name} — "
                f"just 1-2 words. Examples: 'Thanks!', 'OK', 'Done', 'Will do'. "
                f"Write ONLY the message."
            ),
            "mixed_language": (
                f"Write a business chat message in {self.source_name} that naturally "
                f"includes 1-2 English business terms or phrases mixed in, as people "
                f"commonly do in real business communication. "
                f"Keep it to 1-2 sentences. Write ONLY the message."
            ),
            "technical_jargon": (
                f"Write a business chat message in {self.source_name} that includes "
                f"at least one technical acronym or jargon term (like ROI, KPI, SaaS, "
                f"B2B, API, SLA, EBITDA). Keep it to 1-2 sentences. "
                f"Write ONLY the message."
            ),
            "proper_noun": (
                f"Write a business chat message in {self.source_name} that mentions "
                f"a specific person's name and/or a product name. "
                f"Keep it to 1-2 sentences. Write ONLY the message."
            ),
            "ambiguous": (
                f"Write a business chat message in {self.source_name} that is "
                f"intentionally ambiguous — where the referent is unclear without context. "
                f"Examples: 'Let's schedule it', 'Can you handle this?', 'I'll follow up'. "
                f"Keep it to 1-2 sentences. Write ONLY the message."
            ),
            "numbers_dates": (
                f"Write a business chat message in {self.source_name} that includes "
                f"specific numbers, dates, or currency amounts. "
                f"Keep it to 1-2 sentences. Write ONLY the message."
            ),
            "emoji": (
                f"Write a business chat message in {self.source_name} that includes "
                f"1-2 emojis. Keep it to 1-2 sentences. Write ONLY the message."
            ),
            "question": (
                f"Write a business question in {self.source_name} — a direct question "
                f"you'd ask a colleague on Slack/Teams. "
                f"Keep it to 1-2 sentences. Write ONLY the question."
            ),
        }

        prompt = edge_prompts.get(edge_type)
        if not prompt:
            prompt = edge_prompts["short_input"]

        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        content = safe_content(resp) or None
        if not content:
            raise GenerationError("Edge case source text was empty (None content)")
        self.source_text = content.strip()
        if len(self.source_text) < 1:
            raise GenerationError("Edge case source text empty")

    @step(retries=5)
    async def _generate_translation(self, client):
        """Generate the translation in the target language."""
        register = REGISTER_INSTRUCTIONS[self.target_lang]

        # Build context-aware translation prompt
        if self.scenario_type == "thread":
            context_note = (
                "This is a conversation thread. Use the full context to resolve "
                "pronouns and references, but translate the entire thread. "
            )
        else:
            context_note = ""

        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a professional business translator. "
                        f"Translate from {self.source_name} to {self.target_name}. "
                        f"{register} "
                        f"{context_note}"
                        f"Translate accurately and completely. Do not add commentary or notes. "
                        f"Output ONLY the translated text, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": self.source_text,
                },
            ],
        )
        content = safe_content(resp) or None
        if not content:
            raise GenerationError("Translation response was empty (None content)")
        self.translation = content.strip()
        if len(self.translation) < 1:
            raise GenerationError("Translation is empty")
