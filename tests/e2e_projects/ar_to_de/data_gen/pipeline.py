"""Arabic → German translation data generation pipeline.

For each sample:

1. Roll domain (web vs conversational, ~60/40), sub-domain (news /
   blog / chat / review / etc.), topic, length, register.
2. Generate Arabic source text via ``random:medium`` (the seed mixes
   topic + persona so we get good variety across samples).
3. Translate that Arabic source to German via ``random:medium`` with
   a *different* seed (model rotation — the chosen translations come
   from a different generation path than the SFT/DPO model will
   produce, giving DPO a real preference signal to learn).
4. Emit ``[user("Translate to German:\\n\\n<arabic>"), assistant("<german>")]``.

Output is plain text — no JSON wrapper, no tools. The LLM-judge filter
in Phase 3.5 catches low-quality translations before training.
"""

from __future__ import annotations

import random

import liquidrandom

from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    GenerationError,
    Pipeline,
    safe_content,
    step,
)


WEB_SUBDOMAINS = [
    "news_headline",         # 1-2 sentence news lead
    "news_article",          # 2-4 paragraph excerpt
    "blog_post",             # 2-4 paragraph opinion / lifestyle
    "product_description",   # 2-4 sentences, marketing tone
    "forum_post",            # multi-sentence Reddit-style post
    "social_post",           # 1-3 sentence tweet/status
]

CONVERSATIONAL_SUBDOMAINS = [
    "chat_dialogue",         # 3-6 alternating turns (multi-line)
    "customer_review",       # 2-4 sentence review with rating implication
    "comment_thread",        # 2-3 short comments stacked
    "casual_message",        # 1-3 sentence personal message
]

TOPICS = [
    "smartphone reviews",
    "soccer match results",
    "a recipe for kabsa",
    "a travel itinerary in Cairo",
    "remote-work productivity tips",
    "a new restaurant opening",
    "online shopping experiences",
    "weather forecasts",
    "personal finance advice",
    "a movie review",
    "fitness routines",
    "school admissions",
    "cybersecurity news",
    "a startup announcement",
    "a music album release",
    "real estate listings in Riyadh",
    "tech industry layoffs",
    "a flight delay complaint",
    "a wedding planning question",
    "a politics commentary",
    "a healthcare appointment",
    "currency exchange rates",
    "a public transit issue",
    "a book recommendation",
    "a holiday celebration",
    "a coffee shop review",
    "an apartment-hunt experience",
    "a parenting question",
    "an environmental policy update",
    "a museum exhibition",
]

LENGTHS = [
    ("short", "1-2 sentences"),
    ("medium", "3-5 sentences"),
    ("long", "1-2 short paragraphs"),
]


class ArabicToGermanV1(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        # Domain split: 60% web, 40% conversational. Reflects roughly
        # the volume of each kind of text the model will see.
        if random.random() < 0.60:
            self.domain = "web"
            self.subdomain = random.choice(WEB_SUBDOMAINS)
        else:
            self.domain = "conversational"
            self.subdomain = random.choice(CONVERSATIONAL_SUBDOMAINS)

        self.topic = random.choice(TOPICS)
        self.length_label, self.length_desc = random.choice(LENGTHS)

        self.persona = liquidrandom.persona()
        self.seed = (
            f"ar2de-{self.subdomain}-{self.topic}-"
            f"{self.length_label}-{self.persona.name}"
        )

        await self._generate_arabic(client)
        await self._generate_german(client)

        user_msg = f"Translate to German:\n\n{self.arabic}"
        return [
            ChatMLMessage("user", user_msg),
            ChatMLMessage("assistant", self.german),
        ]

    @step(retries=4)
    async def _generate_arabic(self, client):
        """Generate the Arabic source text."""
        subdomain_hint = self.subdomain.replace("_", " ")
        prompt = (
            f"Write a {self.length_desc} {subdomain_hint} in Arabic about "
            f"\"{self.topic}\". "
            f"Use realistic but invented details — specific numbers, names, "
            f"places, dates are fine. "
            f"Match the register and style of a real {subdomain_hint}: "
            f"news headline = formal and concise; chat dialogue = casual "
            f"with natural turn-taking; product description = marketing "
            f"tone; customer review = first-person with an opinion; etc. "
            f"Output ONLY the Arabic text — no headline, no metadata, no "
            f"English at all (except for proper nouns / brand names if "
            f"they're commonly used as-is). No preamble like 'Here is...'"
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip().strip("`'\"")
        if not text:
            raise GenerationError("Arabic source was empty")
        if len(text) < 30:
            raise GenerationError(f"Arabic source too short ({len(text)} chars)")
        if len(text) > 2000:
            raise GenerationError(f"Arabic source too long ({len(text)} chars)")

        # Quick sanity check: there should be a meaningful fraction of
        # Arabic-script characters. Reject otherwise (model sometimes
        # falls back to English when struggling with the topic).
        arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
        if arabic_chars < len(text) * 0.4:
            raise GenerationError(
                f"Source text not predominantly Arabic "
                f"({arabic_chars}/{len(text)} arabic chars)"
            )

        # Reject obvious model preambles.
        for bad in ("Here is", "Below is", "Here's", "Sure,"):
            if text.startswith(bad):
                raise GenerationError(f"Arabic preamble: {text[:40]!r}")

        self.arabic = text

    @step(retries=4)
    async def _generate_german(self, client):
        """Translate the Arabic source to German with a different seed."""
        prompt = (
            f"Translate the following Arabic text into natural, fluent "
            f"German. Preserve meaning, register, and tone — match the "
            f"original's formality. Keep names, numbers, dates, and any "
            f"English loanwords. Output ONLY the German translation — no "
            f"preamble, no quotes, no English explanation, no repetition "
            f"of the Arabic source.\n\n"
            f"Arabic:\n{self.arabic}"
        )
        # Different seed so we rotate through different generator models —
        # this is what gives DPO a real preference signal later (the chosen
        # translations come from a different distribution than what the
        # SFT-trained student will produce).
        translation_seed = f"de-target-{self.seed}"
        resp = await client.chat.completions.create(
            model=f"random:medium:{translation_seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip().strip("`'\"")
        if not text:
            raise GenerationError("German translation was empty")
        if len(text) < 20:
            raise GenerationError(f"German translation too short ({len(text)} chars)")

        # Reject preambles that the judge would penalise.
        for prefix in (
            "Hier ist die Übersetzung:", "Hier ist die Uebersetzung:",
            "Übersetzung:", "Uebersetzung:",
            "Here is", "Here's", "The translation",
        ):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].lstrip(" :\n")

        # Reject if the model emitted Arabic instead of German (model
        # sometimes echoes the source).
        arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
        if arabic_chars > len(text) * 0.10:
            raise GenerationError(
                f"German translation contains too much Arabic "
                f"({arabic_chars}/{len(text)} chars) — likely echoed source"
            )

        # Sanity check: should contain German-ish characters (umlauts /
        # ß) OR at least be in Latin script with reasonable length.
        latin_chars = sum(1 for c in text if "a" <= c.lower() <= "z" or c in "äöüÄÖÜß")
        if latin_chars < len(text) * 0.30:
            raise GenerationError(
                f"German translation not predominantly Latin "
                f"({latin_chars}/{len(text)} latin chars)"
            )

        self.german = text
