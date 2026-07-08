"""Article-summarization data generation pipeline.

For each sample we:

1. Roll a random topic + domain combo (news / blog / tech / business).
2. Generate a 1-3 paragraph article excerpt about it via ``random:medium``.
3. Generate a 2-3 sentence neutral summary via ``random:medium`` (a
   different seed, so we get a different model in the rotation).
4. Emit ``[user("Summarize the following:\\n\\n<excerpt>"),
        assistant("<summary>")]``.

Output is plain text — no JSON wrapper, no tools. Quality varies by
generation model; the LLM-judge filter (Phase 3.5 in the
data_generation skill) is the catch-all sanity check.
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


DOMAINS = [
    "tech_news",
    "business_news",
    "research_writeup",
    "personal_blog",
    "policy_brief",
    "product_announcement",
    "internal_memo",
    "industry_report",
    "interview_recap",
    "incident_report",
]

# Topic seeds — a mix of evergreen and current-feeling areas so the
# generated text feels varied. The pipeline picks a domain + topic
# pair per sample.
TOPICS = [
    "renewable energy adoption",
    "remote-work productivity",
    "open-source supply-chain risk",
    "consumer privacy regulation",
    "battery technology breakthroughs",
    "small-business cashflow management",
    "machine-learning interpretability",
    "urban transit funding",
    "agricultural drought response",
    "fintech compliance changes",
    "telehealth expansion",
    "manufacturing automation",
    "K-12 curriculum shifts",
    "climate adaptation in cities",
    "cybersecurity incident response",
    "biotech clinical trial outcomes",
    "logistics and last-mile delivery",
    "civic data transparency",
    "creator-economy platforms",
    "language preservation efforts",
]

# Length variants — the engine should produce a healthy mix.
LENGTHS = [
    ("very_short", "1 short paragraph (about 50-80 words)"),
    ("short", "1 paragraph (about 80-150 words)"),
    ("medium", "2 paragraphs (about 150-250 words)"),
    ("long", "3 paragraphs (about 250-400 words)"),
]


class ArticleSummarizationV1(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        # Pick a topic + domain + length per sample.
        self.domain = random.choice(DOMAINS)
        self.topic = random.choice(TOPICS)
        self.length_label, self.length_desc = random.choice(LENGTHS)

        # Use a varied seed so the generation rotates across the
        # underlying models. Persona is a cheap source of entropy.
        self.persona = liquidrandom.persona()
        self.seed = f"{self.persona.name}-{self.domain}-{self.topic}"

        await self._generate_excerpt(client)
        await self._generate_summary(client)

        # Build the conversation.
        user_msg = f"Summarize the following:\n\n{self.excerpt}"
        return [
            ChatMLMessage("user", user_msg),
            ChatMLMessage("assistant", self.summary),
        ]

    @step(retries=3)
    async def _generate_excerpt(self, client):
        """Generate a short article-style excerpt."""
        domain_label = self.domain.replace("_", " ")
        prompt = (
            f"Write a {self.length_desc} {domain_label} excerpt about "
            f"\"{self.topic}\". Use realistic but invented details: "
            f"specific numbers, dates, organisation names, or quotes are "
            f"welcome. Write only the article text — no headline, no "
            f"byline, no metadata. Use prose paragraphs (no bullet points)."
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        content = safe_content(resp)
        if not content:
            raise GenerationError("Excerpt was empty")
        self.excerpt = content.strip()
        if len(self.excerpt) < 60:
            raise GenerationError(f"Excerpt too short: {len(self.excerpt)} chars")
        # Reject obvious model preambles that leak into the data.
        for bad in ("Here is", "Below is", "Here's", "I'll write"):
            if self.excerpt.startswith(bad):
                raise GenerationError(f"Excerpt has preamble: {self.excerpt[:40]!r}")

    @step(retries=5)
    async def _generate_summary(self, client):
        """Generate a 2-3 sentence neutral summary of the excerpt."""
        prompt = (
            "Summarize the following text in 2-3 neutral, declarative "
            "sentences (about 30-70 words). Stay strictly faithful — do "
            "not add facts, opinions, or hedging that aren't in the source. "
            "Do NOT include a preamble like \"Here is a summary:\" or "
            "\"Summary:\". Output only the summary text.\n\n"
            f"Text:\n{self.excerpt}"
        )
        # Different seed than the excerpt to rotate models.
        summary_seed = f"summary-{self.seed}"
        resp = await client.chat.completions.create(
            model=f"random:medium:{summary_seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        content = safe_content(resp)
        if not content:
            raise GenerationError("Summary was empty")
        summary = content.strip()
        # Strip common preambles defensively (the judge would penalise them).
        for prefix in (
            "Here is a summary:", "Here's a summary:", "Summary:",
            "Here is the summary:", "In summary,", "In summary:",
        ):
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix):].strip()
        if len(summary) < 20:
            raise GenerationError(f"Summary too short: {len(summary)} chars")
        if len(summary) > 600:
            raise GenerationError(f"Summary too long: {len(summary)} chars")
        # Sanity: 2-5 sentences is the legal range; <2 or >5 is suspect.
        sentences = [s for s in summary.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) < 2:
            raise GenerationError(f"Summary has only {len(sentences)} sentence(s)")
        if len(sentences) > 5:
            raise GenerationError(f"Summary has {len(sentences)} sentences (>5)")
        self.summary = summary
