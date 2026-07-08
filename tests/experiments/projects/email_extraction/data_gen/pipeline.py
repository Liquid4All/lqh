"""Customer-support email-extraction data generation pipeline.

Strategy: roll the *target* fields (intent, urgency, products, sender)
deterministically per sample, ask the LLM to write an email matching
them, then build the ground-truth JSON directly from those rolls.
This avoids the chicken-and-egg of "generate email → extract" where
the extractor would itself be a noisy step.

Output is a 2-message ChatML conversation:
  user      — the verbatim email
  assistant — the structured JSON output
"""

from __future__ import annotations

import json
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


PRODUCT_CATALOG = [
    "Quantum Earbuds Pro", "Quantum Earbuds Lite", "AeroDesk Standing Desk",
    "AeroDesk Mat", "PixelCam 4K", "PixelCam Mini", "BrewMaster Espresso",
    "BrewMaster Grinder", "TrailRun GTX 7", "TrailRun GTX 9 Pro",
    "VeloFlex Yoga Mat", "VeloFlex Foam Roller", "OrbitGlow Smart Bulb",
    "OrbitGlow Hub", "FlowState Notebook", "FlowState Pen Set",
    "PolarPack Cooler", "PolarPack Ice", "TrueNorth Tent",
    "TrueNorth Sleeping Bag", "Solstice Solar Charger",
]

INTENTS = ["question", "complaint", "request", "cancellation"]

# Urgency bias by intent (cumulative weights for 1..5)
URGENCY_BIAS = {
    "question":     [40, 35, 20,  4,  1],
    "complaint":    [10, 25, 30, 25, 10],
    "request":      [25, 35, 25, 12,  3],
    "cancellation": [15, 25, 30, 20, 10],
}

INTENT_TONE_HINT = {
    "question": "polite, curious, asking for clarification or information",
    "complaint": "frustrated; tone scales with urgency from mild annoyance to angry",
    "request": "asking for a specific action (refund, replacement, change of address, technical fix)",
    "cancellation": "explicit about cancelling a subscription, order, or account",
}

URGENCY_HINT = {
    1: "very casual, no rush, almost a hello",
    2: "polite, clearly wants action but no time pressure",
    3: "noticeably wants action soon, slightly impatient",
    4: "frustrated, says things like 'this needs to be sorted today'",
    5: "very angry, threatens legal action, mentions a safety issue, or all-caps phrases",
}


def _pick_urgency(intent: str) -> int:
    weights = URGENCY_BIAS[intent]
    return random.choices(range(1, 6), weights=weights)[0]


def _pick_products() -> list[str]:
    n = random.choices([0, 1, 2, 3], weights=[20, 50, 20, 10])[0]
    if n == 0:
        return []
    return random.sample(PRODUCT_CATALOG, k=n)


class EmailExtractionV1(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        # Roll targets deterministically per sample.
        self.intent = random.choice(INTENTS)
        self.urgency = _pick_urgency(self.intent)
        self.products = _pick_products()
        self.persona = liquidrandom.persona()
        # 15% chance of no sender name (anonymous / no sign-off).
        self.hide_sender = random.random() < 0.15
        self.sender_name = "" if self.hide_sender else self.persona.name

        self.seed = (
            f"ext-{self.intent}-{self.urgency}-"
            f"{self.persona.name}-{len(self.products)}"
        )

        await self._gen_email(client)
        await self._gen_summary(client)

        ground_truth = {
            "sender_name": self.sender_name,
            "intent": self.intent,
            "mentioned_products": list(self.products),
            "urgency": self.urgency,
            "summary": self.summary,
        }
        assistant = json.dumps(ground_truth, ensure_ascii=False)

        return [
            ChatMLMessage("user", self.email_text),
            ChatMLMessage("assistant", assistant),
        ]

    @step(retries=4)
    async def _gen_email(self, client):
        """Have the LLM write a customer-support email matching our rolls."""
        product_clause = (
            f"Mention these specific products by name (use the exact wording): "
            f"{', '.join(self.products)}."
            if self.products
            else "Do NOT mention any product by a proper name; refer to it generically if needed."
        )
        sender_clause = (
            f"End with a sign-off using the name {self.sender_name!r}."
            if not self.hide_sender
            else "Do NOT include a sign-off or signature with a name. End abruptly."
        )
        prompt = (
            f"Write a customer-support email to a company. Format:\n"
            f"  Subject: <subject>\n"
            f"  <empty line>\n"
            f"  <body, 2-4 short paragraphs>\n\n"
            f"Constraints:\n"
            f"- Intent: {self.intent}. Tone: {INTENT_TONE_HINT[self.intent]}.\n"
            f"- Urgency level {self.urgency}/5: {URGENCY_HINT[self.urgency]}.\n"
            f"- {product_clause}\n"
            f"- {sender_clause}\n"
            f"- Sound like a real person; vary phrasing.\n"
            f"- Output ONLY the email (subject + body); no preamble, no quotes."
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip().strip("`'\"")
        if not text:
            raise GenerationError("Email response was empty")
        if "Subject:" not in text:
            raise GenerationError("Email missing 'Subject:' line")
        if len(text) < 80:
            raise GenerationError(f"Email too short ({len(text)} chars)")
        # Verify products appear in the body if we asked for them.
        for p in self.products:
            if p not in text:
                raise GenerationError(f"Product {p!r} missing from email")
        # Verify sender name in sign-off (if requested).
        if not self.hide_sender and self.sender_name not in text:
            raise GenerationError(f"Sign-off missing sender {self.sender_name!r}")
        if self.hide_sender:
            # Avoid the LLM sneaking in a name anyway.
            if self.persona.name in text:
                raise GenerationError(
                    f"Persona name {self.persona.name!r} leaked despite hide_sender"
                )
        self.email_text = text

    @step(retries=3)
    async def _gen_summary(self, client):
        """Generate a 1-sentence neutral summary of the email body."""
        resp = await client.chat.completions.create(
            model=f"random:medium:summary-{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this customer email in ONE neutral sentence "
                    "of at most 25 words. Focus on what the customer wants. "
                    "Output ONLY the sentence, no preamble.\n\n"
                    f"{self.email_text}"
                ),
            }],
        )
        s = safe_content(resp).strip().strip("`'\"")
        if not s:
            raise GenerationError("Summary empty")
        words = s.split()
        if len(words) > 35:
            raise GenerationError(f"Summary too long ({len(words)} words)")
        if len(words) < 5:
            raise GenerationError(f"Summary too short ({len(words)} words)")
        self.summary = s
