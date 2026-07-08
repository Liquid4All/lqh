"""Customer-support email triage data generation pipeline.

Like ``email_extraction``, we roll the *target* triage decision
deterministically per sample (category, priority, next_actions),
then ask the LLM to write an email matching it. Ground truth is
computed directly from the rolls — no extraction step needed.

Output is a 2-message ChatML conversation:
  user      — the verbatim email
  assistant — the triage JSON
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


# Each scenario template provides:
#   - category, priority, next_actions: the ground-truth fields
#   - prompt_hint: what the email should contain so it deserves
#     this triage plan
SCENARIOS: list[dict] = [
    # --- p0 critical ---
    {
        "category": "product_quality", "priority": "p0",
        "next_actions": ["escalate", "loop_in_team"],
        "hint": "describes a SAFETY issue with a product (battery smoking, sharp metal in food, electric shock, child injured). Customer is alarmed.",
    },
    {
        "category": "billing", "priority": "p0",
        "next_actions": ["escalate", "loop_in_team", "offer_refund"],
        "hint": "customer threatens LEGAL ACTION over a billing issue (mentions lawyer, attorney, or chargeback dispute formally).",
    },
    {
        "category": "account", "priority": "p0",
        "next_actions": ["escalate", "loop_in_team"],
        "hint": "customer reports a data breach concern: their personal details appear on someone else's account, or wrong account info shown.",
    },
    # --- p1 high ---
    {
        "category": "billing", "priority": "p1",
        "next_actions": ["offer_refund", "acknowledge"],
        "hint": "customer was charged twice for the same order; says it's clearly visible on their card statement.",
    },
    {
        "category": "shipping", "priority": "p1",
        "next_actions": ["loop_in_team", "acknowledge"],
        "hint": "package marked delivered but is not at the address; customer says it's been 2 days, others on their street received their packages.",
    },
    {
        "category": "product_quality", "priority": "p1",
        "next_actions": ["offer_refund", "loop_in_team"],
        "hint": "product arrived broken (cracked screen, leaking, missing parts). Customer wants replacement or refund.",
    },
    {
        "category": "account", "priority": "p1",
        "next_actions": ["escalate", "loop_in_team"],
        "hint": "customer locked out of paid account, can't access their work, reset email never arrived.",
    },
    # --- p2 normal ---
    {
        "category": "billing", "priority": "p2",
        "next_actions": ["request_more_info"],
        "hint": "customer is confused about an invoice line item but can't say which one — vague email asking for clarification.",
    },
    {
        "category": "shipping", "priority": "p2",
        "next_actions": ["acknowledge"],
        "hint": "customer asks for an update on shipping — package is in transit and only 1 day late, polite tone.",
    },
    {
        "category": "product_quality", "priority": "p2",
        "next_actions": ["acknowledge", "loop_in_team"],
        "hint": "customer reports a minor defect that's annoying but not blocking use; wants the company to know.",
    },
    {
        "category": "account", "priority": "p2",
        "next_actions": ["acknowledge"],
        "hint": "customer wants to update billing address or contact preferences; nothing broken, just admin.",
    },
    # --- p3 low ---
    {
        "category": "feature_request", "priority": "p3",
        "next_actions": ["acknowledge", "loop_in_team"],
        "hint": "customer politely suggests a new feature (e.g. 'would love to be able to pause my subscription instead of cancelling').",
    },
    {
        "category": "feature_request", "priority": "p3",
        "next_actions": ["acknowledge"],
        "hint": "customer asks 'do you have plans to support X?' — informational request.",
    },
    {
        "category": "other", "priority": "p3",
        "next_actions": ["acknowledge", "close"],
        "hint": "POSITIVE feedback — customer thanks the team, says the product is great, no ask attached.",
    },
    {
        "category": "other", "priority": "p3",
        "next_actions": ["request_more_info"],
        "hint": "rambling message that's hard to act on — customer mentions multiple unrelated things, no clear ask.",
    },
]


class EmailTriageV1(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.scenario = random.choice(SCENARIOS)
        self.persona = liquidrandom.persona()
        self.seed = (
            f"triage-{self.scenario['category']}-{self.scenario['priority']}-"
            f"{self.persona.name}"
        )

        await self._gen_email(client)
        await self._gen_rationale(client)

        ground_truth = {
            "category": self.scenario["category"],
            "priority": self.scenario["priority"],
            "next_actions": list(self.scenario["next_actions"]),
            "rationale": self.rationale,
        }

        return [
            ChatMLMessage("user", self.email_text),
            ChatMLMessage("assistant", json.dumps(ground_truth, ensure_ascii=False)),
        ]

    @step(retries=4)
    async def _gen_email(self, client):
        prompt = (
            f"Write a customer-support email that fits this scenario:\n"
            f"  {self.scenario['hint']}\n\n"
            f"Format:\n"
            f"  Subject: <subject>\n"
            f"  <empty line>\n"
            f"  <body, 2-4 short paragraphs>\n"
            f"  <sign-off with the name {self.persona.name!r}>\n\n"
            f"Sound like a real person; vary phrasing. Output ONLY the "
            f"email — no preamble, no quotes."
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
        if self.persona.name not in text:
            raise GenerationError(f"Sign-off missing sender {self.persona.name!r}")
        self.email_text = text

    @step(retries=3)
    async def _gen_rationale(self, client):
        """Generate a 1-2 sentence rationale for the triage plan."""
        prompt = (
            f"Customer email:\n{self.email_text}\n\n"
            f"Triage plan: category={self.scenario['category']}, "
            f"priority={self.scenario['priority']}, "
            f"actions={self.scenario['next_actions']}.\n\n"
            "Write a 1-2 sentence rationale explaining WHY this plan, "
            "referencing specific details from the email. Neutral, "
            "concise. Output ONLY the rationale text."
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:rationale-{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        r = safe_content(resp).strip().strip("`'\"")
        if not r:
            raise GenerationError("Rationale empty")
        words = r.split()
        if len(words) > 60:
            raise GenerationError(f"Rationale too long ({len(words)} words)")
        if len(words) < 6:
            raise GenerationError(f"Rationale too short ({len(words)} words)")
        self.rationale = r
