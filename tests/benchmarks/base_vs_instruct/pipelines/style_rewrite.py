"""Tone/style rewrite datagen for DPO-friendly generation.

The task asks the model to rewrite a flawed support reply according to a style
brief. The target is open-ended but constrained by facts, tone, banned phrases,
and length. This leaves more room for preference learning than single-label or
exact-field tasks.

The reference rewrite (the assistant target) is **LLM-generated and validated**
in-pipeline — never a hard-coded template. An earlier version pinned a single
canned sentence ("Here is the current update: …, so I will keep this <tone>.")
as the gold; that string violates the task's own brief (it opens with a
preamble and parrots the tone words back), so the scorer rated it ~2/10 and SFT
faithfully learned a bad target. Now we generate a real rewrite and reject any
candidate that leaks a banned phrase, drops a required fact, misses the length
window, or carries a greeting/sign-off/meta-commentary. The run-level scorer
filter (``run.py``) is the second line of defence on top of this.

Each sample is a 3-message ChatML conversation:
  system    -> SYSTEM_PROMPT
  user      -> draft + style brief + required facts
  assistant -> polished reply (LLM-written, validated)
"""

from __future__ import annotations

import math
import random
import re

from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    GenerationError,
    Pipeline,
    safe_content,
    step,
)

SYSTEM_PROMPT = (
    "Rewrite customer-support replies to match the requested tone and style. "
    "Preserve all required facts, remove banned phrases, keep the reply concise, "
    "and output ONLY the rewritten reply."
)

_TONES = [
    "warm, calm, and accountable",
    "concise, professional, and direct",
    "empathetic but not apologetic",
    "friendly, precise, and action-oriented",
    "reassuring, plainspoken, and brief",
]
_ISSUES = [
    (
        "a delayed replacement laptop",
        ["replacement laptop ships tomorrow", "tracking email arrives by 6 PM"],
    ),
    (
        "a duplicate subscription charge",
        ["duplicate charge was refunded", "refund posts within 3-5 business days"],
    ),
    (
        "a failed password reset",
        ["temporary access link expires in 30 minutes", "support can reissue it"],
    ),
    (
        "a damaged grocery delivery",
        ["credit has been added to the account", "photos are no longer needed"],
    ),
    (
        "a noisy hotel room",
        ["room change is confirmed", "front desk has the new key ready"],
    ),
    (
        "a missed onboarding call",
        ["new calendar invite is attached", "the setup checklist is unchanged"],
    ),
]
_BANNED = [
    "we apologize for any inconvenience",
    "please be advised",
    "as per our policy",
    "thank you for your patience",
    "at your earliest convenience",
]

# Length window the brief asks for. Validated strictly so a gold never violates
# the very constraint the scorer grades against.
_MIN_WORDS = 45
_MAX_WORDS = 85

# Lower-cased prefixes that signal a greeting / preamble / sign-off — the brief
# requires "output ONLY the rewritten reply", so a candidate starting with any
# of these is rejected.
_PREAMBLE_PREFIXES = (
    "here is", "here's", "here are", "sure", "hi ", "hi,", "hello", "hey",
    "dear ", "greetings", "good morning", "good afternoon",
    "thanks", "thank you", "rewritten", "updated reply", "rewrite:", "reply:",
)

# Lower-cased fragments that signal meta-commentary about the task/tone rather
# than an actual support reply (the canned-template failure mode).
_META_FRAGMENTS = (
    "i will keep", "i'll keep", "as requested", "as you requested",
    "the requested tone", "in the tone", "rewritten reply", "this rewrite",
    "i have rewritten", "here is the rewrite",
)

# Tokens too generic to count toward fact preservation.
_FACT_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "has", "have", "had", "will", "for", "in", "on",
    "at", "by", "your", "you", "it", "its", "this", "that", "with", "no",
}


def _significant_tokens(text: str) -> list[str]:
    """Content tokens of *text*: drop stopwords, keep words >2 chars or digits."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    sig = [t for t in toks if t not in _FACT_STOPWORDS and (len(t) > 2 or t.isdigit())]
    return sig or toks


def _fact_preserved(fact: str, text_tokens: set[str]) -> bool:
    """Whether *fact*'s content survives in *text_tokens*.

    A *stylistic* rewrite is expected to rephrase ("room change is confirmed" ->
    "your room change has been confirmed"), so we require content-word overlap
    rather than a verbatim substring — at most one significant token may be
    reworded away. The scorer filter is the semantic backstop on top of this.
    """
    sig = _significant_tokens(fact)
    if not sig:
        return True
    hits = sum(1 for t in sig if t in text_tokens)
    return hits >= max(1, len(sig) - 1)


def validate_rewrite(
    text: str,
    *,
    tone: str,
    facts: list[str],
    banned: list[str],
    draft: str,
) -> None:
    """Raise ``GenerationError`` if *text* fails the style-rewrite brief.

    First line of defence (the run-level scorer filter is the second). Checks:
    non-empty, no banned phrase, every required fact preserved, length window,
    no greeting/preamble/sign-off, no tone meta-commentary, not a draft copy.
    """
    if not text:
        raise GenerationError("empty rewrite")
    low = text.lower()

    for phrase in banned:
        if phrase.lower() in low:
            raise GenerationError(f"banned phrase leaked: {phrase!r}")

    text_tokens = set(re.findall(r"[a-z0-9]+", low))
    for fact in facts:
        if not _fact_preserved(fact, text_tokens):
            raise GenerationError(f"dropped required fact: {fact!r}")

    n_words = len(text.split())
    if not (_MIN_WORDS <= n_words <= _MAX_WORDS):
        raise GenerationError(f"length out of range ({n_words} words)")

    for prefix in _PREAMBLE_PREFIXES:
        if low.startswith(prefix):
            raise GenerationError(f"greeting/preamble leaked: {text[:40]!r}")

    # No bullet/numbered list (brief says no bullet list).
    if any(marker in text for marker in ("\n-", "\n*", "\n•", "\n1.")):
        raise GenerationError("contains a list")

    # The candidate must not echo the requested tone verbatim or otherwise
    # editorialise about its own writing — that is the canned-template tell.
    if tone.lower() in low:
        raise GenerationError("echoed the requested tone verbatim")
    for frag in _META_FRAGMENTS:
        if frag in low:
            raise GenerationError(f"meta-commentary about the task: {frag!r}")

    if low == draft.strip().lower():
        raise GenerationError("rewrite is identical to the draft")


class StyleRewritePipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.tone = random.choice(_TONES)
        self.issue, self.facts = random.choice(_ISSUES)
        self.banned = random.sample(_BANNED, 2)
        self.seed = f"sr-{self.issue}-{random.randint(0, 1_000_000)}"
        await self._make_draft(client)
        await self._rewrite(client)
        return [
            ChatMLMessage("system", SYSTEM_PROMPT),
            ChatMLMessage("user", self.user_prompt),
            ChatMLMessage("assistant", self.rewrite),
        ]

    @step(retries=4)
    async def _make_draft(self, client) -> None:
        draft = (
            f"{self.banned[0].capitalize()}. Regarding {self.issue}, "
            f"please be advised that {self.facts[0]}. Also, {self.facts[1]}. "
            f"{self.banned[1].capitalize()}. We wanted to circle back with "
            "this update and make sure you understand that the request is "
            "being handled through the normal support process."
        )
        self.draft = draft
        self.user_prompt = (
            f"Rewrite this support reply in a {self.tone} style.\n\n"
            "Required facts to preserve exactly:\n"
            + "\n".join(f"- {fact}" for fact in self.facts)
            + "\n\nBanned phrases to remove:\n"
            + "\n".join(f"- {phrase}" for phrase in self.banned)
            + "\n\nLength: 45-85 words. Use no bullet list, no greeting, "
            "and no sign-off. Output only the rewritten reply.\n\n"
            f"Draft:\n{self.draft}"
        )

    @step(retries=6)
    async def _rewrite(self, client) -> None:
        prompt = (
            f"Rewrite the customer-support draft below in a {self.tone} style.\n\n"
            "Required facts to preserve exactly (keep their meaning):\n"
            + "\n".join(f"- {fact}" for fact in self.facts)
            + "\n\nBanned phrases that must NOT appear anywhere:\n"
            + "\n".join(f"- {phrase}" for phrase in self.banned)
            + "\n\nConstraints:\n"
            f"- Length: {_MIN_WORDS}-{_MAX_WORDS} words.\n"
            "- No bullet list, no greeting, no sign-off.\n"
            "- Do NOT describe the tone or mention that you are rewriting — "
            "just write the reply itself.\n"
            "- Output ONLY the rewritten reply, nothing before or after it.\n\n"
            f"Draft:\n{self.draft}"
        )
        resp = await client.chat.completions.create(
            model=f"random:large:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip().strip("`").strip('"').strip()
        validate_rewrite(
            text,
            tone=self.tone,
            facts=self.facts,
            banned=self.banned,
            draft=self.draft,
        )
        self.rewrite = text
