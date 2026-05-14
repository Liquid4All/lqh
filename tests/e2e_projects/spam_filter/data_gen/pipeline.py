"""Generic spam-filter data generation pipeline.

For each sample we:

1. Roll a filter rule from a fixed catalog of generic rules.
2. Roll a label deterministically 50/50 (yes / no).
3. Generate a message accordingly:
   - **yes**: ask the LLM to write a message that matches the rule.
   - **no, innocent (30%)**: ask for a normal/mundane message.
   - **no, decoy (70%)**: ask for a message that matches a
     *different* rule from the catalog. This forces the model to
     read the rule rather than the surface features of the message.
4. Output a 2-message ChatML conversation:
   user → "Filter rule: ...\n\nMessage:\n..."
   assistant → '{"match": "yes"|"no"}'
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


# 35 generic, user-style spam policies. Worded as "Filter ..." so the
# model gets a consistent imperative. Categories cover sales, phishing,
# urgency, links/attachments, money/crypto, prizes, dating, politics,
# donations, jobs, real estate, support, subscriptions, surface-feature
# rules (caps, exclamations), and identity-based rules (sender / topic).
FILTER_RULES = [
    "Filter messages that want to sell me something",
    "Filter messages asking me to click a link",
    "Filter messages with attachments from senders I don't know",
    "Filter messages that mention cryptocurrency, NFTs, or trading",
    "Filter messages pretending to be from my bank or credit card",
    "Filter messages with urgent deadlines or 'limited time' offers",
    "Filter messages asking for personal information like SSN or password",
    "Filter messages mentioning prizes, sweepstakes, or 'you've won'",
    "Filter unsolicited job offers or recruiter messages",
    "Filter dating-app or romance-related messages",
    "Filter political campaign or fundraising messages",
    "Filter charity donation requests",
    "Filter messages about investing or stock tips",
    "Filter real-estate or property-listing messages",
    "Filter account-verification request messages",
    "Filter life-insurance or extended-warranty offers",
    "Filter loan offers or debt-consolidation pitches",
    "Filter messages mentioning Bitcoin or BTC",
    "Filter automated/no-reply system notifications",
    "Filter messages written entirely in capital letters",
    "Filter messages with three or more exclamation marks",
    "Filter calendar invites and meeting-scheduling messages",
    "Filter messages about food delivery or restaurant offers",
    "Filter travel and flight-deal messages",
    "Filter package-tracking notifications",
    "Filter newsletter and digest emails",
    "Filter subscription-renewal reminders",
    "Filter messages about software updates or system maintenance",
    "Filter messages from coworkers about the current project",
    "Filter messages from my family",
    "Filter messages with phone numbers in the body",
    "Filter messages that contain a coupon or discount code",
    "Filter survey or feedback request messages",
    "Filter messages about home services (plumbing, cleaning, etc.)",
    "Filter messages about medical appointments or prescriptions",
]

# Innocent-message topics for the negative-class anchor. Mundane,
# unambiguous, NOT spammy. The pipeline checks that the chosen rule
# has no overlap with the topic before generating, so we don't end
# up with "innocent message about meetings" while filter rule says
# "filter messages about meetings".
INNOCENT_TOPICS = [
    "a friend asking about weekend hiking plans",
    "a parent reminding their child to call grandma",
    "a roommate saying they finished the laundry",
    "a coworker thanking another for help on a non-project task",
    "a confirmation that a library hold is ready for pickup",
    "a recipe shared between siblings",
    "a dog walker confirming today's walk happened",
    "a quick question about the weather in another city",
    "a friend sharing a song recommendation",
    "someone forwarding a meme they thought was funny",
    "a tutor confirming a session moved an hour later",
    "a neighbor asking to borrow a tool",
    "a gym buddy asking what time the class is",
    "an aunt asking how the new puppy is doing",
    "a quick check-in after a friend's surgery",
    "a kid telling their parent they got home safely",
    "a friend recommending a book they just finished",
]


# Heuristic: which innocent topics overlap with which filter rules?
# Used to skip incompatible (rule, topic) pairs so we don't generate
# an "innocent message about meetings" when the rule is about
# meetings (in which case it would actually be a yes-case).
def _topic_conflicts_with_rule(topic: str, rule: str) -> bool:
    rule_lc = rule.lower()
    topic_lc = topic.lower()
    keywords = [
        ("meeting", "meeting"),
        ("family", "parent"),
        ("family", "aunt"),
        ("family", "kid"),
        ("family", "sibling"),
        ("coworker", "coworker"),
        ("food", "recipe"),
    ]
    for rule_kw, topic_kw in keywords:
        if rule_kw in rule_lc and topic_kw in topic_lc:
            return True
    return False


class SpamFilterV1(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        # Pick the primary filter rule.
        self.filter_rule = random.choice(FILTER_RULES)

        # Roll label 50/50.
        self.label = random.choice(["yes", "no"])

        # For negative class: 30% innocent, 70% decoy (matches a
        # *different* filter). Decoy is the harder case — forces the
        # model to read the rule, not just the surface features.
        self.no_mode: str | None = None
        self.decoy_rule: str | None = None
        self.innocent_topic: str | None = None

        if self.label == "no":
            self.no_mode = random.choices(
                ["innocent", "decoy"], weights=[30, 70],
            )[0]
            if self.no_mode == "innocent":
                # Pick an innocent topic that doesn't accidentally
                # match the filter rule itself.
                topics = [
                    t for t in INNOCENT_TOPICS
                    if not _topic_conflicts_with_rule(t, self.filter_rule)
                ]
                if not topics:
                    topics = INNOCENT_TOPICS
                self.innocent_topic = random.choice(topics)
            else:
                # Decoy: pick a *different* rule and write a message
                # matching it.
                others = [r for r in FILTER_RULES if r != self.filter_rule]
                self.decoy_rule = random.choice(others)

        self.persona = liquidrandom.persona()
        self.seed = (
            f"spam-{self.label}-{self.no_mode or 'pos'}-{self.persona.name}"
        )

        await self._generate_message(client)

        user_prompt = (
            f"Filter rule: {self.filter_rule}\n\n"
            f"Message:\n{self.message}"
        )
        response = json.dumps({"match": self.label})

        return [
            ChatMLMessage("user", user_prompt),
            ChatMLMessage("assistant", response),
        ]

    @step(retries=4)
    async def _generate_message(self, client):
        if self.label == "yes":
            prompt = (
                f"Write a single short message (email, SMS, chat, or DM) that "
                f"clearly matches this filter rule:\n"
                f"\n  {self.filter_rule}\n\n"
                f"The message should look like something a real person would "
                f"actually receive — natural phrasing, not too on-the-nose. "
                f"Length: 1-3 sentences for SMS/chat, or 2-3 short paragraphs "
                f"for email. You can include a Subject line if it's an email. "
                f"Do not include a label, header, or prefix like 'Message:' — "
                f"output ONLY the message text itself."
            )
        elif self.no_mode == "innocent":
            prompt = (
                f"Write a single short, mundane, innocent message about: "
                f"{self.innocent_topic}.\n\n"
                f"It must NOT match this filter rule (so the rule should not "
                f"apply at all):\n"
                f"\n  {self.filter_rule}\n\n"
                f"Length: 1-3 sentences for SMS/chat, or 2-3 short paragraphs "
                f"for email. Output ONLY the message text — no label, no "
                f"prefix, no preamble."
            )
        else:  # decoy
            prompt = (
                f"Write a single short message (email, SMS, chat, or DM) that "
                f"clearly matches this rule:\n"
                f"\n  {self.decoy_rule}\n\n"
                f"It must NOT match this OTHER rule (the message should "
                f"genuinely not apply to it):\n"
                f"\n  {self.filter_rule}\n\n"
                f"Length: 1-3 sentences for SMS/chat, or 2-3 short paragraphs "
                f"for email. Output ONLY the message text — no label, no "
                f"prefix, no preamble."
            )

        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip().strip("`'\"")
        if not text:
            raise GenerationError("Message was empty")
        if len(text) < 20:
            raise GenerationError(f"Message too short ({len(text)} chars)")
        if len(text) > 1500:
            raise GenerationError(f"Message too long ({len(text)} chars)")
        # Reject obvious model preambles.
        for bad in ("Here is", "Here's", "Below is", "Sure,", "Of course"):
            if text.startswith(bad):
                raise GenerationError(
                    f"Message has preamble: {text[:40]!r}"
                )
        self.message = text
