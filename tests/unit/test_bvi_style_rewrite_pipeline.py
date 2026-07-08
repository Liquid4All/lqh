"""Unit tests for the style_rewrite gold validator.

The reference rewrite is now LLM-generated and gated by ``validate_rewrite``
(the in-pipeline first line of defence before the run-level scorer filter).
These tests pin the rejection rules that the old canned-template gold violated.
"""

from __future__ import annotations

import pytest

from lqh.pipeline import GenerationError
from tests.benchmarks.base_vs_instruct.pipelines.style_rewrite import validate_rewrite

TONE = "friendly, precise, and action-oriented"
FACTS = ["credit has been added to the account", "photos are no longer needed"]
BANNED = ["as per our policy", "thank you for your patience"]
DRAFT = (
    "As per our policy. Regarding a damaged grocery delivery, please be advised "
    "that credit has been added to the account. Also, photos are no longer needed."
)

# A compliant rewrite: ~67 words, both facts verbatim, no banned phrase, no
# greeting/preamble/sign-off, no tone echo, no meta-commentary.
GOOD = (
    "Your credit has been added to the account, and photos are no longer needed "
    "for this delivery issue. Everything is now settled on our side, so there is "
    "nothing more you need to do at this point. If any detail still looks wrong, "
    "or you would like a quick summary emailed over, just reply here and we will "
    "sort it out right away for you."
)


def _validate(text: str) -> None:
    validate_rewrite(text, tone=TONE, facts=FACTS, banned=BANNED, draft=DRAFT)


class TestValidateRewrite:
    def test_compliant_rewrite_passes(self) -> None:
        _validate(GOOD)  # must not raise

    def test_empty_rejected(self) -> None:
        with pytest.raises(GenerationError):
            _validate("")

    def test_banned_phrase_rejected(self) -> None:
        with pytest.raises(GenerationError, match="banned"):
            _validate(GOOD + " As per our policy, nothing else is required here.")

    def test_reworded_fact_accepted(self) -> None:
        # A stylistic rewrite rephrases facts (tense/articles/word order) — this
        # must pass; only a genuine content drop should be rejected.
        text = (
            "Good news — a credit has now been added to your account, and you no "
            "longer need to send photos for this delivery issue. Everything is "
            "settled on our side, so there is nothing further to do right now. If "
            "any detail still looks off, just reply here and the team will sort "
            "it out quickly for you today without any extra steps required."
        )
        _validate(text)  # must not raise

    def test_dropped_fact_rejected(self) -> None:
        text = GOOD.replace("photos are no longer needed", "no further pictures needed")
        with pytest.raises(GenerationError, match="required fact"):
            _validate(text)

    def test_too_short_rejected(self) -> None:
        with pytest.raises(GenerationError, match="length"):
            _validate(
                "Credit has been added to the account and photos are no longer needed."
            )

    def test_greeting_preamble_rejected(self) -> None:
        with pytest.raises(GenerationError, match="greeting/preamble"):
            _validate("Here is the current update: " + GOOD)

    def test_tone_echo_rejected(self) -> None:
        text = (
            "Your credit has been added to the account, and photos are no longer "
            f"needed. I will keep this {TONE} and make sure the next step is set "
            "so you do not need to send anything else unless these details look "
            "wrong to you in any meaningful way at all going forward from today."
        )
        with pytest.raises(GenerationError):
            _validate(text)

    def test_meta_commentary_rejected(self) -> None:
        text = (
            "Your credit has been added to the account, and photos are no longer "
            "needed. I will keep this short and clear so the next step is already "
            "set, and you do not need to send anything else unless these specific "
            "details somehow look wrong to you when you review them again later."
        )
        with pytest.raises(GenerationError, match="meta-commentary"):
            _validate(text)

    def test_draft_copy_rejected(self) -> None:
        # Candidate identical to the draft must be rejected. Use GOOD as the
        # "draft" so it clears every other gate and reaches the copy check.
        with pytest.raises(GenerationError, match="identical"):
            validate_rewrite(GOOD, tone=TONE, facts=FACTS, banned=BANNED, draft=GOOD)
