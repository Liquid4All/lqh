"""Unit tests for the voice_satisfaction gold validator and the transformers
v4/v5 preflight verdict.

The gold output is LLM-generated and gated by ``validate_output`` (the
in-pipeline first line of defence before the run-level scorer filter). These
tests pin the SPEC's hard rules: exact field set, score range, allowed tags,
turn-number bounds, and the score↔tag coupling. The preflight tests pin the
pure compatibility decision so the model-mix guard can be trusted offline.
"""

from __future__ import annotations


import pytest

from lqh.pipeline import GenerationError
from tests.benchmarks.base_vs_instruct.pipelines.voice_satisfaction import (
    _extract_json,
    validate_output,
)
from tests.benchmarks.base_vs_instruct.preflight import verdict


def _good(**overrides) -> dict:
    base = {
        "reasoning": "The assistant misheard living room as kitchen, forcing the "
        "user to correct it in turn one before complying.",
        "score": 2,
        "failure_tags": ["wrong_entity", "user_correction"],
        "success_tags": ["graceful_recovery"],
        "failed_turns": [1],
        "successful_turns": [2],
    }
    base.update(overrides)
    return base


class TestValidateOutput:
    def test_compliant_passes(self) -> None:
        validate_output(_good(), num_turns=2)  # must not raise

    def test_smooth_success_passes(self) -> None:
        validate_output(
            _good(
                reasoning="The assistant set the ten-minute timer correctly on the "
                "first try with no issues at all.",
                score=5, failure_tags=[], success_tags=["success"],
                failed_turns=[], successful_turns=[1],
            ),
            num_turns=1,
        )

    def test_missing_field_rejected(self) -> None:
        d = _good()
        del d["score"]
        with pytest.raises(GenerationError, match="missing fields"):
            validate_output(d, num_turns=2)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(GenerationError, match="unexpected fields"):
            validate_output(_good(device="speaker"), num_turns=2)

    @pytest.mark.parametrize("bad", [0, 6, 2.5, "2", True])
    def test_bad_score_rejected(self, bad) -> None:
        with pytest.raises(GenerationError, match="invalid score"):
            validate_output(_good(score=bad), num_turns=2)

    def test_short_reasoning_rejected(self) -> None:
        with pytest.raises(GenerationError, match="reasoning"):
            validate_output(_good(reasoning="bad timer"), num_turns=2)

    def test_unknown_failure_tag_rejected(self) -> None:
        with pytest.raises(GenerationError, match="invalid failure_tags"):
            validate_output(_good(failure_tags=["made_up"]), num_turns=2)

    def test_unknown_success_tag_rejected(self) -> None:
        with pytest.raises(GenerationError, match="invalid success_tags"):
            validate_output(
                _good(score=5, failure_tags=[], success_tags=["amazing"]),
                num_turns=2,
            )

    def test_out_of_range_turn_rejected(self) -> None:
        with pytest.raises(GenerationError, match="out-of-range turn"):
            validate_output(_good(failed_turns=[5]), num_turns=2)

    def test_low_score_requires_failure_tag(self) -> None:
        with pytest.raises(GenerationError, match="failure tag"):
            validate_output(
                _good(score=2, failure_tags=[], success_tags=["success"]),
                num_turns=2,
            )

    def test_high_score_requires_success_tag(self) -> None:
        with pytest.raises(GenerationError, match="success tag"):
            validate_output(
                _good(score=5, failure_tags=[], success_tags=[],
                      failed_turns=[], successful_turns=[1]),
                num_turns=2,
            )

    def test_canceled_neutral_passes(self) -> None:
        # Score 3 ambiguity is encoded as a `canceled` failure tag.
        validate_output(
            _good(
                reasoning="The assistant gave a correct forecast, then the user "
                "said never mind and would check their phone instead.",
                score=3, failure_tags=["canceled"], success_tags=["success"],
                failed_turns=[], successful_turns=[1],
            ),
            num_turns=1,
        )


class TestExtractJson:
    def test_plain(self) -> None:
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self) -> None:
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_invalid_raises(self) -> None:
        with pytest.raises(GenerationError, match="invalid JSON"):
            _extract_json("not json at all")


class TestPreflightVerdict:
    def test_all_good(self) -> None:
        ok, notes = verdict(
            config_loaded=True, model_type="lfm2", causal_lm_ok=True,
            chat_template_ok=True, load_error=None,
        )
        assert ok and notes == []

    def test_config_failed_is_incompatible(self) -> None:
        ok, notes = verdict(
            config_loaded=False, model_type=None, causal_lm_ok=False,
            chat_template_ok=False, load_error="KeyError: lfm2",
        )
        assert not ok
        assert any("architecture" in n for n in notes)

    def test_unknown_architecture_incompatible(self) -> None:
        ok, notes = verdict(
            config_loaded=True, model_type="lfm9", causal_lm_ok=False,
            chat_template_ok=True, load_error=None,
        )
        assert not ok
        assert any("causal-LM mapping" in n for n in notes)

    def test_missing_chat_template_incompatible(self) -> None:
        ok, notes = verdict(
            config_loaded=True, model_type="lfm2", causal_lm_ok=True,
            chat_template_ok=False, load_error=None,
        )
        assert not ok
        assert any("chat template" in n for n in notes)
