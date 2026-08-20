"""Unit tests for lqh/train/reward.py — the GRPO reward engine.

The invariants under test are the ones a silent failure would make
expensive: group reconstruction (an off-by-group misalignment rewards the
wrong completions), the None-not-zero failure contract (0.0 teaches the
policy that a 429 is a bad answer), ledger replay (a resumed run must not
re-bill the judge), and the failure-window abort (a down reward channel
must stop the run, not starve it).

No GPU, no torch: the engine is exercised with a stubbed judge client.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from lqh.train import reward as reward_mod
from lqh.train.reward import (
    EMPTY_COMPLETION_PENALTY,
    FAILURE_MIN_OBSERVED,
    GrpoRewardEngine,
    RewardChannelDown,
    build_reward_funcs,
    iter_groups,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]


class _StubCompletions:
    """Scripted judge: each call pops the next behaviour off the queue.

    A behaviour is either a string (returned verbatim) or an Exception
    (raised). When the queue is empty the last behaviour repeats.
    """

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return _StubResponse(item)


class _StubClient:
    def __init__(self, script: list[Any]) -> None:
        self.completions_stub = _StubCompletions(script)
        self.chat = type("C", (), {"completions": self.completions_stub})()


def _make_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    script: list[Any],
    num_generations: int = 4,
    config: dict[str, Any] | None = None,
) -> tuple[GrpoRewardEngine, _StubCompletions]:
    monkeypatch.setattr(
        GrpoRewardEngine, "_load_scorer_text", staticmethod(lambda cfg: "Be helpful.")
    )
    client = _StubClient(script)
    monkeypatch.setattr(
        GrpoRewardEngine, "_make_client", staticmethod(lambda: client)
    )
    engine = GrpoRewardEngine(
        tmp_path, config or {"base_model": "LiquidAI/LFM2.5-350M"},
        num_generations=num_generations,
    )
    # Deterministic, retry-free tests.
    engine.max_retries = 0
    return engine, client.completions_stub


class _RankingJudge:
    """Judge stub that reads the candidate ids out of the request's
    structured-output schema and ranks them in presentation order (or
    reversed), so tests get a valid permutation without knowing the
    content hash in advance."""

    def __init__(self, *, reverse: bool = False) -> None:
        self.reverse = reverse
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        schema = kwargs["response_format"]["json_schema"]["schema"]
        ids = list(schema["properties"]["ranking"]["items"]["enum"])
        if self.reverse:
            ids.reverse()
        return _StubResponse(
            json.dumps({"reasoning": "test", "ranking": ids})
        )


def _make_rank_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reverse: bool = False,
    num_generations: int = 4,
) -> tuple[GrpoRewardEngine, _RankingJudge]:
    monkeypatch.setattr(
        GrpoRewardEngine, "_load_scorer_text", staticmethod(lambda cfg: "Be helpful.")
    )
    judge = _RankingJudge(reverse=reverse)
    client = type(
        "Client", (), {"chat": type("C", (), {"completions": judge})()}
    )()
    monkeypatch.setattr(
        GrpoRewardEngine, "_make_client", staticmethod(lambda: client)
    )
    engine = GrpoRewardEngine(
        tmp_path, {"base_model": "LiquidAI/LFM2.5-350M"},
        num_generations=num_generations,
    )
    engine.max_retries = 0
    return engine, judge


PROMPT = [{"role": "user", "content": "Say hi."}]


def _batch(g: int, groups: int = 1) -> tuple[list[Any], list[Any], list[str]]:
    prompts, completions, sample_ids = [], [], []
    for gi in range(groups):
        for i in range(g):
            prompts.append(PROMPT)
            completions.append(f"completion {gi}-{i}")
            sample_ids.append(f"sample-{gi}")
    return prompts, completions, sample_ids


# ---------------------------------------------------------------------------
# iter_groups — the misalignment guard
# ---------------------------------------------------------------------------


def test_iter_groups_contiguous() -> None:
    spans = iter_groups(["a", "a", "b", "b"], 2)
    assert spans == [(0, 2), (2, 4)]


def test_iter_groups_rejects_non_divisible() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        iter_groups(["a", "a", "b"], 2)


def test_iter_groups_rejects_interleaved_groups() -> None:
    # The silent-fatal case: completions from two prompts interleaved.
    with pytest.raises(ValueError, match="contiguity"):
        iter_groups(["a", "b", "a", "b"], 2)


# ---------------------------------------------------------------------------
# Group rank reward
# ---------------------------------------------------------------------------


async def test_rank_rewards_map_back_through_the_shuffle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, judge = _make_rank_engine(tmp_path, monkeypatch)
    prompts, completions, sample_ids = _batch(4)
    rewards = await engine.judge_group_rank(
        prompts, completions, sample_id=sample_ids
    )
    # A full permutation of [0, 1/3, 2/3, 1] must come back, regardless
    # of the presentation shuffle.
    assert sorted(rewards) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert len(judge.calls) == 1  # ONE call per group — the design's point


async def test_rank_reward_orientation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judge's best-ranked candidate gets 1.0. Run the same batch
    through a normal and a reversed judge: each completion's two rewards
    must sum to 1 (rank i and rank G-1-i), proving position maps through
    the shuffle consistently rather than by accident."""
    prompts, completions, sample_ids = _batch(4)
    engine_a, _ = _make_rank_engine(tmp_path / "a", monkeypatch)
    rewards_a = await engine_a.judge_group_rank(
        prompts, completions, sample_id=sample_ids
    )
    engine_b, _ = _make_rank_engine(tmp_path / "b", monkeypatch, reverse=True)
    rewards_b = await engine_b.judge_group_rank(
        prompts, completions, sample_id=sample_ids
    )
    for a, b in zip(rewards_a, rewards_b):
        assert a + b == pytest.approx(1.0)


async def test_rank_failure_returns_none_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = _make_engine(
        tmp_path, monkeypatch, script=[RuntimeError("boom")],
    )
    prompts, completions, sample_ids = _batch(4)
    rewards = await engine.judge_group_rank(
        prompts, completions, sample_id=sample_ids
    )
    assert rewards == [None, None, None, None]
    assert engine.stats["rank_failures"] == 1


async def test_rank_non_object_json_is_a_failure_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the judge occasionally emits a bare JSON array despite
    the schema; that must map to None rewards, not an AttributeError that
    kills the whole batch."""
    engine, _ = _make_engine(
        tmp_path, monkeypatch, script=[json.dumps(["c_a", "c_b"])],
    )
    prompts, completions, sample_ids = _batch(4)
    rewards = await engine.judge_group_rank(
        prompts, completions, sample_id=sample_ids
    )
    assert rewards == [None, None, None, None]


async def test_rank_invalid_permutation_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = _make_engine(
        tmp_path, monkeypatch,
        script=[json.dumps({"reasoning": "x", "ranking": ["c_nope"]})],
    )
    prompts, completions, sample_ids = _batch(4)
    rewards = await engine.judge_group_rank(
        prompts, completions, sample_id=sample_ids
    )
    assert rewards == [None, None, None, None]


async def test_candidate_ids_are_unique_at_g8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: overlapping hash-window ids produced duplicate
    candidate ids in production (a duplicated id fails every ranking for
    the group as 'not a permutation'). Ids must be unique per group."""
    captured: list[list[str]] = []

    class _CaptureJudge:
        async def create(self, **kwargs):
            schema = kwargs["response_format"]["json_schema"]["schema"]
            ids = list(schema["properties"]["ranking"]["items"]["enum"])
            captured.append(ids)
            return _StubResponse(
                json.dumps({"reasoning": "x", "ranking": ids})
            )

    monkeypatch.setattr(
        GrpoRewardEngine, "_load_scorer_text", staticmethod(lambda cfg: "s")
    )
    judge = _CaptureJudge()
    client = type("C1", (), {"chat": type("C2", (), {"completions": judge})()})()
    monkeypatch.setattr(
        GrpoRewardEngine, "_make_client", staticmethod(lambda: client)
    )
    engine = GrpoRewardEngine(
        tmp_path, {"base_model": "x"}, num_generations=8,
    )
    for i in range(50):
        texts = [f"t{i}-{j}" for j in range(8)]
        await engine._rank_group(sample_id=f"s{i}", prompt=PROMPT, texts=texts)
    assert len(captured) == 50
    for ids in captured:
        assert len(set(ids)) == 8, f"duplicate candidate ids: {ids}"


async def test_ledger_replay_skips_judge_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, judge = _make_rank_engine(tmp_path, monkeypatch)
    prompts, completions, sample_ids = _batch(4)
    first = await engine.judge_group_rank(prompts, completions, sample_id=sample_ids)
    second = await engine.judge_group_rank(prompts, completions, sample_id=sample_ids)
    assert first == second
    assert len(judge.calls) == 1  # replayed from the ledger, not re-billed
    assert engine.stats["rank_cache_hits"] == 1
    records = list((tmp_path / "rewards").glob("rank_*.json"))
    assert len(records) == 1
    payload = json.loads(records[0].read_text())
    assert payload["kind"] == "group_rank"
    assert sorted(r for r in payload["rewards"]) == pytest.approx(
        [0.0, 1 / 3, 2 / 3, 1.0]
    )


async def test_failed_ledger_records_are_retried_not_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = _make_engine(tmp_path, monkeypatch, script=[RuntimeError("boom")])
    prompts, completions, sample_ids = _batch(4)
    await engine.judge_group_rank(prompts, completions, sample_id=sample_ids)
    # Same content, judge now healthy → must call again, not replay None.
    engine2, judge2 = _make_rank_engine(tmp_path, monkeypatch)
    rewards = await engine2.judge_group_rank(
        prompts, completions, sample_id=sample_ids
    )
    assert None not in rewards
    assert len(judge2.calls) == 1


async def test_failure_window_aborts_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = _make_engine(
        tmp_path, monkeypatch, script=[RuntimeError("down")], num_generations=4,
    )
    prompts, completions, sample_ids = _batch(4, groups=1)
    with pytest.raises(RewardChannelDown):
        # Enough consecutive all-failed groups to cross the observed
        # minimum and the abort fraction.
        for i in range(FAILURE_MIN_OBSERVED // 4 + 1):
            ids = [f"s{i}"] * 4
            await engine.judge_group_rank(prompts, completions, sample_id=ids)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_guard_penalizes_empty_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = _make_engine(tmp_path, monkeypatch, script=["unused"])
    prompts, completions, sample_ids = _batch(4)
    completions[2] = "   "
    penalties = engine.guard_penalties(prompts, completions, sample_id=sample_ids)
    assert penalties[2] == EMPTY_COMPLETION_PENALTY
    assert penalties[0] == 0.0
    assert all(p <= 0 for p in penalties)  # penalties only, never rewards


# ---------------------------------------------------------------------------
# build_reward_funcs
# ---------------------------------------------------------------------------


def test_build_reward_funcs_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        GrpoRewardEngine, "_load_scorer_text", staticmethod(lambda cfg: "criteria")
    )
    monkeypatch.setattr(
        GrpoRewardEngine, "_make_client", staticmethod(lambda: object())
    )
    funcs, weights, engine = build_reward_funcs(
        tmp_path, {"base_model": "x"}, num_generations=8,
    )
    assert len(funcs) == len(weights) == 3
    assert weights[0] == 1.0 and weights[1] == 0.2
    # TRL detects async reward funcs via inspect.iscoroutinefunction.
    import inspect

    assert inspect.iscoroutinefunction(funcs[0])
    assert inspect.iscoroutinefunction(funcs[1])
    assert not inspect.iscoroutinefunction(funcs[2])


def test_zero_weight_skips_judge_funcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ablation arms zero out a judge's weight; the func must be skipped
    entirely so the arm doesn't bill for signal it cannot use."""
    monkeypatch.setattr(
        GrpoRewardEngine, "_load_scorer_text", staticmethod(lambda cfg: "s")
    )
    monkeypatch.setattr(
        GrpoRewardEngine, "_make_client", staticmethod(lambda: object())
    )
    funcs, weights, _ = build_reward_funcs(
        tmp_path,
        {"base_model": "x",
         "reward": {"rank_weight": 0, "absolute_weight": 1.0}},
        num_generations=8,
    )
    names = [f.__name__ for f in funcs]
    assert names == ["judge_absolute", "guard_penalties"]
    funcs, weights, _ = build_reward_funcs(
        tmp_path / "g",
        {"base_model": "x",
         "reward": {"rank_weight": 0, "absolute_weight": 0}},
        num_generations=8,
    )
    assert [f.__name__ for f in funcs] == ["guard_penalties"]


def test_missing_scorer_is_a_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        GrpoRewardEngine, "_make_client", staticmethod(lambda: object())
    )
    monkeypatch.chdir(tmp_path)  # no scorers/ here
    with pytest.raises(ValueError, match="scorer"):
        GrpoRewardEngine(tmp_path, {"base_model": "x"}, num_generations=4)
