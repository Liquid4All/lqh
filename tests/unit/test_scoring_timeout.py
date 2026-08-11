"""Per-sample deadlines, the retry ladder, and fail-open filtering.

Feedback #44: one wedged sample used to hold a whole scoring run hostage —
three outer attempts times three invisible SDK replays times the 270 s
upstream budget is ~40 minutes for a single straggler, and ``run_scoring``
writes nothing until every task has returned, so Ctrl-C wasn't a way out
either. These tests pin the three things that fixed it:

* each sample is bounded by a deadline that covers its retries,
* the deadline starts *after* the semaphore, so queueing is not charged to
  the sample,
* a judge failure in ``run_data_filter`` keeps the row instead of deleting
  user-brought data.

All offline — the judge is a mock.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pyarrow.parquet as pq
import pytest
from openai import APIConnectionError, RateLimitError

from lqh.runner import RunnerResponse
from lqh.scoring import (
    DEFAULT_MAX_RETRIES,
    _RATE_LIMIT_WAIT_CAP_S,
    _rate_limit_wait,
    FAILURE_WARN_FRACTION,
    _MAX_RATE_LIMIT_WAITS,
    failure_warning,
    is_scoring_error,
    run_data_filter,
    run_data_scoring,
    run_scoring,
)

CONVERSATION = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]


def _judge_client(
    *, delay: float = 0.0, content: str = '{"reasoning": "Good", "score": 8}',
    raises: BaseException | None = None,
) -> MagicMock:
    """Mock judge client. *delay* simulates a slow (or wedged) upstream."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]

    async def _create(**_kwargs: Any) -> Any:
        if delay:
            await asyncio.sleep(delay)
        if raises is not None:
            raise raises
        return response

    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


@pytest.fixture
def scorer(tmp_path: Path) -> Path:
    path = tmp_path / "scorer.md"
    path.write_text("Score for correctness, 1-10.")
    return path


# ---------------------------------------------------------------------------
# failure_warning
# ---------------------------------------------------------------------------


class TestFailureWarning:
    def test_silent_when_nothing_failed(self) -> None:
        assert failure_warning(0, 100) is None

    def test_silent_at_or_below_the_threshold(self) -> None:
        # Exactly 10% of 100 is not "above 10%".
        assert failure_warning(int(100 * FAILURE_WARN_FRACTION), 100) is None

    def test_warns_above_the_threshold(self) -> None:
        warning = failure_warning(11, 100)
        assert warning is not None
        assert "11/100" in warning
        assert "11%" in warning

    def test_empty_run_has_nothing_to_warn_about(self) -> None:
        assert failure_warning(0, 0) is None
        assert failure_warning(3, 0) is None


# ---------------------------------------------------------------------------
# run_scoring
# ---------------------------------------------------------------------------


class TestRunScoringDeadline:
    async def test_wedged_sample_is_dropped_not_waited_out(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """A judge that never answers costs the run its deadline, not forever."""
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=3,
        )
        client = _judge_client(delay=30.0)

        result = await asyncio.wait_for(
            run_scoring(
                dataset_path=dataset,
                scorer_path=scorer,
                output_dir=tmp_path / "out",
                client=client,
                sample_timeout=0.2,
            ),
            timeout=10,
        )

        assert result.total == 3
        assert result.scored == 0
        assert result.failed == 3
        # Every sample still has a row, flagged so it stays out of the mean.
        table = pq.read_table(tmp_path / "out" / "results.parquet")
        assert len(table) == 3
        for reasoning in table.column("reasoning").to_pylist():
            assert is_scoring_error(reasoning)
            assert "timed out" in reasoning
        assert result.mean_score == 0.0

        summary = json.loads((tmp_path / "out" / "summary.json").read_text())
        assert summary["num_failed"] == 3

    async def test_deadline_starts_after_the_semaphore(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """Queueing behind concurrency must not be charged to a sample.

        Four samples through one slot at 0.5 s each: the last one starts at
        t=1.5 s, so a clock running from task creation would blow the 1.2 s
        deadline by 0.8 s. Held-slot time is only 0.5 s against that same
        deadline, so the correct behaviour has 2.4x of margin and the broken
        one fails outright — neither verdict rides on a scheduling hiccup.
        """
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=4,
        )
        client = _judge_client(delay=0.5)

        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=client,
            concurrency=1,
            sample_timeout=1.2,
        )

        assert result.scored == 4
        assert result.failed == 0

    async def test_healthy_run_is_untouched_by_the_deadline(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=2,
        )
        client = _judge_client()

        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=client,
        )

        assert result.scored == 2
        assert result.failed == 0
        assert result.mean_score == 8.0
        # One call per sample: a healthy judge is never retried.
        assert client.chat.completions.create.await_count == 2

    async def test_non_positive_timeout_disables_the_bound(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=1,
        )
        client = _judge_client(delay=0.05)

        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=client,
            sample_timeout=0,
        )

        assert result.scored == 1


class TestRetryLadder:
    async def test_default_is_two_attempts_per_sample(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """One retry absorbs a glitch; the second failure ends the sample.

        Guards the ladder against creeping back up — every extra attempt is
        another upstream budget the user waits through.
        """
        assert DEFAULT_MAX_RETRIES == 1
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=2,
        )
        client = _judge_client(raises=RuntimeError("upstream 504"))

        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=client,
        )

        assert result.failed == 2
        assert client.chat.completions.create.await_count == 2 * (DEFAULT_MAX_RETRIES + 1)
        table = pq.read_table(tmp_path / "out" / "results.parquet")
        for reasoning in table.column("reasoning").to_pylist():
            assert "upstream 504" in reasoning

    async def test_rate_limits_do_not_spend_the_retry_budget(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet, monkeypatch,
    ) -> None:
        """A 429 is the server pacing us, not a failed attempt.

        The SDK's retry layer is off for scoring (it multiplies the deadline),
        so this is the only thing left honouring Retry-After. With a 2-attempt
        ladder, treating 429s as failures would fail most of a run the moment
        100-wide concurrency meets a per-minute limit.
        """
        monkeypatch.setattr(
            "lqh.scoring._rate_limit_wait", lambda *a, **kw: 0.01
        )
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=1,
        )

        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content='{"reasoning": "ok", "score": 7}'))
        ]
        calls = {"n": 0}

        async def _create(**_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] <= 3:  # more 429s than the ladder has attempts
                raise RateLimitError(
                    "rate limited",
                    response=httpx.Response(
                        429, request=httpx.Request("POST", "http://x")
                    ),
                    body=None,
                )
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=client,
        )

        assert result.scored == 1
        assert result.failed == 0
        assert calls["n"] == 4

    async def test_rate_limits_are_not_capped_while_a_deadline_is_running(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet, monkeypatch,
    ) -> None:
        """Under a deadline, the deadline decides when to give up — not a
        wait counter. Capping here would fail a sample the server was about
        to serve, with time still on the clock."""
        monkeypatch.setattr(
            "lqh.scoring._rate_limit_wait", lambda *a, **kw: 0.01
        )
        dataset = write_chatml_parquet(tmp_path / "data.parquet", [CONVERSATION])

        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content='{"reasoning": "ok", "score": 7}'))
        ]
        calls = {"n": 0}
        limit = _MAX_RATE_LIMIT_WAITS + 1  # one more 429 than the cap allows

        async def _create(**_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] <= limit:
                raise RateLimitError(
                    "rate limited",
                    response=httpx.Response(
                        429, request=httpx.Request("POST", "http://x")
                    ),
                    body=None,
                )
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=client,
            sample_timeout=10,
        )

        assert result.scored == 1
        assert calls["n"] == limit + 1

    async def test_rate_limit_waits_are_capped(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet, monkeypatch,
    ) -> None:
        """An endless 429 still ends: the cap keeps a sample from spinning
        when the deadline is disabled."""
        monkeypatch.setattr(
            "lqh.scoring._rate_limit_wait", lambda *a, **kw: 0.01
        )
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=1,
        )

        async def _create(**_kwargs: Any) -> Any:
            raise RateLimitError(
                "rate limited",
                response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
                body=None,
            )

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=client,
            sample_timeout=0,  # no deadline — the cap is the only bound
        )

        assert result.failed == 1
        assert client.chat.completions.create.await_count == _MAX_RATE_LIMIT_WAITS + 1

    async def test_model_eval_timeout_keeps_the_model_output(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """When the judge times out, the row must hold what was judged — the
        model's own answer — not the gold reference it was compared against."""
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=1,
        )
        runner = MagicMock()
        runner.complete = AsyncMock(
            return_value=RunnerResponse(content="MODEL ANSWER", model="m", usage=None)
        )

        result = await asyncio.wait_for(
            run_scoring(
                dataset_path=dataset,
                scorer_path=scorer,
                output_dir=tmp_path / "out",
                client=_judge_client(delay=30.0),
                run_inference=True,
                inference_model="m",
                inference_runner=runner,
                sample_timeout=0.2,
            ),
            timeout=10,
        )

        assert result.failed == 1
        table = pq.read_table(tmp_path / "out" / "results.parquet")
        stored = json.loads(table.column("messages").to_pylist()[0])
        assert stored[-1] == {"role": "assistant", "content": "MODEL ANSWER"}

    async def test_inference_timeout_never_stores_the_gold_answer(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """A hang in *inference* must not record the reference as output.

        The sample's context is published before inference runs, precisely so
        the timeout path has something truthful to fall back to. Falling back
        to the input conversation would write the gold assistant turn into
        results.parquet as though the model had produced it.
        """
        gold = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "GOLD"},
        ]
        dataset = write_chatml_parquet(tmp_path / "data.parquet", [gold])

        runner = MagicMock()

        async def _hang(*_args: Any, **_kwargs: Any) -> Any:
            await asyncio.sleep(30.0)

        runner.complete = AsyncMock(side_effect=_hang)

        result = await asyncio.wait_for(
            run_scoring(
                dataset_path=dataset,
                scorer_path=scorer,
                output_dir=tmp_path / "out",
                client=_judge_client(),
                run_inference=True,
                inference_model="m",
                inference_runner=runner,
                sample_timeout=0.2,
                debug=True,
            ),
            timeout=10,
        )

        assert result.failed == 1
        stored = json.loads(
            pq.read_table(tmp_path / "out" / "results.parquet")
            .column("messages")
            .to_pylist()[0]
        )
        assert all(m.get("content") != "GOLD" for m in stored), stored
        assert not any(m.get("role") == "assistant" for m in stored)

        # And it is still debuggable: the call that never answered is replayable.
        entries = [
            json.loads(line)
            for line in (tmp_path / "out" / "debug_low_scores.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(entries) == 1
        assert entries[0]["model_response"] is None
        assert entries[0]["inference_messages_sent"]

    async def test_transient_inference_failure_is_retried(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """Inference gets the same ladder the judge does.

        Scoring clients run with the SDK's retry layer off, so without this
        one dropped connection fails the sample outright with most of the
        deadline unspent.
        """
        dataset = write_chatml_parquet(tmp_path / "data.parquet", [CONVERSATION])
        calls = {"n": 0}

        async def _flaky(*_args: Any, **_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("connection reset")
            return RunnerResponse(content="ANSWER", model="m", usage=None)

        runner = MagicMock()
        runner.complete = AsyncMock(side_effect=_flaky)

        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=_judge_client(),
            run_inference=True,
            inference_model="m",
            inference_runner=runner,
        )

        assert result.scored == 1
        assert result.failed == 0
        assert calls["n"] == 2

    async def test_model_eval_timeout_is_debuggable(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """The sample that hung is the one a user most wants to replay."""
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=1,
        )
        runner = MagicMock()
        runner.complete = AsyncMock(
            return_value=RunnerResponse(content="MODEL ANSWER", model="m", usage=None)
        )

        await asyncio.wait_for(
            run_scoring(
                dataset_path=dataset,
                scorer_path=scorer,
                output_dir=tmp_path / "out",
                client=_judge_client(delay=30.0),
                run_inference=True,
                inference_model="m",
                inference_runner=runner,
                sample_timeout=0.2,
                debug=True,
            ),
            timeout=10,
        )

        entries = [
            json.loads(line)
            for line in (tmp_path / "out" / "debug_low_scores.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(entries) == 1
        assert entries[0]["model_response"] == "MODEL ANSWER"
        assert "timed out" in entries[0]["reasoning"]
        assert list((tmp_path / "out" / "curl_debug").glob("*.sh"))

    async def test_retries_share_the_sample_deadline(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """The bound covers the sample, not each attempt.

        This is the whole point: a per-attempt bound multiplies by the
        ladder, a per-sample bound does not.
        """
        dataset = write_chatml_parquet(
            tmp_path / "data.parquet", [CONVERSATION], num=1,
        )
        client = _judge_client(delay=5.0)

        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await run_scoring(
            dataset_path=dataset,
            scorer_path=scorer,
            output_dir=tmp_path / "out",
            client=client,
            max_retries=3,
            sample_timeout=0.3,
        )
        elapsed = loop.time() - started

        assert result.failed == 1
        # 4 attempts x 5 s would be 20 s if the deadline were per-attempt.
        assert elapsed < 3.0


# ---------------------------------------------------------------------------
# run_data_scoring
# ---------------------------------------------------------------------------


class TestDataScoringDeadline:
    async def test_wedged_sample_lands_in_scores_parquet_as_an_error(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        dataset_dir = tmp_path / "ds"
        write_chatml_parquet(
            dataset_dir / "data.parquet", [CONVERSATION], num=2,
        )
        client = _judge_client(delay=30.0)

        result = await asyncio.wait_for(
            run_data_scoring(
                dataset_dir=dataset_dir,
                scorer_path=scorer,
                client=client,
                sample_timeout=0.2,
            ),
            timeout=10,
        )

        assert result.scored == 0
        assert result.failed == 2
        table = pq.read_table(dataset_dir / "scores.parquet")
        assert table.column("sample_index").to_pylist() == [0, 1]
        for reasoning in table.column("reasoning").to_pylist():
            assert is_scoring_error(reasoning)

    async def test_rows_stay_in_sample_order_when_some_time_out(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """Mixed run: slow samples must not reorder the scores parquet."""
        dataset_dir = tmp_path / "ds"
        write_chatml_parquet(
            dataset_dir / "data.parquet", [CONVERSATION], num=4,
        )

        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content='{"reasoning": "ok", "score": 7}'))
        ]
        calls = {"n": 0}

        async def _create(**_kwargs: Any) -> Any:
            # Every other sample wedges.
            mine = calls["n"]
            calls["n"] += 1
            if mine % 2 == 0:
                await asyncio.sleep(30.0)
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        result = await asyncio.wait_for(
            run_data_scoring(
                dataset_dir=dataset_dir,
                scorer_path=scorer,
                client=client,
                concurrency=1,
                sample_timeout=0.2,
            ),
            timeout=20,
        )

        assert result.total == 4
        assert result.scored + result.failed == 4
        table = pq.read_table(dataset_dir / "scores.parquet")
        assert table.column("sample_index").to_pylist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# run_data_filter — fail open
# ---------------------------------------------------------------------------


class TestDataFilterFailsOpen:
    async def test_judge_error_keeps_the_row(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """User-brought data is not deleted because the judge had a bad minute."""
        input_path = write_chatml_parquet(
            tmp_path / "in.parquet", [CONVERSATION], num=3,
        )
        client = _judge_client(raises=RuntimeError("upstream 502"))

        result = await run_data_filter(
            input_path=input_path,
            scorer_path=scorer,
            output_dataset_dir=tmp_path / "out",
            client=client,
        )

        assert result.total == 3
        assert result.failed == 3
        assert result.kept == 3
        assert result.kept_unjudged == 3
        assert result.dropped == 0
        # The rows survive into the emitted dataset, not just the score table.
        assert len(pq.read_table(tmp_path / "out" / "data.parquet")) == 3
        assert all(pq.read_table(tmp_path / "out" / "scores.parquet").column("kept").to_pylist())

        summary = json.loads((tmp_path / "out" / "summary.json").read_text())
        assert summary["kept_unjudged"] == 3
        assert summary["kept"] == 3
        assert summary["dropped"] == 0

    async def test_timeout_keeps_the_row(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        input_path = write_chatml_parquet(
            tmp_path / "in.parquet", [CONVERSATION], num=2,
        )
        client = _judge_client(delay=30.0)

        result = await asyncio.wait_for(
            run_data_filter(
                input_path=input_path,
                scorer_path=scorer,
                output_dataset_dir=tmp_path / "out",
                client=client,
                sample_timeout=0.2,
            ),
            timeout=10,
        )

        assert result.kept == 2
        assert result.kept_unjudged == 2
        assert len(pq.read_table(tmp_path / "out" / "data.parquet")) == 2

    async def test_threshold_still_drops_judged_rows(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """Fail-open applies to unjudged rows only — a low score still drops."""
        input_path = write_chatml_parquet(
            tmp_path / "in.parquet", [CONVERSATION], num=4,
        )

        scores = iter([9, 2, 9, 2])

        async def _create(**_kwargs: Any) -> Any:
            response = MagicMock()
            response.choices = [
                MagicMock(
                    message=MagicMock(
                        content=json.dumps({"reasoning": "ok", "score": next(scores)})
                    )
                )
            ]
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        result = await run_data_filter(
            input_path=input_path,
            scorer_path=scorer,
            output_dataset_dir=tmp_path / "out",
            client=client,
            threshold=6.0,
            concurrency=1,
        )

        assert result.scored == 4
        assert result.failed == 0
        assert result.kept == 2
        assert result.dropped == 2
        assert result.kept_unjudged == 0

    async def test_mixed_run_counts_add_up(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """kept + dropped == total, with the unjudged rows inside kept."""
        input_path = write_chatml_parquet(
            tmp_path / "in.parquet", [CONVERSATION], num=3,
        )

        outcomes = iter([9, "boom", 2])

        async def _create(**_kwargs: Any) -> Any:
            nxt = next(outcomes)
            if nxt == "boom":
                raise RuntimeError("upstream 502")
            response = MagicMock()
            response.choices = [
                MagicMock(
                    message=MagicMock(
                        content=json.dumps({"reasoning": "ok", "score": nxt})
                    )
                )
            ]
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        result = await run_data_filter(
            input_path=input_path,
            scorer_path=scorer,
            output_dataset_dir=tmp_path / "out",
            client=client,
            threshold=6.0,
            concurrency=1,
            max_retries=0,
        )

        assert result.kept + result.dropped == result.total
        assert result.kept == 2          # the 9 and the unjudged one
        assert result.dropped == 1       # the 2
        assert result.kept_unjudged == 1
        assert result.failed == 1
        assert result.mean_score == 5.5  # (9 + 2) / 2 — the error is excluded


# ---------------------------------------------------------------------------
# Guards the review found untested
# ---------------------------------------------------------------------------


class TestRateLimitWaitClamping:
    """The wait length itself, separately from the wait *count*."""

    @staticmethod
    def _exc(retry_after: str | None) -> RateLimitError:
        headers = {"retry-after": retry_after} if retry_after else {}
        return RateLimitError(
            "rate limited",
            response=httpx.Response(
                429, headers=headers, request=httpx.Request("POST", "http://x")
            ),
            body=None,
        )

    def test_unbounded_clamps_to_the_cap(self) -> None:
        wait = _rate_limit_wait(self._exc("600"), 1, bounded=False)
        assert wait == _RATE_LIMIT_WAIT_CAP_S

    def test_bounded_honours_retry_after(self) -> None:
        # Not clamped down to some smaller constant: retrying earlier than the
        # server asked just earns another 429.
        assert _rate_limit_wait(self._exc("60"), 1, bounded=True, remaining=120.0) == 60.0

    def test_bounded_never_sleeps_past_the_deadline(self) -> None:
        # Sleeping past your own deadline converts time that could have held
        # an attempt into a guaranteed timeout.
        assert _rate_limit_wait(self._exc("600"), 1, bounded=True, remaining=25.0) == 25.0
        assert _rate_limit_wait(self._exc("600"), 1, bounded=True, remaining=-5.0) == 0.0

    def test_missing_header_falls_back_to_backoff(self) -> None:
        assert _rate_limit_wait(self._exc(None), 3, bounded=True, remaining=100.0) == 4.0


class TestConcurrencyGuard:
    async def test_zero_concurrency_does_not_hang(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        """A zero-capacity semaphore is never acquired, and the deadline only
        starts after acquisition — so this used to hang with nothing to stop it."""
        dataset = write_chatml_parquet(tmp_path / "data.parquet", [CONVERSATION])
        result = await asyncio.wait_for(
            run_scoring(
                dataset_path=dataset,
                scorer_path=scorer,
                output_dir=tmp_path / "out",
                client=_judge_client(),
                concurrency=0,
            ),
            timeout=10,
        )
        assert result.scored == 1

    async def test_negative_concurrency_does_not_raise(
        self, tmp_path: Path, scorer: Path, write_chatml_parquet,
    ) -> None:
        dataset = write_chatml_parquet(tmp_path / "data.parquet", [CONVERSATION])
        result = await asyncio.wait_for(
            run_scoring(
                dataset_path=dataset,
                scorer_path=scorer,
                output_dir=tmp_path / "out",
                client=_judge_client(),
                concurrency=-5,
            ),
            timeout=10,
        )
        assert result.scored == 1


class TestPerSourceFailureCounts:
    """A source scored on 3 of its 100 rows must not look like a complete one."""

    async def test_per_source_carries_attempted_and_failed(
        self, tmp_path: Path,
    ) -> None:
        import pyarrow as pa

        from lqh.scoring import score_predictions_by_source

        # 40 predictions across two sources; the judge fails on all of source b.
        n = 40
        rows = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        preds = tmp_path / "preds.parquet"
        pa_table = pa.table({
            "sample_index": list(range(n)),
            "messages": [json.dumps(rows)] * n,
            "source": ["a" if i < 20 else "b" for i in range(n)],
        })
        pq.write_table(pa_table, preds)
        scorer_path = tmp_path / "s.md"
        scorer_path.write_text("score it")

        calls = {"n": 0}

        async def _create(**kwargs: Any) -> Any:
            # Deterministic by call order is not safe under concurrency, so
            # key off the prompt: source b's rows are the last 20 indices and
            # indistinguishable by content, so fail a fixed *count* instead.
            calls["n"] += 1
            if calls["n"] > 20:
                raise RuntimeError("judge down")
            response = MagicMock()
            response.choices = [
                MagicMock(message=MagicMock(content='{"reasoning":"ok","score":7}'))
            ]
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        payload = await score_predictions_by_source(
            predictions_path=preds,
            scorer_path=scorer_path,
            output_dir=tmp_path / "out",
            client=client,
            concurrency=1,
            max_retries=0,
        )

        per_source = payload["per_source"]
        assert per_source["a"]["num_attempted"] == 20
        assert per_source["b"]["num_attempted"] == 20
        # Every source reports what it lost, whatever the split happened to be.
        for label in ("a", "b"):
            entry = per_source[label]
            assert entry["num_failed"] == entry["num_attempted"] - entry["num_scored"]
        assert sum(per_source[s]["num_scored"] for s in ("a", "b")) == 20
        assert "failure_warning" in payload

    async def test_small_source_does_not_warn_on_a_single_error(
        self, tmp_path: Path,
    ) -> None:
        """One transient error in an 8-row source is 12.5% — not a signal, and
        this string rides out to every sweep row."""
        import pyarrow as pa

        from lqh.scoring import score_predictions_by_source

        rows = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        n = 18
        preds = tmp_path / "preds.parquet"
        pq.write_table(
            pa.table({
                "sample_index": list(range(n)),
                "messages": [json.dumps(rows)] * n,
                "source": ["a" if i < 10 else "b" for i in range(n)],
            }),
            preds,
        )
        scorer_path = tmp_path / "s.md"
        scorer_path.write_text("score it")

        calls = {"n": 0}

        async def _create(**_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("one transient error")
            response = MagicMock()
            response.choices = [
                MagicMock(message=MagicMock(content='{"reasoning":"ok","score":7}'))
            ]
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        payload = await score_predictions_by_source(
            predictions_path=preds,
            scorer_path=scorer_path,
            output_dir=tmp_path / "out",
            client=client,
            concurrency=1,
            max_retries=0,
        )

        assert "failure_warning" not in payload


class TestGoldenGenerationRetries:
    """generate_golden is handed the same max_retries=0 client as scoring, but
    it is NOT scoring: it drops a sample on any exception and has no ladder of
    its own unless one is given to it."""

    async def test_transient_error_does_not_cost_a_golden_response(self) -> None:
        from lqh.golden import _golden_from_api

        calls = {"n": 0}

        async def _create(**_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise APIConnectionError(request=httpx.Request("POST", "http://x"))
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="GOLDEN"))]
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        result = await _golden_from_api(
            [0],
            {0: [{"role": "user", "content": "q"}]},
            client,
            "orchestration",
        )

        assert result == {0: "GOLDEN"}
        assert calls["n"] == 2


    async def test_a_wedged_golden_sample_does_not_stall_the_pass(
        self, monkeypatch,
    ) -> None:
        """Adding a retry ladder without a bound would have recreated exactly
        the straggler problem this ticket is about: golden generation waits on
        every task, so 3 attempts x an upstream budget is one sample holding
        up the whole pass."""
        import lqh.golden as golden_mod

        monkeypatch.setattr(golden_mod, "_GOLDEN_TIMEOUT_S", 0.2)

        async def _hang(**_kwargs: Any) -> Any:
            await asyncio.sleep(30.0)

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_hang)

        result = await asyncio.wait_for(
            golden_mod._golden_from_api(
                [0, 1],
                {
                    0: [{"role": "user", "content": "q"}],
                    1: [{"role": "user", "content": "q"}],
                },
                client,
                "orchestration",
            ),
            timeout=10,
        )

        # Dropped, not waited out — a missing golden response is the existing
        # semantics for a sample that could not be generated.
        assert result == {}


class TestWipedSourceIsAlwaysReported:
    async def test_fully_failed_small_source_still_warns(
        self, tmp_path: Path,
    ) -> None:
        """A source that lost everything drops out of the macro-average, so
        the headline is computed over the survivors. That is worth saying
        regardless of how small the source was."""
        import pyarrow as pa

        from lqh.scoring import _THINNING_MIN_SOURCE, score_predictions_by_source

        # 200 good rows + a 19-row source that loses everything: 19/219 is
        # 8.7%, under the global threshold, and 19 is under the per-source
        # thinning floor. Neither existing check can see this one.
        big, small = 200, _THINNING_MIN_SOURCE - 1
        rows = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        n = big + small
        preds = tmp_path / "preds.parquet"
        pq.write_table(
            pa.table({
                "sample_index": list(range(n)),
                "messages": [json.dumps(rows)] * n,
                "source": ["big" if i < big else "small" for i in range(n)],
            }),
            preds,
        )
        scorer_path = tmp_path / "s.md"
        scorer_path.write_text("score it")

        calls = {"n": 0}

        async def _create(**_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] > big:  # concurrency=1 -> the small source is last
                raise RuntimeError("judge down for this source")
            response = MagicMock()
            response.choices = [
                MagicMock(message=MagicMock(content='{"reasoning":"ok","score":8}'))
            ]
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        payload = await score_predictions_by_source(
            predictions_path=preds,
            scorer_path=scorer_path,
            output_dir=tmp_path / "out",
            client=client,
            concurrency=1,
            max_retries=0,
        )

        assert payload["per_source"]["small"]["num_scored"] == 0
        assert payload["per_source"]["small"]["num_attempted"] == small
        assert small / n <= FAILURE_WARN_FRACTION  # global check stays silent
        assert small < _THINNING_MIN_SOURCE       # thinning floor stays silent
        assert "failure_warning" in payload, payload
        assert "NO usable scores" in payload["failure_warning"]
        assert "small" in payload["failure_warning"]

    async def test_wiped_only_run_does_not_get_thinning_prose(
        self, tmp_path: Path,
    ) -> None:
        """The lead sentence must not claim a wiped source was 'scored on only
        part of' its predictions and carries weight in the macro-average —
        both are the opposite of what happened to it."""
        import pyarrow as pa

        from lqh.scoring import _THINNING_MIN_SOURCE, score_predictions_by_source

        big, small = 200, _THINNING_MIN_SOURCE - 1
        rows = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        n = big + small
        preds = tmp_path / "preds.parquet"
        pq.write_table(
            pa.table({
                "sample_index": list(range(n)),
                "messages": [json.dumps(rows)] * n,
                "source": ["big" if i < big else "small" for i in range(n)],
            }),
            preds,
        )
        scorer_path = tmp_path / "s.md"
        scorer_path.write_text("score it")

        calls = {"n": 0}

        async def _create(**_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] > big:
                raise RuntimeError("judge down for this source")
            response = MagicMock()
            response.choices = [
                MagicMock(message=MagicMock(content='{"reasoning":"ok","score":8}'))
            ]
            return response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        payload = await score_predictions_by_source(
            predictions_path=preds,
            scorer_path=scorer_path,
            output_dir=tmp_path / "out",
            client=client,
            concurrency=1,
            max_retries=0,
        )

        warning = payload["failure_warning"]
        assert "no usable scores at all" in warning
        assert "scored on only part" not in warning
        assert "Thinned sources" not in warning
