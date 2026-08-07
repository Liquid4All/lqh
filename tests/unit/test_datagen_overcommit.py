"""Data-gen straggler handling: over-commit, early finish, sample watchdog.

A run used to end only when its very last sample did, so a couple of
items stuck in the retry ladder held a 200-sample run at "198/200" for
minutes. The engine now queues a few spares and stops at the requested
count, cancelling whatever is still crawling.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from lqh.data_gen_validation import pipeline_digest
from lqh.engine import _overcommit_margin, run_pipeline

# Two samples hang forever; everything else returns immediately.
_STRAGGLER_PIPELINE = """
import asyncio
from pathlib import Path
from lqh.pipeline import Pipeline, ChatMLMessage

_started = {"n": 0}

class Straggler(Pipeline):
    async def generate(self, client):
        _started["n"] += 1
        with open("calls.log", "a") as f:
            f.write("x\\n")
        if _started["n"] <= 2:
            await asyncio.sleep(3600)
        return [
            ChatMLMessage(role="user", content="hi"),
            ChatMLMessage(role="assistant", content="ok"),
        ]
"""

# Bring-your-data: two source items never succeed, and every item the
# engine touches is recorded so the num_samples cap can be checked.
_FLAKY_PIPELINE = """
from lqh.pipeline import Pipeline, ChatMLMessage, GenerationError
from lqh.sources import prompts

class Flaky(Pipeline):
    @classmethod
    def source(cls, project_dir):
        return prompts("seeds.txt")

    async def generate(self, client, input=None):
        with open("seen.log", "a") as f:
            f.write(input.prompt + "\\n")
        if input.prompt.startswith("bad"):
            raise GenerationError("nope")
        return [
            ChatMLMessage(role="user", content=input.prompt),
            ChatMLMessage(role="assistant", content="ok"),
        ]
"""

# Pure generation: the first two samples fail permanently (run with
# max_retries=0, so one call is one sample).
_FAIL_FIRST_TWO_PIPELINE = """
from lqh.pipeline import Pipeline, ChatMLMessage, GenerationError

_calls = {"n": 0}

class FailFirstTwo(Pipeline):
    async def generate(self, client):
        _calls["n"] += 1
        if _calls["n"] <= 2:
            raise GenerationError("nope")
        return [
            ChatMLMessage(role="user", content="hi"),
            ChatMLMessage(role="assistant", content="ok"),
        ]
"""

# Every sample takes a beat and records that it ran to completion, so a
# leaked task is visible after the run it belonged to is gone.
_SLOW_PIPELINE = """
import asyncio
from lqh.pipeline import Pipeline, ChatMLMessage

class Slow(Pipeline):
    async def generate(self, client):
        await asyncio.sleep(0.4)
        with open("done.log", "a") as f:
            f.write("x\\n")
        return [
            ChatMLMessage(role="user", content="hi"),
            ChatMLMessage(role="assistant", content="ok"),
        ]
"""

# While "fail.flag" exists: the first two calls fail permanently and the
# rest hang, so the run can be interrupted mid-flight. Without the flag
# every sample succeeds immediately.
_FLAGGED_FAILURE_PIPELINE = """
import asyncio
from pathlib import Path
from lqh.pipeline import Pipeline, ChatMLMessage, GenerationError

_calls = {"n": 0}

class Flagged(Pipeline):
    async def generate(self, client):
        if Path("fail.flag").exists():
            _calls["n"] += 1
            if _calls["n"] <= 2:
                raise GenerationError("nope")
            await asyncio.sleep(3600)
        return [
            ChatMLMessage(role="user", content="hi"),
            ChatMLMessage(role="assistant", content="ok"),
        ]
"""

# Refuses to stop: swallows the first cancellation, then finishes anyway.
_UNCANCELLABLE_PIPELINE = """
import asyncio
from lqh.pipeline import Pipeline, ChatMLMessage

_started = {"n": 0}

class Stubborn(Pipeline):
    async def generate(self, client):
        _started["n"] += 1
        if _started["n"] <= 2:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                with open("swallowed.log", "a") as f:
                    f.write("x\\n")
                await asyncio.sleep(0.3)   # ignores the cancellation
        return [
            ChatMLMessage(role="user", content="hi"),
            ChatMLMessage(role="assistant", content="ok"),
        ]
"""

_HANGING_PIPELINE = """
import asyncio
from lqh.pipeline import Pipeline, ChatMLMessage

class Hangs(Pipeline):
    async def generate(self, client):
        await asyncio.sleep(3600)
        return [ChatMLMessage(role="user", content="hi")]
"""


def _run_bounded(coro, timeout: float = 30.0):
    """Run *coro* with a hard deadline.

    These pipelines contain hour-long sleeps on purpose. A regression in
    the cancellation path should fail the test in seconds, not park CI
    until the sample watchdog fires half an hour later.
    """
    async def main():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(main())


def _write(project: Path, body: str) -> Path:
    (project / "data_gen").mkdir(parents=True, exist_ok=True)
    script = project / "data_gen" / "p.py"
    script.write_text(body)
    return script


def test_margin_scales_and_spares_small_runs() -> None:
    # Draft/validation runs must surface every failure — no spares.
    assert _overcommit_margin(1) == 0
    assert _overcommit_margin(3) == 0
    assert _overcommit_margin(49) == 0
    # Above the threshold: ~3%, floored at 2 and capped at 10.
    assert _overcommit_margin(50) == 2
    assert _overcommit_margin(200) == 6
    assert _overcommit_margin(1000) == 10
    assert _overcommit_margin(100_000) == 10


def test_run_finishes_without_waiting_for_stragglers(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    script = _write(project, _STRAGGLER_PIPELINE)
    out_dir = project / "datasets" / "d"

    progress: list[tuple[int, int]] = []
    started = time.monotonic()
    result = _run_bounded(run_pipeline(
        script_path=script,
        num_samples=60,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type] — pipeline never touches it
        on_progress=lambda c, t: progress.append((c, t)),
    ))
    elapsed = time.monotonic() - started

    # 60 requested + 2 spares: the spares cover the two hung samples, so
    # the run completes instead of blocking on asyncio.sleep(3600).
    assert elapsed < 30
    assert result.total == 60
    assert result.succeeded == 60
    assert len(pq.read_table(out_dir / "data.parquet")) == 60
    assert (project / "calls.log").read_text().count("x") == 62

    # Progress counts what the caller asked for — spares stay invisible,
    # and the count never runs backwards or past the target.
    assert all(total == 60 and 0 <= done <= 60 for done, total in progress)
    assert [c for c, _ in progress] == sorted(c for c, _ in progress)
    assert progress[-1] == (60, 60)


def test_spares_absorb_permanent_failures(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    script = _write(project, _FAIL_FIRST_TWO_PIPELINE)
    out_dir = project / "datasets" / "d"

    result = asyncio.run(run_pipeline(
        script_path=script,
        num_samples=60,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
        max_retries=0,
    ))

    # The spares made up the shortfall, so the caller got everything it
    # asked for — and the counters still add up.
    assert (result.total, result.succeeded, result.failed) == (60, 60, 0)
    assert result.sample_failures == 2  # the reliability signal survives
    assert len(pq.read_table(out_dir / "data.parquet")) == 60


def test_healthy_run_does_not_pay_for_spares(chdir_to_tmp: Path) -> None:
    """Spares stay parked until the run reaches its tail.

    Launching them up front would bill ~3% extra on every run that never
    had a straggler in the first place.
    """
    project = chdir_to_tmp
    script = _write(project, _SLOW_PIPELINE)
    out_dir = project / "datasets" / "d"

    result = _run_bounded(run_pipeline(
        script_path=script,
        num_samples=60,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
    ))

    # All 60 samples take the same time, so the target is reached before
    # any spare gets past the gate.
    assert result.succeeded == 60
    assert (project / "done.log").read_text().count("x") == 60


def test_a_pipeline_that_ignores_cancellation_cannot_corrupt_the_run(
    chdir_to_tmp: Path, monkeypatch,
) -> None:
    """Cancellation is cooperative — an abandoned sample must not write.

    The engine hands control back rather than blocking on a pipeline that
    refuses to stop, so the run's bookkeeping has to be closed to the
    stragglers that outlive it.
    """
    import lqh.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_CANCEL_SETTLE_S", 0.05)
    project = chdir_to_tmp
    script = _write(project, _UNCANCELLABLE_PIPELINE)
    out_dir = project / "datasets" / "d"

    async def main():
        result = await run_pipeline(
            script_path=script,
            num_samples=60,
            output_dir=out_dir,
            client=object(),  # type: ignore[arg-type]
        )
        # Outlive the stubborn samples, which finish after the run did.
        await asyncio.sleep(0.6)
        return result

    result = asyncio.run(asyncio.wait_for(main(), 30))

    assert (project / "swallowed.log").exists()  # they really did ignore it
    assert result.succeeded == 60
    assert len(pq.read_table(out_dir / "data.parquet")) == 60
    # The abandoned samples must not have resurrected the partial file:
    # a headerless partial would be resumed as done work by a later run.
    assert not (out_dir / "data.partial.jsonl").exists()


def test_byo_never_reads_past_num_samples(chdir_to_tmp: Path) -> None:
    """num_samples is a hard cap on source items — spares must not break it.

    Bring-your-data items aren't interchangeable: reading past the cap
    would swap a requested record for one the caller never asked to
    process, and could abort a run on a bug in an out-of-range item.
    """
    project = chdir_to_tmp
    script = _write(project, _FLAKY_PIPELINE)
    (project / "seeds.txt").write_text(
        "\n".join(["bad0", "bad1"] + [f"ok{i}" for i in range(2, 62)])
    )
    out_dir = project / "datasets" / "d"

    result = asyncio.run(run_pipeline(
        script_path=script,
        num_samples=60,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
        max_retries=0,
    ))

    seen = (project / "seen.log").read_text().split()
    assert set(seen) == {"bad0", "bad1"} | {f"ok{i}" for i in range(2, 60)}
    assert "ok60" not in seen and "ok61" not in seen
    # No spares here, so the two dud records are a real shortfall.
    assert (result.total, result.succeeded, result.failed) == (60, 58, 2)
    assert result.sample_failures == 2
    assert len(pq.read_table(out_dir / "data.parquet")) == 58


def test_small_run_generates_exactly_what_was_asked(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    script = _write(project, _STRAGGLER_PIPELINE)
    out_dir = project / "datasets" / "d"

    # num_samples=3 is below the over-commit threshold, and the first two
    # samples hang — so this run genuinely cannot finish early. Give it a
    # short watchdog so the hung pair fails instead of running forever.
    result = _run_bounded(run_pipeline(
        script_path=script,
        num_samples=3,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
        max_retries=0,
        sample_timeout=0.2,
    ))

    assert result.total == 3
    assert result.succeeded == 1
    assert result.failed == 2
    assert (project / "calls.log").read_text().count("x") == 3  # no spares


def test_cancelling_the_run_stops_every_sample(chdir_to_tmp: Path) -> None:
    """A cancelled run must not leave samples spending in the background."""
    project = chdir_to_tmp
    script = _write(project, _SLOW_PIPELINE)
    out_dir = project / "datasets" / "d"

    async def main() -> int:
        run = asyncio.ensure_future(run_pipeline(
            script_path=script,
            num_samples=60,
            output_dir=out_dir,
            client=object(),  # type: ignore[arg-type]
        ))
        await asyncio.sleep(0.2)  # everything is in flight by now
        run.cancel()
        try:
            await run
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.5)  # long enough for orphans to finish
        log = project / "done.log"
        return log.read_text().count("x") if log.exists() else 0

    assert asyncio.run(main()) == 0


def test_error_escaping_a_sample_surfaces_immediately(chdir_to_tmp: Path) -> None:
    """A raising callback must not wait behind the slowest sample.

    Anything that escapes the per-sample handlers (a progress callback
    blowing up is the realistic one) has to stop the run there and then —
    with production defaults a straggler can hold it for hours.
    """
    project = chdir_to_tmp
    script = _write(project, _STRAGGLER_PIPELINE)

    def boom(completed: int, total: int) -> None:
        raise RuntimeError("callback blew up")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="callback blew up"):
        _run_bounded(run_pipeline(
            script_path=script,
            num_samples=60,
            output_dir=project / "datasets" / "d",
            client=object(),  # type: ignore[arg-type]
            on_progress=boom,
        ))

    # Two samples of this pipeline sleep an hour; the error must not wait
    # on them.
    assert time.monotonic() - started < 10


def test_early_finish_leaves_no_unretrieved_future(
    chdir_to_tmp: Path, capsys,
) -> None:
    """The cancel path must not spew asyncio tracebacks on a good run."""
    project = chdir_to_tmp
    script = _write(project, _STRAGGLER_PIPELINE)

    root = logging.getLogger()
    saved = (root.handlers[:], root.level, logging.raiseExceptions)
    logging.raiseExceptions = True
    root.handlers = [logging.StreamHandler(sys.stderr)]
    root.setLevel(logging.ERROR)
    try:
        result = _run_bounded(run_pipeline(
            script_path=script,
            num_samples=60,
            output_dir=project / "datasets" / "d",
            client=object(),  # type: ignore[arg-type]
        ))
        gc.collect()  # future finalizers are where the complaint surfaces
    finally:
        root.handlers, root.level, logging.raiseExceptions = saved

    assert result.succeeded == 60
    err = capsys.readouterr().err
    assert "never retrieved" not in err, err
    assert "Traceback" not in err, err


def test_resume_accepts_a_partial_from_before_over_commit(
    chdir_to_tmp: Path,
) -> None:
    """The partial header is keyed to the requested count, not the work list.

    A cloud continuation (or a local rerun) that picks up a partial
    written before over-commit existed must resume it, not throw away
    paid-for samples.
    """
    project = chdir_to_tmp
    script = _write(project, _STRAGGLER_PIPELINE)
    out_dir = project / "datasets" / "d"
    out_dir.mkdir(parents=True)

    # Header shape the pre-over-commit engine wrote: total == num_samples.
    rows = [json.dumps({"_meta": True, "total": 60, "digest": pipeline_digest(script)})]
    rows += [
        json.dumps({
            "index": i,
            "messages": json.dumps([{"role": "user", "content": f"resumed{i}"}]),
            "audio": None,
            "tools": None,
        })
        for i in range(55)
    ]
    (out_dir / "data.partial.jsonl").write_text("\n".join(rows) + "\n")

    result = asyncio.run(run_pipeline(
        script_path=script,
        num_samples=60,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
    ))

    assert result.resumed_samples == 55
    assert result.succeeded == 60
    assert not (out_dir / "data.partial.stale.jsonl").exists()
    messages = pq.read_table(out_dir / "data.parquet").column("messages").to_pylist()
    assert sum("resumed" in m for m in messages) == 55
    # Only the 5 missing samples plus spares were generated.
    assert (project / "calls.log").read_text().count("x") <= 7


def test_resume_already_in_the_tail_still_releases_spares(
    chdir_to_tmp: Path,
) -> None:
    """Resuming past the tail boundary must not strand the spares.

    58 of 60 resumed, both remaining originals hung: nothing new ever
    completes, so a gate opened only by a completion would never open and
    the spares that exist to replace those two would wait forever.
    """
    project = chdir_to_tmp
    script = _write(project, _STRAGGLER_PIPELINE)
    out_dir = project / "datasets" / "d"
    out_dir.mkdir(parents=True)

    rows = [json.dumps({"_meta": True, "total": 60, "digest": pipeline_digest(script)})]
    rows += [
        json.dumps({
            "index": i,
            "messages": json.dumps([{"role": "user", "content": f"resumed{i}"}]),
            "audio": None,
            "tools": None,
        })
        for i in range(58)
    ]
    (out_dir / "data.partial.jsonl").write_text("\n".join(rows) + "\n")

    result = _run_bounded(run_pipeline(
        script_path=script,
        num_samples=60,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
    ))

    assert result.succeeded == 60
    assert len(pq.read_table(out_dir / "data.parquet")) == 60


def test_failures_survive_a_resume(chdir_to_tmp: Path) -> None:
    """A continuation must not report a clean run after a failed first try.

    Only successes are resumable, so without persisting the failures the
    reliability count resets to zero on every continuation.
    """
    project = chdir_to_tmp
    script = _write(project, _FLAGGED_FAILURE_PIPELINE)
    (project / "fail.flag").touch()
    out_dir = project / "datasets" / "d"

    # First attempt: two samples fail permanently, the rest are still in
    # flight when the run is cut short, leaving a partial behind.
    async def main():
        run = asyncio.ensure_future(run_pipeline(
            script_path=script, num_samples=60, output_dir=out_dir,
            client=object(),  # type: ignore[arg-type]
            max_retries=0,
        ))
        await asyncio.sleep(0.2)
        run.cancel()
        try:
            await run
        except asyncio.CancelledError:
            pass

    asyncio.run(asyncio.wait_for(main(), 30))
    partial = (out_dir / "data.partial.jsonl").read_text()
    assert partial.count('"_failed"') == 2

    # Clear the flag: the continuation generates cleanly, so the only
    # failures in its report are the ones it inherited.
    (project / "fail.flag").unlink()
    result = _run_bounded(run_pipeline(
        script_path=script,
        num_samples=60,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
        max_retries=0,
    ))

    assert result.succeeded == 60
    assert result.sample_failures == 2  # carried across the interruption


def test_sample_watchdog_bounds_a_wedged_pipeline(chdir_to_tmp: Path) -> None:
    project = chdir_to_tmp
    script = _write(project, _HANGING_PIPELINE)
    out_dir = project / "datasets" / "d"

    started = time.monotonic()
    result = asyncio.run(run_pipeline(
        script_path=script,
        num_samples=2,
        output_dir=out_dir,
        client=object(),  # type: ignore[arg-type]
        max_retries=1,
        sample_timeout=0.1,
    ))

    # Two attempts each, then permanent failure — bounded, not hung.
    assert time.monotonic() - started < 10
    assert result.succeeded == 0
    assert result.failed == 2
