"""Tests for run_data_filter — score + filter for bring-your-data-for-scoring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from lqh.scoring import FilterResult, run_data_filter


def _make_input_parquet(path: Path, n: int = 4) -> None:
    rows = []
    for i in range(n):
        msgs = [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
        rows.append(json.dumps(msgs))
    table = pa.table(
        {"messages": rows, "audio": [""] * n, "tools": [""] * n},
        schema=pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ]),
    )
    pq.write_table(table, path)


def _make_client(scores: list[int]) -> MagicMock:
    """Fake AsyncOpenAI where each scoring call returns the next score in order."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()

    call_lock = asyncio.Lock()
    # Pre-build responses; served in deterministic order.
    queue = list(scores)

    async def _create(**kwargs):
        async with call_lock:
            score = queue.pop(0)
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps({"reasoning": f"r{score}", "score": score})
        resp.choices = [choice]
        return resp

    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


def test_run_data_filter_keeps_only_above_threshold(tmp_path: Path) -> None:
    input_path = tmp_path / "in.parquet"
    _make_input_parquet(input_path, n=4)
    scorer = tmp_path / "scorer.md"
    scorer.write_text("score high if content is nice")
    out_dir = tmp_path / "filtered"

    # Scores: 8, 3, 7, 5 — threshold 6 keeps indices 0 and 2
    client = _make_client([8, 3, 7, 5])

    result = asyncio.run(
        run_data_filter(
            input_path=input_path,
            scorer_path=scorer,
            output_dataset_dir=out_dir,
            client=client,
            threshold=6.0,
            concurrency=1,  # deterministic order for mocked scores
            max_retries=0,
        )
    )

    assert isinstance(result, FilterResult)
    assert result.total == 4
    assert result.kept == 2
    assert result.dropped == 2
    assert result.failed == 0

    # data.parquet contains only kept rows, preserving input schema.
    kept = pq.read_table(out_dir / "data.parquet")
    assert kept.num_rows == 2
    assert set(kept.column_names) == {"messages", "audio", "tools"}
    kept_contents = [json.loads(m) for m in kept.column("messages").to_pylist()]
    # Kept should be q0 and q2
    assert {m[0]["content"] for m in kept_contents} == {"q0", "q2"}

    # scores.parquet has one row per input sample with kept flag.
    scores = pq.read_table(out_dir / "scores.parquet")
    assert scores.num_rows == 4
    kept_flags = scores.column("kept").to_pylist()
    assert kept_flags == [True, False, True, False]

    # summary.json reports counts.
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["total"] == 4
    assert summary["kept"] == 2
    assert summary["threshold"] == 6.0
    assert summary["keep_rate"] == 0.5


def test_run_data_filter_empty_input(tmp_path: Path) -> None:
    input_path = tmp_path / "in.parquet"
    table = pa.table(
        {"messages": [], "audio": [], "tools": []},
        schema=pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ]),
    )
    pq.write_table(table, input_path)
    scorer = tmp_path / "s.md"
    scorer.write_text("x")
    out = tmp_path / "out"

    client = _make_client([])
    result = asyncio.run(
        run_data_filter(
            input_path=input_path, scorer_path=scorer,
            output_dataset_dir=out, client=client,
        )
    )
    assert result.total == 0
    assert result.kept == 0
    assert (out / "data.parquet").exists()
    assert (out / "scores.parquet").exists()
    assert (out / "summary.json").exists()


def test_run_data_filter_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            run_data_filter(
                input_path=tmp_path / "nope.parquet",
                scorer_path=tmp_path / "s.md",
                output_dataset_dir=tmp_path / "out",
                client=MagicMock(),
            )
        )
