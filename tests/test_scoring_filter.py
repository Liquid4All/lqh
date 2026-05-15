"""Tests for ``run_data_filter`` — score + filter for bring-your-data-for-scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lqh.scoring import FilterResult, run_data_filter


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def input_parquet(tmp_path: Path) -> Callable[[int], Path]:
    """Factory writing an N-row ChatML parquet with the canonical schema."""

    def _factory(n: int) -> Path:
        rows = [
            json.dumps([
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ])
            for i in range(n)
        ]
        table = pa.table(
            {"messages": rows, "audio": [""] * n, "tools": [""] * n},
            schema=pa.schema([
                pa.field("messages", pa.string()),
                pa.field("audio", pa.string()),
                pa.field("tools", pa.string()),
            ]),
        )
        path = tmp_path / "in.parquet"
        pq.write_table(table, path)
        return path

    return _factory


@pytest.fixture
def scorer_md(tmp_path: Path) -> Path:
    path = tmp_path / "scorer.md"
    path.write_text("score high if content is nice")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_run_data_filter_keeps_only_above_threshold(
    tmp_path: Path,
    input_parquet,
    scorer_md: Path,
    mock_openai_client,
) -> None:
    out_dir = tmp_path / "filtered"

    # Scores 8, 3, 7, 5 — threshold 6 keeps indices 0 and 2.
    result = await run_data_filter(
        input_path=input_parquet(4),
        scorer_path=scorer_md,
        output_dataset_dir=out_dir,
        client=mock_openai_client(scores=[8, 3, 7, 5]),
        threshold=6.0,
        concurrency=1,  # deterministic order for mocked scores
        max_retries=0,
    )

    assert isinstance(result, FilterResult)
    assert result.total == 4
    assert result.kept == 2
    assert result.dropped == 2
    assert result.failed == 0

    kept = pq.read_table(out_dir / "data.parquet")
    assert kept.num_rows == 2
    assert set(kept.column_names) == {"messages", "audio", "tools"}
    kept_contents = [json.loads(m) for m in kept.column("messages").to_pylist()]
    assert {m[0]["content"] for m in kept_contents} == {"q0", "q2"}

    scores = pq.read_table(out_dir / "scores.parquet")
    assert scores.num_rows == 4
    assert scores.column("kept").to_pylist() == [True, False, True, False]

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["total"] == 4
    assert summary["kept"] == 2
    assert summary["threshold"] == 6.0
    assert summary["keep_rate"] == 0.5


async def test_run_data_filter_empty_input(
    tmp_path: Path, scorer_md: Path, mock_openai_client,
) -> None:
    empty_path = tmp_path / "empty.parquet"
    table = pa.table(
        {"messages": [], "audio": [], "tools": []},
        schema=pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ]),
    )
    pq.write_table(table, empty_path)

    out_dir = tmp_path / "out"
    result = await run_data_filter(
        input_path=empty_path,
        scorer_path=scorer_md,
        output_dataset_dir=out_dir,
        client=mock_openai_client(scores=[]),
    )
    assert result.total == 0
    assert result.kept == 0
    assert (out_dir / "data.parquet").exists()
    assert (out_dir / "scores.parquet").exists()
    assert (out_dir / "summary.json").exists()


async def test_run_data_filter_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await run_data_filter(
            input_path=tmp_path / "nope.parquet",
            scorer_path=tmp_path / "s.md",
            output_dataset_dir=tmp_path / "out",
            client=MagicMock(),
        )
