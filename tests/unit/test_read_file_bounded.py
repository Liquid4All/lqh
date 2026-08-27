"""read_file must stay memory-bounded — a big parquet used to OOM-kill the CLI."""

import asyncio

import pyarrow as pa
import pyarrow.parquet as pq

from lqh.tools import handlers
from lqh.tools.handlers import MAX_READ_CHARS, handle_read_file


def _run(**kw):
    return asyncio.run(handle_read_file(**kw)).content


def test_parquet_reads_only_the_preview_rows(tmp_path, monkeypatch):
    table = pa.table({"i": list(range(1000)), "blob": ["x" * 500] * 1000})
    pq.write_table(table, tmp_path / "big.parquet", row_group_size=50)

    # Fail loudly if anything loads the whole file.
    monkeypatch.setattr(
        pq, "read_table", lambda *a, **k: pytest_fail("read_table loaded the whole file")
    )

    out = _run(project_dir=tmp_path, path="big.parquet", limit=5)
    assert "Total rows: 1000" in out
    assert "offset=5" in out
    assert out.count("\n") < 200  # preview only, not 1000 rows

    paged = _run(project_dir=tmp_path, path="big.parquet", offset=995, limit=10)
    assert "Rows 995-999" in paged
    assert "offset=" not in paged.rsplit("Rows", 1)[1]  # no "see more" past the end

    assert "[No rows at offset 5000.]" in _run(
        project_dir=tmp_path, path="big.parquet", offset=5000
    )


def test_parquet_cells_are_clipped(tmp_path):
    pq.write_table(pa.table({"blob": ["y" * 100_000]}), tmp_path / "wide.parquet")
    out = _run(project_dir=tmp_path, path="wide.parquet")
    assert "y" * 1000 not in out


def test_text_read_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(handlers, "MAX_READ_CHARS", 1000)
    (tmp_path / "big.txt").write_text("line\n" * 10_000)
    out = _run(project_dir=tmp_path, path="big.txt")
    assert "of the file was read" in out
    assert len(out) < 5000


def test_small_text_unchanged(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree")
    out = _run(project_dir=tmp_path, path="a.txt")
    assert "3 lines" in out and "three" in out


def pytest_fail(msg):
    raise AssertionError(msg)
