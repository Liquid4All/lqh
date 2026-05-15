"""Tests for ``lqh.sources`` — BYO-data helpers used in pipeline ``source()`` methods."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lqh import sources


# ``project`` is an alias for the ``chdir_to_tmp`` fixture from conftest.
@pytest.fixture
def project(chdir_to_tmp: Path) -> Path:
    """Project directory pinned as the process cwd."""
    return chdir_to_tmp


def _make_image(path: Path) -> None:
    """Minimal 1x1 PNG — enough for ``read_bytes`` / ``mime_type``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


# ---------------------------------------------------------------------------
# image_folder
# ---------------------------------------------------------------------------


def test_image_folder_flat(project: Path) -> None:
    images_dir = project / "images"
    _make_image(images_dir / "a.jpg")
    _make_image(images_dir / "b.png")
    (images_dir / "readme.txt").write_text("not an image")

    items = sources.image_folder("images")
    assert [it.path.name for it in items] == ["a.jpg", "b.png"]
    assert all(it.subfolder == "" for it in items)


def test_image_folder_subfolder_label(project: Path) -> None:
    root = project / "animals"
    _make_image(root / "dog" / "d1.jpg")
    _make_image(root / "cat" / "c1.jpg")
    _make_image(root / "cat" / "c2.jpg")

    items = sources.image_folder("animals", include_subfolder_label=True)
    labels = sorted({it.subfolder for it in items})
    assert labels == ["cat", "dog"]
    assert len(items) == 3


def test_image_folder_non_recursive(project: Path) -> None:
    root = project / "imgs"
    _make_image(root / "top.jpg")
    _make_image(root / "sub" / "nested.jpg")
    items = sources.image_folder("imgs", recursive=False)
    assert [it.path.name for it in items] == ["top.jpg"]


def test_image_folder_missing_raises(project: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sources.image_folder("does_not_exist")


def test_image_folder_empty_raises(project: Path) -> None:
    (project / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        sources.image_folder("empty")


def test_image_item_mime_and_data_url(project: Path) -> None:
    _make_image(project / "images" / "a.jpg")
    [item] = sources.image_folder("images")
    assert item.mime_type() in ("image/jpeg",)
    url = item.as_data_url()
    assert url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_prompts_txt(project: Path) -> None:
    (project / "p.txt").write_text("first\n\nsecond\n")
    items = sources.prompts("p.txt")
    assert [i.prompt for i in items] == ["first", "second"]
    assert all(i.metadata == {} for i in items)


def test_prompts_jsonl_with_metadata(project: Path) -> None:
    lines = [
        json.dumps({"prompt": "hello", "id": 1}),
        json.dumps({"prompt": "world", "id": 2, "cat": "x"}),
    ]
    (project / "p.jsonl").write_text("\n".join(lines) + "\n")
    items = sources.prompts("p.jsonl")
    assert [i.prompt for i in items] == ["hello", "world"]
    assert items[0].metadata == {"id": 1}
    assert items[1].metadata == {"id": 2, "cat": "x"}


def test_prompts_jsonl_bare_strings(project: Path) -> None:
    (project / "p.jsonl").write_text('"one"\n"two"\n')
    items = sources.prompts("p.jsonl")
    assert [i.prompt for i in items] == ["one", "two"]


def test_prompts_jsonl_missing_column_raises(project: Path) -> None:
    (project / "p.jsonl").write_text(json.dumps({"question": "x"}) + "\n")
    with pytest.raises(KeyError):
        sources.prompts("p.jsonl")


def test_prompts_custom_column(project: Path) -> None:
    (project / "p.jsonl").write_text(json.dumps({"question": "x", "id": 3}) + "\n")
    [item] = sources.prompts("p.jsonl", column="question")
    assert item.prompt == "x"
    assert item.metadata == {"id": 3}


def test_prompts_csv(project: Path) -> None:
    path = project / "p.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "cat"])
        w.writerow(["hello", "greeting"])
        w.writerow(["bye", "farewell"])
    items = sources.prompts("p.csv")
    assert [i.prompt for i in items] == ["hello", "bye"]
    assert items[0].metadata == {"cat": "greeting"}


def test_prompts_parquet(project: Path) -> None:
    table = pa.table({"prompt": ["a", "b"], "id": [1, 2]})
    pq.write_table(table, project / "p.parquet")
    items = sources.prompts("p.parquet")
    assert [i.prompt for i in items] == ["a", "b"]
    assert items[0].metadata == {"id": 1}


def test_prompts_unsupported_ext(project: Path) -> None:
    (project / "p.yaml").write_text("x: 1")
    with pytest.raises(ValueError):
        sources.prompts("p.yaml")


# ---------------------------------------------------------------------------
# parquet / jsonl
# ---------------------------------------------------------------------------


def test_parquet_stream(project: Path) -> None:
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    pq.write_table(table, project / "data.parquet")
    rows = list(sources.parquet("data.parquet"))
    assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]


def test_parquet_columns_filter(project: Path) -> None:
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    pq.write_table(table, project / "data.parquet")
    rows = list(sources.parquet("data.parquet", columns=["a"]))
    assert rows == [{"a": 1}, {"a": 2}]


def test_jsonl_stream(project: Path) -> None:
    (project / "d.jsonl").write_text(
        json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n"
    )
    rows = list(sources.jsonl("d.jsonl"))
    assert rows == [{"a": 1}, {"a": 2}]


def test_jsonl_bad_json_raises(project: Path) -> None:
    (project / "bad.jsonl").write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        list(sources.jsonl("bad.jsonl"))


def test_jsonl_non_object_raises(project: Path) -> None:
    (project / "arr.jsonl").write_text("[1, 2, 3]\n")
    with pytest.raises(TypeError):
        list(sources.jsonl("arr.jsonl"))


# ---------------------------------------------------------------------------
# seed_data
# ---------------------------------------------------------------------------


def test_seed_data_txt(project: Path) -> None:
    (project / "seed_data").mkdir()
    (project / "seed_data" / "flowers.txt").write_text("rose\ntulip\n\ndaisy\n")
    assert sources.seed_data("flowers") == ["rose", "tulip", "daisy"]


def test_seed_data_jsonl(project: Path) -> None:
    (project / "seed_data").mkdir()
    (project / "seed_data" / "flowers.jsonl").write_text(
        json.dumps({"name": "rose"}) + "\n" + json.dumps({"name": "tulip"}) + "\n"
    )
    assert sources.seed_data("flowers") == [{"name": "rose"}, {"name": "tulip"}]


def test_seed_data_csv(project: Path) -> None:
    (project / "seed_data").mkdir()
    path = project / "seed_data" / "flowers.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "color"])
        w.writerow(["rose", "red"])
    assert sources.seed_data("flowers") == [{"name": "rose", "color": "red"}]


def test_seed_data_missing_dir_raises(project: Path) -> None:
    with pytest.raises(FileNotFoundError, match="seed_data/"):
        sources.seed_data("flowers")


def test_seed_data_missing_file_raises(project: Path) -> None:
    (project / "seed_data").mkdir()
    with pytest.raises(FileNotFoundError, match="flowers"):
        sources.seed_data("flowers")


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    _make_image(outside / "evil.jpg")
    with pytest.raises(ValueError, match="outside the project directory"):
        sources.image_folder(str(outside))


def test_dotdot_escape_rejected(project: Path) -> None:
    with pytest.raises(ValueError, match="outside the project directory"):
        sources.prompts("../../etc/passwd")
