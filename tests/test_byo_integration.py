"""Integration tests for bring-your-own-data flows end-to-end through the engine.

These tests write real pipeline files that import lqh.sources, load them via
the engine's dynamic loader, and execute them with a mocked AsyncOpenAI
client. No network required, but exercises the full wiring: source()
discovery, per-item instantiation, samples_per_item multiplier, parquet
output, and the lqh.sources helpers.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from lqh.engine import run_pipeline


def _fake_client(reply: str = "ok") -> MagicMock:
    """AsyncOpenAI stand-in: chat.completions.create returns *reply* as content."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()

    async def _create(**kwargs):
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = reply
        resp.choices = [choice]
        return resp

    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


def _write_pipeline(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Bring-your-prompts: source()=prompts(), generate() completes them.
# ---------------------------------------------------------------------------


def test_byo_prompts_end_to_end(project: Path) -> None:
    (project / "prompts.jsonl").write_text(
        json.dumps({"prompt": "hello"}) + "\n" + json.dumps({"prompt": "world"}) + "\n"
    )

    _write_pipeline(
        project / "data_gen" / "p.py",
        """
        from lqh.pipeline import Pipeline, ChatMLMessage
        import lqh.sources as sources

        class CompletePrompts(Pipeline):
            @classmethod
            def source(cls, project_dir):
                return sources.prompts(project_dir / "prompts.jsonl")

            async def generate(self, client, input):
                resp = await client.chat.completions.create(
                    model="random:small",
                    messages=[{"role": "user", "content": input.prompt}],
                )
                return [
                    ChatMLMessage("user", input.prompt),
                    ChatMLMessage("assistant", resp.choices[0].message.content),
                ]
        """,
    )

    output_dir = project / "datasets" / "v1"
    result = asyncio.run(
        run_pipeline(
            script_path=project / "data_gen" / "p.py",
            num_samples=5,  # larger than source length; engine caps at source
            output_dir=output_dir,
            client=_fake_client("greetings"),
            concurrency=1,
            max_retries=0,
        )
    )

    assert result.succeeded == 2
    assert result.total == 2

    table = pq.read_table(output_dir / "data.parquet")
    assert table.num_rows == 2
    rows = [json.loads(m) for m in table.column("messages").to_pylist()]
    assert {r[0]["content"] for r in rows} == {"hello", "world"}
    assert all(r[1]["content"] == "greetings" for r in rows)


# ---------------------------------------------------------------------------
# Seed data: source() yields list[str], samples_per_item multiplier iterates.
# ---------------------------------------------------------------------------


def test_byo_seed_data_with_iteration(project: Path) -> None:
    (project / "seed_data").mkdir()
    (project / "seed_data" / "flowers.txt").write_text("rose\ntulip\n")

    _write_pipeline(
        project / "data_gen" / "p.py",
        """
        from lqh.pipeline import Pipeline, ChatMLMessage
        import lqh.sources as sources

        class Florist(Pipeline):
            @classmethod
            def source(cls, project_dir):
                return sources.seed_data("flowers")

            async def generate(self, client, input):
                resp = await client.chat.completions.create(
                    model="random:small",
                    messages=[{"role": "user", "content": f"About {input}s?"}],
                )
                return [
                    ChatMLMessage("user", f"About {input}s?"),
                    ChatMLMessage("assistant", resp.choices[0].message.content),
                ]
        """,
    )

    output_dir = project / "datasets" / "v1"
    result = asyncio.run(
        run_pipeline(
            script_path=project / "data_gen" / "p.py",
            num_samples=2,
            output_dir=output_dir,
            client=_fake_client("lovely flower"),
            concurrency=1,
            samples_per_item=3,  # 2 seeds × 3 iterations = 6 samples
            max_retries=0,
        )
    )

    assert result.total == 6
    assert result.succeeded == 6

    table = pq.read_table(output_dir / "data.parquet")
    assert table.num_rows == 6
    rows = [json.loads(m) for m in table.column("messages").to_pylist()]
    user_msgs = [r[0]["content"] for r in rows]
    # Each seed appears 3 times
    assert user_msgs.count("About roses?") == 3
    assert user_msgs.count("About tulips?") == 3


# ---------------------------------------------------------------------------
# Image folder: ImageItem flows through generate() unchanged.
# ---------------------------------------------------------------------------


def test_byo_image_folder_end_to_end(project: Path) -> None:
    images = project / "images"
    for name in ("a.jpg", "b.jpg"):
        (images / "dog").mkdir(parents=True, exist_ok=True)
        (images / "dog" / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    _write_pipeline(
        project / "data_gen" / "p.py",
        """
        from lqh.pipeline import Pipeline, ChatMLMessage
        import lqh.sources as sources

        class LabelImages(Pipeline):
            @classmethod
            def source(cls, project_dir):
                return sources.image_folder(
                    project_dir / "images", include_subfolder_label=True,
                )

            async def generate(self, client, input):
                # Touch the item to ensure it's the expected type.
                assert input.subfolder == "dog"
                _ = input.as_data_url()  # forces read_bytes
                resp = await client.chat.completions.create(
                    model="random:small",
                    messages=[{"role": "user", "content": "label?"}],
                )
                return [
                    ChatMLMessage("user", f"label for {input.path.name}"),
                    ChatMLMessage("assistant", resp.choices[0].message.content),
                ]
        """,
    )

    output_dir = project / "datasets" / "v1"
    result = asyncio.run(
        run_pipeline(
            script_path=project / "data_gen" / "p.py",
            num_samples=10,
            output_dir=output_dir,
            client=_fake_client("dog"),
            concurrency=1,
            max_retries=0,
        )
    )
    assert result.succeeded == 2
