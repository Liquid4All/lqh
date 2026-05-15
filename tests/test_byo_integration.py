"""Integration tests for bring-your-own-data flows end-to-end through the engine.

These tests write real pipeline files that import ``lqh.sources``, load them via
the engine's dynamic loader, and execute them with a mocked ``AsyncOpenAI``
client.  No network required, but they exercise the full wiring:
``source()`` discovery, per-item instantiation, ``samples_per_item``
multiplier, parquet output, and the ``lqh.sources`` helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Callable

import pyarrow.parquet as pq
import pytest

from lqh.engine import run_pipeline


@pytest.fixture
def write_pipeline(chdir_to_tmp: Path) -> Callable[[str, str], Path]:
    """Write a pipeline module under the current project directory."""

    def _factory(relative_path: str, body: str) -> Path:
        path = chdir_to_tmp / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(body))
        return path

    return _factory


# ---------------------------------------------------------------------------
# Bring-your-prompts: ``source()=prompts()``, ``generate()`` completes them.
# ---------------------------------------------------------------------------


async def test_byo_prompts_end_to_end(
    chdir_to_tmp: Path,
    write_pipeline,
    mock_openai_client,
) -> None:
    project = chdir_to_tmp
    (project / "prompts.jsonl").write_text(
        json.dumps({"prompt": "hello"}) + "\n"
        + json.dumps({"prompt": "world"}) + "\n"
    )

    pipeline = write_pipeline(
        "data_gen/p.py",
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
    result = await run_pipeline(
        script_path=pipeline,
        num_samples=5,  # larger than source length; engine caps at source
        output_dir=output_dir,
        client=mock_openai_client(content="greetings"),
        concurrency=1,
        max_retries=0,
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


async def test_byo_seed_data_with_iteration(
    chdir_to_tmp: Path,
    write_pipeline,
    mock_openai_client,
) -> None:
    project = chdir_to_tmp
    (project / "seed_data").mkdir()
    (project / "seed_data" / "flowers.txt").write_text("rose\ntulip\n")

    pipeline = write_pipeline(
        "data_gen/p.py",
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
    result = await run_pipeline(
        script_path=pipeline,
        num_samples=2,
        output_dir=output_dir,
        client=mock_openai_client(content="lovely flower"),
        concurrency=1,
        samples_per_item=3,  # 2 seeds × 3 iterations = 6 samples
        max_retries=0,
    )

    assert result.total == 6
    assert result.succeeded == 6

    table = pq.read_table(output_dir / "data.parquet")
    assert table.num_rows == 6
    user_msgs = [json.loads(m)[0]["content"] for m in table.column("messages").to_pylist()]
    assert user_msgs.count("About roses?") == 3
    assert user_msgs.count("About tulips?") == 3


# ---------------------------------------------------------------------------
# Image folder: ImageItem flows through generate() unchanged.
# ---------------------------------------------------------------------------


async def test_byo_image_folder_end_to_end(
    chdir_to_tmp: Path,
    write_pipeline,
    mock_openai_client,
) -> None:
    project = chdir_to_tmp
    images = project / "images" / "dog"
    images.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (images / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    pipeline = write_pipeline(
        "data_gen/p.py",
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
    result = await run_pipeline(
        script_path=pipeline,
        num_samples=10,
        output_dir=output_dir,
        client=mock_openai_client(content="dog"),
        concurrency=1,
        max_retries=0,
    )
    assert result.succeeded == 2
