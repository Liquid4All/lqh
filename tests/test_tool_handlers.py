"""Tests for tool handlers (lqh/tools/handlers.py).

Unit tests verify parameter validation and dispatch logic.  Integration
tests exercise ``handle_list_models`` and ``handle_run_scoring`` against
``api.lqh.ai`` and skip automatically without credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lqh.tools.handlers import (
    _validate_path,
    execute_tool,
    handle_create_file,
    handle_edit_file,
    handle_get_eval_failures,
    handle_list_files,
    handle_read_file,
    handle_run_scoring,
    handle_write_file,
)


# ---------------------------------------------------------------------------
# _validate_path containment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rel_path", "expected"),
    [
        ("file.txt", "file.txt"),
        ("nested/../file.txt", "file.txt"),
        ("../proj2/secret.txt", ValueError),
        ("../outside/secret.txt", ValueError),
    ],
)
def test_validate_path_containment(
    tmp_path: Path,
    rel_path: str,
    expected: str | type[Exception],
) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "nested").mkdir()
    (tmp_path / "proj2").mkdir()
    (tmp_path / "outside").mkdir()

    if expected is ValueError:
        with pytest.raises(ValueError, match="outside the project"):
            _validate_path(project_dir, rel_path)
    else:
        assert _validate_path(project_dir, rel_path) == (project_dir / expected).resolve()


# ---------------------------------------------------------------------------
# Shared workspace fixtures
# ---------------------------------------------------------------------------


SAMPLE_QA = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
]


@pytest.fixture
def scoring_workspace(tmp_path: Path, write_chatml_parquet) -> Path:
    """Project dir with a minimal dataset and scorer for run_scoring tests."""
    ds_dir = tmp_path / "datasets" / "test_ds"
    write_chatml_parquet(ds_dir / "data.parquet", [SAMPLE_QA], audio=True)

    scorer_dir = tmp_path / "evals" / "scorers"
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "test.md").write_text("Score for quality 1-10.")
    return tmp_path


@pytest.fixture
def fake_api_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Inject a debug API token so handlers reach the validation branches."""
    monkeypatch.setenv("LQH_DEBUG_API_KEY", "test-token")
    return "test-token"


# ---------------------------------------------------------------------------
# handle_run_scoring validation
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_api_token")
class TestRunScoringValidation:
    """Parameter validation in ``handle_run_scoring``."""

    async def test_model_eval_requires_run_name(self, scoring_workspace: Path) -> None:
        result = await handle_run_scoring(
            scoring_workspace,
            dataset="datasets/test_ds",
            scorer="evals/scorers/test.md",
            mode="model_eval",
            inference_model="small",
        )
        assert "run_name is required" in result.content

    async def test_model_eval_requires_inference_model(self, scoring_workspace: Path) -> None:
        result = await handle_run_scoring(
            scoring_workspace,
            dataset="datasets/test_ds",
            scorer="evals/scorers/test.md",
            mode="model_eval",
            run_name="test_run",
        )
        assert "inference_model is required" in result.content

    async def test_missing_dataset(self, scoring_workspace: Path) -> None:
        result = await handle_run_scoring(
            scoring_workspace,
            dataset="datasets/nonexistent",
            scorer="evals/scorers/test.md",
            mode="data_quality",
        )
        assert "Error" in result.content

    async def test_missing_scorer(self, scoring_workspace: Path) -> None:
        result = await handle_run_scoring(
            scoring_workspace,
            dataset="datasets/test_ds",
            scorer="evals/scorers/nonexistent.md",
            mode="data_quality",
        )
        assert "Error" in result.content

    async def test_unknown_mode(self, scoring_workspace: Path) -> None:
        result = await handle_run_scoring(
            scoring_workspace,
            dataset="datasets/test_ds",
            scorer="evals/scorers/test.md",
            mode="invalid_mode",
        )
        assert "unknown mode" in result.content

    async def test_duplicate_run_name_rejected(self, scoring_workspace: Path) -> None:
        (scoring_workspace / "evals" / "runs" / "existing_run").mkdir(parents=True)

        result = await handle_run_scoring(
            scoring_workspace,
            dataset="datasets/test_ds",
            scorer="evals/scorers/test.md",
            mode="model_eval",
            run_name="existing_run",
            inference_model="small",
        )
        assert "already exists" in result.content


# ---------------------------------------------------------------------------
# get_eval_failures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_run_factory(tmp_path: Path) -> Callable[[list[float]], str]:
    """Factory writing an eval run dir with ``results.parquet``.

    Returns the relative run path (``evals/runs/test_run``) for use as the
    ``eval_run`` argument to the handler.
    """

    def _factory(scores: list[float], *, name: str = "test_run") -> str:
        run_dir = tmp_path / "evals" / "runs" / name
        run_dir.mkdir(parents=True)

        rows = {
            "sample_index": list(range(len(scores))),
            "messages": [
                json.dumps([
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ])
                for i in range(len(scores))
            ],
            "score": scores,
            "reasoning": [f"Reason {i}" for i in range(len(scores))],
        }
        table = pa.table(
            rows,
            schema=pa.schema([
                pa.field("sample_index", pa.int64()),
                pa.field("messages", pa.string()),
                pa.field("score", pa.float64()),
                pa.field("reasoning", pa.string()),
            ]),
        )
        pq.write_table(table, run_dir / "results.parquet")
        return f"evals/runs/{name}"

    return _factory


class TestGetEvalFailures:
    async def test_returns_failures(self, tmp_path: Path, eval_run_factory) -> None:
        eval_run = eval_run_factory([2.0, 4.0, 7.0, 9.0, 8.0])
        result = await handle_get_eval_failures(
            tmp_path, eval_run=eval_run, threshold=6.0, min_failures=0,
        )
        assert "Failure Cases" in result.content
        assert "Score: 2.0" in result.content
        assert "Score: 4.0" in result.content
        assert "Score: 7.0" not in result.content

    async def test_missing_results(self, tmp_path: Path) -> None:
        (tmp_path / "evals" / "runs" / "empty").mkdir(parents=True)
        result = await handle_get_eval_failures(
            tmp_path, eval_run="evals/runs/empty",
        )
        assert "Error" in result.content

    async def test_no_failures(self, tmp_path: Path, eval_run_factory) -> None:
        eval_run = eval_run_factory([8.0, 9.0, 10.0])
        result = await handle_get_eval_failures(
            tmp_path, eval_run=eval_run, threshold=6.0, min_failures=0,
        )
        assert "No failure cases" in result.content

    async def test_padding_works(self, tmp_path: Path, eval_run_factory) -> None:
        eval_run = eval_run_factory([8.0, 9.0, 7.0])
        result = await handle_get_eval_failures(
            tmp_path, eval_run=eval_run, threshold=6.0, min_failures=2,
        )
        assert "Failure Cases (2 of 3" in result.content


# ---------------------------------------------------------------------------
# File handlers
# ---------------------------------------------------------------------------


class TestFileToolHandlers:
    """File manipulation tools."""

    async def test_create_and_read_file(self, tmp_path: Path) -> None:
        result = await handle_create_file(tmp_path, path="test.txt", content="hello world")
        assert "Created" in result.content

        read = await handle_read_file(tmp_path, path="test.txt")
        assert "hello world" in read.content

    async def test_create_file_rejects_existing(self, tmp_path: Path) -> None:
        (tmp_path / "exists.txt").write_text("x")
        result = await handle_create_file(tmp_path, path="exists.txt", content="y")
        assert "already exists" in result.content

    async def test_write_file_overwrites(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("old")
        result = await handle_write_file(tmp_path, path="file.txt", content="new")
        assert "Wrote" in result.content
        assert (tmp_path / "file.txt").read_text() == "new"

    async def test_edit_file(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello world")
        result = await handle_edit_file(
            tmp_path, path="file.txt", old_string="world", new_string="earth",
        )
        assert "Edited" in result.content
        assert (tmp_path / "file.txt").read_text() == "hello earth"

    async def test_edit_file_rejects_non_unique(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("aaa bbb aaa")
        result = await handle_edit_file(
            tmp_path, path="file.txt", old_string="aaa", new_string="ccc",
        )
        assert "found 2 times" in result.content

    async def test_list_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = await handle_list_files(tmp_path)
        assert "a.txt" in result.content
        assert "b.txt" in result.content

    async def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="outside the project"):
            await handle_read_file(tmp_path, path="../../etc/passwd")


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------


class TestExecuteToolDispatch:
    async def test_unknown_tool(self, tmp_path: Path) -> None:
        result = await execute_tool("nonexistent_tool", {}, tmp_path)
        assert "unknown tool" in result.content

    async def test_dispatches_to_correct_handler(self, tmp_path: Path) -> None:
        (tmp_path / "test.md").write_text("# Test")
        result = await execute_tool("read_file", {"path": "test.md"}, tmp_path)
        assert "# Test" in result.content


# ---------------------------------------------------------------------------
# Integration tests (require API access)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_list_models_returns_models() -> None:
    """Integration: list_models hits the live API."""
    from lqh.tools.handlers import handle_list_models

    result = await handle_list_models()
    assert "Liquid Foundation Models" in result.content
    assert "orchestration" in result.content
    assert "lfm" in result.content.lower()


@pytest.fixture
def make_eval_dataset(write_chatml_parquet) -> Callable[..., Path]:
    """Factory writing a ChatML dataset under ``<project_dir>/datasets/<name>``."""

    def _factory(project_dir: Path, samples: list[list[dict]], *, name: str) -> Path:
        return write_chatml_parquet(
            project_dir / "datasets" / name / "data.parquet", samples, audio=True,
        )

    return _factory


@pytest.fixture
def make_scorer() -> Callable[[Path, str, str], Path]:
    """Factory writing a scorer markdown file."""

    def _factory(project_dir: Path, name: str, body: str) -> Path:
        path = project_dir / "evals" / "scorers" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        return path

    return _factory


@pytest.mark.integration
class TestFullEvalWorkflowIntegration:
    """End-to-end eval workflow exercised against the live API.

    1. Create an eval dataset with questions + reference answers.
    2. Write a scoring rubric.
    3. Run ``model_eval`` with a specific model + system prompt.
    4. Inspect the artifacts.
    """

    async def test_full_eval_workflow(
        self, tmp_path: Path, make_eval_dataset, make_scorer,
    ) -> None:
        make_eval_dataset(tmp_path, [
            [
                {"role": "user", "content": "What is the chemical symbol for water?"},
                {"role": "assistant", "content": "H2O"},
            ],
            [
                {"role": "user", "content": "How many legs does a spider have?"},
                {"role": "assistant", "content": "A spider has 8 legs."},
            ],
            [
                {"role": "user", "content": "What is the speed of light in vacuum, approximately?"},
                {"role": "assistant", "content": "Approximately 300,000 km/s or about 3 x 10^8 m/s."},
            ],
        ], name="qa_eval")

        make_scorer(
            tmp_path, "factual_qa.md",
            "# Factual QA Scoring\n\n"
            "Score the assistant's response for factual accuracy.\n\n"
            "## Criteria\n"
            "- **10**: Perfectly correct, complete answer\n"
            "- **7-9**: Correct with minor omissions or extra detail\n"
            "- **4-6**: Partially correct, key information missing or imprecise\n"
            "- **1-3**: Mostly wrong or irrelevant\n\n"
            "Focus on whether the core fact is correct.\n",
        )

        progress_calls: list[tuple[int, int]] = []

        def on_progress(completed: int, total: int, _concurrency: int) -> None:
            progress_calls.append((completed, total))

        result = await handle_run_scoring(
            tmp_path,
            dataset="datasets/qa_eval",
            scorer="evals/scorers/factual_qa.md",
            mode="model_eval",
            run_name="baseline_small",
            model_size="small",
            inference_model="small",
            inference_system_prompt="Answer factual questions accurately and concisely.",
            on_pipeline_progress=on_progress,
        )

        assert "Model evaluation complete" in result.content
        assert "Error" not in result.content
        # No failures before "scored" in the message.
        assert "failed" not in result.content.lower().split("scored")[0]

        run_dir = tmp_path / "evals" / "runs" / "baseline_small"
        for expected in ("results.parquet", "summary.json", "config.json"):
            assert (run_dir / expected).exists(), expected

        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["num_samples"] == 3
        assert summary["inference_model"] == "small"
        assert summary["inference_system_prompt"] == (
            "Answer factual questions accurately and concisely."
        )
        assert summary["scores"]["mean"] > 0

        config = json.loads((run_dir / "config.json").read_text())
        assert config["inference_model"] == "small"
        assert config["eval_dataset"] == "datasets/qa_eval"
        assert config["scorer"] == "evals/scorers/factual_qa.md"

        results = pq.read_table(run_dir / "results.parquet")
        assert len(results) == 3
        scores = [results.column("score")[i].as_py() for i in range(len(results))]
        assert all(isinstance(s, (int, float)) for s in scores)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    @pytest.mark.parametrize("model", ["small", "medium"])
    async def test_compare_two_models(
        self,
        tmp_path: Path,
        make_eval_dataset,
        make_scorer,
        model: str,
    ) -> None:
        """Run the same eval with each model and verify it stores the model id.

        Per-parameter; ``test_two_models_coexist_in_same_project`` covers
        the cross-run filesystem layout.
        """
        make_eval_dataset(tmp_path, [
            [
                {"role": "user", "content": "Explain what photosynthesis is in one sentence."},
                {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."},
            ],
            [
                {"role": "user", "content": "What causes rain?"},
                {"role": "assistant", "content": "Rain is caused by water vapor condensing in clouds and falling when droplets become heavy enough."},
            ],
        ], name="compare_eval")

        make_scorer(
            tmp_path, "explain.md",
            "Score the response for clarity and accuracy of explanation.\n10 = perfect, 1 = useless.\n",
        )

        result = await handle_run_scoring(
            tmp_path,
            dataset="datasets/compare_eval",
            scorer="evals/scorers/explain.md",
            mode="model_eval",
            run_name=f"run_{model}",
            inference_model=model,
            inference_system_prompt="Explain clearly and accurately.",
        )
        assert "Model evaluation complete" in result.content

        summary = json.loads(
            (tmp_path / "evals" / "runs" / f"run_{model}" / "summary.json").read_text()
        )
        assert summary["inference_model"] == model
        assert summary["num_scored"] > 0
        assert summary["scores"]["mean"] > 0

    async def test_two_models_coexist_in_same_project(
        self, tmp_path: Path, make_eval_dataset, make_scorer,
    ) -> None:
        """Two ``model_eval`` runs in the same project_dir produce sibling run dirs."""
        make_eval_dataset(tmp_path, [[
            {"role": "user", "content": "What causes rain?"},
            {"role": "assistant", "content": "Water vapor condensing in clouds."},
        ]], name="coexist_eval")

        make_scorer(tmp_path, "explain.md", "Score 1-10.")

        for model in ("small", "medium"):
            result = await handle_run_scoring(
                tmp_path,
                dataset="datasets/coexist_eval",
                scorer="evals/scorers/explain.md",
                mode="model_eval",
                run_name=f"run_{model}",
                inference_model=model,
                inference_system_prompt="Be brief.",
            )
            assert "Model evaluation complete" in result.content

        for model in ("small", "medium"):
            summary = json.loads(
                (tmp_path / "evals" / "runs" / f"run_{model}" / "summary.json").read_text()
            )
            assert summary["inference_model"] == model

    async def test_different_system_prompts_same_model(
        self, tmp_path: Path, make_eval_dataset, make_scorer,
    ) -> None:
        """Same model, different system prompts — the prompt-optimization use case."""
        make_eval_dataset(tmp_path, [[
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI where systems learn from data."},
        ]], name="prompt_eval")

        make_scorer(
            tmp_path, "concise.md",
            "Score for conciseness. Shorter, clearer answers score higher.\n"
            "10 = maximally concise and correct. 1 = verbose or wrong.\n",
        )

        verbose = await handle_run_scoring(
            tmp_path,
            dataset="datasets/prompt_eval",
            scorer="evals/scorers/concise.md",
            mode="model_eval",
            run_name="prompt_verbose",
            inference_model="small",
            inference_system_prompt="You are a helpful assistant. Give detailed, thorough explanations.",
        )
        assert "complete" in verbose.content

        concise = await handle_run_scoring(
            tmp_path,
            dataset="datasets/prompt_eval",
            scorer="evals/scorers/concise.md",
            mode="model_eval",
            run_name="prompt_concise",
            inference_model="small",
            inference_system_prompt="Answer in one sentence maximum. Be precise.",
        )
        assert "complete" in concise.content

        summary_a = json.loads(
            (tmp_path / "evals" / "runs" / "prompt_verbose" / "summary.json").read_text()
        )
        summary_b = json.loads(
            (tmp_path / "evals" / "runs" / "prompt_concise" / "summary.json").read_text()
        )
        assert "detailed" in summary_a["inference_system_prompt"]
        assert "one sentence" in summary_b["inference_system_prompt"]

    async def test_system_prompt_path_workflow(
        self, tmp_path: Path, make_eval_dataset, make_scorer,
    ) -> None:
        """Eval using ``system_prompt_path``: write prompt file, run eval, extract failures."""
        make_eval_dataset(tmp_path, [
            [
                {"role": "user", "content": "What is the capital of Germany?"},
                {"role": "assistant", "content": "Berlin."},
            ],
            [
                {"role": "user", "content": "Name a prime number greater than 10."},
                {"role": "assistant", "content": "11 is a prime number."},
            ],
        ], name="prompt_path_eval")

        make_scorer(tmp_path, "qa.md", "Score for factual accuracy, 1-10.")

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "qa_v1.md").write_text(
            "You are a factual Q&A assistant. Answer accurately and concisely."
        )

        result = await handle_run_scoring(
            tmp_path,
            dataset="datasets/prompt_path_eval",
            scorer="evals/scorers/qa.md",
            mode="model_eval",
            run_name="qa_prompt_v1_iter1",
            inference_model="small",
            system_prompt_path="prompts/qa_v1.md",
        )
        assert "complete" in result.content

        config = json.loads(
            (tmp_path / "evals" / "runs" / "qa_prompt_v1_iter1" / "config.json").read_text()
        )
        assert config["system_prompt_path"] == "prompts/qa_v1.md"
        assert "factual Q&A" in config["inference_system_prompt"]

        failures = await handle_get_eval_failures(
            tmp_path,
            eval_run="evals/runs/qa_prompt_v1_iter1",
            threshold=6.0,
            min_failures=1,
        )
        assert "Failure Cases" in failures.content
        assert "Score:" in failures.content
