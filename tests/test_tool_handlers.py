"""Tests for tool handlers (lqh/tools/handlers.py).

Unit tests verify parameter validation and dispatch logic.
Integration tests exercise handle_list_models and handle_run_scoring against the API.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lqh.tools.handlers import (
    ToolResult,
    _validate_path,
    handle_create_file,
    handle_edit_file,
    handle_get_eval_failures,
    handle_list_files,
    handle_read_file,
    handle_run_scoring,
    handle_write_file,
    execute_tool,
)


# ---------------------------------------------------------------------------
# Unit tests for validation logic (no network)
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


class TestRunScoringValidation(unittest.TestCase):
    """Verify parameter validation in handle_run_scoring."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)
        # Create a minimal dataset
        ds_dir = self.project_dir / "datasets" / "test_ds"
        ds_dir.mkdir(parents=True)
        table = pa.table(
            {
                "messages": [json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])],
                "audio": [None],
            },
            schema=pa.schema([pa.field("messages", pa.string()), pa.field("audio", pa.string())]),
        )
        pq.write_table(table, ds_dir / "data.parquet")

        # Create a scorer
        scorer_dir = self.project_dir / "evals" / "scorers"
        scorer_dir.mkdir(parents=True)
        (scorer_dir / "test.md").write_text("Score for quality 1-10.")

    def test_model_eval_requires_run_name(self) -> None:
        result = asyncio.run(
            handle_run_scoring(
                self.project_dir,
                dataset="datasets/test_ds",
                scorer="evals/scorers/test.md",
                mode="model_eval",
                inference_model="small",
                # run_name is missing
            )
        )
        self.assertIn("run_name is required", result.content)

    def test_model_eval_requires_inference_model(self) -> None:
        result = asyncio.run(
            handle_run_scoring(
                self.project_dir,
                dataset="datasets/test_ds",
                scorer="evals/scorers/test.md",
                mode="model_eval",
                run_name="test_run",
                # inference_model is missing
            )
        )
        self.assertIn("inference_model is required", result.content)

    def test_missing_dataset(self) -> None:
        result = asyncio.run(
            handle_run_scoring(
                self.project_dir,
                dataset="datasets/nonexistent",
                scorer="evals/scorers/test.md",
                mode="data_quality",
            )
        )
        self.assertIn("Error", result.content)

    def test_missing_scorer(self) -> None:
        result = asyncio.run(
            handle_run_scoring(
                self.project_dir,
                dataset="datasets/test_ds",
                scorer="evals/scorers/nonexistent.md",
                mode="data_quality",
            )
        )
        self.assertIn("Error", result.content)

    def test_unknown_mode(self) -> None:
        result = asyncio.run(
            handle_run_scoring(
                self.project_dir,
                dataset="datasets/test_ds",
                scorer="evals/scorers/test.md",
                mode="invalid_mode",
            )
        )
        self.assertIn("unknown mode", result.content)

    def test_duplicate_run_name_rejected(self) -> None:
        """If an eval run directory already exists, should error."""
        existing = self.project_dir / "evals" / "runs" / "existing_run"
        existing.mkdir(parents=True)

        result = asyncio.run(
            handle_run_scoring(
                self.project_dir,
                dataset="datasets/test_ds",
                scorer="evals/scorers/test.md",
                mode="model_eval",
                run_name="existing_run",
                inference_model="small",
            )
        )
        self.assertIn("already exists", result.content)


class TestGetEvalFailures(unittest.TestCase):
    """Tests for the get_eval_failures tool handler."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)

    def _make_eval_run(self, scores: list[float]) -> str:
        """Create an eval run with results.parquet containing given scores."""
        run_dir = self.project_dir / "evals" / "runs" / "test_run"
        run_dir.mkdir(parents=True)

        rows = {
            "sample_index": list(range(len(scores))),
            "messages": [
                json.dumps([{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}])
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
        return "evals/runs/test_run"

    def test_returns_failures(self) -> None:
        eval_run = self._make_eval_run([2.0, 4.0, 7.0, 9.0, 8.0])
        result = asyncio.run(
            handle_get_eval_failures(
                self.project_dir, eval_run=eval_run, threshold=6.0, min_failures=0,
            )
        )
        self.assertIn("Failure Cases", result.content)
        self.assertIn("Score: 2.0", result.content)
        self.assertIn("Score: 4.0", result.content)
        self.assertNotIn("Score: 7.0", result.content)

    def test_missing_results(self) -> None:
        (self.project_dir / "evals" / "runs" / "empty").mkdir(parents=True)
        result = asyncio.run(
            handle_get_eval_failures(
                self.project_dir, eval_run="evals/runs/empty",
            )
        )
        self.assertIn("Error", result.content)

    def test_no_failures(self) -> None:
        eval_run = self._make_eval_run([8.0, 9.0, 10.0])
        result = asyncio.run(
            handle_get_eval_failures(
                self.project_dir, eval_run=eval_run, threshold=6.0, min_failures=0,
            )
        )
        self.assertIn("No failure cases", result.content)

    def test_padding_works(self) -> None:
        eval_run = self._make_eval_run([8.0, 9.0, 7.0])
        result = asyncio.run(
            handle_get_eval_failures(
                self.project_dir, eval_run=eval_run, threshold=6.0, min_failures=2,
            )
        )
        # All above threshold but min_failures=2, so bottom 2 returned
        self.assertIn("Failure Cases (2 of 3", result.content)


class TestFileToolHandlers(unittest.TestCase):
    """Basic tests for file manipulation tools."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)

    def test_create_and_read_file(self) -> None:
        result = asyncio.run(
            handle_create_file(self.project_dir, path="test.txt", content="hello world")
        )
        self.assertIn("Created", result.content)

        result = asyncio.run(
            handle_read_file(self.project_dir, path="test.txt")
        )
        self.assertIn("hello world", result.content)

    def test_create_file_rejects_existing(self) -> None:
        (self.project_dir / "exists.txt").write_text("x")
        result = asyncio.run(
            handle_create_file(self.project_dir, path="exists.txt", content="y")
        )
        self.assertIn("already exists", result.content)

    def test_write_file_overwrites(self) -> None:
        (self.project_dir / "file.txt").write_text("old")
        result = asyncio.run(
            handle_write_file(self.project_dir, path="file.txt", content="new")
        )
        self.assertIn("Wrote", result.content)
        self.assertEqual((self.project_dir / "file.txt").read_text(), "new")

    def test_edit_file(self) -> None:
        (self.project_dir / "file.txt").write_text("hello world")
        result = asyncio.run(
            handle_edit_file(self.project_dir, path="file.txt", old_string="world", new_string="earth")
        )
        self.assertIn("Edited", result.content)
        self.assertEqual((self.project_dir / "file.txt").read_text(), "hello earth")

    def test_edit_file_rejects_non_unique(self) -> None:
        (self.project_dir / "file.txt").write_text("aaa bbb aaa")
        result = asyncio.run(
            handle_edit_file(self.project_dir, path="file.txt", old_string="aaa", new_string="ccc")
        )
        self.assertIn("found 2 times", result.content)

    def test_list_files(self) -> None:
        (self.project_dir / "a.txt").write_text("a")
        (self.project_dir / "b.txt").write_text("b")
        result = asyncio.run(
            handle_list_files(self.project_dir)
        )
        self.assertIn("a.txt", result.content)
        self.assertIn("b.txt", result.content)

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(ValueError, msg="outside the project"):
            asyncio.run(
                handle_read_file(self.project_dir, path="../../etc/passwd")
            )


class TestExecuteToolDispatch(unittest.TestCase):
    """Test the execute_tool dispatch function."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmpdir)

    def test_unknown_tool(self) -> None:
        result = asyncio.run(
            execute_tool("nonexistent_tool", {}, self.project_dir)
        )
        self.assertIn("unknown tool", result.content)

    def test_dispatches_to_correct_handler(self) -> None:
        (self.project_dir / "test.md").write_text("# Test")
        result = asyncio.run(
            execute_tool("read_file", {"path": "test.md"}, self.project_dir)
        )
        self.assertIn("# Test", result.content)


# ---------------------------------------------------------------------------
# Integration tests (require API access)
# ---------------------------------------------------------------------------


def _has_api_access() -> bool:
    try:
        from lqh.auth import get_token
        return get_token() is not None
    except Exception:
        return False


@unittest.skipUnless(_has_api_access(), "No API access (set LQH_DEBUG_API_KEY or run /login)")
class TestListModelsIntegration(unittest.TestCase):
    """Integration test for the list_models tool handler."""

    def test_list_models_returns_models(self) -> None:
        from lqh.tools.handlers import handle_list_models

        result = asyncio.run(handle_list_models())
        self.assertIn("Liquid Foundation Models", result.content)
        self.assertIn("orchestration", result.content)
        # Should contain at least one LFM model ID
        self.assertIn("lfm", result.content.lower())


@unittest.skipUnless(_has_api_access(), "No API access (set LQH_DEBUG_API_KEY or run /login)")
class TestFullEvalWorkflowIntegration(unittest.TestCase):
    """End-to-end eval workflow: create dataset -> create scorer -> run eval -> check results.

    This is the core workflow that a user would run through the agent:
    1. Have an eval dataset with questions + reference answers
    2. Create scoring criteria
    3. Run model_eval with a specific model + system prompt
    4. Inspect results
    """

    def test_full_eval_workflow(self) -> None:
        """Complete eval run: dataset -> scorer -> model_eval -> verify outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # --- Step 1: Create eval dataset ---
            eval_dataset_dir = project_dir / "datasets" / "qa_eval"
            eval_dataset_dir.mkdir(parents=True)

            samples = [
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
            ]
            messages = [json.dumps(s) for s in samples]
            table = pa.table(
                {"messages": messages, "audio": [None] * len(messages)},
                schema=pa.schema([pa.field("messages", pa.string()), pa.field("audio", pa.string())]),
            )
            pq.write_table(table, eval_dataset_dir / "data.parquet")

            # --- Step 2: Create scorer ---
            scorers_dir = project_dir / "evals" / "scorers"
            scorers_dir.mkdir(parents=True)
            scorer_path = scorers_dir / "factual_qa.md"
            scorer_path.write_text(
                "# Factual QA Scoring\n\n"
                "Score the assistant's response for factual accuracy.\n\n"
                "## Criteria\n"
                "- **10**: Perfectly correct, complete answer\n"
                "- **7-9**: Correct with minor omissions or extra detail\n"
                "- **4-6**: Partially correct, key information missing or imprecise\n"
                "- **1-3**: Mostly wrong or irrelevant\n\n"
                "Focus on whether the core fact is correct. Do not penalize\n"
                "for verbosity or formatting differences.\n"
            )

            # --- Step 3: Run model_eval via the handler ---
            progress_calls: list[tuple[int, int]] = []

            def on_progress(completed: int, total: int, _concurrency: int) -> None:
                progress_calls.append((completed, total))

            result = asyncio.run(
                handle_run_scoring(
                    project_dir,
                    dataset="datasets/qa_eval",
                    scorer="evals/scorers/factual_qa.md",
                    mode="model_eval",
                    run_name="baseline_small",
                    model_size="small",
                    inference_model="small",
                    inference_system_prompt="Answer factual questions accurately and concisely.",
                    on_pipeline_progress=on_progress,
                )
            )

            # --- Step 4: Verify results ---
            self.assertIn("Model evaluation complete", result.content)
            self.assertNotIn("Error", result.content)
            self.assertNotIn("failed", result.content.lower().split("scored")[0])  # no failures before "scored"

            # Check output files exist
            run_dir = project_dir / "evals" / "runs" / "baseline_small"
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "results.parquet").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "config.json").exists())

            # Verify summary.json contents
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["num_samples"], 3)
            self.assertEqual(summary["inference_model"], "small")
            self.assertEqual(
                summary["inference_system_prompt"],
                "Answer factual questions accurately and concisely.",
            )
            self.assertIn("scores", summary)
            self.assertIn("mean", summary["scores"])
            # Factual questions should score reasonably well
            self.assertGreater(summary["scores"]["mean"], 0)

            # Verify config.json contents
            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(config["inference_model"], "small")
            self.assertEqual(
                config["inference_system_prompt"],
                "Answer factual questions accurately and concisely.",
            )
            self.assertEqual(config["eval_dataset"], "datasets/qa_eval")
            self.assertEqual(config["scorer"], "evals/scorers/factual_qa.md")

            # Verify results.parquet
            results_table = pq.read_table(run_dir / "results.parquet")
            self.assertEqual(len(results_table), 3)
            scores = [results_table.column("score")[i].as_py() for i in range(len(results_table))]
            self.assertTrue(all(isinstance(s, (int, float)) for s in scores))

            # Verify progress was reported
            self.assertEqual(len(progress_calls), 3)
            self.assertEqual(progress_calls[-1], (3, 3))

    def test_compare_two_models(self) -> None:
        """Run the same eval with two different models and compare scores.

        This is the core use case: "which model is better for my task?"
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Setup eval dataset
            eval_dir = project_dir / "datasets" / "compare_eval"
            eval_dir.mkdir(parents=True)

            samples = [
                [
                    {"role": "user", "content": "Explain what photosynthesis is in one sentence."},
                    {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."},
                ],
                [
                    {"role": "user", "content": "What causes rain?"},
                    {"role": "assistant", "content": "Rain is caused by water vapor condensing in clouds and falling when droplets become heavy enough."},
                ],
            ]
            messages = [json.dumps(s) for s in samples]
            table = pa.table(
                {"messages": messages, "audio": [None] * len(messages)},
                schema=pa.schema([pa.field("messages", pa.string()), pa.field("audio", pa.string())]),
            )
            pq.write_table(table, eval_dir / "data.parquet")

            # Scorer
            scorers_dir = project_dir / "evals" / "scorers"
            scorers_dir.mkdir(parents=True)
            (scorers_dir / "explain.md").write_text(
                "Score the response for clarity and accuracy of explanation.\n"
                "10 = perfect, 1 = useless.\n"
            )

            # --- Run 1: small model ---
            result_small = asyncio.run(
                handle_run_scoring(
                    project_dir,
                    dataset="datasets/compare_eval",
                    scorer="evals/scorers/explain.md",
                    mode="model_eval",
                    run_name="run_small",
                    inference_model="small",
                    inference_system_prompt="Explain clearly and accurately.",
                )
            )
            self.assertIn("Model evaluation complete", result_small.content)

            # --- Run 2: medium model ---
            result_medium = asyncio.run(
                handle_run_scoring(
                    project_dir,
                    dataset="datasets/compare_eval",
                    scorer="evals/scorers/explain.md",
                    mode="model_eval",
                    run_name="run_medium",
                    inference_model="medium",
                    inference_system_prompt="Explain clearly and accurately.",
                )
            )
            self.assertIn("Model evaluation complete", result_medium.content)

            # --- Compare ---
            summary_small = json.loads(
                (project_dir / "evals" / "runs" / "run_small" / "summary.json").read_text()
            )
            summary_medium = json.loads(
                (project_dir / "evals" / "runs" / "run_medium" / "summary.json").read_text()
            )

            # Both should have scored something
            self.assertGreater(summary_small["num_scored"], 0)
            self.assertGreater(summary_medium["num_scored"], 0)
            self.assertGreater(summary_small["scores"]["mean"], 0)
            self.assertGreater(summary_medium["scores"]["mean"], 0)

            # Verify they used different models
            self.assertEqual(summary_small["inference_model"], "small")
            self.assertEqual(summary_medium["inference_model"], "medium")

    def test_different_system_prompts_same_model(self) -> None:
        """Same model, different system prompts — the prompt optimization use case."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            eval_dir = project_dir / "datasets" / "prompt_eval"
            eval_dir.mkdir(parents=True)

            samples = [
                [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of AI where systems learn from data."},
                ],
            ]
            messages = [json.dumps(s) for s in samples]
            table = pa.table(
                {"messages": messages, "audio": [None] * len(messages)},
                schema=pa.schema([pa.field("messages", pa.string()), pa.field("audio", pa.string())]),
            )
            pq.write_table(table, eval_dir / "data.parquet")

            scorers_dir = project_dir / "evals" / "scorers"
            scorers_dir.mkdir(parents=True)
            (scorers_dir / "concise.md").write_text(
                "Score for conciseness. Shorter, clearer answers score higher.\n"
                "10 = maximally concise and correct. 1 = verbose or wrong.\n"
            )

            # Prompt A: verbose
            result_a = asyncio.run(
                handle_run_scoring(
                    project_dir,
                    dataset="datasets/prompt_eval",
                    scorer="evals/scorers/concise.md",
                    mode="model_eval",
                    run_name="prompt_verbose",
                    inference_model="small",
                    inference_system_prompt="You are a helpful assistant. Give detailed, thorough explanations.",
                )
            )
            self.assertIn("complete", result_a.content)

            # Prompt B: concise
            result_b = asyncio.run(
                handle_run_scoring(
                    project_dir,
                    dataset="datasets/prompt_eval",
                    scorer="evals/scorers/concise.md",
                    mode="model_eval",
                    run_name="prompt_concise",
                    inference_model="small",
                    inference_system_prompt="Answer in one sentence maximum. Be precise.",
                )
            )
            self.assertIn("complete", result_b.content)

            # Both ran successfully with different prompts stored
            summary_a = json.loads(
                (project_dir / "evals" / "runs" / "prompt_verbose" / "summary.json").read_text()
            )
            summary_b = json.loads(
                (project_dir / "evals" / "runs" / "prompt_concise" / "summary.json").read_text()
            )
            self.assertIn("detailed", summary_a["inference_system_prompt"])
            self.assertIn("one sentence", summary_b["inference_system_prompt"])

    def test_system_prompt_path_workflow(self) -> None:
        """Eval using system_prompt_path: write prompt file, run eval, extract failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Create eval dataset
            eval_dir = project_dir / "datasets" / "prompt_path_eval"
            eval_dir.mkdir(parents=True)
            samples = [
                [
                    {"role": "user", "content": "What is the capital of Germany?"},
                    {"role": "assistant", "content": "Berlin."},
                ],
                [
                    {"role": "user", "content": "Name a prime number greater than 10."},
                    {"role": "assistant", "content": "11 is a prime number."},
                ],
            ]
            messages = [json.dumps(s) for s in samples]
            table = pa.table(
                {"messages": messages, "audio": [None] * len(messages)},
                schema=pa.schema([pa.field("messages", pa.string()), pa.field("audio", pa.string())]),
            )
            pq.write_table(table, eval_dir / "data.parquet")

            # Create scorer
            scorers_dir = project_dir / "evals" / "scorers"
            scorers_dir.mkdir(parents=True)
            (scorers_dir / "qa.md").write_text("Score for factual accuracy, 1-10.")

            # Create a prompt file (the key new feature)
            prompts_dir = project_dir / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "qa_v1.md").write_text(
                "You are a factual Q&A assistant. Answer accurately and concisely."
            )

            # Run eval with system_prompt_path
            result = asyncio.run(
                handle_run_scoring(
                    project_dir,
                    dataset="datasets/prompt_path_eval",
                    scorer="evals/scorers/qa.md",
                    mode="model_eval",
                    run_name="qa_prompt_v1_iter1",
                    inference_model="small",
                    system_prompt_path="prompts/qa_v1.md",
                )
            )
            self.assertIn("complete", result.content)

            # Verify config.json has system_prompt_path
            config = json.loads(
                (project_dir / "evals" / "runs" / "qa_prompt_v1_iter1" / "config.json").read_text()
            )
            self.assertEqual(config["system_prompt_path"], "prompts/qa_v1.md")
            self.assertIn("factual Q&A", config["inference_system_prompt"])

            # Extract failures
            fail_result = asyncio.run(
                handle_get_eval_failures(
                    project_dir,
                    eval_run="evals/runs/qa_prompt_v1_iter1",
                    threshold=6.0,
                    min_failures=1,
                )
            )
            # Should return at least 1 sample (min_failures=1)
            self.assertIn("Failure Cases", fail_result.content)
            self.assertIn("Score:", fail_result.content)


if __name__ == "__main__":
    unittest.main()
