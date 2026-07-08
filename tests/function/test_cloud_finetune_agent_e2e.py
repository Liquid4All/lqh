"""Agent-level cloud fine-tuning e2e tests.

These tests intentionally run the full Python agent loop and tool dispatch,
but replace the orchestration model and CloudBackend with deterministic fakes.
That keeps the tests cheap and repeatable while still proving the current
fine-tuning contract:

* the agent loads the train skill,
* asks permission through the normal start_training path,
* launches on LQH Cloud without a remote argument,
* submits a sweep payload with distinct train/eval datasets and a scorer,
* reads the result back through training_status.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lqh.remote.backend import JobStatus
from tests.harness.harness import E2EHarness
from tests.harness.scenarios import Scenario


BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"


class _FakeCloudBackend:
    submissions: list[dict[str, Any]] = []

    def __init__(self, config: Any, project_dir: Path) -> None:
        self.config = config
        self.project_dir = Path(project_dir)

    async def submit_run(
        self,
        local_run_dir: str,
        config: dict[str, Any],
        module: str = "lqh.train",
    ) -> str:
        run_dir = Path(local_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"fake-cloud-job-{len(self.submissions) + 1}"
        self.submissions.append({
            "job_id": job_id,
            "run_dir": run_dir,
            "config": config,
            "module": module,
        })

        mode = _inner_config(config).get("type", "sft")
        _write_finished_cloud_run(run_dir, config, module, job_id, mode)
        return job_id

    async def sync_progress(self, remote_run_dir: str, local_run_dir: str) -> None:
        run_dir = Path(local_run_dir)
        if not (run_dir / "status.json").exists():
            match = next(
                (s for s in self.submissions if s["run_dir"] == run_dir),
                None,
            )
            if match:
                _write_finished_cloud_run(
                    run_dir,
                    match["config"],
                    match["module"],
                    match["job_id"],
                    _inner_config(match["config"]).get("type", "sft"),
                )

    async def poll_status(self, job_id: str) -> JobStatus:
        return JobStatus(state="completed", current_step=6)

    async def teardown(self, job_id: str) -> None:
        return None


class _FakeHumanClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        return _chat_response(content="Start training")


def _tool_response(name: str, args: dict[str, Any], *, content: str = "") -> Any:
    message = SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(
                id=f"call_{name}",
                type="function",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args),
                ),
            )
        ],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
    )


def _chat_response(content: str) -> Any:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
    )


def _inner_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("type") == "sweep":
        return config.get("base_config", {})
    return config


def _write_finished_cloud_run(
    run_dir: Path,
    config: dict[str, Any],
    module: str,
    job_id: str,
    mode: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "remote_job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "remote_name": "cloud",
                "remote_run_dir": f"cloud:{job_id}",
                "module": module,
                "kind": "train",
                "backend": "cloud",
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "cloud_state.json").write_text(
        json.dumps({"job_id": job_id, "status": "completed", "last_seq": 3}) + "\n"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"state": "completed", "last_update": "2026-06-03T00:00:00Z"}) + "\n"
    )

    if mode == "on_policy_dpo":
        rows = [
            {
                "config_id": "dpo_beta0.1_lr5e-6",
                "primary": 0.42,
                "rc": 0,
                "eval_ce_chosen_delta_ref": -0.08,
                "eval_ce_chosen_p90": 0.71,
                "overrides": {"training": {"learning_rate": 5e-6}, "dpo_beta": 0.1},
            }
        ]
        summary = {
            "mode": "on_policy_dpo",
            "grid_size": "small",
            "n_configs": 1,
            "n_completed": 1,
            "proxy_key": "eval_ce_chosen_mean",
            "rows": rows,
            "winner": {"config_id": rows[0]["config_id"], "primary": rows[0]["primary"]},
        }
        iter_dir = run_dir / "iterations" / "iter_001"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "preference_stats.json").write_text(
            json.dumps(
                {
                    "kept": 8,
                    "pairs_with_both_scored": 8,
                    "gap_p50": 2.0,
                    "gap_p90": 4.0,
                },
                indent=2,
            )
            + "\n"
        )
        (iter_dir / "held_out_eval.json").write_text(
            json.dumps({"mean": 7.2, "delta_vs_baseline": 0.6}, indent=2) + "\n"
        )
        progress = [
            {"phase": "sweep_start", "step": 0, "run_type": "on_policy_dpo"},
            {"phase": "sweep_config_done", "step": 1, "primary": 0.42},
            {"status": "completed", "winner": rows[0]["config_id"]},
        ]
    else:
        rows = [
            {
                "config_id": "sft_lr2e-5_e3",
                "primary": 1.23,
                "rc": 0,
                "overrides": {"training": {"learning_rate": 2e-5, "num_epochs": 3}},
            }
        ]
        summary = {
            "mode": "sft",
            "grid_size": "small",
            "n_configs": 1,
            "n_completed": 1,
            "proxy_key": "eval_loss",
            "rows": rows,
            "winner": {"config_id": rows[0]["config_id"], "primary": rows[0]["primary"]},
        }
        progress = [
            {"phase": "sweep_start", "step": 0, "run_type": "sft"},
            {"phase": "sweep_config_done", "step": 1, "eval_loss": 1.23, "primary": 1.23},
            {"status": "completed", "winner": rows[0]["config_id"]},
        ]

    (run_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (run_dir / "progress.jsonl").open("w") as fh:
        for row in progress:
            fh.write(json.dumps(row) + "\n")


def _write_chatml_dataset(path: Path, pairs: list[tuple[str, str]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for user, assistant in pairs:
        rows.append(
            json.dumps(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"messages": rows}), path)


def _seed_translation_training_project(project_dir: Path) -> None:
    from lqh.remote.compute import save_project_default

    (project_dir / "SPEC.md").write_text(
        "# Translation Fine-Tune\n\n"
        "Translate short English support phrases to concise German. Output only the translation.\n",
        encoding="utf-8",
    )
    _write_chatml_dataset(
        project_dir / "datasets" / "translation_v1" / "data.parquet",
        [
            ("hello", "hallo"),
            ("thank you", "danke"),
            ("goodbye", "auf wiedersehen"),
            ("water", "wasser"),
        ],
    )
    _write_chatml_dataset(
        project_dir / "datasets" / "translation_v1_eval" / "data.parquet",
        [
            ("please", "bitte"),
            ("yes", "ja"),
        ],
    )
    scorer = project_dir / "evals" / "scorers" / "translation_v1.md"
    scorer.parent.mkdir(parents=True, exist_ok=True)
    scorer.write_text(
        "# Translation scorer\n\n"
        "Score 0-10 for correct German translation and no extra text. "
        "Return JSON with score and reasoning.\n",
        encoding="utf-8",
    )
    save_project_default(project_dir, "cloud")


def _scenario(name: str, train_type: str) -> Scenario:
    task = "DPO" if train_type == "on_policy_dpo" else "SFT"
    return Scenario(
        name=name,
        description=(
            f"The project already has train data, held-out eval data, and a scorer. "
            f"Start cloud {task} fine-tuning now, accept permission prompts, then "
            f"check training_status for the finished run."
        ),
        initial_message=f"Start cloud {task} fine-tuning now and check the result.",
        max_turns=4,
        seed_fn=_seed_translation_training_project,
    )


def _training_args(train_type: str, run_name: str) -> dict[str, Any]:
    args: dict[str, Any] = {
        "type": train_type,
        "base_model": BASE_MODEL,
        "dataset": "datasets/translation_v1",
        "eval_dataset": "datasets/translation_v1_eval",
        "scorer": "evals/scorers/translation_v1.md",
        "run_name": run_name,
        "lora": True,
    }
    if train_type == "on_policy_dpo":
        args.update({"num_iterations": 1, "dpo_beta": 0.1, "golden_source": "dataset"})
    return args


def _install_fakes(monkeypatch: pytest.MonkeyPatch, train_type: str, run_name: str) -> None:
    _FakeCloudBackend.submissions = []

    responses = [
        _tool_response("load_skill", {"skill_name": "train"}, content="Loading training workflow."),
        _tool_response(
            "start_training",
            _training_args(train_type, run_name),
            content="Starting cloud fine-tuning with the held-out eval set and scorer.",
        ),
        _tool_response("training_status", {"run_name": run_name}, content="Checking the finished run."),
        _chat_response("Cloud fine-tuning completed. The sweep winner is available in training_status."),
    ]

    async def fake_chat_with_retry(*args: Any, **kwargs: Any) -> Any:
        if responses:
            return responses.pop(0)
        return _chat_response("Done.")

    monkeypatch.setattr("lqh.agent.get_token", lambda: "test-token")
    monkeypatch.setattr("lqh.agent.create_client", lambda token, base_url: object())
    monkeypatch.setattr("lqh.agent.chat_with_retry", fake_chat_with_retry)
    monkeypatch.setattr("tests.harness.harness.require_token", lambda: "test-token")
    monkeypatch.setattr(
        "tests.harness.harness.create_client",
        lambda token, base_url: _FakeHumanClient(),
    )
    monkeypatch.setattr("lqh.remote.cloud.CloudBackend", _FakeCloudBackend)


def _run_agent_e2e(monkeypatch: pytest.MonkeyPatch, train_type: str, run_name: str):
    _install_fakes(monkeypatch, train_type, run_name)
    harness = E2EHarness(_scenario(f"cloud_{train_type}_agent", train_type))
    return asyncio.run(harness.run())


def _assert_common_training_contract(result: Any, train_type: str, run_name: str) -> dict[str, Any]:
    assert result.errors == []
    assert {"load_skill", "start_training", "training_status"} <= result.tools_called()
    assert "train" in result.skills_loaded

    assert len(_FakeCloudBackend.submissions) == 1
    submission = _FakeCloudBackend.submissions[0]
    assert submission["module"] == "lqh.train.sweep"

    outer = submission["config"]
    assert outer["type"] == "sweep"
    inner = outer["base_config"]
    assert inner["type"] == train_type
    assert inner["base_model"] == BASE_MODEL
    assert inner["dataset"] == "datasets/translation_v1/data.parquet"
    assert inner["eval_dataset"] == "datasets/translation_v1_eval/data.parquet"
    assert inner["dataset"] != inner["eval_dataset"]
    assert inner["scorer"] == "evals/scorers/translation_v1.md"
    assert inner["eval_on_checkpoints"] is True
    assert {"dataset", "eval_dataset", "scorer"} <= set(inner["manifest"])

    run_dir = result.project_dir / "runs" / run_name
    assert (run_dir / "remote_job.json").exists()
    assert json.loads((run_dir / "remote_job.json").read_text())["backend"] == "cloud"
    assert json.loads((run_dir / "status.json").read_text())["state"] == "completed"

    status_results = [
        t.content
        for t in result.transcript
        if t.role == "tool_result" and t.tool_name == "training_status"
    ]
    assert status_results, "agent never read training_status"
    assert "completed (remote: LQH Cloud)" in status_results[-1]
    assert "Sweep:" in status_results[-1]
    return inner


def test_agent_starts_cloud_sft_and_reads_sweep_status(monkeypatch: pytest.MonkeyPatch) -> None:
    run_name = f"sft_agent_e2e_{int(time.time())}"
    _assert_common_training_contract(
        _run_agent_e2e(monkeypatch, "sft", run_name),
        "sft",
        run_name,
    )


def test_agent_starts_cloud_dpo_and_reads_dpo_status(monkeypatch: pytest.MonkeyPatch) -> None:
    run_name = f"dpo_agent_e2e_{int(time.time())}"
    result = _run_agent_e2e(monkeypatch, "on_policy_dpo", run_name)
    inner = _assert_common_training_contract(
        result,
        "on_policy_dpo",
        run_name,
    )
    assert inner["num_iterations"] == 1
    assert inner["dpo_beta"] == 0.1

    status = [
        t.content
        for t in result.transcript
        if t.role == "tool_result" and t.tool_name == "training_status"
    ][-1]
    assert "DPO iterations:" in status
    assert "CE(ch)_mean" in status
    assert "eval_rewards" not in status
