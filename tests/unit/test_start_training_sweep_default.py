"""How `start_training` decides whether to sweep.

The policy: **SFT trains once at the defaults in lqh/train/defaults.py; DPO
sweeps.** A sweep runs its configs one after another inside a single job, so
turning it on multiplies the wait — the wrong trade on the first run after a
dataset is ready, when data volume and model size still dominate the score.
DPO keeps sweeping because it is far more sensitive to learning rate and beta
and its defaults are not covered by the SFT calibration study.

An explicit `enable_sweep` from the agent always wins, in both directions.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _make_dataset(project: Path, name: str = "ds") -> str:
    ds_dir = project / "datasets" / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "messages": [json.dumps([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])],
    })
    pq.write_table(table, ds_dir / "data.parquet")
    return f"datasets/{name}"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import lqh.remote.config as remote_config

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(remote_config, "GLOBAL_CONFIG_DIR", home / ".lqh")
    yield home


@pytest.fixture
def launch(tmp_path, isolated_home, monkeypatch):
    """Run start_training up to submission and capture the launch payload."""
    import lqh.tools.handlers as handlers
    from lqh.tools.permissions import grant_training_permission

    project = tmp_path / "proj"
    project.mkdir()
    grant_training_permission(project, project_wide=True)
    monkeypatch.setattr(handlers, "_local_gpu_available", lambda: False)

    scorer = project / "evals" / "scorers" / "s.md"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("# scorer")

    recorded: dict = {}

    async def fake_remote(
        project_dir, run_dir, config, run_name, remote_name, api_key, **kw
    ):
        recorded["config"] = config
        recorded["module"] = kw.get("module")
        return handlers.ToolResult(content="stub: cloud")

    monkeypatch.setattr(handlers, "_execute_start_training_remote", fake_remote)

    def run(**overrides):
        recorded.clear()
        kwargs = dict(
            type="sft",
            base_model="LiquidAI/LFM2.5-1.2B-Instruct",
            dataset=_make_dataset(project, "ds"),
            eval_dataset=_make_dataset(project, "ds_eval"),
            scorer="evals/scorers/s.md",
        )
        kwargs.update(overrides)
        result = asyncio.run(handlers.handle_start_training(project, **kwargs))
        recorded["result"] = result
        return recorded

    return run


def test_sft_trains_once_by_default(launch):
    rec = launch(type="sft")
    assert rec["module"] == "lqh.train"
    assert rec["config"]["type"] == "sft"
    assert "base_config" not in rec["config"]


def test_dpo_sweeps_by_default(launch):
    rec = launch(type="on_policy_dpo")
    assert rec["module"] == "lqh.train.sweep"
    assert rec["config"]["type"] == "sweep"
    assert rec["config"]["base_config"]["type"] == "on_policy_dpo"


def test_explicit_true_sweeps_sft(launch):
    """The /improve late-stage lever: sweeping SFT stays one argument away."""
    rec = launch(type="sft", enable_sweep=True)
    assert rec["module"] == "lqh.train.sweep"
    assert rec["config"]["base_config"]["type"] == "sft"


def test_explicit_false_stops_dpo_sweeping(launch):
    rec = launch(type="on_policy_dpo", enable_sweep=False)
    assert rec["module"] == "lqh.train"
    assert rec["config"]["type"] == "on_policy_dpo"


def test_default_hyperparameters_come_from_the_defaults_module(launch):
    from lqh.train import defaults

    rec = launch(type="sft")
    training = rec["config"]["training"]
    expected = defaults.recommended(run_type="sft", lora=True)
    assert training["learning_rate"] == expected.learning_rate
    assert training["num_epochs"] == expected.num_epochs
    assert training["per_device_batch_size"] == expected.per_device_batch_size
    assert rec["config"]["lora"]["r"] == expected.lora["r"]


def test_explicit_hyperparameters_win_over_the_defaults(launch):
    rec = launch(type="sft", learning_rate=7e-5, num_epochs=1)
    training = rec["config"]["training"]
    assert training["learning_rate"] == 7e-5
    assert training["num_epochs"] == 1


def test_dpo_config_carries_no_num_epochs(launch):
    """DPO is bounded by num_iterations; a stray num_epochs would be ignored
    at best and confusing at worst."""
    rec = launch(type="on_policy_dpo")
    assert "num_epochs" not in rec["config"]["base_config"]["training"]
    assert rec["config"]["base_config"]["num_iterations"] == 5
