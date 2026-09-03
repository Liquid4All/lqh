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


def _make_dataset(project: Path, name: str = "ds", rows: int = 1) -> str:
    ds_dir = project / "datasets" / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "messages": [json.dumps([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])] * rows,
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
    # The batch is dataset-derived, so the expectation must be built with the
    # same row count the handler counted (this fixture writes a 1-row parquet).
    rows = rec["config"]["dataset_rows"]["train_effective"]
    expected = defaults.recommended(run_type="sft", lora=True, train_rows=rows)
    assert training["learning_rate"] == expected.learning_rate
    assert training["num_epochs"] == expected.num_epochs
    assert training["per_device_batch_size"] == expected.per_device_batch_size
    assert rec["config"]["lora"]["r"] == expected.lora["r"]


def test_batch_size_is_derived_from_the_dataset(launch):
    """The step floor is only real if the handler passes the row count through:
    a tiny dataset must not train at the 256-row throughput batch."""
    from lqh.train import defaults

    training = launch(type="sft")["config"]["training"]
    assert training["effective_batch_size"] == defaults.SFT_MIN_EFFECTIVE_BATCH
    assert training["per_device_batch_size"] <= training["effective_batch_size"]


# ---------------------------------------------------------------------------
# Sequence length is derived from the data for text SFT — no tool argument,
# nothing in the confirmation unless rows will be skipped.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_estimate(monkeypatch):
    """Pin the submit-time estimate so tests don't depend on the fixture's
    2-message parquet or on a tokenizer."""
    from lqh.train import seq_length

    state = {"longest": 5_930, "over": 0}

    def _estimate(paths, *, base_model, project_dir, ceiling=None):
        state["paths"] = [str(p) for p in paths]
        state["base_model"] = base_model
        return seq_length.SeqEstimate(
            longest_tokens=state["longest"], rows=1,
            rows_over_ceiling=state["over"], source="tokenizer",
        )

    monkeypatch.setattr(seq_length, "estimate_longest_row_tokens", _estimate)
    return state


def test_sft_sequence_length_is_derived_from_the_data(launch, fake_estimate):
    from lqh.train import defaults

    training = launch(type="sft")["config"]["training"]
    assert training["max_seq_length"] == defaults.derived_seq_length(5_930) == 6144
    assert training["auto_seq_length"] is True
    # Train AND eval files feed the estimate; the longest eval row matters too.
    assert any("ds_eval" in p for p in fake_estimate["paths"])
    assert any("/ds/" in p for p in fake_estimate["paths"])


def test_sft_sequence_length_without_a_tokenizer_still_derives(launch):
    """No stub: the conftest blocks the Hub, so the character fallback runs.
    The 2-message fixture row lands on the 1024 floor."""
    from lqh.train import defaults

    training = launch(type="sft")["config"]["training"]
    assert training["max_seq_length"] == defaults.SEQ_LENGTH_GRANULARITY
    assert training["auto_seq_length"] is True


def test_sequence_length_is_capped_at_the_ceiling_and_skips_are_announced(
    launch, fake_estimate
):
    from lqh.train import defaults

    fake_estimate["longest"] = 90_000
    fake_estimate["over"] = 3
    rec = launch(type="sft")
    assert rec["config"]["training"]["max_seq_length"] == defaults.MAX_SEQ_LENGTH_CEILING
    assert rec["config"]["dataset_rows"]["over_limit"] == 3
    from lqh.tools.handlers import _training_data_line

    line = _training_data_line(rec["config"])
    assert "3 conversations too long" in line
    assert "seq" not in line.lower()


def test_no_skip_note_in_the_normal_case(launch, fake_estimate):
    from lqh.tools.handlers import _training_data_line

    rec = launch(type="sft")
    assert "over_limit" not in rec["config"]["dataset_rows"]
    line = _training_data_line(rec["config"])
    assert "too long" not in line
    assert "seq" not in line.lower()


def test_dpo_keeps_the_fixed_length_and_is_never_auto(launch, fake_estimate):
    from lqh.train import defaults

    training = launch(type="on_policy_dpo")["config"]["base_config"]["training"]
    assert training["max_seq_length"] == defaults.MAX_SEQ_LENGTH
    assert training["auto_seq_length"] is False
    assert "paths" not in fake_estimate  # the estimator did not even run


def test_sweep_children_inherit_the_derived_length(launch, fake_estimate):
    rec = launch(type="sft", enable_sweep=True)
    assert rec["config"]["base_config"]["training"]["max_seq_length"] == 6144


def test_expert_env_override_pins_the_length(launch, fake_estimate, monkeypatch):
    monkeypatch.setenv("LQH_MAX_SEQ_LENGTH", "16384")
    training = launch(type="sft")["config"]["training"]
    assert training["max_seq_length"] == 16384
    assert training["auto_seq_length"] is False
    assert "paths" not in fake_estimate  # pinned: no estimate needed


def test_expert_env_override_rejects_nonsense(launch, monkeypatch):
    for bad in ("abc", "12", "9999999"):
        monkeypatch.setenv("LQH_MAX_SEQ_LENGTH", bad)
        rec = launch(type="sft")
        assert rec["result"].ok is False
        assert rec["result"].error_kind == "config"
        assert "LQH_MAX_SEQ_LENGTH" in rec["result"].content
        assert "config" not in rec


def test_start_training_schema_has_no_sequence_length_argument():
    """The whole point: users never see the knob."""
    from lqh.tools.definitions import get_all_tools

    tool = next(t for t in get_all_tools() if t["function"]["name"] == "start_training")
    props = tool["function"]["parameters"]["properties"]
    assert "max_seq_length" not in props
    assert not any("seq_len" in k or "context_length" in k for k in props)


def test_batch_size_scales_with_rows_and_honours_the_epoch_override(
    launch, tmp_path
):
    """Row count AND the caller's epochs must both reach the derivation — with
    fewer epochs the same rows need a smaller batch to keep the step count."""
    from lqh.train import defaults

    project = tmp_path / "proj"
    big = _make_dataset(project, "big", rows=3_000)

    three = launch(type="sft", dataset=big)["config"]["training"]
    assert three["effective_batch_size"] == defaults.sft_effective_batch(3_000, 3)
    assert three["effective_batch_size"] == 90

    one = launch(type="sft", dataset=big, num_epochs=1)["config"]["training"]
    assert one["effective_batch_size"] == defaults.sft_effective_batch(3_000, 1)
    assert one["effective_batch_size"] == 30
    assert one["num_epochs"] == 1


def test_repeat_weighted_rows_size_the_batch(launch, tmp_path):
    """`repeat` multiplies what the run actually trains on, so the batch must
    be derived from the effective count, not the raw one."""
    from lqh.train import defaults

    project = tmp_path / "proj"
    _make_dataset(project, "small_src", rows=1_000)
    rec = launch(
        type="sft",
        dataset=[{"path": "datasets/small_src", "repeat": 4}],
    )
    assert rec["config"]["dataset_rows"]["train_effective"] == 4_000
    assert rec["config"]["training"]["effective_batch_size"] == (
        defaults.sft_effective_batch(4_000, 3)
    )


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
