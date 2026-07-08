"""Compatibility entry point for the cloud DPO agent e2e.

This replaces the old SSH-remote prototype coverage with the current
cloud-first fine-tuning path. The detailed SFT/DPO assertions live in
``test_cloud_finetune_agent_e2e.py``.
"""

from __future__ import annotations

import time

import pytest

from tests.function.test_cloud_finetune_agent_e2e import (
    _assert_common_training_contract,
    _run_agent_e2e,
)


def test_full_pipeline_cloud_dpo_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    run_name = f"dpo_full_pipeline_e2e_{int(time.time())}"
    result = _run_agent_e2e(monkeypatch, "on_policy_dpo", run_name)
    _assert_common_training_contract(result, "on_policy_dpo", run_name)
