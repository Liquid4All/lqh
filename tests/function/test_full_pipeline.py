"""Compatibility entry point for the cloud fine-tuning agent e2e.

The fine-tuning product path is cloud-first: the agent calls
``start_training`` with no remote argument, the tool routes to LQH Cloud, and
``training_status`` reads the cloud run mirror.
"""

from __future__ import annotations

import time

import pytest

from tests.function.test_cloud_finetune_agent_e2e import (
    _assert_common_training_contract,
    _run_agent_e2e,
)


def test_full_pipeline_cloud_sft_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    run_name = f"sft_full_pipeline_e2e_{int(time.time())}"
    result = _run_agent_e2e(monkeypatch, "sft", run_name)
    _assert_common_training_contract(result, "sft", run_name)
