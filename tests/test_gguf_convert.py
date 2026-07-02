"""Unit tests for the HF→GGUF conversion worker helpers.

The heavy path (llama.cpp convert/quantize, PEFT merge) only runs inside
the cloud sandbox, but the env parsing, binary resolution, model-dir
discovery, and smoke-test gating are pure and must stay correct.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lqh.remote import gguf_convert


def test_main_requires_source_and_quants(monkeypatch):
    monkeypatch.delenv("LQH_GGUF_SOURCE_URL", raising=False)
    monkeypatch.delenv("LQH_GGUF_QUANTS", raising=False)
    assert gguf_convert.main() == 2


def test_main_parses_env_and_forwards(monkeypatch):
    captured = {}

    def fake_run_gguf(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(gguf_convert, "run_gguf", fake_run_gguf)
    monkeypatch.setenv("LQH_GGUF_SOURCE_URL", "https://r2/presigned")
    monkeypatch.setenv("LQH_GGUF_QUANTS", "Q4_K, Q8_0 ,")
    monkeypatch.setenv("LQH_GGUF_IS_LORA", "1")
    monkeypatch.setenv("LQH_GGUF_BASE_MODEL", "LiquidAI/LFM2.5-1.2B")
    monkeypatch.setenv("LQH_GGUF_INCLUDE_F16", "1")
    monkeypatch.setenv("LQH_GGUF_HF_REPO", "me/repo")
    monkeypatch.setenv("LQH_GGUF_HF_PRIVATE", "0")
    monkeypatch.setenv("LQH_GGUF_FILENAME", "abc-model-lora.tar.gz")
    monkeypatch.setenv("LQH_PROJECT_ID", "proj")
    monkeypatch.setenv("LQH_JOB_ID", "job-1")
    monkeypatch.setenv("LQH_GGUF_SOURCE_ARTIFACT_ID", "art-1")

    assert gguf_convert.main() == 0
    assert captured["quants"] == ["Q4_K", "Q8_0"]
    assert captured["is_lora"] is True
    assert captured["base_model"] == "LiquidAI/LFM2.5-1.2B"
    assert captured["include_f16"] is True
    assert captured["hf_repo"] == "me/repo"
    assert captured["hf_private"] is False
    assert captured["project_id"] == "proj"
    assert captured["source_artifact_id"] == "art-1"


def test_main_returns_1_on_worker_failure(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("convert exploded")

    monkeypatch.setattr(gguf_convert, "run_gguf", boom)
    monkeypatch.setenv("LQH_GGUF_SOURCE_URL", "https://r2/presigned")
    monkeypatch.setenv("LQH_GGUF_QUANTS", "Q4_K")
    assert gguf_convert.main() == 1


def test_bin_env_override(monkeypatch):
    monkeypatch.setenv("LLAMA_QUANTIZE_BIN", "/custom/llama-quantize")
    assert gguf_convert._bin("llama-quantize", "LLAMA_QUANTIZE_BIN") == "/custom/llama-quantize"


def test_bin_falls_back_to_bare_name(monkeypatch, tmp_path):
    monkeypatch.delenv("LLAMA_CLI_BIN", raising=False)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))  # no build/ dir
    assert gguf_convert._bin("llama-cli", "LLAMA_CLI_BIN") == "llama-cli"


def test_find_hf_model_dir(tmp_path):
    root = tmp_path / "src"
    model = root / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"\x00")
    assert gguf_convert._find_hf_model_dir(root) == model


def test_find_hf_model_dir_missing_weights(tmp_path):
    root = tmp_path / "src"
    model = root / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")  # config but no weights
    with pytest.raises(RuntimeError):
        gguf_convert._find_hf_model_dir(root)


def test_smoke_test_rejects_empty_output(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")

    def fake_run(cmd, *, capture=False):
        return subprocess.CompletedProcess(cmd, 0, stdout=gguf_convert._SMOKE_PROMPT, stderr="")

    monkeypatch.setattr(gguf_convert, "_run", fake_run)
    with pytest.raises(RuntimeError, match="no output"):
        gguf_convert._smoke_test(gguf)


def test_smoke_test_rejects_nonzero_exit(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")

    def fake_run(cmd, *, capture=False):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="load error")

    monkeypatch.setattr(gguf_convert, "_run", fake_run)
    with pytest.raises(RuntimeError, match="smoke test failed"):
        gguf_convert._smoke_test(gguf)


def test_smoke_test_passes_on_generation(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")

    def fake_run(cmd, *, capture=False):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=gguf_convert._SMOKE_PROMPT + " Paris.", stderr=""
        )

    monkeypatch.setattr(gguf_convert, "_run", fake_run)
    gguf_convert._smoke_test(gguf)  # no raise
