"""Unit tests for batch-size calibration helpers that don't need a GPU.

The probe itself needs CUDA, but the formula/bucket/effective-batch math
and the backend client wiring are pure and must stay correct.
"""

from __future__ import annotations

from lqh.train import calibrate


def test_seq_len_bucket():
    assert calibrate.seq_len_bucket(0) == 2048
    assert calibrate.seq_len_bucket(2048) == 2048
    assert calibrate.seq_len_bucket(2049) == 3072
    assert calibrate.seq_len_bucket(4096) == 4096


def test_profile_key_shape():
    k = calibrate.profile_key(
        base_model="LiquidAI/LFM2-1.2B",
        method="lora",
        gpu_type="A100-80GB",
        modality="text",
        seq_len=2048,
        lora_rank=32,
        dtype="bf16",
        image_id="im-abc",
    )
    assert k["base_model"] == "LiquidAI/LFM2-1.2B"
    assert k["training_method"] == "lora"
    assert k["max_seq_len_bucket"] == 2048
    assert k["lora_rank"] == 32


def test_apply_preserves_effective_batch():
    cfg: dict = {}
    # target effective = 16; safe micro = 4 → accum 4
    accum = calibrate._apply(cfg, 4, 16)
    assert cfg["per_device_batch_size"] == 4
    assert accum == 4
    assert cfg["gradient_accumulation_steps"] == 4
    # safe micro = 8 → accum 2
    calibrate._apply(cfg, 8, 16)
    assert cfg["gradient_accumulation_steps"] == 2
    # safe micro larger than effective → accum floors at 1
    calibrate._apply(cfg, 32, 16)
    assert cfg["gradient_accumulation_steps"] == 1


def test_get_cached_profile_noop_without_env(monkeypatch):
    monkeypatch.delenv("LQH_BASE_URL", raising=False)
    monkeypatch.delenv("LQH_API_TOKEN", raising=False)
    assert calibrate._get_cached_profile({"base_model": "x"}) is None


def test_get_cached_profile_parses_response(monkeypatch):
    monkeypatch.setenv("LQH_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LQH_API_TOKEN", "tok")

    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"profile": {"measured_micro_batch": 8}}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["auth"] = headers["Authorization"]
        return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)
    prof = calibrate._get_cached_profile({"base_model": "x"})
    assert prof == {"measured_micro_batch": 8}
    assert captured["url"] == "https://api.example.com/v1/cloud/batch_profile"
    assert captured["auth"] == "Bearer tok"


def test_report_oom_downgrade_noop_outside_cloud(monkeypatch):
    monkeypatch.delenv("LQH_JOB_ID", raising=False)
    monkeypatch.delenv("LQH_API_TOKEN", raising=False)
    # Should simply return without raising or calling the network.
    calibrate.report_oom_downgrade({"base_model": "x", "training": {}})
