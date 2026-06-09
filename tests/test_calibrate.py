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
    # non-divisible micro-batches should not undershoot the requested
    # effective batch.
    calibrate._apply(cfg, 48, 256)
    assert cfg["gradient_accumulation_steps"] == 6


def test_ensure_batch_defaults_targets_effective_batch():
    cfg: dict = {}
    calibrate.ensure_batch_defaults(cfg, default_micro_batch=256)
    assert cfg["per_device_batch_size"] == 256
    assert cfg["gradient_accumulation_steps"] == 1
    assert cfg["effective_batch_size"] == 256


def test_ensure_batch_defaults_honors_explicit_batch_shape():
    cfg = {"per_device_batch_size": 8, "gradient_accumulation_steps": 4}
    calibrate.ensure_batch_defaults(cfg, default_micro_batch=4)
    assert cfg["per_device_batch_size"] == 8
    assert cfg["gradient_accumulation_steps"] == 4
    assert cfg["effective_batch_size"] == 32


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


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_name(_idx):
        return "FakeGPU"


class _FakeTorch:
    cuda = _FakeCuda()


def _patch_torch(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())


def test_autotune_applies_cached_value_below_configured(monkeypatch):
    """A cached measured value smaller than the configured micro-batch
    must still apply (the old `micro >= cur_micro` guard ignored it and
    re-probed on every run)."""
    _patch_torch(monkeypatch)
    monkeypatch.setattr(
        calibrate, "_get_cached_profile", lambda key: {"measured_micro_batch": 64}
    )
    monkeypatch.setattr(
        calibrate,
        "_probe_micro_batch",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not probe on cache hit")),
    )
    cfg = {
        "per_device_batch_size": 256,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 256,
    }
    calibrate.maybe_autotune_batch_size(
        cfg, model=object(), tokenizer=object(), base_model="m", method="lora", lora_rank=32
    )
    assert cfg["per_device_batch_size"] == 64
    assert cfg["gradient_accumulation_steps"] == 4  # effective 256 preserved


def test_autotune_cached_value_respects_admin_cap(monkeypatch):
    _patch_torch(monkeypatch)
    monkeypatch.setattr(
        calibrate,
        "_get_cached_profile",
        lambda key: {"measured_micro_batch": 128, "admin_max_micro_batch": 32},
    )
    cfg = {"per_device_batch_size": 4, "gradient_accumulation_steps": 4}
    calibrate.maybe_autotune_batch_size(
        cfg, model=object(), tokenizer=object(), base_model="m", method="lora", lora_rank=32
    )
    assert cfg["per_device_batch_size"] == 32


def test_autotune_probes_full_range_despite_small_config(monkeypatch):
    """Old run configs carry per_device_batch_size=4; the probe must
    still search the full candidate range, not cap at the configured
    value (the bug that froze discovery at 4, GPU_TYPE_2.md)."""
    _patch_torch(monkeypatch)
    monkeypatch.setattr(calibrate, "_get_cached_profile", lambda key: None)
    seen = {}

    def fake_probe(model, tokenizer, *, seq_len, max_micro_batch, target_headroom, gradient_checkpointing):
        seen["max_micro_batch"] = max_micro_batch
        return 96, 25_000

    monkeypatch.setattr(calibrate, "_probe_micro_batch", fake_probe)
    posted = {}

    def fake_post(key, **kwargs):
        posted.update(kwargs)
        return True

    monkeypatch.setattr(calibrate, "_post_profile", fake_post)
    cfg = {
        "per_device_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 64,
    }
    calibrate.maybe_autotune_batch_size(
        cfg, model=object(), tokenizer=object(), base_model="m", method="lora", lora_rank=32
    )
    assert seen["max_micro_batch"] == max(calibrate._PROBE_BATCHES)
    assert cfg["per_device_batch_size"] == 96
    assert cfg["gradient_accumulation_steps"] == 1  # ceil(64/96)
    assert posted["micro_batch"] == 96
    assert posted["source"] == "probe"


def test_autotune_probe_cap_uses_admin_ceiling(monkeypatch):
    _patch_torch(monkeypatch)
    monkeypatch.setattr(
        calibrate,
        "_get_cached_profile",
        lambda key: {"measured_micro_batch": None, "admin_max_micro_batch": 48},
    )
    seen = {}

    def fake_probe(model, tokenizer, *, seq_len, max_micro_batch, target_headroom, gradient_checkpointing):
        seen["max_micro_batch"] = max_micro_batch
        return 48, 10_000

    monkeypatch.setattr(calibrate, "_probe_micro_batch", fake_probe)
    monkeypatch.setattr(calibrate, "_post_profile", lambda key, **kw: True)
    cfg = {"per_device_batch_size": 256, "effective_batch_size": 256}
    calibrate.maybe_autotune_batch_size(
        cfg, model=object(), tokenizer=object(), base_model="m", method="lora", lora_rank=32
    )
    assert seen["max_micro_batch"] == 48
    assert cfg["per_device_batch_size"] == 48
