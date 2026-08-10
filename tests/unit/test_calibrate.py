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
    assert cfg["effective_batch_size"] == 16
    # safe micro = 8 → accum 2
    calibrate._apply(cfg, 8, 16)
    assert cfg["gradient_accumulation_steps"] == 2
    assert cfg["effective_batch_size"] == 16
    # safe micro larger than effective → accum floors at 1
    calibrate._apply(cfg, 32, 16)
    assert cfg["gradient_accumulation_steps"] == 1
    assert cfg["effective_batch_size"] == 16


def test_apply_rounds_down_and_records_the_realized_batch():
    """A micro-batch that does not divide the target must UNDER-shoot it.

    Rounding up overshot by up to 2x (target 53, probed micro 48 → true batch
    96), which halved the optimizer-step count the SFT target is derived to
    guarantee (lqh.train.defaults.sft_effective_batch). The realized value is
    written back because sft.py's step accounting reads these fields.
    """
    cfg: dict = {}
    calibrate._apply(cfg, 48, 53)
    assert cfg["per_device_batch_size"] == 48
    assert cfg["gradient_accumulation_steps"] == 1
    assert cfg["effective_batch_size"] == 48

    for micro, target in ((96, 256), (128, 132), (64, 75), (24, 53)):
        cfg = {}
        accum = calibrate._apply(cfg, micro, target)
        realized = cfg["per_device_batch_size"] * accum
        assert realized == cfg["effective_batch_size"]
        assert realized <= target
        # Never collapse to a fraction of the target either.
        assert realized > target // 2


def test_derived_sft_batches_survive_the_probe_ladder():
    """End-to-end on the real numbers: every batch the SFT derivation can
    produce, against every micro-batch the probe can pick, still clears the
    step floor the derivation promised."""
    from lqh.train import defaults

    for rows, epochs in ((1_790, 3), (4_433, 3), (500, 3), (8_000, 2), (300, 3)):
        target = defaults.sft_effective_batch(rows, epochs)
        for micro in calibrate._PROBE_BATCHES:
            if micro > target:
                continue
            cfg: dict = {}
            calibrate._apply(cfg, micro, target)
            steps = defaults.optimizer_steps(
                train_rows=rows,
                num_epochs=epochs,
                effective_batch_size=cfg["effective_batch_size"],
            )
            assert steps >= defaults.SFT_MIN_HEALTHY_OPTIMIZER_STEPS, (
                rows, epochs, target, micro, cfg, steps,
            )


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


def test_apply_never_exceeds_effective_batch_target():
    cfg = {}
    accum = calibrate._apply(cfg, micro=128, target_effective=16)
    assert cfg["per_device_batch_size"] == 16
    assert accum == 1


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
    assert cfg["per_device_batch_size"] == 16


def test_autotune_probes_memory_not_the_effective_target(monkeypatch):
    """The probe measures the MEMORY ceiling; the effective target only caps
    what gets applied.

    The batch profile it writes back is shared across runs and keyed on
    model/GPU/seq-len/rank/dtype/image with no batch component. Capping the
    probe at this run's target would publish "this dataset was small" as "this
    model cannot fit more than N per device", and every later run of that model
    on that GPU would inherit it from the cache — which matters now that the
    text SFT LoRA target scales with the dataset."""
    _patch_torch(monkeypatch)
    # A probe only measures past the target when the measurement can be
    # published, so this path needs a reachable backend (as a cloud sandbox has).
    monkeypatch.setattr(calibrate, "_profile_writes_enabled", lambda: True)
    monkeypatch.setattr(calibrate, "_get_cached_profile", lambda key: None)
    seen = {}

    def fake_probe(model, tokenizer, *, max_micro_batch, **kwargs):
        seen["max_micro_batch"] = max_micro_batch
        seen.update(kwargs)
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
    # Probed to the top of the ladder, not to this run's target of 64.
    assert seen["max_micro_batch"] == max(calibrate._PROBE_BATCHES)
    assert seen["pair_batch"] is False
    # Applied: capped at the target, so the true optimizer batch stays 64.
    assert cfg["per_device_batch_size"] == 64
    assert cfg["gradient_accumulation_steps"] == 1
    assert cfg["effective_batch_size"] == 64
    # Cached: the measurement (96), which is what the profile column means.
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

    def fake_probe(model, tokenizer, *, max_micro_batch, **kwargs):
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


def test_autotune_dpo_method_keys_separately_and_probes_pairs(monkeypatch):
    """DPO must not consume an SFT-cached batch: its key is dpo-prefixed
    and its probe is pair-shaped."""
    _patch_torch(monkeypatch)
    monkeypatch.setattr(calibrate, "_profile_writes_enabled", lambda: True)
    seen_key = {}

    def fake_get(key):
        seen_key.update(key)
        return None

    monkeypatch.setattr(calibrate, "_get_cached_profile", fake_get)
    seen = {}

    def fake_probe(model, tokenizer, **kwargs):
        seen.update(kwargs)
        return 16, 30_000

    monkeypatch.setattr(calibrate, "_probe_micro_batch", fake_probe)
    posted_key = {}

    def fake_post(key, **kwargs):
        posted_key.update(key)
        return True

    monkeypatch.setattr(calibrate, "_post_profile", fake_post)
    cfg = {"per_device_batch_size": 256, "effective_batch_size": 256}
    calibrate.maybe_autotune_batch_size(
        cfg, model=object(), tokenizer=object(), base_model="m", method="dpo_lora", lora_rank=32
    )
    assert seen_key["training_method"] == "dpo_lora"
    assert posted_key["training_method"] == "dpo_lora"
    assert seen["pair_batch"] is True
    assert cfg["per_device_batch_size"] == 16


def test_oom_downgrade_skipped_when_checkpointing_disabled(monkeypatch):
    """The OOM self-heal must obey the same cache contract as the probe:
    an uncheckpointed run's OOM must not downgrade the shared
    (checkpointing-ON) profile."""
    monkeypatch.setenv("LQH_JOB_ID", "job-1")
    monkeypatch.setenv("LQH_API_TOKEN", "tok")
    _patch_torch(monkeypatch)

    def fail_post(key, **kwargs):
        raise AssertionError("must not post a downgrade when checkpointing is off")

    monkeypatch.setattr(calibrate, "_post_profile", fail_post)
    calibrate.report_oom_downgrade(
        {
            "base_model": "m",
            "training": {"gradient_checkpointing": False, "per_device_batch_size": 64},
        }
    )


def test_oom_downgrade_posts_when_checkpointing_on(monkeypatch):
    monkeypatch.setenv("LQH_JOB_ID", "job-1")
    monkeypatch.setenv("LQH_API_TOKEN", "tok")
    _patch_torch(monkeypatch)
    posted = {}

    def fake_post(key, **kwargs):
        posted.update(key)
        posted.update(kwargs)
        return True

    monkeypatch.setattr(calibrate, "_post_profile", fake_post)
    calibrate.report_oom_downgrade(
        {
            "base_model": "m",
            "type": "dpo",
            "training": {"per_device_batch_size": 64},
        }
    )
    assert posted["micro_batch"] == 32
    assert posted["source"] == "downgraded"
    assert posted["training_method"] == "dpo_lora"


def test_autotune_no_cache_io_when_checkpointing_disabled(monkeypatch):
    """The shared cache is measured with gradient checkpointing ON; a run
    with it disabled must not consume cached values nor write back its
    own (much smaller) probe result."""
    _patch_torch(monkeypatch)
    monkeypatch.setattr(
        calibrate,
        "_get_cached_profile",
        lambda key: {"measured_micro_batch": 256, "admin_max_micro_batch": None},
    )

    def fake_probe(model, tokenizer, **kwargs):
        assert kwargs["gradient_checkpointing"] is False
        return 8, 40_000

    monkeypatch.setattr(calibrate, "_probe_micro_batch", fake_probe)

    def fail_post(key, **kwargs):
        raise AssertionError("must not write back when checkpointing is off")

    monkeypatch.setattr(calibrate, "_post_profile", fail_post)
    cfg = {
        "per_device_batch_size": 4,
        "effective_batch_size": 64,
        "gradient_checkpointing": False,
    }
    calibrate.maybe_autotune_batch_size(
        cfg, model=object(), tokenizer=object(), base_model="m", method="lora", lora_rank=32
    )
    # The cached 256 (measured WITH checkpointing) must not apply; the
    # local probe result does.
    assert cfg["per_device_batch_size"] == 8


def test_autotune_caps_the_probe_when_it_cannot_publish(monkeypatch):
    """No backend to POST to (local / SSH-direct without a job token) means a
    measurement above the target is measured, discarded, and re-measured next
    run. Cap the probe at what this run can apply, and skip the write-back —
    a truncated probe is a lower bound, not a measurement."""
    _patch_torch(monkeypatch)
    monkeypatch.setattr(calibrate, "_profile_writes_enabled", lambda: False)
    monkeypatch.setattr(calibrate, "_get_cached_profile", lambda key: None)
    seen: dict = {}
    posted: dict = {}

    def fake_probe(model, tokenizer, *, max_micro_batch, **kwargs):
        seen["max_micro_batch"] = max_micro_batch
        return min(96, max_micro_batch), 25_000

    monkeypatch.setattr(calibrate, "_probe_micro_batch", fake_probe)
    monkeypatch.setattr(
        calibrate, "_post_profile", lambda key, **kw: posted.update(kw) or True
    )
    cfg = {"per_device_batch_size": 4, "effective_batch_size": 64}
    calibrate.maybe_autotune_batch_size(
        cfg, model=object(), tokenizer=object(), base_model="m", method="lora",
        lora_rank=32,
    )
    assert seen["max_micro_batch"] == 64
    assert cfg["per_device_batch_size"] == 64
    assert posted == {}
