"""Safe batch-size auto-tuning for cloud training (GPU_TYPE.md Design §6).

Flow, run once per (base_model, method, gpu_type, seq-len bucket, lora_rank,
dtype, image) combo:

  1. GET the cached batch profile from the backend. On a hit, apply the
     cached micro-batch and return (zero probe cost).
  2. On a miss, probe: try increasing micro-batches, measure
     ``torch.cuda.max_memory_reserved`` *including a representative
     generate()* (the documented OOM happened in checkpoint-eval generate,
     not the train step), reserve headroom, pick the largest safe batch.
  3. POST the result back immediately via the scoped job token so a later
     preemption doesn't lose the measurement.

Effective batch size is preserved by adjusting gradient_accumulation_steps.

Everything here is best-effort: ANY failure (no GPU, backend down, probe
error) leaves ``training_cfg`` untouched so the run proceeds on the
configured/default batch. Auto-tuning must never crash a training run.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Target fraction of VRAM to occupy at peak. The remaining headroom
# (~15%) covers fragmentation and the eval-time generate() spike.
_DEFAULT_HEADROOM = 0.85

# Micro-batch candidates probed in ascending order.
_PROBE_BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64]


def _in_cloud() -> bool:
    """True inside a backend-managed cloud sandbox (job token present)."""
    return bool(os.environ.get("LQH_JOB_ID")) and bool(os.environ.get("LQH_API_TOKEN"))


def _api_base() -> str | None:
    base = os.environ.get("LQH_BASE_URL", "").rstrip("/")
    if not base:
        return None
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def seq_len_bucket(seq_len: int) -> int:
    """Round a sequence length up to the nearest 1024 so near-equal seq
    lengths share one profile."""
    if seq_len <= 0:
        return 2048
    return int(math.ceil(seq_len / 1024.0) * 1024)


def profile_key(
    *,
    base_model: str,
    method: str,
    gpu_type: str,
    modality: str,
    seq_len: int,
    lora_rank: int,
    dtype: str,
    image_id: str,
) -> dict[str, Any]:
    return {
        "base_model": base_model,
        "training_method": method,
        "gpu_type": gpu_type,
        "modality": modality,
        "max_seq_len_bucket": seq_len_bucket(seq_len),
        "lora_rank": lora_rank,
        "dtype": dtype,
        "image_id": image_id,
    }


def _get_cached_profile(key: dict[str, Any]) -> dict[str, Any] | None:
    base = _api_base()
    token = os.environ.get("LQH_API_TOKEN")
    if not base or not token:
        return None
    try:
        import httpx

        params = {k: str(v) for k, v in key.items()}
        r = httpx.get(
            base + "/cloud/batch_profile",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        if r.status_code != 200:
            return None
        return r.json().get("profile")
    except Exception:
        return None


def _post_profile(
    key: dict[str, Any],
    *,
    micro_batch: int,
    grad_accum: int,
    peak_mem_mb: int | None,
    headroom: float,
    source: str,
    stable: bool,
) -> None:
    base = _api_base()
    token = os.environ.get("LQH_API_TOKEN")
    if not base or not token:
        return
    try:
        import httpx

        body = dict(key)
        body.update(
            measured_micro_batch=micro_batch,
            measured_grad_accum=grad_accum,
            measured_peak_mem_mb=peak_mem_mb,
            headroom_target=headroom,
            source=source,
            stable=stable,
        )
        httpx.post(
            base + "/cloud/batch_profile",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
    except Exception:
        return


def _apply(training_cfg: dict[str, Any], micro: int, target_effective: int) -> int:
    """Set per_device_batch_size to ``micro`` and adjust grad accumulation
    to preserve the configured effective batch size. Returns grad_accum."""
    micro = max(1, micro)
    accum = max(1, round(target_effective / micro))
    training_cfg["per_device_batch_size"] = micro
    training_cfg["gradient_accumulation_steps"] = accum
    return accum


def _probe_micro_batch(model, tokenizer, *, seq_len: int, start: int, target_headroom: float):
    """Probe increasing micro-batches with a real fwd+backward and a short
    generate(), returning (safe_micro_batch, peak_mem_mb) or (None, None).

    Conservative: stops at the first batch that exceeds the memory budget
    or OOMs, and uses the last batch that fit."""
    import torch

    if not torch.cuda.is_available():
        return None, None
    device = next(model.parameters()).device
    if device.type != "cuda":
        return None, None
    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(total * target_headroom)
    vocab = max(2, int(getattr(tokenizer, "vocab_size", 0) or 32000))

    safe: int | None = None
    peak: int | None = None
    candidates = [b for b in _PROBE_BATCHES if b >= max(1, start)] or [max(1, start)]
    was_training = model.training
    for b in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            input_ids = torch.randint(0, vocab, (b, seq_len), device=device)
            attn = torch.ones_like(input_ids)
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            # Eval-time generate peak — the documented OOM source. Include
            # it so the chosen batch survives checkpoint-eval, not just the
            # train step.
            model.eval()
            with torch.no_grad():
                prompt = input_ids[:, : min(seq_len, 32)]
                model.generate(prompt, max_new_tokens=8, do_sample=False)
            used = int(torch.cuda.max_memory_reserved(device))
            if used <= budget:
                safe, peak = b, used // (1024 * 1024)
            else:
                break
        except Exception:
            # OOM (torch.cuda.OutOfMemoryError) or any probe error — the
            # previous safe batch stands.
            break
        finally:
            torch.cuda.empty_cache()
    if was_training:
        model.train()
    else:
        model.eval()
    return safe, peak


def maybe_autotune_batch_size(
    training_cfg: dict[str, Any],
    *,
    model,
    tokenizer,
    base_model: str,
    method: str,
    lora_rank: int = 0,
    modality: str = "text",
) -> None:
    """Calibrate per_device_batch_size in place when auto-tuning is on.

    Enabled by ``training.auto_batch`` (default: on in cloud sandboxes,
    off locally where there is no backend to cache against). Never raises.
    """
    try:
        enabled = bool(training_cfg.get("auto_batch", _in_cloud()))
        if not enabled:
            return
        import torch

        if not torch.cuda.is_available():
            return

        seq_len = int(training_cfg.get("max_seq_length", 2048))
        dtype = "bf16" if training_cfg.get("bf16", True) else "fp32"
        gpu_type = os.environ.get("LQH_GPU_TYPE") or torch.cuda.get_device_name(0)
        image_id = os.environ.get("LQH_IMAGE_ID", "")
        key = profile_key(
            base_model=base_model,
            method=method,
            gpu_type=gpu_type,
            modality=modality,
            seq_len=seq_len,
            lora_rank=lora_rank,
            dtype=dtype,
            image_id=image_id,
        )

        cur_micro = int(training_cfg.get("per_device_batch_size", 4))
        cur_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
        target_effective = max(1, cur_micro * cur_accum)

        cached = _get_cached_profile(key)
        if cached and cached.get("measured_micro_batch"):
            micro = int(cached["measured_micro_batch"])
            cap = cached.get("admin_max_micro_batch")
            if cap:
                micro = min(micro, int(cap))
            accum = _apply(training_cfg, micro, target_effective)
            print(
                f"calibrate: cached micro_batch={micro} grad_accum={accum} "
                f"(effective {target_effective}, gpu={gpu_type})",
                flush=True,
            )
            return

        headroom = float(training_cfg.get("batch_headroom", _DEFAULT_HEADROOM))
        safe, peak = _probe_micro_batch(
            model, tokenizer, seq_len=seq_len, start=cur_micro, target_headroom=headroom
        )
        if not safe:
            print("calibrate: probe found no safe batch; keeping configured default", flush=True)
            return
        accum = _apply(training_cfg, safe, target_effective)
        _post_profile(
            key,
            micro_batch=safe,
            grad_accum=accum,
            peak_mem_mb=peak,
            headroom=headroom,
            source="probe",
            stable=True,
        )
        print(
            f"calibrate: probed micro_batch={safe} grad_accum={accum} peak={peak}MB "
            f"(effective {target_effective}, gpu={gpu_type})",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — telemetry must never crash the run
        print(f"calibrate: auto-tune skipped ({exc})", flush=True)


def report_oom_downgrade(config: dict[str, Any]) -> None:
    """Self-heal: after an OOM on a config that used a cached batch size,
    write back a halved micro-batch flagged unstable so the next run uses
    the smaller value (GPU_TYPE.md §6). Best-effort; never raises.

    Called from the trainer's OOM handler. Does NOT run for preemption —
    preemption is not a memory problem.
    """
    try:
        if not _in_cloud():
            return
        import torch

        training_cfg = config.get("training", {}) or {}
        if not bool(training_cfg.get("auto_batch", True)):
            return
        lora_cfg = config.get("lora", {}) or {}
        lora_enabled = bool(lora_cfg.get("enabled", True))
        method = "lora" if lora_enabled else "full"
        lora_rank = int(lora_cfg.get("r", 32)) if lora_enabled else 0
        seq_len = int(training_cfg.get("max_seq_length", 2048))
        dtype = "bf16" if training_cfg.get("bf16", True) else "fp32"
        gpu_type = os.environ.get("LQH_GPU_TYPE") or (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        )
        key = profile_key(
            base_model=config.get("base_model", ""),
            method=method,
            gpu_type=gpu_type,
            modality=config.get("modality", "text"),
            seq_len=seq_len,
            lora_rank=lora_rank,
            dtype=dtype,
            image_id=os.environ.get("LQH_IMAGE_ID", ""),
        )
        cur_micro = int(training_cfg.get("per_device_batch_size", 4))
        downgraded = max(1, cur_micro // 2)
        _post_profile(
            key,
            micro_batch=downgraded,
            grad_accum=int(training_cfg.get("gradient_accumulation_steps", 1)) * 2,
            peak_mem_mb=None,
            headroom=float(training_cfg.get("batch_headroom", _DEFAULT_HEADROOM)),
            source="downgraded",
            stable=False,
        )
        print(f"calibrate: OOM self-heal — wrote downgraded micro_batch={downgraded}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"calibrate: OOM downgrade skipped ({exc})", flush=True)
